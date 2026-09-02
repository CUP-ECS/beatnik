# Which blob-aware far field to build: a recommendation

**Status:** RECOMMENDATION NOTE — not a task, and not Beatnik work. It answers
one question: of the three *permanent-fix* routes named in
[tasks/add-canopy.md](add-canopy.md) **X1** ("The option set, and which one this
points at") — option 2 (Cartesian-Taylor), option 3 (black-box FMM), option 4
(kernel-independent FMM) — which should Canopy build?

It does not change X1, whose recommendation is deliberately conditional, and it
does not touch the near-term unblock (X1 option 1), which stays the right first
move whichever permanent route is chosen. The implementation lives in
[tasks/canopy0.md](canopy0.md) **C1**/**F6**, in the Canopy repository, and the
only acceptance test Beatnik owns is X1's.

**Companion documents:** [tasks/canopy-bbFMM.md](canopy-bbFMM.md) (option 3 in
detail), [tasks/canopy-kIndp.md](canopy-kIndp.md) (option 4 in detail),
[tasks/canopy0.md](canopy0.md) **F6** (what a second basis costs inside Canopy),
`canopy-questions.md` at the repository root (the reference author's derivation
for option 2).

## Recommendation

**Option 3, black-box FMM (Fong & Darve), as the permanent fix.** The four
criteria split as follows, and only one of them is close.

| Criterion | 2 — Cartesian-Taylor | 3 — bbFMM | 4 — KIFMM |
| --- | --- | --- | --- |
| GPU parallelism | good | **best** | good, with new machinery |
| Distributed cost | **best** | worst | middle |
| Canopy refactor | smallest code, real derivation risk | **moderate, no math risk** | largest |
| Accuracy reachable | $\sim\!10^{-2}$–$10^{-4}$, structurally capped | **tunable past $10^{-8}$** | $10^{-6}$–$10^{-9}$, floored by conditioning |
| Memory at that accuracy | trivial | **the one real cost** | lowest per unit accuracy |
| Error argument under softening | valid | **valid, unweakened** | invalid (canopy-kIndp.md) |

Accuracy is what decides it. Beatnik's whole reason for wanting a blob-aware far
field is to make `--br-approximation fmm` defensible at something near
direct-solve tolerance (add-canopy.md **Problem**, and X1's exit criterion of
$\tau_A \le 10^{-10}$ with `near_softening_factor = 0`). Option 2 cannot get
into that neighborhood at any tolerable DOF count — that is
[canopy0.md](canopy0.md) **F6(d)**, quantified below — and option 4's
representation error is floored by a regularized pseudo-inverse in exactly the
band being asked for, with its convergence theory invalidated by softening
rather than merely weakened (canopy-kIndp.md, "Where softening breaks it,
exactly"). Option 3 is the only route whose far-field error is a *tunable
truncation* with a convergence estimate that is unaffected by $b>0$ — softening
in fact makes Chebyshev interpolation converge slightly faster, since it moves
the kernel's singularity off the real axis.

That recommendation carries two conditions, both cheap and both measurements
rather than design work:

1. **Measure `n_unique_ops` before choosing the interpolation order $n$.** The
   memory cost of option 3 is (per-operator matrix) × (realized M2L operator
   keys) × (occupied depths), and only the first factor is a textbook number.
   See [Accuracy versus memory](#accuracy-versus-memory-cost).
2. **Treat SVD compression of the M2L operators as mandatory, not an
   optimization.** Fong & Darve present it as a constant-factor win; here it is
   what makes $n \ge 8$ representable at all.

**Option 2 is still the right answer if the measurement says Beatnik only needs
$10^{-4}$.** That is not a hedge, it is where the decision rule sits: see
[What would change this recommendation](#what-would-change-this-recommendation).
add-canopy.md **T5** is the measurement, and it is not yet run.

## Ability to parallelize on a GPU

Canopy is Kokkos throughout and its M2L is already the right shape for all three
routes, which is the single most useful fact in this comparison. The work-
dominant kernel is a team-per-target `Kokkos::parallel_for` walking a CSR pair
list, with each pair naming an index into a precomputed operator table held in a
device view (`canopy/src/Canopy_DownwardSweep.hpp:1471-1545`, the table at
`:337-341` and its build at `:1073-1108`). The inner loop is a dense
$N_t \times N_s$ matvec against a per-key operator — at the default $P=8$,
$N_t = (P{+}1)(P{+}2)/2 = 45$ and $N_s = (P{+}1)^2 = 81$
(`canopy/src/Canopy_LaplaceKernel.hpp:157`, `:488`).

**All three routes keep that structure and change only what fills the table.**
None of them needs a new traversal, a new pair list, or a new launch topology.

- **Option 3** replaces the operator with the evaluated kernel matrix
  $K_l(\bar{x}_p, \bar{y}_q)$ — a dense real $n^3 \times n^3$ block, three of
  them (one per kernel component). This is the *most* GPU-friendly of the three:
  no index arithmetic, no branching, no complex arithmetic, high arithmetic
  intensity, and it is batched-GEMM-shaped, so pairs sharing an operator key can
  be aggregated into a single GEMM rather than a team each. The remaining four
  operators (P2M, M2M, L2L, L2P) are also dense interpolation matvecs — the same
  shape again.
- **Option 2** replaces the operator with $(-1)^{|q|}b_{p+q}(R)$, a dense real
  $N_q \times N_q$ block with $N_q = \binom{p+3}{3}$. Same shape, much smaller.
  The one GPU trap is computing $b_n(R)$ *on the fly* per pair: the ladder
  $\partial_a P_m = -(2m{+}1)r_a P_{m+1}$ and the multi-index recurrence
  (`canopy-questions.md` §3) are sequential in order and want symmetric-tensor
  index arithmetic, so a per-pair device evaluation is register-heavy and
  divergent. It should be precomputed into Canopy's existing operator table
  exactly as the harmonic operator is, at which point the device code is a
  smaller version of option 3's.
- **Option 4**'s five operators are all dense matvecs too, so the unaccelerated
  form is fine on device — but its performance case rests on FFT-accelerating
  M2L (canopy-kIndp.md, "What survives intact"), which means batched forward and
  inverse transforms per box, per level, per component plus grid-embedding and
  extraction passes. That is genuinely new device machinery in the sweep, not a
  table swap, and it is the only place among the three where the GPU work
  changes character rather than dimension.

**A win common to all three, worth stating because it is immediate.** Canopy's
far-field gradient is today six extra `eval_phi` calls per target — a central
finite difference at $h = 10^{-5} w_{\rm self}$, carrying its own
`TODO: replace with analytical derivatives`
(`canopy/src/Canopy_LaplaceKernel.hpp:851-879`). L2P therefore costs $7\times$
what a value evaluation costs, on the *hottest per-particle kernel in the
sweep*. Every one of the three routes deletes that: option 3 because the
interpolated quantity is the vector kernel, option 4 because the local
representation is point sources, option 2 because a Taylor local expansion
differentiates analytically. It also deletes the $\sim\!10^{-10}$ plateau that
add-canopy.md **R1** warns must not be confused with the softening bias
(canopy0.md **F7**).

**A second win common to all three, and it is the larger one.** Removing the
`near_softening_factor` floor (`canopy/src/Canopy_CommunicationPlan.hpp:347-361`)
moves 5%–97% of pairs — the measured range in add-canopy.md **Problem** — out of
P2P and into M2L. P2P is $O(\text{ncrit}^2)$ per leaf pair; M2L is one operator
apply per cell pair. On a GPU that is a shift from the least favorable work in
the sweep to the most favorable, and it dwarfs any of the differences between
the three routes.

## Ability to use in a distributed environment

Nothing in any of the three routes touches the distribution: the tree builder,
partitioner, dual-tree traversal, communication plan and MAC are all
kernel-agnostic, and the plan already resolves level-mismatched pairs by MAC
(`canopy/src/Canopy_CommunicationPlan.hpp:338-361`). The only distributed
quantity that changes is **how many numbers per cell cross the network**, and
that scales linearly with far-field DOF per cell in two places: the shared-cell
multipole `MPI_Allreduce` (`canopy/src/Canopy_UpwardSweep.hpp:537-581`) and the
shared-cell local `MPI_Allreduce` in the downward sweep
(`canopy/src/Canopy_DownwardSweep.hpp:499-501`, `:1786`), plus the ghost
multipole exchange the plan sets up.

Today, at $P=8$ and `NComps = 3`, a cell carries $45 \times 3$ complex
coefficients = **270 doubles**. Comparable-accuracy DOF counts per cell,
$\times 3$ components:

| route | DOF per cell per component | doubles per cell | vs today |
| --- | --- | --- | --- |
| harmonic, $P=8$ (today) | 45 complex | 270 | 1.0× |
| option 2, $p=4$ | 35 real | 105 | 0.39× |
| option 2, $p=6$ | 84 real | 252 | 0.93× |
| option 4, $k=6$ | 152 real | 456 | 1.7× |
| option 4, $k=8$ | 296 real | 888 | 3.3× |
| option 3, $n=6$ | 216 real | 648 | 2.4× |
| option 3, $n=8$ | 512 real | 1536 | 5.7× |
| option 3, $n=10$ | 1000 real | 3000 | 11× |

(Option 2's count is $\binom{p+3}{3}$; option 4's is $6k^2-12k+8$; option 3's is
$n^3$ — canopy-kIndp.md, "The three costs X1 does not state", cost 3.)

**Option 3 is the worst of the three on this axis and it is not close.** A
surface representation (option 4) beats a volume one on count, and a low-order
Cartesian basis beats both. Two things keep it from being decisive:

- The comm volume that matters at scale is dominated by the **shared-cell**
  count, which is a property of the partition and the number of ranks, not of
  the kernel; the per-cell payload is a multiplier on a term that is already
  small relative to the P2P halo. Against that, the P2P halo *shrinks* by the
  factor in the previous section when the softening floor goes away. The net
  distributed traffic almost certainly falls for every route, even option 3 at
  $n=8$; that is worth measuring rather than asserting, and it is an
  add-canopy.md **T8**-shaped measurement.
- The per-level operator tables of options 3 and 4 are **replicated on every
  rank** and never communicated, so they cost per-GPU memory rather than
  bandwidth. That moves the option-3 problem from the network to the
  [memory section](#accuracy-versus-memory-cost), where it is real.

One distributed property is worth calling out as *preserved* by all three, since
losing it would have been disqualifying: the operator tables depend only on
(level, relative offset, $b$) and not on particle positions, so they survive
every `migrate` / `rebalance` / `auto_maintain`
(canopy-bbFMM.md cost 1; canopy-kIndp.md cost 1). They are a
solver-construction cost, not a per-evaluation cost, which is what keeps
add-canopy.md **R5**'s "maintain before every solve" discipline affordable.

## How much refactoring Canopy needs

A large part of the work is **common to all three routes** and is already
scoped in canopy0.md **F6(b)**/**F6(c)**. It is the bulk of the mechanical
change and it is not a differentiator:

- Generalize coefficient storage from `complex_type***` to a real `coeff_type`
  through both sweeps (`canopy/src/Canopy_UpwardSweep.hpp:63,73`;
  `canopy/src/Canopy_DownwardSweep.hpp:108,118`), and drop the
  $(P{+}1)(P{+}2)/2$ coefficient-count assumption (`:66`, `:111`) and the shared
  $A_{n,m}$ table (`canopy/src/Canopy_UpwardSweep.hpp:235`).
- Make `kernel_type` a template parameter rather than a typedef
  (`canopy/src/Canopy_Solver.hpp:112`, canopy0.md **F1**), so the blob-aware
  kernel is selectable and the existing harmonic path stays testable beside it.
- Re-key the M2L operator cache with the physical cell width. The builder's
  documented property — the operator *"depends only on (dd, ii, jj, kk); no
  physical width enters"* (`canopy/src/Canopy_DownwardSweep.hpp:1073-1077`) —
  holds because $1/r$ is homogeneous. Softening introduces the absolute length
  $\sqrt b$ and that property dies for **every** route.
- Re-derive or discard the five width-normalization conventions that exist for
  FP32 conditioning at depth (canopy0.md **F6(c)**, citing
  `canopy/src/Canopy_LaplaceKernel.hpp:226-229`, `:267-272`, `:369-372`,
  `:685-687`, `:795-799`).
- Delete the finite-difference L2P (`:851-879`).

What differs is the **new kernel struct** — the six static methods
`p2m_contribution`, `m2m_translate`, `m2l_translate`, `m2l_build_operator`,
`l2l_translate`, `l2p_evaluate` (`canopy/src/Canopy_LaplaceKernel.hpp:142-147`,
`:517`) — and the differences are about *risk*, not line count:

- **Option 2 — smallest, with the only real derivation risk.** M2L is the sole
  kernel-touching operator (canopy0.md **F6(e)**); M2M and L2L are binomial
  Taylor shifts that never see the kernel; the derivative ladder and the
  arbitrary-order recurrence are both written out in `canopy-questions.md`
  §§1-3, and the reference treecode already implements the order-2 tensors
  (`treecode.py:56-81`), which gives a component-level oracle to test against.
  The residual risk is the one the reference author flagged himself — "the only
  place that needs care is the sign and normalization convention between the
  moment definition and the $(-1)^{|q|}$ multiplier" (`canopy-questions.md`
  §108) — plus symmetric-tensor indexing, which is where hand-derived Cartesian
  FMMs habitually go wrong. Not large, but it is mathematics that has to be
  correct at every order used, and getting it subtly wrong produces a plausible
  velocity field, which is precisely the failure mode add-canopy.md exists to
  bound.
- **Option 3 — moderate, with essentially no derivation risk.** All five
  operators are rewritten, but four of them are Chebyshev interpolation
  matrices, derivable from a textbook formula and unit-testable in isolation
  against polynomial exactness, and the fifth is "call the kernel at $n^3$ pairs
  of nodes" — the same three lines P2P already runs
  (`canopy/src/Canopy_P2P.hpp:799-803`). There is no basis identity to get
  wrong, no sign convention to reconcile, and no order-dependent algebra. The
  one substantial new piece of *engineering* is the SVD compression of the
  operator tables, which needs a dense-linear-algebra dependency Canopy does not
  currently have (its `CMakeLists.txt` finds Kokkos, Cabana and GTest, and no
  BLAS/LAPACK) — but it runs once, on host, at construction.
- **Option 4 — largest, and the only one carrying an unresolved theory
  question.** All five operators, *plus* a regularized pseudo-inverse
  precomputation per level (a mandatory LAPACK dependency, not an optional one),
  *plus* the FFT machinery its performance case rests on, *plus* — before any of
  that — the off-surface probe canopy-kIndp.md says must be run first, because
  the property that makes KIFMM trustworthy (a residual on one check surface
  bounds the error everywhere outside it) is a consequence of harmonicity and is
  lost at $b>0$. A route that has to replace its error argument with a
  measurement campaign before the first operator is written is the wrong answer
  to a question whose entire subject is a bounded error.

## Accuracy versus memory cost

This is the section that carries the recommendation, and it has two independent
parts: how fast each basis converges per degree of freedom, and what the
per-key operator tables cost once softening has destroyed the level-independent
cache.

### Convergence per DOF

- **Option 2 converges too slowly to reach Beatnik's band.** A Cartesian Taylor
  expansion truncated at order $p$ has relative error $\sim (c\,w/R)^{p+1}$,
  with $w$ the cell half-width, $R$ the centre separation and $c$ a geometric
  factor between 1 and $\sqrt3$ (worst case at a box corner). At standard
  admissibility $R/w \approx 3$ the ratio is between $1/3$ and $0.58$, so each
  additional order buys between 0.24 and 0.48 decades while the DOF count grows
  as $\binom{p+3}{3} \sim p^3/6$. Reaching $10^{-6}$ therefore wants $p \approx
  11$–$24$, i.e. **364 to 2925 DOF per cell** — and an M2L needing
  $\partial^\alpha\varphi$ to order $2p$. That is canopy0.md **F6(d)**'s
  conclusion arrived at from the convergence rate instead of the derivative
  count, and the two agree: *the basis that admits softening trivially is the
  basis that does not scale to high accuracy.* The reference author's "order
  $p=2$ is enough" (`canopy-questions.md` §5) is not in tension with this — it
  is a statement about beating the **regularization** problem, which $p=2$ does,
  and X1's closing paragraph already separates the two claims.
- **Option 3 converges geometrically in $n$, and softening helps.** Chebyshev
  interpolation error falls like $\rho^{-n}$ with $\rho$ set by the distance from
  the interpolation region to the nearest kernel singularity in the mapped
  complex plane; admissibility guarantees $\rho > 1$, and $b>0$ moves the
  singularity off the real axis entirely, which can only increase $\rho$
  (canopy-bbFMM.md, "The math it is rooted in"). Published behavior puts
  $n \approx 4$–5 near $10^{-4}$ and $n \approx 8$ near $10^{-8}$
  (canopy-bbFMM.md cost 2, carried from Fong & Darve — **not independently
  verified here**). So $10^{-8}$ costs 512 DOF per cell, against option 2's
  ~1000-plus for $10^{-6}$: option 3 is the more DOF-efficient basis in exactly
  the band that matters, despite being the *less* efficient one at $10^{-2}$.
- **Option 4 is the most DOF-efficient of the three and still the wrong choice
  here**, because its ceiling is not set by the representation but by the
  conditioning of the check-to-equivalent inversion, whose regularization
  parameter floors the achievable error in the $10^{-6}$–$10^{-9}$ band
  (canopy-kIndp.md cost 2). Option 3's interpolation has no inverse problem in
  it at all, and its SVD compression is a speed optimization that can be dialed
  back to buy accuracy. For a $10^{-10}$ target this inverts the usual ordering
  of the two black-box methods.

### Memory, and the number that must be measured first

Per-key operator storage, at the shapes above:

| route | per-key operator | bytes per key |
| --- | --- | --- |
| harmonic, $P=8$ (today) | $45\times81$ complex | 58 KB |
| option 2, $p=4$ | $35\times35$ real | 9.8 KB |
| option 2, $p=6$ | $84\times84$ real | 56 KB |
| option 3, $n=4$ | $3\times 64^2$ real | 98 KB |
| option 3, $n=6$ | $3\times 216^2$ real | 1.1 MB |
| option 3, $n=8$ | $3\times 512^2$ real | 6.3 MB |
| option 3, $n=10$ | $3\times 1000^2$ real | 24 MB |
| option 4, $k=8$, FFT form | $3\times(2k)^3$ complex | 196 KB |

**The multiplier on all of these is unmeasured, and it is the first thing to
find out.** Canopy's table is shaped `(Nt, Ns, n_unique_ops)`
(`canopy/src/Canopy_DownwardSweep.hpp:1085`), where `n_unique_ops` is the number
of *realized* $(dd, ii, jj, kk)$ keys — capped at
`M2L_OP_COUNT_CAP = 32768` (`:304`), with an overflow valve routing excess pairs
to a fallback path (`:1029-1047`). The textbook V-list figure of 316 offsets
applies to a uniform tree with $dd = 0$; an adaptive tree on a thin bubble
surface realizes depth-mismatched keys too, and nobody has counted them. Then
softening multiplies the count again by the number of occupied depths, since the
key must carry the physical width.

Worked, with $L$ occupied depths and $N_{\rm keys}$ realized keys:

```
option 2, p=4 :  9.8 KB × N_keys × L    N_keys=2000, L=6  →   118 MB
option 3, n=6 :  1.1 MB × N_keys × L    N_keys=2000, L=6  →    13 GB
option 3, n=8 :  6.3 MB × N_keys × L    N_keys=2000, L=6  →    76 GB
```

Uncompressed option 3 is therefore **not representable** above about $n=4$ on
any realistic GPU, and this is a stronger statement than canopy-bbFMM.md's,
which assumed the 316-offset figure. The Fong & Darve compressed form is what
fixes it, and it must be the *shared-basis* form — one $n^3 \times r$ pair of
bases per level, with a small $r \times r$ core per key — rather than a per-key
truncated SVD:

```
shared basis + per-key core, n=8, r=100:
  bases  : 2 × 512 × 100 × 3 × 8 B × L            ≈ 15 MB
  cores  : 100² × 3 × 8 B × N_keys × L            ≈ 2.9 GB   (N_keys=2000, L=6)
  at r=50                                          ≈ 720 MB
```

That is affordable, and it also fixes the flop side: an uncompressed $n=8$ M2L
is $n^6 = 2.6\times10^8$ multiply-adds per pair per component against today's
$45\times81 = 3645$, while the compressed form is $r^2$. **SVD compression is
therefore load-bearing for both memory and time, and the recommendation is
conditional on it.** Option 2's tables need no compression at any order it can
usefully reach; option 4's FFT form is similarly memory-light, which is the
strongest thing in its favor and is not enough to overcome its accuracy floor
and its broken error argument.

One further note, since Canopy's normalization conventions exist for it: none of
these accuracy figures survive single precision. The $10^{-8}$-and-below
discussion presupposes `scalar_type = double` throughout the sweep, and the
width-normalization rework (canopy0.md **F6(c)**) should be done with that
assumption stated rather than inherited.

## What would change this recommendation

Three measurements, in order of how much they would move it.

1. **add-canopy.md T5's $\tau_A$ requirement.** If Beatnik turns out to need only
   $10^{-4}$ per evaluation — because claim B's trajectory rung, derived from
   the measured $10^4$ amplification, tolerates it — then option 2 is the better
   engineering answer: 118 MB of tables instead of 720 MB, less network traffic
   than Canopy sends today, a smaller device kernel, and the derivation already
   supplied and already implemented at order 2 in the reference. **Option 2 is
   not a worse method; it is a method with a ceiling, and whether the ceiling
   binds is a measurement Beatnik has not made.** T5 is the gate, and X1's
   discriminator — passing with `near_softening_factor = 0` — is cleared by
   either route.
2. **`n_unique_ops` on a real milestone-0 tree.** If it is near 316 rather than
   thousands, uncompressed option 3 at $n=6$ becomes comfortable and the SVD
   condition relaxes to an optimization. If it is at the 32768 cap, option 3
   needs the compressed form at *every* order and option 2 gains substantially.
   This is a one-line instrumentation of the table build
   (`canopy/src/Canopy_DownwardSweep.hpp:1070`) and needs no new physics.
3. **Whether the $\epsilon_{ilm}$ contraction moves inside M2L.** Doing so cuts
   the descending local expansions from nine to three, which is a real win at
   option 3's DOF counts — but it is an interface change, and Beatnik's adapter
   currently owns that contraction (add-canopy.md "Approach"; canopy-bbFMM.md,
   "Does it need only one solve?"). It is a Canopy-side decision. If it is taken,
   add-canopy.md **T2** is the only Beatnik file that moves, and the $\epsilon$
   sign convention must be stated at the new boundary.

Beatnik does not require any particular route. It requires whatever $\tau_A$
claim A asserts, and add-canopy.md **T5** is what states that number.
