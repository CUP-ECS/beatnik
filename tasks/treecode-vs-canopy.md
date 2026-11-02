# Does the reference treecode supply the softened far-field math Canopy lacks?

**Status:** FINDINGS ONLY. Nothing implemented, nothing decided. Written to
answer a question asked against [`treecode.md`](treecode.md) and
[`canopy.md`](canopy.md), and to close `treecode.md`'s open question 3 ("Does
Canopy's FMM already subsume this?"), which was deliberately left unexamined
because `../canopy` had not been opened.

**Sources read.** `~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`
(all 138 lines) and, on the Canopy side,
`canopy/src/Canopy_LaplaceKernel.hpp` (all 887 lines),
`canopy/src/Canopy_Solver.hpp`, `canopy/src/Canopy_P2P.hpp`,
`canopy/src/Canopy_CommunicationPlan.hpp`, plus the kernel-coupling points in
`Canopy_UpwardSweep.hpp` and `Canopy_DownwardSweep.hpp`. No code was changed and
nothing was built or run; every accuracy number quoted here is carried over from
[`treecode.md` §1](treecode.md) or [`canopy.md` F1/F4](canopy.md).

---

## 0. The short answer

**The premise needs one correction, and then the answer is "partly — and the
part it supplies is the part that matters".**

The correction: **M2M, M2L, L2L and L2P are not missing from Canopy's Laplace
kernel.** All five operators are implemented, in the solid-harmonic basis, with
Greengard theorem citations on each — `p2m_contribution`
(`Canopy_LaplaceKernel.hpp:230`), `m2m_translate` (`:273`, Thm 5.22),
`m2l_translate` (`:373`, Thm 5.23), `l2l_translate` (`:688`, Thm 5.26),
`l2p_evaluate` (`:800`), plus a precomputed-operator M2L path
(`m2l_build_operator`, `:516`). What is missing is not the operators; it is
**softening in them**. `canopy.md` F1 is the statement of that gap: the far field
expands the bare $1/r$, and softening lives only in P2P
(`Canopy_P2P.hpp:799-803`, `883-896`), kept relevant by a floor in the MAC
(`Canopy_CommunicationPlan.hpp:347-361`).

So the real question is: **does `treecode.py` supply the mathematics for a
*softened* M2M/M2L/L2L/L2P?** Answer:

| Operator | Does the treecode contain it? |
| --- | --- |
| **P2M** (softened) | **Yes, complete.** `treecode.py:33-35`. And it is kernel-independent, so it needs no softening at all. |
| **M2P** (softened) | **Yes, complete to order 2.** `_expansion_batch`, `treecode.py:56-81`. This is the *only* kernel-dependent operator in a Cartesian-Taylor tree method, and it is the seed of M2L. |
| **M2M** | **No — and it does not need to.** The Python recomputes moments per level (`treecode.py:30-35`) rather than translating them. But M2M in a Cartesian-Taylor basis is a **binomial shift of moments, independent of the kernel** — no softening enters, so there is nothing to derive. |
| **M2L** | **No object, but the derivative pattern.** No local expansion exists anywhere in the file; the traversal stack only ever pushes children (`treecode.py:118-137`). What it does give is $\partial^\alpha K$ for $|\alpha| \le 2$ in closed form, and a Cartesian M2L *is* $\partial^{\alpha+\beta}K(R)$. So it gives the first two rungs of the ladder, not the ladder. |
| **L2L** | **No — and it does not need to.** L2L in a Cartesian-Taylor basis is a Taylor shift of derivative coefficients: also kernel-independent. |
| **L2P** | **No — and it does not need to.** Truncated Taylor evaluation; kernel-independent. |

The structural fact underneath that table: **in a Cartesian Taylor tree method,
M2L (equivalently M2P) is the only operator that knows what the kernel is.**
M2M, L2L and L2P are combinatorics on moments and derivative coefficients. So
the treecode supplies exactly the kernel-dependent piece, and the pieces it
omits are the ones that need no softening-specific derivation.

**But it is not transplantable into Canopy's kernel as written**, for three
reasons in §2, and the cheap way to use it is not option (b) of `canopy.md` C1
— it is a fourth option that C1 does not list. That is §3.

---

## 1. What the treecode's expansion actually is, precisely

`_expansion_batch(rv, G, D, Q, blob, order)` with `rv = x_t - c` and
`d = y_s - c` computes the order-2 Taylor expansion of the vector kernel in the
**source** offset:

$$
K(x_t - y_s) = K(rv - d) \approx K(rv) \;-\; d_a\,\partial_a K(rv)
  \;+\; \tfrac12 d_a d_b\, \partial_a \partial_b K(rv),
$$

contracted against the moments $G = \sum g$, $D = \sum d\otimes g$,
$Q = \sum d\otimes d\otimes g$ (`treecode.py:33-35`), with the cross product
taken outside the contraction. Written in terms of the softened potential
$\phi_b(r) = (r^2+b)^{-1/2}$:

- $K = -\nabla\phi_b$ (`treecode.py:58`: `rv / w^{1.5}`),
- `dK` (`:61`) $= -\nabla\nabla\phi_b = I/w^{3/2} - 3\,rv\otimes rv/w^{5/2}$,
- `ddK` (`:66-75`) $= -\nabla^3\phi_b$, the three-index $-3(I\otimes rv)_{\rm sym}/w^{5/2} + 15\,rv^{\otimes3}/w^{7/2}$,

with $w = |rv|^2 + b$ throughout — which is the "blob-aware" property
`treecode.md` flags as load-bearing. **So the concrete new mathematics the file
contains is: closed-form second and third derivatives of the Plummer potential.**
Nothing more, and that is genuinely the hard-to-get-wrong part of the physics.

Two incidental notes that matter downstream:

- The cross product is linear in the source strength, so it commutes with the
  expansion. This confirms `canopy.md` F2 from the other side: a Canopy-side
  implementation should produce the $3\times3$ tensor and let the caller
  contract, exactly as F2 prescribes — it should *not* bake the cross product
  into the kernel the way the Python does.
- The treecode's acceptance radius is the **exact** $\max_j |y_j - c|$
  (`treecode.py:31`), where Canopy's MAC uses the geometric
  $\sqrt3\,(h_A+h_B)$ (`Canopy_CommunicationPlan.hpp:346`). `canopy.md` F4
  already notes that $\sqrt3 h$ *overstates* the source extent for a cell
  straddling a sheet. The exact radius is tighter, cheap to compute in the
  upward sweep, and **is an accuracy-per-cost win on a two-dimensional
  distribution that is independent of every kernel question in this document.**
  See §4.

---

## 2. Why it cannot be dropped into `Canopy_LaplaceKernel.hpp`

Three obstructions, in increasing order of how much work they represent.

### 2a. Storage type and basis (mechanical, but wide)

The sweeps *are* templated on `KernelType`
(`UpwardSweep<MemorySpace, ExecutionSpace, KernelType>`,
`Canopy_Solver.hpp:116-119`), so a new kernel struct is pluggable in principle —
but they hard-assume this kernel's storage: `complex_type***` views
(`Canopy_UpwardSweep.hpp:63,73`; `Canopy_DownwardSweep.hpp:108,118`),
`num_coeffs_per_cell = (P+1)(P+2)/2` (`:66`, `:111`), and the shared
$A_{n,m}$ table built to $2P$ (`Canopy_UpwardSweep.hpp:235`). A
Cartesian-Taylor kernel wants **real** symmetric-tensor storage of size
$(p{+}1)(p{+}2)(p{+}3)/6$ and no A-table. It fits only by generalizing a
`coeff_type` typedef through both sweeps, or by wasting every imaginary half.
`Canopy_Solver.hpp:112` also fixes `kernel_type` as a typedef rather than a
template parameter, which `canopy.md` F1 already records.

### 2b. Harmonicity — the reason no substitution exists

Canopy's M2M/M2L/L2L are the solid-harmonic addition theorems (Greengard 5.22,
5.23, 5.26). Those are valid **because $1/r$ is harmonic**. The softened
potential is not:

$$
\nabla^2 (r^2+b)^{-1/2} \;=\; -\,\frac{3b}{(r^2+b)^{5/2}} \;\ne\; 0 .
$$

So there is no softened coefficient one can substitute into `m2l_translate` to
make it evaluate the softened kernel. `canopy.md` C1 step 3(b) already states
this ("$1/\sqrt{r^2+\varepsilon^2}$ is not harmonic, so the existing
solid-harmonic M2L does not apply unchanged") and asks, under **Additional
information needed**, "whether option (b) is achievable without replacing the
expansion basis."

**This document answers that question: no.** Not by reading the treecode, and
not in the solid-harmonic basis at all. The treecode does not solve the problem
— it *evades* it, by using a basis (Cartesian Taylor) that requires only
smoothness of the kernel, never harmonicity. Any softened far field in Canopy
therefore means a second expansion basis alongside the existing one, not a
patch to it.

### 2c. Loss of scale invariance — the expensive obstruction

This is the finding least visible from the treecode side and the one most likely
to be underestimated.

Canopy's whole numerical conditioning strategy rests on $1/r$ being
**homogeneous of degree $-1$**. Every operator is "scale-normalized" against
cell half-width on that basis: P2M produces $\bar M = M/w^{n+1}$
(`Canopy_LaplaceKernel.hpp:226-229`), M2M applies $(w_c/w_p)^{j+1}$ (`:267-272`),
M2L expands $\rho^{-(n+j+1)}$ as $(w_s/\rho)^{n+1}(w_t/\rho)^j$
(`:369-372`), L2L applies $(w_c/w_p)^j$ (`:685-687`), L2P consumes
$\bar L = L\,w^n$ (`:795-799`). The stated purpose is FP32 conditioning at
depth.

The sharper consequence is the **precomputed M2L operator cache**. Operators are
keyed on $(dd, ii, jj, kk)$ — a depth difference and an integer offset in
half-widths — and `Canopy_DownwardSweep.hpp:1075` states the property outright:
the operator *"depends only on (dd, ii, jj, kk); no physical width enters"*. That
is true precisely because the kernel has no absolute length scale.

**A softened kernel has one: $\sqrt b$.** $(r^2+b)^{-1/2}$ is not
homogeneous, so:

- the operator cache key must additionally carry the physical cell width (or
  $w/\sqrt b$), which multiplies the distinct-key count by the number of
  occupied depths and degrades the reuse that the path exists to buy;
- every one of the five width-normalization conventions above has to be
  re-derived, because they are not merely rescalings any more.

So the far-field softening work in `canopy.md` C1 option (b) is not "add $b$
to a few denominators, as the treecode does". It is a second expansion basis,
new real-valued storage through both sweeps, and a rebuilt M2L operator-cache
keying scheme. That is a large task, and the estimate should say so.

### 2d. And a note on why the order-2 seed is not the whole ladder

For a Cartesian Taylor FMM truncated at multipole order $p_M$ and local
order $p_L$, M2L needs $\partial^\alpha\phi_b$ for
$|\alpha| \le p_M+p_L+1$. The treecode supplies $|\alpha| \le 3$. To
match Canopy's default $P=8$ one would need up to 17th derivatives and
$\sim\!(19)(20)(21)/6 \approx 1330$ tensor slots per pair — the
$O(p^3)$-vs-$O(p^2)$ growth that is the standard reason solid harmonics
win at high order. **A softened Cartesian FMM is therefore only attractive at low
order**, i.e. in the $10^{-3}$-accuracy regime `treecode.md` §1 measures, not
in the $10^{-6}$ regime `canopy.md` F1 wants. The two facts have to be held
together: the basis that admits softening is the basis that does not scale to
high accuracy.

The literature pointer `canopy.md` C1 asks for does exist, and is a
low-order Cartesian-Taylor FMM with softening: **Dehnen's `falcON`**
(W. Dehnen, *ApJ* **536**, L39, 2000; *JCP* **179**, 27, 2002) is a Cartesian
Taylor-expansion FMM built for softened gravity with a full M2L, and
Warren & Salmon's hashed oct-tree work is the Cartesian-multipole precedent.
**Neither was read for this document** — they are named as the check C1 step 3
should run, not as verified results.

---

## 3. Is a new kernel option to Canopy feasible?

Yes, and there is a much cheaper version than the one `canopy.md` C1 currently
contemplates. Four options, ranked by lift.

### Option A — a softened **M2P** mode over Canopy's *existing* interaction list

**This is the cheap one, and it is not in C1's list of options.**

The observation: Canopy's dual-tree traversal already produces
**(target cell, source cell)** accepted pairs
(`Canopy_CommunicationPlan.hpp:549-665`) and already has a working softened P2P
for the rejected ones. A "treecode mode" does not need a new traversal, a new
tree, a new partitioner or a new communication plan. It needs to replace *one
step*: where the downward sweep currently does M2L into a local expansion and
then L2L/L2P down to particles, evaluate the source cell's multipole **directly
at each particle in the target cell** — M2P — and skip L2L and L2P entirely.

What that requires:

| Piece | Lift |
| --- | --- |
| Cartesian moments in the upward sweep (P2M) | Small. Kernel-independent; `treecode.py:33-35` is the formula. |
| M2M for those moments | Small. Binomial shift, kernel-independent — the one operator the Python skips and a real implementation should have (`treecode.md` §3 says the same). |
| Softened M2P | **Small, and transcribed.** `treecode.py:56-81`, $\sim\!80$ lines of explicit index loops. This is `treecode.md`'s own estimate for the same code. |
| M2L / L2L / L2P | **Deleted from the path.** Not implemented, not needed. |
| Storage | Real, $(p{+}1)(p{+}2)(p{+}3)/6 \times$ NComps per cell. Still needs the `coeff_type` generalization of §2a, but *only* in the upward sweep and the new M2P driver — the local-expansion machinery is untouched. |
| Operator cache | **Not applicable.** M2P has no per-offset operator to cache, so §2c's loss of scale invariance costs nothing here. |

What it buys, which is exactly what `canopy.md` wants:

- **The far field carries the softening.** `near_softening_factor` becomes
  unnecessary rather than load-bearing, and F1's systematic
  $\tfrac32\varepsilon^2/R^2$ gradient bias — the one that does not shrink
  with order and makes $10^{-6}$ unreachable at any cost — **disappears**.
  The remaining error is ordinary truncation error, which *does* respond to
  $\theta$ and order.
- Analytic gradients (see §4), replacing the finite-difference L2P.
- Reference fidelity: it is the reference's own algorithm and its own
  $\theta$/order/`ncrit` knobs, which `treecode.md` §3 notes are already
  parsed by Beatnik and currently warned-and-mapped.

What it costs:

- **Accuracy ceiling $\sim\!10^{-3}$** at $\theta=0.3$, order 2
  (`treecode.md` §1), with the error a plateau in $N$. Higher order is
  available but pays §2d's $O(p^3)$.
- **Complexity goes $O(N)\to O(N\log N)$** and per-target work rises: every
  particle in a target cell re-evaluates every accepted source multipole,
  instead of the cell paying M2L once and amortizing via L2L/L2P. On a leaf of
  `ncrit` particles that is an `ncrit`-fold increase in far-field arithmetic.
  **This is the real cost of option A and it is a measurement, not an estimate.**
- Does nothing for `canopy.md` C2/C3/C4 — the ordering, cheap-refresh and
  reproducibility gaps are in the tree and partitioner, not the kernel.

### Option B — a full softened Cartesian-Taylor FMM (C1 option (b), done properly)

Everything in §2b/2c/2d. Second basis, real storage through both sweeps, M2L
derivative tensors to $p_M+p_L+1$, and a re-keyed operator cache. Retains
$O(N)$ and the L2L/L2P amortization; still capped at low order by §2d.
**Substantially larger than C1's current framing suggests** — and worth doing
only after option A has measured whether a softening-consistent far field
actually buys the accuracy the consumer needs.

### Option C — status quo

Keep the bare far field and `near_softening_factor`, accept F1's
$\tfrac32/\text{factor}^2$ gradient bias, and state the achievable tolerance
at the $10^{-2}$–$10^{-3}$ level. This is C1 option (a).

**Note the collision:** option C's honest accuracy claim and option A's ceiling
are *the same number*, $\sim\!10^{-3}$. That reframes the choice usefully —
option A is not "trade accuracy for fidelity", it is "reach the same accuracy
with an error that is controllable by $\theta$ and order instead of one that
is a fixed bias, and get analytic gradients and reference-faithful knobs along
the way."

### Option D — do nothing in Canopy; port the treecode into Beatnik

`treecode.md` §3's plan. Independent of everything above and does not close
`canopy.md` C1.

---

## 4. Two findings that are independent of any of this

Both fall out of the reading and are actionable on their own.

**4a. Canopy's L2P gradient is a central finite difference, not analytic.**
`Canopy_LaplaceKernel.hpp:851-879`: six extra potential evaluations at
$h = 10^{-5} w_{\rm self}$, with an in-code `TODO: replace with analytical
derivatives` and a comment recording that a previously *fixed* step was the root
cause of a premature full-rollup NaN. Consequences:

- **Some of the gradient error `canopy.md` C1 and C5 are about to measure is FD
  error, not softening bias or truncation.** The relative FD error at the
  roundoff/truncation optimum is $\sim\!10^{-10}$–$10^{-11}$, so it does
  not threaten a $10^{-3}$ budget — but it is a **third** plateau in the
  $P$-scan that C1 step 1 and risk **R1** are built around, and R1's
  "truncation falls, bias plateaus" discriminator has to account for it. Worth
  stating in C1 before the scan is read.
- Both options A and B produce analytic gradients as a side effect: the
  treecode's kernel *is* the gradient of $\phi_b$, so the derivative tensors
  are what the expansion evaluates directly.

**4b. The exact node radius is a free accuracy win on a sheet.** `treecode.py:31`
uses $\max_j |y_j - c|$; `Canopy_CommunicationPlan.hpp:346` uses
$\sqrt3(h_A+h_B)$. `canopy.md` F4 already identifies that the geometric bound
overstates the source extent for a cell straddling a two-dimensional
distribution — conservative and therefore safe, but it converts pairs to P2P
that a tighter bound would accept. Computing the exact radius per cell in the
upward sweep and using it in `mac_satisfied` is a small, kernel-independent
change that reduces near-field cost at fixed accuracy on exactly the
distribution `canopy.md` C5 exists to measure. **It should be measured as part of
C5 rather than assumed**, since it also interacts with the near-softening floor.

---

## 5. Answers to the two questions as asked

**"Does the treecode kernel contain the M2M/M2L/L2L/L2P mathematics missing from
Canopy's Laplace kernel?"** Those operators are not missing — they exist in the
solid-harmonic basis. What is missing is softening in them, and the treecode
contains: the softened M2P/M2L *seed* (closed-form $\partial^2\phi_b$ and
$\partial^3\phi_b$, complete and correct, order 2 only), the Cartesian P2M,
and nothing else — no local expansion, no M2L object, no M2M, no L2L, no L2P. It
is not a deficiency of the file: in the Cartesian-Taylor basis those four are
kernel-independent combinatorics, so the treecode supplies precisely the one
operator family that has to know about $b$. It is a **basis change**, not a
patch, and it does not transplant into the existing kernel (§2).

**"Would adding a new kernel option to Canopy be feasible?"** Yes. Option A
above — a softened M2P mode reusing Canopy's existing tree, partition,
interaction list and P2P, dropping M2L/L2L/L2P from the path — is the feasible
one, and it is cheaper than `canopy.md` C1 option (b) by a wide margin while
delivering the property C1 exists to get: a far field that carries the softening,
with no systematic bias floor. It costs $O(N)\to O(N\log N)$ and an
`ncrit`-fold rise in per-target far-field arithmetic, and it caps accuracy at
$\sim\!10^{-3}$ — which is the same number option C's honest claim lands on.
Whether that trade is worth it turns on a measurement nobody has: the actual
cost of M2P-per-particle against M2L+L2L+L2P-per-cell on this problem.

**And `treecode.md` open question 3 is now answered: no, Canopy does not subsume
the treecode.** `grep -i "barnes\|treecode\|m2p"` over `canopy/src/` returns
nothing; there is no opening-angle mode, no monopole mode, and no
multipole-at-a-point evaluation anywhere. `mac_theta` is a spherical MAC used to
*build M2L pairs* (`Canopy_CommunicationPlan.hpp:338-352`), not a Barnes–Hut
opening angle. A treecode is therefore **not** a configuration of T3a today. It
could become one, via option A — and that, rather than a separate
`BRSolverTreecode`, may be the better home for `treecode.md` §3's work, since it
inherits Canopy's distributed tree instead of re-deciding
`treecode.md`'s open question 2 (ring vs. allgather).

---

## 6. What this document does not settle

1. **The per-target cost of M2P vs. the per-cell cost of M2L+L2L+L2P** on this
   problem, on MI300A. Option A's viability rests entirely on it, and it needs a
   build and a run. It is a strictly easier measurement than `treecode.md`'s
   open question 1 (the Kokkos direct-sum crossover) and partly subsumes it.
2. **Whether Dehnen 2000/2002 in fact gives a usable softened Cartesian M2L**,
   and to what order. Named from memory; not read. This is `canopy.md` C1's
   "Additional information needed" and remains open.
3. **How much of the $10^{-3}$-vs-$10^{-6}$ gap the consumer actually
   needs.** Every option above lands at $10^{-3}$ except option B at
   moderate order. `canopy.md` F1 asserts $10^{-6}$ is unreachable with a
   bare far field; this document adds that it is also out of reach for a
   *low-order* softened one. If $10^{-6}$ is a hard requirement, the
   remaining path is option B at an order high enough to pay §2d's
   $O(p^3)$ — and that should be costed before anything else here is built.