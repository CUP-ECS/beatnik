# Canopy as Beatnik's far-field Birkhoff-Rott solver

**Status:** IN PROGRESS — **T1**, **T2**, **T3** and **T4** are **DONE**; every
task from **T5**
on is NOT STARTED. The findings and the measured numbers are complete, and no upstream work
gates the sequence: Canopy's derivative ladder is validated at the production
order. Three constraints bind inside **T5** and none of them gates it: no scan
point above $p=3$ may be adopted as the production order (**R11**), the scan
runs in one launch against one state, and step 7's sweep is budgeted against
`pdebug`'s one-hour cap.

## Problem

`--br-approximation fmm` **runs** after **T3**, and one virtual is still a
stub. `BRSolverFMM::computeInterfaceVelocity` is live
([src/Beatnik_BRSolverFMM.hpp:140](../../src/Beatnik_BRSolverFMM.hpp#L140)) and
only `::computeSurfaceRieszScalar` is still `BEATNIK_NOT_IMPLEMENTED`
([src/Beatnik_BRSolverFMM.hpp:198](../../src/Beatnik_BRSolverFMM.hpp#L198)) —
**T7** alone owns what remains — while the adapter beneath them is real and
Canopy-backed
([src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp)).
The remaining throw is therefore in the BR solver, not in the far field, and it
is reachable only under `--bernoulli-scalar-mode surface-riesz`.
`fmm` is nevertheless the **default**
([src/Beatnik_Params.hpp:107](../../src/Beatnik_Params.hpp#L107)), so the default
far-field path is the one that throws. The only working Birkhoff-Rott evaluator
is `BRSolverDirect`, an $O(N_tN_s)$ ring-exchanged pairwise sum
([src/Beatnik_BRSolverDirect.hpp:105-166](../../src/Beatnik_BRSolverDirect.hpp#L105-L166))
called three times per accepted timestep
([src/Beatnik_ZModelSolver.hpp:223](../../src/Beatnik_ZModelSolver.hpp#L223)) and a
fourth time under `--bernoulli-scalar-mode surface-riesz`
([:255](../../src/Beatnik_ZModelSolver.hpp#L255)).

The end state: `--br-approximation fmm` runs the same physics through Canopy's
fast multipole method, in sub-$N^2$ time, with a **stated, measured** bound on
how far its velocity departs from `BRSolverDirect`'s on the same state; and two
new `milestone`-tier members demonstrate that bound on the real milestone-0
problem at all 81 checkpointed steps of a 2000-step run, at both subdivision
levels.

### The far field Beatnik asks Canopy for

Canopy's far field is a **basis**, selected by a template parameter on `Solver`
and `createSolver`; the contract every basis satisfies is specified in
[abstract-solver-backend.md](../../../canopy/tasks/abstract-solver-backend.md). Two bases ship:

| Basis | Expands | Coefficients | Accuracy knob |
| --- | --- | --- | --- |
| `LaplaceKernel` (solid-harmonic, the **default**) | bare $1/r$ | $(P{+}1)(P{+}2)/2$ complex per cell | $P$, against a near-field softening floor |
| `CartesianTaylorBasis` | the Plummer-softened $\varphi(r)=(r^2{+}b)^{-1/2}$ directly | $\binom{p+3}{3}$ real per cell | $p$, a pure truncation |

**Beatnik selects `CartesianTaylorBasis`, and the geometry is why.** Beatnik's
kernel carries $b=\varepsilon^2=6.25\times10^{-4}$ with $\varepsilon=0.025$ on a
bubble of radius $0.25$, so $\varepsilon$ is **5% of the bubble diameter** and
the softening perturbs the kernel by $\tfrac32\varepsilon^2/r^2=3.8\times10^{-3}$
even at maximum separation. The softening is not a short-range regularization on
this problem — it is a length comparable to the geometry — so a far field that
expands the bare kernel and pushes the softened pairs into the direct sum is
being asked to treat most of the domain as near field. `CartesianTaylorBasis`
expands the softened kernel itself, at every order, and takes
`near_softening_factor = 0`.

That configuration is demonstrated rather than projected. Canopy drives a
four-solve pipeline on this basis at ranks 1-6 at `near_softening_factor = 0`
with an explicit `softening = 0.025` — Beatnik's own $\varepsilon$ — and
reproduces a direct softened sum
([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
T4). The measured figures are in [Current state](#canopy) and they are what sets
this document's production order.

The two bases are not equally validated for this problem and the selection is
not a preference: **T4** and **T5** measure both on Beatnik's own geometry, and
the solid-harmonic basis with the floor disabled is **T4**'s negative case —
the configuration Canopy measured missing the same comparison by three orders on
the potential and by $5.2\times10^{-2}$ on the gradient.

### The fidelity target, and where it comes from

**$\tau_A \le 10^{-3}$ max relative velocity error against `BRSolverDirect` on
the same state.** That is the accuracy of the reference implementation's own
default far field: the Python's Barnes-Hut treecode at its defaults
($\theta=0.3$, order 2, `ncrit` 64) agrees with the direct sum to
$\sim\!10^{-3}$ relative velocity, measured across mesh sizes and confirmed
against the reference's own README ([treecode.md](../treecode.md) §1) —
$1.6\times10^{-3}$ at 642 vertices and $4.8\times10^{-4}$ at 2562, which are
exactly milestone-0's two levels. The reference runs that path by default
(`--br-approximation treecode`), so a Beatnik FMM at $\tau_A\le10^{-3}$ is not a
degraded mode: it is the fidelity the physics this port reproduces was produced
at.

The comparison is closer than a shared tolerance. The reference treecode expands
the **same** softened kernel in the **same** Cartesian-Taylor basis, to order 2 —
its `K`, `dK`, `ddK` tensors in `_expansion_batch` are the order-0/1/2 derivative
tensors of $\varphi$, and Canopy's basis is unit-tested against exactly that
contraction
([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md) T3,
whose $|p|=1$ check asserts the local coefficients against the reference's
$K/dK/ddK$). What differs is the traversal, the acceptance criterion, and — the
one that costs an order — whether there is a target-side expansion at all.

#### Three knobs, three different ways of not transferring

`FmmParams` copies the reference's three defaults exactly: `mac_theta` 0.3,
`order` 2, `ncrit` 64 (`treecode.py:101-103`). None of the three carries the
reference's behaviour across unchanged, and each fails differently.

`treecode.py` is in the reference implementation's repository, not in this one
or in Canopy's: `~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/`. No
task depends on having it — [treecode.md](../treecode.md) is the in-tree record
of what it does — but the line numbers below are against that file.

**`mac_theta` is a different predicate.** The reference accepts iff
`node.radius < theta * s`, with $s$ the distance from the **target point** to the
source node center and `node.radius` the source node's **actual particle-cloud
radius** (`treecode.py:31`, `:124`) — no target box, and the contents' extent
rather than the cell's. Canopy accepts iff $R\theta>\sqrt3\,(h_A+h_B)$ with $h$ a
geometric half-width (`CommunicationPlan::mac_satisfied`,
`canopy/src/Canopy_CommunicationPlan.hpp:338-352`). No $\theta$ makes the two the
same set. Beatnik keeps $\theta=0.3$ and accepts Canopy's predicate;
**Deliberate deviations** records why reshaping it is not attempted.

**`order` is the same quantity and not the same accuracy.** Both now denote the
Cartesian Taylor truncation order $p$, which `mac_theta` never did. But the
reference is a treecode: `_expansion_batch` is evaluated at
`rv = target - node.center`, the offset to the **actual target point**
(`treecode.py:121-126`), so its only truncation is on the source side. An FMM
accumulates the far field into a local expansion about the **target box center**
and evaluates that instead, which truncates a second time. Canopy's L2P gradient
is analytic,

$$
\partial_a u(x) \;=\; \sum_p \frac{a^{p-e_a}}{(p-e_a)!}\,\ell_p^A ,
\qquad a = x - c_A ,
$$

and differentiating a degree-$p$ polynomial leaves a degree-$(p{-}1)$ one. So at
order $p$ the potential is accurate to order $p$ and **the gradient only to
$p-1$**. Beatnik reads only the gradient — both contractions in
[The physics maps onto one Canopy solve](#the-physics-maps-onto-one-canopy-solve-exactly)
are contractions of $T_{cj}$, and the potential is never used — so **the reference's
order-2 velocity is Canopy's order-3 gradient**, and Beatnik's production order
is **3**.

That is measured, not inferred. The fingerprint of a target-side loss is the
ratio of the absolute errors, $1/W$ with $W$ the target cell half-width, where a
source-side loss would give $1/R$: Canopy measured $29.47$ against $1/W=29.88$
and $218.5$ against $223.7$ on two domains differing 12-fold in scale, while
$1/R$ at the MAC edge was $2.59$ and $19.4$
([cartesian-taylor-basis-progress-log.md](../../../canopy/tasks/cartesian-taylor-basis-progress-log.md)
§T4).

**`ncrit` is the same quantity feeding a wider near field**, which is
[The far field has to be live to be measured](#the-far-field-has-to-be-live-to-be-measured).

#### The order that reaches the target

Canopy accepts equal-size cells only beyond $R/w>2\sqrt3/\theta$ with $w$ a
half-width (`canopy/src/Canopy_CommunicationPlan.hpp:338-352`), and a
Cartesian-Taylor truncation at order $p$ leaves a relative error
$\sim(cw/R)^{p+1}$ on the potential and $\sim(cw/R)^{p}$ on the gradient. Canopy
measured $c\approx1$ in half-widths directly — $1.062$, $0.987$ and $0.939$ at
$R/w=8$, $16$, $32$
([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md) T3)
— so the model for the quantity Beatnik reads is
$\epsilon_{\rm grad}\approx(\theta/2\sqrt3)^{p}$:

| `mac_theta` | $R/w$ accepted | gradient at $p=2$ | at $p=3$ | at $p=4$ | $p$ for $10^{-3}$ | DOF/cell $\binom{p+3}{3}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 (the reference's, and Beatnik's) | $>11.55$ | $7.5\times10^{-3}$ | $6.5\times10^{-4}$ | $5.6\times10^{-5}$ | **3** | 20 |
| 0.5 (Canopy's own default) | $>6.93$ | $2.1\times10^{-2}$ | $3.0\times10^{-3}$ | $4.3\times10^{-4}$ | 4 | 35 |

The model is confirmed at all three points Canopy has measured on a volumetric
cloud, within 20% throughout: $9.0\times10^{-3}$ against $7.5\times10^{-3}$ at
$(\theta,p)=(0.3,2)$, $7.07\times10^{-4}$ against $6.5\times10^{-4}$ at
$(0.3,3)$, and $1.87\times10^{-2}$ against $2.1\times10^{-2}$ at $(0.5,2)$.

$p=3$ is therefore the production order and the smallest that reaches $\tau_A$ at
Beatnik's $\theta$. **It is still a model on Beatnik's geometry**: it is
evaluated at an idealized equal-cell pairing on a volumetric cloud, and the
sheet's anisotropic leaf occupancy and the depth-mismatched cell pairs an
adaptive tree realizes are outside it. **T5** measures the curve on real
milestone-0 states and picks the production order; nothing may be compiled into a
test before it does.

#### The far field has to be live to be measured

At $\theta$ the near field reaches $\sqrt3/\theta$ **cell widths** — $5.77$ at
$\theta=0.3$. Beatnik's sources are a 2-manifold, so the leaves the surface
occupies form a two-dimensional grid and that neighbourhood covers
$\pi(\sqrt3/\theta)^2\approx105$ of them. A surface of $N$ vertices at leaf
occupancy `ncrit` refines until a cell holds `ncrit`, reaching $N/\texttt{ncrit}$
occupied leaves in $\log_4$ rather than $\log_8$ levels
([canopy0.md](../../../canopy/tasks/canopy0.md) F4), so a far field that carries
any of the field at all needs

$$
N \;\gg\; \pi\,(\sqrt3/\theta)^2 \cdot \texttt{ncrit} ,
$$

which at $\theta=0.3$ and the reference's `ncrit = 64` is $N\gg6720$. Milestone-0
is 642 and 2562 vertices. **A comparison run at the default `ncrit` on either
level compares two direct sums**, passes at any order and measures nothing — the
trap [treecode.md](../treecode.md) §1 documents on the treecode side, here with a
number on it. Canopy met the same wall and answered it by pairing 8640 particles
with `ncrit = 8` rather than by lowering the particle count
(`canopy/tests/tstCartesianTaylorSolve.hpp:96-107`).

`FmmParams::ncrit` stays at the reference's 64, which is the right default at
production vertex counts and wrong only at milestone-0's. **T4**, **T5** and
**T6** each run at an `ncrit` satisfying the bound at their vertex count, and
each **asserts** the far field is live through the P2P pair fraction rather than
assuming it. Two consequences follow and neither is a defect in this work:

- At 2562 vertices the bound needs $\texttt{ncrit}\lesssim24$; `ncrit = 8` gives
  320 occupied leaves against 105, a factor of 3, which is live but not
  generous.
- At 642 vertices it has **no solution**: `ncrit = 8` gives 80 occupied leaves,
  still inside the 105-leaf near field, and lower values degenerate the tree.
  The 642-vertex level cannot exercise a far field at $\theta=0.3$ under any
  `ncrit`. That is a property of the mesh — the reference is barely engaging
  there too, which is why its $1.6\times10^{-3}$ at 642 is its *worst* of the
  three sizes in [treecode.md](../treecode.md) §1. **T6**'s L3 member therefore
  carries claim B and a claim A that is mostly a P2P comparison, and says so;
  the far-field accuracy claim rests on the L4 member.

### Why not tighter than $10^{-3}$

A 2000-step FMM-driven trajectory cannot be compared against the direct gold set
at the existing `--rtol 1e-10 --atol 1e-12`, and no order reaches far enough to
change that. The divergence is measured: a one-ulp seed ($5.6\times10^{-17}$ on
`vertices`) grows to $8.5\times10^{-13}$ by step 2000 at level 3
([milestone0-progress-log.md:320-332](../milestone0-progress-log.md)), about
$10^4$ amplification, power-law rather than exponential. A $10^{-3}$
perturbation injected at **every** evaluation is thirteen orders above the seed
that already exhausts the tightest rung, so the FMM-driven and direct-driven
trajectories decorrelate long before step 2000 — the same conclusion
[treecode.md](../treecode.md) §1 reaches for the reference treecode, and for the
same reason. Chasing $10^{-10}$ instead would want $p\approx11$-24 on the potential and one
order more on the gradient Beatnik actually reads — 364-2925 DOF per cell
([canopy-kernel-rec.md](../../../canopy/tasks/canopy-kernel-rec.md)) — for a
trajectory comparison that still would not pass.

The consequence shapes **T6**: its two claims are a per-evaluation bound and a
stability-plus-divergence-horizon measurement, not one loosened gold comparison.

**Out of scope.** Any change to Canopy — including any reshaping of its
acceptance criterion (**X1** names the one conditional dependency and its
acceptance test; it does not design it), and the fix for the operator cache
[Current state](#canopy) records as emptying on every build. Any new CLI option —
the surface is closed, and every knob this work needs is already parsed
([examples/02_adaptive_mesh_bubble/InputFile.hpp:444-470](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L444-L470)).
`Face` and `Triangle3` source quadrature, which still throw
([src/Beatnik_SourceQuadrature.hpp:377-473](../../src/Beatnik_SourceQuadrature.hpp#L377-L473)).
Periodic boundaries, which this mesh does not have. Restart of Canopy's internal
state.

## Approach

### The physics maps onto one Canopy solve, exactly

Canopy's `NComps` is the number of simultaneous independent charge components,
and with `NComps = 3` and `compute_gradient = true` one traversal produces, per
target $i$, the $3\times3$ tensor (`DownwardSweep::gradient()`, accumulated for
near pairs by `canopy/src/Canopy_P2P.hpp:883-896`)

$$
T_{cj}(i) \;=\; \texttt{gradient}(i,c,j) \;=\; -\sum_s q_{c,s}\,
  \frac{\delta_j}{(r^2+\varepsilon^2)^{3/2}}, \qquad \delta = x_i - y_s .
$$

That is Beatnik's kernel with $\varepsilon^2 = b$, up to sign
([src/Beatnik_BRSolverBase.hpp:29-31](../../src/Beatnik_BRSolverBase.hpp#L29-L31)).
Loading the three charge components with the three components of the
area-weighted source vector makes both contractions Beatnik needs **local
post-processing of that tensor**, with no second traversal and no communication:

- **Velocity.** With $q_c = \omega_s S_{c,s}$, the Birkhoff-Rott sum
  $\sum_s (\delta\times S_s)\omega_s K$ is $-\epsilon_{ijk}T_{kj}$, i.e.
  component-wise $u_0 = T_{12}-T_{21}$, $u_1 = T_{20}-T_{02}$,
  $u_2 = T_{01}-T_{10}$. The cross product is linear in the source strength, so
  it commutes with any expansion of the kernel — which is why it belongs here and
  not inside a basis.
- **Riesz scalar.** With $q_c = \omega_s G_{c,s}$,
  $\sum_s (\delta\cdot G_s)\omega_s K = -\operatorname{tr} T$.

The two evaluations do **not** share a tensor: the velocity contracts against
the sheet strength and the Riesz scalar against a different vector field. They
are two `solve()` calls over one tree, which Canopy supports directly —
`solve()` re-reads the charge slice and zeroes its outputs each call.

Under `CartesianTaylorBasis` the far-field gradient is **analytic**: L2P
differentiates the local Taylor expansion in closed form rather than by central
difference
([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md),
"The far field is three scalar passes"), so the gradient carries no
finite-difference step-size error and no third error component for **T5**'s scan
to disentangle. What it does carry is one order less accuracy than the same
solve's potential, which is why the production order is 3 and not the
reference's 2 — see
[The order that reaches the target](#the-order-that-reaches-the-target). The
solid-harmonic basis retains its finite-difference L2P and pays the same order
loss on top of the step-size error; that difference is one of the things **T5**
measures between the two.

$1/4\pi$ and `br_sign` on the velocity, and $-1/4\pi^2$ unsigned on the Riesz
scalar, are applied exactly once, in the adapter, matching `BRSolverDirect`
([src/Beatnik_BRSolverDirect.hpp:126-127](../../src/Beatnik_BRSolverDirect.hpp#L126-L127),
[:206-208](../../src/Beatnik_BRSolverDirect.hpp#L206-L208)).

### Two decompositions, and the round trip between them

Beatnik's sources are owned mesh vertices under Tessera's decomposition; the
result must land back on the vertex that produced it, on the rank that owns it.
Canopy owns its own decomposition and **permutes and migrates the caller's
array**: `setup` and every maintenance path redistribute particles across ranks
and reorder the local array, with the within-AoSoA order after migration
explicitly unspecified (`canopy/src/Canopy_TreePartitioner.hpp:576-580`), and
nothing in `canopy/src/` carries a caller-supplied identity through it. The
far-field abstraction did not change this: the tree builder, partitioner, MAC,
dual-tree traversal and communication plan are basis-blind and identical for both
bases
([abstract-solver-backend.md](../../../canopy/tasks/abstract-solver-backend.md),
"Out of scope").

The round trip is therefore Beatnik's job, and it is the **tag-reverse
handshake** proven on the `origin/develop-canopy` branch
(`src/FmmBRSolver.hpp:198-296`, `325-480` on that branch — the structured-mesh
predecessor of this work, verified at 1 and 4 ranks): a tag member travels with
each particle through every Canopy migration, and a `Cabana::Distributor` built
from the tags currently held routes fresh data in and results out.

It is simpler here than there. Under the `Vertex` quadrature the sources are
exactly the owned vertices in owned order — `pointCount` returns the vertex
count and `generate` writes rows $[0, \texttt{ownedVertexCount()})$ from
`mesh.positions()`
([src/Beatnik_SourceQuadrature.hpp:194-197](../../src/Beatnik_SourceQuadrature.hpp#L194-L197),
[:225-256](../../src/Beatnik_SourceQuadrature.hpp#L225-L256)) — and the targets are
the same rows of the same array. So **source index, target index and output row
are one integer**, and the tag is `(origin_rank, owned_index)` rather than
develop-canopy's `(rank, i, j)`.

Per evaluation, the sequence is:

1. `quadrature.generate` → `(Ns,3)` points and `(Ns,3)` area-weighted strengths,
   owned rows only. Already written; the adapter does not reimplement it.
2. Pack into a mesh-ordered AoSoA: position, charge, output, tag.
3. First evaluation of the run: deep-copy to the Canopy-ordered AoSoA and
   `setup`. Later evaluations: build the forward `Distributor` from the tags the
   Canopy-ordered AoSoA currently holds, migrate the fresh mesh-ordered tuples
   into it, then `auto_maintain`.
4. `solve<Position, Charge>(particles, compute_gradient = true)`.
5. Contract the tensor into the output member, in place, on Canopy's ordering.
6. Reverse-`Distributor` keyed on `tag.origin_rank`, then scatter into the
   caller's `(Nt,3)` view by `tag.owned_index`.

Steps 3-6 are three collectives per evaluation on top of Canopy's own, nine per
timestep. That cost is real and is measured by **T8**, not assumed.

### Why the adapter holds the Canopy state and the BR solver does not

`FarFieldSolver` ([src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp))
is the only header permitted to name a Canopy type; `BRSolverFMM` names none
today and must still name none afterwards. Canopy's `Solver` is a persistent
object whose tree, partition and communication plan are the thing being reused
across evaluations, and whose `P_ORDER`, `NComps` **and now the far-field basis**
are compile-time template parameters. All of those are Canopy facts, so all of
them live behind the adapter: the adapter owns the `Solver`, the two AoSoAs and
the first-call flag, and `BRSolverFMM` remains the thirty lines that turn a
`(mesh, geometry, state, quadrature)` tuple into a call.

Two compile-time parameters against runtime values is a genuine mismatch —
`FmmParams::order` is an `int`
([src/Beatnik_Params.hpp:229](../../src/Beatnik_Params.hpp#L229)) and the
basis is a `FarFieldBasis` enum (**T1**). The adapter resolves it by dispatching
the runtime `(basis, order)` pair onto an explicitly enumerated set of
instantiations and throwing for anything else — see the conventions table. Since
each arm is a distinct C++ type, what the adapter actually holds is a type-erased
handle to one of them, constructed once and persistent thereafter; **T2** step 5
is where that lands. It does not silently round and it does not silently
substitute a basis.

**Canopy's basis parameter is defaulted to the solid-harmonic basis**, so an
adapter that forgets to name the basis compiles, runs, and produces the bare-$1/r$
far field with no diagnostic. The dispatch must name the basis explicitly on
every instantiation; **R2** is the risk entry and **T4**'s negative case is what
catches it.

### The two milestone members: two claims, one binary

Each member asserts **two different claims**, in one binary, one launch, because
they need different trajectories and only one of them can be a gold comparison.

**Claim A — the per-evaluation bound.** The trajectory is driven by
`BRSolverDirect`, so the run is bit-identical to the existing member and all 81
gold comparisons still run at `--rtol 1e-10 --atol 1e-12`; that is what proves it
is the right trajectory. At each of those 81 states the FMM velocity is
additionally evaluated **on the same state** and compared against the direct
velocity, asserting a max relative error $\le\tau_A$. Same input, same state, no
chaotic amplification in the way — the only comparison that isolates the far
field, and the only one that can carry a number as tight as $\tau_A$.

**Claim B — the trajectory stays physical, and decorrelates no sooner than
measured.** The same binary then runs 2000 steps FMM-driven. It **cannot** assert
a gold rung: at $\tau_A\approx10^{-3}$ per evaluation the trajectories separate
well before step 2000 and a passing rung would only mean the rung was loose. It
asserts instead the properties that remain meaningful after decorrelation, and
that are exactly the ones develop-canopy's full-roll-up blow-up violated
(`tasks/fmm_premature_nan.md` on that branch: an unsoftened multipole ~35x too
large seeded a runaway to NaN, while the direct solver completed the identical
deck):

- the run reaches step 2000 and every velocity is finite;
- the entity counts never change;
- the volume drift tracks the reference's own `kRefVolumeDrift` series within a
  bound **T5** measures — a conserved integral survives decorrelation of the
  pointwise field, which is what makes it the useful check here;
- the **divergence horizon** — the first step at which the run exceeds each rung
  against the gold set — is reported as a ladder and asserted against an envelope
  **T5** measures. This is M0-D1's instrument applied to a new perturbation
  source: it turns "the trajectory decorrelates" from an untestable fact into a
  regression signal, because a later change that makes it decorrelate *sooner*
  fires while a loose single rung would not.

Both of **T6**'s compiled numbers — $\tau_A$ and the horizon envelope — are
measurements **T5** produces. A number chosen before the measurement is a number
tuned to whatever the code did.

### Conventions

| Choice | Rule |
| --- | --- |
| Canopy visibility | `Beatnik_FarFieldInterface.hpp` is the **only** header that may name a Canopy type, include a Canopy header, or hold a Canopy object. The `FarFieldBasis` enum is a Beatnik type naming no Canopy type and lives with the other mode enums in `Beatnik_Types.hpp`; the adapter is where it becomes a Canopy basis. Verified by `grep -n "include.*Canopy_\|Canopy::" src/*.hpp src/*.in` naming only `Beatnik_FarFieldInterface.hpp`. A bare `grep -l Canopy src/*.hpp` is **not** the check: sixteen headers match the word in prose comments, `Beatnik_Params.hpp` among them. |
| Build guard | Everything Canopy-facing sits behind `BEATNIK_ENABLE_CANOPY` ([src/CMakeLists.txt:1-6](../../src/CMakeLists.txt#L1-L6), [:78-79](../../src/CMakeLists.txt#L78-L79)). A `~canopy` build must still compile every header and still throw the existing configuration error ([src/Beatnik_CreateBRSolver.hpp:66-69](../../src/Beatnik_CreateBRSolver.hpp#L66-L69)). |
| Failure behavior | A violated precondition throws `std::logic_error` for "this code is unwritten" and `std::runtime_error` for "this build or configuration cannot do it", matching [src/Beatnik_CreateBRSolver.hpp:45-49](../../src/Beatnik_CreateBRSolver.hpp#L45-L49). Never return a truncated or best-effort field: a plausible wrong velocity is the failure mode this whole document exists to bound. |
| New parameters | Added to `FmmParams` ([src/Beatnik_Params.hpp:167-396](../../src/Beatnik_Params.hpp#L167-L396)) with a defaulted member and a comment stating units, the meaning of the default, and which Canopy knob it reaches. Never a new constructor parameter, never a new CLI option. |
| Runtime dispatch | `(FmmParams::basis, FmmParams::order)` selects among an explicitly enumerated set of instantiations; an unsupported pair throws naming the supported set. Never silently rounded, never silently substituted. The set and the compile-time cost of extending it are documented on the dispatch. |
| Basis is always named | Every Canopy `Solver`/`createSolver` instantiation names its basis explicitly. The parameter is defaulted upstream and the default is not the basis Beatnik wants. |
| Enums over bools | A mode selector is an enum or tag type, never a bool or a magic number. |
| Comments | Units, sign convention, and which side of a difference is which, on the declaration. The sign of Canopy's gradient output and the direction of $\delta$ are the two most misread things on this path and must be stated at every boundary they cross. |
| Provenance | Any routine derived from `origin/develop-canopy`'s `src/FmmBRSolver.hpp`, from Canopy, or from the reference Python cites the file and line range on the routine. |
| Accuracy claims | Every stated tolerance names **which field it is on — potential or gradient** — plus the source distribution, the rank counts, **the basis**, `order`, `ncrit`, `max_depth`, `mac_theta`, `softening`, `near_softening_factor` and the realized P2P pair fraction. The two fields differ by a full order at fixed `order`, so a figure without its field named is unreadable, and a figure without its P2P fraction may be a direct sum wearing an FMM's name. A bare tolerance is not a claim and may not be compiled into a test. |
| Citing Canopy | Cite Beatnik by `file:line`. Cite Canopy **source** by symbol name — `FmmConfig::near_softening_factor`, `Solver::auto_maintain` — with a line number only where the symbol is in a file the far-field abstraction does not restructure (`Canopy_P2P.hpp`, `Canopy_CommunicationPlan.hpp`, `Canopy_TreeBuilder.hpp`, `Canopy_TreePartitioner.hpp`). A stale line number into a restructured header points at unrelated code and is worse than no citation. Cite Canopy **design documents** by relative path into `../../../canopy/tasks/`; this repository keeps no copy of them, so a citation that resolves inside `tasks/canopy/` is stale by construction. |
| Test tier | New correctness tests are `unit` unless a task says otherwise. **The gate does not change**: it stays at five `regression` members and 60 launches (CLAUDE.md "Minimum test set"). The two new members go in the `milestone` tier, which is not the gate. |
| Formatting | Do not run clang-format, `clangformat.sh` or `cabana-format`. Match the surrounding style by hand. |

### Deliberate deviations

- **No `local` or `clustered` far field, and no treecode.** The CLI already maps
  all three onto `fmm` with a warning
  ([examples/02_adaptive_mesh_bubble/InputFile.hpp:456-463](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L456-L463)),
  and [treecode.md](../treecode.md) §3 costs a standalone treecode port and
  recommends against it on performance grounds. Canopy's Cartesian-Taylor basis
  is the reference treecode's expansion under an $O(N)$ traversal, which is what
  makes porting the treecode itself redundant rather than merely expensive.
- **No treecode-driven gold set is generated for claim B.** Comparing an
  FMM-driven Beatnik trajectory against a Python *treecode*-driven one looks like
  the apples-to-apples trajectory test claim B cannot otherwise be. It is not:
  the two share an expansion but not an acceptance criterion and not a traversal,
  so their per-evaluation errors differ in detail, and two $10^{-3}$ methods do
  not track each other any better than either tracks the direct sum. It would
  cost a third 2000-step Python reference run and buy a comparison with the same
  decorrelation problem.
- **Canopy's own decomposition is accepted, not fought.** The alternative —
  asking Canopy to adopt the mesh's decomposition — would give up the load
  balancing that is the reason Canopy partitions separately, and Canopy exposes
  no such mode. The tag round trip is the price.
- **The solid-harmonic basis stays reachable through `FmmParams::basis`.** It is
  not the production path, but it is the negative control that proves the basis
  selector is live, and it is the only way **T5** can report what the choice
  bought. Removing it from the enum would make **T4**'s negative case
  unwriteable.
- **`auto_maintain` rather than `setup` per evaluation.** `setup` every stage is
  simpler and always correct, but pays a full global tree build plus repartition
  three times per timestep ([canopy0.md](../../../canopy/tasks/canopy0.md) F3(b)).
  develop-canopy measured `auto_maintain` returning the cheapest `Migrate`
  action for all 14 calls of a five-step run with results identical to the
  setup-every-step baseline. **T8** measures whether that holds on a deforming
  surface, where [canopy0.md](../../../canopy/tasks/canopy0.md) F3(c) predicts
  `Rebalance` instead. Note what the choice no longer buys: under
  `CartesianTaylorBasis` even the cheapest `Migrate` rebuilds the whole M2L
  operator table, because every maintenance path recomputes the root box and so
  clears the cache — see [Current state](#canopy). The remaining argument for
  `auto_maintain` is the tree build and the repartition, not the operators.
- **No attempt to reshape Canopy's acceptance criterion.** Matching the
  reference's MAC would mean a target-point, cloud-radius predicate inside
  `CommunicationPlan`, which is shared by both bases, and it would not repay the
  work. $\theta$ can already be widened to reproduce the reference's acceptance
  *radius* — about $\theta=0.735$ once the target-box term is accounted for —
  and doing so costs $p=5$ for $10^{-3}$ on the gradient, 56 DOF per cell and a
  derivative ladder to $|k|=10$, against $p=3$, 20 DOF and $|k|=6$ at
  $\theta=0.3$. The target-side expansion is what makes an FMM an FMM, and it is
  paid for in both the acceptance radius and the gradient's order; Beatnik takes
  Canopy's predicate at the reference's $\theta$ and pays in order instead.
- **No attempt to correct the kernel on Beatnik's side.** Handing Canopy a bare
  far field and adding
  $\sum_{r<R_c}[K_{\rm soft}-K_{\rm bare}]\times S_s$ by direct summation is
  algebraically a near-field softening floor by another name, and it is what
  selecting `CartesianTaylorBasis` makes unnecessary. It buys nothing and is not
  attempted.

## Current state

### Beatnik

- `--br-approximation fmm` **runs** (**T3**).
  `BRSolverFMM::computeInterfaceVelocity` is live
  ([src/Beatnik_BRSolverFMM.hpp:140](../../src/Beatnik_BRSolverFMM.hpp#L140));
  only `::computeSurfaceRieszScalar` still throws
  ([:198](../../src/Beatnik_BRSolverFMM.hpp#L198)) via
  `BEATNIK_NOT_IMPLEMENTED`
  ([src/Beatnik_Types.hpp:86](../../src/Beatnik_Types.hpp#L86)), and it is
  reached only under `--bernoulli-scalar-mode surface-riesz`. It throws rather
  than returning a wrong field, which is the safe direction.
- `FarFieldSolver`'s three methods are **real and Canopy-backed** (**T2**).
  `setSources` / `evaluateCurl` / `evaluateDot` are gone, replaced by
  `evaluateVelocity`, `evaluateRieszScalar` and `diagnostics()`
  ([src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp)); the
  adapter owns the type-erased six-arm dispatch, the persistent Canopy
  `Solver`, the `FmmConfig` builder and the tag-reverse round trip.
  **T3** calls into `evaluateVelocity` and **T4** measured what it returns;
  `evaluateRieszScalar` has no caller until **T7**. In a `~canopy` build the two
  evaluations throw `std::runtime_error` naming `+canopy` instead.
- **No Beatnik header includes a Canopy header.** The build already finds and
  links Canopy under `+canopy`
  ([CMakeLists.txt:79-81](../../CMakeLists.txt#L79), [src/CMakeLists.txt:78-79](../../src/CMakeLists.txt#L78-L79)),
  and the tuolumne environment builds with it
  ([systems/tuolumne/claude.md](../../systems/tuolumne/claude.md) §2).
- `FmmConfig`'s `ncrit` and `max_depth` have **no default initializer** in
  Canopy (`FmmConfig::ncrit`, `FmmConfig::max_depth`), so a default-constructed
  `FmmConfig` builds an arbitrary tree. `FmmParams` supplies both, and every
  other member `FmmConfig` requires, with a default (**T1**).
- The mode enums this work extends live in `src/Beatnik_Types.hpp`
  (`BRApproximation` at [:163](../../src/Beatnik_Types.hpp#L163),
  `FarFieldBasis` at [:189](../../src/Beatnik_Types.hpp#L189),
  `BernoulliScalarMode` at [:197](../../src/Beatnik_Types.hpp#L197),
  `ViscosityMode` at [:206](../../src/Beatnik_Types.hpp#L206),
  `KernelBlobMode` at [:219](../../src/Beatnik_Types.hpp#L219)).
- The CLI parses `--br-treecode-theta/-order/-ncrit`
  ([examples/02_adaptive_mesh_bubble/InputFile.hpp:478-480](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L478-L480))
  and nothing else FMM-facing. No new option may be added.
- `Beatnik_Test_Milestone0Frozen.cpp` sets
  `p.zmodel.br_approximation = BRApproximation::Direct` explicitly
  ([tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp:509](../../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp#L509)),
  with a comment stating that `fmm` would add an approximation error the
  comparison cannot separate from round-off divergence. That statement is correct
  and is why **T6** adds members rather than a flag.
- The `milestone` tier has two members and eight launches, measured at **37.25
  minutes** under the runner's `# flux: -t 60m` and `-q pdebug`
  ([scripts/tuolumne/run_milestone.flux:5-7](../../scripts/tuolumne/run_milestone.flux#L5-L7)).
  There is **not** room for two more members of comparable cost.

### Canopy

Cited by symbol, per the conventions table.

- The far field is selected by a template parameter on `Solver` and
  `createSolver`, **defaulted to the solid-harmonic `LaplaceKernel`**. The
  contract a basis satisfies — its traits, its five operators, its three-stage
  M2L, its auxiliary tables and its overflow policy — is specified in
  [abstract-solver-backend.md](../../../canopy/tasks/abstract-solver-backend.md).
  The runtime API Beatnik calls is unchanged by the abstraction:
  `setup<PositionIdx, ChargeIdx>(particles, count_before_migration)`,
  `solve<PositionIdx, ChargeIdx>(particles, compute_gradient)`,
  `auto_maintain<PositionIdx, ChargeIdx>(particles)`, `num_local_particles()`,
  `potential()`, `gradient()`.
- **`CartesianTaylorBasis` is built and measured**, in
  [cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
  T1-T6, all **DONE**. It expands $\varphi(r)=(r^2+b)^{-1/2}$ directly, in real
  coefficients, $\binom{p+3}{3}$ per cell per component; requires
  `Scalar = double` by `static_assert`; has an analytic L2P gradient; and its
  operator keys carry the tree level (`key_needs_level = true`), so its operator
  table is larger than the solid-harmonic basis's by roughly the number of
  occupied depths.
- **What it was measured at**, on 8640 particles in a volumetric cube,
  `ncrit = 8`, `max_depth = 6`, `replication_depth = 2`, `softening = 0.025`,
  `near_softening_factor = 0`, four solves with `migrate / rebalance / migrate`
  between them, ranks 1-6, against a direct softened sum:

  | arm | potential | gradient |
  | --- | --- | --- |
  | $\theta=0.3$, $p=3$ | $1.9263\times10^{-5}$ | $7.0718\times10^{-4}$ |
  | $\theta=0.3$, $p=2$ | $2.655\times10^{-4}$ | $8.996\times10^{-3}$ |
  | $\theta=0.5$, $p=2$ | $9.9667\times10^{-4}$ | $1.8652\times10^{-2}$ |

  and `LaplaceKernel` at the same `near_softening_factor = 0` and the same
  positive `softening` missing it by three orders on the potential
  ($1.894\times10^{-2}$) and reaching $5.168\times10^{-2}$ on the gradient. This
  is a volumetric cloud, not a sheet; **T5** is what measures the same curve on
  Beatnik's geometry.
- **The derivative ladder is validated to $|k|=6$, which is $2p$ at the
  production order.** Above the closed forms at $|k|\le3$ the oracle is a
  Richardson-extrapolated finite difference, and the M2L reaches $b_{p+q}$ at
  $|p+q|=2p$, so $p=3$ needs $|k|=6$. Canopy measured it, per degree, each with
  its own Richardson step divisor located by an eleven-point scan rather than
  carried over:

  | $\vert k\vert$ | divisor | tolerance | achieved worst | margin |
  | --- | --- | --- | --- | --- |
  | 4 | $L/24$ | $4\times10^{-6}$ | $3.080\times10^{-7}$ | 13.0x |
  | 5 | $L/24$ | $3\times10^{-4}$ | $2.531\times10^{-5}$ | 11.9x |
  | 6 | $L/16$ | $2\times10^{-2}$ | $1.184\times10^{-3}$ | 16.9x |

  No tolerance was widened to pass a degree, and perturbing one recurrence
  coefficient by $0.1\%$ pushed all three above their own bounds (6500x, 913x,
  142x), so degrees 5 and 6 are exercised rather than enumerated. **The sample
  set is Beatnik's own band**: $b=6.25\times10^{-4}$ is $\varepsilon^2$ at
  $\varepsilon=0.025$, and the interior scales $|r|/\sqrt b \in
  \{9.276, 15.46, 74.2\}$ are now permanent — the worst deviation at both
  $|k|=5$ and $|k|=6$ falls at that $b$. The achieved deviation grows about 40x
  per degree, which tracks the roundoff floor of a $|k|$-th difference rather
  than the ladder's conditioning, so it is oracle resolution and not recurrence
  error ([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
  T6; its R8 is closed for $p=3$). **It remains unvalidated above $p=3$** —
  $p=4$ would need $|k|=8$ — and the oracle's 1-D stencils now abort loudly on
  an order they do not carry rather than silently reusing a lower one. That
  bounds **T5**'s scan, not the production path; see **R11**.
- **The M2L operator cache retains nothing on a moving distribution.**
  `set_root_half_width` clears the **entire** cache whenever the root half-width
  changes and does so only for a `key_needs_level` basis
  (`canopy/src/Canopy_DownwardSweep.hpp:406-416`);
  `Solver::_push_root_half_width` runs immediately before every one of the three
  `_downward.setup()` calls, and `TreeBuilder::build` recomputes the root box
  from the particles on **every** maintenance path — `migrate` included
  (`canopy/src/Canopy_TreeBuilder.hpp:608-621`). There is no `FmmConfig` knob
  that pins the box: the six bounding-box tolerances pad it by fractions of its
  own width, so a padded box drifts too. Canopy measured the consequence and it
  is total: **zero keys retained at every one of 336 builds**, 3.9x the cached
  key count constructed over four solves, with the drift a smooth 0.18-0.40% per
  build so that a power-of-two tolerance would not fire either
  ([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
  T5 and R6). Beatnik's surface deforms every RK stage. The magnitude on
  Beatnik's tree is **T8** step 2; the fix is Canopy's and is out of scope.
- **The M2L operator table** is bounded by
  `min(M2L_OP_COUNT_CAP, byte_budget / bytes_per_key)`, with
  `M2L_OP_COUNT_CAP = 32768` and the byte budget an `FmmConfig` field defaulting
  to 2 GB. Keys beyond the cap route to a per-pair translate fallback —
  different arithmetic, slower, and asserted equal to the table path
  ([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
  T3) — and `total_fallback_pair_count()` reports how many pairs took it. At
  $p\le4$ the per-key operator is $\le9.8$ KB
  ([canopy-kernel-rec.md](../../../canopy/tasks/canopy-kernel-rec.md),
  "Memory"), so the **count cap binds first** and the byte budget is not the
  constraint. Realized counts on the cloud above, at np=1: **25438** unique keys
  at $\theta=0.3$ and 7374 at $\theta=0.5$, on 2941 cells, with 246 and 0
  fallback pairs. Canopy also observed the $\theta=0.3$ table *saturating* the
  32768 cap once the trajectory was amplified, so the cap is a live constraint
  at this $\theta$ rather than a distant one. This is **R6**.
- `FmmConfig::near_softening_factor` still exists, defaults to **4.0**, and
  still forces close pairs into P2P. Beatnik sets it to 0 under
  `CartesianTaylorBasis`; it is meaningful only under the solid-harmonic basis,
  and it has **no effect at all unless `softening > 0`**.
- **`FmmConfig::softening` defaults to $-1.0$, which selects distribution-based
  auto-softening** at first `setup()` — an effective $\varepsilon$ that moves
  with the particle distribution. It is a **length**; Beatnik's `blob()` is a
  squared length
  ([src/Beatnik_Params.hpp:125-128](../../src/Beatnik_Params.hpp#L125-L128)).
  Leaving it defaulted is not a fallback but a silent substitution of a
  different kernel, and it additionally disables `near_softening_factor`. It
  must be set explicitly positive; `build_m2l_operators` aborts loudly on
  `softening <= 0`, which covers only the zero-or-negative case and not the
  auto-softening one.
- `solve()` after motion without maintenance is silently wrong, `migrate()` is
  not cheap, and a deforming surface picks `Rebalance` essentially every stage
  ([canopy0.md](../../../canopy/tasks/canopy0.md) F3(a)-(c)). None of this is
  basis-dependent and none of it changed.
- Zoltan2's `multijagged` is non-deterministic across runs, so the far-field path
  is not bitwise reproducible run to run at fixed rank count
  ([canopy0.md](../../../canopy/tasks/canopy0.md) F3(c)).
- **The np=4 defect is narrower than it was.** `SingleSolve.PotentialNComps3`
  and `SingleSolve.PotentialAndGradientNComps3` still fail at exactly 4 ranks
  (`max_pot_rel_err = 0.00207` against a $10^{-3}$ budget,
  `canopy/README.md:562-588`), under `LaplaceKernel` at $P=8$ and
  `softening = 0`. But `CartesianTaylorSolve` drives `NComps = 3` with the
  gradient compared and **passes at ranks 1-6**, which is the closest existing
  proxy for Beatnik's configuration. Beatnik's gate and milestone tier both run
  at 4 ranks. This is **R4**.
- No Canopy test gives any rank zero particles
  ([canopy0.md](../../../canopy/tasks/canopy0.md) F5), and no Canopy test
  measures accuracy on a non-volumetric source distribution
  ([canopy0.md](../../../canopy/tasks/canopy0.md) F4) — `CartesianTaylorSolve`
  is a volumetric cube. Both gaps are on this path.

## Progress log

[add-canopy-progress-log.md](add-canopy-progress-log.md) holds what actually
happened: the reasoning behind decisions this document states flatly, the
measured numbers behind its claims, and things only running revealed. **Read it
before implementing any task, changing any signature, or reopening a question
this document treats as settled** — in particular before compiling any tolerance
into any test, since a measured number in the log always outranks an estimate
here.

## Task sequence

### T1 — `FmmParams` carries everything `FmmConfig` needs, and names the basis — **DONE**

**Depends on:** none.

**Fill in:** [src/Beatnik_Params.hpp](../../src/Beatnik_Params.hpp) (`FmmParams`),
[src/Beatnik_Types.hpp](../../src/Beatnik_Types.hpp) (the new enum, beside the
existing mode enums at [:163-193](../../src/Beatnik_Types.hpp#L163-L193)),
[examples/02_adaptive_mesh_bubble/InputFile.hpp](../../examples/02_adaptive_mesh_bubble/InputFile.hpp)
(only where an already-parsed key must reach a new member),
[README.md](../../README.md).

**Reference:** `FmmConfig`'s full member list, the two members with no default
initializer (`ncrit`, `max_depth`), the six bounding-box padding factors, the
operator byte budget and the `near_softening_factor` comment
(`canopy/src/Canopy_Solver.hpp`, `struct FmmConfig`); `FmmParams`
([src/Beatnik_Params.hpp:167-396](../../src/Beatnik_Params.hpp#L167-L396));
develop-canopy's `makeCanopyConfig` for the full mapping it needed
(`src/FmmBRSolver.hpp:588-607` on that branch).

**Do:**

1. Add `enum class FarFieldBasis { CartesianTaylor, SolidHarmonic }` to
   `Beatnik_Types.hpp`, in the style of the mode enums already there, with a
   comment stating which Canopy basis each maps to and that
   `CartesianTaylor` is the validated production path. It names no Canopy type.
2. Add `FarFieldBasis basis = FarFieldBasis::CartesianTaylor` to `FmmParams`.
3. Extend `FmmParams` with every remaining knob the adapter must set:
   `max_depth`, `near_softening_factor`, `m2l_op_table_byte_budget`, `ncrit_tol`,
   `replication_depth`, `imbalance_tolerance`, and the six bounding-box padding
   factors. Every one gets a default initializer and a comment naming units, the
   default's meaning, and the `FmmConfig` member it reaches.
4. Default `near_softening_factor` to **0**, and state on the declaration that a
   non-zero value is meaningful only under `FarFieldBasis::SolidHarmonic` —
   under `CartesianTaylor` the far field already carries the blob, so the floor
   only moves work into P2P and slows the solve without improving it.
5. **Raise `order`'s default from 2 to 3**, and restate its comment. It is the
   basis's order knob, and under `CartesianTaylor` it is the Cartesian Taylor
   truncation order $p$ — the same *quantity* `--br-treecode-order` denotes in
   the reference, and not the same *accuracy*. The declaration must say why the
   two differ: the reference is a treecode with no target-side expansion, so its
   order-2 velocity has the truncation order of an FMM's order-2 potential,
   which is an FMM's order-3 gradient, and Beatnik reads only the gradient. Give
   the measured figures — $8.996\times10^{-3}$ at $p=2$ against
   $7.0718\times10^{-4}$ at $p=3$, both at $\theta=0.3$ — so the default is
   traceable to a measurement rather than to a preference. `--br-treecode-order`
   still overrides, so a Python command line that passes 2 explicitly still gets
   2. Three places state or qualify the default and all three move together.
   Correct the `FmmParams` doc comment
   ([src/Beatnik_Params.hpp:132-140](../../src/Beatnik_Params.hpp#L132-L140)) and
   the CLI comment
   ([examples/02_adaptive_mesh_bubble/InputFile.hpp:478-479](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L478-L479)),
   both of which say the treecode numbers do not mean the same thing to the two
   algorithms; say per knob which way each fails to transfer rather than
   deleting the warning. Then correct the schema line `printSchema` emits
   ([examples/02_adaptive_mesh_bubble/InputFile.hpp:1098](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L1098)),
   which reads `-> FMM expansion order (2)`. `order` is the one knob where the
   "option names and defaults match the Python script exactly" promise genuinely
   breaks, and the schema is where a user reads it, so that line must carry
   **both** numbers — the Python's 2 and Beatnik's 3 — rather than overwriting
   one with the other. `printSchema` mirrors the option table and README moves
   with it in the same change
   ([:1015-1022](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L1015-L1022)).
6. **Leave `ncrit` at 64** and state the constraint that makes it right only at
   production vertex counts: under Canopy's MAC the near field reaches
   $\sqrt3/\theta$ cell widths, which on a 2-manifold is
   $\pi(\sqrt3/\theta)^2\approx105$ occupied leaves at $\theta=0.3$, so a live far
   field needs $N\gg105\cdot\texttt{ncrit}$ — $N\gg6720$ at this default. Below
   that the solve is a direct sum with FMM bookkeeping, silently and at full
   accuracy. Put the inequality on the declaration, not the conclusion, so it
   re-evaluates at a different `mac_theta`. See
   [The far field has to be live to be measured](#the-far-field-has-to-be-live-to-be-measured).
7. Choose and document a `max_depth` default. It has no counterpart in the
   treecode knob set, is bounded at 19 by the `uint64_t` Morton key
   (`canopy/src/Canopy_TreeBuilder.hpp:176-181`), and sets the finest cell width
   as (root box width) / $2^{\rm max\_depth}$. Two forces pull against each
   other and both belong in the comment: a sheet reaches
   $N/\texttt{ncrit}$ leaves in $\log_4$ rather than $\log_8$ levels, so it wants
   depth ([canopy0.md](../../../canopy/tasks/canopy0.md) F4); and every occupied depth multiplies
   `CartesianTaylorBasis`'s realized operator-key count, which the 32768-key cap
   bounds. develop-canopy ran 19 and hit a depth-driven finite-difference blow-up
   at roll-up — a mechanism the analytic Taylor L2P removes, so that particular
   reason to fear 19 does not transfer. State the reasoning for whatever is
   chosen.
8. **Do not add a `softening` member that duplicates `eps`.** Canopy's
   `softening` is a length and Beatnik's `blob()` is a squared length —
   $\varepsilon^2$ under `Length` and $\varepsilon$ under `Matlab`
   ([src/Beatnik_Params.hpp:125-128](../../src/Beatnik_Params.hpp#L125-L128)) — so
   the adapter must pass $\sqrt{\texttt{blob()}}$, which is `eps` under `Length`
   and $\sqrt{\texttt{eps}}$ under `Matlab`. Deriving it from `blob()` at the one
   call site is what makes the two modes correct without a second source of
   truth. State this on the declaration; getting it wrong is a silent factor of
   $\varepsilon$ in the softening length, and under `CartesianTaylor` that value
   goes into the far field's own $w=r^2+b$ rather than only into P2P, so the
   error is no longer confined to close pairs.
9. `FmmParams` must not be constructible into a state that builds an arbitrary
   Canopy tree. Since every Beatnik member is default-initialized, this reduces
   to giving `ncrit` and `max_depth` defensible values and validating them where
   the adapter builds the `FmmConfig`.
10. Update README's parameter documentation in the same change.

**Exit criterion:** `spack install` succeeds; a `FmmParams` default-constructed
and passed through the adapter's config builder yields an `FmmConfig` whose every
member is initialized, whose `softening` is positive rather than Canopy's
auto-softening sentinel, and whose `near_softening_factor` is 0 (all asserted by
**T4**'s test, which is where a runnable check first exists); `FmmParams`
default-constructs with `order == 3`, `mac_theta == 0.3` and `ncrit == 64`;
README lists every new member with its default and states why `order` departs
from `--br-treecode-order`'s 2 while `mac_theta` and `ncrit` do not. No new CLI
option appears in `--help`, and `--br-treecode-order 2` still yields
`order == 2`.

**Met.** `FarFieldBasis` is in
[src/Beatnik_Types.hpp:169-193](../../src/Beatnik_Types.hpp#L169-L193) with a
`toString` beside the other ten enums' (that file's one-table-per-enum
invariant), and `FmmParams`
([src/Beatnik_Params.hpp:133-396](../../src/Beatnik_Params.hpp#L133-L396)) now
carries all sixteen members. `spack install` succeeded after touching
`examples/02_adaptive_mesh_bubble/adaptive_mesh_bubble.cpp` — 29 CXX objects
compiled including `adaptive_mesh_bubble.cpp.o`, zero `error:` lines, so the
header change was genuinely compiled rather than no-op'd past. Verified by
reading, which is where the checkable part of this criterion lives: the
declarations give `order = 3`, `mac_theta = 0.3`, `ncrit = 64`, `max_depth = 10`,
`near_softening_factor = 0.0`, `ncrit_tol = 0.1`, `replication_depth = 3`,
`imbalance_tolerance = 0.10`, the six bbox factors at 0.10 and
`m2l_op_table_byte_budget` at 2 GiB; the `br-treecode-order` parse lambda is
untouched and still assigns `fmm.order` from the argument, so an explicit 2 still
yields 2; and diffing the parse-table key set and the `printSchema` option-token
set against `HEAD` returns **identical** on both, which is the proof that no new
CLI option appears in `--help`. The `printSchema` order line now carries both
numbers ("Python 2, Beatnik 3") with the gradient-versus-potential reason.

**What is deferred, and what this task did not prove.** The `FmmConfig` clause of
the criterion above is **T4**'s: there is no config builder yet and T1 added none,
so "every member initialized, `softening` positive, `near_softening_factor` 0" is
asserted nowhere runnable. What T1 owes and delivered is that the values exist and
are defensible on their declarations. Nothing was run: no binary was invoked, and
`--help` was checked by `grep` on the static string literal `printSchema` emits
rather than by executing it.

**Canopy isolation holds.** `grep -n "include.*Canopy_\|Canopy::" src/*.hpp
src/*.in` returns nothing — no Beatnik header includes a Canopy header or names a
Canopy type, and neither new type is guarded by `BEATNIK_ENABLE_CANOPY`. See the
progress log's `## T1`.

---

### T2 — `FarFieldSolver` backed by Canopy: the adapter and the round trip — **DONE**

**Depends on:** T1.

**This is the task that first names a Canopy type in Beatnik code.** T1 reads
`canopy/src/Canopy_Solver.hpp` to derive its defaults and its comments, but names
no Canopy type, includes no Canopy header and holds no Canopy object; no task
before this one may. Every Canopy reading decision, every signature that
Canopy's actual API forces, and every departure from the interface as it stands
today is recorded in the log by this task.

**Fill in:** [src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp)
(the whole class body and its three signatures),
[src/CMakeLists.txt](../../src/CMakeLists.txt) if the header's guard structure
changes, [README.md](../../README.md).

**Reference:**

- `origin/develop-canopy`'s `src/FmmBRSolver.hpp` — the working predecessor.
  `packGridParticles` (`:150-196`), `buildForwardDistributor` and its
  five-step tag-reverse handshake (`:198-296`), the full pipeline including the
  first-call branch (`:325-480`), the cross-product contraction (`:414-426`),
  the reverse distribute and scatter (`:428-470`), and the persistent-state
  members with the reasoning on each (`:609-644`).
- Canopy's API, by symbol: `FmmConfig`; the `Solver` constructor and its
  softening branch; the basis template parameter and its default;
  `setup<PositionIdx, ChargeIdx>` and its "count BEFORE migration" contract;
  `solve<PositionIdx, ChargeIdx>(particles, compute_gradient)`;
  `auto_maintain<PositionIdx, ChargeIdx>` and its three-way `MaintenanceAction`;
  and the accessors `num_local_particles()`, `potential()`, `gradient()`. All in
  `canopy/src/Canopy_Solver.hpp`. The basis itself is
  `canopy/src/Canopy_CartesianTaylorBasis.hpp`.
- The gradient's shape and sign (`DownwardSweep::gradient()`,
  `canopy/src/Canopy_P2P.hpp:883-896`), and that P2P skips pairs with
  $r^2<10^{-24}$ (`canopy/src/Canopy_P2P.hpp:881`) — for this kernel the self
  term contributes exactly zero to the gradient, so skipping it is correct rather
  than merely tolerable.
- What the quadrature hands over: owned rows only, one source per owned vertex,
  in owned order
  ([src/Beatnik_SourceQuadrature.hpp:194-197](../../src/Beatnik_SourceQuadrature.hpp#L194-L197),
  [:225-256](../../src/Beatnik_SourceQuadrature.hpp#L225-L256)), and the R9
  discipline that makes emitting a ghost a rank-count-dependent magnitude error
  ([:216-220](../../src/Beatnik_SourceQuadrature.hpp#L216-L220)).
- The prefactors and signs to reproduce
  ([src/Beatnik_BRSolverDirect.hpp:126-127](../../src/Beatnik_BRSolverDirect.hpp#L126-L127),
  [:206-208](../../src/Beatnik_BRSolverDirect.hpp#L206-L208)) and the convention
  block they come from
  ([src/Beatnik_BRSolverBase.hpp:34-52](../../src/Beatnik_BRSolverBase.hpp#L34-L52)).

**Do:**

1. **Replace the three signatures rather than adding overloads beside them.**
   `setSources( source_points )` cannot express what Canopy needs: the charges
   must be present at `setup` time (`setup<PositionIdx, ChargeIdx>` reads both
   slices), and the maintenance/solve split is not where the current interface
   puts it. Delete the current three methods and their doc comments and write
   the shape Canopy forces. There are exactly **two** callers to update, both
   stubs today: `BRSolverFMM::computeInterfaceVelocity`
   ([src/Beatnik_BRSolverFMM.hpp:113](../../src/Beatnik_BRSolverFMM.hpp#L113)) and
   `::computeSurfaceRieszScalar` ([:140](../../src/Beatnik_BRSolverFMM.hpp#L140)).
   The three replacements, with the tree maintenance decided internally so
   `BRSolverFMM` needs no knowledge of Canopy's lifecycle:

   ```cpp
   using point_view = Kokkos::View<Real* [3], device_type>;
   using vector_view = Kokkos::View<Real* [3], device_type>;
   using scalar_view = Kokkos::View<Real*, device_type>;

   void evaluateVelocity( const point_view& sources,
                          const vector_view& strengths,
                          const ZModelParams& params, vector_view& velocity );

   void evaluateRieszScalar( const point_view& sources,
                             const vector_view& gradients,
                             const ZModelParams& params, scalar_view& scalar );

   const FarFieldDiagnostics& diagnostics() const;
   ```

   Each departure from the current three is forced, and the reason belongs on
   the declaration:

   - **There is no `targets` argument, because Canopy has no target list.**
     `solve<PositionIdx, ChargeIdx>(particles, compute_gradient)`
     (`Canopy_Solver.hpp:237-273`) sizes its `_potential` and `_gradient` views
     to `_num_local` — the particle count — and evaluates the field at the
     particle positions only. Under the `Vertex` quadrature source row, target
     row and output row are one integer, so one array serves both sides; see
     [Two decompositions, and the round trip between them](#two-decompositions-and-the-round-trip-between-them).
     Evaluating at points that are not sources would mean padding the particle
     set with zero-charge targets, which this consumer does not need.
   - **The views are concrete, not templated.** `SourceQuadratureBase`'s
     `point_view` and `strength_view`
     ([src/Beatnik_SourceQuadrature.hpp:102-105](../../src/Beatnik_SourceQuadrature.hpp#L102-L105))
     and `BRSolverBase`'s `vector_view` and `scalar_view` are already
     `Kokkos::View<Real* [3], device_type>` and `Kokkos::View<Real*, device_type>`.
     The `TODO(types): templated pending Tessera/Canopy interface` comments on
     the current three methods are resolved by this task and go away rather than
     being carried forward.
   - **`ZModelParams` replaces the bare `Real blob`.** One object supplies
     `blob()` — whose square root is taken at this one call site per T1 step 8 —
     together with `br_sign`, `blob_mode` and `source_quadrature`, so there is no
     second source of truth for the softening length and no separate sign
     argument to forget. It is also what the two prefactors are read from — see
     [The physics maps onto one Canopy solve](#the-physics-maps-onto-one-canopy-solve-exactly)
     for which one goes on which contraction.
   - **The charges travel with the points in the same call**, because Canopy
     needs them at tree-construction time: `setup<PositionIdx, ChargeIdx>` reads
     both slices. That is what `setSources( source_points )` could not express.
   - **A non-`Vertex` quadrature is rejected here**, with `std::runtime_error`
     naming the quadrature in force. `ZModelParams::source_quadrature` defaults
     to `SourceQuadrature::Face`
     ([src/Beatnik_Params.hpp:95](../../src/Beatnik_Params.hpp#L95)), whose
     `generate` still throws, so today the round trip is protected only by that
     throw. If `Face` is ever implemented the sources stop being the vertices and
     the one-integer tag silently becomes wrong, so the guard belongs here rather
     than being inherited from a stub.
2. Define the AoSoA member layout as an enum, not bare indices: position
   `double[3]`, charge `double[3]`, output `double[3]`, tag `int[2]`
   = `(origin_rank, owned_index)`. develop-canopy's `FmmField` namespace
   (`src/FmmBRSolver.hpp:55-61` on that branch) is the precedent; two tag
   components suffice here because the source list is one-dimensional.
3. Implement the round trip as the six steps in **Approach**. The forward
   `Distributor` is rebuilt every evaluation; caching it across `Migrate`-action
   evaluations is a named future optimization, not part of this task.
4. Build the `FmmConfig` from `FmmParams` plus `ZModelParams::blob()` per T1
   step 8, and validate it: throw naming the offending member if `ncrit` or
   `max_depth` is non-positive, or `max_depth > 19`. **Set `softening` to
   $\sqrt{\texttt{blob()}}$ explicitly and throw if it is not strictly
   positive.** `FmmConfig::softening` defaults to $-1.0$, which selects
   distribution-based auto-softening at first `setup()`: an effective
   $\varepsilon$ that moves with the particle distribution, is not Beatnik's,
   and additionally disables `near_softening_factor`. Canopy aborts on
   `softening <= 0` inside `build_m2l_operators`, which catches zero and
   negative values but *not* the sentinel, since the sentinel is replaced before
   the operators are built. The check has to be here.
5. Dispatch `(FmmParams::basis, FmmParams::order)` onto **six** instantiations,
   **naming the Canopy basis explicitly in every one**, and throw naming the
   supported set for anything else. The set is `CartesianTaylor` at orders 0, 2,
   3, 4 and 5 — 0 for **T4**'s monopole-only negative case, 3 for the production
   path, and the neighbours **T5** scans either side of it — plus **one**
   `SolidHarmonic` arm at order **3**, so **T4**'s negative case and **T5**'s
   basis comparison are built and the comparison is at equal order. Note on the
   dispatch that orders 4 and 5 are **scan-only**: Canopy's derivative-ladder
   oracle stops at $|k|=6$, so they are measurable but not adoptable as the
   production order (**R11**). Add a `static_assert` or equivalent that no
   instantiation relies on Canopy's defaulted basis parameter; a silently
   solid-harmonic solve is **R2**.

   **The dispatch is type-erased, and that is what lets the adapter hold a
   persistent solver at all.** Canopy's basis is a template-*template* parameter
   and its order a non-type template parameter
   (`Canopy_Solver.hpp:125-128`), so every arm is a distinct C++ type and
   `FarFieldSolver` cannot name the `Solver` as a typed member. It holds a
   `std::unique_ptr<Impl>` over an abstract `Impl` declaring the two evaluates
   and the diagnostics, with one
   `template <template <class, int, int> class FarField, int P_ORDER> struct ImplFor : Impl`
   per arm, constructed once on the first evaluation from the runtime
   `(basis, order)` pair and persistent thereafter — which is what makes the
   tree, the partition and the communication plan reusable across evaluations.
   State on the dispatch what an extra arm costs: `Beatnik_FarFieldInterface.hpp`
   reaches every translation unit that creates a BR solver, through
   `Beatnik_CreateBRSolver.hpp`, so each arm instantiates Canopy's whole pipeline
   — `TreeBuilder`, `TreePartitioner`, `CommunicationPlan`, `UpwardSweep`,
   `DownwardSweep`, `P2P` — in all of them. Adding a second `SolidHarmonic` arm
   is one line and that is the price of it.
6. **`FarFieldDiagnostics` is the third signature, and it is the only route by
   which any later task can read a Canopy number.** It is a Beatnik POD naming
   no Canopy type, declared in `Beatnik_FarFieldInterface.hpp` beside
   `FarFieldSolver` and **outside** `BEATNIK_ENABLE_CANOPY` so a `~canopy` build
   still compiles it. Every field traces to an accessor Canopy already exposes:

   - the maintenance action the evaluation took — a nested
     `enum class Maintenance { Setup, Migrate, Rebalance, Rebuild }` mirroring
     `Solver::MaintenanceAction` with the first-call `Setup` case added;
   - the global particle count Canopy holds, reduced from
     `num_local_particles()`;
   - the P2P pair fraction, per the next step;
   - `m2l_n_unique_ops()`, `m2l_op_cache_size()`, `m2l_op_keys_built_count()`,
     `total_fallback_pair_count()` and `total_m2l_pair_count()`, all off
     `Solver::downward()`;
   - the basis, the order and the softening **length** actually instantiated,
     since the conventions table requires every accuracy figure to name them.

   In a `~canopy` build the two evaluates throw and `diagnostics()` returns the
   default-constructed member.

   Add `const far_field_type& farField() const` to `BRSolverFMM`
   ([src/Beatnik_BRSolverFMM.hpp](../../src/Beatnik_BRSolverFMM.hpp)) in the same
   change. `_far_field` is private and `BRSolverBase`'s virtuals return `void`,
   so without it a test holding a `BRSolverFMM` cannot reach the diagnostics
   without naming a Canopy type. The consumers are **T4** steps 2 and 7, **T5**
   steps 1 and 5, and **T8** steps 1 and 2; settling the surface here is what
   keeps those three tasks from each reopening it.
7. **Compute the P2P pair fraction in the adapter — Canopy reports no such
   counter.** `P2P`'s public surface is `num_ghost_particles()`,
   `ghost_positions()` and `ghost_charges()` and nothing else
   (`canopy/src/Canopy_P2P.hpp:158-163`), and Canopy's own accuracy test settles
   for an `m2l_n_unique_ops() > 0` liveness guard
   (`canopy/tests/tstCartesianTaylorSolve.hpp:96-107`), which is weaker than
   **R1** requires. The ingredients are host-side and already public:
   `Solver::comm_plan().p2p_plan().neighbor_lists` is an
   `unordered_map<MortonKey, vector<MortonKey>>` over this rank's owned leaves
   (`canopy/src/Canopy_CommunicationPlan.hpp:116-134`, accessor at `:220`), and
   `Solver::builder().cells()` is a globally replicated `vector<CellInfo>`
   carrying `global_count` per cell
   (`canopy/src/Canopy_TreeBuilder.hpp:84-92`, accessor at `:187`). A leaf's
   particles live on exactly one rank after partitioning, so summing
   $n_T \sum_{S \in \mathrm{nbrs}(T)} n_S$ over owned leaves and reducing gives
   the global P2P pair count with no double counting; the fraction is that over
   the square of the global source count. It is a few thousand host-side lookups
   per evaluation, and it is the number every accuracy claim in **T4**, **T5**,
   **T6** and **T8** has to carry, so it is computed unconditionally rather than
   behind a diagnostic flag.
8. Every rank must enter every collective the same number of times per
   evaluation, **including a rank that owns zero sources**. Canopy has no test
   for a zero-particle rank ([canopy0.md](../../../canopy/tasks/canopy0.md) F5) and Beatnik's
   decomposition can produce one. Do not branch the collective sequence on a
   local count.
9. `#include <Canopy_Solver.hpp>` and every Canopy-typed member sit behind
   `BEATNIK_ENABLE_CANOPY`; the class must still compile, and its methods must
   still throw `std::runtime_error` naming the missing build option, in a
   `~canopy` build.
10. **The file header's claim that "Canopy has not been read while writing this
   header"** ([src/Beatnik_FarFieldInterface.hpp:19-21](../../src/Beatnik_FarFieldInterface.hpp#L19-L21))
   becomes false with this task and is deleted by it. The softened-kernel
   paragraph at [:36-41](../../src/Beatnik_FarFieldInterface.hpp#L36-L41) is
   **T3**'s to correct and must be left standing: rewrite the class body around
   it rather than clearing the header wholesale.

**Exit criterion:** `spack install` succeeds with `+canopy`, and succeeds again
with the Beatnik spec flipped to `~canopy`, the second build's `FarFieldSolver`
compiling and throwing `std::runtime_error` naming `+canopy` rather than failing
to compile. A `spack`-mode checkout has no `cmake -DBeatnik_ENABLE_CANOPY=OFF` to
reach for, so the `~canopy` half is checked by temporarily setting the
development environment's Beatnik spec to `~canopy`, running
`spack concretize -f && spack install`, confirming the guarded class, then
restoring `+canopy` and reinstalling. Only Beatnik's variant changes, so no
dependency rebuilds, and the committed
[systems/tuolumne/spack.yaml](../../systems/tuolumne/spack.yaml) snapshot is not
edited because the change is reverted. Further:
`grep -n "include.*Canopy_\|Canopy::" src/*.hpp src/*.in` names only
`Beatnik_FarFieldInterface.hpp`; and
`grep -n "CartesianTaylorBasis\|LaplaceKernel" src/Beatnik_FarFieldInterface.hpp`
shows a basis named on every one of the six dispatch arms. No behavioral claim is
made by this task — **T4** is where correctness is first checked.

**Met.** Both builds are green and both greps pass.

*What was verified.* `spack install` succeeds with `+canopy` — 29 CXX objects
from a cleaned build directory, `adaptive_mesh_bubble.cpp.o` among them — and
all six arms are genuinely emitted, not merely written: `nm -C` on the
installed `adaptive_mesh_bubble` finds 289 defined symbols for each of
`CartesianTaylorBasis<double, p, 3>` at $p = 0, 2, 3, 4, 5$ and 290 for
`LaplaceKernel<double, 3, 3>`. That is a consequence of a departure recorded
in the log: the arm is selected in `FarFieldSolver`'s **constructor** rather
than at the first evaluation, so the dispatch switch is instantiated even
though `BRSolverFMM`'s virtuals still throw and nothing yet calls an
evaluation. `spack install` succeeds again with the development environment's
Beatnik spec flipped to `~canopy`: the installed `Beatnik_Config.hpp` carries
`/* #undef BEATNIK_ENABLE_CANOPY */`, the binary holds **zero** `Canopy::`
symbols, and with a temporary explicit instantiation of
`FarFieldSolver<Kokkos::HIP, Kokkos::HIPSpace>` added to the example driver
every member instantiates — constructor, both evaluations, `diagnostics()`,
`params()` and `throwNoCanopy` — and the binary carries the
`std::runtime_error` text naming `Beatnik_ENABLE_CANOPY=ON, spack '+canopy'`.
The temporary instantiation and the spec flip were both reverted and `+canopy`
reinstalled; the committed
[systems/tuolumne/spack.yaml](../../systems/tuolumne/spack.yaml) snapshot was
not edited. `grep -n "include.*Canopy_\|Canopy::" src/*.hpp src/*.in` names
only `Beatnik_FarFieldInterface.hpp`, and
`grep -n "CartesianTaylorBasis\|LaplaceKernel" src/Beatnik_FarFieldInterface.hpp`
shows a basis named on all six arms.

*The exit criterion could not be met as written, and a one-line CMake fix was
needed.* `Beatnik_ENABLE_CANOPY` was set from `Canopy_FOUND` alone, so
`~canopy` — which passes only `-DBeatnik_REQUIRE_CANOPY=OFF` — installed a
`+canopy` binary in any environment that has Canopy installed, this one
included. The macro in [CMakeLists.txt](../../CMakeLists.txt) now keys
`Beatnik_ENABLE_*` off `Beatnik_REQUIRE_*`, whose default is still `_FOUND`.
The log records the diagnosis.

*What was **not** verified, and is not claimed.* **Nothing was run.** No
binary was invoked, no job was submitted, no test was executed, and no
tolerance, budget, speedup or accuracy figure is asserted anywhere in this
change. The adapter's arithmetic — the two contractions, the two prefactors,
the sign of Canopy's gradient, and above all the tag-reverse round trip — is
checked by **compilation only**. **T4** is where any of it is first executed
and where **R3**'s dropped-or-duplicated-source failure mode is actually
caught; the round trip's own validity check and the `FarFieldDiagnostics`
global particle count exist to make that cheap, but neither has ever been
exercised. The one measured number here is a compile cost: on tuolumne the six
arms take the project from 30 to 78 CPU-minutes over the same 29 translation
units, a factor of 2.6.

---

### T3 — `BRSolverFMM::computeInterfaceVelocity` — **DONE**

**Depends on:** T2.

**Fill in:** [src/Beatnik_BRSolverFMM.hpp](../../src/Beatnik_BRSolverFMM.hpp)
(`computeInterfaceVelocity` only; `computeSurfaceRieszScalar` is T7),
[src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp)
(the softened-kernel paragraph in the file header only — step 2 requires it and
the original list omitted it; the adapter's code is T2's and is not reopened),
[README.md](../../README.md).

**Reference:** the direct implementation this must agree with, step for step
([src/Beatnik_BRSolverDirect.hpp:105-165](../../src/Beatnik_BRSolverDirect.hpp#L105-L165))
— note it reallocates the output to `ownedVertexCount()` and zeroes it before
accumulating ([:111-115](../../src/Beatnik_BRSolverDirect.hpp#L111-L115)), calls
`quadrature.generate` itself ([:117-119](../../src/Beatnik_BRSolverDirect.hpp#L117-L119)),
and applies `br_sign/4\pi` once ([:125-127](../../src/Beatnik_BRSolverDirect.hpp#L125-L127));
the caller's contract ([src/Beatnik_ZModelSolver.hpp:219-224](../../src/Beatnik_ZModelSolver.hpp#L219-L224)),
which reallocates `vertex_dot` to the owned count and expects the prefactors
already applied.

**Do:**

1. Generate sources through the quadrature, call the adapter, write the
   `(N_owned, 3)` velocity. Overwrite, do not accumulate — the declaration says
   overwritten ([src/Beatnik_BRSolverBase.hpp:137-139](../../src/Beatnik_BRSolverBase.hpp#L137-L139)).
2. Correct the softened-kernel `@note` on `computeInterfaceVelocity`'s doc
   comment — **one** block, not two, and it is on the method rather than in the
   file header ([:104-109](../../src/Beatnik_BRSolverFMM.hpp#L104-L109); the
   second `@note` at [:111-114](../../src/Beatnik_BRSolverFMM.hpp#L111-L114) is
   about MPI, is accurate, and needs no change) — and the matching
   paragraph in `Beatnik_FarFieldInterface.hpp:37-44` (T2 rewrote the class
   body around it and moved it from `:36-41`). Both say the expanded
   kernel is the softened one, which is true under `CartesianTaylorBasis` and
   false under the solid-harmonic basis the parameter defaults to — so the
   statement needs the basis attached to it, not deleting. The claim that "an
   acceptance criterion tuned on the bare kernel is optimistic" near self-contact
   does not apply to a basis whose $w=r^2+b$ carries the blob at every order;
   replace it with what does bind, which is Taylor truncation at the accepted
   $R/w$, and point at this document.
3. Same correction in [README.md](../../README.md): whatever it says about the FMM
   path's accuracy must become the measured statement, and it must say the
   default `--br-approximation fmm`
   ([src/Beatnik_Params.hpp:107](../../src/Beatnik_Params.hpp#L107)) is not the
   validated path until **T5** has run.

**Exit criterion:** a two-step `--br-approximation fmm` run of
`examples/02_adaptive_mesh_bubble` at the milestone-0 configuration completes
without throwing, at 1 and 4 ranks, submitted as a batch script under
`scripts/tuolumne/` and read from its `.log`; and the same run with
`--br-approximation direct` still produces the checkpoint it produces today.
No accuracy claim — that is **T4**.

**Met.** All four launches through
[scripts/tuolumne/t3_fmm_velocity.flux](../../scripts/tuolumne/t3_fmm_velocity.flux)
at the milestone-0 configuration, level 3, 2 steps, `--source-quadrature
vertex`, on HIP. **FMM:** job `f3Zr7NFJbe2s`, `fmm_np1` and `fmm_np4` both
**rc=0**, each writing 3 checkpoint steps + latest; the run banner reads `BR
fmm, quadrature vertex, velocity full`, so the FMM path was genuinely taken,
and neither launch exited through `BEATNIK_NOT_IMPLEMENTED` nor through the
adapter's `~canopy` `std::runtime_error` — the two ways "fmm ran" can be a lie
if the build did not pick the change up. **Direct:** the same job's `direct_np1`
and `direct_np4` also rc=0, against the pre-change baseline job `f3ZqzxPyjYes`
run from the installed binary before any source was edited.

The direct half is **not** a bitwise result, and bitwise was never the right
yardstick: two runs of the *same* binary with the *same* command line differ by
1.2e-15 (np1) and 1.9e-15 (np4) of field RMS, so `h5diff`'s exact compare fails
on a rerun of an unmodified binary. Measured against that noise floor, baseline
vs. after is **1.5e-15 (np1) and 1.0e-15 (np4)** of field RMS — the same size at
np1 and *smaller* at np4 than the binary's own run-to-run spread, i.e. no
detectable change to the direct path. Step 0 is bitwise identical in every case.

One result beyond the criterion, and it is about **R3** rather than accuracy:
gid-matched across rank counts, the FMM's np1-vs-np4 spread is **2.224e-15** of
field RMS — *identical* to the direct solver's own np1-vs-np4 spread of
2.224e-15 — and the `/vertices/gid` sets match exactly at both rank counts. The
round trip therefore neither dropped nor duplicated a source at 1 or 4 ranks on
its first-ever execution.

**Not claimed.** No accuracy statement of any kind about the far field. At
`--icosphere-subdivisions 3` (642 vertices) with `ncrit = 64`, the README's own
liveness inequality ($N \gg 105\cdot\texttt{ncrit} = 6720$) puts this run two
orders below where the far field carries any of the field, so the solve is
expected to be all or nearly all P2P — an FMM that is a direct sum with FMM
bookkeeping around it. A passing T3 is therefore **not** evidence that the
expansion, the basis selector, the M2L path or the acceptance criterion is
correct, and the fmm-vs-direct agreement at this configuration was deliberately
not computed as a number so it cannot be quoted as one. No diagnostic was read
off `farField().diagnostics()`: nothing needed debugging, and the example driver
has no path that prints them. **T4** owns all of it.

---

### T4 — Unit test: the FMM velocity against the direct velocity, same state — **DONE**

**Depends on:** T3.

**Fill in:** `tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp` (new),
[tests/unit_tests/CMakeLists.txt](../../tests/unit_tests/CMakeLists.txt)
(`BEATNIK_UNIT_TEST_SOURCES`, [:42](../../tests/unit_tests/CMakeLists.txt#L42)),
`scripts/tuolumne/t4_fmm_vs_direct.flux` (new).

**The rank sweep needs a script of its own, and this is why.** Neither existing
path can run a `unit`-tier member at ranks 1-6. The tier registers every test at
exactly one rank
([tests/unit_tests/CMakeLists.txt:72](../../tests/unit_tests/CMakeLists.txt#L72),
`set(_beatnik_unit_ranks 1)`), so a `ctest` entry never sweeps; in `spack` mode
there is no build tree and therefore no `ctest` at all (CLAUDE.md "Build mode").
And the tier runner pins its allocation to one node
([scripts/tuolumne/unit_tests.flux:3](../../scripts/tuolumne/unit_tests.flux#L3))
while deriving its launch width from `BEATNIK_UNIT_RANKS` as
$\lceil n/4 \rceil$ nodes, so `BEATNIK_UNIT_RANKS=5` or `6` asks for two nodes
inside a one-node allocation. `t4_fmm_vs_direct.flux` therefore allocates two
nodes and loops ranks 1-6 over this one binary, in the shape
[scripts/tuolumne/t3_fmm_velocity.flux](../../scripts/tuolumne/t3_fmm_velocity.flux)
established. The tier runner is left alone: it stays the one-rank regression
path, and this member lands in it for free because it discovers its tests.

**Reference:** the tier's registration and its "self-validating, non-zero on
failure" contract
([tests/unit_tests/CMakeLists.txt:11-40](../../tests/unit_tests/CMakeLists.txt#L11-L40)),
the assertion helper (`tests/unit_tests/Beatnik_TestAssert.hpp`), and
`Beatnik_Test_TangentialRelaxation.cpp` as the tier's rank-count-aware precedent;
`Beatnik_Test_Milestone0Frozen.cpp`'s `makeParams` for a params set that has
already been measured against
([tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp:474-575](../../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp#L474-L575));
develop-canopy's `tests/tstFmmVsExact.hpp` for the shape of the comparison it
made.

**Do:**

1. Build a solver at the milestone-0 configuration, `setup()`, and advance a
   handful of steps with the **direct** solver so the state is a real one with a
   non-zero sheet strength. At step 0 the sheet strength is identically zero
   (`--initial-potential-strength 0`), so a step-0 comparison is vacuous — the
   test must assert the strength is non-zero before comparing.
2. **Choose the vertex count and `ncrit` together so the far field is live**,
   and assert that it is. At $\theta=0.3$ a 2-manifold's near field covers about
   105 occupied leaves, so the comparison needs $N/\texttt{ncrit}\gg105$; the
   default `ncrit = 64` gives 10 leaves at 642 vertices and 40 at 2562, both of
   which make this test a comparison of two direct sums that passes at any
   order. Use subdivision level 4 (2562 vertices) with `ncrit = 8` — 320
   occupied leaves — and **assert the measured P2P pair fraction is below a
   stated bound** rather than assuming it. Record the chosen pair and the
   realized fraction in the log; if no affordable pair clears the bound, that is
   the finding and **T5** inherits it. See
   [The far field has to be live to be measured](#the-far-field-has-to-be-live-to-be-measured).
3. Construct both BR solvers and evaluate **both on that same state**, then
   assert max relative and max absolute velocity error against a budget whose
   value and full qualification list are recorded in the log.
4. Three negative cases, because each proves a different thing and a comparison
   that has only ever seen agreeing data has not been tested:
   - **The order knob is live.** `order = 0` (monopole-only Taylor) must exceed
     the budget while the production order 3 passes, and `order = 2` must land
     between them — roughly an order above 3's error, which is the signature of
     the gradient truncating at $p$ rather than $p+1$. This is what makes the error a
     truncation rather than a bias, and it is the property the whole basis choice
     rests on.
   - **The basis selector is live.** `FarFieldBasis::SolidHarmonic` with
     `near_softening_factor = 0` must exceed the budget by orders of magnitude.
     A pass here means the adapter is not selecting the basis it says it is —
     see **R2**.
   - **The blob reaches the far field.** A second `BRSolverFMM` built at a
     different `eps` must produce a different FMM velocity on the same state.
     Under a bare-kernel far field it would not, which is the cheapest available
     proof that $b$ is inside $w$. It has to be a second solver rather than a
     perturbation of a live one: Canopy fixes the softening at `Solver`
     construction and pushes it into the M2L operator tables, so the adapter
     holds the value and **throws** if a later evaluation presents a different
     `blob()` — a guard, not a defect, and **T7** meets it too.
5. Rank counts 1-6, since that is the gate's sweep and Canopy's
   three-component gradient path is known wrong at exactly 4 under
   `LaplaceKernel` (`canopy/README.md:562-588`). If 4 ranks fails, that is the finding: record
   it, and do not widen the budget to accommodate it — see **R4**.
6. Include a variant where at least one rank owns zero sources, if the
   decomposition can be made to produce one at these vertex counts; if it
   cannot, say so in the log rather than leaving the case silently uncovered.
7. Assert the FMM result is finite everywhere, and that the global source count
   Canopy reports equals the global owned vertex count — the cheap independent
   check on the round trip (**R3**).
8. **Assert the adapter's softening and near-field floor, which is the half of
   T1's exit criterion that first becomes runnable here.** Assert
   `diagnostics().softening` equals $\sqrt{\texttt{blob()}}$ and is strictly
   positive — a value of $-1$ is Canopy's auto-softening sentinel and a
   different kernel — and that `params().near_softening_factor`
   ([src/Beatnik_FarFieldInterface.hpp:516](../../src/Beatnik_FarFieldInterface.hpp#L516))
   is 0. The `FmmConfig` members themselves are **not** observable from a test
   by construction: `FmmConfig` is a Canopy type and the conventions table
   confines those to the adapter, and `FarFieldDiagnostics`
   ([:205-275](../../src/Beatnik_FarFieldInterface.hpp#L205-L275)) exposes
   `softening` but carries no `near_softening_factor` field. So the assertion is
   on the adapter's inputs and on the softening it reports, and an evaluation
   that completes at all is what shows the config it built was accepted.
9. **Close the `~canopy` instantiation gap permanently**, with an explicit
   instantiation of `FarFieldSolver` in this test source, guarded to the
   `~canopy` build. `Beatnik_CreateBRSolver.hpp` preprocesses out the
   `new BRSolverFMM<...>` line, which is the class's only construction site, so
   a `~canopy` build instantiates `FarFieldSolver` nowhere and the guarded half
   of the header is parsed but never definition-checked. One line in the tree
   that instantiates it is what keeps that half from rotting silently the next
   time someone edits it.

**Exit criterion:** `flux batch scripts/tuolumne/t4_fmm_vs_direct.flux` reports
`Beatnik_Test_FmmVsDirect` green at every one of ranks 1, 2, 3, 4, 5 and 6,
against a budget recorded with its qualification list, and
`flux batch scripts/tuolumne/unit_tests.flux` reports the tier green at one
rank; and each of the three negative cases fails, naming its own reason — the
order case naming truncation, the basis case naming the basis in force, the blob
case naming the softening length — rather than merely exiting non-zero.

**Met.** `Beatnik_Test_FmmVsDirect` is green at **every one of ranks 1, 2, 3, 4,
5 and 6** through
[scripts/tuolumne/t4_fmm_vs_direct.flux](../../scripts/tuolumne/t4_fmm_vs_direct.flux),
and the `unit` tier is green at one rank through
[scripts/tuolumne/unit_tests.flux](../../scripts/tuolumne/unit_tests.flux). The
gate is unchanged: five `regression` members, 60 launches.

**The far field was live, and that is asserted rather than assumed.** At 2562
vertices with `ncrit = 8` the realized `p2p_pair_fraction` is **0.2537** against
a compiled bound of 0.75 and against 1.0 for a solve with no far field at all,
with 130968 M2L pairs and **zero** operator-table fallbacks, so three quarters
of the pairs go through the expansion and the number below is one code path
rather than a mixture (**R6**). The measured error is **5.008e-4** of the direct
field's own scale — `max|u_fmm − u_direct| = 2.5804e-6` over
`max|u_direct| = 5.15236e-3` — against a **2.0e-3** budget carrying the full
qualification list (velocity/gradient, milestone-0 icosphere at level 4 after
five direct steps, ranks 1-6, `CartesianTaylor`, `order` 3, `ncrit` 8,
`max_depth` 10, `mac_theta` 0.3, `softening` 0.025, `near_softening_factor` 0,
P2P fraction 0.2537). That budget is **not** $\tau_A$; **T5** measures that.

**The rank sweep found nothing, which is the result.** The relative error spans
5.00728e-4 to 5.00934e-4 over the six rank counts — a spread of
$4.1\times10^{-4}$ of the error itself, which is the *same order* as the
run-to-run difference the same binary shows on one rank count
($1.1\times10^{-4}$, measured against the earlier sweep), so the rank count
contributes nothing detectable. `global_particle_count` is exactly 2562 at every
one, so the round trip neither dropped nor duplicated a source (**R3**). **R4
did not fire**: np=4 is 5.00822e-4, inside that band and 3e-9 above np=1's.
That bounds Canopy's np=4 gradient defect further at this basis, order and
softening; it does not retire it, and 4 stays in the sweep.

**All three negative cases fail, each naming its own reason.** `order = 0` gives
0.407338 (204x the budget) and `order = 2` gives 5.9658e-3, so the three orders
are strictly ordered with $e_2/e_3 = 11.91$ against Canopy's own 12.7 — the
signature of the gradient truncating at $p$ rather than $p+1$, i.e. a truncation
and not a bias. `SolidHarmonic` at `near_softening_factor = 0` gives 4.66e-3, so
it misses the budget and sits 9.3x above the production arm *at the same order*.
Doubling the softening length moves the FMM velocity by 0.139041 relative while
it still tracks the direct solver *at that softening length* to 5.439e-4 — the
half that proves $b$ is inside $w$ — and the adapter's softening-stability
guard threw as designed. Steps 7 and 8 passed: finite everywhere, the global source count
equals the global owned vertex count, `diagnostics().softening` equals
$\sqrt{\texttt{blob()}}$ and is strictly positive rather than Canopy's $-1$
sentinel, and `params().near_softening_factor` is 0.

**One prediction in this document did not survive contact, and it is R2's.** The
test first required the solid-harmonic arm to exceed the budget by 100x, from
**R2**'s "tens of percent"; it measured 4.66e-3 and failed that one check at
every rank count with everything else green. The prediction describes a
*self-contacting* sheet. Here every accepted far-field pair sits at
$R\approx0.11$-$0.22$ against $\sqrt b = 0.025$, so the bare and softened
kernels differ by $\tfrac32 b/R^2$ — 2-8% on the closest accepted pairs, less
beyond — and a whole-field separation of about a decade is what that predicts.
The **test's basis-separation constant** was re-derived accordingly (a 5x ratio
against the production arm, plus the independent requirement that the arm miss
the budget); `kVelocityBudget` was not touched. **R2** should be read as a
roll-up statement, and **T5** should scan a deformed state if it wants the
figure R2 is actually about.

**Not verified.** No trajectory claim of any kind: this compares a *single
evaluation* on a state the direct solver produced, which is claim A's shape and
**not** claim B's — **R7** is untouched and **T6** owns it. Nothing about the
Riesz scalar (**T7**), which still throws. No $\tau_A$ and no parameter scan
(**T5**); the 5.008e-4 above is one point, at one order, on one undeformed
geometry five steps off its initial condition, and is not a statement about
achievable far-field fidelity. Nothing about `max_depth`: the fallback count was
0, so **R6** has no evidence to act on and the value was not revised. Nothing
about the remeshing path — connectivity is frozen, so the adapter's fallback to
`setup()` on a changed source set never fired. Nothing about the maintenance
histogram (**T8**): every arm builds a fresh solver and evaluates once, so the
action was `Setup` every time. And the `~canopy` branch of this test — the
explicit instantiation plus its throw assertions — is compiled-in but
**unexecuted**: this machine builds `+canopy` and no `~canopy` build was made
for this task.

**A rank owning zero sources is not reachable from the mesh decomposition** at
this vertex count (minimum owned counts 2562, 1233, 807, 595, 469, 386 at ranks
1-6), so the test constructs the case at the adapter's interface instead: the
last rank ships its rows to rank 0 and evaluates on an empty source list. Every
rank returned, the global count stayed 2562, and rank 0's field moved by
2.5e-16 to 8.0e-16 relative — round-off.

---

### T5 — Measure the achievable far-field fidelity, and publish it — **NOT STARTED**

**Depends on:** T4. Canopy's derivative ladder is validated at $|k|=2p$ through
$p=3$, so the production order and everything below it are unblocked
([Current state](#canopy)). **Scan points at $p\ge4$ are not**: $p=4$ reaches
$|k|=8$ and Canopy's oracle stops at 6. Those points may be measured and
reported — the arithmetic is very likely fine and the curve is worth having —
but **none may be adopted as the production order** until Canopy extends the
oracle to that degree, because a silent recurrence error there would present as
exactly the plateau **R1** describes and be attributed to truncation. Step 6
below states the consequence.

**This task produces every number later tasks key off.**

**Fill in:** a measurement driver under `tests/regression_tests/` registered in
the "Measurement drivers — IN NO TIER" section
([tests/CMakeLists.txt:536-572](../../tests/CMakeLists.txt#L536-L572)); a batch
script under `scripts/tuolumne/`;
[add-canopy-progress-log.md](add-canopy-progress-log.md);
[README.md](../../README.md).

**Reference:** the convergence model this must confirm or correct, with its
constants ([canopy-kernel-rec.md](../../../canopy/tasks/canopy-kernel-rec.md), "Convergence per DOF"
and "Memory"); the acceptance predicate that sets $R/w$
(`canopy/src/Canopy_CommunicationPlan.hpp:338-352`); the reference treecode's own
accuracy at the same two mesh sizes ([treecode.md](../treecode.md) §1); the
measurement-driver loop's rule that it appends to **neither** manifest
([tests/CMakeLists.txt:548-556](../../tests/CMakeLists.txt#L548-L556)); M0-D1's
ladder, which is the instrument step 7 reuses
([milestone0-progress-log.md](../milestone0-progress-log.md)).

Step 7 rebuilds none of M0-D1's machinery. Three in-tree artifacts are what it
reuses, and each carries a property that is expensive to rediscover:

- [tests/regression_tests/Beatnik_Test_Milestone0Run.cpp](../../tests/regression_tests/Beatnik_Test_Milestone0Run.cpp) —
  the measurement driver in the no-tier section, taking `argv[1..3]` as level,
  steps and checkpoint-every. It pins
  `p.zmodel.br_approximation = BRApproximation::Direct`
  ([:184](../../tests/regression_tests/Beatnik_Test_Milestone0Run.cpp#L184)),
  which is the one line a T5 driver parameterizes rather than copies.
- [tests/regression_tests/milestone0_ladder.py](../../tests/regression_tests/milestone0_ladder.py) —
  the ladder itself: `pair --run DIR --ref DIR [--label L] [--json OUT]`, taking
  a gold directory of `.npz` or a second run directory of `.h5` on the right.
  From one comparator run per step at the tightest rung it derives a **lower
  bound** on the first failing step, then confirms that step with real
  invocations at the rung before reporting it. It reports per field, and the two
  right-hand sides are **not over the same field set** — `sheet_vector` is absent
  from the reference `.npz`, so a Beatnik-vs-Beatnik horizon is measured over a
  strictly larger set than a Beatnik-vs-Python one and the two numbers are not
  directly comparable. Step 7 reports both, so which is which has to be stated.
- [scripts/tuolumne/milestone0_divergence.flux](../../scripts/tuolumne/milestone0_divergence.flux) —
  the budget-guarded sweep: measured per-row wall estimates, cheapest-first
  ordering, a guard that **skips** a launch whose estimate does not fit the
  remaining budget rather than starting it and having it killed at the wall, and
  `BEATNIK_M0_MODE=probe` (25 steps rather than 2000) as how those estimates were
  measured in the first place.

**Do:**

1. Scan, on real milestone-0 states at both subdivision levels: `order` over the
   dispatched set, `mac_theta`, `ncrit`, `max_depth`, and both values of
   `basis`. Report max relative and max absolute velocity error against
   `BRSolverDirect` on the **same** state, and the fraction of pairs Canopy
   handled in P2P. **Every point carries its P2P fraction or it is not a
   point**: at the default `ncrit` neither milestone-0 level has a live far
   field, so a scan that does not vary `ncrit` measures the direct sum at every
   order. `ncrit` is therefore a scan axis and not a fixed background, and the
   scan must state for each level the `ncrit` at which the far field first
   carries a stated fraction of the field.

   **The whole scan runs in one launch against one shared state**, in the shape
   [tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp:607-630](../../tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp#L607-L630)
   already uses: the spin-up is done once and every arm is evaluated against
   that one state through a single helper
   ([:708](../../tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp#L708)). That makes
   arm-to-arm differences **exact** within the launch. It is not a convenience:
   the spin-up is downstream of five timesteps and is not bitwise reproducible,
   so a figure taken in one launch and compared against one taken in another
   carries a floor of about $10^{-4}$ **of the error itself** — the same binary
   at np=1 measured $5.00765\times10^{-4}$ on one job and
   $5.00819\times10^{-4}$ on another. The scan resolves differences smaller than
   that: between adjacent `max_depth` values, and between neighbouring
   `mac_theta`. Any figure that does have to cross launches — a rank sweep, or
   step 7's repeats — carries that floor and states it beside itself.
2. **Report potential and gradient error separately at every point.** They
   differ by a full order at fixed `order` — the target-side L2P truncation
   explained in [The order that reaches the target](#the-order-that-reaches-the-target)
   — and Beatnik reads only the gradient. A scan that reports one number per
   point cannot be read against the reference's documented figure, which is a
   source-side-only velocity and so corresponds to the *potential* column.
3. **Read the scan as a scan.** Under `CartesianTaylor` the gradient error
   should fall with `order` at the rate
   $\epsilon_{\rm grad}\approx(\theta/2\sqrt3)^{p}$ predicts — confirmed within
   20% at three points on a volumetric cloud, see
   [The order that reaches the target](#the-order-that-reaches-the-target) — and
   then flatten into Canopy's own floating-point floor. The `SolidHarmonic`
   comparison is at **one** order, the production order 3, since that is the only
   solid-harmonic arm **T2** dispatches.

   **The discriminator is a curve against a fixed point, not a gap measured in
   decades.** The bias is in the kernel rather than in the truncation, so no
   order rescues the solid-harmonic arm: what proves the selector is live is
   that the `CartesianTaylor` curve keeps *falling* with `order` at the model's
   rate while that single point does not move with `order` at all. The two
   landing on top of each other is the alarm. Measuring the solid-harmonic
   curve's own shape would need dispatch arms **T2** does not build — one line
   each, and not required for this scan.

   **Do not read the size of the gap as the signal on this geometry.** At
   milestone-0's smooth sphere it is about a decade: **T4** measured
   $4.66\times10^{-3}$ against $5.008\times10^{-4}$ at $p=3$, a factor of 9.3,
   while `CartesianTaylor` at $p=2$ gives $5.9658\times10^{-3}$ — *above* the
   solid-harmonic point. So a gap-threshold test is satisfied by an ordinary
   $p=2$ point and would read as a broken selector. A separation of tens of
   percent is a self-contact figure (**R2**), reachable only where accepted
   separations approach $\sqrt b$, and no milestone-0 configuration in this tree
   reaches it. Report the smooth-sphere separation, and record that the
   self-contact figure is unmeasured here rather than chasing it. Record which
   regime each observed level is in.
4. Confirm or correct the error model in **Problem**. Its constant $c\approx1$
   was measured on a volumetric cloud at idealized equal-cell separations; the
   realized $R/w$ distribution on a thin bubble surface with depth-mismatched
   cell pairs is what actually sets the rate, and no Canopy test has ever
   measured accuracy on a non-volumetric distribution
   ([canopy0.md](../../../canopy/tasks/canopy0.md) F4). A rate differing from
   the model in *exponent* rather than in constant would mean the sheet's
   geometry changes which term dominates, and is the finding. A measured rate far *better* than
   the model most likely means the far field never engaged, which at these vertex
   counts is easy to hit by accident and which [treecode.md](../treecode.md) §1
   documents as the same trap on the treecode side — check the P2P pair fraction
   before believing it.
5. Measure the operator table: the realized key count `n_unique_ops`, the bytes
   it occupies, and `total_fallback_pair_count()`, at each `max_depth` and
   `order` in the scan. `CartesianTaylorBasis` keys carry the tree level, so the
   count scales with occupied depth against a 32768-key cap; a non-zero fallback
   count means some pairs took different arithmetic and the accuracy number is
   a mixture. This is **R6**.
6. Record $\tau_A$ — the best max relative velocity error achieved at an
   affordable order and P2P fraction, with its full qualification list — in the
   log, and publish the validated parameter set and the achieved fidelity in
   README. State the production `order` and why it was chosen over the next one
   up and the next one down. If the measured production order differs from T1's
   compiled default of 3, changing that default is part of this task and README
   moves with it — **except upward past 3**, which needs Canopy's oracle
   extended to $|k|=2p$ first. If the scan says $p=4$ is wanted, record the
   figure, leave the default at 3, and say in the log that the raise is pending
   that extension rather than pending a Beatnik change.
7. Measure the **divergence horizon** claim B needs, by running the milestone-0
   configuration to 2000 steps. This is M0-D1's measurement with a
   per-evaluation perturbation as the seed instead of a one-ulp initial
   condition, and it is what turns claim B's envelope into a measured number.
   Report the volume-drift series alongside, since claim B asserts against
   `kRefVolumeDrift` and needs a measured bound.

   **The published envelope is FMM-driven against the in-tree Python gold set**,
   because that is the quantity **T6** step 3 asserts, and an envelope measured
   against anything else does not transfer to it. A direct-driven 2000-step run
   is reported **alongside** as attribution: it separates the FMM's
   per-evaluation perturbation from the Beatnik-versus-Python drift that is
   present either way. The two ladders nearly coincide and the reason is already
   measured, so it need not be re-derived — the direct path tracks the gold set
   to $8.5\times10^{-13}$ at step 2000
   ([milestone0-progress-log.md:320-332](../milestone0-progress-log.md)), so at
   any rung at or above $10^{-10}$ they agree. Note that the two right-hand
   sides are not over the same field set; the **Reference** entry on the ladder
   says which.

   **R8** is unchanged and binds here: more than one FMM-driven run, the
   envelope set from the *earliest* observed horizon with margin, and the
   run-to-run spread recorded separately from the direct-versus-FMM gap.

   **Budget the sweep before submitting it.** The FMM-driven per-step cost is
   **unmeasured** — **T8** owns it — so it cannot be extrapolated from M0-D1's
   direct figures (L3 HIP np1 $0.005385$ s/step, np4 $0.014058$; L4 HIP np1
   $0.008792$, np4 $0.019211$; L3 SERIAL np1 $0.043030$, np4 $0.021891$; L4
   SERIAL np1 $0.644068$, np4 $0.187007$). Probe at a reduced step count first
   and set the per-row estimates from that measurement, in the shape
   [milestone0_divergence.flux](../../scripts/tuolumne/milestone0_divergence.flux)
   already uses, rather than submitting a blind 2000-step sweep into `pdebug`'s
   one-hour cap. That script's own rule applies here: a sweep that does not fit
   is a **finding for the log**, not a reason to lengthen the walltime, change
   queue, or quietly run fewer steps.

**Exit criterion:** the log carries the full scan with every entry's
qualification list, a separate potential and gradient column at every point, and
a P2P pair fraction at every point; a stated $\tau_A$ and production parameter
set, including the `ncrit` at which each subdivision level first has a live far
field; the operator key-count and fallback-count table from step 5; and the
divergence-horizon ladder and volume-drift bound from step 7. README carries the validated parameter set
and the achieved fidelity for the gradient. The task is complete whichever value
$\tau_A$ takes — if it is above $10^{-3}$ at every affordable order, that is the
finding, and **X1** is what it implies.

---

### T6 — The two milestone-tier FMM members — **NOT STARTED**

**Depends on:** T5 (for $\tau_A$, the horizon envelope, the volume-drift bound
and the per-level `ncrit`) and T4 (for the comparison harness). No upstream gate:
Canopy's derivative ladder is validated at $|k|=6$, which is $2p$ at the
production order.

**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp` and
`Beatnik_Test_Milestone0FmmL4.cpp` (new),
[tests/CMakeLists.txt](../../tests/CMakeLists.txt)
(`BEATNIK_MILESTONE_TEST_SOURCES` at [:430-435](../../tests/CMakeLists.txt#L430-L435)
and the two `_beatnik_args_<stem>_abs` / `_rel` pairs),
[scripts/tuolumne/run_milestone.flux](../../scripts/tuolumne/run_milestone.flux)
(the walltime), [CLAUDE.md](../../CLAUDE.md) ("Minimum test set", the tier's member
count and launch count), [README.md](../../README.md).

**Reference:**

- The existing members, whose structure these mirror:
  [tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp](../../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp)
  in full — its per-level literal blocks (`:295-403`), `goldForStep`'s
  suffix-based lookup and why it is not built from a time (`:417-440`),
  `runComparator`'s four distinguished outcomes (`:451-468`), `makeParams`
  (`:474-575`), the owned-count partition check (`:682-712`), the volume-drift
  check against the reference's own series (`:760-788`), `compareStep`
  (`:795-814`), the step loop (`:837-900`), the negative case and why exit 2
  must not be accepted (`:926-945`), and the cost report (`:952-987`).
  [Beatnik_Test_Milestone0FrozenL4.cpp](../../tests/regression_tests/Beatnik_Test_Milestone0FrozenL4.cpp)
  is the whole second member: 46 lines, of which three carry content.
- The tier's registration loop, which keys arguments by source stem and
  `FATAL_ERROR`s on a source with no argument list
  ([tests/CMakeLists.txt:475-535](../../tests/CMakeLists.txt#L475-L535)), and the
  tier's rank set as a property of the tier
  ([:437-439](../../tests/CMakeLists.txt#L437-L439)).
- The measured cost that constrains this: 37.25 minutes for the tier's current
  eight launches under `# flux: -t 60m`, `-q pdebug`
  ([scripts/tuolumne/run_milestone.flux:5-7](../../scripts/tuolumne/run_milestone.flux#L5-L7)).

**Do:**

1. **Two source stems, one body**, exactly as the existing pair: the L4 file
   `#define`s `BEATNIK_M0_LEVEL 4` and `#include`s the L3 file. Every per-level
   literal is selected by `BEATNIK_M0_LEVEL` and re-derived for that level.
   **Do not transfer a literal between levels** — the entity counts, the two
   carried scalars, the polyhedral deficit, the final `time` and the 81-entry
   reference volume-drift series all differ, and the existing members' blocks
   are the values to reuse (they are that level's, already re-derived).
2. **Claim A.** Use T5's per-level `ncrit`, and report the realized P2P pair
   fraction in each member's log so a reader can see how much far field the
   claim actually exercised. At 2562 vertices there is an `ncrit` that makes it
   live; **at 642 there is none** — the near field covers about 105 occupied
   leaves at $\theta=0.3$ and the level has at most 80 even at `ncrit = 8` — so
   the L3 member's claim A is largely a P2P comparison and must say so on the
   assertion rather than present itself as a far-field bound. It is still worth
   asserting: it is the round trip, the tag handshake and the contraction under
   test, all of which are rank-count-dependent and none of which the L4 member
   covers at L3's decomposition. The far-field accuracy claim rests on L4.
   Drive the trajectory with `BRSolverDirect` — the run must stay
   bit-identical to the existing member, so the 81 gold comparisons run at
   `--rtol 1e-10 --atol 1e-12` unchanged and prove the trajectory is the right
   one. At each of the 81 checkpointed steps, additionally evaluate the FMM
   velocity **on that same state** and assert max relative error
   $\le\tau_A$. Report the per-step value at 17 digits so the series is in the
   log without re-running.
3. **Claim B.** Then run 2000 steps FMM-driven and assert the four properties
   **Approach** names: the run reaches step 2000; every velocity is finite; the
   entity counts never change (both paths of the existing member — Tessera's
   global counts every step and an `MPI_Allreduce` over owned counts at every
   compared step); and the volume drift tracks `kRefVolumeDrift` within T5's
   measured bound. Then compare against the gold set as a **ladder**, reporting
   the first failing step at each rung, and assert only that each rung's horizon
   is no *earlier* than T5's measured envelope. Do **not** assert a rung passes
   at step 2000; at $\tau_A\approx10^{-3}$ none will, and a rung that does is
   loose enough to be meaningless.
4. **A stop is a reported stop step, never a shorter pass**, and a comparator
   exit of 2 (could not load) is never conflated with 1 (compared and
   disagreed). Both properties are the existing member's and both are load-bearing
   here: claim B's whole purpose is to notice the FMM destabilizing the physics.
5. Keep the existing member's negative case — the final state against the step-0
   gold, which must exit exactly 1 — and add one for claim A: a state
   deliberately perturbed by more than $\tau_A$ must fail the same-state
   comparison. Add one for claim B's horizon: an artificially early horizon
   envelope must fail, so the envelope assertion is known to be live rather than
   trivially satisfied.
6. Register both stems with their own gold directory, add the two argument-list
   pairs, and **raise the runner's walltime**: the tier goes from 2 members and
   8 launches to 4 and 16, and each new launch runs two 2000-step trajectories
   plus 81 extra FMM evaluations — roughly twice an existing launch, so on the
   37.25-minute measurement the tier lands near two hours. Measure the tier run
   and set `-t` from the measurement, not from an estimate. If the honest number
   exceeds what `-q pdebug` allows, changing the queue is part of this task and
   must be stated in the runner's header comment.
7. Update CLAUDE.md's "Minimum test set" tier paragraph with the new member
   count and launch count, and state explicitly that **the gate is unchanged** —
   still five `regression` members and 60 launches.

**Additional information needed:** $\tau_A$, the horizon envelope and the
volume-drift bound. All three come from **T5**, and none is invented here. If
$\tau_A$ lands above $10^{-3}$, this task still lands — it compiles in the
measured $\tau_A$, and README and the log state plainly that the
reference-treecode-parity claim is pending **X1**. It does not loosen the number
silently and it does not wait.

**Exit criterion:** `ctest -L milestone -R Milestone0Fmm` passes at ranks 1 and 4
on SERIAL and HIP, and
`flux batch scripts/tuolumne/run_milestone.flux` reports all 16 launches green
inside its walltime; each member's log carries the 81-entry claim-A error series
and claim B's per-rung horizon table; and all three negative cases fire — the
final state against the step-0 gold exits exactly 1, the perturbed-state claim-A
case fails naming $\tau_A$, and the artificially early horizon envelope fails
naming the rung it was early on.

---

### T7 — The Riesz-scalar path — **NOT STARTED**

**Depends on:** T3.

**Fill in:** [src/Beatnik_BRSolverFMM.hpp](../../src/Beatnik_BRSolverFMM.hpp)
(`computeSurfaceRieszScalar`),
[src/Beatnik_FarFieldInterface.hpp](../../src/Beatnik_FarFieldInterface.hpp) (the
dot contraction), a case added to **T4**'s test.

**Reference:** the direct implementation
([src/Beatnik_BRSolverDirect.hpp:184-241](../../src/Beatnik_BRSolverDirect.hpp#L184-L241)),
including that it is **not** multiplied by `br_sign` while the velocity is — an
asymmetry reproduced deliberately ([:203-208](../../src/Beatnik_BRSolverDirect.hpp#L203-L208));
`generateGradient` and the two state models' different expressions for $G$
([src/Beatnik_SourceQuadrature.hpp:256-290](../../src/Beatnik_SourceQuadrature.hpp#L256-L290));
the caller, which is collective and reached the same number of times on every
rank ([src/Beatnik_ZModelSolver.hpp:248-257](../../src/Beatnik_ZModelSolver.hpp#L248-L257));
that this is a **second** `solve()` over the same tree with different charges,
which Canopy supports (`Canopy::Solver::solve`).

**Do:** contract $\Psi = -\operatorname{tr}T$ with the $-1/4\pi^2$ prefactor and
no `br_sign`. Reuse the tree and the round trip from the velocity evaluation
where the state has not changed between them; if that reuse is not safe, say why
in the log and pay the second round trip rather than reusing a stale partition.
There is no gold file for this combination and none can exist — the reference
raises for `treecode` + `surface-riesz` (`mesh_solver.py:605`), recorded as risk
R5 in [framework.md](../framework.md) — so `BRSolverDirect` is the reference,
which is what the ladder does everywhere else.

**Exit criterion:** **T4**'s test gains a Riesz-scalar case that passes at ranks
1-6 against `BRSolverDirect` on the same state, at a budget recorded with its
qualification list; and a `--bernoulli-scalar-mode surface-riesz
--br-approximation fmm` two-step run completes at 1 and 4 ranks.

---

### T8 — Cost: the maintenance policy and the actual speedup — **NOT STARTED**

**Depends on:** T4.

**Fill in:** a batch script under `scripts/tuolumne/`;
[add-canopy-progress-log.md](add-canopy-progress-log.md);
[README.md](../../README.md) (and its "Future Optimizations" section only if the
user approves an entry).

**Reference:** what `migrate()` actually costs, per call
([canopy0.md](../../../canopy/tasks/canopy0.md) F3(b)); the prediction that a deforming surface picks
`Rebalance` rather than `Migrate` essentially every stage, and that a plan
rebuild is a serial host-side dual-tree traversal executed on every rank (F3(c),
citing `canopy/src/Canopy_CommunicationPlan.hpp:549-665`); develop-canopy's
measured action histogram — all 14 `auto_maintain` calls of a five-step run
returned `Migrate`, and a 1400-step roll-up run returned
`Migrate=1180 Rebalance=3019 Rebuild=0` (`tasks/integrate_canopy.md` row AM and
`tasks/fmm_premature_nan.md` Validation, on that branch); the profiling
convention `BEATNIK_SCOPED_TIMER_DETAILED` and the action-histogram destructor
(`src/FmmBRSolver.hpp:122-140`, `:669-680` on that branch).

**Do:**

1. Report the `MaintenanceAction` histogram per run, and the wall-clock split
   across pack, forward distribute, forward migrate, `auto_maintain`, `solve`,
   contract, reverse distribute, reverse migrate, scatter. Nine of these per
   timestep is the cost model that decides whether this path is worth running.
2. Report how much of `solve` is operator construction, per maintenance action.
   The cache does **not** amortize here and the measurement is of how much that
   costs, not of whether it happens: `CartesianTaylorBasis` sets
   `key_needs_level = true`, every maintenance path recomputes the root box from
   the particles, and `set_root_half_width` then clears the entire operator
   cache — so every `Migrate`, `Rebalance` and `Rebuild` rebuilds the whole
   table. Canopy measured zero keys retained across 336 builds on a moving
   distribution
   ([cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md)
   T5 and R6). Report `m2l_op_keys_built_count()` per build against
   `m2l_op_cache_size()`, so the log states the retention on **Beatnik's** tree
   rather than inheriting Canopy's; a non-zero retention here would mean
   Beatnik's root box is not drifting, which is itself worth knowing. This
   share, times nine evaluations per timestep, is what decides whether the far
   field is affordable at all, and if it dominates then the fix is Canopy's
   (out of scope) and belongs in the log as a sized request rather than as a
   Beatnik workaround.
3. Measure `fmm` against `direct` wall-clock per step at several vertex counts,
   and report the **P2P pair fraction** alongside — with
   `near_softening_factor = 0` the near field is set by `ncrit` and the MAC
   rather than by a softening floor, so the fraction is a property of
   `mac_theta` and the tree and is the thing that trades against `order`. State
   the crossover vertex count, or state that there is none in the measured range.
4. Do not change the maintenance policy inside this task. If the measurement
   says `auto_maintain` is the wrong choice, that is a finding and a follow-up
   task, not a drive-by edit.

**Exit criterion:** the log carries the action histogram, the per-phase
wall-clock split, the operator-construction share from step 2 and the
`fmm`-versus-`direct` per-step comparison with the P2P pair fraction at each
size, all at stated vertex counts, rank counts, backends, basis and order;
README states the measured speedup and the vertex count above which `fmm` is
faster, or states that none was found.

---

### X1 — Conditional external dependency: a geometrically-converging basis — **NOT STARTED, NOT IMPLEMENTED HERE**

**Depends on:** T5, which is what decides whether it is needed at all.

**On the current estimate this task is not expected to fire.** It is recorded so
the fallback is named rather than re-derived under pressure, and so a session
reading a disappointing **T5** result knows where the work lives.

**The trigger.** **T5** reports the `order` needed for $\tau_A\le10^{-3}$ at an
acceptable P2P fraction, together with the realized operator-key count. This task
fires if either bound is exceeded: the order needed is outside the dispatched
set and extending it is unaffordable, or the key count at that order and
`max_depth` pushes `total_fallback_pair_count()` off zero because
`CartesianTaylorBasis` keys carry the tree level against a 32768-key cap, which
Canopy has already been measured at 78% of. On the gradient a Cartesian-Taylor
expansion buys $\log_{10}(2\sqrt3/\theta)$ decades per order — 1.06 at Beatnik's
`mac_theta = 0.3` and 0.47 at standard admissibility — while its DOF count grows
as $\binom{p+3}{3}\sim p^3/6$, so the basis is cheap in the band this work
targets and expensive an order of magnitude below it
([canopy-kernel-rec.md](../../../canopy/tasks/canopy-kernel-rec.md),
"Convergence per DOF").

**Where the work would live:** a black-box (Chebyshev interpolation) basis in
Canopy, on the same far-field contract, designed in
[canopy-bbFMM.md](../../../canopy/tasks/canopy-bbFMM.md). It converges geometrically in the
interpolation order rather than algebraically, and softening helps rather than
hurts — $b>0$ moves the kernel singularity off the real axis, which can only
increase the convergence rate. Its cost is memory: the per-key operator is
1.1 MB at $n=6$ against `CartesianTaylorBasis`'s 9.8 KB at $p=4$, so it is only
representable in the compressed shared-basis form, which is why it is its own
design and not a variant of this one
([canopy-kernel-rec.md](../../../canopy/tasks/canopy-kernel-rec.md), "Memory").

**What Beatnik requires of it, and what it does not.** No interface change:
`FmmConfig`, `setup`, `solve`, `auto_maintain` and the gradient's shape and sign
all stay, and the basis is selected through the same template parameter, so the
only Beatnik file that moves is **T2**'s adapter and the only Beatnik change is
one enumerator in `FarFieldBasis` and one more dispatch arm. Beatnik does not
care how the basis is built.

**Exit criterion**, which is the acceptance test in Beatnik's terms and the only
part of this task Beatnik owns: **T4**'s test passes at ranks 1-6 with
$\tau_A\le10^{-3}$ at a P2P pair fraction and an operator-table size **T8**
reports as affordable, and **T6**'s claim A passes at that $\tau_A$ at all 81
states at both levels. A route that lands above $10^{-3}$ has not cleared it —
record the achieved figure and what binds it, and do not widen $\tau_A$ to
accommodate it, because $10^{-3}$ is the reference implementation's own number
and not a Beatnik preference.

## Known risks

**R1 — a truncation plateau is read as a hard floor, or the reverse.** Under
`CartesianTaylorBasis` the error is a truncation and `order` is the knob, which
is the opposite of what a bare-kernel far field does; both nonetheless flatten
eventually, one into Canopy's floating-point floor and one into a kernel bias.
**Presents as:** an error curve that stops falling with `order`. **Distinguished
by** where it stopped and by the second curve: **T5** step 1 scans both bases, and
a `CartesianTaylor` curve that plateaus at the same level as the `SolidHarmonic`
curve is not a truncation plateau at all — it is the selector not selecting.
A third reading is available here and is the cheapest to hit: **the far field
never engaged.** At the default `ncrit` neither milestone-0 level has one, and a
solve that is entirely P2P agrees with `BRSolverDirect` to round-off at every
order — a flat curve at $10^{-15}$ rather than at $10^{-3}$, which reads as
success. **Distinguished by** the P2P pair fraction, which every scan point and
every test must report. **Do:** no tolerance may be compiled into any test
before **T5** has been read; every tolerance carries the qualification list the
conventions table requires, which now includes the basis, the field it is on and
the P2P fraction.

**R2 — the adapter silently gets the solid-harmonic far field.** Canopy's basis
template parameter is defaulted to `LaplaceKernel`, so an instantiation that omits
it compiles and runs and produces a bare-$1/r$ far field. With
`near_softening_factor = 0` — which **T1** makes the default — there is neither a
floor nor a blob in the expansion, so the far field carries a kernel bias no
order corrects.

**How large the bias is depends on the geometry, and the two regimes are far
apart.** On the milestone-0 smooth sphere it is about a decade: **T4** measured
$4.66\times10^{-3}$ against the production arm's $5.008\times10^{-4}$ at ranks
1-6, both bases at `order` 3, `ncrit` 8, `max_depth` 10, `mac_theta` 0.3,
`softening` 0.025, `near_softening_factor` 0, P2P fraction 0.2537. Tens of
percent — the mechanism behind develop-canopy's full-roll-up NaN — describes a
**self-contacting sheet**, where accepted separations approach $\sqrt b$ and
$\tfrac32 b/R^2$ stops being a small correction. No configuration in this tree
reaches that, so the decade is the figure to expect and the tens of percent is
not available as a threshold.

**Presents as:** the `CartesianTaylor` and `SolidHarmonic` arms landing on top of
each other — a far field that does not move when the basis does. It does **not**
present as a uniform order-of-magnitude miss: the arms are separated by a decade
on this geometry, and `CartesianTaylor` at $p=2$ already sits above the
solid-harmonic point, so any test keyed to the size of the gap rather than to the
*shape* of the order curve is measuring the geometry instead of the selector
(**T5** step 3). **Do:** **T2** step 5 names the basis on every instantiation and
asserts that none relies on the default; **T4**'s second negative case is the
runtime proof, and it is the one case whose *passing* would be the alarm.

**R3 — the round trip silently drops or duplicates a source.** A tag mismatch
does not crash; it produces a velocity that is wrong on some vertices, or wrong
everywhere by a factor that changes with the rank count — the same signature as
the R9 ghost-emission bug the quadrature warns about
([src/Beatnik_SourceQuadrature.hpp:216-220](../../src/Beatnik_SourceQuadrature.hpp#L216-L220)).
**T4**'s rank sweep is what catches it, which is why 1-6 and not just 1, and
**T4** step 7's global-count check is the cheap independent discriminator.

**R4 — Canopy's np=4 defect is absorbed into Beatnik's budget.** `SingleSolve`'s
three-component gradient bodies fail at exactly 4 ranks at $2\times$ over a
$10^{-3}$ budget (`canopy/README.md:562-588`), under `LaplaceKernel` at $P=8$
with `softening = 0`. The evidence is narrower than it was —
`CartesianTaylorSolve` drives `NComps = 3`, compares the gradient and passes at
ranks 1-6 — but that is a different basis, order, softening and distribution, so
it bounds the defect rather than retiring it. Beatnik's gate runs at 4 ranks and
the milestone tier runs at 1 and 4, and Beatnik's own $\tau_A$ is $10^{-3}$ — the
same order as the defect, so this would not present as an obvious outlier. An
error that appears at exactly one rank count is a decomposition-bug signature,
not a budget signature: if **T4** fails at 4 ranks only, record it and raise it
upstream — do **not** widen the budget, and do not drop 4 from the sweep.

**R5 — `solve()` after motion without maintenance returns a defined-but-wrong
field.** Canopy's `solve()` uses the leaf membership, communication plan and P2P
neighbour lists cached by the *last* setup or maintenance call, so a particle
that has moved out of its leaf still contributes to its old leaf's multipole and
still gets its old leaf's near-field list. Nothing raises
([canopy0.md](../../../canopy/tasks/canopy0.md) F3(a)). This is the single most dangerous property of
the API for this consumer, because the three RK stages each move every source.
**T2** must call maintenance before every `solve()`, unconditionally, and must
not add a "the positions barely moved" fast path — that precondition can only be
checked upstream.

**R6 — the operator table overflows and part of the far field takes different
arithmetic.** `CartesianTaylorBasis` keys carry the tree level, so the realized
key count scales with the number of occupied depths against a 32768-key cap;
a thin sheet at a generous `max_depth` is exactly the shape that reaches it.
The cap is closer than it looks: Canopy measured 25438 keys at $\theta=0.3$ on
8640 particles at `ncrit = 8` — 78% of the cap — and saw the table saturate it
outright once the trajectory was amplified, while $\theta=0.5$ on the same cloud
needed only 7374. Beatnik runs at $\theta=0.3$, which is the expensive end.
Overflow is not an error: those pairs route to a per-pair translate that is
slower and bitwise different, with one warning. **Presents as:** an accuracy
number that is a mixture of two code paths, and a speedup worse than the pair
counts predict. **Distinguished by** `total_fallback_pair_count()`, which
**T5** step 5 reports at every scan point. **Do:** if it is non-zero at the
production configuration, lower `max_depth` or the order before touching the cap,
and record which and why — the byte budget is not the binding constraint at
$p\le4$ and raising it will not help.

**R7 — claim A passes and the FMM still destroys the physics.** Claim A measures
the far field on states the *direct* solver produced. develop-canopy's failure
was not there: the FMM tracked the exact solver acceptably for 1362 steps and
then a single corrupted node seeded a runaway to whole-field NaN
(`tasks/fmm_premature_nan.md` Background). Claim B is the guard, and it is why
the two claims are in one binary rather than claim A alone. Its finiteness,
entity-count and volume-drift assertions are the parts that fire in that
scenario — and note that no gold comparison at any rung would, which is why
claim B does not rest on one.

**R8 — claim B's horizon envelope is set from a single run of a
non-reproducible path.** Zoltan2's partition is non-deterministic across runs
([canopy0.md](../../../canopy/tasks/canopy0.md) F3(c)), so two FMM-driven runs of the same deck take
different summation orders and diverge from each other as well as from the
direct path. An envelope measured once will be tripped by the noise.
**Presents as:** **T6**'s horizon assertion failing intermittently, on no code
change. **Do:** **T5** step 7 must measure the horizon more than once and set the
envelope from the *earliest* observed horizon with margin, and the log must
record the run-to-run spread separately from the direct-versus-FMM gap. A single
number with no spread beside it is not a usable envelope.

**R9 — the milestone tier outgrows its walltime and reports a timeout as a
failure.** The tier is at 37.25 of 60 minutes with eight launches, and **T6**
takes it to sixteen launches each doing roughly twice the work — near two hours
on that measurement, which may also exceed what `-q pdebug` allows. A scheduler
kill and a real failure look similar in a log skimmed quickly. **T6** step 6 must
measure the tier run and set `-t` and the queue from it; if the honest number is
unwieldy, splitting claim A and claim B into separate members is the fallback, at
the cost of a third 2000-step trajectory per level.

**R10 — progress stalls waiting on an upstream task.** Two upstream items appear
in this document and they are not the same kind of thing, which is the confusion
to avoid. **X1** is a conditional nobody expects to fire: every task is
independent of it, **T6** lands with whatever $\tau_A$ **T5** measures, and the
deliverable without it is a working, measured, bounded-error fast path at the
reference implementation's own fidelity. Canopy's ladder validation (**R11**)
was the other, and it has landed: the ladder is checked at $|k|=6$, which is
$2p$ at the production order, so **no task here waits on anything upstream**.
What survives of it binds only which scan point may become the production order,
not whether any task may run. A session that reads either item as a gate on the
sequence has misread it.

**R11 — a scan point above $p=3$ is promoted to the production order.** The
derivative ladder is validated at $|k|=2p$ through $p=3$ and no further;
**T2**'s dispatch set reaches $p=5$ and **T5** scans it, so the scan produces
points at $|k|=8$ and $|k|=10$ that no oracle covers. The recurrence divides by
$w$ and weights terms by $k_j(k_j-1)$, so a conditioning loss would be worst at
small $|r|/\sqrt b$ — Beatnik's own band, which is why the measured margins
matter: 13.0x, 11.9x and 16.9x at degrees 4, 5 and 6, at $b=\varepsilon^2$ and
$|r|/\sqrt b$ sampled from inside the band rather than bracketing it. Nothing
suggests it degrades at 8 or 10. **Presents as:** an error curve that keeps
falling above $p=3$ at a rate slightly off the model, or one that stops falling
there — both indistinguishable from **R1**'s truncation plateau, and neither
attributable without an oracle. **Do:** report those points as measurements and
do not adopt one as the production order; **T5** step 6 states it, and the
raise is an upstream request for another degree of oracle, not a Beatnik
change. Canopy's stencils abort loudly on an order they do not carry, so the
extension fails visibly rather than producing a mis-differenced reference.

**R12 — the operator rebuild dominates and the far field is never faster.**
Every maintenance call empties the M2L operator table, unconditionally and by
construction, because the basis is level-keyed and the root box is recomputed
from the particles at every build. Beatnik calls maintenance nine times per
timestep on a surface that deforms continuously, so the table is rebuilt nine
times per timestep, at 25438 keys and 20 DOF per cell at the production
configuration. **Presents as:** correct results throughout and `fmm` slower than
`direct` at every vertex count **T8** measures — the crossover simply never
arriving. **Distinguished from** an ordinary constant-factor loss by **T8** step
2's per-build key count: a rebuild cost shows as `m2l_op_keys_built_count()`
climbing by the full cache size at every build. **Do:** it is not a correctness
risk and no Beatnik task fixes it. If **T8** finds it dominant, the log records
the measured share as a sized request against Canopy — plausibly keying the
cache on the physical $R$ rather than on the level, which
[cartesian-taylor-basis.md](../../../canopy/tasks/cartesian-taylor-basis.md) R6
names — and README states the crossover as "none found, bounded by operator
reconstruction" rather than reporting a speedup that does not exist. Do not
respond by lowering `max_depth` to shrink the table: that trades a cost problem
for an accuracy one and **T5** owns `max_depth`.
