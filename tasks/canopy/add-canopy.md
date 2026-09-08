# Canopy as Beatnik's far-field Birkhoff-Rott solver

**Status:** NOT STARTED — every task below is NOT STARTED. The findings and the
measured numbers are complete.

## Problem

`--br-approximation fmm` throws. `BRSolverFMM::computeInterfaceVelocity` and
`::computeSurfaceRieszScalar` are `BEATNIK_NOT_IMPLEMENTED` stubs
([src/Beatnik_BRSolverFMM.hpp:113-154](../../src/Beatnik_BRSolverFMM.hpp#L113-L154)),
and so is every method of the adapter behind them
([src/Beatnik_FarFieldInterface.hpp:125-188](../../src/Beatnik_FarFieldInterface.hpp#L125-L188)).
`fmm` is nevertheless the **default**
([src/Beatnik_Params.hpp:106](../../src/Beatnik_Params.hpp#L106)), so the default
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
[abstract-solver-backend.md](abstract-solver-backend.md). Two bases ship:

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

The two bases are not equally validated for this problem and the selection is
not a preference: **T4** and **T5** measure both, and the solid-harmonic basis
with the floor disabled is **T4**'s negative case.

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
tensors of $\varphi$, and Canopy's basis is unit-tested against them
([abstract-solver-backend.md](abstract-solver-backend.md) T12). What differs is
the traversal (FMM's M2M/M2L/L2L against the treecode's M2P, i.e. $O(N)$ against
$O(N\log N)$) and the acceptance criterion. So `FmmParams::order` and
`--br-treecode-order` finally denote the same quantity — the Cartesian Taylor
truncation order — while `mac_theta` still does not, because Canopy's predicate
is the exafmm spherical MAC and the reference's is a Barnes-Hut opening angle
([canopy0.md](canopy0.md) F4).

**The order that target needs is an estimate until T5 measures it.** Canopy
accepts M2L iff $R\theta>\sqrt3\,(h_A+h_B)$ with $h$ a half-width
(`canopy/src/Canopy_CommunicationPlan.hpp:338-352`), so equal-size cells are
accepted only beyond $R/w>2\sqrt3/\theta$. A Cartesian-Taylor truncation at order
$p$ has relative error $\sim(cw/R)^{p+1}$ with $c\in[1,\sqrt3]$
([canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF"):

| `mac_theta` | $R/w$ accepted | decades per order | $p$ for $10^{-3}$ | DOF/cell $\binom{p+3}{3}$ |
| --- | --- | --- | --- | --- |
| 0.3 (`FmmParams` default) | $>11.5$ | 0.82-1.06 | 2-3 | 10-20 |
| 0.5 (Canopy default) | $>6.9$ | 0.60-0.84 | 3-4 | 20-35 |

So the target is expected to land at $p=2$-$4$, which is cheap — and $p=2$ is
already `FmmParams::order`'s default. This is a model, not a measurement: the
error constant, the sheet's anisotropic leaf occupancy and the depth-mismatched
cell pairs an adaptive tree realizes are all outside it. **T5** measures the
curve and picks the production order; nothing may be compiled into a test before
it does.

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
same reason. Chasing $10^{-10}$ instead would want $p\approx11$-24 and 364-2925
DOF per cell ([canopy-kernel-rec.md](canopy-kernel-rec.md)), for a trajectory
comparison that still would not pass.

The consequence shapes **T6**: its two claims are a per-evaluation bound and a
stability-plus-divergence-horizon measurement, not one loosened gold comparison.

**Out of scope.** Any change to Canopy (**X1** names the one conditional
dependency and its acceptance test; it does not design it). Any new CLI option —
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
difference ([abstract-solver-backend.md](abstract-solver-backend.md) T12), so the
gradient carries no finite-difference step-size error and no third error
component for **T5**'s scan to disentangle. The solid-harmonic basis retains its
finite-difference L2P; that difference is one of the things **T5** measures
between the two.

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
bases ([abstract-solver-backend.md](abstract-solver-backend.md), "Out of scope").

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
([src/Beatnik_Params.hpp:147-148](../../src/Beatnik_Params.hpp#L147-L148)) and the
basis is a `FarFieldBasis` enum (**T1**). The adapter resolves it by dispatching
the runtime `(basis, order)` pair onto an explicitly enumerated set of
instantiations and throwing for anything else — see the conventions table. It
does not silently round and it does not silently substitute a basis.

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
| Canopy visibility | `Beatnik_FarFieldInterface.hpp` is the **only** header that may name a Canopy type, include a Canopy header, or hold a Canopy object. The `FarFieldBasis` enum is a Beatnik type naming no Canopy type and lives with the other mode enums in `Beatnik_Types.hpp`; the adapter is where it becomes a Canopy basis. Verified by `grep -l Canopy src/*.hpp` naming exactly `Beatnik_FarFieldInterface.hpp` and `Beatnik_Config.hpp.in`. |
| Build guard | Everything Canopy-facing sits behind `BEATNIK_ENABLE_CANOPY` ([src/CMakeLists.txt:1-6](../../src/CMakeLists.txt#L1-L6), [:78-79](../../src/CMakeLists.txt#L78-L79)). A `~canopy` build must still compile every header and still throw the existing configuration error ([src/Beatnik_CreateBRSolver.hpp:66-69](../../src/Beatnik_CreateBRSolver.hpp#L66-L69)). |
| Failure behavior | A violated precondition throws `std::logic_error` for "this code is unwritten" and `std::runtime_error` for "this build or configuration cannot do it", matching [src/Beatnik_CreateBRSolver.hpp:45-49](../../src/Beatnik_CreateBRSolver.hpp#L45-L49). Never return a truncated or best-effort field: a plausible wrong velocity is the failure mode this whole document exists to bound. |
| New parameters | Added to `FmmParams` ([src/Beatnik_Params.hpp:141-152](../../src/Beatnik_Params.hpp#L141-L152)) with a defaulted member and a comment stating units, the meaning of the default, and which Canopy knob it reaches. Never a new constructor parameter, never a new CLI option. |
| Runtime dispatch | `(FmmParams::basis, FmmParams::order)` selects among an explicitly enumerated set of instantiations; an unsupported pair throws naming the supported set. Never silently rounded, never silently substituted. The set and the compile-time cost of extending it are documented on the dispatch. |
| Basis is always named | Every Canopy `Solver`/`createSolver` instantiation names its basis explicitly. The parameter is defaulted upstream and the default is not the basis Beatnik wants. |
| Enums over bools | A mode selector is an enum or tag type, never a bool or a magic number. |
| Comments | Units, sign convention, and which side of a difference is which, on the declaration. The sign of Canopy's gradient output and the direction of $\delta$ are the two most misread things on this path and must be stated at every boundary they cross. |
| Provenance | Any routine derived from `origin/develop-canopy`'s `src/FmmBRSolver.hpp`, from Canopy, or from the reference Python cites the file and line range on the routine. |
| Accuracy claims | Every stated tolerance names the source distribution, the rank counts, **the basis**, `order`, `ncrit`, `max_depth`, `mac_theta`, `softening` and `near_softening_factor` it was measured at. A bare tolerance is not a claim and may not be compiled into a test. |
| Citing Canopy | Cite Beatnik by `file:line`. Cite Canopy by **symbol name** — `FmmConfig::near_softening_factor`, `Solver::auto_maintain` — with a line number only where the symbol is in a file the far-field abstraction does not restructure (`Canopy_P2P.hpp`, `Canopy_CommunicationPlan.hpp`, `Canopy_TreeBuilder.hpp`, `Canopy_TreePartitioner.hpp`). A stale line number into a restructured header points at unrelated code and is worse than no citation. |
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
  three times per timestep ([canopy0.md](canopy0.md) F3(b)).
  develop-canopy measured `auto_maintain` returning the cheapest `Migrate`
  action for all 14 calls of a five-step run with results identical to the
  setup-every-step baseline. **T8** measures whether that holds on a deforming
  surface, where canopy0.md F3(c) predicts `Rebalance` instead.
- **No attempt to correct the kernel on Beatnik's side.** Handing Canopy a bare
  far field and adding
  $\sum_{r<R_c}[K_{\rm soft}-K_{\rm bare}]\times S_s$ by direct summation is
  algebraically a near-field softening floor by another name, and it is what
  selecting `CartesianTaylorBasis` makes unnecessary. It buys nothing and is not
  attempted.

## Current state

### Beatnik

- `--br-approximation fmm` **throws** from `BRSolverFMM`'s two virtuals
  ([src/Beatnik_BRSolverFMM.hpp:126](../../src/Beatnik_BRSolverFMM.hpp#L126),
  [:153](../../src/Beatnik_BRSolverFMM.hpp#L153)) via `BEATNIK_NOT_IMPLEMENTED`
  ([src/Beatnik_Types.hpp:86](../../src/Beatnik_Types.hpp#L86)). It throws rather
  than returning a wrong field, which is the safe direction.
- `FarFieldSolver`'s three methods throw the same way
  ([src/Beatnik_FarFieldInterface.hpp:129](../../src/Beatnik_FarFieldInterface.hpp#L129),
  [:156](../../src/Beatnik_FarFieldInterface.hpp#L156),
  [:187](../../src/Beatnik_FarFieldInterface.hpp#L187)). The header states outright
  that Canopy had not been read when it was written, and its `setSources` /
  `evaluateCurl` split does not match Canopy's `setup` / `auto_maintain` /
  `solve` split — **T2** owns the signature change.
- **No Beatnik header includes a Canopy header.** The build already finds and
  links Canopy under `+canopy`
  ([CMakeLists.txt:79-81](../../CMakeLists.txt#L79), [src/CMakeLists.txt:78-79](../../src/CMakeLists.txt#L78-L79)),
  and the tuolumne environment builds with it
  ([systems/tuolumne/claude.md](../../systems/tuolumne/claude.md) §2).
- `FmmParams` carries three members — `mac_theta` (default 0.3), `order`
  (default 2), `ncrit` (default 64)
  ([src/Beatnik_Params.hpp:141-152](../../src/Beatnik_Params.hpp#L141-L152)). It has
  **no** basis selector, no `max_depth`, no `near_softening_factor`, no operator
  byte budget and none of Canopy's six bounding-box tolerances, all of which
  `FmmConfig` requires and two of which (`ncrit`, `max_depth`) have **no default
  initializer** in Canopy (`FmmConfig::ncrit`, `FmmConfig::max_depth`), so a
  default-constructed `FmmConfig` builds an arbitrary tree.
- The mode enums this work extends live in `src/Beatnik_Types.hpp`
  (`BRApproximation` at [:163](../../src/Beatnik_Types.hpp#L163),
  `BernoulliScalarMode` at [:171](../../src/Beatnik_Types.hpp#L171),
  `KernelBlobMode` at [:193](../../src/Beatnik_Types.hpp#L193)).
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
  [abstract-solver-backend.md](abstract-solver-backend.md).
- `CartesianTaylorBasis` expands $\varphi(r)=(r^2+b)^{-1/2}$ directly, in real
  coefficients, $\binom{p+3}{3}$ per cell per component. It requires
  `Scalar = double` by `static_assert`, its L2P gradient is analytic, and its
  operator keys carry the tree level, so its operator table is larger than the
  solid-harmonic basis's by roughly the number of occupied depths.
- The M2L operator table is bounded by
  `min(M2L_OP_COUNT_CAP, byte_budget / bytes_per_key)`, with
  `M2L_OP_COUNT_CAP = 32768` and the byte budget an `FmmConfig` field defaulting
  to 2 GB. Keys beyond the cap route to a per-pair translate fallback — different
  arithmetic, slower, not wrong — and `total_fallback_pair_count()` reports how
  many pairs took it. At $p\le4$ the per-key operator is $\le9.8$ KB
  ([canopy-kernel-rec.md](canopy-kernel-rec.md), "Memory"), so the **count cap
  binds first** and the byte budget is not the constraint.
- Operators are held in a cache keyed by the canonicalized M2L key and the
  kernel parameters, persisting across topology changes; only the key→index map
  is rebuilt when the interaction list is invalidated. For a deforming surface at
  fixed $b$ that is a cost win, and **T8** is where it shows up.
- `FmmConfig::near_softening_factor` still exists and still forces close pairs
  into P2P. Beatnik sets it to 0 under `CartesianTaylorBasis`; it is meaningful
  only under the solid-harmonic basis.
- `FmmConfig::softening` is a **length**; Beatnik's `blob()` is a squared length
  ([src/Beatnik_Params.hpp:125-128](../../src/Beatnik_Params.hpp#L125-L128)).
- `solve()` after motion without maintenance is silently wrong, `migrate()` is
  not cheap, and a deforming surface picks `Rebalance` essentially every stage
  ([canopy0.md](canopy0.md) F3(a)-(c)). None of this is basis-dependent and none
  of it changed.
- Zoltan2's `multijagged` is non-deterministic across runs, so the far-field path
  is not bitwise reproducible run to run at fixed rank count
  ([canopy0.md](canopy0.md) F3(c)).
- Canopy's three-component gradient path — the one this work rests on — is
  labelled `unit`, not `regression`, and fails at exactly 4 ranks
  (`SingleSolve.PotentialNComps3`, `max_pot_rel_err = 0.00207` against a
  $10^{-3}$ budget, `canopy/README.md:368-370`). Beatnik's gate and milestone
  tier both run at 4 ranks. This is **R4**.
- No Canopy test gives any rank zero particles ([canopy0.md](canopy0.md) F5), and
  no Canopy test measures accuracy on a non-volumetric source distribution
  ([canopy0.md](canopy0.md) F4). Both gaps are on this path.

## Progress log

[add-canopy-progress-log.md](add-canopy-progress-log.md) holds what actually
happened: the reasoning behind decisions this document states flatly, the
measured numbers behind its claims, and things only running revealed. **Read it
before implementing any task, changing any signature, or reopening a question
this document treats as settled** — in particular before compiling any tolerance
into any test, since a measured number in the log always outranks an estimate
here.

## Task sequence

### T1 — `FmmParams` carries everything `FmmConfig` needs, and names the basis — **NOT STARTED**

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
(`canopy/src/Canopy_Solver.hpp`, `struct FmmConfig`); `FmmParams` as it stands
([src/Beatnik_Params.hpp:141-152](../../src/Beatnik_Params.hpp#L141-L152));
develop-canopy's `makeCanopyConfig` for the full mapping it needed
(`src/FmmBRSolver.hpp:588-607` on that branch).

**Do:**

1. Add `enum class FarFieldBasis { CartesianTaylor, SolidHarmonic }` to
   `Beatnik_Types.hpp`, in the style of the mode enums already there, with a
   comment stating which Canopy basis each maps to and that
   `CartesianTaylor` is the validated production path. It names no Canopy type.
2. Add `FarFieldBasis basis = FarFieldBasis::CartesianTaylor` to `FmmParams`.
3. Extend `FmmParams` with every remaining knob the adapter must set:
   `max_depth`, `near_softening_factor`, `m2l_operator_byte_budget`, `ncrit_tol`,
   `replication_depth`, `imbalance_tolerance`, and the six bounding-box padding
   factors. Every one gets a default initializer and a comment naming units, the
   default's meaning, and the `FmmConfig` member it reaches.
4. Default `near_softening_factor` to **0**, and state on the declaration that a
   non-zero value is meaningful only under `FarFieldBasis::SolidHarmonic` —
   under `CartesianTaylor` the far field already carries the blob, so the floor
   only moves work into P2P and slows the solve without improving it.
5. Restate `order`'s comment: it is the basis's order knob, and under
   `CartesianTaylor` it is the Cartesian Taylor truncation order $p$ — the same
   quantity `--br-treecode-order` denotes in the reference. Correct the
   `FmmParams` doc comment
   ([src/Beatnik_Params.hpp:132-140](../../src/Beatnik_Params.hpp#L132-L140)) and
   the CLI comment
   ([examples/02_adaptive_mesh_bubble/InputFile.hpp:478-479](../../examples/02_adaptive_mesh_bubble/InputFile.hpp#L478-L479)),
   both of which say the treecode numbers do not mean the same thing to the two
   algorithms. That is true of `mac_theta` and no longer true of `order`; say
   which is which rather than deleting the warning.
6. Choose and document a `max_depth` default. It has no counterpart in the
   treecode knob set, is bounded at 19 by the `uint64_t` Morton key
   (`canopy/src/Canopy_TreeBuilder.hpp:176-181`), and sets the finest cell width
   as (root box width) / $2^{\rm max\_depth}$. Two forces pull against each
   other and both belong in the comment: a sheet reaches
   $N/\texttt{ncrit}$ leaves in $\log_4$ rather than $\log_8$ levels, so it wants
   depth ([canopy0.md](canopy0.md) F4); and every occupied depth multiplies
   `CartesianTaylorBasis`'s realized operator-key count, which the 32768-key cap
   bounds. develop-canopy ran 19 and hit a depth-driven finite-difference blow-up
   at roll-up — a mechanism the analytic Taylor L2P removes, so that particular
   reason to fear 19 does not transfer. State the reasoning for whatever is
   chosen.
7. **Do not add a `softening` member that duplicates `eps`.** Canopy's
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
8. `FmmParams` must not be constructible into a state that builds an arbitrary
   Canopy tree. Since every Beatnik member is default-initialized, this reduces
   to giving `ncrit` and `max_depth` defensible values and validating them where
   the adapter builds the `FmmConfig`.
9. Update README's parameter documentation in the same change.

**Exit criterion:** `spack install` succeeds; a `FmmParams` default-constructed
and passed through the adapter's config builder yields an `FmmConfig` whose every
member is initialized (asserted by **T4**'s test, which is where a runnable check
first exists); README lists every new member with its default. No new CLI option
appears in `--help`.

---

### T2 — `FarFieldSolver` backed by Canopy: the adapter and the round trip — **NOT STARTED**

**Depends on:** T1.

**This is the task that first opens `../canopy`.** T1 must not, and no earlier
task may name a Canopy type. Every Canopy reading decision, every signature that
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
   The recommended shape is one method per contraction, each taking points,
   source vectors, the blob and the output view, with the tree
   maintenance decided internally — that keeps `BRSolverFMM` free of any
   knowledge of Canopy's lifecycle. Record the chosen signatures in the log.
2. Define the AoSoA member layout as an enum, not bare indices: position
   `double[3]`, charge `double[3]`, output `double[3]`, tag `int[2]`
   = `(origin_rank, owned_index)`. develop-canopy's `FmmField` namespace
   (`src/FmmBRSolver.hpp:55-61` on that branch) is the precedent; two tag
   components suffice here because the source list is one-dimensional.
3. Implement the round trip as the six steps in **Approach**. The forward
   `Distributor` is rebuilt every evaluation; caching it across `Migrate`-action
   evaluations is a named future optimization, not part of this task.
4. Build the `FmmConfig` from `FmmParams` plus `ZModelParams::blob()` per T1
   step 7, and validate it: throw naming the offending member if `ncrit` or
   `max_depth` is non-positive, or `max_depth > 19`.
5. Dispatch `(FmmParams::basis, FmmParams::order)` onto an enumerated set of
   instantiations, **naming the Canopy basis explicitly in every one**. Start
   with the set the measurement needs — at minimum `CartesianTaylor` at every
   order **T5** will scan, plus one `SolidHarmonic` instantiation so **T4**'s
   negative case and **T5**'s comparison can be built — and throw naming the
   supported set for anything else. Add a `static_assert` or equivalent that no
   instantiation relies on Canopy's defaulted basis parameter; a silently
   solid-harmonic solve is **R2**.
6. Every rank must enter every collective the same number of times per
   evaluation, **including a rank that owns zero sources**. Canopy has no test
   for a zero-particle rank ([canopy0.md](canopy0.md) F5) and Beatnik's
   decomposition can produce one. Do not branch the collective sequence on a
   local count.
7. `#include <Canopy_Solver.hpp>` and every Canopy-typed member sit behind
   `BEATNIK_ENABLE_CANOPY`; the class must still compile, and its methods must
   still throw `std::runtime_error` naming the missing build option, in a
   `~canopy` build.

**Exit criterion:** `spack install` succeeds with `+canopy` **and** with the
Canopy dependency's headers made unavailable (or with
`Beatnik_ENABLE_CANOPY=OFF` configured by hand), the second build's
`FarFieldSolver` throwing `std::runtime_error` naming `+canopy` rather than
failing to compile; `grep -l Canopy src/*.hpp` names only
`Beatnik_FarFieldInterface.hpp` and `Beatnik_Config.hpp.in`; and
`grep -n "CartesianTaylorBasis" src/Beatnik_FarFieldInterface.hpp` shows the
basis named on every `Solver` instantiation the dispatch builds. No behavioral
claim is made by this task — **T4** is where correctness is first checked.

---

### T3 — `BRSolverFMM::computeInterfaceVelocity` — **NOT STARTED**

**Depends on:** T2.

**Fill in:** [src/Beatnik_BRSolverFMM.hpp](../../src/Beatnik_BRSolverFMM.hpp)
(`computeInterfaceVelocity` only; `computeSurfaceRieszScalar` is T7),
[README.md](../../README.md).

**Reference:** the direct implementation this must agree with, step for step
([src/Beatnik_BRSolverDirect.hpp:105-166](../../src/Beatnik_BRSolverDirect.hpp#L105-L166))
— note it reallocates the output to `ownedVertexCount()` and zeroes it before
accumulating ([:112-115](../../src/Beatnik_BRSolverDirect.hpp#L112-L115)), calls
`quadrature.generate` itself ([:117-119](../../src/Beatnik_BRSolverDirect.hpp#L117-L119)),
and applies `br_sign/4\pi` once ([:126-127](../../src/Beatnik_BRSolverDirect.hpp#L126-L127));
the caller's contract ([src/Beatnik_ZModelSolver.hpp:219-224](../../src/Beatnik_ZModelSolver.hpp#L219-L224)),
which reallocates `vertex_dot` to the owned count and expects the prefactors
already applied.

**Do:**

1. Generate sources through the quadrature, call the adapter, write the
   `(N_owned, 3)` velocity. Overwrite, do not accumulate — the declaration says
   overwritten ([src/Beatnik_BRSolverBase.hpp:137-139](../../src/Beatnik_BRSolverBase.hpp#L137-L139)).
2. Correct the two `@note` blocks in the file header
   ([:102-107](../../src/Beatnik_BRSolverFMM.hpp#L102-L107)) and the matching
   paragraph in `Beatnik_FarFieldInterface.hpp:36-41`. Both say the expanded
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
   ([src/Beatnik_Params.hpp:106](../../src/Beatnik_Params.hpp#L106)) is not the
   validated path until **T5** has run.

**Exit criterion:** a two-step `--br-approximation fmm` run of
`examples/02_adaptive_mesh_bubble` at the milestone-0 configuration completes
without throwing, at 1 and 4 ranks, submitted as a batch script under
`scripts/tuolumne/` and read from its `.log`; and the same run with
`--br-approximation direct` still produces the checkpoint it produces today.
No accuracy claim — that is **T4**.

---

### T4 — Unit test: the FMM velocity against the direct velocity, same state — **NOT STARTED**

**Depends on:** T3.

**Fill in:** `tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp` (new),
[tests/unit_tests/CMakeLists.txt](../../tests/unit_tests/CMakeLists.txt)
(`BEATNIK_UNIT_TEST_SOURCES`, [:42](../../tests/unit_tests/CMakeLists.txt#L42)).

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
2. Construct both BR solvers and evaluate **both on that same state**, then
   assert max relative and max absolute velocity error against a budget whose
   value and full qualification list are recorded in the log.
3. Three negative cases, because each proves a different thing and a comparison
   that has only ever seen agreeing data has not been tested:
   - **The order knob is live.** `order = 0` (monopole-only Taylor) must exceed
     the budget while the production order passes. This is what makes the error a
     truncation rather than a bias, and it is the property the whole basis choice
     rests on.
   - **The basis selector is live.** `FarFieldBasis::SolidHarmonic` with
     `near_softening_factor = 0` must exceed the budget by orders of magnitude.
     A pass here means the adapter is not selecting the basis it says it is —
     see **R2**.
   - **The blob reaches the far field.** Perturbing the softening length handed
     to Canopy must change the FMM velocity. Under a bare-kernel far field it
     would not, which is the cheapest available proof that $b$ is inside $w$.
4. Rank counts 1-6, since that is the gate's sweep and Canopy's
   three-component gradient path is known wrong at exactly 4
   (`canopy/README.md:368-370`). If 4 ranks fails, that is the finding: record
   it, and do not widen the budget to accommodate it — see **R4**.
5. Include a variant where at least one rank owns zero sources, if the
   decomposition can be made to produce one at these vertex counts; if it
   cannot, say so in the log rather than leaving the case silently uncovered.
6. Assert the FMM result is finite everywhere, and that the global source count
   Canopy reports equals the global owned vertex count — the cheap independent
   check on the round trip (**R3**).

**Exit criterion:** `ctest -R Beatnik_Test_FmmVsDirect` passes at ranks 1-6 (and
`BEATNIK_UNIT_RANKS=4 flux batch scripts/tuolumne/unit_tests.flux` passes in
spack mode) against a budget recorded with its qualification list; and each of
the three negative cases fails, naming its own reason — the order case naming
truncation, the basis case naming the basis in force, the blob case naming the
softening length — rather than merely exiting non-zero.

---

### T5 — Measure the achievable far-field fidelity, and publish it — **NOT STARTED**

**Depends on:** T4. **This task produces every number later tasks key off.**

**Fill in:** a measurement driver under `tests/regression_tests/` registered in
the "Measurement drivers — IN NO TIER" section
([tests/CMakeLists.txt:536-570](../../tests/CMakeLists.txt#L536-L570)); a batch
script under `scripts/tuolumne/`;
[add-canopy-progress-log.md](add-canopy-progress-log.md);
[README.md](../../README.md).

**Reference:** the convergence model this must confirm or correct, with its
constants ([canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF"
and "Memory"); the acceptance predicate that sets $R/w$
(`canopy/src/Canopy_CommunicationPlan.hpp:338-352`); the reference treecode's own
accuracy at the same two mesh sizes ([treecode.md](../treecode.md) §1); the
measurement-driver loop's rule that it appends to **neither** manifest
([tests/CMakeLists.txt:548-556](../../tests/CMakeLists.txt#L548-L556)); M0-D1's
ladder, which is the instrument step 6 reuses
([milestone0-progress-log.md](../milestone0-progress-log.md)).

**Do:**

1. Scan, on real milestone-0 states at both subdivision levels: `order` over the
   dispatched set, `mac_theta`, `ncrit`, `max_depth`, and both values of
   `basis`. Report max relative and max absolute velocity error against
   `BRSolverDirect` on the **same** state, and the fraction of pairs Canopy
   handled in P2P.
2. **Read the scan as a scan.** Under `CartesianTaylor` the error should fall
   with `order` at the rate the table in **Problem** predicts and then flatten
   into Canopy's own floating-point floor; under `SolidHarmonic` it should
   plateau well above that regardless of order, because the bias is in the
   kernel rather than the truncation. Two curves of different *shape* is the
   finding; two curves of the same shape means the basis selector is not doing
   what it claims. Record which regime each observed level is in.
3. Confirm or correct the decades-per-order table in **Problem**. It is a model
   with an unmeasured constant, evaluated at an idealized equal-cell geometry;
   the realized $R/w$ distribution on a thin bubble surface with depth-mismatched
   cell pairs is what actually sets the rate. A measured rate far *better* than
   the model most likely means the far field never engaged, which at these vertex
   counts is easy to hit by accident and which [treecode.md](../treecode.md) §1
   documents as the same trap on the treecode side — check the P2P pair fraction
   before believing it.
4. Measure the operator table: the realized key count `n_unique_ops`, the bytes
   it occupies, and `total_fallback_pair_count()`, at each `max_depth` and
   `order` in the scan. `CartesianTaylorBasis` keys carry the tree level, so the
   count scales with occupied depth against a 32768-key cap; a non-zero fallback
   count means some pairs took different arithmetic and the accuracy number is
   a mixture. This is **R6**.
5. Record $\tau_A$ — the best max relative velocity error achieved at an
   affordable order and P2P fraction, with its full qualification list — in the
   log, and publish the validated parameter set and the achieved fidelity in
   README. State the production `order` and why it was chosen over the next one
   up and the next one down.
6. Measure the **divergence horizon** claim B needs: run the milestone-0
   configuration FMM-driven and direct-driven to 2000 steps and report, as a
   ladder, the first step at which the two exceed each rung. This is M0-D1's
   measurement with a per-evaluation perturbation as the seed instead of a
   one-ulp initial condition, and it is what turns claim B's envelope into a
   measured number. Report the volume-drift series alongside, since claim B
   asserts against `kRefVolumeDrift` and needs a measured bound.

**Exit criterion:** the log carries the full scan with every entry's
qualification list; a stated $\tau_A$ and production parameter set; the operator
key-count and fallback-count table from step 4; and the divergence-horizon ladder
and volume-drift bound from step 6. README carries the validated parameter set
and the achieved fidelity for the gradient. The task is complete whichever value
$\tau_A$ takes — if it is above $10^{-3}$ at every affordable order, that is the
finding, and **X1** is what it implies.

---

### T6 — The two milestone-tier FMM members — **NOT STARTED**

**Depends on:** T5 (for $\tau_A$, the horizon envelope and the volume-drift
bound) and T4 (for the comparison harness).

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
2. **Claim A.** Drive the trajectory with `BRSolverDirect` — the run must stay
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
([canopy0.md](canopy0.md) F3(b)); the prediction that a deforming surface picks
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
2. Report how much of `solve` is operator construction. Canopy's operator cache
   is keyed by geometry and persists across topology changes, so a `Rebalance`
   rebuilds the key→index map but not the operators themselves; on a deforming
   surface at fixed $b$ the per-key build should amortize to near zero after the
   first few hundred steps. Whether it does is the difference between
   `Rebalance`-every-stage being affordable and not.
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
`CartesianTaylorBasis` keys carry the tree level against a 32768-key cap. A
Cartesian-Taylor expansion buys 0.24-0.48 decades per order at standard
admissibility and 0.82-1.06 at Beatnik's `mac_theta = 0.3`, while its DOF count
grows as $\binom{p+3}{3}\sim p^3/6$ — so the basis is cheap in the band this work
targets and expensive an order of magnitude below it
([canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF").

**Where the work would live:** a black-box (Chebyshev interpolation) basis in
Canopy, on the same far-field contract, designed in
[canopy-bbFMM.md](canopy-bbFMM.md). It converges geometrically in the
interpolation order rather than algebraically, and softening helps rather than
hurts — $b>0$ moves the kernel singularity off the real axis, which can only
increase the convergence rate. Its cost is memory: the per-key operator is
1.1 MB at $n=6$ against `CartesianTaylorBasis`'s 9.8 KB at $p=4$, so it is only
representable in the compressed shared-basis form, which is why it is its own
design and not a variant of this one
([canopy-kernel-rec.md](canopy-kernel-rec.md), "Memory").

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
**Do:** no tolerance may be compiled into any test before **T5** has been read,
and every tolerance carries the qualification list the conventions table
requires, which now includes the basis.

**R2 — the adapter silently gets the solid-harmonic far field.** Canopy's basis
template parameter is defaulted to `LaplaceKernel`, so an instantiation that omits
it compiles and runs and produces a bare-$1/r$ far field. With
`near_softening_factor = 0` — which **T1** makes the default — that is not a
small error: with no floor and no blob in the expansion the far field is wrong by
tens of percent, which is the mechanism behind develop-canopy's full-roll-up NaN.
**Presents as:** **T4** failing by orders of magnitude at every order, with the
`order = 0` and production-order cases indistinguishable. **Do:** **T2** step 5
names the basis on every instantiation and asserts that none relies on the
default; **T4**'s second negative case is the runtime proof, and it is the one
case whose *passing* would be the alarm.

**R3 — the round trip silently drops or duplicates a source.** A tag mismatch
does not crash; it produces a velocity that is wrong on some vertices, or wrong
everywhere by a factor that changes with the rank count — the same signature as
the R9 ghost-emission bug the quadrature warns about
([src/Beatnik_SourceQuadrature.hpp:216-220](../../src/Beatnik_SourceQuadrature.hpp#L216-L220)).
**T4**'s rank sweep is what catches it, which is why 1-6 and not just 1, and
**T4** step 6's global-count check is the cheap independent discriminator.

**R4 — Canopy's np=4 defect is absorbed into Beatnik's budget.** Canopy's
three-component gradient path fails at exactly 4 ranks at $2\times$ over a
$10^{-3}$ budget (`canopy/README.md:368-370`). Beatnik's gate runs at 4 ranks and
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
([canopy0.md](canopy0.md) F3(a)). This is the single most dangerous property of
the API for this consumer, because the three RK stages each move every source.
**T2** must call maintenance before every `solve()`, unconditionally, and must
not add a "the positions barely moved" fast path — that precondition can only be
checked upstream.

**R6 — the operator table overflows and part of the far field takes different
arithmetic.** `CartesianTaylorBasis` keys carry the tree level, so the realized
key count scales with the number of occupied depths against a 32768-key cap;
a thin sheet at a generous `max_depth` is exactly the shape that reaches it.
Overflow is not an error: those pairs route to a per-pair translate that is
slower and bitwise different, with one warning. **Presents as:** an accuracy
number that is a mixture of two code paths, and a speedup worse than the pair
counts predict. **Distinguished by** `total_fallback_pair_count()`, which
**T5** step 4 reports at every scan point. **Do:** if it is non-zero at the
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
([canopy0.md](canopy0.md) F3(c)), so two FMM-driven runs of the same deck take
different summation orders and diverge from each other as well as from the
direct path. An envelope measured once will be tripped by the noise.
**Presents as:** **T6**'s horizon assertion failing intermittently, on no code
change. **Do:** **T5** step 6 must measure the horizon more than once and set the
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

**R10 — progress stalls waiting on X1.** Every task is independent of it, and
**T6** lands with whatever $\tau_A$ **T5** measures. The failure mode is a session
reading **X1** as a gate on the whole document. It is not: it is a conditional
that the current estimate says will not fire, and the deliverable without it is a
working, measured, bounded-error fast path at the reference implementation's own
fidelity.
