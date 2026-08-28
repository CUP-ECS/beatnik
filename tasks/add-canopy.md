# Canopy as Beatnik's far-field Birkhoff-Rott solver

**Status:** NOT STARTED — every task below is NOT STARTED. The findings and the
measured numbers are complete.

## Problem

`--br-approximation fmm` throws. `BRSolverFMM::computeInterfaceVelocity` and
`::computeSurfaceRieszScalar` are `BEATNIK_NOT_IMPLEMENTED` stubs
([src/Beatnik_BRSolverFMM.hpp:113-154](../src/Beatnik_BRSolverFMM.hpp#L113-L154)),
and so is every method of the adapter behind them
([src/Beatnik_FarFieldInterface.hpp:125-188](../src/Beatnik_FarFieldInterface.hpp#L125-L188)).
The only working Birkhoff-Rott evaluator is `BRSolverDirect`, an $O(N_t N_s)$
ring-exchanged pairwise sum
([src/Beatnik_BRSolverDirect.hpp:105-166](../src/Beatnik_BRSolverDirect.hpp#L105-L166))
called three times per accepted timestep
([src/Beatnik_ZModelSolver.hpp:223](../src/Beatnik_ZModelSolver.hpp#L223)) and a
fourth time under `--bernoulli-scalar-mode surface-riesz`
([:255](../src/Beatnik_ZModelSolver.hpp#L255)).

The end state: `--br-approximation fmm` runs the same physics through Canopy's
fast multipole method, in sub-$N^2$ time, with a **stated, measured** bound on
how far its velocity departs from `BRSolverDirect`'s on the same state; and two
new `milestone`-tier members demonstrate that bound on the real milestone-0
problem at all 81 checkpointed steps of a 2000-step run, at both subdivision
levels.

**The far field's accuracy is the whole difficulty, and it is bounded by
something outside this repository.** Canopy's multipole far field expands the
**unsoftened** $1/r$ Laplace kernel; the Plummer softening exists only in the
near-field P2P kernel
(`canopy/src/Canopy_P2P.hpp:799-803`, `883-896`), and is kept relevant by a floor
that forces any pair whose cell-centre separation is below
`near_softening_factor` $\cdot\ \varepsilon$ out of M2L and into P2P
(`canopy/src/Canopy_CommunicationPlan.hpp:347-361`). Beatnik's kernel carries
$b = \varepsilon^2 = 6.25\times10^{-4}$ with $\varepsilon = 0.025$ on a bubble of
radius $0.25$, so $\varepsilon$ is **5% of the bubble diameter** and the
softening perturbs the kernel by $\tfrac32\varepsilon^2/r^2 = 3.8\times10^{-3}$
even at maximum separation. The consequence is measured, not estimated
(provenance: log topic *read-only survey*):

| `near_softening_factor` | max relative velocity error, level 3 | level 4 | fraction of pairs in the near field |
| --- | --- | --- | --- |
| 4 (Canopy's default) | $2.0\times10^{-2}$ | $1.8$–$2.7\times10^{-2}$ | 4.9% |
| 8 | $5.0$–$7.9\times10^{-3}$ | $5.3$–$7.1\times10^{-3}$ | 23% |
| 12 | $0.8$–$1.8\times10^{-3}$ | $0.8$–$1.7\times10^{-3}$ | 54% |
| 20 | $1.7\times10^{-4}$ | $1.3\times10^{-4}$ | 97% |

Three things follow, and they set the shape of everything below.

1. **This is a kernel error, not a truncation error.** It is flat in step
   (measured at steps 25, 100, 500, 1000, 1500, 2000), flat in resolution (642
   against 2562 vertices), and independent of expansion order — the same
   plateau-in-$N$ pathology the reference treecode has
   ([tasks/treecode.md](treecode.md) §1), from a different cause. Canopy's FMM
   machinery itself is excellent: on the **bare** kernel it matches a brute-force
   sum to $2\times10^{-16}$ relative (measured, log topic *read-only survey*).
2. **The floor is not tunable out of the way on this geometry.** Reaching
   $10^{-4}$ requires 97% of pairs in P2P, i.e. the direct sum with a tree
   attached. It is also what caps the speedup: at factor 4 the near field is a
   fixed ~5% *fraction* of the domain independent of $N$, so the ceiling is a
   bounded ~20x over the direct sum, not an asymptotic win.
3. **A softened far field in Canopy is therefore a precondition** for any claim
   at direct-solve tolerance. It is not implementable inside Canopy's existing
   expansion basis — the solid-harmonic addition theorems (Greengard Thms 5.22,
   5.23, 5.26, cited at `canopy/src/Canopy_LaplaceKernel.hpp:273`, `:373`,
   `:688`) hold because $1/r$ is harmonic and
   $\nabla^2(r^2+b)^{-1/2} = -3b/(r^2+b)^{5/2} \ne 0$. That work is
   [tasks/canopy0.md](canopy0.md) **C1**/**F6**, in the Canopy repository. **X1**
   below states the acceptance test Beatnik applies to it and nothing more.

Everything else in this document is independent of that precondition and worth
building without it: the adapter, the round trip, the solver, the unit tests and
the measurement harness all work identically whichever far field Canopy has, and
the measurement harness is what turns the precondition from an argument into a
number.

**Out of scope.** Any change to Canopy (**X1** names the dependency and its
acceptance test; it does not design it). Any new CLI option — the surface is
closed, and every knob this work needs is already parsed
([examples/02_adaptive_mesh_bubble/InputFile.hpp:444-470](../examples/02_adaptive_mesh_bubble/InputFile.hpp#L444-L470)).
`Face` and `Triangle3` source quadrature, which still throw
([src/Beatnik_SourceQuadrature.hpp:377-473](../src/Beatnik_SourceQuadrature.hpp#L377-L473)).
Periodic boundaries, which this mesh does not have. Restart of Canopy's internal
state.

## Approach

### The physics maps onto one Canopy solve, exactly

Canopy's `NComps` is the number of simultaneous independent charge components,
and with `NComps = 3` and `compute_gradient = true` one traversal produces, per
target $i$, the $3\times3$ tensor
(`canopy/src/Canopy_DownwardSweep.hpp:120-126`, accumulated by
`canopy/src/Canopy_P2P.hpp:883-896`)

$$
T_{cj}(i) \;=\; \texttt{gradient}(i,c,j) \;=\; -\sum_s q_{c,s}\,
  \frac{\delta_j}{(r^2+\varepsilon^2)^{3/2}}, \qquad \delta = x_i - y_s .
$$

That is Beatnik's kernel with $\varepsilon^2 = b$, up to sign
([src/Beatnik_BRSolverBase.hpp:29-31](../src/Beatnik_BRSolverBase.hpp#L29-L31)).
Loading the three charge components with the three components of the
area-weighted source vector makes both contractions Beatnik needs **local
post-processing of that tensor**, with no second traversal and no communication:

- **Velocity.** With $q_c = \omega_s S_{c,s}$, the Birkhoff-Rott sum
  $\sum_s (\delta\times S_s)\omega_s K$ is $-\epsilon_{ijk}T_{kj}$, i.e.
  component-wise $u_0 = T_{12}-T_{21}$, $u_1 = T_{20}-T_{02}$,
  $u_2 = T_{01}-T_{10}$. The cross product is linear in the source strength, so
  it commutes with any expansion of the kernel — which is why it belongs here and
  not inside a kernel.
- **Riesz scalar.** With $q_c = \omega_s G_{c,s}$,
  $\sum_s (\delta\cdot G_s)\omega_s K = -\operatorname{tr} T$.

The two evaluations do **not** share a tensor: the velocity contracts against
the sheet strength and the Riesz scalar against a different vector field. They
are two `solve()` calls over one tree, which Canopy supports directly —
`solve()` re-reads the charge slice and zeroes its outputs each call
(`canopy/src/Canopy_Solver.hpp:196-218`).

$1/4\pi$ and `br_sign` on the velocity, and $-1/4\pi^2$ unsigned on the Riesz
scalar, are applied exactly once, in the adapter, matching `BRSolverDirect`
([src/Beatnik_BRSolverDirect.hpp:126-127](../src/Beatnik_BRSolverDirect.hpp#L126-L127),
[:206-208](../src/Beatnik_BRSolverDirect.hpp#L206-L208)).

### Two decompositions, and the round trip between them

Beatnik's sources are owned mesh vertices under Tessera's decomposition; the
result must land back on the vertex that produced it, on the rank that owns it.
Canopy owns its own decomposition and **permutes and migrates the caller's
array**: `setup` and every maintenance path redistribute particles across ranks
and reorder the local array, with the within-AoSoA order after migration
explicitly unspecified (`canopy/src/Canopy_TreePartitioner.hpp:576-580`), and
nothing in `canopy/src/` carries a caller-supplied identity through it.

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
([src/Beatnik_SourceQuadrature.hpp:194-197](../src/Beatnik_SourceQuadrature.hpp#L194-L197),
[:225-256](../src/Beatnik_SourceQuadrature.hpp#L225-L256)) — and the targets are
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

`FarFieldSolver` ([src/Beatnik_FarFieldInterface.hpp](../src/Beatnik_FarFieldInterface.hpp))
is the only header permitted to name a Canopy type; `BRSolverFMM` names none
today and must still name none afterwards. Canopy's `Solver` is a persistent
object whose tree, partition and communication plan are the thing being reused
across evaluations, and whose `P_ORDER` and `NComps` are **compile-time template
parameters** (`canopy/src/Canopy_Solver.hpp:104-105`). Both facts are Canopy
facts, so both live behind the adapter: the adapter owns the `Solver`, the two
AoSoAs and the first-call flag, and `BRSolverFMM` remains the thirty lines that
turn a `(mesh, geometry, state, quadrature)` tuple into a call.

`P_ORDER` fixed at compile time and `FmmParams::order` being a runtime value
([src/Beatnik_Params.hpp:147-148](../src/Beatnik_Params.hpp#L147-L148)) is a
genuine mismatch. The adapter resolves it by dispatching a runtime order onto an
explicitly enumerated set of instantiations and throwing for anything else — see
the conventions table. It does not silently round.

### The two milestone members: two claims, one binary

A 2000-step FMM-driven trajectory **cannot** be compared against the direct gold
set at the existing `--rtol 1e-10 --atol 1e-12`, and no choice of FMM
configuration changes that. The divergence is measured: a one-ulp seed
($5.6\times10^{-17}$ on `vertices`) grows to $8.5\times10^{-13}$ by step 2000
at level 3 ([tasks/milestone0-progress-log.md:320-332](milestone0-progress-log.md)),
about $10^4$ amplification, power-law rather than exponential. Passing at
$10^{-10}$ from a per-evaluation perturbation would need that perturbation at
$\sim\!10^{-15}$ relative — the direct sum. So each member asserts **two
different claims**, in one binary, one launch:

**Claim A — the per-evaluation bound, and the provable form of "within the same
error tolerance as the direct solve".** The trajectory is driven by
`BRSolverDirect`, so the run is bit-identical to the existing member and all 81
gold comparisons still run at `--rtol 1e-10 --atol 1e-12`; that is what proves it
is the right trajectory. At each of those 81 states the FMM velocity is
additionally evaluated **on the same state** and compared against the direct
velocity, asserting a max relative error $\le \tau_A$. Same input, same state,
no chaotic amplification in the way — the only comparison that isolates the far
field. The claim is met at $\tau_A = 10^{-10}$, the comparator's own rung.

**Claim B — the trajectory, at a measured tolerance.** The same binary then runs
2000 steps FMM-driven and compares against the same gold set as a **ladder**,
reporting the first failing step at each rung rather than passing or failing one
number, exactly as M0-D1 built for the direct path. It also asserts the run
completes 2000 steps, the entity counts never change, and the volume drift
tracks the reference's own series — the checks that would have caught the
full-roll-up blow-up on develop-canopy (`tasks/fmm_premature_nan.md` on that
branch: an unsoftened multipole ~35x too large seeded a runaway to NaN, while the
direct solver completed the identical deck).

Claim B's rung is a measurement, not a choice: **T6** derives it from **T5**'s
$\tau_A$ and the measured amplification, records the derivation in the log, and
compiles in the result. A rung chosen before the measurement is a rung tuned to
whatever the code did.

### Conventions

| Choice | Rule |
| --- | --- |
| Canopy visibility | `Beatnik_FarFieldInterface.hpp` is the **only** header that may name a Canopy type, include a Canopy header, or hold a Canopy object. Every other file goes through `FarFieldSolver`. Verified by `grep -l Canopy src/*.hpp` naming exactly that file and `Beatnik_Config.hpp.in`. |
| Build guard | Everything Canopy-facing sits behind `BEATNIK_ENABLE_CANOPY` ([src/CMakeLists.txt:1-6](../src/CMakeLists.txt#L1-L6), [:78-79](../src/CMakeLists.txt#L78-L79)). A `~canopy` build must still compile every header and still throw the existing configuration error ([src/Beatnik_CreateBRSolver.hpp:66-69](../src/Beatnik_CreateBRSolver.hpp#L66-L69)). |
| Failure behavior | A violated precondition throws `std::logic_error` for "this code is unwritten" and `std::runtime_error` for "this build or configuration cannot do it", matching [src/Beatnik_CreateBRSolver.hpp:45-49](../src/Beatnik_CreateBRSolver.hpp#L45-L49). Never return a truncated or best-effort field: a plausible wrong velocity is the failure mode this whole document exists to bound. |
| New parameters | Added to `FmmParams` ([src/Beatnik_Params.hpp:141-152](../src/Beatnik_Params.hpp#L141-L152)) with a defaulted member and a comment stating units, the meaning of the default, and which Canopy knob it reaches. Never a new constructor parameter, never a new CLI option. |
| Runtime order dispatch | `FmmParams::order` selects among an explicitly enumerated set of `P_ORDER` instantiations; an unsupported value throws naming the supported set. Never silently rounded. The set and the compile-time cost of extending it are documented on the dispatch. |
| Enums over bools | A mode selector is an enum or tag type, never a bool or a magic number. |
| Comments | Units, sign convention, and which side of a difference is which, on the declaration. The sign of Canopy's gradient output and the direction of $\delta$ are the two most misread things on this path and must be stated at every boundary they cross. |
| Provenance | Any routine derived from `origin/develop-canopy`'s `src/FmmBRSolver.hpp`, from Canopy, or from the reference Python cites the file and line range on the routine. |
| Accuracy claims | Every stated tolerance names the source distribution, the rank counts, `P_ORDER`, `ncrit`, `max_depth`, `mac_theta`, `softening` and `near_softening_factor` it was measured at. A bare tolerance is not a claim and may not be compiled into a test. |
| Test tier | New correctness tests are `unit` unless a task says otherwise. **The gate does not change**: it stays at five `regression` members and 60 launches (CLAUDE.md "Minimum test set"). The two new members go in the `milestone` tier, which is not the gate. |
| Formatting | Do not run clang-format, `clangformat.sh` or `cabana-format`. Match the surrounding style by hand. |

### Deliberate deviations

- **No `local` or `clustered` far field, and no treecode.** The CLI already maps
  all three onto `fmm` with a warning
  ([examples/02_adaptive_mesh_bubble/InputFile.hpp:456-463](../examples/02_adaptive_mesh_bubble/InputFile.hpp#L456-L463)),
  and [tasks/treecode.md](treecode.md) §3 costs a standalone treecode port and
  recommends against it on performance grounds. Nothing here revisits that.
- **Canopy's own decomposition is accepted, not fought.** The alternative —
  asking Canopy to adopt the mesh's decomposition — would give up the load
  balancing that is the reason Canopy partitions separately, and Canopy exposes
  no such mode. The tag round trip is the price.
- **The Riesz-scalar path is implemented even though no gold file covers it.**
  The reference raises for `treecode` + `surface-riesz`
  (`mesh_solver.py:605`), so no Python run exercises the combination — recorded
  as risk R5 in [tasks/framework.md](framework.md) and unchanged by this work.
  **T7** implements it against `BRSolverDirect` as the reference, which is what
  the ladder does everywhere else.
- **`auto_maintain` rather than `setup` per evaluation.** `setup` every stage is
  simpler and always correct, but pays a full global tree build plus repartition
  three times per timestep ([tasks/canopy0.md](canopy0.md) F3(b)).
  develop-canopy measured `auto_maintain` returning the cheapest `Migrate`
  action for all 14 calls of a five-step run with results identical to the
  setup-every-step baseline. **T8** measures whether that holds on a deforming
  surface, where canopy0.md F3(c) predicts `Rebalance` instead.
- **No attempt to correct the softening on Beatnik's side.** Handing Canopy
  `softening = 0` for a machine-accurate bare field and adding
  $\sum_{r<R_c}[K_{\rm soft}-K_{\rm bare}]\times S_s$ by direct summation is
  algebraically identical to Canopy's own near-field floor and leaves the same
  residual — the table in **Problem** *is* that residual, measured. It buys
  nothing and is not attempted.

## Current state

- `--br-approximation fmm` **throws** from `BRSolverFMM`'s two virtuals
  ([src/Beatnik_BRSolverFMM.hpp:126](../src/Beatnik_BRSolverFMM.hpp#L126),
  [:153](../src/Beatnik_BRSolverFMM.hpp#L153)) via `BEATNIK_NOT_IMPLEMENTED`
  ([src/Beatnik_Types.hpp:86](../src/Beatnik_Types.hpp#L86)). It throws rather
  than returning a wrong field, which is the safe direction.
- `FarFieldSolver`'s three methods throw the same way
  ([src/Beatnik_FarFieldInterface.hpp:129](../src/Beatnik_FarFieldInterface.hpp#L129),
  [:156](../src/Beatnik_FarFieldInterface.hpp#L156),
  [:187](../src/Beatnik_FarFieldInterface.hpp#L187)). The header states outright
  that Canopy had not been read when it was written, and its `setSources` /
  `evaluateCurl` split does not match Canopy's `setup` / `auto_maintain` /
  `solve` split — **T2** owns the signature change.
- **No Beatnik header includes a Canopy header.** The build already finds and
  links Canopy under `+canopy`
  ([CMakeLists.txt:79-81](../CMakeLists.txt#L79), [src/CMakeLists.txt:78-79](../src/CMakeLists.txt#L78-L79)),
  and the tuolumne environment builds with it
  ([systems/tuolumne/claude.md](../systems/tuolumne/claude.md) §2).
- `FmmParams` carries three members — `mac_theta`, `order`, `ncrit`
  ([src/Beatnik_Params.hpp:141-152](../src/Beatnik_Params.hpp#L141-L152)). It has
  **no** `max_depth`, no `softening`, no `near_softening_factor` and none of
  Canopy's six bounding-box tolerances, all of which `FmmConfig` requires and two
  of which (`ncrit`, `max_depth`) have **no default initializer** in Canopy
  (`canopy/src/Canopy_Solver.hpp:54-55`), so a default-constructed `FmmConfig`
  builds an arbitrary tree.
- The CLI parses `--br-treecode-theta/-order/-ncrit`
  ([examples/02_adaptive_mesh_bubble/InputFile.hpp:478-480](../examples/02_adaptive_mesh_bubble/InputFile.hpp#L478-L480))
  and nothing else FMM-facing. No new option may be added.
- `Beatnik_Test_Milestone0Frozen.cpp` sets
  `p.zmodel.br_approximation = BRApproximation::Direct` explicitly, with a
  comment stating that `fmm` would add an approximation error the comparison
  cannot separate from round-off divergence
  ([tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp:506-511](../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp#L506-L511)).
  That statement is correct and is why **T6** adds members rather than a flag.
- The `milestone` tier has two members and eight launches, measured at **37.25
  minutes** under the runner's `# flux: -t 60m`
  ([scripts/tuolumne/run_milestone.flux:5](../scripts/tuolumne/run_milestone.flux#L5)).
  There is **not** room for two more members of comparable cost.
- Canopy's far field does not carry softening, and no Canopy test measures
  accuracy in any softened configuration or on any non-volumetric source
  distribution ([tasks/canopy0.md](canopy0.md) F4). Canopy's own
  three-component gradient path — the one this work rests on — is labelled
  `unit`, not `regression`, and is known wrong at exactly 4 ranks
  (`max_pot_rel_err = 0.00207` against a $10^{-3}$ budget, `canopy/README.md`
  Known Issues). Beatnik's gate and milestone tier both run at 4 ranks.

## Progress log

[tasks/add-canopy-progress-log.md](add-canopy-progress-log.md) holds what
actually happened: the reasoning behind decisions this document states flatly,
the measured numbers behind its claims, and things only running revealed. **Read
it before implementing any task, changing any signature, or reopening a question
this document treats as settled** — in particular before compiling any tolerance
into any test, since a measured number in the log always outranks an estimate
here.

## Task sequence

### T1 — `FmmParams` carries everything `FmmConfig` needs — **NOT STARTED**

**Depends on:** none.

**Fill in:** [src/Beatnik_Params.hpp](../src/Beatnik_Params.hpp) (`FmmParams`),
[examples/02_adaptive_mesh_bubble/InputFile.hpp](../examples/02_adaptive_mesh_bubble/InputFile.hpp)
(only where an already-parsed key must reach a new member),
[README.md](../README.md).

**Reference:** `FmmConfig`'s full member list and the two members with no
default initializer (`canopy/src/Canopy_Solver.hpp:53-78`); the six bounding-box
padding factors and their meaning (`:44-52`); the `near_softening_factor`
comment, which states the far field is unsoftened (`:70-77`);
`FmmParams` as it stands ([src/Beatnik_Params.hpp:141-152](../src/Beatnik_Params.hpp#L141-L152));
develop-canopy's `makeCanopyConfig` for the full mapping it needed
(`src/FmmBRSolver.hpp:588-607` on that branch).

**Do:**

1. Extend `FmmParams` with every knob the adapter must set: `max_depth`,
   `softening_factor` (the multiple of `ZModelParams::eps` handed to Canopy —
   see step 3), `near_softening_factor`, `ncrit_tol`, `replication_depth`,
   `imbalance_tolerance`, and the six bounding-box padding factors. Every one
   gets a default initializer and a comment naming units, the default's meaning,
   and the `FmmConfig` member it reaches.
2. Choose and document a `max_depth` default. It has no counterpart in the
   treecode knob set, is bounded at 19 by the `uint64_t` Morton key
   (`canopy/src/Canopy_TreeBuilder.hpp:176-181`), and sets the finest cell width
   as (root box width) / $2^{\rm max\_depth}$. develop-canopy ran 19 and hit a
   depth-driven finite-difference blow-up at roll-up; state the reasoning for
   whatever is chosen.
3. **Do not add a `softening` member that duplicates `eps`.** Canopy's
   `softening` is a length and Beatnik's `blob()` is a squared length —
   $\varepsilon^2$ under `Length` and $\varepsilon$ under `Matlab`
   ([src/Beatnik_Params.hpp:125-128](../src/Beatnik_Params.hpp#L125-L128)) — so
   the adapter must pass $\sqrt{\texttt{blob()}}$, which is `eps` under `Length`
   and $\sqrt{\texttt{eps}}$ under `Matlab`. Deriving it from `blob()` at the one
   call site is what makes the two modes correct without a second source of
   truth. State this on the declaration; getting it wrong is a silent factor of
   $\varepsilon$ in the softening length.
4. `FmmParams` must not be constructible into a state that builds an arbitrary
   Canopy tree. Since every Beatnik member is default-initialized, this reduces
   to giving `ncrit` and `max_depth` defensible values and validating them where
   the adapter builds the `FmmConfig`.
5. Update README's parameter documentation in the same change.

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

**Fill in:** [src/Beatnik_FarFieldInterface.hpp](../src/Beatnik_FarFieldInterface.hpp)
(the whole class body and its three signatures),
[src/CMakeLists.txt](../src/CMakeLists.txt) if the header's guard structure
changes, [README.md](../README.md).

**Reference:**

- `origin/develop-canopy`'s `src/FmmBRSolver.hpp` — the working predecessor.
  `packGridParticles` (`:150-196`), `buildForwardDistributor` and its
  five-step tag-reverse handshake (`:198-296`), the full pipeline including the
  first-call branch (`:325-480`), the cross-product contraction (`:414-426`),
  the reverse distribute and scatter (`:428-470`), and the persistent-state
  members with the reasoning on each (`:609-644`).
- Canopy's API: `FmmConfig` (`canopy/src/Canopy_Solver.hpp:53-78`), the
  constructor and its softening branch (`:151-179`),
  `setup<PositionIdx, ChargeIdx>` and its "count BEFORE migration" contract
  (`:181-196`), `solve<PositionIdx, ChargeIdx>(particles, compute_gradient)`
  (`:198-239`), `auto_maintain<PositionIdx, ChargeIdx>` and its three-way
  `MaintenanceAction` (`:348-363`, `:120-126`), and the accessors
  `num_local_particles()`, `potential()`, `gradient()` (`:472-475`).
- The gradient's shape and sign (`canopy/src/Canopy_DownwardSweep.hpp:120-126`,
  `canopy/src/Canopy_P2P.hpp:883-896`), and that P2P skips pairs with
  $r^2 < 10^{-24}$ (`:881`) — for this kernel the self term contributes exactly
  zero to the gradient, so skipping it is correct rather than merely tolerable.
- What the quadrature hands over: owned rows only, one source per owned vertex,
  in owned order
  ([src/Beatnik_SourceQuadrature.hpp:194-197](../src/Beatnik_SourceQuadrature.hpp#L194-L197),
  [:225-256](../src/Beatnik_SourceQuadrature.hpp#L225-L256)), and the R9
  discipline that makes emitting a ghost a rank-count-dependent magnitude error
  ([:216-220](../src/Beatnik_SourceQuadrature.hpp#L216-L220)).
- The prefactors and signs to reproduce
  ([src/Beatnik_BRSolverDirect.hpp:126-127](../src/Beatnik_BRSolverDirect.hpp#L126-L127),
  [:206-208](../src/Beatnik_BRSolverDirect.hpp#L206-L208)) and the convention
  block they come from
  ([src/Beatnik_BRSolverBase.hpp:34-52](../src/Beatnik_BRSolverBase.hpp#L34-L52)).

**Do:**

1. **Replace the three signatures rather than adding overloads beside them.**
   `setSources( source_points )` cannot express what Canopy needs: the charges
   must be present at `setup` time (`setup<PositionIdx, ChargeIdx>` reads both
   slices), and the maintenance/solve split is not where the current interface
   puts it. Delete the current three methods and their doc comments and write
   the shape Canopy forces. There are exactly **two** callers to update, both
   stubs today: `BRSolverFMM::computeInterfaceVelocity`
   ([src/Beatnik_BRSolverFMM.hpp:113](../src/Beatnik_BRSolverFMM.hpp#L113)) and
   `::computeSurfaceRieszScalar` ([:140](../src/Beatnik_BRSolverFMM.hpp#L140)).
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
   step 3, and validate it: throw naming the offending member if `ncrit` or
   `max_depth` is non-positive, or `max_depth > 19`.
5. Dispatch `FmmParams::order` onto an enumerated set of `P_ORDER`
   instantiations. Start with the set the measurement needs — at minimum the
   value **T5** will scan over — and throw naming the set for anything else.
   develop-canopy ran `P_ORDER = 10` after finding 6 insufficient at roll-up
   (`src/FmmBRSolver.hpp:46` on that branch; `tasks/fmm_premature_nan.md`
   Background).
6. Every rank must enter every collective the same number of times per
   evaluation, **including a rank that owns zero sources**. Canopy has no test
   for a zero-particle rank (canopy0.md F5) and Beatnik's decomposition can
   produce one. Do not branch the collective sequence on a local count.
7. `#include <Canopy_Solver.hpp>` and every Canopy-typed member sit behind
   `BEATNIK_ENABLE_CANOPY`; the class must still compile, and its methods must
   still throw `std::runtime_error` naming the missing build option, in a
   `~canopy` build.

**Exit criterion:** `spack install` succeeds with `+canopy` **and** with the
Canopy dependency's headers made unavailable (or with
`Beatnik_ENABLE_CANOPY=OFF` configured by hand), the second build's
`FarFieldSolver` throwing `std::runtime_error` naming `+canopy` rather than
failing to compile; `grep -l Canopy src/*.hpp` names only
`Beatnik_FarFieldInterface.hpp` and `Beatnik_Config.hpp.in`. No behavioral claim
is made by this task — **T4** is where correctness is first checked.

---

### T3 — `BRSolverFMM::computeInterfaceVelocity` — **NOT STARTED**

**Depends on:** T2.

**Fill in:** [src/Beatnik_BRSolverFMM.hpp](../src/Beatnik_BRSolverFMM.hpp)
(`computeInterfaceVelocity` only; `computeSurfaceRieszScalar` is T7),
[README.md](../README.md).

**Reference:** the direct implementation this must agree with, step for step
([src/Beatnik_BRSolverDirect.hpp:105-166](../src/Beatnik_BRSolverDirect.hpp#L105-L166))
— note it reallocates the output to `ownedVertexCount()` and zeroes it before
accumulating ([:112-115](../src/Beatnik_BRSolverDirect.hpp#L112-L115)), calls
`quadrature.generate` itself ([:117-119](../src/Beatnik_BRSolverDirect.hpp#L117-L119)),
and applies `br_sign/4\pi` once ([:126-127](../src/Beatnik_BRSolverDirect.hpp#L126-L127));
the caller's contract ([src/Beatnik_ZModelSolver.hpp:219-224](../src/Beatnik_ZModelSolver.hpp#L219-L224)),
which reallocates `vertex_dot` to the owned count and expects the prefactors
already applied.

**Do:**

1. Generate sources through the quadrature, call the adapter, write the
   `(N_owned, 3)` velocity. Overwrite, do not accumulate — the declaration says
   overwritten ([src/Beatnik_BRSolverBase.hpp:137-139](../src/Beatnik_BRSolverBase.hpp#L137-L139)).
2. Update the file header: the two `@note` blocks currently describe an FMM that
   expands the softened kernel
   ([:102-107](../src/Beatnik_BRSolverFMM.hpp#L102-L107)), and
   `Beatnik_FarFieldInterface.hpp:36-41` says the same. It does not. Replace
   both with what is true, the measured floor from **Problem**, and a pointer to
   this document.
3. Same correction in [README.md](../README.md): whatever it says about the FMM
   path's accuracy must become the measured statement, and it must say the
   default `--br-approximation fmm`
   ([src/Beatnik_Params.hpp:106](../src/Beatnik_Params.hpp#L106)) is not the
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
[tests/unit_tests/CMakeLists.txt](../tests/unit_tests/CMakeLists.txt)
(`BEATNIK_UNIT_TEST_SOURCES`).

**Reference:** the tier's registration and its "self-validating, non-zero on
failure" contract
([tests/unit_tests/CMakeLists.txt:11-40](../tests/unit_tests/CMakeLists.txt#L11-L40)),
the assertion helper (`tests/unit_tests/Beatnik_TestAssert.hpp`), and
`Beatnik_Test_TangentialRelaxation.cpp` as the tier's rank-count-aware precedent;
`Beatnik_Test_Milestone0Frozen.cpp`'s `makeParams` for a params set that has
already been measured against
([tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp:474-575](../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp#L474-L575));
develop-canopy's `tests/tstFmmVsExact.hpp` for the shape of the comparison it
made.

**Do:**

1. Build a solver at the milestone-0 configuration, `setup()`, and advance a
   handful of steps with the **direct** solver so the state is a real one with a
   non-zero sheet strength. At step 0 the sheet strength is identically zero
   (`--initial-potential-strength 0`), so a step-0 comparison is vacuous — the
   test must assert the strength is non-zero before comparing.
2. Construct both BR solvers and evaluate **both on that same state**, then
   assert max relative and max absolute velocity error. Both directions:
   the error is under budget, **and** a deliberately degraded configuration
   (`near_softening_factor` = 1) is over it — a comparison that has only ever
   seen agreeing data has not been tested.
3. Rank counts 1-6, since that is the gate's sweep and Canopy's
   three-component gradient path is known wrong at exactly 4
   (canopy/README.md Known Issues). If 4 ranks fails, that is the finding: record
   it, and do not widen the budget to accommodate it — see **R4**.
4. Include a variant where at least one rank owns zero sources, if the
   decomposition can be made to produce one at these vertex counts; if it
   cannot, say so in the log rather than leaving the case silently uncovered.
5. Assert the FMM result is finite everywhere. A NaN reaching the caller is what
   develop-canopy's roll-up failure looked like from outside.

**Exit criterion:** `ctest -R Beatnik_Test_FmmVsDirect` passes at ranks 1-6 (and
`BEATNIK_UNIT_RANKS=4 flux batch scripts/tuolumne/unit_tests.flux` passes in
spack mode), asserting a max relative velocity error under a budget whose value
and full qualification list are recorded in the log; and the same test fails,
naming the softening bias, when built with `near_softening_factor = 1`.

---

### T5 — Measure the achievable far-field fidelity, and publish it — **NOT STARTED**

**Depends on:** T4. **This task produces the number every later task keys off.**

**Fill in:** a measurement driver under `tests/regression_tests/` registered in
the "Measurement drivers — IN NO TIER" section
([tests/CMakeLists.txt:536-570](../tests/CMakeLists.txt#L536-L570)); a batch
script under `scripts/tuolumne/`;
[tasks/add-canopy-progress-log.md](add-canopy-progress-log.md);
[README.md](../README.md).

**Reference:** the survey measurement this must reproduce and extend, with its
method and provenance (log topic *read-only survey*); the mechanism being
measured (`canopy/src/Canopy_CommunicationPlan.hpp:338-361`); Canopy's
finite-difference far-field gradient, which contributes its own much lower
plateau and must not be mistaken for the softening bias
(`canopy/src/Canopy_LaplaceKernel.hpp:851-879`, and canopy0.md F7); the
measurement-driver loop's rule that it appends to **neither** manifest
([tests/CMakeLists.txt:548-556](../tests/CMakeLists.txt#L548-L556)).

**Do:**

1. Scan, on real milestone-0 states at both subdivision levels: `P_ORDER` over
   the dispatched set, `near_softening_factor` over at least {4, 8, 12, 20},
   `mac_theta`, `ncrit`, `max_depth`. Report max relative and max absolute
   velocity error against `BRSolverDirect` on the **same** state, and the
   fraction of pairs Canopy handled in P2P.
2. **Read the scan as a scan.** Truncation error falls as `P_ORDER` rises; the
   softening bias plateaus; Canopy's finite-difference L2P contributes a third,
   much lower plateau. Three components, not two — "the error stopped falling"
   is not by itself the softening bias. Record which plateau each observed level
   is.
3. Confirm or correct the survey's table in **Problem**. It was measured outside
   Beatnik, with an exact pairwise cutoff rather than Canopy's cell-centre MAC,
   and it carries no FMM truncation error at all — so it is a *floor* on
   Canopy's achievable error, and Canopy's own number must be at least as large.
   A measured number below it means the method being measured is not the method
   the table models; find out which before believing it.
4. Record $\tau_A$ — the best max relative velocity error achieved, with its
   full qualification list — in the log, and publish the validated parameter set
   and the achieved fidelity in README.
5. Measure the amplification **T6** needs: apply a perturbation of size
   $\tau_A$ to one velocity evaluation of a direct run and report the resulting
   `vertices` divergence at steps 25, 100, 500, 1000, 2000. That is what turns
   $\tau_A$ into claim B's rung instead of a guess. The direct path's own
   measured growth series ([tasks/milestone0-progress-log.md:320-332](milestone0-progress-log.md))
   is the comparison.

**Exit criterion:** the log carries the full scan with every entry's
qualification list, a stated $\tau_A$, an explicit statement of which of the
three plateaus binds, and the perturbation-to-trajectory mapping from step 5;
README carries the validated parameter set and the achieved fidelity for the
gradient. The task is complete whichever value $\tau_A$ takes — if it is
$2\times10^{-2}$, that is the finding, and **X1** is what it implies.

---

### T6 — The two milestone-tier FMM members — **NOT STARTED**

**Depends on:** T5 (for $\tau_A$ and claim B's rung) and T4 (for the comparison
harness).

**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp` and
`Beatnik_Test_Milestone0FmmL4.cpp` (new),
[tests/CMakeLists.txt](../tests/CMakeLists.txt)
(`BEATNIK_MILESTONE_TEST_SOURCES` and the two `_beatnik_args_<stem>_abs` /
`_rel` pairs), [scripts/tuolumne/run_milestone.flux](../scripts/tuolumne/run_milestone.flux)
(the walltime), [CLAUDE.md](../CLAUDE.md) ("Minimum test set", the tier's member
count and launch count), [README.md](../README.md).

**Reference:**

- The existing members, whose structure these mirror:
  [tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp](../tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp)
  in full — its per-level literal blocks (`:295-403`), `goldForStep`'s
  suffix-based lookup and why it is not built from a time (`:412-440`),
  `runComparator`'s four distinguished outcomes (`:451-468`), `makeParams`
  (`:474-575`), the owned-count partition check (`:682-712`), the volume-drift
  check against the reference's own series (`:760-788`), `compareStep`
  (`:795-814`), the step loop (`:837-900`), the negative case and why exit 2
  must not be accepted (`:926-945`), and the cost report (`:952-987`).
  [Beatnik_Test_Milestone0FrozenL4.cpp](../tests/regression_tests/Beatnik_Test_Milestone0FrozenL4.cpp)
  is the whole second member: three lines.
- The tier's registration loop, which keys arguments by source stem and
  `FATAL_ERROR`s on a source with no argument list
  ([tests/CMakeLists.txt:475-535](../tests/CMakeLists.txt#L475-L535)), and the
  tier's rank set as a property of the tier
  ([:438-439](../tests/CMakeLists.txt#L438-L439)).
- The measured cost that constrains this: 37.25 minutes for the tier's current
  eight launches under `# flux: -t 60m`
  ([scripts/tuolumne/run_milestone.flux:5](../scripts/tuolumne/run_milestone.flux#L5)).

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
   $\le \tau_A$. Report the per-step value at 17 digits so the series is in the
   log without re-running.
3. **Claim B.** Then run 2000 steps FMM-driven and compare against the same gold
   set as a **ladder** — report the first failing step at each of several rungs
   rather than asserting one — and assert: the run reaches step 2000; the entity
   counts never change (both paths of the existing member: Tessera's global
   counts every step and an `MPI_Allreduce` over owned counts at every compared
   step); the volume drift tracks `kRefVolumeDrift` within a measured bound;
   every velocity is finite. The compiled rung comes from **T5** step 5.
4. **A stop is a reported stop step, never a shorter pass**, and a comparator
   exit of 2 (could not load) is never conflated with 1 (compared and
   disagreed). Both properties are the existing member's and both are load-bearing
   here: claim B's whole purpose is to notice the FMM destabilizing the physics.
5. Keep the existing member's negative case — the final state against the step-0
   gold, which must exit exactly 1 — and add one for claim A: a state
   deliberately perturbed by more than $\tau_A$ must fail the same-state
   comparison. A bound that has only ever seen agreeing data has not been tested.
6. Register both stems with their own gold directory, add the two argument-list
   pairs, and **raise the runner's walltime**: the tier goes from 2 members and
   8 launches to 4 and 16, and each new launch runs two 2000-step trajectories
   plus 81 extra FMM evaluations. Measure the tier run and set `-t` from the
   measurement, not from an estimate.
7. Update CLAUDE.md's "Minimum test set" tier paragraph with the new member
   count and launch count, and state explicitly that **the gate is unchanged** —
   still five `regression` members and 60 launches.

**Additional information needed:** claim B's rung, and whether $\tau_A$ from
**T5** permits claim A at $10^{-10}$ at all. Both come from **T5**. If $\tau_A$
is far above $10^{-10}$, this task still lands — it compiles in the measured
$\tau_A$, and README and the log state plainly that the direct-solve-tolerance
claim is pending **X1**. It does not loosen $10^{-10}$ silently and it does not
wait.

**Exit criterion:** `ctest -L milestone -R Milestone0Fmm` passes at ranks 1 and 4
on SERIAL and HIP, and
`flux batch scripts/tuolumne/run_milestone.flux` reports all 16 launches green
inside its walltime; each member's log carries the 81-entry claim-A error series
and claim B's per-rung first-failing-step table; and both negative cases fire —
the final state against the step-0 gold exits exactly 1, and the perturbed-state
claim-A case fails for the stated reason rather than merely exiting non-zero.

---

### T7 — The Riesz-scalar path — **NOT STARTED**

**Depends on:** T3.

**Fill in:** [src/Beatnik_BRSolverFMM.hpp](../src/Beatnik_BRSolverFMM.hpp)
(`computeSurfaceRieszScalar`),
[src/Beatnik_FarFieldInterface.hpp](../src/Beatnik_FarFieldInterface.hpp) (the
dot contraction), a case added to **T4**'s test.

**Reference:** the direct implementation
([src/Beatnik_BRSolverDirect.hpp:184-241](../src/Beatnik_BRSolverDirect.hpp#L184-L241)),
including that it is **not** multiplied by `br_sign` while the velocity is — an
asymmetry reproduced deliberately ([:203-208](../src/Beatnik_BRSolverDirect.hpp#L203-L208));
`generateGradient` and the two state models' different expressions for $G$
([src/Beatnik_SourceQuadrature.hpp:256-290](../src/Beatnik_SourceQuadrature.hpp#L256-L290));
the caller, which is collective and reached the same number of times on every
rank ([src/Beatnik_ZModelSolver.hpp:248-257](../src/Beatnik_ZModelSolver.hpp#L248-L257));
that this is a **second** `solve()` over the same tree with different charges,
which Canopy supports (`canopy/src/Canopy_Solver.hpp:198-218`).

**Do:** contract $\Psi = -\operatorname{tr}T$ with the $-1/4\pi^2$ prefactor and
no `br_sign`. Reuse the tree and the round trip from the velocity evaluation
where the state has not changed between them; if that reuse is not safe, say why
in the log and pay the second round trip rather than reusing a stale partition.
There is no gold file for this combination and none can exist (the reference
raises, `mesh_solver.py:605`), so `BRSolverDirect` is the reference.

**Exit criterion:** **T4**'s test gains a Riesz-scalar case that passes at ranks
1-6 against `BRSolverDirect` on the same state, at a budget recorded with its
qualification list; and a `--bernoulli-scalar-mode surface-riesz
--br-approximation fmm` two-step run completes at 1 and 4 ranks.

---

### T8 — Cost: the maintenance policy and the actual speedup — **NOT STARTED**

**Depends on:** T4.

**Fill in:** a batch script under `scripts/tuolumne/`;
[tasks/add-canopy-progress-log.md](add-canopy-progress-log.md);
[README.md](../README.md) (and its "Future Optimizations" section only if the
user approves an entry).

**Reference:** what `migrate()` actually costs, per call
([tasks/canopy0.md](canopy0.md) F3(b), citing
`canopy/src/Canopy_Solver.hpp:241-305` and `:625-645`); the prediction that a
deforming surface picks `Rebalance` rather than `Migrate` essentially every
stage, and that a plan rebuild is a serial host-side dual-tree traversal
executed on every rank (F3(c), citing
`canopy/src/Canopy_CommunicationPlan.hpp:549-665`); develop-canopy's measured
action histogram — all 14 `auto_maintain` calls of a five-step run returned
`Migrate`, and a 1400-step roll-up run returned
`Migrate=1180 Rebalance=3019 Rebuild=0` (`tasks/integrate_canopy.md` row AM and
`tasks/fmm_premature_nan.md` Validation, on that branch); the profiling
convention `BEATNIK_SCOPED_TIMER_DETAILED` and the action-histogram destructor
(`src/FmmBRSolver.hpp:122-140`, `:669-680` on that branch).

**Do:**

1. Report the `MaintenanceAction` histogram per run, and the wall-clock split
   across pack, forward distribute, forward migrate, `auto_maintain`, `solve`,
   contract, reverse distribute, reverse migrate, scatter. Nine of these per
   timestep is the cost model that decides whether this path is worth running.
2. Measure `fmm` against `direct` wall-clock per step at several vertex counts,
   and report the **P2P pair fraction** alongside, since the near-field floor is
   what caps the speedup. State the crossover vertex count, or state that there
   is none in the measured range.
3. Do not change the maintenance policy inside this task. If the measurement
   says `auto_maintain` is the wrong choice, that is a finding and a follow-up
   task, not a drive-by edit.

**Exit criterion:** the log carries the action histogram, the per-phase
wall-clock split and the `fmm`-versus-`direct` per-step comparison with the P2P
pair fraction at each size, all at stated vertex counts, rank counts and
backends; README states the measured speedup and the vertex count above which
`fmm` is faster, or states that none was found.

---

### X1 — External precondition: a softened far field in Canopy — **NOT STARTED, NOT IMPLEMENTED HERE**

**Depends on:** T5 (which measures how much it is needed).

This task is **not** Beatnik work and this document does not design it. It
exists so the dependency is named, its acceptance test is stated in Beatnik's
terms, and no session mistakes its absence for an oversight.

**Where the work lives:** [tasks/canopy0.md](canopy0.md) **C1**, whose option
set and cost are in **F6** and **F8** of that document. The short version: the
solid-harmonic addition theorems require harmonicity, the Plummer potential is
not harmonic, so a softened far field is a **second expansion basis** alongside
the existing one — plus real-valued storage through both sweeps, and a re-keyed
M2L operator cache, since softening introduces the absolute length scale
$\sqrt b$ that the cache's documented "no physical width enters" property
(`canopy/src/Canopy_DownwardSweep.hpp:1075`) depends on not existing.

**What Beatnik requires of it, and what it does not:**

- The kernel Canopy's far field expands must be
  $\delta\,(b+r^2)^{-3/2}$ with the **same** $b$ P2P uses, so
  `near_softening_factor` becomes unnecessary rather than load-bearing.
- Beatnik requires **no interface change**: `FmmConfig`, `setup`, `solve`,
  `auto_maintain` and the gradient's shape and sign all stay as they are. If the
  work changes any of them, T2's adapter is the only Beatnik file that moves.
- Beatnik does **not** require $10^{-6}$ from a low-order Cartesian-Taylor
  basis, which canopy0.md F6(d) shows is out of reach. It requires whatever
  $\tau_A$ claim A asserts, and **T5** is what states that number.

**Exit criterion**, which is the acceptance test in Beatnik's terms and the only
part of this task Beatnik owns: **T4**'s test passes at ranks 1-6 with
$\tau_A \le 10^{-10}$ and with `near_softening_factor = 0` (the floor disabled
entirely), and **T6**'s claim A passes at $10^{-10}$ at all 81 states at both
levels. Passing with the floor disabled is the discriminator: it is what
distinguishes a far field that carries the softening from one that merely avoids
the pairs where it matters.

## Known risks

**R1 — The softening bias is mistaken for expansion error and "fixed" by raising
`P_ORDER`.** Both present as a relative-error figure over budget. The
distinguishing measurement is the scan: truncation error falls as `P_ORDER`
rises, the softening bias plateaus. develop-canopy hit exactly this and recorded
the resolution — `P_ORDER` 10→16 gave **byte-identical** results
(`tasks/fmm_premature_nan.md` Resolution), which is the signature of a wrong
kernel rather than an under-resolved expansion. The scan has **three** plateaus,
not two: Canopy's finite-difference L2P contributes its own at
$\sim\!10^{-10}$ (canopy0.md F7). No tolerance may be set before **T5** has
been read.

**R2 — A tolerance gets chosen before it is measured.** The failure is
structural, not careless: **T6** cannot be written without a number, so a session
that starts T6 before T5 will invent one, and an invented tolerance is
indistinguishable in the source from a measured one. T6's dependency on T5 is
therefore not an ordering preference. Every compiled tolerance carries its
qualification list in a comment, and a tolerance without one is not a claim.

**R3 — The round trip silently drops or duplicates a source.** A tag mismatch
does not crash; it produces a velocity that is wrong on some vertices, or wrong
everywhere by a factor that changes with the rank count — the same signature as
the R9 ghost-emission bug the quadrature warns about
([src/Beatnik_SourceQuadrature.hpp:216-220](../src/Beatnik_SourceQuadrature.hpp#L216-L220)).
**T4**'s rank sweep is what catches it, which is why 1-6 and not just 1. A
cheap independent check worth having in the test: the global source count
Canopy reports must equal the global owned vertex count.

**R4 — Canopy's np=4 defect is absorbed into Beatnik's budget.** Canopy's
three-component gradient path fails at exactly 4 ranks at
$2\times$ over a $10^{-3}$ budget (canopy/README.md Known Issues, and
canopy0.md C6). Beatnik's gate runs at 4 ranks and the milestone tier runs at 1
and 4. An error that appears at exactly one rank count is a decomposition-bug
signature, not a budget signature: if **T4** fails at 4 ranks only, record it and
raise it upstream — do **not** widen the budget, and do not drop 4 from the
sweep.

**R5 — `solve()` after motion without maintenance returns a defined-but-wrong
field.** Canopy's `solve()` uses the leaf membership, communication plan and P2P
neighbour lists cached by the *last* setup or maintenance call, so a particle
that has moved out of its leaf still contributes to its old leaf's multipole and
still gets its old leaf's near-field list. Nothing raises (canopy0.md F3(a)).
This is the single most dangerous property of the API for this consumer, because
the three RK stages each move every source. **T2** must call maintenance before
every `solve()`, unconditionally, and must not add a "the positions barely moved"
fast path — that is canopy0.md C3's job, upstream, where the precondition can be
checked.

**R6 — The milestone tier outgrows its walltime and starts reporting a timeout
as a failure.** The tier is at 37.25 of 60 minutes with eight launches, and
**T6** takes it to sixteen launches each doing roughly twice the work. A
scheduler kill and a real failure look similar in a log skimmed quickly. **T6**
step 6 must measure the tier run and set `-t` from it; if the honest number is
large enough to be unwieldy, splitting claim A and claim B into separate members
is the fallback, at the cost of a third 2000-step trajectory.

**R7 — Claim A passes and the FMM still destroys the physics.** Claim A measures
the far field on states the *direct* solver produced. develop-canopy's failure
was not there: the FMM tracked the exact solver acceptably for 1362 steps and
then a single corrupted node seeded a runaway to whole-field NaN
(`tasks/fmm_premature_nan.md` Background). Claim B is the guard, and it is why
the two claims are in one binary rather than claim A alone. Its finiteness,
entity-count and volume-drift assertions are the parts that fire in that
scenario — a gold comparison at a loose rung would not.

**R8 — The measured floor is treated as an estimate and re-litigated.** The
table in **Problem** is a measurement on the real gold states, flat across six
steps, two levels and four factor values. It is not a bound derived from a
formula, and the $\tfrac32\varepsilon^2/R^2$ per-pair expression in canopy0.md
F1 is the (pessimistic by ~5x) formula it supersedes. **T5** confirms it in-tree
with Canopy's own MAC and its own truncation error, and it can come out
*larger*; a session that measures something much smaller has measured a
different quantity — most likely a configuration in which the far field never
engages, which at these vertex counts is easy to hit by accident and which
[tasks/treecode.md](treecode.md) §1 documents as the same trap on the treecode
side.

**R9 — Progress stalls waiting on X1.** Every task except T6's tolerance is
independent of the Canopy-side kernel work, and T6 lands with whatever $\tau_A$
**T5** measures. The failure mode is a session reading **X1** as a gate on the
whole document. It is not: the deliverable without X1 is a working, measured,
bounded-error fast path plus a stated gap, which is strictly more useful than a
stalled one.
