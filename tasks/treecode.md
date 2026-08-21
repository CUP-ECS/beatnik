# The reference treecode — what it is, how accurate it is, what porting it costs

**Status:** FINDINGS ONLY. Nothing here is implemented, and nothing here changes
a decision yet. Written to answer three questions asked against
[`framework.md`](framework.md)'s recorded deviation *"The treecode is replaced,
not ported"* (line 125-129).

**Sources read.** `~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`
(138 lines, the whole algorithm), `zmodel3d/mesh_solver.py` (the dispatch and the
params), `examples/run_adaptive_mesh_bubble.py` (the CLI),
`scripts/benchmark_br.py`, the reference `README.md`, and on the Beatnik side
[src/Beatnik_BRSolverBase.hpp](../src/Beatnik_BRSolverBase.hpp),
[src/Beatnik_BRSolverDirect.hpp](../src/Beatnik_BRSolverDirect.hpp),
[src/Beatnik_BRSolverFMM.hpp](../src/Beatnik_BRSolverFMM.hpp),
[src/Beatnik_SourceQuadrature.hpp](../src/Beatnik_SourceQuadrature.hpp),
[src/Beatnik_Params.hpp](../src/Beatnik_Params.hpp),
[src/Beatnik_Types.hpp](../src/Beatnik_Types.hpp),
[examples/02_adaptive_mesh_bubble/InputFile.hpp](../examples/02_adaptive_mesh_bubble/InputFile.hpp).
**`../canopy` was not opened**, per the question as asked; every FMM statement
below is about the classical algorithm from the literature, not about Canopy's
implementation of it.

New measurements in this document were produced by a throwaway sweep script
(`/tmp/stewartj/treecode_sweep.py`, not committed) driving the reference's own
`potential_mesh_birkhoff_rott_velocity` on a login node — serial NumPy, no
scheduler, no Beatnik build involved. The reference tree is read-only and was
not modified.

---

## 0. What the treecode is, in one paragraph

`treecode.py` is a **Barnes-Hut octree** evaluation of exactly the same
discrete sum `BRSolverDirect` computes:

```
u(x_t) = (1/4pi) sum_s  delta_ts x S_s / (b + |delta_ts|^2)^{3/2},   delta_ts = x_t - y_s
```

It builds one octree over the *source* points (`_build`, lines 27-53), stores a
monopole/dipole/quadrupole moment set on every node, and then for each target
walks the tree from the root: a node whose radius is small enough relative to its
distance from the target is **accepted** and evaluated through a truncated Taylor
expansion of the kernel about the node centre (`_expansion_batch`, lines 56-81);
otherwise the walk descends into its children, and at a leaf that is still not
accepted the sources are summed directly (`_direct_batch`, lines 84-93). It
returns the same unsigned, `1/4pi`-included quantity as the direct evaluator, so
the caller-side `br_sign` handling in
`mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sources` (lines 420-431) is
identical for both paths. **It is the reference's default**
(`--br-approximation treecode`, `parse_args` line 233) and the only fast path the
reference's documented "configuration (a)" workhorse uses.

Three details are load-bearing and easy to lose in a port:

1. **The expansion is blob-aware.** `_expansion_batch` expands
   `K(r) = r/(b + r^2)^{3/2}`, *with* `b` carried through — `w = |r|^2 + b`
   appears in every denominator (line 57). This is not the bare Coulomb
   expansion with a softened near field bolted on. The reference's own note says
   why (`benchmark_br.py` line 82-83): expanding the bare kernel makes the
   roll-up develop spurious structure, because at the separations that matter on
   a self-approaching sheet `b` is comparable to `r^2`.
2. **The acceptance test uses an exact node radius, not a box diagonal.**
   `node.radius` is `max_j |y_j - centre|` over the node's own sources (line 31),
   and acceptance is `node.radius < theta * |x_t - centre|` (line 124). Cheap and
   tighter than a geometric box bound.
3. **The moments are computed directly, per node, over that node's whole source
   set** — `G = sum g`, `D = sum d ⊗ g`, `Q = sum d ⊗ d ⊗ g` with
   `d = y - centre` (lines 33-35), evaluated *before* the node is subdivided.
   That is `O(N log N)` work done redundantly at every level rather than one
   upward M2M pass; see §2.

A note that matters for the port's scope: **the treecode covers the velocity
only.** The surface Riesz scalar path
(`_source_riesz_scalar`, `mesh_solver.py:588-605`) supports `direct`, `local` and
`clustered` and **raises for `treecode`** (line 605). This is already recorded on
the Beatnik side at [Beatnik_BRSolverBase.hpp:163-167](../src/Beatnik_BRSolverBase.hpp#L163-L167)
and [Beatnik_FarFieldInterface.hpp:165-170](../src/Beatnik_FarFieldInterface.hpp#L165-L170).

---

## 1. How accurate is it against the direct solve?

**Answer: about `1e-3` relative velocity at the reference's own defaults
(`theta=0.3`, `order=2`, `ncrit=64`), and that error is essentially independent
of mesh size — it does not converge as `N` grows.** The reference's README makes
the same claim ("agrees with the direct sum to ~1e-3 relative velocity at
theta=0.3"); the sweep below confirms it and fills in the shape of the
`theta`/`order` trade.

Measured on the reference's own benchmark configuration (icosphere radius 0.5,
`potential = 0.5x + 0.3y`, `A=0.3 g=1.0 eps=0.025 mu=0.002`,
`use_matlab_blob=False` so `b = eps^2 = 6.25e-4`). `relmax` is
`max_t |u_tree - u_direct| / max_t |u_direct|`; `relL2` is the Frobenius-norm
relative error. **`vertex` quadrature is the one Beatnik implements**, so it is
the primary table.

### `vertex` quadrature, N = 10242 sources (icosphere subdivisions 5)

Direct reference: 16.17 s per evaluation, `|u|max = 3.604e-01`.

| order | theta=0.7 | 0.5 | 0.3 (default) | 0.2 | 0.1 |
| --- | --- | --- | --- | --- | --- |
| 0 (monopole) | 1.3e-01 | 6.7e-02 | 2.5e-02 | 1.0e-02 | 2.1e-03 |
| 1 (+dipole) | 5.3e-02 | 1.8e-02 | 2.9e-03 | 8.0e-04 | 8.3e-05 |
| **2 (+quadrupole, default)** | 3.8e-02 | 8.1e-03 | **8.1e-04** | 2.4e-04 | 5.9e-06 |

Speedups over the serial NumPy direct sum at the same rows, order 2:
`30x / 19x / 10x / 5.8x / 1.6x`. `ncrit` is nearly inert on accuracy
(`8.1e-04` at 64, `1.3e-03` at 16, `4.8e-04` at 256) and non-monotone on cost in
NumPy, where `ncrit` also sets how much work is vectorized.

### The same at other sizes, order 2, `vertex` quadrature

| N sources | theta=0.5 | 0.3 | 0.2 |
| --- | --- | --- | --- |
| 42 | 0 (exact) | 0 (exact) | 0 (exact) |
| 162 | 6.5e-03 | 7.6e-16 | 7.6e-16 |
| 642 | 8.7e-03 | 1.6e-03 | 1.5e-04 |
| 2562 | 4.7e-03 | 4.8e-04 | 1.1e-04 |
| 10242 | 8.1e-03 | 8.1e-04 | 2.4e-04 |

`face` quadrature (2x the sources, at face centroids) behaves the same to within
a factor of ~1.3: at N_vert=10242 / 20480 sources, order 2 theta 0.3 gives
`9.4e-04`.

Four things to read off these numbers.

- **The default is a `1e-3`-accuracy method, deliberately.** `theta=0.3` at
  order 2 sits at `~1e-3` and buys ~10x over serial direct at 10k sources. The
  reference chose the knee, not the accurate end.
- **The error does not shrink with resolution.** 4.8e-04 at 2562 sources,
  8.1e-04 at 10242. A treecode's relative error is set by `theta` and `order`,
  not by `h`; refining the mesh does not refine the far field. This is the
  property that makes it incomparable with the direct sum at gate tolerances.
- **`order` is worth more than `theta` per unit cost.** Going 0->2 at fixed
  `theta=0.3` buys 30x accuracy for ~3x cost; going `theta` 0.3->0.1 at fixed
  order 2 buys 140x accuracy for 6x cost, but only because at `theta=0.1`
  almost nothing is accepted and the method degenerates toward direct.
- **At small N the treecode silently *is* the direct sum.** The exact zeros and
  the `1e-16`s above are not accuracy wins; they are runs where the acceptance
  test never fires (or `N <= ncrit`, line 110, which short-circuits to
  `_direct_batch` outright). **Every mesh in the current gate is in this
  regime** — the regression tier runs the 162-vertex icosphere, and M0's
  milestone tier runs 642 (subdivisions 3) and 2562 (subdivisions 4). At 162
  vertices, `theta=0.3` order 2 is bit-comparable to direct; at 642 it is
  already `1.6e-03` off.

### What this means for validation, concretely

M0-D1 measured Beatnik's divergence horizon against the Python gold sets
([milestone0-progress-log.md](milestone0-progress-log.md), Steps 2-4): starting
from a **1-ulp** difference in the initial condition, the trajectories stay
inside `1e-12/1e-14` for only 775-1350 of 2000 steps, and stay inside
`1e-10/1e-12` for all 2000. A treecode path injects a **`1e-3`** relative
perturbation into every velocity evaluation — nine orders of magnitude above the
perturbation that already exhausts the tightest rung. So:

- **A treecode run can never be trajectory-compared against a direct gold set**,
  at any tolerance the existing ladder uses. Not at `1e-10`, not at `1e-6`.
- The only meaningful validation of a treecode implementation is a
  **single-evaluation, same-input** comparison against `BRSolverDirect`: same
  mesh, same state, one call, assert `relmax < f(theta, order)` using the table
  above as the expected bound. That is a `unit`-tier test, not a `regression`- or
  `milestone`-tier one, and it does **not** change the gate (still five members,
  60 launches).
- The same argument already applies to `BRSolverFMM` and is why
  [framework.md](framework.md) points test 3 at the Python **direct** gold file.
  A ported treecode does not escape it; it inherits it.

---

## 2. How the treecode differs from a classical FMM

Both are tree methods for the same sum, and both are `O(N log N)`-or-better, so
the distinction is easy to blur. The differences that matter here:

| | Barnes-Hut treecode (`treecode.py`) | Classical FMM |
| --- | --- | --- |
| Expansions used | **Multipole only** (far-field expansion about a *source* cell centre) | **Multipole *and* local** (a Taylor expansion about a *target* cell centre) |
| Passes | Build + moments, then an independent per-target walk. **No downward pass.** | Upward pass (P2M, M2M), an interaction-list pass (M2L), a downward pass (L2L, L2P) |
| Operators | P2M and M2P only | P2M, M2M, M2L, L2L, L2P, plus P2P near field |
| Interaction unit | (target point, source node) | (target node, source node) |
| Complexity | `O(N log N)` — every target descends its own path from the root | `O(N)` — targets are handled in batches, one per leaf, via the local expansion |
| Accuracy control | opening angle `theta` + fixed order `p` | expansion order `p` + a fixed, geometry-based interaction list |
| Error behaviour | Uncontrolled-but-empirical: no rigorous bound in general, and the observed error is a plateau in `N` (see §1) | Prescribed: `p` is chosen from a target accuracy and the error bound is `O(theta^{p+1})` with the interaction list fixed |
| Practical accuracy | ~`1e-3` at the reference default | routinely tuned to `1e-6`-`1e-9`, at higher `p` and higher cost |
| Code size | 138 lines of NumPy | thousands of lines; translation operators dominate |
| Parallel structure | trivially parallel over targets; the tree build is the only shared object | needs a locally-essential tree, a global partition (usually space-filling-curve), and communication in both passes |

Concretely, in the reference's terms: `_expansion_batch` is a **P2M/M2P pair
fused** — the moments `G, D, Q` are the multipole coefficients (P2M), and the
Taylor kernel derivatives `K, dK, ddK` applied to them are the direct evaluation
of that multipole at a single target (M2P). There is no object anywhere in
`treecode.py` corresponding to a local expansion, no `M2L`, and consequently no
downward traversal — `treecode_velocity_unsigned`'s loop (lines 120-137) is a
single explicit stack that only ever pushes children.

Two further departures from the *classical* treecode that a port must reproduce
or consciously reject:

- **`_build` recomputes moments from scratch at every level** (lines 30-35 run on
  the node's full `idx` before subdivision). A classical implementation does one
  upward M2M pass. Same answer up to floating-point associativity, `O(N log N)`
  vs `O(N)` build. Cheap in NumPy where each level is one vectorized `einsum`;
  the wrong shape for a Kokkos port, which should do the upward pass.
- **The traversal is target-batched, not per-target.** Each stack entry is
  `(node, array of still-active target ids)` (line 119), so the expansion is
  evaluated as a vector op over all targets that accepted that node. This exists
  purely to avoid a Python per-target loop; it is a NumPy artifact, and it is the
  *opposite* of the right GPU shape (see §3).

**Bottom line for the recorded deviation:** the framework's claim that the
treecode and an FMM are "a different algorithm with a different error structure"
([Beatnik_BRSolverFMM.hpp:24-40](../src/Beatnik_BRSolverFMM.hpp#L24-L40)) is
correct, and the two `1e-3`-vs-`1e-6+` accuracy regimes make it stronger than the
header states: a Beatnik FMM will not merely fail to match a Python treecode
tightly, it will be *more accurate than it*, in a way no tolerance choice
reconciles.

---

## 3. What porting the treecode into Beatnik would cost

### The good news first

**The CLI needs no change, and the port would *remove* a deviation.**
`--br-approximation treecode`, `--br-treecode-theta`, `--br-treecode-order` and
`--br-treecode-ncrit` are all already in the reference's `parse_args` (lines
232-244), already parsed by
[InputFile.hpp:444-470](../examples/02_adaptive_mesh_bubble/InputFile.hpp#L444-L470),
and already carried in `FmmParams` (`mac_theta`, `order`, `ncrit` —
[Beatnik_Params.hpp:138-152](../src/Beatnik_Params.hpp#L138-L152)) where the
comment concedes the mapping "is nominal; the numbers do not mean the same thing
to the two algorithms." Porting the treecode makes those three knobs mean exactly
what the reference means by them. Under the project's CLI convention
([framework.md](framework.md) line 75) this is the rare change that adds a
capability with **zero** new options: `treecode` stops being warned-and-mapped
and becomes its own `BRApproximation` enumerator.

The seams are also already cut correctly. `BRSolverBase` is a two-method
interface, `createBRSolver` is a `switch`, and the quadrature already hands over
exactly what a tree wants — `(Ns,3)` points and `(Ns,3)` area-weighted strengths
from `SourceQuadratureBase::generate`. A `BRSolverTreecode` is a sibling of
`BRSolverDirect`, not a modification of anything.

### The actual work

**Estimate: a moderate task, comparable to T2c (the direct BR solver) plus a
tree. One focused task, 700-1000 lines of new header with the project's doc
density, plus ~250 lines of unit test.** Broken down:

| Piece | Lift | Notes |
| --- | --- | --- |
| Kernel expansion (`_expansion_batch` -> device function) | **Small.** ~80 lines. | Pure arithmetic, no data structure. Three closed-form tensors (`K`, `dK`, `ddK`) contracted against `G`, `D`, `Q`. Direct transcription of lines 56-81; the `einsum`s expand into explicit loops over 3 indices. The one trap is keeping `b` inside `w` everywhere. |
| Near field (`_direct_batch`) | **None.** | `BRSolverDirect`'s inner loop already is this, restricted to a source range. Factor it into a `KOKKOS_INLINE_FUNCTION` both solvers call. |
| Tree build | **Medium.** The real work. | The Python recursion (lines 27-53) must become a flat, device-resident structure: a node array (centre 3, radius 1, `G` 3, `D` 9, `Q` 27 = **40 doubles/node**), child indices, and leaf source ranges over a **tree-order permutation** of the source arrays. Standard approach: Morton-key sort of the sources, build levels bottom-up, one upward pass for the moments (i.e. do the M2M the Python skips). Kokkos-parallel per level. |
| Traversal | **Medium.** Restructure, not translate. | The Python's target-batched stack is the wrong shape for a GPU. The right shape is one thread (or team) per target with a small fixed-depth explicit stack in registers/scratch, exactly the classical BH kernel. This is a rewrite of lines 118-137, not a port of them — but it is a well-known ~120-line kernel. Depth bound must be asserted, not assumed. |
| **Distributed-memory** | **Medium, and the design decision.** | The Python is serial with all sources in one array. Two options: **(a)** reuse `BRSolverDirect::ringAccumulate` — circulate each rank's source block, build a small tree per arriving block, traverse it for local targets. `O(Ns/P)` memory, `P` tree builds per call, and it reproduces the existing collective structure exactly, including "every rank calls it the same number of times." Accuracy is *slightly* worse than the serial treecode because each per-rank tree is shallower and its root cells are geometrically worse. **(b)** `MPI_Allgatherv` all sources, build one identical global tree per rank, traverse for local targets. Matches the Python's accuracy exactly, `O(Ns)` memory and a redundant `O(Ns log Ns)` build on every rank. At `1e5`-`1e6` sources (b) is only 3-40 MB and is the honest first implementation; (a) is what scales. **This choice should be made explicitly and recorded, not defaulted into.** |
| Riesz scalar | **Trivial, as a decision.** | The reference raises for `treecode` + `surface-riesz` (`mesh_solver.py:605`). The faithful port is for `BRSolverTreecode::computeSurfaceRieszScalar` to reject that combination by name and task ID, exactly as the port's other unimplemented configurations do. Do *not* silently fall back to direct. |
| Wiring | **Small.** ~40 lines. | `BRApproximation::Treecode` in [Beatnik_Types.hpp:163](../src/Beatnik_Types.hpp#L163) + its `toString`; a `case` in [Beatnik_CreateBRSolver.hpp:58](../src/Beatnik_CreateBRSolver.hpp#L58); drop `treecode` from the warned-and-mapped branch in `InputFile.hpp`; a `TreecodeParams` (or reuse `FmmParams`, renamed intent) . README + `framework.md` deviation edits. |
| Test | **Small-medium.** ~250 lines, `unit` tier. | Per §1: one-shot velocity comparison against `BRSolverDirect` on the same state, asserting `relmax` under a `theta`/`order`-dependent bound, at a mesh **large enough that acceptance actually fires** — subdivisions 4+ (2562 vertices), since 162 is bit-identical to direct and would pass vacuously. Also at 2+ rank counts, to pin the distributed choice above. **Adds no gate member.** |

### What makes it *not* harder than it looks

- The blob-aware expansion is the subtle part of the physics, and it is 25 lines
  of the reference that transcribe literally.
- No `MPI_Allreduce`-shaped novelty: option (a) reuses a collective loop that is
  already written, tested and documented for deadlock-freedom.
- No new CLI, no new gold data, no `regression`/`milestone` tier change.

### What makes it harder than it looks

- **The device tree build is the whole job.** Everything else is arithmetic.
  Morton sort + level-wise build + upward moment pass on GPU is a known but
  non-trivial ~300 lines, and it is rebuilt every RK stage (the surface moves),
  so its cost is not amortizable the way `FarFieldSolver::setSources`' doc
  speculates for the FMM ([Beatnik_FarFieldInterface.hpp:106-113](../src/Beatnik_FarFieldInterface.hpp#L106-L113)).
- **The 10x speedup measured in §1 is against serial NumPy, and does not
  transfer.** `BRSolverDirect` is already a fully parallel `RangePolicy` over
  targets with no temporaries; on a GPU it is arithmetic-dense, coalesced, and
  extremely fast per FLOP. A treecode replaces that with pointer chasing,
  divergent per-target stacks and a serial-ish build. **The crossover N where a
  Kokkos treecode beats the Kokkos direct sum is unknown and must be measured
  before any of this is justified on performance grounds** — it is plausibly well
  above `1e5` vertices per GPU.
- **`1e-3` accuracy is a physics decision, not just a speed one.** Any Beatnik
  treecode run is a different problem from a direct run past a few hundred steps.

### The honest recommendation

**Port it only if the goal is reference fidelity, not performance.** Two distinct
reasons one might want it, and they lead to different answers:

1. *"I want to reproduce a reference `--br-approximation treecode` run."* Then
   port it, accept option (b) for the distribution so the accuracy matches, and
   expect the comparison to be per-evaluation rather than per-trajectory. Worth
   the moderate lift; it closes a real deviation with no CLI cost.
2. *"I want the `O(N^2)` sum to stop being the runtime."* Then the treecode is
   the wrong tool and the FMM (T3a) is the right one — a `1e-3` method whose
   error plateaus in `N` is not what a production code should be integrating
   with, and the framework's original judgement stands.

A cheap intermediate exists and is worth naming: implement the **expansion and
near-field device functions only**, plus a host-side serial tree, as a
`unit`-tier oracle that quantifies "how much does a treecode-grade far field
perturb this problem" on real Beatnik state — without committing to a
production-quality distributed device tree. That is perhaps a fifth of the full
lift and it answers the physics question that decides whether the rest is worth
doing.

---

## Open questions this document does not answer

1. **Where is the Kokkos crossover?** Needs a measured `BRSolverDirect` cost
   curve on MI300A at 1e4-1e6 vertices. Until that exists, no performance claim
   about a treecode port is defensible.
2. **Ring (a) or allgather (b) for the distribution?** §3 lays out the trade;
   the choice belongs to whoever takes the task.
3. **Does Canopy's FMM already subsume this?** Deliberately unexamined —
   `../canopy` was not opened, per the question as asked. If Canopy exposes an
   opening-angle/monopole mode, a "treecode" could conceivably be a
   configuration of T3a rather than a new solver, which would change the lift
   estimate substantially.
