# zmodel3d-amr → Beatnik C++ port

**Status:** IN PROGRESS — last updated 2026-08-06

## Problem

Port the Python library at `~/research-bridges/zmodel-steve/zmodel3d-amr` to C++
with **MPI** for distributed parallelism and **Kokkos** for on-node/GPU
parallelism, inside Beatnik.

The Python is a working, validated serial reference: an adaptive-triangle
higher-order 3D z-model that evolves a closed vortex sheet (a rising
Rayleigh-Taylor bubble) with metric-based dynamic remeshing. It does not scale —
the Birkhoff-Rott evaluation is `O(N²)` in NumPy, the remesher is a Python loop
over edges, and nothing is distributed. Beatnik is where it becomes a
production-scale code.

This file records what the framework commits built, and lays out the remaining
work in order.

### Read this first: the port source is NOT `solver.py`

The task brief named `zmodel3d/solver.py` as the port source. **That file is the
structured-grid z-model and is not what the adaptive-mesh bubble driver calls.**
`run_adaptive_mesh_bubble.py` imports from `zmodel3d/__init__.py`
(re-exporting `mesh_solver.py` and `mesh.py`) and from
`zmodel3d/dynamic_remesh.py`; it never imports `solver.py`. `solver.py` is 346
lines of `(ny, nx, 3)` structured-grid arrays; `mesh_solver.py` is 1832 lines of
unstructured triangle-surface code.

Tracing everything to `solver.py` would have produced a framework whose
provenance comments pointed at unrelated code. The port therefore traces to the
**real** origin, and cites `solver.py` additionally where a routine has a
recognizable structured-grid ancestor there (the Bernoulli potential, the RK3
tableau, the BR kernel). The five real source files:

| File | Lines | What it holds |
| --- | --- | --- |
| `examples/run_adaptive_mesh_bubble.py` | 1887 | CLI, control flow, checkpoints, diagnostics, plotting |
| `zmodel3d/mesh_solver.py` | 1832 | states, BR velocity, the RHS, RK3, red-green AMR, quality repair |
| `zmodel3d/mesh.py` | ~760 | surface primitives, AMR indicators, sphere generators, refinement |
| `zmodel3d/dynamic_remesh.py` | ~1170 | metric remeshing, nonlocal proximity, exact triangle distances |
| `zmodel3d/mesh_quality.py` | 169 | valence-equalizing flips, tangential relaxation, isotropic cleanup |

`~/research-bridges/zmodel-steve/zmodel3d-amr/` is **strictly read-only**.

## Approach

### What the framework commits built

Four commits on `rising-bubble-redesign`:

1. `763537e` — the header framework, `src/Beatnik_*.hpp`
2. `d91be1e` — the example driver, `examples/02_adaptive_mesh_bubble/`
3. `ef8059c` — the regression harness, `tests/regression_tests/` + ctest + CI
4. *(this commit)* — `README.md` and this file

Every library body throws
`std::logic_error("<Class>::<method> not implemented")`. **The documentation is
the deliverable**: each mathematical stub carries the equation it discretizes,
the meaning and units of every argument, the sign and normalization conventions,
and the assumptions inherited from the Python — so a later session can implement
it from the header alone, without reading the Python.

#### Conventions established

| Convention | Choice |
| --- | --- |
| Namespace | `Beatnik` |
| Library style | header-only, all headers in `src/` |
| Header naming | `Beatnik_` prefix |
| C++ standard | C++17 |
| Parallelism | Kokkos + MPI from the start; no serial-only signatures |
| Stub bodies | `BEATNIK_NOT_IMPLEMENTED("Class", "method")` |
| Traceability | `// Port of <file>::<fn> (lines N-M)` on every ported routine |
| Container types | templated on `<ExecutionSpace, MemorySpace>`, each view alias carrying `// TODO(types): templated pending Tessera/Canopy interface; collapse to a concrete type once known.` |

#### The headers

*Foundations* — `Beatnik_Types.hpp` (scalars, the stub macro, the CLI enums),
`Beatnik_Params.hpp` (one struct per parameter group, defaults transcribed from
`parse_args`).

*Adapters, one per external dependency* — `Beatnik_MeshInterface.hpp` (Tessera:
surface, adjacency, topological edits), `Beatnik_IOInterface.hpp` (Tessera HDF5:
checkpoints), `Beatnik_FarFieldInterface.hpp` (Canopy: FMM kernel sums).
**Neither `../tessera` nor `../canopy` was opened**; the interfaces are shaped by
what the Python algorithms need. No other header names a type from either
library.

*Communication* — `Beatnik_Communication.hpp` names every distributed-memory
point as a documented stub: halo exchange (vertices, fields), the ghost
scatter-add, the four reductions (sum, min, max, all-finite), refinement-mark
reconciliation, redistribution, and the checkpoint gather/broadcast. Each records
what moves, which MPI operation is expected, and **what invariant breaks without
it** — the last being the part that cannot be reconstructed from code later.

*Mathematics* — `Beatnik_MeshGeometry.hpp` (areas, normals, gradients, cotangent
Laplacian, mean-curvature normal, enclosed volume and its gradient),
`Beatnik_SurfaceState.hpp` (the two state models and the carried material
position), `Beatnik_VolumeProjection.hpp`, `Beatnik_ZModelSolver.hpp` (the RHS),
`Beatnik_TimeIntegrator.hpp` (TVD-RK3 + adaptive dt).

*Abstractions the task asked for* — `Beatnik_SourceQuadrature.hpp`
(`SourceQuadratureBase` + `createSourceQuadrature`, with `Vertex`, `Face`,
`Triangle3`) and `Beatnik_BRSolverBase.hpp` + `Beatnik_BRSolverDirect.hpp` +
`Beatnik_BRSolverFMM.hpp` + `Beatnik_CreateBRSolver.hpp`.

*Adaptivity* — `Beatnik_AdaptiveMesh.hpp` (indicator-driven red-green AMR),
`Beatnik_DynamicRemesh.hpp` (the default path), `Beatnik_MeshQuality.hpp`
(valence cleanup).

*Setup and orchestration* — `Beatnik_InitialCondition.hpp`,
`Beatnik_Restart.hpp` (restart isolated so it cannot block the non-restart
tests), `Beatnik_Diagnostics.hpp`, `Beatnik_Solver.hpp` (the control flow,
transcribed line-for-line against `main`).

#### Deliberate deviations from the Python

Each of these is a decision, not an omission:

- **`local` and `clustered` BR approximations are not ported.** They are stepping
  stones between the direct sum and the treecode; Beatnik has a real FMM. Their
  CLI names map to `fmm` with a warning.
- **The treecode is replaced, not ported.** Canopy's FMM is a different algorithm
  with a different error structure, so a Beatnik `fmm` run is not expected to
  match a Python `treecode` run tightly. Test 3 therefore compares against the
  Python **direct** gold file.
- **Only the `vertex` source quadrature will be implemented.** All three remain
  selectable. Recorded in the README.
- **Plotting, video, and the plane-section diagnostic are not ported.** Their
  options are accepted and ignored with a warning.
- **The state models are one class with a runtime tag**, not two duck-typed
  dataclasses, so the RHS and integrator are not templated on the model.

### What is NOT yet true

**The C++ has never been compiled or run.** No `spack install`, no `--help`
invocation, no `ctest`. Verification was deliberately deferred to a following
session at the user's direction. Treat every C++ file as *designed* but not
*validated*.

The Python side **was** run and does pass: `compare_output.py` matches the
positive fixture, fails the negative one, and the fixtures regenerate
reproducibly.

## Progress log

- 2026-08-06 — Read the five Python sources in full. Established the conventions
  above and landed the four framework commits. Discovered that the brief's named
  port source (`solver.py`) is the wrong file and traced to the real sources
  instead; recorded above. Wrote and **ran** `compare_output.py` and its
  fixtures; the C++ was not built (see task V0). Next: V0.

---

# Task sequence

Tasks are ordered. Each names the headers and functions it fills in, its Python
counterpart, and its **exit criterion**. Coarse-grained tasks carry an
"Additional information needed" section, as required.

**Dependency-opening is deferred to specific tasks.** `../tessera` is opened
first in **M1**; `../canopy` first in **F1**. No earlier task should open either
— that is the whole point of the three adapter headers.

---

## V0 — Make it build and run to a stub *(do this first)*

**Why first:** everything below assumes a compiling baseline. Roughly two dozen
headers were written without a compiler; expect ordinary errors — missing
includes, `typename` on dependent types, `auto`-returning stubs whose deduced
type is `void`, unused-parameter warnings under `-Werror` if the build enables
it, and the `Kokkos::View<Real*[3], device_type>` aliases needing adjustment for
the Kokkos 5 API.

**Do:**
1. `spack env activate ~/spack_envs/tuolumne_beatnik && spack install`
   (spack mode — this checkout's `profile.local.sh`; **never** hand-run `cmake`).
   Note the header-only rebuild caveat in `systems/tuolumne/claude.md` §3.
2. Fix compile errors **without changing documented semantics**. If a fix
   requires a semantic decision, record it here rather than deciding silently.
3. `adaptive_mesh_bubble --help` must exit 0 and print the schema.
4. A real invocation must parse, echo its configuration, and exit with a
   `std::logic_error` from a stub — **not** an argument-parsing error and not a
   crash.
5. `ctest -L unit` in the spack build tree: `Beatnik_Test_PythonCompare` passes
   and `Beatnik_Test_PythonCompare_Negative` passes by failing (`WILL_FAIL`).

**Also check:** the live `~/spack_envs/tuolumne_beatnik/spack.yaml` has drifted
from the committed `systems/tuolumne/spack.yaml` snapshot (live has `kokkos@5`
and a `tessera` spec; the snapshot has `kokkos@4` and none). Pre-existing drift,
not from these commits, but CLAUDE.md requires the snapshot to track the live
env — resync it here. The working tree also carries uncommitted `CMakeLists.txt`
edits (`Kokkos 4`→`5`, `find_package(SILO)` commented out) that belong with that
resync.

**Exit criterion:** all five steps above succeed, and the README "Known Issues"
entry about the framework never having been compiled is deleted.

---

## Phase 1 — Regression test 1: initial conditions, 0 timesteps

Compare the Python driver's startup checkpoint against Beatnik with the same
defaults. Validates mesh generation and problem setup with no dynamics at all.

### T1a — Generate the gold file *(human step, no code)*

Run the Python driver with default arguments plus `--steps 0
--checkpoint-dir <dir> --source-quadrature vertex --br-approximation direct`,
and commit `<dir>/checkpoint_t*_step0000000.npz` under
`tests/regression_tests/gold/`.

**`--source-quadrature vertex` is not optional here.** The Python default is
`face`, and the C++ port only implements `vertex`; a `face` gold file is not
comparable. (At 0 timesteps the quadrature is never evaluated, so this gold file
would in fact be identical either way — but generating every gold file the same
way avoids the trap at T2a, where it matters.)

This is the exact python command run. The step 0 NPZ file is used as the gold file here:
`python examples/run_adaptive_mesh_bubble.py --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --viscosity-mode laplace-beltrami --br-approximation direct --isotropic-cleanup --checkpoint-every-steps 1 --no-video --steps 0 --source-quadrature vertex`

**Exit criterion:** a gold `.npz` is committed (DONE) TODO: Verify `compare_output.py` loads it
without a structural complaint when compared against itself. 

### T1b — Icosphere generation and mesh geometry

**Fill in:**
- `Beatnik_MeshInterface.hpp`: `SurfaceMesh::generateIcosphere`, `adopt`,
  `buildEdgeAdjacency`, `buildVertexAdjacency`, `buildFaceAdjacency`,
  the accessors. ← *Python:* `mesh.py::icosphere_mesh` (362-461),
  `::edges_from_faces` (227-237), `::vertex_adjacency` (141-147)
- `Beatnik_MeshGeometry.hpp`: `MeshGeometry::compute`,
  `SurfaceOperators::{triangleQuality, edgeLengths, faceEdgeExtents,
  enclosedVolume, volumeGradient, areaWeightedMean}`.
  ← *Python:* `mesh_solver.py::_mesh_geometry_arrays` (216-236),
  `mesh.py::triangle_quality` (101-124), `face_areas` (81-85),
  `face_normals` (127-138), `run_adaptive_mesh_bubble.py::mesh_enclosed_volume`
  (1036-1040), `::mesh_volume_gradient` (1043-1051)
- `Beatnik_Communication.hpp`: `haloExchangeVertices`, `haloScatterAdd`,
  `allReduceSum`, `allReduceMin`, `allReduceMax`, `allReduceAllFinite`.

**This task opens `../tessera`** — see M1, which it depends on. Run M1 first, or
merge the two; they are separated only so the Tessera-shaped decisions are
recorded distinctly.

**Exit criterion:** a 1-rank run reproduces the Python's vertex and face counts
(162 / 320 at the default subdivision 2) and its enclosed volume and minimum
edge length to `1e-14` relative.

### T1c — Initial condition and checkpoint write

**Fill in:**
- `Beatnik_InitialCondition.hpp`: `build`, and the fast path (default sphere, no
  vorticity, no polar mode) only. Leave `applyShapeDeformation`,
  `applyPolarMode` and `seedInitialVorticity` throwing.
  ← *Python:* `run_adaptive_mesh_bubble.py::main` (1215-1240),
  `::apply_initial_geometry` (714-717)
- `Beatnik_SurfaceState.hpp`: `resize`, `seedMaterialPosition`,
  `updateSheetVector`, `centerPotential`, `allFinite`.
  ← *Python:* `mesh_solver.py::potential_sheet_vector` (364-367),
  `::_area_weighted_scalar_mean` (239-244)
- `Beatnik_IOInterface.hpp`: `timeKey`, `write`.
  ← *Python:* `::checkpoint_time_key` (951-952), `::save_state_checkpoint`
  (955-990)
- `Beatnik_Restart.hpp`: `coldStart`.
- `Beatnik_Solver.hpp`: `setup`, and enough of `solve` to exit immediately at
  `steps == 0`; `finalize`.

**This task opens `../tessera`'s HDF5 I/O** — see M2.

**Exit criterion:** **regression test 1 passes at 0 timesteps.**
`compare_output.py beatnik.h5 gold.npz --rtol 1e-12 --atol 1e-14` exits 0, at
ranks 1, 2 and 4. Register it in the `regression` tier — **which changes the
ship gate, so confirm with the user first** (CLAUDE.md "Minimum test set").

---

## Phase 2 — Regression test 2: a few timesteps, `direct` BR

A direct BR solve is straightforward, so a failure here is unambiguously a bug in
the surrounding mathematics rather than in the far field.

### T2a — Generate the gold file *(human step)*

Rerun the Python with `--steps 5 --source-quadrature vertex
--br-approximation direct --no-dynamic-remesh --refine-every 0
--checkpoint-every-steps 1`, and commit the resulting `.npz` files.

**Adaptivity is off deliberately.** Test 2 isolates the *evolution*; refinement
and remeshing introduce their own ordering and tie-breaking differences (risks R4
and R7) which would confound the comparison. Test 2 must not be the first place
adaptivity is exercised.

**Exit criterion:** gold files for steps 1-5 committed.

### T2b — Surface differential operators

**Fill in:** `Beatnik_MeshGeometry.hpp`:
`SurfaceOperators::{faceScalarGradient, surfaceGradient,
cotangentLaplacianScalar, graphLaplacianScalar, graphLaplacianVector,
meanCurvatureNormal, projectTangent}`.
← *Python:* `mesh_solver.py::_face_scalar_gradient` (938-961),
`::surface_gradient` (964-986), `::cotangent_laplacian_scalars` (1020-1059),
`::graph_laplacian_scalars` (1004-1017), `::graph_laplacian_vectors` (989-1001),
`::mean_curvature_normal` (1068-1110), `::_project_tangent` (247-256)

**Exit criterion:** a unit test (tier `unit`) confirming, on the default
icosphere, that `meanCurvatureNormal` returns `≈ -2/R · n̂_out` (the
Meyer-Desbrun-Schroeder-Barr identity — the definitive sign check) and that
`surfaceGradient` of a linear function reproduces its tangential projection to
`1e-12`.

### T2c — Vertex quadrature and the direct BR solver

**Fill in:** `Beatnik_SourceQuadrature.hpp`:
`VertexQuadrature::{generate, generateGradient}`;
`Beatnik_BRSolverDirect.hpp`: `computeInterfaceVelocity`,
`computeSurfaceRieszScalar`.
← *Python:* `mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet` (768-792,
the `vertex` branch), `::_source_velocity_direct_unsigned` (437-454),
`::_source_riesz_scalar_direct` (457-489)

Do **not** port the NumPy target-chunking (line 445): it exists to bound a
temporary array that the Kokkos formulation never allocates.

**Exit criterion:** a `unit` test comparing the induced velocity on the default
icosphere against a hard-coded reference computed from the Python, to `1e-13`
relative, at ranks 1 and 4.

### T2d — The RHS, volume projection, and the integrator

**Fill in:** `Beatnik_ZModelSolver.hpp`:
`computeRightHandSidePotential`, `computeBernoulliPotential`,
`applyVelocityMode`, `computeScalarViscosity`, `computeSurfaceTension`;
`Beatnik_VolumeProjection.hpp`: `removeVolumeFlux`, `projectToVolume`;
`Beatnik_TimeIntegrator.hpp`: `step`, `chooseStepSize`;
`Beatnik_Solver.hpp`: `solve`, `advanceOneStep`, `checkpointDue`;
`Beatnik_Diagnostics.hpp`: `compute`, `writeProgressLine` (minus the two
nonlocal-gap fields, which need T3-era machinery — report `+inf`/NaN as the
Python does when they are unavailable).
← *Python:* `mesh_solver.py::potential_mesh_rhs` (1236-1269),
`::_bernoulli_scalar_from_velocity` (912-935), `::_interface_velocity` (259-269),
`::_scalar_viscosity` (1062-1065), `::_surface_tension_velocity` (1113-1139),
`::_remove_discrete_volume_flux` (285-295),
`::potential_mesh_rk3_step` (1291-1311),
`run_adaptive_mesh_bubble.py::project_state_to_volume` (1054-1077),
`::choose_step_dt` (889-901), `::main` (1398-1636)

**Watch:** the surface-tension term is added **before** the volume projection
(`mesh_solver.py:1134-1138`), and `potential_dot` is re-centred **after** the
viscous term (lines 1264-1268). Both are documented in the headers; both change
the answer if reordered.

**Exit criterion:** **regression test 2 passes at 5 timesteps**, `--rtol 1e-10`,
at ranks 1, 2 and 4. Volume drift stays below `1e-12` relative.

---

## Phase 3 — Regression test 3: `fmm` BR via Canopy

### T3a — Canopy far-field adapter

**Fill in:** `Beatnik_FarFieldInterface.hpp`:
`FarFieldSolver::{setSources, evaluateCurl, evaluateDot}`;
`Beatnik_BRSolverFMM.hpp`: both methods.

**This task opens `../canopy`.** First task permitted to.

**Additional information needed before a fine-grained design can be tasked:**
- What Canopy exposes: a generic three-component kernel evaluation, or fixed
  Laplace/Coulomb kernels only? The BR kernel is a **softened** `1/r²` field
  (`δ/(b+r²)^{3/2}`), not the bare one — Canopy's existing
  `near_softening_factor` (see README, FMM tuning) suggests softening is already
  a first-class concept there, but its exact form must be checked against
  `ZModelParams::blob()`.
- Whether the cross-product contraction can be folded into the kernel or must be
  applied to three separate scalar solves.
- Whether the tree can be reused across RK3 stages, or must be rebuilt when
  sources move.
- How Canopy's `ncrit`/`mac_theta`/`max_depth` map onto `FmmParams`. The README
  already records validated values for the *structured* solver
  (`mac_theta 0.4`, `max_depth 19` hard max, `ncrit 64`); whether they carry over
  to an unstructured surface is unknown.
- Whether the existing `Rebalance` dreg-cache issue (README "Known Issues",
  Canopy #22) affects this path.

**Exit criterion:** a `unit` test comparing `BRSolverFMM` against
`BRSolverDirect` on the same surface, agreeing to the FMM's advertised
tolerance; and the `--br-approximation fmm` path runs 5 steps without aborting.

### T3b — Regression test 3

**Exit criterion:** **regression test 3 passes**: Beatnik `fmm` versus the
**Python `direct`** gold file from T2a, at a tolerance loosened to the FMM's
accuracy (expect `--rtol 1e-6`, to be pinned by T3a's measurement — record the
number here once known). Not against a Python `treecode` run; see R6.

---

## Phase 4 — Adaptivity

### T4a — Indicator-driven red-green AMR

**Fill in:** all of `Beatnik_AdaptiveMesh.hpp`;
`Beatnik_MeshInterface.hpp::refine`;
`Beatnik_MeshQuality.hpp::{improveConnectivityByFlips,
improveQualityTangential}`; `Beatnik_Communication.hpp::reconcileRefinementMarks`.
← *Python:* `mesh.py::{area_change_indicator, curvature_change_indicator,
curvature_resolution_indicator, refine_marked_faces,
projected_red_green_face_count}`, `mesh_solver.py::{refine_potential_mesh_state,
_quality_preserving_refinement_marks, _balance_red_green_refinement,
_expand_marked_face_rings, _limit_marked_fraction, _drop_faces_below_min_edge,
improve_mesh_connectivity_by_edge_flips, improve_mesh_quality_tangential}`

**Additional information needed before a fine-grained design can be tasked:**
- The `max_faces` greedy accept loop (`mesh_solver.py:1501-1512`) is inherently
  sequential — each trial depends on the previous acceptance — and quadratic. A
  distributed replacement must preserve the *intent* (respect the cap, prefer
  high scores, keep the closure valid) but will not reproduce the serial mark
  set. Deciding the replacement needs Tessera's partitioning model, which is only
  known after M1. See R4.
- Whether Tessera's refinement is conforming red-green natively or whether
  Beatnik must drive it edge by edge.

**Exit criterion:** a run with `--no-dynamic-remesh --refine-every 5` completes
20 steps; face counts match a Python run of the same configuration at ranks 1
and 4 **when `--max-faces` is not binding**. Where it binds, only the
non-refinement fields are compared, and the divergence is recorded here.

### T4b — Dynamic remeshing

**Fill in:** all of `Beatnik_DynamicRemesh.hpp`;
`Beatnik_MeshInterface.hpp::{splitEdges, collapseEdges, flipEdges, compact}`.
← *Python:* all of `dynamic_remesh.py`

**Additional information needed before a fine-grained design can be tasked:**
- **The nonlocal proximity query is the hardest single item in the port.** It is
  a genuinely global spatial search over face centroids with two exclusion
  criteria (topological rings and material-coordinate distance), and no ghost
  depth makes it local. Candidate approaches — a distributed ArborX tree, reusing
  Canopy's tree, or a two-level scheme — cannot be chosen without knowing what
  Tessera and Canopy already build. Requires M1 and T3a.
- The exclusion sets are *per-face variable-size index sets*, which suits neither
  a `Kokkos::View` nor a distributed query cleanly. A ring-depth-bounded CSR
  representation is the obvious first attempt but needs sizing against real
  meshes.
- Collapse safety (link condition + geometric test) across a rank boundary needs
  a two-phase owner-decides protocol; whether Tessera supplies one is unknown.

**Exit criterion:** a default-configuration run completes 50 steps without
aborting, at ranks 1 and 4; volume drift below `1e-10`; minimum triangle quality
stays above `--remesh-min-quality`.

### T4c — Isotropic cleanup

**Fill in:** `Beatnik_MeshQuality.hpp::{valenceEqualizingFlips,
tangentialRelaxation, isotropicCleanup}`.
← *Python:* `mesh_quality.py` (44-167)

**Exit criterion:** with cleanup on, a run that reaches a tightening roll-up does
not die on the "curvature sliver"; the valence histogram stays concentrated at 6.
Compare the *statistics* against a Python run, not the flip set — see R7.

---

## Phase 5 — Remaining coverage

### T5a — Shape deformations and initial vorticity

**Fill in:** `Beatnik_InitialCondition.hpp::{applyShapeDeformation,
applyPolarMode, seedInitialVorticity}`.
← *Python:* `run_adaptive_mesh_bubble.py::apply_initial_geometry` (719-886),
`::_apply_polar_mode` (698-710)

**Watch:** the vorticity profile tables differ between the two state models —
the sheet-vector profiles are the *derivatives* of the potential ones. The header
carries both columns. Also: `applyPolarMode` needs a Legendre \(P_\ell\); use the
three-term recurrence, not the explicit polynomial.

**Exit criterion:** gold-file comparisons at 0 timesteps for
`--initial-shape mushroom-seed`, `--initial-shape skirt-seed`, and
`--polar-mode 2 --polar-amp 0.05`.

### T5b — Restart

**Fill in:** `Beatnik_IOInterface.hpp::read`; `Beatnik_Restart.hpp::load`;
`Beatnik_Communication.hpp::broadcastFromRoot`.
← *Python:* `::load_state_checkpoint` (993-1033), `::main` (1199-1214)

**Exit criterion:** a run checkpointed at step 5 and restarted reaches step 10
and matches a Python restart of the same checkpoint. **It will not match an
uninterrupted 10-step run** — see R3, and do not write a test that expects it to.

### T5c — Field filtering, redistribution, and the sheet-vector model

**Fill in:** `Beatnik_Solver.hpp::filterCirculationField`;
`Beatnik_ZModelSolver.hpp::computeRightHandSideSheet`;
`Beatnik_SurfaceState.hpp::projectSheetTangent`.
← *Python:* `::filter_circulation_field` (923-948), `mesh_solver.py::mesh_rhs`
(1207-1233), `::mesh_rk3_step` (1272-1288)

**Exit criterion:** a 5-step gold comparison with `--state-model sheet-vector`.

### T5d — Load balancing

**Fill in:** `Beatnik_Communication.hpp::redistribute`.

**Additional information needed:** Tessera's partitioning and migration API,
which T4b will have exposed. Correctness does not require this task; throughput
does — after a few hundred steps the refined spiral concentrates on one rank, and
the BR evaluation cost is linear in local target count.

**Exit criterion:** a 200-step run at 16 ranks shows a vertex-count imbalance
below 1.2×, and the per-step time scales.

---

## Dependency-opening tasks

### M1 — Open `../tessera`: mesh model

**First task permitted to read `../tessera`.** Reconcile
`Beatnik_MeshInterface.hpp` against what Tessera actually provides: the storage
model, the owned/ghost partition, adjacency, the topological edit operations, and
whether the `MeshEditResult` parent/weight scheme matches how Tessera reports
field transfer. **Rewrite the adapter; do not spread Tessera types outward.**

Record here: what Tessera provides, what it does not, and what Beatnik must
implement itself.

### M2 — Open `../tessera`: HDF5 I/O

Reconcile `Beatnik_IOInterface.hpp`. The checkpoint **schema** is fixed by the
gold files (see the table in that header) and is not negotiable; what is
negotiable is whether Tessera writes it directly, or Beatnik gathers and writes.
Also settle the dataset paths, and update `FIELD_MAP` at the top of
`compare_output.py` in the same change.

### F1 — Open `../canopy`

Folded into T3a above.

---

# Known risks

## R1 — Initial-mesh reproducibility in regression test 1 *(highest)*

Test 1 compares Beatnik's **generated** initial mesh against the Python gold
mesh. If Beatnik's mesh generation is not reproducible against the Python
algorithm — different vertex counts, different refinement decisions, or positions
differing by more than the comparison tolerance — **the test fails for reasons
that have nothing to do with the correctness of the solver**, and there is no way
to tell the two apart from the failure output.

The icosphere is the least risky case: the base icosahedron is a fixed literal
table, subdivision midpoints are `(v_a+v_b)/‖v_a+v_b‖`, and both codes use IEEE
double. Vertex *ordering* will differ, but that is exactly what
`compare_output.py`'s quantized sort exists to absorb. The `latlon` generator
involves `sin`/`cos` at computed angles, where the two languages' libm may differ
in the last bit — usually within tolerance, but not guaranteed.

**Mitigation, if it bites:** have Beatnik **read the initial mesh from the gold
file** rather than generating it. `SurfaceMesh::adopt` and
`Comm::broadcastFromRoot` exist for exactly this; the driver would take a
`--restart-from <gold>.npz`-shaped path. Mesh generation then becomes a
*separate* validated concern.

**Sequencing decision — state it plainly:** **test 1 as sequenced above assumes
the generated-mesh path** (T1b generates, T1c compares). If T1b's exit criterion
fails on counts or positions, switch test 1 to the read-from-gold path
immediately rather than tuning tolerances, and add a distinct task:

> **T1b′ — Validate initial mesh generation.** Compare Beatnik's generated
> icosphere against the gold mesh *in isolation*, with its own tolerance and its
> own failure report, decoupled from the solver comparison. Exit criterion: the
> vertex sets agree to `1e-14`, or the discrepancy is characterized and the
> tolerance for it justified here.

## R2 — Reductions are not reproducible across rank counts

`MPI_Allreduce` with `MPI_SUM` on floating point is not associative, and GPU
partial sums are not reproducible run to run. Two reduced quantities —
`initial_volume` and `initial_min_edge` — are carried for the whole run, and
*every* adaptive dt and *every* proximity radius scales off them. So a 4-rank run
and a 6-rank run take slightly different timesteps and diverge.

**Consequence for testing:** never compare across rank counts at tight tolerance.
Compare each rank count against the *same* gold file at a tolerance that
accommodates the trajectory difference, and expect that tolerance to grow with
step count. Regression tests 1 and 2 are short precisely to keep it small.

## R3 — A restart does not reproduce an uninterrupted run

Checkpoints do not carry `reference_face_area` or `reference_face_curvature`, so
loading one re-bases the area- and curvature-change AMR indicators to "no change
so far". The next refinement decision therefore differs from what an
uninterrupted run would have made, and the trajectories diverge. This is a
genuine behavioral difference, not numerical, and no tolerance hides it.

Faithful to the Python, which drops the fields deliberately. Isolated in
`Beatnik_Restart.hpp`. **Do not write a test that expects restart continuity.**
The third indicator (sagitta) is absolute and therefore restart-invariant, which
is the shape of a future mitigation if one is ever wanted.

## R4 — The `max_faces` greedy loop does not parallelize

`_quality_preserving_refinement_marks` (1501-1512) walks seeds in descending
score, tentatively adds each, re-closes the mark set, and keeps it only if the
projected count still fits. Each trial depends on the previous acceptance. Any
distributed replacement will produce a *different* mark set.

Acceptable — the cap is a resource limit, not physics — but it means **a run
where `--max-faces` binds will not match the Python face for face.** T4a's exit
criterion is written around that. When the cap binds, compare the fields, not the
mesh.

## R5 — `surface-riesz` + fast far field has no gold file

The Python raises for `bernoulli_scalar_mode=surface-riesz` under `treecode`
(`mesh_solver.py:605`), so no Python run can produce a gold file for it. Beatnik
supports the combination. It is therefore **unvalidatable against the reference**
and should be treated as unverified until someone constructs a gold file another
way (e.g. a Python `direct` + `surface-riesz` run, compared against Beatnik
`fmm` + `surface-riesz` at loosened tolerance).

## R6 — FMM ≠ treecode

Canopy's FMM and the Python's Barnes-Hut treecode are different algorithms with
different error structures. `--br-treecode-theta/-order/-ncrit` are mapped onto
`FmmParams` nominally; the numbers do not mean the same thing.

Handled by comparing test 3 against the Python **direct** gold file. Do not
compare against a Python `treecode` run and do not tune tolerances until it
passes — that would be fitting one approximation to another.

Second-order concern: the FMM expands a **softened** `1/r²` field, and near
self-contact the sheet separation approaches `√b`, where the softening dominates
the geometry. An acceptance criterion tuned on the bare kernel is optimistic
there. Measure the error against `BRSolverDirect` *at roll-up*, not only on the
initial sphere.

## R7 — Order-dependent mesh operations

Three passes are explicitly order-dependent in the reference and will not
reproduce their serial results in parallel:

- `_valence_equalizing_flips` uses a `touched` set to keep one pass's flips
  independent, making the result depend on edge iteration order.
- `flip_edges_for_quality` and `improve_mesh_connectivity_by_edge_flips` rebuild
  their edge map after each accepted flip.
- `collapse_short_edges` processes shortest-first, and each collapse changes
  which later collapses are safe.

Compare **statistics** (valence histogram, quality distribution, face count) not
the exact operation set. A parallel implementation that reaches a comparable
mesh quality is correct even though it made different edits.

## R8 — Ghost depth versus exchange count in the RHS

The RHS is a **two-ring** stencil on the potential: one surface gradient builds
the sheet vector, and a second is taken of the Bernoulli potential. With a
one-face-deep ghost layer the potential must be exchanged **twice** per RHS
evaluation. The easy bug is a single exchange of a single-deep halo, which is
wrong only near partition boundaries and only by a small amount — so it produces
a plausible-looking solution with a seam that moves when the rank count changes.

Watch for it explicitly at T2d: run the same configuration at 1 and 4 ranks and
plot the difference field. A seam localized on partition boundaries is this bug;
uniformly distributed noise is R2.

## R9 — Silent double-counting of ghosts

Two independent places where including ghost entities is silently wrong: the
quadrature must emit sources for **owned** entities only, and the enclosed volume
and edge-length reductions must sum over **owned** faces and edges only. Both
produce a result that is smoothly wrong and scales with the ghost fraction, i.e.
changes with the rank count and the partition shape.

`Beatnik_SourceQuadrature.hpp` and `Beatnik_MeshGeometry.hpp` both carry the
warning. The cheap detector: the enclosed volume of the initial sphere has a
closed form; check it against `4πR³/3` (minus the known polyhedral deficit) at
several rank counts.

## R10 — Ambiguity in `compare_output.py`'s quantized matching

The comparator pairs vertices by quantizing coordinates to a grid of
`--match-eps` (default `1e-9`) and sorting on the integer key. This needs
`match_eps` to sit comfortably between two scales: **much larger** than the
coordinate disagreement between the two codes, and **much smaller** than the
shortest edge.

Adaptive refinement squeezes that window from below — `--remesh-h-min` defaults
to `1.5e-3` and the tight set to `8e-4`, still six orders above the default eps,
so there is room today. But a run that refines far harder, or a comparison after
enough steps that positions disagree by more than `1e-9`, will start reporting
ambiguous pairings. That is reported as a failure, not hidden, which is the
correct behavior — but the fix is to think about which of the two bounds was
violated, not to reflexively widen `--max-ambiguous`.

## R11 — The gold files use the Python's default quadrature

`--source-quadrature` defaults to `face` in the Python and only `vertex` is
implemented in the port, and under the potential state model the two build the
sheet strength *differently* (exact per-face gradient versus area-averaged
per-vertex gradient), differing at `O(h)`.

**Every gold file must be generated with `--source-quadrature vertex`.** A gold
file made with the default will disagree with a correct Beatnik run by an amount
that looks exactly like a subtle discretization bug. Written into T1a and T2a;
repeated here because it is the single easiest way to lose a week.
