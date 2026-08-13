# zmodel3d-amr → Beatnik C++ port

**Status:** IN PROGRESS

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

**As of T1c there is a driver path and a checkpoint, and regression
test 1 passes.** A `--steps 0` run generates the icosphere, initializes and
re-centres the fields, seeds the material coordinate, computes the two carried
scalars, and writes an HDF5 checkpoint that matches the Python gold file at
`--rtol 1e-12 --atol 1e-14` at ranks 1-6 on SERIAL and HIP.

**There is still no timestep.** `Solver::solve` implements a `steps == 0` guard
and nothing else; at `steps > 0` it throws. **T2b has landed, so the surface
differential operators are no longer stubs** —
`SurfaceOperators::{faceScalarGradient, surfaceGradient,
cotangentLaplacianScalar, graphLaplacianScalar, graphLaplacianVector,
meanCurvatureNormal, projectTangent}` and `SurfaceState::updateSheetVector` are
implemented and validated against the Python reference (see T2b's completion
note). **T2c has landed too**, so the vertex source quadrature and the direct BR
evaluation are implemented and validated as well. What is still a stub:

- **No RHS, no integrator, no volume projection.** T2d. **The BR evaluation now
  exists and is validated on the vertex rule** (T2c): `VertexQuadrature` and
  `BRSolverDirect` are implemented and checked against the Python reference. What
  is still missing is the *calling* of them — the BR solver and the quadrature
  are constructed by `Solver::setup` and no production path invokes them yet,
  which is T2d's.
- **No adaptivity.** T4a/T4b/T4c, still blocked on the disjoint-editing-families
  design question and on Tessera's G5b/G5c/G5d.
- **`/vertices/u1` in a `--steps 0` checkpoint is still present-but-meaningless**,
  though no longer because of a stub: `SurfaceState::updateSheetVector` is
  implemented (T2b), but nothing on the 0-timestep path *calls* it, so
  `initializeFields` leaves the field zero — a *defined* value and not a correct
  one. `Tessera::writeMesh` writes the whole vertex pack unconditionally and
  `compare_output.py` skips the field `state_model` does not select, so nothing
  depends on it yet. T2d is where the RHS starts calling it every stage.
- **`CheckpointIO::read` and `RestartReader::load` still throw** — T5b. Writing is
  validated; reading is not, in either direction.
- **`InitialCondition` implements the fast path only.** `applyShapeDeformation`,
  `applyPolarMode` and `seedInitialVorticity` throw (T5a), so any non-default
  `--initial-shape`, `--polar-amp` or `--initial-potential-strength` aborts rather
  than silently producing a sphere. `--mesh-kind latlon` likewise.
- **`finalize()`'s "last finite state" is not yet distinct from "current".** The
  last-finite bookkeeping lives in the step loop, which is T2d's, so at 0
  timesteps the two coincide and a passing gate says nothing about that path.

Treat the C++ as *buildable, structurally sound, and validated exactly as far as
T1c's exit criterion reaches* — mesh generation, the initial condition, the two
carried scalars, and the checkpoint write. Nothing that evolves in time has been
checked against the Python.

The Python side **was** run and does pass: `compare_output.py` matches the
positive fixture, fails the negative one, and the fixtures regenerate
reproducibly.

## Progress log

The session-by-session record has moved to **[`framework-progress-log.md`](framework-progress-log.md)**.

**Read it as needed.** This document states what is true now; the log records how
each of those statements was arrived at — the semantic decisions and why they
were forced, the signature changes and what made them unavoidable, the bugs that
only running revealed, and the numbers measured on real hardware. Before
implementing a task, changing a signature, or reopening a question this document
treats as settled, read the log entries for the tasks yours depends on: the
reasoning behind a flat statement here is usually there and nowhere else.
References to the log below cite its section headings, which are task IDs.

**Append to it** at the end of any task that decides something, changes a
signature, measures something, or finds a bug — a new `## <task ID>` section at
the bottom of the log, named for the task.

**Keep this document dateless.** It states the current design, not its history.
When a task completes, mark its heading and its outcome paragraph `**DONE**` and
`**Met.**` with no date, fold anything that is now simply true into the relevant
section, and put the reasoning, the measurements and the things that only
running revealed in the log instead. Cite the log by task ID, never by date —
that is what keeps the two files from drifting back into parallel narratives.

---

# Task sequence

Tasks are ordered. Each names the headers and functions it fills in, its Python
counterpart, and its **exit criterion**. Coarse-grained tasks carry an
"Additional information needed" section, as required.

**Dependency-opening is deferred to specific tasks.** `../tessera` is opened
first in **M1**; `../canopy` first in **F1**. No earlier task should open either
— that is the whole point of the three adapter headers.

---

## V0 — Make it build and run to a stub *(do this first)* — **DONE**

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

**Met.** All five succeeded; step 2 was vacuous (zero compile
errors). The spack.yaml resync above was already done by the working tree's
uncommitted edits. Steps 3-4 run via the new
`scripts/tuolumne/run_v0_smoke.flux` — **not** interactively, see the login-node
rule in CLAUDE.md. Step 5 ran the two `compare_output.py` invocations directly,
since spack mode has no build tree for `ctest`. Details and the four latent
framework bugs fixed on the way are in the progress log, under *V0 and T1a*.

---

## Phase 1 — Regression test 1: initial conditions, 0 timesteps

Compare the Python driver's startup checkpoint against Beatnik with the same
defaults. Validates mesh generation and problem setup with no dynamics at all.

### T1a — Generate the gold file *(human step, no code)* — **DONE**

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

**Exit criterion:** a gold `.npz` is committed, and `compare_output.py` loads it
without a structural complaint when compared against itself. **Met** —
`tests/regression_tests/initial_conditions/gold.npz`, self-compare at
`--rtol 1e-12 --atol 1e-14` exits 0 (162 vertices, 320 faces, `potential`,
162/162 unambiguous). The five carried scalars, and the two T1b must reproduce,
are in the progress log, under *V0 and T1a*.

### T1b — Icosphere generation and mesh geometry — **DONE**

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

**Met**, by the new `unit`-tier test `Beatnik_Test_MeshGeometry`
(`tests/unit_tests/Beatnik_Test_MeshGeometry.cpp`), run through
`flux batch scripts/tuolumne/unit_tests.flux` at 1 rank on the **HIP** backend:
**15/15 checks**, and both reference scalars matched the T1a values to all 17
printed digits, not merely to `1e-14`:

```
enclosed volume 0.063235073124669514 vs T1a 0.063235073124669514
min edge        0.068976121063816842 vs T1a 0.068976121063816842
local V 162 (owned 162), owned E 480, local F 320 (owned 320)
```

The test checks more than the criterion: `V/E/F = 162/480/320`, Euler
characteristic 2, halo depth 2, every edge having exactly two incident faces,
vertex adjacency symmetric with `2E` entries and no self-loop, face adjacency of
degree exactly 3 and reciprocal, and two whole-surface identities that pin
`MeshGeometry::compute` and `volumeGradient`
(\f$\sum_v A_v = \sum_f A_f\f$ and \f$\sum_v p_v\cdot\partial V/\partial p_v =
3V\f$, the latter by Euler's homogeneous-function theorem). Details, the adapter
rework that preceded it, and the six signature changes it forced are in the
progress log, under *M1 adapter rework and T1b*.

### T1c — Initial condition and checkpoint write — **DONE**

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

  **As built:** `resize` became `initializeFields( mesh )` and
  `seedMaterialPosition` / `centerPotential` / `allFinite` now take the mesh,
  because T1c also discharged M1's deferred follow-up — the three evolved fields
  live in the Tessera vertex user pack, so `SurfaceState` holds no storage at
  all. `remap` was deleted. **`updateSheetVector` was the one method of the five
  T1c left throwing**, deferred to T2b on a stated dependency: its body *is*
  `SurfaceOperators::surfaceGradient`, which was T2b's, and at 0 timesteps under
  the `potential` model the sheet vector is never read. **T2b implemented it**;
  the fill-in list under T2b now names it. See the signature table in the
  progress log, under *T1c*.
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

**Met**, by the new `regression`-tier test
`Beatnik_Test_InitialConditions`
(`tests/regression_tests/Beatnik_Test_InitialConditions.cpp`), run through
`flux batch scripts/tuolumne/run_regression_minset.flux`. The gate was
pre-authorized for this change, so it now has **exactly one member** and is no
longer vacuous.

The criterion asked for ranks 1, 2 and 4; the **whole gate** was run — SERIAL and
HIP at ranks 1, 2, 3, 4, 5, 6, twelve launches — and every one passed:
`[gate] PASS (label=regression)`. The criterion's three rank counts are a
verified subset, not the extent of what was checked.

```
initial_volume    all 12 configurations within 1 ulp of the T1a value
initial_min_edge  0.068976121063816842 at every rank count, both backends,
                  rel 0 -- bit-identical to T1a everywhere
comparator        vertices max|e| 5.551115e-17 (max rel 2.92e-15), potential
                  max|e| 0, remesh_material_position max|e| 5.551115e-17,
                  faces 320/320 identical after remap, 162/162 unambiguous
```

The measured `initial_volume` takes exactly **three** distinct values across all
twelve configurations, spanning **2 ulp**:

| value | where |
| --- | --- |
| `6.32350731246694997e-02` | SERIAL np1, HIP np5 |
| `6.32350731246695136e-02` | SERIAL np2/3/4/6, HIP np1/2/3/4/6 — the T1a value, bitwise |
| `6.32350731246695275e-02` | SERIAL np5 |

i.e. a total spread of `2.78e-17` absolute, `4.4e-16` relative — four orders
inside the `1e-12` tolerance. **No tolerance was touched.** The discrimination
that establishes this is summation order and not R9 is recorded under R2 and R9
below and is built into the test, so it is re-measured on every gate run.

The test checks more than the criterion: the entity counts and Euler
characteristic, the halo depth, the **owned-set partition** (the per-rank
`ownedX` counts summed with a plain `MPI_Allreduce` must equal 162/480/320 —
R9's precondition, checked rather than assumed), both carried scalars against
the T1a literals at 17 digits, the volume-to-`4πR³/3` ratio, and a **negative
case** requiring the comparator to exit exactly `1` and not `2`.

---

## Phase 2 — Regression test 2: a few timesteps, `direct` BR

A direct BR solve is straightforward, so a failure here is unambiguously a bug in
the surrounding mathematics rather than in the far field.

### T2a — Generate the gold file *(human step)* — **DONE**

Rerun the Python with `python examples/run_adaptive_mesh_bubble.py --steps 10 --source-quadrature vertex --br-approximation direct --no-dynamic-remesh --refine-every 0 --checkpoint-every-steps 1 --no-video --checkpoint-dir results`, and commit the resulting `.npz` files.

**Adaptivity is off deliberately.** Test 2 isolates the *evolution*; refinement
and remeshing introduce their own ordering and tie-breaking differences (risks R4
and R7) which would confound the comparison. Test 2 must not be the first place
adaptivity is exercised.

**Exit criterion:** gold files for every step of the run committed.

**Met.** Eleven `.npz` files — **steps 0 through 10**, one per step — are committed
under `tests/regression_tests/direct-solve-10-steps/gold/`, with the generating
command recorded in that directory's `README.md`. The whole 10-step run is
covered, not the steps 1-5 the criterion originally asked for, which is why T2d's
exit criterion now reads "all 10 timesteps at ranks 1-5".

Four properties of the set were checked rather than assumed, because each is
something a later comparison would otherwise silently depend on:

- **Same schema as the T1a gold**, key for key: `vertices`, `faces`,
  `potential`, `remesh_material_position`, `state_model`, `step`, `time`,
  `initial_volume`, `initial_min_edge`. So `compare_output.py` needs no change;
  self-compares of steps 0, 5 and 10 at `--rtol 1e-12 --atol 1e-14` exit 0 with
  `162/162` unambiguous and `320/320` faces identical after remap.
- **Step 0 is bitwise identical to `initial_conditions/gold.npz`** in all four
  arrays. The two gold sets therefore describe the same problem despite T1a's
  command listing `--A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --viscosity-mode
  laplace-beltrami` explicitly and T2a's not: those are the Python's defaults.
  This also means test 2 at step 0 re-tests exactly what regression test 1 does.
- **`state_model` is `potential`** in every file, so the sheet-vector path (T5c)
  is not implicated and `/vertices/u1` stays the present-but-meaningless field
  M2 documented.
- **The mesh never changes: 162 vertices / 320 faces at every step**, confirming
  `--no-dynamic-remesh --refine-every 0` did what the "adaptivity is off
  deliberately" note above requires. The carried scalars are constant across all
  eleven files and bit-identical to T1a's, so nothing re-based them mid-run.

`time` is **not** a uniform `0.003` per step — it is `0.003` exactly at step 1
and then drifts (`0.0059999881751648708` at step 2, `0.029996631612342662` at
step 10). The adaptive dt is live in this configuration, so T2d must reproduce
`choose_step_dt`, not step at a fixed dt; comparing against these files with a
hardcoded dt will fail on `time` first and on the fields second.

### T2b — Surface differential operators — **DONE**

**Fill in:** `Beatnik_MeshGeometry.hpp`:
`SurfaceOperators::{faceScalarGradient, surfaceGradient,
cotangentLaplacianScalar, graphLaplacianScalar, graphLaplacianVector,
meanCurvatureNormal, projectTangent}`;
`Beatnik_SurfaceState.hpp::updateSheetVector`.
← *Python:* `mesh_solver.py::_face_scalar_gradient` (938-961),
`::surface_gradient` (964-986), `::cotangent_laplacian_scalars` (1020-1059),
`::graph_laplacian_scalars` (1004-1017), `::graph_laplacian_vectors` (989-1001),
`::mean_curvature_normal` (1068-1110), `::_project_tangent` (247-256),
`::potential_sheet_vector` (364-367)

`updateSheetVector` is T2b's, not T1c's: T1c deferred it here on a stated
dependency (its body *is* `surfaceGradient`), and both "What is NOT yet true"
above and `src/Beatnik_SurfaceState.hpp` said so while this list omitted it.

**Exit criterion:** a unit test (tier `unit`) confirming, on the default
icosphere, that `meanCurvatureNormal` returns `≈ -2/R · n̂_out` (the
Meyer-Desbrun-Schroeder-Barr identity — the definitive sign check) and that
`surfaceGradient` of a linear function reproduces its tangential projection to
`1e-12`.

**Met**, by the new `unit`-tier test `Beatnik_Test_T2bOperators`
(`tests/unit_tests/Beatnik_Test_T2bOperators.cpp`), run through
`flux batch scripts/tuolumne/unit_tests.flux` at 1 rank on the **HIP** backend
(the default execution space; the tier registers one suffix-less binary):
**31/31 checks**, tier 4/4. The gate is untouched — the new test is `unit` and
the `regression` tier still has exactly one member. Every reference number is a
hard-coded literal computed from the read-only Python on the default icosphere,
and none was adjusted; **no tolerance was changed anywhere.**

**The sign, first, because it is the check the criterion calls definitive.**
`meanCurvatureNormal` is strictly inward at **all 162 vertices**, tested as
\f$\Delta_{LB}x\cdot\hat n_{\text{out}} < 0\f$ against the *exact analytic*
outward normal \f$(p-c)/\|p-c\|\f$ — available because every icosphere vertex
lies exactly on the sphere, and therefore dependent on nothing under test. Zero
violations. Its magnitude averages `8.0177647933837246` against \f$2/R = 8\f$
exactly, **0.22% high**, inside the `1e-1` bound the test states a priori from
\f$(h/R)^2 = 0.076\f$; its direction is antiparallel to the radial to
`1.39e-04`. The per-vertex extremes (`7.9184808270587634` / `9.0760095262647997`,
the icosphere's valence-5 versus valence-6 vertices) are pinned against the
Python rather than against a tolerance fitted to that spread.

**Agreement with the Python is at `1e-15` or better on every compared quantity**
— three decades inside the criterion's `1e-12`, for T1b's reason: Tessera's
icosphere positions and the Python's differ only in their last bits. Thirteen
order-invariant summary scalars were compared (`max`, `min`, `sum` of
magnitudes, so no vertex-order matching is needed); the table is in the progress
log, under *T2b*. The exact identities:

```
faceScalarGradient   max|g_f - P_f a| = 1.7056134324626197e-15  (exact per face)
surfaceGradient      max|g . n_v|     = 1.5619656867989929e-16  (the projection)
projectTangent       max|v . n_v|     = 3.0631241924871614e-16
updateSheetVector    max|S . n_v|     = 9.8860206384425047e-17
updateSheetVector    max|(g x S).n + |g|^2| = 4.4408920985006262e-16
cotangentLaplacianScalar(const)  max|.| = 0   EXACTLY, not to a tolerance
graphLaplacianScalar(const)      max|.| = 0   EXACTLY
```

all bounded a priori at `1e-13` absolute. The test checks more than the
criterion: the two constant-field identities above (which catch a stencil that
forgot to difference, and which every non-constant test would pass), the
cotangent Laplacian's **dissipative sign** via the energy form
\f$\sum_i A_i\phi_i(\Delta_s\phi)_i = -0.91960120772791898 < 0\f$,
`projectTangent` reproducing \f$P_v a\f$ to `2.5e-16`, `updateSheetVector`'s
rotation sense via the signed identity
\f$(\nabla_s\phi\times S)\cdot\hat n = -\|\nabla_s\phi\|^2\f$ (which a magnitude
check cannot see — flipping the minus sign reverses it and leaves every
\f$|S|\f$ unchanged), and both graph Laplacians against the Python, which is
what validates their taking the **unique** one-ring rather than a per-face
scatter.

**The criterion's second half cannot hold as literally written, and that is a
property of the operator, not of the port.** `surfaceGradient` is an
area-weighted average of the per-face in-plane gradients, then projected. For a
linear \f$\phi = a\cdot p\f$ the *face* gradient is exactly \f$P_f a\f$ — checked
above at `1.7e-15` — but the average of \f$P_f a\f$ over mutually tilted faces is
not \f$P_v a\f$, and the projection does not repair the difference. The
discrepancy is \f$O((h/R)^2)\f$ and the **reference itself measures
`2.3416899365234726e-02`** on this mesh, 1.7% of \f$|a|\f$; no correct
implementation makes it `1e-12`. So `1e-12` is spent on the two statements that
are true and that a wrong implementation fails: Beatnik reproduces the
reference's `surface_gradient` of the same linear function on the same mesh —
that discrepancy scalar included — to `1.2e-15`, and the exact half of the claim,
no normal component, to `1.6e-16`. The untrue reading was replaced by a stronger
true one and the discrepancy is reported as a number, not absorbed into a
tolerance.

**Two signature changes and one widening**, plus a latent bug in *both* tier
wrappers that this task's second `unit` test exposed — `flux run` was eating the
manifest on stdin, so the tier reported a green `PASS (3/3 tests)` while silently
skipping the new test, and the **gate** wrapper had the same bug and would have
silently run only one member per backend once T2d registered regression test 2.
All in the progress log, under *T2b*.

### T2c — Vertex quadrature and the direct BR solver — **DONE**

**Fill in:** `Beatnik_SourceQuadrature.hpp`:
`VertexQuadrature::{generate, generateGradient}`;
`Beatnik_BRSolverDirect.hpp`: `computeInterfaceVelocity`,
`computeSurfaceRieszScalar`.
← *Python:* `mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sheet` (768-792,
the `vertex` branch), `::_source_velocity_direct_unsigned` (437-454),
`::_source_riesz_scalar_direct` (457-489)

Do **not** port the NumPy target-chunking (line 445): it exists to bound a
temporary array that the Kokkos formulation never allocates.

**Four `const mesh_type&` parameters must widen to `mesh_type&`**, and this task
is where it happens because it is the first to write a body that touches the
mesh. Every accessor a quadrature or a BR kernel needs is **non-const**:
`SurfaceMesh::positions()` (`src/Beatnik_MeshInterface.hpp:821`), `potential()`
(`:834`), `sheetVector()` (`:846`) and `faceVertices()` (`:877`). The first
three return Cabana slices of a non-const member; the fourth calls
`ensureGeometry()`, which builds and caches `Tessera::MeshGeometry` against
`generation()` on first call after a topology edit. So a `const mesh_type&`
parameter cannot read the positions, let alone the fields — the same constraint
that forced `RestartReader::coldStart( const mesh_type& )` →
`coldStart( mesh_type& )` at T1c, and the same reason
`MeshGeometry::compute` takes an explicit `vertex_count`.

| Declaration | File |
| --- | --- |
| `SourceQuadratureBase::generate` / `::generateGradient` | `Beatnik_SourceQuadrature.hpp:139`, `:158` |
| `VertexQuadrature`, `FaceQuadrature`, `Triangle3Quadrature` overrides of both | `Beatnik_SourceQuadrature.hpp:199`, `:211`, `:257`, `:269`, `:332`, `:344` |
| `BRSolverBase::computeInterfaceVelocity` / `::computeSurfaceRieszScalar` | `Beatnik_BRSolverBase.hpp:148`, `:169` |
| `BRSolverDirect` overrides of both | `Beatnik_BRSolverDirect.hpp:98`, `:130` |
| `BRSolverFMM` overrides of both | `Beatnik_BRSolverFMM.hpp:113`, `:140` |

**Callers to update: none.** `Solver::setup` constructs the quadrature and the
BR solver and nothing calls either yet (the only other mention in the tree is a
comment, `src/Profiling.hpp:17`), so the widening is free here and will not be
at T2d. `state` stays `const`: the vertex rule reads `mesh.sheetVector()`, which
T2d's RHS refreshes through `SurfaceState::updateSheetVector` before each
evaluation, so the quadrature never needs to write the state.

**Exit criterion:** **a `regression`-tier test** comparing the induced velocity
on the default icosphere against a hard-coded reference computed from the
Python, to `1e-13` relative. The tier, not the test, supplies the rank sweep
(the convention T1c set), so the criterion's ranks 1 and 4 are a subset of the
gate's 1-6 on SERIAL and HIP. **This grows the ship gate to two members** —
pre-authorized, so it does not need re-confirming, but it does mean registering
one binary per backend with the `_<BACKEND>` suffix the gate selects on, never
one suffix-less binary (which the installed path skips entirely, a silent
zero-test pass).

**Met**, by the new `regression`-tier test `Beatnik_Test_BirkhoffRott`
(`tests/regression_tests/Beatnik_Test_BirkhoffRott.cpp`), run through
`flux batch scripts/tuolumne/run_regression_minset.flux`. **The ship gate now
has two members** — pre-authorized above — so it ran **24 launches**
(2 members x SERIAL and HIP x ranks 1, 2, 3, 4, 5, 6) and every one passed:
`[gate] PASS (label=regression)`, with `Beatnik_Test_BirkhoffRott` reporting
**29/29 checks** in all twelve of its configurations. The criterion's ranks 1
and 4 are a verified subset, not the extent of what was checked.

Agreement with the reference is at **`1e-15` or better on every compared
quantity** — two decades inside the criterion's `1e-13`, and the worst
disagreement seen anywhere in the sweep is `1.30e-15`. **No tolerance was
touched and no reference number was adjusted.** Worst relative error over all
twelve configurations, against literals computed from the read-only Python on
`mesh.icosphere_mesh( subdivisions=2, radius=0.25, center=(0,0,0.25) )` with
\f$\phi = a\cdot p\f$, `a = (0.3, -0.7, 1.1)`:

| Quantity | Python | worst rel over 12 configs |
| --- | --- | --- |
| source `max\|S\|` | `1.3193451648051979` | `1.7e-16` |
| source `sum\|S\|` | `167.62467266803066` | `3.4e-16` |
| velocity `max\|u\|` | `0.71412231153219252` | `3.1e-16` |
| velocity `min\|u\|` | `0.21419090124870638` | `1.3e-15` |
| velocity `sum\|u\|` | `68.015172526189744` | `6.3e-16` |
| velocity `sum u_x` | `-13.809091739775855` | `1.3e-16` |
| velocity `sum u_y` | `32.221214059476992` | `2.2e-16` |
| velocity `sum u_z` | `-50.633336379178147` | `2.8e-16` |
| riesz `max` | `0.22717497673577594` | `6.1e-16` |
| riesz `min` | `-0.22717497673577597` | `1.2e-16` |
| riesz `sum\|psi\|` | `18.452052083854369` | `3.9e-16` |

The two source scalars are T2b's published values, reproduced bit for bit from
the same reference call — so the sheet strength feeding the kernel is pinned
before the kernel is compared, and a disagreement localizes to the BR sum.

The test checks more than the criterion: the entity counts and halo depth, the
**owned-set partition** (R9's precondition, summed with a plain
`MPI_Allreduce` rather than read from Tessera), the **global source count**
(exactly 162 at every rank count — the direct R9 detector, since a
ghost-inclusive list would make it 200-400 here while every velocity number
moved smoothly and plausibly), the potential's ghost rows after the explicit
`haloExchange()` the sheet-vector update requires, three **signed** velocity
components (which a reversed \f$\delta\times S\f$ fails and the magnitudes do
not), the Riesz scalar's negative `min` (which pins the \f$-1/4\pi^2\f$ sign),
and a **negative case** requiring `--br-sign -1` to negate the velocity
*bitwise* while leaving the Riesz scalar alone.

That negative case is where running found something. `br_sign` independence of
the Riesz path is **not** bitwise on HIP, because `generateGradient` re-runs
`surfaceGradient`, whose atomic face scatter is not bitwise reproducible — the
first version of the check demanded equality and failed on HIP at all six rank
counts while passing on SERIAL at all six. It is asserted at `1e-13` relative
instead, measured at `2.4e-16`, with the discriminator recorded: a `br_sign`
leak would make that number `2.0`. Details in the progress log, under *T2c*.

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

**Exit criterion:** **regression test 2 (direct-solve-10-steps) passes at all 10 timesteps**, `--rtol 1e-10`,
at ranks 1, 2, 3, 4, and 5. Volume drift stays below `1e-12` relative.

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

### T4a/T4b — the disjoint editing families *(OPEN DESIGN QUESTION — read before either)*

**Recorded by the M1 adapter rework. Not resolved, deliberately.**

Tessera has **two disjoint families of topological edit and a mesh belongs to
exactly one of them**:

| Family | Operations | Invariant maintained | `Level` |
| --- | --- | --- | --- |
| **Hierarchical** | `refine()`, `refineLocal()` | 2:1 level balance + conforming closure | authoritative |
| **Remesh** | `splitEdges()`, and `collapseEdges()`/`flipEdges()`/`compact()` when they land | conformity and manifoldness only | advisory |

**Verified, not inferred** (`../tessera/src/Tessera_EditFamily.hpp`, README
*Editing families*): a mesh carries an `EditFamily` tag, `None` until its first
topological edit and **fixed thereafter**; every entry point calls
`requireEditFamily()` and **throws `std::runtime_error`** naming both families
when the tag disagrees. Beatnik cannot make this a Beatnik-side check, cannot
order its calls around it, and cannot catch-and-continue. Tessera's reason is
that `refine()`'s 2:1 invariant is stated in *level differences* and is coherent
only because `refine()` performs the uniform 1→4 red split: bisecting one edge
of a triangle produces children whose edges have mixed levels, so a
`splitEdges()` child merely inherits its parent's level.

**Beatnik's default configuration runs both**: T4a is `refine()`, T4b is
split/collapse/flip, and `run_adaptive_mesh_bubble.py::main` interleaves them
inside the step loop (refine every `--refine-every` steps, dynamic remesh every
step unless `--no-dynamic-remesh`). So this must be settled before either task
is implemented. The four candidates below are what Tessera's API actually
admits. **No resolution is presumed, and nothing in T1b's code is shaped around
one.**

#### Option 1 — Two mesh objects, one per family, transferring state between them

*What is verified:* a mesh can be built from a triangle soup
(`buildFromTriangleSoup` + `distribute`, replicated input) or from per-rank
patches plus canonical keys (`buildFromTriangleSoupDistributed`), and
`writeMesh`/`readMesh` round-trips a mesh **together with its whole vertex user
pack** across a change of rank count. So a transfer is expressible.

*What it costs Beatnik:* the soup builders carry **positions and connectivity
only** — not the user pack — so the three vertex fields would have to be moved
separately, keyed by gid, and **the new build renumbers gids**. That is where
provenance is lost, and it is not incidental: `buildFromTriangleSoupDistributed`
requires a *canonical key per local vertex* whose contract is "rank-independent,
equal iff the same vertex, collisions throw", and `makeVertexKey` is structured
(a base index, or the sorted pair of a midpoint's two parents). A vertex that
arrived by several rounds of remeshing has no such provenance available to
Beatnik, so Beatnik would have to invent a key scheme — which is exactly the
kind of topology bookkeeping the adapter exists to avoid. The one path that
*does* carry the pack is an HDF5 round trip per phase switch: a collective
MPI-IO write plus a read, every time the two phases alternate, i.e. potentially
every step. Correct, and far too expensive.

*Unknown:* whether an in-memory "clone into a fresh mesh on the same partition,
carrying the pack" is feasible Tessera-side. It does not exist in the README and
would be a new capability.

#### Option 2 — Drop one family from the default configuration

**Dropping the remesh family** (i.e. `--no-dynamic-remesh` becomes the only
supported mode) is expressible *today* and T4a's exit criterion already runs
that configuration. But dynamic remeshing is the Python's default and is what
holds triangle quality through the roll-up; T4c exists because a run without it
dies on the "curvature sliver". This trades a correctness-adjacent capability
for a scheduling convenience and is the weakest of the four.

**Dropping the hierarchical family** is the interesting direction, and Tessera
says so itself: README *Future Optimizations* records that "the driving
consumer, Beatnik's z-model remesher, is entirely edge-addressed and never calls
`refine()`". `dynamic_remesh.py` is indeed entirely edge-addressed. And the AMR
indicators, which mark **faces**, translate: marking a face means splitting its
three edges, which is precisely a red split, and `splitEdges()` performs exactly
the marked bisections **conforming on exit with no closure layer and no 2:1
pass**.

*What it costs Beatnik:* T4a's exit criterion compares face counts against a
Python `refine_marked_faces` run, and a split-based refinement will produce a
*different* face count wherever the closure pattern differs — `splitEdges` gives
2/3/4-child patterns where red-green gives a transient closure layer. That
criterion would need restating; risks R4 and R7 already accept this class of
divergence, so the precedent exists. `Level` also becomes advisory, so nothing
bounds the level jump across a refinement front.

*What is verified:* `splitEdges( mesh, halo, edgeMask )` takes a host
`std::vector<char>` sized `numOwnedEdges()`, the **edge owner** decides, the
decision is propagated to every rank holding an incident face, the result is
conforming, and `rebuildHalo()` is called on the way out at the recorded depth.
It is `split_selected_edges` directly. Conformity — not 2:1 balance — is what the
surface operators need.

*Unknown:* whether the face-mark → edge-mask translation reproduces the Python's
`projected_red_green_face_count` closely enough for `--max-faces` accounting; and
whether repeated non-uniform bisection degrades triangle quality faster than
red-green does. Tessera measured red-green's worst radius ratio saturating by
round 11 and flat through round 16; it publishes **no equivalent measurement for
`splitEdges`**, so this is a real gap in the evidence rather than a formality.

**This is the leading candidate on cost**: no copy, no lost provenance, no new
Tessera capability, and one editing family for the whole run. It is *not* free of
prerequisites — a remesh-only Beatnik still needs coarsening, so it blocks on
G5b (collapse) and G5c (flip), which block T4b anyway.

#### Option 3 — Rebuild the mesh between the AMR phase and the remesh phase

One logical surface, re-created as a fresh `Mesh` (hence `EditFamily::None`)
whenever the phase changes.

*What is verified:* only that a fresh mesh is untagged. Everything else is
option 1's transfer problem, unchanged — the pack does not come along with a
soup rebuild.

*What it costs Beatnik:* option 1's cost, **per phase change**, plus the loss of
the partition (a fresh `distribute`/`loadBalance`) and of gid continuity. Since
`main` interleaves the two phases inside the step loop, that is paid every step.
Strictly worse than option 1 with no compensating benefit.

#### Option 4 — Push a change upstream into Tessera

Tessera has already scoped the full version: README *Future Optimizations*,
"Unify the two editing families by extending the level model to **anisotropic
bisection**", pointing at `../tessera/tasks/edge-split.md` Decision 1's
alternative. It needs per-edge levels plus a compatible balance rule, maintained
by the mark-propagation fixpoint, the closure patterns **and the HDF5 format**
alike. Tessera calls it "a much larger design than the guard it would replace,
and one no known consumer needs" — naming Beatnik as the consumer that does not
need it *because* its remesher is edge-addressed. It is not started.

*What it costs Beatnik:* no implementation, but it blocks on an upstream design
task that Tessera has explicitly deprioritized on the grounds that option 2 is
available.

**The narrower upstream ask is the part worth remembering:** options 1 and 3 both
want the same much smaller thing — *an in-memory clone of a mesh into a fresh
`Mesh` on the same partition, carrying the vertex user pack and needing no
canonical keys*. That is a fraction of the anisotropic-bisection design and would
make either option practical. Whether Tessera would take it is unknown and has
not been asked.

#### What a later session should do first

Not pick from this list on paper. The one measurement that discriminates is the
unknown under option 2: **does a `splitEdges`-only refinement hold triangle
quality comparably to red-green over many rounds?** If it does, option 2 is
clearly right and the question closes. If it does not, the choice is between
option 4's narrow ask and living with `--no-dynamic-remesh`. That measurement is
a Tessera-side experiment, not a Beatnik one, and it does not need any of T4a or
T4b written first.

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

**Fill in:** `Beatnik_IOInterface.hpp::read`; `Beatnik_Restart.hpp::load`.
← *Python:* `::load_state_checkpoint` (993-1033), `::main` (1199-1214)

**M2 CHANGE — `Comm::broadcastFromRoot` is no longer on this path.** `read` is
`Tessera::readMesh`, which reconstructs the mesh and all three vertex fields
collectively with no rank holding the global mesh. Two traps M2 found and both
recorded on `CheckpointIO::read`: the halo comes back **1-deep** and must be
widened with `rebuildHalo( mesh, halo, 2 )` before any RHS evaluation (R8), and
a structural mismatch with the build is an **`MPI_Abort` inside Tessera**, not a
catchable exception — so this task's cheap checks must run *before* the call.

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

### M1 — Open `../tessera`: mesh model — **DONE**

**First task permitted to read `../tessera`.** Reconcile
`Beatnik_MeshInterface.hpp` against what Tessera actually provides: the storage
model, the owned/ghost partition, adjacency, the topological edit operations, and
whether the `MeshEditResult` parent/weight scheme matches how Tessera reports
field transfer. **Rewrite the adapter; do not spread Tessera types outward.**

**Met.** `src/Beatnik_MeshInterface.hpp` rewritten against the real
API and building clean. Tessera is now a hard dependency
(`find_package(Tessera REQUIRED)`, `Tessera::Tessera` on `DEPENDS_ON`, and
`depends_on("tessera")` added to the out-of-repo spack package). Bodies remain
`BEATNIK_NOT_IMPLEMENTED`.

#### What Tessera provides

Tessera is a distributed unstructured triangle-mesh library over Cabana +
Kokkos. Everything below is public API, read from `README.md` and the headers it
names.

| Beatnik needs | Tessera call | Notes |
| --- | --- | --- |
| Entity storage | `Mesh<Scalar,Dim,VertexFields<>,EdgeFields<>,FaceFields<>,Mem,Exec,Mode>` | **Three** kinds — vertices, edges, faces — each a Cabana AoSoA, owned-first, with a compile-time **user field pack** the caller declares. |
| Icosphere | `buildIcosphere( mesh, subdiv )` | Same golden-ratio table and normalized-midpoint rule as the Python. Unit sphere only. Replicated on every rank. |
| Adopt a soup | `buildFromTriangleSoup( mesh, soup )` | Host, serial, **replicated** input. |
| Partition | `facePartitionByAxis( mesh, axis )` | Deterministic geometric block partition; no communication. |
| Distribute | `distribute( mesh, halo, faceOwner )` | Cuts to owned + 1-deep ghost, builds the three halo plans, rebuilds both CSRs. |
| Halo exchange | `haloExchange( mesh, halo )` | Collective; syncs the **whole tuple of all three kinds** at once. |
| Vertex one-ring | `buildVertexStencil( mesh, 1 )` | CSR, local indices, ascending. Also `mesh.vertexEdges()` / `mesh.vertexFaces()` directly. |
| Edge list + edge→face | edge AoSoA `EdgeField::Verts` / `::Faces` | Maintained continuously; **nothing to build**. |
| Geometry accessor | `buildMeshGeometry( mesh )` | Derives per-face/per-edge **local** vertex indices from the gid storage; device-capturable. Covers owned + ghost. |
| Raw primitives | `faceArea`, `faceNormalRaw`, `edgeVector`, `cotangentAtCorner` | Unoriented, convention-free. |
| Weighted stencil apply | `applyStencil( mesh, stencil, w, in, out )` | Caller builds `w`; owned rows only. |
| Face→vertex reduce | `reduceVertexFromFaces( ... , op )` | Caller's op; one thread per owned vertex, no atomics. |
| Conforming AMR | `refine( mesh, halo, mask )` | **Red-green conforming refinement is NATIVE** (answers T4a directly). 1→4 red split + cross-rank 2:1 balance + transient closure. |
| Refinement field transfer | `RefinePolicy` | Pluggable per-field midpoint blend; default is the linear average. |
| Migration | `migrate( mesh, halo, dest )` | External per-owned-face assignment. Moves whole tuples; **all user fields follow automatically**. Rebuilds ownership, ghosts and halo plans. |
| Load balance | `loadBalance( mesh, halo )` / `computeLoadBalance` | Zoltan2 geometric MultiJagged (never RCB — broken on Tuolumne). |
| External partition input | `ownedFaceCentroids/Gids/Weights( mesh )` | The documented Canopy contract; available to Beatnik unchanged (relevant at T3a). |
| Global min | `globalMin( mesh, x )` | The only global reduction. |
| Stale-handle safety | `generation()` + `GenerationHandle` | Every handle is stamped; copying a stale one **aborts**. `haloExchange` does not bump it. |
| I/O | `writeMesh` / `readMesh` (+ XDMF) | M2's territory. |

**Added since M1 was first written** (Tessera branch `conforming-refinement`) — the calls that close G1-G4 and G6-G8 and that Beatnik
must now use instead of rolling its own. Each is documented in Tessera's README
API section.

| Beatnik needs | Tessera call | Notes |
| --- | --- | --- |
| **Two-ring halo** (the RHS; R8) | `distribute( mesh, halo, faceOwner, depth )`, `rebuildHalo( mesh, halo, depth )`, `mesh.haloDepth()` | Set **`depth = 2` once at setup**. `refine()`, `splitEdges()` and `migrate()` *preserve* it, so nothing downstream re-states it. `halo.depth == 0` means never built and is treated as 1. |
| Two-ring stencil | `buildVertexStencil( mesh, 2 )` | Now **complete** at depth ≥ 2, and **throws `std::invalid_argument`** when `k > mesh.haloDepth()` instead of returning silently short rows. |
| Ghost scatter-add | `haloScatterAddVertices<FieldIndex>( mesh, halo )` (also `...Edges` / `...Faces`, and the plan-level `haloScatterAdd<F>( comm, aosoa, plan )`) | **One named field per call**, scalar or `Scalar[N]` (componentwise) — not a whole-tuple call like `haloExchange`. Assemble by looping **owned** faces into local (possibly ghost) slots, then scatter-add. Ghosts are left untouched, so follow with `haloExchange()` if kernels read them; calling it twice **double-counts**. |
| Global sum / max / all-finite | `globalSum`, `globalMax`, `globalAllFinite`, plus the existing `globalMin` | `globalAllFinite` takes a **verdict, not data** — Beatnik does its own Kokkos `isfinite` sweep and hands over the bool. `globalSum` on `double` is **not** bitwise reproducible across rank counts (this is R2, now stated by Tessera too). |
| Global entity counts | `globalOwnedVertices/Edges/Faces/Euler( mesh )` | `long long`, exact. Replaces Beatnik's `global*Count()`. |
| Face→face adjacency | `buildFaceAdjacency( mesh )` → `FaceAdjacency<MemorySpace>` | Collective, generation-guarded. **Two halves:** `nbrGid`/`nbrOwner` are always valid and are what a *topological* consumer (AMR mark growth, remesh conflict resolution) uses; the local-index `csr` is usable only where `numNonResident == 0`, which a *geometric* consumer must check rather than assume. Rows sorted by neighbour gid, so row order is rank-count invariant. |
| Caller-driven edge split | `splitEdges( mesh, halo, edgeMask, policy )` → `SplitResult` | Host `std::vector<char>` sized `numOwnedEdges()`; the edge **owner** decides. Every incident face becomes 2, 3 or 4 children, **conforming on exit with no closure and no 2:1 pass**. Rebuilds the halo. This is `dynamic_remesh.py::split_selected_edges` directly. |
| Distributed initial build | `buildIcosphereDistributed( mesh, halo, subdiv, depth )`, `buildFromTriangleSoupDistributed( mesh, halo, localSoup, localVertexKeys, depth )` + `VertexKey` / `makeVertexKey` | No rank holds the global mesh, and `distribute()` is not on this path. Position multisets are **bitwise identical** to the replicated path. `buildIcosphere` + `distribute` remains supported and is right for the default subdivision-2 sphere. |
| Lat/lon sphere | `generateLatLonSphere<Scalar>( nLat, nLon )` / `buildLatLonSphere( mesh, nLat, nLon )` | Serves `--mesh-kind latlon`. Exact poles, no seam duplicate, fixed quad diagonal, CCW-outward. **libm reproducibility caveat** — positions are not bit-reproducible across machines, which is exactly R1's `latlon` concern. |
| Load-balance mode | `loadBalance( mesh, halo, tol, LoadBalanceMode::…, &stats )` / `computeLoadBalance(...)` | Default is **`Sampled`** — the only mode measured run-to-run reproducible; nothing is gathered to rank 0 in `Sampled` (`O(nparts)`) or `Distributed` (zero). `LoadBalanceStats::rootSolveFaces` reports it. |

**One new constraint that did not exist at M1: the two editing families are
disjoint and enforced by a throw**, so `refine()` (T4a) and
`splitEdges()`/`collapseEdges()`/`flipEdges()` (T4b) cannot run on the same
mesh. Laid out in full, with the four candidate resolutions, under
"T4a/T4b — the disjoint editing families" in the task sequence above. Noted on
all four declarations in `Beatnik_MeshInterface.hpp` so it cannot be met for
the first time as a runtime throw.
**Calls the M1 adapter rework and T1b introduced.** The two tables
above say what Tessera *offers*; this one says what Beatnik now actually calls,
so a reader can see the adapter's whole Tessera surface in one place.

| Beatnik entry point | Tessera calls it makes |
| --- | --- |
| `SurfaceMesh::generateIcosphere` | `buildIcosphere` → in-place scale/translate on `vertexSlice<Position>` → `facePartitionByAxis(mesh, 2)` → `distribute( mesh, halo, faceOwner, 2 )` → `haloExchange` → `globalSum` (the orientation check) |
| `SurfaceMesh::adopt` | `TriangleSoup` fill → `buildFromTriangleSoup` → the same partition/distribute/exchange |
| `faceVertices()` / `edgeVertices()` | `buildMeshGeometry( mesh )`, cached on `generation()`; returns its `faceVerts` / `edgeVerts` |
| `vertexOneRing()` | `buildVertexStencil( mesh, 1 )`, cached on `generation()` |
| `edgeAdjacency()` | host read of `edgeSlice<EdgeField::Faces>` + `faceSlice<FaceField::Gid>`, resolved to local indices; cached on `generation()` |
| `faceAdjacency()` | `buildFaceAdjacency( mesh )` (**collective**), cached on `generation()`; exposes `nbrGid`/`nbrOwner`/`numNonResident` as well as the CSR |
| `globalVertexCount` / `globalEdgeCount` / `globalFaceCount` / `globalEulerCharacteristic` | `globalOwnedVertices` / `Edges` / `Faces` / `Euler` |
| `haloDepth()` | `mesh.haloDepth()` |
| `haloExchange()` | `haloExchange( mesh, halo )` |
| `haloScatterAddVertexField<FieldId>()` | `haloScatterAddVertices<userVertexField<FieldId>()>( mesh, halo )` |
| `setVertices()` | writes `vertexSlice<Position>` over the owned range |
| `Comm::allReduceSum/Min/Max/AllFinite` | one `MPI_Allreduce` each on the caller's `MPI_Comm`, identical datatype and op to `Tessera::globalSum/Min/Max/AllFinite` (see the note in that header on why they take a comm rather than a mesh) |

Two structural facts drive most of the adapter:

1. **Connectivity is stored as global ids**, so a `Mesh` is not device-capturable
   and `faces()` cannot return a `View<LocalIndex*[3]>` from storage. Kernels
   index `MeshGeometry::faceVerts`, rebuilt per topology generation.
2. **Field transfer is Tessera's job, not the caller's.** `refine()` interpolates
   midpoints through the policy, `migrate()` ships whole tuples, `haloExchange()`
   syncs whole tuples. A per-vertex field held in a `Kokkos::View` *outside* the
   mesh is silently dropped by refinement and silently stale after migration.

#### Does `MeshEditResult`'s parent/weight scheme match? **No.**

Tessera reports no parent map and needs none — the default `RefinePolicy` blend
*is* the `(0.5, 0.5)` weights the old struct encoded, applied inside `refine()`.
`MeshEditResult` is therefore **deleted** (no other header named it) and replaced
by `MeshEditReport`, which carries what `Tessera::RefineResult` actually gives:
2:1 balance rounds, local split-edge count, post-edit owned counts.

The consequence is the largest M1 change: **Beatnik's evolved per-vertex state
must live inside the Tessera mesh as vertex user fields**, declared in the
adapter as `VertexFields<Real, Real[3], Real[3]>` =
`{Potential, SheetVector, MaterialPosition}` and named from outside via
`Beatnik::VertexFieldId`. The linear average is the correct transfer rule for
all three, so `DefaultRefinePolicy` is used unchanged.

#### Gaps — what Tessera does NOT provide

*Of the eleven recorded at M1, **eight closed Tessera-side** (branch
`conforming-refinement`) — G1, G2, G3, G4, G5a, G6, G7,
G8 — via the calls in the "Added since M1" table above. Only G5b, G5c and G5d
remain open, and they are what still blocks T4b/T4c.*

**G5 — No topological edit except the face-mask refine. — PARTIALLY CLOSED.**
  - **G5a — caller-driven edge split. — DONE.** `splitEdges( mesh, halo,
    edgeMask )` bisects exactly the marked edges, the owner deciding, every
    incident face becoming 2, 3 or 4 children — **conforming on exit with no
    closure and no 2:1 pass**. This is `split_selected_edges` directly.
  - **G5b — edge collapse. — OPEN.** Does not exist at any level. The data model
    has no coarsening path, so neither the link condition nor a cross-rank
    owner-decides protocol has anywhere to attach. Tessera task
    `../tessera/tasks/edge-collapse.md` (NOT STARTED, largest of the eleven; hard
    dependency on halo depth ≥ 2, which has landed).
  - **G5c — edge flip. — OPEN.** Does not exist. Needs both incident faces, so a
    correct one needs the edge-coordinator machinery — which
    `buildFaceAdjacency` (G4) now exposes, so the prerequisite is met.
    `../tessera/tasks/edge-flip.md` (NOT STARTED).
  - **G5d — compaction. — OPEN.** Prerequisite of collapse rather than a
    consequence of it in Tessera's own ordering.
    `../tessera/tasks/mesh-compaction.md` (NOT STARTED).
**T4b (dynamic remeshing) and T4c's flips are still blocked** — the split third
of `dynamic_remesh.py` is now expressible, the collapse and flip thirds are not.

#### What Beatnik must implement itself (and legitimately may)

None of these is haloing or partitioning:

- The **discretization conventions** Tessera deliberately refuses: outward normal
  orientation, the vertex-area definition, the cotangent weight fill for
  `applyStencil`, the curvature sign. This is `Beatnik_MeshGeometry.hpp`, exactly
  as scoped.
- **Scale and translate** the unit icosphere to `radius` / `center`, and verify
  the winding is outward (positive enclosed volume) rather than assume it.
- **Owned-only iteration discipline** (risk R9): Tessera exposes `numOwnedX()`
  and orders entities owned-first, but enforces nothing. Owned edges *do* form a
  global partition, so the edge-length reduction has a correct answer available.

#### One contract the adapter encapsulates

  mark-propagation fixpoint internally (`MPI_Allreduce`-guarded, hard-capped,
  round count reported). An arbitrary rank-local mask is a legal input, so
  `Beatnik_Communication.hpp::reconcileRefinementMarks` has no work left to do.

Two smaller shape changes, both recorded inline in the header: `adopt()` now
requires its arrays replicated on **every** rank (not rank 0), because
`buildFromTriangleSoup` has no communication and `distribute()` relies on
replication; and `refine()`'s mask is a **host `std::vector<char>` sized
`ownedFaceCount()`**, not a device view sized `Nf`, so a device-computed AMR
indicator must round-trip to the host.

### M2 — Open `../tessera`: HDF5 I/O — **DONE**

Reconcile `Beatnik_IOInterface.hpp`. The checkpoint **schema** is fixed by the
gold files (see the table in that header) and is not negotiable; what is
negotiable is whether Tessera writes it directly, or Beatnik gathers and writes.
Also settle the dataset paths, and update `FIELD_MAP` at the top of
`compare_output.py` in the same change.

**Met.** `src/Beatnik_IOInterface.hpp` rewritten against
`Tessera_HDF5Writer.hpp` / `Tessera_HDF5Reader.hpp` / `Tessera_IoCommon.hpp`,
building clean. Bodies remain `BEATNIK_NOT_IMPLEMENTED`; T1c implements `write`
and T5b `read`.

#### The open question, decided: **Tessera writes it.**

`Tessera::writeMesh( mesh, stem )` is a collective MPI-IO write of every rank's
**owned** entities, exactly once, into a clean partition of `<stem>.h5`. Dense
global vertex/edge indices come from an `MPI_Exscan` over owned-only counts, and
connectivity is translated into them *before* the write — which is precisely the
local-to-global renumbering the pre-M2 header identified as the gather path's
one genuinely error-prone step. It also carries the whole vertex user pack, so
all three Beatnik fields are written for free.

So the gather is not merely unnecessary, it is worse on every axis: O(global)
memory on rank 0, a serialized write, and a hand-rolled reimplementation of the
hard part. **`Beatnik_Communication.hpp::gatherForCheckpoint` is deleted** (the
M1 precedent: delete rather than keep a shim no caller could correctly consume).

What Tessera does *not* write is the scalar metadata — its only root-attribute
types are `int` and `uint64`, and the checkpoint needs a `double` time, two
`double` scalars and a string. Beatnik appends a `/beatnik` group from rank 0
after `writeMesh` closes the file, between barriers.

#### The dataset paths, settled

`FIELD_MAP` in `compare_output.py` and `H5_PATH` in `make_fixtures.py` were both
updated to match, and the two committed `.h5` fixtures regenerated.

| `.npz` key | HDF5 dataset | Written by |
| --- | --- | --- |
| `vertices` | `/vertices/position` | Tessera |
| `faces` | `/faces/verts` (u64) | Tessera |
| `potential` | `/vertices/u0` | Tessera |
| `sheet_vector` | `/vertices/u1` | Tessera |
| `remesh_material_position` | `/vertices/u2` | Tessera |
| the five scalars | `/beatnik/<name>` | Beatnik |

`/faces/verts` holds **dense global vertex indices** written at the same exscan
offsets as `/vertices/position`, so it indexes that table's rows directly — the
`.npz` `faces` convention exactly, needing only a dtype widening.

#### Four consequences that are not obvious from the paths

1. **`/vertices/u<N>` is a POSITIONAL name**, and the mapping to a meaning is
   `Beatnik::VertexFieldId`'s declaration order. Nothing in the file says `u0`
   is the potential, so **reordering that enum silently relabels every
   checkpoint on disk** — which is the one failure mode `FIELD_MAP` exists to
   prevent, now reachable through a file that does not mention any Beatnik name.
   Mitigation: the writer also emits `/beatnik/vertex_field_names`, and
   `compare_output.py` **verifies** `FIELD_MAP` against it (`LoadError` on
   disagreement). Deliberately a cross-check and not an inference — resolving
   paths *from* the declaration would make the script agree with whatever the
   writer did, including a silent reordering. The stale pre-M2 note in
   `Beatnik_MeshInterface.hpp` claiming the schema is "keyed by name, not by
   this index" is corrected.
2. **A Beatnik checkpoint always carries BOTH state fields.** `u0` and `u1` are
   slots in one Cabana tuple and `writeMesh` writes the pack unconditionally,
   while the Python writes `potential` *or* `sheet_vector` and never both.
   Left alone this fails every comparison on `sheet_vector: present in cpp
   only`. `compare_output.py` now compares only the field `state_model` selects
   and skips the inactive one; `remesh_material_position` keeps the strict
   both-or-neither rule, because there a one-sided presence is a real signal.
3. **Reading is `readMesh`, not `adopt`.** `Tessera::readMesh` rebuilds the whole
   distributed mesh *and every vertex user field* with a fresh block partition
   and a `migrate()`, so a checkpoint round-trips **across a change of rank
   count** and no rank ever holds the global mesh. The pre-M2 read path (rank 0
   reads, `broadcastFromRoot`, `adopt`) is gone, and with it the `state` /
   `material` out-parameters of both `read` and `write` — under the M1 field
   pack the solution *is* in the mesh. `broadcastFromRoot` survives for its other
   caller only: the R1 read-the-gold-file mitigation, whose input is a `.npz`.
4. **Two traps in `readMesh` for T5b.** (a) The halo it leaves is **1-deep** — it
   hands a freshly-constructed halo, whose `depth == 0` reads as the historical
   1, to `migrate()` — so `read` must follow with `rebuildHalo( mesh, halo, 2 )`
   or the two-ring RHS is wrong at partition boundaries (R8). (b) A structural
   mismatch (precision, dim, refinement mode, the vertex field pack) is an
   **`MPI_Abort` inside Tessera, not an exception**, so it cannot become the
   `std::runtime_error` `Beatnik_Restart.hpp` promises; the checks Beatnik *can*
   make must happen before the call, not after.

#### One deviation from the Python, recorded

`_latest` is a **rank-0 symlink**, not a second full write. A byte copy is not
equivalent either — the `<stem>.xmf` sidecar names its `.h5` by stem, so a copied
`_latest.xmf` would point at the wrong file, while a symlinked one resolves
correctly in the same directory.

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

### Measured at T1c: the cross-rank spread of the two scalars

Both carried scalars, at ranks 1-6 on SERIAL and HIP — twelve configurations,
from `Beatnik_Test_InitialConditions`, which reports them on every gate run:

- **`initial_min_edge` — spread ZERO.** `0.068976121063816842` at every rank
  count on both backends, bit-identical to the T1a Python value. Expected:
  `MPI_MIN` on a fixed set of values is order-independent, so this is
  reproducible *by construction* and not by luck. It is the reduction the
  adaptive dt and both proximity radii key off, which is the good half of the
  news.
- **`initial_volume` — spread 2 ulp**, `2.78e-17` absolute / `4.4e-16` relative,
  taking three distinct values (table in T1c's completion note). The T1a value
  is hit bitwise in 9 of the 12 configurations.

**What this means for a tolerance:** nothing needs one. `4.4e-16` is four orders
inside `1e-12`, and the two `1e-14`-relative comparisons T1b and T1c both make
still pass. Do not read the spread as a budget to spend — it is the *observed*
noise floor of one `globalSum` over 320 faces, and R2's warning is about
*trajectories*, where the spread compounds over steps. T2d is where it starts to
matter.

**Two things the numbers say that the risk statement did not.**

1. **The spread is NOT primarily cross-rank.** np1 (where `MPI_SUM` has nothing
   to add) already differs between SERIAL and HIP, and np5 lands 1 ulp *high* on
   SERIAL and 1 ulp *low* on HIP. So most of it is the on-node
   `Kokkos::parallel_reduce` tree order, not `MPI_Allreduce` — which
   `Beatnik_MeshGeometry.hpp`'s DETERMINISM note already predicted for the
   assembled fields, and which R2 attributed to the collective.
2. **It does not trend with rank count.** np6, which has the largest ghost
   fraction of the six (owned F 53-54 against local F 91-184), reproduces the
   T1a value exactly on both backends. That is the observation that discriminates
   R2 from R9 — see R9.

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

**Largely retired by G1.** Tessera's halo now takes a depth, so the fix is
`distribute( mesh, halo, faceOwner, /*depth=*/2 )` once at setup — preserved
across `refine()`/`splitEdges()`/`migrate()` — and `buildVertexStencil( mesh, 2 )`
throws rather than returning short rows if the depth was not set. What remains of
the risk is only forgetting to set it, which is now loud rather than silent.

The RHS is a **two-ring** stencil on the potential: one surface gradient builds
the sheet vector, and a second is taken of the Bernoulli potential. With a
one-face-deep ghost layer the potential must be exchanged **twice** per RHS
evaluation — and **that does not work**: a second `haloExchange()` refreshes the
same 1-deep ghost set rather than widening it, which is why depth, not exchange
count, is the answer. The easy bug is a single exchange of a single-deep halo,
which is wrong only near partition boundaries and only by a small amount — so it
produces a plausible-looking solution with a seam that moves when the rank count
changes.

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

### Checked at T1c, and NOT biting — with the reasoning, not just the verdict

R9 and R2 present *identically* in a failure report: both make `initial_volume`
disagree with the gold value by an amount that varies with the rank count. T1c
ran the rank sweep and told them apart on three independent grounds, all three
now mechanized inside `Beatnik_Test_InitialConditions` so they are re-checked on
every gate run rather than being a one-off measurement:

1. **The owned sets partition the global sets exactly.** Summing each rank's
   `ownedVertexCount/ownedEdgeCount/ownedFaceCount` with a plain
   `MPI_Allreduce(MPI_SUM)` gives **162 / 480 / 320** at every rank count on both
   backends — the serial totals, once each. Deliberately summed independently of
   Tessera's own `globalOwnedX`, because owned-versus-local is exactly the
   distinction R9 turns on and two agreeing paths are worth more than one. This
   is the precondition the volume and edge-length reductions need, and R9 says to
   check it rather than assume it.
2. **The deviation does not scale with the ghost fraction.** It is ±1 ulp and it
   does not trend: np1 (no ghosts at all) deviates, np6 (ghosts outnumbering owned
   entities roughly 2:1) is exact, and np5 deviates in *opposite directions* on
   the two backends. A ghost-inclusion bug is monotone in the ghost fraction and
   orders of magnitude larger — it would put np6 furthest out, not exactly on the
   reference.
3. **The closed form is unmoved.** `volume / (4πR³/3) = 0.96616074859858714` at
   every rank count — the polyhedral deficit of the subdivision-2 icosphere,
   a property of the triangulation and not of the partition. Double-counting even
   a handful of ghost faces would move this in the second or third digit.

**Verdict: summation order (R2), not double-counting (R9).** So the tolerance was
left alone, which is what the discrimination was for.

The same three checks are the template for T2d, which faces the harder version of
this ambiguity: R8's seam-versus-noise question. Note grounds 1 and 3 are
*structural* and stay decisive there, while ground 2 (the no-trend argument) gets
weaker once positions evolve, because a real seam bug also moves with the
partition.

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