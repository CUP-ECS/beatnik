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

**There is a timestep, and it reproduces the reference for ten steps.** **T2b has
landed, so the surface differential operators are no longer stubs** —
`SurfaceOperators::{faceScalarGradient, surfaceGradient,
cotangentLaplacianScalar, graphLaplacianScalar, graphLaplacianVector,
meanCurvatureNormal, projectTangent}` and `SurfaceState::updateSheetVector` are
implemented and validated against the Python reference (see T2b's completion
note). **T2c has landed too**, so the vertex source quadrature and the direct BR
evaluation are implemented and validated as well. **T2d has landed**: the RHS,
the volume projection, the TVD-RK3 integrator with adaptive dt, the step loop,
the diagnostics and regression test 2 are written, and the **whole 36-launch
gate is green** — three members × {SERIAL, HIP} × ranks 1-6. Ten timesteps now
match the Python gold set at `--rtol 1e-10`, with the adaptive dt reproduced and
the volume drift pinned to the reference's own. See T2d's completion note and the
`## T2d — completion` log entry.

What is still a stub:

- **Adaptivity is half-open. T4a is `**DONE**` and green at the full gate** —
  see that entry's Met. paragraph. `--no-dynamic-remesh
  --refine-every N` now refines, through `Tessera::splitEdges()`; everything
  else on the post-step path still throws, by name and by task ID, so the
  reference's *default* configuration (dynamic remeshing every step) aborts at
  setup rather than silently running a different problem. The **three
  post-refine quality repairs are rejected too** — `--flip-passes > 0` (T4d),
  `--smooth-iters > 0` (T4c, and it defaults to `1`) and `--isotropic-cleanup`
  (T4d, and it defaults to *on*) — so a refining run must pass
  `--flip-passes 0 --smooth-iters 0 --no-isotropic-cleanup`. The editing-family
  question that blocked all of Phase 4 is **settled** (Phase 4, *The
  editing-family question — RESOLVED*): T4b, T4c and T4e are implementable
  against Tessera as it stands, and only **T4d** — coarsening, flips and
  isotropic cleanup — still waits on Tessera's G5b/G5c/G5d.
- **`/vertices/u1` in a `--steps 0` checkpoint is still present-but-meaningless**,
  though no longer because of a stub: `SurfaceState::updateSheetVector` is
  implemented (T2b), but nothing on the 0-timestep path *calls* it, so
  `initializeFields` leaves the field zero — a *defined* value and not a correct
  one. `Tessera::writeMesh` writes the whole vertex pack unconditionally and
  `compare_output.py` skips the field `state_model` does not select, so nothing
  depends on it yet. At `steps > 0` T2d's RHS refreshes it every stage.
- **`CheckpointIO::read` and `RestartReader::load` still throw** — T5b. Writing is
  validated; reading is not, in either direction.
- **`InitialCondition` implements the fast path only.** `applyShapeDeformation`,
  `applyPolarMode` and `seedInitialVorticity` throw (T5a), so any non-default
  `--initial-shape`, `--polar-amp` or `--initial-potential-strength` aborts rather
  than silently producing a sphere. `--mesh-kind latlon` likewise.
- **`Solver::filterCirculationField` and `ZModelSolver::computeRightHandSideSheet`
  still throw** — T5c. Both had their `mesh` parameter widened to `mesh_type&` at
  T2d, so T5c implements bodies only.
- **`BRSolverFMM`'s two methods still throw** — T3a.

What is **implemented but unexercised**, which is a different and quieter kind of
gap: **three of the six dt controls are inert in every configuration any test
runs** — `--max-sheet-dt-product`, `--dt-switch-time` and `--t-end` are each off
by default, so neither the `max_sheet_dt_product` branch nor either caller-side
clamp has ever executed under a test (T5e's table names the defaults and the
line numbers). **T5e and T5f close this**, and neither is stubbed: these are
bodies with no coverage, not `throw`s.

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

**Why first:** everything below assumes a compiling baseline.

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

**Exit criterion:** all five steps above succeed, and the README "Known Issues"
entry about the framework never having been compiled is deleted.

**Met.** All five succeeded; step 2 was vacuous (zero compile
errors). Steps 3-4 run via the new
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

### T2d — The RHS, volume projection, and the integrator — **DONE**

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
at ranks 1, 2, 3, 4, and 5. The per-step volume drift matches the **reference's
own** per-step drift to `1e-3` relative, with `1e-9` absolute as the blow-up cap.
*(Restated at T2d from "stays below `1e-12` relative", which was written a priori
and is an order of magnitude below what the discretization can deliver: the
Python's own drift, measured from the eleven gold files, is `5.19e-12` at step 1
and `5.17e-11` at step 10. Agreement with the reference is the stronger check —
it fails if Beatnik conserves volume better than the Python as well as worse —
and `1e-3` is a decade above the round-off floor of the drift ratio itself.)*

**Met.** The whole 36-launch gate is green — three members × {SERIAL, HIP} ×
ranks 1-6, run on 2026-08-14 inside a 2-node allocation, `[gate] PASS`, zero
`[FAIL]` lines. `Beatnik_Test_DirectSolve10Steps` reports **107/107 checks** on
rank 0 and 83/83 on every other rank, at every one of the twelve
backend × rank-count configurations, so the criterion's ranks 1-5 are a verified
subset of the gate's 1-6 on both backends. Inside that: the per-step `time`
matches its gold literal to `~2e-16` at all ten steps (the adaptive dt is
reproduced, not stubbed); all ten per-step `compare_output.py` invocations exit
0 at `--rtol 1e-10 --atol 1e-12`; the last-finite round trip matches the step-10
gold; the negative case against the step-0 gold exits exactly 1; and the mesh
stays 162/480/320 with halo depth 2 throughout.

**The volume-drift bound was restated, once, against a measurement.** The first
gate run failed in exactly one way — the per-step drift, ten checks per launch,
in all twelve configurations, growing *linearly in the step count* (`5.19e-12`
at step 1 to `5.17e-11` at step 10) and **identical on both backends at every
rank count**. Backend and rank independence rule out R2 and R8; linear growth
rules out round-off. That leaves RK3 truncation of a rate-only projection, which
the reference has too — so it was measured rather than asserted. Computing the
enclosed volume of the eleven gold `.npz` files offline with `enclosedVolume`'s
own convention gives the Python drift series, and it agrees with Beatnik's to
within one to two ulps of the drift ratio at every step. The `1e-12` bound was
therefore a priori and below the discretization's floor; it is replaced by
agreement with the reference at `1e-3` relative plus a `1e-9` absolute blow-up
cap — strictly stronger, since it also fails a run that conserves volume *better*
than the Python. Both series at 17 digits, and the ulp arithmetic that fixes the
tolerance, are in the progress log under *T2d — completion*.

**One launch-environment trap, recorded in the README.** The gate's
`BEATNIK_TEST_SCRATCH` must be on a **parallel** filesystem. Pointed at
tuolumne's node-local `/tmp`, ranks 1-4 pass and ranks 5-6 die inside
`H5FD__mpio_open` — which reads exactly like the multi-rank solver bug this test
exists to catch, and is not one.

**What was built.** Every method on this task's fill-in list has a body, plus four
that were not on it and had to be (recorded under `## T2d` in the log):
`SurfaceState::{faceSheetVector, maxSheetStrength, projectSheetTangent}` and the
non-const `Solver::mesh()`. `spack install` succeeds and installs all three gate
binaries per backend; the gate manifest carries the new line with its two
arguments.

**What is deliberately NOT built, and stays throwing:**
`Solver::filterCirculationField` and `ZModelSolver::computeRightHandSideSheet`
(T5c), `BRSolverFMM`'s bodies (T3a), all of adaptivity (T4a-T4e).
`VolumeProjection::projectToVolume` **is** implemented but is unreachable from
any configuration that exists today — every reference call site is inside a
refine or remesh branch — so it is written and unexercised. **T4a is NOT where
it first runs**, contrary to what this note originally said: the reference gates
it on a *repair* having run (`flips > 0 or smooth_iters > 0 or
isotropic_cleanup`), and T4a rejects all three. It first runs at T4c or T4d,
whichever lands first.

**The six diagnostics fields that report `NaN`/`+inf` are a deliberate gap, not a
bug:** the four AMR indicators need T4a and the two nonlocal-gap fields need
T4b's spatial query. Nothing in `Diagnostics` is checkpointed or compared, so the
gap is legibility only — and `NaN`/`+inf` are exactly what the Python's own
formatter prints when *it* has nothing, so a Beatnik progress line still diffs
against a Python one.

---

## Phase 3 — Regression test 3: `fmm` BR via Canopy

**Phase 3 may be deferred, and Phase 4 does not wait for it.** The direct BR
solver is implemented and validated (T2c, T2d) and is a complete far field, just
a slow one, so every task in Phases 4 and 5 can be implemented and tested with
`--br-approximation direct`. No task in Phase 4 depends on T3a except **T4e**,
and only optionally — Canopy's tree is one of the candidate vehicles for the
proximity query, not a requirement. Phase 3 remains the right next step for
*throughput*; it is not on the critical path for *capability*.

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

### The editing-family question — RESOLVED

**Decision: Beatnik never calls `Tessera::refine()`. Every topological edit, in
every configuration, goes through the Remesh family — `splitEdges()` today,
`collapseEdges()` / `flipEdges()` / `compact()` when Tessera lands them. One
family for the whole run, so Tessera's `EditFamily` guard is a backstop that
should never fire.**

Three alternatives were considered and **dropped**: holding two mesh objects and
transferring state between them, rebuilding the mesh at each phase change, and
asking Tessera upstream for the unified anisotropic-bisection level model (or its
narrower variant, an in-memory clone of a mesh carrying its vertex user pack).
Each existed to resolve a conflict that finding 1 shows does not arise. **Nothing
needs to be asked of Tessera to unblock T4a**, and the three findings below also
change what T4a is, not merely which call it makes.

#### Finding 1 — the two adaptivity modes are mutually exclusive in the reference

The premise the survey rested on — "Beatnik's default configuration runs both" —
is **false**. `run_adaptive_mesh_bubble.py:1424` reads

```python
if (not args.dynamic_remesh) and args.refine_every > 0 and step % args.refine_every == 0:
```

and the dynamic-remesh branch at `:1469-1471` is guarded by `if
args.dynamic_remesh`. The driver runs the indicator-driven refiner **or** the
metric remesher, never both, and the choice is a command-line constant for the
whole run. `src/Beatnik_AdaptiveMesh.hpp:16-22` already said so
("Only under `--no-dynamic-remesh` … the driver runs one or the other, never
both").

So even had Beatnik kept `Tessera::refine()` for the AMR path and `splitEdges()`
for the remesh path, no single mesh would ever see both families. The conflict is
not between the two adaptivity modes.

**Where a real interleave does survive**, and it is the only one: the refine
branch calls `improve_mesh_connectivity_by_edge_flips` and, under
`--isotropic-cleanup`, `mesh_quality.isotropic_cleanup`'s flip passes
(`run_adaptive_mesh_bubble.py:1440-1462`). Flips are Remesh-family. So a
`refine()`-based T4a could never grow its quality repairs, while a
`splitEdges()`-based one gains them the day Tessera's G5c lands.
`improve_mesh_quality_tangential` (`mesh_solver.py:1775-1800`) moves vertices and
changes no connectivity, so it belongs to no family and is legal either way.

#### Finding 2 — `splitEdges()` *is* `refine_marked_faces`; `Tessera::refine()` is not

The Python's refinement and Tessera's `splitEdges()` are the same algorithm.
`mesh.py::refine_marked_faces` creates midpoints on the three edges of every
**marked** face only, then retriangulates every face on the bit pattern of its
own split edges — `split_count` 1 → 2 children (median from the midpoint),
2 → 3 children, 3 → 4 children (red) — with **no cascade**: `existing_midpoint`
only ever finds midpoints minted by a marked face, so a neighbour's split does
not propagate. That is `splitEdges()`'s contract verbatim
(`../tessera/docs/design.md` → *Edge-addressed splitting*, the |S| table), for a
mask that is "every edge of every marked face".

`Tessera::refine()` is **not** the same algorithm, and the difference is not
cosmetic. Its conforming closure is **transient**: every call un-closes the whole
closure layer, refines the red layer, and rebuilds the closure from scratch
(`../tessera/docs/design.md` → *The closure layer*). The Python's green and blue
children are **permanent** — they are ordinary faces of the next round and can be
bisected again. Two consequences:

- **Face counts diverge from the Python from round 2 onward** under `refine()`,
  because round 2 does not see round 1's closure faces at all. Under
  `splitEdges()` they agree by construction whenever the mark sets agree.
- **Per-face state is churned by the closure.** Tessera copies a parent's face
  user fields to its closure children and, on un-close, "keeps the lowest-gid
  child's values and discards the rest". Beatnik's reference area and reference
  curvature (below) are exactly such state, and they take a lossy round trip
  through every closure/un-close cycle. Under `splitEdges()` nothing is ever
  un-closed and the inheritance is exact.

So the fidelity argument runs the *opposite* way to the survey's assumption:
`splitEdges()` is the higher-fidelity port, and `refine()` is the deviation.

#### Finding 3 — the quality unknown is measured, and `splitEdges()` wins

The survey named one discriminating measurement — *does a `splitEdges`-only
refinement hold triangle quality comparably to red-green over many rounds?* — and
recorded that Tessera published no such number. **It does now**, and both
measurements are in Tessera's gate, byte-identical at ranks 1-5 on both backends.

| | metric as published | worst, converted to \f$Q = R/2r\f$ |
| --- | --- | --- |
| `splitEdges()`, length-driven rounds (`../tessera/tests/test_split_edges.cpp:100-142,915-944`, case 8, 7 rounds; `../tessera/tests/test_split_edges_depth.cpp`, 10 rounds) | min inradius/circumradius per round `0.3780 0.3780 0.2815` **repeating with period 3**; min angle `33.203°` flat | **1.776** |
| `refine()` red-green, 16 rounds on a shrinking cap (`../tessera/tests/test_conforming_quality.cpp:36-78`, `../tessera/docs/design.md`) | max \f$Q\f$ by family: red `1.0278`, green `1.5672`, blue `2.2344` (`2.5254` before the geometric diagonal tie-break) | **2.234** |

The two are the same quantity: \f$Q = abc(a+b+c)/16A^2 = R/2r\f$, and
\f$r/R = 1/2Q\f$, so an equilateral triangle is \f$Q = 1\f$ / ratio `0.5`, and
`0.2815 → Q = 1.776`. **The `splitEdges()` path's worst element is better than
red-green's**, and its sequence does not drift downward — it is exactly periodic
with period 3, measured to ten rounds (R12).

Two structural reasons it should hold, so the number is not being read as luck:
Tessera's two-edge tie-break joins the midpoint of the **longer** split edge to
its opposite corner (`docs/design.md`, *The two-edge tie-break*), which is
Rivara's longest-edge rule; and Beatnik's masks are the two cases that rule is
good for — "all three edges of a marked face" (T4a, always the red split) and
"every edge longer than its target" (T4b, exactly case 8's workload).

**The honest scope of the evidence:** the bound belongs to the *mask*, not to
`splitEdges()`. Tessera's depth study (R12) drove ten rounds under a
longer-than-mean mask and got an exactly periodic trajectory — but the same
machinery under a shorter-than-mean or a length-blind mask degrades geometrically
to a zero minimum angle, with no floor at all. So the two structural reasons
above are not decoration: they *are* the guarantee. **Every edge Beatnik marks
must be marked because it exceeds a target length.** Carried as **R12**, which
states the constraint, the monitoring signals, and the mask transform that
restores the bound if a non-length term is ever needed.

#### What the decision buys, beyond unblocking

- **`_balance_red_green_refinement` becomes three lines of edge-mask logic.** The
  Python's promotion rule (`mesh_solver.py:1543-1580`) is "an unmarked face with
  ≥ 2 split edges is promoted to a full red split, to fixpoint", plus a
  quality test on the one-edge case. On an edge mask that is *mark the third edge
  of any face with two marked edges*, iterated to a global fixpoint — no face
  marks, no closure, no 2:1 machinery.
- **`projected_red_green_face_count` becomes one global sum.** Under an edge
  mask, the post-split face count is exactly \f$\sum_f (|S_f| + 1)\f$ — local
  arithmetic plus one `globalSum`, evaluated before any edit.
- **R4's quadratic greedy loop is no longer needed.** Because the projection is
  a cheap closed form rather than a fixpoint, `--max-faces` can be enforced by a
  **global threshold search on the score** (bisect the score threshold until the
  projected count fits) instead of the reference's sequential accept loop. Same
  intent, parallel, deterministic, and \f$O(\log)\f$ projections instead of
  \f$O(N_{\text{seeds}})\f$ closures. R4's warning that a capped run will not
  match the Python face-for-face still stands.
- **`Comm::reconcileRefinementMarks` stays deleted-in-effect.** `splitEdges()`
  agrees the mask across ranks itself, the **edge owner** deciding
  (`../tessera/src/Tessera_EdgeSplit.hpp`, *Distributed structure* step 1). What
  Beatnik must reconcile is only its own fixpoint's termination test, which is one
  `MPI_Allreduce(MPI_LOR)`.

#### Deliberate deviations this decision introduces

- **`Tessera::refine()` is not used, so the mesh has no `Level` and no 2:1
  balance.** Nothing in the z-model needs either: the surface operators need
  conformity, which `splitEdges()` guarantees on exit, and the Python has no level
  model to reproduce.
- **The two-edge diagonal differs from the Python's.** Tessera chooses the
  shorter diagonal geometrically; `mesh.py::refine_marked_faces` uses a fixed
  rotation-dependent diagonal. Beatnik keeps Tessera's — it is the better element
  and it is what Tessera's quality measurement was taken with. Consequence: a
  Beatnik and a Python refinement of the same mark set have the same V/E/F but not
  the same connectivity wherever a face had exactly two split edges. Risk **R13**.
- **The refine branch ships without its quality repairs** until Tessera's G5c
  lands. `--flip-passes > 0` and `--isotropic-cleanup` are **rejected at setup**
  in the refine configuration, by name and task ID, never silently skipped — the
  rule T2d established in `Solver::requireSupportedConfiguration`.

#### Conventions for the whole of Phase 4

| Convention | Choice |
| --- | --- |
| Editing family | Remesh, always. `SurfaceMesh::refine` is **deleted**, not left unused. |
| Edit entry point | `SurfaceMesh::splitEdges( const std::vector<char>& edgeMask )`, mask sized `ownedEdgeCount()`, `1` = bisect. Matches `Tessera::splitEdges`'s own convention rather than inventing a second. |
| Mark representation | An **edge** mask, everywhere: the only thing passed to Tessera, and the only thing R12 constrains. Face-level indicators are translated to edges at the point of use. **T4a correction:** the *verdict* is nonetheless carried per face, in a `RefineMark` face user field, because route (a) has to halo-exchange it and `haloExchange()` addresses fields by compile-time member index — a mark outside the mesh cannot cross a rank boundary. So "no face mask is stored" is true of Beatnik-side storage and false of the mesh field; the edge mask is derived from it before every use. |
| Per-face carried state | Tessera **face user fields**, named from outside via `Beatnik::FaceFieldId` — the same pattern M1 established for vertices, for the same reason (a `Kokkos::View` outside the mesh is silently dropped by a split and silently stale after `migrate`). **T4a correction: three slots, not two** — `FaceFields<Real, Real, Real>` = `{ReferenceArea, ReferenceCurvature, RefineMark}`. See the row above for why the third is there, and R14 for what it does to the checkpoint. |
| Fixpoint loops | Terminate on a global `MPI_Allreduce(MPI_LOR)`, never a rank-local test, and carry a hard round cap that **throws** when hit rather than proceeding with a partial mark set. |
| Caps and budgets | Global quantities. A per-rank cap is a different algorithm and a rank-count-dependent one. |
| Unsupported configuration | Rejected in `Solver::requireSupportedConfiguration` before the first step, naming the method and the task ID. Never skipped. |

---

### T4a — Indicator-driven refinement through `splitEdges()` — **DONE**

**Met.** Gate run 4 (`f3SRwVuXai8X`) is green: **all 48 launches**, four
`regression` members x {SERIAL, HIP} x ranks 1, 2, 3, 4, 5, 6, ending
`[gate] PASS (label=regression)` with zero failures.
`Beatnik_Test_RefineSplitEdges` passes **86/86 checks in each of its twelve
configurations** — `--no-dynamic-remesh --refine-every 5 --flip-passes 0
--smooth-iters 0 --no-isotropic-cleanup --max-faces 1400` at `--area-threshold
1e-4 --curvature-change-threshold 1e-4` (the threshold deviation is the one
recorded below and in the test header, because at the defaults the criterion's
command refines nothing). Every pass satisfies \f$V-E+F=2\f$, every owned edge
names exactly two incident faces with both locally resident after every edit,
and the global face count equals `projectedFaceCount`'s prediction **exactly**
at every pass and every configuration. The failure direction is confirmed:
`--flip-passes 2` exits from `requireSupportedConfiguration` naming
`MeshQuality::improveConnectivityByFlips` and T4d, not from a Tessera
`EditFamily` throw.

**The measured floor is `0.119`.** The whole-run global minimum \f$r/R\f$ is
`0.119876446958` at ranks 1-4 and `0.119867784111` at ranks 5-6, identical on
both backends. (The `0.119867826031` this document previously called the ranks
5-6 run minimum is pass 3's minimum; the run minimum is pass 4's, four ulp-scale
digits lower, because the mesh keeps evolving for five more steps after the last
mark. The floor is unaffected and the ranks 1-4 figure is unchanged.)

**The two R12 series**, per pass and per the round index, matching what was
measured before to twelve significant digits:

- global minimum \f$r/R\f$: `0.304119905237` → `0.123117984672` →
  `0.119867826031` (ranks 5-6; `0.119877418574` at ranks 1-4) →
  `0.119867784111` (ranks 5-6; `0.119876446958` at ranks 1-4);
- global count of faces below \f$r/R\f$ `0.25`: `0` → `4` → `96` (ranks 5-6;
  `94` at ranks 1-4) → unchanged at pass 4.

Both decline monotonically — R12's *shape-problem* signature, and it is the
**reference algorithm's** behaviour, not Beatnik's: the Python's own series is
reproduced to twelve digits on the two uncapped passes. No mitigation was
applied; see R12. Gate run 3's hang in `Beatnik_Test_InitialConditions_MPI_SERIAL`
at 6 ranks did not recur — that member was green at every rank count in run 4,
so the hang was a transient and not a finding.

**Depends on:** none — T2d is `**DONE**` and nothing else is required. In
particular this task needs **no** new Tessera capability.

**Fill in:**
- `Beatnik_MeshInterface.hpp`: add `face_fields = Tessera::FaceFields<Real, Real>`
  and the `FaceFieldId` enum alongside `VertexFieldId`; **delete**
  `SurfaceMesh::refine` (`src/Beatnik_MeshInterface.hpp:1258`) and the
  editing-family warning block above it (`:1184-1195`, whose "Beatnik's default
  configuration wants both" is the same false premise corrected in *Finding 1*);
  change `SurfaceMesh::splitEdges` (`:1296`) from
  `const EdgeListView&` to `const std::vector<char>& edgeMask`; add
  `faceEdges()` (see **Do** step 2).
- `Beatnik_AdaptiveMesh.hpp`: all of it, restructured onto an edge mask —
  `areaChangeIndicator`, `curvatureChangeIndicator`,
  `curvatureResolutionIndicator`, `markFaces`, `limitMarkedFraction`,
  `expandMarkedRings`, `balanceRedGreen`, `projectedFaceCount`, `selectMarks`,
  `refine`, `resetReferenceState`.
- `Beatnik_Communication.hpp`: **delete** `reconcileRefinementMarks`
  (`src/Beatnik_Communication.hpp:351`) — M1 already recorded that Tessera leaves
  it nothing to do, and this task confirms it for `splitEdges()` too.
- `Beatnik_Solver.hpp`: `requireSupportedConfiguration` — drop the
  `refine_every > 0` rejection, add rejections for `--flip-passes > 0`
  (T4d), `--isotropic-cleanup` (T4d) **and `--smooth-iters > 0`** under
  `--no-dynamic-remesh`; call the refiner from `advanceOneStep` on the
  `--refine-every` cadence.

  **The `--smooth-iters` rejection is not optional and was missing from this
  list.** `FilterParams::smooth_iters` defaults to `1`
  (`src/Beatnik_Params.hpp:588`) and the reference's refine branch calls the
  tangential pass with it *unconditionally* once anything is marked
  (`run_adaptive_mesh_bubble.py:1446-1450`), so without it a default
  `--refine-every N` run reaches `MeshQuality::improveQualityTangential` — a
  throwing **T4c** method with no task ID on its message — several steps in.
  It names the method and T4c, like the other three. The existing
  `--redistribute-every > 0` rejection stays (also T4c).

← *Python:* `mesh.py::{area_change_indicator (215-218),
curvature_change_indicator (221-224), face_curvature_indicator (172-174),
cotangent_vertex_curvature (150-169), curvature_resolution_indicator (203-212),
refine_marked_faces, projected_red_green_face_count (251-271)}`,
`mesh_solver.py::{refine_potential_mesh_state (1374-1431),
_quality_preserving_refinement_marks (1454-1512),
_expand_marked_face_rings (1515-1530), _limit_marked_fraction (1434-1451),
_balance_red_green_refinement (1543-1580), _single_green_split_quality
(1606-1623), _drop_faces_below_min_edge (1626-1635)}`
← *Tessera:* `../tessera/src/Tessera_EdgeSplit.hpp` (the whole header comment),
`../tessera/docs/design.md` → *Edge-addressed splitting*

**Callers to update: none.** Nothing in `src/`, `examples/` or `tests/` calls
`SurfaceMesh::refine`, `SurfaceMesh::splitEdges`, `reconcileRefinementMarks` or
any `AdaptiveMesh` method today; the only occurrences are the declarations
themselves and prose references in `Beatnik_MeshQuality.hpp:157`,
`Beatnik_Restart.hpp:153` and `Beatnik_SurfaceState.hpp:165`, which must be
re-pointed but compile either way. So every signature change here is free, and it
will not be after T4b.

**Do:**

1. **Correct the stale header statements first**, so the headers stop
   contradicting the code that is about to be written. Two of them:
   - `Beatnik_AdaptiveMesh.hpp:410-412` claims the refined state is built with
     `reference_face_area=None` and the reference is "re-based to the
     post-refinement geometry". It is not — see step 4 — and a session
     implementing from that comment gets the indicator wrong in a way no
     structural test catches. The file header's list at `:33-43` is right.
   - `Beatnik_MeshInterface.hpp:1184-1195` says "Beatnik's default configuration
     wants both" editing families. It does not; see *Finding 1* above.

2. **Get face → edge.** Tessera has no face→edge CSR: it has
   `mesh.vertexEdges()`, and the edge AoSoA carries `EdgeField::Verts` and
   `EdgeField::Faces`. Beatnik already resolves the second of those into local
   indices in `SurfaceMesh::edgeAdjacency()` (M1's call table). Build the
   inverse — per owned face, its three local edge indices — and cache it on
   `generation()` exactly as `edgeAdjacency()` and `vertexOneRing()` do. This is
   the one new adapter accessor the task needs and it belongs with the others,
   not inline in the AMR code.

3. **Translate marks to edges, with the owner able to see them.** An owned edge
   may be incident on a face owned by another rank, and `splitEdges()` takes the
   verdict from the **edge owner**. So a face-level decision must reach that rank.
   Two routes, and the task must choose one and say which in the log: (a) write
   the per-face score into a face user field, `haloExchange()`, and let each rank
   evaluate its owned edges from locally-resident faces; or (b) evaluate the
   indicators on ghost faces too, which is legal because all three are local
   geometric quantities over a face's own vertices and their one-rings, and the
   halo is 2-deep. Route (a) is preferred: it is one exchange, it does not
   duplicate arithmetic, and it does not depend on the one-ring of a ghost face's
   corner being complete. **Verify, do not assume, that both incident faces of
   every owned edge are locally resident at halo depth 2** — T1b's test already
   asserts every edge has exactly two incident faces at ranks 1-6, so the check is
   a re-assertion, not new work.

4. **Get the reference-area transfer right; it is not inheritance.**
   `mesh.py::refine_marked_faces` gives each child of a subdivided face
   `parent_ref * child_area / parent_area`, keeps an unsplit face's reference
   unchanged, and **resets** reference *curvature* to the child's current value
   for subdivided faces only. Tessera inherits face user fields **verbatim** and
   `RefinePolicy` covers vertex fields only
   (`../tessera/src/Tessera_RefinePolicy.hpp` — the two hooks are
   `interpolatePosition` and `interpolateVertexField`), so neither rule is
   expressible through the policy. Do it with two local passes around the call:
   - **before** `splitEdges()`, replace the stored reference area by the ratio
     \f$\sigma_f = A^{\text{ref}}_f / A_f\f$;
   - **after** it, restore \f$A^{\text{ref}}_f = \sigma_f A_f\f$.

   This reproduces the Python **exactly**, for both cases at once: children of one
   parent share \f$\sigma\f$, so \f$A^{\text{ref}}_{\text{child}} = (A^{\text{ref}}_p/A_p)
   A_{\text{child}}\f$; and an unsplit face's corners are untouched by a split, so
   its area — and therefore its reference — is unchanged. The one departure is
   that the Python normalizes by `sum(child_areas)` rather than the parent's own
   area; the two differ only in floating-point association.

   Reference *curvature* needs a "was I subdivided" discriminator. Use the fact
   that a face with `|S| = 0` **keeps its gid** while a subdivided parent's gid is
   retired (`../tessera/src/Tessera_EdgeSplit.hpp`, the |S| table and the child-gid
   exscan): snapshot the owned face gids before the call and reset the reference
   curvature of exactly those faces whose gid is not in the snapshot. State in the
   log what the measured new-face count was, so a silently-empty snapshot is not
   mistaken for "nothing refined".

5. **Seed, cap, expand, balance — in that order**, matching
   `_quality_preserving_refinement_marks`. The seed and the score are per face
   (`markFaces`'s existing doc comment is correct and stays); everything after is
   per edge:
   - `limitMarkedFraction` keeps the `max(1, ceil(f·N_f))` highest-scoring seeds.
     The ranking is **global**; implement it as a threshold search on the score
     with one `globalSum` of the count per probe, not a distributed sort.
   - `expandMarkedRings` is `--refine-neighbor-rings` breadth-first steps over
     face adjacency. Use `faceAdjacency()`'s `nbrGid`/`nbrOwner` half, which M1
     records as the one valid for a *topological* consumer; do **not** use the
     local-index CSR unless `numNonResident == 0`.
   - `balanceRedGreen` is the fixpoint of *mark the third edge of any face with
     exactly two marked edges*, plus the one-edge quality promotion against
     \f$\max(\tau_{\text{floor}}, f_q q_{\text{parent}})\f$. Terminate on
     `MPI_Allreduce(MPI_LOR)`; cap the rounds and **throw** on the cap.
   - `projectedFaceCount` is \f$\sum_f (|S_f|+1)\f$ over owned faces plus one
     `globalSum`. `--max-faces` is enforced by bisecting the score threshold until
     the projection fits, **not** by the reference's greedy accept loop (R4).

6. **Re-base the reference, and implement the volume-projection gate as the
   reference writes it — which under this task's configuration means the
   projection does NOT run.** The Python gates `project_state_to_volume` on a
   repair having *actually happened*, not on the refinement having happened:
   `if not args.no_preserve_volume and (flips > 0 or args.smooth_iters > 0 or
   args.isotropic_cleanup)` (`run_adaptive_mesh_bubble.py:1465-1468`). Under the
   only configuration this build accepts — flips, smoothing and cleanup all
   rejected at setup — all three are false. So implement the same gate, in full
   rather than folded to `false`, and **do not** call the projection
   unconditionally to "exercise it": that would be a deviation from the
   reference dressed as coverage. `VolumeProjection::projectToVolume` therefore
   remains **unexercised** and first executes at T4c/T4d, whichever lands first.
   (An earlier revision of this entry, and T2d's `Affects:` note, both said T4a
   was where it first runs. Both were wrong for this reason.)

**Exit criterion:** a `regression`-tier test, registered per backend with the
`_<BACKEND>` suffix the gate selects on — **this grows the ship gate to four
members / 48 launches, the user has confirmed this is allowed**
(CLAUDE.md "Minimum test set"). It must show, at the gate's ranks 1-6 on SERIAL
and HIP:

- a `--no-dynamic-remesh --refine-every 5 --flip-passes 0 --smooth-iters 0
  --no-isotropic-cleanup` run completes 20 steps without aborting, and the
  global V/E/F satisfy Euler \f$V-E+F=2\f$ and conformity after every
  refinement pass. **`--no-isotropic-cleanup` is part of the command**:
  `CleanupParams::enabled` defaults to `true` (`src/Beatnik_Params.hpp:376-378`),
  so without it the run is rejected by the very rejection this task adds;
- the **global** face count after each pass equals this rank's own
  `projectedFaceCount` prediction for that pass **exactly** — the check that
  catches a mask that was reconciled differently than it was projected, and the
  one that fails loudly if the balance fixpoint did not converge;
- the face count agrees with a Python run of the same configuration where
  `--max-faces` is not binding, and where it binds only the non-refinement
  fields are compared and the divergence is recorded in the log (R4);
- the minimum triangle radius ratio over the whole run stays above a floor
  **measured on the first run and recorded here**, not guessed, and **not** taken
  from Tessera's `kMinRadiusRatioFloor` (R12);
- the per-pass diagnostics log **both** R12 signals against the round index: the
  global minimum \f$r/R\f$ **and** the global count of faces below \f$r/R\f$
  `0.25`;
- **the failure direction:** a run with `--refine-every 5 --flip-passes 2` exits
  non-zero from `requireSupportedConfiguration` with a message naming
  `MeshQuality::improveConnectivityByFlips` and **T4d** — not from a Tessera
  `EditFamily` throw, and not by silently running without flips.

---

### T4b — Metric-driven dynamic remeshing: the sizing field and the split pass — **DONE**

**Met.** Gate run 1 for this task (`f3SpT4MZbqMh`) is green at all **60
launches** — five `regression` members × {SERIAL, HIP} × ranks 1-6, ending
`[gate] PASS (label=regression)` with zero failures. The four pre-existing
members are unchanged (`Beatnik_Test_RefineSplitEdges` still 86/86, which is
also the check that T4a's shape literals did not move when its `r/R` kernel was
single-sourced into `SurfaceOperators::radiusRatioStats`).
`Beatnik_Test_DynamicRemeshSplit` passes **377/377 checks in each of its twelve
configurations**, running
`--dynamic-remesh --remesh-every 1 --remesh-collapse-factor 0
--remesh-max-collapses 0 --remesh-smooth-iters 0 --remesh-flip-min-gain 1e12
--no-isotropic-cleanup` for 20 steps at `--remesh-sagitta-tolerance 0.002
--remesh-h-max 0.06` (the two-knob deviation recorded below, forced by the same
R15 trap that moved T4a's thresholds — at the defaults the sizing field is
pinned at its upper clamp and the pass is all-or-nothing).

What was verified, at every one of the twenty passes: `V - E + F = 2` globally;
every owned edge naming exactly two incident faces with both locally resident;
`splits == split_candidates` with the per-pass cap never binding, i.e. **every**
edge longer than `split_factor · max(target, h_min)` entering the mask; and the
volume drift held at zero to `1e-14` — which is where
`VolumeProjection::projectToVolume` **first executes**, T4b being the first task
whose branch reaches it. The face and vertex counts reproduce the reference's
`320 → 560 → 800 → 1040 → 1160 → 1400` **exactly at every step**, not only at
pass 1, and the two R12 signals reproduce the reference's to all twelve
significant digits at every step. Every printed number is **byte-identical
across {SERIAL, HIP} × ranks {1, 3, 6}** — the sizing field, the eight gradation
sweeps and the mask are rank-count invariant, which is what the new per-vertex
halo exchange buys.

**R12's answer for this mask is the HEALTHY signature, and that is the headline.**
The global minimum \f$r/R\f$ runs
`0.4865 → 0.3739 → 0.2485 → 0.2485 → 0.2815` and then *stays* at `0.2815`; the
population below `0.25` runs `0 → 0 → 120 → 120 → 0` and stays at zero. It dips
and recovers, and the last third of the run sets no new low — exactly what R12
predicts for a purely length-driven mask, and exactly what T4a's mask did not
do. The measured floor is `0.248`.

**The failure direction is confirmed five ways**: `--remesh-proximity` exits
naming `DynamicRemesh::nonlocalFaceCentroidDistance` and T4e;
`--remesh-collapse-factor 0.45`, `--remesh-smooth-iters 1`,
`--remesh-flip-min-gain 1e-3` and `--isotropic-cleanup` each exit naming their
method and T4d. None is a Tessera `EditFamily` throw and none runs silently.

**T4d's question, answered from the run:** the missing coarsening does **not**
bite within 20 steps. The minimum triangle quality falls `0.977 → 0.625` over
the five real passes and then *recovers* to `0.673`, never approaching
`--remesh-min-quality` `0.18`. There is no crossing step to report.

**Depends on:** T4a (the edge-mask plumbing, the face-user-field pack, and the
face→edge accessor are all T4a's, and this task must not re-invent them).

The default configuration. `dynamic_remesh.py` is three thirds — split, collapse,
flip — and **only the split third is expressible against Tessera today** (G5b,
G5c, G5d are open; see M1's gap section). This task is the split third and the
sizing field that drives it; T4d is the rest. Splitting them this way is
deliberate: the sizing field, the gradation smoothing and the pass structure are
the bulk of `dynamic_remesh.py` and none of it is blocked.

**Fill in:**
- `Beatnik_DynamicRemesh.hpp` — the target-length (sizing) field from the
  sagitta tolerance \f$h = \sqrt{8\,\text{tol}/\kappa}\f$ clamped to
  `[h_min, h_max]`, the gradation smoothing (`target_gradation_factor` 1.35,
  `target_gradation_iterations` 8), the split selection
  (`edge length > split_factor · target`), the multi-pass loop
  (**`passes` = 1**), the per-pass split cap, and the diagnostics.

  **The pass count is 1, not 2.** `2` is `DynamicRemeshParams.passes`'s
  dataclass default (`dynamic_remesh.py:31`), which the driver overrides with
  `--remesh-passes`, default **1** (`run_adaptive_mesh_bubble.py:420`).
  `RemeshParams::passes` (`src/Beatnik_Params.hpp:265`) already said 1; this
  entry was the thing that was wrong.
- `Beatnik_Solver.hpp` — the rejections below, and the call to the remesher from
  `advanceOneStep` on the `--remesh-every` cadence
  (`src/Beatnik_Solver.hpp:149`, the branch sketched at `:42-48`). This was
  missing from the list; the task is not expressible without it, since
  `--dynamic-remesh` is rejected outright until it is written.

← *Python:* `dynamic_remesh.py`, everything except `collapse_short_edges`,
`flip_edges_for_quality`, `tangential_smooth_vertices` and the proximity paths.

**Hard constraint on the mask (R12):** an edge may enter the split mask **only**
because its length exceeds `split_factor · target`. That is the one family
Tessera measured as shape-bounded, and it is bounded because it is a coarse
Rivara longest-edge rule — self-correcting. Do **not** union in edges for any
other reason (a curvature term, a vorticity term, a region tag); those edges
carry no bound, and a length-blind mask drives the minimum angle to zero within
ten rounds. If a later requirement genuinely needs a non-length term, the mask
must first be made longest-edge-consistent — promote each marked edge to the
longest edge of its incident faces, to fixpoint — before `splitEdges()` sees it,
and the new rule must be run as a fifth family in
`../tessera/tests/test_split_edges_depth.cpp` before it is committed to (the
three edit sites are named in R12).

**Explicitly out of scope, and why:**
- Collapse and flip → **T4d**, blocked upstream.
- **The nonlocal proximity query → T4e.** This document called it "the hardest
  single item in the port" and treated it as blocking. It is not: both switches
  that reach it, `DynamicRemeshParams.use_proximity` and `.surgical_proximity`,
  **default to `False`** (`dynamic_remesh.py:33,41`), so it is off on the default
  path. It is isolated into its own late task rather than allowed to hold up the
  remesher.

**Additional information needed before T4d can be tasked, and answered by this
task:** whether a split-only remesher holds triangle quality at all over a
roll-up, and for how many steps — i.e. how badly the missing coarsening bites,
measured rather than argued. Record the step count at which the minimum quality
crosses `--remesh-min-quality`.

**ANSWERED: it does not cross, within 20 steps, and the trace is not even
monotone.** The global minimum triangle quality \f$4\sqrt3 A/\sum\ell^2\f$ falls
`0.977 → 0.769 → 0.625 → 0.625 → 0.673` across the five real passes and then
holds at `0.673`, drifting in its sixth digit for the remaining fifteen steps.
`--remesh-min-quality` is `0.18`. So on this problem the collapse third is not
what holds quality up over this horizon — the split mask's own
longest-edge-consistency is — and T4d's case rests on longer runs and on the
tighter roll-up, not on this one.

**Exit criterion — RESTATED, because the document's original conflicts with
itself and the conflict is structural.** The original asked for a
`--dynamic-remesh` run that completes 20 steps *and* for "a configuration that
would need a collapse" to exit non-zero naming T4d. **Every `--dynamic-remesh`
run needs a collapse**: `dynamic_remesh_arrays` (`dynamic_remesh.py:141-172`)
calls `collapse_short_edges` unconditionally on every pass, and
`flip_edges_for_quality` + `tangential_smooth_vertices` whenever
`splits > 0 or collapses > 0 or min_quality < 0.18`. There is no
`--dynamic-remesh-split-only` switch and none was added. The two halves are
therefore satisfiable only by making the *configuration* the discriminator: a
run is accepted when the reference's own knobs make the three unimplemented
thirds no-ops, and rejected otherwise. That is what the restatement below says.

A `regression`-tier test registered per backend, showing at the gate's ranks 1-6
on SERIAL and HIP:

- a `--dynamic-remesh --remesh-collapse-factor 0 --remesh-max-collapses 0
  --remesh-smooth-iters 0 --remesh-flip-min-gain 1e12 --no-isotropic-cleanup`
  run completes 20 steps without aborting, and the global V/E/F satisfy Euler
  \f$V-E+F=2\f$ and conformity after every pass. **Confirm the split pass
  actually fires at the default sizing parameters and, if it does not, lower the
  sagitta tolerance until the scheduled passes are real and record the
  deviation** — R15's trap, exactly as it bit T4a's thresholds;
- every edge longer than `split_factor · target` at the end of a pass is either
  split in the next pass or blocked by `h_min` — asserted, not inspected;
- the R12 pair — the global minimum \f$r/R\f$ and the global count of faces
  below \f$r/R\f$ `0.25` — is logged per pass against the round index, and shows
  the **healthy** signature (the minimum cycles and sets no new low in the last
  third of the run; the count returns to zero between dips) rather than a
  monotone per-round decline. If it declines monotonically, say so and record
  it; do not apply an R12 mitigation, which is a separate task with its own gold
  set;
- against an offline split-only Python reference: pass-1 face and vertex counts
  agree **exactly**, and the global minimum \f$r/R\f$ and sub-`0.25` count agree
  at every pass to the precision T4a achieved (twelve significant digits). Per
  R13, do not expect the *counts* to agree past pass 1;
- the per-step enclosed-volume drift is **measured and written into this
  document as 17-digit literals** (T2d's convention — its `kGoldVolumeDrift` is
  for a fixed mesh and must not be reused), with `1e-9` absolute retained as the
  blow-up cap;
- **the failure direction:** `--remesh-proximity` exits non-zero from
  `requireSupportedConfiguration` naming T4e, and each of
  `--remesh-collapse-factor 0.45` (the default), `--remesh-smooth-iters 1`,
  `--remesh-flip-min-gain 1e-3` and `--isotropic-cleanup` exits non-zero naming
  its method and **T4d** — not from a Tessera `EditFamily` throw, and not by
  silently running without them.

**The measured volume drift, as 17-digit literals.** It is **exactly zero at
every step**, on both backends at every rank count, and so is the reference's —
because under `--dynamic-remesh` the driver projects the state back to the
initial volume after *every* remesh step (`run_adaptive_mesh_bubble.py:1513-1516`,
gated on the remesh having run rather than on it having changed anything). The
reference's series carries one non-zero entry, step 17's
`2.22044604925031308e-16`, which is one ulp of the ratio. A series of zeros is
only a meaningful assertion if the bound is tight enough to fail a build that
skipped the projection, so the test's bound is `1.0e-14`: T2d measured the
drift of a fixed-connectivity run *without* the projection as
`5.1697091052460564e-11` by step 10, growing linearly, which is three decades
above it. The `1e-9` absolute cap is kept as the coarser blow-up detector the
criterion names.

---

### T4c — Tangential relaxation — **NOT STARTED**

**Depends on:** none. Independent of T4a and T4b, and deliberately so: it changes
**no connectivity**, so it belongs to no editing family and could be implemented
before either. Sequenced here only because nothing calls it until T4a does.

**Fill in:**
- `Beatnik_MeshQuality.hpp::improveQualityTangential` **and
  `::tangentialRelaxation`** (`src/Beatnik_MeshQuality.hpp:138`, `:211`). They are
  the same operator reached from different callers — the header at `:141` already
  says so — and `tangentialRelaxation` was assigned to no task at all, the hole
  T4b found for `tangentialSmooth`. One private kernel, both entry points
  delegating to it. `tangentialRelaxation` therefore ships **implemented but
  unexercised**, its only caller `isotropicCleanup` being T4d: the same
  standing as `VolumeProjection::projectToVolume` between T2d and T4b.
- `Beatnik_Solver.hpp::requireSupportedConfiguration` — drop **two** rejections,
  not one: `redistribute_every > 0` (`src/Beatnik_Solver.hpp:829-834`) and
  `--smooth-iters > 0` under `--refine-every` (`:719-723`), which T4a added
  naming this task by ID. Every other rejection there stays.
- `Beatnik_Solver.hpp::advanceOneStep` — **the two call sites, neither of which
  exists in code today.** Without them the task is not expressible, and the
  original fill-in list named neither: the post-refine tangential pass
  (`src/Beatnik_Solver.hpp:462-478`, where the reference's three repairs are
  currently a comment and `flips` is a literal `0`) and the whole
  `--redistribute-every` branch, transcribed in the control-flow comment at
  `:56-57` and absent below it.

← *Python:* `mesh_solver.py::improve_mesh_quality_tangential` (1775-1832),
`run_adaptive_mesh_bubble.py::main` (1446-1451, the post-refine call; 1557-1565,
the redistribute branch)

The displacement is the neighbour-centroid Laplacian **projected onto the local
tangent plane**, so the interface geometry is not normal-smoothed. Both operators
exist and are validated: `SurfaceOperators::graphLaplacianVector` and
`::projectTangent` (T2b). The `reset_reference` argument is real — the driver
passes `reset_reference = (smooth_iters == 0)` to the flips and `False` here
(`run_adaptive_mesh_bubble.py:1440-1451`), so the re-basing is *not*
unconditional.

**`--redistribute-every` is not load balancing and this task is not blocked on
T5d.** The reference's branch is the tangential pass plus
`VolumeProjection::projectToVolume`, gated on `smooth_iters > 0` rather than on
`--no-preserve-volume` alone (`:1564-1565`), and nothing else — no
repartitioning, the Python being serial. `Comm::redistribute` stays T5d's.

**Dropping the `--smooth-iters` rejection is where the volume projection first
runs on the REFINE path.** The refine branch's gate is
`flips > 0 || smooth_iters > 0 || isotropic_cleanup`
(`src/Beatnik_Solver.hpp:475-479`), transcribed in full by T4a precisely so that
deleting a rejection turns it on; `--smooth-iters` defaults to `1`. T4b already
executed the projection on the remesh path, so what is new here is the refine
path only.

**Exit criterion:** a `unit`-tier test on the default icosphere — the tier, so
the ship gate stays at five members / 60 launches — showing:

- **tangency, stated so it is satisfiable.** At `iterations = 1`,
  \f$\max_v |\Delta x_v\cdot\hat n_v| \le 10^{-13}\max_v|\Delta x_v|\f$; the
  reference measures `2.05e-17`. Two things forbid the per-vertex ratio the
  criterion originally asked for: **42 of the 162 vertices move by exactly
  zero** (icosahedral symmetry makes their neighbour-centroid offset exactly
  radial), so \f$|\Delta x\cdot\hat n|/|\Delta x|\f$ is `0/0` on a quarter of
  the mesh; and the identity is **per sweep, not cumulative** — at
  `iterations = 3` the accumulated \f$\max|\Delta x\cdot\hat n|\f$ against the
  pre-pass normals is `2.05e-6`, because each sweep re-projects against the
  geometry the previous one moved. At `iterations > 1`, assert tangency **per
  sweep**;
- **the mean triangle quality rises and V/E/F are unchanged.** Reference mean
  \f$4\sqrt3 A/\sum\ell^2\f$: `0.98852866623246283` →
  `0.99027290116169975` at `iterations = 1` and → `0.99244442526171672` at
  `iterations = 3`. The **minimum** quality *decreases* slightly
  (`0.97727413140883002` → `0.97721067116745464`), which is a property of the
  operator and not of the port — report it, do not assert it. Displacement
  scale: \f$\max|\Delta x|\f$ `0.00079728040863246894` at `iterations = 1`,
  `0.0021023925151415252` at `iterations = 3`, against a shortest edge of
  `0.068976121063816842`, so the pass is not a no-op on this mesh and R15's trap
  does not apply;
- **the failure direction, with the separation measured rather than asserted:** a
  deliberately un-projected displacement changes the enclosed volume by
  `1.606e-2` relative against the projected pass's `3.898e-6`, a factor of
  ~4100. A tangency check that cannot see that factor has no teeth;
- **rank-count invariance of every scalar above**, with the tier run once at
  `BEATNIK_UNIT_RANKS=4` (`scripts/tuolumne/unit_tests.flux:99`). The pass
  recomputes normals each sweep and so needs a position halo exchange *between*
  sweeps; at one rank that precondition is unobservable, and getting it wrong
  moves a seam with the rank count rather than failing.

**Session prompt:**

````
Read `tasks/framework.md` and implement `T4c` — tangential relaxation: the
neighbour-centroid Laplacian displacement projected onto the local tangent
plane, plus the two reference call sites that reach it.

Read these before starting, and skip the rest of the document:
- `tasks/framework.md`: the `T4c` task entry, the Conventions table under
  "Conventions established", the "Conventions for the whole of Phase 4" table,
  and risks **R8** (a multi-sweep pass and the ghost depth), **R9** (owned-only
  iteration) and **R15** (a pass that changes nothing is indistinguishable from
  a correct one).
- `tasks/framework-progress-log.md`: `## T2b` — both operators this task
  composes are implemented and validated there, and its `Affects: T4c` bullet
  carries the umbrella-vector scalars the test cross-checks against; `## T2d` —
  decision 9 (why the post-step passes throw rather than skip) and decision 4
  (owned range out, whole local range for anything a face loop scatters into);
  `## T4a` — route (a), the `EdgeField::Faces`-is-partial-after-any-edit bug, and
  that `resetReferenceState` is also what *initializes* the face pack; `## T4b` —
  why the reference's smoothing sweep is Jacobi and what reading it as
  Gauss-Seidel would have cost.
- `src/Beatnik_MeshQuality.hpp:110-220` — `tangentialRelaxation` (`:138`) and
  `improveQualityTangential` (`:211`), both throwing.
- `src/Beatnik_Solver.hpp:33-60` (the control flow transcribed against `main`),
  `:440-500` (the refine branch and its volume-projection gate), `:715-725` and
  `:826-835` (the two rejections to delete).
- Read-only reference, under `~/research-bridges/zmodel-steve/zmodel3d-amr/`:
  `zmodel3d/mesh_solver.py:1775-1832`,
  `examples/run_adaptive_mesh_bubble.py:1438-1451` and `:1557-1565`.

The document is stale in these small ways. Correct them as part of this task:
- T4a's **Do** step 6 and T2d's "What is deliberately NOT built" paragraph both
  say `VolumeProjection::projectToVolume` first executes at T4c or T4d,
  whichever lands first. T4b ran it first — its **Met.** paragraph and `## T4b`
  in the log both say so. What is true of this task is narrower and is already
  stated in its own entry: it is where the projection first runs on the
  **refine** path.
- "What is NOT yet true": the `--smooth-iters > 0` (T4c) clause and the sentence
  requiring a refining run to pass `--flip-passes 0 --smooth-iters 0
  --no-isotropic-cleanup` both stop being true of `--smooth-iters`.
- `src/Beatnik_MeshQuality.hpp:182-186` says "the caller re-bases explicitly
  after both" the flip and the smooth pass. The reference passes
  `reset_reference=False` to the tangential pass at **both** of its call sites
  (`:1449`, `:1562`), so that sentence is true only of the flip path — scope it
  to T4d.

Decisions already made — do not reopen, and record them in
`tasks/framework-progress-log.md` under `## T4c`:
- **One kernel, two entry points.** `improveQualityTangential` and
  `tangentialRelaxation` are the same operator; implement one private device
  kernel and have both delegate. `tangentialRelaxation` ships **unexercised** —
  its only caller `isotropicCleanup` is T4d — which is `projectToVolume`'s
  standing between T2d and T4b, not an oversight. Say so in the log.
- **Compose T2b's validated operators; do not write a new stencil.**
  `SurfaceOperators::graphLaplacianVector` over `mesh.vertexOneRing()` is exactly
  the reference's sorted unique-neighbour centroid offset. T2b measured its `max`
  as `0.012663750374617330` and the reference's own raw offset on the same mesh
  is `0.012663750374617372` — agreement at `4e-17`, which is what localizes a
  failure to the projection rather than to the stencil. `::projectTangent` is the
  projection.
- **The sweep is Jacobi, not Gauss-Seidel.** The reference builds the whole
  displacement array from `vertices` and only then adds it (`:1804-1812`), so one
  sweep is independent of vertex order and therefore of the partition. Reading it
  as Gauss-Seidel is a faithful-looking transcription that is partition
  dependent — the trap `## T4b` records for the gradation sweep.
- **Recompute the vertex normals every sweep, and halo-exchange positions
  between sweeps.** Positions live in the mesh, so that is `mesh.haloExchange()`;
  `haloExchangeVertexView` (T4b) exists for views held *outside* the mesh and is
  not needed here. Without the exchange, boundary vertices relax against stale
  neighbours and the seam moves with the rank count.
- **Do not re-base the AMR reference state after this pass.** Both reference call
  sites pass `reset_reference=False`, and with `--flip-passes 0` the flip pass
  returns having changed nothing, so nothing re-bases. An
  `AdaptiveMesh::resetReferenceState` call here would change every subsequent
  refinement decision.
- **The `--redistribute-every` branch needs no repartitioning**, so this task is
  not blocked on T5d. See the entry.

Constraints specific to this task:
- **The reference's CLI is the CLI** — no new switch, and only the two named
  rejections come out of `requireSupportedConfiguration`. The others stay, by
  name and task ID.
- Out of scope by name and task ID: `DynamicRemesh::tangentialSmooth` (T4d — a
  port of `dynamic_remesh.py::tangential_smooth_vertices`, a *different*
  function), `isotropicCleanup` and both flip passes (T4d, blocked on Tessera
  G5c), and `Diagnostics::compute`'s four `NaN` AMR indicator fields (T4a's named
  open follow-up, which runs inside three other gate members).

Running on the cluster — this machine uses Flux:

- Do not run executables directly. No `flux run`, no bare `mpirun`, and no
  invoking a test binary or the example driver on the login node — not a
  single-rank smoke test, not `--help`. On a login node an interactive launch
  does not fail, it blocks forever waiting for an allocation that never comes.
  Everything that executes goes through a batch script submitted to the `pdebug`
  queue. Building is the exception and is done on the login node.
- **Build with `spack install`, never `cmake`/`make`** — this checkout is spack
  mode: `spack env activate ~/spack_envs/tuolumne_beatnik && spack install`.
  `Beatnik` is a CMake INTERFACE library that does not track its headers as
  dependencies, so a header-only change can report a sub-second no-op build;
  `touch examples/02_adaptive_mesh_bubble/adaptive_mesh_bubble.cpp` first
  (`systems/tuolumne/claude.md` §3).
- **Do not write a new batch script.** Both tier wrappers already exist and
  **discover** their tests, so a newly registered test lands in them for free:
  `scripts/tuolumne/unit_tests.flux` (queue `pdebug`, `--nodes=1 --exclusive
  -t 20m`, no account flag on this system) and the ship gate
  `scripts/tuolumne/run_regression_minset.flux`. Each writes
  `<job-name>.<jobid>.log` to the submitting directory and exits non-zero if any
  test failed, so the job's own status is meaningful.
- The exit criterion needs the unit tier twice — once at the default one rank,
  once at four:

  ```bash
  jobid=$(flux batch scripts/tuolumne/unit_tests.flux)
  while flux jobs "$jobid" 2>/dev/null | grep -q "$jobid"; do sleep 30; done
  flux job status "$jobid"
  ```

  then again with `BEATNIK_UNIT_RANKS=4` exported before `flux batch`. Poll every
  30 seconds and continue the moment the job leaves the queue; do not sleep for
  the walltime.
- **Then run the ship gate, because this task edits `Solver::advanceOneStep` and
  all five `regression` members execute it** (CLAUDE.md "Minimum test set"):
  `BEATNIK_TEST_SCRATCH=/p/lustre5/stewartj/beatnik/gate_scratch flux batch
  scripts/tuolumne/run_regression_minset.flux`, polled the same way. That path
  must stay on the parallel filesystem — pointed at tuolumne's node-local `/tmp`,
  ranks 1-4 pass and ranks 5-6 die inside `H5FD__mpio_open`, which reads exactly
  like the multi-rank solver bug the gate exists to catch. All 60 launches must
  end `[gate] PASS (label=regression)` with the five members' check counts
  unchanged — `Beatnik_Test_RefineSplitEdges` 86/86 and
  `Beatnik_Test_DynamicRemeshSplit` 377/377 are the two that would move if the
  new call site perturbed a mesh it should not.
- Leaving the queue is not success. Read `flux job status` and then the `.log`
  file before concluding anything: a job killed at its walltime or lost to a node
  failure disappears from the queue exactly like one that passed.
- If a job is still pending after ten minutes, stop polling and report the queue
  state instead of waiting silently. Cancel any job you started and are no longer
  waiting on (`flux cancel <jobid>`) before you finish.
- **Do not touch the rank-to-GPU binding.** Both wrappers launch with
  `--ntasks=N --nodes=ceil(N/4) --exclusive --gpus-per-task=1 --cores-per-task=24
  --setopt=mpibind=verbose:1`; tuolumne is 4 ranks per node. A wrong binding does
  not fail, it oversubscribes one device and returns a plausible number.
- Both wrappers read their manifests on **fd 3**, not stdin, because `flux run`
  inherits and consumes its caller's stdin and swallowed every remaining manifest
  line — a green tier that silently skipped the new test. Do not move that back.
- **Do not run `clang-format`.** Formatting is the user's job; write and edit in
  the style of the surrounding code.
- Stop and report rather than work around. If the build fails twice for the same
  reason, or a job dies twice the same way, write up what you tried and what the
  error was, and stop. Do not loosen a tolerance, drop a case, or substitute a
  smaller run — that silently changes what **DONE** means and the substitution is
  invisible in the diff.

Exit criterion: as restated in the `T4c` entry — read it there rather than from
this block. Every reference number in it was measured from the read-only Python
on the default icosphere; hard-code them as 17-digit literals, the convention
every other test in this tree follows, and do not adjust one to make a check
pass.

When done: commit and push the work, then mark `T4c` **DONE** in
`tasks/framework.md` with a **Met.** paragraph stating what was actually
verified, and append a `## T4c` section to `tasks/framework-progress-log.md`
covering the decisions above, any signature changed and what forced it, bugs only
running revealed, and the numbers measured — with an `**Affects:**` line naming
the later task IDs your findings change (T4d at least: it inherits the shared
kernel, and its `isotropicCleanup` is `tangentialRelaxation`'s first caller).
Delete this `**Session prompt:**` block from the `T4c` entry in the same edit;
it describes work that is now finished.
````

---

### T4d — Coarsening, flips, and isotropic cleanup — **BLOCKED (upstream)**

**Depends on:** T4b, T4c, **and Tessera G5b (edge collapse), G5c (edge flip) and
G5d (compaction)** — all three `NOT STARTED` in `../tessera/tasks/`
(`edge-collapse.md`, `edge-flip.md`, `mesh-compaction.md`), with Tessera's own
ordering being compaction → flip → collapse. **This task cannot start until they
land**, and that is the only thing standing between Phase 4 and the reference's
full behaviour.

**Fill in:** `Beatnik_MeshInterface.hpp::{collapseEdges, flipEdges, compact}`;
the collapse and flip thirds of `Beatnik_DynamicRemesh.hpp`
(`collapseShortEdges`, `flipEdgesForQuality`) **and `tangentialSmooth`**, which
was assigned to no task until T4b assigned it here — it is a port of
`dynamic_remesh.py::tangential_smooth_vertices`, *not* of
`mesh_solver.py::improve_mesh_quality_tangential` (that one is T4c), and it runs
inside the same `if changed or needs_quality_repair` block as the flips;
`Beatnik_MeshQuality.hpp::{improveConnectivityByFlips, valenceEqualizingFlips,
isotropicCleanup}`; the `requireSupportedConfiguration` rejections T4a and T4b
added. Note `tangentialSmooth` moves vertices and changes no connectivity, so
unlike the other two it is **not** blocked on a Tessera gap — it is here only
because it is unreachable while the flips are.
← *Python:* `dynamic_remesh.py::{collapse_short_edges, flip_edges_for_quality,
tangential_smooth_vertices}`,
`mesh_quality.py` (44-167),
`mesh_solver.py::improve_mesh_connectivity_by_edge_flips` (1704-1772)

**Additional information needed, to be answered against Tessera when the three
land, not now:** whether `collapseEdges` supplies the two-phase owner-decides
protocol for the link condition and the geometric safety test across a rank
boundary, or leaves it to the caller; and whether `compact()` invalidates the
face user fields the way `refine()`/`splitEdges()` invalidate edge user fields.

**Exit criterion:** a default-configuration run completes 50 steps at ranks 1
and 4; volume drift below `1e-10`; the minimum triangle quality stays above
`--remesh-min-quality` for the whole run; and with `--isotropic-cleanup` a run
that reaches a tightening roll-up does not die on the "curvature sliver", with
the valence histogram staying concentrated at 6. Compare **statistics** against
the Python, never the flip or collapse set (R7).

---

### T4e — Nonlocal proximity queries — **NOT STARTED (opt-in)**

**Depends on:** T4b, and T3a if Canopy's tree is the chosen vehicle.

Off by default on both switches, so nothing else waits for it. A genuinely global
spatial search over face centroids with two exclusion criteria — topological
rings (`proximity_exclusion_rings` 3) and material-coordinate distance — that no
ghost depth makes local.

**Additional information needed before a fine-grained design:** whether to build a
distributed ArborX tree, reuse Canopy's FMM tree (T3a will have opened Canopy and
can answer), or use a two-level scheme; and how to represent the per-face
variable-size exclusion sets, for which a ring-depth-bounded CSR is the obvious
first attempt but needs sizing against a real refined mesh. **A distributed
spatial-index library would be a new third-party dependency and is a decision for
the user, not for the implementing session** — the alternative that avoids it is
reusing Canopy's tree.

**Exit criterion:** with `--remesh-use-proximity`, a run in which two sheet lobes
approach to within `proximity_activation_distance` refines the approaching faces
and not their topological neighbours, at ranks 1 and 4, with the marked set
identical at both rank counts.

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

### T5e — Gold sets for the three inert dt controls *(human step)* — **NOT STARTED**

**Depends on:** none. T2d is `**DONE**`, which is all this needs. **Implementable
now** — it is not gated behind Phase 3 or Phase 4, and it is sequenced here
because it is coverage rather than capability, not because anything precedes it.

The reference's adaptive timestepping is `choose_step_dt`
(`run_adaptive_mesh_bubble.py:889-901`) plus two clamps the *caller* applies in
the step loop (`:1406-1410`) and the `--t-end` loop break (`:1402-1403`). All of
it is ported and has a body. Three of its controls are nevertheless dead in every
gold file that exists, because each defaults to off:

| Control | Default | Reference | Port |
| --- | --- | --- | --- |
| `--max-sheet-dt-product` | `0.0` (off) | `:897-900`, `max_sheet_strength` `:904-920` | `src/Beatnik_TimeIntegrator.hpp:277-300`, `src/Beatnik_SurfaceState.hpp:474-544` |
| `--dt-switch-time` / `--dt-after-switch` | `-1.0` (off) / `0.001` | `:1407-1408` | `src/Beatnik_Solver.hpp:410-412` |
| `--t-end` | unset | `:1402-1403`, `:1409-1410` | `src/Beatnik_Solver.hpp:337-339`, `:413-414` |

This task produces the gold sets. **T5f** consumes them; splitting them is the
pattern T1a/T2a set, because generating a gold file means running the read-only
Python and committing the result, and that is a distinct verifiable act from
writing a test.

**Fill in:** no source. Three new gold directories under
`tests/regression_tests/`, each with a `README.md` carrying its exact generating
command, exactly as `tests/regression_tests/direct-solve-10-steps/README.md`
does.

**Reference:** `tests/regression_tests/direct-solve-10-steps/README.md` for the
base command and the directory convention; T2a above for why each option on it
is there.

**Do:**

1. **Start from T2a's command and change nothing else.** Adaptivity stays off
   (`--no-dynamic-remesh --refine-every 0`) for T2a's reason: this task isolates
   the *dt controls*, and refinement would move `h_min` discontinuously and
   confound them. `--source-quadrature vertex` is not optional (R11).

   ```
   python examples/run_adaptive_mesh_bubble.py --steps 10 \
     --source-quadrature vertex --br-approximation direct \
     --no-dynamic-remesh --refine-every 0 \
     --checkpoint-every-steps 1 --no-video --checkpoint-dir <dir>
   ```

2. **Pick each control's value so the clamp actually binds, and record how it was
   picked.** A value that never binds produces the T2a trajectory exactly and
   makes T5f pass while testing nothing — R15. The window is bounded on both
   sides: a clamp below `--min-dt` (`2.5e-4`) is swallowed by the floor, which is
   applied *inside* the `min` (`Beatnik_TimeIntegrator.hpp:293-298`), and a clamp
   above the unthrottled `dt` (`0.003`) never fires. So every value must land
   strictly inside `(2.5e-4, 3e-3)`.
   - **`--max-sheet-dt-product C`.** `C` divides `max|S|`, which is not known a
     priori. Probe it first — print `max_sheet_strength(state)` from a `--steps 1`
     run — then set `C = 0.5 · 0.003 · max|S|`, which puts the clamp at half the
     adaptive dt. Record the probed `max|S|` and the chosen `C` in the README.
   - **`--dt-switch-time 0.012 --dt-after-switch 0.001`.** T2a's series reaches
     `t = 0.015` at step 5, so the switch arms partway through and clamps roughly
     the back half of the run to `0.001`. `0.001` is the reference default and sits
     inside the window.
   - **`--t-end 0.02`** with `--steps 10`. The unclamped run passes `0.018` around
     step 6, so step 7 is *truncated* to land exactly on `0.02` and the loop then
     breaks — one file exercises both halves of the `--t-end` path, the
     short final step and the early exit.

   These three are **starting values, not results.** Confirm from the produced
   files that each bound, and if one did not, adjust it and record the value that
   worked rather than committing a set that reproduces T2a.

3. **Commit under three sibling directories**, named for the control:
   `dt-max-sheet-product/gold/`, `dt-switch-time/gold/`, `dt-t-end/gold/`, each
   with its own `README.md`. Keeping them separate rather than in one directory is
   what lets T5f name a single gold set per sub-run and report which control
   failed.

**Exit criterion:** the three gold directories are committed with their
`README.md` commands, `compare_output.py` self-compares one file from each at
`--rtol 1e-12 --atol 1e-14` and exits 0, and — the check that makes the set worth
having — **the per-step `time` series of each set is shown to differ from the T2a
series in the expected direction and at the expected step**, with the numbers
written into the README:

- `dt-max-sheet-product`: every step's dt is ≈ half T2a's, from step 1;
- `dt-switch-time`: the series tracks T2a until `t` crosses `0.012` and the
  per-step dt is `0.001` thereafter;
- `dt-t-end`: the last file has `time` equal to `0.02` to within `1e-14`, and the
  set has **fewer than 11 files**, i.e. the loop broke before the step budget.

A set whose `time` series matches T2a's is a failed generation, not a passing
one; that is the whole point of the criterion.

### T5f — Regression test for the three dt controls — **NOT STARTED**

**Depends on:** T5e.

**Fill in:**
- `tests/regression_tests/Beatnik_Test_DtControls.cpp` — one new test binary
  driving three sub-runs, one per control, each against its T5e gold set.
- `tests/CMakeLists.txt` — add the source to `BEATNIK_REGRESSION_TEST_SOURCES`
  (`:209-216`) **and** the matching `_beatnik_args_Beatnik_Test_DtControls_abs`
  / `_rel` pair (`:240-262`); a missing argument list is a configure-time
  `FATAL_ERROR` (`:264-273`), not a silent launch. Add the three gold
  directories to the install rules (`:454-471`), following the T2a block —
  a regression test installed without its gold data is not installed.

**Reference:** `tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp` for
the whole shape — the per-step gold lookup by `_step%07d.npz` suffix (which is
why it takes a gold *directory* and not a file, `tests/CMakeLists.txt:257-262`),
the `compare_output.py` invocation, and the rank-0-reports convention.
`src/Beatnik_TimeIntegrator.hpp:228-303` and `src/Beatnik_Solver.hpp:404-438`
for what is under test.

**Callers to update: none.** This task adds a test and its registration; no
signature in `src/` changes. `Solver::requireSupportedConfiguration`
(`src/Beatnik_Solver.hpp:547-571`) rejects none of the three controls today, so
all three run as-is and nothing needs unblocking first.

**Do:**

1. **Find the gold set by step suffix, never by time key.** The checkpoint name
   embeds the time (`CheckpointIO::timeKey`, `src/Beatnik_IOInterface.hpp:463`,
   six fractional digits), and the time is the quantity under test — rebuilding
   the filename from a computed time would make the test agree with whatever dt
   it produced. T2d's test already does it this way.
2. **Assert the clamp bound, per sub-run, as a count and not as a vibe.** Derive
   from the gold `time` series how many steps each control was expected to alter,
   and require exactly that many. This is R15's mitigation and it is the check
   that separates "the clamp works" from "the clamp is a no-op and the gold set
   was mis-generated".
3. **Report which sub-run failed.** Three configurations in one binary means a
   bare non-zero exit is ambiguous; name the control in the failure line.
4. Keep each sub-run at the step count T5e generated and no more — the gate pays
   for every step at twelve configurations.

**Exit criterion:** a `regression`-tier test registered per backend with the
`_<BACKEND>` suffix the gate selects on — **this grows the ship gate by one
member to 60 launches (T4a has landed, so the gate is at 48), so confirm with the user
before registering it** (CLAUDE.md "Minimum test set", and the same note on
`tests/CMakeLists.txt:53`). Registering one suffix-less binary instead is a
silent zero-test pass on the installed path (T2c's warning). It must show, at the
gate's ranks 1-6 on SERIAL and HIP:

- all three sub-runs pass `compare_output.py` against their T5e gold set at
  every step, at `--rtol 1e-10 --atol 1e-12` (T2d's tolerance — these runs are
  the same length and the same physics, so a looser one would be unexplained);
- the per-step `time` matches its gold literal to `1e-14` relative in all three,
  which is where a mis-ordered clamp shows up first;
- the bound-step count of each sub-run equals the count derived from its gold
  series (step 2 above);
- dt is **identical on every rank** — reduce the per-step dt with
  `MPI_Allreduce`/`MPI_MIN` and `MPI_MAX` and require the two to be bitwise
  equal. `max|S|` is an `MPI_MAX` and so is order-independent by construction,
  but that is the claim under test, and a rank-dependent dt is the one failure
  here that a single-rank test cannot see;
- **the failure direction:** re-running the `--max-sheet-dt-product` sub-run with
  the control at its inert `0.0` against the *same* gold set fails, and fails on
  the **`time` mismatch at step 1** with that step named — not merely by exiting
  non-zero, and not on a field comparison several steps later.

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

**One constraint that did not exist at M1: the two editing families are disjoint
and enforced by a throw**, so `refine()` and
`splitEdges()`/`collapseEdges()`/`flipEdges()`/`compact()` cannot run on the same
mesh. **Beatnik's resolution is to use the Remesh family exclusively and never
call `refine()`** — see *The editing-family question — RESOLVED* under Phase 4 for
the reasoning and for what T4a does instead. The constraint is noted on all four
declarations in `Beatnik_MeshInterface.hpp` so it cannot be met for the first
time as a runtime throw; T4a deletes the `refine()` declaration outright.
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
remain open, and they are what still blocks T4d.*

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
**What the three open gaps actually block, restated after the Phase 4 design
settled:** the split third of `dynamic_remesh.py` is expressible today and is
**T4b**; the collapse and flip thirds, and every flip in `mesh_quality.py`, are
**T4d** and wait on G5b/G5c/G5d. Nothing else in Phase 4 is blocked.

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

One smaller shape change, recorded inline in the header: `adopt()` now
requires its arrays replicated on **every** rank (not rank 0), because
`buildFromTriangleSoup` has no communication and `distribute()` relies on
replication.

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

**Measured at T4a, and it is worse than "will not match the Python".** A
capped pass is not rank-count invariant *within Beatnik either*. The threshold
search converges to a value pinned between two adjacent scores, so an ulp-level
difference in a score near the cut — which R2 guarantees across rank counts —
flips a mark. Measured on the T4a configuration: pass 3 gives **1372 faces at
ranks 1-4 and 1390 at ranks 5-6, identically on both backends**, which locates
it in the cross-rank reduction order rather than in the on-node atomics. The
uncapped passes are invariant across all twelve configurations. So a test over a
capped pass may assert the cap, the projection identity and the structural
invariants, and must not assert a face count.

**Downgraded by the Phase 4 design, but not retired.** The *cost* half of this
risk is gone: under an edge mask the projected face count is
\f$\sum_f(|S_f|+1)\f$, one local sum plus one `globalSum`, so the cap can be
enforced by a global threshold search on the score — \f$O(\log)\f$ projections,
parallel and deterministic — instead of the reference's \f$O(N_{\text{seeds}})\f$
sequential accept loop. The *divergence* half stands unchanged: a threshold
search accepts a different mark set than a greedy walk, so a capped run still
will not match the Python face for face.

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
## R12 — Triangle shape at depth is bounded **by the mask**, not by `splitEdges()`

Phase 4 rests on repeated `splitEdges()` holding triangle shape over many rounds.
Tessera measured this to depth (`../tessera/tests/test_split_edges_depth.cpp`,
`unit` tier, not in Tessera's gate; rationale and the four families at `:30-72`,
knobs `TESSERA_SPLIT_DEPTH_ROUNDS` default 30 and `TESSERA_SPLIT_DEPTH_FACES`
default 2 000 000 at `:517-520`). The assumption holds, **conditionally**, and
the condition is a constraint on Beatnik's code rather than a property of
Tessera's.

Global minimum radius ratio \f$r/R\f$ (`0.5` equilateral, `0` degenerate) per
round, repeated `splitEdges()` with no intervening `migrate()`:

| mask rule (`Family`, `:305-311`) | rounds | min \f$r/R\f$ trajectory | verdict |
| --- | --- | --- | --- |
| `AboveMean` — split iff **longer** than the mean | 10, F 320 → 3 276 800 | `0.3780 0.3780 0.2815` repeating | **bounded — exactly periodic, period 3.** Min angle `33.203°` every round; population below `0.30` is exactly 0 outside the dip rounds |
| `BelowMean` — split iff **shorter** than the mean | 7, F → 1 179 680 | `0.1953 0.0568 0.0169 0.0068 0.0031 0.0015 0.0007` | unbounded — halves per round; min angle `24.96° → 0.21°` |
| `HashThird` — length-blind (hash of midpoint position mod 3) | 27, F → 2 340 916 | `0.1953` → `<1e-4` by round 7, `0.0000` from round 8 | unbounded; 96.7% of faces below `0.30` by round 27 |
| `CapHash` — the same, inside a fixed geodesic cap | 30 | `0.2238` → `0.0000` by round 11 | unbounded **and localised** — a small region collapses, the rest untouched |

Every round line is byte-identical across np 1, 2, 4, 5 × {SERIAL, HIP} ×
{Serial, Default} — 7 configurations, 39 distinct lines, zero spread. These are
properties of the global mesh, not of a decomposition.

**Why, and why the red-green intuition does not transfer.** Tessera's red-green
`refine()` is shape-bounded independently of round count because its closure is
*transient*: un-close discards the whole closure layer every round, so every
visible triangle is one of finitely many retriangulations of a red triangle.
`splitEdges()` has no such reset — a \f$|S|=1\f$ median-cut child is an ordinary
face on the next call and can be cut again, so the reachable similarity classes
are unbounded in the round count. **`splitEdges()` cannot and does not offer a
shape guarantee.** What supplies the bound in the friendly case is that the rule
is *length-driven*, making it a coarse relative of Rivara longest-edge bisection:
it attacks the long edge of a stretched triangle, which is exactly the edge whose
bisection improves shape. It is self-correcting. Splitting short edges is the
same machinery run backwards.

**The condition on Phase 4:** every edge in the mask must be there **because it
exceeds a target length**. T4b's `split_selected_edges` driven by "edge longer
than the local target" is precisely the family that came out periodic. A mask
that starts from a length criterion and then adds edges for another reason — a
curvature term, a vorticity term, a region tag — inherits **none** of the bound
for those edges. This is stated as a hard constraint in T4a and T4b.

**How it would present:** a long run's minimum triangle quality declining slowly,
with no single bad edit to point at — which reads exactly like an accumulating
solver bug, and is not one.

**What distinguishes it.** Record the global minimum \f$r/R\f$ per refinement
pass against the **round index** from T4a's first run, and alongside it the
**count of faces below a fixed \f$r/R\f$ of `0.25`** — the cheaper and earlier
signal of the two:

- **Healthy:** the minimum *cycles*. It dips and recovers to a value it has held
  before, and never sets a new low late in a run. The sub-`0.25` count returns to
  zero between dips.
- **Shape problem:** monotone decline at roughly a constant factor per round,
  independent of the physics and reproducible from the mesh history alone. The
  sub-`0.25` count becomes a stable fraction of the mesh (~17% in the below-mean
  family) — this distinguishes "a few bad cells at a feature" from "the mesh is
  going bad" far earlier than the minimum does.
- **Solver problem:** tracks the roll-up, not the round index.

**If it does decline, the fix is on Beatnik's side.** Tessera deliberately did
*not* add a quality constraint inside `splitEdges()`: refusing to split an edge
whose child would fall below a floor would bisect fewer edges than asked on a
predicate the caller cannot see, contradicting the "bisects EXACTLY the marked
edges" contract, and turning a visible quality problem into silent
under-refinement. In order of preference:

1. **Make the mask longest-edge-consistent.** Before calling, promote each marked
   edge to the longest edge of its incident faces (Rivara propagation). This is
   what converts the unbounded families into the bounded one, and it is a pure
   mask transform written above Tessera.
2. **Filter the mask against a shape predicate and log what was dropped**
   (`_single_green_split_quality` is the reference predicate), so the resulting
   under-refinement is visible rather than silent.
3. Only if neither suffices, ask Tessera for an **opt-in mask filter that returns
   which marks it dropped** — that is the shape it should take; it is not built
   today because no consumer needs it.

**Before committing to a mask rule that is not purely length-driven**, add it as
a fifth family in `test_split_edges_depth.cpp` and run it: one batch job, ~1m15s
at np4, tells you whether the rule is in the bounded class. Three edit sites,
~40 lines total — the `Family` enum and `familyName` (`:305-326`), a `case` in
`buildMask` (`:333-397`), and a `driveFamily` call in `run` (`:493-500`). The
rule must be a pure function of **global mesh geometry** — Tessera's families
hash the midpoint *position*, not gids, because gids come from an `MPI_Exscan`
and are not rank-count invariant (`:52-54`).

### MEASURED AT T4a — and T4a's mask is **NOT** in the bounded family

Phase 4's *Finding 3* claims Beatnik's two masks are "the two cases that rule is
good for", naming T4a's "all three edges of a marked face" as always-the-red-split
and therefore bounded. **That is wrong, and T4a measured it.** The global minimum
\f$r/R\f$ per refinement pass, and the population below `0.25`:

| pass | Python `min r/R` | Python `< 0.25` | Beatnik `min r/R` | Beatnik `< 0.25` |
| --- | --- | --- | --- | --- |
| 0 (initial) | `0.486497704566` | 0 | `0.486497704566` | 0 |
| 1 | `0.304119905237` | 0 | `0.304119905237` | 0 |
| 2 | `0.123117984672` | 4 | `0.123117984672` | 4 |
| 3 | `0.119867830292` | 101 | `0.119877` / `0.119868` | 94 / 96 |
| 4 | `0.119867790771` | 101 | `0.119876` / `0.119868` | 94 / 96 |

The minimum **does not cycle** and the sub-`0.25` count **does not return to
zero**: both decline monotonically, and the count settles at ~7% of the mesh.
That is this risk's *shape-problem* signature, arriving on the very first task
that could produce it.

**It is the reference algorithm's, not Beatnik's and not `splitEdges()`'s.** The
Python columns above are the reference's own numbers, computed offline from its
checkpoints with the same \f$8A^2/((a+b+c)abc)\f$ formula; Beatnik reproduces
the first two passes to twelve significant digits *including the count*, on both
backends at every rank count. The pass-3/4 spread is R4's capped-pass divergence
and nothing else. So the decline is a faithful port, and the correct response is
to record it rather than to "fix" it away from the reference.

**The mechanism, which this risk predicts once the mask is read carefully.** A
*red* face's four children are similar to their parent and are fine. The **green
transition faces** at the red region's boundary are not: they are bisected on
whichever edge their neighbour happened to red, **not** on their own longest
edge. Those edges are not length-driven, they inherit none of the bound, and the
next pass cuts the previous pass's green children again — `splitEdges()` has no
reset, so a \f$|S|=1\f$ child is an ordinary face next call. Phase 4's claim
holds for the red interior and fails for exactly the faces that set the minimum.

**What this does *not* undermine.** T4b's mask — "every edge longer than its
target" — is genuinely the family Tessera measured as periodic, and is
unaffected. What is retired is the assumption that *T4a's* mask inherits the
bound by being a red split.

### MEASURED AT T4b — the length-driven mask IS in the bounded family, and the CAP can take it out

T4b ran the same two signals per pass over twenty steps of split-only dynamic
remeshing, and got the **healthy** signature this risk describes, in both of its
halves:

| pass | min \f$r/R\f$ | \f$< 0.25\f$ | splits |
| --- | --- | --- | --- |
| 0 (initial) | `0.486497704566` | 0 | — |
| 1 | `0.373875540852` | 0 | 120 |
| 2 | `0.248492357897` | **120** | 120 |
| 3 | `0.248490855246` | **120** | 120 |
| 4 | `0.281539942917` | **0** | 60 |
| 5 | `0.281537474137` | 0 | 120 |
| 6-20 | `0.2815` → `0.281492866851` | 0 | 0 |

The minimum **dips and recovers**; the sub-`0.25` population appears and
**returns to zero**; the last third of the run sets no new low. The residual
decline from pass 5 on is in the sixth digit and tracks the roll-up rather than
the round index — this risk's *solver* axis, and at that magnitude simply the
bubble deforming under a fixed connectivity. Beatnik reproduces the reference's
column to all twelve digits at **every** pass, and the whole table is
byte-identical across {SERIAL, HIP} × ranks {1, 3, 6}. The measured floor is
`0.248`, which lives in `Beatnik_Test_DynamicRemeshSplit.cpp` as
`kMinRadiusRatioFloor` — again **not** Tessera's `0.25`, which this run would
fail by four ulp at passes 2 and 3.

So the two masks now have measurements and they say opposite things, which is
this risk's whole point: the bound belongs to the *mask*, not to `splitEdges()`.

**And there is a third finding, which this risk did not anticipate: the per-pass
cap can move a length-driven mask OUT of the bounded family.** At the reference's
*default* sizing parameters the same configuration marks all 480 edges and
`--remesh-max-splits 300` truncates the mask; measured on the reference itself,
that run's minimum \f$r/R\f$ goes to `0.204341652937` at pass 1 with **32** faces
below `0.25`, then **64** — and it never returns to zero for the remaining
eighteen steps. A truncated mask is no longer "every edge longer than its
target": the surviving edges are the top-\f$N\f$ by ratio, which is a
*rank*-driven rule, and the neighbours left unsplit are exactly the transition
faces that carry no bound. Practical consequence for any later task: **a capped
pass is not just R4's count divergence, it is an R12 exposure**, and a
configuration whose cap binds every pass should expect the shape-problem
signature.

**What to do about it is a later task's decision, and the options are this
risk's ordered list unchanged** (Rivara mask promotion first, a caller-side shape
filter with logging second, an opt-in Tessera filter last). T4a deliberately did
none of them: each changes every refinement decision away from the reference,
which is a semantic deviation that needs its own gold set and its own task.

**Do not set the floor a priori.** T4a's exit criterion requires the floor to be
measured on the first run and written into this document, because a guessed floor
either fires spuriously or never fires at all. Do **not** reuse Tessera's
`kMinRadiusRatioFloor` (`0.25`, `../tessera/tests/test_split_edges.cpp:100-140`)
as that floor: it is a statement about case 8's mask, not about `splitEdges()`,
and its comment carries an explicit SCOPE paragraph (`:123-136`) saying so — and
on the T4a configuration a `0.25` floor fails outright, with 96 faces below it.

**THE MEASURED FLOOR IS `0.119`.** The global minimum \f$r/R\f$ over the whole
20-step T4a run is `0.119867784111` at ranks 5-6 and `0.119876446958` at ranks
1-4, on both backends — a spread of `7.2e-5` relative, entirely inside the capped
pass's mark divergence. (An earlier revision gave the ranks 5-6 figure as
`0.119867826031`; that is pass *3*'s minimum, not the run's. Pass 4 marks
nothing but the mesh evolves five more steps, and the minimum drifts down in its
tenth significant digit. Neither the floor nor the ranks 1-4 figure moves.) `0.119` is that minimum rounded **down** to three digits,
so a run reproducing the measurement clears it by roughly 700x the observed
spread while a run that sets a genuinely new low fails. It lives in
`tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp` as
`kMinRadiusRatioFloor`, and **it is a floor for this configuration and this
number of passes**, not a property of the method: four passes reach `0.1199`, and
nothing here says a fortieth pass would.

**One threshold worth borrowing rather than inventing:** the depth diagnostic
reports the tail population at `r/R` of `0.30, 0.25, 0.20, 0.15, 0.10`
(`kTail`, `:98-102`). The `0.25` this risk asks T4a to log is one of them, so
Beatnik's diagnostic and Tessera's are directly comparable — keep it there.

## R13 — Beatnik's two-edge diagonal differs from the Python's

For a face with exactly two split edges, `Tessera::splitEdges` cuts the quad
along its **shorter** diagonal, decided geometrically and tie-broken on `EdgeKey`
(`../tessera/docs/design.md` → *The two-edge tie-break*).
`mesh.py::refine_marked_faces` uses a fixed diagonal determined by which pair of
edges was split. Beatnik keeps Tessera's: it is the better-shaped element, and it
is what Tessera's quality numbers were measured with.

**Consequence:** a Beatnik and a Python refinement of the same mark set have
identical V, E and F but **not** identical connectivity, wherever a face had
exactly two split edges. Vertex positions are unaffected — the midpoints are the
same points either way — so a comparison of positions and fields still holds;
only face-for-face connectivity comparison does not.

**Measured at T4a, and the consequence runs one level further than the paragraph
above says.** Identical V/E/F holds for *one* refinement of *one* agreed mesh.
From the pass after that the two codes are integrating **different meshes**, so
their indicators differ and they select different mark sets — and then even the
counts diverge. Measured on the T4a configuration: pass 1 agrees exactly
(320 → 452 faces, 162 → 228 vertices), pass 2 gives **788 faces here against 796
there**. Nothing is wrong on either side. What *does* survive is the shape
statistics: Beatnik reproduces the Python's global minimum \f$r/R\f$ and its
count below `0.25` to twelve significant digits at both of those passes, i.e.
the worst elements are literally the same elements even where the counts differ.
So a Python comparison for an adaptive run is a **one-pass** comparison of
counts plus an all-pass comparison of shape statistics.

**Measured again at T4b, and there it did NOT bite — which is worth stating so
the risk is read as conditional rather than as a law.** T4b's masks are partial
from the first pass (120 of 480 edges), so faces with exactly two split edges do
arise and the two codes do choose their diagonals by different rules. Even so,
Beatnik reproduced the reference's per-step face and vertex counts **exactly at
all twenty steps**, along with both shape signals to twelve digits. The honest
reading is that on this mesh the two rules agree wherever the case arises — not
that the risk is retired. A later task that sees a late-step count divergence
under a partial mask should check the diagonal before assuming a bug, and a
later task that sees agreement should not conclude the rules are the same.

Practically the same class of divergence as R7, and handled the same way: compare
counts and statistics, not the edit set. It is recorded separately because it is
*deterministic and structural* rather than order-dependent, so unlike R7 it will
reproduce identically at every rank count — and a tester who sees a stable,
reproducible connectivity difference will otherwise reasonably conclude it is a
bug.

## R14 — Face user fields silently widen the checkpoint

T4a adds `FaceFields<Real, Real>` to the mesh type. `Tessera::writeMesh` writes
the face user pack unconditionally, as its own datasets, exactly as it does the
vertex pack — so every checkpoint gains `/faces/u0` and `/faces/u1`.

Harmless **today**: `compare_output.py` reads HDF5 only through `FIELD_MAP`, so
datasets it does not name are ignored, and the gold `.npz` files carry no face
data to disagree with. Two consequences that are not:

1. **Every checkpoint written before T4a becomes unreadable by a post-T4a
   binary.** `Tessera::readMesh` treats a field-pack mismatch as an
   **`MPI_Abort` inside Tessera, not a catchable exception** (M2's trap (b)), so
   this surfaces as a hard abort with no Beatnik-side message. Nothing depends on
   it yet — `CheckpointIO::read` still throws (T5b) — but T5b must not be written
   assuming the two packs are compatible.
2. **`/faces/u<N>` is positional**, exactly as `/vertices/u<N>` is, so reordering
   `Beatnik::FaceFieldId` silently relabels every checkpoint on disk. M2 mitigated
   the vertex case with `/beatnik/vertex_field_names` plus a `compare_output.py`
   cross-check; **T4a extended the same mechanism** rather than inventing a
   second one. `CheckpointIO::write` now emits `/beatnik/face_field_names` from
   `AdaptiveMesh::face_field_names`, under the same `static_assert` against
   `FaceFieldId::Count`, and `compare_output.py::check_face_field_names`
   verifies it. One difference from the vertex case, stated because it changes
   what the check *is*: no face dataset appears in `FIELD_MAP` and none is
   compared — the gold `.npz` files carry no per-face state — so the check is
   against a spelled-out `FACE_FIELD_NAMES` tuple rather than against the path
   table.

**The pack is THREE slots, not the two this risk was written against.**
`FaceFields<Real, Real, Real>` = `{ReferenceArea, ReferenceCurvature,
RefineMark}`; every checkpoint gains `/faces/u0`, `/faces/u1` **and**
`/faces/u2`. The third is scratch between refinement passes and is in the pack
only because route (a) has to halo-exchange it, and `haloExchange()` addresses
fields by their compile-time Cabana member index — so a mark held outside the
mesh cannot cross a rank boundary at all. It is zeroed after every pass, so a
checkpoint carries zeros unless it was written mid-pass, which nothing does.

## R15 — A dt clamp that never binds is indistinguishable from a correct one

T5e/T5f test three controls that are **`min`s against the adaptive dt**. If the
value chosen for one of them sits outside the window where it can bite — above
the unthrottled `--dt` of `0.003`, or below `--min-dt` `2.5e-4`, where the floor
inside the `min` swallows it (`src/Beatnik_TimeIntegrator.hpp:293-298`) — then
the run reproduces the T2a trajectory exactly. The gold set generated from it is
a copy of T2a's, the test compares Beatnik against it and passes, and **the
control was never executed.** A completely unimplemented clamp passes the same
test just as well.

This is the failure mode a gold-file comparison is structurally blind to, because
both the working and the dead implementation produce identical output. It costs
nothing at the time and shows up much later as a control that was believed
covered and is not.

**How it would present:** it would not. That is the risk. A green gate, three
extra members, and no signal.

**What distinguishes it, and why both halves are needed:** the clamp's effect on
the *trajectory*, asserted as a number at both ends.

- **At generation (T5e):** the per-step `time` series of each gold set must
  differ from T2a's, in a stated direction and at a stated step. Written into
  T5e's exit criterion, because that is the last point at which a bad value is
  cheap to fix.
- **At test time (T5f):** the count of steps on which the clamp bound must equal
  the count derived from the gold series. This is the half that survives someone
  later regenerating a gold set with a different value — the count is re-derived
  from the gold rather than hard-coded, so the two stay consistent.

**Do not respond to a suspected instance by loosening a tolerance**; the
comparison is not failing, which is the problem. Recompute the window from
`max|S|` and `--min-dt` and check that the chosen value is inside it.

The same trap applies to any future `min`-shaped control. The general form:
**a clamp is only tested by a configuration in which it changes the answer**, so
its test must assert the change, not merely the agreement.
