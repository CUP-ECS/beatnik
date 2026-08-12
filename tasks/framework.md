# zmodel3d-amr → Beatnik C++ port

**Status:** IN PROGRESS — last updated 2026-08-12

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

**As of T1c (2026-08-12) there IS a driver path and a checkpoint, and regression
test 1 passes.** A `--steps 0` run generates the icosphere, initializes and
re-centres the fields, seeds the material coordinate, computes the two carried
scalars, and writes an HDF5 checkpoint that matches the Python gold file at
`--rtol 1e-12 --atol 1e-14` at ranks 1-6 on SERIAL and HIP.

**There is still no timestep.** `Solver::solve` implements a `steps == 0` guard
and nothing else; at `steps > 0` it throws. So the whole of the mathematics
below T1b remains a stub, and in particular:

- **No RHS, no integrator, no volume projection, no BR evaluation.** T2b, T2c and
  T2d. The BR solver and the quadrature are *constructed* by `Solver::setup` and
  never called.
- **No adaptivity.** T4a/T4b/T4c, still blocked on the disjoint-editing-families
  design question and on Tessera's G5b/G5c/G5d.
- **`SurfaceState::updateSheetVector` still throws**, deferred to T2b on a stated
  dependency (it *is* `surfaceGradient`, which is T2b's). Consequence in the
  checkpoint: `/vertices/u1` is written unconditionally by `Tessera::writeMesh`
  and is present-but-meaningless — `initializeFields` left it zero, which is a
  *defined* value and not a correct one. `compare_output.py` skips the field
  `state_model` does not select, so nothing depends on it yet.
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

- 2026-08-06 — Read the five Python sources in full. Established the conventions
  above and landed the four framework commits. Discovered that the brief's named
  port source (`solver.py`) is the wrong file and traced to the real sources
  instead; recorded above. Wrote and **ran** `compare_output.py` and its
  fixtures; the C++ was not built (see task V0). Next: V0.

- 2026-08-07 — **V0 and T1a complete.** Next: T1b (which needs M1 first).

  **V0.** `spack install` built the whole framework **with zero compile errors**
  — the anticipated round of `typename`/include/Kokkos-5-alias fixes never
  materialized, so no semantic decisions had to be made and none are recorded
  here. `--help` exits 0; the T1a command line parses, echoes its configuration
  (`execution space HIP`, `ranks 1`), and dies with
  `error: Solver::setup not implemented` — a stub, as required, not a parse
  error and not a signal. `Beatnik_Test_PythonCompare` passes and `_Negative`
  fails as its `WILL_FAIL` expects; both were run as the bare
  `compare_output.py` invocations `tests/CMakeLists.txt` registers, since spack
  mode has no build tree to run `ctest` in. The `systems/tuolumne/spack.yaml`
  drift noted in the task was already resolved by the working tree's uncommitted
  edits, which are now correct against the live env.

  **T1a.** `compare_output.py gold.npz gold.npz --rtol 1e-12 --atol 1e-14`
  exits 0 and reports the expected structure: 162 vertices, 320 faces,
  `state_model potential`, 162/162 unambiguous matches at the default
  `--match-eps`, and the **five** carried scalars present
  (`initial_volume 6.3235073124669514e-02`,
  `initial_min_edge 6.8976121063816842e-02`). Those two are the values T1b's
  exit criterion must reproduce to `1e-14` relative.

  **CORRECTED 2026-08-12 by T1c: "the four carried scalars" above said four; it
  is FIVE.** The authoritative set is not prose anywhere — it is the gold
  `.npz`'s own 0-d keys and `compare_output.py`'s `REQUIRED_FIELDS`, and both
  were read directly rather than inferred:

  | `.npz` key | dtype | compared |
  | --- | --- | --- |
  | `state_model` | `<U9` | exactly |
  | `time` | `float64` | rtol/atol |
  | `step` | `int64` | exactly |
  | `initial_volume` | `float64` | rtol/atol |
  | `initial_min_edge` | `float64` | rtol/atol |

  `step` is the one the count of four dropped. M2's dataset table already said
  "the five scalars" and was right; M2's *prose* said "a `double` time, two
  `double` scalars and a string", which is four and was also wrong — corrected in
  `Beatnik_IOInterface.hpp` in the same change. `CheckpointIO::write` emits
  exactly these five under `/beatnik/`, plus `/beatnik/vertex_field_names`.

  **Four framework bugs found and fixed by actually running things.** All were
  latent — every one would have hit the first person to submit the ship gate,
  and none is visible by reading:
  1. `flux run` on a **login node** does not fail, it blocks forever waiting for
     an allocation that never comes. Cost a hung session. Now recorded as a
     project-wide guideline in CLAUDE.md and beside the launch template in
     `systems/tuolumne/claude.md` §4.
  2. `# flux: --time=N` is not a valid `flux batch` option (`-t Nm` is), so
     `run_regression_minset.flux` **could not be submitted at all**.
  3. `flux batch` copies the script into a `/var/tmp` spool, so the
     `BASH_SOURCE`-based `BEATNIK_REPO` fallback resolved to `/var/tmp/scripts`
     and the resolver source failed. The script's own comment predicted this;
     the code did not implement it. Both flux scripts now walk up from `PWD`
     (which flux does preserve) as the fallback.
  4. `scripts/lib/beatnik_env.sh` was not `set -u` clean — it read
     `BEATNIK_SYSTEM` and nine other caller-supplied knobs unguarded, aborting
     any `set -u` batch script on the first read. Fixed once, at the top of the
     resolver, with a `: "${KNOB:=}"` block rather than by patching each site.

  New: `scripts/tuolumne/run_v0_smoke.flux`, the batch wrapper for V0 steps 3-4.
  It carries the T1a command line, so it becomes the precursor to regression
  test 1 as the stubs fill in.

- 2026-08-11 — **Eight of M1's eleven Tessera gaps closed upstream.** Read
  `../tessera/tasks/{halo-depth,halo-scatter-add,global-reductions,face-adjacency,
  distributed-coarse-build,latlon-sphere,distributed-loadbalance-solve}.md` and
  `../tessera/README.md`. G1, G2, G3, G4, G6, G7 and G8 are implemented, tested
  at TIER `regression` on SERIAL + HIP at ranks 1-5, and documented; **G5a
  (`splitEdges`) is implemented too**, which the brief did not list. Gaps section
  updated and a second "Added since M1" table records the exact calls Beatnik now
  makes. Only **G5b (collapse), G5c (flip) and G5d (compaction)** remain, and
  they are what still blocks T4b and T4c.

  Three consequences beyond the gap list, each recorded where it bites:
  1. **R8 is largely retired** — `distribute(..., depth=2)` once at setup, and a
     `k > haloDepth` stencil now throws instead of returning short rows.
  2. **`refine()` rebuilds the halo itself**, so the adapter's `refine` →
     identity `migrate` → `haloExchange` contract is obsolete and must be dropped
     from `Beatnik_MeshInterface.hpp`.
  3. **New constraint: the hierarchical and remesh editing families are disjoint
     and enforced by a throw.** `refine()` (T4a) and `splitEdges()`/future
     collapse/flip (T4b) cannot run on the same mesh, and Beatnik's default
     configuration runs both. This is a design question for T4a/T4b, not an
     implementation detail.

  No Beatnik code changed in this session; the adapter still needs reworking
  against the above before T1b.

- 2026-08-11 — **M2 complete.** `Beatnik_IOInterface.hpp` rewritten against
  Tessera's real HDF5 I/O; the decision, the settled paths and the four
  non-obvious consequences are in the M2 section below. `FIELD_MAP` and
  `H5_PATH` updated and the two `.h5` fixtures regenerated in the same change.
  `gatherForCheckpoint` deleted from `Beatnik_Communication.hpp`;
  `Beatnik_Restart.hpp` and the `VertexFieldId` note in
  `Beatnik_MeshInterface.hpp` corrected to match. `spack install` clean;
  `Beatnik_Test_PythonCompare` passes and `_Negative` fails as its `WILL_FAIL`
  expects, both run as the bare `compare_output.py` invocations
  `tests/CMakeLists.txt` registers; the T1a gold self-compare still exits 0. The
  new `FIELD_MAP`-vs-declaration guard was checked by hand-writing a file with a
  permuted `/beatnik/vertex_field_names` — it exits 2 naming both sides.

  Two things worth carrying forward:
  1. **`clangformat.sh` is not safe to run repo-wide right now.** The tree is not
     clang-format-clean at HEAD (26 pending replacements in this one header
     before the edit), and the pass reflows doc comments — mangling the markdown
     tables that *are* the deliverable — across a dozen files nobody touched. The
     edits here were written to the format instead; the four files' replacement
     counts went 26→3, 5→5, 27→27, 0→0, i.e. no new drift. Worth a decision later
     on whether to format the whole tree once, deliberately, in its own commit.
  2. **T1c's exit criterion is unaffected but its comparison is not free.**
     `compare_output.py` now has two behaviours that only a real Beatnik file
     exercises (the inactive state field, the field-name cross-check), so the
     regenerated fixtures were shaped to look like one: Tessera paths, `uint64`
     faces, a present-but-wrong `sheet_vector`, and the name declaration.

- 2026-08-11 — **M1 adapter rework + T1b complete.** Next: T1c.

  **The adapter rework.** `Beatnik_MeshInterface.hpp` and
  `Beatnik_Communication.hpp` rewritten against Tessera as it actually now
  stands. The previous revision's "M1 GAP" text described a Tessera that no
  longer exists — eight of the eleven gaps had closed upstream — so most of it
  was replaced by the real call rather than merely annotated. What drove each
  change:

  1. **R8 retired.** `distribute( mesh, halo, faceOwner, depth )` exists, so
     `SurfaceMesh::halo_depth = 2` is a compile-time constant passed exactly once
     from `distributeReplicated()`, and `refine()`/`splitEdges()`/`migrate()`
     preserve it. All the old short-row and exchange-twice guidance is gone;
     `buildVertexStencil` now **throws** on `k > haloDepth()`, so the two-ring
     stencil is either correct or loud. `haloDepth()` is exposed so a test can
     assert the depth rather than infer it from a stencil that happens not to be
     short — and the new test does.
  2. **The obsolete refinement contract dropped.** `Tessera::refine()` calls
     `rebuildHalo()` itself now, so the documented `refine` → identity `migrate`
     → `haloExchange` sequence is gone from `refine()`, from `redistribute()`
     (which is no longer forced to serve as the re-halo), and from
     `haloExchange()`'s precondition. The `@pre` there now says the plans are
     *always* live, which is a stronger and simpler statement.
  3. **The disjoint editing families.** Noted on all four affected declarations
     (`refine`, `splitEdges`, `collapseEdges`, `flipEdges`), each stating that
     the families are mutually exclusive per mesh and that violation is a throw
     from inside Tessera and **not** a Beatnik-side check. Analysed — **not
     resolved** — under "T4a/T4b — the disjoint editing families" above: four
     options, what each costs Beatnik, what each assumes that was actually
     verified in Tessera's README or task files, and what remains unknown. The
     one measurement that would discriminate is named there. Nothing in T1b's
     code is shaped around a presumed winner.

  Beyond those three, the stale gap text for G3 (reductions), G4 (face
  adjacency), G5a (`splitEdges`), G6 (distributed build), G7 (lat/lon) and G8
  (load-balance modes) was replaced by the real calls and their real caveats;
  `reconcileRefinementMarks` now documents that Tessera runs the fixpoint and is
  left throwing deliberately, so a T4a caller cannot keep a reconciliation step
  and believe it is doing something.

  **Six signature changes, every one forced by Tessera's storage model rather
  than by convenience.** Recorded here because the headers are the deliverable
  and a silent reshape would be the wrong way to do this:

  | Was | Now | Why it could not stay |
  | --- | --- | --- |
  | `Comm::haloExchangeField( mesh, field )` | **deleted** | Tessera's gather is whole-tuple; there is no per-field exchange and cannot cheaply be one. Kept as a shim it would have implied a cost model Beatnik does not have. M2 precedent (`gatherForCheckpoint`). Its two doc cross-references (T2b's `surfaceGradient`, `ZModelSolver`) were repointed rather than left dangling. |
  | `Comm::haloScatterAdd( mesh, field )` | `haloScatterAdd<FieldId>( mesh )` | Tessera accumulates a field **inside** the mesh AoSoA, addressed by its compile-time Cabana member index. An external `Kokkos::View` has no such index, so the old signature is not implementable by anyone. |
  | `MeshGeometry::compute( vertices, faces )` | `compute( vertices, vertex_count, faces )` | `position_slice` is a Cabana slice behind a generation guard that forwards `operator()` only — it exposes no extent, so `Nv` cannot be recovered from it. |
  | `edgeLengths( vertices, faces, lengths )` | `edgeLengths( vertices, edges, lengths )` | The Python's body *is* the derivation of the unique edge set from faces, because NumPy has no edge list. Tessera maintains that set as a first-class entity kind, so rederiving it — on device, where a hash set is exactly what one does not want — would reimplement storage the mesh already has. |
  | `areaWeightedMean` returning `Real` | kept, **plus** `areaWeightedMeanPartials` | A mean is not reducible: `allReduceSum` of per-rank means is not the global mean. The single-`Real` form cannot express reduce-both-then-divide, so the distributed path needs the partials. The original is kept and documented as whole-surface-only. |
  | `edgeAdjacency()` → `void`, `faceAdjacency()` → `AdjacencyCsr` | `EdgeFaceIncidence`, `FaceAdjacencyCsr` | Both were throwing stubs whose return types could not carry the answer. Edge-to-face needs an incidence **count** (the closed-surface check must not be confused by a non-resident face); face-to-face has two halves and `numNonResident` is the precondition a geometric consumer must check rather than assume. |

  **Two semantic decisions, both about distributed assembly.**

  - **Beatnik needs no scatter-add for its geometry, and the four pre-T1b
    `@note MPI` blocks that said it did were wrong.** Tessera's local face set is
    *the owned faces plus every face incident on an owned vertex*, so a loop over
    **all locally held faces** gives every owned vertex its complete
    incident-face set with no communication and no double-counting. A
    scatter-add after such a loop would **double-count**. The rule, its two
    corollaries (pass the whole local set to `compute`/`volumeGradient`; pass the
    **owned** range to `enclosedVolume`/`edgeLengths`) and the fact that the two
    conventions are opposite are stated once in `Beatnik_MeshGeometry.hpp`'s
    header and cross-referenced from the routines. `haloScatterAddVertexField`
    is kept for the other pattern (owned-face loop into a mesh-resident field),
    which nothing in T1b uses.
  - **Orientation is verified, not repaired.** The Python re-winds each face
    against the outward direction (`icosphere_mesh` 452-461). `generateIcosphere`
    instead reduces the enclosed volume and **throws** unless it is positive: a
    Tessera generator that needs repairing is a Tessera defect to report, and
    silently absorbing it would hide the one failure mode that flips every normal
    and curvature downstream with nothing else noticing.

  **T1b's numbers, and how good they are.** `Beatnik_Test_MeshGeometry` at
  1 rank on HIP: 15/15, and both T1a reference scalars matched to all 17 printed
  digits, not merely to `1e-14`. That is better than expected and has a cause
  worth recording: **Tessera's icosphere base table and its 20-face list are the
  same literals as the Python's**, and its midpoint rule
  `normalize3( 0.5*(a+b) )` differs from the Python's `(a+b)/‖a+b‖` only in
  multiplying by a reciprocal where NumPy divides. So risk R1's worst case — a
  generated mesh that disagrees with the gold mesh for reasons unrelated to the
  solver — does **not** bite for the icosphere, and the T1b′ fallback task is not
  needed. The `latlon` half of R1 stands: Tessera documents its lat/lon positions
  as not bit-reproducible across libm implementations.

  **Two things only running revealed.**
  1. **A manifest-relative-path bug in `unit_tests.flux`, whose failure mode was
     worse than a plain failure.** The installed manifest names its script *and
     its data files* relative to the manifest's own directory; the first version
     prefixed only the script and ran from the submitting cwd. The positive
     comparator case then failed loudly on a missing fixture — but the
     `py-fail` negative case **passed for the wrong reason**, because a missing
     file also exits non-zero. The tally read 2/3 instead of 1/3. Fixed by
     running the whole invocation from the manifest directory; the reasoning is
     in a comment there because the class of bug (a WILL_FAIL case passing
     vacuously) is invisible in a green log.
  2. **No compile errors at all**, again, as at V0 — so no semantic decision was
     forced by the compiler and none is recorded on that account.

  **New and changed build/test wiring.**

  - `tests/unit_tests/` **created as a tier**: `Beatnik_TestAssert.hpp` (a
    header-only recorder: boolean / exact-integer / relative-tolerance checks,
    failures accumulated not aborted, a greppable `[PASS|FAIL] <name> (p/n
    checks)` tally, exit 0 only if every check passed *and at least one ran*),
    `Beatnik_Test_MeshGeometry.cpp`, and a `CMakeLists.txt` registered from
    `tests/CMakeLists.txt`. Deliberately **not** gtest and not through
    `test_harness.cmake`: in spack mode there is no build tree and therefore no
    ctest, so a unit test must be authoritative about its own verdict. That one
    property serves both modes, since ctest's default success criterion *is* exit
    code zero — so `add_test` needs no `PASS_REGULAR_EXPRESSION`.
  - **Both tiers install under `+testing`, and the variant needed no packaging
    change.** Verified by reading the spack package rather than assuming:
    `+testing` already sets `Beatnik_ENABLE_TESTING` **and**
    `Beatnik_INSTALL_TEST_EXECUTABLES`, and `setup_run_environment` already
    prepends `share/Beatnik/tests` to `PATH` — the same pattern Tessera's
    `package.py` uses for `Tessera_INSTALL_TEST_EXECUTABLES` /
    `share/Tessera/tests`. So `package.py` is unchanged. What was missing was the
    regression tier's *data*, which no target owns; `tests/CMakeLists.txt` now
    installs it. **Structure is preserved rather than made
    relocatable-by-lookup**: the installed tree mirrors the repo's `tests/`
    layout under one root, so the same relative paths work against either, and
    `compare_output.py` — which takes both files as explicit CLI arguments and
    performs no lookup — needed no change. Verified by listing the prefix, not
    inferred from a clean install:

    | Installed path (under `$(spack location -i beatnik)`) | What |
    | --- | --- |
    | `share/Beatnik/tests/Beatnik_Test_MeshGeometry` | the unit binary, on `PATH` |
    | `share/Beatnik/tests/beatnik_unit_manifest.txt` | unit tier, `exe` / `py-pass` / `py-fail` lines |
    | `share/Beatnik/tests/beatnik_gate_manifest.txt` | regression tier (still empty) |
    | `share/Beatnik/tests/regression_tests/compare_output.py` | the comparator |
    | `share/Beatnik/tests/regression_tests/fixtures/synthetic_{gold.npz,match.h5,perturbed.h5}` | its fixtures |
    | `share/Beatnik/tests/regression_tests/initial_conditions/gold.npz` | the T1a gold, for T1c |

    `make_fixtures.py` is deliberately not installed: it regenerates the
    fixtures and is a development tool, not something a run needs.
  - `scripts/tuolumne/unit_tests.flux` **runs the whole tier and fails the job if
    any test fails.** It discovers its tests rather than naming them, so T2b's
    and T2c's land in it for free. `ctest -L unit` in tree mode, the manifest
    otherwise. The manifest's three line kinds exist because the tier is not
    homogeneous: `py-fail` is where ctest's `WILL_FAIL` has to live when there is
    no ctest, and getting it wrong is how a green log hides a broken comparator
    (see above). `BEATNIK_UNIT_TARGETS` is accumulated in
    `test_harness.cmake` as well as in `tests/unit_tests/`, so one manifest
    covers both registration styles.
  - **`clangformat.sh` is still not safe to run repo-wide**, for the reason M2
    recorded. The edits here were written to the format and then measured
    per file: `Beatnik_MeshInterface.hpp` 27→**0**, `Beatnik_Communication.hpp`
    5→**0**, `Beatnik_MeshGeometry.hpp` 23→**23**, `Beatnik_ZModelSolver.hpp`
    0→0, and both new test files 0. No new drift anywhere, and two files are now
    clean. One trap found: reflowing a comment paragraph is safe, reflowing a
    `\f[ … \f]` display-math block or a markdown table is not — three math blocks
    in T2b's untouched docs were briefly mangled and restored, and two tables had
    to be *narrowed* rather than wrapped.

  The ship gate is untouched: the new test is `unit`, the `regression` tier is
  still empty, and `Beatnik_Test_PythonCompare` still passes with
  `_Negative` still failing as its `WILL_FAIL` expects (both now run from the
  install prefix as well as from ctest).

- 2026-08-12 — **T1c complete. THE SHIP GATE NOW HAS A MEMBER.** Next: T2a
  (generate the 5-step gold files), then T2b.

  **The gate change.** `Beatnik_Test_InitialConditions` is registered in the
  `regression` tier — regression test 1, the whole driver path at 0 timesteps
  against the T1a Python gold checkpoint. Before this the tier was empty and
  `run_regression_minset.flux` reported PASS having launched nothing; that note
  is now gone from the wrapper, from `tests/CMakeLists.txt` and from README
  "Known Issues". **What must pass before anything ships has changed**, which was
  pre-authorized for this task. What the gate covers is still only what exists:
  mesh generation, the initial condition, and the checkpoint write. There is no
  timestep and no adaptivity, so a green gate does not say the solver integrates
  anything.

  The gate ran **twelve launches** — SERIAL and HIP at ranks 1-6 — and all
  twelve passed. The unit tier is unchanged and still green: 3/3, with
  `Beatnik_Test_MeshGeometry` 15/15, `Beatnik_Test_PythonCompare` passing and
  `_Negative` failing as its `WILL_FAIL` expects.

  **The measured numbers** are in T1c's completion note above and under R2. The
  short version: `initial_min_edge` is bit-identical to the Python at every rank
  count on both backends; `initial_volume` spans 2 ulp (`4.4e-16` relative) and
  hits the T1a value bitwise in 9 of 12 configurations; the comparator's worst
  vertex error is `5.551115e-17`. No tolerance was changed anywhere.

  **The scalar-count contradiction, resolved.** The document said "four carried
  scalars" (T1a) and "the five scalars" (M2's table) and described four in M2's
  prose. It is **five**, and the authority is not prose: the gold `.npz`'s 0-d
  keys and `compare_output.py`'s `REQUIRED_FIELDS`, both read directly. `step`
  (`int64`) is what the count of four dropped. Corrected in T1a's note above and
  in `Beatnik_IOInterface.hpp`'s prose; M2's table was already right.

  **Five signature changes. Four forced, ONE for convenience — labelled as
  such.**

  | Was | Now | Why it could not stay |
  | --- | --- | --- |
  | `SurfaceState` owning three `Kokkos::View`s | **no storage**; every method takes the mesh | M1 booked this and deferred it here. Under the vertex user field pack the three fields *are* slots in Tessera's AoSoA, and a Beatnik-side copy is silently dropped by `refine()` and silently stale after `migrate()`. The views were also **never allocated**, so every accessor returned an empty view — see "only running revealed" below. |
  | `SurfaceState::resize( vertex_count )` | `initializeFields( mesh )` | Tessera owns the allocation, so there is nothing a vertex count is for. What is left is *initialization*, and its contract **inverted**: the old doc said "called after every mesh edit", which is now actively wrong — `refine()` interpolates the pack, so zeroing afterwards would destroy the solution. A rename was the only way to stop that doc-comment being followed. |
  | `SurfaceState::remap( edit )` | **deleted** | It was written against `MeshEditResult`, which M1 deleted. Tessera transfers the pack itself, so there is no parent map to consume and no work to do. Same precedent as `gatherForCheckpoint` (M2) and `haloExchangeField` (M1): delete rather than keep a shim no caller could correctly use. |
  | `RestartReader::coldStart( const mesh_type& )` | `coldStart( mesh_type& )` | `faceVertices()`/`edgeVertices()` build and cache `Tessera::MeshGeometry` against `generation()` on first call, so they cannot be `const` and neither can any caller. Forced by the M1 storage model (connectivity is gids; the local-index views are derived). |
  | the four proximity distance/factor fields in the example's `ClArgs` | in `RemeshParams` | **Forced, though it looks like tidying.** `Solver::setup`'s documented step 3 is the resolution against `initial_min_edge`, and the solver is handed only a `SolverParams` — so a factor living in the driver's own struct was unreachable at the one place able to use it, and step 3 could not have been written at all. CLI names and defaults unchanged. |
  | `SurfaceMesh` exposing no Tessera mesh | `tesseraMesh()` / `tesseraHalo()` | `Tessera::writeMesh`/`readMesh` are whole-mesh operations over storage, connectivity, ownership and the user pack at once; no subset of the facade can stand in. Scoped to the sibling adapters, which the contract already permits ("no **other** Beatnik header"), and documented as a deliberate hole with a named caller. |

  **Three things only running revealed**, and one of them was the framework's
  fault:

  1. **`SurfaceState`'s three views were unbacked, and nothing had ever noticed.**
     M1 recorded it as a follow-up and T1c was the first code to *use* the state,
     so this is the first task that could have hit it. Had the views been left
     alone and merely allocated, the result would have been a checkpoint whose
     `/vertices/u<N>` datasets were Tessera's uninitialized storage while
     Beatnik's own arrays held the real values — a plausible-looking file that
     fails the comparison for a reason no field name points at.
  2. **A read-only working directory in the installed gate path.** The gate
     wrapper runs each test *from the manifest's directory* so manifest-relative
     data paths resolve (the fix `unit_tests.flux` already carries), and that
     directory is inside a spack install prefix. A test that writes output
     therefore cannot default to `.`, and would fail **only** on the installed
     path and never under `ctest`. The wrapper now exports an absolute
     `BEATNIK_TEST_SCRATCH` under the submitting directory and the test resolves
     `BEATNIK_TEST_SCRATCH` → `TMPDIR` → `.`.
  3. **`clangformat.sh` is still not safe repo-wide, and this run has the receipt.**
     Running clang-format over the touched files mangled `Beatnik_IOInterface.hpp`'s
     markdown schema table (`| By |` split across two lines) and
     `Beatnik_InitialCondition.hpp`'s three-way ASCII branch diagram. Reverted;
     the code regions were formatted by explicit `--lines=` ranges and the prose
     was narrowed by hand, as M2 and T1b did. Measured per file: **no new drift
     anywhere, and three files improved** — `Beatnik_Solver.hpp` 10→**1**,
     `Beatnik_IOInterface.hpp` 3→**1**, `Beatnik_SurfaceState.hpp` 0→0,
     `Beatnik_MeshInterface.hpp` 0→0, `Beatnik_Restart.hpp` 0→0,
     `Beatnik_InitialCondition.hpp` 2→2, `Beatnik_Params.hpp` 1→1,
     `InputFile.hpp` 351→351, `adaptive_mesh_bubble.cpp` 23→23,
     `Beatnik_TestAssert.hpp` 0→0, and the new test file 0. (Counted with
     clang-format 21, not the 14 the CMake `find_package` asks for, so these are
     comparable to each other and to HEAD but not to M2's and T1b's numbers.)

  **No compile errors that forced a semantic decision** — the third time running
  in a row (V0, T1b, T1c). The two build failures were both mine and neither was
  ambiguous: a generated shim `#include`d the test source relative to `tests/`
  instead of `tests/regression_tests/`, and the shared `_beatnik_compare_dir`
  path variables were defined below their new first consumer. Recorded because
  "no compile errors" keeps being true and is worth knowing about this framework,
  not because these were interesting.

  **New and changed build/test wiring.**

  - `tests/regression_tests/Beatnik_Test_InitialConditions.cpp`, **one binary per
    enabled backend** — `Beatnik_Test_InitialConditions_MPI_{SERIAL,OPENMP,HIP}`,
    generated from one source by a per-backend shim that pins
    `BEATNIK_TEST_EXEC_SPACE`. Not one binary on `DefaultExecutionSpace` the way
    the `unit` tier does it, and the reason is mechanical rather than stylistic:
    the gate selects a backend by the target's `_<BACKEND>` suffix, so a
    suffix-less target is **skipped entirely** by the installed path — a silent
    zero-test pass — and on tuolumne the default space is HIP, so one binary
    could not honestly answer for SERIAL.
  - **How a multi-rank `regression` entry is represented** (the tier's first, so
    this sets the convention): **the rank set is a property of the gate, not of
    the test.** The test is registered once per backend and the sweep stays where
    it already lives — `BEATNIK_TEST_MPI_RANKS` for ctest, `BEATNIK_GATE_RANKS`
    in the wrapper — with the test reading its own comm size and adapting. So
    T1c's ranks 1/2/4 are a verified *subset* of the gate's 1-6, and the gate
    definition is unchanged and still single-sourced. The rejected alternative
    was a per-test rank list on the manifest line: it would let one test gate at
    fewer ranks than the gate claims, which is the "gate silently shrinks"
    failure CLAUDE.md forbids.
  - **The gate manifest format widened to `<target> [args...]`**, paths relative
    to the manifest's own directory — the same convention the unit manifest's
    `py-pass`/`py-fail` lines already use, and needed for the same reason (a
    test's data must travel with the manifest that names it). A bare target name
    is still valid, so this is a widening and not a replacement. The wrapper
    splits the line, filters the backend suffix **on field 1**, and runs from the
    manifest directory.
  - **The wrapper now fails when the manifest names nothing runnable.** It
    previously reported PASS after zero launches, which was correct while the tier
    was empty by design and is a false green now that it is not.
  - **The multi-rank verdict question, settled** —
    `Beatnik_TestAssert.hpp` deferred it to "the task that first needs one". Every
    rank calls `report()`, so the log names which rank failed, and the returned
    exit codes are reduced with `MPI_Allreduce(MPI_MAX)` at the call site. The
    reduction is deliberately **not** inside `Recorder`: a collective there would
    deadlock in exactly the case that matters most, one rank taking an exception
    path and never reaching `report()` while its peers block in the reduce.
  - **A real negative case, and the T1b trap closed by construction.** The test
    runs the comparator a second time against the deliberately mismatched
    `synthetic_gold.npz` and requires exit status **exactly 1**, not merely
    non-zero — because `compare_output.py` returns 1 for "compared and disagreed"
    and 2 for a `LoadError`. That distinction is what stops a mis-plumbed path
    passing as a detected mismatch, which is precisely how T1b's `py-fail` case
    passed vacuously. Every input path is also checked for existence first.
    Verified by reading the log, not by trusting the exit code: the negative run
    reports `vertex count: cpp=162 gold=12`.

- 2026-08-12 — **Formatting is now the user's job, not a session step.** The
  clang-format *tooling* stays exactly as it was — `.clang-format`,
  `clangformat.sh`, `cmake/FindCLANG_FORMAT.cmake`, the
  `find_package(CLANG_FORMAT 14)` call and the `cabana-format` target are all
  untouched — but CLAUDE.md no longer asks a session to conform to the formatter
  and now forbids running it: the user formats by hand when they choose to. The
  entries below calling `clangformat.sh` "not safe to run repo-wide" are part of
  why. Gate re-verified at ranks 1-6 on SERIAL and HIP.

---

# Task sequence

Tasks are ordered. Each names the headers and functions it fills in, its Python
counterpart, and its **exit criterion**. Coarse-grained tasks carry an
"Additional information needed" section, as required.

**Dependency-opening is deferred to specific tasks.** `../tessera` is opened
first in **M1**; `../canopy` first in **F1**. No earlier task should open either
— that is the whole point of the three adapter headers.

---

## V0 — Make it build and run to a stub *(do this first)* — **DONE 2026-08-07**

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

**Met 2026-08-07.** All five succeeded; step 2 was vacuous (zero compile
errors). The spack.yaml resync above was already done by the working tree's
uncommitted edits. Steps 3-4 run via the new
`scripts/tuolumne/run_v0_smoke.flux` — **not** interactively, see the login-node
rule in CLAUDE.md. Step 5 ran the two `compare_output.py` invocations directly,
since spack mode has no build tree for `ctest`. Details and the four latent
framework bugs fixed on the way are in the progress log.

---

## Phase 1 — Regression test 1: initial conditions, 0 timesteps

Compare the Python driver's startup checkpoint against Beatnik with the same
defaults. Validates mesh generation and problem setup with no dynamics at all.

### T1a — Generate the gold file *(human step, no code)* — **DONE 2026-08-07**

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
without a structural complaint when compared against itself. **Met 2026-08-07** —
`tests/regression_tests/initial_conditions/gold.npz`, self-compare at
`--rtol 1e-12 --atol 1e-14` exits 0 (162 vertices, 320 faces, `potential`,
162/162 unambiguous). See the progress log for the two carried scalars T1b must
reproduce.

### T1b — Icosphere generation and mesh geometry — **DONE 2026-08-11**

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

**Met 2026-08-11**, by the new `unit`-tier test `Beatnik_Test_MeshGeometry`
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
rework that preceded it, and the signature changes it forced are in the progress
log.

### T1c — Initial condition and checkpoint write — **DONE 2026-08-12**

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
  all. `remap` was deleted. **`updateSheetVector` is the one method of the five
  still throwing**, deferred to T2b on a stated dependency: its body *is*
  `SurfaceOperators::surfaceGradient`, which is T2b's and is itself a stub, and
  at 0 timesteps under the `potential` model the sheet vector is never read. The
  reason is written on the declaration so the next reader does not have to
  re-derive it. See the signature table in the 2026-08-12 progress-log entry.
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

**Met 2026-08-12**, by the new `regression`-tier test
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

### T4a/T4b — the disjoint editing families *(OPEN DESIGN QUESTION — read before either)*

**Recorded 2026-08-11 by the M1 adapter rework. Not resolved, deliberately.**

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

### M1 — Open `../tessera`: mesh model — **DONE 2026-08-07**

**First task permitted to read `../tessera`.** Reconcile
`Beatnik_MeshInterface.hpp` against what Tessera actually provides: the storage
model, the owned/ghost partition, adjacency, the topological edit operations, and
whether the `MeshEditResult` parent/weight scheme matches how Tessera reports
field transfer. **Rewrite the adapter; do not spread Tessera types outward.**

**Met 2026-08-07.** `src/Beatnik_MeshInterface.hpp` rewritten against the real
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

**Added since M1 was first written** (2026-08-09/10, Tessera branch
`conforming-refinement`) — the calls that close G1-G4 and G6-G8 and that Beatnik
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
disjoint and enforced.** `refine()`/`refineLocal()` are the *hierarchical* family
(`Level` authoritative, 2:1 balance, conforming closure); `splitEdges()` — and
`collapseEdges()`/`flipEdges()`/`compact()` when they land — are the *remesh*
family (`Level` advisory). A mesh is tagged on its first topological edit and
**each entry point throws** if the other family is then used on it. Beatnik's AMR
path (T4a, `refine()`) and its dynamic-remesh path (T4b, split/collapse/flip)
therefore **cannot run on the same mesh**, and the default configuration runs
both. Deciding which family Beatnik lives in — or how the two are staged — is a
design question that must be settled at T4a/T4b and cannot be deferred to the
implementation. **The four candidate resolutions, what each costs Beatnik, what
each assumes that has actually been verified, and what remains unknown are laid
out under "T4a/T4b — the disjoint editing families" in the task sequence above.**
It is noted on the `refine`, `splitEdges`, `collapseEdges` and `flipEdges`
declarations in `Beatnik_MeshInterface.hpp` so it cannot be met for the first
time as a runtime throw.

**Calls the M1 adapter rework and T1b introduced** (2026-08-11). The two tables
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

*As recorded at M1 (2026-08-07). **Eight of the eleven have since been closed
Tessera-side** (2026-08-09/10, branch `conforming-refinement`) — G1, G2, G3, G4,
G5a, G6, G7, G8. Only G5b, G5c and G5d remain open, and they are what still
blocks T4b/T4c. The calls that close the eight are in the "Added since M1"
table above.*

**G1 — No halo deeper than 1. — DONE.** Was the blocker for R8: the Beatnik RHS
is a **two-ring** stencil, and `buildVertexStencil(mesh, 2)` was *silently
incomplete* within one hop of a partition boundary. `distribute()` and
`rebuildHalo()` now take a `depth`; `refine()` and `migrate()` preserve it;
`buildVertexStencil` **throws** when `k > mesh.haloDepth()` instead of returning
short rows. Beatnik passes `depth = 2` once at setup.

**G2 — No ghost scatter-add. — DONE.** `haloScatterAdd` and the three kind-named
wrappers now exist, one named field per call. `Beatnik_Communication.hpp::
haloScatterAdd` forwards to it rather than implementing one.

**G3 — Only `globalMin`; no sum, max, or all-finite. — DONE.** `globalSum`,
`globalMax`, `globalAllFinite` and `globalOwnedVertices/Edges/Faces/Euler` are
now library calls. Beatnik's four `allReduce*` and both `global*Count()` forward
to them instead of hand-rolling `MPI_Allreduce`.

**G4 — No face→face adjacency through shared edges. — DONE.**
`buildFaceAdjacency( mesh )` is collective, built on `refine()`'s edge
coordinator, and returns both a local-index CSR and always-valid
`nbrGid`/`nbrOwner`. Serves T4a's mark growth (topological half, no precondition)
and T4b's proximity-exclusion rings.

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

**G6 — The coarse mesh is replicated on every rank before it is cut. — DONE.**
`buildIcosphereDistributed` and `buildFromTriangleSoupDistributed` build without
any rank holding the global mesh, and produce bitwise-identical geometry to the
replicated path. Irrelevant at the default subdivision 2 (162 vertices), so
Beatnik keeps `buildIcosphere` + `distribute` there and switches only for a large
initial mesh.

**G7 — No lat/lon sphere generator. — DONE.** `generateLatLonSphere` /
`buildLatLonSphere` now live beside the icosphere, with the exact-pole,
no-seam-duplicate and fixed-diagonal details pinned. `--mesh-kind latlon` is
still not on any regression path, and the libm reproducibility caveat Tessera
documents is the same one R1 raises.

**G8 — The load-balance solve is gathered to rank 0. — DONE.**
`LoadBalanceMode` now offers `GatherRoot` (the old path, kept as reference),
`Distributed` (rank 0 receives **zero** faces) and `Sampled` (`O(nparts)`), with
**`Sampled` the default** because it is the only one measured run-to-run
reproducible. T5d uses the default and reports `LoadBalanceStats::rootSolveFaces`.

#### What Beatnik must implement itself (and legitimately may)

None of these is haloing or partitioning:

- The **discretization conventions** Tessera deliberately refuses: outward normal
  orientation, the vertex-area definition, the cotangent weight fill for
  `applyStencil`, the curvature sign. This is `Beatnik_MeshGeometry.hpp`, exactly
  as scoped.
- **Scale and translate** the unit icosphere to `radius` / `center`, and verify
  the winding is outward (positive enclosed volume) rather than assume it.
- ~~The **lat/lon triangle soup** (G7).~~ Now Tessera's
  (`generateLatLonSphere`).
- ~~The four **global reductions** (G3).~~ Now Tessera's; Beatnik's
  `Beatnik_Communication.hpp` wrappers forward rather than call `MPI_Allreduce`.
- **Owned-only iteration discipline** (risk R9): Tessera exposes `numOwnedX()`
  and orders entities owned-first, but enforces nothing. Owned edges *do* form a
  global partition, so the edge-length reduction has a correct answer available.

#### Two contracts the adapter now encapsulates

- **The mandatory post-refine sequence.** `Tessera::refine()` leaves each rank
  holding only its refined owned entities and **clears the halo**. A
  `haloExchange()` in between is a silent no-op on an empty plan, and a second
  `refine()` without a re-halo *throws*. `SurfaceMesh::refine()` therefore
  performs `refine` → identity `migrate` → `haloExchange` itself and never
  returns with a cleared halo. **Superseded:** `refine()` now calls
  `rebuildHalo()` itself, at the recorded depth, so the identity-`migrate`
  workaround is gone and the halo is valid on return. Drop it from the adapter.
- **Marks do not need reconciling.** Tessera runs the cross-rank 2:1
  mark-propagation fixpoint internally (`MPI_Allreduce`-guarded, hard-capped,
  round count reported). An arbitrary rank-local mask is a legal input, so
  `Beatnik_Communication.hpp::reconcileRefinementMarks` has no work left to do.

Two smaller shape changes, both recorded inline in the header: `adopt()` now
requires its arrays replicated on **every** rank (not rank 0), because
`buildFromTriangleSoup` has no communication and `distribute()` relies on
replication; and `refine()`'s mask is a **host `std::vector<char>` sized
`ownedFaceCount()`**, not a device view sized `Nf`, so a device-computed AMR
indicator must round-trip to the host.

### M2 — Open `../tessera`: HDF5 I/O — **DONE 2026-08-11**

Reconcile `Beatnik_IOInterface.hpp`. The checkpoint **schema** is fixed by the
gold files (see the table in that header) and is not negotiable; what is
negotiable is whether Tessera writes it directly, or Beatnik gathers and writes.
Also settle the dataset paths, and update `FIELD_MAP` at the top of
`compare_output.py` in the same change.

**Met 2026-08-11.** `src/Beatnik_IOInterface.hpp` rewritten against
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

### Measured at T1c (2026-08-12): the cross-rank spread of the two scalars

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

### Checked at T1c (2026-08-12), and NOT biting — with the reasoning, not just the verdict

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
