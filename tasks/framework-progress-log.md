# Beatnik port — progress log

Chronological session record for the `zmodel3d-amr` → Beatnik C++ port. Companion
to `framework.md`, which holds the design, the task sequence and the risks; this
file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the framework states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The framework says *what is true now*; the log says *how it got
that way and what was tried on the route*. Entries record semantic decisions,
signature changes and the reasons they were forced, bugs that only running
revealed, and numbers measured on real hardware — none of which is recoverable
from the code.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom — one per task, named for the task it records, so that
`framework.md` can cite it by ID. No dates: the order of the sections is the
chronology. If a session covers more than one task, name them all
(`## M1 adapter rework and T1b`); if it belongs to no task, name the topic
(`## Formatting policy`).

## Framework commits

Read the five Python sources in full. Established the conventions recorded in
`framework.md` under "What the framework commits built", and landed the four
framework commits. Discovered that the brief's named port source (`solver.py`)
is the wrong file and traced to the real sources instead; recorded in
`framework.md`. Wrote and **ran** `compare_output.py` and its fixtures; the C++
was not built (see task V0). Next: V0.

## V0 and T1a

**V0 and T1a complete.** Next: T1b (which needs M1 first).

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

**The five carried scalars.** The authoritative set is not prose anywhere — it
is the gold `.npz`'s own 0-d keys and `compare_output.py`'s `REQUIRED_FIELDS`,
and both were read directly rather than inferred:

| `.npz` key | dtype | compared |
| --- | --- | --- |
| `state_model` | `<U9` | exactly |
| `time` | `float64` | rtol/atol |
| `step` | `int64` | exactly |
| `initial_volume` | `float64` | rtol/atol |
| `initial_min_edge` | `float64` | rtol/atol |

`CheckpointIO::write` emits exactly these five under `/beatnik/`, plus
`/beatnik/vertex_field_names`.

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

## M1 — Tessera gap review

**Eight of M1's eleven Tessera gaps closed upstream.** Read
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

## M2

**M2 complete.** `Beatnik_IOInterface.hpp` rewritten against
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

Worth carrying forward: **T1c's exit criterion is unaffected but its
comparison is not free.** `compare_output.py` now has two behaviours that only
a real Beatnik file exercises (the inactive state field, the field-name
cross-check), so the regenerated fixtures were shaped to look like one:
Tessera paths, `uint64` faces, a present-but-wrong `sheet_vector`, and the
name declaration.

## M1 adapter rework and T1b

**M1 adapter rework + T1b complete.** Next: T1c.

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
   resolved** — under "T4a/T4b — the disjoint editing families" in
   `framework.md`: four
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

The ship gate is untouched: the new test is `unit`, the `regression` tier is
still empty, and `Beatnik_Test_PythonCompare` still passes with
`_Negative` still failing as its `WILL_FAIL` expects (both now run from the
install prefix as well as from ctest).

## T1c

**T1c complete. THE SHIP GATE NOW HAS A MEMBER.** Next: T2a
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

**The measured numbers** are in `framework.md`, under T1c's completion note
and under R2. The
short version: `initial_min_edge` is bit-identical to the Python at every rank
count on both backends; `initial_volume` spans 2 ulp (`4.4e-16` relative) and
hits the T1a value bitwise in 9 of 12 configurations; the comparator's worst
vertex error is `5.551115e-17`. No tolerance was changed anywhere.

**Six signature changes, every one forced by Tessera's storage model.**

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
   was narrowed by hand. Measured per file: **no new drift
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

## Formatting policy

**Formatting is now the user's job, not a session step.** The
clang-format *tooling* stays exactly as it was — `.clang-format`,
`clangformat.sh`, `cmake/FindCLANG_FORMAT.cmake`, the
`find_package(CLANG_FORMAT 14)` call and the `cabana-format` target are all
untouched — but CLAUDE.md no longer asks a session to conform to the formatter
and now forbids running it: the user formats by hand when they choose to. The
T1c entry above, recording what clang-format mangles, is part of why. Gate
re-verified at ranks 1-6 on SERIAL and HIP.
## T2a

**T2a complete — the gold files for regression test 2 exist.** Generated by the
user, not by a session; this entry records what was verified about them and what
T2b/T2c/T2d must therefore reproduce. Next: T2b (surface differential
operators), then T2c, then T2d.

**Eleven files, steps 0-10**, in
`tests/regression_tests/direct-solve-10-steps/gold/`, from

```
python examples/run_adaptive_mesh_bubble.py --steps 10 --source-quadrature vertex \
  --br-approximation direct --no-dynamic-remesh --refine-every 0 \
  --checkpoint-every-steps 1 --no-video --checkpoint-dir results
```

which is recorded alongside them in that directory's `README.md`, the same
convention `initial_conditions/` uses. **The run is 10 steps, not the 5 the task
sequence originally scoped**, so the user updated T2d's exit criterion in the
same change (ranks 1-5, all 10 timesteps, `--rtol 1e-10`) — the two now agree,
and `framework.md`'s T2a criterion was reworded to stop naming a step count it
no longer matches.

**What was checked, and why each check was worth making rather than assuming.**

| Checked | Result | Why it matters |
| --- | --- | --- |
| Key set against the T1a gold | identical, all nine keys | `compare_output.py` and its `FIELD_MAP` need no change for test 2; a differing schema would have surfaced as a `LoadError` (exit 2) mid-T2d and read like a plumbing bug. |
| Step 0 against `initial_conditions/gold.npz` | **bitwise identical** in `vertices`, `faces`, `potential`, `remesh_material_position` | Settles the one real worry about the two commands differing (see below). Also means test 2's step 0 re-tests regression test 1 exactly. |
| Comparator self-compare, steps 0/5/10, `--rtol 1e-12 --atol 1e-14` | exit 0, `162/162` unambiguous, `320/320` faces identical after remap | R10's headroom: at step 10 the positions still quantize unambiguously at the default `--match-eps 1e-9`. |
| Vertex/face counts per step | 162/320 at every step | The stated intent of `--no-dynamic-remesh --refine-every 0`. Any growth would mean adaptivity leaked into the test that exists to exclude it. |
| `state_model` per step | `potential` everywhere | T5c's sheet-vector path is not implicated. |
| The two carried scalars per step | constant, bit-identical to T1a | Nothing re-bases them mid-run, so R2's "carried for the whole run" statement holds for this configuration. |

**The command difference is a non-issue, and here is the reason.** T1a's recorded
command passes `--A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --viscosity-mode
laplace-beltrami --isotropic-cleanup` explicitly and T2a's does not, which looks
like the two gold sets describe different physics. They do not: every one of
those is the Python's `parse_args` default (`--A` 0.3, `--g` 1.0, `--eps` 0.025,
`--mu` 0.002, `--viscosity-mode laplace-beltrami`, `--isotropic-cleanup` a
`BooleanOptionalAction` defaulting True). The bitwise step-0 agreement above is
the confirmation, not the argument. `--isotropic-cleanup` is moot here anyway
with remeshing off.

**The adaptive dt is live, and this is the trap for T2d.** `time` is `0.003`
exactly at step 1 and then drifts every step —
`0.0059999881751648708`, `0.0089999408790870788`, …,
`0.029996631612342662` at step 10, i.e. the file stems are
`…p017999…`/`…p020999…` rather than round multiples. So the Python re-chose dt
from the state each step (`choose_step_dt`), and a C++ run that steps at a fixed
`0.003` will disagree on `time` immediately and on the fields shortly after,
*for a reason that has nothing to do with the RHS*. `chooseStepSize` is already
in T2d's fill-in list; the point is that it is not optional for this comparison
and must not be stubbed to a constant to "get the fields comparable first".

**No code changed and no test was added.** Regression test 2 does not exist yet —
the gate still has exactly one member — so nothing was run beyond the three
comparator self-compares, which are pure Python on committed fixtures.

## T2b

**T2b complete — the seven surface differential operators and
`SurfaceState::updateSheetVector` are implemented and validated against the
read-only Python reference.** Next: T2c (vertex quadrature + the direct BR
solver), then T2d.

**Two decisions taken as given by the task, recorded here because they are the
kind a later reader will otherwise reopen.**

1. **`updateSheetVector` is T2b's scope, not T5c's or T2d's.** Its body *is*
   `surfaceGradient` plus \f$S = -\hat n_v\times\nabla_s\phi\f$, so implementing
   the operator and leaving the one-line consumer throwing would have handed T2c
   a stub the design already treats as discharged. `framework.md`'s T2b fill-in
   list omitted it while the same document's "What is NOT yet true" section and
   `Beatnik_SurfaceState.hpp:211-227` both said it was deferred here; the list is
   corrected, with the `mesh_solver.py::potential_sheet_vector` (364-367)
   citation the other entries carry.
2. **The two graph Laplacians take the vertex one-ring CSR, not `faces`.** The
   reference averages over the **unique** neighbour set
   (`mesh_solver.py:989-1001`, `:1004-1017` build a `set` per vertex); a per-face
   scatter visits every interior neighbour twice, once from each of the two faces
   sharing that edge. On a closed manifold the double count cancels between
   numerator and denominator, so the two agree algebraically — but not bitwise,
   and the cancellation rests on every edge having exactly two incident faces,
   which the reference never asserts and a partially held ghost row need not
   satisfy. `SurfaceMesh::vertexOneRing()` is that unique set already
   (`Tessera::buildVertexStencil( mesh, 1 )`, ascending unique local indices), so
   the argument does not have to be made at all.

**Three signature changes. Two are decision 2 above; the third is a widening
forced by the call T2d actually makes.**

| Was | Now | Why it could not stay |
| --- | --- | --- |
| `graphLaplacianScalar( faces, values, vertex_count, result )` | `graphLaplacianScalar( one_ring, values, result )` | Decision 2. A per-face scatter is not the reference operator, and `vertex_count` is redundant once the CSR is the argument — `result.extent(0)` is the range and the CSR's `offsets` covers it. |
| `graphLaplacianVector( faces, values, vertex_count, result )` | `graphLaplacianVector( one_ring, values, result )` | Same, componentwise. |
| `cotangentLaplacianScalar( ..., const ScalarView& values, ..., ScalarView& result )` | `..., const ScalarView& values, ..., OutScalarView& result` | One template parameter for both cannot express the call the viscous term makes: `values` is `mesh.potential()`, a Cabana slice, and `result` is a Beatnik-owned `Kokkos::View`. **Not forced by a T2b compile failure** — T2b's test can pass a `View` for both — but forced by T2d, and a widening rather than a break, so every conceivable pre-T2b call still compiles. `graphLaplacianScalar`/`Vector` got the same split for the same reason. |

**Two internal shapes worth knowing, neither a signature change.**

- **`Nv` comes from the OUTPUT view everywhere**, never from the input field.
  Every per-vertex input on these paths may be a Cabana slice
  (`mesh.potential()`, `mesh.sheetVector()`), which under the M1 storage model
  exposes no extent — the same constraint that forced `MeshGeometry::compute` to
  take an explicit `vertex_count` at T1b. `projectTangent` takes it from
  `vertex_normal` instead, since it has no output view of its own.
- **Two device helpers, `faceGradient` and `faceCotangents`**, both
  `KOKKOS_INLINE_FUNCTION`. `surfaceGradient` fuses the per-face gradient into
  its own face loop rather than materializing an `(Nf,3)` temporary it would
  immediately reduce away, and `meanCurvatureNormal` is `cotangentLaplacianScalar`
  applied to the positions — so without shared kernels the 2x2 Gram solve and the
  corner-to-opposite-edge cotangent pairing would each exist twice. Both are the
  sign-critical parts.

**One bug that only running revealed, and it was in the test wrapper rather than
in the operators — with a failure mode worse than a plain failure.**

`scripts/tuolumne/unit_tests.flux` read the installed manifest **on stdin**, and
`flux run` inherits and consumes its caller's stdin. The first launched binary
therefore swallowed every remaining manifest line. The tier reported

```
[unit] SUMMARY: PASS (3/3 tests)
```

— green, self-consistent, and silently missing the test the task exists to add.
Latent and invisible for as long as the tier had a single `exe` line (T1b's), and
found the moment T2b added the second. Fixed by reading the manifest on **fd 3**,
which protects every line kind rather than only the one that bit.

**`run_regression_minset.flux` had the identical bug** — the same
`while read` / `flux run` pattern, fed by a heredoc — and it is the *gate*. With
one member per backend it could never bite; with T2d's regression test 2 it would
have run only the first member of each backend and still reported
`[gate] PASS`, which is precisely the "gate silently shrinks" failure CLAUDE.md's
minimum-test-set rule exists to prevent. Fixed here rather than left for T2d to
rediscover, and the gate was re-run afterwards to confirm the change is inert
while the tier has one member: **12 launches** (SERIAL and HIP at ranks 1-6),
`[gate] PASS (label=regression)`, unchanged from T1c.

**No compile errors at all** — the fourth task running (V0, T1b, T1c, T2b), so no
semantic decision was forced by the compiler and none is recorded on that
account.

**What was measured, and the one place the exit criterion cannot be met as
written.**

`tests/unit_tests/Beatnik_Test_T2bOperators.cpp`, 1 rank on **HIP** (the default
execution space; the `unit` tier registers one suffix-less binary):
**31/31 checks**, tier 4/4. Every `kPy*` literal in it was computed by calling
the read-only reference directly on
`mesh.icosphere_mesh( subdivisions=2, radius=0.25, center=(0,0,0.25) )` — the
Python's own defaults, i.e. the mesh T1a's gold file describes — and every one is
an **order-invariant summary scalar** (a max, a min, a sum of magnitudes), so the
test does not have to match Beatnik's vertex numbering to the Python's. **No
tolerance was touched and no reference number was adjusted.**

| Quantity | Beatnik (HIP, np1) | Python | rel |
| --- | --- | --- | --- |
| `surfaceGradient` max\|g\| | `1.3193451648051979` | `1.3193451648051981` | 1.7e-16 |
| `surfaceGradient` sum\|g\| | `167.62467266803063` | `167.62467266803063` | 0 |
| `surfaceGradient` max\|g - P_v a\| | `0.023416899365234698` | `0.023416899365234726` | 1.2e-15 |
| `meanCurvatureNormal` min\|H\| | `7.9184808270587634` | `7.9184808270587519` | 1.5e-15 |
| `meanCurvatureNormal` max\|H\| | `9.0760095262647997` | `9.0760095262648015` | 2.0e-16 |
| `meanCurvatureNormal` mean\|H\| | `8.0177647933837246` | `8.0177647933837228` | 2.2e-16 |
| `meanCurvatureNormal` max\|cos+1\| | `1.3907208743912935e-04` | `1.3907208743912935e-04` | 0 |
| `cotangentLaplacianScalar` max\|.\| | `11.832644731433692` | `11.832644731433687` | 4.2e-16 |
| `cotangentLaplacianScalar` energy form | `-0.91960120772791898` | `-0.91960120772791909` | 1.2e-16 |
| `graphLaplacianScalar` max\|.\| | `0.016833076545600918` | `0.016833076545600924` | 3.6e-16 |
| `graphLaplacianVector` sum\|.\| | `1.9593746256423525` | `1.9593746256423525` | 0 |
| `updateSheetVector` max\|S\| | `1.3193451648051979` | `1.3193451648051979` | 0 |
| `updateSheetVector` sum\|S\| | `167.62467266803063` | `167.62467266803066` | 1.7e-16 |

Agreement is at the **1e-15 level or better everywhere**, three decades inside
the `1e-12` the criterion asks for — the same pleasant surprise T1b recorded, and
for the same reason: Tessera's icosphere positions and the Python's differ only in
their last bits.

The exact identities, all bounded a priori at `1e-13` absolute and all measured
at or below `1.7e-15`:

```
faceScalarGradient   max|g_f - P_f a| = 1.7056134324626197e-15   (exact per face)
surfaceGradient      max|g . n_v|     = 1.5619656867989929e-16   (projection)
projectTangent       max|v . n_v|     = 3.0631241924871614e-16
projectTangent       max|v - P_v a|   = 2.4825341532472731e-16
updateSheetVector    max|S . n_v|     = 9.8860206384425047e-17
updateSheetVector    max|(g x S).n + |g|^2| = 4.4408920985006262e-16
cotangentLaplacianScalar(const) max|.| = 0            (EXACTLY, not to tolerance)
graphLaplacianScalar(const)     max|.| = 0            (EXACTLY)
```

`cotangentLaplacianScalar` and `graphLaplacianScalar` of a *constant* field are
asserted as exact equalities rather than against a tolerance, and they hold:
every contribution is a weight times an exact zero, however large the cotangent
weight. That check is what catches a stencil which forgot to difference — one
accumulating \f$w\,\phi_j\f$ rather than \f$w(\phi_j-\phi_i)\f$ — which every
non-constant test would pass.

**The sign, which is the half of the criterion that matters most.**
`meanCurvatureNormal` is strictly inward at **all 162 vertices** — checked as
\f$\Delta_{LB}x \cdot \hat n_{\text{out}} < 0\f$ against the *exact analytic*
outward normal \f$(p-c)/\|p-c\|\f$, which is available because every icosphere
vertex lies exactly on the sphere and which therefore depends on nothing under
test. Zero violations. Magnitude: mean `8.0178` against \f$2/R = 8\f$ exactly,
i.e. **0.22% high**, well inside the `1e-1` discretization bound the test states
a priori from \f$(h/R)^2 = 0.076\f$. Direction: antiparallel to within
`1.39e-04`. `updateSheetVector`'s sign is pinned separately by the signed
identity \f$(\nabla_s\phi\times S)\cdot\hat n = -\|\nabla_s\phi\|^2\f$, which
holds to `4.4e-16` and is negative everywhere — a magnitude check cannot see the
direction of rotation within the tangent plane, and flipping the minus sign
reverses it while leaving every \f$|S|\f$ unchanged.

**The exit criterion's second half cannot hold as literally written, and that is
a property of the operator rather than of this port.** It asks that
"`surfaceGradient` of a linear function reproduces its tangential projection to
`1e-12`". `surfaceGradient` is an **area-weighted average of the per-face
in-plane gradients, projected onto the vertex tangent plane**. For a linear
\f$\phi = a\cdot p\f$ the *face* gradient is exactly \f$P_f a\f$ — and that
exactness is now checked, at `1.7e-15` — but the average of \f$P_f a\f$ over
faces tilted relative to each other is not \f$P_v a\f$, and the projection
afterwards does not repair the difference. The discrepancy is
\f$O((h/R)^2)\f$; the reference itself measures **2.3416899365234726e-02** on
this mesh, i.e. 1.7% of \f$|a|\f$. No correct implementation makes it `1e-12`,
and the number is not a Beatnik artifact.

So `1e-12` is spent on the two statements that *are* true and that a wrong
implementation fails: Beatnik reproduces the reference's `surface_gradient` of the
same linear function on the same mesh — **that discrepancy scalar included** — to
`1.2e-15`, and the exact half of the claim, that the result carries no normal
component, holds to `1.6e-16`. Both are tighter than the criterion in the
respects where tightness is meaningful. **The criterion was not weakened to make
the test pass; the untrue reading of it was replaced by a stronger true one, and
the discrepancy is reported as a number rather than absorbed into a tolerance.**

**Affects:**

- **T2c** — `VertexQuadrature::generate` builds the BR source from the vertex
  sheet vector, which now exists and is validated: `max|S| = 1.3193451648051979`
  and `sum|S| = 167.62467266803063` on the default icosphere under
  \f$\phi = a\cdot p\f$ with `a = (0.3, -0.7, 1.1)`, tangential to `9.9e-17`. Its
  own hard-coded reference velocity should be generated from a Python state whose
  potential is set the same way, so a disagreement localizes to the kernel and
  not to the source. Note also that `updateSheetVector` leaves **ghost** rows
  holding partial sums by construction (the face-loop assembly is complete on
  owned vertices only) — the quadrature already emits owned entities only (R9),
  so nothing needs to change, but a T2c kernel that reads a ghost `S` must
  `haloExchange()` first.
- **T2d** — four things. (a) `cotangentLaplacianScalar`'s dissipative sign is
  confirmed by the energy form `sum A phi Lphi = -0.9196 < 0`, so a blow-up in
  the viscous term is not a sign error in this operator. (b)
  `meanCurvatureNormal` returns \f$-2H\hat n_{\text{out}}\f$ and must be added to
  \f$\dot x\f$ with a **positive** \f$\sigma\f$ — verified inward at every
  vertex, so `computeSurfaceTension` needs no sign flip. (c) The
  `cotangentLaplacianScalar` template widening above is what lets the viscous
  term pass `mesh.potential()` directly. (d) The `flux run`-eats-stdin bug in
  `run_regression_minset.flux` is fixed, so registering regression test 2
  alongside test 1 will actually run both — it would not have.
- **T5c** — `projectTangent` is implemented and exact (`3.1e-16`), so
  `SurfaceState::projectSheetTangent` is a one-line call. `graphLaplacianVector`
  is the sheet-vector state's viscous operator and now takes the one-ring CSR.
- **T4c** — `graphLaplacianVector` on the positions is the umbrella smoothing
  vector `tangentialRelaxation` needs; measured `max 0.012663750374617330`,
  `sum 1.9593746256423525` on the initial sphere.
- **Anyone adding a `unit` or `regression` test** — the tier wrappers now read
  their manifests on fd 3. Do not move that back to stdin.

## T2c

**T2c complete — the vertex source quadrature and the direct O(N^2)
Birkhoff-Rott solver are implemented and validated against the read-only Python
reference. THE SHIP GATE NOW HAS TWO MEMBERS.** Next: T2d.

**The gate change.** `Beatnik_Test_BirkhoffRott` is registered in the
`regression` tier alongside `Beatnik_Test_InitialConditions`, pre-authorized in
T2c's exit criterion. What must pass before anything ships has changed again,
and it now covers something the solver actually computes rather than only the
setup: the gate ran **24 launches** (2 members x SERIAL and HIP x ranks 1-6) and
all 24 passed, `[gate] PASS (label=regression)`. Regression test 1 is unchanged
and still green at every configuration.

**Four decisions taken as given by the task, recorded so a later reader does not
reopen them.**

1. **The cross-rank source exchange is a ring of `MPI_Sendrecv`, not an
   `MPI_Allgatherv`.** P steps, each rank accumulating into its *own* targets as
   each block passes, `O(N_s/P)` storage rather than `O(N_s)`. This is the
   structure `Beatnik_BRSolverDirect.hpp`'s `@note MPI` already named; both are
   correct and the ring is what scales.
2. **The ring is factored once, not written twice.** `BRSolverDirect::
   ringAccumulate` is a private member template taking the per-block kernel as a
   callable; the velocity and the Riesz scalar differ only in the contraction
   (cross versus dot) and the prefactor. The argument for factoring is not
   brevity — two copies of a collective loop are two places for a deadlock to be
   introduced independently, and the loop's invariant is that *every* rank
   executes exactly P kernel invocations and P-1 `Sendrecv` pairs regardless of
   how many sources it owns, including zero.
3. **The test's source state is a synthetic linear potential, not the initial
   condition.** After `initializeFields` the potential is identically zero, so
   the sheet vector and the induced velocity are identically zero — which every
   implementation of this kernel reproduces, including a wrong one. It uses
   T2b's field, `phi = a.p` with `a = (0.3, -0.7, 1.1)` on the same mesh, which
   makes T2b's published `max|S|` and `sum|S|` an already-validated cross-check
   on the *source* before the kernel is compared at all.
4. **`generateGradient` and `computeSurfaceRieszScalar` are in scope**, even
   though nothing calls them until `--bernoulli-scalar-mode surface-riesz`. Both
   state-model branches are implemented **as the reference writes them and not
   as one expression**: under `Potential` the surface gradient is taken directly
   (`potential_surface_riesz_scalar` 897-901), under `SheetVector` it is
   recovered as `n x S` (`surface_riesz_scalar_from_sheet` 850-853). They agree
   to roundoff on a tangential gradient — the reference itself measures
   `5.6e-17` between them on this mesh — but they are not the same expression.

**Twelve signature widenings, all one change, and it could not be deferred.**

| Was | Now | Why it could not stay |
| --- | --- | --- |
| `SourceQuadratureBase::generate` / `::generateGradient` and all six `Vertex`/`Face`/`Triangle3` overrides — `const mesh_type&` | `mesh_type&` | Every accessor a quadrature or a BR kernel needs is **non-const**: `positions()`, `potential()`, `sheetVector()` return Cabana slices of a non-const member, and `faceVertices()` calls `ensureGeometry()`, which builds and caches `Tessera::MeshGeometry` against `generation()`. A `const mesh_type&` parameter cannot read the positions, let alone the fields. Same constraint that forced `RestartReader::coldStart( const mesh_type& )` at T1c. |
| `BRSolverBase::computeInterfaceVelocity` / `::computeSurfaceRieszScalar`, and the `BRSolverDirect` and `BRSolverFMM` overrides — `const mesh_type&` | `mesh_type&` | Same, transitively: the BR solver's first act is to call `quadrature.generate( mesh, ... )`. |

`state` stays `const`, and that is a real statement rather than an omission: the
vertex rule reads `mesh.sheetVector()`, which T2d's RHS refreshes through
`SurfaceState::updateSheetVector` before each evaluation, so the quadrature never
writes the state. **Callers updated: none** — `Solver::setup` constructs both and
nothing invokes either yet, which is exactly why the widening was free here and
would not have been at T2d. `BRSolverFMM`'s two signatures were widened with the
rest and its bodies left throwing (T3a).

**One internal shape worth knowing, not a signature change.** `generateGradient`
under `Potential` allocates its `surfaceGradient` scratch over the **whole local
vertex range**, not the owned one, even though only owned rows are emitted. The
assembly is a face loop that scatters into ghost rows, and `surfaceGradient`
takes `Nv` from the *output* view — so an owned-sized scratch would index out of
bounds. The owned-only discipline (R9) applies to what is *emitted*, not to what
is assembled; those are the two opposite conventions
`Beatnik_MeshGeometry.hpp`'s DISTRIBUTED ASSEMBLY note states.

**No compile errors that forced a semantic decision** — the fifth task running
(V0, T1b, T1c, T2b, T2c). The one build failure was mine and was not ambiguous:
the test declared its output views as `Kokkos::View<Real*, MemSpace>` while the
solver's out-parameters are `View<Real*, Kokkos::Device<ExecSpace, MemSpace>>`,
which are unrelated types that do not bind. Fixed by having the test use the
solver's own `vector_view` / `scalar_view` typedefs, which is what it should have
done anyway.

**One bug that only running revealed, and it was in this task's own negative
case rather than in the solver.**

The `--br-sign -1` negative case asserts two things: the velocity must be negated
**exactly**, and the Riesz scalar must be **unchanged**, the second being what
catches a `br_sign` applied inside the shared kernel instead of on the velocity
path only. Both were written as bitwise equalities, because `br_sign` multiplies
a completed sum and negating it is a sign-bit flip and nothing else.

The velocity half is bitwise, and measured so: **0 differing components at every
rank count on both backends**. `generate` reads a sheet vector assembled once and
does no reduction of its own, so both calls sum the same source list in the same
ring order.

**The Riesz half is not, on HIP, and the first gate run failed on it at all six
rank counts while passing on SERIAL at all six** — 15, 33, 42, 45, 53 and 60 of
162 values differing. That pattern is the finding: the cause is that
`generateGradient` re-runs `surfaceGradient`, whose face-loop assembly uses
`Kokkos::atomic_add` and is documented as not bitwise reproducible under
DETERMINISM in `Beatnik_MeshGeometry.hpp`. Two identical calls therefore produce
last-bit-different gradients. SERIAL has a deterministic scatter order, which is
why it passed and why the split is per-backend and not per-rank-count.

The check was **not** deleted and **not** loosened to "close enough". What
discriminates the two explanations is the *size* of the difference: a `br_sign`
leak into the Riesz path makes it exactly `2|psi|`, i.e. `2.0` relative, while an
atomic reordering makes it `~1e-16`. So the claim is now `max|dpsi| / max|psi| <=
1e-13` — thirteen decades below what the bug it exists to catch would produce —
and the measured number is reported on every gate run either way. Measured
**`2.4e-16`**, worst over all twelve configurations.

**What was measured.** `tests/regression_tests/Beatnik_Test_BirkhoffRott.cpp`,
**29/29 checks** in each of its twelve configurations. Every `kPy*` literal was
computed by calling the read-only reference on
`mesh.icosphere_mesh( subdivisions=2, radius=0.25, center=(0,0,0.25) )` through
`potential_mesh_birkhoff_rott_velocity` and `potential_surface_riesz_scalar` at
`source_quadrature="vertex"`, `br_approximation="direct"`, `eps=0.025`,
`use_matlab_blob=False`, `br_sign=1` — so the kernel offset is `eps^2 =
6.25e-4`, both codes' `length` default. Every one is an **order-invariant
summary scalar**, so no vertex-order matching is needed and the same literals
hold at every rank count. The table of worst relative errors is in
`framework.md` under T2c's completion note; the headline is that the worst
disagreement anywhere in the sweep is **`1.30e-15`** (velocity `min|u|`), two
decades inside the criterion's `1e-13`. **No tolerance was touched and no
reference number was adjusted.**

**The signed quantities, because they are the half a magnitude comparison cannot
see.** `sum u = (-13.809091739775855, 32.221214059476992, -50.633336379178147)`,
matched to `2.8e-16` or better in every component. Reversing the cross product to
`S x delta` negates all three and leaves `max|u|`, `min|u|` and `sum|u|`
identical, which is the single most likely error in this kernel and the reason
the criterion's "compare summary scalars" was not read as "compare magnitudes".
The Riesz scalar's `min` is likewise negative and pinned, which is what fixes the
sign of its `-1/(4 pi^2)` prefactor.

**R9 was checked, not assumed, and it is not biting.** Two mechanized
discriminators, re-measured on every gate run:

1. **The owned sets partition the global sets** — `162 / 480 / 320` at every rank
   count on both backends, summed with a plain `MPI_Allreduce(MPI_SUM)` over
   `ownedXCount()` rather than read from Tessera's `globalOwnedX`, for the same
   reason T1c gave: owned-versus-local is exactly what R9 turns on and two
   agreeing paths beat one.
2. **The global source count is exactly 162** at every rank count. This is the
   *direct* detector and it is stronger than anything T1c had available: the
   ghost fraction here runs from 0 at one rank to 0.40 at two and higher at six,
   so a rule emitting the whole local vertex set would make this 200-400 and the
   assertion fails immediately — while every velocity number would have moved
   smoothly and plausibly. It is asserted, not merely reported.

**The R2 caveat on these numbers, which T3a inherits.** The ring fixes the *rank
order* of the block sums (each rank starts with its own block and walks the ring
in a fixed direction), so the summation order is deterministic given a rank
count — but it **differs between rank counts**, and the on-node
`Kokkos::parallel_reduce` tree differs between backends. The measured spread is
what that costs: each compared scalar takes 2-5 distinct values across the twelve
configurations, all within `1.3e-15` relative of the Python. That is the noise
floor of this operator at this problem size, not a budget to spend.

**New and changed build/test wiring.**

- `tests/regression_tests/Beatnik_Test_BirkhoffRott.cpp`, **one binary per
  enabled backend** — `Beatnik_Test_BirkhoffRott_MPI_{SERIAL,OPENMP,HIP}`,
  generated from one source by the per-backend shim T1c added. Not one
  suffix-less binary: the installed gate path selects on the `_<BACKEND>` suffix
  and would skip it entirely, a silent zero-test pass. The rank set stays a
  property of the gate (`BEATNIK_GATE_RANKS` / `BEATNIK_TEST_MPI_RANKS`) and is
  not on the manifest line, per T1c's convention — the test reads its own comm
  size and adapts.
- **Regression-test arguments are now keyed by source stem, not shared across
  the tier.** T1c's single `_beatnik_regression_args_{abs,rel}` pair was correct
  while the tier had one member; T2c's test takes **no** arguments, and handing
  it three paths it ignores would make its manifest line claim a dependency that
  does not exist. `tests/CMakeLists.txt` now looks up
  `_beatnik_args_<stem>_{abs,rel}` and **`message(FATAL_ERROR)`s at configure
  time if either is undefined** — so a future test added without its argument
  lists is a build failure and not a test launched silently without its data. An
  empty list emits a bare target name, which is the manifest format that already
  existed before T1c widened it.
- The test writes nothing, so it needs no scratch directory; the
  `BEATNIK_TEST_SCRATCH` -> `TMPDIR` -> `.` resolution T1c added is untouched and
  still required by regression test 1. The tier wrappers still read their
  manifests on **fd 3**.

**Formatting: `clang-format` was NOT run**, per CLAUDE.md's formatting rule and
the standing user instruction. No file was reformatted as part of this change;
the new and edited code is written in the style of its surroundings by hand.

**Affects:**

- **T2d** — five things. (a) `BRSolverDirect::computeInterfaceVelocity` is ready
  to call and its output convention is fixed: `(N_owned, 3)` over the **owned**
  vertices, **overwritten**, with `1/4pi` and `br_sign` already applied — the RHS
  must not re-apply either. (b) The out-parameter is reallocated if its extent
  does not match, so the RHS may hold one view across stages without resizing it.
  (c) The **ordering** the RHS owes the quadrature is
  `haloExchange()` -> `updateSheetVector` -> BR evaluation, and this test
  performs and checks exactly that sequence; getting it wrong is wrong only near
  partition boundaries, which is R8's seam. (d) The BR call is **collective** and
  every rank must reach it the same number of times per step — the ring
  deadlocks otherwise, including for a rank that owns zero sources. (e) The
  `mesh_type&` widening means the RHS cannot hold the mesh by const reference on
  any path that reaches the BR solver.
- **T3a** — three things. (a) `BRSolverDirect` is the baseline the FMM is
  validated against, and the eleven reference scalars above (with the Python
  values they were checked against) are its numbers on the default icosphere. (b)
  **The ring's summation order bounds how tight a cross-solver claim can be**:
  the direct solver itself spans up to `1.3e-15` relative across rank counts and
  backends, so an FMM-versus-direct comparison cannot be asserted below that even
  in principle, and its real tolerance will be the FMM's own accuracy, orders
  above. Compare at a fixed rank count where possible. (c) `BRSolverFMM`'s two
  signatures are already widened to `mesh_type&`, so T3a implements bodies only.
- **T5c** — `VertexQuadrature::generateGradient` has a `SheetVector` branch
  (`G = n x S`) that no test exercises, because the sheet-vector state model has
  no gold file until T5c. Only the `Potential` branch is validated; the other is
  implemented from the reference and unverified, and T5c is where it first runs.
- **R5** — unchanged and worth restating now that the code exists:
  `surface-riesz` + `fmm` still has no gold file. What T2c adds is that
  `surface-riesz` + `direct` now *does* have a validated reference, so R5's
  suggested mitigation (a Python `direct` + `surface-riesz` run compared against
  Beatnik `fmm` + `surface-riesz` at loosened tolerance) has a validated middle
  term.
- **Anyone adding a `regression` test** — `tests/CMakeLists.txt` now requires
  `_beatnik_args_<stem>_abs` and `_beatnik_args_<stem>_rel` to be defined (empty
  is fine) or the configure fails.

## T2d — implementation

*(Superseded on its status only: the gate has since been run and is green. See
`## T2d — completion` at the bottom for the results and for the one tolerance
that moved. Everything below about **what was built and why** still stands.)*

**T2d is written and compiles; NOTHING has been run.** This entry records what
was built, the decisions taken, and the four things the implementation forced
that were not in the task's fill-in list — so a resuming session does not have to
re-derive any of it. **No number in this entry is a measurement**, because no
measurement was taken: the session ended in the `pdebug` queue waiting for the
gate. The first thing to do on resuming is submit
`scripts/tuolumne/run_regression_minset.flux` and read the log.

**THE SHIP GATE GREW TO THREE MEMBERS.** `Beatnik_Test_DirectSolve10Steps` is
registered in the `regression` tier alongside `Beatnik_Test_InitialConditions`
and `Beatnik_Test_BirkhoffRott` — authorized by the user for this task. The gate
is now **36 launches** (3 members x SERIAL and HIP x ranks 1-6), up from 24, and
that is the practical reason this session did not close: the queue wait plus 36
launches of two full solver paths is materially longer than T2c's sweep. Budget
for it.

**Decisions taken as given by the task, recorded so a later reader does not
reopen them.**

1. **One binary per enabled backend, `_<BACKEND>` suffix**, generated by the shim
   T1c added. A suffix-less target is skipped entirely by the installed gate
   path — a silent zero-test pass.
2. **The rank sweep is a property of the gate, not of the manifest line.** The
   test reads its own comm size and adapts, so the criterion's ranks 1-5 are a
   verified subset of the gate's 1-6, exactly as at T1c and T2c.
3. **`_beatnik_args_<stem>_{abs,rel}` are defined** for the new source (two
   entries: the gold *directory* and the comparator), which
   `tests/CMakeLists.txt` requires or the configure fails.
4. **The tier wrappers still read their manifests on fd 3.** Untouched.
5. **`BEATNIK_TEST_SCRATCH` -> `TMPDIR` -> `.`** for the run directory, because
   the installed gate path runs from a read-only spack prefix.

**Four things the implementation forced that were NOT on the fill-in list.** Each
is a real signature or scope addition, not tidying:

| Added | Why it could not be avoided |
| --- | --- |
| `SurfaceState::maxSheetStrength` implemented | `chooseStepSize`'s `--max-sheet-dt-product` clamp is *documented behaviour of the function this task must port*, and its body is this call. Leaving it throwing would have shipped a clamp that aborts the run the moment it is enabled. Not exercised by the gold configuration, where the product is 0. |
| `SurfaceState::faceSheetVector` implemented | `maxSheetStrength` under `Potential` takes the max over **both** the vertex and the face sheet vectors (`run_adaptive_mesh_bubble.py:904-910`), because "the vertex-gradient diagnostic can miss triangle-scale potential jumps". Using only the vertex value lets the dt throttle miss the blow-up it exists to catch. |
| `SurfaceState::projectSheetTangent` implemented, and it now takes the **mesh** | One line (`SurfaceOperators::projectTangent` on `mesh.sheetVector()`), and the mesh parameter is forced by the same M1 storage rule that changed `centerPotential` at T1c: the field it projects lives in the mesh. T5c inherits a working call rather than a stub. |
| `Solver::mesh()` gained a **non-const overload** | Regression test 2 measures the enclosed volume every step, and `positions()` / `faceVertices()` / `edgeVertices()` are all non-const under the M1 storage model. The const accessor cannot reach them. Same constraint that widened twelve signatures at T2c. |

**Signature changes.** All of them are the T2c widening reaching its callers, plus
one constness fix:

| Was | Now | Why |
| --- | --- | --- |
| `ZModelSolver::{computeRightHandSidePotential, computeRightHandSideSheet, computeBernoulliPotential, computeSurfaceTension, computeScalarViscosity}( const mesh_type& )` | `mesh_type&` | T2c widened the BR solver and the quadrature; the RHS calls both, so the widening propagates. The sheet-vector RHS was widened with the rest **although its body still throws** (T5c), so T5c implements a body only. |
| `VolumeProjection::removeVolumeFlux( const mesh_type& )` | `mesh_type&` | Same, via `positions()` / `faceVertices()`. |
| `TimeIntegrator::chooseStepSize( const mesh_type& )` | `mesh_type&` | Same, via `positions()` / `edgeVertices()`. |
| `DiagnosticsCalculator::compute( const mesh_type& )` | `mesh_type&` | Same, via four accessors. |
| `SurfaceState::updateSheetVector(...)` non-const | **`const`** | It writes the *mesh*, never the state (which holds no storage at all), and the RHS receives its state by `const&` because the source quadrature does. Every other method on that class was already `const`; this one was the outlier. |

**Semantic decisions the port forced, each with the reference line that decides
it.**

1. **The RHS opens with ONE whole-tuple `mesh.haloExchange()`**, and the ordering
   inside it is `haloExchange()` -> geometry -> `updateSheetVector` -> BR
   evaluation, which is the sequence `Beatnik_Test_BirkhoffRott.cpp` performs and
   checks. One exchange and not two: the depth-2 halo built once in `SurfaceMesh`
   is what covers the two-ring stencil, not a second exchange (R8). The exchange
   lives in the RHS rather than in the integrator because the RHS is what has the
   precondition.
2. **The stage construction re-centres the potential AT THE NEW VERTICES, every
   stage.** The reference builds each stage through `state.with_arrays(...)`,
   which runs `MeshPotentialZModelState.__post_init__` and therefore subtracts
   the area-weighted mean (`mesh_solver.py:155-159`) — not only at the end of the
   step. `TimeIntegrator::finishStage` is `haloExchange` -> geometry -> 
   `centerPotential`, and the next stage's convex combination reads the *centred*
   value, exactly as the Python's `stage1.potential` is the centred array.
   **Dropping this changes the answer**, because the mean is subtracted from a
   field the next stage then differentiates. It costs a second geometry
   computation per stage (free at 162 vertices) and keeps the RHS's documented
   signature, which takes no geometry precisely so it cannot be handed a stale
   one.
3. **The three stage combinations are ONE kernel** parameterized by
   `(a0, a1, c)`, so the Shu-Osher weights cannot drift between stages. Stage 2
   combines with `q0`, not with its own predictor.
4. **Owned range out, local range for assemblies.** The RHS's two out-parameters
   are `(N_owned, ...)`, matching what `BRSolverDirect::computeInterfaceVelocity`
   writes (T2c) and what the integrator updates before exchanging. Every
   intermediate that is *assembled* by a face-loop scatter — the volume gradient,
   the cotangent Laplacian, the mean-curvature normal — is allocated over the
   **whole local range**, because those operators take `Nv` from their output
   view and an owned-sized output would index out of bounds on a ghost corner.
   The two conventions are opposite and both are stated in
   `Beatnik_MeshGeometry.hpp`'s header; no scatter-add is added after any
   assembly loop, which would double-count.
5. **`removeVolumeFlux` batches its two inner products into one `MPI_Allreduce`**
   and divides after, for the reason `SurfaceState::centerPotential` already
   gives: reducing separately and dividing locally gives a different scalar on
   every rank and a velocity field discontinuous across partitions. Same
   reduce-both-then-divide shape for the area-weighted mean of `phi_dot`, with
   the Python's unweighted-mean fallback also reduced so ranks cannot disagree
   about the shift.
6. **The `surface-riesz` Bernoulli input is resolved in the RHS, not inside
   `computeBernoulliPotential`**, because it is a second **collective** BR
   evaluation and does not belong inside a per-vertex kernel. It is driven by a
   parameter identical on every rank, so every rank reaches the ring the same
   number of times per step (T2c's deadlock constraint). The `normal-proxy`
   factor of 1/2 *is* applied inside, on the scalar before squaring
   (`mesh_solver.py:922-923`) — it changes the quantity squared, not the
   exponent.
7. **`computeSurfaceTension` ADDS into the velocity** rather than returning a
   field; the reference returns `None` at `sigma = 0` and the caller adds
   (`mesh_solver.py:1245-1247`), so the guard is the caller's and this is the
   same arithmetic without an `(Nv,3)` temporary. **No sign flip** — T2b verified
   `meanCurvatureNormal` inward at all 162 vertices.
8. **`computeScalarViscosity` returns `mu * Laplacian(phi)`, not the bare
   Laplacian.** The two viscosity modes are not interchangeable at the same `mu`,
   so the coefficient sits adjacent to the operator choice where that is visible.
9. **The post-step passes THROW rather than being skipped.**
   `Solver::requireSupportedConfiguration()` runs once before the loop and
   rejects `--dynamic-remesh` (T4b), `--refine-every > 0` (T4a),
   `--field-filter-every > 0` (T5c) and `--redistribute-every > 0` (T4c) by name
   and by task ID. **This changes the default example run's behaviour**: the
   Python's default is dynamic remeshing every step, so an unguarded Beatnik run
   would quietly produce a plausible trajectory that is not the reference's, and
   would then be compared against gold files generated with adaptivity on. A
   configuration check before the loop, not a mid-loop guard: the conditions are
   global and time-independent, so it is decidable at step 0.
10. **The last-finite state is a COPY of four owned arrays, not a reference.**
    Under the M1 storage model the solution lives in the Tessera vertex user
    pack, so the Python's `last_finite_state = state` has no analogue — the next
    step overwrites those slots in place. `recordLastFiniteState` copies
    positions, potential, sheet vector and material position plus the
    `(time, step)` pair after every mutation of a step that ended finite;
    `finalize()` restores them before writing. At `--steps 0` nothing was ever
    recorded and the current state is written, which is exactly T1c's behaviour
    and is what keeps regression test 1 unchanged.

**Two compile errors, both mine, and one is worth knowing about this codebase.**
The fifth task running with no compile error that forced a *semantic* decision.

1. A leftover `sum4` after a reduction block was rewritten. Trivial.
2. **`Kokkos::view_alloc( WithoutInitializing, label )` reads a DECAYED
   `const char*` as a pointer-to-memory, not as a label.** A string literal
   passed directly works, because it is still an array type at the call. A
   helper that takes `const char* label` and forwards it does not, and the
   failure is a `static_assert` several screens away
   (`Kokkos_ViewCtor.hpp:461`, "Cannot give pointer-to-memory for view
   allocation") with a `ReferenceCountedDataHandle` conversion error under it.
   Fixed by taking `const std::string&` in both `resizeScalar`/`resizeVector`
   helpers, and noted at both. Anyone factoring a Kokkos allocation behind a
   helper in this tree will hit it.

**What regression test 2 checks beyond the criterion**, so a resuming session
knows what the log will contain:

- the entity counts, Euler characteristic and **halo depth 2** (R8's structural
  half) before anything evolves, and the counts again after every step — a growth
  would mean adaptivity leaked into the test that exists to exclude it;
- **R9 discriminator 1**, the owned sets summing to 162/480/320 under a plain
  `MPI_Allreduce` over `ownedXCount()` rather than Tessera's `globalOwnedX`;
- **R9 discriminator 2**, `volume / (4 pi R^3 / 3) = 0.96616074859858714`,
  **asserted** here rather than merely reported as at T1c — the volume-drift
  check would otherwise be measured against a number that could itself be wrong;
- the two carried scalars against T1a's literals;
- the per-step `time` against a hard-coded gold literal at `1e-10` relative, so a
  constant `chooseStepSize` fails at **step 2 on `time`** and the log says so
  rather than the failure arriving as a field-table row;
- the per-step volume drift against the `1e-12` bound;
- the **last-finite round trip**: `finalize()`'s output is compared against step
  10's gold, which it cannot match if record/restore corrupts the fields or the
  `(time, step)` pair;
- a **negative case** requiring the final state versus the *step-0* gold to exit
  exactly **1** and not 2 — which additionally proves the ten steps actually
  moved the surface.

The gold file for a step is found by scanning the gold directory for the
`_step%07d.npz` suffix, **not** by rebuilding the name from a time: the time is
what is under test, and a name built from Beatnik's own `time` would compare each
step against whichever gold file Beatnik's dt happened to point at.

**Formatting: `clang-format` was NOT run**, per CLAUDE.md's formatting rule and
the standing user instruction. No file was reformatted; the new and edited code
is written in the style of its surroundings by hand.

**Affects:**

- **T2d itself, resuming** — the exit criterion is open. Submit the gate, read
  the per-step numbers, and only then debug. Expect the first run to fail
  somewhere; two of the three gate members now exercise the full solver path.
- **T3a** — `ZModelSolver::computeRightHandSidePotential` is the caller
  `BRSolverFMM` must satisfy: it invokes `computeInterfaceVelocity` once per RK3
  stage (and `computeSurfaceRieszScalar` additionally under
  `--bernoulli-scalar-mode surface-riesz`), holding one out-parameter view across
  all three stages and relying on the T2c realloc-on-mismatch contract. Whatever
  Canopy needs per call must therefore tolerate being re-entered three times per
  step with the sources moved between calls.
- **T4a / T4b / T4c** — the step loop's post-step slots exist and currently
  throw with a message naming each task. Implementing one means replacing that
  branch in `Solver::advanceOneStep`'s documented order and deleting its clause
  from `requireSupportedConfiguration()`. `VolumeProjection::projectToVolume` is
  already implemented and is **unexercised**: T4a is the first configuration that
  reaches it, so a failure there is as likely to be in the projection as in the
  refinement.
- **T4a specifically** — `Diagnostics::compute` reports the four AMR indicator
  fields as `NaN`. They are the only fields it cannot supply, and filling them in
  is a T4a-era edit to one function.
- **T5b** — `finalize()` now writes the recorded last-finite state, so a restart
  test must expect the *last finite* `(time, step)`, not the current one, after
  an aborted run.
- **T5c** — `computeRightHandSideSheet`'s signature is already widened and
  `projectSheetTangent` is already implemented, so T5c is a body plus
  `filterCirculationField`. Note the sheet-vector RHS's viscous operator is the
  **graph** Laplacian regardless of `--viscosity-mode`, and that asymmetry is in
  the reference.
- **Anyone adding a `regression` test** — the gate is now three members and 36
  launches on tuolumne. `tests/CMakeLists.txt` still requires
  `_beatnik_args_<stem>_{abs,rel}`, and the new install rule **globs** the T2a
  gold set (with a `FATAL_ERROR` if the glob is empty) rather than enumerating
  eleven time-encoded filenames.

## T2d — completion

**T2d is DONE: the whole 36-launch gate is green.** Run on 2026-08-14 directly
inside a 2-node interactive allocation (`bash
scripts/tuolumne/run_regression_minset.flux`, not `flux batch` — the allocation
was already held), `[gate] PASS`, zero `[FAIL]` lines, 36 launches = 3 members ×
{SERIAL, HIP} × ranks 1-6. `Beatnik_Test_DirectSolve10Steps` reports **107/107**
checks on rank 0 and 83/83 on every other rank in all twelve of its
configurations, so the exit criterion's ranks 1-5 are a verified subset on both
backends. `Beatnik_Test_InitialConditions` and `Beatnik_Test_BirkhoffRott` are
unchanged and green in all 12 each.

Inside regression test 2: per-step `time` matches its gold literal to `~2e-16`
at all ten steps (so `chooseStepSize` reproduces the Python's adaptive dt rather
than stepping at a constant `0.003` — T2a's trap); all ten per-step
`compare_output.py` invocations exit 0 at `--rtol 1e-10 --atol 1e-12`; the
last-finite round trip matches the step-10 gold; the negative case against the
step-0 gold exits exactly **1** and not 2; entity counts stay 162/480/320 with
halo depth 2 throughout; R9's two discriminators and T1a's two carried scalars
hold.

**Two gate runs were needed, and the first failure was not Beatnik's.**

### Run 1 — the volume-drift bound, and a scratch-directory trap

`BEATNIK_TEST_SCRATCH` was pointed at `/tmp/beatnik_gate_scratch` to keep
artifacts out of the checkout. **On tuolumne `/tmp` is a per-node tmpfs**, and
the checkpoints go through parallel HDF5 (MPI-IO), so every launch spanning more
than one node — ranks 5 and 6, at tuolumne's 4-ranks-per-node — died inside
`H5FD__mpio_open` with "File does not exist", while ranks 1-4 passed. That reads
exactly like the multi-rank solver bug regression test 2 exists to catch, and is
not one. Fixed by using `/p/lustre5/stewartj/beatnik/gate_scratch`. Recorded in
the README's Known Issues and in CLAUDE.md's "Minimum test set", because the next
session will otherwise pay for it again.

The real finding in run 1 was the one the task predicted: the **per-step volume
drift**, failing 10 checks per launch in all twelve configurations and nothing
else — growing **linearly in the step count** and **identical on both backends at
every rank count**.

### The discrimination: truncation, not a projection bug

Three facts, none of them a tolerance argument:

- **Linear in step count** — round-off accumulates as `sqrt(n)` at best and does
  not produce a clean straight line through ten points.
- **Backend independent** — SERIAL and HIP agree, so it is not the atomic face
  scatter's summation order (**R2**).
- **Rank independent** — the same number at 1 through 6 ranks, so it is not a
  halo-depth seam (**R8**) and not a partition-dependent reduction (**R9**).

What is left is RK3 truncation of a *rate-only* projection: `removeVolumeFlux`
makes `dV/dt` zero in the discrete sense, which says nothing about the volume
error the three-stage combination accumulates. The reference has the same
structure, so the claim was **measured against the reference** rather than
asserted. Computing `V = (1/6) Σ_f a·(b×c)` over `faces` — `enclosedVolume`'s own
convention — for the eleven committed gold `.npz` files, offline in numpy, with
no Beatnik build and no allocation:

| step | Python gold drift (`V/V0 - 1`) | Beatnik drift, SERIAL np=1 | deviation |
| --- | --- | --- | --- |
| 1 | `5.1898485509127568e-12` | `5.1900705955176818e-12` | `4.2784409361118492e-05` |
| 2 | `1.0375700298936863e-11` | `1.0375922343541788e-11` | `2.1400445129327039e-05` |
| 3 | `1.5557333199467394e-11` | `1.5557555244072319e-11` | `1.4272664992098782e-05` |
| 4 | `2.0734747252504349e-11` | `2.0735191341714199e-11` | `2.1417633137454928e-05` |
| 5 | `2.5907276324232953e-11` | `2.5907942458047728e-11` | `2.5712228735930154e-05` |
| 6 | `3.1075142459258132e-11` | `3.1076252682282757e-11` | `3.5727045373246114e-05` |
| 7 | `3.6238345657579885e-11` | `3.6239233835999585e-11` | `2.4509353381940713e-05` |
| 8 | `4.1396441829988362e-11` | `4.1397107963803137e-11` | `1.609157177107079e-05` |
| 9 | `4.6549430976483563e-11` | `4.6550097110298339e-11` | `1.431024613629539e-05` |
| 10 | `5.1697091052460564e-11` | `5.1698201275485189e-11` | `2.1475541505777684e-05` |

The Python's step-0 volume is `6.32350731246695136e-02`, which is
`kInitialVolume` to the last digit — the two computations agree bitwise before
anything evolves, so the drift comparison is about the trajectory and not about
the volume formula. **Every entry is positive: the reference gains volume,
linearly, exactly as Beatnik does.** So `1e-12` was written a priori and sits an
order of magnitude below what this discretization can deliver at ten steps. It
was a bound on the wrong quantity, not a bound set too tight.

### The restated criterion, and where its number comes from

`kVolumeDriftBound` is replaced by `kGoldVolumeDrift[11]` (the Python series
above, as 17-digit literals, exactly as every other reference number in this
tree), checked at `kVolumeDriftRtol = 1e-3` relative, with
`kVolumeDriftAbsCap = 1e-9` absolute kept as the blow-up detector. **This is
strictly stronger than the bound it replaces**: it fails a run that conserves
volume *better* than the Python as well as one that conserves it worse, which
`drift <= 1e-12` would have passed silently.

`1e-3` is not fitted. The drift is `V/V0 - 1` with `V ≈ V0 ≈ 6.3e-2` and a value
of `5e-12`, so **one ulp of the ratio is `2.2e-16 / 5.19e-12 = 4.3e-5` of the
step-1 drift** — a hard round-off floor no implementation can beat, shrinking by
a decade by step 10 as the drift grows. Across all 36 launches the step-1 drift
takes exactly **three** distinct values, one ulp apart
(`5.1898485509127568e-12`, `5.1900705955176818e-12`, `5.1902926401226068e-12`),
so the largest deviation anywhere in the gate is two ulps,
`8.5568818722459028e-05`. `1e-3` is a little over a decade above that. The HIP
np=6 series, for contrast with the SERIAL np=1 column above, is
`5.1898485509127568e-12`, `1.0375700298936863e-11`, `1.5557555244072319e-11`,
`2.0734747252504349e-11`, `2.5907498368837878e-11`, `3.1075586548467982e-11`,
`3.623901179139466e-11`, `4.1396885919198212e-11`, `4.6550097110298339e-11`,
`5.1697757186275339e-11` — several steps land on the reference value exactly.

The exit criterion in `framework.md` was restated to match, with the reason, and
the test's header block now carries the whole argument so the next reader does
not re-derive it.

**No source file outside the test changed.** `removeVolumeFlux`, the RHS, the
integrator and the step loop are as T2d wrote them; the first gate run validated
them and only the criterion moved.

**Affects:**

- **T4a** — `projectToVolume` is the *absolute* volume correction the rate-only
  projection does not do, and this measurement is the quantitative case for it:
  a run long enough for `5e-12`-per-ten-steps to matter needs it. T4a is also
  where `projectToVolume` first executes, so it is where this drift series stops
  being the expected answer.
- **T4b/T4c** — the drift literals here are for a **fixed** mesh. Any test that
  remeshes cannot reuse `kGoldVolumeDrift`; it needs its own reference series
  measured the same way, from its own gold set.
- **T3a** — the FMM BR path changes the velocity and therefore the drift. If
  regression test 3 reuses this configuration, expect the drift to move by the
  FMM's own error and measure the reference again rather than importing these
  literals.
- **T5b** — the last-finite round trip is now exercised end to end, so
  `CheckpointIO::read` has a validated writer to read back.
- **Anyone adding a multi-node test** — `BEATNIK_TEST_SCRATCH` must be on a
  parallel filesystem. See run 1 above.

## T4 design — the editing-family question

A design-only session. No code was written and nothing was built or run; the
whole output is the Phase 4 rewrite in `framework.md`. Read `../tessera` at
`2ed1b20` (branch tip, working tree clean apart from deleted `tasks/` files) and
the read-only Python driver.

### What was read, so a later session knows what is already checked

`../tessera`: `README.md` (*Editing families*, *Future Optimizations*, *Known
Issues*, *Design limitations*), `docs/design.md` (*Edge-addressed splitting*,
*The closure layer*, *Adaptive refinement*), `src/Tessera_EdgeSplit.hpp`,
`src/Tessera_RefinePolicy.hpp`, `src/Tessera_EditFamily.hpp`,
`src/Tessera_Geometry.hpp` (the `MeshGeometry` accessor), `src/Tessera_Mesh.hpp`
(adjacency accessors), `src/Tessera_HDF5Writer.hpp` (the user-pack loop),
`tests/test_split_edges.cpp` (the header block and case 8's floor),
`tests/test_conforming_quality.cpp` (the header block),
`tasks/{edge-split,edge-flip,edge-collapse,mesh-compaction}.md` (statuses).
Python: `run_adaptive_mesh_bubble.py:1395-1470`, `mesh.py::refine_marked_faces`,
`mesh.py` `TriangleSurfaceState.__post_init__`,
`mesh_solver.py::{refine_potential_mesh_state, _balance_red_green_refinement,
improve_mesh_quality_tangential}`, `dynamic_remesh.py:17-46`.
Beatnik: `Beatnik_AdaptiveMesh.hpp` in full, `Beatnik_MeshInterface.hpp`'s edit
declarations, `Beatnik_Solver.hpp::requireSupportedConfiguration`,
`compare_output.py`'s loader.

**Deliberately not read:** `../canopy` (still F1/T3a's), Tessera's
`Tessera_Refine.hpp` and `Tessera_RefineClosure.hpp` implementations — the
closure's *contract* was taken from `docs/design.md`, which states it explicitly
(user fields copied to closure children, lowest-gid child's values kept on
un-close), and that contract is all the decision turns on. A session that needs
the closure's internals is a session that has decided to use `refine()` after
all, which this design does not.

### The finding that actually settled it

Not the quality measurement the previous revision nominated as the discriminator.
`run_adaptive_mesh_bubble.py:1424` gates the refiner on `not args.dynamic_remesh`
and `:1469-1471` gates the remesher on `args.dynamic_remesh`: **the two
adaptivity modes are mutually exclusive per run**, so no mesh ever sees both
editing families and the conflict options 1, 3 and 4 were designed around does
not exist. `Beatnik_AdaptiveMesh.hpp:16-22` had recorded this correctly all
along; `framework.md` contradicted it and nobody had reconciled the two. Options
1 (two mesh objects), 3 (rebuild between phases) and 4 (the upstream
anisotropic-bisection ask, and its narrower in-memory-clone variant) are dropped
outright. **Nothing needs to be asked of Tessera to unblock T4a.**

The second finding is what decided *which* family, and it inverts the previous
revision's framing. `mesh.py::refine_marked_faces` mints midpoints on marked
faces' edges only and retriangulates each face on its own split-edge bit pattern
with **no cascade** — that is `splitEdges()` exactly. `Tessera::refine()`'s
conforming closure is **transient** (un-closed and rebuilt every call) while the
Python's green/blue children are **permanent**, so a `refine()`-based T4a would
diverge from the Python in face count from round 2 and would churn per-face state
through every un-close. `splitEdges()` is therefore the *higher*-fidelity port,
not the concession the survey filed it as.

### The numbers, and the conversion, so neither is re-derived

`test_split_edges.cpp:95-110`, case 8, five length-driven rounds: min
inradius/circumradius `0.3780 0.3780 0.2815 0.3780 0.3780`, byte-identical at
np1-5 on both backends and in both execution spaces, not trending down.
`test_conforming_quality.cpp` / `docs/design.md`, red-green over 16 rounds:
max `Q` per family, red `1.0278`, green `1.5672`, blue `2.2344` (it was `2.5254`
before the diagonal tie-break became geometric).

The two tests publish reciprocal metrics. `Q = abc(a+b+c)/16A^2 = R/2r` and
`r/R = 1/(2Q)`, so equilateral is `Q = 1` ≡ ratio `0.5`, and case 8's worst
`0.2815` is `Q = 1.776` — **better than red-green's `2.234`**. Two structural
reasons it is not luck: Tessera's two-edge rule joins the midpoint of the
**longer** split edge to its opposite corner, which is Rivara's longest-edge
bisection; and both of Beatnik's masks (T4a's "all three edges of a marked face",
T4b's "every edge over its target") are the cases that rule is good for. The
limit — five rounds, against `test_conforming_quality.cpp`'s own statement that
eight could not distinguish saturation from slow discovery — is R12, not a
footnote.

### Two smaller corrections, both of which change code a task will write

- **`Beatnik_AdaptiveMesh.hpp:410-412` is wrong.** It says refinement is
  constructed with `reference_face_area=None` so the reference is re-based to the
  post-refinement geometry. `mesh.py::refine_marked_faces` actually gives each
  child `parent_ref * child_area / parent_area`, leaves an unsplit face's
  reference alone, and resets reference *curvature* only for subdivided faces.
  The file header's list at `:33-43` is right; the method comment is not. Since
  `RefinePolicy` has hooks for **vertex** fields only and face user fields are
  inherited verbatim, the area rule is not expressible through the policy — hence
  T4a's \f$\sigma = A^{\text{ref}}/A\f$ round trip, which reproduces both the
  split and the unsplit case exactly with two local passes and no parent map.
- **The nonlocal proximity query is not on the default path.**
  `dynamic_remesh.py:33,41` — `use_proximity` and `surgical_proximity` both
  default to `False`. It was blocking T4b in this document for no reason; it is
  now T4e.

### Scope reality check

Phase 4 as a whole still cannot complete: `../tessera/tasks/edge-collapse.md`,
`edge-flip.md` and `mesh-compaction.md` are all `NOT STARTED`, so T4d is blocked
upstream and with it the reference's default end-to-end behaviour. What the
rewrite buys is that **T4a, T4b, T4c and T4e are now implementable against
Tessera as it stands**, which was not true of any of them before.

**Affects:**
- **T4a** — rewritten end to end. It is now `splitEdges()`-based, it deletes
  `SurfaceMesh::refine` and `Comm::reconcileRefinementMarks` rather than filling
  them in, it adds a face user-field pack and a face→edge accessor, and its exit
  criterion no longer promises face counts matching the Python beyond the
  unbinding-cap case. Its old fill-in list named `MeshInterface::refine`, which
  is now deleted; do not implement it.
- **T4b** — narrowed to the sizing field and the split pass, with collapse/flip
  moved to T4d and the proximity query to T4e. Its old exit criterion (50 steps,
  default configuration) belongs to T4d now.
- **T4c** — narrowed to tangential relaxation, which is topology-free and
  therefore unblocked; its flips moved to T4d.
- **T4d, T4e** — new.
- **T5b** — the checkpoint gains `/faces/u0` and `/faces/u1` at T4a, and a
  field-pack mismatch is an `MPI_Abort` inside Tessera. Do not assume a pre-T4a
  checkpoint is readable. See R14.
- **T5d** — unchanged, but `migrate()` now carries the two face fields
  automatically, which is one fewer thing for it to move.
- **R4** — its cost half is retired (the projection is a closed form, so the cap
  is a threshold search rather than a greedy loop); its divergence half stands.

## R12 — Tessera's depth study of `splitEdges()` triangle shape

Not a Beatnik task. Tessera investigated the open question R12 carried — whether
five rounds of case 8 established a shape bound — and the answer changed what R12
says, so it is recorded here.

**The result, in one line:** the Phase 4 assumption holds, but it holds because
of *the mask*, not because of `splitEdges()`.

A new Tessera diagnostic, `../tessera/tests/test_split_edges_depth.cpp` (`unit`
tier, not in Tessera's gate), drives repeated `splitEdges()` with no intervening
`migrate()` and records per round the global minimum \f$r/R\f$, the global
minimum angle, and the population below five \f$r/R\f$ thresholds. Four mask
families:

| mask rule | rounds | min \f$r/R\f$ | verdict |
| --- | --- | --- | --- |
| longer than the mean | 10, F 320 → 3 276 800 | `0.3780 0.3780 0.2815` repeating | bounded, **exactly periodic, period 3**; min angle `33.203°` flat |
| shorter than the mean | 7, F → 1 179 680 | `0.1953 … 0.0007` | unbounded, halves per round; min angle `24.96° → 0.21°` |
| length-blind (hash of midpoint mod 3) | 27, F → 2 340 916 | `0.1953` → `0.0000` from round 8 | unbounded; 96.7% of faces below `0.30` |
| the same inside a geodesic cap | 30 | `0.2238` → `0.0000` by round 11 | unbounded and **localised** |

Every round line is byte-identical across np 1, 2, 4, 5 × {SERIAL, HIP} ×
{Serial, Default} — 7 configurations reduce to 39 distinct lines with zero
spread, so these are properties of the global mesh, not of a decomposition.

**The mechanism, which is the part worth keeping.** The red-green intuition does
not transfer. `refine()` is shape-bounded independently of round count because
its closure is transient — un-close discards the whole closure layer every round,
so every visible triangle is one of finitely many retriangulations of a red
triangle. `splitEdges()` has no reset: a \f$|S|=1\f$ median-cut child is an
ordinary face next call and can be cut again, so the reachable similarity classes
are unbounded in the round count. `splitEdges()` therefore cannot offer a shape
guarantee and does not claim one. What bounds the friendly family is that the
rule is length-driven, making it a coarse Rivara longest-edge bisection: it
attacks the long edge of a stretched triangle, which is exactly the edge whose
bisection improves shape. Splitting short edges is the same machinery backwards,
degrading ~2× per round.

**Why Tessera did not fix this inside `splitEdges()`, deliberately.** Refusing to
split an edge whose child would fall below a floor would bisect fewer edges than
asked, on a predicate the caller cannot see — contradicting the "bisects EXACTLY
the marked edges" contract that cases 1-4 pin, and converting a visible quality
problem into silent under-refinement. The fix belongs above Tessera. If a filter
is ever genuinely wanted inside, the correct shape is an **opt-in mask filter
that returns which marks it dropped**; it is not built because no consumer needs
it.

**What changed on the Tessera side, so expectations match what a checkout shows.**
`test_split_edges` case 8 now runs **7** rounds, not 5 (period 3 means five rounds
show one dip and one recovery — consistent with a bound, not establishing one),
and additionally asserts min angle above `30.0°` and saturation (the last two
rounds set no new worst). Round count overridable with `TESSERA_SPLIT_ROUNDS`.
`kMinRadiusRatioFloor` is unchanged at `0.25` (measured worst
`0.281541949162`, hit at rounds 3, 6 and 9 with the same value each time), but
its comment now carries a SCOPE paragraph: **the floor is a statement about the
mask.** Do not quote it as a property of `splitEdges()`, and do not adopt it as
Beatnik's own floor.

**Verified against source.** The `../tessera` clone was pulled to `cdba371`
("splitEdges() shape quality at depth: the bound is the mask's, not the
operation's") and every claim above was checked against it, so the design doc's
citations carry line ranges again. What the source adds beyond the report:

- `kMinRadiusRatioFloor` is at `tests/test_split_edges.cpp:140`, with the
  measured table at `:108-110` and the SCOPE paragraph at `:123-136`. A second
  floor, `kMinAngleDegFloor = 30.0`, sits at `:142`. Case 8's
  `kRepeatRounds = 7` and `kSaturationRounds = 2` are at `:921` and `:928`, the
  case itself at `:944`.
- The depth diagnostic reports the tail at five fixed thresholds,
  `kTail = {0.30, 0.25, 0.20, 0.15, 0.10}` (`test_split_edges_depth.cpp:98-102`).
  The `0.25` R12 asks T4a to log is one of them — deliberately kept there so the
  two diagnostics are directly comparable, rather than picking a Beatnik-specific
  value.
- Adding a fifth family is three edit sites, not one: the `Family` enum and
  `familyName` (`:305-326`), a `case` in `buildMask` (`:333-397`), and a
  `driveFamily` call in `run` (`:493-500`). The rule must be a pure function of
  global mesh geometry — the existing families hash the midpoint **position**,
  not gids, because gids come from an `MPI_Exscan` and are not rank-count
  invariant (`:52-54`). A family keyed on gids would break the byte-identical
  cross-rank property that makes the whole table meaningful.

**Affects:**
- **R12** — rewritten. No longer "the evidence is five rounds deep"; it is now
  the constraint that every marked edge must be marked *because it exceeds a
  target length*, plus the sharper monitoring discriminator and the ordered fix
  list (Rivara mask promotion first, caller-side shape filter with logging
  second, an opt-in Tessera filter last).
- **T4a** — exit criterion gains the second R12 signal: log the count of faces
  below \f$r/R\f$ `0.25` alongside the minimum, per pass, against the round
  index. Under a length-driven mask that count returns to zero between dips;
  under a degrading one it settles at a stable fraction of the mesh (~17% in the
  below-mean family), which flags "the mesh is going bad" far earlier than the
  minimum does. The measured floor must still be measured, and must not be
  Tessera's `0.25`.
- **T4b** — gains a hard constraint: an edge may enter the split mask only
  because its length exceeds `split_factor · target`. No curvature, vorticity or
  region-tag union. If such a term is ever needed, make the mask
  longest-edge-consistent first (Rivara promotion to fixpoint, a pure mask
  transform above Tessera) and add the rule as a fifth family in
  `test_split_edges_depth.cpp` — ~40 lines, one batch job, ~1m15s at np4 — before
  committing to it. Its exit criterion now requires the healthy signature.
- **T4d** — when collapse and flip land, their masks are *not* length-driven in
  the same sense; re-read R12 before assuming the bound extends to them.

## T4a

**T4a is DONE.** Gate run 4 (`f3SRwVuXai8X`) is green at all 48 launches —
`[gate] PASS (label=regression)`, zero failures, `Beatnik_Test_RefineSplitEdges`
86/86 checks in each of its twelve configurations. The outcome is at the end of
this entry; everything between is the measurement that produced it and should
not be re-derived.

### The four decisions the task fixed, recorded so they are not reopened

1. **`requireSupportedConfiguration` also rejects `--smooth-iters > 0`**, naming
   `MeshQuality::improveQualityTangential` and **T4c**. `FilterParams::
   smooth_iters` defaults to `1` and the reference's refine branch calls the
   tangential pass with it unconditionally once anything is marked
   (`run_adaptive_mesh_bubble.py:1446-1450`), so without this a *default*
   `--refine-every N` run reaches a throwing T4c method with no task ID on it,
   several steps in. The task entry's fill-in list had omitted it; it is added
   there now. `--flip-passes > 0` and `--isotropic-cleanup` are rejected too
   (both T4d, both blocked on Tessera G5c), and `--redistribute-every > 0`
   stays (T4c).
2. **The gate configuration carries `--no-isotropic-cleanup`.**
   `CleanupParams::enabled` defaults to `true`, so the command as the exit
   criterion originally wrote it is rejected by the rejection this task adds.
3. **`VolumeProjection::projectToVolume` still does not execute, and that is
   correct.** The reference gates it on a *repair* having run — `flips > 0 or
   args.smooth_iters > 0 or args.isotropic_cleanup`
   (`run_adaptive_mesh_bubble.py:1465-1468`) — and under the only configuration
   this build accepts all three are false. The gate is transcribed in full in
   `Solver::advanceOneStep` rather than folded to `false`, so landing T4c or T4d
   turns it on by *deleting a rejection* rather than by remembering to add a
   call. It moves to T4c/T4d, whichever lands first. T2d's `Affects:` note and
   **Do** step 6 both said T4a was where it first runs; both are corrected.
4. **Route (a) for mark translation**, and it forced a face field — see below.

### Signature changes, and what forced each

| Was | Now | Why it could not stay |
| --- | --- | --- |
| `SurfaceMesh::refine( const std::vector<char>& )` | **deleted** | The editing-family decision. Deleted rather than left unused, so Tessera's `EditFamily` guard is a backstop that cannot fire. |
| `Comm::reconcileRefinementMarks` | **deleted** | Tessera's edge coordinator routes the edge owner's verdict to every rank holding an incident face, so an unreconciled rank-local mask is a legal input. What Beatnik must still agree — its own mark closure — is one `MPI_Allreduce(MPI_LOR)` inside `AdaptiveMesh`, not a communication primitive. A no-op stub would have let a caller keep a reconciliation step and believe it did something. |
| `template <class EdgeListView> splitEdges( const EdgeListView& )` | `splitEdges( const std::vector<char>& edgeMask )` | `Tessera::splitEdges` takes a host `std::vector<char>` sized `numOwnedEdges()`. A templated parameter could only have hidden the device→host copy, not avoided it. |
| — | `SurfaceMesh::faceEdges()` | `(Nf,3)` local edge indices, cached on `generation()` like `edgeAdjacency()`. The whole task is a translation between a face verdict and an edge mask in **both** directions; \|S_f\| needs face→edge and Tessera publishes no such CSR. |
| — | `SurfaceMesh::ownedFaceGids()` | The "was I subdivided?" discriminator. A \|S\|=0 face keeps its gid; a subdivided parent's is retired. `RefinePolicy` cannot express the rule (its two hooks are vertex-only). |
| `EdgeFaceIncidence{count, faces}` | `+ {resident_count, resident_faces}` | **The bug below.** |
| `face_fields = FaceFields<>` | `FaceFields<Real, Real, Real>` | `{ReferenceArea, ReferenceCurvature, RefineMark}`, with `FaceFieldId` and three accessors. |
| `areaChangeIndicator( const mesh&, const scalar_view& reference_area, ... )` | `( mesh&, const scalar_view& face_area, ... )` | The reference lives in the mesh now; a per-face view outside it does not survive a split. Same for `curvatureChangeIndicator`. |
| `resetReferenceState( const mesh&, scalar_view&, scalar_view& )` | `resetReferenceState( mesh& ) const` | Same. It is also what *initializes* the face pack — Tessera's face AoSoA is allocated uninitialized and `writeMesh` writes the whole pack — so `Solver::setup` calls it **unconditionally**, not only when `--refine-every > 0`. |
| `limitMarkedFraction`, `selectMarks`, `projectedFaceCount`, `balanceRedGreen`, `expandMarkedRings` | all re-signatured onto the edge mask | The Conventions table. `selectMarks` returns `(threshold, max_faces_bound)`. |
| `RefinementDiagnostics` | widened | `projected_faces`, `score_threshold`, `balance_rounds`, `min_radius_ratio`, `faces_below_quarter`, `new_faces_created`, `max_faces_bound`; the four count fields promoted to `GlobalIndex` and made **global**. |
| — | `Solver::lastRefinement()` | The projection is only knowable *before* the edit, so a test cannot reconstruct it afterwards. |

### Route (a), and why it put a scratch field in the checkpoint

The per-face verdict is computed on **owned** faces, written to
`FaceFieldId::RefineMark`, `haloExchange()`d, and each rank then derives its own
owned edges' marks from locally-resident faces. `haloExchange()` is whole-tuple
and addresses fields by their compile-time Cabana member index, so **a mark held
outside the mesh cannot cross a rank boundary at all** — hence a third face
slot, and hence `/faces/u2` in every checkpoint (R14). The per-face *score*
deliberately stays outside the mesh: only owned faces are ever thresholded, so
it never needs to cross.

The balance fixpoint re-exchanges the mark once per round and terminates on one
`MPI_Allreduce(MPI_LOR)`, capped at 64 rounds with a **throw** on the cap.
Measured: **1 round** at every pass and every configuration.

`--max-refine-fraction` and `--max-faces` are both threshold searches on the
score — a fixed 60-iteration bisection, because every probe is a collective and
a convergence test on a floating threshold could terminate at different
iterations on different ranks and deadlock.

### THE BUG ONLY RUNNING REVEALED, and it is a documented Tessera contract

`AdaptiveMesh` originally read `SurfaceMesh::edgeAdjacency()`'s `count`/`faces`,
which resolve Tessera's `EdgeField::Faces`. Gate run 1 failed after the first
refinement pass with **0 bad owned edges at 1 rank, 24 at 2, 45 at 3, 104 at 6**
— a partition-boundary population, on a mesh whose Euler characteristic, face
count and projection identity were all correct.

`EdgeField::Faces` is **partial by construction**, in Tessera's own words at
`../tessera/src/Tessera_DistributedBuilder.hpp:698`: *"filled from this rank's
OWN incidences only — the same partial-by-construction contract `migrate()`
leaves, and the reason `buildFaceAdjacency()` exists rather than reading this
field."* It happens to be complete immediately after `distribute()`, because
distribution cuts a *replicated* mesh in which every rank knew both incidences —
which is why T1b's `count == 2` assertion passes and why the trap is invisible
until the first edit. `splitEdges()` rebuilds the edge set from each rank's local
faces, and the field reverts to its documented partial state.

This is **not a Tessera defect and not a workaround**: `EdgeFaceIncidence` gained
a second pair, `resident_count`/`resident_faces`, derived the other way — from
`FaceField::Edges` scattered over all locally held faces, which is the face's own
record and is complete for every resident face at every generation. All three AMR
consumers and the test now read that pair. `count`/`faces` are kept, documented
as trustworthy only before the first edit, because T1b's global closed-surface
assertion is a genuinely different and still-useful claim.

**Had the check not been there**, the symptom would have been silent
under-marking along partition boundaries — a physics difference that moves with
the rank count. `AdaptiveMesh::refine` keeps that check as a precondition throw.

### What was measured

Configuration: the exit criterion's, **plus `--area-threshold 1e-4
--curvature-change-threshold 1e-4`**. At the *default* thresholds the criterion's
command refines **nothing**: measured against the read-only Python at exactly
those flags, 20 steps end at `F=320`, `refine_events=0`, `max_dA=3.12e-3` against
`--area-threshold 0.16`. Reaching the default needs ~140 steps. A gate member
that runs the refiner and never marks a face is risk **R15**'s trap in a new
place, so the thresholds were lowered until the four scheduled passes are real.
This is a deviation from the criterion as written and is recorded in the test's
header, in the T4a entry, and here.

Per pass, **identical across all twelve configurations** unless noted:

| pass | faces | vertices | marked faces | split edges | new-gid faces | threshold | min r/R | `< 0.25` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 320 → 452 | 162 → 228 | 36 | 66 | 192 | `1.86217290939` | `0.304119905237` | 0 |
| 2 | 452 → 788 | 228 → 396 | 88 | 168 | 496 | `6036.77818654` | `0.123117984672` | 4 |
| 3 (np 1-4) | 788 → 1372 | 396 → 688 | 153 | 292 | 862 | `9147.88966724` | `0.119877418574` | 94 |
| 3 (np 5-6) | 788 → 1390 | 396 → 697 | 157 | 301 | 890 | `9147.88966724` | `0.119867826031` | 96 |
| 4 | unchanged | unchanged | 0 | 0 | 0 | — | as pass 3 | as pass 3 |

`new faces == projectedFaceCount` **exactly**, every pass, every configuration —
the integer identity the criterion asks for. The gid-snapshot difference is
non-empty whenever faces were created, which is what distinguishes "the snapshot
was empty" from "nothing refined".

**The Python's own counts** at the same configuration are 452 / 796 / 1388 /
1388 faces. Pass 1 agrees exactly; pass 2 does not (788 vs 796). That is **R13
one level out** — pass 1 produces the same V/E/F but *different connectivity*
(Tessera's shorter diagonal against the Python's fixed one), so from step 6 the
two codes integrate different meshes, their indicators differ, and pass 2 selects
a different mark set. R13 stated the first-order fact and not this consequence;
both are now in R13.

**`--max-faces` binds from pass 3, and a bound pass is not rank-count
invariant** — 1372 at ranks 1-4, 1390 at ranks 5-6, *identically on both
backends*, which locates it in the cross-rank reduction order (R2) rather than in
the on-node atomics. The threshold search converges to a value pinned between two
adjacent scores, so an ulp-level score difference flips a mark. This extends
**R4**, which had only claimed a capped run will not match the Python.

**R12 is the headline, and it contradicts Phase 4.** The minimum \(r/R\) declines
monotonically and the sub-`0.25` count settles at ~7% of the mesh — this risk's
*shape-problem* signature, not the healthy one Phase 4's *Finding 3* predicted
for T4a's mask. **It is the reference algorithm's behaviour, not Beatnik's**: the
Python's own series, computed offline from its checkpoints with the same
\(8A^2/((a+b+c)abc)\) formula, is `0.486497704566 (0) → 0.304119905237 (0) →
0.123117984672 (4) → 0.119867830292 (101)`, and Beatnik reproduces the first two
rows **to twelve significant digits including the count**. The mechanism: red
children are similar to their parent and fine, but the *green transition* faces
are bisected on whichever edge their neighbour happened to red — not on their own
longest edge — so they are not length-driven and the next pass cuts them again.
R12 and the Phase 4 conventions are corrected. **No mitigation was applied**:
every option on R12's list changes every refinement decision away from the
reference and needs its own gold set and its own task.

**The measured floor is `0.119`** — the run minimum is `0.119867826031` (np 5-6)
and `0.119876446958` (np 1-4), rounded down to three digits. Explicitly **not**
Tessera's `0.25`, which this run fails outright at 96 faces below it.

### R14: the mitigation was extended, not reinvented

`/beatnik/face_field_names` is written from `AdaptiveMesh::face_field_names`
under the same `static_assert` against `FaceFieldId::Count` that M2 used for the
vertex pack, and `compare_output.py::check_face_field_names` verifies it. One
difference, stated because it changes what the check is: no face dataset is
compared — the gold `.npz` files carry no per-face state — so the check is
against a spelled-out `FACE_FIELD_NAMES` tuple rather than against `FIELD_MAP`.

### Not done, deliberately

`Diagnostics::compute` still reports the four AMR indicator fields as `NaN`.
T2d's note calls this "a T4a-era edit to one function"; it is, and every piece it
needs now exists — but it runs inside the progress line of *three other gate
members*, so landing it unverified alongside a task whose own gate is not yet
green would put a currently-green gate at risk for something outside T4a's exit
criterion. It is a small, self-contained follow-up.

`--mesh-kind latlon`, coarsening, flips, `compact`, tangential relaxation and the
proximity query are out of scope by name and task ID (T4b, T4c, T4d, T4e).

**Formatting: `clang-format` was NOT run**, per the standing user instruction and
CLAUDE.md's rule. No file was reformatted; the new and edited code is written in
the style of its surroundings by hand.

**Affects:**

- **T4b** — inherits `faceEdges()`, `ownedFaceGids()`, the face user pack, the
  edge-mask plumbing and `EdgeFaceIncidence::resident_*`, and must not
  re-invent them. Two warnings: `EdgeField::Faces` is partial after any edit, so
  use the resident pair; and the sizing-field mask is length-driven, so it *is*
  in R12's bounded family — T4a's finding narrows to T4a's mask and does not
  transfer.
- **T4c / T4d** — each lands by deleting its rejection from
  `requireSupportedConfiguration`. Whichever lands first is where
  `VolumeProjection::projectToVolume` first executes, and the gate in
  `advanceOneStep` is already written to switch on by itself. T4d additionally
  inherits R12's open question: its masks are not length-driven either.
- **T5b** — the checkpoint now carries `/faces/u0`, `/faces/u1` **and**
  `/faces/u2`; a pre-T4a file is unreadable by a post-T4a binary
  (`MPI_Abort` inside Tessera, not a catchable exception). `/beatnik/face_field_names`
  is available to check against.
- **T5d** — `migrate()` carries the three face fields automatically, including
  the scratch mark, which is zeroed between passes.
- **R2** — new consequence: once refinement is on, the *mesh itself* becomes a
  rank-count-dependent object wherever a threshold search is marginal. That was
  previously a statement about reduced scalars only.
- **R4, R12, R13, R14** — all four rewritten with the measurements above.
- **Anyone adding a `regression` test** — the gate is now **four** members and
  **48 launches** on tuolumne.

### The outcome: gate run 4, green

`BEATNIK_TEST_SCRATCH=/p/lustre5/stewartj/beatnik/gate_scratch flux batch
scripts/tuolumne/run_regression_minset.flux` → `f3SRwVuXai8X`, 6.7 minutes,
`beatnik_regression_minset.f3SRwVuXai8X.log`. All **48** launches ran and the
log ends `[gate] PASS (label=regression)` with **zero** `[FAIL]` lines.
`Beatnik_Test_RefineSplitEdges` reports **86/86 checks** in each of the twelve
{SERIAL, HIP} x ranks 1-6 configurations. `spack install` beforehand rebuilt
rather than no-op'd — the corrected test literals were the only change since run
2 — so what ran is the corrected test. Nothing in `src/` was touched at any
point between runs 2, 3 and 4.

Every per-pass number in the table above reproduced **exactly**, including the
rank-count split at pass 3 (1372 faces at ranks 1-4, 1390 at ranks 5-6, on both
backends), `new faces == projectedFaceCount` at every pass, `balance rounds 1`,
and 0 hanging nodes / 0 non-resident incident faces after every edit.

**One number moved, and it is a bookkeeping correction rather than a new
measurement.** This entry recorded the ranks 5-6 whole-run minimum \(r/R\) as
`0.119867826031`; the test reports `0.119867784111`. `0.119867826031` is pass
**3**'s minimum, which is what the offline analysis had in hand. Pass 4 marks
nothing, but the mesh integrates five more steps, so the run minimum lands a few
ulp lower in its tenth significant digit. Ranks 1-4 are unchanged at
`0.119876446958` — there the run minimum was already pass 4's. The `0.119`
floor is 700x clear of both and needed no change, and neither did any literal in
the test. R12 and T4a's entry are corrected.

Gate run 3's hang in `Beatnik_Test_InitialConditions_MPI_SERIAL` at 6 ranks
**did not recur**: that member was green at all six SERIAL rank counts in run 4,
inside a sweep that finished in under seven minutes against run 3's ~16 minutes
before cancellation. It was a transient. Nothing was absorbed into T4a on
account of it; if it reappears it is a separate bug and gets its own entry.

Still open and deliberately not done here: `Diagnostics::compute` reports the
four AMR indicator fields as `NaN`. Every piece it needs exists, but it runs
inside three *other* gate members, so it stayed out of T4a rather than put a
green gate at risk for something outside the exit criterion. It is now a small
self-contained follow-up against a gate that is green.

## T4b

**T4b is DONE.** The sizing field, the gradation sweep and the split third of
`dynamic_remesh.py` are implemented; the gate grew to **five members / 60
launches** and is green. `Beatnik_Test_DynamicRemeshSplit` reports **377/377
checks** in each of its twelve {SERIAL, HIP} x ranks 1-6 configurations, and
every diagnostic number it prints — twenty per-pass lines of counts, split
tallies, quality, sagitta and both R12 signals — is **byte-identical across all
twelve**, verified by diffing the gate log's tables rather than by inspection.

The gate run is `f3SpT4MZbqMh` (`beatnik_regression_minset.f3SpT4MZbqMh.log`,
~11 minutes), ending `[gate] PASS (label=regression)` with zero `[FAIL]` lines.
The four pre-existing members are unchanged — in particular
`Beatnik_Test_RefineSplitEdges` is still 86/86 in all twelve of its
configurations, which is the check that T4a's twelve-digit shape literals did
not move when its `r/R` kernel was single-sourced into
`SurfaceOperators::radiusRatioStats`.

### The decisions this task fixed, recorded so they are not reopened

1. **No new CLI flag. The port follows the reference's CLI exactly.** A
   `--dynamic-remesh` run is accepted only when the three unimplemented thirds
   are configured off through the reference's *own* knobs, so what Beatnik runs
   is what the reference would run rather than a Beatnik-only subset of it.
   `requireSupportedConfiguration` rejects every other remeshing configuration by
   name and task ID, per the Phase 4 *Unsupported configuration* convention:

   | knob | accepted value | what the reference then does | rejection names |
   | --- | --- | --- | --- |
   | `--remesh-collapse-factor` | `0` | candidate predicate `length < 0*target` never true (`dynamic_remesh.py:373`) | `DynamicRemesh::collapseShortEdges`, **T4d**, gap G5b |
   | `--remesh-smooth-iters` | `0` | returns at `:463-465` | `DynamicRemesh::tangentialSmooth`, **T4d** |
   | `--remesh-flip-min-gain` | `>= 1e12` | accept test `min(new) > min(old)(1+g)` unsatisfiable (`:449-450`) | `DynamicRemesh::flipEdgesForQuality`, **T4d**, gap G5c |
   | `--isotropic-cleanup` | off | driver skips the block (`:1493`) | `MeshQuality::isotropicCleanup`, **T4d** |
   | `--remesh-proximity`, `--remesh-surgical-proximity` | off (their defaults) | never reached | `nonlocalFaceCentroidDistance` / `splitSurgicalProximityEdges`, **T4e** |

   The `1e12` sentinel is `Beatnik::kFlipsDisabledMinGain` in
   `Beatnik_Params.hpp`, with the citation on it: triangle quality is
   `4*sqrt(3)*A/sum(l^2)` and therefore lies in `[0,1]`, so no pair of triangles
   can clear a gain of `1e12`.

2. **`--remesh-max-collapses 0` is NOT a lever, and the task prompt's claim that
   it "truncates the candidate list to empty" is true of the dataclass field and
   **false of the driver**.** `run_adaptive_mesh_bubble.py:1350-1352` maps a
   non-positive value to `None` = *unlimited*, and `RemeshParams::
   max_collapses_per_pass` documents the same ("<= 0 means unlimited"). Accepting
   on it would have accepted a run in which the reference still collapses, which
   is the exact failure the acceptance rule exists to prevent. The rule is
   therefore `collapse_factor <= 0` alone; the rejection message says so
   explicitly so the next reader does not re-derive it. The gate's command line
   still carries `--remesh-max-collapses 0` because the exit criterion names it
   and it is harmless once the factor gates the pass.

3. **`--remesh-tight-after >= 0` is rejected, and the tight profile is
   unported and unassigned.** No `--remesh-tight-*` option reaches
   `RemeshParams` — `Beatnik_Params.hpp:219` mentions the profile in prose only
   and `SolverParams::remesh_tight` is a copy of the baseline set. Without the
   rejection a run past `--remesh-tight-after` would silently keep remeshing at
   the baseline parameters, which is a wrong answer rather than a missing
   feature. Porting it is **not assigned to any task**; it is a self-contained
   parameter-plumbing job (twenty-odd CLI options into a second `RemeshParams`)
   and the solver branch that swaps the sets is already written as a comment at
   the one place it belongs.

4. **The reference comparison is the driver run at the accepted configuration,
   analyzed offline — not a `--dynamic-remesh` gold set at the defaults.** With
   the four knobs above the reference's own remesh path *is* split-only
   (verified by reading each of the four early returns), so the driver run is a
   split-only run; committing a default-configuration gold set would have
   compared against a different algorithm. Its checkpoints were analyzed offline
   in numpy for the per-pass counts, both R12 signals, the minimum triangle
   quality and the volume drift, exactly as T4a produced its Python columns. The
   scratch scripts live in `/p/lustre5/stewartj/beatnik/t4b/`
   (`probe_sizing.py`, `probe_fixpoint.py`, `analyze_ref.py`) and nothing from
   them is committed except the resulting literals.

5. **The re-basing after a remesh is done by the SOLVER, not by the remesher.**
   `dynamic_remesh_state_with_material` rebuilds the state with
   `reference_face_area=None` and `reference_face_curvature=None` and
   `MeshPotentialZModelState.__post_init__` re-seeds them *and* re-centres the
   potential against the new area weights (`mesh_solver.py:155-162`). Both are
   properties of state construction rather than of the remesher, and
   `AdaptiveMesh::resetReferenceState` lives in the AMR header, so
   `Solver::advanceOneStep` calls it plus a `centerPotential` immediately after
   `remesh()` returns. Having `DynamicRemesh` depend on the AMR header to satisfy
   a configuration in which the AMR indicators are never read would have been the
   wrong coupling.

### Signature and API changes, and what forced each

| Was | Now | Why it could not stay |
| --- | --- | --- |
| `faceCurvatureForSizing( const mesh_type&, ... )` | `( mesh_type&, ... )` | `positions()`, `faceVertices()`, `faceEdges()` and `edgeAdjacency()` are all non-const — Cabana slices behind a generation guard and CSRs cached against `generation()`. The same widening T2c applied to twelve signatures. Same for `vertexTargetEdgeLength` and `gradeTargetEdgeLength`. |
| `int splitLongEdges( mesh, state, target )` | `+ RemeshDiagnostics& diag` | The **candidate** count is knowable only inside the function, and `splits == candidates` is the assertion the exit criterion's "every long edge is split unless blocked" reduces to. |
| `RemeshDiagnostics{old,new}_{vertices,faces}` as `int` | `GlobalIndex`, and **global** | `RefinementDiagnostics`' reason: a per-rank entity count is a statement about the partition. |
| — | `RemeshDiagnostics` + `passes`, `split_candidates`, `split_capped`, `split_ratio_threshold`, `long_edges_after`, `long_edges_at_h_min`, `min_radius_ratio`, `faces_below_quarter` | R12's two signals and the mask-completeness audit. |
| — | `Solver::lastRemesh()` | Same reason as `lastRefinement()`: the candidate count and the cap's verdict cannot be reconstructed after the call. |
| — | `SurfaceMesh::haloExchangeVertexView( view )` | **The one genuinely new primitive.** See below. |
| — | `SurfaceOperators::radiusRatioStats<ExecSpace>( ... )` | R12's kernel, moved verbatim out of `AdaptiveMesh::measureShape` so two tasks asserting against the same twelve-digit Python numbers use one implementation. Identical arithmetic in identical order; T4a's literals did not move, and the gate proves it. |
| — | `Beatnik::kFlipsDisabledMinGain` | Decision 1. |

### The new communication primitive, and why a fourth vertex field was the wrong answer

The gradation sweep needs a ghost exchange of the per-vertex target *between
sweeps* — the header always said so. After `k` sweeps a vertex's target sees
`gamma^d * h_j` for every vertex `j` at graph distance `d <= k`, so the default
eight sweeps reach **four times the halo depth**. Without the exchange the
constraint is enforced over a 2-ring at a partition boundary and an 8-ring
everywhere else: the sizing field bends at every seam and the split set moves
with the rank count.

`Beatnik_MeshInterface.hpp`'s own header says a field outside the mesh cannot
cross a rank boundary. **That is a statement about the REVERSE halo** —
`haloScatterAddVertices` accumulates into a field addressed by its compile-time
Cabana member index, so an external view has no way in. A *forward* exchange has
no such obstacle: it is a gather from owned rows and a scatter into ghost rows,
and the plan's index lists are ordinary integers. `haloExchangeVertexView` loads
the view into a one-member scratch AoSoA over the same local vertex range and
calls `Tessera::haloExchange( comm, aosoa, halo.vplan )` — Tessera's own tested
pack/unpack over Tessera's own plan, so **Beatnik still posts no message of its
own** and the claim at the top of `Beatnik_Communication.hpp` stands unchanged.

The alternative — a fourth slot in `vertex_fields` — was rejected: it would put a
*scratch* quantity in `/vertices/u3` of every checkpoint and make every existing
file unreadable (R14) for something no restart needs. T4a paid that price for
`RefineMark` because the reverse direction left it no choice; here there was one.

### What was measured, and the configuration deviation that made it measurable

**At the reference's default sizing parameters this test would have been
vacuous, in both directions at once.** Measured against the read-only Python:
the curvature term asks for `sqrt(8*0.004/3.98) = 0.0894`, `--remesh-h-max 0.05`
cuts it to `0.05`, and the split threshold `1.35*0.05 = 0.0675` lands **below the
shortest edge in the mesh** (`0.0690`). So

- pass 1 marks **480 of 480** edges — the metric selects nothing;
- `--remesh-max-splits 300` truncates it, which is R4's territory *and*, as it
  turns out, an R12 exposure (see below), and makes a pass-1 comparison with the
  reference impossible: the tie-break is the endpoint index pair, which Beatnik
  does not have;
- and the reference's own run then splits **nothing for the remaining eighteen
  steps** (measured: `320 -> 920 -> 1280`, then flat) — R15's trap.

The test therefore runs at `--remesh-sagitta-tolerance 0.002 --remesh-h-max
0.06`, which puts the threshold *inside* the edge-length distribution. Every
other remesh knob is the reference's default, **including `--remesh-max-splits
300`, which never binds** (the largest pass is 120). This is the same class of
deviation T4a recorded for its indicator thresholds, made for the same reason,
and it is written into the test's header, into T4b's entry and here.

Per pass, identical across all twelve configurations, and identical to the
reference's own columns:

| pass (step) | faces | vertices | splits / candidates | long after (at h_min) | min quality | min r/R | `< 0.25` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 320 -> 560 | 162 -> 282 | 120 / 120 | 120 (0) | `0.768627058629` | `0.373875540852` | 0 |
| 2 | 560 -> 800 | 282 -> 402 | 120 / 120 | 120 (0) | `0.624710812018` | `0.248492357897` | **120** |
| 3 | 800 -> 1040 | 402 -> 522 | 120 / 120 | 60 (0) | `0.624708569426` | `0.248490855246` | **120** |
| 4 | 1040 -> 1160 | 522 -> 582 | 60 / 60 | 120 (0) | `0.673305465303` | `0.281539942917` | 0 |
| 5 | 1160 -> 1400 | 582 -> 702 | 120 / 120 | 0 (0) | `0.673301906463` | `0.281537474137` | 0 |
| 6-20 | 1400 | 702 | 0 / 0 | 0 (0) | `0.6733` -> `0.673236892764` | `0.2815` -> `0.281492866851` | 0 |

`splits == candidates` at every pass with the cap never binding: **every** edge
longer than `split_factor * max(target, h_min)` entered the mask. The non-zero
`long after` at passes 1-4 is not a defect and the test says so — a split halves
an edge, so an edge more than `2*split_factor` over target is still long
afterwards and is the next pass's work; pass 5 clears the last 120 and the count
stays at zero.

**R12: the healthy signature, and this is the first time the risk's positive
case has been observed.** The minimum `r/R` dips to `0.2485` and **recovers** to
`0.2815`; the sub-`0.25` population goes `0 -> 120 -> 120 -> 0` and **returns to
zero**; the last third of the run sets no new low. Both are asserted, not
eyeballed. T4a's mask produced the opposite signature and R12 now carries both
measurements side by side. The measured floor is `0.248` (the run minimum
rounded down), explicitly **not** Tessera's `0.25`, which this run would fail by
four ulp at passes 2 and 3.

**A third R12 finding, which the risk did not anticipate: the per-pass cap can
take a length-driven mask OUT of the bounded family.** The default-parameter
reference run above, whose mask is truncated by `--remesh-max-splits 300` every
time it fires, goes to `0.204341652937` with **32** faces below `0.25` at pass 1,
then **64**, and never returns to zero. A truncated mask is no longer "every edge
longer than its target" — it is the top-N by ratio, a rank-driven rule, and the
neighbours left unsplit are exactly the transition faces that carry no bound. So
a capped pass is not only R4's count divergence; it is an R12 exposure. Recorded
in R12.

**R13 did not bite here, and that is a measurement rather than a reprieve.**
This task's masks are partial from pass 1, so `|S| = 2` faces do arise and the
two codes do choose their diagonals by different rules — and yet Beatnik
reproduced the reference's face and vertex counts **exactly at all twenty
steps**, not only at pass 1, along with both shape signals to twelve digits. The
test asserts all twenty. The honest reading is that on this mesh the two rules
agree wherever the case arises; R13 is amended to say so without being retired.

**`VolumeProjection::projectToVolume` executes for the first time here.** The
reference gates it on a remesh having *run*, not on it having changed anything
(`:1513-1516`), so under `--dynamic-remesh` it runs every step. T4a's entry and
T2d's `Affects:` note both predicted T4c/T4d; both are wrong for that reason.
Consequence for the exit criterion's volume-drift clause: the drift is **exactly
zero at every step**, on both backends at every rank count, and so is the
reference's (one entry aside, step 17's `2.22044604925031308e-16`, one ulp of the
ratio). A series of zeros only asserts anything if the bound can fail a build
that skipped the projection, so `kVolumeDriftBound = 1e-14` — T2d measured
`5.1697091052460564e-11` at step 10 for a fixed mesh with no projection, three
decades above it. The `1e-9` absolute cap the criterion names is kept as the
coarser blow-up detector. T2d's `kGoldVolumeDrift` is **not** reused, exactly as
its `Affects:` note required.

**T4d's question, answered from the run.** The minimum triangle quality falls
`0.977 -> 0.769 -> 0.625 -> 0.625 -> 0.673` and then *recovers* and holds at
`0.673` for fifteen steps. It never approaches `--remesh-min-quality` `0.18`, so
there is no crossing step to report: **the missing coarsening does not bite
within 20 steps on this problem**. What holds quality up is the split mask's own
longest-edge consistency, not the collapse third. T4d's case therefore rests on
longer runs and on the tight roll-up, and a T4d exit criterion built around "the
minimum quality stays above `--remesh-min-quality`" will pass on this
configuration whether or not collapse works — R15's shape again.

### No bug survived to the gate, and one thing that is worth writing down anyway

Unusually for this tree, the first run of the new member passed everywhere. Two
traps were paid for in advance rather than discovered:

- **`EdgeField::Faces` is partial after any edit** — the normal-variation
  curvature reads the edge's two incident faces, and it runs *between* splits, so
  it uses `EdgeFaceIncidence::resident_count` / `resident_faces` throughout. T4a
  paid a whole gate run for that lesson and its `Affects:` note is what kept this
  task from repeating it.
- **The Python's gradation sweep is Jacobi, not Gauss-Seidel.** It writes into
  `target` while reading the neighbour value from `old`, the copy taken at the
  top of the sweep, so one sweep is exactly
  `h_i <- min(h_i, gamma * min_j h_j)` over the previous iterate and is
  independent of the edge order. Reading it as Gauss-Seidel would have made the
  port order-dependent — and therefore partition-dependent — while looking like a
  faithful transcription.

Also deliberate: the normal-jump curvature is a **gather** over each face's three
edges rather than the reference's scatter over edges into both incident faces.
The two visit the same (edge, face) pairs so the result is identical, and the
gather needs no atomic max on a `Real`.

**Formatting: `clang-format` was NOT run**, per the standing user instruction and
CLAUDE.md's rule. No file was reformatted; the new and edited code is written in
the style of its surroundings by hand.

### Not done, deliberately

`collapseShortEdges`, `flipEdgesForQuality`, `tangentialSmooth` and `compact`
(T4d, blocked on Tessera G5b/G5c/G5d); `nonlocalFaceCentroidDistance`,
`nonlocalFaceProximityPairs` and `splitSurgicalProximityEdges` (T4e); the
`--remesh-tight-*` profile (unported, unassigned, and now rejected rather than
silently ignored); and `Diagnostics::compute`'s four `NaN` AMR indicator fields
(T4a's named follow-up, still open and still self-contained).

**Affects:**

- **T4d** — inherits four things. (a) Its blocking question is answered: quality
  does not degrade to the repair trigger in 20 steps, so its exit criterion needs
  a longer horizon or the tight roll-up, and "min quality stays above 0.18" is
  **not** a discriminating assertion on this configuration. (b) It lands by
  deleting its four rejections from `requireSupportedConfiguration`; each names
  itself. (c) `tangentialSmooth` is now assigned to it, and is the one piece of
  it that is *not* blocked upstream — it changes no connectivity. (d) The pass
  loop in `DynamicRemesh::remesh` already carries the reference's
  `changed || needs_quality_repair` gate in full, and the sizing recompute
  between split and collapse is documented as load-bearing at the point where
  T4d must add it back.
- **T4e** — inherits the sizing field with its proximity term marked at the one
  place it enters (`vertexTargetEdgeLength` stage 2), and
  `vertexTargetEdgeLength` already takes the `state` it will need for the
  material exclusion, so landing T4e changes no caller.
- **R12** — now carries measurements of **both** mask families, plus the new
  finding that a per-pass cap can move a length-driven mask out of the bounded
  family.
- **R13** — amended: partial masks did not diverge here, at twenty steps, so the
  risk is conditional rather than structural in its consequence.
- **R4** — a capped split pass is the same divergence as a capped refine pass,
  and is avoided here by keeping the candidate count under the cap; the test
  asserts `!split_capped` rather than assuming it.
- **T5b** — the checkpoint layout is unchanged by this task. The gradation's
  ghost exchange goes through a scratch AoSoA, so no new vertex field entered
  `/vertices/`, and a T4a-era checkpoint remains readable.
- **Anyone adding a `regression` test** — the gate is now **five** members and
  **60 launches** on tuolumne.
