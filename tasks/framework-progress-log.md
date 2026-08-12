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
