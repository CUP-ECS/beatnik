# Grouped HDF5/XDMF checkpoint output

**Status:** T1 DONE, T2 DONE

## Problem

A Beatnik run with `--checkpoint-dir` writes one `<prefix>_t<timekey>_step<step>.h5`
plus one same-stemmed `.xmf` sidecar per checkpoint
([src/Beatnik_IOInterface.hpp:547](../src/Beatnik_IOInterface.hpp#L547)). Paraview's
XDMF readers are not file-series readers, so those N sidecars are N unrelated
datasets: each must be opened separately and there is no time slider over the
sequence. Nothing downstream can recover the grouping, because grouping has to be
stated in the light data.

Tessera now states it. `Tessera::MeshSeries`
([../tessera/src/Tessera_XdmfSeries.hpp:83](../../tessera/src/Tessera_XdmfSeries.hpp#L83))
is a caller-held handle that writes each frame exactly as before and additionally
maintains **one master `.xmf`** — an XDMF temporal collection naming every frame with
its time — rewritten after every frame.

**End state.** A run with `--checkpoint-dir out --checkpoint-prefix checkpoint` that
takes N checkpoints leaves, in addition to everything it leaves today, a single
`out/checkpoint.xmf` that Paraview's temporal XDMF3 reader opens as one dataset with
N timesteps on the time slider. Per-frame `.h5` files, per-frame `.xmf` sidecars, the
`/beatnik` scalar group and the `_latest` symlinks are all unchanged in layout, so
`compare_output.py`, the five gate tests and any future `readMesh` restart are
unaffected.

**Out of scope.** Reopening a series across a restart (see
[Deliberate deviations](#deliberate-deviations)); any change to the per-frame HDF5
schema; any new CLI option; any change to which states get checkpointed or when.

## Read this first

The brief asks for the I/O interface to be updated to use Tessera's new API. Three
things codebase inspection established that the brief could not:

1. **A naive port throws on every existing gate test.**
   `MeshSeries::write( mesh, frameStem, time )` throws `std::runtime_error` when
   `time` is not *strictly* greater than the previous frame's
   ([../tessera/src/Tessera_XdmfSeries.hpp:110-118](../../tessera/src/Tessera_XdmfSeries.hpp#L110-L118)).
   Beatnik's `Solver::finalize()` re-writes the last-finite state, which carries the
   **same `(time, step)`** as the previous checkpoint whenever the last accepted step
   also checkpointed — stated outright at
   [src/Beatnik_Solver.hpp:373-378](../src/Beatnik_Solver.hpp#L373-L378) ("the same
   filename as `setup`'s startup checkpoint, written twice"). Gate test 2 runs with
   `--checkpoint-every-steps 1`
   ([tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp:372](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp#L372)),
   so this fires there, and at `--steps 0` (gate test 1) it fires on the second
   frame. T1 therefore carries an explicit equal-time rule; see
   [Conventions](#conventions).

2. **Time can be equal but never smaller.** `recordLastFiniteState()` runs
   immediately before `checkpointDue()`
   ([src/Beatnik_Solver.hpp:597-600](../src/Beatnik_Solver.hpp#L597-L600)), so the
   state `finalize()` restores is always at or after the last checkpoint's step. A
   decreasing time is therefore not a case to accommodate but a bug, and T1 throws on
   it rather than tolerating it.

3. **A restart cannot reach any of this.** `CheckpointIO::read` is a
   `BEATNIK_NOT_IMPLEMENTED` stub
   ([src/Beatnik_IOInterface.hpp:636-642](../src/Beatnik_IOInterface.hpp#L636-L642))
   and `RestartReader::load` throws (framework.md T5b, NOT STARTED), so
   `--restart-from` cannot complete. The series-across-restart question is real but
   unreachable, and this design defers it to T5b rather than guessing an answer now.

## Approach

`CheckpointIO` gains one `Tessera::MeshSeries` member, constructed with the master
stem, and `write()` routes its frame through it instead of calling
`Tessera::writeMesh` directly. Nothing else moves: the mkdir, the barriers, the
rank-0 `/beatnik` append and the `_latest` relink all stay where they are and in the
order they are in.

Routing through `MeshSeries` rather than having Beatnik accumulate its own
`std::vector<Tessera::XdmfTimeStep>` and call `Tessera::writeXdmfSeries` is the
decision worth recording. Both work; `MeshSeries` is the intended surface, it keeps
`TIMER_WRITE_MESH` spanning the frame (it goes through the public timed `writeMesh`,
[../tessera/src/Tessera_HDF5Writer.hpp:446](../../tessera/src/Tessera_HDF5Writer.hpp#L446)),
it keeps the `.xmfindex` restart record that a future series-reopen will consume, and
it means the atomic temp-file-plus-rename master write is not reimplemented here. The
cost is that its throw-on-equal-time has to be worked around from the outside, which
is the one place Beatnik does its own thing.

`MeshSeries` is held by value, not lazily. `CheckpointIO` is itself constructed
lazily, only when `--checkpoint-dir` is set
([src/Beatnik_Solver.hpp:1041-1046](../src/Beatnik_Solver.hpp#L1041-L1046)), and
`MeshSeries` writes nothing until its first `write()`, so a value member costs a
string and an empty vector and no I/O.

### Conventions

| Choice | Rule |
| --- | --- |
| Master stem | `<directory>/<prefix>`, i.e. `out/checkpoint.xmf`. No frame is ever named `<prefix>.h5`, so there is no collision, and it is the only `.xmf` in the directory with no step in its name — Tessera's own "the one to open" convention. |
| Master stem, computed once | A `private static std::string masterStem( const std::string& directory, const std::string& prefix )`, called from the constructor's member-init list. `_series` is declared **after** `_prefix` so declaration-order initialization sees both already moved-into. |
| Equal-time frame | When `header.time == ` the last appended frame's time **and** the stem is identical: call `Tessera::writeMesh( mesh.tesseraMesh(), stem, time )` directly (the **timed** overload, so the sidecar is byte-identical to what `MeshSeries` would have written) and do **not** append to the series. The master already names that exact frame at that exact time, so nothing is lost and there is nothing to warn about. |
| Decreasing time, or equal time with a different stem | `throw std::runtime_error` naming both stems and both times. These are unreachable given fact 2 of [Read this first](#read-this-first); reaching one means an invariant broke and the loud failure is the point. |
| Series state Beatnik keeps | `_last_frame_time` and `_last_frame_stem`, private, only meaningful when `_series.numFrames() > 0`. `MeshSeries` exposes `numFrames()` and `masterStem()` but not the last time or stem, so the guard needs its own. Do not add a separate frame counter — `_series.numFrames()` is the counter. |
| Adapter contract | Unchanged and load-bearing: `Tessera::MeshSeries` may be named **only** in `Beatnik_IOInterface.hpp`. No other Beatnik header may acquire a Tessera or HDF5 type ([src/Beatnik_IOInterface.hpp:23-25](../src/Beatnik_IOInterface.hpp#L23-L25)). |
| Public signatures | None change. `CheckpointIO::write( header, mesh )` keeps its signature and its return value (the timestamped `.h5` path). See [Callers](#callers). |
| Test tier | `unit`, as a standalone self-validating binary under `tests/unit_tests/`, per that directory's convention ([tests/unit_tests/CMakeLists.txt:11-40](../tests/unit_tests/CMakeLists.txt#L11-L40)). The `regression` tier is the ship gate and is **not** touched. |
| Test output path | `$BEATNIK_TEST_SCRATCH`, else `$TMPDIR`, else `"."`, then a subdirectory unique per `(execution space, rank count)` — copy the resolution order and the reasoning from [tests/regression_tests/Beatnik_Test_InitialConditions.cpp:271-282](../tests/regression_tests/Beatnik_Test_InitialConditions.cpp#L271-L282). A relative default fails only on the installed path, where the manifest directory is read-only. |
| File headers | BSD-3-Clause block with `SPDX-License-Identifier: BSD-3-Clause` on any new file, in the comment style of its file type. |
| Formatting | **Do not run clang-format, `clangformat.sh` or the `cabana-format` target.** Match the surrounding style by hand. |

### Deliberate deviations

- **`<prefix>_latest.xmf` keeps pointing at the newest *frame* sidecar, not at the
  master.** It is the "latest checkpoint" alias and a restart consumes
  `<prefix>_latest.h5`; repointing it at a collection master would break that pairing.
  The master is a third thing beside the pair, and the README says which to open.
- **The per-frame `.xmf` sidecars now carry a `<Time Value=>` child**, because
  `MeshSeries::write` goes through the timed `writeMesh` overload rather than the
  timeless one that
  [src/Beatnik_IOInterface.hpp:547](../src/Beatnik_IOInterface.hpp#L547) calls today.
  This is a change to emitted light data, accepted rather than avoided: it makes a
  single frame self-describing, and nothing reads those sidecars —
  `compare_output.py` reads `.h5` datasets only.
- **A series is not reopened across a restart, and this is not worked around.** On a
  hypothetical restart the master `<prefix>.xmf` would be rewritten with only the
  post-restart frames while Tessera appends to the pre-existing `<prefix>.xmfindex`
  ([../tessera/src/Tessera_XdmfSeries.hpp:158-171](../../tessera/src/Tessera_XdmfSeries.hpp#L158-L171)),
  leaving the two describing different frame lists. Unreachable today (fact 3 above),
  and the fix belongs with the restart path: framework.md **T5b** owns it. T3 records
  it in README "Known Issues" so it is not discovered by surprise there.
- **The equal-time frame is still rewritten**, not skipped. Its content is expected to
  be byte-identical to the frame already on disk, but that follows from
  `recordLastFiniteState()` sitting where it does, and today's documented behaviour is
  "written twice, harmless because `writeMesh` truncates". Preserving the write keeps
  this change confined to the light data.

## Current state

- `CheckpointIO::write` works and is exercised by all five gate members. It calls the
  **timeless** `Tessera::writeMesh( mesh, stem )`
  ([src/Beatnik_IOInterface.hpp:547](../src/Beatnik_IOInterface.hpp#L547)) and knows
  nothing about series. There is no master `.xmf`, no `.xmfindex`, and no `<Time>`
  element in any file Beatnik emits.
- `CheckpointIO::read` is a stub that **throws** (`BEATNIK_NOT_IMPLEMENTED`,
  [src/Beatnik_IOInterface.hpp:636-642](../src/Beatnik_IOInterface.hpp#L636-L642)) —
  it does not return a wrong header. Restart is therefore loudly unavailable, not
  quietly broken.
- Tessera at `../tessera` HEAD (`2ba15cd`) has `Tessera_XdmfSeries.hpp`, and
  `Tessera.hpp` includes it, so `#include <Tessera.hpp>` is all Beatnik needs. Tessera
  installs `src/*.hpp` by directory glob
  ([../tessera/CMakeLists.txt:117-118](../../tessera/CMakeLists.txt#L117-L118)), so
  the new header needs no install-list edit. **But the installed Tessera in this
  spack environment predates the commit** — T1 step 1 exists for that reason.
- The `unit` tier has **six** members as of T2 (five before it) and the `regression` tier
  has five, with the gate at 60 launches on tuolumne. This work adds one `unit` member
  and no gate member. The "three members" this line used to claim counted only
  `BEATNIK_UNIT_TEST_SOURCES` and missed `Beatnik_Test_PythonCompare` and its negative
  case, which are in the same manifest and the same tier.

### Callers

`CheckpointIO::write`'s signature does not change, so its callers are listed for
completeness rather than for editing. All three are `Solver::writeCheckpoint()`
([src/Beatnik_Solver.hpp:1051-1068](../src/Beatnik_Solver.hpp#L1051-L1068)), reached
from `setup()` step 6 ([src/Beatnik_Solver.hpp:303](../src/Beatnik_Solver.hpp#L303)),
from `advanceOneStep()` when `checkpointDue()`
([src/Beatnik_Solver.hpp:599-600](../src/Beatnik_Solver.hpp#L599-L600)), and from
`finalize()` ([src/Beatnik_Solver.hpp:380-384](../src/Beatnik_Solver.hpp#L380-L384)).
**No file outside `src/Beatnik_IOInterface.hpp` is edited by T1.**

## Progress log

[tasks/grouped-io-progress-log.md](grouped-io-progress-log.md). **Read it before
starting any task here**, before changing a signature this document states, and before
reopening a question this document treats as settled — a completed task may have
changed what a later task's **Do** steps should say, and the log's `**Affects:**` line
is the index to that.

## Task sequence

### T1 — `CheckpointIO::write` emits the grouped master through `Tessera::MeshSeries` — **DONE**

**Met.** Tessera was reinstalled first (R5 confirmed live: the installed prefix had no
`Tessera_XdmfSeries.hpp` before it), then `spack install` of Beatnik succeeded.
`scripts/tuolumne/grouped_io_t1.flux` — the new exit-criterion script — ran
`adaptive_mesh_bubble --steps 4 --checkpoint-every-steps 2 --checkpoint-dir
/p/lustre5/stewartj/beatnik/grouped_io/grouped_io_t1` at 4 ranks on 1 node with the
template's rank-to-GPU binding, as **job `f3T3fdHDWBQT`** (`CD`, rc=0). It measured, in
the log rather than by eye: `CollectionType="Temporal"` count **1**, `<Time Value=`
count **3**, distinct frame `.h5` count **3** — the frames at
`(t=0, step 0)`, `(t=0.0045, step 2)` and `(t=0.0075, step 4)`, with `finalize()`'s
repeat of the last one appearing **once**. `checkpoint.xmfindex` likewise holds exactly
three lines. The failure direction held: no
`Tessera::MeshSeries::write: time must be strictly increasing` anywhere in the log.
Five earlier submissions (`f3T3V6SDq8JB`, `f3T3cdLxTFvf`, `f3T3dQg57dWF`,
`f3T3eARVtREw`, `f3T3etQ7s1Yj`) failed on the script and the example's defaults, not on
this change — see the progress log.

**Depends on:** none.

**Fill in:** [src/Beatnik_IOInterface.hpp](../src/Beatnik_IOInterface.hpp) only — the
file header comment, `CheckpointIO`'s constructor, `write()`, and three new private
members plus one new private static helper.

**Reference:**
- `Tessera::MeshSeries`, its constructor, `write()`, `numFrames()` and `masterStem()`:
  [../tessera/src/Tessera_XdmfSeries.hpp:83-150](../../tessera/src/Tessera_XdmfSeries.hpp#L83-L150).
  Its validation rules — strictly increasing time, frame directory equal to master
  directory — are at lines 110-127.
- The timed `Tessera::writeMesh` overload used by the equal-time branch:
  [../tessera/src/Tessera_HDF5Writer.hpp:446-456](../../tessera/src/Tessera_HDF5Writer.hpp#L446-L456).
- The temporal-collection text and what Paraview requires of it:
  [../tessera/src/Tessera_Xdmf.hpp:247-310](../../tessera/src/Tessera_Xdmf.hpp#L247-L310)
  and Tessera's README "Time series: one Paraview dataset instead of N".
- The five-step order `write()` must preserve:
  [src/Beatnik_IOInterface.hpp:496-509](../src/Beatnik_IOInterface.hpp#L496-L509).

**Do:**

1. **Install the Tessera that has the header, first.** This checkout is `spack` mode
   (`BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh` to confirm before
   building). `tessera@develop` is a `develop` spec pointing at
   `../tessera`, which is at `2ba15cd`; `spack install` Tessera before Beatnik, or the
   `#include <Tessera.hpp>` will resolve to a prefix with no `Tessera_XdmfSeries.hpp`
   and the failure will look like a Beatnik error. Read
   `systems/tuolumne/claude.md` first, and do not `spack install` against the
   production environment while a production job is live.
2. Add the private static `masterStem( directory, prefix )` returning
   `directory.empty() ? prefix : directory + "/" + prefix`. Reuse it nowhere else —
   `checkpointStem()` keeps its own construction.
3. Add the three private members in this order after `_prefix`:
   `Tessera::MeshSeries _series;`, `double _last_frame_time = 0.0;`,
   `std::string _last_frame_stem;`. Initialize `_series( masterStem( _directory, _prefix ) )`
   in the constructor's member-init list, after `_prefix`. Comment on `_last_frame_*`
   that they are meaningful only when `_series.numFrames() > 0`.
4. Replace step 2 of `write()` — the bare `Tessera::writeMesh( mesh.tesseraMesh(), stem )`
   call — with the three-way decision, in this order:
   - `_series.numFrames() == 0`, or `time > _last_frame_time`: `_series.write( mesh.tesseraMesh(), stem, time )`, then set `_last_frame_time`/`_last_frame_stem`.
   - `time == _last_frame_time` **and** `stem == _last_frame_stem`: `Tessera::writeMesh( mesh.tesseraMesh(), stem, time )` and leave the series untouched. Comment *why* (the master already names this frame at this time) — the reason is not recoverable from the code.
   - otherwise: `throw std::runtime_error` naming both stems and both times and saying this is an invariant break, not a supported input.
   Compare `time` as `double`, from `static_cast<double>( header.time )`, once, into a
   local — `Real` may be `float` and the series stores `double`.
5. Leave steps 1, 3, 4 and 5 of `write()` byte-for-byte alone: the mkdir must still
   precede everything (the master lands in the same directory), and the barrier before
   the rank-0 `/beatnik` reopen must still be there.
6. Rewrite the file-header comment: a new section stating what the output directory
   now contains (N frames, N frame sidecars, one master, one `.xmfindex`, two
   `_latest` symlinks), which file to open in Paraview and that it needs the
   *temporal* XDMF3 reader, the equal-time rule and its reason, and that the
   `_latest.xmf` alias deliberately still names a frame. Update the "FILE NAMING"
   section, which currently enumerates the two files a save writes.

**Exit criterion:** `spack install` of Beatnik succeeds, and a batch-submitted run
(`flux batch` a script under `scripts/tuolumne/`; never launch interactively from a
login node) of `adaptive_mesh_bubble --steps 4 --checkpoint-every-steps 2
--checkpoint-dir <scratch>/grouped_io_t1` at 4 ranks leaves `<...>/checkpoint.xmf`
containing exactly one `CollectionType="Temporal"` and a `<Time Value=` count equal to
the number of **distinct** checkpoint stems in the directory — i.e. the duplicate
final frame appears once, not twice. And the failure direction: the run does **not**
throw `Tessera::MeshSeries::write: time must be strictly increasing`, which is what an
unguarded port produces at `finalize()`.

### T2 — A `unit`-tier test asserting the master's XDMF text — **DONE**

**Met, with one stated correction to the criterion's own arithmetic.**
`tests/unit_tests/Beatnik_Test_CheckpointSeries.cpp` was added and registered, and
appears in the installed manifest (`exe Beatnik_Test_CheckpointSeries`) after
`spack install`. Runs, all with `BEATNIK_TEST_SCRATCH` on lustre:

- **1 rank, job `f3T413MkAgo1`** — `[unit] SUMMARY: PASS (6/6 tests)`, with
  `Beatnik_Test_CheckpointSeries` PASS at 35/35 checks.
- **4 ranks, job `f3T413VBb5oM`** (`BEATNIK_UNIT_RANKS=4`) —
  `Beatnik_Test_CheckpointSeries` PASS on **all four ranks** (35/35 on rank 0, 3/3 on
  the other three, which run only the collective checks). The tier as a whole is
  `FAIL (5/6)`, and the one failure is **pre-existing and by design**:
  `Beatnik_Test_T2bOperators` asserts `comm_size == 1` deliberately
  ([tests/unit_tests/Beatnik_Test_T2bOperators.cpp:188-199](../tests/unit_tests/Beatnik_Test_T2bOperators.cpp#L188-L199)),
  so the tier can never be green at four ranks and this criterion's "green at four" was
  never achievable. Nothing was relaxed to accommodate it.
- **Member count: the tier now has SIX members, not four.** The criterion's "four, not
  three" counted only `BEATNIK_UNIT_TEST_SOURCES`; the manifest also carries
  `Beatnik_Test_PythonCompare` and its negative case. Five before, six now.

**Failure direction verified by actually doing it**, not by inspection: the equal-time
branch was temporarily replaced with an unconditional `_series.write()`, rebuilt, and
run as job `f3T3yTeaTTom`, where the test failed with
`Tessera::MeshSeries::write: time must be strictly increasing ... has time 0.200000 and
the previous frame had 0.200000` — the exact message, not an error elsewhere. The branch
was then restored (`git diff` against the T1 commit clean) and rebuilt, so what is
pushed is the guarded version and the two runs above are of that build.

**Depends on:** T1 **DONE**.

**Fill in:** new `tests/unit_tests/Beatnik_Test_CheckpointSeries.cpp`; one entry added
to `BEATNIK_UNIT_TEST_SOURCES` in
[tests/unit_tests/CMakeLists.txt:43-54](../tests/unit_tests/CMakeLists.txt#L43-L54).
That list is single-sourced — the same loop applies the `unit` label, appends to the
installed manifest and installs the binary
([tests/unit_tests/CMakeLists.txt:79-98](../tests/unit_tests/CMakeLists.txt#L79-L98))
— so no other build file is touched.

**Reference:**
- Test shape, `Recorder`, `BEATNIK_CHECK_TRUE`/`_EQ`/`_CLOSE`, and the
  `MPI_Allreduce( MPI_MAX )` single-verdict `main`:
  [tests/unit_tests/Beatnik_Test_TangentialRelaxation.cpp:910-947](../tests/unit_tests/Beatnik_Test_TangentialRelaxation.cpp#L910-L947)
  and [tests/unit_tests/Beatnik_TestAssert.hpp:183-240](../tests/unit_tests/Beatnik_TestAssert.hpp#L183-L240).
- Building a mesh without a solver:
  [tests/unit_tests/Beatnik_Test_MeshGeometry.cpp:111-113](../tests/unit_tests/Beatnik_Test_MeshGeometry.cpp#L111-L113)
  (`mesh_type mesh( MPI_COMM_WORLD ); mesh.generateIcosphere( ... )`).
- Scratch-path resolution: the Conventions table row, and
  [tests/regression_tests/Beatnik_Test_InitialConditions.cpp:271-282](../tests/regression_tests/Beatnik_Test_InitialConditions.cpp#L271-L282).
- What the emitted text must look like:
  [../tessera/src/Tessera_Xdmf.hpp:247-310](../../tessera/src/Tessera_Xdmf.hpp#L247-L310).

**Do:**

1. Drive `CheckpointIO` **directly**, not through `Solver` — this is a `unit` test and
   the solver loop is not what is under test. Build an icosphere with
   `generateIcosphere`, call `AdaptiveMesh::resetReferenceState( mesh )` so the face
   user pack is not written from uninitialized memory, construct
   `CheckpointIO( MPI_COMM_WORLD, dir, "checkpoint" )`, and call `write()` four times:
   times `0.0, 0.1, 0.2` with steps `0, 1, 2`, then a fourth call repeating
   `(0.2, 2)` — the `finalize()` shape from fact 1 of
   [Read this first](#read-this-first).
2. Assert nothing about field *values*. The vertex user pack is not initialized by
   this test, so `u0`/`u1`/`u2` hold whatever was in memory; the subject here is the
   emitted XDMF text.
3. On rank 0 only, read `<dir>/checkpoint.xmf` as text and assert: exactly one
   `CollectionType="Temporal"`; exactly **three** occurrences of `<Time Value=`, with
   values `0`, `0.1`, `0.2` (parse and compare with a tolerance — the writer's `%g`
   formatting is not a fixed string); three `<Topology` and three `<Geometry`; and
   that each child names only its own frame's `.h5` basename, i.e. each of the three
   frame basenames appears in the file and no basename appears in a sibling's child
   block. Assert every frame `.h5` and every frame `.xmf` still exists.
4. Reduce the verdict with `MPI_Allreduce( MPI_MAX )` in `main`, and have every rank
   call `report()`, so the log names which rank failed. The `CheckpointIO::write`
   calls are collective and must happen on every rank; only the text assertions are
   rank-0.
5. Both failure directions, each asserting the master is left **byte-unchanged**:
   record the master's bytes, then (a) call `write()` with a *decreasing* time and
   confirm T1's `std::runtime_error` is thrown — catch it and check the message names
   both stems, not merely that something threw; (b) confirm the fourth
   equal-time-equal-stem call of step 1 did **not** add a fourth `<Time Value=`.
   Direction (a) is the guard doing its job; direction (b) is the equal-time rule
   doing its job, and the two are distinguishable only by asserting on both.
6. Add the source to `BEATNIK_UNIT_TEST_SOURCES` with a one-line comment naming this
   task, matching the existing entries' style.

**Exit criterion:** the `unit` tier is green at one rank and at four —
`BEATNIK_UNIT_RANKS=4 flux batch scripts/tuolumne/unit_tests.flux` — with
`Beatnik_Test_CheckpointSeries` among the members that ran, and the tier's member
count in the log is **four**, not three. And the failure direction: temporarily
reverting T1's equal-time branch to an unconditional `_series.write()` makes this test
fail with the `time must be strictly increasing` message rather than passing or
erroring elsewhere. Restore the branch afterwards.

### T3 — Documentation and the ship-gate re-run — **NOT STARTED**

**Depends on:** T1 **DONE**, T2 **DONE**.

**Fill in:** [README.md](../README.md), [docs/design.md](../docs/design.md),
[CLAUDE.md](../CLAUDE.md).

**Reference:** the README's run section
([README.md:160-175](../README.md#L160-L175)) and the `Checkpoint / restart` option
row ([README.md:195](../README.md#L195)); Tessera's README "Time series: one Paraview
dataset instead of N" for the Paraview-reader caveat worth restating rather than
rediscovering.

**Do:**

1. README: add a short "Checkpoint output" subsection after the run block listing what
   a `--checkpoint-dir` run leaves and naming `<prefix>.xmf` as the file to open in
   Paraview, with the `Xdmf3ReaderT` caveat and the `grep -c '<Time Value='` check.
   The CLI surface does not change, so the option table needs no edit.
2. README "Known Issues": the restart-versus-series inconsistency from
   [Deliberate deviations](#deliberate-deviations) — what it is, that it is unreachable
   while `CheckpointIO::read` throws, and that framework.md T5b owns it. State whether
   it is a regression from this work or pre-existing (it is new, and latent).
3. docs/design.md: it has no I/O section today. Add one only if the reasoning does not
   already live in the `Beatnik_IOInterface.hpp` header comment T1 rewrote; a second
   copy is how these drift. Prefer a one-line pointer to that header.
4. Re-run the **whole** ship gate and record the result:
   `scripts/tuolumne/run_regression_minset.flux` — `regression` label x {SERIAL, HIP} x
   ranks 1-6 = 60 launches, with `BEATNIK_TEST_SCRATCH` on a parallel filesystem (the
   checkpoints go through MPI-IO; a node-local scratch fails every multi-node launch).
   Update CLAUDE.md "Minimum test set" so the "full sweep as of T4c" stamp names this
   work instead, and state that the tier still has five members — this change adds a
   `unit` member and no gate member.

**Exit criterion:** the gate is 60/60 green in a single sweep whose log is cited in the
progress log by job ID, and `grep -rn "xmf" README.md` shows the new subsection. The
failure direction the gate covers here is regression, not feature: the five members
compare `.h5` datasets through `compare_output.py` and must be **unchanged** by a
change that only adds light data — a member that now fails localizes the break to the
frame write, since nothing else moved.

## Known risks

**R1 — Paraview opens the master but shows one timestep.** Presents as "the fix did
not work" with a valid-looking file. Almost always the reader, not the file: only the
*temporal* XDMF3 reader (`Xdmf3ReaderT`) walks a temporal collection. The
distinguishing measurement is `grep -c '<Time Value=' out/checkpoint.xmf` — if that
is N, the file is right and the reader choice is wrong. T2 asserts on the text for
exactly this reason: it removes Paraview from the loop.

**R2 — Two frames collide on one stem.** `timeKey()` formats to six decimals
([src/Beatnik_IOInterface.hpp:464-483](../src/Beatnik_IOInterface.hpp#L464-L483)), so
two checkpoints less than 1e-6 apart in time produce the same `timeKey`; the stems
then differ only by the step field, but if the step also matched, the second write
would silently overwrite the first while the master listed one `h5name` twice at two
different times. Pre-existing for the `.h5` files; this change makes it visible in the
master. Presents as a Paraview animation that appears to freeze on one geometry for
two steps. Distinguished from R1 by looking at the child `<DataItem>` paths: R1 has N
distinct paths, R2 has a repeat. Not fixed here — the fix is a step-only stem, which
would break the filename pairing with the Python gold files that
`compare_output.py` depends on.

**R3 — The equal-time branch masks a real regression later.** If a future change makes
`finalize()` write a *different* state at the same `(time, step)`, the branch would
rewrite the frame and the master would still be correct, so nothing would complain —
but the frame's contents would have changed identity. The guard against this is
already in T1: the branch requires the **stem** to match as well as the time, and
anything else throws. If a later task moves `recordLastFiniteState()` relative to
`checkpointDue()`, that throw is what will fire, and it should be read as a real
signal rather than relaxed.

**R4 — The master is rewritten every frame, O(frames) rank-0 text per frame.** At a
few thousand checkpoints this is still immaterial against a collective HDF5 write, and
it is what buys a valid master from a killed run
([../tessera/src/Tessera_XdmfSeries.hpp:25-33](../../tessera/src/Tessera_XdmfSeries.hpp#L25-L33)).
It would present as rank-0 time growing with the frame index in a long production run.
Measure before acting: `TIMER_WRITE_MESH` does **not** cover it (the master write
happens after the timed `writeMesh` returns), so a profile that shows no growth there
has not ruled this out.

**R5 — Beatnik built against a Tessera prefix that predates `2ba15cd`.** Presents as a
compile error on `Tessera::MeshSeries` being undeclared, or — worse, if some other
Tessera version is on the include path — as `Tessera.hpp` failing to find
`Tessera_XdmfSeries.hpp`, which reads as a Beatnik include bug. T1 step 1 exists to
prevent it. Confirm with `ls $(spack location -i tessera)/include/Tessera_XdmfSeries.hpp`
before concluding anything else about a build failure in T1.
