# Large roll-up — the default rising bubble past 200,000 vertices

**Status:** NOT STARTED

## Problem

Beatnik reproduces the Python reference's default rising-bubble configuration
(`tasks/milestone1.md`, COMPLETE): README configuration (a) of
`~/research-bridges/zmodel-steve/zmodel3d-amr`, run through
`examples/02_adaptive_mesh_bubble` with the full metric remesher — split,
collapse, quality flips, guarded tangential smoothing — and the isotropic
cleanup. That configuration is a **small** run. Its sizing field holds the mesh
at 518–610 vertices to `time` 1.24, and no Beatnik dynamic-remesh run has
exceeded about 10,000 vertices.

**End state.** A production run of `adaptive_mesh_bubble` with configuration
(a)'s physics — `--A 0.3 --g 1.0 --mu 0.002`, a sphere,
`--viscosity-mode laplace-beltrami`, `--br-approximation direct`,
`--source-quadrature vertex`, `--adaptive-dt`, `--dynamic-remesh`,
`--isotropic-cleanup` — with resolution, step size and kernel
desingularization set by D1, that:

- carries **at least 200,000 vertices** at some checkpoint;
- runs as far in physical time as it stays finite, toward the `t_end` D1 sets,
  which is at least 2.5;
- writes checkpoints to Lustre throughout, and is **restartable**, so it spans
  as many chained batch jobs as it needs.

Restart (`--restart-from`) is implemented as part of this work (T5b).

**Out of scope:**

- **The Canopy FMM** (`--br-approximation fmm`). It is slower than the direct
  sum at every measured size, up to icosphere level 6, and has never run with a
  changing vertex count (Approach).
- **Comparison with the Python reference beyond the T5b restart check.** The
  reference is serial `O(N^2)` Python at about 1.1 s/step on 600 vertices
  (`tasks/milestone1-progress-log.md` `## M1-G3`), so no gold set exists at
  this size and none is generated.
- **Nonlocal proximity and surgery** (`--remesh-proximity`,
  `--remesh-surgical-proximity`, task T4e in `tasks/framework.md`). A run that
  reaches a sheet self-approach ends there; R3.
- The tight remesh profile (`--remesh-tight-after`), field filtering and the
  sheet-vector model (T5c), load balancing (T5d), plotting and video.

## Approach

**Measure, decide, pilot, then run.** P1 measures cost, memory and early
stability at the two candidate starting resolutions with no code change. D1
turns those numbers into the production parameter set. T5b makes restart work.
V1j/V1 run the parameter set to `t_end` at reduced resolution, across one
restart, which shows that the configuration survives before the expensive run
is spent. R1j/R1 is the production run.

Decisions, with their reasons:

- **Direct Birkhoff-Rott sum.** `BRSolverDirect` circulates the source block
  around a ring of ranks
  ([src/Beatnik_BRSolverDirect.hpp:100-106](../src/Beatnik_BRSolverDirect.hpp#L100-L106),
  `ringAccumulate` at `:274`), so its per-rank work is `O(N^2 / P)`. The FMM
  costs 15–40× the direct estimate per step at levels 4–6 on HIP, with no
  crossover in range (`tasks/canopy_profiling-progress-log.md`, table "(d)
  `fmm` against estimated `direct` per step"). Every FMM test also freezes
  connectivity: `Beatnik_Test_Milestone0Fmm` sets
  `p.dynamic_remesh = false; p.amr.refine_every = 0`
  (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1059-1063`). The
  adapter is written to fall back to a full `setup()` when the source map goes
  stale ([src/Beatnik_FarFieldInterface.hpp:1235-1245](../src/Beatnik_FarFieldInterface.hpp#L1235-L1245)),
  but no run has exercised that path. Switching to the FMM would need its own
  task.
- **Resolution comes from a fine starting mesh held by `--remesh-h-max`.**
  The alternative is to grow from a coarse one. An icosphere at subdivision
  level `n` has `10·4^n + 2` vertices:
  - level 6: 40,962 vertices, mean edge about `4.1e-3` on the radius-0.25
    sphere (icosahedron edge `1.0515 R`, halved per level);
  - level 7: 163,842 vertices, mean edge about `2.05e-3`.

  Growing from 600 vertices is capped at `max_splits_per_pass` = 300 per pass
  ([src/Beatnik_Params.hpp:675](../src/Beatnik_Params.hpp#L675)), so it would
  take at least about 660 remesh steps. The default `h_max` = 0.05 (`:643`)
  would also coarsen a fine start: the collapse pass removes up to
  `max_collapses_per_pass` = 120 edges per pass (`:679`). P1 measures both the
  hold and the coarsening; D1 picks the starting level and `h_max`.
- **The step size is set explicitly.** The adaptive dt is
  `dt0 · min(1, h_min/h_min⁰)^p`, floored at `--min-dt`
  ([src/Beatnik_TimeIntegrator.hpp:176-230](../src/Beatnik_TimeIntegrator.hpp#L176-L230),
  defaults [src/Beatnik_Params.hpp:836-878](../src/Beatnik_Params.hpp#L836-L878):
  `dt` 0.003, `min_dt` 2.5e-4, `dt_edge_power` 1). `h_min⁰` is the starting
  mesh's own minimum edge, so the rule is **relative**. A level-7 start begins
  at `dt` = 0.003 whatever its edge length, and the floor stops dt following
  the mesh below it. D1 sets `--dt`, `--min-dt` and `--dt-edge-power` from P1's
  stability arms.
- **`--eps` is decided in D1, not assumed.** At the default `--eps 0.025` the
  kernel's smoothing length is about 12× a level-7 edge, so extra vertices
  resolve the smoothing rather than finer roll-up. A smaller eps resolves more
  of the sheet but shortens the dynamics' time scale and the stable dt. That is
  a physics choice, and D1 records it with its reason.
- **Restart is implemented here.** `RestartReader::load` and
  `CheckpointIO::read` throw (Current state). T5b keeps its ID from
  `tasks/framework.md`, whose T5b entry points here. A multi-segment run is
  chained by restarting each segment from the previous segment's
  `checkpoint_latest.h5`.
- **Instrumentation is in the batch scripts, not the code.** The progress line
  is a port of the reference's format
  ([src/Beatnik_Diagnostics.hpp:348-379](../src/Beatnik_Diagnostics.hpp#L348-L379))
  and stays unchanged. Wall time per step comes from timestamping rank 0's
  progress lines as they arrive. Memory comes from `/usr/bin/time -f
  rss_kb=%M` around each rank (peak RSS), and, for the long jobs, from a
  background `ps -C adaptive_mesh_bubble -o rss=` sampler on the first node.

### Production configuration

The values below are D1's to set. Until D1 is **DONE**, every row reads `D1`
and no task after D1 may start.

| Setting | Option | Value | Source |
| --- | --- | --- | --- |
| Physics | `--A 0.3 --g 1.0 --mu 0.002 --viscosity-mode laplace-beltrami --br-approximation direct --source-quadrature vertex --adaptive-dt --dynamic-remesh --isotropic-cleanup` | fixed | configuration (a) |
| Kernel desingularization | `--eps` | D1 | D1 |
| Starting mesh | `--icosphere-subdivisions` | D1 | P1 |
| Sizing | `--remesh-h-max`, `--remesh-h-min`, `--remesh-sagitta-tolerance` | D1 | P1 |
| Edit caps | `--remesh-max-splits`, `--remesh-max-collapses` | D1 | P1 |
| Step size | `--dt`, `--min-dt`, `--dt-edge-power` | D1 | P1 |
| End time | `--t-end` | D1 (≥ 2.5) | D1 |
| Checkpoint cadence | `--checkpoint-every-time` | D1 | P1 |
| Ranks / nodes, segment walltime | — | D1 | P1 |
| Pilot resolution | starting level and `h_max` for V1j | D1 | P1 |
| Pilot quality floor | minimum `minQ` V1 asserts | D1 | P1 |

### Conventions

| Convention | Why | Check |
| --- | --- | --- |
| Every new script and source file carries the BSD-3-Clause header block with `SPDX-License-Identifier: BSD-3-Clause`, in the file type's comment style. | Project rule (CLAUDE.md). | `grep -L SPDX-License-Identifier` over the diff's new files is empty. |
| Batch scripts live in `scripts/tuolumne/large_rollup_*.flux`, find `BEATNIK_REPO` by walking up from `PWD`, source `scripts/lib/beatnik_env.sh` first, and are `set -u` clean. | `flux batch` spools the script to `/var/tmp`, so `BASH_SOURCE` points at the spool (`systems/tuolumne/claude.md` §5). | By reading: the `beatnik_find_repo` block of `scripts/tuolumne/rising_bubble_amr_sweep.flux` is reused; `source` precedes every other command. |
| P1 and T5b scripts use the **dev** env (no `BEATNIK_USE_PROD`). V1j and R1j scripts set `BEATNIK_USE_PROD=1`, after the prod env has been pulled to the intended commits and `spack install`ed. | Short probes follow the working tree. Long jobs must not have their binary rebuilt under them; a reinstall during a live run SIGBUS-killed one (rc 135, CLAUDE.md). | `grep -n BEATNIK_USE_PROD scripts/tuolumne/large_rollup_*.flux`. |
| Long jobs run from a staged copy: `stage/bin/adaptive_mesh_bubble` copied from `beatnik_exe adaptive_mesh_bubble`, plus `stage/COMMIT` (`git rev-parse HEAD` and the `git status --porcelain` count, written at staging time). The job echoes `stage/COMMIT`, never `git rev-parse HEAD`. | The tree moves between submission and start. Beatnik, Tessera and Canopy are header-only in the example, so the copied executable freezes the code (`scripts/tuolumne/m1_milestone_run.flux:45-56`). | By reading the script; the job log's first lines echo `stage/COMMIT`. |
| Queue: 1–2 nodes with `-t` ≤ 45m go to `pdebug`; anything longer or wider goes to `pbatch`. `-t Nm`, never `--time`. | `systems/tuolumne/claude.md` "Choosing a queue". | `grep -n '# flux: -q\|# flux: -t' scripts/tuolumne/large_rollup_*.flux` against each script's node count and `-t`. |
| All output under `/p/lustre5/stewartj/beatnik/large_rollup/<task ID>/`. | Checkpoints are MPI-IO, so node-local scratch fails across nodes (CLAUDE.md). | `grep -n lustre5` in each script. |
| Every launch inside a job carries its own `flux run -t` cap, and prints one summary line `[large_rollup] <tag> rc=<rc> wall=<s>` after it. | One hung launch must not eat the job, and the post-job task reads the summary lines. | By reading the script; the job log has one summary line per launch. |
| Never run clang-format, `clangformat.sh` or `cabana-format`; edit C++ in the surrounding style. | CLAUDE.md. | The diff touches no line the task did not change. |
| Comments and docs describe the code as it is now: no "previously", "no longer", "used to". | CLAUDE.md. | `git diff \| grep -nE 'previously\|no longer\|used to\|is gone'` is empty. |
| A change to an example's accepted arguments or behaviour updates `README.md` in the same change; a fixed Known Issue is deleted from README "Known Issues" in the change that fixes it. | CLAUDE.md. | By reading the diff of `README.md`. |
| Restart code takes its communicator from the mesh (`mesh.comm()`) and names no `MPI_COMM_WORLD`. | Library code is communicator-agnostic. | `git diff src/ \| grep MPI_COMM_WORLD` is empty. |
| Every collective in `read`/`load` is reached by every rank in the same order. A check that can fail is decided on every rank from replicated data (or reduced first), and the throw follows the decision on every rank. | A throw that only some ranks reach deadlocks the rest. | By reading: no collective sits after a rank-dependent branch or return. |
| Restart failures the code can detect throw `std::runtime_error` naming the path and the reason, **before** `Tessera::readMesh` is called. A silent cold start is never a fallback. | A structural mismatch inside `readMesh` is an uncatchable `MPI_Abort` ([src/Beatnik_IOInterface.hpp:757-765](../src/Beatnik_IOInterface.hpp#L757-L765)), and a silent cold start produces a plausible run of the wrong thing ([src/Beatnik_Restart.hpp:162-167](../src/Beatnik_Restart.hpp#L162-L167)). | The unit test's failure directions (T5b). |
| Kernels use `KOKKOS_LAMBDA` and capture locals, never `this`; every View is labelled; no View is allocated inside a per-step loop; the host reads device data only after a fence or `deep_copy`. | These pass on SERIAL and fail on HIP. | By reading; `grep -n 'KOKKOS_CLASS_LAMBDA\|\[=\]\|\[&\]'` in the diff. |
| A new test joins `BEATNIK_UNIT_TEST_SOURCES` in `tests/unit_tests/CMakeLists.txt` with a comment saying what it checks and that it needs `BEATNIK_TEST_SCRATCH`; it decides its own verdict and returns non-zero on failure. | The unit tier's registration and installed-path convention (`tests/unit_tests/CMakeLists.txt:12-40`). | The test appears in `scripts/tuolumne/unit_tests.flux`'s output at `BEATNIK_UNIT_RANKS=1` and `4`. |
| New code builds warning-free in the dev env (`spack install`, spack mode; never `cmake`/`make`). | The checkout is spack mode (`profile.local.sh`). | `spack install` output has no new warning from a changed file. |
| Math written into this file or the log uses KaTeX `$…$`; math in session replies uses Unicode. | CLAUDE.md "Math in markdown". | By reading. |

### Deliberate deviations

- **No reference comparison at production size.** The run is checked for
  internal consistency: volume drift, minimum quality, finiteness, and the
  pilot's survival at reduced resolution. No `.npz` gold exists or is made at
  this size (Out of scope).
- **A restarted run is not the uninterrupted run.** On load, the AMR change
  indicators re-base to "no change so far", faithfully to the reference
  ([src/Beatnik_Restart.hpp:28-56](../src/Beatnik_Restart.hpp#L28-L56)). The
  dynamic-remesh path re-bases them after every remesh anyway
  (`Solver::rebuildState`, `tasks/milestone1.md` Current state). Three
  differences remain after a restart:
  - `readMesh`'s fresh block partition changes the reduction order;
  - the gid order changes Tessera's tie-breaks;
  - the gid-renumbering cadence counter `DynamicRemesh::_remesh_calls`
    ([src/Beatnik_DynamicRemesh.hpp:1436-1440](../src/Beatnik_DynamicRemesh.hpp#L1436-L1440))
    restarts at 0.

  So no task asserts that a restarted dynamic-remesh run equals an
  uninterrupted one. T5b's exactness check uses a frozen mesh.

## Current state

- **Restart throws.** `RestartReader::load` is `BEATNIK_NOT_IMPLEMENTED`
  ([src/Beatnik_Restart.hpp:177-185](../src/Beatnik_Restart.hpp#L177-L185)),
  and so is `CheckpointIO::read`
  ([src/Beatnik_IOInterface.hpp:767-773](../src/Beatnik_IOInterface.hpp#L767-L773)).
  `Solver::setup` dispatches to `load` under `--restart-from`
  ([src/Beatnik_Solver.hpp:241-262](../src/Beatnik_Solver.hpp#L241-L262)). The
  writer is complete: a checkpoint carries the mesh, the three vertex fields
  and the `/beatnik` scalars `state_model`, `time`, `step`, `initial_volume` and
  `initial_min_edge` (schema at `src/Beatnik_IOInterface.hpp:70-90`).
  `read`'s doc comment states the two `readMesh` traps:
  - the halo comes back 1-deep and must be widened to 2;
  - a structural mismatch is an `MPI_Abort`.

  README "Known Issues" carries the restart master-XDMF problem
  (`README.md:817-832`): each process's `MeshSeries` starts empty, so a
  restart rewrites `<prefix>.xmf` with only post-restart frames while Tessera
  appends to `<prefix>.xmfindex`.
- **`--restart-from`, `--t-end`, `--checkpoint-every-time` and
  `--icosphere-subdivisions` are parsed** by the example
  (`examples/02_adaptive_mesh_bubble/InputFile.hpp:303,358-360,558-562`). A
  restart's `--steps` is a local budget (`src/Beatnik_Restart.hpp:89-94`).
- **Nonlocal gap diagnostics are not computed.** `Diagnostics::compute` sets
  `min_nonlocal_distance` and `gap_edge_ratio` to `inf`, and the two exact-gap
  fields to `nan`
  ([src/Beatnik_Diagnostics.hpp:311-318](../src/Beatnik_Diagnostics.hpp#L311-L318)),
  so `minGap=inf` on every progress line means "not measured".
  `--exact-gap-diagnostics` reaches only those fields.
- **Long runs measured so far** (before T4d's collapse, flip, smoothing and
  cleanup passes existed). Four variants of the level-3 split-only bubble
  (`--remesh-collapse-factor 0 --remesh-smooth-iters 0
  --remesh-flip-min-gain 1e12 --no-isotropic-cleanup`), 4 ranks HIP, job
  `f3e92SU62uew`, script `scripts/tuolumne/rising_bubble_amr_sweep.flux`, log
  `/p/lustre5/stewartj/beatnik/rising_bubble_amr_sweep/rb_amr_sweep.f3e92SU62uew.log`:

  | Variant | Stopped at `t` | Last progress line: `t`, `V`, `minQ`, `minEdge`, `dt` |
  | --- | --- | --- |
  | A | 1.6898 | 1.6074, 10,066, 0.046, `5.1e-4`, `5.0e-5` |
  | B | 2.3769 | 2.3118, 6,775, 0.11, `7.5e-4`, `6.5e-5` |
  | C | 2.2989 | 2.2113, 5,643, 0.083, `1.2e-3`, `1.05e-4` |
  | D | 2.5268 | 2.5134, 4,956, 0.033, `5.0e-4`, `5.0e-5` |

  Every variant stopped on `nonfinite mesh-RHS state`, with `minQ` falling
  over its last few progress lines and dt at or near its `--min-dt` of 5e-5.
  Variant A ran 8457 steps in 476 s, about 0.056 s/step at 2,000–10,000
  vertices.
- **Direct-sum cost at fixed connectivity**: the direct estimate is 0.108
  s/step at level 6 HIP np1 and 0.125 s/step at np4
  (`tasks/canopy_profiling-progress-log.md`, table "(d)"). It is an estimate
  from the `[acc]` evaluations, not a timed run. Multi-rank scaling of the
  direct sum at more than 41k vertices is unmeasured.
- **Per-rank RSS grows slowly** in 4-rank dynamic-remesh runs (README Known
  Issues, `README.md:878-893`): 6–22 MB per rank over 600 steps at about 1040
  faces. Whether it plateaus is unmeasured.

## Progress log

Session record: [large_rollup-progress-log.md](large_rollup-progress-log.md).
Consult it before implementing a task, changing a signature, or reopening a
question this document treats as settled.

> Each task is reviewed with `review-task` before any task that depends on it
> starts. A task is ready to build on when it is marked **DONE** and the log has
> a `## <ID> review` section whose verdict is `Passed`. The document's status
> becomes COMPLETE only when a `review-task` sweep of the whole feature passes.

## Task sequence

### P1 — measure cost, memory and early stability at levels 6 and 7 — **NOT STARTED**

**Depends on:** none.
**Fill in:** `scripts/tuolumne/large_rollup_probe.flux` (new). No source change.
**Removes:** nothing.
**Reference:** `scripts/tuolumne/rising_bubble_amr_sweep.flux` for the
variant-loop shape and repo discovery; `systems/tuolumne/claude.md` §4–5 for
the launch line.
**Do:**

1. Base arguments, every launch: configuration (a)'s physics (Production
   configuration, first row), `--eps 0.025`, `--steps 60`,
   `--progress-time-interval 0`. The progress line then prints at step 1 and
   every 6 steps (`Solver::writeProgressIfDue`,
   `src/Beatnik_Solver.hpp:994-1012`). Add `--checkpoint-dir <out>/<tag>
   --checkpoint-every-steps 60` so each launch writes one checkpoint at its
   final size.
2. Launches, each `flux run -t 6m`, with `--gpus-per-task=1
   --cores-per-task=24` and the node count derived from the ranks:

   | Tag | Level | `--remesh-h-max` | `--dt` | Ranks |
   | --- | --- | --- | --- | --- |
   | `L6_np1` | 6 | `4.1e-3` | 0.003 | 1 |
   | `L6_np4` | 6 | `4.1e-3` | 0.003 | 4 |
   | `L7_np1` | 7 | `2.05e-3` | 0.003 | 1 |
   | `L7_np4` | 7 | `2.05e-3` | 0.003 | 4 |
   | `L7_np4_dt1e-3` | 7 | `2.05e-3` | 0.001 | 4 |
   | `L7_np4_hmax_default` | 7 | (default 0.05) | 0.003 | 4 |

   The last row is the control. It shows the collapse pass coarsening a fine
   start when `h_max` does not hold it.
3. Wrap each rank's executable in `/usr/bin/time -f "rss_kb=%M"`. Pipe the
   launch's stdout through a loop that prefixes each line with
   `$(date +%s.%N)`. Write each launch's log to `<out>/<tag>/launch.log`, then
   print the summary line.
4. Job: 1 node, `-t 45m`, `pdebug`, dev env, output under
   `/p/lustre5/stewartj/beatnik/large_rollup/P1/`. Expected run time is about
   15 minutes. That is an estimate: level-7 direct at about
   `(163842/40962)^2 × 0.108 ≈ 1.7` s/step np1 gives about 100 s for 60
   steps, and the six caps bound the worst case at 36 min. If the job is still
   running at 20 minutes, wait for it in this session and record the overrun
   in the log.
5. Record in the log, per launch:
   - `V`/`F` at step 1 and step 60;
   - median s/step over the progress intervals after step 1;
   - setup wall (launch start to the step-1 line);
   - dt at step 60;
   - `minQ`, `minEdge` and `volRel` at the last line;
   - peak `rss_kb` per rank;
   - the stop line, if any.

   From these, extrapolate the production cost: s/step at 200,000 vertices
   for np 4, 8 and 16 (state the scaling exponent you assume and why), times
   the step count to `t = 2.5` at the measured dt.

**Exit criterion:** the log's `## P1` section has the table above filled for
all six launches. Each launch either reached `step=60`, or has its stop line or
its cap `rc` recorded with the last progress line before it. The control
direction holds: `L7_np4_hmax_default`'s `F` at step 60 is **below** its
step-1 `F`, while `L7_np4`'s step-60 `F` is within 5 % of its step-1 `F` or
above it. If the control does not coarsen, `h_max` is not what holds the
resolution; record that and do not proceed to D1 until the log explains it.

### D1 — choose the production parameter set — **NOT STARTED**

**Depends on:** P1.
**Fill in:** this document's [Production configuration](#production-configuration)
table; the log's `## D1`. No code or script.
**Removes:** the `D1` placeholders in that table.
**Reference:** P1's measurements; Current state's long-run table; R1–R4.
**Do:** this is a decision session held with the user. The session prepares
options and their measured consequences, and the user chooses; the session
does not fill the table alone. Present:

- **eps:** 0.025 against a value tied to the edge length. For each, the
  stable dt P1 implies and the step count to `t_end`.
- **Starting level and sizing:** level 7 held at `h_max` ≈ `2.05e-3`, against
  level 6 grown by AMR. Include `h_min`, sagitta and the edit caps.
- **Step size:** `--dt`, `--min-dt` and `--dt-edge-power`, from the
  `L7_np4` and `L7_np4_dt1e-3` arms.
- **Ranks and nodes:** from the np1/np4 scaling and P1's cost extrapolation.
- **Segment walltime and checkpoint cadence:** `pbatch`'s maximum `-t`, from
  `flux queue list` or the system documentation, and checkpoint write cost
  from P1's final checkpoint.
- **`t_end`:** at least 2.5.
- **Pilot resolution:** about 4–16× fewer vertices than production, at the
  same eps and dt rule, and the pilot's quality floor.

**Exit criterion:** every row of the Production configuration table holds a
value and a source, and the log's `## D1` section records the alternatives
considered, the measured number behind each chosen value, and the estimated
production node-hours. Failure direction: if P1's extrapolation puts the
production run above the node-hours the user accepts, D1 records that and
stops. A smaller target or an FMM task is then a new design decision, not a
change made here.

### T5b — restart from a Beatnik checkpoint — **NOT STARTED**

**Depends on:** none. It may run in parallel with P1 and D1.
**Fill in:**
- `CheckpointIO::read`
  ([src/Beatnik_IOInterface.hpp:767-773](../src/Beatnik_IOInterface.hpp#L767-L773))
  and the restart side of the master-series handling in the same class;
- `RestartReader::load`
  ([src/Beatnik_Restart.hpp:177-185](../src/Beatnik_Restart.hpp#L177-L185));
- the `src/Beatnik_Solver.hpp:247-248` comment that says `load` throws;
- `tests/unit_tests/Beatnik_Test_Restart.cpp` (new), and its entry in
  `tests/unit_tests/CMakeLists.txt`;
- `scripts/tuolumne/large_rollup_t5b.flux` (new);
- `README.md`.

**Removes:**
- `BEATNIK_NOT_IMPLEMENTED` in `read` and in `load`;
- the `(void)` casts in both;
- every comment calling either a stub: `src/Beatnik_Solver.hpp:247-248`,
  `src/Beatnik_IOInterface.hpp:256-261`,
  `src/Beatnik_MeshInterface.hpp:344-345`;
- the README Known Issue on the restart master XDMF (`README.md:817-832`);
- the README sentence at `:826` saying `--restart-from` cannot complete.

**Reference:**
- `run_adaptive_mesh_bubble.py::load_state_checkpoint` (lines 993-1033) and
  `::main` (1199-1214) in the reference tree;
- `read`'s and `load`'s doc comments, which already specify the contract:
  - checks before `readMesh`;
  - `rebuildHalo( mesh, halo, 2 )` after it;
  - the missing-field fallbacks;
  - `RestartState.from_checkpoint` and `amr_reference_rebased` both true;
- `Tessera_IO.hpp` in `../tessera` for `readMesh` and `MeshSeries`. **This task
  is the first to open them.**

**Do:**

1. Read `readMesh` and `MeshSeries`. Record in the log whether `readMesh`
   preserves vertex gids, and whether a `MeshSeries` can reopen an existing
   `<prefix>.xmfindex`.
2. Implement `read`. Make the cheap checks first, decided identically on every
   rank:
   - the file exists;
   - `/beatnik` is present;
   - `state_model` is recognized.

   Then call `readMesh`, widen the halo to depth 2, and read the `/beatnik`
   scalars into the `CheckpointHeader`.
3. Implement `load` on `read`. Fill `RestartState` from the header, with
   `from_checkpoint` and `amr_reference_rebased` true. `Solver::setup` already
   seeds the material position, re-bases the AMR state and carries the
   scalars (`src/Beatnik_Solver.hpp:262-300`); change `setup` only where the
   loaded state needs it, and enumerate any changed signature's callers in the
   log.
4. Resolve the master-series issue in this change. Reopen the series from
   `<prefix>.xmfindex` if Tessera supports it, so the master lists pre- and
   post-restart frames. Otherwise, a restart that would write into a directory
   whose master already exists throws, naming the directory and the remedy (a
   new `--checkpoint-dir`). Record which, and why, in the log. Either way the
   master and the `.xmfindex` never disagree silently.
5. The unit test, at the tier's 1 rank and at `BEATNIK_UNIT_RANKS=4`:
   - **round trip:** write a checkpoint of a level-2 cold start with
     non-trivial `potential` and material position, read it into a fresh
     mesh, and assert:
     - the five header scalars are equal exactly;
     - `V`/`E`/`F` are equal;
     - every vertex's position and fields are equal exactly, paired by gid if
       gids survive, otherwise by exact position;
     - the halo depth is 2;
   - **failure directions:** a missing path throws `std::runtime_error` whose
     message contains the path; a file written by `Tessera::writeMesh` with no
     `/beatnik` group throws naming `/beatnik`. Neither aborts.
6. The script (dev env, 1 node, `pdebug`, `-t 30m`, expected under 10 minutes)
   runs, under `/p/lustre5/stewartj/beatnik/large_rollup/T5b/`:
   - **frozen-mesh restart against the reference:**
     1. Beatnik, configuration (a) with `--no-dynamic-remesh --refine-every 0`,
        level 2, 4 ranks HIP, 10 steps, checkpoint every 5 steps.
     2. Beatnik restarted from the step-5 `.h5` for `--steps 5`, at 4 ranks and
        at 1 rank, each into a new `--checkpoint-dir`.
     3. `m1d3_refrun.py convert` of the step-5 `.h5`.
     4. The reference restarted from it for 5 steps (`m1d3_refrun.py run`,
        reference stage `/p/lustre5/stewartj/beatnik/M1-G3j/stage`).
     5. `compare_output.py` of each Beatnik step-10 `.h5` against the
        reference's step-10 `.npz`, at its defaults.
   - **dynamic-remesh restart:**
     1. Configuration (a), 4 ranks HIP, 70 steps with a checkpoint at step 50.
     2. A restart from step 50 for `--steps 20`.
     3. Print each run's final progress line.
   - **failure direction:** `--restart-from /nonexistent.h5` exits non-zero,
     with the path in its error message.
7. README: delete the Known Issue and the stale sentence. Document the
   `--restart-from` semantics: a local `--steps` budget; the AMR indicators
   re-base; the run is not trajectory-identical to an uninterrupted one; and
   the master-series behaviour chosen in step 4.

**Exit criterion:**
- the unit tier passes at 1 and 4 ranks through
  `scripts/tuolumne/unit_tests.flux`, including `Beatnik_Test_Restart`, whose
  failure-direction checks each print their own PASS line;
- the T5b job shows both restarted Beatnik step-10 files matching the
  reference restart under `compare_output.py` defaults (`RESULT: match`);
- the dynamic-remesh restart reaches step 70 with `volRel` ≤ `1e-12`, and its
  `F` is within 10 % of the uninterrupted run's step-70 `F`;
- the nonexistent-path launch exits non-zero naming the path;
- the regression gate (`scripts/tuolumne/run_regression_minset.flux`) passes,
  since `Solver::setup` is on every member's path.

### V1j — pilot run to `t_end`, across one restart, submitted — **NOT STARTED**

**Depends on:** D1, T5b.
**Fill in:** `scripts/tuolumne/large_rollup_segment.flux` (new; also used by
R1j), and the stage directory.
**Submits:** `scripts/tuolumne/large_rollup_segment.flux`, run on D1's pilot
rank count in `pbatch`, from stage
`/p/lustre5/stewartj/beatnik/large_rollup/V1/stage`, writing to
`/p/lustre5/stewartj/beatnik/large_rollup/V1/out/`. Expected run time:
- D1's pilot estimate (P1's s/step scaled to the pilot size, times its step
  count to `t_end`) plus 10 %;
- `-t` 1.5× that, capped at `pbatch`'s maximum. If it does not fit one job,
  the segment script is submitted as a chain (below).

**Removes:** nothing.
**Do:**

1. Pull the prod env's beatnik source clone to the T5b commit (or later) and
   `spack install` the prod env, with **no production job live** (R5). Copy
   the prod view's `adaptive_mesh_bubble` into the stage directory's `bin/`
   and write `stage/COMMIT`.
2. The segment script takes the arguments file, an output directory and an
   optional restart path:
   - it cold starts, or `--restart-from <out>/checkpoint_latest.h5` when one
     exists;
   - it runs to `--t-end`, or to the walltime less a 10-minute margin, by
     capping `flux run -t`;
   - it samples RSS on the first node every 60 s into `<out>/rss.log`;
   - it timestamps rank 0's progress lines.
3. **The pilot always crosses at least one restart.** Segment 1 runs with
   `--t-end` set to half of `t_end`. Segment 2 is submitted with
   `flux batch --dependency=afterok:<segment-1 id>` and restarts from the
   latest checkpoint to `t_end`. Further segments follow the same pattern if
   needed.
4. Append the job IDs to the log's `**Jobs:**` line.

**Exit criterion:** every segment's job is accepted by the scheduler and its
ID is on the log's `**Jobs:**` line. The stage directory holds `bin/`, the
arguments file and `COMMIT`. The prod env's installed beatnik spec has the
commit that `stage/COMMIT` names.

### V1 — read the pilot — **NOT STARTED**

**Depends on:** V1j.
**Fill in:** the log's `## V1`; this document's Current state, if the pilot
changes a fact stated there.
**Removes:** nothing.
**Do:** record the following from the progress lines and `rss.log`:
- `t`, `V`, `F`, `minQ`, `minEdge` and `dt`, every 0.1 time units;
- s/step against `V`;
- RSS per rank against step;
- the restart point (the last pre-restart step and `t`, and the first
  post-restart progress line).

If the pilot stopped, record its stop line, the last three progress lines,
and which of R1–R3 the measurements point to, by the distinctions those risks
give. Then return to D1. This task does not change parameters.

**Exit criterion:**
- **Success:** the last segment's final progress line has `t ≥ t_end − dt`,
  with no `nonfinite` stop line in any segment.
- At every progress line: `|volRel| ≤ 1e-12`, and `minQ` at or above D1's
  pilot floor.
- The post-restart segment's first progress line has `step` and `t` greater
  than the restart checkpoint's.
- **Failure direction:** a pilot that stopped early, or broke the volume or
  quality bound, is recorded as a failed V1 with the diagnosis above, and R1j
  does not start.

### R1j — production run, submitted — **NOT STARTED**

**Depends on:** V1.
**Fill in:** the stage directory; no new script (it reuses
`large_rollup_segment.flux`).
**Submits:** `scripts/tuolumne/large_rollup_segment.flux`, on D1's production
nodes in `pbatch` at D1's segment walltime, as a dependency chain of segments.
Stage `/p/lustre5/stewartj/beatnik/large_rollup/R1/stage`, output
`/p/lustre5/stewartj/beatnik/large_rollup/R1/out/`. Expected run time: D1's
production estimate, refined by V1's measured s/step against `V`. Each
segment's `-t` is D1's segment walltime.
**Removes:** nothing.
**Do:**
1. Re-check that the prod env is at the commit V1 ran. Re-stage only if the
   commit changed, and never `spack install` the prod env while a V1 or R1
   segment is live.
2. Submit the first segment. Submit enough dependent segments
   (`--dependency=afterok`) to cover the estimate plus 25 %.
3. Record every job ID.

**Exit criterion:** every segment is accepted, with its ID on the log's
`**Jobs:**` line, and the stage's `COMMIT` matches V1's.

### R1 — read the production run — **NOT STARTED**

**Depends on:** R1j.
**Fill in:** the log's `## R1`; this document's Status and Current state;
README "Known Issues" if the run exposes a defect.
**Removes:** nothing.
**Do:**
- Record V1's quantities at production size.
- Record the first progress line with `V ≥ 200000` and the checkpoint that
  holds it.
- Record the final `t` and the stop reason, if any.
- Record the total node-hours.

If a segment failed for a reason other than a nonfinite state (walltime
without a checkpoint, a node failure), resubmit the chain from the latest
checkpoint. The same failure twice stops the task.

**Exit criterion:**
- **Success:** a checkpoint whose `/vertices` count is at least 200,000; the
  run reached `t_end`, or stopped on a `nonfinite` state with the cause
  diagnosed per R1–R3; and `|volRel| ≤ 1e-12` at every progress line.
- **Failure direction:** a run that never reached 200,000 vertices before
  stopping is a failed R1. It is recorded with its maximum `V` and stop cause,
  and goes back to D1.

## Known risks

**R1 — the fine start is unstable at its starting dt.** The adaptive dt is
relative to the starting mesh, so a level-7 start begins at the same `dt` as a
level-2 one. *Presents as:* a nonfinite stop within the first tens of steps at
level 7 but not at level 6, while `minQ` stays above about 0.3. *Distinguished
from R2 by:* `minQ` and `minEdge` at the last progress line both stay near
their starting values. P1's `L7_np4_dt1e-3` arm is the direct test. *Do:*
D1 lowers `--dt`; do not raise `--min-dt` to compensate.

**R2 — mesh quality collapses late.** The pre-T4d sweeps stopped with `minQ`
0.03–0.11 and dt at or near its floor (Current state). The T4d passes
(collapse, quality flips, guarded smoothing, cleanup) exist to prevent this,
but none has run past `t` 1.33 or above about 2,000 vertices. *Presents as:*
`minQ` falling steadily over many progress lines before a nonfinite stop,
with dt at `--min-dt`. *Do:* record the trend. Remedies (`--remesh-min-quality`,
smoothing iterations, flip gain) are D1's to choose, not V1's or R1's.

**R3 — the sheet approaches itself, and nothing measures it.** The
nonlocal-distance diagnostics are not computed (Current state), and proximity
refinement and surgery are T4e, out of scope. A rising bubble's bottom jet
eventually reaches its top. *Presents as:* a nonfinite stop with `minQ`
healthy, and `minEdge` and dt collapsing over the last few progress lines.
*Distinguished from R2 by:* `minQ` staying well above the floor. Confirm by
measuring the closest non-adjacent face pair in the last finite checkpoint
offline. *Do:* this is the run's physical end in this port. Record `t`; do
not count it as a defect.

**R4 — the direct sum does not scale with ranks at this size.** The level-6
estimate was no faster at np4 than at np1 (Current state). *Presents as:*
P1's `L7_np4` s/step not below `L7_np1`'s. *Do:* D1 sizes the run on the
measured scaling, not on `N^2/P`. The FMM is not a drop-in fallback (Approach).

**R5 — the prod binary is replaced under a live job.** *Presents as:* a
segment dying with rc 135 (SIGBUS) mid-run. *Do:* never `spack install` the
prod env while a V1 or R1 segment is live. Restart the chain from the latest
checkpoint.

**R6 — memory grows with steps rather than with V.** README's RSS Known Issue,
over a run far longer than the 600 steps it was measured on. *Presents as:*
`rss.log` rising while `V` is flat. *Distinguished from mesh growth by:* RSS
divided by `V` rising. *Do:* record it in V1. If the projected RSS at R1's
length exceeds a rank's memory, segment walltime bounds it, because each
restart starts a fresh process.

**R7 — restart and the checkpoint series.** A restart into the same
`--checkpoint-dir` could leave the master `<prefix>.xmf` and `.xmfindex`
disagreeing, or `checkpoint_latest` pointing at a pre-restart frame.
*Presents as:* ParaView showing only post-restart frames, or a segment
restarting from an earlier time than the previous one reached. *Do:* T5b
resolves it; V1 checks that each segment's first progress line is later than
the previous segment's last checkpoint.

**R8 — `h_min` binds before 200,000 vertices.** At level 7 the starting edge
`2.05e-3` is close to the default `h_min` 0.0015, so refinement has little
headroom, and the sizing field clamps where the roll-up needs finer edges.
*Presents as:* `minEdge` pinned at `h_min` for many progress lines while
`max|S|` grows. *Do:* D1 sets `h_min` with this in view. A pinned `minEdge`
is not a defect.
