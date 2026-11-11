# Large roll-up — the default rising bubble past 200,000 vertices

**Status:** IN PROGRESS

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
V1j, V1k, V1c and V1 run the pilot: the parameter set at reduced resolution,
to `t_end`, across chained restarts. That shows the configuration survives before
the expensive run is spent. R1j/R1 is the production run.

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
  `CheckpointIO::read` restore a run (T5b, Current state). T5b keeps its ID from
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

D1 sets these values with the user; each source names the log section that
measured it, or "user, `## D1 — re-cost after V1`" (or the earlier "user,
`## D1`") for a pure choice.

**Cost.** Production is the V1 pilot's configuration with every length scaled
by $s = 0.85$ ($h_{max} = 0.85 \cdot 4.1\times10^{-3}$; `h_min`, sagitta and
eps likewise; `--dt` and `--min-dt` by $s^2$). The projection is arithmetic
from V1's logs, not a run (log `## D1 — re-cost after V1`, script
`/p/lustre5/stewartj/beatnik/large_rollup/D1/d1_recost.py`). Remeshing steps
cost $35\,(V/136{,}952)^{2.37}$ s/step on np4 (log `## V1`, Cost); the pilot's
trajectory, 4,604 steps to $t = 2.5$ (3,566 measured to $t = 2.2086$, the
rest at seg9's dt and `V` rate), is scaled to $1/s^2$ steps and $1/s^2$
vertices at each time. At $s = 1$ the model gives 34,585 s of compute to step
3566, against V1's measured 32,166 s. At $s = 0.85$: about 6,400 steps, `V`
crossing 200,000 at $t \approx 2.29$ and reaching about 235,000 at 2.5, 125
s/step at the end, **67.9 h of wall on np4, 67.9 node-h, 84.9 with the ×1.25
chain margin**. The pilot continuation (V1c) is about 13.1 h of compute, 4
segments, about 16 node-h. The budget is about 105 node-h for both. The fit
is extrapolated 1.7× in `V` past the largest measured point; V1c measures
s/step from 137,000 to about 170,000 and so checks the first part of it.

| Setting | Option | Value | Source |
| --- | --- | --- | --- |
| Physics | `--A 0.3 --g 1.0 --mu 0.002 --viscosity-mode laplace-beltrami --br-approximation direct --source-quadrature vertex --adaptive-dt --dynamic-remesh --isotropic-cleanup` | fixed | configuration (a) |
| Kernel desingularization | `--eps` | `--eps 1.36e-2` (about $3.9\,h_{max}$, the pilot's ratio) | `## V1` (pilot `--eps 1.6e-2` finite to $t = 2.224$), scaled by 0.85; user, `## D1 — re-cost after V1` |
| Starting mesh | `--icosphere-subdivisions` | `--icosphere-subdivisions 6` (40,962 vertices; mean edge `4.1e-3`, under the split threshold $1.35\,h_{max} = 4.71\times10^{-3}$) | `## V1` (the pilot's start); user, `## D1 — re-cost after V1` |
| Sizing | `--remesh-h-max`, `--remesh-h-min`, `--remesh-sagitta-tolerance` | `--remesh-h-max 3.49e-3 --remesh-h-min 2.55e-3 --remesh-sagitta-tolerance 0.0136` (projected `V` 200,000 at $t \approx 2.29$, about 235,000 at 2.5) | the pilot row scaled by 0.85; projection log `## D1 — re-cost after V1`; user, same section |
| Edit caps | `--remesh-max-splits`, `--remesh-max-collapses` | `--remesh-max-splits 1000 --remesh-max-collapses 400` | P1b (about 70 edits/step in the control); user, `## D1`; remeshing cost with these caps measured by the pilot (`## V1`, Cost) |
| Step size | `--dt`, `--min-dt`, `--dt-edge-power` | `--dt 2.31e-3 --min-dt 1.45e-4 --dt-edge-power 2` ($dt \approx 172\,h_{min}^2$; about 6,400 steps to $t = 2.5$; the floor keeps $dt\,\mu/minEdge^2 \le 0.4$ down to `minEdge` $8.5\times10^{-4}$) | `## P1c` (ratio $dt\,\mu/h_{min}^2 = 0.343$ in every arm); the pilot's 4,604 steps (`## V1`) ×$1/0.85^2$; user, `## D1 — re-cost after V1` |
| End time | `--t-end` | `--t-end 2.5` | user, `## D1` |
| Checkpoint cadence | `--checkpoint-every-time`, `--checkpoint-every-steps` | `--checkpoint-every-time 0.025 --checkpoint-every-steps 20` (the two are ORed, `src/Beatnik_Solver.hpp:678-682`; one interval is at most 20 steps, 0.70 h at the projected end's 125 s/step; about 420 files of up to about 93 MB) | `## V1` (Cost; checkpoint 16.2 MB at 40,962 vertices); user, `## D1 — re-cost after V1` |
| Ranks / nodes, segment walltime | — | np4 on 1 node; segments of `-t 240m` in `pbatch` with a 230-minute launch cap, about 18 segments (22 with the ×1.25 margin). Two intervals (1.4 h) fit the cap's 1.92 h half; peak per-rank RSS about 31.6–43.1 GB, under the 64 GB bound | `## P1c` (np8/np4 s/step 0.97, so np8 doubles node-h); `## V1` Memory (7–10 GB per rank per hour; 128 GB per MI300A APU); user, `## D1 — re-cost after V1` |
| Pilot resolution | starting level and `h_max` for V1j, V1k, V1c | `--icosphere-subdivisions 6 --remesh-h-max 4.1e-3 --remesh-h-min 3.0e-3 --remesh-sagitta-tolerance 0.016 --eps 1.6e-2 --dt 3.2e-3 --min-dt 2e-4 --dt-edge-power 2`, np4 on 1 node. V1j and V1k ran nine `-t 90m` segments to $t = 2.224$; V1c continues from the step-3566 checkpoint with `--checkpoint-every-steps 20` added, in `-t 240m` segments (about 1,040 steps, 13.1 h, 4 segments) | `## P1c` (`L6_np4_eps1.6e-2` stable at ratio 0.343); `## V1` (Cost); user, `## D1` and `## D1 — re-cost after V1` |
| Pilot quality floor | minimum `minQ` V1 asserts | `minQ` ≥ 0.12 | user, `## D1` (pre-T4d long runs ended at 0.033–0.11; 0.18 is the repair trigger) |

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
| Restart failures the code can detect throw `std::runtime_error` naming the path and the reason, **before** `Tessera::readMesh` is called. A silent cold start is never a fallback. | A structural mismatch inside `readMesh` is an uncatchable `MPI_Abort` ([src/Beatnik_IOInterface.hpp:936-942](../src/Beatnik_IOInterface.hpp#L936-L942)), and a silent cold start produces a plausible run of the wrong thing ([src/Beatnik_Restart.hpp:185-190](../src/Beatnik_Restart.hpp#L185-L190)). | The unit test's failure directions (T5b). |
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

- **Restart works (T5b).** `RestartReader::load`
  ([src/Beatnik_Restart.hpp:204-237](../src/Beatnik_Restart.hpp#L204-L237))
  refuses a checkpoint whose `state_model` differs from the run's, then fills
  `RestartState` from `CheckpointIO::read`
  ([src/Beatnik_IOInterface.hpp:944-955](../src/Beatnik_IOInterface.hpp#L944-L955)).
  `read` makes the cheap checks in `readHeader` (`:977`, rank 0, broadcast),
  continues the checkpoint series in `resumeSeries` (`:1146`), then calls
  `Tessera::readMesh` and widens the halo to depth 2; vertex, edge and face
  gids survive the read. `Solver::setup` dispatches to `load` under
  `--restart-from`
  ([src/Beatnik_Solver.hpp:241-313](../src/Beatnik_Solver.hpp#L241-L313)),
  prints the reference's `restarted from <path> at step=… t=… with V=… F=…`
  line, and seeds the material position only when the file lacks it. A
  checkpoint carries the mesh, the three vertex fields and the `/beatnik`
  scalars `state_model`, `time`, `step`, `initial_volume` and
  `initial_min_edge` (schema at `src/Beatnik_IOInterface.hpp:70-90`). A
  structural mismatch between the file and the build is still an `MPI_Abort`
  inside `readMesh`.

  **The series across a restart (R7).** A restart into the directory holding
  `<prefix>.xmfindex` reopens the series, so the master `<prefix>.xmf` lists
  the pre- and post-restart frames; a restart point earlier than the series'
  last frame throws before anything is read, naming the index and the remedy
  (a new `--checkpoint-dir`). Chaining from `checkpoint_latest.h5` into the
  same directory is always accepted (README "Restarting a run").
- **`--restart-from`, `--t-end`, `--checkpoint-every-time` and
  `--icosphere-subdivisions` are parsed** by the example
  (`examples/02_adaptive_mesh_bubble/InputFile.hpp:303,358-360,558-562`). A
  restart's `--steps` is a local budget (`src/Beatnik_Restart.hpp:92-96`).
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

  **The V1 pilot** (with the T4d passes; Pilot resolution row, np4 HIP, nine
  chained segments, log `## V1`) runs to $t = 2.224$ (step 3624) with no
  nonfinite stop, and ends on its ninth segment's cap, not at `t_end`. `V`
  grows 40,962 → 136,952; $\lvert volRel \rvert \le 4.2\times10^{-15}$ and
  `minQ` ≥ 0.30 at all 218 progress lines. `minEdge` holds at
  $0.45\,h_{max} = 1.845\times10^{-3}$ from $t \approx 0.5$ to 2.0 with dt
  about `5.85e-4`, then falls to `9.2e-4`–`1.3e-3` with dt `2.0e-4`–`3.6e-4`.
  Cost rises from 0.12 s/step at 40,962 vertices to 35 s/step at 137,000
  (about $V^{2.37}$), 3,566 checkpointed steps in 11.9 node-h.
- **Direct-sum cost at fixed connectivity**: the direct estimate is 0.108
  s/step at level 6 HIP np1 and 0.125 s/step at np4
  (`tasks/canopy_profiling-progress-log.md`, table "(d)"). It is an estimate
  from the `[acc]` evaluations, not a timed run. Multi-rank scaling of the
  direct sum at more than 41k vertices is unmeasured.
- **Per-rank RSS grows slowly** in 4-rank dynamic-remesh runs (README Known
  Issues, `README.md:905-920`): 6–22 MB per rank over 600 steps at about 1040
  faces. In the V1 pilot it does not plateau within an 80-minute segment:
  per-rank RSS rises from about 1.3 GB to 5–13 GB in each segment, RSS per
  vertex rises 2.2–4.2× while `V` grows 4–57 %, and a restart resets it (log
  `## V1`, Memory).

## Progress log

Session record: [large_rollup-progress-log.md](large_rollup-progress-log.md).
Consult it before implementing a task, changing a signature, or reopening a
question this document treats as settled.

> Each task is reviewed with `review-task` before any task that depends on it
> starts. A task is ready to build on when it is marked **DONE** and the log has
> a `## <ID> review` section whose verdict is `Passed`. The document's status
> becomes COMPLETE only when a `review-task` sweep of the whole feature passes.

## Task sequence

### P1 — measure cost, memory and early stability at levels 6 and 7 — **DONE**

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

**Exit criterion:** the log's `## P1` section has a table with one row per
launch of **Do** 2, holding every quantity **Do** 5 lists, for all six
launches. Each launch either reached `step=60`, or has its stop line or
its cap `rc` recorded with the last progress line before it. The control
direction holds: `L7_np4_hmax_default`'s `F` at step 60 is **below** its
step-1 `F`, while `L7_np4`'s step-60 `F` is within 5 % of its step-1 `F` or
above it. If the control does not coarsen, `h_max` is not what holds the
resolution; record that and do not proceed to D1 until the log explains it.

**Met.** Two jobs (log `## P1`). In `f3j99ZVqFY9d`, the six **Do** 2 launches
were recorded: `L6_np1` and `L6_np4` reached step 60 (0.120 / 0.118 s/step,
no remeshing). `L7_np1` was killed by its cap (rc 142) after step 6.
`L7_np4`, `L7_np4_dt1e-3` and `L7_np4_hmax_default` stopped `nonfinite
mesh-RHS state` at steps 12, 33 and 12. Each has its last progress line in the
log. Every level-7 arm blew up with `minQ` 0.974, so the control direction
could not be read at step 60. By the user's decision, the follow-up job
`f3j9KFoF6Yto` (`scripts/tuolumne/large_rollup_probe_b.flux`) repeated the
pair at `--dt 4e-4`. There, `L7_np4_hmax_default_dt4e-4`'s F went 327678 →
319952 (below), while `L7_np4_dt4e-4`'s F stayed 327680 → 327680 (within
5 %): the control direction holds. P1b also showed the level-7 blow-up is the
explicit viscous limit $dt\,\mu/h_{min}^2 \approx 0.5$: `--mu 0` at
`--dt 1e-3` and `--dt-edge-power 2` at `--dt 8e-4` both reached step 60.
Level-7 np4 direct cost is 0.44 s/step (4.5 s/step while remeshing heavily).
Peak RSS is ≤ 1.7 GB/rank.

### P1c — stability and cost at an edge-tied eps, np4 and np8 — **DONE**

**Depends on:** P1.
**Fill in:** `scripts/tuolumne/large_rollup_probe_c.flux` (new). No source change.
**Removes:** nothing.
**Reference:** `scripts/tuolumne/large_rollup_probe_b.flux` for the base
arguments, launch line, caps, RSS wrapper, line timestamps and summary lines;
log `## P1` (P1b subsection) for the arms being compared against, and `## D1`
for the sizing, step-size and edit-cap values these launches carry.
**Do:**

1. Base arguments, every launch: configuration (a)'s physics (Production
   configuration, first row), `--steps 60`, `--progress-time-interval 0`,
   `--dt-edge-power 2`, `--remesh-max-splits 1000 --remesh-max-collapses 400`,
   `--checkpoint-dir <out>/<tag> --checkpoint-every-steps 60`.
2. Launches, each `flux run -t 8m`, with `--gpus-per-task=1
   --cores-per-task=24` and the node count derived from the ranks (4 ranks
   per node):

   | Tag | Level | `--eps` | `--remesh-h-max` | `--remesh-h-min` | `--remesh-sagitta-tolerance` | `--dt` | `--min-dt` | Ranks |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | `L7_np4_eps0.025` | 7 | 0.025 | `2.05e-3` | `1.5e-3` | 0.004 | `8e-4` | `5e-5` | 4 |
   | `L7_np4_eps8e-3` | 7 | `8e-3` | `2.05e-3` | `1.5e-3` | 0.004 | `8e-4` | `5e-5` | 4 |
   | `L7_np8_eps8e-3` | 7 | `8e-3` | `2.05e-3` | `1.5e-3` | 0.004 | `8e-4` | `5e-5` | 8 |
   | `L6_np4_eps1.6e-2` | 6 | `1.6e-2` | `4.1e-3` | `3.0e-3` | 0.016 | `3.2e-3` | `2e-4` | 4 |

   `8e-3` is about four level-7 mean edges (`2.05e-3`). The first row is the
   baseline: P1b's stable `L7_np4_dt8e-4_pow2` arm with the raised caps and
   the floor `5e-5`, so a change between rows 1 and 2 is eps's alone. The
   last row is the pilot candidate: every length doubled from level 7 and dt
   scaled by 4, with eps at the same ratio to its edge.
3. RSS and timestamps as in P1b: `/usr/bin/time -f "rss_kb=%M"` per rank,
   each stdout line prefixed with `$(date +%s.%N)`, each launch's log in
   `<out>/<tag>/launch.log`, then the summary line.
4. Job: 2 nodes, `-t 45m`, `pdebug`, dev env, output under
   `/p/lustre5/stewartj/beatnik/large_rollup/P1c/`. Expected run time is
   about 10 minutes, an estimate from P1b (7.0 min for four level-7 arms at
   0.44 s/step, 4.5 s/step when remeshing heavily); the four caps bound the
   worst case at 32 min. The 1-node launches use one node of the allocation.
5. Record in the log's `## P1c`, per launch, the quantities P1's **Do** 5
   lists, plus $dt\,\mu/h_{min}^2$ at the last line (on the measured
   `minEdge`) and `max|S|` at the last line. From them:
   - the stable dt at eps `8e-3` against eps 0.025, and the steps to
     $t = 2.5$ each implies, by P1b's method;
   - the np8/np4 s/step ratio at eps `8e-3`, separately for intervals with
     and without remesh events if both occur;
   - the production node-hours at np8 (2 nodes), bracketed as in `## D1`,
     against 35 node-h.

**Exit criterion:** the log's `## P1c` section has one row per launch of
**Do** 2 holding every quantity **Do** 5 lists, for all four launches. Each
launch either reached `step=60` or has its stop line or cap `rc` recorded
with the last progress line before it. The baseline `L7_np4_eps0.025` reaches
`step=60` with $dt\,\mu/h_{min}^2 \le 0.41$ at its last line. If it does not,
the raised caps or the lowered floor changed what P1b measured; record that,
and D1 does not resume until the log explains it. A nonfinite stop in an eps
`8e-3` arm while the baseline reaches step 60 is a measured result (eps
`8e-3` is unstable at this dt rule), recorded with its last progress lines,
not a task failure.

**Met.** Job `f3j9ihiaeo7u` (log `## P1c`), exit 0. All four launches rc 0
and reached step 60 with no stop line: `L7_np4_eps0.025`, `L7_np4_eps8e-3` and
`L7_np8_eps8e-3` at $t = 0.0234$, `L6_np4_eps1.6e-2` at $t = 0.0937$. The
baseline's $dt\,\mu/h_{min}^2$ at step 60 is 0.343 (dt 3.13e-4, `minEdge`
0.00135), within 0.41 and matching P1b's `L7_np4_dt8e-4_pow2`. Every arm ended
at the same ratio, 0.343. Eps `8e-3` gives the same dt sequence as 0.025, so
the steps to $t = 2.5$ are about 8,000–17,100 for both. np8/np4 s/step is
0.428 / 0.440 = 0.97: no rank speedup over 4 → 8. No launch remeshed, so the
raised caps and the remesh-interval scaling are unmeasured.

### D1 — choose the production parameter set — **DONE**

D1 sizes production and the pilot from the V1 pilot's measurements (log
`## V1`): its s/step against `V`, its step count, its vertex growth and its
per-rank memory growth. The alternatives and the numbers behind each value are
in the log's `## D1` and `## D1 — re-cost after V1`.

**Met.** Every row of the Production configuration table holds a value and a
source. The cost paragraph gives production at $s = 0.85$ as about 6,400
steps, 67.9 h of wall on np4, 67.9 node-h (84.9 with the ×1.25 margin),
computed from V1's fit $35\,(V/136{,}952)^{2.37}$ s/step, its 4,604-step
trajectory (3,566 measured steps to $t = 2.2086$, then seg9's dt `2.81e-4` and
33 vertices per step) and its growth to 136,952 vertices; the model reproduces
the pilot's own compute within 7.5 % (34,585 s against 32,166 s). The budget
is about 105 node-h: production plus V1c's 16. Segment sizing at `-t 240m`
(launch cap 230 min, 3.83 h): one `--checkpoint-every-steps 20` interval is
0.70 h at the projected end-of-run 125 s/step (0.32 h for the pilot at 58), so
two fit in 1.92 h; peak per-rank RSS at 7–10 GB/h is 31.6–43.1 GB at
$s = 0.85$ (30.3–41.8 for the pilot), under half of one MI300A APU's 128 GB.
The log's `## D1 — re-cost after V1` records the options, the user's
answers and the arithmetic. The next task is V1c (continue the pilot), then
V1, then R1j; V1 depends on V1c, R1j on V1.

**Depends on:** P1, P1c. D1 also reads the log's `## V1`; V1 itself is not
DONE and does not gate D1.
**Fill in:** this document's [Production configuration](#production-configuration)
table and its cost paragraph; the task entries D1's choices change (the pilot
tasks and R1j, below); a new section `## D1 — re-cost after V1` appended at
the end of the log (a top-level section, not a subsection of the existing
`## D1`, so that the reopened D1 gets its own review). No code or script.
**Removes:** every table value and cost figure D1 replaces, with the sentences
that cite it, including the level-7 row's node-hour figures wherever they
appear in this document.
**Reference:** log `## V1` (Cost, Memory, Trajectory); log `## D1` (the
current values and their alternatives); `## P1c` (np8/np4 s/step 0.97); R1,
R2, R6 and R8.
**Do:** this is a decision session held with the user. The session prepares
options and their measured consequences, and the user chooses; the session
does not fill the table alone. Present each of these, with the arithmetic
labelled as arithmetic:

- **Cost against `V`.** The pilot's s/step grows about as $V^{2.37}$ for
  $V \ge 65{,}000$ (35 s/step at 136,952, np4 HIP), and about 97 % of it is
  outside the direct sum. Project production node-hours for each candidate
  resolution from that fit, from the pilot's step count (3,566 steps to
  $t = 2.2086$, about 4,600 to 2.5), and from the vertex growth the pilot
  showed (3.3× its start by $t = 2.22$).
- **What fits the budget.** For each option, the projected node-hours and
  what it gives up: a smaller vertex target; a coarser `--remesh-h-max` or
  `--remesh-sagitta-tolerance` that bounds `V` growth (R8); a larger budget;
  or a profiling task, ahead of R1j, that finds which phase holds the 97 %
  (a new task, added to the sequence only if the user chooses it).
- **Segment walltime against checkpoint cadence.** One
  `--checkpoint-every-time` interval, in wall time (steps per interval ×
  projected s/step at the segment's end), must fit well inside the launch cap.
  Otherwise a segment writes no checkpoint and the chain repeats without
  progress, because `scripts/tuolumne/large_rollup_segment.flux` exits 0 on a
  cap kill whenever any `checkpoint_latest.h5` exists. Each cap also discards
  the work after the last checkpoint (`## V1j` Findings 1); in the pilot that
  was about 6,200 s of 42,600, 15 % (`## V1`, Cost).
- **Segment walltime against memory (R6).** Per-rank RSS grows 7–10 GB per
  hour of process life and resets at a restart, so the segment walltime bounds
  peak RSS. Choose `-t` so that the rate × walltime stays well under a rank's
  memory.
- **The pilot.** Whether a pilot runs again: continue the existing one from
  `V1/out/checkpoint_t00002p208576_step0003566.h5` with segments sized as
  above, re-run it at a re-sized resolution, or go to R1j on V1's evidence to
  $t = 2.224$. Name the tasks that follow.

**Exit criterion:** every row of the Production configuration table holds a
value and a source; the cost paragraph above the table gives projected
production node-hours computed from V1's measured s/step against `V` and step
count; the chosen segment walltime fits at least two checkpoint intervals at
the projected end-of-segment s/step and keeps the projected per-rank RSS under
half a rank's memory; the log's `## D1 — re-cost after V1` section records the
alternatives, the numbers behind each choice, and the next task; and the task
sequence names that task with its dependencies. Failure direction: if no
option the user accepts fits the node-hours the user accepts, D1 records that
and stops. A smaller target, a profiling task or an FMM task is then a new
design decision, not a change made here.

### T5b — restart from a Beatnik checkpoint — **DONE**

**Met.** Job `f3jB613LsKju` (`scripts/tuolumne/large_rollup_t5b.flux`, dev
env at `d94ff8e`): the frozen-mesh step-5 restarts at 4 ranks and at 1 rank
both reach step 10 and give `RESULT: match` against the Python reference
restarted from the same file; the dynamic-remesh restart from step 50 reaches
step 70 with `volRel` 0 and F 1032 against the uninterrupted run's 1032; a
restart into the series' own directory leaves master and index both listing
steps 0, 5, 10; `--restart-from /nonexistent.h5` exits 1 naming the path. The
unit tier passes 9/9 at 1 rank (`f3jB5zmpxqgb`); at `BEATNIK_UNIT_RANKS=4`
(`f3jB6TgWinzj`) every member passes except `Beatnik_Test_T2bOperators`'
deliberate `comm_size == 1`, and `Beatnik_Test_Restart` passes on every rank
with its three failure-direction PASS lines at both rank counts. The
regression gate passes, 60 launches (`f3jB6qVdovmd`). Resolution of Do 4:
the series is reopened from `<prefix>.xmfindex`; a restart point earlier than
its last frame throws. Log `## T5b`.

**Depends on:** none. It may run in parallel with P1 and D1.
**Fill in:**
- `CheckpointIO::read`
  ([src/Beatnik_IOInterface.hpp:944-955](../src/Beatnik_IOInterface.hpp#L944-L955))
  and the restart side of the master-series handling in the same class
  (`readHeader` at `:977`, `resumeSeries` at `:1146`);
- `RestartReader::load`
  ([src/Beatnik_Restart.hpp:204-237](../src/Beatnik_Restart.hpp#L204-L237));
- the comment on the restart branch of `Solver::setup`
  (`src/Beatnik_Solver.hpp:245-252`);
- `tests/unit_tests/Beatnik_Test_Restart.cpp` (new), and its entry in
  `tests/unit_tests/CMakeLists.txt`;
- `scripts/tuolumne/large_rollup_t5b.flux` (new);
- `README.md`.

**Removes** (all removed in `0774838`; line numbers are those before it):
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
- `../tessera/src/Tessera_HDF5Reader.hpp` for `readMesh` and
  `../tessera/src/Tessera_XdmfSeries.hpp` for `MeshSeries`. **This task is the
  first to open them.**

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
- the unit tier passes through `scripts/tuolumne/unit_tests.flux` at 1 rank,
  and at `BEATNIK_UNIT_RANKS=4`, read member by member, every member passes
  except `Beatnik_Test_T2bOperators`, whose `comm_size == 1` check fails by
  design at more than one rank
  (`tests/unit_tests/Beatnik_Test_T2bOperators.cpp:183-199`). At both rank
  counts `Beatnik_Test_Restart` passes, and its failure-direction checks each
  print their own PASS line;
- the T5b job shows both restarted Beatnik step-10 files matching the
  reference restart under `compare_output.py` defaults (`RESULT: match`);
- the dynamic-remesh restart reaches step 70 with `volRel` ≤ `1e-12`, and its
  `F` is within 10 % of the uninterrupted run's step-70 `F`;
- the nonexistent-path launch exits non-zero naming the path;
- the regression gate (`scripts/tuolumne/run_regression_minset.flux`) passes,
  since `Solver::setup` is on every member's path.

### V1j — pilot run to `t_end`, across one restart, submitted — **DONE**

**Met.** The user pulled the production clones and ran the production
`spack install` by hand. This session checked the result: beatnik
`4b33843` (contains `d94ff8e`), tessera `308885b` and canopy `9cb0682` are
each an ancestor-check pass. The production view's `adaptive_mesh_bubble`
(`beatnik-develop-jrgarvack…`, dated 2026-11-09 18:47, after beatnik HEAD's
18:25) embeds `Beatnik_GIT_COMMIT_HASH` `4b33843`, the production beatnik
HEAD that `stage/COMMIT` names. `git diff 4b33843 dbd8e0f -- src examples
tests CMakeLists.txt` is empty. `V1/stage/` holds `bin/adaptive_mesh_bubble`
(`cmp`-identical to the view), `args.txt`, `COMMIT`, `segment1.flux` and
`segment2.flux`. The scheduler accepted both segments: `f3jBPQzmiXhq` (seg1,
running at the check) and `f3jBPi5KJgJb` (seg2, `D`,
`depends:after-success=f3jBPQzmiXhq`). Their IDs are on the log's
`## V1j` **Jobs:** line. The segment script passes `bash -n` and carries the
SPDX header and `set -u`. The run's output is for V1 to read.

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

### V1k — pilot continuation to `t_end`, submitted — **DONE**

V1j's two segments end short of `t_end`. Seg1 `f3jBPQzmiXhq` reached
`--t-end 1.25` (`seg1 end=t_end`). Seg2 `f3jBPi5KJgJb` restarted at step 1764,
$t = 1.25$. Its 80-minute launch cap killed it at step 2326, $t = 1.577$
(`seg2 rc=142`, `end=cap checkpoint=checkpoint_t00001p577250_step0002326.h5`,
job log `V1/large_rollup_segment.f3jBPi5KJgJb.log`). Its last progress line
is step 2322, $t = 1.575$, dt 5.83e-4, V 81,028, `minQ` 0.67, `volRel`
-4.4e-16 (`V1/out/seg2.launch.log`).

**Depends on:** V1j.
**Fill in:** the stage directory only: new files
`V1/stage/segment3.flux` … `segment9.flux`. No change to
`scripts/tuolumne/large_rollup_segment.flux`, `args.txt`, `bin/` or `COMMIT`.
**Submits:** `scripts/tuolumne/large_rollup_segment.flux` as staged copies, as
a chain of seven segments seg3–seg9. Each is 1 node, np4, `-t 90m` (launch cap
`flux run -t 80m`), in `pbatch`, from stage
`/p/lustre5/stewartj/beatnik/large_rollup/V1/stage`, writing to
`/p/lustre5/stewartj/beatnik/large_rollup/V1/out/` (job logs in `V1/`).
Expected run time:
- about 1,580 steps remain from $t = 1.577$ to $2.5$ at dt about 5.85e-4;
- seg2 ended at about 10.6 s/step at V 81k, rising with V (seg2's launch
  log; seg1 ran 0.13 s/step at 41k with no remeshing, and 5.0 s/step at 64k);
- at 10–20 s/step that is 4.4–8.8 h, i.e. 4–7 segments at the 80-minute cap,
  10.5 node-h at most.
**Removes:** nothing.
**Do:**

1. Do not rebuild or re-stage. The chain continues the V1j run with the
   staged binary, `args.txt` and `COMMIT` unchanged, so every segment runs
   the same code.
2. Write `segment3.flux` as a byte copy of `segment1.flux`. Seg2 has
   finished, so seg3 has no dependency. Write `segment<k>.flux` for
   $k = 4…9$ as the same copy plus `# flux: --dependency=afterok:<seg k−1
   id>`, after seg $k-1$ is accepted, as V1j did for seg2.
3. Submit every segment with the arguments
   `V1/stage/args.txt V1/out 2.5 seg<k> V1/out/checkpoint_latest.h5` (full
   paths). Each restarts from the latest checkpoint, into the same
   `--checkpoint-dir`, which T5b's `resumeSeries` accepts.
4. A segment that starts after the run has reached `t_end` restarts at
   $t = 2.5$. `Solver::solve` breaks before its first step
   (`src/Beatnik_Solver.hpp:359-361`), so the segment rewrites the
   equal-time checkpoint and exits 0. Surplus segments are therefore harmless,
   and V1 reads past them. A `nonfinite` stop exits non-zero and breaks the
   `afterok` chain, so later segments never run.
5. Append every job ID to the log's `**Jobs:**` line.

**Exit criterion:** all seven segments are accepted by the scheduler, seg4–seg9
each with an `afterok` dependency on its predecessor. Every ID is on the log's
`**Jobs:**` line. `segment3.flux` is `cmp`-identical to `segment1.flux`, and
each later copy differs from it only by its dependency directive. `args.txt`,
`bin/adaptive_mesh_bubble` and `COMMIT` in the stage are unchanged from V1j.

**Met.** At the 2026-11-10 06:09 check all seven segments were accepted: seg3
`f3jGfa2qqqNX` `RUN`, and seg4 `f3jGg4ENVXpK`, seg5 `f3jGhvGwFoiF`, seg6
`f3jGi9bd7VU3`, seg7 `f3jGiNrJD7yh`, seg8 `f3jGic4He3u9` and seg9
`f3jGiqL31AFH` each `DEPEND` with `after-success=` its predecessor's ID.
`cmp segment1.flux segment3.flux` is silent. `diff segment1.flux segment<k>.flux`
for $k = 4…9$ prints only `8a9 > # flux: --dependency=afterok:<seg k−1 id>`.
`bin/adaptive_mesh_bubble` is `cmp`-identical to the
`beatnik-develop-jrgarvack5rup44kphc4vjxpp7jxzt6q` install that `COMMIT` names.
`args.txt` (18:50:15) and `COMMIT` (18:50:20) keep their 2026-11-09 timestamps,
and `COMMIT` names `4b33843`. Seg3's launch log shows the restart at step 2326,
$t = 1.577$. IDs and logs: progress log `## V1k`.

### V1c — pilot continuation from $t = 2.2086$ to `t_end`, submitted — **DONE**

The nine V1j/V1k segments end on seg9's cap at $t = 2.224$; the newest
checkpoint is `V1/out/checkpoint_t00002p208576_step0003566.h5` (step 3566,
$t = 2.2086$, 136,952 vertices), which `V1/out/checkpoint_latest.h5` names.
At 35–58 s/step one `--checkpoint-every-time 0.025` interval is 1.4–2.0 h, so
the 80-minute cap of `-t 90m` segments cannot reach the next checkpoint (log
`## V1`, "The 90-minute chain cannot finish it"). V1c continues the run in
`-t 240m` segments with a 20-step checkpoint cadence added.

**Depends on:** D1, V1k.
**Fill in:**
- `scripts/tuolumne/large_rollup_segment.flux`: the `-t` directive becomes
  `240m`; the launch cap and the cap-detection threshold are derived from one
  value, the job's remaining time from `flux job timeleft` (present in the
  installed Flux: `Usage: flux-job timeleft [JOBID]`) less a 10-minute
  margin, in place of the literal `flux run -t 80m` (`:154`) and `4740`
  (`:188`). If `flux job timeleft` fails or the cap it gives is not positive,
  the segment prints `[large_rollup] FAIL:` with the reason and exits non-zero
  before launching. The header comment's "How a segment ends" paragraph
  states the derived cap. The exit logic (`:176-196`) is otherwise unchanged,
  and the script stays generic for R1j.
- New stage files only: `V1/stage/args_c.txt` (`args.txt` plus
  `--checkpoint-every-steps 20`) and `V1/stage/segment10.flux` …
  `segment14.flux`. `args.txt`, `bin/`, `COMMIT` and `segment1.flux` …
  `segment9.flux` are unchanged.

**Submits:** `V1/stage/segment10.flux` … `segment14.flux`, staged copies of
the edited `scripts/tuolumne/large_rollup_segment.flux`, each 1 node, np4,
`-t 240m` (launch cap 230 min), in `pbatch`, from stage
`/p/lustre5/stewartj/beatnik/large_rollup/V1/stage`, writing to
`/p/lustre5/stewartj/beatnik/large_rollup/V1/out/` (job logs in `V1/`).
Expected run time: about 1,040 steps and 13.1 h of compute (log
`## D1 — re-cost after V1`, arithmetic from seg9's rates and the
$V^{2.37}$ fit), plus at most one 20-step interval (0.32 h) lost per cap:
four segments; the fifth is margin and, started after `t_end`, exits at once
(V1k **Do** 4). At most 20 node-h.
**Removes:** the literal `80m` launch cap and `4740` threshold in
`scripts/tuolumne/large_rollup_segment.flux`, with the comments that state
them.
**Do:**

1. Do not rebuild or re-stage the binary. The continuation runs the staged
   `bin/adaptive_mesh_bubble` and `COMMIT` that V1j staged, so every segment
   of the pilot runs the same code.
2. Edit the repository script as **Fill in** says, check it with `bash -n`,
   and commit it before staging.
3. Write `segment10.flux` as a byte copy of the committed script, with no
   dependency. Write `segment<k>.flux` for $k = 11…14$ as the same copy plus
   `# flux: --dependency=afterok:<seg k−1 id>`, after seg $k-1$ is accepted.
4. Submit every segment with the arguments `V1/stage/args_c.txt V1/out 2.5
   seg<k> V1/out/checkpoint_latest.h5` (full paths).
5. Append every job ID to the log's `## V1c` **Jobs:** line.

**Exit criterion:** all five segments are accepted by the scheduler,
seg11–seg14 each with an `afterok` dependency on its predecessor, and every ID
is on the log's `## V1c` **Jobs:** line. `segment10.flux` is `cmp`-identical
to the committed `scripts/tuolumne/large_rollup_segment.flux`, and each later
copy differs from it only by its dependency directive. `diff args.txt
args_c.txt` shows only the added `--checkpoint-every-steps 20`. `bin/` and
`COMMIT` are unchanged from V1j (`cmp` against the install `COMMIT` names).
The committed script has no `80m` or `4740` literal (`grep -nE '80m|4740'` is
empty), its cap comes from `flux job timeleft`, and the non-positive-cap path
exits non-zero before `flux run` (by reading).

**Met.** At the 2026-11-11 09:35 check all five segments were accepted: seg10
`f3jVdhyQ8zwy` `SCHED`, and seg11 `f3jVdzNzL25D`, seg12 `f3jVeGSiafEX`, seg13
`f3jVeXfVm1jV` and seg14 `f3jVf89kr2Hd` each `DEPEND` with `after-success=`
its predecessor's ID. The script commit is `57903c4`; `bash -n` passes and
`grep -nE '80m|4740'` on it is empty. Its cap is `flux job timeleft` less
600 s, passed as `flux run -t "${CAP}s"`, with cap detection at the cap less
60 s; a failed `timeleft`, a non-integer or a non-positive cap calls
`cap_fail`, which prints `[large_rollup] FAIL:` and the `rc=1 wall=0.0`
summary and exits 1 before the RSS sampler and `flux run` (read). `cmp
segment10.flux` against the script, both the working copy and
`git show 57903c4:`, is silent; `diff` against the script for seg11–seg14
prints only `8a9 > # flux: --dependency=afterok:<seg k−1 id>`, each the right
predecessor. `diff args.txt args_c.txt` prints only `4a5 >
--checkpoint-every-steps 20`. `bin/adaptive_mesh_bubble` is `cmp`-identical to
the `beatnik-develop-jrgarvack5rup44kphc4vjxpp7jxzt6q` install; `args.txt`,
`COMMIT` and `segment1.flux` keep their 2026-11-09 timestamps, and `COMMIT`
names `4b33843`. IDs and logs: progress log `## V1c`.

### V1 — read the pilot — **NOT STARTED**

V1 reads the pilot through V1c's last segment. The reading of seg1–seg9 is
in log `## V1`: no `nonfinite` stop,
$\lvert volRel \rvert \le 4.22\times10^{-15}$ and `minQ` ≥ 0.300 at all 218
progress lines, every
restart later than its checkpoint, and seg9 ending on its cap at $t = 2.224$.
V1 records the whole pilot in a new top-level log section `## V1 — after V1c`.

**Depends on:** V1c.
**Fill in:** the log's `## V1 — after V1c`; this document's Current state, if
the pilot changes a fact stated there.
**Removes:** nothing.
**Do:** record the following from the progress lines and `rss.log`:
- `t`, `V`, `F`, `minQ`, `minEdge` and `dt`, every 0.1 time units;
- s/step against `V`;
- RSS per rank against step;
- each restart point, seg2 through the last segment that ran (the last
  pre-restart step and `t`, the `restarted from … at step= t=` line, and the
  first post-restart progress line).

Read every segment's job log and `<tag>.launch.log` from V1j, V1k and V1c
(seg1–seg14). A segment that restarted at `t_end` and took no step is
recorded as surplus. `V1/analysis/v1_read_pilot.py` reads seg1–seg9; extend
a copy for seg10–seg14.

Against the production projection (log `## D1 — re-cost after V1`): record
V1c's measured s/step against `V` beside $35\,(V/136{,}952)^{2.37}$, and the
pilot's steps from 3566 to `t_end` beside the projected 1,040. Re-derive the
production node-h at $s = 0.85$ with the measured values in place of the
extrapolated ones. If that figure, with the ×1.25 margin, exceeds the 85
node-h production share of the budget, R1j does not start and the log says
so; re-sizing is D1's.

If the pilot stopped, record its stop line, the last three progress lines,
and which of R1–R3 the measurements point to, by the distinctions those risks
give. Then return to D1. This task does not change parameters.

**Exit criterion:**
- **Success:** the last segment's final progress line has `t ≥ t_end − dt`,
  with no `nonfinite` stop line in any segment.
- At every progress line: `|volRel| ≤ 1e-12`, and `minQ` at or above D1's
  pilot floor.
- Every post-restart segment's first progress line has `step` and `t` greater
  than its restart checkpoint's.
- The re-derived production node-h is recorded.
- **Failure direction:** a pilot that stopped early (a `nonfinite` stop, or
  its last segment ending on its cap short of `t_end`), or broke the volume or
  quality bound, is recorded as a failed V1 with the diagnosis above, and R1j
  does not start.

### R1j — production run, submitted — **NOT STARTED**

**Depends on:** V1.
**Fill in:** the stage directory; no new script (it reuses
`large_rollup_segment.flux` as V1c leaves it).
**Submits:** staged copies of `scripts/tuolumne/large_rollup_segment.flux`,
each 1 node, np4, `-t 240m` (launch cap 230 min) in `pbatch`, as an `afterok`
chain. Stage `/p/lustre5/stewartj/beatnik/large_rollup/R1/stage`, output
`/p/lustre5/stewartj/beatnik/large_rollup/R1/out/`; each copy's `--output`
directive names `R1/`, created before submitting. Expected run time: 67.9 h of
compute, about 18 segments, 22 with the ×1.25 margin (cost paragraph above
the Production configuration table), refined by V1's re-derived estimate.
**Removes:** nothing.
**Do:**
1. Re-check that the prod env is at the commit V1 ran. Re-stage only if the
   commit changed, and never `spack install` the prod env while a V1 or R1
   segment is live. If the commit is unchanged, copy V1's staged
   `bin/adaptive_mesh_bubble` and `COMMIT`.
2. Write `R1/stage/args.txt` from the production rows: Physics, Kernel,
   Starting mesh, Sizing, Edit caps, Step size and Checkpoint cadence, plus
   `--steps 1000000 --progress-time-interval 0.01`, as V1j's `args.txt`. The
   segment script appends `--checkpoint-dir` and `--t-end 2.5`.
3. Submit the first segment cold. Submit enough dependent segments
   (`--dependency=afterok`), each restarting from
   `R1/out/checkpoint_latest.h5`, to cover the estimate plus 25 %.
4. Record every job ID.

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
level-2 one. The instability is the explicit Laplace-Beltrami viscous limit,
$dt\,\mu/h_{min}^2 \approx 0.5$ (log `## P1`, P1b subsection): a dt linear in
$h$ cannot track a limit in $h^2$, and at the default `--min-dt` 2.5e-4 the
floor itself is unstable once `minEdge` falls below about `1.0e-3`. *Presents
as:* a nonfinite stop within the first tens of steps at level 7 but not at
level 6, while `minQ` stays above about 0.3. *Distinguished from R2 by:* `minQ`
and `minEdge` at the last progress line both stay near their starting values.
P1b's `L7_np4_dt1e-3_mu0` arm (stable at the `--dt` where the viscous arm blew
up) is the direct test. *Do:* D1 keeps $dt\,\mu/h_{min}^2 \lesssim 0.4$,
evaluated on the measured `minEdge`, not on `h_min`, e.g. with
`--dt-edge-power 2` and a lowered `--min-dt`. Lowering `--dt` alone is not
enough.

**R2 — mesh quality collapses late.** The pre-T4d sweeps stopped with `minQ`
0.03–0.11 and dt at or near its floor (Current state). The T4d passes
(collapse, quality flips, guarded smoothing, cleanup) exist to prevent this.
The V1 pilot, with them, keeps `minQ` ≥ 0.30 to $t = 2.224$ and 136,952
vertices (Current state). *Presents as:*
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

**R8 — `h_min` binds before 200,000 vertices.** The sizing field clamps at
`--remesh-h-min` (Sizing row), so where the roll-up needs finer edges than
that, refinement stops and `V` grows more slowly than the projection.
*Presents as:* `minEdge` pinned at `h_min` for many progress lines while
`max|S|` grows. *Do:* D1 sets `h_min` with this in view. A pinned `minEdge`
is not a defect.
