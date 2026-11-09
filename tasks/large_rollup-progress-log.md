# Large roll-up — progress log

Session record for large_rollup. Companion to `large_rollup.md`, which holds the
design, the task sequence and the risks; this file holds what actually
happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `large_rollup.md` can cite it
by ID. No dates: the order of the sections is the chronology. If a session
covers more than one task, name them all; if it belongs to no task, name the
topic.

**Begin each implementing section with `**Commits:**`** — the SHAs or the
range the task produced. A reviewer finds the task's diff from this line;
without it, the diff has to be guessed from file history, and a guess that
includes a neighbor's commit reviews the wrong code. **Follow it with
`**Jobs:**`** when jobs verified the exit criterion — each job's ID, script,
output path, and the direction it showed — so a review can verify the task
from that output instead of queueing another run.

Reviews of a task are recorded here too, as `## <ID> review` sections written
by `review-task`, and a feature-wide sweep as `## Sweep`. They follow the same
rules, end with the same `**Affects:**` line, and are how a later session
learns what a review required.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

## P1

**Commits:** `560cb03` (`scripts/tuolumne/large_rollup_probe.flux`); this
section and the P1 exit-criterion wording fix in `large_rollup.md` are the
following commit.
**Jobs:** `f3j99ZVqFY9d`, `scripts/tuolumne/large_rollup_probe.flux`, job log
`/p/lustre5/stewartj/beatnik/large_rollup/P1/large_rollup_P1.f3j99ZVqFY9d.log`,
per-launch logs `/p/lustre5/stewartj/beatnik/large_rollup/P1/<tag>/launch.log`
(checkpoints beside them). It did **not** show the control direction at step
60: no level-7 launch reached step 60. Over steps 1–6 the direction holds (see
below).

**Status: NOT DONE — open on a user decision.** Every level-7 launch went
nonfinite within 6–33 steps, so the control-direction clause of the exit
criterion cannot be evaluated as written. The question is with the user.

### Decisions (taken before the run)

- **Waiting is handed to the loop controller.** **Do** 4's "wait in-session
  past 20 minutes" became "wait at most 10 minutes, then hand off". The job
  ran 14.84 min (start 14:07:46, end 14:22:38, about 1.5 min queued in
  `pdebug`), inside the 15-minute estimate. The session handed off at 9 min
  and resumed after the job ended.
- The launch matrix, `--steps 60`, `--eps 0.025`, the `6m` caps and the control
  row ran exactly as the task fixes them; no launch was added, dropped or
  resized.
- Implementation choices inside **Fill in**: `--setopt=mpibind=verbose:1` on
  every launch (binding recorded in each `launch.log`; np1 sees GPUs 0–3, np4
  one GPU per rank); the launch's start epoch is written as the first line of
  each `launch.log`; the job log repeats each launch's progress/stop/rss lines
  before its `[large_rollup] <tag> rc= wall=` line.
- Exit-criterion wording in `large_rollup.md` P1 corrected from "the table
  above" to the per-launch record of **Do** 5 (cosmetic, as the prompt
  directed).

### Dev env rebuild

`spack -e ~/spack_envs/tuolumne_beatnik install` was **not** a no-op: it
rebuilt `tessera@develop` (`aikc55n`) and `beatnik@develop` (`ckhvawf`). The
first invocation hit the session's 95 s command timeout while still building.
It finished in the background. A second `install` (no lock contention) then
completed beatnik in 43 s. The binary reports `Beatnik 0.1.0 (01e874f…)`, the
HEAD at submission. A third invocation started while the first held the stage
lock failed with `LockTimeoutError`; that one is harmless.

### Per-launch record

s/step is the median of the per-step time over progress intervals after step
1, from the line timestamps. Setup is launch start to the `step=    1` line.
"Last line" is the last progress line before the stop. `rss_kb` is the peak
per rank from `/usr/bin/time`.

| Tag | rc | Outcome | V / F step 1 | V / F last line (step) | s/step | Setup s | dt last | minQ / minEdge / volRel last | max\|S\| last | Peak RSS per rank (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `L6_np1` | 0 | reached step 60 | 40962 / 81920 | 40962 / 81920 (60) | 0.120 | 8.4 | 1.87e-3 | 0.974 / 0.0027 / -4.4e-16 | 7.8e-2 | 1.07 |
| `L6_np4` | 0 | reached step 60 | 40962 / 81920 | 40962 / 81920 (60) | 0.118 | 7.4 | 1.87e-3 | 0.974 / 0.0027 / 0 | 7.8e-2 | 1.05, 1.06, 1.06, 1.06 |
| `L7_np1` | 142 | killed by 6m cap after step 6; no stop line | 163842 / 327680 | 163842 / 327680 (6) | 1.134 (1→6) | 7.1 | 2.39e-3 | 0.974 / 0.00169 / -4.4e-16 | **3.45e4** | — (killed before `time` reported) |
| `L7_np4` | 1 | `stopping at step=12 t=0.0307: nonfinite mesh-RHS state` | 163842 / 327680 | 163842 / 327680 (6) | 0.442 (1→6) | 8.3 | 2.39e-3 | 0.974 / 0.00169 / -1.7e-15 | **3.45e4** | 1.55, 1.52, 1.57, 1.51 |
| `L7_np4_dt1e-3` | 1 | `stopping at step=33 t=0.0260: nonfinite mesh-RHS state` | 163842 / 327680 | 165042 / 330080 (30) | 0.442 (steps 1→24) | 8.3 | 1.00e-3 (24: 6.86e-4) | 1.1e-4 / 3.7e29 / 9e96 (30: garbage) | 5.3e42 | 1.57, 1.52, 1.60, 1.61 |
| `L7_np4_hmax_default` | 1 | `stopping at step=12 t=0.0338: nonfinite mesh-RHS state` | 163841 / 327678 | 163753 / 327502 (6) | 2.84 (1→6) | 9.6 | 2.73e-3 | 0.835 / 0.00196 / 0 | 5.4e2 | 1.73, 1.59, 1.59, 1.55 |

Notes on the table:

- **Level 6 never remeshed.** `remesh_events=0` on every line, with V and F
  constant. With `h_max` 4.1e-3 and the starting edge about 3.95e-3, nothing
  needed splitting in 60 steps, and nothing was collapsed because `h_max` holds
  the mesh. So the level-6 numbers are pure fixed-connectivity cost. The one
  slow interval in each level-6 run (0.39 s and 0.18 s per step over steps
  48→54) is a one-off, not a trend.
- **Level-7 `dt1e-3` trajectory.** `max|S|` went 6.6e-4 (step 1), 3.3e-3 (6),
  6.1e-3 (12), **0.30 (18)**, **252 (24)**, then 5e42 at step 30, with 286,600
  remesh events by then. `minQ` stayed 0.974 and `minEdge` 1.97e-3 → 1.48e-3
  through step 24. The blow-up precedes any remeshing; the remesh storm and the
  slow steps (17–21 s/step) come after it.
- **`L7_np1`'s cap kill is the same blow-up.** Its step-6 line is
  bit-identical in `max|S|` (3.44960e4) to `L7_np4`'s. `L7_np4` then took 150 s
  for steps 6→12 before the nonfinite stop; at about 2.6× np4's per-step cost,
  np1 would need more than the 354 s it had left.
- **Control (`L7_np4_hmax_default`).** It coarsens from step 1:
  `remesh_events` 1 → 89, F 327678 → 327502 at step 6, while `L7_np4`'s F stays
  at 327680. The control direction therefore **holds over steps 1–6**, but the
  exit criterion asks for step 60, which neither launch reached.

### What failed: level 7 at `eps` 0.025 is unstable from the first steps

Every level-7 arm grows `max|S|` by orders of magnitude within 0.015–0.02 time
units, with `minQ` at its starting 0.974 and `minEdge` near its start. That is
R1's signature ("nonfinite stop within the first tens of steps at level 7 but
not at level 6, while `minQ` stays above about 0.3"). But R1's direct test, the
`dt1e-3` arm, **also blew up**. Cutting `--dt` threefold delayed the blow-up
only from before `t` 0.0157 to about `t` 0.018–0.023; it did not prevent it. So
R1's prescribed remedy ("D1 lowers `--dt`") is not shown to work at a
threefold reduction.

Hypothesis, not tested: an explicit **viscous** stability limit. Adaptive dt
scales linearly with the minimum edge (`chooseStepSize`,
`src/Beatnik_TimeIntegrator.hpp:262-274`, `dt_edge_power` 1). Explicit
Laplace-Beltrami viscosity, however, needs $dt \lesssim c\,h^2/\mu$. Measured
$dt\,\mu/h_{min}^2$:

| Run, line | dt | minEdge | $dt\,\mu/h^2$ | Stable? |
| --- | --- | --- | --- | --- |
| L6, step 6 | 2.39e-3 | 3.38e-3 | 0.42 | yes |
| L6, step 60 | 1.87e-3 | 2.70e-3 | 0.51 | yes |
| L7_np4_dt1e-3, step 18 | 7.07e-4 | 1.52e-3 | 0.61 | growing |
| L7_np4, step 6 | 2.39e-3 | 1.69e-3 | 1.67 | blown up |

The stable rows sit at 0.42–0.51 and the unstable ones at 0.61 and above. That
is consistent with a limit near $c \approx 0.5$, which a linear-in-$h$ dt
cannot track as the mesh refines. If it holds, a level-7 start needs
$dt \lesssim 0.5 \times (1.5\times10^{-3})^2 / 0.002 \approx 5.6\times10^{-4}$
once edges reach `h_min`, or `--dt-edge-power 2`. Neither was run: either one
changes the launch matrix and is the user's call.

### Cost extrapolation (arithmetic, not a run)

The only level-7 cost points not contaminated by the blow-up are the step 1→6
intervals at fixed connectivity (`remesh_events=0`): **1.134 s/step np1, 0.442
s/step np4**. `dt1e-3` repeats 0.442 over steps 1→24.

- **Vertex exponent.** np1 level 6→7 is 0.120 → 1.134, ×9.45 for ×4 in V, an
  exponent of 1.62. That is approaching the direct sum's $N^2$ as the GPU
  fills; level 6 at np1 is not saturated. I assume **$N^2$** from level 7 up:
  pessimistic by at most a factor $1.22^{0.38} \approx 1.08$.
- **Rank exponent (R4).** Level 6: np1 0.120 vs np4 0.118, no speedup, as R4
  predicted from the estimate. Level 7: 1.134 / 0.442 = 2.57× on 4 ranks, so
  $P^{-0.68}$, not $P^{-1}$. I assume **$P^{-0.68}$** for 8 and 16 ranks
  (optimistic if efficiency keeps falling with P; $N^2/P$ given alongside as
  the bound).
- s/step at 200,000 vertices: $0.442 \times (200000/163842)^2 = 0.659$ at np4;
  then **np4 0.66, np8 0.41, np16 0.26** with $P^{-0.68}$ (with $N^2/P$: 0.66,
  0.33, 0.16).
- Steps to $t = 2.5$. The only measured level-7 dt before blow-up is 6.86e-4
  (`dt1e-3` arm, step 24), which gives about **3,650 steps**: np4 about 40 min,
  np8 25 min, np16 16 min of BR cost. At level 6's 1.87e-3 it would be 1,340
  steps. Both dt values come from runs that are either unstable (level 7) or
  never refined (level 6), and remesh cost is not included, so the step count
  is the weak link, not s/step. If the viscous limit holds, dt at
  `h_min` 1.5e-3 is at most about 5.6e-4, giving 4,500+ steps.
- **Memory:** 1.06 GB/rank at 41k vertices (np1 and np4 alike, so the
  footprint is mostly per-process/runtime, not per-vertex), 1.5–1.7 GB/rank at
  164k on np4. Not a constraint on any segment size considered.

### Risks the numbers bear on

- **R1** — level 7 is unstable at the starting dt, but a threefold dt cut
  does not cure it; the dt the run needs is unknown. Viscous $h^2$ hypothesis
  above.
- **R4** — confirmed at level 6 (no np4 speedup); partly at level 7 (2.57×
  on 4 ranks).
- **R8** — not reached: no launch refined, and `minEdge` reached `h_min` 1.5e-3
  only in the blown-up `dt1e-3` arm.

**Affects:** D1 — cannot choose a level-7 start, a dt or an `h_min` until the
level-7 instability is explained; the s/step and RSS numbers above are its
cost inputs. V1j/R1j — segment sizing changes with the step count (3,650 to
4,500+ steps at level-7 dt, against 1,340 at level 6's dt); RSS is not a
constraint. R1 (risk) — its "D1 lowers `--dt`" remedy is unproven at 3×.
