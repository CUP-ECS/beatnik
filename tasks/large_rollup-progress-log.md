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

**Commits:** `560cb03` (`scripts/tuolumne/large_rollup_probe.flux`), `6cef416`
(this section's first part and the P1 exit-criterion wording fix), and
`c938e46` (`scripts/tuolumne/large_rollup_probe_b.flux`, the P1b
subsection below, P1 marked **DONE**).
**Jobs:**
- `f3j99ZVqFY9d`, `scripts/tuolumne/large_rollup_probe.flux`, job log
  `/p/lustre5/stewartj/beatnik/large_rollup/P1/large_rollup_P1.f3j99ZVqFY9d.log`,
  per-launch logs `/p/lustre5/stewartj/beatnik/large_rollup/P1/<tag>/launch.log`
  (checkpoints beside them). It did **not** show the control direction at step
  60, because no level-7 launch reached step 60. Over steps 1–6 the direction
  holds (see below).
- `f3j9KFoF6Yto`, `scripts/tuolumne/large_rollup_probe_b.flux`, job log
  `/p/lustre5/stewartj/beatnik/large_rollup/P1/large_rollup_P1b.f3j9KFoF6Yto.log`,
  per-launch logs under the same `P1/<tag>/launch.log`. It **showed** the
  control direction at step 60 at `--dt 4e-4` (P1b subsection).

The first job left P1 open on a user decision: every level-7 launch went
nonfinite within 6–33 steps, so the control-direction clause could not be
evaluated. The user chose a follow-up probe, P1b, recorded below.

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

### P1b — level-7 viscous-limit probe (user decision)

**User decision:** after the first job, the user was asked how P1 should
proceed. The answer was to add a follow-up probe (P1b) at level 7 testing the
viscous-limit hypothesis (`--dt-edge-power 2` and/or `--dt 4e-4`, so
$dt\,\mu/h^2 \lesssim 0.4$), plus the control row at the same dt, then
re-evaluate the control direction at step 60. P1b therefore extends P1's fixed
launch matrix with the user's approval. The second script,
`scripts/tuolumne/large_rollup_probe_b.flux`, is outside P1's **Fill in** for
the same reason. It copies the first script's base arguments, launch line,
caps and logging, with level 7 on 4 ranks for every arm.

Implementation choice within that decision: a fourth, diagnostic arm,
`L7_np4_dt1e-3_mu0`. It is P1's blown-up `dt1e-3` arm with `--mu 0`, the
direct discriminator for "viscosity is the cause". The job ran 7.0 min (queued
3.9 min in `pdebug`), exit 0; no rebuild between the two jobs.

| Tag | rc | Outcome | V / F step 1 | V / F step 60 | s/step | Setup s | dt step 60 | minQ / minEdge / volRel step 60 | max\|S\| step 60 | $dt\,\mu/h^2$ step 60 | Peak RSS per rank (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `L7_np4_dt4e-4` | 0 | reached step 60 | 163842 / 327680 | 163842 / 327680 | 0.441 | 12.5 | 2.50e-4 (= `--min-dt`) | 0.974 / 0.00135 / -2.2e-16 | 1.06e-2 | 0.27 | 1.16, 1.18, 1.14, 1.15 |
| `L7_np4_dt8e-4_pow2` | 0 | reached step 60 | 163842 / 327680 | 163842 / 327680 | 0.438 | 8.7 | 3.13e-4 | 0.974 / 0.00135 / 0 | 1.49e-2 | 0.34 | 1.15, 1.15, 1.16, 1.18 |
| `L7_np4_hmax_default_dt4e-4` | 0 | reached step 60 | 163841 / 327678 | 159978 / 319952 | 4.51 | 10.0 | 3.48e-4 | 0.704 / 0.00186 / 6.7e-16 | 1.66e-2 | 0.20 | 1.50, 1.53, 1.55, 1.48 |
| `L7_np4_dt1e-3_mu0` | 0 | reached step 60 | 163842 / 327680 | 163842 / 327680 | 0.441 | 15.2 | 6.25e-4 | 0.974 / 0.00135 / 1.3e-15 | 2.81e-2 | — ($\mu = 0$) | 1.14, 1.16, 1.15, 1.29 |

(s/step is the median over the post-step-1 intervals. The step 54→60 interval,
about 0.81 s/step, includes the step-60 checkpoint write.)

**Control direction holds at step 60.** The control's F fell from 327678 to
319952 (−2.4 %, 4,195 remesh events). `L7_np4_dt4e-4`, at the same dt with
`h_max` 2.05e-3, held F at 327680 (0 %). `h_max` is what holds the fine
start's resolution. P1's `L7_np4` / `L7_np4_hmax_default` pair at `--dt 0.003`
cannot show the same thing, because both blow up first.

**The viscous-limit hypothesis is confirmed.** At `--dt 1e-3`, `mu` 0.002
blew up (P1: `max|S|` 252 at step 24, nonfinite at step 33). `mu` 0 at the
same dt ran to step 60 and `t` 0.0416, past where the viscous run died, with
`max|S|` growing smoothly to 2.8e-2. Both arms that keep
$dt\,\mu/h^2 \le 0.34$ are stable. Across both jobs, every stable line sits at
0.20–0.51 and every unstable one at 0.61 or above. The limit for explicit
Laplace-Beltrami viscosity here is near $dt\,\mu/h_{min}^2 \approx 0.5$.

Two consequences for D1:

- **Linear dt scaling hits the floor.** `L7_np4_dt4e-4` reached
  `--min-dt` 2.5e-4 by step 60, because `minEdge` fell from 1.97e-3 to 1.35e-3
  with no remeshing: the deforming sphere compresses edges below `h_min`
  without a split or collapse. At the floor, stability needs
  $h \ge \sqrt{2.5\times10^{-4}\cdot0.002/0.5} = 1.0\times10^{-3}$. Below that,
  the floor itself is unstable. `--dt-edge-power 2` tracks the limit at a
  constant ratio (about 0.34–0.41 in the run) and stayed off the floor; it is
  the natural choice, with `--min-dt` lowered rather than raised.
- **Remeshing dominates cost when it is active.** The control did about 70
  remesh edits per step and cost 4.5 s/step. The arms whose mesh `h_max` held
  cost 0.44 s/step, so remeshing at that rate is about 10× the direct BR sum.
  No arm of either job refined (`refine_events=0` throughout), so
  refinement-driven remesh cost at 200k vertices is still unmeasured.

**Revised extrapolation.** BR s/step is unchanged: level-7 np4 is 0.44 s/step
over 60 steps on three arms, confirming P1's step 1→6 number. At 200k
vertices that gives np4 0.66, np8 0.41 and np16 0.26 s/step. dt is set by the
viscous limit at $c \approx 0.4$, i.e. $dt \approx 200\,h_{min}^2$: about
4.5e-4 at $h$ 1.5e-3, 3.6e-4 at 1.35e-3, and 2.0e-4 at 1.0e-3. Taking about
3.5e-4 gives about **7,100 steps to $t = 2.5$**. BR alone is then about 78 min
at np4, 49 at np8 and 31 at np16. If remeshing runs at the control's rate
throughout (4.5 s/step at np4, 164k vertices), the run is about 9 h at np4.
The truth lies between these and is set by how much the run remeshes, which
P1 cannot measure.

**Memory:** 1.14–1.29 GB/rank on the held-mesh arms and 1.48–1.55 GB/rank on
the remeshing control, at about 164k vertices on 4 ranks. That matches P1's
1.5–1.7 GB/rank on the remeshing/blown-up arms. Remeshing adds about
0.35 GB/rank. Memory is not a constraint.

### Risks, updated by P1b

- **R1** is confirmed as a dt instability, and P1b identifies it as the
  explicit viscous limit $dt \lesssim 0.5\,h^2/\mu$. Its remedy should read
  "D1 keeps $dt\,\mu/h_{min}^2 \lesssim 0.4$, e.g. `--dt-edge-power 2` with a
  lowered `--min-dt`". "Lowers `--dt`" alone is not enough when dt scales
  linearly with $h$, because `--min-dt` 2.5e-4 becomes unstable once $h$ falls
  below 1.0e-3.
- **R4** — unchanged (level-6 np4 no faster; level-7 np4 2.57× on 4 ranks).
- **R8** — `minEdge` fell below `h_min` 1.5e-3 (to 1.35e-3) by compression,
  not refinement, so `h_min` does not bound `minEdge` and does not bound the
  viscous dt. D1 should size dt on the measured `minEdge`, not on `h_min`.

**Affects:** D1 — the step size must respect the viscous limit
$dt\,\mu/h_{min}^2 \lesssim 0.4$; `--dt-edge-power 2` with a lowered `--min-dt`
is the measured-stable choice, and `--dt 0.003` at level 7 is not viable. Its
cost inputs are 0.44 s/step BR at 164k vertices np4 (0.66 projected at 200k)
and about 4.5 s/step while remeshing heavily, about 7,100 steps to $t = 2.5$,
and RSS ≤ 1.7 GB/rank. V1j/R1j — segment sizing must use about 7,100 steps
and a remesh-dependent s/step between 0.66 and about 5 (np4), not P1's
first-job 1,340–4,500 range. R1 (risk text) — its remedy is superseded by the
viscous-limit finding; D1 or a design-doc correction should restate it.

## P1 review

**Verdict:** Passed
**Reviewed:** `560cb03^..964fab9` (code: `scripts/tuolumne/large_rollup_probe.flux`,
`scripts/tuolumne/large_rollup_probe_b.flux`; the rest is `tasks/`)
**Evidence:** the implementing session's jobs, output read; no verification job,
since all four checks pass and the criterion is a measurement record, not a
tolerance near its bound.
- `f3j99ZVqFY9d`, `scripts/tuolumne/large_rollup_probe.flux`, log
  `/p/lustre5/stewartj/beatnik/large_rollup/P1/large_rollup_P1.f3j99ZVqFY9d.log`:
  COMPLETED, exit 0, queued 1.6 min, ran 14.84 min. Echoed HEAD `01e874f`,
  porcelain 3; `git diff 01e874f 964fab9 -- src examples tests CMakeLists.txt`
  is empty, so the binary is the one every P1 commit describes. Six summary
  lines: `L6_np1` rc 0, `L6_np4` rc 0, `L7_np1` rc 142 (cap), `L7_np4`,
  `L7_np4_dt1e-3`, `L7_np4_hmax_default` rc 1 with `nonfinite mesh-RHS state`
  stop lines at steps 12, 33, 12.
- `f3j9KFoF6Yto`, `scripts/tuolumne/large_rollup_probe_b.flux`, log
  `/p/lustre5/stewartj/beatnik/large_rollup/P1/large_rollup_P1b.f3j9KFoF6Yto.log`:
  COMPLETED, exit 0, queued 3.9 min, ran 7.0 min. Echoed HEAD `6cef416`
  (only `tasks/` and scripts after `01e874f`). All four arms rc 0 and reached
  step 60.
- Both directions, read from each `<tag>/launch.log`: control
  `L7_np4_hmax_default_dt4e-4` F 327678 (step 1) → 319952 (step 60),
  `remesh_events` 4195; held `L7_np4_dt4e-4` F 327680 → 327680. At `--dt
  0.003` the same pair shows F 327678 → 327502 against 327680 → 327680 at
  step 6 before both blow up. Every V/F, dt, `minQ`, `minEdge`, `volRel`,
  `max|S|`, stop line and `rss_kb` spot-checked in the two tables matches the
  launch logs.

The control pair was read at `--dt 4e-4` (P1b) rather than at the `--dt
0.003` the exit criterion names, and P1b's script lies outside **Fill in**;
both are the user's recorded decision in `## P1`, and are settled.

1. [nit] `tasks/large_rollup-progress-log.md` `## P1` — the peak-RSS columns
   use different units: the first table divides `rss_kb` by 10⁶ (`L7_np4`
   1546632 → 1.55), the P1b table by 2²⁰ (`L7_np4_dt4e-4` 1218968 → 1.16).
   "Matches P1's 1.5–1.7 GB/rank" therefore compares GB with GiB (P1's
   range is 1.44–1.65 GiB). No conclusion changes; memory is not a
   constraint either way. Accepted.
2. [follow-up] `tasks/large_rollup.md` Known risks R1 — its remedy ("D1
   lowers `--dt`; do not raise `--min-dt`") is contradicted by P1b's
   viscous-limit finding ($dt\,\mu/h_{min}^2 \lesssim 0.4$; `--dt-edge-power
   2` with a lowered `--min-dt`). Restating a risk's remedy is a design
   correction for the user, to be taken up when D1's prompt is written.
3. [follow-up] P1 measured no refinement (`refine_events=0` throughout), so
   remesh cost at 200,000 vertices while refining is unmeasured; D1's cost
   estimate brackets it at 0.66–about 5 s/step (np4).

**Affects:** none beyond what `## P1`'s **Affects:** already states (D1, V1j/R1j, the R1 risk text).

## D1

**Commits:** this section only (D1 is open, so nothing else is committed for it yet).

D1 is **not DONE**. The first session prepared eleven choices from P1's
record, with no new measurement, and put them to the user. The user's answers
are below as decisions. Answer 1 picks an edge-tied eps, which needs a new
probe, **P1c**, before dt and the step count can be fixed. That probe is a new
task and is not in the design, so D1 stops here. The Production configuration
table is left at `D1`, because the exit criterion requires every row to be
filled and the eps, step-count and node-hour rows depend on P1c.

### Decisions (the user's answers)

| # | Choice | Decision | Alternatives considered | Measured basis |
| --- | --- | --- | --- | --- |
| 1 | `--eps` | **edge-tied, about `8e-3`** (about $4h$ at level 7); value fixed after P1c | 0.025, configuration (a)'s value and the only one measured | At 0.025 the smoothing length is about 12 level-7 edges (design Approach). No P1 launch ran a smaller eps, so its stable dt and step count are unmeasured. |
| 2 | Start and sizing | **level 7, `--remesh-h-max 2.05e-3`, `--remesh-h-min 1.5e-3`, `--remesh-sagitta-tolerance 0.004`** | level 7 with `h_min` 1.0e-3; level 7 with sagitta about 3e-5; level 6 grown by AMR | P1b: F 327680 → 327680 held, against 327678 → 319952 for the default-`h_max` control. With sagitta 0.004, curvature sizing falls below `h_max` only where $\kappa > 8\cdot0.004/(2.05\times10^{-3})^2 \approx 7600$, so growth comes from stretched edges splitting at $1.35\,h_{max}$. Whether V passes 200,000 is unmeasured, and the pilot shows the growth ratio. |
| 3 | Edit caps | **`--remesh-max-splits 1000 --remesh-max-collapses 400`** | the defaults, 300 / 120 | The P1b control made about 70 edits per step at 4.51 s/step, against 0.44 held. The cost of the raised caps is unmeasured. |
| 4 | Step size | **`--dt 8e-4 --dt-edge-power 2 --min-dt 5e-5`** | `--dt 7e-4` with the same power and floor; `--min-dt 2.5e-5` | P1b's `L7_np4_dt8e-4_pow2` held $dt\,\mu/h_{min}^2$ at 0.34–0.41 and stayed off the floor. A floor of 5e-5 is stable down to $h = \sqrt{5\times10^{-5}\cdot0.002/0.4} = 5.0\times10^{-4}$. At eps 0.025 this gives about 7,100 steps to $t = 2.5$ (minEdge near 1.35e-3) up to about 16,000 (minEdge near the collapse floor $0.45\,h_{max} = 9.2\times10^{-4}$). P1c must re-measure this at the chosen eps. |
| 5 | Ranks / nodes | **np8 on 2 nodes** | np4 on 1 node; np16 on 4 nodes | Level-7 scaling is $P^{-0.68}$ (np1 1.134, np4 0.442 s/step). At 200k vertices the direct-sum cost is 0.41 s/step at np8. How remeshing cost scales with ranks is unmeasured. |
| 6 | Segment walltime | **12h** in `pbatch` | 24h (the `pbatch` maximum); 6h | `flux queue list`: `pbatch` TDEFAULT 12h, TLIMIT 1d. |
| 7 | Checkpoint cadence | **`--checkpoint-every-time 0.025`** | 0.01; 0.05 | A level-7 checkpoint is 64.57 MB, about 79 MB at 200k. A write costs about 2.2 s at 164k (P1b step 54→60 ran 0.81 s/step against 0.44), about 2.7 s at 200k. That makes about 100 files, about 8 GB and about 4.5 min of writes to $t = 2.5$. |
| 8 | `--t-end` | **2.5** | 3.0 | No Beatnik run has passed $t$ 2.53 (design Current state table). |
| 9 | Pilot | **level 6, `--remesh-h-max 4.1e-3 --remesh-h-min 3.0e-3 --remesh-sagitta-tolerance 0.016`, `--dt 2.8e-3 --min-dt 2e-4 --dt-edge-power 2`, np4 on 1 node, two segments at `-t 90m` in `pbatch`** | level 6 with production's `h_min` and sagitta; level 5 | Level 6 holds at `h_max` 4.1e-3 and costs 0.118 s/step np4. Doubling the lengths leaves the curvature threshold $\propto \mathrm{tol}/h^2$ unchanged and scales the viscous dt by 4. The dt row was scaled from `--dt 7e-4`. At the chosen `--dt 8e-4` the same rule gives `3.2e-3`, which is open for P1c (below). |
| 10 | Pilot quality floor | **`minQ` ≥ 0.12** | 0.18 (the repair trigger); 0.05 | The pre-T4d long runs ended at `minQ` 0.033–0.11. `--remesh-min-quality` 0.18 triggers the repair pass, so `minQ` dips below 0.18 are repaired, not failures. |
| 11 | Node-hour budget | **about 35 node-h** for R1j | about 15 node-h | the estimate below |

### Production node-hours, under the decisions (eps 0.025 inputs)

np8 on 2 nodes, `--dt 8e-4`, 7,100–16,000 steps:

| Regime | s/step | Wall | Node-h | ×1.25 (R1j chain margin) |
| --- | --- | --- | --- | --- |
| Direct sum only | 0.41 | 0.8–1.8 h | 1.6–3.6 | 4.6 |
| Remeshing every step, scaled $P^{-0.68}$ | 3.12 | 6.2–13.9 h | 12.3–27.7 | 34.7 |
| Remeshing every step, not scaled | 5.0 | 9.9–22.2 h | 19.7–44.4 | 55.6 |

The budget answer was framed for np4's worst case (25.4 node-h). At np8 the
worst case is within 35 node-h only if remeshing scales with ranks as the
direct sum does. If it does not scale, the worst case exceeds the budget
(44 node-h, 56 with the margin). The edge-tied eps and the raised caps can
raise the step count and the remesh cost, and neither is measured yet. So D1
cannot yet say whether the run fits the budget. The failure direction is
neither met nor ruled out.

### Open, for P1c and the resumed D1

- **P1c probe (proposed shape, for the design):** P1b-shaped, 60 steps,
  level 7, `--remesh-h-max 2.05e-3`, the decided step-size and edit-cap flags,
  at the chosen eps, on np4. An np8 arm on 2 nodes in `pdebug` (≤ 45 min) would
  also measure the np8 direct-sum and remesh scaling that the budget question
  now turns on.
- The pilot's eps under "same eps" (production's edge-tied value, or the same
  ratio to level 6's edge, about `1.6e-2`), and the pilot's `--dt` at the
  chosen production `--dt 8e-4` (`3.2e-3` by the ×4 rule).
- The design's Approach `--eps` bullet stays true ("decided in D1"). Adding
  P1c to the task sequence is a design change, for design-doc-writer.

**Affects:** D1 — blocked on P1c; table untouched. P1c (new, not in the design)
— measures stable dt, step count and cost at the edge-tied eps with the raised
caps, ideally with an np8 arm. V1j — pilot is level 6 np4 on 1 node, two
segments at `-t 90m` in `pbatch`; its eps and `--dt` await P1c. R1j —
production is np8 on 2 nodes, 12h segments; the segment count and fit within
35 node-h await P1c.

### Resumed after P1c — questions prepared (2026-11-09)

The resumed session read `## P1c` and `## P1c review` and put four items to the
user; no measurement was added and the table is still at `D1`. Cosmetic
design-doc corrections made in the same commit: the doc's **Status** reads
IN PROGRESS, and D1 **Do** cites P1c's four arms (ratio 0.343) under step size
and P1c's np8/np4 0.97 under ranks and nodes.

1. **Final production eps** — recommended `--eps 8e-3`: stable to $t = 0.023$
   with dt identical to 0.025 (`## P1c` derived result 1); later stability is
   unmeasured (P1c review follow-up 3), V1 shows it.
2. **Pilot eps and `--dt`** — recommended `--eps 1.6e-2 --dt 3.2e-3`: P1c's
   `L6_np4_eps1.6e-2` ran stable at ratio 0.343, 0.118 s/step np4.
3. **Ranks and nodes (decision 5)** — recommended np4 on 1 node (11.1–23.8
   node-h remeshing every step, 29.7 with the ×1.25 margin, inside 35) over
   np8 on 2 nodes (21.6–47.5, 59.4 with the margin; fits 35 only at the
   8,000-step end). Remesh cost while refining, and its rank scaling, are
   unmeasured and bracket both.
4. **Segment count** — at np4, worst case about 23.8 h wall, 29.7 h with the
   margin: three 12h `pbatch` segments for R1j.

**Affects:** D1 — waits on the user's four answers; table untouched.

### User's answers to the four post-P1c questions (2026-11-09)

1. **`--eps 8e-3`** for production (decision 1 fixed).
2. **Pilot `--eps 1.6e-2 --dt 3.2e-3`** (decision 9's dt row superseded).
3. **np8 on 2 nodes, node-hour budget raised to about 60 node-h** (decision 5
   kept; decision 11 raised from 35).
4. **"three 12h segments, np4 on 1 node, pbatch"**.

Answers 3 and 4 conflict on ranks and nodes: 3 keeps np8 on 2 nodes with a
budget sized for np8's worst case (59.4 node-h with the margin), while the
option chosen in 4 names np4 on 1 node. The segment count (three 12h `pbatch`
segments) is the same either way, but the R1j node count and the node-hour
estimate are not, so the table is not filled until the user resolves it.

**Affects:** D1 — waits on the ranks/nodes clarification; table untouched.

### Final decisions and close-out (2026-11-09)

**Commits:** `422f0c3` (questions; doc status and D1 **Do** cite P1c),
`95dbe0c` (answers 1–4 recorded), and the commit that adds this subsection,
fills the Production configuration table and marks D1 DONE.

**Clarification answer:** R1j runs **np4 on 1 node, three 12h `pbatch`
segments, with a node-hour budget of about 60 node-h** (headroom over np4's
worst case). The user also asked how to raise the allowed node-hours; that is
the task loop's job-limit policy, outside the design, and goes back to the user
in the result.

Decisions beside the earlier eleven:

| # | Choice | Decision | Alternatives considered | Measured basis |
| --- | --- | --- | --- | --- |
| 1 (final) | `--eps` | **`8e-3`** | 0.025 | `## P1c`: `L7_np4_eps8e-3` stable to $t = 0.023$, dt sequence identical to 0.025 (8.0e-4 → 3.13e-4), `max|S|` equal to five figures. Later stability unmeasured; V1 shows it. |
| 5 (final) | Ranks / nodes | **np4 on 1 node** | np8 on 2 nodes; np16 | `## P1c` derived result 2: np8/np4 s/step 0.428/0.440 = 0.97 at twice the node-hours. |
| 9 (final) | Pilot eps, dt | **`--eps 1.6e-2 --dt 3.2e-3`** | `--dt 2.8e-3`; pilot at production eps `8e-3` | `## P1c`: `L6_np4_eps1.6e-2` stable through step 60 at ratio 0.343, 0.118 s/step np4. |
| 11 (final) | Node-hour budget | **about 60 node-h** | 35 node-h | the table below |
| 12 | R1j segments | **three 12h segments (`-t 720m`) in `pbatch`** | two 12h segments | np4 worst case 23.8 h wall, 29.7 with the ×1.25 margin. |

**Production node-hours, np4 on 1 node, 200,000 vertices, 8,000–17,100
steps** (`## P1c` derived results 1–3; remesh s/step from P1b's control, 4.51
at 164k on np4, about 5.0 at 200k):

| Regime | s/step (np4, 200k) | Wall | Node-h | ×1.25, upper |
| --- | --- | --- | --- | --- |
| Direct sum only | 0.656 | 1.5–3.1 h | 1.5–3.1 | 3.9 |
| Remeshing every step | 5.0 | 11.1–23.8 h | 11.1–23.8 | 29.7 |

Inside the 60 node-h budget in both regimes, so the failure direction is not
met. The bracket's uncertainty is remesh cost while refining, which no probe
measured (P1 review follow-up 3; no P1c launch remeshed); V1 measures it at
pilot scale.

**Pilot (V1j) estimate:** level 6, 40,962 vertices (4× fewer than production),
about 2,000–4,300 steps to $t = 2.5$ by $dt \approx 172\,h_{min}^2$ (`minEdge`
2.7e-3 down to $0.45 \times 4.1\times10^{-3}$). Wall at np4 is about 4–8.5 min
direct-sum only (0.118 s/step) and about 40–90 min if it remeshes every step at
P1b's ×10 rate (about 1.2 s/step). Two 90m `pbatch` segments cover it, about
3 node-h at most.

**Affects:** V1j — reads the `Pilot resolution` row (level 6, sizing, `--eps
1.6e-2`, `--dt 3.2e-3 --min-dt 2e-4 --dt-edge-power 2`, np4 on 1 node, two
90m `pbatch` segments) and the `Pilot quality floor` row (`minQ` ≥ 0.12), with
the production physics, caps, `--t-end` and checkpoint rows. R1j — reads every
production row: np4 on 1 node, three 12h `pbatch` segments, `--eps 8e-3`, the
level-7 sizing, caps, step size, `--t-end 2.5`, `--checkpoint-every-time
0.025`; expected 11.1–23.8 node-h. No Approach or Known-risks sentence is
made wrong by these values.

## P1c

**Commits:** `b014788` (`scripts/tuolumne/large_rollup_probe_c.flux`, new);
the docs commit that adds this section and marks P1c DONE in
`tasks/large_rollup.md`.
**Jobs:** `f3j9ihiaeo7u`, `scripts/tuolumne/large_rollup_probe_c.flux`, job log
`/p/lustre5/stewartj/beatnik/large_rollup/P1c/large_rollup_P1c.f3j9ihiaeo7u.log`,
per-launch logs `/p/lustre5/stewartj/beatnik/large_rollup/P1c/<tag>/launch.log`
for the four tags below (checkpoints beside each). COMPLETED, exit 0; queued
11.4 min in `pdebug`, ran 2.6 min. Echoed HEAD `9e97da8`, porcelain 3
(`commands.sh`, the P1c prompt, the uncommitted script), exe
`.spack-env/view/bin/adaptive_mesh_bubble`, `beatnik@develop` `ckhvawf`, HIP
6.4.43484. All four launches rc 0 and reached step 60. **The baseline
`L7_np4_eps0.025` reached step 60 with $dt\,\mu/h_{min}^2 = 0.343$**
($\le 0.41$), matching P1b's `L7_np4_dt8e-4_pow2` line for line (step-60 dt
3.13e-4, `minEdge` 0.00135, `max|S|` 1.49e-2): the raised caps and the floor
`5e-5` changed nothing at this horizon.

An earlier submission, `f3j9ZmJVgkBy`, failed at `shell_init` (exception
type=exec, nothing ran) because the `--output` directory `.../large_rollup/P1c`
did not exist: the Flux shell opens the log before the script's own `mkdir`.
The directory was created and the unchanged script resubmitted. The job's
`jobid =` provenance line prints `unknown` (`FLUX_JOB_ID` is not set in the
batch shell); the job ID is the one above.

### Decisions (taken before the run)

- The launch matrix, caps, `--steps 60` and job size are exactly P1c's
  **Do** 1–4; no launch was added, dropped or resized.
- The baseline reproduces P1b's `L7_np4_dt8e-4_pow2` plus only the raised caps
  and `--min-dt 5e-5`. `--remesh-h-min 1.5e-3` and
  `--remesh-sagitta-tolerance 0.004` are the example's defaults
  (`examples/02_adaptive_mesh_bubble/InputFile.hpp:1148-1149`), so passing
  them explicitly changes nothing relative to P1b.
- `--mu 0.002` is in the base arguments (configuration (a)'s physics).
- Implementation choice: the launch table carries a node-count field (1 for
  np4, 2 for np8) beside the rank count; the launch line is otherwise P1b's.
- No source change (`git diff 01e874f HEAD -- src examples tests
  CMakeLists.txt` empty). `spack -e ~/spack_envs/tuolumne_beatnik install`
  re-ran beatnik `ckhvawf`'s build stages in 41 s, rc 0, before submission.
- Each solver header confirms the arguments took effect, e.g. `physics A 0.3,
  g 1, eps 0.008, mu 0.002` and `time steps 60, dt 0.0008, adaptive on,
  min_dt 5e-05` for `L7_np4_eps8e-3`.

### Per-launch record

s/step is the median over the post-step-1 progress intervals (every launch has
one slow interval, 48→54: 0.55–0.78 s/step at level 7 and 0.18 at level 6.
The median ignores it). Setup is launch start to the `step=    1` line. RSS is
peak `rss_kb` per rank in **GB = `rss_kb`/10⁶**. The ratio column is
$dt\,\mu/h_{min}^2$ at step 60 on the measured `minEdge`, with $\mu = 0.002$.

| Tag | rc | Outcome | V / F step 1 | V / F step 60 | s/step | Setup s | dt step 60 | minQ / minEdge / volRel step 60 | max\|S\| step 60 | $dt\,\mu/h^2$ step 60 | Peak RSS per rank (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `L7_np4_eps0.025` | 0 | reached step 60, t 0.0234 | 163842 / 327680 | 163842 / 327680 | 0.443 | 12.2 | 3.13e-4 | 0.974 / 0.00135 / 2.2e-16 | 1.48732e-2 | 0.343 | 1.21, 1.26, 1.21, 1.20 |
| `L7_np4_eps8e-3` | 0 | reached step 60, t 0.0234 | 163842 / 327680 | 163842 / 327680 | 0.440 | 11.8 | 3.13e-4 | 0.974 / 0.00135 / 0 | 1.48730e-2 | 0.343 | 1.21, 1.24, 1.20, 1.28 |
| `L7_np8_eps8e-3` | 0 | reached step 60, t 0.0234 | 163842 / 327680 | 163842 / 327680 | 0.428 | 9.2 | 3.13e-4 | 0.974 / 0.00135 / 2.2e-16 | 1.48730e-2 | 0.343 | 1.17, 1.16, 1.17, 1.16, 1.18, 1.17, 1.16, 1.17 |
| `L6_np4_eps1.6e-2` | 0 | reached step 60, t 0.0937 | 40962 / 81920 | 40962 / 81920 | 0.118 | 7.2 | 1.25e-3 | 0.974 / 0.0027 / 0 | 5.93112e-2 | 0.343 | 1.05, 1.06, 1.06, 1.05 |

No stop lines. `remesh_events=0`, `refine_events=0` and `flips=0` on every
progress line of every launch: nothing was split or collapsed, so the raised
caps were never exercised.

**np8 binding (mpibind, verbose):** two shells, each printing `task 0..3 nths
24 gpus 0..3` with cpus `1-7,9-15,17-23,97-103,105-111,113-119` (task 0)
through `73-79,81-87,89-95,169-175,177-183,185-191` (task 3): one GPU and 24
cores per rank, four ranks per node, on both nodes. The binding is correct, so
the np8 s/step is not an oversubscription artifact.

### Derived results

**1. Stable dt and steps to $t = 2.5$, eps `8e-3` against 0.025.** Both are
stable through step 60, and their dt sequences are identical (8.00e-4 →
3.13e-4), because the `--dt-edge-power 2` rule sets dt from `minEdge` alone and
eps does not enter it. The trajectories agree to five significant figures in
`max|S|` and differ only in `centroid_z` (2.50094e-1 vs 2.50104e-1). The
measured rule is $dt \approx 172\,h_{min}^2$ (ratio 0.343 in every arm, level 6
included). By P1b's method, the steps to $t = 2.5$ are about **8,000** at
`minEdge` 1.35e-3 and about **17,100** at the collapse floor
$0.45\,h_{max} = 9.2\times10^{-4}$, for both eps. That is about 12 % above
D1's 7,100–16,000, which used $c = 0.4$ ($200\,h^2$) where the run's rule gives
0.343. Caveat: 60 steps reach only $t = 0.023$, where the sheet strength is
small (`max|S|` 1.5e-2) and the eps difference has barely acted. "Stable at
eps `8e-3`" is therefore shown only for the start; whether a smaller eps
destabilises the rolled-up sheet later is not measured here.

**2. np8/np4 s/step at eps `8e-3`.** 0.428 / 0.440 = **0.97**, an exponent of
$P^{-0.04}$ from 4 to 8 ranks, against $P^{-0.68}$ from 1 to 4 (P1). Every
interval is a no-remesh interval, so this is the direct-sum-plus-fixed-mesh
cost only. **The ratio with remesh events is unmeasured**, because no launch
remeshed. At 200,000 vertices ($N^2$ from 163,842): np4 0.656, np8 0.638
s/step. Doubling the nodes buys about 3 %.

**3. Production node-hours at np8 (2 nodes), against 35 node-h.** D1's three
regimes, re-derived with the measured np8 cost and the 8,000–17,100 steps:

| Regime | s/step (np8, 200k) | Wall | Node-h | ×1.25 (R1j chain margin), upper |
| --- | --- | --- | --- | --- |
| Direct sum only | 0.638 | 1.4–3.0 h | 2.8–6.1 | 7.6 |
| Remeshing every step, scaled as the direct sum (×0.97) | 4.86 | 10.8–23.1 h | 21.6–46.2 | 57.8 |
| Remeshing every step, not scaled | 5.0 | 11.1–23.8 h | 22.2–47.5 | 59.4 |

The direct-sum regime fits 35 node-h with a wide margin. The two remeshing
regimes now coincide, because the direct sum itself does not scale to np8:
D1's middle row (3.12 s/step, $P^{-0.68}$) is not supported by the
measurement. A run that remeshes every step fits 35 node-h at the 8,000-step
end (21.6–22.2, 27.0–27.8 with the margin) and exceeds it at the 17,100-step
end (46–48, 58–59 with the margin). Whether the run fits is set by how much it
remeshes and how far `minEdge` falls, neither of which 60 steps measures.

For comparison (not a run): np4 on 1 node costs the same wall for half the
node-hours: direct only 1.5–3.1 node-h, remeshing every step 11.1–23.8
node-h, 29.7 with the margin, inside 35 throughout.

### Suggestions for D1 (not run)

- Reconsider decision 5 (np8 on 2 nodes). At 164k vertices np8 is 3 % faster
  than np4 on the measured intervals, at twice the node-hours. np4 on 1 node
  fits 35 node-h in every regime above; np8 does not at the long end.
- If D1 wants the eps `8e-3` stability shown past the start, a longer pilot
  (V1j) is where it shows, not another 60-step probe.

**Affects:** D1 — eps `8e-3` is stable at the start with the decided step-size
rule, and gives the same dt as 0.025 (3.13e-4 at `minEdge` 1.35e-3; measured
rule $dt \approx 172\,h_{min}^2$, ratio 0.343); steps to $t = 2.5$ are about
8,000–17,100; np8 is 0.97× np4's s/step (no rank speedup measured), so np8
production is 2.8–6.1 node-h direct-only and 21.6–47.5 node-h if it remeshes
every step (≤ 35 only at the short end), while np4 on 1 node fits 35 node-h
throughout; the raised edit caps and the remesh-interval scaling are
unmeasured (no launch remeshed). V1j — the pilot candidate
`L6_np4_eps1.6e-2` (`--eps 1.6e-2`, `--dt 3.2e-3`, `--min-dt 2e-4`) is stable
through step 60 at the same 0.343 ratio, 0.118 s/step np4; by the same rule
the pilot needs about 2,000–4,300 steps to $t = 2.5$ (`minEdge` 2.7e-3 to
$0.45 \times 4.1\times10^{-3}$), about 4–8.4 min of step cost at np4 before
remeshing.

## P1c review

**Verdict:** Passed
**Reviewed:** `b014788^..dea2ae9` (code: `scripts/tuolumne/large_rollup_probe_c.flux`
in `b014788`; `dea2ae9` is `tasks/` only)
**Evidence:** the implementing session's job, output read; no verification job,
since all four checks pass and the criterion is a measurement record whose
bound ($dt\,\mu/h_{min}^2 \le 0.41$) the baseline clears at 0.343.
- `f3j9ihiaeo7u`, `scripts/tuolumne/large_rollup_probe_c.flux`, log
  `/p/lustre5/stewartj/beatnik/large_rollup/P1c/large_rollup_P1c.f3j9ihiaeo7u.log`:
  COMPLETED, exit 0, queued 11.4 min in `pdebug`, ran 2.6 min (Flux
  accounting). Echoed HEAD `9e97da8`, porcelain 3, dev-env exe, `beatnik`
  `ckhvawf`. `git diff 01e874f dea2ae9 -- src examples tests CMakeLists.txt`
  is empty, so the binary is the code every P1c commit describes. The script
  was uncommitted at run time; each `<tag>/launch.log`'s first line echoes
  the full argument list, and all four match `b014788`'s `BASE` and
  `LAUNCHES` exactly.
- Four summary lines, all rc 0 (walls 40.5, 40.0, 35.6, 14.4 s); every
  launch reached `step=60` with no stop line. Each solver header confirms its
  eps (0.025, 0.008, 0.008, 0.016), `mu 0.002` and `min_dt`.
- Baseline direction: `L7_np4_eps0.025` step 60 dt 3.13e-4, `minEdge`
  0.00135, ratio 0.343 ≤ 0.41. Every table cell (V/F at steps 1 and 60,
  median s/step 0.443 / 0.440 / 0.428 / 0.118 recomputed from the line
  timestamps, setup 12.2 / 11.8 / 9.2 / 7.2 s, dt, `minQ`/`minEdge`/`volRel`,
  `max|S|`, peak `rss_kb`) matches the launch logs. The derived steps to
  $t = 2.5$ (about 7,990 and 17,170 by $dt = 172\,h^2$) and the node-hour
  table re-derive from the stated inputs.

1. [nit] The job ran with `large_rollup_probe_c.flux` uncommitted, so the
   echoed HEAD does not pin the script that ran (Conventions: provenance via
   the echoed commit). The per-launch argument echo shows the run matches
   `b014788`, so no conclusion is affected. Accepted.
2. [nit] The provenance line prints `jobid = unknown` because `FLUX_JOB_ID` is
   unset in the batch shell; the log records the ID. Accepted; later scripts
   could use `flux getattr jobid`.
3. [follow-up] eps `8e-3` is shown stable only to $t = 0.023$ (`max|S|`
   1.5e-2), and no launch remeshed, so the raised edit caps and remesh cost
   scaling with ranks are unmeasured. `## P1c` states both; D1 and the pilot
   (V1) carry them.
4. [follow-up] `## P1c` suggests D1 reconsider decision 5 (np8 on 2 nodes),
   since np8 is 0.97× np4's s/step at twice the node-hours. That is D1's
   decision with the user, not a finding here.

**Affects:** none beyond what `## P1c`'s **Affects:** already states (D1, V1j).
