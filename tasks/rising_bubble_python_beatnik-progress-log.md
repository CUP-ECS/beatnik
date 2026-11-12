# Rising bubble — Beatnik against the Python reference — progress log

Session record for rising_bubble_python_beatnik. Companion to
`rising_bubble_python_beatnik.md`, which holds the design, the task sequence
and the risks; this file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `rising_bubble_python_beatnik.md`
can cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task,
name the topic.

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

## Design measurements

**Commits:** none (design only).

Python reference kernel cost, one tuolumne login-node core,
`/usr/tce/bin/python3` (numpy 2.1.2), random points, blob `1.6e-3`:

| N | direct velocity (`_source_velocity_direct_unsigned`) | direct Riesz scalar | treecode velocity (θ 0.3, order 2, ncrit 64, points on the r = 0.25 sphere) |
| --- | --- | --- | --- |
| 2,000 | 0.40 s | 0.17 s | — |
| 6,500 | 5.86 s | 2.46 s | 1.09 s |
| 13,000 | — | — | 2.60 s |

The Riesz scalar is not evaluated under `--bernoulli-scalar-mode
normal-speed` (the default), so a TVD-RK3 step costs three velocity
evaluations: about 18 s/step direct at 6,500 vertices before remeshing
overhead. The reference's whole-step cost at 600 vertices, direct, is about
1.1 s/step (`tasks/large_rollup.md`, Out of scope), mostly remeshing overhead
at that size. These give the 10–25 h estimate for a direct run to `t = 3.45`
and the 2–5 h estimate for the treecode run.

Flux queue limits read with `flux queue list`: pdebug 1 h maximum (16 nodes),
pbatch 24 h.

No production job was live (`flux jobs -a --since=-3d`: all
`large_rollup_segment` jobs `CD`). The production checkout is at `4b33843` on
`large-rollup`; `git diff --stat 4b33843..de299e2 -- src examples` is empty.

**Affects:** T1 — the production install already carries current `src/`;
T3j — the Python walltimes (300m direct, 240m treecode) come from these numbers.

## Production env reinstall

**Commits:** none (the `systems/tuolumne/spack-production.yaml` snapshot update
is in the working tree, uncommitted).

The live production `spack.yaml` carried `beatnik … profiling_level=2`, the
dev setting; systems doc §1 makes production `profiling_level=1`. It was
changed to 1 (canopy was already `~profiling`). The committed snapshot
`spack-production.yaml` was stale (kokkos@4, the pre-17.1 trilinos spec, no
tessera, no `gsl` guard) and now equals the live file; it differs from
`spack.yaml` only in the beatnik and canopy profiling settings.

`flux jobs` showed no active job immediately before the install. The
production beatnik checkout was fast-forwarded `4b33843 → de299e2` from the
local dev checkout (the 18 commits are not on origin). `spack concretize -f &&
spack install` with `HIPCC_*_FLAGS_APPEND` unset exited 0, rebuilding only
beatnik (`inrvs7n`, 12m27s); the view's `adaptive_mesh_bubble` links to it.

**Affects:** T1 — Do 1 finds the install current; stage `inrvs7n`.
