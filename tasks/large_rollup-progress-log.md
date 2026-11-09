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

(No entries yet.)
