# Canopy M2L operator-key demand: measurement, then the cap — progress log

Session record for `add-canopy-t6`. Companion to `add-canopy-t6.md`, which holds
the design, the task sequence and the risks; this file holds what actually
happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `add-canopy-t6.md` can cite it
by ID. No dates: the order of the sections is the chronology. If a session
covers more than one task, name them all; if it belongs to no task, name the
topic.

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

This topic's measurements are the point of it, so two are worth calling out as
the ones every later reader will come back for: **T5**, the level-4 demand
series, and **T9a**, the first level-4 claim-A error measured at zero fallback.
Record both at full precision with their step numbers, rank counts and
backends — a demand figure without its rank count is unreadable, because the key
set is per rank.

Note also that T6's own tier-run result is logged in
`tasks/canopy/add-canopy-progress-log.md` (task T0 here writes it), not in this
file. This log starts from the instrumentation work.

(No entries yet.)
