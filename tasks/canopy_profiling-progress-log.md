# Canopy profiling and optimization — progress log

Session record for canopy_profiling. Companion to
[`canopy_profiling.md`](canopy_profiling.md), which holds the design, the task
sequence and the risks; this file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `canopy_profiling.md` can cite
it by ID. No dates: the order of the sections is the chronology. If a session
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

Two things this programme specifically wants in every measuring entry, because
its numbers are only comparable when they are pinned:

- the **Canopy commit hash** the measured binary was built from (and that
  `git -C ../canopy status --porcelain` was empty), plus the flux job IDs;
- the **(level, np, backend, steps, `sample_every`, FmmParams overrides)** of
  every row quoted, and the build-and-queue cost of the task (install wall,
  queue wait, job wall), so later tasks can budget from measurements rather
  than estimates.

(No entries yet.)
