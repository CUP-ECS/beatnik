# An abstract far-field backend for Canopy — progress log

Session record for abstract-solver-backend. Companion to
[abstract-solver-backend.md](abstract-solver-backend.md), which holds the
design, the task sequence and the risks; this file holds what actually
happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `abstract-solver-backend.md`
can cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task, name
the topic.

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

Three things this project in particular will want back later, so record them
where they arise:

- **The commit the golden reference data was generated at** (T1), and every
  later regeneration with the reason for it. Reference data whose provenance is
  unknown cannot be trusted to mean anything.
- **The profiling breakdown before and after T3 and T4** (R3). A `constexpr`
  that quietly became a runtime value has no correctness signal at all, and
  these two numbers are the only record that it did not happen.
- **The measured `n_unique_ops` and `n_unique_ops * bytes_per_key`** from T8's
  instrumentation (R7). The current figure is a claim in a tuning comment, not a
  measurement, and it is the number most likely to decide whether a
  compressed-operator basis is worth building at all.

(No entries yet.)
