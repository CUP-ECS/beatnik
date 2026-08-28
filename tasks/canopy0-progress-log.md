# Canopy as the far-field engine for a distributed vortex-sheet solver — progress log

Session record for canopy. Companion to `canopy.md`, which holds the design, the
task sequence and the risks; this file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `canopy.md` can cite it by ID.
No dates: the order of the sections is the chronology. If a session covers more
than one task, name them all; if it belongs to no task, name the topic.

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

Every accuracy number written here must carry the qualification list the
design's conventions table requires: the source distribution, the rank counts,
`P_ORDER`, `ncrit`, `max_depth`, `mac_theta` and the softening it was measured
at. A bare tolerance is not a measurement.

## Topic: second read-only pass — treecode comparison merged into the design

No task was started and no code was built or run. A second read-only pass opened
`src/Canopy_LaplaceKernel.hpp` in full (887 lines) alongside a reference
Barnes–Hut treecode (`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`,
138 lines) to answer the question **C1** had left open under *Additional
information needed*: whether a softened far field is reachable without replacing
the expansion basis. The findings were written up separately first
(`tasks/treecode-vs-canopy.md`) and have now been merged into `canopy.md` as
**F6**, **F7**, **F8**, F4's exact-node-radius note, C1's revised four-option
step 3, **C11**, **R7** and **R8**. `tasks/treecode-vs-canopy.md` is retained as
the provenance record of that pass; `canopy.md` is now the authoritative
statement.

What the pass changed in the design's substance, as opposed to adding to it:

- **A premise in the original framing was wrong.** M2M/M2L/L2L/L2P are *not*
  missing from Canopy's kernel — all five operators are implemented in the
  solid-harmonic basis with Greengard theorem citations
  (`src/Canopy_LaplaceKernel.hpp:230`, `:273`, `:373`, `:688`, `:800`, plus the
  precomputed-operator path at `:516`). What is missing is softening *inside*
  them. F1's conclusion is unchanged; only the diagnosis of why is.
- **C1's open question is answered: no.** The solid-harmonic addition theorems
  hold because $1/r$ is harmonic, and $(r^2+b)^{-1/2}$ is not
  ($\nabla^2\phi_b = -3b/(r^2+b)^{5/2}$). There is no coefficient substitution
  that makes the existing M2L evaluate the softened kernel; a softened far field
  is a second expansion basis, not a patch. Recorded as F6(a).
- **The cost of that basis was underestimated in C1's original framing.** Beyond
  the second basis, softening breaks the scale invariance every operator's width
  normalization rests on, and re-keys the precomputed M2L operator cache — whose
  documented property is that the operator *"depends only on (dd, ii, jj, kk); no
  physical width enters"* (`src/Canopy_DownwardSweep.hpp:1075`), true only for a
  kernel with no absolute length scale. F6(c).
- **A cheaper option existed that C1 did not list.** A softened M2P mode reusing
  the existing tree, partition, interaction list and P2P — dropping M2L/L2L/L2P
  from the path rather than softening them — removes the bias entirely at
  materially less work. It is now C1 option (d), and F8 holds the analysis. Its
  viability rests on one unmeasured number (per-target M2P vs. per-cell
  M2L+L2L+L2P), which is why it is a measurement in C1 step 3 and a risk (R7),
  not a recommendation.
- **The accuracy landscape collapsed to one number.** The status quo's honest
  claim and the M2P ceiling are both $\sim\!10^{-3}$. So the choice between
  them is about the *kind* of error (fixed bias vs. truncation controllable by
  `mac_theta` and order), not its size. $10^{-6}$ is out of reach for a
  low-order softened far field too, which makes "what does the consumer actually
  need" the cheapest open question on the list (R8).
- **Canopy's far-field gradient is a finite difference**, not analytic:
  six extra potential evaluations at $h = 10^{-5}w_{\rm self}$ with an in-code
  `TODO` (`src/Canopy_LaplaceKernel.hpp:851-879`). It contributes a third,
  $\sim\!10^{-10}$ plateau to the `P_ORDER` scan C1 step 1 and R1 are built
  around. F7, task C11.

Also: every `\f$`/`\f[` in `canopy.md` was converted to KaTeX `$`/`$$` per the
repository's *Math in markdown* rule. The Doxygen delimiters had been rendering
as prose in a plain markdown reader, with CommonMark stripping the backslash from
every escaped punctuation character inside them.

Nothing here is a measurement. Every accuracy figure in F6/F8 is an estimate or
a number carried from `tasks/treecode.md`; the qualification list the conventions
table requires cannot be supplied for any of them, and no tolerance should be set
from them.

**Affects:** **C1** — its option set, its *Additional information needed* (now
narrowed to the Dehnen 2000/2002 literature check) and its exit criterion all
changed; do not start it from the pre-merge text. **C5** — gains step 5, the
exact-node-radius measurement. **C8** — a `Gradient`-only far field cannot skip
the potential internally until C11 lands. **C11** — new, and worth doing before
C1 step 1's scan is interpreted. **R1** — the scan has three plateaus, not two.
