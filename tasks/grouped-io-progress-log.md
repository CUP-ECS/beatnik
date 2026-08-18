# Grouped HDF5/XDMF checkpoint output — progress log

Session record for grouped-io. Companion to
[tasks/grouped-io.md](grouped-io.md), which holds the design, the task sequence and
the risks; this file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states flatly,
the measured numbers behind a claim, or the history of a file you are about to change.
The design says *what is true now*; the log says *how it got that way and what was
tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a signature,
measures something, or finds a bug. Add a new `## <task ID>` section at the bottom,
named for the task it records, so `grouped-io.md` can cite it by ID. No dates: the
order of the sections is the chronology. If a session covers more than one task, name
them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan this
entry changes, one clause each on how, or `none`. A finding that invalidates a later
task is worthless if the session starting that task has to read the whole log to
notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards: semantic
decisions and what forced them, signature changes and why they could not stay as they
were, bugs that only running revealed, measured numbers, and approaches tried that did
not work. Record too where the implementation departed from the task's stated **Do**
steps, and why — a task marked `**DONE**` that was done differently than it was
written is the quietest way for a design to stop describing the code.

## T1

**Session-level decision recorded on request, before any code was written: T3 does not
re-run the ship gate.** This work adds light data only and touches no `.h5` dataset, so
the 60-launch `regression` sweep was deliberately not run; correctness of frame content
is verified separately, outside this session. T3's exit criterion as written in
[grouped-io.md](grouped-io.md) still names 60/60 green, and that half of it is
**superseded** — CLAUDE.md's "full sweep as of T4c" stamp is left alone rather than
restamped as if a sweep had happened. Everything else in that criterion still applies.
Also decided: one commit per task, three total, staged by explicit path and pushed to
`origin/rising-bubble-redesign`.

**R5 was live, not hypothetical.** Before anything else,
`ls $(spack location -i tessera)/include/Tessera_XdmfSeries.hpp` returned ENOENT: the
installed Tessera prefix predated `2ba15cd` even though the `../tessera` worktree was at
it. `spack install tessera` (40s) fixed it. Had Beatnik been built first, the failure
would have read as a Beatnik include bug. The check is worth keeping in that order.

**Implemented exactly as the T1 Do steps say**, with no deviations: the private static
`masterStem( directory, prefix )`, `_series` declared after `_prefix` and initialized
from it in the member-init list, `_last_frame_time` / `_last_frame_stem` commented as
meaningful only when `_series.numFrames() > 0`, and the three-way decision replacing
step 2 of `write()` with `time` converted once into a `double` local. Steps 1, 3, 4 and
5 of `write()` are untouched. No public signature changed; no file outside
`src/Beatnik_IOInterface.hpp` was edited for the code change.

**Only running revealed how much of the batch-script cost is not the I/O change.** Five
submissions failed before the passing one, and the useful part is *why*, because a T2 or
T3 session writing a new script will hit the same walls:

1. `f3T3V6SDq8JB` — `# flux: --time=5` in the header. `flux batch` rejects `--time` at
   argument-parsing time; `-t 5m` is the form. This is trap 1 in
   `systems/tuolumne/claude.md` §5 and the template still carries the invalid form at
   line 5, which is how it got copied.
2. `f3T3cdLxTFvf` — the template's `BASH_SOURCE`-based `BEATNIK_REPO` fallback resolved
   to the `/var/tmp` spool `flux batch` copies the script into, so the resolver source
   failed with `/var/tmp/scripts/lib/beatnik_env.sh: No such file`. Trap 2 in the same
   section. Fixed by lifting `beatnik_find_repo()` verbatim from
   `scripts/tuolumne/unit_tests.flux:67-86`, which walks up from `PWD`.
3. `f3T3dQg57dWF`, `f3T3eARVtREw`, `f3T3etQ7s1Yj` — three successive *unimplemented*
   solver passes reached by the example's own defaults, each aborting before step 1 so
   only the startup checkpoint was ever written: `MeshQuality::isotropicCleanup` (T4d),
   `DynamicRemesh::collapseShortEdges` (T4d, blocked on Tessera G5b),
   `BRSolverFMM::computeInterfaceVelocity`, then `FaceQuadrature::generate`. The fix is
   not a weakened run: the script now passes exactly T4b's split-only configuration
   (`--source-quadrature vertex --br-approximation direct --remesh-collapse-factor 0
   --remesh-smooth-iters 0 --remesh-flip-min-gain 1e12 --no-isotropic-cleanup`, from
   `tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:179-186`), none of which
   touches the I/O path. **`adaptive_mesh_bubble` at its documented defaults cannot
   complete a single step in this checkout** — worth knowing before writing any future
   example-driving script, and the reason is recorded in the script's own header comment.

**Measured, job `f3T3fdHDWBQT` (4 ranks, 1 node, the template binding verbatim):**
`CollectionType="Temporal"` = 1, `<Time Value=` = 3, distinct frame `.h5` = 3, and
`checkpoint.xmfindex` = 3 lines. The three master times are `0`,
`0.0045000000000015002` and `0.0074999581581878952` — note the third is **not** exactly
`0.0075`: adaptive `dt` lands the last step slightly short, while `timeKey()`'s six
decimals round it to `00000p007500`. So the `.xmfindex`'s `%.17g` time and the stem's
time key genuinely disagree in the last digits, by construction and harmlessly. A future
series-reopen (T5b) must key on the **stem**, not on re-deriving a stem from the
recorded time.

**The equal-time branch fired and is what the run proves.** `finalize()` re-wrote
`(t=0.0074999581581878952, step 4)` — the same stem as the step-4 checkpoint — and the
master gained no fourth timestep and the `.xmfindex` no fourth line. R4's cost is
visible but immaterial at this size: the master grew 2560 -> 7430 bytes over three
frames.

**Affects:** T2 — the test must construct `CheckpointIO` directly (it already planned
to), which sidesteps every one of the five failures above; but its scratch directory
must be cleared per run for the same reason the T1 script does it, since a leftover
`.xmfindex` from a previous run is *appended to* and would make a `<Time Value=` count
unreadable. T3 — the gate re-run in its step 4 is superseded by the decision above, and
its README subsection should state the `Xdmf3ReaderT` requirement and the
`grep -c '<Time Value='` check, both of which this run exercised.
