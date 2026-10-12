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

## T0

**Inputs, pinned.** The four T9b FMM logs: `Fmm_HIP` (job `f3d7zP5no6pf`),
`Fmm_SERIAL` (`f3d7zNwy8tb9`), `FmmL4_HIP` (`f3d7zPLmZqxj`) and `FmmL4_SERIAL`
(`f3d7zPD9mXyy`). All four were built from Beatnik `0411c47` (12 files dirty)
and Canopy `bd10c8f`, branch `investigate-m2l-cap`, clean. The build was
`canopy +profiling` at level 1 and `beatnik profiling_level=2`; every
`print_*` table was printing. The configuration is CartesianTaylor, order 3,
`ncrit 8`, `mac_theta 0.3`, `max_depth 10`, 2000 steps, np1 and np4 in one
allocation per log. T0 itself had no build, no job and no queue cost; the
reducer reads a 37 MB log in about 0.5 s on the login node.

**Decisions.**
- **T0 owns the probe's row format.** `[cost]`, `[acc]` and `[cost-total]`
  rows are one line each: `[<tag>] key=value key=value ...`, whitespace
  separated, in the style of `[t6] COST`. Keys match `[A-Za-z_][A-Za-z0-9_.]*`
  and appear once per row. Values are numbers where they parse, else strings.
  A value that is not available is printed as `-1`, never omitted; it is
  excluded from statistics and counted as a sentinel. `run=` is required on
  every row, and `step=` on `[cost]` and `[acc]`. `run` groups one launch's
  rows, so a script that runs several configurations in one log gives each its
  own `run`. All other keys are read generically, so T2 can add fields without
  changing the reducer. Canopy breakdown blocks in a probe log
  (`canopy_report=1`) need `--format probe`. That mode parses and counts them
  but does not attribute them; without it they are unattributable and the
  reducer fails.
- **Block rules.** A block is the `solve() phase breakdown (N MPI rank[s])`
  header followed by the rows `Total solve`, `Upward sweep`, `Downward sweep`
  and `P2P`, in that order. Any other line inside a block is skipped. It is
  **truncated** when the next header, a launch boundary or EOF arrives first,
  which exits 1 naming the header's line. Other fatal errors: a block that
  cannot be attributed, a rank count different from the launch's, rows out of
  order, and `|Total − (Up + Down + P2P)| > 2.5e-3` s, which is the rounding
  of four `%.3f` values.

**Attribution, as the logs actually are.**
- Each claim-A state prints `claim A step S comparator exit`, then the
  standalone `fmm.computeInterfaceVelocity` runs. Each state's single block
  therefore follows its comparator note, and blocks are attributed to the most
  recent such note. There are 81 states with one block each.
- Claim B starts at the first `claim B step 0 relative drift` note. Block `k`
  (0-based) is step `k//3 + 1`, RK3 stage `k%3`.
- **The `+1` evaluation is the final probe:** the standalone `fmm_type probe`
  that `Beatnik_Test_Milestone0Fmm.cpp` builds on claim B's final state, after
  the loop, to read the realized P2P fraction. It is not a timestep, and it is
  excluded from claim-B sums. Hence 6082 = 81 + 3 × 2000 + 1.
- Each `claim B step S` note's first occurrence is cross-checked against
  `3S` blocks. All 80 notes per launch hold at both np. At np4 every rank
  prints each note, so boundaries are de-duplicated by (claim, step); a
  non-rank-0 copy never preceded rank 0's blocks in these logs.

**Verified.**
- Every log reduces to 12164 blocks, equal to `grep -c`, with 0 errors.
- The `FmmL4_HIP` np4 block with the `[H2SCR]` line at 344082, and the three
  intrusions in `Fmm_HIP`, reduce cleanly.
- Deleting block 3000's last two rows (scratch copy) exits 1:
  `trunc.log:126009: breakdown block truncated (next block header at line
  126047 before its four rows; got total, upward)`.
- Cutting the log at line 126015 exits 1 on the claim-B count (2919, not 6000
  or 6001).
- The Frozen logs reduce to 0 blocks, exit 0.
- `--self-test` covers the grammar's four rejection cases, the sentinel count,
  and the truncated, missing-evaluation and outside-launch block cases.

**Per-launch tables.** Times are seconds per `solve()`, as Canopy's rank-mean
print gives them. **P2P, and Upward at level 3, sit at the `%.3f` print
resolution (1 ms) and are not measurements at that size.** Claim A is the 81
direct-driven states; claim B excludes the final probe. The claim-A Upward
maxima of 0.150–0.317 s on HIP are each launch's first block, the cold first
evaluation; SERIAL has no such outlier.

| level | backend | np | claim | Total med / p90 / max (s) | Upward med / p90 / max | Downward med / p90 / max | P2P med / p90 / max | Downward share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | HIP | 1 | A | 0.036 / 0.053 / 0.217 | 0.001 / 0.001 / 0.194 | 0.034 / 0.050 / 0.064 | 0.001 / 0.001 / 0.001 | 88.2 % |
| 3 | HIP | 1 | B | 0.027 / 0.044 / 0.063 | 0.001 / 0.001 / 0.002 | 0.025 / 0.042 / 0.062 | 0.001 / 0.001 / 0.001 | 93.7 % |
| 3 | HIP | 4 | A | 0.024 / 0.032 / 0.225 | 0.002 / 0.002 / 0.192 | 0.018 / 0.025 / 0.031 | 0.004 / 0.006 / 0.006 | 68.0 % |
| 3 | HIP | 4 | B | 0.017 / 0.023 / 0.032 | 0.002 / 0.002 / 0.003 | 0.014 / 0.019 / 0.029 | 0.002 / 0.003 / 0.006 | 79.4 % |
| 3 | Serial | 1 | A | 0.034 / 0.054 / 0.069 | 0.001 / 0.001 / 0.001 | 0.031 / 0.051 / 0.066 | 0.002 / 0.002 / 0.002 | 90.8 % |
| 3 | Serial | 1 | B | 0.030 / 0.047 / 0.069 | 0.001 / 0.001 / 0.001 | 0.027 / 0.044 / 0.066 | 0.002 / 0.002 / 0.002 | 89.8 % |
| 3 | Serial | 4 | A | 0.017 / 0.025 / 0.030 | 0.001 / 0.002 / 0.002 | 0.015 / 0.023 / 0.028 | 0.001 / 0.001 / 0.001 | 88.7 % |
| 3 | Serial | 4 | B | 0.014 / 0.020 / 0.033 | 0.001 / 0.001 / 0.004 | 0.012 / 0.018 / 0.031 | 0.001 / 0.001 / 0.001 | 88.4 % |
| 4 | HIP | 1 | A | 0.331 / 0.371 / 0.627 | 0.002 / 0.002 / 0.150 | 0.327 / 0.367 / 0.623 | 0.002 / 0.003 / 0.003 | 98.1 % |
| 4 | HIP | 1 | B | 0.325 / 0.371 / 0.404 | 0.002 / 0.002 / 0.002 | 0.321 / 0.366 / 0.399 | 0.002 / 0.003 / 0.003 | 98.7 % |
| 4 | HIP | 4 | A | 0.196 / 0.221 / 0.576 | 0.007 / 0.008 / 0.317 | 0.180 / 0.205 / 0.566 | 0.006 / 0.010 / 0.011 | 90.8 % |
| 4 | HIP | 4 | B | 0.194 / 0.223 / 0.251 | 0.006 / 0.008 / 0.012 | 0.176 / 0.201 / 0.222 | 0.010 / 0.018 / 0.038 | 90.8 % |
| 4 | Serial | 1 | A | 0.422 / 0.483 / 0.498 | 0.003 / 0.004 / 0.004 | 0.404 / 0.465 / 0.480 | 0.015 / 0.016 / 0.017 | 95.6 % |
| 4 | Serial | 1 | B | 0.429 / 0.489 / 0.539 | 0.003 / 0.003 / 0.004 | 0.410 / 0.471 / 0.520 | 0.015 / 0.016 / 0.017 | 95.7 % |
| 4 | Serial | 4 | A | 0.199 / 0.222 / 0.241 | 0.005 / 0.007 / 0.008 | 0.191 / 0.213 / 0.230 | 0.004 / 0.005 / 0.005 | 95.3 % |
| 4 | Serial | 4 | B | 0.223 / 0.245 / 0.270 | 0.005 / 0.007 / 0.013 | 0.213 / 0.235 / 0.258 | 0.004 / 0.005 / 0.006 | 95.4 % |

**Share of a claim-B timestep inside `solve()`**: summed claim-B `solve()`
divided by the `[t6] COST claimB=` wall. That wall also holds `auto_maintain`,
the adapter's distribute and migrate phases, the non-BR step work, 80 volume
checks, the checkpoints, and every level-1 Canopy table print with its
`MPI_Reduce`.

| level | backend | np | claim B summed `solve()` (s) | `[t6] COST claimB` (s) | ratio |
| --- | --- | --- | --- | --- | --- |
| 3 | HIP | 1 | 170.0 | 222.4 | 0.765 |
| 3 | HIP | 4 | 101.9 | 215.1 | 0.474 |
| 3 | Serial | 1 | 181.1 | 2789.0 | 0.065 |
| 3 | Serial | 4 | 84.7 | 6392.4 | 0.013 |
| 4 | HIP | 1 | 1753.5 | 1966.0 | 0.892 |
| 4 | HIP | 4 | 1067.5 | 1361.1 | 0.784 |
| 4 | Serial | 1 | 2369.0 | 4905.7 | 0.483 |
| 4 | Serial | 4 | 1210.0 | 8162.1 | 0.148 |

**Trend over claim B.** Each cell is summed `solve()` per step (s/step) in a
250-step bin (`--bin-steps 250`; 25-step bins are available from the same
command), with that bin's downward share of summed `solve()` in parentheses.

| level | backend | np | 1-250 | 251-500 | 501-750 | 751-1000 | 1001-1250 | 1251-1500 | 1501-1750 | 1751-2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | HIP | 1 | 0.039 (90.6 %) | 0.108 (95.5 %) | 0.075 (92.9 %) | 0.063 (91.4 %) | 0.079 (92.8 %) | 0.110 (94.6 %) | 0.110 (94.8 %) | 0.097 (93.8 %) |
| 3 | HIP | 4 | 0.029 (74.1 %) | 0.059 (83.9 %) | 0.046 (78.4 %) | 0.041 (75.5 %) | 0.050 (78.9 %) | 0.063 (81.1 %) | 0.062 (79.8 %) | 0.058 (79.1 %) |
| 3 | Serial | 1 | 0.044 (80.9 %) | 0.117 (92.7 %) | 0.083 (89.0 %) | 0.066 (85.9 %) | 0.083 (88.7 %) | 0.115 (91.6 %) | 0.115 (91.6 %) | 0.101 (90.7 %) |
| 3 | Serial | 4 | 0.024 (84.5 %) | 0.052 (91.8 %) | 0.037 (86.8 %) | 0.032 (84.7 %) | 0.040 (87.4 %) | 0.054 (89.7 %) | 0.051 (89.6 %) | 0.047 (88.7 %) |
| 4 | HIP | 1 | 0.347 (97.6 %) | 0.653 (98.4 %) | 0.812 (98.6 %) | 0.974 (98.8 %) | 1.073 (98.8 %) | 1.096 (98.8 %) | 1.031 (98.8 %) | 1.027 (98.7 %) |
| 4 | HIP | 4 | 0.268 (91.7 %) | 0.406 (92.1 %) | 0.492 (91.4 %) | 0.553 (91.1 %) | 0.651 (91.0 %) | 0.640 (90.9 %) | 0.624 (90.0 %) | 0.635 (89.3 %) |
| 4 | Serial | 1 | 0.573 (92.9 %) | 0.927 (95.8 %) | 1.093 (95.5 %) | 1.257 (95.5 %) | 1.395 (96.0 %) | 1.456 (96.2 %) | 1.383 (95.8 %) | 1.393 (96.0 %) |
| 4 | Serial | 4 | 0.324 (94.0 %) | 0.470 (94.7 %) | 0.565 (94.9 %) | 0.643 (95.1 %) | 0.735 (95.4 %) | 0.717 (96.0 %) | 0.692 (95.9 %) | 0.694 (95.8 %) |

**Does the downward share grow with step count?**
- **At level 4, the downward *time* grows and the share barely moves.**
  - At HIP np1, the median Downward rises from 0.103 to 0.362 s per
    evaluation, 3.5x, from bin 1-250 to bin 1251-1500. Upward stays at
    0.001–0.002 s. The share goes from 97.6 % to 98.8 %; it starts near
    saturation, so the share cannot show the growth.
  - At HIP np4 the share falls slightly, from 91.7 % to 89.3 %, because P2P
    grows from 0.003 to 0.017 s while Downward grows 2.4x.
  - Per-step `solve()` grows 3.2x (np1), 2.4x (np4), 2.5x (SERIAL np1) and
    2.3x (SERIAL np4). It plateaus from about step 1000, matching the
    occupied-depth-7 onset.
- **At level 3 the growth is not monotone.** It peaks at 251-500, dips at
  751-1000 and rises again.
- **This is consistent with, but does not isolate, the operator-rebuild
  prediction.** A rebuild-dominated Downward should grow with the number of
  unique operators, which grows with tree depth, and so should the M2L kernel.
  Level-1 timers cannot separate `ilist_s4_op_table_build` from the M2L
  kernel. T3's level-2 split is what decides it.

**What only reading the logs revealed.**
- **On SERIAL, most of a claim-B step is outside `solve()`.** The share inside
  `solve()` is 6.5 % at level 3 np1, 1.3 % at level 3 np4, 48 % at level 4 np1
  and 15 % at level 4 np4. HIP's share is 47–89 %.
  - SERIAL np4's claim-B wall exceeds np1's (2.29x at level 3, 1.66x at
    level 4), while its `solve()` sum is *lower* (0.47x, 0.51x).
  - **The SERIAL rank-scaling problem is therefore not in `solve()`.** It is
    in `auto_maintain`, the adapter phases or the non-BR step: one or more of
    the parts the level-1 `solve()` breakdown cannot see.
- **HIP np4 also loses about half of a level-3 step outside `solve()`**:
  101.9 s inside against 215.1 s wall. The same holds, smaller, at level 4
  (0.78).
- **The level-1 reporting cost is inside every one of these walls.** Each
  evaluation prints the UpwardSweep, DownwardSweep, P2P and `solve()` tables,
  each behind an `MPI_Reduce`, plus about 2.5 M2L operator-table lines (30 410
  per log); about 37 MB per log. T2's `canopy_report` ratio is what bounds this part of
  the outside-`solve()` share.

**Affects:**
- T2 — the probe must emit [cost]/[acc]/[cost-total] in T0's key=value
  grammar, with field names matching what the reducer reads.
- T2 — `run=` is required on every row, and `step=` on `[cost]` and `[acc]`;
  missing values print `-1`.
- T2 — a `canopy_report=1` launch's log is read with `--format probe`.
- T3 — the per-row maintenance-count check (counts = 3 × steps) is not in the
  reducer yet; T3 adds it once T2 fixes the field names.
- T3 and O5 — the outside-`solve()` share is large: 11–53 % on HIP, 52–99 %
  on SERIAL. The `ff_maintain` split and the adapter phases need the same
  weight as `ff_solve` when ranking.
- O7 — SERIAL np4's slowdown is entirely outside `solve()`: its `solve()` sum
  is about half np1's.
