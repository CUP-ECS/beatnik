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

## T1

**Canopy commits.** `aa56e64` (the switch, the test case, the README
subsection), `2dc4072` (the test job script) and `c182348` (two more gated
prints, below), all on `profiling-canopy`, not pushed.

**Signatures added** (`../canopy/src/Canopy_Profiling.hpp`, outside the
`#ifdef CANOPY_ENABLE_PROFILING` block, so they exist in every build):
`inline bool Canopy::Profiling::reporting_enabled()` and
`inline void Canopy::Profiling::set_reporting( bool )`. Default on. In a
build without profiling `set_reporting` is ignored and `reporting_enabled()`
returns `false`. The declaration comment states the collective precondition
(identical on every rank; a mismatch deadlocks and is not detected).

**Gated.** `print_solve_breakdown` and `print_timing_table` return before their
`MPI_Reduce`, on every rank, so every `CANOPY_PRINT_*` macro is silenced. The
three `auto_maintain` `[Canopy diag]` lines and the per-rank M2L operator-table
line are gated as the task says.

**Departure: three more prints gated than the task named.** All are rank-0
`fprintf`s with no collective around them:
- `TreePartitioner::partition_cells`'s `[Canopy diag] partition` line, with its
  `[H2SCR]` scratch line inside the same block. That is 4719 lines each in the
  T9b L4 HIP log. Gated in `aa56e64`; `Canopy_TreePartitioner.hpp` now
  includes `Canopy_Profiling.hpp`.
- `TreePartitioner::refresh_ownership`'s `[Canopy diag] refresh_ownership`
  line, and P2P's level-3 `[Canopy diag] p2p_inter_leaf_pairs` line. Both were
  gated in `c182348`. Only running revealed them: the first test job
  (`f3diBYPMfS4X`) failed `LaplaceSolve.reportingSwitch` at np 1–6 on one
  `refresh_ownership` line from rank 0. The first smoke job (`f3diBaGUea6b`)
  showed 15–24 of the same lines per `canopy_report=0` launch.

After `c182348`, no `[Canopy diag]` or `[Canopy Diagnostics]` print in `src/`
is ungated.

**The test case.** `LaplaceSolve.reportingSwitch` in `tests/tstLaplaceSolve.hpp`
does the following:
- Builds the `with_laplace_solve` particle set and runs one `setup()` and one
  `solve()` with reporting off.
- Captures stdout and stderr with gtest's `CaptureStdout` / `CaptureStderr`.
- Asserts zero `[Canopy Diagnostics]` and zero `[Canopy diag]` lines on every
  rank, and that `timer_totals()` gained `downward_total` (and that it is
  > 0).
- Restores the flag through an RAII guard.

It checks `[Canopy diag]` as well as `[Canopy Diagnostics]`, which is stricter
than the exit criterion; that is what caught the `refresh_ownership` line.

**Verified** (job `f3diM2F6p3YB`, Canopy `c182348`, clean, `build-profiling/`
at `Canopy_PROFILING_LEVEL=2`, queue wait 0 s, wall 6.8 min):
- `Canopy_Test_{LaplaceSolve,CartesianTaylorSolve}_MPI_SERIAL_np_[1-6]` all
  completed through `canopy_ctest` against the existing budget rows.
- The new case printed `canopy_diagnostics_lines=0 canopy_diag_lines=0` on all
  21 ranks of np 1–6, with `downward_total` 0.015–0.036 s.
- The existing cases still print: 8917 `[Canopy Diagnostics]` lines in the
  log.
- **Failure direction**, run once in the same job: a LaplaceSolve binary built
  from `aa56e64` with the `print_solve_breakdown` gate deleted
  (`tests/Canopy_Test_LaplaceSolve_MPI_SERIAL.nogate`, built by hand, never
  committed) fails `reportingSwitch` at np1. Its captured output carries the
  `[Canopy Diagnostics] solve() phase breakdown (1 MPI rank)` block. The
  gate is in the committed source.

Each Canopy stem build was about 160 s (`make -j 32`, both stems).

**Affects:** none.

## T2

**Beatnik signatures changed** (uncommitted when measured; committed with this
entry):
- `src/Profiling.hpp`: the eleven stale `FmmBRSolver` keys are replaced by
  `TIMER_FF_EVALUATE_TOTAL` and the ten phase keys `TIMER_FF_{PACK,
  FWD_DISTRIBUTE, FWD_MIGRATE, MAINTAIN, SOLVE, CONTRACT, REV_DISTRIBUTE,
  REV_MIGRATE, SCATTER, DIAGNOSTICS}` (strings `ff_*`).
  - New: a never-reset `timer_totals()` and `reset_timer_totals()`;
    `accumulate` feeds both.
  - `fmm_phase_entries()` and `BEATNIK_PRINT_FMM_TIMERS` now name the new
    phases; nothing calls them.
- `FarFieldDiagnostics`: `static constexpr int num_maintenance = 4`,
  `long long evaluations`, and `long long
  maintenance_counts[num_maintenance]`.
  - They are cumulative over the solver's life and indexed by
    `static_cast<int>( Maintenance )`.
  - They are incremented at the end of a completed `evaluate`.
- `Solver::brSolver() const` returns `const br_solver_type&`.
- `Beatnik_Probe_FmmCost` (new), in `BEATNIK_DRIVER_SOURCES`.

**Where each phase's timer sits in `evaluate`:**
- `ff_maintain` includes `createSolver` on the first call, as well as
  `setup` / `autoMaintain`.
- `ff_fwd_migrate` includes the setup path's `resize` + `deep_copy`.
- `ff_fwd_distribute` includes the `MPI_LAND` reduce.
- `ff_diagnostics` covers everything from `readDiagnostics` to the end.

The forward-distributor and reverse-migrate scopes are immediately-invoked
lambdas, the idiom the reverse distributor already used.

**Probe decisions.**
- **A sixth key, `run=<label>`.** The default is
  `L<level>_<space>_np<N>_ncrit<n>_order<p>_rep<r>`. The smoke script passes
  its own label.
- **Max and mean.** Rank-local time and count fields print as `<key>` (mean
  over ranks) and `<key>.max`.
- **Last evaluation of the window.** The global fields of `[cost]`
  (`fb_range_guard`, `fb_count_cap`, `fb_dropped`, `m2l_pairs`,
  `m2l_fallback`, `p2p_frac`, `particles`) are the window's last evaluation,
  since `FarFieldDiagnostics` holds only the last one.
- **`direct_step_est` uses maxima over ranks.** It is `3 * direct_wall +
  (last step wall − last step's ff_evaluate_total)`, with each term the max
  over ranks.
- **Extra Canopy keys.** `builder_build` and `sort_by_leaf` were added to the
  task's list, because T3 has to split `ff_maintain` into tree build,
  partition, comm plan and migrate.
- **Sentinels.** Canopy keys timed only by `CANOPY_SCOPED_TIMER_DETAILED`
  (`dn_build_ilist`, `ilist_*`, `commplan_dtt_traversal`) print `-1` below
  Canopy level 2.
- **A non-row header line.** One `[fmmcost] header` line per launch records
  the configuration. The reducer ignores it.
- **Scheduler.** `flux batch --flags=waitable` is refused on this machine
  ("only the instance owner can submit with FLUX_JOB_WAITABLE"); submit plain
  and wait with `flux job status`.

**README.** The user approved the "Future Optimizations" → "Higher profiling
levels" line. The existing entry was stale, since level 2 is now in use, so it
was rewritten: level 2 times the ten `ff_*` phases; level 1 (an unfenced
per-evaluation total) and level 3 (per-rank counts per phase) remain the
opportunity.

**Env.** `canopy@develop +profiling profiling_level=2` in
`~/spack_envs/tuolumne_beatnik/spack.yaml:16` and the snapshot
`systems/tuolumne/spack.yaml`. `systems/tuolumne/claude.md` §1 says so.
Canopy's `profiling_level` reaches Beatnik's translation units through the
exported `Canopy` INTERFACE target (`../canopy/src/CMakeLists.txt:29-31`).

**Departure: two installs, not one.**
- The first (`spack concretize -f` + `spack install`, 662 s wall; Beatnik
  10 min 25 s) built Canopy `aa56e64`.
- The `refresh_ownership` line (T1 above) failed the strict diagnostic checks
  in both first jobs, so Canopy `c182348` needed a second `spack install`
  (576 s; Beatnik 9 min 14 s).
- Both were real rebuilds, not header no-ops. The installed Canopy headers
  were checked for the new gates before submitting.

**Verified** (job `f3diM2PdyufM`, `pdebug`, one node, Beatnik `93b8847` plus
this task's uncommitted edits, Canopy `c182348` clean, queue wait 2 s, job wall
2.75 min, launch walls 13–37 s). Binding was `t6b_key_demand.flux`'s:
`--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1`, one node at
np 1 and 4. HIP, CartesianTaylor order 3, `ncrit 8`, `mac_theta 0.3`,
`m2l_op_count_cap 65536`.
- **`bogus=1`** exits 1 with `unknown key 'bogus'`.
- **All 20 `[cost]` rows** (L4 np1 ×2 launches at steps 1/20/40/60, and
  L5/L6 at np1 and np4 at steps 1/5/10) pass every check:
  - no `-1` in any field;
  - the ten phases sum to `ff_evaluate_total` within 0.28 % (worst row);
  - `ff_solve` is within 0.24 % of `upward_total + downward_total +
    p2p_total`;
  - the maintenance counts equal `evals` equal 3 × `steps`.
- **`[acc]` at L4**: `acc_rel` 5.01e-4 to 5.07e-4 at all 8 rows.
- **Canopy lines.** Zero `[Canopy Diagnostics]` and zero `[Canopy diag]` lines
  in all five `canopy_report=0` launches. 1290 and 200 in the
  `canopy_report=1` launch.
- **Reduction.** `fmm_profile_reduce.py --format probe` reduces the log.

**Measured.** These are early states only (steps 1–60 at L4, 1–10 at L5/L6).
Per-step cost grows along the trajectory (T0, R3), so none of these numbers is
a long-run figure.

| run | s/step (wall/steps) | `ff_evaluate_total` / step wall | `ff_solve` / eval | `ff_maintain` / eval | op-build share of `ff_solve` | actions (setup/migrate/rebalance) | `[acc]` rel | direct est s/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4 np1 rep0 | 0.310 | 0.991 | 0.092 | 0.0086 | 0.579 | 1/166/13 | 5.07e-4 | 0.0087–0.0108 |
| L5 np1 | 0.830 | 0.995 | 0.228 | 0.024 | 0.485 | 1/18/11 | 1.62e-3 | 0.028–0.029 |
| L5 np4 | 0.685 | 0.987 | 0.178 | 0.019 | 0.403 | 1/18/11 | 1.62e-3 | 0.036–0.043 |
| L6 np1 | 2.907 | 0.998 | 0.651 | 0.388 | 0.367 | 1/9/20 | 2.71e-3 | 0.108–0.110 |
| L6 np4 | 1.837 | 0.994 | 0.439 | 0.192 | 0.418 | 1/9/20 | 2.71e-3 | 0.114–0.120 |

(`ff_solve` and `ff_maintain` are the last window's per-evaluation means. The
op-build share and the actions are run totals.)

- **Reporting overhead (R1).** The `canopy_report=1` / `=0` step-wall ratio at
  L4 np1 is 1.018 on the run totals (18.957 / 18.628 s), and 1.020 in the last
  window.
- **The operator table is rebuilt at every evaluation.** At L4, every 20-step
  window has `keys_built_delta` = 449880 = 60 × `cache_size` 7498. At L5 np1 it
  is 208080 = 15 × 13872, and at L6 np1 435960 = 15 × 29064.
  - `ilist_s4_op_table_build` is 37–58 % of `ff_solve`.
  - `dn_build_ilist` is about 93 % of `downward_total` at L4 (0.084 of
    0.090 s per eval). `m2l_kernel` is only 4 ms.
  - The prior is confirmed on Beatnik's tree.
- **`ff_evaluate_total` is 98.7–99.8 % of step wall.** The non-BR step work is
  negligible on HIP at these sizes.
- **Maintenance grows with N.** `ff_maintain` is 8.5 % of eval time at L4,
  14–16 % at L5 and 24–30 % at L6. At L6, rebalance is 20 of 30 actions.
  `comm_plan_build` (0.13–0.29 s/eval) dominates it, and
  `commplan_dtt_traversal` is 0.10–0.19 s of that.
- **Range-guard fallback first appears at L6**: 3328–3360 pairs, with
  `fb_count_cap = 0`. There is none at L4 or L5 in these windows.
- **Adapter phases are small but grow with ranks.** Everything outside
  `ff_solve` and `ff_maintain` is 1.4–3.2 % of `ff_evaluate_total` at np1 and
  2.7–4.7 % at np4. The worst case is L5 np4, where the forward and reverse
  distribute and migrate phases are 4.2 % together. `ff_diagnostics` is
  0.5–2.9 %.
- **No crossover in range.** `fmm` over estimated `direct` per step is about
  33x at L4, 29x at L5 np1, 16x at L5 np4, 27x at L6 np1 and 16x at L6 np4.
  The direct sum on HIP scales better than N² over this range.
- **R7.** The L4 np1 direct estimate (0.0087–0.0094 s/step at steps 20–60)
  is 1.02–1.11x M0-D1's measured 0.0085 s/step. Step 1's 0.0108 s carries the
  first step's setup.
- **R8: level 6 fits one MI300A at np1**, at 2.9 s/step early.

**What only running revealed.**
- **Claim A's metric exceeds $\tau_A$ at L5 and L6 on the unoptimized
  build:** 1.62e-3 and 2.71e-3, against `kTauA = 1.5e-3`.
  - At L5 there is no fallback at all. The error grows as the P2P pair
    fraction falls (0.25 at L4, 0.06 at L5, 0.015 at L6), i.e. as more of the
    field is carried by truncated order-3 expansions.
  - $\tau_A$ was established at L3/L4 only.
- **Canopy's `migrate_total`, `rebalance_total`, `setup_total`, `partition`
  and `redistribute` read 0 in every row.**
  - `auto_maintain` runs `_finish_topology_{change,stable}` directly, which time
    only `builder_build`, `repartition`, `sort_by_leaf` and `comm_plan_build`.
    `migrate()` / `rebalance()` / `setup()`, which own those totals, are not
    on its path.
  - Its own comparison `_builder.build` and its `redistribute` call are
    untimed. At L4 the timed children account for only about 40 % of
    `ff_maintain` (0.0034 of 0.0086 s).

**Affects:**
- T3 — the sizing rates above are early-state rates, so size the matrix
  with T0's growth factors (up to 3.2x at L4). The whole matrix is far cheaper
  than the design feared: L6 np1 fits at 2.9 s/step.
- T3 and Conventions — "`[acc]` ≤ 1.5e-3 at every SMS state" cannot hold at L5,
  which the SMS must include: the baseline itself is 1.62e-3 there. The accuracy
  bound needs restating for L5/L6 (e.g. no regression against T3's baseline)
  before any O task. **This needs the user's decision.**
- T3 and O5 — `ff_maintain`'s Canopy split is incomplete. `migrate_total` and
  `rebalance_total` never fire under `auto_maintain`, and about 60 % of L4
  `ff_maintain` is in untimed code. O5 may need timers for the comparison
  build and `redistribute` first.
- O1 — confirmed: the operator table is rebuilt at every evaluation, at 37–58 %
  of `ff_solve`.
- O3 — range-guard fallback is absent at L4/L5 in early states and about
  3.3e3 pairs at L6.
- O6 — the adapter phases are at most 4.7 % of `ff_evaluate_total` (L5 np4),
  just under its 5 % trigger at these states, and rising with rank count.

## L5/L6 accuracy check (after T2)

T2 measured Beatnik's claim-A metric above `kTauA` = 1.5e-3 at L5 (1.62e-3)
and L6 (2.71e-3). This topic asks two things: does the Python reference lose
the same accuracy at those sizes, and which knob closes the gap. No code under
`src/` changed. The new files are
`tests/regression_tests/reference_treecode_levels.py`, a reference-error tool
that reads Beatnik `.h5` checkpoints, and two `pdebug` runners,
`scripts/tuolumne/cprof_ref_levels.flux` and
`scripts/tuolumne/cprof_knob_arms.flux`.

**Provenance.**
- Beatnik `86c32ac` and Canopy `c182348`, clean. Dev env as installed at T2.
- Reference `zmodel3d-amr` `ec7d7bf`, clean `zmodel3d/`; `/usr/tce/bin/python3`.
- Job `f3dipDdn7yZZ` (reference, 574 s wall) and job `f3dipDmkg7V9` (knob
  arms, 885 s wall). Both rc 0, both submitted together, queue wait under
  1 min.
- HIP, binding as T2.

**The reference on Beatnik's own states.** These are direct-driven
`Beatnik_Test_Milestone0Run` checkpoints at steps 25 and 50, HIP np1. The
reference runs θ 0.3, order 2. The value is claim A's metric.

| vertices | step | reference `ncrit` 64 (its default) | reference `ncrit` 8 |
| --- | --- | --- | --- |
| 2562 (L4) | 25 / 50 | 5.45e-4 / 5.03e-4 | 1.73e-3 / 1.59e-3 |
| 10242 (L5) | 25 / 50 | 9.18e-4 / 9.25e-4 | 1.17e-3 / 1.19e-3 |
| 40962 (L6) | 25 / 50 | 9.29e-4 / 9.42e-4 | 1.02e-3 / 9.57e-4 |

The L4 `ncrit` 64 value matches T9r's step-25 figure, 5.83e-4. That run was
on the gold state rather than this checkpoint.

**The reference does not lose accuracy with N.** At its default it sits at
about 9.3e-4 at L5 and L6. At `ncrit` 8 it is *worse* at L4 and *better* at L6.
So Beatnik's growth (5.1e-4 → 1.6e-3 → 2.7e-3 at `ncrit` 8) is not shared by
the reference. By the user's acceptance rule ("acceptable if the Python
suffers the same loss"), Beatnik at `ncrit` 8 is **not** acceptable at L5/L6.

**Beatnik knob arms.** These come from `Beatnik_Probe_FmmCost`, FMM-driven,
50 steps, `[acc]` at steps 1/25/50. L4 runs at np1 and L5/L6 at np4. One knob
is changed per row from the baseline of `ncrit` 8, order 3, θ 0.3.

| arm | `[acc]` 1 / 25 / 50 | s/step | P2P frac | `fb_range_guard` (last) |
| --- | --- | --- | --- | --- |
| L4 base | 5.01e-4 / 5.06e-4 / 5.07e-4 | 0.319 | 0.250 | 0 |
| L4 `ncrit` 64 | 1.74e-4 / 1.72e-4 / 1.71e-4 | **0.057** | 0.852 | 0 |
| L4 order 4 | 5.6e-5 / 5.6e-5 / 5.7e-5 | 0.881 | 0.250 | 0 |
| L4 θ 0.2 | 1.16e-4 / 1.16e-4 / 1.17e-4 | 0.469 | 0.563 | 0 |
| L5 base | 1.63e-3 / 1.63e-3 / 9.43e-4 | 0.734 | 0.058 | 0 |
| L5 `ncrit` 64 | 5.44e-4 / 5.40e-4 / 4.97e-4 | **0.330** | 0.424 | 0 |
| L5 order 4 | 1.90e-4 / 1.92e-4 / 1.72e-4 | 1.619 | 0.058 | 0 |
| L5 θ 0.2 | 2.06e-4 / 2.00e-4 / 1.82e-4 | 1.473 | 0.127 | 31614 |
| L6 base | 2.71e-3 / 2.71e-3 / 2.77e-3 | 1.904 | 0.015 | 15652 |
| L6 `ncrit` 64 | 1.45e-3 / 1.49e-3 / 1.50e-3 | **0.817** | 0.092 | 0 |
| L6 order 4 | 3.8e-4 / 3.8e-4 / 3.7e-4 | 3.707 | 0.015 | 15652 |
| L6 θ 0.2 | 5.2e-4 / 5.2e-4 / 5.6e-4 | 3.608 | 0.034 | 252066 |

The direct-cost estimates per step were 0.006 (L4 np1), 0.034 (L5 np4) and
0.110 s (L6 np4).

**What it shows.**
- **`ncrit` 64 is both more accurate and 2.2–5.6x faster** at every level.
  - At L4 and L5 it is better than the reference's own default.
  - At L6 it is 1.5e-3, still 1.6x the reference's 9.3e-4.
- **Leaf size explains most of the gap but not all of it.** At fixed
  `ncrit`, Beatnik's error grows with N (`ncrit` 64: 1.7e-4 → 5.4e-4 →
  1.5e-3); the reference's does not.
  - That residual is consistent with the FMM's two-sided truncation
    accumulating over a deeper tree. This was not isolated here.
  - Order 4 removes it: 3.8e-4 at L6 even at `ncrit` 8, for about 2x the
    cost.
  - **Caveat:** the CartesianTaylor derivative ladder is validated only to
    |k| = 6 (`../canopy/tasks/02_oracle_extension.md`), so order-4 numbers are
    from an arm that is not yet oracle-checked.
- **Production already defaults to `ncrit` 64** (`FmmParams::ncrit`,
  `src/Beatnik_Params.hpp:256`). Two things run at 8:
  - The milestone FMM members, deliberately: at 2562 vertices, 64 makes
    85 % of pairs P2P. That fails the L4 member's liveness bound
    `kP2PFractionBound = 0.75`. At L3 the M2L count is exactly 0.
  - This document's probe default and every canopy_profiling measurement so
    far.
- **Cost at production `ncrit`.** `fmm` over estimated `direct` per step is
  9x at L4, 10x at L5 np4 and 7x at L6 np4, against 33x, 22x and 17x at
  `ncrit` 8. There is still no crossover in range. The operator build is
  39–49 % of `ff_solve` at `ncrit` 64.

**Affects:**
- T3 — the cost model should be measured at production `ncrit` 64, perhaps
  with an `ncrit` 8 arm for the milestone configuration. At 64 the costs and
  the phase shares differ materially (L4 is 5.6x cheaper). **User decision
  pending.**
- Conventions (accuracy bound) — at `ncrit` 64, `[acc]` is ≤ 1.5e-3 at L4
  and L5, and at L6 it is 1.45–1.50e-3, at the bound. The L5/L6 τ_A problem
  largely disappears if T3 measures at 64.
- add-canopy milestone tier — unchanged by this. Its `ncrit` 8 is a
  deliberate far-field stress configuration (header of
  `Beatnik_Test_Milestone0Fmm.cpp`), and changing it fails the liveness bound.
  **User decision pending.**
- O1–O5 — any optimization should be judged at production `ncrit` 64 as well.
  The residual N-growth of the error at fixed `ncrit` is a candidate follow-up,
  with order 4 as the measured fix.

## Decision: optimize at `ncrit 8`, accept the L5/L6 accuracy loss

The user decided, after the L5/L6 accuracy check:
- **All canopy_profiling measurement stays at `ncrit 8`.** At `ncrit 64` much
  of the far field becomes P2P and would hide the unoptimized code the
  programme targets.
- **The accuracy loss above level 4 is accepted.** It comes from leaf size and
  order, not from any planned optimization. Only O1 (a quantized root box
  changes the tree) can move it, and the guard catches that.

Design changes made with this entry:
- Conventions' accuracy row and the common O criterion: ≤ 1.5e-3 at level 4,
  and no regression beyond T3's run-to-run spread against T3's baseline at
  levels 5 and 6.
- A deliberate deviation records the `ncrit 8` choice.
- V1 gains a step re-measuring the final speedup at production `ncrit 64`.
- README Known Issues records the N-growth of the error against the
  reference.

**Affects:**
- T3 — the `[acc]` baseline at L5/L6 is the regression reference. Record the
  spread (R2) at L5 as well as L4.
- O1–O7 — use the new accuracy criterion.
- V1 — the extra `ncrit 64` measurement.

## T3

**Provenance, pinned.**
- **Code.** Beatnik `d45aeed` checked out. The installed dev binary is
  `beatnik-develop-ckhvawf` (installed at T2, 11:14 the same day), built from
  `86c32ac`; every later commit is docs or scripts. Canopy `c182348`, clean
  porcelain in every job.
- **Toolchain.** `canopy@develop +profiling profiling_level=2`, cce@20.0.0,
  dev env `~/spack_envs/tuolumne_beatnik`.
- **No install** in this task. The Beatnik porcelain the jobs logged is
  `.claude/settings.json`, two untracked files, and this task's uncommitted
  reducer and script, none of them compiled.
- **Configuration.** HIP and SERIAL, CartesianTaylor order 3, θ 0.3,
  `ncrit 8`, `m2l_op_count_cap 65536`, `canopy_report 0`, except the named
  `ncrit` arm.
- **Binding.** `launch()` copied from `cprof_knob_arms.flux`: `flux run
  --ntasks=<np> --nodes=1 --exclusive --gpus-per-task=1 --cores-per-task=24
  --setopt=mpibind=verbose:1`, for both backends. mpibind gave each np4 rank
  its own GPU and a disjoint 24-core set.

All jobs are `pdebug`, one node, `-t 58m`, budget 3300 s. All exited rc 0 with
no `SKIPPED` or `FAIL` line. Submitted together at 12:45:16–19; queue wait
1–2 s each.

| job | `BEATNIK_CPB_SET` | runs (level steps sample_every) | launch walls | job wall |
| --- | --- | --- | --- | --- |
| `f3dj4QS4bmGB` | `hip_np1_l4` | L4 np1 2000 50 | 2018 s | 34.0 min |
| `f3dj4QaHyn1D` | `hip_np1_l5l6` | L6 np1 250 25; L5 np1 800 50 | 915 + 1406 s | 39.1 min |
| `f3dj4QiSupuD` | `hip_np4_l4` | L4 np4 2000 50 | 1411 s | 23.9 min |
| `f3dj4Qt8o8hu` | `hip_np4_l5l6` | L6 np4 250 25; L5 np4 800 50 | 564 + 1025 s | 26.9 min |
| `f3dj4R2EmD3D` | `hip_np4_ncrit` | L5 np4 800 50 at `ncrit` 64 and 32 | 277 + 458 s | 12.7 min |
| `f3dj4RASfEVu` | `serial_np1` | L4 np1 500 25 | 885 s | 15.9 min |
| `f3dj4RJLmQaT` | `serial_np4` | L4 np4 500 25 | 1931 s | 32.6 min |
| `f3dj4RSg5NSs` | `spread_hip_np4` | SMS L5 np4 300 25 ×2; SMS L4 np4 1200 50 ×2 | 301 + 293 + 744 + 744 s | 35.1 min |
| `f3djNiFw5apw` | `sms_hip_np1` | SMS L5 np1 300 25; SMS L4 np1 1200 50 | 384 + 1038 s | 24.1 min |

The last job was submitted at 13:25, after the matrix, for the reason under
"Departures".

**Decisions, made before this task and recorded here.**
- **Every measurement runs at `ncrit 8`**, order 3, θ 0.3,
  `m2l_op_count_cap 65536`, the probe's defaults. `ncrit` 32 and 64 appear
  only in the level-5 arm, which informs V1's production-`ncrit` statement.
- **The accuracy baseline is per state.** Level 4 is held to ≤ 1.5e-3; levels
  5 and 6 are recorded as they are and guarded by "no regression beyond the
  measured spread". So the spread was measured at level 4 and level 5.
- **The per-row count check lives in the reducer** (below).
- **Untimed maintenance is reported as untimed.** T3 adds no timer; O5 owns
  that.
- **No new probe key.** None was needed for (a)–(f). O10's split does need
  keys the probe lacks, so O10 adds them.

**Decisions made in this task.**
- **One script with named sets**, `canopy_profiling_baseline.flux`, selected
  by `BEATNIK_CPB_SET`. It has nine T3 sets and four SMS sets; the
  `BEATNIK_CPS_SWEEP` row override and the budget guard are as in
  `cprof_knob_arms.flux`. A row is `label level np steps sample_every space
  knob estimate`; `space` picks `Beatnik_Probe_FmmCost_MPI_{HIP,SERIAL}`.
- **Eight jobs, not one or two.** Each holds at most about 2300 s of
  estimated work, so no row depended on another finishing early. pdebug had
  12 free nodes and ran all eight at once.
- **The spread is measured at HIP np4**, which carries both the partitioner
  and device-reduction sources of non-reproducibility. It uses the SMS rows
  exactly, twice in one allocation. The matrix's own np4 runs, on other nodes,
  give a cross-node comparison at level 4.
- **Level 6 is outside the SMS.** Its spread is unmeasured and its rows cost
  540–891 s. A tree-changing O task adds T3's level-6 rows instead (design,
  "SMS definition").
- **O-task IDs are kept, not renumbered.** Earlier log entries cite O1, O3 and
  O6 by ID, so the sections were reordered in place, O8–O10 added, and O3
  marked NOT TAKEN.

**The reducer change** (`tests/regression_tests/fmm_profile_reduce.py`).
- **`cost_count_check`** runs on every `[cost]` row at parse time. It needs
  `steps`, `evals` and the four `maint_*` keys as non-negative integers;
  anything else is malformed and fatal (`ReduceError`, exit 1).
- **A well-formed row failing `maint sum == evals == 3*steps`** goes to
  `rows["excluded"]`. It never enters `reduce_probe`'s statistics, and
  `print_probe` reports it as `EXCLUDED [cost] run=<r> step=<s>: <reason>`
  beside a per-run `count check: N passed, M excluded` line. It does not
  change the exit status.
- **The `[acc]` row at the same step is kept.** It is a standalone evaluation
  outside the window.
- **`--states`** prints one line per `[cost]` row, joined to its `[acc]` row:
  - s/step, `3×ff_evaluate_total / step_wall`, `ff_solve` and `ff_maintain`
    per evaluation;
  - **untimed maintenance** (`ff_maintain − builder_build − repartition −
    sort_by_leaf − comm_plan_build`);
  - op-build share of `ff_solve`, adapter share, the action counts, P2P
    fraction, `fb_range_guard`, `acc_rel` and `direct_step_est`.

  It is the per-state before/after view the common O criterion asks for.
- **`--self-test`** gains a passing fixture (the existing probe rows, now
  with counts), a failing row (maint sum 59 against `evals` 60), and two
  malformed rows (a missing `maint_rebuild`, `evals=-1`). The failing row's
  `step_wall=9.99` must not reach any statistic.
- **Verified.**
  - The self-test passes.
  - A copy with the equality forced true fails it with an `AssertionError`.
  - The T2 smoke log, the knob-arms log and the T9b milestone log all still
    reduce, rc 0.
  - Every one of the nine T3 logs reduces, rc 0, with
    **328 `[cost]` rows passing and none excluded**.

**What only running revealed.**
- **The time step is not fixed:** the input enables adaptive timestepping,
  deliberately (see the decision below). At level 4, dt is 0.003 to about step
  100, then falls steadily to 0.00035 by step 1150 and stays near 0.0004 to
  step 2000. So t = 1.573 at step 1000 and only 1.962 at step 2000.
  - dt's plateau coincides with the per-step cost plateau (T0, R3).
  - Per unit of physical time, the late trajectory costs about 30x the
    early one.

  **Last states, R3:**
  - L4 2000 steps → t = 1.962; L4 1200 (SMS) → t = 1.646;
  - L5 800 → t = 1.540, which matches L4 at about step 990, the plateau
    onset; L5 300 (SMS) → t = 0.800;
  - L6 250 → t = 0.688; SERIAL L4 500 → t = 1.154.
- **`[acc]` is bit-reproducible.** `acc_rel` agrees to all nine printed
  digits, at every matched state, between:
  - the two repetitions in one job, at L4 and L5;
  - runs on different nodes;
  - np1 and np4;
  - SERIAL and HIP.

  The action histogram (e.g. 1/1363/4636/0 at L4) is identical at np1 and np4
  and on both backends. **R2's premise does not hold for the accuracy metric;
  only the step wall varies.**
- **Range-guard fallback is not a level-6-only effect.** It appears at level
  4 from step 250 (36 of 41 rows non-zero, peak 6166 pairs) and at level 5 from
  step 200 (peak 16018). T2's "none at L4/L5" was a statement about steps
  1–60. It costs little: `m2l_kernel`, which contains the fallback, is at most
  4.3 % of any HIP row.
- **The operator-table copy is a separate 9–18 %.** `ilist_s4_op_table_copy`
  re-copies every unique column on the host and re-uploads the whole table at
  every interaction-list build, so O1's retention does not shrink it
  (`Canopy_DownwardSweep.hpp:2142-2172`).
- **SERIAL's cost is the adapter's distributors**, not maintenance:
  `ff_fwd_distribute` + `ff_rev_distribute` are 52 % (np1) and 87 % (np4) of
  the step. T0 could only say "outside `solve()`".
- **The non-BR step grows at HIP np4.** It is 2.4 % of the step at step 50
  and 8.4 % at step 2000 (L4), against ≤ 0.7 % at np1. `direct_step_est`
  inherits it, because it adds the FMM step's non-BR time: L4 np4 is 0.020 →
  0.082 s/step. **The np4 `fmm`/`direct` ratios are therefore understated;**
  the np1 ratios are the clean ones.
- **Level 5 grows like level 4.** At np1 it goes from 0.98 to 2.40 s/step
  over 800 steps (2.5x); at np4 from 0.76 to 1.77. The sizing's early rates
  (0.83, 0.69) held only for the first 50 steps.
- **`auto_maintain` rebalances almost every evaluation:** 77 % at L4, 98 % at
  L5 and L6. Rebuild never fires.

**(a) Phase split per step, % of step wall** (each per-evaluation phase × 3 /
step wall), early, middle and late rows.
- *op build* is `ilist_s4_op_table_build`; *op copy* is
  `ilist_s4_op_table_copy`.
- *classify + rest of ilist* is `dn_build_ilist` minus those two: mostly
  `ilist_s3_classify_pairs`, plus 1–4 % untimed between stages.
- *rest of downward + L2P* is `downward_total − dn_build_ilist − m2l_kernel`.
- *maint untimed* is `ff_maintain` minus the four timed children.
- *non-BR* is `step_wall − 3 × ff_evaluate_total`.

Per evaluation, divide by 3.

| run | step (t) | s/step | maint timed | maint untimed | op build | op copy | classify + rest of ilist | `m2l_kernel` | rest of downward + L2P | up + P2P | adapter + diag | non-BR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L4 HIP np1 | 50 (0.15) | 0.327 | 2.8 | 4.6 | 51.7 | 15.5 | 15.7 | 3.7 | 1.5 | 2.0 | 1.7 | 0.7 |
| L4 HIP np1 | 1050 (1.59) | 1.248 | 4.5 | 4.4 | 54.4 | 18.5 | 13.3 | 1.9 | 0.8 | 1.0 | 0.9 | 0.2 |
| L4 HIP np1 | 2000 (1.96) | 1.200 | 4.7 | 4.7 | 53.2 | 18.1 | 14.4 | 1.8 | 0.9 | 1.1 | 1.0 | 0.2 |
| L4 HIP np4 | 50 (0.15) | 0.307 | 3.8 | 5.8 | 38.4 | 13.0 | 7.5 | 2.2 | 13.8 | 6.2 | 6.7 | 2.4 |
| L4 HIP np4 | 1050 (1.59) | 0.795 | 6.1 | 4.5 | 41.3 | 14.2 | 6.6 | 1.5 | 11.1 | 6.7 | 3.2 | 4.8 |
| L4 HIP np4 | 2000 (1.96) | 0.825 | 6.1 | 4.9 | 40.3 | 13.9 | 6.5 | 1.3 | 7.5 | 7.8 | 3.3 | 8.4 |
| L5 HIP np1 | 50 (0.15) | 0.978 | 17.1 | 5.2 | 35.2 | 11.1 | 21.7 | 3.8 | 0.8 | 2.3 | 2.3 | 0.3 |
| L5 HIP np1 | 450 (1.08) | 1.854 | 14.4 | 4.8 | 41.4 | 14.7 | 18.1 | 2.8 | 1.1 | 0.9 | 1.5 | 0.2 |
| L5 HIP np1 | 800 (1.54) | 2.404 | 12.8 | 5.0 | 42.6 | 16.2 | 16.8 | 3.4 | 0.7 | 1.1 | 1.3 | 0.1 |
| L5 HIP np4 | 50 (0.15) | 0.756 | 14.3 | 4.6 | 31.2 | 10.7 | 10.5 | 2.0 | 15.0 | 5.5 | 4.8 | 1.2 |
| L5 HIP np4 | 450 (1.08) | 1.274 | 15.2 | 5.0 | 37.5 | 13.9 | 11.1 | 1.8 | 7.2 | 3.7 | 2.7 | 1.9 |
| L5 HIP np4 | 800 (1.54) | 1.770 | 13.0 | 5.1 | 37.7 | 13.9 | 10.9 | 1.9 | 9.5 | 3.7 | 2.4 | 1.8 |
| L6 HIP np1 | 25 (0.07) | 3.096 | 26.6 | 7.6 | 23.4 | 8.2 | 25.9 | 3.9 | 0.5 | 0.7 | 2.9 | 0.1 |
| L6 HIP np1 | 150 (0.44) | 3.587 | 26.7 | 7.4 | 24.9 | 9.1 | 23.7 | 3.9 | 0.6 | 1.0 | 2.5 | 0.1 |
| L6 HIP np1 | 250 (0.69) | 4.161 | 24.0 | 7.8 | 28.1 | 9.9 | 22.5 | 3.7 | 0.6 | 0.9 | 2.3 | 0.1 |
| L6 HIP np4 | 25 (0.07) | 1.927 | 21.5 | 6.2 | 30.1 | 11.4 | 14.0 | 2.4 | 8.2 | 2.9 | 2.7 | 0.5 |
| L6 HIP np4 | 150 (0.44) | 2.176 | 23.3 | 6.3 | 30.3 | 11.6 | 13.3 | 2.4 | 6.3 | 2.8 | 2.8 | 0.7 |
| L6 HIP np4 | 250 (0.69) | 2.482 | 21.4 | 6.5 | 32.4 | 12.6 | 12.8 | 2.2 | 5.8 | 2.9 | 2.6 | 0.7 |
| L4 SERIAL np1 | 25 (0.07) | 1.515 | 0.9 | 1.4 | 10.4 | 2.3 | 4.5 | 14.3 | 0.9 | 2.6 | 62.4 | 0.4 |
| L4 SERIAL np1 | 275 (0.75) | 1.632 | 3.6 | 1.7 | 17.6 | 4.7 | 4.4 | 14.3 | 1.3 | 2.4 | 49.6 | 0.3 |
| L4 SERIAL np1 | 500 (1.15) | 2.027 | 3.3 | 2.6 | 22.2 | 6.2 | 4.3 | 13.8 | 1.2 | 2.0 | 44.0 | 0.3 |
| L4 SERIAL np4 | 25 (0.07) | 3.483 | 0.4 | 0.6 | 3.6 | 1.0 | 0.7 | 1.7 | 2.1 | 0.6 | 89.0 | 0.3 |
| L4 SERIAL np4 | 275 (0.75) | 3.553 | 1.9 | 0.6 | 4.6 | 1.2 | 0.9 | 2.1 | 2.3 | 0.6 | 85.5 | 0.3 |
| L4 SERIAL np4 | 500 (1.15) | 4.228 | 1.4 | 0.6 | 5.3 | 1.5 | 0.9 | 2.4 | 2.8 | 0.7 | 84.0 | 0.3 |
| L5 HIP np4 `ncrit 32` | 50 (0.15) | 0.332 | 4.3 | 5.9 | 34.9 | 11.3 | 6.4 | 2.2 | 15.9 | 8.4 | 7.8 | 2.7 |
| L5 HIP np4 `ncrit 32` | 450 (1.08) | 0.563 | 10.6 | 4.5 | 36.1 | 12.6 | 6.7 | 2.0 | 10.8 | 7.6 | 5.2 | 3.6 |
| L5 HIP np4 `ncrit 32` | 800 (1.54) | 0.712 | 8.7 | 4.8 | 41.6 | 13.5 | 7.4 | 1.5 | 6.4 | 8.1 | 3.6 | 4.2 |
| L5 HIP np4 `ncrit 64` | 50 (0.15) | 0.321 | 11.2 | 5.4 | 34.3 | 11.0 | 4.7 | 1.6 | 13.0 | 9.1 | 6.8 | 2.6 |
| L5 HIP np4 `ncrit 64` | 450 (1.08) | 0.268 | 10.9 | 6.7 | 25.8 | 7.8 | 4.2 | 4.8 | 11.2 | 13.4 | 8.7 | 6.1 |
| L5 HIP np4 `ncrit 64` | 800 (1.54) | 0.401 | 7.8 | 5.3 | 34.7 | 11.7 | 4.6 | 1.4 | 9.7 | 12.5 | 5.6 | 6.5 |

**Untimed maintenance, explicitly.** As a share of `ff_maintain` (per-eval
seconds in log rows):
- **L4:** 42–62 %, which is 4.4–5.8 % of step wall at np1 and np4.
- **L5:** 23–28 %; **L6:** 21–24 %.
- **SERIAL:** 23–62 %.

The timed part is dominated by `comm_plan_build`, and
`commplan_dtt_traversal` is about two-thirds of that. At L6 np1 they are 0.26
and 0.17 s per evaluation of a 0.35–0.44 s `ff_maintain`. `builder_build` +
`repartition` + `sort_by_leaf` are under 3 % of step wall everywhere.

**(b) and (c) Actions, operator build, retention, depth and fallback.**
- **Retention is about zero at every row past step 1.** Keys built per
  evaluation equal `unique_ops` (per-run medians 0.96–1.01). At step 1 it is
  exactly 2/3: one of the first three evaluations reuses the setup's table.
- **Occupied depth reaches 7** at level 4 by step 700 (np1; 800 as an np4
  rank mean) and at level 5 by step 200. Level 6 starts at 7.

| run | actions setup/migrate/rebalance/rebuild | op build / `ff_solve` min–med–max | keys built per eval / `unique_ops` (median) | occupied depths (first ≥ 7) | `fb_range_guard` max (rows > 0) |
| --- | --- | --- | --- | --- | --- |
| L4 HIP np1 | 1/1363/4636/0 | 0.57–0.61–0.63 | 1.00 | 5–8 (step 700) | 6166 (36/41) |
| L4 HIP np4 | 1/1363/4636/0 | 0.43–0.50–0.54 | 1.01 | 5–8 (step 800) | 6166 (36/41) |
| L5 HIP np1 | 1/45/2354/0 | 0.47–0.52–0.55 | 0.96 | 6–8 (step 200) | 16018 (13/17) |
| L5 HIP np4 | 1/45/2354/0 | 0.42–0.47–0.50 | 0.98 | 6–8 (step 200) | 16018 (13/17) |
| L6 HIP np1 | 1/10/739/0 | 0.37–0.39–0.43 | 0.99 | 7–8 (step 1) | 21768 (11/11) |
| L6 HIP np4 | 1/10/739/0 | 0.44–0.45–0.47 | 0.99 | 7–8 (step 1) | 21768 (11/11) |
| L4 SERIAL np1 | 1/358/1141/0 | 0.30–0.38–0.45 | 1.00 | 5–6 (never) | 2000 (11/21) |
| L4 SERIAL np4 | 1/358/1141/0 | 0.34–0.37–0.41 | 1.00 | 5–6 (never) | 2000 (11/21) |
| L5 HIP np4 `ncrit 32` | 1/546/1853/0 | 0.43–0.48–0.53 | 0.99 | 5–7 (step 800) | 2128 (11/17) |
| L5 HIP np4 `ncrit 64` | 1/638/1761/0 | 0.38–0.46–0.48 | 1.00 | 5–6 (never) | 504 (8/17) |

**(d) `fmm` against estimated `direct` per step, and the crossover.**
`direct_step_est` is from `[acc]`, the median over the run's states.

| run | fmm s/step early / mid / late | direct est s/step (median) | fmm / direct early / mid / late | P2P fraction early / late |
| --- | --- | --- | --- | --- |
| L4 HIP np1 | 0.327 / 1.248 / 1.200 | 0.0089 | 38.0 / 139.7 / 123.2 | 0.250 / 0.362 |
| L4 HIP np4 | 0.307 / 0.795 / 0.825 | 0.0508 | 15.1 / 14.5 / 10.1 | 0.250 / 0.362 |
| L5 HIP np1 | 0.978 / 1.854 / 2.404 | 0.0278 | 35.7 / 66.0 / 84.3 | 0.057 / 0.078 |
| L5 HIP np4 | 0.756 / 1.274 / 1.770 | 0.0569 | 18.1 / 21.7 / 26.7 | 0.057 / 0.078 |
| L6 HIP np1 | 3.096 / 3.587 / 4.161 | 0.1078 | 28.5 / 32.9 / 38.6 | 0.015 / 0.016 |
| L6 HIP np4 | 1.927 / 2.176 / 2.482 | 0.1253 | 15.6 / 17.4 / 19.5 | 0.015 / 0.016 |
| L4 SERIAL np1 | 1.515 / 1.632 / 2.027 | 0.6444 | 2.4 / 2.5 / 3.1 | 0.252 / 0.263 |
| L4 SERIAL np4 | 3.483 / 3.553 / 4.228 | 0.1886 | 18.6 / 18.9 / 22.2 | 0.252 / 0.263 |
| L5 HIP np4 `ncrit 32` | 0.332 / 0.563 / 0.712 | 0.0541 | 7.9 / 10.3 / 10.4 | 0.254 / 0.372 |
| L5 HIP np4 `ncrit 64` | 0.321 / 0.268 / 0.401 | 0.0485 | 7.7 / 5.7 / 6.8 | 0.424 / 0.623 |

- **No crossover in range, at any `ncrit` measured.**
  - At matched early states (step 50, or 25 at L6) the clean np1 ratio is
    38, 36 and 29 at levels 4, 5 and 6. Late states are worse: 123 at L4,
    84 at L5.
  - The direct estimate scales as N^0.82 from L4 to L5 and N^0.98 from L5
    to L6; the GPU is not saturated, so it is not yet N². The early FMM
    step scales as N^0.79 and N^0.83.
  - So the early-state ratio falls by about 0.87x per level. Extrapolated,
    that reaches 1 only after about 25 more levels: not a crossover in any
    reachable range.
- **Leaf size changes the ratio, not its sign.** At level 5 np4 the late
  ratio is 26.7 at `ncrit 8`, 10.4 at 32 and 6.8 at 64.
- **The `ncrit 64` arm** extends the "L5/L6 accuracy check" point along the
  trajectory: 0.27–0.40 s/step, and `[acc]` 3.9e-4 to 7.9e-4 at every state,
  under τ_A.

**(e) Outside `solve()`, medians, % of step wall.**

| run | `ff_solve` | `ff_maintain` | adapter + `ff_diagnostics` | non-BR | adapter + diag / `ff_evaluate_total` |
| --- | --- | --- | --- | --- | --- |
| L4 HIP np1 | 89.0 | 9.8 | 1.0 | 0.2 | 1.0 |
| L4 HIP np4 | 78.7 | 11.8 | 3.4 | 4.8 | 3.6 |
| L5 HIP np1 | 78.7 | 19.6 | 1.5 | 0.2 | 1.5 |
| L5 HIP np4 | 75.4 | 19.9 | 2.8 | 1.7 | 2.8 |
| L6 HIP np1 | 62.9 | 34.2 | 2.6 | 0.1 | 2.6 |
| L6 HIP np4 | 67.1 | 29.5 | 2.8 | 0.7 | 2.8 |
| L4 SERIAL np1 | 40.7 | 5.0 | 54.2 | 0.3 | 54.4 |
| L4 SERIAL np4 | 10.8 | 2.0 | 86.8 | 0.3 | 87.0 |
| L5 HIP np4 `ncrit 32` | 77.6 | 14.2 | 4.8 | 3.4 | 5.0 |
| L5 HIP np4 `ncrit 64` | 73.0 | 15.9 | 7.2 | 5.6 | 7.6 |

- On HIP the adapter phases plus `ff_diagnostics` are 1.0–3.4 % (medians);
  the single-state maximum is 6.7 % at L4 np4, step 50.
- On SERIAL they are the step (O7).
- `ff_maintain` is the larger outside-`solve()` share on HIP: 10–34 %.

**(f) The accuracy baseline, `acc_rel` at every state.** Identical at np1 and
np4 and on SERIAL and HIP, so one list per level is the baseline for every SMS
row. The lists are rounded to three digits for reading; because `[acc]`
reproduces exactly, an O task compares against the full nine-digit values in
the job logs.
- **Level 4** (2000 steps, every 50): max **1.099e-3** (step 1000), min
  5.01e-4. **≤ τ_A = 1.5e-3 at all 41 states.**
  1: 5.01e-04, 50: 5.07e-04, 100: 5.17e-04, 150: 6.09e-04, 200: 5.47e-04, 250: 6.04e-04, 300: 5.15e-04, 350: 7.62e-04, 400: 5.28e-04, 450: 5.85e-04, 500: 6.32e-04, 550: 6.16e-04, 600: 8.15e-04, 650: 6.62e-04, 700: 7.56e-04, 750: 8.10e-04, 800: 7.76e-04, 850: 7.46e-04, 900: 6.56e-04, 950: 8.28e-04, 1000: 1.10e-03, 1050: 5.87e-04, 1100: 5.74e-04, 1150: 7.47e-04, 1200: 6.02e-04, 1250: 7.94e-04, 1300: 8.66e-04, 1350: 1.08e-03, 1400: 7.67e-04, 1450: 9.23e-04, 1500: 6.03e-04, 1550: 7.09e-04, 1600: 9.03e-04, 1650: 7.96e-04, 1700: 6.73e-04, 1750: 5.28e-04, 1800: 6.77e-04, 1850: 8.12e-04, 1900: 5.37e-04, 1950: 5.30e-04, 2000: 5.33e-04
- **Level 5** (800 steps, every 50, and the SMS's every 25 to 300): max
  **1.634e-3** (step 1), min 8.27e-4.
  1: 1.63e-03, 50: 9.43e-04, 100: 1.57e-03, 150: 1.36e-03, 200: 9.44e-04, 250: 8.90e-04, 300: 9.09e-04, 350: 1.40e-03, 400: 1.26e-03, 450: 8.27e-04, 500: 1.18e-03, 550: 1.17e-03, 600: 1.16e-03, 650: 1.16e-03, 700: 1.14e-03, 750: 1.38e-03, 800: 1.44e-03
  SMS states not in the line above: 25: 1.63e-03, 75: 1.36e-03, 125: 1.15e-03, 175: 1.12e-03, 225: 8.35e-04, 275: 1.31e-03
- **Level 6** (250 steps, every 25): max **2.771e-3** (step 50), min 1.591e-3,
  falling along the trajectory.
  1: 2.71e-03, 25: 2.71e-03, 50: 2.77e-03, 75: 2.62e-03, 100: 2.58e-03, 125: 2.36e-03, 150: 2.18e-03, 175: 2.03e-03, 200: 1.95e-03, 225: 1.82e-03, 250: 1.59e-03

**Run-to-run spread (R2)**, HIP np4, the SMS rows.
- **`[acc]`: zero** at every state, at both levels, within the job and across
  nodes.
- **Step wall, L4 1200 steps, rep0 against rep1 in one job:**
  - summed wall over the run 730.0 against 728.9 s, −0.15 %;
  - per state, median |Δ| 0.70 %, max 4.6 % (step 800).
- **Step wall, L4, rep0 against the matrix run on another node** (25 matched
  states): summed −0.31 %; per state, median 1.6 %, max 4.6 % (step 250).
- **Step wall, L5 300 steps, rep0 against rep1:**
  - summed 282.3 against 280.0 s, −0.83 %;
  - per state, median 1.6 %, max 6.9 % at step 1 (4.4 % excluding step 1).
- The L5 matrix run (`sample_every 50`) is not a matched-window comparison
  with the SMS's 25-step windows, so no cross-node L5 figure is quoted.
- **Gain threshold (twice the spread):**
  - run level: 0.6 % at L4 (cross-node), 1.7 % at L5;
  - per state: 9 % at L4, 14 % at L5.

**R1 — instrumentation against the milestone tier.** T0 has no per-step
claim-B wall, so two comparisons were made against T9b's FmmL4 logs, at the same
configuration and trajectory.
- **Summed `solve()` per step in T0's 250-step bins.** Probe `ff_solve` over
  T9b `solve()`:
  - np1: 0.95, 1.00, 1.00, 1.01, 1.01, 1.02, 1.01, 1.03;
  - np4: 1.02, 1.02, 1.06, 1.04, 1.01, 1.01, 1.01, 1.02.
- **The 2000-step wall.** The probe's summed step wall is 1993.8 s against
  `[t6] COST claimB` 1966.0 s at np1 (1.014), and 1383.5 against 1361.1 s at
  np4 (1.016).
- **The gap is under 10 %,** so O-task gains may be quoted from the probe.
- The two builds differ in both directions: T9b printed level-1 tables, worth
  about 2 % (T2), and the probe fences and runs level-2 timers. Their sum is
  1.4–1.6 %. V1's tier walls bound the fence part alone.

**R7 — the direct estimate.** At L4 HIP np1, `direct_step_est` has a median
of 0.0089 s/step over 40 states (range 0.0086–0.0097), against M0-D1's
0.0085. The ratio is **1.05** (1.01–1.14), within 10 %, so it is **not
applied**. No `fmm`/`direct` statement above would change with it: every HIP
ratio is at least 10.

**SMS, measured.**
- **`sms_hip_np1`** (job `f3djNiFw5apw`): 384 + 1038 s, job wall 24.1 min.
  - Against the matrix's L4 np1 run on another node: `[acc]` is identical
    at all 25 states, summed wall +0.29 %, per state median 1.0 % and max
    4.7 %. That is the np1 cross-node spread, consistent with np4's.
  - Two copies are about 2.85 ks against the 3.3 ks budget, so an np1 on/off
    control fits one job with 14 % margin.
- **`sms_hip_np4`** is the spread job's rep0/rep1: 297 + 744 s.
- **SERIAL** is T3's own rows: 885 s (np1), 1931 s (np4).
- The design's "SMS definition" carries the arguments.

**Departures from the task's Do steps.**
- **A ninth job.** The HIP np1 SMS baseline needed one: the matrix ran L5
  np1 at `sample_every 50`, so the SMS's 25-step states had no np1 wall
  baseline. `sms_hip_np1` was run once, after the matrix, as that baseline.
- **Step 3's deletion rule removed O3.** Step 3's "add an O task for any
  phase above 10 %" added three (O8, O9, O10).
- **O6 now depends on the tasks that shrink the rest of the step.** Its
  trigger is a share, and those tasks raise it.

**Affects:**
- **O1** — first in the sequence. The operator build is 25–54 % of HIP step
  wall and is rebuilt at every evaluation over the full trajectory. Its
  failure direction is restated to T3's measured keys-built pattern. Its job
  adds T3's level-6 rows.
- **O1, O2 and the Conventions accuracy row** — the measured `[acc]` spread
  is **zero**, so "no regression beyond T3's spread" at levels 5 and 6 forbids
  any increase at any state. A task that changes the tree cannot pass that
  literally. **The user must rule on the L5/L6 bound before O1 closes.**
  (Ruled: see "Decision: no accuracy regression in any O task".)
- **O2, O4** — unchanged triggers, sequenced directly after O1.
- **O3** — NOT TAKEN: `m2l_kernel`, which contains the fallback, is at most
  4.3 % of any HIP row.
- **O5** — scope set by measurement. The comm-plan DTT is the first target
  (53–73 % of `ff_maintain` at L5/L6). Untimed maintenance is 42–62 % of
  `ff_maintain` (about 5 % of step wall) at L4 and 21–28 % at L5/L6, so new
  timers are needed only if O5 targets level 4. Rebalance is chosen on 77–98 %
  of evaluations.
- **O6** — trigger not met (1.0–3.4 %). It now depends on O1, O5, O8, O9 and
  O10 and is re-evaluated after them.
- **O7** — target identified: the adapter's forward and reverse distributors
  on SERIAL (52 % of the step at np1, 87 % at np4).
- **O8 (new)** — operator-table assembly copy and upload, 9–18 %.
- **O9 (new)** — pair classification, `ilist_s3_classify_pairs`, 4–17 %.
- **O10 (new)** — downward exchange and per-depth loop at np4, 6–11 % (max
  18 %). It adds the probe keys T3 could not.
- **V1** — the crossover statement is "none between levels 4 and 6 at `ncrit`
  8, 32 or 64". R1's gap is 1.4–1.6 %. The `ncrit 64` L5 np4 trajectory is
  0.27–0.40 s/step with `[acc]` ≤ 7.9e-4.
- **S0** — O10's share is zero at np1 and 6–18 % at np4, so it grows with rank
  count. HIP np4's non-BR step also grows to 8 % late.

## Decision: no accuracy regression in any O task

The user ruled after T3, answering the question its **Affects:** line raised.
- **Every optimization task holds accuracy from pre- to post-optimization,
  in aggregate.**
  - At every level, the maximum `acc_rel` over that level's SMS states after
    the task is ≤ the maximum before it. "Before" is the previous O task's
    value, or T3's for the first.
  - Individual states may move either way.
  - Because `[acc]` is bit-reproducible (T3), the comparison is exact; no
    spread allowance applies to accuracy.
  - This entry was first written as a per-state rule. The user clarified that
    "no regression" means in aggregate, and chose the worst state per level
    as the aggregate.
  - T3's maxima over the SMS states: **L4 1.099e-3** (step 1000) and **L5
    1.634e-3** (step 1). For a task that adds level 6: **L6 2.771e-3**
    (step 50, 250 steps).
  - Level 4's ≤ τ_A = 1.5e-3 still holds, implied by T3's baseline (max
    1.099e-3).
- **This replaces the relative guard above level 4** ("Decision: optimize at
  `ncrit 8`"), which allowed "no regression beyond T3's spread" at levels 5
  and 6 only. It now applies at every level, including tree-changing tasks.
  Their individual states may move as the MAC sees different cells (R4), but
  the level's worst state may not rise.
- **Adaptive timestepping is enabled in the input on purpose,** to evolve the
  simulation quickly enough. T3's falling dt (0.003 to about 0.0004 at level
  4) is that, not a defect. Physical-time and step-count comparisons are
  therefore not interchangeable. The SMS compares at matched steps, which are
  matched states because the trajectory is bit-reproducible.

Design changes made with this entry: the Conventions accuracy row, the common O
exit criterion, the SMS definition's accuracy bullet, R2 and R4, and README
Known Issues ("guards every level against regression").

**Affects:**
- O1–O10 — per level, the maximum `[acc]` over the SMS states must not rise
  against the previous task, exactly.
- O1, O2 — a quantized or retained tree may move individual states, but fails
  if it raises any level's worst state; there is no τ_A or spread allowance.
- V1 — unchanged; it re-asserts claim A on the milestone tier.

## O1

**Provenance, pinned.**
- **Code.** Beatnik `58937d2` "canopy_profiling O1: quantize_root_half_width
  from Beatnik, with counters", the exact code every job ran (each logs
  `beatnik commit = 58937d2`, dirty 4 files: `.claude/settings.json`, this
  doc, two untracked non-compiled files). Canopy `c182348`, 0 porcelain lines
  in every job. **No Canopy file changed.**
- **Toolchain.** `canopy@develop +profiling profiling_level=2`, cce@20.0.0,
  dev env `~/spack_envs/tuolumne_beatnik`, `profile.local.sh` (spack mode).
- **Install.** One `spack install`, 15:03:44, 11 min 37 s wall (Beatnik
  10 min 53 s; Canopy re-staged in 41 s, unchanged source). It was a real
  rebuild, about 110 CPU-min: every `.cpp` under `src tests examples` was
  touched first, per the header-only caveat in `systems/tuolumne/claude.md`
  §3. The installed `Beatnik_Params.hpp` has `= true`. The later edit to that
  header's doc comment (this section's ruling) changed no code and was not
  reinstalled.
- **Configuration.** As T3: CartesianTaylor order 3, θ 0.3, `ncrit 8`,
  `m2l_op_count_cap 65536`, `canopy_report 0`. Binding is
  `canopy_profiling_baseline.flux`'s `launch()`, unmodified.

All jobs `pdebug`, submitted 15:04:42, queue waits 0–86 s, all rc 0, no
`SKIPPED` or `FAIL`. The reducer's count check (`fmm_profile_reduce.py
--format probe --states`) passed every `[cost]` row with 0 excluded.

| job | name | rows (label level steps every) | launch walls | job wall |
| --- | --- | --- | --- | --- |
| `f3dk9yN1mqFM` | `beatnik_t4_fmm_vs_direct` | `t4_fmm_vs_direct.flux`, 2 nodes | — | 1.18 min |
| `f3dk9xouhPJF` | `beatnik_o1_hip_np4_l6` | `l6q0`, `l6q1` L6 np4 250 25 | 557 + 442 s | 17.1 min |
| `f3dk9xfGba3h` | `beatnik_o1_hip_np4` | `smsl5q0` 5 300 25, `smsl4q0` 4 1200 50, then `q1` | 297 + 722 + 190 + 383 s | 27.0 min |
| `f3dk9xY4X4aT` | `beatnik_o1_hip_np1` | same rows at np1 | 391 + 1052 + 302 + 493 s | 37.7 min |
| `f3dk9xwujWWB` | `beatnik_o1_serial_np1` | `smsl4q0`, `smsl4q1` 4 500 25 | 1056 + 829 s | 31.8 min |
| `f3dk9y61haqV` | `beatnik_o1_serial_np4_q0` | `smsl4q0` 4 500 25 | 1913 s | 32.4 min |
| `f3dk9yE6BftT` | `beatnik_o1_serial_np4_q1` | `smsl4q1` 4 500 25 | 1773 s | 30.0 min |

T3 bases: `f3dj4RSg5NSs` rep0 (HIP np4), `f3djNiFw5apw` (HIP np1),
`f3dj4Qt8o8hu` (L6 np4), `f3dj4RASfEVu` and `f3dj4RJLmQaT` (SERIAL np1, np4).

**Decisions, made before this task and recorded here.**
- The SMS keeps cap 65536, T3's configuration.
- Overflow clears are counted, not avoided, and their windows are exempt from
  the 90 % retention criterion.
- The counters are Beatnik-only; O1 changes no Canopy file.
- FmmVsDirect runs as its own two-node job in the same sitting.
- The accuracy guard is the per-level aggregate ("Decision: no accuracy
  regression in any O task").
- The knob is passed explicitly in both arms (`quantize_root_half_width=0`
  or `=1`), so the measurement does not depend on the installed default.

**Decisions made in this task.**
- **Installed with default `true`,** so FmmVsDirect exercises the quantized
  path.
- **A separate L6 job.** The SMS definition adds T3's level-6 rows for a
  tree-changing task; two L6 np4 arms plus the np4 SMS pair (about 3.2 ks)
  did not fit one 3.3 ks budget.
- **`Impl::readDiagnostics` returns the root half-width,** so `evaluate` can
  compare it without a second accessor.
- **The default stays `true` despite the level-4 `[acc]` rise,** by the
  user's ruling (next section).
- **The first evaluation stays uncounted** by the width counter, by the
  user's ruling (below, "Bugs only running revealed").

**Signatures.**
- `FmmParams::quantize_root_half_width` (`bool`, default `true`), after
  `m2l_op_count_cap` in `src/Beatnik_Params.hpp`; mapped in `buildConfig` to
  `FmmConfig::quantize_root_half_width`.
- `FarFieldDiagnostics::root_width_changes` and `op_cache_overflow_clears`,
  `long long`, cumulative, rank-local, no collective. They are counted at the
  end of `evaluate` for `evaluations > 0`. A width change is a width that
  differs from the member `_root_half_width`. Otherwise an overflow clear is
  `cache_after != cache_before + keys_built_delta`.
- `Impl::readDiagnostics` now returns `double` (`down.root_half_width()`).
- Probe: key `quantize_root_half_width=<0|1>` (default `FmmParams{}`'s),
  echoed in the header line; `[cost]` keys `width_changes`,
  `overflow_clears` and their `.max`, after `occupied_depths`.

**Bugs only running revealed.**
- **The step-1 window cannot reach 90 % retention by construction.** With
  the knob on it reads $r = 0.66$ even after discounting crossings. The first
  evaluation's cold table fill lies inside that window, and the counter never
  counts the first evaluation, so the 0 → w width is not a "change". Canopy's
  own `set_root_half_width` does treat it as one (it clears an empty cache).
  Discounting the cold fill as one full build gives $r = 1.000$. **Ruled:**
  the first evaluation stays uncounted (no rebuild), and the criterion is
  read as excluding the cold fill.

**Results.**
- **Retention, knob on.** Discounted $r \ge 0.993$ in every window after
  step 1 at every level, backend and rank count. `width_changes` is **0** and
  `overflow_clears` **0** in every row, so the rebuilt keys are all
  "remainder" (new keys in a surviving cache): crossings 0, clears 0. The
  caches grew without a clear to 30 975 (L4 np4), 33 534 (L4 np1), 19 797 (L5
  np4), 24 528 (L5 np1), 32 511 (L6 np4), 27 582 (SERIAL np1) and 22 266
  (SERIAL np4) keys, against the cap of 65 536. L4 np1's 33 534 is already past
  `FmmParams`' own default cap of 32 768.
- **Failure direction, knob off.** `[acc]` is bit-identical to T3 at every
  state of every run. Per-run `kb/ev/uo` medians are 0.955–1.000, exactly
  0.667 at step 1. `width_changes` is ≥ 0.96 × evals and `overflow_clears`
  is 0 in every window after step 1. The step wall is within T3's per-state
  spread on HIP except L4 step 1 (+12.5 % np1, −13.4 % np4). Run-level it is
  −3.5 % at L4 np4, outside T3's 0.6 %, in the faster direction. SERIAL np1 is
  +19.2 % on another node (`tuolumne1015` against T3's `1016`), and T3 has no
  SERIAL spread. Gains are therefore quoted only from on/off pairs inside one
  job.
- **Depth (R4).** Knob on: occupied depth starts one level deeper (L4 6, L5
  7, L6 8 throughout). `max_depth 10` never binds, and `fb_dropped` is 0 in
  every row.
- **Accuracy.** Per-level max, knob on against knob off (= T3):
  - L4 **1.402e-3** (step 800) against 1.099e-3; over SERIAL's 500 steps,
    1.134e-3 (step 1) against 8.13e-4.
  - L5 1.212e-3 (step 75) against 1.634e-3.
  - L6 2.248e-3 (step 250) against 2.771e-3.

  **The guard fails at level 4.** Mean over the SMS states: L4
  6.63e-4 → 9.31e-4, L5 1.203e-3 → 1.066e-3, L6 2.301e-3 → 2.146e-3.
  Knob-on `[acc]` is bit-identical across HIP/SERIAL and np1/np4. It is
  geometry (R4), not a wrong-width column (R5): the rise is already full at
  step 1, before any column is reused. Knob on, the error sits flat at about
  0.9–1.1e-3 (L4), 1.0–1.2e-3 (L5) and 2.0–2.25e-3 (L6) along the trajectory.
- **FmmVsDirect: PASS at ranks 1–6** (39/39 checks at np1, 41/41 per rank
  otherwise), HIP, installed default `true`. **Gap:** the binary
  instantiates only `Kokkos::DefaultExecutionSpace`
  (`tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp:1193-1195`), so there is
  no SERIAL FmmVsDirect. SERIAL's evidence is the SERIAL probe `[acc]` rows.

**Per-state numbers.** `op build/eval` is `ilist_s4_op_table_build` per
evaluation. `r (disc.)` is the crossing-discounted retention. Both arms ran
in the same job except SERIAL np4.

**L4_HIP_np4** — summed step wall off 709.1 s → on 372.5 s (-47.5 %); summed op build 268.3 → 0.3 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.726 | 0.628 | 0.636 | -12.4 % | 0.0304 → 0.01893 | 0.662 | 6 |
| 50 | 0.316 | 0.318 | 0.208 | -34.2 % | 0.0399 → 0.00001 | 1.000 | 6 |
| 100 | 0.329 | 0.324 | 0.216 | -34.3 % | 0.0393 → 0.00003 | 1.000 | 6 |
| 150 | 0.348 | 0.347 | 0.213 | -38.8 % | 0.0406 → 0.00001 | 1.000 | 6 |
| 200 | 0.358 | 0.357 | 0.221 | -38.3 % | 0.0395 → 0.00006 | 0.999 | 6 |
| 250 | 0.427 | 0.423 | 0.217 | -49.2 % | 0.0467 → 0.00001 | 1.000 | 6 |
| 300 | 0.494 | 0.491 | 0.214 | -56.7 % | 0.0600 → 0.00002 | 1.000 | 6 |
| 350 | 0.545 | 0.544 | 0.234 | -57.1 % | 0.0673 → 0.00012 | 0.998 | 6.25 |
| 400 | 0.537 | 0.530 | 0.260 | -51.6 % | 0.0643 → 0.00029 | 0.996 | 7 |
| 450 | 0.533 | 0.521 | 0.257 | -51.8 % | 0.0638 → 0.00020 | 0.997 | 7 |
| 500 | 0.557 | 0.543 | 0.272 | -51.2 % | 0.0693 → 0.00018 | 0.998 | 7 |
| 550 | 0.619 | 0.579 | 0.306 | -50.6 % | 0.0748 → 0.00009 | 0.999 | 7 |
| 600 | 0.672 | 0.624 | 0.318 | -52.7 % | 0.0768 → 0.00011 | 0.999 | 7 |
| 650 | 0.689 | 0.642 | 0.344 | -50.1 % | 0.0827 → 0.00007 | 0.999 | 7 |
| 700 | 0.679 | 0.638 | 0.349 | -48.6 % | 0.0826 → 0.00004 | 1.000 | 7 |
| 750 | 0.649 | 0.606 | 0.354 | -45.5 % | 0.0799 → 0.00003 | 1.000 | 7 |
| 800 | 0.648 | 0.634 | 0.351 | -45.8 % | 0.0833 → 0.00004 | 1.000 | 7 |
| 850 | 0.663 | 0.639 | 0.389 | -41.3 % | 0.0817 → 0.00003 | 1.000 | 7 |
| 900 | 0.745 | 0.732 | 0.379 | -49.1 % | 0.0921 → 0.00001 | 1.000 | 7 |
| 950 | 0.783 | 0.770 | 0.395 | -49.6 % | 0.1017 → 0.00002 | 1.000 | 7 |
| 1000 | 0.741 | 0.728 | 0.369 | -50.2 % | 0.0957 → 0.00003 | 1.000 | 7 |
| 1050 | 0.809 | 0.788 | 0.375 | -53.6 % | 0.1018 → 0.00002 | 1.000 | 7 |
| 1100 | 0.829 | 0.801 | 0.408 | -50.8 % | 0.1010 → 0.00004 | 1.000 | 7 |
| 1150 | 0.796 | 0.772 | 0.398 | -50.0 % | 0.0985 → 0.00001 | 1.000 | 7 |
| 1200 | 0.842 | 0.824 | 0.394 | -53.2 % | 0.1058 → 0.00001 | 1.000 | 7 |

**L5_HIP_np4** — summed step wall off 280.4 s → on 179.1 s (-36.1 %); summed op build 91.1 → 0.2 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.149 | 1.069 | 1.156 | +0.6 % | 0.0550 → 0.04097 | 0.666 | 7 |
| 25 | 0.720 | 0.712 | 0.448 | -37.8 % | 0.0751 → 0.00001 | 1.000 | 7 |
| 50 | 0.781 | 0.787 | 0.527 | -32.5 % | 0.0761 → 0.00002 | 1.000 | 7 |
| 75 | 0.787 | 0.785 | 0.560 | -28.8 % | 0.0806 → 0.00004 | 1.000 | 7 |
| 100 | 0.803 | 0.791 | 0.580 | -27.8 % | 0.0799 → 0.00003 | 1.000 | 7 |
| 125 | 0.799 | 0.789 | 0.587 | -26.5 % | 0.0774 → 0.00003 | 1.000 | 7 |
| 150 | 0.881 | 0.869 | 0.610 | -30.8 % | 0.0823 → 0.00003 | 1.000 | 7 |
| 175 | 0.878 | 0.866 | 0.617 | -29.7 % | 0.0860 → 0.00008 | 0.999 | 7 |
| 200 | 0.970 | 0.953 | 0.631 | -34.9 % | 0.1039 → 0.00011 | 0.999 | 7 |
| 225 | 1.061 | 1.051 | 0.657 | -38.1 % | 0.1154 → 0.00007 | 1.000 | 7 |
| 250 | 1.077 | 1.073 | 0.652 | -39.5 % | 0.1279 → 0.00008 | 1.000 | 7 |
| 275 | 1.192 | 1.181 | 0.636 | -46.6 % | 0.1462 → 0.00006 | 1.000 | 7 |
| 300 | 1.372 | 1.345 | 0.633 | -53.9 % | 0.1652 → 0.00012 | 0.999 | 7 |

**L4_HIP_np1** — summed step wall off 1040.6 s → on 484.3 s (-53.5 %); summed op build 549.4 → 0.3 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.381 | 0.428 | 0.438 | +15.0 % | 0.0419 → 0.02908 | 0.667 | 6 |
| 50 | 0.343 | 0.346 | 0.243 | -29.2 % | 0.0561 → 0.00001 | 1.000 | 6 |
| 100 | 0.352 | 0.359 | 0.263 | -25.3 % | 0.0561 → 0.00001 | 1.000 | 6 |
| 150 | 0.384 | 0.390 | 0.269 | -29.9 % | 0.0605 → 0.00000 | 1.000 | 6 |
| 200 | 0.394 | 0.399 | 0.268 | -32.0 % | 0.0609 → 0.00000 | 1.000 | 6 |
| 250 | 0.490 | 0.514 | 0.246 | -49.8 % | 0.0858 → 0.00000 | 1.000 | 6 |
| 300 | 0.676 | 0.680 | 0.242 | -64.2 % | 0.1211 → 0.00002 | 1.000 | 6 |
| 350 | 0.810 | 0.811 | 0.252 | -68.9 % | 0.1409 → 0.00019 | 0.998 | 7 |
| 400 | 0.755 | 0.762 | 0.267 | -64.6 % | 0.1308 → 0.00028 | 0.998 | 7 |
| 450 | 0.716 | 0.733 | 0.289 | -59.6 % | 0.1282 → 0.00034 | 0.998 | 7 |
| 500 | 0.781 | 0.801 | 0.320 | -59.0 % | 0.1418 → 0.00016 | 0.999 | 7 |
| 550 | 0.870 | 0.887 | 0.385 | -55.7 % | 0.1540 → 0.00008 | 1.000 | 7 |
| 600 | 0.912 | 0.925 | 0.436 | -52.2 % | 0.1627 → 0.00005 | 1.000 | 7 |
| 650 | 0.918 | 0.932 | 0.487 | -46.9 % | 0.1645 → 0.00006 | 1.000 | 7 |
| 700 | 0.936 | 0.956 | 0.502 | -46.4 % | 0.1683 → 0.00004 | 1.000 | 7 |
| 750 | 0.962 | 0.968 | 0.520 | -45.9 % | 0.1729 → 0.00002 | 1.000 | 7 |
| 800 | 1.016 | 1.023 | 0.481 | -52.7 % | 0.1846 → 0.00004 | 1.000 | 7 |
| 850 | 1.029 | 1.036 | 0.517 | -49.8 % | 0.1850 → 0.00002 | 1.000 | 7 |
| 900 | 1.129 | 1.147 | 0.501 | -55.6 % | 0.2066 → 0.00001 | 1.000 | 7 |
| 950 | 1.181 | 1.201 | 0.517 | -56.2 % | 0.2144 → 0.00001 | 1.000 | 7 |
| 1000 | 1.097 | 1.100 | 0.504 | -54.1 % | 0.1999 → 0.00001 | 1.000 | 7 |
| 1050 | 1.230 | 1.243 | 0.515 | -58.1 % | 0.2244 → 0.00000 | 1.000 | 7 |
| 1100 | 1.199 | 1.221 | 0.571 | -52.4 % | 0.2173 → 0.00002 | 1.000 | 7 |
| 1150 | 1.139 | 1.148 | 0.550 | -51.7 % | 0.2044 → 0.00000 | 1.000 | 7 |
| 1200 | 1.225 | 1.229 | 0.537 | -56.2 % | 0.2216 → 0.00001 | 1.000 | 7 |

**L5_HIP_np1** — summed step wall off 376.9 s → on 293.5 s (-22.1 %); summed op build 147.6 → 0.2 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.040 | 1.027 | 1.226 | +17.9 % | 0.0765 → 0.05778 | 0.667 | 7 |
| 25 | 0.922 | 0.918 | 0.770 | -16.5 % | 0.1156 → 0.00002 | 1.000 | 7 |
| 50 | 1.001 | 0.992 | 0.922 | -7.9 % | 0.1161 → 0.00004 | 1.000 | 7 |
| 75 | 1.032 | 1.030 | 0.980 | -5.0 % | 0.1249 → 0.00003 | 1.000 | 7 |
| 100 | 1.031 | 1.035 | 0.998 | -3.2 % | 0.1240 → 0.00002 | 1.000 | 7 |
| 125 | 1.002 | 1.004 | 0.998 | -0.4 % | 0.1164 → 0.00001 | 1.000 | 7 |
| 150 | 1.065 | 1.080 | 1.043 | -2.1 % | 0.1286 → 0.00002 | 1.000 | 7 |
| 175 | 1.103 | 1.136 | 1.028 | -6.8 % | 0.1422 → 0.00011 | 0.999 | 7 |
| 200 | 1.257 | 1.296 | 1.029 | -18.1 % | 0.1700 → 0.00013 | 0.999 | 7 |
| 225 | 1.351 | 1.398 | 1.046 | -22.6 % | 0.1940 → 0.00005 | 1.000 | 7 |
| 250 | 1.472 | 1.524 | 1.002 | -31.9 % | 0.2153 → 0.00004 | 1.000 | 7 |
| 275 | 1.733 | 1.755 | 0.965 | -44.3 % | 0.2503 → 0.00003 | 1.000 | 7 |
| 300 | 1.870 | 1.902 | 0.942 | -49.6 % | 0.2725 → 0.00005 | 1.000 | 7 |

**L6_HIP_np4** — summed step wall off 538.4 s → on 427.3 s (-20.6 %); summed op build 164.2 → 0.3 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.178 | 2.161 | 2.239 | +2.8 % | 0.1268 → 0.08543 | 0.666 | 8 |
| 25 | 1.927 | 1.929 | 1.637 | -15.0 % | 0.1931 → 0.00002 | 1.000 | 8 |
| 50 | 1.985 | 1.965 | 1.645 | -17.1 % | 0.1938 → 0.00003 | 1.000 | 8 |
| 75 | 2.032 | 2.021 | 1.758 | -13.5 % | 0.2005 → 0.00003 | 1.000 | 8 |
| 100 | 2.090 | 2.070 | 1.823 | -12.8 % | 0.2055 → 0.00001 | 1.000 | 8 |
| 125 | 2.058 | 2.038 | 1.793 | -12.9 % | 0.2007 → 0.00005 | 1.000 | 8 |
| 150 | 2.176 | 2.158 | 1.696 | -22.1 % | 0.2207 → 0.00005 | 1.000 | 8 |
| 175 | 2.217 | 2.197 | 1.679 | -24.3 % | 0.2229 → 0.00003 | 1.000 | 8 |
| 200 | 2.324 | 2.323 | 1.682 | -27.6 % | 0.2413 → 0.00007 | 1.000 | 8 |
| 225 | 2.323 | 2.321 | 1.692 | -27.2 % | 0.2456 → 0.00006 | 1.000 | 8 |
| 250 | 2.482 | 2.504 | 1.662 | -33.0 % | 0.2683 → 0.00004 | 1.000 | 8 |

**L4_SERIAL_np1** — summed step wall off 1026.5 s → on 806.4 s (-21.4 %); summed op build 138.0 → 0.2 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.377 | 1.557 | 1.726 | +25.3 % | 0.0398 → 0.02754 | 0.667 | 6 |
| 25 | 1.515 | 1.842 | 1.650 | +8.9 % | 0.0524 → 0.00001 | 1.000 | 6 |
| 50 | 1.527 | 1.853 | 1.492 | -2.3 % | 0.0528 → 0.00001 | 1.000 | 6 |
| 75 | 1.434 | 1.912 | 1.554 | +8.4 % | 0.0523 → 0.00002 | 1.000 | 6 |
| 100 | 1.481 | 1.862 | 1.553 | +4.9 % | 0.0523 → 0.00001 | 1.000 | 6 |
| 125 | 1.436 | 1.854 | 1.640 | +14.2 % | 0.0524 → 0.00001 | 1.000 | 6 |
| 150 | 1.534 | 1.919 | 1.439 | -6.2 % | 0.0596 → 0.00000 | 1.000 | 6 |
| 175 | 1.463 | 1.867 | 1.444 | -1.3 % | 0.0534 → 0.00000 | 1.000 | 6 |
| 200 | 1.430 | 1.933 | 1.602 | +12.0 % | 0.0595 → 0.00000 | 1.000 | 6 |
| 225 | 1.538 | 1.887 | 1.468 | -4.6 % | 0.0638 → 0.00000 | 1.000 | 6 |
| 250 | 1.592 | 1.909 | 1.456 | -8.5 % | 0.0986 → 0.00000 | 1.000 | 6 |
| 275 | 1.632 | 2.096 | 1.617 | -0.9 % | 0.0956 → 0.00003 | 1.000 | 6 |
| 300 | 1.952 | 2.106 | 1.579 | -19.1 % | 0.1258 → 0.00002 | 1.000 | 6 |
| 325 | 1.936 | 2.097 | 1.838 | -5.1 % | 0.1372 → 0.00031 | 0.997 | 7 |
| 350 | 2.058 | 2.376 | 1.843 | -10.4 % | 0.1364 → 0.00005 | 0.999 | 7 |
| 375 | 2.393 | 2.400 | 1.843 | -23.0 % | 0.1311 → 0.00035 | 0.997 | 7 |
| 400 | 1.874 | 2.227 | 1.799 | -4.0 % | 0.1087 → 0.00020 | 0.998 | 7 |
| 425 | 1.875 | 2.093 | 1.836 | -2.1 % | 0.1184 → 0.00026 | 0.998 | 7 |
| 450 | 1.784 | 2.126 | 1.586 | -11.1 % | 0.1153 → 0.00037 | 0.997 | 7 |
| 475 | 1.891 | 2.293 | 1.553 | -17.9 % | 0.1279 → 0.00015 | 0.999 | 7 |
| 500 | 2.027 | 2.419 | 1.462 | -27.9 % | 0.1468 → 0.00016 | 0.999 | 7 |

**L4_SERIAL_np4** — summed step wall off 1875.5 s → on 1736.5 s (-7.4 %); summed op build 78.2 → 0.2 s.

| step | T3 s/step | off s/step | on s/step | on vs T3 | op build/eval off → on (s) | on: r (disc.) | on: depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.820 | 2.686 | 2.754 | -2.3 % | 0.0280 → 0.01854 | 0.662 | 6 |
| 25 | 3.483 | 3.554 | 3.526 | +1.2 % | 0.0415 → 0.00001 | 1.000 | 6 |
| 50 | 3.706 | 3.553 | 3.536 | -4.6 % | 0.0385 → 0.00001 | 1.000 | 6 |
| 75 | 3.695 | 3.297 | 3.560 | -3.7 % | 0.0387 → 0.00002 | 1.000 | 6 |
| 100 | 3.581 | 3.467 | 3.587 | +0.2 % | 0.0379 → 0.00004 | 0.999 | 6 |
| 125 | 3.666 | 3.631 | 3.598 | -1.9 % | 0.0381 → 0.00001 | 1.000 | 6 |
| 150 | 3.732 | 3.601 | 3.379 | -9.5 % | 0.0413 → 0.00001 | 1.000 | 6 |
| 175 | 3.821 | 3.807 | 3.304 | -13.5 % | 0.0386 → 0.00001 | 1.000 | 6 |
| 200 | 3.658 | 3.785 | 3.618 | -1.1 % | 0.0386 → 0.00011 | 0.998 | 6 |
| 225 | 3.636 | 3.863 | 3.350 | -7.9 % | 0.0403 → 0.00001 | 1.000 | 6 |
| 250 | 3.928 | 3.822 | 3.157 | -19.6 % | 0.0501 → 0.00000 | 1.000 | 6 |
| 275 | 3.553 | 3.716 | 3.189 | -10.2 % | 0.0541 → 0.00002 | 1.000 | 6 |
| 300 | 3.737 | 3.744 | 3.390 | -9.3 % | 0.0620 → 0.00002 | 1.000 | 6 |
| 325 | 3.886 | 3.851 | 3.476 | -10.6 % | 0.0665 → 0.00021 | 0.996 | 6.25 |
| 350 | 3.787 | 3.729 | 3.498 | -7.6 % | 0.0654 → 0.00004 | 0.999 | 6.25 |
| 375 | 3.986 | 3.804 | 3.468 | -13.0 % | 0.0658 → 0.00045 | 0.993 | 7 |
| 400 | 3.778 | 3.895 | 3.533 | -6.5 % | 0.0596 → 0.00014 | 0.998 | 7 |
| 425 | 3.791 | 3.849 | 3.524 | -7.0 % | 0.0634 → 0.00017 | 0.997 | 7 |
| 450 | 3.947 | 3.925 | 3.540 | -10.3 % | 0.0633 → 0.00022 | 0.997 | 7 |
| 475 | 4.045 | 4.051 | 3.618 | -10.6 % | 0.0660 → 0.00018 | 0.998 | 7 |
| 500 | 4.228 | 4.109 | 3.641 | -13.9 % | 0.0736 → 0.00017 | 0.998 | 7 |

**L4 `[acc]`** (knob off = T3 at every state, bit-identical)

| step | T3 | O1 (on) | change |
| --- | --- | --- | --- |
| 1 | 5.0076e-04 | 1.1338e-03 | +126.4 % |
| 25 | 5.0618e-04 | 1.1244e-03 | +122.1 % |
| 50 | 5.0700e-04 | 1.1144e-03 | +119.8 % |
| 75 | 5.1582e-04 | 1.0797e-03 | +109.3 % |
| 100 | 5.1726e-04 | 1.0380e-03 | +100.7 % |
| 125 | 5.2790e-04 | 1.0660e-03 | +101.9 % |
| 150 | 6.0878e-04 | 1.0114e-03 | +66.1 % |
| 175 | 5.0836e-04 | 9.5737e-04 | +88.3 % |
| 200 | 5.4742e-04 | 8.7885e-04 | +60.5 % |
| 225 | 5.8915e-04 | 8.6536e-04 | +46.9 % |
| 250 | 6.0439e-04 | 8.1017e-04 | +34.0 % |
| 275 | 5.3496e-04 | 8.3881e-04 | +56.8 % |
| 300 | 5.1550e-04 | 8.5650e-04 | +66.1 % |
| 325 | 6.0725e-04 | 9.3266e-04 | +53.6 % |
| 350 | 7.6153e-04 | 9.4444e-04 | +24.0 % |
| 375 | 8.1324e-04 | 1.0166e-03 | +25.0 % |
| 400 | 5.2754e-04 | 1.0679e-03 | +102.4 % |
| 425 | 4.5806e-04 | 1.0547e-03 | +130.3 % |
| 450 | 5.8536e-04 | 4.5554e-04 | -22.2 % |
| 475 | 6.2214e-04 | 3.6193e-04 | -41.8 % |
| 500 | 6.3220e-04 | 1.1160e-03 | +76.5 % |
| 550 | 6.1619e-04 | 1.1877e-03 | +92.8 % |
| 600 | 8.1491e-04 | 8.1949e-04 | +0.6 % |
| 650 | 6.6159e-04 | 6.6914e-04 | +1.1 % |
| 700 | 7.5563e-04 | 6.2743e-04 | -17.0 % |
| 750 | 8.0964e-04 | 7.5714e-04 | -6.5 % |
| 800 | 7.7639e-04 | 1.4025e-03 | +80.6 % |
| 850 | 7.4588e-04 | 7.7219e-04 | +3.5 % |
| 900 | 6.5644e-04 | 8.9084e-04 | +35.7 % |
| 950 | 8.2779e-04 | 9.3676e-04 | +13.2 % |
| 1000 | 1.0994e-03 | 8.6207e-04 | -21.6 % |
| 1050 | 5.8722e-04 | 7.4657e-04 | +27.1 % |
| 1100 | 5.7386e-04 | 1.1242e-03 | +95.9 % |
| 1150 | 7.4691e-04 | 1.0894e-03 | +45.9 % |
| 1200 | 6.0176e-04 | 9.6332e-04 | +60.1 % |

**L5 `[acc]`** (knob off = T3 at every state, bit-identical)

| step | T3 | O1 (on) | change |
| --- | --- | --- | --- |
| 1 | 1.6337e-03 | 9.9253e-04 | -39.2 % |
| 25 | 1.6290e-03 | 1.1482e-03 | -29.5 % |
| 50 | 9.4280e-04 | 1.0225e-03 | +8.5 % |
| 75 | 1.3560e-03 | 1.2115e-03 | -10.7 % |
| 100 | 1.5696e-03 | 1.2073e-03 | -23.1 % |
| 125 | 1.1468e-03 | 1.0647e-03 | -7.2 % |
| 150 | 1.3600e-03 | 1.1266e-03 | -17.2 % |
| 175 | 1.1184e-03 | 1.1143e-03 | -0.4 % |
| 200 | 9.4368e-04 | 1.0901e-03 | +15.5 % |
| 225 | 8.3459e-04 | 9.8452e-04 | +18.0 % |
| 250 | 8.8988e-04 | 9.9999e-04 | +12.4 % |
| 275 | 1.3101e-03 | 9.9609e-04 | -24.0 % |
| 300 | 9.0874e-04 | 9.0339e-04 | -0.6 % |

**L6 `[acc]`** (knob off = T3 at every state, bit-identical)

| step | T3 | O1 (on) | change |
| --- | --- | --- | --- |
| 1 | 2.7071e-03 | 2.2306e-03 | -17.6 % |
| 25 | 2.7075e-03 | 2.1532e-03 | -20.5 % |
| 50 | 2.7711e-03 | 2.2300e-03 | -19.5 % |
| 75 | 2.6176e-03 | 2.1874e-03 | -16.4 % |
| 100 | 2.5796e-03 | 2.0923e-03 | -18.9 % |
| 125 | 2.3586e-03 | 2.0673e-03 | -12.4 % |
| 150 | 2.1782e-03 | 2.0423e-03 | -6.2 % |
| 175 | 2.0270e-03 | 2.0557e-03 | +1.4 % |
| 200 | 1.9477e-03 | 2.1975e-03 | +12.8 % |
| 225 | 1.8242e-03 | 2.1028e-03 | +15.3 % |
| 250 | 1.5914e-03 | 2.2484e-03 | +41.3 % |

**Affects:**
- **O2** — its trigger (crossing rebuilds over 5 % of SMS wall) is **not
  met**: zero octave crossings in 1200 (L4), 300 (L5) and 250 (L6) steps at
  every rank count.
- **O4** — its trigger is **not met**. After O1, `ilist_s4_op_table_build`
  is 0.2–0.3 s summed per run, under 0.1 % of SMS step wall.
- **O8** — the operator-table copy is unchanged by retention. It now carries
  a larger share of a shorter step.
- **V1** — **triggered, before the next O task**: `[acc]` moved more than
  20 % at states at every level (L4 −42 to +130 %, L5 −39 to +18 %, L6
  −20.5 to +41.3 %). The milestone FMM members leave this knob at its
  default, so they now run quantized. Their claim A (≤ τ_A at 81 states to
  step 2000, level 4) has not been re-run, and the SMS already reaches
  1.402e-3 of the 1.5e-3 at level 4.
- **Overflow-clear follow-up** — none occurred at cap 65 536, so no
  follow-up is opened. But a surviving cache accumulates keys: at `FmmParams`'
  default cap 32 768, L4 np1 would have overflowed by step 1200. A larger
  default cap, or eviction instead of a full clear, is the follow-up if a
  run at the default shows clears.
- **Every later O task** — the accuracy baseline is now O1's maxima: L4
  1.402e-3, L5 1.212e-3, L6 2.248e-3.

## Decision: adopt O1 despite the level-4 `[acc]` rise

The user ruled after O1's measurements, on being shown the speedup and the
accuracy change side by side:
- **The accuracy trade-off is worth the speedup.** `quantize_root_half_width`
  is adopted with default `true`, and O1 is DONE.
- **This is an exception for O1, not a change to the guard.** "Decision: no
  accuracy regression in any O task" still binds every later task. Its
  "before" is now O1's values: L4 1.402e-3 (1.134e-3 on SERIAL's 500-step
  SMS), L5 1.212e-3, L6 2.248e-3.
- **What was traded.** L4's worst SMS state rose 28 % (1.099e-3 to
  1.402e-3), still under τ_A, and its mean rose 40 %. L5's and L6's worst
  states fell 26 % and 19 %. In return the summed HIP step wall fell 47–53 %
  at L4, 22–36 % at L5 and 21 % at L6.
- **The step-1 retention reading.** The width counter keeps not counting the
  first evaluation; the 90 % retention criterion excludes the cold fill.

**Affects:**
- O2–O10 — the accuracy baseline is O1's maxima above, compared exactly.
- V1 — must re-assert claim A on the milestone tier with the quantized
  default; level 4 has only a 7 % margin on the SMS states.
- README — "FMM tunables" carries the knob's row; Known Issues records the
  exception and the un-re-run milestone claim.

## Decision: defer V1; a milestone failure from O1 is accepted

The user ruled after O1 closed:
- **The milestone FMM members may fail claim A because of O1, and that is
  acceptable.** They inherit `quantize_root_half_width = true` and have not
  been re-run. Level 4's worst SMS state is 1.402e-3, 7 % under `kTauA`
  1.5e-3, and the milestone checks 81 states to step 2000, past the SMS's
  1200.
- **If they fail, increase the allowable error.** Raise `kTauA`
  (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp`) to cover the
  measured worst. Record the run, the step and the new value here. Do not
  turn quantization off to make the old bound pass. This overrides that
  file's "do not loosen this literal" for this cause, and its comment now
  says so.
- **V1 is deferred.** O1's 20 % rule triggered V1 before the next O task. The
  user chose not to run the multi-hour milestone tier now and to continue
  with optimizations. V1 runs at the end of Phase O as originally sequenced.

Recorded where a session cannot miss it: `CLAUDE.md` (milestone tier),
the `kTauA` comment, V1's entry and the status line in
`canopy_profiling.md`, and README Known Issues.

**Affects:**
- V1 — runs at the end of Phase O. Its step 1 may need a raised `kTauA`;
  "16/16 green" is then met at the raised bound.
- Every O task until V1 — no milestone run is required by the 20 % rule. The
  per-level SMS accuracy guard (O1's maxima as baseline) still binds.

## Decision: O2 and O4 not taken; O8 before O5

The user ruled after O1, on its measured triggers and post-O1 shares:
- **O2 NOT TAKEN.** Zero octave crossings in every O1 knob-on run, so the
  crossing-rebuild cost is 0 % against its 5 % trigger.
- **O4 NOT TAKEN.** `ilist_s4_op_table_build` is 0.02–0.11 % of summed SMS
  step wall after O1, against its 5 % trigger.
- **O8 runs before O5.** These are post-O1 shares of HIP step wall, as
  medians of `3 × phase / step_wall` over each knob-on run's `[cost]` rows:

  | target | L4 np1 | L4 np4 | L5 np1 | L5 np4 | L6 np4 |
  | --- | --- | --- | --- | --- | --- |
  | `ilist_s4_op_table_copy` (O8) | 43.2 | 28.4 | 19.8 | 20.8 | 17.8 |
  | `ff_maintain` (O5) | 17.7 | 23.2 | 34.9 | 35.3 | 42.6 |
  | `comm_plan_build` (in O5) | 10.5 | 9.3 | 26.2 | 22.2 | 28.2 |
  | `ilist_s3_classify_pairs` (O9) | 17.9 | 6.1 | 21.0 | 14.8 | 14.6 |

  On SERIAL the forward and reverse distributors (O7) are 67 % of the step
  at np1 and 92 % at np4.

  O8 leads at level 4, and O1 makes it more tractable: the cache now persists
  across evaluations, so most of the per-evaluation host copy and full
  upload is redundant. Its exit criterion is a bitwise-equal table, so it
  cannot move `[acc]`. O5 leads at levels 5 and 6 and comes next.

**Affects:**
- O8 — next. Its dependency "O4 resolved" is met.
- O5 — after O8; its T3 shares are superseded by the table above.

## O8

**Provenance, pinned.**
- **Code.** Canopy `046fdc6` "DownwardSweep: assemble the M2L operator table
  on the device (O8)" plus `56f3932` (the test-job script only). Every job ran
  HEAD `56f3932` with **0 porcelain lines**. Beatnik `40fc8c7`
  "canopy_profiling O8: device_op_table_assembly from Beatnik", logged by
  every SMS job (dirty 6 files, none compiled: `.claude/settings.json`, this
  task's design doc, and four untracked notes/scripts).
- **Toolchain.** `canopy@develop +profiling profiling_level=2`, cce@20.0.0,
  dev env `~/spack_envs/tuolumne_beatnik`, `profile.local.sh` (spack mode).
  Canopy standalone: `build-profiling/` under `tuolumne_trilinos`; the four
  stem targets built in 7 min 5 s on the login node.
- **Install.** One `spack install`, 12:33:05–12:44:12, **11 min 7 s** wall,
  108 CPU-min (Beatnik 10 min 35 s; Canopy and tessera re-staged in about
  30 s). Every `.cpp` under `src tests examples` was touched first
  (`systems/tuolumne/claude.md` §3). The installed `Canopy_Solver.hpp`,
  `Canopy_DownwardSweep.hpp` and `Beatnik_Params.hpp` carry the knob.
- **Configuration.** As O1: CartesianTaylor order 3, θ 0.3, `ncrit 8`,
  `m2l_op_count_cap 65536`, `canopy_report 0`. Both arms pass
  `quantize_root_half_width=1` and `device_op_table_assembly=0|1`
  explicitly. Binding is `canopy_profiling_baseline.flux`'s `launch()`,
  unmodified.

All jobs `pdebug`, rc 0, no `SKIPPED` or `FAIL`, every `[cpb] total wall`
present. The reducer's count check (`fmm_profile_reduce.py --format probe
--states`) passed every `[cost]` row with 0 excluded.

| job | name | node | rows (label level steps every) | launch walls | queue wait | job wall |
| --- | --- | --- | --- | --- | --- | --- |
| `f3dvCN5tgncw` | `canopy-cprof-o8` | — | Canopy stems + stale check | — | 1 s | 10.0 min |
| `f3dvNa7ovPnF` | `beatnik_o8_hip_np1` | 1016 | `smsl5a1`, `smsl5a0` 5 300 25; `smsl4a1`, `smsl4a0` 4 1200 50 | 247 + 301 + 256 + 499 s | 1 s | 22.2 min |
| `f3dvNaGqG2Fy` | `beatnik_o8_hip_np4` | 1042 | same rows at np4 | 139 + 195 + 236 + 376 s | 49 s | 16.1 min |
| `f3dvNaQDiShd` | `beatnik_o8_serial_np1` | 1011 | `smsl4a1`, `smsl4a0` 4 500 25 | 801 + 838 s | 80 s | 27.7 min |
| `f3dvNaYRcUAK` | `beatnik_o8_serial_np4_a1` | 1015 | `smsl4a1` 4 500 25 | 1808 s | 177 s | 30.5 min |
| `f3dvNafzSoaP` | `beatnik_o8_serial_np4_a0` | 1014 | `smsl4a0` 4 500 25 | 1681 s | 177 s | 28.4 min |

`a1` is the knob on (the new path), `a0` off (the host-copy path). O1 bases,
all knob-on (`q1`): `f3dk9xY4X4aT` (HIP np1, node 1026), `f3dk9xfGba3h` (HIP
np4, 1019), `f3dk9xwujWWB` (SERIAL np1, 1015), `f3dk9yE6BftT` (SERIAL np4,
1012). The SMS jobs were submitted together at 12:44:33; the Canopy job went
out at 12:22, before the install. It uses the trilinos env and
`build-profiling/`, not the dev env, so it could not collide with the install,
and a failure there would have saved one.

**Decisions, made before this task and recorded here.**
- **O8 gets a runtime knob,** so the failure-direction arm runs in the same
  job and on the same node as the change. O1 showed cross-node walls moving
  19 % on SERIAL np1, where T3 measured no spread. The knob:
  - Canopy `FmmConfig::device_op_table_assembly`, default **on**, beside
    `quantize_root_half_width`, reaching `DownwardSweep` through
    `set_device_op_table_assembly()` in the `Solver` constructor, as
    `m2l_op_count_cap` does (`quantize_root_half_width` itself goes to
    `TreeBuilder`, not the sweep);
  - Beatnik `FmmParams::device_op_table_assembly`, default on, mapped in
    `buildConfig`;
  - probe key `device_op_table_assembly=<0|1>`, echoed in `[fmmcost]
    header`;
  - a README "FMM tunables" row, and the Canopy README `FmmConfig` table row.

  Off is the previous host-copy path, unchanged.
- **The "before" is O1's knob-on SMS**, with `quantize_root_half_width=1`
  passed explicitly in both arms.
- **Accuracy baseline is O1's per-level maxima, compared exactly:** L4
  1.402e-3 (1.134e-3 over SERIAL's 500 steps), L5 1.212e-3. The table must
  be bitwise equal, so `[acc]` must be bit-identical to O1 at every state.
- **No level-6 rows.** O8 does not change the tree.
- **The milestone tier is not run** (V1 deferred), and `kTauA` is untouched.

**Decisions made in this task.**
- **Device copy of the cache, not "upload only new columns into the table".**
  `_m2l_op_cache_device` (`memory_space`, same extents as the host cache)
  holds the first `_m2l_op_cache_device_size` columns bit for bit. Each build
  uploads only `[device_size, cache_size)` (one contiguous LayoutLeft
  `deep_copy`), uploads the `n_unique_ops` source-column indices looked up in
  `_m2l_op_cache_index` (never by position: `ops[]` order is the stage-3 merge
  order), and gathers with a flat-index functor (`M2LOpTableGather`, t
  fastest), then fences inside the `ilist_s4_op_table_copy` timer so the
  phase stays comparable with the host path, whose `deep_copy` fenced.
- **Lifecycle.** `clear_m2l_op_cache()` drops the device copy with the host
  cache. That covers all three clear sites: `set_m2l_kernel_params`,
  `set_root_half_width`, and stage 4's cap checks (`_m2l_op_cache_size >
  effective_op_cap` and the overflow refill). `_grow_m2l_op_cache()` grows the
  device copy alongside, preserving its uploaded prefix device to device.
- **A disagreement throws instead of re-uploading.** `_sync_m2l_op_cache_device`
  allocates only when there is no copy. Extents that differ from the host
  cache, or a device size above the host size, throw `std::logic_error`
  ("out of step with the host cache"). A full re-upload on mismatch would
  have hidden a missed invalidation.
- **Host backends keep no copy.** Where
  `SpaceAccessibility<execution_space, HostSpace>::accessible` (SERIAL), "on"
  gathers straight from the host cache with the same functor. SERIAL
  therefore never exercises the device copy; HIP covers it, in the unit case
  and in the SMS.
- **Toggling the knob neither dirties the interaction list nor empties the
  cache.** The table is identical either way.
- **The baseline script's knob column accepts a comma-separated list**
  (`${_knob//,/ }`), so one row passes both keys. Single-knob rows (T3, O1)
  are unchanged.

**Signatures.**
- Canopy: `FmmConfig::device_op_table_assembly` (`bool`, default `true`);
  `DownwardSweep::set_device_op_table_assembly(bool)` and
  `device_op_table_assembly()`; private `_m2l_op_cache_device`,
  `_m2l_op_cache_device_size`, `_sync_m2l_op_cache_device()`,
  `M2LOpTableGather<CacheView>`.
- Beatnik: `FmmParams::device_op_table_assembly` (`bool`, default `true`)
  after `quantize_root_half_width` in `src/Beatnik_Params.hpp`, mapped in
  `buildConfig`.
- Probe: key `device_op_table_assembly=<0|1>` (default `FmmParams{}`'s),
  echoed in the header after `quantize_root_half_width`. No `[cost]` field
  changed.

**Canopy unit case.** `DownwardSweepCaching.deviceOpTableAssemblyBitwise`
(`tests/tstDownwardSweep.hpp`) builds one tree (the caching fixture,
LaplaceKernel P = 6, 1000 particles) through two sweeps, knob off and on, in
lock-step. After each build it matches columns **by key** and `memcmp`s every
column. The steps:
1. a first fill at cap 1;
2. the default cap, so the cache **grows** (`_grow_m2l_op_cache` with a
   non-empty prefix; asserted `cache_size > 1`);
3. a rebuild that realizes no new key (asserted `keys_built` unchanged);
4. an explicit `clear_m2l_op_cache()` and refill;
5. a cap of half the cache, which empties it at stage 4's cap check.

At np1 that is 612 keys, then 306.

**Canopy tests, job `f3dvCN5tgncw`: all PASS.** `DownwardSweep` and
`CartesianTaylorSolve` at SERIAL np 1–6 and HIP np 1–4, every entry
`completed` within budget. The new case is `[ OK ]` on every rank of every
DownwardSweep entry.

**Stale-column check, by hand, not committed.** Two HIP DownwardSweep
binaries, built from deliberately broken sources and run in the same job on
the bitwise case only, at np 1 and 2. **Both fail.**
- **`.stale_grow`** (the device copy's growth skips its prefix copy) **fails
  the bitwise comparison** itself: `countOpTableMismatches` = 1 at np1 (at
  two stages), and 4 failures over the two ranks at np2.
- **`.stale_clear`** (`clear_m2l_op_cache()` leaves the device copy in place)
  fails, but **not** at the bitwise comparison. After the explicit clear, the
  refill of the same tree puts the same keys in the same positions (612,
  merge order), so the stale columns happen to hold the right bits and step 4
  compares equal. It fails at step 5: the refill there is 306 columns, and
  `_grow_m2l_op_cache` copies the stale 612-column prefix into the new
  306-column view, which HIP rejects (`hipMemcpyAsync ... hipErrorInvalidValue`,
  an exception in the test body).

  So a skipped clear is caught as a hard error, not as a wrong table. A
  skipped clear whose refill differed in key order but not in extent would
  reach the bitwise comparison. This fixture cannot produce one, because one
  tree's merge order is stable.

**Bugs only running revealed.**
- **The device gather is slightly slower than the host loop on SERIAL np1.**
  It is a copy-phase regression, not a step-wall one. `ilist_s4_op_table_copy`
  is consistently 3–5 % slower per evaluation knob-on (e.g. 14.08 → 14.80 ms
  at step 25), plus outlier windows (steps 125, 325, 350: +6 to +9 ms). Summed
  over the run it is 32.4 → 35.9 s (+11 %), about +0.45 % of the 775 s step
  wall. The same-job step wall is nonetheless 4.9 % *lower* knob-on, and
  SERIAL's per-state spread here is ±25 %, so it is not visible at step
  level. SERIAL np4's copy is unchanged (16.7 → 16.8 s). The likely cause is
  the flat-index functor's per-element 64-bit div/mod, where the old host
  loop indexed directly. The fix is a column-per-iteration functor on host
  backends. Not done in O8: it would need a second install.
- **Step 1 at L5 is 25–30 % slower knob-on, and it is launch order, not the
  knob.** The L5 `a1` run was the first launch in each HIP job. Its step-1
  excess sits in `m2l_kernel` (+0.28 s/step at np4) and `upward_total`
  (+0.17 s), which O8 does not touch, while its s4 copy is 5.2 against
  42.7 ms. At L4, where `a1` ran third, step 1 is 14–24 % *faster* knob-on.

**Results.**
- **Gain (on against off, same job).** Summed SMS step wall:
  - L4 HIP np1 **−49.0 %** (490.8 → 250.5 s), np4 **−38.0 %** (366.0 →
    226.9 s);
  - L5 HIP np1 **−22.2 %** (292.0 → 227.3 s), np4 **−30.5 %** (181.4 →
    126.1 s).

  Per state after step 1: L4 np1 −35.6 to −54.2 %, np4 −28.8 to −44.3 %; L5
  np1 −18.3 to −25.8 %, np4 −28.9 to −35.5 %. Every HIP state after step 1
  clears the per-state thresholds (9 % L4, 14 % L5) and every run the
  run-level ones. SERIAL np1 −4.9 % (same job) and np4 +7.6 % (two jobs, two
  nodes) are not claimed: T3 measured no SERIAL spread, and O1 saw 19 %
  across nodes.
- **The targeted phase.** Summed `ilist_s4_op_table_copy`, off → on: L4 np1
  202.1 → 5.15 s (−97.5 %), L4 np4 115.9 → 1.96 s (−98.3 %), L5 np1 59.9 →
  5.31 s (−91.1 %), L5 np4 39.4 → 0.95 s (−97.6 %). Per evaluation it is
  0.27–0.89 ms at np4 against 17–49 ms, and grows slowly along the trajectory
  with `n_unique_ops`. Step 1 holds the cold fill's full upload (1.5–5.2 ms).
- **`ilist_s5_device_upload` did not grow.** Summed, off → on: L4 np1 1.40 →
  1.21 s, L4 np4 0.82 → 0.76, L5 np1 1.53 → 1.21, L5 np4 0.45 → 0.36, SERIAL
  np1 1.36 → 1.35, np4 0.72 → 0.76. It is 0.0–0.5 % of the step in every arm.
  Stage 5's uploads were not absorbing deferred work from stage 4.
- **Accuracy: bit-identical.** `[acc]` is identical, as printed, to O1's
  knob-on value at **every** state of **every** run, in **both** arms: 25
  states × 2 HIP runs at L4, 13 × 2 at L5, 21 × 2 SERIAL. Per-level maxima
  are unchanged: L4 **1.4025e-3** (step 800), SERIAL's 500 steps 1.1338e-3
  (step 1), L5 **1.2115e-3** (step 75). No state moved, so V1 is not
  triggered by O8.
- **Failure direction (off against O1's on).**
  - **`[acc]`:** bit-identical at every state.
  - **Step wall, run level:** L4 np1 +1.3 %, np4 −1.8 %; L5 np1 −0.5 %,
    np4 +1.3 %. L5 is within T3's 1.7 %. L4 is outside T3's 0.6 % in both
    directions, as O1's own control was (−3.5 % at L4 np4), on different
    nodes from O1's (1016 against 1026, 1042 against 1019).
  - **Step wall, per state after step 1:** L4 −5.6 to +4.0 % (np1) and −3.3
    to +0.6 % (np4); L5 −4.1 to +1.6 % and −1.4 to +3.5 %. All within T3's
    9 % and 14 %.
  - **Step 1 off against O1:** +0.1 % (L4 np1), −1.2 % (L5 np1), −29.4 % (L4
    np4), −15.1 % (L5 np4). The np4 values are outside the per-state spread,
    like O1's own step-1 outliers.
  - **`s4 copy`, off against O1:** within ±3 % per run on HIP (np1 202.1
    against 207.7 s). The gain therefore vanishes with the knob off, and it
    is quoted only from the same-job pairs.
  - **SERIAL, cross-node:** np1 +1.1 % run level (1011 against 1015; per
    state −20 to +37 %), np4 −5.2 % (1014 against 1012).
- **Post-O8 shares** (medians of `3 × phase / step_wall` over each knob-on
  run's `[cost]` rows, HIP; off arm in brackets):

  | target | L4 np1 | L4 np4 | L5 np1 | L5 np4 |
  | --- | --- | --- | --- | --- |
  | `ilist_s4_op_table_copy` (O8) | 2.0 (41.6) | 0.9 (31.0) | 2.2 (20.0) | 0.7 (21.6) |
  | `ff_maintain` (O5) | 38.0 (20.7) | 39.0 (25.1) | 44.4 (35.1) | 46.7 (35.7) |
  | `comm_plan_build` (in O5) | 20.8 (11.6) | 16.4 (10.4) | 34.3 (26.5) | 30.0 (21.3) |
  | `ilist_s3_classify_pairs` (O9) | 28.0 (18.6) | 9.1 (6.7) | 27.4 (21.5) | 18.5 (13.8) |

  The off-arm shares reproduce O1's knob-on shares, computed the same way
  from O1's logs, within 2.1 points (copy 43.4 / 30.7 / 20.8 / 22.2,
  `ff_maintain` 18.6 / 25.0 / 34.8 / 34.9). The table in "Decision: O2 and O4
  not taken; O8 before O5" differs from both by up to 2.6 points, so its
  reduction differs in detail; compare like with like.

**Per-state numbers.** `on vs off` is within one job (two jobs for SERIAL
np4); `off vs O1` is the failure direction. `[acc]` is the knob-on value.
`=` means off, on and O1 agree to every printed digit.

**L4_HIP_np1** — summed step wall O1 484.3 s, off 490.8 s, on 250.5 s (on vs off -49.0 %; off vs O1 +1.3 %); summed s4 copy O1 207.7, off 202.1, on 5.15 s; summed s5 upload O1 1.25, off 1.40, on 1.21 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.438 | 0.438 | 0.331 | -24.3 % | +0.1 % | 29.48 → 28.30 → 2.251 | 0.61 → 0.40 | 1.1338e-03 | = |
| 50 | 0.243 | 0.251 | 0.154 | -38.8 % | +3.4 % | 29.17 → 28.78 → 0.614 | 0.37 → 0.32 | 1.1144e-03 | = |
| 100 | 0.263 | 0.270 | 0.167 | -38.0 % | +2.4 % | 29.49 → 29.15 → 0.663 | 0.41 → 0.33 | 1.0380e-03 | = |
| 150 | 0.269 | 0.272 | 0.176 | -35.6 % | +1.3 % | 29.56 → 29.15 → 0.645 | 0.38 → 0.33 | 1.0114e-03 | = |
| 200 | 0.268 | 0.269 | 0.171 | -36.6 % | +0.4 % | 29.62 → 28.92 → 0.655 | 0.37 → 0.34 | 8.7885e-04 | = |
| 250 | 0.246 | 0.252 | 0.156 | -38.2 % | +2.4 % | 29.03 → 28.32 → 0.591 | 0.34 → 0.32 | 8.1017e-04 | = |
| 300 | 0.242 | 0.252 | 0.151 | -39.9 % | +4.0 % | 28.92 → 28.37 → 0.532 | 0.34 → 0.30 | 8.5650e-04 | = |
| 350 | 0.252 | 0.260 | 0.160 | -38.3 % | +3.2 % | 28.29 → 28.09 → 0.591 | 0.36 → 0.31 | 9.4444e-04 | = |
| 400 | 0.267 | 0.276 | 0.163 | -41.0 % | +3.4 % | 32.97 → 32.74 → 0.694 | 0.32 → 0.28 | 1.0679e-03 | = |
| 450 | 0.289 | 0.297 | 0.163 | -45.0 % | +2.8 % | 37.71 → 38.01 → 0.710 | 0.33 → 0.29 | 4.5554e-04 | = |
| 500 | 0.320 | 0.327 | 0.174 | -46.8 % | +2.0 % | 44.87 → 44.26 → 0.766 | 0.34 → 0.28 | 1.1160e-03 | = |
| 550 | 0.385 | 0.393 | 0.194 | -50.7 % | +2.1 % | 57.10 → 56.14 → 1.072 | 0.35 → 0.30 | 1.1877e-03 | = |
| 600 | 0.436 | 0.431 | 0.218 | -49.3 % | -1.3 % | 63.01 → 59.74 → 1.338 | 0.38 → 0.33 | 8.1949e-04 | = |
| 650 | 0.487 | 0.472 | 0.238 | -49.7 % | -2.9 % | 70.42 → 65.46 → 1.677 | 0.39 → 0.35 | 6.6914e-04 | = |
| 700 | 0.502 | 0.474 | 0.232 | -51.1 % | -5.6 % | 74.47 → 67.98 → 1.876 | 0.42 → 0.37 | 6.2743e-04 | = |
| 750 | 0.520 | 0.492 | 0.237 | -51.7 % | -5.5 % | 79.06 → 71.38 → 1.928 | 0.40 → 0.36 | 7.5714e-04 | = |
| 800 | 0.481 | 0.479 | 0.227 | -52.7 % | -0.4 % | 73.85 → 71.07 → 1.727 | 0.38 → 0.36 | 1.4025e-03 | = |
| 850 | 0.517 | 0.533 | 0.245 | -54.1 % | +3.3 % | 80.58 → 79.51 → 2.097 | 0.42 → 0.35 | 7.7219e-04 | = |
| 900 | 0.501 | 0.516 | 0.237 | -54.1 % | +2.9 % | 78.84 → 77.42 → 2.104 | 0.43 → 0.36 | 8.9084e-04 | = |
| 950 | 0.517 | 0.537 | 0.255 | -52.5 % | +3.9 % | 78.48 → 77.31 → 2.105 | 0.43 → 0.37 | 9.3676e-04 | = |
| 1000 | 0.504 | 0.520 | 0.245 | -52.8 % | +3.3 % | 76.89 → 76.22 → 2.022 | 0.44 → 0.35 | 8.6207e-04 | = |
| 1050 | 0.515 | 0.533 | 0.255 | -52.1 % | +3.5 % | 78.19 → 77.68 → 2.092 | 0.43 → 0.35 | 7.4657e-04 | = |
| 1100 | 0.571 | 0.589 | 0.277 | -52.9 % | +3.1 % | 87.16 → 86.20 → 2.599 | 0.47 → 0.38 | 1.1242e-03 | = |
| 1150 | 0.550 | 0.566 | 0.259 | -54.2 % | +2.8 % | 84.55 → 83.85 → 2.649 | 0.40 → 0.37 | 1.0894e-03 | = |
| 1200 | 0.537 | 0.553 | 0.254 | -54.0 % | +2.9 % | 82.59 → 81.83 → 2.553 | 0.40 → 0.37 | 9.6332e-04 | = |

**L4_HIP_np4** — summed step wall O1 372.5 s, off 366.0 s, on 226.9 s (on vs off -38.0 %; off vs O1 -1.8 %); summed s4 copy O1 116.2, off 115.9, on 1.96 s; summed s5 upload O1 0.87, off 0.82, on 0.76 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.636 | 0.449 | 0.387 | -13.8 % | -29.4 % | 20.07 → 20.06 → 1.488 | 0.29 → 0.24 | 1.1338e-03 | = |
| 50 | 0.208 | 0.209 | 0.142 | -31.8 % | +0.6 % | 18.75 → 19.28 → 0.274 | 0.23 → 0.22 | 1.1144e-03 | = |
| 100 | 0.216 | 0.216 | 0.147 | -32.1 % | -0.1 % | 18.84 → 19.21 → 0.272 | 0.23 → 0.21 | 1.0380e-03 | = |
| 150 | 0.213 | 0.213 | 0.147 | -31.0 % | +0.1 % | 18.68 → 18.98 → 0.257 | 0.23 → 0.22 | 1.0114e-03 | = |
| 200 | 0.221 | 0.218 | 0.154 | -29.1 % | -1.4 % | 18.15 → 18.42 → 0.288 | 0.23 → 0.22 | 8.7885e-04 | = |
| 250 | 0.217 | 0.212 | 0.147 | -30.3 % | -2.3 % | 17.30 → 17.47 → 0.278 | 0.23 → 0.22 | 8.1017e-04 | = |
| 300 | 0.214 | 0.212 | 0.149 | -29.9 % | -0.9 % | 17.52 → 17.56 → 0.278 | 0.22 → 0.21 | 8.5650e-04 | = |
| 350 | 0.234 | 0.232 | 0.165 | -28.8 % | -0.9 % | 18.40 → 18.49 → 0.289 | 0.22 → 0.21 | 9.4444e-04 | = |
| 400 | 0.260 | 0.255 | 0.178 | -30.5 % | -1.7 % | 20.55 → 20.68 → 0.337 | 0.21 → 0.20 | 1.0679e-03 | = |
| 450 | 0.257 | 0.249 | 0.168 | -32.6 % | -3.0 % | 22.43 → 22.39 → 0.392 | 0.21 → 0.20 | 4.5554e-04 | = |
| 500 | 0.272 | 0.266 | 0.174 | -34.7 % | -2.4 % | 25.45 → 25.31 → 0.411 | 0.21 → 0.20 | 1.1160e-03 | = |
| 550 | 0.306 | 0.299 | 0.185 | -38.0 % | -2.3 % | 29.72 → 29.42 → 0.457 | 0.21 → 0.20 | 1.1877e-03 | = |
| 600 | 0.318 | 0.312 | 0.197 | -36.6 % | -2.0 % | 32.56 → 32.20 → 0.540 | 0.22 → 0.21 | 8.1949e-04 | = |
| 650 | 0.344 | 0.337 | 0.207 | -38.6 % | -2.3 % | 37.56 → 37.29 → 0.631 | 0.23 → 0.21 | 6.6914e-04 | = |
| 700 | 0.349 | 0.340 | 0.204 | -40.0 % | -2.8 % | 38.42 → 38.16 → 0.644 | 0.23 → 0.21 | 6.2743e-04 | = |
| 750 | 0.354 | 0.342 | 0.203 | -40.6 % | -3.3 % | 40.19 → 39.72 → 0.688 | 0.23 → 0.21 | 7.5714e-04 | = |
| 800 | 0.351 | 0.343 | 0.203 | -40.8 % | -2.4 % | 39.90 → 39.55 → 0.691 | 0.23 → 0.21 | 1.4025e-03 | = |
| 850 | 0.389 | 0.378 | 0.211 | -44.2 % | -2.9 % | 46.01 → 45.52 → 0.762 | 0.23 → 0.21 | 7.7219e-04 | = |
| 900 | 0.379 | 0.371 | 0.206 | -44.3 % | -2.3 % | 46.24 → 45.97 → 0.757 | 0.23 → 0.21 | 8.9084e-04 | = |
| 950 | 0.395 | 0.389 | 0.222 | -42.9 % | -1.7 % | 45.94 → 45.72 → 0.767 | 0.23 → 0.21 | 9.3676e-04 | = |
| 1000 | 0.369 | 0.364 | 0.214 | -41.1 % | -1.4 % | 41.35 → 41.12 → 0.709 | 0.23 → 0.21 | 8.6207e-04 | = |
| 1050 | 0.375 | 0.371 | 0.222 | -40.2 % | -1.1 % | 42.65 → 42.53 → 0.785 | 0.23 → 0.21 | 7.4657e-04 | = |
| 1100 | 0.408 | 0.406 | 0.233 | -42.5 % | -0.6 % | 48.14 → 48.04 → 0.868 | 0.24 → 0.22 | 1.1242e-03 | = |
| 1150 | 0.398 | 0.393 | 0.227 | -42.3 % | -1.2 % | 45.03 → 44.92 → 0.847 | 0.23 → 0.21 | 1.0894e-03 | = |
| 1200 | 0.394 | 0.390 | 0.228 | -41.6 % | -1.0 % | 44.99 → 44.80 → 0.823 | 0.24 → 0.21 | 9.6332e-04 | = |

**L5_HIP_np1** — summed step wall O1 293.5 s, off 292.0 s, on 227.3 s (on vs off -22.2 %; off vs O1 -0.5 %); summed s4 copy O1 61.9, off 59.9, on 5.31 s; summed s5 upload O1 1.61, off 1.53, on 1.21 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.226 | 1.211 | 1.509 | +24.7 % | -1.2 % | 62.38 → 60.53 → 11.166 | 1.54 → 1.94 | 9.9253e-04 | = |
| 25 | 0.770 | 0.773 | 0.580 | -24.9 % | +0.4 % | 61.74 → 60.19 → 3.659 | 1.48 → 1.22 | 1.1482e-03 | = |
| 50 | 0.922 | 0.937 | 0.720 | -23.1 % | +1.6 % | 63.93 → 62.40 → 4.765 | 1.66 → 1.30 | 1.0225e-03 | = |
| 75 | 0.980 | 0.974 | 0.770 | -21.0 % | -0.6 % | 65.00 → 62.83 → 5.260 | 1.79 → 1.54 | 1.2115e-03 | = |
| 100 | 0.998 | 1.002 | 0.798 | -20.3 % | +0.3 % | 65.41 → 63.18 → 5.673 | 1.89 → 1.53 | 1.2073e-03 | = |
| 125 | 0.998 | 1.008 | 0.785 | -22.1 % | +1.0 % | 65.66 → 63.16 → 5.676 | 1.94 → 1.39 | 1.0647e-03 | = |
| 150 | 1.043 | 1.001 | 0.818 | -18.3 % | -4.1 % | 65.89 → 61.42 → 5.741 | 1.85 → 1.49 | 1.1266e-03 | = |
| 175 | 1.028 | 1.007 | 0.812 | -19.3 % | -2.0 % | 67.25 → 63.16 → 5.737 | 1.92 → 1.45 | 1.1143e-03 | = |
| 200 | 1.029 | 1.027 | 0.794 | -22.7 % | -0.2 % | 71.31 → 69.53 → 6.327 | 1.78 → 1.39 | 1.0901e-03 | = |
| 225 | 1.046 | 1.015 | 0.793 | -21.8 % | -3.0 % | 73.92 → 70.12 → 6.337 | 1.60 → 1.39 | 9.8452e-04 | = |
| 250 | 1.002 | 1.002 | 0.759 | -24.3 % | +0.1 % | 74.81 → 73.69 → 7.599 | 1.61 → 1.30 | 9.9999e-04 | = |
| 275 | 0.965 | 0.976 | 0.726 | -25.6 % | +1.2 % | 75.14 → 74.35 → 7.214 | 1.57 → 1.05 | 9.9609e-04 | = |
| 300 | 0.942 | 0.941 | 0.698 | -25.8 % | -0.1 % | 75.18 → 74.35 → 6.564 | 1.35 → 1.02 | 9.0339e-04 | = |

**L5_HIP_np4** — summed step wall O1 179.1 s, off 181.4 s, on 126.1 s (on vs off -30.5 %; off vs O1 +1.3 %); summed s4 copy O1 40.4, off 39.4, on 0.95 s; summed s5 upload O1 0.47, off 0.45, on 0.36 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.156 | 0.981 | 1.272 | +29.7 % | -15.1 % | 45.52 → 42.69 → 5.179 | 0.63 → 0.54 | 9.9253e-04 | = |
| 25 | 0.448 | 0.452 | 0.292 | -35.5 % | +0.8 % | 41.84 → 40.68 → 0.873 | 0.47 → 0.40 | 1.1482e-03 | = |
| 50 | 0.527 | 0.529 | 0.367 | -30.6 % | +0.4 % | 41.84 → 40.72 → 0.883 | 0.50 → 0.42 | 1.0225e-03 | = |
| 75 | 0.560 | 0.553 | 0.392 | -29.0 % | -1.4 % | 42.37 → 41.12 → 0.854 | 0.51 → 0.41 | 1.2115e-03 | = |
| 100 | 0.580 | 0.586 | 0.413 | -29.5 % | +1.0 % | 42.34 → 41.32 → 0.876 | 0.53 → 0.43 | 1.2073e-03 | = |
| 125 | 0.587 | 0.603 | 0.428 | -29.0 % | +2.8 % | 42.15 → 41.04 → 0.890 | 0.51 → 0.41 | 1.0647e-03 | = |
| 150 | 0.610 | 0.621 | 0.442 | -28.9 % | +1.9 % | 42.31 → 41.60 → 0.903 | 0.51 → 0.43 | 1.1266e-03 | = |
| 175 | 0.617 | 0.632 | 0.446 | -29.5 % | +2.5 % | 42.83 → 41.97 → 1.007 | 0.51 → 0.41 | 1.1143e-03 | = |
| 200 | 0.631 | 0.648 | 0.446 | -31.1 % | +2.6 % | 46.31 → 45.04 → 1.004 | 0.50 → 0.39 | 1.0901e-03 | = |
| 225 | 0.657 | 0.654 | 0.449 | -31.4 % | -0.4 % | 48.65 → 47.01 → 1.113 | 0.53 → 0.41 | 9.8452e-04 | = |
| 250 | 0.652 | 0.666 | 0.451 | -32.2 % | +2.1 % | 50.31 → 48.75 → 1.381 | 0.49 → 0.38 | 9.9999e-04 | = |
| 275 | 0.636 | 0.659 | 0.443 | -32.7 % | +3.5 % | 49.51 → 48.62 → 1.390 | 0.46 → 0.35 | 9.9609e-04 | = |
| 300 | 0.633 | 0.635 | 0.436 | -31.3 % | +0.2 % | 48.49 → 47.13 → 1.319 | 0.45 → 0.35 | 9.0339e-04 | = |

**L4_SERIAL_np1** — summed step wall O1 806.4 s, off 815.0 s, on 775.3 s (on vs off -4.9 %; off vs O1 +1.1 %); summed s4 copy O1 28.3, off 32.4, on 35.91 s; summed s5 upload O1 1.41, off 1.36, on 1.35 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.726 | 2.018 | 1.517 | -24.8 % | +16.9 % | 19.59 → 20.31 → 21.682 | 1.13 → 1.07 | 1.1338e-03 | = |
| 25 | 1.650 | 1.884 | 1.826 | -3.1 % | +14.2 % | 14.24 → 14.08 → 14.801 | 0.87 → 0.90 | 1.1244e-03 | = |
| 50 | 1.492 | 1.689 | 1.876 | +11.1 % | +13.2 % | 14.07 → 13.86 → 14.636 | 0.88 → 0.90 | 1.1144e-03 | = |
| 75 | 1.554 | 1.501 | 1.542 | +2.7 % | -3.4 % | 14.39 → 14.18 → 14.849 | 0.92 → 0.94 | 1.0797e-03 | = |
| 100 | 1.553 | 1.607 | 1.689 | +5.1 % | +3.5 % | 19.08 → 14.46 → 15.463 | 1.00 → 0.97 | 1.0380e-03 | = |
| 125 | 1.640 | 1.736 | 1.719 | -1.0 % | +5.8 % | 21.74 → 14.57 → 22.763 | 1.01 → 0.96 | 1.0660e-03 | = |
| 150 | 1.439 | 1.718 | 1.437 | -16.4 % | +19.4 % | 18.60 → 14.33 → 15.566 | 0.94 → 0.95 | 1.0114e-03 | = |
| 175 | 1.444 | 1.971 | 1.467 | -25.6 % | +36.6 % | 18.60 → 14.60 → 19.659 | 0.98 → 0.94 | 9.5737e-04 | = |
| 200 | 1.602 | 1.822 | 1.525 | -16.3 % | +13.7 % | 16.32 → 14.48 → 15.293 | 0.95 → 0.91 | 8.7885e-04 | = |
| 225 | 1.468 | 1.605 | 1.352 | -15.8 % | +9.4 % | 14.72 → 14.72 → 15.257 | 0.93 → 0.90 | 8.6536e-04 | = |
| 250 | 1.456 | 1.472 | 1.539 | +4.5 % | +1.1 % | 14.56 → 14.37 → 14.921 | 0.83 → 0.88 | 8.1017e-04 | = |
| 275 | 1.617 | 1.465 | 1.416 | -3.3 % | -9.4 % | 14.71 → 14.43 → 14.967 | 0.84 → 0.87 | 8.3881e-04 | = |
| 300 | 1.579 | 1.476 | 1.588 | +7.6 % | -6.5 % | 15.30 → 14.88 → 15.381 | 0.89 → 0.90 | 8.5650e-04 | = |
| 325 | 1.838 | 1.637 | 1.385 | -15.4 % | -10.9 % | 17.16 → 16.86 → 25.867 | 0.92 → 0.89 | 9.3266e-04 | = |
| 350 | 1.843 | 1.477 | 1.395 | -5.5 % | -19.9 % | 19.18 → 18.99 → 28.301 | 0.91 → 0.91 | 9.4444e-04 | = |
| 375 | 1.843 | 1.468 | 1.426 | -2.8 % | -20.3 % | 21.92 → 30.33 → 31.925 | 0.87 → 0.88 | 1.0166e-03 | = |
| 400 | 1.799 | 1.584 | 1.436 | -9.4 % | -11.9 % | 21.11 → 34.02 → 35.625 | 0.87 → 0.89 | 1.0679e-03 | = |
| 425 | 1.836 | 1.656 | 1.509 | -8.9 % | -9.8 % | 22.79 → 36.87 → 38.166 | 0.87 → 0.88 | 1.0547e-03 | = |
| 450 | 1.586 | 1.743 | 1.429 | -18.0 % | +9.9 % | 25.18 → 36.50 → 37.629 | 0.88 → 0.85 | 4.5554e-04 | = |
| 475 | 1.553 | 1.565 | 1.740 | +11.2 % | +0.7 % | 25.53 → 41.14 → 42.210 | 0.88 → 0.87 | 3.6193e-04 | = |
| 500 | 1.462 | 1.519 | 1.728 | +13.8 % | +3.9 % | 27.29 → 44.39 → 45.213 | 0.88 → 0.86 | 1.1160e-03 | = |

**L4_SERIAL_np4** — summed step wall O1 1736.5 s, off 1646.8 s, on 1772.4 s (on vs off +7.6 %; off vs O1 -5.2 %); summed s4 copy O1 15.7, off 16.7, on 16.80 s; summed s5 upload O1 0.74, off 0.72, on 0.76 s.

| step | O1 s/step | off s/step | on s/step | on vs off | off vs O1 | s4 copy/eval O1 → off → on (ms) | s5 upload/eval off → on (ms) | `[acc]` (on) | vs O1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.754 | 2.392 | 2.703 | +13.0 % | -13.1 % | 14.24 → 15.88 → 14.870 | 0.50 → 0.60 | 1.1338e-03 | = |
| 25 | 3.526 | 3.075 | 3.625 | +17.9 % | -12.8 % | 8.85 → 9.54 → 9.835 | 0.48 → 0.48 | 1.1244e-03 | = |
| 50 | 3.536 | 3.384 | 3.543 | +4.7 % | -4.3 % | 8.88 → 9.34 → 9.894 | 0.48 → 0.49 | 1.1144e-03 | = |
| 75 | 3.560 | 3.386 | 3.481 | +2.8 % | -4.9 % | 9.11 → 9.55 → 10.091 | 0.48 → 0.49 | 1.0797e-03 | = |
| 100 | 3.587 | 3.498 | 3.410 | -2.5 % | -2.5 % | 9.14 → 9.69 → 10.259 | 0.49 → 0.50 | 1.0380e-03 | = |
| 125 | 3.598 | 3.377 | 3.453 | +2.3 % | -6.1 % | 9.24 → 9.83 → 10.246 | 0.49 → 0.52 | 1.0660e-03 | = |
| 150 | 3.379 | 3.094 | 3.584 | +15.9 % | -8.4 % | 9.16 → 9.85 → 10.264 | 0.48 → 0.50 | 1.0114e-03 | = |
| 175 | 3.304 | 3.309 | 3.616 | +9.3 % | +0.2 % | 9.20 → 9.75 → 10.138 | 0.49 → 0.51 | 9.5737e-04 | = |
| 200 | 3.618 | 3.230 | 3.549 | +9.9 % | -10.7 % | 9.07 → 9.77 → 10.040 | 0.49 → 0.50 | 8.7885e-04 | = |
| 225 | 3.350 | 3.245 | 3.404 | +4.9 % | -3.1 % | 9.02 → 9.83 → 9.988 | 0.47 → 0.50 | 8.6536e-04 | = |
| 250 | 3.157 | 3.038 | 3.526 | +16.0 % | -3.8 % | 8.87 → 9.69 → 9.817 | 0.47 → 0.50 | 8.1017e-04 | = |
| 275 | 3.189 | 3.092 | 3.519 | +13.8 % | -3.0 % | 9.04 → 9.78 → 9.917 | 0.47 → 0.50 | 8.3881e-04 | = |
| 300 | 3.390 | 3.147 | 3.480 | +10.6 % | -7.1 % | 9.40 → 10.13 → 10.292 | 0.47 → 0.50 | 8.5650e-04 | = |
| 325 | 3.476 | 3.402 | 3.576 | +5.1 % | -2.1 % | 9.85 → 10.70 → 10.732 | 0.47 → 0.52 | 9.3266e-04 | = |
| 350 | 3.498 | 3.271 | 3.625 | +10.8 % | -6.5 % | 10.00 → 10.90 → 10.709 | 0.48 → 0.52 | 9.4444e-04 | = |
| 375 | 3.468 | 3.514 | 3.487 | -0.8 % | +1.3 % | 11.11 → 11.96 → 11.793 | 0.47 → 0.53 | 1.0166e-03 | = |
| 400 | 3.533 | 3.620 | 3.543 | -2.1 % | +2.5 % | 12.61 → 13.08 → 12.658 | 0.48 → 0.51 | 1.0679e-03 | = |
| 425 | 3.524 | 3.357 | 3.583 | +6.7 % | -4.8 % | 13.56 → 13.84 → 13.564 | 0.47 → 0.54 | 1.0547e-03 | = |
| 450 | 3.540 | 3.330 | 3.538 | +6.2 % | -5.9 % | 13.03 → 13.61 → 13.221 | 0.47 → 0.52 | 4.5554e-04 | = |
| 475 | 3.618 | 3.271 | 3.697 | +13.0 % | -9.6 % | 14.85 → 15.52 → 14.963 | 0.47 → 0.51 | 3.6193e-04 | = |
| 500 | 3.641 | 3.260 | 3.694 | +13.3 % | -10.5 % | 15.45 → 16.21 → 15.405 | 0.47 → 0.51 | 1.1160e-03 | = |

**Affects:**
- **O5** — next, and now the largest target by far. `ff_maintain` is 38–47 %
  of the HIP step after O8 (21–36 % before), and `comm_plan_build` alone is
  16–34 %. Its stated shares (T3, and the post-O1 table) are superseded by
  the post-O8 table above.
- **O9** — `ilist_s3_classify_pairs` rose to 28 % at L4 and L5 np1, and to
  9–19 % at np4, from 7–22 %. At np1 it is now level with `comm_plan_build`.
- **O6** — its trigger is a share of the step, and every adapter share rose
  by the step's 22–49 % fall. Recompute its trigger from O8's knob-on rows
  (these logs), not T3's.
- **O10** — the downward sweep outside the interaction list is unchanged in
  seconds and larger as a share. Recompute from these logs too.
- **O7** — SERIAL is unchanged at step level. The O8 gather is 3–5 % slower
  than the host loop in `ilist_s4_op_table_copy` at SERIAL np1 (about 0.45 %
  of step), and a column-wise host functor would recover it. Under 1 % of
  any SERIAL step, so O7's target (the distributors, 67–92 %) is unchanged.
- **V1** — not triggered by O8: `[acc]` is bit-identical to O1 at every SMS
  state. V1 stays deferred to the end of Phase O. `device_op_table_assembly`
  is on by default in `FmmParams`, so the milestone tier will exercise it.
