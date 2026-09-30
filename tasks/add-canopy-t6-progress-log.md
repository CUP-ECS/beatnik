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

## T0

Documentation only. No code, no tolerance, no walltime; `git diff --stat`
touched nothing outside `tasks/`.

### Decisions taken as given by the task, recorded so they are not reopened

- **The `## T6` section of `tasks/canopy/add-canopy-progress-log.md` was
  completed IN PLACE, not appended to.** `grep -c '^## T6$'` returns 1, which
  it did before this task as well — the check is against duplication, not
  presence. Both of that section's self-describing scaffolds are gone: the
  opening "INCOMPLETE BY DESIGN" note now says what the tier run showed, and
  the closing "What is deliberately missing from this entry" subsection was
  replaced by the results themselves plus a *What this entry does not
  establish* subsection carrying what is genuinely still open.
- **No `Met.` paragraph was written and T6 stays IN PROGRESS.** The tier came
  back red; nothing is met. T9b owns that paragraph after a green re-run.
- **`scripts/tuolumne/run_milestone.flux` was not touched**, which is a
  deliberate departure from `T6-handoff.log` §6 step 3 (set `-t` from the
  measured total). The reasoning is in the `## T6` section: the run that
  produced the measurement is red, and the member that dominates its cost is
  the one about to change. `CLAUDE.md` and `README.md` likewise still say the
  `1440m` is pbatch's ceiling and not a measurement, which is still true.

### Where the task document's figures disagreed with the log — the log won

Four, all recorded in `## T6` at the log's value:

| figure | document | log |
| --- | --- | --- |
| tier total | 8.687 h | **31 265.7 s = 8.685 h** (broker line, `31265.7s`) |
| FMM members' cost | 29 237 s | **28 831.6 s** (sum of the eight `[t6] COST` claimA+claimB) |
| frozen pair's cost | about 2 028 s | **2 192.4 s** (sum of the eight `[m0t3] COST wall=`) |
| level-3 claim-A peak | `3.0958e-4` | **`3.0977653582364744e-4`** at SERIAL np1 |

The level-3 figure is the one worth understanding rather than just correcting:
`3.0958115097968656e-4` is the **pre-submission check's** SERIAL np1 value
(`f3ayuwpvn8yV`), and the tier run is a different run. The tier's four level-3
launches read `3.0977653582364744e-4`, `3.0956117473918756e-4`,
`3.0945486992828075e-4` and `3.0957978709796177e-4`, all at step 250 — so the
margin is **3.23x** and the four-digit agreement stands, but no specific
17-digit level-3 number is reproducible between runs. Also **the two
per-launch cost sums are the only accounting available**: the runner prints no
per-launch wall and printed no `SUMMARY` line at all, because it failed before
reaching it.

### What the log showed that the task's Do steps did not anticipate

- **Four failing assertion sites, not three.** The τ_A bound fails in **both**
  forms the member asserts it in — `p.rel <= kTauA` at
  `Beatnik_Test_Milestone0Fmm.cpp:1461` and
  `p.max_abs <= kTauA * p.scale` at `:1462` — beside `p.m2l_fallback == 0` at
  `:1456` and `diag.global_m2l_fallback_pair_count == 0LL` at `:1923`. That is
  what makes the failed-check arithmetic close exactly: `71 + 4 + 4 + 1 = 80 =
  3097 − 3017` at SERIAL np1 and `71 + 6 + 6 + 1 = 84 = 3097 − 3013` at HIP
  np4. The `:1456` count is **71 of 81 states** in every one of the four
  launches — the ten clean states are steps 0 through 225.
- **The fallback counts are globally reduced, and np4 overflows LESS than
  np1.** At step 1375: 13 274 (SERIAL np1), 13 420 (HIP np1), **5 300**
  (SERIAL np4), **5 392** (HIP np4). The cap is per rank, so four ranks carry
  four times the key budget. The task's "about 13 000 fallback pairs by step
  1375" is an **np1** figure and does not describe np4.
- **The fallback peak is at step 1650, not 1375.** 16 283 pairs (SERIAL np1)
  and 16 370 (HIP np1), where the error peaks at 1375. This is **R3** observed
  before the demand counter existed: a cap sized from the error peak would be
  short elsewhere.
- **The `[Canopy]` cap warning appears in only two of the four failing
  launches.** 2 732 occurrences in `FmmL4` SERIAL np1 and 2 733 in `FmmL4` HIP
  np1, and **zero** in either np4 launch despite non-zero fallback at 71
  states there. So **the warning's presence in a log is not a per-launch test
  for overflow** — the fallback counter is. Worth knowing before T5 or T9a
  greps a log for it.
- **Claim A at level 4 on HIP is much cheaper than the task estimated.**
  84.543 s at np1 and 100.154 s at np4, against the task's "roughly 140 s
  each". The four claim-A halves together are about **35 minutes**, and the
  two HIP ones about **three**. The direction is favourable — T3 and T5 fit
  `pdebug` with more room than the task claimed.
- **Two of the estimate table's predictions were badly wrong, in opposite
  directions.** `FmmL4` SERIAL np1's claim B was predicted at ~21 800 s from
  the level-3 SERIAL/HIP trajectory ratio of 9.2x and measured **5 304 s** —
  the realized level-4 ratio is **2.24x**, so the single largest term was
  **4.3x high**, which is most of the gap between the predicted 13-17 h and
  the measured 8.685 h. The level-3 SERIAL np4 penalty, which the level-3 HIP
  pair had contradicted, is **real and confirmed at both levels**: 2.35x
  slower than np1 at level 3, 1.52x at level 4.

### Also edited

`tasks/canopy/add-canopy.md`, T6's pointer only: the "Where it stands"
blockquote (`:1787`) now names this document and states the tier's red result,
and the status line at `:3` no longer says "the confirming full-tier run is
outstanding", which the run made false. **T6's status is unchanged at IN
PROGRESS** in both places, and no tolerance, `-t` or exit criterion was
touched. Correcting `:3` is a small departure from this task's stated "pointer
only" scope, taken because leaving a known-false claim in a design doc's
status line is worse than the scope creep.

**Affects:**
- **T1** — none. T0 is the motivation for T1, not a constraint on it.
- **T3, T5** — read every fallback and demand figure **per rank**, and expect
  np4 to overflow less than np1 rather than four times as much; the onset is
  step 250 and the fallback peak step 1650 while the error peak is step 1375,
  so the demand series must be printed at all 81 states and its own peak
  located rather than assumed to sit at 1375 (**R3**). Do not use the
  `[Canopy]` warning as an overflow indicator. Budget T5's HIP claim-A halves
  at about 85 s and 100 s, not 140 s each.
- **T8** — the cost facts its walltime planning needs are the eight
  `[t6] COST` rows now in `## T6`, the level-4 SERIAL/HIP ratio of **2.24x**
  (not level 3's 9.2x) and the SERIAL np4 penalty measured at both levels.
- **T9b** — owns `run_milestone.flux`'s `-t`, T6's `**Met.**` paragraph and
  T6's DONE mark, none of which this task wrote. The measured 8.685 h on a red
  run is the floor to size `-t` from, not the figure: the member that
  dominates it is the one T6-T8 change.

## T1

Canopy only. Two cmake trees, one `pdebug` job for the exit criterion and a
second for a question the first one raised.

### What was added, and the signatures

All in `canopy/src/Canopy_DownwardSweep.hpp` unless said otherwise. The clone
edited is the Beatnik env's spack `develop` source at
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`, **not**
`/g/g20/stewartj/research-bridges/canopy-dev/Canopy`, so T2 and T4 pick these
up with no push and pull. Nothing was committed or pushed; the clone carries
them as working-tree modifications at `develop` commit
`d3145e019a61951315bda93e71c7f691fa90642d`.

| addition | where |
| --- | --- |
| `static constexpr int M2L_DEMAND_COUNT_CAP = 1048576;` | beside `M2L_OP_COUNT_CAP` |
| `int _m2l_demanded_op_count = -1;` | beside `_m2l_realized_keys` |
| `bool _m2l_demand_saturated = false;` | beside it |
| `int m2l_n_demanded_ops() const` | after `m2l_n_unique_ops()` |
| `bool m2l_demand_saturated() const` | after it |
| `std::vector<int> m2l_cells_at_depth() const` | after those, **ungated**, **by value** |
| `#include <unordered_set>` | with the other standard includes |
| `n_demanded_ops=` and `demand_saturated=` | the `[Canopy Diagnostics] M2L operator table:` printf |
| `LS_DEMAND_BUDGET_KEYS = 1`, `testM2LKeyDemandConstrained`, `testM2LKeyDemandDefault`, and their two `TEST()` registrations | `canopy/tests/tstLaplaceSolve.hpp` |
| `run_t1_demand.flux`, `run_t1_repro.flux` | `canopy/scripts/tuolumne/` (new, untracked) |

`m2l_cells_at_depth()` returns **by value** because `_all_at_depth_local` is a
`std::vector<std::vector<int>>` of per-depth cell-index *lists* and the
per-depth *count* vector exists nowhere as state — there is nothing to hand
back by const reference. No existing signature changed and nothing was
removed, so every number the suite has ever produced still stands.

### How the counter is kept read-only

The demanded set is a function-local `std::unordered_set<M2LKey, M2LKeyHash>`
declared inside the merge block under `#ifdef CANOPY_ENABLE_PROFILING`, and
`key` is inserted into it **before** the `ops.size() < effective_op_cap` test,
so it sees every distinct key the merge sees rather than only the admitted
ones. It touches none of `ops`, `key_to_op`, `local_to_global`, `pair_op_idx`,
`_m2l_realized_keys`, the operator cache or the fallback tables; its only
consumer is `_m2l_demanded_op_count`, written once after the loop. Keys in
`local_ops[t]` are **already canonical** — canonicalization happens once at
the classify pass's single hash site — so they are inserted as they are.
Insertion stops at `M2L_DEMAND_COUNT_CAP` and sets `_m2l_demand_saturated`
(**R2**), and the flag is **cleared at the top of every build**, so one
saturating build cannot make every later one read as saturated. `-1` is the
"not compiled in" sentinel and 0 is a legal count (**R7**): both new cases
assert the sentinel explicitly in the `~profiling` branch, naming why a 0
there would be a wrong answer rather than a missing one.

### Builds

Two trees inside the same clone, beside spack's `build-linux-rhel8-zen4-*`
directories, which were left alone. Both configured with the arguments
`run_cmake_tuolumne.sh` uses, under
`spack env activate ${HOME}/spack_envs/tuolumne_trilinos` — Canopy is
**manual** mode and a different environment from the Beatnik checkout this
work was driven from.

| tree | cmake | resolved | build |
| --- | --- | --- | --- |
| `build-t1-prof-on` | `-DCanopy_ENABLE_PROFILING=ON -DCanopy_PROFILING_LEVEL=2` | `level=2` | 2 m 55 s |
| `build-t1-prof-off` | `-DCanopy_ENABLE_PROFILING=OFF -DCanopy_PROFILING_LEVEL=2` | **`level=0`** | 2 m 30 s |

The `OFF` tree was deliberately given `Canopy_PROFILING_LEVEL=2` as well, so
the run exercises the **kill switch** rather than routing around it: cmake
reported `Profiling, Enable FMM phase timing via MPI_Wtime (level=0)` and the
binary emitted **zero** `[Canopy Diagnostics]` lines against the `ON` tree's
1 536. Only `Canopy_Test_LaplaceSolve_MPI_SERIAL` was built; the full tree was
not needed and was not built.

### Measured: job `f3bPfi66qz4X`, both trees, ranks 1-6

`scripts/tuolumne/run_t1_demand.flux`, `pdebug`, `--time-limit=40`, one job
running `ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL` in each tree.
**`100% tests passed, 0 tests failed out of 6` in both** (50.50 s and
45.51 s), combined rc 0. Over the **21 `(nprocs, rank)` pairs** at ranks 1-6:

| case | `ON` tree | `OFF` tree |
| --- | --- | --- |
| constrained, `eff_cap=1` | `realized=1` everywhere; **`demanded` 111 .. 718**, strictly > 1 at every pair; `saturated=0`; `fallback` 186 .. 1 702, positive everywhere | identical but **`demanded=-1`** |
| default budget | **`demanded == realized` at every pair**, realized **111 .. 686**; `saturated=0`; `fallback=0` | identical but **`demanded=-1`** |

The default-budget realized range **111 .. 686 reproduces exactly** the
per-rank figure `LS_BUDGET_KEYS`'s existing comment records for the frozen
configuration, which is a free cross-check that the counter is counting the
right set.

The `[Canopy Diagnostics]` line now reads, at a one-column cap and at the
default, on the same rank and tree:

```
n_unique_ops=1   n_demanded_ops=604 demand_saturated=0 ... effective_cap=1
n_unique_ops=604 n_demanded_ops=604 demand_saturated=0 ... effective_cap=32768
```

which is the whole point of the task in two lines: the realized count follows
the cap and the demanded count does not.

**The budget that buys exactly one column is `Kernel::bytes_per_key * 1` =
21 952 B** for `LaplaceKernel<double, 6, 1>`, recorded as
`LS_DEMAND_BUDGET_KEYS = 1` and multiplied by the trait rather than written as
a literal. **T6 extends these cases and needs that number.** One column rather
than a larger cap is deliberate: it pins the realized side at exactly 1 at
every rank count, so `demanded > realized` is a claim with a known left-hand
side. `LS_BUDGET_KEYS = 64` would not do — its own comment records that at 256
columns some rank at np=3 overflows nothing, which makes a strict inequality
vacuous rather than false.

### The bug only running revealed: the exit criterion's identity test cannot be met as written, and that is not the counter's fault

**The `+profiling` and `~profiling` realized output are NOT byte-identical.**
Stating that plainly because **R1** turns on it. What is true, and what
actually discharges R1:

1. **The two new cases' realized figures match the two trees line for line at
   all 21 `(nprocs, rank)` pairs** — `realized`, `fallback`,
   `m2l_realized_keys()` contents and `cells_at_depth` — with `demanded=` the
   only field that differs. That is the comparison the exit criterion names,
   and it passes exactly.
2. The **whole** `[laplace-solve]` output is identical between the trees at
   **np 1, 2, 4 and 5**.
3. At **np 3 and np 6** other tests' solves differ. The clearest instance:
   `crossRankAgreement`'s solve — the *first* solve in the process — reports
   `n_unique_ops` 285 and 189 at ranks 0 and 1 in one tree and 273 and 204 in
   the other, and `opTableByteBudget`'s tight-budget fallback reads 123 against
   127.
4. **A second job settled it.** `f3bPhuxP1JNT`
   (`scripts/tuolumne/run_t1_repro.flux`) ran **three identical ctest passes
   per tree** at np 3 and 6. The `ON` tree disagreed with itself on all three
   pairings, and **so did the `OFF` tree**: its pass 1 and pass 3 differ in
   `fallback`, `realized_keys`, `cells_at_depth` and `n_unique_ops`, with
   `demanded=-1` throughout. The `nprocs=3 rank=1` first-solve `n_unique_ops`
   read `204, 204, 204` in `ON` passes 1-2 and `189` in pass 3 — the same value
   the between-tree diff had flagged.

So the instability is **pre-existing run-to-run nondeterminism in the
tree/partition path at np ≥ 3**, reproduced by unmodified code, and the
between-tree difference is not evidence about the instrumentation. It is
invisible in CI because no assertion pins `n_unique_ops`: all six tests pass
through it. Two further facts narrow it — `initial_hash` is
`0xb6ad437608ad69b7` in **every** line of both jobs, so the particle set is
fixed and the variation is downstream of it; and `cells_at_depth` itself
varies, so what moves is the **tree**, not just the key count. It was not
chased further: it is outside T1's scope and T1's own measurements are stable
under it.

**`demanded` was stable across all three `ON` passes** (the np3+np6 nine-line
set read `111, 116, 156, 160, 174, 175, 217, 281, 337` every pass) even where
`realized` moved, which is reassuring but should not be read as a guarantee.

### The caveat that matters most for T5

**This validation ran on a `key_needs_level=0` basis.** The diagnostics line
says so: `LaplaceKernel`'s solid-harmonic keys collapse every level onto one
column. The demand question at level 4 is a **`key_needs_level=1`** question —
CartesianTaylor retains the absolute level, so every occupied depth multiplies
the key count, which is the mechanism the whole topic turns on. T1 proves the
counter counts the merge's distinct canonical keys correctly; it measures
nothing about how many of them a level-keyed basis realizes.
`m2l_cells_at_depth()` exists for exactly that reason and was verified
non-empty on every rank (it read `[1,8,13..64,0..30]` across the 21 pairs — a
4-level tree), but its consumer is T2 and T5.

### Departures from T1's stated Do steps

- **`M2L_OP_COUNT_CAP` and `m2l_effective_op_cap()` were not touched**, as
  instructed. T1 only observes.
- **The `~profiling` branch of both new cases asserts the `-1` sentinel**
  rather than skipping. Step 9 specifies only the `+profiling` assertions, but
  a case that compiles to nothing in the `OFF` tree would make the exit
  criterion's "the `OFF` tree builds the same suite and passes with
  `m2l_n_demanded_ops()` returning `-1`" unverifiable from the log.
- **`_m2l_demand_saturated` is cleared at the start of each build.** Steps 2-4
  do not say to, and without it the flag is monotone for the life of the sweep
  while the count it qualifies is per build.
- **The new Canopy flux scripts carry no BSD-3-Clause header**, because no
  sibling script in `canopy/scripts/tuolumne/` does and Canopy's own
  `CLAUDE.md` states no such rule. Beatnik's header rule is about Beatnik
  files. The two new `.hpp` edits are to existing files, which keep theirs.
- **clang-format was not run** on either file, per the standing rule.

**Affects:**
- **T2** — the accessors are exactly as this document specified, so T2's plan
  stands: `m2l_n_demanded_ops()`, `m2l_demand_saturated()` and
  `m2l_cells_at_depth()` (by value, `std::vector<int>`, one entry per depth,
  ungated). T2's `local_m2l_occupied_depths` should count the **non-zero**
  entries of that vector, not its size — the vector is `max_depth + 1` long
  and the measured trees have trailing zeros.
- **T4** — the edits are in the Beatnik env's clone, so a `spack install`
  after adding `+profiling` picks them up with no push and pull. Note that
  spack builds this clone with its own `build-linux-rhel8-zen4-*` trees, which
  T1 did not touch; the two `build-t1-prof-*` trees are additions beside them.
- **T5** — two things. First, **T1's numbers are on a level-blind basis** and
  say nothing about how many keys a `key_needs_level=1` basis realizes; read
  `m2l_cells_at_depth()` beside the demand figure, since occupied depth is the
  multiplier. Second, **a repeated demand measurement may not reproduce**: the
  tree itself is run-to-run nondeterministic at np ≥ 3 in the Canopy test
  path, so T5 should report whether its own series is reproducible across two
  runs rather than assume a single run is the number. If it is not, the peak
  is a distribution and the cap must be sized from the worst observed, not the
  mean.
- **T6** — the one-column budget is `bytes_per_key * LS_DEMAND_BUDGET_KEYS`
  with `LS_DEMAND_BUDGET_KEYS = 1` (21 952 B at `LaplaceKernel<double, 6, 1>`),
  and the two cases it extends are `testM2LKeyDemandConstrained` and
  `testM2LKeyDemandDefault` in `canopy/tests/tstLaplaceSolve.hpp`. Both already
  branch on `CANOPY_ENABLE_PROFILING`, so a count-cap knob added there must
  keep working in the `OFF` tree.
- **T8** — `m2l_demand_saturated()` is the flag its branch B reads outright; it
  was `false` on every measurement here, so no saturated case has yet been
  observed and its presentation is still untested in practice.

## T2

Beatnik only, one file: `src/Beatnik_FarFieldInterface.hpp`, **+73 lines, −0**.
`git diff --stat` touched nothing else — no test, no script, no CMake, no
Canopy file.

### Decisions taken as given by the task, recorded so they are not reopened

- **The exit criterion ran through the existing
  `scripts/tuolumne/t6_l3_member.flux`**, invoked as
  `flux batch scripts/tuolumne/t6_l3_member.flux HIP`. No new `.flux` script
  was written and that one was not edited: its `-t 58m` is sized for all four
  backend/rank combinations, so a HIP-only run simply finishes early (617 s of
  58 m used). The script announces the backend override loudly, which is
  expected here and is not a finding.
- **`+profiling` was NOT added to the canopy spec.** That is T4. T2 is
  deliberately built and run against a `~profiling` canopy, which is the only
  reason the `-1` sentinel is observable at all — under `+profiling` the field
  would carry a count and the R7 check would have nothing to see.
- **Nothing was committed, pushed, or edited in the Canopy clone.** T1's
  additions are still uncommitted working-tree modifications at `develop`
  commit `d3145e0`, and `canopy@=develop` is a `spack develop` spec, so
  `spack install` compiled them in place.

### The four fields, exactly as implemented

All appended to `FarFieldDiagnostics` immediately after `local_m2l_op_cap`,
each documenting units, rank-locality and the Canopy accessor it mirrors, in
the style of the five fields above it.

| field | type, default | source |
| --- | --- | --- |
| `local_m2l_demanded_op_count` | `int`, **`-1`** | `down.m2l_n_demanded_ops()` verbatim |
| `local_m2l_demand_saturated` | `bool`, `false` | `down.m2l_demand_saturated()` verbatim |
| `local_m2l_cells_at_max_depth` | `int`, `0` | derived, see below |
| `local_m2l_occupied_depths` | `int`, `0` | derived, see below |

The last two come from **one scan** of `down.m2l_cells_at_depth()`, and the
rule is not the obvious one. The vector is `max_depth + 1` long and carries
trailing zeros, so:

- `local_m2l_occupied_depths` is the count of its **non-zero** entries, not
  its `size()`;
- `local_m2l_cells_at_max_depth` is the value of its **last non-zero** entry —
  the cell count at the deepest *occupied* depth — so a tree shallower than
  `max_depth` 10 reports a real count instead of an uninformative 0. The loop
  assigns on every non-zero entry, which leaves the last one.

**The `-1` and the `0` are kept apart in all three places R7 names**: the
field default is `-1` for the count and `false` for the flag; the population
copies Canopy's value verbatim with no clamping or `std::max`; and both doc
comments say in as many words that `0` is a legal measurement and `-1` means
"this Canopy build carries no profiling". The two derived fields are
documented as having **no `-1` case at all**, because `m2l_cells_at_depth()`
is ungated — their `0` means "empty vector, i.e. before `setup()`" and never
"unavailable". Conflating the two sentinels across the gated and ungated
halves of this block is the specific mistake the comments are written against.

`std::vector` is never **named** in Beatnik — the scan binds the accessor's
return with `const auto` — so no `#include <vector>` was added and the adapter
header's include list is unchanged.

### Callers: none needed editing, as the task enumerated

`grep -rn readDiagnostics src/ tests/ examples/` returns exactly four hits and
all four are in this file: the doc reference (`:734`), the pure virtual
(`:772`), the single override (`:875`) and the one call site (`:1491`,
`_impl->readDiagnostics( _diagnostics )`). The struct gained members and no
signature moved, so nothing outside the override was touched.

### Build

`spack install` in the dev env, exit 0. Two packages rebuilt:

| package | hash | time |
| --- | --- | --- |
| `canopy@develop` | `2cqynij` | **37 s** |
| `beatnik@develop` | `4bhhtbd` | **11 m 13 s** |

**The canopy rebuild is the load-bearing half of that table.** It is the first
compile of T1's working-tree edits, confirming T1's `**Affects:** T2` claim
that no push and pull is needed — `spack develop` picked the modified clone up
by itself. The installed header carries the three new accessors
(`grep -c` = 3 in the store's `Canopy_DownwardSweep.hpp`).

**The first instantiation of `m2l_cells_at_depth()` on the CartesianTaylor arm
produced no template error**, which the task flagged as the plausible failure
and as Beatnik's to fix. Nothing needed fixing; the accessor is an ordinary
non-template member of the sweep, so the arm's basis does not reach it.

The system doc's **header-only rebuild caveat applies and was paid**: this is
an INTERFACE library whose `HEADERS_PUBLIC` are not dependencies, so
`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp` and the
`examples/01_rising_bubble` driver were `touch`ed before installing. Without
that the install would have reported a sub-second no-op and the job below would
have run the **old** binary while reading as a pass. An 11-minute install is the
evidence the touch worked.

### Measured: job `f3bQ3AzdBncF`, HIP, ranks 1 and 4

`Beatnik_Test_Milestone0Fmm`, level 3, full 2000-step fidelity, at commit
`f94c9db` + 1 modified file:

| launch | wall | checks |
| --- | --- | --- |
| HIP np1 | **316 s** | `3097/3097` |
| HIP np4 | **301 s** | `3097/3097` rank 0, `2919/2919` ranks 1-3 |
| total | **617 s** | `[t6l3] SUMMARY: PASS (2/2 launches)` |

Against T0's **314 s** HIP np1 baseline that is **+0.6 %** — run-to-run noise,
and the expected result for four fields that no code path yet reads. The check
counts are identical to the ones T0 records for the passing level-3 launches,
so the member is unchanged rather than merely still green. The SERIAL half was
not run and nothing is claimed for it.

### How the `-1` sentinel was established — statically, by three facts

Nothing in Beatnik prints the field until T3's probe exists, so T2's R7 check
is a static argument, and it needs all three legs:

1. The env concretizes canopy as **`~profiling`** — `spack find --variants
   canopy` reads
   `canopy@develop~cuda~examples~ipo~openmp~openmptarget~profiling+rocm...`.
2. The installed `Canopy::Canopy` INTERFACE target exports **no
   `INTERFACE_COMPILE_DEFINITIONS` property at all**
   (`share/cmake/Canopy/Canopy_Targets.cmake:61-66` sets only
   `INTERFACE_COMPILE_FEATURES`, `INTERFACE_INCLUDE_DIRECTORIES`,
   `INTERFACE_LINK_LIBRARIES` and `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`), and
   `CANOPY_ENABLE_PROFILING` appears **nowhere** in Beatnik's own CMake. So the
   macro is undefined in every Beatnik translation unit.
3. With it undefined, Canopy's `_m2l_demanded_op_count` keeps its `-1` member
   default (`Canopy_DownwardSweep.hpp:726`): the only write to it is at
   `:1769-1771`, inside `#ifdef CANOPY_ENABLE_PROFILING`.

Therefore `local_m2l_demanded_op_count` reads **`-1`, not `0`**, in this build,
and `local_m2l_demand_saturated` reads `false`. **The runtime confirmation is
T3's**, which must run before T4 turns `+profiling` on — after T4 the `-1` is
no longer reachable in this env and the chance to observe it is gone.

### What only building or running revealed

- **`flux batch --flags=waitable` is refused on this instance.** It exits 1
  with `flux-batch: ERROR: only the instance owner can submit with
  FLUX_JOB_WAITABLE`, so the `flux batch --flags=waitable` + `flux job wait`
  recipe does not work here at all. **`flux job status <jobid>` alone does**:
  it blocked until completion and exited `rc=0` on a job submitted without the
  flag. Every later task that submits a job (T3, T5, T9a, T9b) should skip
  straight to `flux job status` rather than spending a submission discovering
  this.
- **Two of the four new fields already carry real data in this `~profiling`
  build.** `m2l_cells_at_depth()` is ungated, so `local_m2l_cells_at_max_depth`
  and `local_m2l_occupied_depths` are live *now*; only the two demand fields
  are sentinel-valued until T4. That splits T3's probe output into two classes
  and is the reason its failure-direction check has something to compare
  against.
- **The `spack install` took 11 m 13 s and exceeded a single command timeout**,
  so it was backgrounded and waited on. Budget a rebuild of this env at about
  twelve minutes when only a header changed but a consumer `.cpp` was touched.

### Departures from T2's stated Do steps

- **`tasks/add-canopy-t6.md:3` was corrected from `**Status:** NOT STARTED` to
  `**Status:** IN PROGRESS`.** Not in scope as written, and taken on T0's own
  precedent: three of this document's tasks are DONE, so the line was
  known-false, and T0 argued that leaving a known-false claim in a design doc's
  status line is worse than the scope creep. No other line outside the T2 entry
  was touched.
- **clang-format was not run**, per the standing rule, on a file that is
  clang-formatted. The new block was written by hand in the surrounding style;
  if the user's next format pass moves a line in it, that is expected and is
  not a regression.

**Affects:**
- **T3** — it consumes all four fields, and three things shape its probe.
  First, **only `local_m2l_demanded_op_count` and
  `local_m2l_demand_saturated` are sentinel-valued** before T4; the two
  depth-derived fields are live already, so a probe that prints all four
  against this build should show `-1`/`0` beside two real counts, and a
  probe that showed `-1` for the depth fields would be reporting its own bug.
  Second, **T3's runtime `-1` observation is the last chance to take it** —
  T4 makes the sentinel unreachable in this env. Third, use `flux job status`
  directly; `--flags=waitable` is refused.
- **T5** — the demand series it prints comes off
  `local_m2l_demanded_op_count`, and `local_m2l_occupied_depths` is the
  multiplier to read beside it (T1's caveat: the level-4 question is a
  `key_needs_level=1` question and T1's validation was level-blind). Both are
  **rank-local and unreduced** in the struct exactly as in Canopy, so T5 must
  report per rank and must not average.
- **T7** — `local_m2l_op_cap` is unchanged and still the number the demand
  count is read against; the count cap it plumbs through `FmmParams` shows up
  in this struct through that existing field, not through a new one.
- **T4, T6, T8, T9a, T9b** — none. T2 added no gate member, changed no
  tolerance, no cap and no signature, and the level-3 member's cost is
  unmoved at 316 s / 301 s on HIP.

## T3

Beatnik only, three files: a new
`tests/regression_tests/Beatnik_Probe_FmmKeyDemand.cpp` (**783 lines**), a new
`scripts/tuolumne/t6b_key_demand.flux` (**216 lines**), and **+7 lines, −0** in
`tests/CMakeLists.txt` — one entry appended to `BEATNIK_DRIVER_SOURCES` with
its comment. `git diff --stat` touched nothing else: no header, no member, no
Canopy file, no `README.md`, no `CLAUDE.md`, no `docs/testing.md`.

### Decisions taken as given by the task, recorded so they are not reopened

- **T3 ran before T4 and `+profiling` was NOT added to
  `/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml`.** The env still
  concretizes `canopy@develop~cuda~examples~ipo~openmp~openmptarget~profiling`
  (confirmed by `spack find --variants canopy` at submission time, and echoed
  into the job log), which is the only reason the `-1` sentinel is observable
  at all. After T4 it is unreachable in this env and the observation cannot be
  retaken. Turning the variant on is T4's task.
- **`scripts/tuolumne/t6b_key_demand.flux` carries only the level-3 validation
  launch.** T5 extends this same file with the level-4 matrix; that matrix was
  not written now. The file's header says so, and says why the level-4 queue
  and `-t` decision belongs to T5 (one level-4 FMM trajectory is 2 373 s at HIP
  np1, per T0).
- **Validation is HIP at np1 and np4, level 3 only.** np4 is the only launch
  that exercises the per-rank unreduced printing at all. **The `_MPI_SERIAL`
  target builds and `beatnik_exe` resolves it — the runner resolves it and
  prints the path — but it is NOT launched and nothing is claimed for it.**
- **No `README.md`, `CLAUDE.md` or `docs/testing.md` change, confirmed rather
  than assumed.** `grep -n` over the three for `BEATNIK_DRIVER_SOURCES`,
  `Beatnik_Test_FmmScan`, `Beatnik_Test_Milestone0Run` and "Measurement driver"
  returns **exactly one hit**, `README.md:662` — and it is not documentation of
  the driver loop but a *Known Issues* reproduce recipe,
  `Beatnik_Test_Milestone0Run <L> 2000 25 fmm 8 3`, for the comparator's
  mis-pairing bug. The probe does not appear in that recipe and does not change
  it. So: none of the three documents the loop, the probe adds no tier member,
  and no example's accepted arguments move — no change is owed to any of them.

### The probe, as implemented

`argv[1]` is the level and there is no second argument — no step-count
override, deliberately. The trajectory is claim A's: **direct**, frozen
connectivity, 2000 steps, checkpoint every 25, which is the gold sets' own 81
states. At each of them the preconditions are established in
`evaluateClaimA`'s order (halo exchange, geometry at current positions, sheet
vector) and `computeInterfaceVelocity` is called **once**, on the FMM solver
only. **No direct comparator and no comparator subprocess** — this measures
demand, not error, and dropping them is most of why it is so much cheaper than
the member.

The FMM solver is constructed **once, outside the loop**, exactly as claim A
does. That is load-bearing rather than tidy: a solver rebuilt per state would
report a cold cache at every state and `keys_built_delta` — R4's whole
instrument — would measure nothing.

### Three shape departures from T3's stated Do steps

- **The header is printed AFTER the step-0 evaluation, not before it.** Step 4
  asks the header to carry `bytes_per_key` and whether the build reports demand
  at all, and **neither exists until one evaluation has run**:
  `FarFieldDiagnostics` is default-constructed before that, so a header emitted
  first would have printed `bytes_per_key=0` and `demand_available` read off a
  `-1` that was the struct's initializer rather than Canopy's answer — which is
  precisely the reading R7 exists to prevent. Step 0 is therefore measured
  first, the header printed from its diagnostics, then step 0's own row. The
  header still appears above every row in the log, which is what step 4 is for.
- **`local_m2l_cells_at_max_depth` is in the row, beyond step 3's eleven
  fields.** T3's exit criterion checks it in the same rows as
  `local_m2l_occupied_depths`, so it has to be printed there. `keys_built_delta`
  is likewise printed per row rather than only in the trailer, because R4's
  signature is the per-evaluation increment and reading it from a peak alone
  would not show that it is the same at *every* state (it is — see below).
- **`demand_available` is reduced both ways (`MPI_MIN` and `MPI_MAX`) and the
  two compared**, rather than read off rank 0. It is a compile-time property and
  so cannot legitimately differ between ranks; a disagreement would mean the
  ranks are not running the same binary, which is worth failing on here rather
  than inferring later from a ragged demand column.

Nothing else departed. The registration is one line in `BEATNIK_DRIVER_SOURCES`
and no other CMake change — the loop already supplies the per-backend generated
TU, the build, the install and the absence of a label, an `add_test` and a
manifest line.

### Build

`spack install` in the dev env, exit 0, **12 m 20 s**, `beatnik@develop` hash
`4bhhtbd`. `canopy@develop` was **cached and did not rebuild** — nothing under
`canopy/` changed since T2 compiled T1's working-tree edits. **No `touch` was
needed and none was done**: `tests/CMakeLists.txt` changed, so cmake reran and
the two new targets are new translation units rather than stale ones. The
12-minute figure is a full reconfigure-and-rebuild and matches T2's 11 m 13 s;
the header-only no-op trap T2 warns about does not apply to a change that moves
a CMake list.

`HIPCC_LINK_FLAGS_APPEND` and `HIPCC_COMPILE_FLAGS_APPEND` were cleared before
installing, per the system doc. The install exceeded a single command timeout,
as T2 found, and was backgrounded and waited on.

### Measured: job `f3bQSGSss6RD`, HIP, level 3, ranks 1 and 4

`-q pdebug -t 30m`, at commit `0377307` + 3 modified files. `flux jobs` reports
`COMPLETED` returncode `0` after **93.20 s**.

| launch | runner wall | trajectory wall | 81 evaluations | checks |
| --- | --- | --- | --- | --- |
| HIP np1 | **24 s** | 15.216 s | 4.0435 s | `174/174` |
| HIP np4 | **43 s** | 35.580 s | ~3.65 s per rank | `174/174` x 4 |
| total | **67 s** | — | — | `SUMMARY: PASS (2/2 launches)` |

**Far below the budget.** The task sized this against level-3 claim A's 166 s;
the probe is 24 s because it omits the direct comparator and the per-state
Python comparator subprocess. `-t 30m` is accordingly very generous, and T5
should size the level-4 matrix from the FMM evaluation cost rather than from
this total.

**np4 is SLOWER than np1 here** — 43 s against 24 s — which is the level-3
rank-4 penalty T0 measured at both levels, unchanged: 642 vertices over four
ranks is communication-dominated and the 81 evaluations themselves are
marginally *cheaper* per rank (3.65 s against 4.04 s).

### The `-1` sentinel, observed at runtime for the first time (R7)

T2 established it statically from three facts and said the runtime confirmation
was T3's. It is now taken, and it is unambiguous. Over all **405** rows (81 x 1
plus 81 x 4):

- `demand=-1` in **405 of 405**; rows reading demand as anything else: **0**.
- `demand_saturated=0` in **405 of 405**.
- `occupied_depths` ranges **4 to 6**, `cells_at_max_depth` is **non-zero in
  every row**, and neither is ever negative. **The ungated half is live**, so a
  `-1` there would have been the probe's own bug and was not.
- Both headers print `demand_available=0` and the loud
  `*** DEMAND UNAVAILABLE ***` line naming `~profiling` as the cause and saying
  in as many words that `-1` is not zero demand.
- The trailer reports `first_exceed_step=-1`, `peak_demand=-1`,
  `peak_demand_step=-1` rather than "never exceeded" — the sentinel propagates
  into the derived figures instead of becoming a measurement there.

R5's check passed at both rank counts: `ncrit` 8, `order` 3,
`cartesian-taylor`, `mac_theta` 0.3, `max_depth` 10, `near_softening_factor` 0,
all echoed out of `fmm.farField().params()` and matched against the literals.

### What only running revealed

- **R4's "cache retains nothing" signature is confirmed, and it is total.**
  `keys_built_delta == unique_ops` in **405 of 405 rows** — zero mismatches, at
  both rank counts, at every one of the 81 states. Every evaluation rebuilds
  the *entire* admitted column set; the persistent cache carries nothing across
  a rebuild. `keys_built` reaches **250 098** at np1 by step 2000 against a
  resident `cache` of **3 592**. This is measured at level 3, where the cap does
  not bind, so it is the *clean* form of the signature: the rebuild is not a
  consequence of overflow. **T8's branch A — raise the cap — therefore pays the
  full cap in rebuild cost at every evaluation**, and this is the evidence its
  criterion asks for, available before any level-4 run.
- **Level 3 cannot exercise the overflow path at all, and this run is a
  mechanism check rather than a measurement.** Peak `unique_ops` is **6 404**
  against the **32 768** cap, `global_m2l_fallback` is **0** in all 405 rows,
  and the `[Canopy] M2L op count exceeded cap` warning appears **zero** times.
  That is the expected result and not a contradiction of T0: the level-3 member
  declares `kFarFieldIsLive = false` and `kP2PFractionBound = 1.0`
  (`Beatnik_Test_Milestone0Fmm.cpp:468`), and the probe measured
  `p2p_pair_fraction` between **0.70847** and **0.94521** there — level 3 is
  mostly a direct sum by design. **Do not read T3's zero fallback as evidence
  about level 4.**
- **The p2p fraction at level 3 exceeds the level-4 bound at most states, and
  the probe is right not to assert on it.** 0.945 is far past level 4's
  `kP2PFractionBound = 0.75`. A probe that had copied the level-4 bound as a
  compiled literal would have failed 405 times against a correct run — the
  concrete reason the "probe assertions: none" convention is not merely
  fastidious.
- **The per-rank columns really do differ, so the unreduced printing earns its
  cost.** At step 1000, np4: `unique_ops` 852 / 809 / 902 / 980,
  `occupied_depths` 6 / 5 / 6 / 6, `cells_at_max_depth` 2 / 19 / 4 / 6. A mean
  over those four would have reported a tree that no rank has.
- **`flux job status <jobid>` worked exactly as T2 reported**, on a job
  submitted without `--flags=waitable`, and blocked to completion. No
  submission was spent rediscovering the `FLUX_JOB_WAITABLE` refusal.
- **The probe writes 82 `.h5` per launch, not 81** — the 81 numbered
  checkpoints plus `checkpoint_latest.h5`, beside the grouped-IO master
  `checkpoint.xmf`. Worth knowing before a later task counts files to decide a
  series is complete.

### Departures from the standing rules: none

clang-format was **not** run, per the standing rule, on two files written by
hand in the surrounding style. Nothing was committed or pushed, in this repo or
in the Canopy clone; T1's Canopy edits remain uncommitted working-tree
modifications at `develop` commit `d3145e0` and `spack develop` continues to
compile them in place.

**Affects:**
- **T4** — the `-1` observation it makes unreachable **has now been taken**, so
  T4 is unblocked with nothing left to preserve. After T4 this same runner at
  the same level should print `demand_available=1` and a real count; the
  level-3 numbers above (peak `unique_ops` 6 404, `op_cap` 32 768, zero
  fallback) are the *before* half of the comparison, and demand at level 3 must
  come back **at or below 6 404 x (something small)** and certainly under the
  cap — a level-3 demand above 32 768 after T4 would mean the counter, not the
  tree, is wrong.
- **T5** — inherits this runner and this output format verbatim, and three
  things shape what it does with them. First, **level 3 proves nothing about
  overflow** (`kFarFieldIsLive = false` there); the level-4 matrix is the whole
  measurement. Second, **the demand series must be read per rank from the `row`
  lines**, whose rank tag is `rank=<r>/<n>`; the trailer already locates the
  demand peak and its own step and the first cap exceedance *from the demand and
  cap columns*, per **R3**, and the `[Canopy]` warning is not consulted anywhere
  in the probe. Third, **budget from the evaluation cost, not from this
  total**: the probe is 24 s at level 3 against claim A's 166 s because it drops
  the direct comparator, so the level-4 launches are dominated by the 81 FMM
  evaluations and the 2000-step direct trajectory, not by comparison.
- **T8** — **branch A's rebuild-cost test is already answered in the
  unfavourable direction.** `keys_built_delta == unique_ops` at 405 of 405
  states means the operator cache retains nothing between evaluations even where
  the cap does not bind, so raising the cap raises the per-evaluation rebuild
  cost by the full amount of the raise. Branch A must be costed against that,
  not against a hoped-for cache hit.
- **T6, T7, T9a, T9b** — none. T3 added no gate member and no milestone member
  (both manifests verified: 0 probe lines, milestone still 12 non-comment
  lines), changed no tolerance, no cap and no signature, and touched no member
  binary.
