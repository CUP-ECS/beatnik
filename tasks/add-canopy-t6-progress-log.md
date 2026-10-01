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


## T4

Environment and documentation, plus two probe submissions. Three repo files
changed — `systems/tuolumne/spack.yaml` (**+1 −1**),
`systems/tuolumne/claude.md` (**+8 −1**) and
`scripts/tuolumne/t6b_key_demand.flux` (**+22 −11**, comments only) — and one
file outside the repo, the live env's
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml`. No source file, no
test, no CMake, no Canopy file.

### Decisions taken as given by the task, recorded so they are not reopened

- **The matrix is T3's, unchanged: level 3, HIP, np1 and np4.** No SERIAL
  launch, no other rank count, no level-4 matrix — T5 owns level 4. Holding the
  matrix fixed is the whole reason T3's numbers are usable as a before-half.
- **The Canopy clone was left uncommitted.** T1's edits to
  `src/Canopy_DownwardSweep.hpp` and `tests/tstLaplaceSolve.hpp` still sit as
  working-tree modifications on `develop` commit `d3145e0` (confirmed with
  `git status --short` before installing), and `canopy@=develop` is a
  `spack develop` spec, so `spack install` compiled them in place. No `git
  pull`, `git commit` or `git push` was run in that clone.
- **Nothing was read from this run as evidence about overflow.** The level-3
  member declares `kFarFieldIsLive = false`
  (`Beatnik_Test_Milestone0Fmm.cpp:468`) and this run measured
  `global_p2p_frac` between **0.71000 and 0.95413**, so level 3 is mostly a
  direct sum, exactly as T3 found. `global_m2l_fallback` is 0 in all 405 rows
  and the cap never binds. The demand question is level 4's.
- **`systems/tuolumne/spack-production.yaml` was not touched.** The production
  canopy spec stays `~profiling`.

### The spec change, and what it actually set

`+profiling` was added bare to the `canopy@develop` spec at `:16` of both the
live env and its committed snapshot, placed immediately after the version to
match the `beatnik@develop` spec's own variant ordering on the next line. The
two files `diff` empty afterwards, as the task requires.

`profiling_level` was deliberately **not** set, and the resolution confirms
that was enough: the installed target now reads

```
INTERFACE_COMPILE_DEFINITIONS "CANOPY_ENABLE_PROFILING;CANOPY_PROFILING_LEVEL=1"
```

in `share/cmake/Canopy/Canopy_Targets.cmake:62`, against T2's finding that the
`~profiling` build exported **no such property at all**. `spack find --variants
canopy` moved from `...~openmptarget~profiling+rocm...` to
`...~openmptarget+profiling+rocm...`. That is the one-line proof the define
reaches every Beatnik translation unit through the INTERFACE target, with no
Beatnik CMake change, as the Conventions table predicted.

### A concretize step the task's Do list does not mention, and why it is needed

**`spack install` alone is not enough here; `spack concretize -f` is required
first.** A plain `spack concretize` after the spec edit fails outright:

```
==> Error: Spack concretizer internal error. ... is unsatisfiable. Couldn't
concretize without changing the existing environment.
```

Adding the variant changes canopy's hash, which changes beatnik's dependency
hash, and the env's `unify` setting will not move an already-concrete root
incrementally. `spack concretize -f` is the system doc's own prescription
(`systems/tuolumne/claude.md:130-140`) and it succeeded. **It moved no package
version**: the lockfile carries **62** concrete specs before and after with an
empty added-set and an empty removed-set, so the force-reconcretize is not a
hidden dependency bump. Record this for any later task that edits a spec in
this env — the error message looks alarming and is routine.

| package | hash before | hash after | build |
| --- | --- | --- | --- |
| `canopy@develop` | `2cqynij` | **`w4woraj`** | **32 s** |
| `beatnik@develop` | `4bhhtbd` | **`nnbspfy`** | **12 m 52 s** |

### Beatnik genuinely rebuilt

**12 m 52 s**, `INSTALL_RC=0`, against T2's 11 m 13 s and T3's 12 m 20 s for a
full reconfigure-and-rebuild. The header-only no-op trap does not apply and
**no `touch` was done and none was needed**: the dependency hash moved, so
spack rebuilds the dependent unconditionally. `HIPCC_LINK_FLAGS_APPEND` and
`HIPCC_COMPILE_FLAGS_APPEND` were cleared before concretizing and installing,
per the system doc. The install exceeded a single command timeout, as T2 and T3
both found, and was backgrounded and waited on.

### Measured: job `f3bQk9QtwPnw`, HIP, level 3, ranks 1 and 4

`scripts/tuolumne/t6b_key_demand.flux` unchanged but for its comments, `-q
pdebug -t 30m`, **61 s total wall**, `[t6b] SUMMARY: PASS (2/2 launches)`,
`flux job status` rc 0. The runner's provenance line reads `[t6b] canopy =
canopy@develop+profiling` — the cheapest proof the job ran the rebuilt binary.

Every check the task names, against T3's before-half (job `f3bQSGSss6RD`):

| check | T3 (`~profiling`) | T4 (`+profiling`) |
| --- | --- | --- |
| `demand_available` in both headers | `0` | **`1`** |
| `*** DEMAND UNAVAILABLE ***` | in both headers | **absent, 0 occurrences** |
| rows with `demand=-1` | **405 of 405** | **0 of 405** |
| `n_demanded_ops=` in the Canopy line | absent | **405 occurrences** |
| `demand_saturated` | `0` in 405 | `0` in 405 |
| `global_m2l_fallback` | `0` in 405 | `0` in 405 |
| `occupied_depths` | 4 to 6 | 4 to 6 |
| `cells_at_max_depth` zero rows | 0 | 0 |
| peak demand | `-1` (sentinel) | **6 198**, np1 step 1600 |

Demand at level 3 comes back far **under** the 32 768 cap, as the task
required: peak 6 198 against it, and four orders under the 1 048 576
`M2L_DEMAND_COUNT_CAP`. `demand_saturated` was never set.

### The level-3 demand series — T5's control prior

Per-rank peaks over the 81 checkpointed states, from the trailer lines:

| run | np1 peak (step) | np4 peaks, ranks 0-3 |
| --- | --- | --- |
| `f3bQk9QtwPnw` (A) | **6 198** (step 1600) | 2 332 / 1 864 / 1 932 / 1 926 |
| `f3bQmUaxP3eP` (B) | **5 938** (step 1575) | 2 135 / 2 073 / 2 039 / 1 876 |

np1 demand ranges **828 to 6 198** across the 81 states in run A; the np4
per-rank minimum over all rows is **285**. The series rises with tree depth
rather than with step: every state at `occupied_depths=4` sits in the 828-864
band, and the peaks are all `occupied_depths=6` states.

**`demand == unique_ops` in 405 of 405 rows, in both runs.** That is *not* the
level-blindness result T1 had on `LaplaceKernel` and must not be read as one.
Here the cap never binds — peak realized 6 198 against a 32 768 `op_cap`, zero
fallback — so every demanded key is admitted by construction, on any basis.
The identity is forced by the absence of overflow, and it says nothing about
what a `key_needs_level=1` basis demands at level 4 where the cap does bind.
`bytes_per_key=3200` and `key_needs_level=1` are echoed in every Canopy line,
confirming the CartesianTaylor basis is the one being measured.

### What moved against T3, and the second job that explains it

**The realized columns moved, and the task asked for this to be said plainly
rather than waved off as noise.** T3 against T4 run A, over the same 405
`(nranks, rank, step)` keys:

| field | rows differing |
| --- | --- |
| `unique_ops` | **367 / 405** (66 of 81 at np1) |
| `keys_built_delta` | 367 / 405 |
| `cells_at_max_depth` | 308 / 405 |
| `global_m2l_pairs` | 342 / 405 |
| `occupied_depths` | 34 / 405 |
| `global_m2l_fallback` | **0 / 405** |

Peak `unique_ops` fell from T3's **6 404** to **6 198**. Taken alone that is an
**R1** signature — the demand counter changing an answer — so it was not taken
alone. **A second submission of the same `+profiling` binary, job
`f3bQmUaxP3eP`, settles it.** Run A against run B differs in **369 of 405**
rows of `unique_ops`, the same fields at the same magnitude as the T3-to-T4
comparison, with peak np1 demand at 5 938 against A's 6 198. Two runs of one
binary disagree with each other as much as the two binaries disagree, so
**`+profiling` is not the cause and R1 stands undischarged-by-nothing: the
counter is read-only.** `global_m2l_fallback` is 0 in all 405 rows of all three
runs, and `op_cap` is 32 768 in all of them.

**This extends T1's nondeterminism finding from np ≥ 3 to np1.** T1 measured
the Canopy test path disagreeing with itself only at np 3 and 6; here the
level-3 Beatnik trajectory diverges at **np1** as well. The shape is
informative: all three runs share an **identical contiguous prefix** — np1
steps 0 through 175, the first eight checkpoints, agreeing in every column —
and then diverge at step 200 and stay diverged. At np4 the prefix ends earlier,
at step 125. A pure tree-build nondeterminism would be expected to show at step
0; an identical prefix that breaks partway and never recovers points at the
**2000-step direct trajectory** itself diverging, with the tree following the
positions. It was not chased further: it is outside T4's scope, it is present
in the unmodified path, and no assertion anywhere pins these columns.

Walltimes moved with it and should not be read as a `+profiling` cost: np1
trajectory 15.216 s at T3 against 11.311 s here, and np1 `eval_wall_total`
4.0435 s against 3.7517 s — the instrumented build ran *faster*, which is
itself a sign the difference is trajectory shape rather than overhead.

### Comment corrections to `scripts/tuolumne/t6b_key_demand.flux`

Comment-only, no executable line touched. The header at `:26-31` carried
"**RUN THIS BEFORE T4.**" as a live imperative and described the env as
concretizing `canopy ~profiling`; both became false the moment this task
landed. It now records that the `-1` observation **was taken**, names T3's job
`f3bQSGSss6RD` and points at `## T3` here, then says what the script shows
under `+profiling` — including that a `-1` in the demand column now means the
binary was not built against `+profiling`, and to check the `spack find
--variants canopy` line the script echoes before trusting any row. The PASS
summary at `:204-206` likewise told the reader the demand column is the `-1`
sentinel; it now says the column is a real count, that `0` is a legal
measurement there, and keeps the reminder that the two depth columns were live
all along.

### Departures from T4's stated Do steps

- **`spack concretize -f` was run before `spack install`**, which step 4 does
  not mention. It is not optional — see above; a plain concretize errors out.
- **A second probe submission was taken** beyond the exit criterion's one run.
  Without it the 367-row `unique_ops` movement would have been an unresolved
  R1 signature sitting in the log, and the task explicitly asked for movement
  to be reported plainly rather than treated as noise. Reporting it *and*
  explaining it costs 61 s.
- **`systems/tuolumne/claude.md:51` was rewritten rather than amended**, as
  step 5 requires. It said "The two differ only in `profiling_level` (dev 2,
  prod 1)"; it now names both differences, in both packages, and says why
  canopy's `+profiling` is dev-only.
- **clang-format was not run**, per the standing rule.
- Nothing was committed or pushed, in this repo or in the Canopy clone.

**Affects:**
- **T5** — four things, and the last is the expensive one. First, it inherits
  this runner with its comments already corrected for a `+profiling` world; no
  further comment work is owed. Second, **the level-3 np1 control series is a
  distribution, not a number**: two runs of one binary gave peak demand 6 198
  and 5 938, about a 4 % spread, so T5's step-2 control must be compared
  against that band and a level-4 figure must come from the **worst observed**
  of at least two runs, exactly as T1's `**Affects:** T5` line already warned —
  now confirmed at np1, which T1 could not claim. Third, **`demand ==
  unique_ops` at level 3 is an artefact of the cap not binding**, not a
  property of the basis, so T5 must not treat a level-4 `demand > realized` as
  a surprise or a level-3 regression. Fourth, `spack concretize -f` is needed
  after any spec edit in this env and moves no versions.
- **T7, T8** — the demand counter is now compiled into every Beatnik binary
  this env builds, so any later run of any member or probe carries it at no
  extra build cost. T8's branch B reads `m2l_demand_saturated()`, which has
  still **never been observed set** — 405 rows here, 405 at T3, and all of
  T1's — so its presentation remains untested in practice.
- **T9a, T9b** — the env they will run against is now `canopy +profiling`.
  That is a different binary from the one T6's tier run used, and
  `CANOPY_PROFILING_LEVEL=1` adds MPI_Wtime phase timing to the FMM path. No
  cost was measurable here (this run was *faster* than T3's, inside the
  trajectory's own run-to-run spread), but neither is it a controlled
  measurement, and a tier run is hours rather than seconds. If a milestone
  walltime moves, look here first.
- **T6** — none directly, but note that `systems/tuolumne/spack.yaml` no longer
  describes the environment T6's tier run used.

## T5

Two repo files changed, both of them edits rather than additions:
`scripts/tuolumne/t6b_key_demand.flux` (the level-4 matrix, `-t 60m`, a
per-job scratch path, and the header and summary comments the matrix made
false) and `tasks/add-canopy-t6.md` (the two stale figures this task was told
to correct, outside the T5 entry). **No C++, no CMake, no Canopy file, no
`spack install`.** The probe binary that produced every number below is the one
T4 installed: `canopy@develop+profiling`, echoed in all three job logs.

### Decisions taken as given by the task, recorded so they are not reopened

- **The repeat measurement is separate `flux batch` submissions, not passes
  inside one job.** Independent allocations separate a run-to-run effect from
  an allocation-fixed one, and give independent logs to correlate.
- **Three draws, not two** — the task's two, plus one, on the user's
  instruction that each job is minutes. Each submission is about 170 s, so the
  third draw costs nothing and turns a two-point spread into a three-point one.
- **The peak is reported as the worst observed per (rank count, rank)**, with
  the run-to-run spread stated. Never a mean and never a single draw: T8 sizes
  a cap from this number.
- **The level-3 launch is a reproducibility check against a measured band**,
  not a test of whether level-3 demand is under the cap — already measured
  twice at T4.
- **Every rank-local field is reported per rank and unreduced**, and
  `global_m2l_fallback` rather than the `[Canopy]` warning is the overflow
  observable.

### The script, as extended

The matrix became `_matrix=( "3:1" "4:1" "4:4" )`, as `<level>:<ranks>`, and
the loop reads the level out of each entry instead of a single `_level`. The
level-3 control runs **first**, so an apparatus problem shows against its known
band before the expensive pair spends any time. `-t` moved from `30m` to
`60m`, which also covers the optional SERIAL level-4 pair; that pair was
**not** added, because the HIP result is not ambiguous. The rank-to-node
binding, the `rm -rf` + `mkdir -p` scratch handling and the provenance block
are untouched. The failure tags gained `L${_level}`, which they needed the
moment the matrix spanned two levels.

### The scratch path had to become per job, and the first attempt is why

**T5's first attempt lost a launch to a filesystem collision of its own
making.** The script's scratch path was `${ROOT}/${_target}_L${_level}_np${_np}`
— keyed by target, level and rank count but **not by job** — which was correct
while only T3 and T4 ran it one submission at a time. Two submissions of the
three-draw matrix were then put in the queue together (`f3bZ1YdD8SFy` and
`f3bZ1YnwyidM`) and wrote the same checkpoint files. The second died at step
1325 of its level-4 np4 launch:

```
#007: ../../src/H5FDsec2.c line 941 in H5FD__sec2_lock(): unable to lock file,
      errno = 11, error message = 'Resource temporarily unavailable'
[FAIL]   check 118: unexpected exception: Beatnik::CheckpointIO::write: cannot
      reopen '.../keydemand_sub4_HIP_np4/checkpoint_t00001p693400_step0001325.h5'
      to append the /beatnik scalar group.
MPIDI_Cray_shared_mem_coll_bcast(515): collective tags 14 and 1 do not match
```

53 of 81 states, rc 255. **Both of that overlapping pair were discarded, not
just the one that died** — a collision that does not abort is the worse case,
because the probe's particle-count round trip would then read another job's
checkpoint and report a plausible series. The fix is a `job${_jobid}/` path
component, taken from `FLUX_JOB_ID` if set, else `flux getattr jobid`, else the
PID. **`FLUX_JOB_ID` is not set for a batch script** — flux sets it for the
tasks the shell launches, not for the batch instance's init program — so the
first fixed draw landed in `jobpid3233048` and only the `flux getattr jobid`
fallback produces a traceable `jobf3bZ93jf57sM`. Both are collision-safe; only
the second is traceable, which is why the chain has three links.

### The three draws

All three submissions `COMPLETED` rc 0, `SUMMARY: PASS (3/3 launches)`,
`174/174 checks` in all nine launches, at commit `9303470` + 2 modified files.

| draw | jobid | job wall | L3 np1 | L4 np1 | L4 np4 | scratch |
| --- | --- | --- | --- | --- | --- | --- |
| D | `f3bZ3aqyro5y` | **169.46 s** | 19 s | 57 s | 73 s | `jobpid3233048` |
| E | `f3bZ93jf57sM` | **171.64 s** | 20 s | 56 s | 75 s | `jobf3bZ93jf57sM` |
| F | `f3bZAPsQynnB` | **171.47 s** | 26 s | 56 s | 69 s | `jobf3bZAPsQynnB` |

A fourth complete draw exists and corroborates every figure below without being
counted in any spread: `f3bYxvcsXMWo`, 175.32 s, the pre-fix submission that
ran alone and so could not collide. Its level-4 np1 peak is **37 518 at step
1650**, inside D/E/F's own 0.46 % spread.

**Nine launches, 486 rows each in D, E and F — 81 states x (1 + 1 + 4) ranks,
exactly.** The series is complete in every launch; no launch was shortened and
no state was skipped. Total cost of the measurement: **8.5 minutes of job
wall** across three submissions, against the `-t 60m` the script now carries
and the 8.687 h the failing tier run spent to learn less.

### `demand_saturated` was never set

**0 of 1944 rows** across all four complete draws, at both rank counts and both
levels. `M2L_DEMAND_COUNT_CAP` ($2^{20}$) is nowhere near binding, so **every peak
below is a measurement and not a lower bound**, and the failure direction the
exit criterion specifies did not occur. `demand_available=1` in all nine
headers and `demand=-1` in 0 of 1944 rows, so no figure here is the `~profiling`
sentinel. R5's parameter check passed in all nine launches: `ncrit` 8, `order`
3, `cartesian-taylor`, `mac_theta` 0.3, `max_depth` 10,
`near_softening_factor` 0, `bytes_per_key` 3200, `byte_budget` 2147483648,
`op_cap` 32768.

### The number T8 is waiting on

**Worst observed per (rank count, rank), over draws D, E and F:**

| rank count | rank | worst demand | step | draw | D / E / F | spread | depths | `cells_at_max_depth` | `unique_ops` | `keys_built_delta` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **np1** | 0 | **37 678** | **1650** | E | 37 504 / 37 678 / 37 504 | **0.46 %** | **7** | 168 | **32 768** | **32 768** |
| np4 | 0 | 17 144 | 1375 | E | 17 013 / 17 144 / 16 890 | 1.50 % | 7 | 32 | 17 144 | 17 144 |
| np4 | 1 | 16 309 | 1350 | E | 16 259 / 16 309 / 16 303 | 0.31 % | 7 | 37 | 16 309 | 16 309 |
| np4 | 2 | 15 981 | 1875 | D | 15 981 / 15 892 / 15 955 | 0.56 % | 7 | 31 | 15 981 | 15 981 |
| np4 | 3 | 16 055 | 1350 | F | 16 032 / 15 950 / 16 055 | 0.66 % | 7 | 34 | 16 055 | 16 055 |

**The single number is 37 678 demanded operator keys, at HIP np1 rank 0, step
1650.** It is **1.150x** the 32 768 count cap.

- **Implied table size at that peak: 37 678 x 3200 B = 120 569 600 B =
  114.99 MiB = 0.112 GiB.** Against the 2 GiB byte budget, which buys 671 088
  columns, so **the byte budget is not the constraint and is not close to
  being one** — the count cap binds by **17.8x** at the measured peak. T8's
  branch A costs about 115 MiB of table per rank, not gigabytes.
- **`keys_built_delta` at the np1 peak is 32 768** — the cap, not the demand,
  because only admitted keys are built. That is the whole admitted table
  rebuilt in that one evaluation.
- **`occupied_depths` at the peak is 7**, against 5 at step 0 and a level-4
  range of **5 to 8**. Level 3's range is 4 to 6. Every occupied depth
  multiplies the key count on a `key_needs_level` basis, and the demand series
  tracks the depth count rather than the step: all eight states at
  `occupied_depths=5` sit in a 10 882–10 902 band, and every state above 30 000
  is at depth 7.
- **np4's worst rank demands 17 144 — less than half np1's, not a quarter.**
  The cap is per rank, so the four-rank tree's largest local key set is 45.5 %
  of the single-rank set, and **no np4 rank ever reaches the cap**
  (`first_exceed_step=-1` for all four ranks in all three draws).

### Where the demand peak sits against the three steps the failure already has

**R3 is confirmed, and the three steps are four.** The demand peak is at
**step 1650 — the fallback peak's step, not the error peak's.**

| step | what is there | demand at it (np1, worst of three) |
| --- | --- | --- |
| 250 | fallback first becomes non-zero (40–70 pairs) | 18 396, **under the cap** |
| **900** | **demand first exceeds the cap**, in all three draws | 33 124 / 33 142 / 33 124 |
| 1375 | claim A's error peak (`1.2513e-3`) | 36 622 / 36 630 / 36 600 |
| **1650** | the fallback peak **and the demand peak** | **37 504 / 37 678 / 37 504** |

**A cap sized from the error peak at step 1375 would be short by about 1 050
keys at step 1650.** Demand at 1375 is 97.2 % of the peak — close, but the two
steps are genuinely distinct and the ordering is stable across all three draws.
`first_exceed_step` is **900** in all three draws, not step 250: the document's
"first trips it at step 250" describes when **fallback** starts, which is a
different event, and 250 is 56 % of the cap.

### The finding that was not in any Do step: fallback at level 4 is not all cap-driven

**At np4 no rank's demand ever reaches the cap, and fallback is still non-zero
at 71 of 81 states, in all three draws.** Per-rank peak demand is 15 892–17 144
against a 32 768 cap, `first_exceed_step` is `-1` for all four ranks in all
three draws, and `global_m2l_fallback` is still non-zero from step 250 onward,
peaking at **6 884 pairs at step 1550**. No key can have been refused for a
count-cap reason in any of those 71 states, so **some other refusal path routes
those pairs to the per-pair fallback.** The key space itself is the obvious
candidate — `dd` over [-6,6] (`Canopy_CartesianTaylorBasis.hpp:469`) and
`ii,jj,kk` over [-32,32] (`Canopy_DownwardSweep.hpp:526`) are hard
representability bounds, not budgets — but **this entry does not identify the
mechanism and should not be read as having done so.**

At np1 the same thing shows as a split rather than a total: fallback is non-zero
at 71 of 81 states, of which **34 to 35 have demand at or under the cap**
(steps 250 through 1775) and only 36 to 37 exceed it.

**Why this matters more than the peak does.** The level-4 member asserts
`p.m2l_fallback == 0` (`Beatnik_Test_Milestone0Fmm.cpp:1456`) and fails it at 71
of 81 states at **both** rank counts. Raising the count cap can only fix the
states whose fallback is cap-driven. At np4 that is **zero states**, and at np1
about half. **A cap raise alone therefore cannot turn the level-4 member green**,
whatever value it is raised to — which is a constraint on T8's branch A that the
document's branch criteria do not currently carry, and a reason T9a's `pdebug`
check exists before T9b is submitted. **Deciding what follows is T8's, not
this entry's.**

### R4 at level 4: the same total result as level 3

`keys_built_delta == unique_ops` in **405 of 405** level-4 rows in every draw —
all 81 np1 states and all 324 np4 (state, rank) pairs — exactly as T3 and T4
measured at level 3. The operator cache retains nothing between evaluations at
level 4 either, and now also in the regime where the cap **does** bind: at the
37 np1 over-cap states the increment is the full 32 768, so each of those
evaluations rebuilds the entire admitted table. **Raising the cap raises the
per-evaluation rebuild cost by the full amount of the raise, at every one of
the 81 states.** T3 already recorded this conclusion as T8's to draw; nothing
here re-litigates it, and the level-4 increment at the peak is recorded above
as the task asked.

### The level-3 control, and the one place this entry misses its exit criterion

**Stated plainly: the control does not stay inside the 5 938 – 6 198 np1 band
the exit criterion names.** The three draws peak at **5 728** (step 1925),
**6 624** (step 1575) and **5 948** (step 1575) — D below the band, E above it,
F inside it — and the pre-fix draw A peaked at **5 586** (step 400), further
below. The three-draw spread is **15.64 %**, against the 4 % the band was drawn
from.

Everything the control is actually testing passes, in all three draws:

| check | result |
| --- | --- |
| rows per launch | 81, complete |
| `demand > op_cap` | **0 of 81** in every draw |
| `global_m2l_fallback` | **0** in all 243 rows |
| `demand_saturated` | **0** in all 243 rows |
| `demand == unique_ops` | **81 of 81** in every draw |
| `occupied_depths` | 4 to 6, as T3 and T4 |
| demand minimum | **828** in all three draws, identical |

**The band is the thing that was wrong, not the apparatus.** It was a two-draw
min/max (6 198 and 5 938) presented as a band, and five draws of the same
binary now span **5 586 to 6 624** — a ±8.5 % window around roughly 6 100, with
the peak's *step* moving from 400 to 1925 across draws. That is the level-3
trajectory nondeterminism T4 characterised, sampled three more times; a
two-point range cannot bound it. The substantive readings are stable to the
point of being identical where they should be — the 828 minimum, the 81/81
identity, the zero fallback — and the measurement apparatus demonstrably did
not move. **The exit criterion's numeric band is nevertheless not met as
written, and no level-4 figure above depends on it.** It is recorded this way
rather than widened silently; the fix, if one is wanted, is to state the
control as a tolerance on the band's own sample count, which is a document edit
and not a measurement.

Also worth noting for anyone who greps: **the `[Canopy] M2L op count exceeded
cap` warning count exactly equals the np1 over-cap state count** in each draw
(37, 36, 37) and is **zero in every np4 launch** despite non-zero fallback at 71
states there. So the warning is faithful at np1 and silent at np4 — T0's point,
reproduced, and the reason the demand-against-`op_cap` comparison is the
instrument.

### The full per-rank series

The 81-state series at all five (rank count, rank) pairs, as the worst of the
three draws at each state, plus the per-draw values at np1 so the spread is
visible state by state. `demand` is per rank and unreduced; `fb` is Canopy's
global reduction and is the same on every rank by construction.

**Level 4, np1 rank 0.** `demand (worst)` is the max over D, E and F at that state; the three columns beside it are the individual draws, and the remaining columns come from the draw that supplied the worst value.

| step | demand (worst) | D | E | F | uniq | depths | cmax | kbd | fb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | **10902** | 10902 | 10902 | 10902 | 10902 | 5 | 552 | 10902 | 0 |
| 25 | **10902** | 10902 | 10902 | 10902 | 10902 | 5 | 564 | 10902 | 0 |
| 50 | **10902** | 10902 | 10902 | 10902 | 10902 | 5 | 560 | 10902 | 0 |
| 75 | **10902** | 10902 | 10902 | 10902 | 10902 | 5 | 550 | 10902 | 0 |
| 100 | **10890** | 10888 | 10882 | 10890 | 10890 | 5 | 550 | 10890 | 0 |
| 125 | **10892** | 10892 | 10892 | 10892 | 10892 | 5 | 558 | 10892 | 0 |
| 150 | **10902** | 10898 | 10902 | 10898 | 10902 | 5 | 562 | 10902 | 0 |
| 175 | **12060** | 11660 | 12060 | 11642 | 12060 | 6 | 12 | 12060 | 0 |
| 200 | **12392** | 12392 | 11684 | 11674 | 12392 | 6 | 16 | 12392 | 0 |
| 225 | **14538** | 14374 | 14476 | 14538 | 14538 | 6 | 42 | 14538 | 0 |
| 250 | **18396** | 18396 | 17978 | 18116 | 18396 | 6 | 119 | 18396 | 40 |
| 275 | **17770** | 17770 | 17480 | 17416 | 17770 | 6 | 126 | 17770 | 204 |
| 300 | **20920** | 20770 | 20774 | 20920 | 20920 | 6 | 240 | 20920 | 1536 |
| 325 | **21456** | 21456 | 21368 | 21278 | 21456 | 6 | 286 | 21456 | 1872 |
| 350 | **18880** | 18880 | 18822 | 18676 | 18880 | 6 | 280 | 18880 | 920 |
| 375 | **18950** | 18826 | 18950 | 18830 | 18950 | 6 | 302 | 18950 | 1968 |
| 400 | **19178** | 19178 | 19150 | 19178 | 19178 | 6 | 316 | 19178 | 976 |
| 425 | **17824** | 17824 | 17728 | 17706 | 17824 | 6 | 278 | 17824 | 364 |
| 450 | **17504** | 17496 | 17502 | 17504 | 17504 | 6 | 370 | 17504 | 584 |
| 475 | **19160** | 19144 | 19080 | 19160 | 19160 | 6 | 374 | 19160 | 666 |
| 500 | **22500** | 22476 | 22476 | 22500 | 22500 | 6 | 402 | 22500 | 1096 |
| 525 | **22296** | 22210 | 22296 | 22230 | 22296 | 6 | 416 | 22296 | 1140 |
| 550 | **23698** | 23698 | 23614 | 23644 | 23698 | 6 | 428 | 23698 | 1240 |
| 575 | **23418** | 23414 | 23418 | 23276 | 23418 | 6 | 368 | 23418 | 1184 |
| 600 | **23166** | 23102 | 23166 | 23070 | 23166 | 6 | 328 | 23166 | 1108 |
| 625 | **23786** | 23738 | 23740 | 23786 | 23786 | 6 | 340 | 23786 | 1120 |
| 650 | **23880** | 23880 | 23718 | 23676 | 23880 | 6 | 316 | 23880 | 860 |
| 675 | **25134** | 25134 | 24820 | 24804 | 25134 | 6 | 376 | 25134 | 1134 |
| 700 | **27006** | 26702 | 27006 | 26354 | 27006 | 7 | 8 | 27006 | 1192 |
| 725 | **24656** | 24656 | 24598 | 24656 | 24656 | 6 | 294 | 24656 | 894 |
| 750 | **25116** | 25116 | 25116 | 25110 | 25116 | 6 | 332 | 25116 | 1036 |
| 775 | **26776** | 26772 | 26776 | 26290 | 26776 | 7 | 8 | 26776 | 1044 |
| 800 | **29168** | 29168 | 28128 | 28328 | 29168 | 7 | 28 | 29168 | 1646 |
| 825 | **28418** | 28408 | 28418 | 27590 | 28418 | 7 | 24 | 28418 | 1440 |
| 850 | **31568** | 31466 | 30948 | 31568 | 31568 | 7 | 40 | 31568 | 1940 |
| 875 | **29156** | 29086 | 29156 | 29064 | 29156 | 7 | 40 | 29156 | 1948 |
| 900 | **33138** | 33138 | 32916 | 33130 | 32768 | 7 | 60 | 32768 | 2858 |
| 925 | **32814** | 32814 | 32530 | 32798 | 32768 | 7 | 48 | 32768 | 2346 |
| 950 | **31078** | 31078 | 31068 | 31078 | 31078 | 7 | 32 | 31078 | 1768 |
| 975 | **32112** | 31896 | 31696 | 32112 | 32112 | 7 | 38 | 32112 | 2012 |
| 1000 | **34182** | 33916 | 34182 | 34116 | 32768 | 7 | 72 | 32768 | 4878 |
| 1025 | **33700** | 33472 | 33700 | 33108 | 32768 | 7 | 60 | 32768 | 3816 |
| 1050 | **33442** | 33442 | 33118 | 33150 | 32768 | 7 | 72 | 32768 | 4183 |
| 1075 | **33952** | 33946 | 33782 | 33952 | 32768 | 7 | 90 | 32768 | 5630 |
| 1100 | **32408** | 32408 | 32398 | 32404 | 32408 | 7 | 60 | 32408 | 3132 |
| 1125 | **31650** | 31548 | 31596 | 31650 | 31650 | 7 | 54 | 31650 | 2556 |
| 1150 | **33046** | 32908 | 33002 | 33046 | 32768 | 7 | 96 | 32768 | 4014 |
| 1175 | **34116** | 33992 | 33902 | 34116 | 32768 | 7 | 120 | 32768 | 5998 |
| 1200 | **32934** | 32876 | 32934 | 32890 | 32768 | 7 | 90 | 32768 | 3532 |
| 1225 | **34828** | 34828 | 34492 | 34790 | 32768 | 7 | 108 | 32768 | 7747 |
| 1250 | **35258** | 35258 | 34698 | 35212 | 32768 | 7 | 120 | 32768 | 9074 |
| 1275 | **35196** | 35136 | 35022 | 35196 | 32768 | 7 | 88 | 32768 | 7097 |
| 1300 | **34594** | 34540 | 34594 | 34530 | 32768 | 7 | 102 | 32768 | 7226 |
| 1325 | **36490** | 36488 | 36478 | 36490 | 32768 | 7 | 128 | 32768 | 12073 |
| 1350 | **36528** | 36508 | 36528 | 36518 | 32768 | 7 | 128 | 32768 | 12478 |
| 1375 | **36630** | 36622 | 36630 | 36600 | 32768 | 7 | 132 | 32768 | 13368 |
| 1400 | **35684** | 35670 | 35684 | 35662 | 32768 | 7 | 120 | 32768 | 10788 |
| 1425 | **35908** | 35712 | 35908 | 35868 | 32768 | 7 | 118 | 32768 | 10866 |
| 1450 | **36626** | 36620 | 36116 | 36626 | 32768 | 7 | 104 | 32768 | 10928 |
| 1475 | **35348** | 35330 | 34908 | 35348 | 32768 | 7 | 126 | 32768 | 10117 |
| 1500 | **34066** | 34066 | 34048 | 34064 | 32768 | 7 | 152 | 32768 | 8285 |
| 1525 | **34078** | 34078 | 34042 | 34078 | 32768 | 7 | 142 | 32768 | 8252 |
| 1550 | **34656** | 34656 | 34640 | 34656 | 32768 | 7 | 160 | 32768 | 10486 |
| 1575 | **32082** | 32082 | 32036 | 32082 | 32082 | 7 | 48 | 32082 | 2960 |
| 1600 | **33134** | 33134 | 33074 | 33134 | 32768 | 7 | 84 | 32768 | 4044 |
| 1625 | **33924** | 33924 | 33898 | 33924 | 32768 | 7 | 112 | 32768 | 6340 |
| 1650 | **37678** | 37504 | 37678 | 37504 | 32768 | 7 | 168 | 32768 | 15503 |
| 1675 | **33412** | 33412 | 33340 | 33412 | 32768 | 7 | 142 | 32768 | 6202 |
| 1700 | **33962** | 33962 | 33834 | 33932 | 32768 | 7 | 133 | 32768 | 7092 |
| 1725 | **30370** | 30216 | 30198 | 30370 | 30370 | 7 | 32 | 30370 | 1976 |
| 1750 | **30264** | 30156 | 30226 | 30264 | 30264 | 7 | 28 | 30264 | 1732 |
| 1775 | **32080** | 32080 | 31950 | 32080 | 32080 | 7 | 64 | 32080 | 3012 |
| 1800 | **34484** | 34484 | 34340 | 34484 | 32768 | 7 | 120 | 32768 | 8570 |
| 1825 | **35302** | 35302 | 35302 | 35302 | 32768 | 7 | 156 | 32768 | 13291 |
| 1850 | **34744** | 34744 | 34740 | 34744 | 32768 | 7 | 140 | 32768 | 10368 |
| 1875 | **36522** | 36442 | 36416 | 36522 | 32768 | 7 | 128 | 32768 | 13450 |
| 1900 | **36824** | 36824 | 36772 | 36818 | 32768 | 7 | 98 | 32768 | 11325 |
| 1925 | **36534** | 36534 | 36512 | 36480 | 32768 | 7 | 92 | 32768 | 10509 |
| 1950 | **36560** | 36560 | 36552 | 36538 | 32768 | 7 | 96 | 32768 | 10498 |
| 1975 | **35430** | 35430 | 35254 | 35338 | 32768 | 7 | 74 | 32768 | 7067 |
| 2000 | **33692** | 33692 | 33600 | 33560 | 32768 | 8 | 7 | 32768 | 3879 |

**Level 4, np4, all four ranks.** Each cell is the worst of the three draws at that (state, rank). `fb` is the global reduction, from draw E.

| step | r0 | r1 | r2 | r3 | depths r0-r3 | fb (global) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8151 | 5969 | 5980 | 5823 | 5/5/5/5 | 0 |
| 25 | 8098 | 5949 | 5994 | 5926 | 5/5/5/5 | 0 |
| 50 | 8083 | 5975 | 6033 | 5872 | 5/5/5/5 | 0 |
| 75 | 8079 | 5936 | 5981 | 5834 | 5/5/5/5 | 0 |
| 100 | 8075 | 5935 | 5947 | 5775 | 5/5/5/5 | 0 |
| 125 | 8106 | 5967 | 5970 | 5855 | 5/5/5/5 | 0 |
| 150 | 8153 | 5998 | 6019 | 5870 | 5/5/5/5 | 0 |
| 175 | 8489 | 6550 | 6497 | 6083 | 6/6/6/6 | 0 |
| 200 | 8404 | 6660 | 6580 | 6384 | 6/6/6/6 | 0 |
| 225 | 9357 | 7103 | 6916 | 7087 | 6/6/6/6 | 0 |
| 250 | 11061 | 8683 | 8622 | 8438 | 6/6/6/6 | 60 |
| 275 | 10173 | 8177 | 8333 | 7873 | 6/6/6/6 | 110 |
| 300 | 11433 | 9036 | 9171 | 9002 | 6/6/6/6 | 1556 |
| 325 | 11882 | 9837 | 9728 | 9643 | 6/6/6/6 | 1896 |
| 350 | 11131 | 8685 | 8576 | 8609 | 6/6/6/6 | 920 |
| 375 | 10812 | 8776 | 8739 | 8712 | 6/6/6/6 | 1920 |
| 400 | 11210 | 9117 | 9099 | 8830 | 6/6/6/6 | 976 |
| 425 | 10939 | 8850 | 8929 | 8684 | 6/6/6/6 | 364 |
| 450 | 10922 | 8919 | 8958 | 8823 | 6/6/6/6 | 604 |
| 475 | 11551 | 9442 | 9637 | 9186 | 6/6/6/6 | 666 |
| 500 | 13076 | 11079 | 10997 | 10741 | 6/6/6/6 | 1096 |
| 525 | 13402 | 11341 | 11153 | 10940 | 6/6/6/6 | 1164 |
| 550 | 13773 | 12015 | 11996 | 11725 | 6/6/6/6 | 1288 |
| 575 | 13267 | 11428 | 11731 | 11466 | 6/6/6/6 | 1184 |
| 600 | 12939 | 11416 | 11174 | 11081 | 6/6/6/6 | 1108 |
| 625 | 13394 | 11301 | 11852 | 11398 | 6/6/6/6 | 1032 |
| 650 | 13027 | 11317 | 11501 | 11015 | 6/6/6/6 | 860 |
| 675 | 13547 | 11939 | 11939 | 11762 | 6/6/6/6 | 1134 |
| 700 | 14232 | 12435 | 12681 | 12312 | 7/7/7/7 | 1096 |
| 725 | 13300 | 11199 | 11763 | 11485 | 6/6/6/6 | 894 |
| 750 | 13590 | 11883 | 12202 | 11873 | 6/6/6/6 | 1036 |
| 775 | 14169 | 12177 | 12476 | 12580 | 7/7/7/7 | 1044 |
| 800 | 14218 | 12909 | 12911 | 12812 | 7/7/7/7 | 1662 |
| 825 | 14075 | 12532 | 12663 | 12510 | 7/7/7/7 | 1440 |
| 850 | 15136 | 13387 | 13644 | 13521 | 7/7/7/7 | 1964 |
| 875 | 14517 | 13173 | 13270 | 13017 | 7/7/7/7 | 1948 |
| 900 | 16100 | 14518 | 14694 | 14408 | 7/7/7/7 | 2488 |
| 925 | 15663 | 14329 | 14653 | 14106 | 7/7/7/7 | 2300 |
| 950 | 15165 | 13672 | 13822 | 13746 | 7/7/7/7 | 1768 |
| 975 | 15587 | 14164 | 14314 | 13784 | 7/7/7/7 | 1996 |
| 1000 | 16317 | 14824 | 14730 | 14508 | 7/7/7/7 | 2968 |
| 1025 | 15742 | 14781 | 14624 | 14394 | 7/7/7/7 | 2622 |
| 1050 | 15710 | 14581 | 14596 | 14253 | 7/7/7/7 | 3460 |
| 1075 | 16226 | 14867 | 14977 | 14481 | 7/7/7/7 | 4134 |
| 1100 | 15874 | 14618 | 14600 | 14349 | 7/7/7/7 | 3132 |
| 1125 | 15242 | 13871 | 13974 | 13716 | 7/7/7/7 | 2524 |
| 1150 | 15717 | 14216 | 14159 | 13989 | 7/7/7/7 | 3756 |
| 1175 | 16020 | 14949 | 14613 | 14553 | 7/7/7/7 | 4212 |
| 1200 | 15840 | 14928 | 14422 | 14548 | 7/7/7/7 | 3376 |
| 1225 | 16647 | 15764 | 15383 | 15493 | 7/7/7/7 | 4332 |
| 1250 | 16304 | 15531 | 15089 | 15313 | 7/7/7/7 | 4796 |
| 1275 | 16178 | 15345 | 14911 | 14973 | 7/7/7/7 | 3920 |
| 1300 | 15855 | 14915 | 14510 | 14685 | 7/7/7/7 | 4252 |
| 1325 | 16608 | 15824 | 15371 | 15613 | 7/7/7/7 | 5188 |
| 1350 | 16970 | 16309 | 15720 | 16055 | 7/7/7/7 | 5096 |
| 1375 | 17144 | 16027 | 15919 | 15893 | 7/7/7/7 | 5262 |
| 1400 | 16674 | 15543 | 15549 | 15436 | 7/7/7/7 | 4944 |
| 1425 | 16355 | 15175 | 15195 | 15075 | 7/7/7/7 | 5116 |
| 1450 | 16189 | 15272 | 15387 | 15344 | 7/7/7/7 | 4670 |
| 1475 | 15538 | 14774 | 14829 | 14915 | 7/7/7/7 | 5454 |
| 1500 | 15513 | 14557 | 14621 | 14331 | 7/7/7/7 | 6080 |
| 1525 | 15447 | 14583 | 14720 | 14392 | 7/7/7/7 | 6158 |
| 1550 | 15676 | 15046 | 14932 | 14724 | 7/7/7/7 | 6832 |
| 1575 | 14653 | 13752 | 13852 | 13520 | 7/7/7/7 | 2960 |
| 1600 | 15092 | 14127 | 14172 | 14227 | 7/7/7/7 | 3640 |
| 1625 | 15378 | 14535 | 14292 | 14177 | 7/7/7/7 | 4512 |
| 1650 | 16657 | 15518 | 15659 | 15488 | 7/7/7/7 | 6290 |
| 1675 | 15487 | 14337 | 14381 | 14334 | 7/7/7/7 | 5380 |
| 1700 | 15713 | 14688 | 14680 | 14501 | 7/7/7/7 | 4964 |
| 1725 | 14083 | 13078 | 13091 | 12996 | 7/7/7/7 | 1976 |
| 1750 | 13647 | 12913 | 12880 | 12782 | 7/7/7/7 | 1660 |
| 1775 | 14470 | 13734 | 13854 | 13547 | 7/7/7/7 | 3012 |
| 1800 | 15558 | 14779 | 14853 | 14591 | 7/7/7/7 | 4864 |
| 1825 | 16044 | 15275 | 15189 | 15399 | 7/7/7/7 | 6278 |
| 1850 | 15970 | 15209 | 15112 | 15205 | 7/7/7/7 | 5584 |
| 1875 | 16969 | 16054 | 15981 | 15931 | 7/7/7/7 | 5542 |
| 1900 | 16654 | 15707 | 15866 | 15859 | 7/7/7/7 | 4738 |
| 1925 | 16143 | 15339 | 15492 | 15483 | 7/7/7/7 | 4514 |
| 1950 | 16629 | 15705 | 15640 | 15539 | 7/7/7/7 | 4240 |
| 1975 | 16006 | 14958 | 14854 | 14868 | 7/7/7/7 | 3214 |
| 2000 | 14997 | 14054 | 14052 | 14193 | 8/8/7/7 | 2934 |

**Level 3, np1 rank 0 — the control.** All three draws, unreduced; `op_cap` is 32 768 in every row and `global_m2l_fallback` is 0 in every row.

| step | D | E | F |
| --- | --- | --- | --- |
| 0 | 864 | 864 | 864 |
| 25 | 864 | 864 | 864 |
| 50 | 864 | 864 | 864 |
| 75 | 864 | 864 | 864 |
| 100 | 864 | 864 | 864 |
| 125 | 852 | 852 | 852 |
| 150 | 852 | 852 | 852 |
| 175 | 828 | 828 | 828 |
| 200 | 850 | 848 | 846 |
| 225 | 1338 | 1360 | 872 |
| 250 | 4348 | 3724 | 3800 |
| 275 | 1784 | 1784 | 1100 |
| 300 | 4130 | 4400 | 4432 |
| 325 | 3068 | 2944 | 2944 |
| 350 | 3748 | 3748 | 3748 |
| 375 | 3182 | 3156 | 3070 |
| 400 | 5416 | 5254 | 5538 |
| 425 | 4106 | 3988 | 4102 |
| 450 | 2174 | 2188 | 2188 |
| 475 | 2116 | 2174 | 2206 |
| 500 | 2442 | 2510 | 2630 |
| 525 | 2540 | 2632 | 2692 |
| 550 | 2540 | 2522 | 2508 |
| 575 | 2508 | 2508 | 2508 |
| 600 | 2974 | 2994 | 3018 |
| 625 | 2864 | 3014 | 2864 |
| 650 | 2714 | 2880 | 2992 |
| 675 | 1876 | 2018 | 2090 |
| 700 | 1756 | 1880 | 1746 |
| 725 | 1862 | 1990 | 2002 |
| 750 | 2186 | 2038 | 2186 |
| 775 | 2984 | 3168 | 3206 |
| 800 | 3302 | 3102 | 3358 |
| 825 | 2670 | 2438 | 3236 |
| 850 | 1984 | 1694 | 1984 |
| 875 | 1482 | 1746 | 1534 |
| 900 | 1472 | 1878 | 1616 |
| 925 | 1590 | 1594 | 1650 |
| 950 | 1614 | 1584 | 1586 |
| 975 | 1984 | 1984 | 1984 |
| 1000 | 3062 | 1966 | 2986 |
| 1025 | 3800 | 3800 | 3800 |
| 1050 | 3890 | 4012 | 4392 |
| 1075 | 4798 | 4700 | 4402 |
| 1100 | 3954 | 4344 | 3510 |
| 1125 | 2226 | 2226 | 2226 |
| 1150 | 1994 | 1994 | 1994 |
| 1175 | 1960 | 1902 | 1902 |
| 1200 | 2562 | 2596 | 2562 |
| 1225 | 2682 | 2702 | 2562 |
| 1250 | 3618 | 3660 | 3606 |
| 1275 | 4466 | 4624 | 5078 |
| 1300 | 5634 | 5512 | 5134 |
| 1325 | 2706 | 4636 | 5076 |
| 1350 | 3990 | 5420 | 5308 |
| 1375 | 3176 | 4724 | 4690 |
| 1400 | 3200 | 3516 | 3488 |
| 1425 | 3242 | 3508 | 3480 |
| 1450 | 3544 | 3510 | 3544 |
| 1475 | 3502 | 4714 | 4176 |
| 1500 | 4824 | 6246 | 5592 |
| 1525 | 4514 | 5948 | 5258 |
| 1550 | 5012 | 5794 | 5012 |
| 1575 | 5278 | 6624 | 5948 |
| 1600 | 5596 | 6196 | 5608 |
| 1625 | 3730 | 3704 | 3730 |
| 1650 | 3588 | 3560 | 3590 |
| 1675 | 3638 | 3624 | 3642 |
| 1700 | 3210 | 3210 | 3216 |
| 1725 | 3216 | 3210 | 3216 |
| 1750 | 3010 | 3012 | 3010 |
| 1775 | 2648 | 2646 | 2648 |
| 1800 | 3348 | 3162 | 3350 |
| 1825 | 2976 | 2810 | 2826 |
| 1850 | 3516 | 3648 | 3458 |
| 1875 | 4110 | 4102 | 4108 |
| 1900 | 4722 | 4752 | 4766 |
| 1925 | 5728 | 5178 | 5746 |
| 1950 | 3820 | 3854 | 3854 |
| 1975 | 3652 | 3244 | 3694 |
| 2000 | 3576 | 3166 | 3606 |

### Departures from the standing rules: none

clang-format was **not** run. Nothing was committed or pushed in this repo or
in the Canopy clone, whose T1 edits remain uncommitted working-tree
modifications at `develop` commit `d3145e0`. **No `spack install` was run and
none was needed** — T5 changed a `.flux` script and a markdown file, and the
task's own instruction was to stop and say why if a rebuild appeared necessary.
It did not. The four jobs this task started all reached `COMPLETED`; none was
left running and none needed `flux cancel`.

### The two document corrections this task was told to make

Both in `tasks/add-canopy-t6.md`, outside the T5 entry:

- `:16-19` said the level-4 run "routes about 13 000 pairs to the per-pair
  fallback" without saying at which rank count. It now says **at np1**, and
  gives np4's 5 300 (SERIAL) and 5 392 (HIP) at the same step with the
  per-rank cap as the reason.
- `:64-67` said "HIP np1 and np4 roughly 140 s each. All four together are
  about 38 minutes". It now carries the tier run's measured **84.543 s** and
  **100.154 s**, **about 35 minutes** for all four claim-A halves, and about
  three minutes for the two HIP ones, cited to
  `tasks/canopy/add-canopy-progress-log.md:2377-2386`.

**Affects:**
- **T8** — has its number: **37 678 keys worst-observed at np1, 17 144 at the
  worst np4 rank**, 1.150x the cap at np1, 115 MiB of table, run-to-run spread
  under 1.5 %, `demand_saturated` never set so branch B is **not** forced by
  saturation. Two things constrain the decision beyond the count. First,
  **branch A cannot by itself turn the member green**: at np4 the cap is never
  reached and fallback is still non-zero at 71 of 81 states, so a cap raise
  fixes zero np4 states and about half the np1 ones. Second, **branch A's
  rebuild cost is the full cap at every one of the 81 evaluations**, confirmed
  at level 4 in the regime where the cap binds. Size any raise against the
  demand peak at **step 1650**, not the error peak at 1375 — the gap is about
  1 050 keys.
- **T9a** — the pure-FMM-path check cannot be expected to come back clean from
  a cap change alone, for the np4 reason above. Its distinguishing measurement
  is still the fallback column, and this entry says that column has a
  non-cap-driven component at level 4 that nothing in T6–T8 as currently
  specified addresses. Budget: the probe's level-4 HIP pair is **113 s** of
  launch time, so a `pdebug` re-measurement after any cap change is minutes.
- **T6, T7** — none. The knob is worth having whichever way T8 falls, and the
  default stays 32 768; this entry changed no cap, no signature and no
  tolerance.
- **T9b** — none directly, except that the scratch path is now per job, so a
  tier runner that ever reuses this script's scratch root cannot collide with
  a probe run.

## T6

Canopy only. No Beatnik file was touched and no `spack install` was run — T7
owns the Beatnik side. Two existing cmake trees rebuilt (one target each), one
`pdebug` job, and `git diff --stat` in the Canopy clone touches three source
files plus one new untracked script.

### Decisions taken as given by the task, recorded so they are not reopened

- **`with_laplace_solve` gained a SECOND optional parameter**, `int
  m2l_op_count_cap = 0`, applied beside the byte budget at `:906-909`,
  mirroring the byte budget's pattern exactly rather than introducing a config
  struct. Without it the "small count cap with a generous byte budget" case
  cannot be driven at all.
- **Both new assertions are UNGATED.** `m2lOpCountCapConstrained`'s `eff_cap`,
  `realized` and `fallback` checks and `m2lKeyDemandDefault`'s
  `eff_cap == 32768` check are claims about the cap in force and the columns
  admitted, not about profiling state, so they sit outside the
  `CANOPY_ENABLE_PROFILING` branch and were verified in the `~profiling` tree
  as well. Only the `demanded >` comparison is gated.
- **BOTH cap-printing sites were updated, not just the overflow message.** The
  overflow `fprintf` (`:1817-1826`) and the `[Canopy Diagnostics]` profiling
  `printf` (`:1879-1891`) each printed the constant `M2L_OP_COUNT_CAP` as their
  "count cap" field; both now print `_m2l_op_count_cap`. The task's Do step 6
  names only the first, and leaving the second would have made every profiling
  log report `count_cap=32768` for a run configured otherwise — exactly the
  class of silent wrong number this topic exists to avoid.
- **Nothing was committed, pushed or pulled in the Canopy clone.** T1's
  instrumentation is still uncommitted working-tree modifications at `develop`
  commit `d3145e0`, and `canopy@=develop` is a `spack develop` spec in the
  Beatnik env, so a pull would move `develop` past the commit those edits sit
  on. T6's changes were added to the same working tree.
- **The new flux script carries no BSD-3-Clause header**, per T1's precedent:
  no sibling in `canopy/scripts/tuolumne/` has one.
- **clang-format was not run** on any of the three edited files.

### What was added, and the signatures

All in the Beatnik env's spack `develop` source at
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`.

| addition | where |
| --- | --- |
| `int m2l_op_count_cap = 32768;` | `src/Canopy_Solver.hpp:116`, in `FmmConfig` after `m2l_op_table_byte_budget` |
| `_downward.set_m2l_op_count_cap( cfg.m2l_op_count_cap );` | `src/Canopy_Solver.hpp:217`, beside the byte-budget route at `:213` |
| `void set_m2l_op_count_cap( int )` | `src/Canopy_DownwardSweep.hpp:350-359` |
| `int m2l_op_count_cap() const` | `:361` |
| `int _m2l_op_count_cap = M2L_OP_COUNT_CAP;` | `:855`, beside `_m2l_op_table_byte_budget` at `:847` |
| `#include <stdexcept>` | with the other standard includes |
| `LS_COUNT_CAP_KEYS = 4`, `LS_DEFAULT_OP_COUNT_CAP = 32768` | `tests/tstLaplaceSolve.hpp`, after `LS_DEMAND_BUDGET_KEYS` |
| `testM2LOpCountCapConstrained`, `testM2LOpCountCapBounds`, their two `TEST()` registrations | `tests/tstLaplaceSolve.hpp` |
| `run_t6_count_cap.flux` | `canopy/scripts/tuolumne/` (new, untracked) |

**One existing signature changed**, and it is a test-only one:
`with_laplace_solve` is now
`( Fn&& after, std::size_t m2l_op_table_byte_budget = 0, int
m2l_op_count_cap = 0 )`. Both knobs default to 0 and every existing call site
is unchanged. `m2l_effective_op_cap()`'s signature did not change — only the
term it reads, from `M2L_OP_COUNT_CAP` to `_m2l_op_count_cap` — so none of its
six callers needed editing, as the task's step 8 predicted.

The `[laplace-solve]` driver line gained **one field**, `op_count_cap=%d`,
between `op_budget` and `op_cap`. That is additive: no existing field's value
moved, but a whole-line diff of the *other* tests against a pre-T6 log will
differ by that field by construction. The demand cases print their own lines
and are unaffected.

### Why the cap is 4 and not 1

`LS_COUNT_CAP_KEYS = 4` rather than reusing `LS_DEMAND_BUDGET_KEYS = 1`:
a cap of 1 would pass identically if the count cap were accepted and then
silently ignored while the one-column *byte* budget was applied instead. Four
distinguishes the two knobs. The byte budget is left at `FmmConfig`'s 2 GB
default in that case — worth **97 823 columns** at `LaplaceKernel<double, 6,
1>`'s 21 952 B per key — and the case asserts that explicitly, so a realized
count of 4 can only have come from the count cap.

`LS_DEFAULT_OP_COUNT_CAP = 32768` is written as a **literal**, deliberately
against this file's own "derive, never literal" convention: the claim is that
the default did not *move*, and a value read off the class would track any
move and assert nothing.

### Builds

Both of T1's trees, same environment (`spack env activate
${HOME}/spack_envs/tuolumne_trilinos`; Canopy is **manual** mode), one target
each, on the login node. `make -j Canopy_Test_LaplaceSolve_MPI_SERIAL`
recompiled one TU and relinked in each; both clean on the first attempt, no
warnings surfaced. Nothing was reconfigured.

### Measured: job `f3bZMnmTkWbZ`, both trees, ranks 1-6

`scripts/tuolumne/run_t6_count_cap.flux`, `pdebug`, one node,
`--time-limit=40`, one job running `ctest -V -R
Canopy_Test_LaplaceSolve_MPI_SERIAL` in each tree.
**`100% tests passed, 0 tests failed out of 6` in both**, combined rc 0. The
suite is now **eight** gtest cases per rank count (six plus the two new ones);
the six ctest "tests" are the six rank counts.

Over the **21 `(nprocs, rank)` pairs**:

| case | `ON` tree | `OFF` tree |
| --- | --- | --- |
| `m2lOpCountCapConstrained`, cap 4 | `eff_cap=4 realized=4` everywhere, `budget=2147483648`; **`demanded` 111 .. 718**, strictly > 4 at every pair; `saturated=0`; `fallback` **175 .. 1 677**, positive everywhere | identical but **`demanded=-1`** |
| `m2lKeyDemandDefault` | `eff_cap=32768` at every pair | same |
| `m2lOpCountCapBounds` | `default_cap=32768 zero_cap_eff=0 floored_eff=1 negative_raises=1` | same |

**The constrained case's demanded range 111 .. 718 is exactly T1's
one-column-budget range**, which is a free cross-check worth keeping: the two
caps refuse from the same demanded set, so the count cap is not perturbing
what the merge sees, only how many of those keys it admits.

### Byte-identity, scoped to the two demand cases

Scoped per the task, because T1 measured pre-existing run-to-run
nondeterminism in the LaplaceSolve tree/partition path at np ≥ 3 that the
unmodified `OFF` tree exhibits against itself. The comparison run was
`f3bZMnmTkWbZ` against T1's `f3bPfi66qz4X`
(`canopy_t1_demand.f3bPfi66qz4X.log`, still in the Canopy checkout root), full
`[laplace-solve] ... m2l_demand_constrained` and `... m2l_demand_default`
lines, sorted and diffed:

| case | tree | result |
| --- | --- | --- |
| `m2l_demand_constrained` | `ON` | **byte-identical**, 21/21 |
| `m2l_demand_constrained` | `OFF` | **byte-identical**, 21/21 |
| `m2l_demand_default` | `ON` | **byte-identical**, 21/21 |
| `m2l_demand_default` | `OFF` | **byte-identical**, 21/21 |

That is every field of those lines — `eff_cap`, `realized`, `demanded`,
`saturated`, `fallback`, the verbatim `realized_keys` string and
`cells_at_depth`. **Which it was, for the record: byte-identical, with no
appeal to the documented np-3/np-6 instability needed.** It did not fire in
these cases this time, exactly as T1 measured it does not.

### What only building or running revealed

- **Nothing failed.** Both trees compiled on the first attempt and the single
  job came back rc 0 on the first submission, so there is no bug to record
  here. Recording the absence deliberately: a knob whose default is the
  constant it replaces is a change with no runtime surface until something
  sets it, and the measurement confirms that rather than assuming it.
- **A count cap of 0 cannot be driven through `with_laplace_solve`**, because
  0 is its "leave the config default" sentinel — the pattern the byte budget
  established and the decision above fixed. Rather than invent a second
  sentinel, the zero case is checked by `testM2LOpCountCapBounds`, which
  constructs a bare `DownwardSweep` (its constructor is two `MPI_Comm_*`
  calls, nothing collective or allocating) and asserts on
  `m2l_effective_op_cap()` with no solve at all. That also makes the
  negative-raises and still-floored-by-the-budget checks free. Zero columns
  then implies the full-fallback path by construction: the merge's admit test
  is `ops.size() < effective_op_cap`, which no key satisfies at 0.
- **`DownwardSweep` had no `throw` of its own**, so `#include <stdexcept>` was
  added. The idiom copied is `TreeBuilder`'s `std::runtime_error`
  (`Canopy_TreeBuilder.hpp:179`); `std::invalid_argument` would have been more
  precise but would have been the only one in the library.
- The `ctest` wall was **about the same as T1's** despite two added cases —
  one of them does no solve and the other is one more frozen-configuration
  solve. `--time-limit=40` was far more than needed and was kept from T1's
  runner unchanged.

### Departures from T6's stated Do steps

- **Do step 6 was done twice**, at the profiling printf as well as the
  overflow message — see the decisions above.
- **Step 8's caller list dropped one entry.** It names "the profiling printf
  (`:1683`)" *and* "the two message sites", but the profiling printf **is**
  one of the two message sites; after T1's and T6's insertions there are
  exactly two in this file, at `:1822` and `:1888`, plus the three
  non-printing reads at `:1478`, `:1934` and `:1955` and Beatnik's
  `Beatnik_FarFieldInterface.hpp:890`. The T6 entry now lists five, not six.
- **The task document's `Canopy_DownwardSweep.hpp` citations were corrected**
  throughout, not only in the T6 entry — they predated T1's roughly 124
  inserted lines and T6 added about 100 more. `tasks/abstract-solver-backend.md`
  in the Canopy checkout was likewise updated: the constant's stale `:343`
  became `:607`, and the note now says the cap is configurable with that
  constant as its default, still a count and still floored by the budget.
- **The deviation note gained the CartesianTaylor arithmetic** per step 7, and
  the contrast is sharper than the note implied: at 58 KB per key at $P=8$ the
  count cap binds by a factor of about **1.1**, and at CartesianTaylor order
  3's 3200 B per key the 2 GB budget buys **671 088** columns, so the count
  cap binds by a factor of **20** and is the only constraint that ever binds
  there.

**Affects:**
- **T7** — the Canopy side is exactly as this document specified and the
  plumbing target is `FmmConfig::m2l_op_count_cap` (`Canopy_Solver.hpp:116`),
  an `int` defaulting to 32768. Two things shape T7's work. First, **a
  negative value throws** from `DownwardSweep::set_m2l_op_count_cap()` during
  the `Canopy::Solver` constructor, so a `FmmParams` member that reaches it
  unvalidated turns a Beatnik CLI typo into a constructor exception, not a
  clamp — decide deliberately whether `FmmParams` validates earlier or lets it
  throw. Second, **0 is legal and means zero columns**, so the
  "0 means leave the default" convention `with_laplace_solve` uses must NOT be
  copied into `FmmParams`: the Beatnik member should carry 32768 as its own
  default, matching `m2l_op_table_byte_budget`'s pattern, not a 0 sentinel.
- **T8** — the knob it chooses a value for now exists and is reachable, and
  two measurements here bear on the choice. The byte budget is **not** the
  binding constraint on the CartesianTaylor arm by a factor of 20, so T8 is
  choosing the count and only the count. And `m2l_demand_saturated()` was
  `false` at every pair here as it was in T1, so branch B's presentation is
  still untested in practice.
- **T9a, T9b** — none. T6 changed no default, no tolerance, no Beatnik gate
  member and no milestone member — the two new cases live in Canopy's own
  `Canopy_Test_LaplaceSolve_MPI_SERIAL` suite — so Beatnik's gate is still
  five `regression` members and 60 launches.

## T7

Beatnik only, no Canopy file touched. Four existing files edited —
`src/Beatnik_Params.hpp` (**+47 −7**), `src/Beatnik_FarFieldInterface.hpp`
(**+5 −2**), `tests/regression_tests/Beatnik_Probe_FmmKeyDemand.cpp`
(**+59 −12**) and `README.md` (**+2 −1**) — and one new script,
`scripts/tuolumne/t7_cap_knob.flux`. One `spack install`, two `pdebug` jobs.

### Decisions taken as given by the task, recorded so they are not reopened

- **The default is 32768 and T7 did not change it.** `FmmParams` ships the
  constant that was already in force, so every existing configuration's
  overflow set is unchanged bit for bit. **T8 chooses the value**, and a T7
  that had also raised the cap would have destroyed T8's before/after
  measurement.
- **No CLI option and no Python counterpart**, matching
  `m2l_op_table_byte_budget`. The probe's `argv[2]` is a measurement driver's
  argument, not a CLI surface on the solver.
- **`FmmParams` does not validate.** A negative value throws from
  `DownwardSweep::set_m2l_op_count_cap()` inside the `Canopy::Solver`
  constructor, and Beatnik lets it: with no CLI option the only route to one is
  a programmer's literal, Canopy already rejects it, and the byte budget beside
  it is likewise unvalidated here. The doc comment states this rather than the
  code policing it.
- **0 is legal and means zero.** Canopy's test-only `with_laplace_solve`
  convention, where 0 means "leave the config default", is NOT copied — T6's
  `**Affects:** T7` line asked for exactly this and it was followed.
- **`Beatnik_Test_Milestone0Fmm.cpp` was not touched.** Its `makeFmmParams` is
  T8's.
- **clang-format was not run**, per the standing rule. Nothing was committed or
  pushed, in this repo or in the Canopy clone; T1's and T6's Canopy edits remain
  uncommitted working-tree modifications at `develop` commit `d3145e0`, and
  `spack develop` compiled them in place. The clone was not pulled.

### The probe's changed argument contract

**`argv[2]` is new and optional: `m2l_op_count_cap`, absent meaning the
`FmmParams` default.** The ARGUMENTS block previously said "there is no option
surface here and none may be added"; it now says there is no *option* surface
(the arguments are positionals, from the batch script) and restates the
step-count refusal as the standing rule it actually is — "the step count stays
the compiled `kSteps` and no argument will ever move it". The cap is admitted on
that same reasoning: it shortens nothing, all 81 states are still measured, and
it is printed in the header so a run records what it was configured with.

Three mechanical consequences worth knowing before the next edit:

- **`makeFmmParams` gained a defaulted parameter**, `int op_count_cap = -1`,
  where `-1` is the "absent" sentinel and **not** reachable from the command
  line: `runProbe` rejects a negative `argv[2]` through `rec.fail` first, the
  way the level is rejected. So the sentinel cannot collide with a caller's
  value, and an omitted argument leaves the member's configuration exactly as
  `Beatnik_Test_Milestone0Fmm.cpp` has it.
- **An empty `argv[2]` is a trap the runner pays for, not the probe.**
  `std::atoi("")` is 0, which is a *legal* cap meaning "admit no column", so a
  script that passed an empty string would measure a different configuration
  and still exit 0. `t7_cap_knob.flux` therefore builds its argument vector
  conditionally and omits the argument rather than passing `""`; the comment
  there says why.
- **The `[t6probe] header` line gained `op_count_cap=` between `byte_budget=`
  and `op_cap=`.** Any script parsing that line positionally will need
  updating; `t6b_key_demand.flux` does not parse it and was not touched. The
  distinction the two fields carry is the whole point: `op_count_cap` is what
  was **configured** (read back out of `fmm.farField().params()`), `op_cap` is
  what is **in force** (`m2l_effective_op_cap()`, the smaller of the count cap
  and what the byte budget buys).

### The doctrine paragraph, and the figures that replaced its arithmetic

`Beatnik_Params.hpp`'s byte-budget paragraph reasoned from "at `order` ≤ 4 a
column costs at most about 9.8 KB, so the full 32768 keys occupy roughly
0.3 GiB", which makes the two constraints look close. At CartesianTaylor order 3
a column is **3200 B**, so 2 GiB buys **671 088** columns and the count cap
binds by a factor of **20** — it is the only constraint that ever binds on this
path. That figure and T5's level-4 peak (**37 678** keys worst-observed at HIP
np1 step 1650, 1.150x the cap, 115 MiB of table, 17 144 at the worst np4 rank)
are now carried in the comment. The statement that lowering the byte budget is
the wrong lever is kept.

`FarFieldDiagnostics::local_m2l_op_cap`'s comment no longer says "Canopy's own
32768-key count cap": it names `FmmParams::m2l_op_count_cap` and says the
default is 32768 and that the cap is configurable rather than a constant.

### Build

`spack install` in the dev env, **rc 0 in 5 m 47 s** (347 s), against T3's
12 m 20 s and T4's 12 m 52 s for a full tree. `HIPCC_LINK_FLAGS_APPEND` and
`HIPCC_COMPILE_FLAGS_APPEND` were cleared before installing, per the system doc.
`canopy@develop+profiling` hash `w4woraj` was cached and did not rebuild;
`beatnik@develop` came back `nnbspfy`. No `touch` was needed — `Beatnik_Params.hpp`
is a header, but `Beatnik_Probe_FmmKeyDemand.cpp` is a TU in the same build and
changed too, so the header-only no-op trap the system doc warns about did not
apply. The install fit inside one command timeout for the first time in this
topic, which is a consequence of the trim below rather than of anything else.

**THE BUILD WAS TRIMMED AND THE TRIM WAS KEPT FOR THE FINAL BUILD, ON THE
USER'S INSTRUCTION.** The prompt's own plan was to trim for iteration and then
revert and run one full `spack install`; the user said to skip the full build as
well. So **"`spack install` succeeds" is verified for the two exit-criterion
targets, not for the whole project**: `tests/CMakeLists.txt` had every
`BEATNIK_REGRESSION_TEST_SOURCES` entry, three of four
`BEATNIK_MILESTONE_TEST_SOURCES`, two of three `BEATNIK_DRIVER_SOURCES` and
`add_subdirectory(unit_tests)` commented out, the root `CMakeLists.txt` had
`add_subdirectory(examples)` commented out, and
`cmake/test_harness/test_harness.cmake` had `set(BEATNIK_TEST_DEVICES HIP)`
appended after the device loop. The build log confirms the trim bit: exactly two
targets compiled, at 25 % and 50 %. Two things follow and both matter.

- **The trims are fully reverted in the tree and were never committed.**
  `git diff --name-only` carries no `cmake/` file, no `CMakeLists.txt` and no
  `tests/CMakeLists.txt`; T7's diff is four files plus the new script. A stale
  trim is how the gate silently shrinks, so this was checked rather than
  assumed.
- **The INSTALLED VIEW IS STILL THE TRIMMED ONE.** The spack prefix currently
  holds only `Beatnik_Test_Milestone0Fmm_MPI_HIP` and
  `Beatnik_Probe_FmmKeyDemand_MPI_HIP`; there is no regression binary, no unit
  test, no example and no SERIAL target in it, and
  `beatnik_gate_manifest.txt` is correspondingly empty. **Any later task that
  runs the gate, the unit tier, the milestone tier or a SERIAL anything must
  `spack install` the reverted tree first.** It is one full install away and
  nothing is lost, but a gate run against this prefix would report a clean pass
  over zero tests.

### Measured: job `f3bZYbHfZ7bM` — the failure-direction pair

`scripts/tuolumne/t7_cap_knob.flux`, `-q pdebug -t 30m`, HIP level 3 np1, two
launches, at commit `37f9128` + 8 modified files. `flux job status` rc 0,
`SUMMARY: PASS (2/2 launches)`, **37 s total job wall** (21 s and 16 s), both
launches `[PASS] Beatnik_Probe_FmmKeyDemand (174/174 checks)` and 81 rows.

| launch | `op_count_cap` | `op_cap` | fallback over 81 rows | `unique_ops` | `demand` | `first_exceed_step` |
| --- | --- | --- | --- | --- | --- | --- |
| no `argv[2]` | **32768** | **32768** | **0 in 81 of 81** | 828 – 5 790 | 828 – 5 790 | `-1` |
| `argv[2]=1024` | **1024** | **1024** | **non-zero in 72 of 81**, peak 14 451 at step 400 | 828 – **1 024** | 828 – **6 178** | **225** |

**Both header fields move together, and that is the whole result.** A knob that
is accepted and dropped produces a byte-identical run at the default, which is
why one launch could not have shown anything; and `op_count_cap` moving while
`op_cap` stayed at 32768 would have meant Beatnik stored the value and Canopy
never saw it. `op_cap` is `m2l_effective_op_cap()` read back through the
diagnostics, so the second field moving is Canopy's own answer.

Four internal consistencies, none of them assumed:

- **`unique_ops` clamps at exactly 1 024 while `demand` still reaches 6 178.**
  Keys are refused, not un-demanded — the demand counter is measuring the same
  key set in both launches and only admission moved.
- **The 72 non-zero-fallback rows are exactly the 72 rows with
  `demand > op_cap`.** Not approximately: the two sets coincide.
- **The 9 zero-fallback rows in the capped launch are the 9 shallow states**,
  steps 0 through 200, all at `occupied_depths=4` with demand 828–864, which is
  genuinely under 1 024. So the cap binds from step 225 onward and not before,
  and `first_exceed_step=225` agrees.
- **The `[Canopy] M2L op count exceeded cap` warning appears 72 times**, again
  exactly the over-cap count. T5 found the warning faithful at np1 and silent at
  np4; this is np1 and it is faithful.

The default launch's peak demand is **5 790 at step 1300**, inside the five-draw
5 586 – 6 624 level-3 band `## T5` widened, so the apparatus is where it was.
`demand == unique_ops` in all 81 default rows, as at T3 and T4 and for the same
reason — the cap does not bind there. `global_p2p_frac` is 0.70876 – 0.94509 at
the default and 0.70737 – 0.94337 capped, both far past level 4's
`kP2PFractionBound = 0.75`, which is the level-3 member's declared
`kFarFieldIsLive = false` showing through and the reason the probe asserts on
nothing.

### Measured: job `f3bZYbR41Y31` — the default direction

`scripts/tuolumne/t6_l3_member.flux HIP`, rc 0, `SUMMARY: PASS (2/2 launches)`,
**615 s total**. The runner prints its own loud
`*** BACKEND SET OVERRIDDEN: HIP ***` line, which is correct and expected — the
member's full share of the tier is SERIAL and HIP, and only HIP was run.

| launch | wall | checks |
| --- | --- | --- |
| HIP np1 | **308 s** | **`3097/3097`** |
| HIP np4 | 307 s | `3097/3097` rank 0, `2919/2919` on the other three |

**308 s against T2's 316 s budget and the same 3097/3097 check count**, which is
the comparison that matters — the trajectory is run-to-run nondeterministic and
the wall time is a budget, not an assertion. np4 came free in the same job and
was not required by the exit criterion.

### What only building or running revealed

- **Nothing failed.** No build error, no failed launch, no resubmission. Both
  jobs came back rc 0 on the first submission. Recorded deliberately: a knob
  whose default is the constant it replaces has no runtime surface until
  something sets it, and the capped launch is what turns that from an
  assumption into a measurement.
- **A default-only verification would have proved nothing, and this is worth
  stating because it is cheap to get wrong.** The level-3 member passing
  unchanged is consistent with the knob being routed AND with it being dead
  code. Only the capped launch separates them, and it costs 16 s.
- **The trim is a large lever on this topic's iteration cost** — 5 m 47 s
  against 12 m 52 s, better than 2x — and the build log's target percentages
  (25 %, 50 %) are the cheapest check that it bit. The `FATAL_ERROR` guards on
  the argument-list loops never fired, because sources were commented out and
  argument lists were not.
- **`pdebug` absorbed both submissions immediately**; neither job spent
  measurable time in `SCHED`.

### Departures from T7's stated Do steps

- **`README.md` WAS edited, against Do step 6's expectation.** The step's own
  test — does an example's accepted arguments change? — is satisfied: they do
  not, the member has no CLI option, and the probe is a measurement driver in no
  tier rather than an example. But `README.md:349-351` is a table titled "The
  FMM-only members, none of which has a CLI option", enumerating exactly these
  public `FmmParams` members, and CLAUDE.md's "Keep `README.md` in sync" rule
  covers a public API addition. A row for `m2l_op_count_cap` was added and the
  byte-budget row beside it corrected, since it said "Canopy's own 32768-key
  count cap binds first" of what is now a configurable default. Recorded as a
  departure rather than done quietly.
- **The final build was NOT the full untrimmed one the prompt specified**, on
  the user's instruction — see Build above for what that scopes the
  `spack install` claim to and for the state the installed view is left in.
- **`max_depth`'s doc comment (`:272-291`) was left alone.** It also refers to
  "Canopy's 32768-key cap", which is now this member's default rather than a
  constant, but Do step 3 names only the byte-budget paragraph and editing it
  would have been a drive-by. Flagged here instead: it is stale in wording, not
  in arithmetic.

**Affects:**
- **T8** — the knob it sets exists, is reachable, and is **demonstrated** to
  reach Canopy rather than assumed to. Four things shape its work. First,
  **T8 changes a value and nothing else**: the plumbing, the doc comments and
  the README row are done, so a cap change is one literal in
  `FmmParams::m2l_op_count_cap`'s default or in the level-4 member's
  `makeFmmParams`. Second, **the probe can now drive a cap from the command
  line**, so T8 can measure a candidate cap at level 4 without rebuilding —
  `beatnik_exe Beatnik_Probe_FmmKeyDemand_MPI_HIP 4 <cap>` — which makes
  branch A's "does the demand fit" question a 60 s launch rather than an
  install. Third, **`t7_cap_knob.flux` is the shape of a before/after pair** and
  T8's exit criterion wants exactly that at level 4; it is level-3-and-np1 by
  construction and should be copied rather than edited. Fourth, **the installed
  view is trimmed** (see Build), so T8 must `spack install` the reverted tree
  before running any member other than `Beatnik_Test_Milestone0Fmm_MPI_HIP`.
- **T9a, T9b** — the installed view is trimmed, and **T9b's full milestone tier
  cannot run against this prefix at all**: three of its four members and the
  entire SERIAL half are not built. T9b's Do step 1 already requires a
  finalizing `spack install` before submitting, so this costs nothing as long as
  that step is not skipped on the grounds that a recent install exists. Nothing
  else: T7 changed no default, no tolerance, no gate member and no milestone
  member.
- **The gate is unchanged** — still five `regression` members and 60 launches on
  tuolumne. T7 added no test to any tier; the probe is in none, and the only new
  file is a batch script.
