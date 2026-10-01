# Canopy M2L operator-key demand: measurement, then the cap

**Status:** IN PROGRESS

## Problem

`Beatnik_Test_Milestone0FmmL4` fails on all four of its launches (SERIAL and
HIP, np1 and np4). The proximate cause is Canopy's per-rank M2L operator-column
count cap:

```
[Canopy] M2L op count exceeded cap 32768 (count cap 32768, byte budget
2147483648 B at 3200 B per key); remaining pairs route to fallback path.
```

At `--icosphere-subdivisions 4` the level-4 interaction list stays under the cap
through step 225, first trips it at step 250, and by step 1375 routes about
13 000 pairs to the per-pair fallback. Four assertion sites fail as a result:
the purity precondition `p.m2l_fallback == 0`
(`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1456`) at 71 of the 81
states, the same check on claim B's final state (`:1923`), and — at four to six
of the 81 states — **both** forms in which the member asserts the accuracy
bound, `p.rel <= kTauA` (`:1461`) and `p.max_abs <= kTauA * p.scale` (`:1462`),
peaking at `1.2513e-3` against `kTauA = 1.0e-3` (defined at `:320`).

**The τ_A exceedance is not yet a measurement of the FMM.** The fallback path is
the same mathematics reassociated — a different code path, bitwise different
from the table path (`src/Canopy_CartesianTaylorBasis.hpp:510-517`) — and every
state above τ_A is a state with non-zero fallback. Level-4 claim A has therefore
never been measured on a pure FMM path. τ_A must not be widened on this
evidence.

**The demand is unmeasurable at the current cap, and that is the first thing to
fix.** The serial merge refuses a key once `ops.size()` reaches
`effective_op_cap` (`src/Canopy_DownwardSweep.hpp:1616-1645`), so
`n_unique_ops` (`:1663`), `_m2l_realized_keys` (`:1697`) and every consumer of
`m2l_n_unique_ops()` (`:934-937`) saturate at the cap by construction. They read
32768 whether the tree wants 33 000 keys or 300 000. The fallback pair count is
not a proxy: it counts *pairs* refused a column, not *keys* refused one, and
many pairs share a key.

So the cap cannot be sized. This document makes the demand observable, measures
it on Beatnik's own level-4 geometry, and then either raises the cap or reduces
the demand on that number.

### Why this does not need a 24-hour job

The failing tier run took 8.687 h over 16 launches, and the measured level-3
cost breakdown (`tasks/canopy/add-canopy-progress-log.md:2085-2100`) shows why
almost none of that is needed here:

```
claim A (2000 direct steps + 81 FMM evaluations + 82 comparator calls)  166.051 s
claim B (2000 FMM-driven steps)                                        2313.059 s
```

**Claim B is 93% of the member and is irrelevant to the demand question.** Key
demand is a property of one interaction-list build at one geometry, and the
geometries of interest are the *direct-driven* states claim A already visits —
the same states the cap trips at. Driving the direct trajectory is cheap; the
2000-step FMM-driven trajectory is what costs hours.

Measured claim-A costs at level 4, from the failing tier run: SERIAL np1
1432 s, SERIAL np4 523 s, HIP np1 and np4 roughly 140 s each. All four together
are about 38 minutes, and the two HIP launches alone about 5 minutes. **The
measurement fits `-q pdebug -t 60m`.** Only the final validation (T9b) needs
`pbatch`, and T9a exists so that it is not submitted until it is expected to
pass.

### Out of scope

- **Widening `kTauA`.** It is a claim about a parameter set, and no number
  measured on a fallback-contaminated path is evidence about it.
- **Changing the gate.** The `regression` tier keeps exactly five members and
  60 launches; nothing here adds or removes one.
- **The `milestone` tier's membership.** It keeps its four members and sixteen
  launches. The probe (T3) is registered in no tier.
- **The unpairable level-4 window at steps 1350–1900.** A known T5 finding,
  neither a pass nor a failure, and independent of the cap.
- **Setting `run_milestone.flux`'s `-t` from a measurement.** It stays at
  `-t 1440m` until a tier run passes; re-timing from a failing run whose most
  expensive member does a different amount of work than the fixed version would
  bake in a cost that is about to change.

## Approach

Four moves, in order, each cheap enough to verify before the next commits to
anything:

1. **Make demand observable** (T1, T2). Count the distinct canonical keys the
   merge *sees*, independently of how many it *admits*. The merge loop at
   `src/Canopy_DownwardSweep.hpp:1602-1645` already iterates each thread's
   distinct keys, and those keys are already canonical (`:1367-1370`, call site
   `:1495`), so the counter is one hash insert per key already in hand — not per
   pair. It feeds nothing in the solve.
2. **Validate the counter without an HPC job** (T6's unit test, runnable at T1
   time). `tests/tstLaplaceSolve.hpp:736` already takes
   `m2l_op_table_byte_budget` as its one configuration knob and applies it at
   `:848-849`. Driving it at a budget worth one column makes demand exceed
   realized by a known margin on a tree small enough to run in seconds.
3. **Measure on the real geometry** (T3, T4, T5). A standalone probe that drives
   the direct trajectory and evaluates the FMM once per checkpointed state,
   printing the demand series. Minutes on HIP.
4. **Act on the number** (T6, T7, T8). Plumb the count cap through `FmmConfig`
   and `FmmParams` with the default unchanged, then choose between raising it at
   the level-4 member and reducing the demand — on the measurement, with both
   branches specified in advance.

### The two facts that shape every decision here

**CartesianTaylor keys carry the tree level.**
`src/Canopy_CartesianTaylorBasis.hpp:486` sets `key_needs_level = true`, and
`canonicalize_key` is the identity (`:497-500`), because the operator is
physical — $b_k(R)$ at the real translation vector, whose length is the integer
offset times the half-width at the deeper depth — so two pairs with the same
integer offset at different levels have different operators and a level-blind
key would alias them (`:471-486`). Every occupied depth therefore multiplies the
realized key count. `src/Beatnik_Params.hpp:272-291` predicts exactly this
failure: a self-contacting roll-up drives the occupied-depth count up, "so every
occupied depth multiplies the realized key count against Canopy's 32768-key
cap", and "the number is reasoned, not measured, and the measurement is owed:
lower it only on evidence of realized overflow at the production
configuration". T5 supplies that evidence.

**Raising the cap costs rebuild time, not just memory.** For a
`key_needs_level` basis, `set_root_half_width` clears the entire operator cache
whenever the root half-width changes (`src/Canopy_DownwardSweep.hpp:406-416`,
recorded at `Canopy_CartesianTaylorBasis.hpp:481-485`), so on a drifting
bounding box the cache empties on every rebuild and every admitted column is
built again. A cap of $N$ keys on a drifting box means up to $N$ operator builds
per rebuild. `local_m2l_op_keys_built`
(`src/Beatnik_FarFieldInterface.hpp:279-283`) is the counter that shows it:
"climbing by the full cache size at every build is the signature of a cache that
retains nothing, which is what a level-keyed basis on a drifting bounding box
does." **So T5 must measure build cost alongside the key count, and T8 decides
on both.**

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Demand counter gating | `#ifdef CANOPY_ENABLE_PROFILING` | Zero cost and zero memory in a production build. Two mechanisms set the define, for two consumers. **Into Beatnik** (T4): Canopy's exported INTERFACE target (`canopy/src/CMakeLists.txt:27-33`), so a `canopy +profiling` spec is sufficient and no Beatnik CMake change is needed. **Into a standalone Canopy build** (T1's own two builds): the cmake option `Canopy_ENABLE_PROFILING` (`canopy/CMakeLists.txt:252-277`), as `-DCanopy_ENABLE_PROFILING=ON` or `=OFF` — `OFF` is an authoritative kill switch there, forcing the level to 0 whatever `Canopy_PROFILING_LEVEL` says. |
| "Unavailable" sentinel | `-1` | Zero demand is legal — a tree with no M2L pairs realizes no keys — so `0` must never mean "not compiled in". Every accessor and diagnostic field returns `-1` in a `~profiling` build. |
| Diagnostic set bound | `M2L_DEMAND_COUNT_CAP = 1048576` | $2^{20}$ keys, about 56 MB of `unordered_set`. Chosen to exceed what the 2 GiB byte budget could ever buy at this basis's 3200 B per key (671 088 columns), so a *saturated* demand counter already decides T8's branch without needing the exact figure. Reported through a separate `demand_saturated` flag, never conflated with the count. |
| Naming | `m2l_n_demanded_ops()`, `_m2l_demanded_op_count`, `local_m2l_demanded_op_count` | Mirrors the existing `m2l_n_unique_ops()` / `_m2l_realized_keys` / `local_m2l_unique_op_count` triple exactly. "Demanded" against "unique/realized" is the distinction the whole document turns on. |
| Cap knob name | `FmmConfig::m2l_op_count_cap`, `FmmParams::m2l_op_count_cap` | Mirrors `m2l_op_table_byte_budget` (`Canopy_Solver.hpp:95`, `Beatnik_Params.hpp:395`) in name, placement and plumbing. |
| Cap knob default | `32768`, the current `M2L_OP_COUNT_CAP` | Every existing configuration's overflow set is unchanged bit-for-bit, which is the property the deviation note protects. |
| CLI exposure | None | `m2l_op_table_byte_budget` has no CLI option and no Python counterpart; the count cap follows it. README needs no change, since no example's accepted arguments move. |
| Probe tier | None | The probe goes in the **"Measurement drivers — IN NO TIER"** loop (`tests/CMakeLists.txt:585-665`), which is the milestone tier's loop stopped short: no `LABELS`, no `add_test`, no manifest append, and installed so `beatnik_exe` resolves it. It therefore appears in neither `beatnik_milestone_manifest.txt` nor `beatnik_gate_manifest.txt`, which in `spack` mode are the **only** observables — this checkout has no build tree of its own and `ctest -L milestone` reports zero tests whatever the probe does. |
| Probe assertions | None | It measures. A probe that asserts is a test that will be tuned; this one exits 0 unless it cannot run at all. |
| Scratch root | `BEATNIK_TEST_SCRATCH` on `/p/lustre5` | Checkpoints go through MPI-IO; a node-local scratch fails every launch spanning more than one node. |
| Formatting | Never run clang-format, `clangformat.sh` or `cabana-format` | Write in the style of the surrounding code and leave formatting to the user. |

### Deliberate deviations

- **The demand counter is profiling-gated, so the level-4 member cannot assert
  on it.** An always-on counter would let `assertClaimA` check demand directly,
  but it would also build an unbounded key set on every interaction-list build
  in every production configuration. The member keeps asserting
  `p.m2l_fallback == 0` (`Beatnik_Test_Milestone0Fmm.cpp:1456`), which is the
  observable that matters: zero fallback *is* zero refused keys. Demand is a
  sizing instrument, not a gate.
- **`M2L_OP_COUNT_CAP` becomes a default rather than a floor.** Today
  `m2l_effective_op_cap()` (`Canopy_DownwardSweep.hpp:324-331`) is
  `min(M2L_OP_COUNT_CAP, byte_budget / bytes_per_key)` and the constant is a
  hard floor on purpose — the deviation note at
  `canopy/tasks/abstract-solver-backend.md:303-311` keeps it so that a pure byte
  budget cannot move which pairs overflow. T6 preserves that property by a
  different mechanism: the cap remains a count, is still floored by the byte
  budget, and its default is the same constant — so the overflow set moves only
  for a configuration that explicitly asks. The note's cited line for the
  constant (`src/Canopy_DownwardSweep.hpp:343`) is stale; it is at `:550`.
- **The probe re-derives claim A's setup instead of sharing it.**
  `Beatnik_Test_Milestone0Fmm.cpp` is 2201 lines and deliberately carries no
  step-count or claim selector — "a knob that can silently shorten a 2000-step
  run is how a truncated run reads as a shorter pass" (`:113-115`). Factoring
  its driving loop into a shared header would mix a refactor of a currently
  failing member into its own diagnosis. The probe duplicates roughly 200 lines
  of parameter setup and accepts that cost; T3 states which constants must agree
  and how that is checked.
- **The dev env's canopy spec gains `+profiling` rather than a third env being
  created.** Canopy is an INTERFACE library, so the variant costs a Beatnik
  recompile and no more, and Beatnik already runs its own `+profiling
  profiling_level=2`. Level 1 (the bare `+profiling` default,
  `canopy/CMakeLists.txt:257-277`) is enough: the demand counter and the
  existing `[Canopy Diagnostics]` line are gated on `CANOPY_ENABLE_PROFILING`
  alone, not on the level.

## Current state

**Canopy** (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`):

- `M2L_OP_COUNT_CAP = 32768` is a `static constexpr int` at
  `src/Canopy_DownwardSweep.hpp:550` with no runtime path to change it.
  `m2l_effective_op_cap()` (`:324-331`) floors it by
  `_m2l_op_table_byte_budget / KernelType::bytes_per_key`.
- The byte budget *is* configurable: `FmmConfig::m2l_op_table_byte_budget`
  (`src/Canopy_Solver.hpp:95`) is routed to `set_m2l_op_table_byte_budget()`
  (`Canopy_DownwardSweep.hpp:309-313`) from the Solver constructor (`:192`).
  At CartesianTaylor order 3 a column is $20\times20\times8=3200$ bytes
  (`Canopy_CartesianTaylorBasis.hpp:506-508`), so 2 GiB buys 671 088 columns and
  **the count cap binds by a factor of 20 on Beatnik's path**.
- The merge admits keys until `effective_op_cap` and assigns `-1` beyond it
  (`:1616-1645`), warning once per build. `n_unique_ops` (`:1663`) is the
  admitted count. There is **no counter for the refused keys** — this is the gap
  T1 closes.
- `m2l_n_unique_ops()` (`:934-937`), `m2l_realized_keys()` (`:926-929`),
  `m2l_op_cache_size()` (`:433`) and `m2l_op_keys_built_count()` (`:435`) are
  ungated public accessors. The `[Canopy Diagnostics] M2L operator table:` line
  at `:1665-1693` is gated on `CANOPY_ENABLE_PROFILING`.
- `_all_at_depth_local` and `_leaves_at_depth_local` (`:484-487`, filled
  `:1111-1113`) hold the per-depth occupied-cell counts but have **no public
  accessor**. T1 adds one, because the occupied-depth count is what explains a
  level-keyed basis's key count.
- No test drives the count cap. `tests/tstLaplaceSolve.hpp:736` takes
  `m2l_op_table_byte_budget` and applies it at `:848-849`, reporting the
  effective cap at `:1057` — the nearest existing fixture, and the one T1's and
  T6's validation tests extend.
- Canopy builds in **manual** mode — out-of-tree cmake + make under
  `spack env activate ${HOME}/spack_envs/tuolumne_trilinos`
  (`canopy/systems/tuolumne/claude.md` §1 and §3), a different environment from
  the one that builds Beatnik. **This clone carries no cmake build tree**: the
  `build-linux-rhel8-zen4-*` directories in it are spack's. A second clone of
  the same repo at `/g/g20/stewartj/research-bridges/canopy-dev/Canopy` — same
  remote, same `develop` commit — owns the `build-tuolumne` tree that
  `canopy/scripts/tuolumne/run_ctest_laplace_solve.flux:30-31` runs `ctest` in.
  T1's and T6's builds are configured **in this clone**, not that one, so T2 and
  T4 read their edits with no cross-clone push and pull.

**Beatnik** (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/beatnik`):

- `FarFieldDiagnostics` (`src/Beatnik_FarFieldInterface.hpp:214-348`) carries
  `local_m2l_unique_op_count` (`:273`), `local_m2l_op_cache_size` (`:277`),
  `local_m2l_op_keys_built` (`:283`), `local_m2l_bytes_per_key` (`:293`) and
  `local_m2l_op_cap` (`:301`), and beside them T2's four:
  `local_m2l_demanded_op_count` (`:315`), `local_m2l_demand_saturated`,
  `local_m2l_cells_at_max_depth` and `local_m2l_occupied_depths`. All nine are
  populated in the single `readDiagnostics` override at `:875-918`
  (declared `:772`, called `:1491`). `local_m2l_unique_op_count`'s
  comment describes it as read "against Canopy's per-rank 32768-key cap" — it
  is, and it saturates there.
- `FmmParams` (`src/Beatnik_Params.hpp:167`) carries `mac_theta = 0.3` (`:194`),
  `order = 3` (`:229`), `ncrit = 64` (`:256`), `max_depth = 10` (`:291`) and
  `m2l_op_table_byte_budget` (`:395`), the last routed to `FmmConfig` at
  `Beatnik_FarFieldInterface.hpp:1068`. **There is no count-cap member.**
- `m2l_op_table_byte_budget`'s doc comment (`Beatnik_Params.hpp:382-394`) states
  the current doctrine: "the constraint to act on is the count cap, and the
  response to realized overflow is a lower `max_depth` or `order`, not a
  smaller table." T7 makes the count cap itself actionable and must update that
  paragraph; T8's demand-reduction branch is the `max_depth` lever it names.
- The level-4 member runs `kNcrit = 8`
  (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:340`),
  `kProductionOrder = 3` (`:345`), `kVertices = 2562` (`:511`),
  `kP2PFractionBound = 0.75` (`:544`), `max_depth` 10 asserted at `:1385`.
  `ncrit = 8` is near its floor: the liveness inequality
  $N\gg\pi(\sqrt3/\theta)^2\cdot\texttt{ncrit}$ (`Beatnik_Params.hpp:238-251`)
  is about 840 at `ncrit = 8` against 2562 vertices, and 6720 at the default 64.
  **`ncrit = 8` is itself a demand driver** — it deepens the tree to make the
  far field live at all — which is why T8 cannot simply raise it.
- The milestone tier has four members and sixteen launches, registered at
  `tests/CMakeLists.txt:439-495`, run by
  `scripts/tuolumne/run_milestone.flux` at `-q pbatch -t 1440m`.
- **An assertion-free installed binary already has a home.**
  `tests/CMakeLists.txt:585-665` is the **"Measurement drivers — IN NO TIER"**
  loop, documented in place as the milestone loop stopped short of the point
  where it applies a label: it writes one generated translation unit per backend
  pinning `BEATNIK_TEST_EXEC_SPACE`, builds `<stem>_MPI_<BACKEND>`, installs to
  `share/Beatnik/tests` so `beatnik_exe` resolves it — and then applies no
  `LABELS`, calls no `add_test` and appends to neither manifest. Its source list
  is `BEATNIK_DRIVER_SOURCES` (`:617-630`) and it carries two members,
  `Beatnik_Test_Milestone0Run.cpp` and `Beatnik_Test_FmmScan.cpp`. There is no
  `tools/` directory in this repo and no other home for such a binary.
- `scripts/tuolumne/t6_l3_member.flux` is the precedent for a `pdebug`
  single-member script that invokes an installed test binary directly by reading
  its arguments out of `beatnik_milestone_manifest.txt`, which it locates by
  scanning `$PATH` for the file (`:120-136`) — a manifest is a data file and
  `which` cannot find one. T3's and T5's runners follow its structure.
- **T6 remains IN PROGRESS in `tasks/canopy/add-canopy.md`.** The failing tier
  run's numbers are recorded nowhere; T0 records them.

**Environment** (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml`, whose
committed snapshot `systems/tuolumne/spack.yaml` is byte-identical to it):
`canopy@develop` at `:16` carries **no** `+profiling`; `beatnik@develop` at
`:17` carries `+testing +canopy +examples +profiling profiling_level=2` — a
Beatnik variant, unrelated to `CANOPY_ENABLE_PROFILING`. This checkout is
**`spack` mode**: build with `spack install`, never `cmake`/`make`, and there is
no build tree and therefore no `ctest`.

## Progress log

`tasks/add-canopy-t6-progress-log.md`. Read it before implementing any task
below, before changing a signature this document names, and before reopening a
question this document treats as settled — it carries the measured numbers
behind every claim here, and the `**Affects:**` line of each entry names the
later tasks whose stated plan that entry changed.

## Task sequence

### T0 — Record T6's tier-run result; leave T6 IN PROGRESS — **DONE**

**Depends on:** none.
**Fill in:** `tasks/canopy/add-canopy-progress-log.md` — the `## T6` section
already present at `:1930` is **completed in place**, never appended to with a
second heading. That section opens "INCOMPLETE BY DESIGN" and its closing
subsection (`:2205-2215`) reserves three things for the session that reads the
tier job: the tier-run results, the `**Affects:**` line and a `**Met.**`
paragraph. This task writes the first two and replaces that closing subsection
with what is actually outstanding. **No `Met.` paragraph is written here and T6
stays IN PROGRESS** — the tier came back red, so nothing is met; T9b owns that
paragraph after a green re-run. Also `tasks/canopy/add-canopy.md` (T6's pointer
only).
**Reference:** the numbers come from the tier job's own log,
`beatnik_milestone.f3azynKNQFCb.log` in the repo root (job `f3azynKNQFCb`,
`[milestone] FAIL (label=milestone)` at `:20612`), and from `T6-handoff.log` in
the repo root, which names the job and every failure branch. **Confirm every
figure below against that log rather than transcribing it** — a number carried
between documents unchecked is how a measurement becomes folklore. The estimate
table this supersedes is at
`tasks/canopy/add-canopy-progress-log.md:2146-2154`; the level-3 cost format to
match is at `:2085-2100`.
**Do:**
1. Record into the existing `## T6` section: 12 of 16 launches green; all four
   `Milestone0FmmL4` launches red on both backends at both rank counts; the
   frozen pair and both level-3 FMM members fully green.
2. Record the failure precisely — the three failing assertions
   (`Beatnik_Test_Milestone0Fmm.cpp:1456` twice, and the τ_A bound), the
   `[Canopy]` cap message, fallback exactly 0 at all 81 states in all four
   level-3 launches, level-4 clean through step 225, first trip at step 250,
   about 13 000 fallback pairs by step 1375.
3. Record the four level-4 claim-A worst errors at 17 digits with their steps
   (SERIAL np1 `0.0012513022396595567`, SERIAL np4
   `0.0012474182160902654`, HIP np1 `0.0012498660607555461`, HIP np4
   `0.0012435185167586275`; all peak at step 1375), the realized P2P fractions
   (0.2057–0.3976), the count of states above τ_A per launch (4, 4, 5, 6), and
   the offending steps (1000, 1350, 1375, 1475; plus 1325 at HIP np1 and 975 at
   HIP np4).
4. Record that level 3 peaked at `3.0958e-4` at step 250, a 3.2x margin.
5. Record the measured costs: 31 265.7 s total (8.687 h) against the 24 h cap;
   FMM members 29 237 s; frozen pair about 2 028 s; the per-launch
   claim-A/claim-B split for both FMM members at both backends and rank counts.
6. Record what was clean at level 4: all five horizon rungs on envelope, all
   three negative cases fired in every launch, unpairable steps exactly
   1350–1900, `FrozenL4` green on both backends.
7. State that the τ_A exceedance occurs only at states with non-zero fallback
   and is therefore not a measurement of the expansion, and that
   `run_milestone.flux` stays at `-t 1440m`.
8. Point T6 in `tasks/canopy/add-canopy.md` at this document. Its status is
   already IN PROGRESS (`:3` and `:1785`) and stays so, and its "Where it
   stands" blockquote (`:1790-1793`) already cites `## T6` in the progress log
   and `T6-handoff.log`; what it does not yet cite is this document, which is
   where the cap's diagnosis now lives. Change no tolerance and no `-t`.

**Exit criterion:** `grep -c '^## T6$' tasks/canopy/add-canopy-progress-log.md`
returns 1 — it returned 1 before this task as well, so what this checks is that
the section was completed **in place and not duplicated**. That section carries
all four failing launches, the step-250 onset, the step-1375 peak, the 8.687 h
total and an `**Affects:**` line, and no longer claims the tier results are
missing; `git diff --stat` shows no file changed outside the two task documents;
and `grep -n 'pbatch -t 1440m' scripts/tuolumne/run_milestone.flux` still
matches.

**Met.** The `## T6` section of `tasks/canopy/add-canopy-progress-log.md` was
completed **in place** — `grep -c '^## T6$'` returns 1 — and its
"INCOMPLETE BY DESIGN" opening and "What is deliberately missing" closing
subsection are both gone, replaced by the tier run's numbers: job
`f3azynKNQFCb`, **12 of 16 launches green**, all four `Milestone0FmmL4`
launches red, **31 265.7 s = 8.685 h** against the 24 h cap, fallback clean
through step 225 and non-zero at all 71 states from step 250, the four
step-1375 worst errors at 17 digits (`0.0012513022396595567`,
`0.0012474182160902654`, `0.0012498660607555461`, `0.0012435185167586275`),
the per-launch claim-A/claim-B cost table, and an `**Affects:**` line naming
T7, T8, R9 and this document. `git diff --stat` shows only files under
`tasks/`; no code, script, tolerance or walltime changed, and
`scripts/tuolumne/run_milestone.flux` still carries `# flux: -t 1440m` and
`# flux: -q pbatch` at `:5` and `:7`. **Three figures in this task's Do steps
disagreed with the log and the log won** — the total is 8.685 h not 8.687 h,
the FMM members sum to 28 831.6 s not 29 237 s, and the frozen pair to
2 192.4 s not 2 028 s — and the failing-assertion count is **four** sites,
not three, because the τ_A bound fails in both of the forms the member
asserts it in. All are recorded in `## T0` of the progress log. The exit
criterion's own `grep -n 'pbatch -t 1440m'` pattern matches nothing in the
pristine tree either, since the two flux directives are on separate lines;
that is noted there too. **No `Met.` paragraph was written for T6 and T6
stays IN PROGRESS**, per this task's own instruction — T9b owns that after a
green re-run.

---

### T1 — Canopy: count and expose the demanded key set, profiling-gated — **DONE**

**Depends on:** none.
**Fill in:** `canopy/src/Canopy_DownwardSweep.hpp` (the merge loop at
`:1602-1645`, the accessor block near `:926-937`, the member block near
`:695-760`, the constant block near `:548-557`, the profiling printf at
`:1665-1693`); `canopy/tests/tstLaplaceSolve.hpp` (a new case using the existing
budget-parameterized driver at `:736`).
**Reference:** the accessor triple to mirror is `m2l_n_unique_ops()`
(`:934-937`), `m2l_realized_keys()` (`:926-929`), `m2l_op_cache_size()`
(`:433`). `M2LKey` and `M2LKeyHash` are at `:583-613`. The guarantee that
`local_ops[t]` holds already-canonical keys is at `:1367-1370` with the
canonicalization call at `:1495`. The gating precedent is the printf at
`:1665-1693`. The define comes from the cmake option at
`canopy/CMakeLists.txt:252-277` for this task's own builds, and propagates to
Beatnik through `canopy/src/CMakeLists.txt:27-33`.
**Do:**
1. Add `static constexpr int M2L_DEMAND_COUNT_CAP = 1048576;` beside
   `M2L_OP_COUNT_CAP` (`:550`), commented with the reasoning in the conventions
   table above — that it exceeds the 671 088 columns 2 GiB buys at 3200 B per
   key, so saturation is itself an answer.
2. Add two members: `int _m2l_demanded_op_count = -1;` and
   `bool _m2l_demand_saturated = false;`. Comment that `-1` means "not compiled
   with profiling" and that `0` is a legal count.
3. In the merge loop, under `#ifdef CANOPY_ENABLE_PROFILING`, hold a local
   `std::unordered_set<M2LKey, M2LKeyHash>` and insert `key` **before** the cap
   test, so the set sees every distinct key the merge sees. Stop inserting once
   the set reaches `M2L_DEMAND_COUNT_CAP` and set `_m2l_demand_saturated`.
   Reserve the set at the same figure `local_ops` reserves per thread.
4. Write the set's size to `_m2l_demanded_op_count` after the loop. Outside the
   `#ifdef`, leave it `-1`.
5. Add `int m2l_n_demanded_ops() const` and
   `bool m2l_demand_saturated() const`. Document on the declarations: rank-local,
   unreduced, `-1` when unavailable, counted over canonical keys, and that the
   count is what the cap *would* have to be to admit every key.
6. Add `std::vector<int> m2l_cells_at_depth() const`, returning **by value** a
   vector whose entry `d` is `_all_at_depth_local[d].size()` — one entry per
   depth — so a consumer can see the occupied-depth count that explains a
   level-keyed basis's key total. `_all_at_depth_local` (`:487`) is a
   `std::vector<std::vector<int>>` of per-depth cell-index lists: already host
   state with no device mirror to take a side of (`_d_all_at_depth` at `:491` is
   the separate device copy), and the per-depth *count* vector exists nowhere as
   state, so there is nothing to return by const reference. Ungated — it reads
   state the sweep already maintains at `:1113` and `:1125`.
7. Extend the `[Canopy Diagnostics] M2L operator table:` printf (`:1678-1692`)
   with `n_demanded_ops=` and `demand_saturated=`.
8. **The demanded set must feed nothing.** It must not touch `ops`,
   `key_to_op`, `local_to_global`, `pair_op_idx`, `_m2l_realized_keys`, the
   operator cache or the fallback tables. Assert this by test, not by
   inspection: see the exit criterion.
9. Add a `tstLaplaceSolve.hpp` case driving `with_laplace_solve` at a budget
   worth exactly one column. Assert `m2l_n_unique_ops() == 1`,
   `m2l_n_demanded_ops() > 1`, `m2l_demand_saturated() == false`, and
   `total_fallback_pair_count() > 0`. Add a second case at the default budget
   asserting `m2l_n_demanded_ops() == m2l_n_unique_ops()` and
   `total_fallback_pair_count() == 0`.

**Additional information needed:** none. The one figure this task cannot supply
is what `m2l_n_demanded_ops()` reads on Beatnik's level-4 geometry; T5 supplies
it.

**Exit criterion:** two cmake trees in
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`, one configured
`-DCanopy_ENABLE_PROFILING=ON` and one `=OFF`, both under
`spack env activate ${HOME}/spack_envs/tuolumne_trilinos`. The `ON` tree's test
suite passes including both new cases — the constrained-budget case reports
demand strictly greater than the realized 1, and the default-budget case reports
demand equal to realized with zero fallback. In the failure direction: the `OFF`
tree builds the same suite and passes with `m2l_n_demanded_ops()` returning
`-1`, and the constrained-budget case's *realized* count, fallback pair count
and `m2l_realized_keys()` contents are identical between the two trees — which
is what proves the counter changed no answer.

**Met.** Two cmake trees in the Canopy clone at
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`, both configured with
`run_cmake_tuolumne.sh`'s arguments under
`spack env activate ${HOME}/spack_envs/tuolumne_trilinos`:
**`build-t1-prof-on`** (`-DCanopy_ENABLE_PROFILING=ON
-DCanopy_PROFILING_LEVEL=2`) and **`build-t1-prof-off`**
(`-DCanopy_ENABLE_PROFILING=OFF -DCanopy_PROFILING_LEVEL=2`, which cmake
resolved to `level=0` — the kill switch exercised, not bypassed). Both built
`Canopy_Test_LaplaceSolve_MPI_SERIAL` clean and both ran the suite at ranks
1-6 in one `pdebug` job, `f3bPfi66qz4X`
(`scripts/tuolumne/run_t1_demand.flux`): **`100% tests passed, 0 tests failed
out of 6` in each tree**, combined rc 0.

The two new cases, over all **21 `(nprocs, rank)` pairs** at ranks 1-6:

- **`m2lKeyDemandConstrained`** at a one-column budget (21 952 B, one
  `bytes_per_key`): `eff_cap=1`, **`realized=1`** everywhere, and
  **`demanded` from 111 to 718** — strictly greater than the realized 1 at
  every pair, `saturated=0` everywhere, `fallback` from 186 to 1 702, always
  positive.
- **`m2lKeyDemandDefault`** at the default budget: **`demanded == realized` at
  every one of the 21 pairs**, with realized from **111 to 686** — exactly
  the per-rank range `LS_BUDGET_KEYS`'s comment already records for the frozen
  configuration — and `fallback=0` everywhere.
- In the `OFF` tree both cases report **`demanded=-1`** at every pair, never
  0, and emit **zero** `[Canopy Diagnostics]` lines against the `ON` tree's
  1 536.

**On R1: the two builds' realized output is NOT byte-identical, and the
`~profiling` build is not byte-identical to itself either** — which is what
makes the counter exonerated rather than suspect. The two new cases' realized
figures (`realized`, `fallback`, `m2l_realized_keys()` contents,
`cells_at_depth`) **match line for line between the trees at all 21 pairs**,
with `demanded=` the only differing field, and the whole `[laplace-solve]`
output is identical at **np 1, 2, 4 and 5**. At **np 3 and np 6** other tests'
solves differ — `crossRankAgreement`'s `n_unique_ops` reads 285/189 in one
tree and 273/204 in the other. A second job, `f3bPhuxP1JNT`
(`scripts/tuolumne/run_t1_repro.flux`), ran three identical passes per tree
and found **the `OFF` tree disagreeing with itself in the same fields at the
same two rank counts**, so this is pre-existing run-to-run nondeterminism in
the tree/partition path at np ≥ 3, not a write from the instrumentation. Full
detail in `## T1` of the progress log.

**The budget that buys exactly one column is `Kernel::bytes_per_key * 1`** —
21 952 B for `LaplaceKernel<double, 6, 1>` — recorded as
`LS_DEMAND_BUDGET_KEYS = 1` and derived from the trait, never a literal. T6
extends these same cases and needs it.

---

### T2 — Beatnik: mirror demand onto `FarFieldDiagnostics` — **DONE**

**Depends on:** T1 **DONE**.
**Fill in:** `src/Beatnik_FarFieldInterface.hpp` — the struct at `:214-301` and
the single `readDiagnostics` override at `:829-845`.
**Reference:** `local_m2l_unique_op_count` (`:273`) and `local_m2l_op_cap`
(`:300`) are the fields to sit beside; `local_m2l_op_keys_built`'s comment
(`:279-283`) states the cache-thrash signature these fields are read against.
**Do:**
1. Add `int local_m2l_demanded_op_count = -1;`,
   `bool local_m2l_demand_saturated = false;`,
   `int local_m2l_cells_at_max_depth = 0;` and
   `int local_m2l_occupied_depths = 0;`. Document that `-1` means the Canopy
   build carries no profiling, distinct from a genuine zero. The last two are
   derived from `m2l_cells_at_depth()` by a single scan, and the rule is not
   the obvious one: the vector runs to `max_depth + 1` entries and carries
   trailing zeros, so `local_m2l_occupied_depths` is the count of its
   **non-zero** entries rather than its size, and
   `local_m2l_cells_at_max_depth` is the value of its **last non-zero** entry —
   the cell count at the deepest *occupied* depth — so that a tree shallower
   than `max_depth` 10 reports a real count rather than an uninformative 0.
2. Populate all four in `readDiagnostics` beside the existing five.
3. **Callers of the changed interface**, enumerated: the pure virtual
   declaration at `:726`, its one override at `:829`, and the one call site at
   `:1418` (`_impl->readDiagnostics( _diagnostics )`). No signature changes —
   the struct gains members — so no caller needs editing. No other file in
   `src/`, `tests/` or `examples/` mentions `readDiagnostics`.

**Exit criterion:** `spack install` of the env succeeds, and an existing FMM
test that reads `farField().diagnostics()` — `Beatnik_Test_Milestone0Fmm`'s
level-3 member is the cheapest at 314 s on HIP np1 — still passes unchanged. In
the failure direction: against a `~profiling` canopy the new field reads `-1`
and not `0` (**R7**) — within T2 that is only statically true, since the
accessors return their `-1` member default and `spack.yaml:16` still carries no
`+profiling`, and nothing in Beatnik prints the field until T3's probe exists;
the runtime confirmation is T3's own failure-direction criterion, which runs
before T4 turns `+profiling` on.

**Met.** `spack install` of the dev env succeeded in **11 m 13 s** (beatnik
`4bhhtbd`), preceded by a **37 s** canopy rebuild that is the first compile of
T1's working-tree edits — and therefore the first instantiation of
`m2l_cells_at_depth()` on the **CartesianTaylor** arm, which produced no
template error and needed no Beatnik-side fix.

`Beatnik_Test_Milestone0Fmm`'s level-3 member then ran unchanged at HIP np1 and
np4 in job **`f3bQ3AzdBncF`** (`scripts/tuolumne/t6_l3_member.flux HIP`, commit
`f94c9db` + 1 modified file): **`[t6l3] SUMMARY: PASS (2/2 launches)`**, with
**`[PASS] Beatnik_Test_Milestone0Fmm (3097/3097 checks)`** at np1 and
`3097/3097` on rank 0 plus `2919/2919` on ranks 1-3 at np4 — the same check
counts T0 records for the passing level-3 launches. Wall times **316 s at np1
and 301 s at np4**, 617 s together, against T0's **314 s** np1 baseline: a
**+0.6 %** difference at np1, i.e. run-to-run noise, which is the expected
result for four fields nothing yet reads. The backend override is announced in
the log as designed; the SERIAL half was not run and is not claimed.

**The `-1` sentinel is established statically, by three facts together**
(**R7**). The env concretizes canopy as **`~profiling`** (`spack find
--variants canopy`); the installed `Canopy::Canopy` INTERFACE target exports
**no `INTERFACE_COMPILE_DEFINITIONS` property at all**
(`share/cmake/Canopy/Canopy_Targets.cmake:61-66`) and `CANOPY_ENABLE_PROFILING`
appears nowhere in Beatnik's own CMake, so the macro is undefined in every
Beatnik translation unit; and in that case Canopy's `_m2l_demanded_op_count`
keeps its **`-1`** member default, the only write to it being inside
`#ifdef CANOPY_ENABLE_PROFILING` (`Canopy_DownwardSweep.hpp:1769-1771`). So
`local_m2l_demanded_op_count` is `-1` and not `0` in this build. Nothing in
Beatnik prints it yet, by design; **T3's probe is what observes it at
runtime.**

---

### T3 — Beatnik: the demand probe binary — **DONE**

**Depends on:** T2 **DONE**.
**Fill in:** a new `tests/regression_tests/Beatnik_Probe_FmmKeyDemand.cpp`;
`tests/CMakeLists.txt` — one entry appended to `BEATNIK_DRIVER_SOURCES`
(`:617-630`) and nothing else, since that loop already supplies build, install,
no label and no manifest line; a new `scripts/tuolumne/t6b_key_demand.flux`
carrying the level-3 validation launch this task's exit criterion needs, which
T5 **extends** with the level-4 matrix rather than creating.
**Reference:** the state-driving sequence to reproduce is `evaluateClaimA`
(`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1397-1437`) — one whole
tuple halo exchange, geometry at current positions, then the sheet vector, in
the order `Beatnik_ZModelSolver.hpp` steps 0-2 establish them. The FMM knobs are
that member's `makeFmmParams` (`:1027-1032`), which sets only `ncrit = kNcrit`
and is level-independent, so it carries over verbatim.

**The level-parameterization precedent is `Beatnik_Test_Milestone0Run.cpp`, not
the member.** The member's `makeParams` (`:945`) reads the compile-time
`kSubdivisions` (`:250`, from `BEATNIK_M0_FMM_LEVEL` at `:230-235`) and cannot
take a level at runtime, and its `kVertices` (`:429`, `:511`) and
`kP2PFractionBound` (`:468`, `:544`) live in per-level `#if` arms.
`Beatnik_Test_Milestone0Run.cpp` is the driver that already solves this:
`makeParams( int subdivisions, int steps, … )` (`:175`) takes the level as a
parameter, `verticesForLevel( int )` (`:158`) computes `10*4^L + 2` rather than
tabulating it — the arithmetic step 6's round-trip check needs — and `argv[1]`
is the level (`:283`). The five values step 7 checks are all level-independent,
so the probe needs no per-level literal table.

The registration home is the **"Measurement drivers — IN NO TIER"** loop
(`tests/CMakeLists.txt:585-665`), whose two existing members,
`Beatnik_Test_FmmScan.cpp` and `Beatnik_Test_Milestone0Run.cpp`, are the closest
structural precedents; `Beatnik_Test_FmmScan.cpp`'s file header states the
in-no-tier contract and the P2P-fraction discipline the probe inherits. Because
that loop generates one translation unit per backend, the probe's targets are
`Beatnik_Probe_FmmKeyDemand_MPI_SERIAL` and `..._MPI_HIP`.

The launch and binding pattern is `scripts/tuolumne/t6_l3_member.flux`, whose
binding block is `:200-208`; the direct/FMM comparison harness pattern is
`scripts/tuolumne/t3_fmm_velocity.flux`.
**Do:**
1. Drive the **direct** trajectory at a level selected on the command line (3 or
   4), at the milestone-0 configuration, `--checkpoint-every-steps 25`, 2000
   steps. No step-count override: the probe takes the level and nothing that can
   shorten the run.
2. At each checkpointed step, establish the preconditions exactly as
   `evaluateClaimA` does and call `computeInterfaceVelocity` on the FMM solver
   once. Do **not** run the direct solver or any comparator — the probe measures
   demand, not error.
3. Print one row per state: step, `local_m2l_demanded_op_count`,
   `local_m2l_unique_op_count`, `local_m2l_op_cap`,
   `local_m2l_demand_saturated`, `global_m2l_fallback_pair_count`,
   `global_m2l_pair_count`, `p2p_pair_fraction`, `local_m2l_op_cache_size`,
   `local_m2l_op_keys_built`, `local_m2l_occupied_depths`, and the wall time of
   that single FMM evaluation. **Per rank and unreduced** for the rank-local
   fields — the key set is per rank, and a mean would hide the rank that
   actually overflows.
4. Print a header line carrying the level, `ncrit`, `order`, `mac_theta`,
   `max_depth`, `bytes_per_key`, the byte budget, and whether the Canopy build
   reports demand at all (i.e. whether the field is `-1`). A run whose header
   says `-1` is not a measurement and must say so in one loud line.
5. Print a trailer: the step at which demand first exceeds the cap, the peak
   demand and its step, the peak `local_m2l_op_keys_built` increment per
   evaluation, and the total wall time.
6. Assert nothing about any measured value. Exit non-zero only if the run cannot
   proceed — a throw, a non-finite velocity, or a particle-count round-trip
   mismatch against the level's vertex count.
7. The parameter set must not be allowed to drift from the member's. Echo
   `ncrit`, `order`, `basis`, `mac_theta` and `max_depth` out of
   `fmm.farField().params()` and compare them against compiled-in literals that
   match `kNcrit` (`:340`), `kProductionOrder` (`:345`) and the values asserted
   at `:1376-1387`, failing loudly on a mismatch. All five are level-independent
   — `ncrit` 8, `order` 3, `CartesianTaylor`, `mac_theta` 0.3, `max_depth` 10 —
   so the check is one table, not one per level. A probe measuring a different
   configuration than the member is worse than no probe.

**Exit criterion:** `spack install` succeeds and **both**
`beatnik_exe Beatnik_Probe_FmmKeyDemand_MPI_SERIAL` and
`beatnik_exe Beatnik_Probe_FmmKeyDemand_MPI_HIP` resolve; the per-backend suffix
is not optional, since the driver loop names its targets `<stem>_MPI_<BACKEND>`
and `beatnik_exe` in installed mode is `command -v <basename>` exactly
(`scripts/lib/beatnik_env.sh:269-278`).

**The tier check is against the two manifests, not against `ctest`.** Locate
each by scanning `$PATH` for the file the way
`scripts/tuolumne/t6_l3_member.flux:120-136` does, then confirm that
`grep -c Beatnik_Probe_FmmKeyDemand` returns 0 in both
`beatnik_milestone_manifest.txt` and `beatnik_gate_manifest.txt`, and that the
milestone manifest still carries its **12** non-comment lines (four members x
three backends). Neither manifest is reachable through `which` — they are data
files, not executables — and a `grep -c` against a path that does not exist
prints nothing and exits 2, which reads as a pass.

Then a `pdebug` submission of `scripts/tuolumne/t6b_key_demand.flux` runs the
probe at level 3 on HIP at np1 and np4 and exits 0, the np4 launch showing the
rank-local fields printed once per rank rather than reduced.

In the failure direction: that run is against the `~profiling` canopy this env
still concretizes, so it must print the loud "demand unavailable" line with
`local_m2l_demanded_op_count` at `-1` and `local_m2l_demand_saturated` at
`false` — never demand as 0 (**R7**) — while in the same rows
`local_m2l_occupied_depths` and `local_m2l_cells_at_max_depth` read real
non-zero counts, because `m2l_cells_at_depth()` is ungated and live in this
build. A probe reporting `-1` for those two is reporting its own bug.

**Met.** `spack install` in the dev env exited 0 in **12 m 20 s**
(`beatnik@develop` hash `4bhhtbd`; `canopy@develop` was cached and did not
rebuild), and both per-backend targets resolve through `beatnik_exe` to
`…/.spack-env/view/share/Beatnik/tests/Beatnik_Probe_FmmKeyDemand_MPI_SERIAL`
and `…_MPI_HIP`. **R6, against both installed manifests located by scanning
`$PATH`** — both files were *found*, so the `grep -c` reads are real and not the
exit-2 silence the task warns about: `grep -c Beatnik_Probe_FmmKeyDemand`
returns **0** in `beatnik_milestone_manifest.txt` and **0** in
`beatnik_gate_manifest.txt`, and the milestone manifest still carries its **12**
non-comment lines (the gate manifest's 15 are likewise unmoved). Job
**`f3bQSGSss6RD`** (`-q pdebug`, `-t 30m`) reports `COMPLETED` with returncode
**0** after **93.2 s**, both launches passing: HIP np1 in **24 s** and HIP np4 in
**43 s**, each rank reporting `[PASS] Beatnik_Probe_FmmKeyDemand (174/174
checks)` — five such lines, one from np1 and four from np4, which is the
per-rank printing itself. The probe's own clocks: trajectory wall **15.216 s**
(np1) and **35.580 s** (np4), of which the 81 FMM evaluations are **4.0435 s**
and about **3.65 s** per rank. The `_MPI_SERIAL` target was resolved by the
runner and deliberately **not launched**; nothing is claimed for it.

**The failure direction held exactly.** Across all **405** rows (81 states x 1
rank plus 81 x 4) the demand column is **`-1` in every one** and
`demand_saturated` is **`0` in every one** — zero rows read demand as `0`
(**R7**) — and the header printed the loud `*** DEMAND UNAVAILABLE ***` line in
both launches with `demand_available=0`. In those same 405 rows
`local_m2l_occupied_depths` ranges **4 to 6** and `local_m2l_cells_at_max_depth`
is **non-zero in every row**, with no negative value in either column: the
ungated half is live, so the probe is not reporting its own bug. The per-rank
columns genuinely differ at np4 — `unique_ops` 852 / 809 / 902 / 980 at step
1000 — so the rank-local fields are unreduced rather than four copies of one
number. **R5**: the five knobs echoed out of `fmm.farField().params()` matched
the member's compiled literals at both rank counts (`ncrit` 8, `order` 3,
`cartesian-taylor`, `mac_theta` 0.3, `max_depth` 10, plus
`near_softening_factor` 0).

**What the run does not establish, and must not be read as establishing:
anything about overflow.** Level 3's peak `unique_ops` is **6 404** against the
**32 768** cap, `global_m2l_fallback` is **0** in all 405 rows and the
`[Canopy] M2L op count exceeded cap` warning appears **zero** times — because
the level-3 member itself declares `kFarFieldIsLive = false` and
`kP2PFractionBound = 1.0` (`Beatnik_Test_Milestone0Fmm.cpp:468`), and the probe
measured `p2p_pair_fraction` between **0.708** and **0.945** there. T5's
level-4 matrix is the only place the demand question is answerable. See
`tasks/add-canopy-t6-progress-log.md` `## T3`, whose `**Affects:**` line carries
this and the **R4** signature the run did turn up.

---

### T4 — Turn on `CANOPY_ENABLE_PROFILING` in the dev env — **DONE**

**Depends on:** T1 **DONE** (there is nothing to read before the counter
exists).
**Fill in:** two files carrying the same spec, edited together —
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml:16`, the live env, and
`systems/tuolumne/spack.yaml:16`, its committed snapshot in this repo. The
`canopy@develop` spec gains `+profiling` in each.
`systems/tuolumne/spack-production.yaml:8` is **not** touched — the production
canopy spec keeps no `+profiling`. Also `systems/tuolumne/claude.md:51`.
**Reference:** the variant is declared at
`.spack/package_repos/COMPASS/spack_pkgs/spack_repo/compass/packages/canopy/package.py:39`
with `profiling_level` at `:41`; the CMake resolution is
`canopy/CMakeLists.txt:252-277` and the INTERFACE propagation
`canopy/src/CMakeLists.txt:27-33`. Beatnik's own `+profiling profiling_level=2`
at `spack.yaml:17` is a different variant and is left alone. The rule that binds
the live env and its snapshot together is `systems/tuolumne/claude.md:43-44`,
restated in §2 at `:64-66`, with the snapshot-to-env table at `:46-49`.
**Do:**
1. Add `+profiling` to the `canopy@develop` spec in **both** files — bare, so
   the level resolves to 1. Do not set `profiling_level`: the demand counter and
   the `[Canopy Diagnostics]` line are gated on `CANOPY_ENABLE_PROFILING` alone,
   and level 2 adds detailed sub-phase timers whose overhead the probe does not
   need.
2. Keep the two byte-identical, which they are now: `diff` between them is the
   check. The snapshot is what a later session reads when the live env is not to
   hand, and a snapshot that has drifted describes a build nobody ran.
3. **The intended state of both source clones is the working tree as it
   stands.** The canopy clone is at `develop` commit `d3145e0` plus T1's
   uncommitted edits to `src/Canopy_DownwardSweep.hpp` and
   `tests/tstLaplaceSolve.hpp` — the demand counter itself, and the whole reason
   the variant is being turned on — and the beatnik clone is at its current
   `HEAD`. Do not pull either clone and do not commit the canopy one:
   `canopy@=develop` is a `spack develop` spec (`spack.yaml:36-37`), so
   `spack install` compiles those working-tree edits in place, and a pull would
   move `develop` past the commit they sit on.
4. `spack install`. This targets the **development** env; do not touch the
   production env, and never `spack install` against the production env while a
   production job is live — a running job whose executable pages change takes a
   SIGBUS (rc=135).
5. Update `systems/tuolumne/claude.md:51`. It states a single difference between
   the two committed snapshots; after this task there are two, in different
   packages — beatnik's `profiling_level` (dev 2, prod 1) and canopy's
   `+profiling`, which is dev-only.

**Exit criterion:** the probe from T3, run at level 3 on one node, prints a
header line reporting demand as a non-negative integer rather than `-1`, and its
log carries a `[Canopy Diagnostics] M2L operator table:` line containing
`n_demanded_ops=`. In the failure direction: `spack spec` for the env shows
`canopy ... +profiling` and `beatnik` rebuilt against it, so a stale Beatnik
binary compiled without the define cannot be the thing that ran.

**Met.** `+profiling` was added to the `canopy@develop` spec in both
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml:16` and
`systems/tuolumne/spack.yaml:16`, which remain byte-identical (`diff` is
empty); `systems/tuolumne/spack-production.yaml:8` was not touched. Bare
`+profiling` resolved the level to 1 as intended: the installed
`Canopy::Canopy` target now carries
`INTERFACE_COMPILE_DEFINITIONS "CANOPY_ENABLE_PROFILING;CANOPY_PROFILING_LEVEL=1"`,
where T2 recorded it exporting no such property at all. **Beatnik genuinely
rebuilt** — `spack concretize -f` moved canopy's hash `2cqynij` to `w4woraj`
and beatnik's `4bhhtbd` to `nnbspfy` while changing no package version (62
concrete specs before and after, no additions or removals), and `spack install`
exited 0 with canopy at **32 s** and `beatnik@develop` at **12 m 52 s**. That
is a full rebuild, not the sub-second no-op the header-only caveat warns about,
so no `touch` was needed and none was done.

The probe then ran unchanged at level 3 on HIP at np1 and np4, job
**`f3bQk9QtwPnw`**, `[t6b] SUMMARY: PASS (2/2 launches)` in 61 s, with the
runner's provenance line reading `canopy = canopy@develop+profiling`. Both
headers print **`demand_available=1`**, the `*** DEMAND UNAVAILABLE ***` line
appears **zero** times where T3 had it in both headers, and
`[Canopy Diagnostics] M2L operator table:` carries `n_demanded_ops=` on all
**405** rank-evaluations. Over those 405 rows: **demand is `-1` in none**
(T3: 405 of 405), `demand_saturated=0` everywhere, `global_m2l_fallback=0`
everywhere, `occupied_depths` 4 to 6 and `cells_at_max_depth` non-zero in every
row. Peak demand is **6 198 at np1 step 1600**, far under both the 32 768
`op_cap` and the 1 048 576 `M2L_DEMAND_COUNT_CAP`, so the counter is measuring
the tree and not overflowing.

**R1 is discharged by measurement rather than assumed.** The realized columns
*did* move against T3 — `unique_ops` differs in 367 of 405 rows — so a second
run of the **same** `+profiling` binary was taken (job **`f3bQmUaxP3eP`**) to
tell an instrumentation effect from run-to-run noise. It disagrees with the
first `+profiling` run in **369 of 405** rows, the same fields and the same
magnitude as the T3-to-T4 comparison, and peak np1 demand reads 5 938 against
6 198. The movement is therefore the pre-existing trajectory and tree
nondeterminism T1 documented, not the counter; `global_m2l_fallback` is 0 in
all 405 rows of all three runs. See `tasks/add-canopy-t6-progress-log.md`
`## T4`, whose `**Affects:**` line carries what this costs T5.

---

### T5 — Measure the level-4 demand series — **NOT STARTED**

**Depends on:** T3 **DONE**, T4 **DONE**.
**Fill in:** `scripts/tuolumne/t6b_key_demand.flux`, which T3 created for its
level-3 validation launch and this task extends with the level-4 matrix; results
into `tasks/add-canopy-t6-progress-log.md`.
**Reference:** copy the runner structure, repo-root discovery, provenance block
and rank-to-node binding from `scripts/tuolumne/t6_l3_member.flux` — the binding
must be copied exactly, because a wrong binding does not fail, it
oversubscribes one device and returns a plausible number.
**Do:**
1. Write the script with `-q pdebug -t 60m`, one node, exclusive. Never launch
   interactively from a login node: submit with `flux batch` and read the
   `.log`.
2. **Run the level-4 HIP np1 and np4 matrix twice, as two separate `flux batch`
   submissions** — not two passes inside one job. A single run is not the
   number: two submissions of one `+profiling` binary at level 3 disagree in
   369 of 405 rows of `unique_ops` and gave np1 peak demand 6 198 and 5 938,
   about a 4 % spread, so the peak is a draw from a distribution. Two
   independent allocations separate a run-to-run effect from an
   allocation-fixed one. Each submission's level-4 HIP pair is about five
   minutes.
3. Run level 3 at HIP np1 in each submission as the control. Its job is
   **reproducibility against a measured band**, not whether demand is under the
   cap — that is already measured: np1 peak demand **6 198 at step 1600** and
   **5 938 at step 1575**, zero `global_m2l_fallback` in all 405 rows of both
   runs, `demand_saturated` never set, np4 per-rank peaks
   2 332 / 1 864 / 1 932 / 1 926 and 2 135 / 2 073 / 2 039 / 1 876. A control
   outside that band by much more than the observed 4 % means the measurement
   apparatus moved, not the tree. Note also that `demand == unique_ops` at
   level 3 is an artefact of the cap not binding there, so a level-4
   `demand > realized` is the expected reading and not a level-3 regression.
4. Add SERIAL np1 and np4 at level 4 to a submission only if the HIP result is
   ambiguous. Budget from the measured claim-A costs: SERIAL np1 1432 s, SERIAL
   np4 523 s. All five launches together are about 38 minutes and fit `-t 60m`;
   if they do not, split the submission rather than raising `-t`.
5. Set `BEATNIK_TEST_SCRATCH` to a per-launch directory under `/p/lustre5`,
   removed and recreated immediately before each launch so a stale checkpoint
   cannot be read back as this run's output.
6. Record into the log, for **both** submissions: the full 81-state demand
   series for level 4 at both rank counts; the step at which demand first
   exceeds 32768; whether `demand_saturated` was ever set; the occupied-depth
   count against demand at the peak; and the per-evaluation
   `local_m2l_op_keys_built` increment at the peak.
7. Report the peak as the **worst observed** value per (rank count, rank)
   across the two submissions, with its step, and state the run-to-run spread
   between them. Never a mean, and never a single draw: T8 sizes a cap from
   this number.
8. Record the implied table size at the worst-observed peak, as
   $\texttt{demand}\times3200$ bytes, so T8 has the memory figure beside the
   count.
9. Place the demand peak's step against the three steps the failure already
   has: the cap's **onset at step 250**, claim A's **error peak at step 1375**,
   and the **fallback peak at step 1650**. Say which step drives each rather
   than assuming they coincide.

**Additional information needed:** none — this task produces the number every
later task is waiting on.

**Exit criterion:** the log carries the level-4 demand series from **both**
submissions at both HIP rank counts, a peak named as the worst observed across
them with its step and rank, the run-to-run spread between the two
submissions stated, and a level-3 control series from each whose demand stays
inside the 5 938 – 6 198 np1 band and never exceeds `local_m2l_op_cap`. In the
failure direction: if `demand_saturated` is set at any state, the log says so
explicitly and records that the measurement is a lower bound of $2^{20}$ —
which is already sufficient to select T8's demand-reduction branch, and must
not be reported as a peak.

---

### T6 — Canopy: make the count cap configurable, default unchanged — **NOT STARTED**

**Depends on:** T1 **DONE**. Independent of T5 — the knob is worth having
whichever way the measurement falls, and building it in parallel with T5 is
fine. Choosing its *value* is T8.
**Fill in:** `canopy/src/Canopy_Solver.hpp` (the config struct near `:84-95`,
the constructor route near `:192`);
`canopy/src/Canopy_DownwardSweep.hpp` (`m2l_effective_op_cap()` at `:324-331`, a
new setter beside `:309-313`, the member block near `:755-758`, the constant at
`:550`, the overflow message at `:1626-1640`);
`canopy/tasks/abstract-solver-backend.md:303-311`;
`canopy/tests/tstLaplaceSolve.hpp`.
**Reference:** `m2l_op_table_byte_budget` is the exact template — declared at
`Canopy_Solver.hpp:95`, routed at `:192`, set at
`Canopy_DownwardSweep.hpp:309-313`, read at `:324-331`, stored at `:758`,
defaulted by a constant at `:556-557`.
**Do:**
1. Add `int m2l_op_count_cap = 32768;` to `FmmConfig`, documented as a per-rank
   bound on the operator **column count**, the companion of the byte budget, and
   the thing a level-keyed basis on a deep tree actually runs out of.
2. Add `set_m2l_op_count_cap( int )` beside the byte-budget setter, invalidating
   the interaction list the same way (`:311`), since it sizes a table built
   there. Reject a value below 0 loudly; 0 is legal and means every pair takes
   the overflow path, matching the byte budget's documented "smaller than one
   column is legal" behaviour (`:305-307`).
3. Change `m2l_effective_op_cap()` to floor the configured cap by the byte
   budget's column count, replacing the `M2L_OP_COUNT_CAP` term.
4. Keep `M2L_OP_COUNT_CAP = 32768` as the **default** the config initializer and
   `_m2l_op_count_cap` member take, so a configuration that sets nothing gets
   today's cap and today's overflow set. Rewrite the comment at `:528-549` to
   say the cap is now configurable with that default, and why the default is
   what preserves every existing answer.
5. Route it from the Solver constructor at `:192`.
6. Update the overflow message (`:1626-1640`) to print the configured cap
   alongside the byte budget, so a log says which of the two bound.
7. Update the deviation note at `abstract-solver-backend.md:303-311`: the cap is
   still a count and still floored by the byte budget, the default still makes
   today's overflow set unchanged, and the constant's line is `:550`, not
   `:343`. Note that at CartesianTaylor order 3 the per-key cost is 3200 B, not
   the note's 58 KB at $P=8$, so the count cap binds by a factor of 20 on that
   basis and is the only constraint that ever binds there.
8. **Callers of `m2l_effective_op_cap()`**, enumerated: the merge's
   `effective_op_cap` read (`:1330`), the cache-overflow guard (`:1733`), the
   `:1755` bound check, the two message sites (`:1635-1636`, `:1688-1689`), the
   profiling printf (`:1683`), and in Beatnik `readDiagnostics`
   (`Beatnik_FarFieldInterface.hpp:890`). The signature does not change, so
   none needs editing; all are listed because each reads a number whose
   provenance moves from a constant to a config field.
9. Extend the T1 test cases: one driving a small `m2l_op_count_cap` with a
   generous byte budget and asserting realized equals that cap while demand
   exceeds it, and one asserting that at the default the effective cap is still
   32768 and the realized key set is byte-identical to a pre-change run.

**Exit criterion:** in the Canopy checkout, a `+profiling` build's suite passes;
a case at `m2l_op_count_cap = 4` reports `m2l_n_unique_ops() == 4` with
`m2l_n_demanded_ops() > 4` and non-zero fallback; and a default-configured case
reports `m2l_effective_op_cap() == 32768` with `m2l_realized_keys()` identical
to the same case before this task. In the failure direction: a negative
`m2l_op_count_cap` raises rather than clamping, and `m2l_op_count_cap = 0`
yields zero columns with every pair on the fallback path.

---

### T7 — Beatnik: plumb the count cap through `FmmParams` — **NOT STARTED**

**Depends on:** T6 **DONE**.
**Fill in:** `src/Beatnik_Params.hpp` (a new member beside `:395`, and the
doc comment at `:382-394`); `src/Beatnik_FarFieldInterface.hpp:1068`.
**Reference:** `m2l_op_table_byte_budget` at `Beatnik_Params.hpp:395` routed at
`Beatnik_FarFieldInterface.hpp:1068` is the exact pattern — no CLI option, no
Python counterpart, reaching one `FmmConfig` member.
**Do:**
1. Add `int m2l_op_count_cap = 32768;` to `FmmParams`, documented as reaching
   `FmmConfig::m2l_op_count_cap`, with no CLI option, and with the reason it
   exists: under `FarFieldBasis::CartesianTaylor` the keys carry the tree level,
   so occupied depth multiplies the key count and this is the cap that binds.
   Cite the measured level-4 peak from T5.
2. Route it at `Beatnik_FarFieldInterface.hpp:1068` beside the byte budget.
3. Rewrite the doctrine paragraph at `Beatnik_Params.hpp:382-394`. It currently
   says "the response to realized overflow is a lower `max_depth` or `order`,
   not a smaller table" — true about the *byte budget*, and it must now also say
   that the count cap is the constraint that binds, that it is configurable, and
   what the realized level-4 demand is. Keep the statement that lowering the
   byte budget is the wrong lever.
4. Update `README.md` only if an example's accepted arguments change. They do
   not — this member has no CLI option — so confirm and record that rather than
   editing.

**Exit criterion:** `spack install` succeeds and the level-3 FMM member still
passes unchanged at HIP np1 (measured 314 s), demonstrating that a default
`FmmParams` produces the same cap and the same answers. In the failure
direction: a scratch run of the T3 probe with `m2l_op_count_cap` set to 1024
reports `local_m2l_op_cap == 1024` and non-zero fallback at level 3, where the
default cap yields exactly zero — which is what proves the knob reaches Canopy.

---

### T8 — Decide: raise the cap, or reduce the demand — **NOT STARTED**

**Depends on:** T5 **DONE**, T7 **DONE**.
**Fill in:** whichever the decision selects — `FmmParams::m2l_op_count_cap`'s
default or the level-4 member's `makeFmmParams`
(`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp`), or
`FmmParams::max_depth` (`src/Beatnik_Params.hpp:291`). The decision and its
evidence go in the progress log either way.
**Reference:** the memory arithmetic is `demand × bytes_per_key` at
3200 B (`Canopy_CartesianTaylorBasis.hpp:506-508`); the rebuild-cost signature
is `local_m2l_op_keys_built` (`Beatnik_FarFieldInterface.hpp:279-283`) against
the cache-clear rule for a `key_needs_level` basis
(`Canopy_DownwardSweep.hpp:406-416`); the `max_depth` lever and its standard of
evidence are at `Beatnik_Params.hpp:272-291`; the `ncrit` liveness inequality is
at `:238-251`.
**Do:** take the branch the measurement selects, and record which criterion
decided it.

- **Branch A — raise the cap.** Take this if T5's peak demand is at most about
  4x the current cap (roughly 130 000 keys, about 420 MB per rank at 3200 B per
  key) **and** the per-evaluation `local_m2l_op_keys_built` increment at the peak
  does not dominate the evaluation's wall time. Set
  `FmmParams::m2l_op_count_cap` to the measured peak rounded up with headroom,
  at the level-4 member. Confirm the byte budget still does not bind:
  $\texttt{cap}\times3200<2^{31}$ holds up to 671 088 keys.
- **Branch B — reduce the demand.** Take this if the peak is 10x the cap or
  more, if `demand_saturated` was set, or if the rebuild cost dominates. The cap
  is then not the fix, because a level-keyed basis on a drifting bounding box
  rebuilds every admitted column on every rebuild. Lower `FmmParams::max_depth`
  from 10 — the lever `Beatnik_Params.hpp:272-291` names, on exactly the
  evidence it asks for — and re-run T5 to confirm the demand falls below the cap
  and the realized P2P fraction stays under `kP2PFractionBound = 0.75`
  (`Beatnik_Test_Milestone0Fmm.cpp:544`), since a shallower tree moves work into
  the near field.
- **Not available: raising `ncrit`.** It would reduce depth and therefore
  demand, but `ncrit = 8` (`:340`) is near its floor already: the liveness
  inequality puts the far field's existence at $N\gg840$ against 2562 vertices,
  and the default 64 would require $N\gg6720$. Raising it buys a cheaper solve
  that is a direct sum wearing an FMM's name, which is the one failure mode
  claim A's P2P-fraction bound exists to catch. Record it as considered and
  rejected; do not take it without a new liveness measurement.
- **Not available: a level-blind key.** `canonicalize_key` cannot zero `max_d`
  for this basis — the operator is physical, so two pairs with the same integer
  offset at different levels have different operators and would alias
  (`Canopy_CartesianTaylorBasis.hpp:471-486`). This would be a change to the
  basis's normalization, not to a cap.

**Exit criterion:** the progress log names the branch, the measured number that
selected it, and the criterion it was tested against; the selected change is in
the tree; and a T5 re-run at level 4 on HIP np1 and np4 reports
`global_m2l_fallback_pair_count == 0` at all 81 states. In the failure
direction: the same re-run at the *unchanged* configuration still reports
non-zero fallback from step 250, so the re-run is known to be measuring the
change and not a flake.

---

### T9a — Confirm claim A on a pure FMM path, before any long job — **NOT STARTED**

**Depends on:** T8 **DONE**.
**Fill in:** no source changes. A `pdebug` submission of the T5 script at the
post-T8 configuration, plus the progress log.
**Reference:** `kTauA = 1.0e-3` (`Beatnik_Test_Milestone0Fmm.cpp:320`); the
contaminated peaks T0 recorded.
**Do:**
1. Re-run the probe at level 4, HIP np1 and np4, and confirm zero fallback at
   all 81 states and demand at or below the cap.
2. Extend the probe run, or run the level-4 member's claim A by other cheap
   means, to obtain the **fallback-free** worst relative error and its step.
   Claim A at level 4 is 140 s at HIP np1 and 1432 s at SERIAL np1 — both fit
   `pdebug`.
3. Record that number against τ_A. This is the first honest measurement of
   level-4 claim A.
4. **If it still exceeds τ_A on a pure path, stop here and do not submit T9b.**
   That is a finding about the expansion at level 4 — a new task, at order,
   `mac_theta` or the bound's derivation — not a number to widen. Record it and
   raise it.

**Exit criterion:** the log carries a level-4 worst relative error measured at
zero fallback, at 17 digits, with its step and the realized P2P fraction, and
states whether it is under `kTauA`. In the failure direction: the run's own
per-state fallback column is zero at every state, so a "pure path" claim is
backed by the measurement rather than assumed from T8.

---

### T9b — Re-run the milestone tier; close T6 — **NOT STARTED**

**Depends on:** T9a **DONE** and its measured error under `kTauA`.
**Fill in:** `scripts/tuolumne/run_milestone.flux` (`-t` only, and only after
the run); `tasks/canopy/add-canopy.md` (T6 status); both progress logs.
**Reference:** the runner's own header comment says how to set `-t` from a tier
run. The gate is untouched: five `regression` members, 60 launches on tuolumne.
**Do:**
1. Finalize the env before submitting: pull the canopy and beatnik clones to the
   intended commits and `spack install` **first**, so the binary reflects them.
   Never `spack install` against the production env while a production job is
   live.
2. Submit the full milestone tier at `-q pbatch -t 1440m` — all four members,
   both backends, ranks 1 and 4, sixteen launches.
3. Only once it is green, set `-t` from the measured total with headroom, and
   record the measurement that set it.
4. Mark T6 **DONE** in `tasks/canopy/add-canopy.md`, replace the provisional
   walltime wording, and state the measured tier cost. Update the `milestone`
   tier description in `CLAUDE.md` and `docs/testing.md` if the total or the
   `-t` changes what they claim.
5. Confirm the gate is unchanged and say so: `regression` still has five members
   and 60 launches on tuolumne.

**Exit criterion:** `scripts/tuolumne/run_milestone.flux` reports
`SUMMARY: PASS (16/16 launches)`; `run_milestone.flux`'s `-t` is set from that
run's measured total; `tasks/canopy/add-canopy.md` shows T6 **DONE** with the
measured cost; and the gate runner
`scripts/tuolumne/run_regression_minset.flux` still reports five members and 60
launches. In the failure direction: a red tier leaves `-t` at `1440m` and T6 at
IN PROGRESS, and the failing member's per-check detail lines are read before any
tolerance is touched.

## Known risks

**R1 — The demand counter changes an answer.** An accidental write into `ops`,
`key_to_op` or `pair_op_idx` from the instrumentation would move the overflow
set, and it would present as a *tolerance* failure somewhere unrelated, not as a
diagnostic bug. Distinguishing measurement: T1's exit criterion compares the
realized key list, fallback count and `m2l_realized_keys()` contents between
`+profiling` and `~profiling` builds. They must be identical. If they differ,
the counter is not read-only and nothing measured with it is usable.

**R2 — The demanded set exhausts memory.** The CartesianTaylor key space is
bounded — `max_d` over `max_depth + 1` levels, `dd` over
$[-6,6]$ (`Canopy_CartesianTaylorBasis.hpp:469`), and `ii,jj,kk` over
$[-32,32]$ (`Canopy_DownwardSweep.hpp:526`) — but that product is about 39
million keys at `max_depth` 10, far more than fits. `M2L_DEMAND_COUNT_CAP` at
$2^{20}$ bounds the set at roughly 56 MB against the level-3 member's measured
peak RSS of 1 060 488 kB. Presentation if the bound is hit:
`demand_saturated` set, and the reported count is a lower bound, not a peak.
T5's exit criterion requires that case to be reported as a lower bound; T8
treats it as branch B outright.

**R3 — Demand is measured but the peak is not where the error peaks.** Three
steps are already distinct: the cap first trips at step 250, claim A's error
peaks at step 1375, and the fallback pair count peaks at step 1650. Demand and
error need not be monotone in each other, and sizing the cap from the error peak
rather than the demand peak would leave the cap short at some other step. T5
step 9 records the demand peak against all three and says which step drives
each.

**R4 — Branch A is taken and the rebuild cost destroys the run.** Raising the
cap on a `key_needs_level` basis with a drifting bounding box means rebuilding
every admitted column on every rebuild
(`Canopy_DownwardSweep.hpp:406-416`). This presents as a *timeout*, not as a
wrong answer, and a timeout in a `pbatch` job is expensive to diagnose.
Distinguishing measurement: `local_m2l_op_keys_built`'s increment per
evaluation, printed by the probe at every state. If it climbs by the full cache
size each time, the cache retains nothing and branch A's cost scales with the
cap. T8's branch-A criterion tests this before committing, and T9a re-measures
at `pdebug` scale before T9b is submitted.

**R5 — The probe measures a different configuration than the member.** The
probe re-derives claim A's parameter setup rather than sharing it, so a drift in
`ncrit`, `order`, `mac_theta`, `max_depth` or the softening would make the
measurement inapplicable — and it would present as a *plausible* demand series,
with no symptom at all. T3 step 7 echoes those five out of
`fmm.farField().params()` and fails loudly against compiled-in literals matching
`kNcrit` (`:340`), `kProductionOrder` (`:345`) and the values asserted at
`:1371-1386`.

**R6 — The probe is picked up by the tier runner.** A label or a manifest line
would put an assertion-free binary into a tier, where it would report `PASS`
unconditionally and inflate the tier's member count. The
`BEATNIK_DRIVER_SOURCES` loop (`tests/CMakeLists.txt:585-665`) forecloses this
by construction — it applies no `LABELS`, calls no `add_test` and appends to
neither manifest — and T3's exit criterion checks both installed manifests.
**`ctest` is not the check.** In `spack` mode this checkout has no build tree of
its own, and the tree spack builds in registers only
`Beatnik_Example_02_adaptive_mesh_bubble_help`, so `ctest -N -L milestone`
reports zero tests whether the probe is labelled or not — a green reading there
is evidence of nothing.

**R7 — `~profiling` demand reads as zero rather than unavailable.** A `0`
returned instead of `-1` would read as "the tree wants no keys", which would
retire the whole question with a wrong answer. The sentinel is fixed by
convention above; T2's and T3's failure-direction exit criteria both check that
a `~profiling` build reports `-1` and says so loudly.

**R8 — Branch B lowers `max_depth` and kills the far field.** A shallower tree
moves work into the near field, and past some depth the solve becomes a direct
sum that agrees with `BRSolverDirect` to round-off at any order — reading as a
*pass* with a better error than before. Distinguishing measurement: the realized
P2P pair fraction, which claim A bounds at `kP2PFractionBound = 0.75`
(`:544`) and the probe prints at every state. T8's branch B requires it to stay
under that bound.

**R9 — τ_A still fails on a pure path.** Entirely possible: the contaminated
peak is 1.25x over, and removing the contamination may not close that gap. It
would present identically to today's failure — a τ_A exceedance at level 4 —
which is why T9a exists as a `pdebug` step before the `pbatch` tier run, and why
its step 4 forbids proceeding. The distinguishing measurement is the fallback
column: zero at every state means the number is about the expansion, and the
response is a task at `order` or `mac_theta`, never a wider τ_A.
