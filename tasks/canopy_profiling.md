# Canopy profiling and optimization, driven through Beatnik

**Status:** IN PROGRESS — T0, T1, T2, T3, O1, O8 and O5 done; V1 is triggered by O1's `[acc]` moves but **deferred by ruling** to the end of Phase O (log, "Decision: defer V1; a milestone failure from O1 is accepted"); O2 and O4 NOT TAKEN (triggers unmet after O1); **O9 next**, then O10 (sequence in [Phase O](#phase-o--optimize-single-node)).

## Problem

Beatnik's `--br-approximation fmm` path is accurate enough to ship — it holds
the milestone tier's claim A ($\tau_A = 1.5\times10^{-3}$) at icosphere levels
3 and 4 — but it is **far slower than the direct sum it is meant to replace**.
At level 4 (2562 vertices, HIP, np1) the FMM costs about 44x `direct` per step
over the first 25 steps and about 140x over a 2000-step trajectory, and its
per-step cost climbs 2.7x along that trajectory (0.434 to 1.189 s/step) while
`direct` stays flat at 0.0085 s/step
([canopy/add-canopy-progress-log.md](canopy/add-canopy-progress-log.md) `## T5`,
"The step-7 sweep does not fit"). Nothing has located that cost
phase by phase on Beatnik's workload, and no vertex count is known at which the
FMM wins.

This document profiles Canopy **as Beatnik drives it**, then optimizes Canopy
(and the thin Beatnik adapter in front of it) against those measurements. All
sessions run in this checkout (`beatnik/`); Canopy source is edited in the
sibling clone `../canopy` (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`),
which the dev spack env builds in place as a `spack develop` spec. Profiling
Canopy therefore means building and running **Beatnik**, because the workload
— a deforming closed 2-D sheet re-evaluated at every RK3 stage — exists only
here.

**End state, single node (Phase P and Phase O):**

- The log carries a phase-by-phase cost model of one FMM evaluation and of one
  timestep, at levels 4, 5 and 6, on HIP at np 1 and 4 (SERIAL at level 4).
- The maintenance-action histogram, the operator-construction share, and the
  `fmm`-versus-`direct` per-step ratio with the P2P pair fraction are recorded
  at each measured size, together with the crossover vertex count (or the
  statement that there is none in range).
- Every optimization that measurement justified has landed in Canopy or the
  adapter, each with its own measured gain and an accuracy check.
- The milestone tier and the regression gate are green on the optimized build,
  and README states the measured speedup and the crossover.

**End state, multi-node (Phase S):** the same instrument, run at levels 6–7 on
16–64 ranks, locates the communication and maintenance costs that only appear
at scale, and the optimizations that measurement justifies have landed.

**Out of scope:**

- AMR and every other port capability (`tasks/milestone1.md`, `tasks/framework.md`).
- The Riesz-scalar FMM path (add-canopy T7).
- Raising the FMM order to 4, which waits on `../canopy/tasks/02_oracle_extension.md`.
- The FMM's numerical method: basis, MAC and softening. Optimizations may change
  the tree and so the answer, but only within the accuracy bound in
  [Conventions](#conventions).
- Reproducibility across runs, which is add-canopy R8.

## Approach

**Measure first, through one instrument, then optimize in order of measured
share.** Phase P builds the instrument once and produces the cost model. Phase O
takes optimizations one at a time in the order Phase P's numbers rank them.
Each O task re-runs the same fixed **standard measurement set** (SMS, defined
by T3), so every gain is measured against the same baseline at the same
states. Phase S repeats the measurement at scale.

### The instrument

The instrument has three layers, all landed in one build (T1 + T2):

1. **Canopy's own timers**, at `CANOPY_PROFILING_LEVEL` 2.
   - `CANOPY_SCOPED_TIMER` and `CANOPY_SCOPED_TIMER_DETAILED` accumulate into a
     process-global registry plus a never-reset `timer_totals()`
     (`../canopy/src/Canopy_Profiling.hpp:134-164`).
   - Level 2 is the level that times the host operator build
     (`TIMER_ILIST_S4_OP_TABLE_BUILD`), the dual-tree traversal
     (`TIMER_COMMPLAN_DTT_TRAVERSAL`) and the rest of the interaction-list
     stages (`:89-109`).
   - The dev env's canopy spec today carries `+profiling` with no
     `profiling_level`, which resolves to **level 1**
     (`../canopy/CMakeLists.txt:255-283`; the spack package passes no level for
     `default`, `package.py:129-133`). T2 raises it to 2.
2. **A run-time reporting switch in Canopy** (T1). In a profiling build, every
   `solve()`, `migrate()`, `rebalance()` and setup prints Min/Max/Mean tables,
   each behind an `MPI_Reduce`. The print macros are at
   `../canopy/src/Canopy_Profiling.hpp:470-540`; the call sites are
   `Canopy_Solver.hpp:317,381-382,409-410,653-654,697,725`,
   `Canopy_DownwardSweep.hpp:3106-3107`, `Canopy_UpwardSweep.hpp:715` and
   `Canopy_P2P.hpp:1269`. Alongside them are the `[Canopy diag] auto_maintain`
   lines (`Canopy_Solver.hpp:457-543`) and the per-rank
   `[Canopy Diagnostics] M2L operator table` line
   (`Canopy_DownwardSweep.hpp:2003-2020`).
   
   That output perturbs what it measures, and it floods the log: the level-4
   HIP milestone log from T9b is 38 MB with 79 072 `[Canopy Diagnostics]` lines
   (`beatnik_milestone_Milestone0FmmL4_HIP.f3d7zPLmZqxj.log`). The switch turns
   printing off while leaving accumulation on. The default stays "print", so no
   existing Canopy test or log consumer changes.
3. **Beatnik-side phase timers and a probe** (T2).
   - The timers go in `FarFieldSolver::evaluate`, around the nine phases of the
     round trip.
   - New per-evaluation counters go in `FarFieldDiagnostics`.
   - A new measurement driver, `Beatnik_Probe_FmmCost`, runs an **FMM-driven**
     frozen-mesh trajectory and reads both timer registries around every
     timestep. At sampled states it also evaluates a standalone direct solver
     and a standalone FMM solver side by side, which gives two things per
     sampled state at no extra job:
     - the per-step `direct` cost, so the crossover comes out of the same run;
     - claim A's error metric, so every cost run is also an accuracy check.

### Why the trajectory is FMM-driven

The maintenance path Canopy picks (`Migrate`, `Rebalance` or `Rebuild`;
`../canopy/src/Canopy_Solver.hpp:186-191,442-544`) depends on how far the
particles moved since the last evaluation. A direct-driven trajectory sampled
every 25 steps — the shape of `Beatnik_Probe_FmmKeyDemand` — presents Canopy
with 25 steps of motion at once, which is not the motion it sees in
production: one RK3 stage at a time, three evaluations per step
(`src/Beatnik_TimeIntegrator.hpp:155,162,168`, each through
`ZModelSolver::computeRightHandSidePotential` to
`_br_solver->computeInterfaceVelocity` at `src/Beatnik_ZModelSolver.hpp:223`).
Only an FMM-driven run measures the production maintenance pattern, so the
probe drives one, and pays the FMM cost the document exists to reduce.

### Matched states, not matched step counts

Per-step cost grows along the trajectory as the bubble deforms and the tree
deepens. Occupied depths go from 5 to 8 at level 4
(`../canopy/tasks/tree-opt.md`, "Measured on a downstream application's
configuration"). A short run therefore under-predicts a long one by up to 2.7x,
and README Known Issues records that as a budgeting hazard. Every comparison in
this document is made **at the same (level, step) states**, and every
measurement job carries a remaining-budget guard rather than trusting an
estimate. The canonical guard is `scripts/tuolumne/t5_divergence.flux`'s
skip-on-remaining-budget: its estimates were 2.7x wrong and it still protected
the measurement.

Matching physical time across levels is not affordable. The adaptive dt scales
off the minimum edge, so level 5 needs about 2x the steps of level 4 to reach
the same time, and level 6 about 4x, each step costing more. **Levels 5 and 6
are therefore measured over a shorter part of the trajectory than level 4:**
T3 measured level 5 to step 800 (t = 1.54, about level 4's step 990) and level
6 to step 250 (t = 0.69), against level 4's 2000 steps (t = 1.96). Level 4
alone carries the late-trajectory measurement. T3 records the physical time
each measured state corresponds to, so the comparison is explicit.

### What is already known, and what it predicts

This is the prior Phase P must confirm or refute. It is not a conclusion to
build on.

- **The M2L operator table is rebuilt from scratch at every evaluation.**
  - `CartesianTaylorBasis` sets `key_needs_level = true`
    (`../canopy/src/Canopy_CartesianTaylorBasis.hpp:486`).
  - Every maintenance path calls `_push_root_half_width()` before
    `_downward.setup` (`Canopy_Solver.hpp:648,694,722`, definition `:801-804`).
  - `set_root_half_width` clears the whole cache on any change of the width,
    compared as an exact double (`Canopy_DownwardSweep.hpp:451-460`).
  - The root box is recomputed from the moving particles, so it changes at
    every build. The probe in `add-canopy-t6` measured
    `keys_built_delta == unique_ops` in 405 of 405 rows at level 3
    ([add-canopy-t6-progress-log.md](add-canopy-t6-progress-log.md) `## T3`,
    "What only running revealed").
  - That is **three full table rebuilds per timestep**, one per evaluation. The
    host build is a serial loop over keys
    (`Canopy_CartesianTaylorBasis.hpp:1186-1240`), up to 37 678 columns at
    3200 B each at level 4.
- **A fix for the rebuild already exists in Canopy, and Beatnik does not turn it
  on.** `FmmConfig::quantize_root_half_width` (`Canopy_Solver.hpp:126`, default
  `false`) rounds the root half-width up to a power of two, so the width is
  bit-identical across every build whose box stays inside one octave and the
  cache survives. On Canopy's own drift fixture, builds 2–4 rebuilt 5.3–7.1 %
  of the admitted columns, against 100 % with it off
  (`../canopy/tasks/tree-opt.md` B2; `../canopy/README.md`, "The M2L operator
  cache empties on every build"). `FarFieldSolver::buildConfig`
  (`src/Beatnik_FarFieldInterface.hpp:1124`) never set it before O1, and
  `FmmParams` (`src/Beatnik_Params.hpp:167-434`) had no field for it.
- **At level 4 the downward sweep dominates an evaluation.** The T9b level-4
  HIP log's per-solve breakdown reads Downward 97.6 %, Upward 1.3 %, P2P 1.1 %
  of `solve()` at steady state. The interaction-list build, and with it the
  operator-table rebuild, sits inside the downward timer.
- **Range-guard fallback is 1.58 % of M2L pairs over the level-4 series and
  4.2 % at the worst state.** At HIP it costs 52–109x a GEMM pair
  (`../canopy/README.md`, "Batch the per-pair M2L fallback on the device"). That
  makes it a share of M2L **kernel** time comparable to the whole GEMM path.
- **Tree balancing does not pay.** `tree_balance_max_level_delta` was measured
  in Canopy's A3 to add fallback work and save nothing
  (`../canopy/tasks/tree-opt.md` A3 `**Met.**`). It is not a candidate here.
- **The adapter does host work on every evaluation that is not FMM work.**
  `p2pPairCountLocal()` (`src/Beatnik_FarFieldInterface.hpp:1298-1323`) builds
  a host hash every evaluation and is always computed. It is followed by a
  9-element `MPI_Allreduce` (`:1589`) and the forward distributor's own claims
  migrate (`:1237`). These are sized in T3, not assumed.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Where sessions run | This checkout, `beatnik/`, in `spack` build mode (`BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh` reports `profile.local.sh`) | Profiling Canopy needs Beatnik's binary; the dev env builds both. |
| Where Canopy is edited | `../canopy`, branch **`profiling-canopy`** | The dev env's `canopy@=develop` develop spec has no `path:`, so it builds `<env>/canopy` (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml:36-37`). |
| Canopy conventions | `../canopy/CLAUDE.md` applies to every Canopy file: build only named targets, no clang-format, README "Future Optimizations" and "Known Issues" discipline, KaTeX math. License banner is "Copyright (c) 2025 by the Canopy authors" (`../canopy/src/Canopy_Solver.hpp:1-10`) | Canopy is its own project; its rules govern its files. Beatnik files follow this repo's `CLAUDE.md`. |
| Commits | Separate commits in each repo. A Canopy change is **committed before** the Beatnik build that measures it. Every measurement job logs `git -C ../canopy rev-parse HEAD` and `git -C ../canopy status --porcelain`, and the Beatnik log entry names the Canopy hash | A `spack develop` build compiles the working tree, so an uncommitted Canopy edit is otherwise an unrecorded variable in every number. |
| Beatnik build | `spack env activate ~/spack_envs/tuolumne_beatnik && spack install` (see `systems/tuolumne/claude.md` §3, including the `HIPCC_*_FLAGS_APPEND` unset) | `spack` mode; never `cmake`/`make` in this checkout. Canopy is header-only and is included by every Beatnik TU that creates a BR solver, so **any Canopy edit is a full Beatnik rebuild, about 10–11 min wall** ([canopy/add-canopy-progress-log.md](canopy/add-canopy-progress-log.md) `## T5`, "Cost and mechanics"). |
| Canopy standalone build | Out-of-tree tree `../canopy/build-profiling/`, configured once by T1 with `../canopy/run_cmake_tuolumne.sh` under `spack env activate ${HOME}/spack_envs/tuolumne_trilinos`. Build **only the stems a task names**: `make -j Canopy_Test_<Stem>_MPI_<DEVICE>` | `../canopy/systems/tuolumne/claude.md` §1–3. ccache makes a re-build of a named stem cheap. |
| Canopy test runs | `ctest` inside a flux batch script under `../canopy/scripts/tuolumne/`, with an anchored regex. The env and `GLIBC_TUNABLES` come from `../canopy/systems/tuolumne/claude.md` §4–5. SERIAL at np 1–6, HIP at np 1–4 | Canopy's tests register HIP at np 1–4 only (`../canopy/run_cmake_tuolumne.sh`). |
| Spack env for measurement | **dev env** (`BEATNIK_USE_PROD` unset). Every new script under `scripts/tuolumne/` refuses `BEATNIK_USE_PROD=1` loudly, as `t5_divergence.flux:143` does | `canopy +profiling` is dev-only (`systems/tuolumne/claude.md` §1); a production binary has no timers to read. |
| Profiling levels | dev env: `canopy +profiling profiling_level=2` (T2), `beatnik profiling_level=2` (already). Production unchanged: `canopy ~profiling`, `beatnik profiling_level=1` | Level 2 is where the operator-build and DTT timers live. |
| Measurement driver | `Beatnik_Probe_FmmCost`, in the **"Measurement drivers — IN NO TIER"** loop (`tests/CMakeLists.txt:584-675`): no `LABELS`, no `add_test`, no manifest. Installed, launched by `beatnik_exe` | A probe measures and asserts nothing. A measurement that carries a pass criterion gets tuned to pass. |
| Accuracy bound for any optimization | Claim A's metric, `max|u_fmm − u_direct| / max|u_direct|` over owned vertices (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1447-1490`). **No regression in aggregate, at every level:** the worst value over that level's SMS states after the task is ≤ the worst value before it (the previous task's, or T3's for the first), compared exactly. Individual states may move either way. Level 4 therefore also stays within `kTauA` = 1.5e-3 (`:320`). Every change is still reported per state, against the previous task and against T3 | The user's ruling after T3 (log, "Decision: no accuracy regression in any O task"). `[acc]` is bit-reproducible across repetitions, nodes, rank counts and backends (T3), so "no regression" needs no tolerance. $\tau_A$ alone would not do: it was set at level 4 from the reference treecode's own error, and the unoptimized `ncrit 8` baseline already exceeds it at L5 (1.6e-3) and L6 (2.7e-3) ([log](canopy_profiling-progress-log.md) "L5/L6 accuracy check"). |
| Backend priority | **HIP first.** SERIAL is measured at level 4 and must stay correct, but a SERIAL-only slowness (np4 slower than np1 on SERIAL: 2.21x at level 3, 1.37x at level 4) is O7, the last optimization task | MI300A production runs HIP. |
| Rank counts | Phase P/O: np 1 and 4, one node. Phase S: 16–64 ranks | One node holds 4 MI300A APUs. |
| New file headers | Beatnik: the BSD-3 block with `SPDX-License-Identifier: BSD-3-Clause` exactly as `src/Profiling.hpp:1-10`. Canopy: Canopy's banner | `CLAUDE.md` "Match the license/header convention". |
| Formatting | Never run clang-format, `clangformat.sh` or `cabana-format`, in either repo | The user formats by hand. |
| Sentinel for "not compiled in" | `-1`, never `0` | Zero is a legal count or time. This matches `FarFieldDiagnostics`' existing demand fields (`src/Beatnik_FarFieldInterface.hpp:214-399`). |

#### Build and run economy (applies to every task)

Builds take about 10 minutes and queues and runs take hours, so wasted builds
and jobs are the dominant cost of this programme.

1. **One `spack install` per task, at most.** Finish *all* of a task's Canopy
   and Beatnik edits, compile-check the Canopy side with the standalone stem
   build, and only then install. Never install to "see if it compiles".
2. **Never `spack install` the dev env while a job that uses it is queued or
   running.** A queued job runs whatever binary is installed *at launch*, and a
   running job whose binary is overwritten takes a SIGBUS (rc=135). The window
   while a job runs is for analysis, reading code and writing the *next*
   change, not for installing it.
3. **One job per measurement, many configurations per job.**
   - Each measurement job runs its whole matrix in one allocation, with a
     remaining-budget guard. Copy `t5_divergence.flux`'s shape and the
     `BEATNIK_T5_SWEEP` row override.
   - `pdebug` (60-minute cap, short queue) is the default.
   - `pbatch` is used only for the milestone tier and Phase S.
   - When an np1 job and an np4 job are both needed, submit them together; they
     do not share a node.
4. **Canopy unit tests and the Beatnik probe go out together.** Submit both
   jobs as soon as both builds finish, in the same sitting.
5. **The expensive checks run at phase boundaries, not per task.** The
   milestone tier (4.2 h wall split, 8.7 node-hours) and the regression gate
   run at V1 and V2 only. Per-task accuracy comes free from the probe.
6. **Read logs; do not attach.** Wait with `flux job status <id>` or by polling
   the `.log`. `flux job attach` forwards SIGTERM and can kill the job.
7. **Mine before you run.** Anything an existing log already answers is read
   from the log (T0), not re-measured.

### Deliberate deviations

- **The probe re-derives claim A's setup instead of sharing it with
  `Beatnik_Test_Milestone0Fmm.cpp`.** The same choice was made for
  `Beatnik_Probe_FmmKeyDemand` (`tasks/add-canopy-t6.md`, Deliberate
  deviations). Factoring a 2201-line tier member to share 200 lines would mix a
  refactor into a measurement. The probe duplicates the parameter setup and the
  `fieldDifference`/`fieldScale` helpers, and its header names the constants
  that must agree with the member.
- **Canopy's default stays "print every table".** The reporting switch is
  opt-out and only the probe opts out. Canopy's own tests, and every Beatnik
  test that runs in a profiling build, keep printing exactly as now.
- **Beatnik's `src/Profiling.hpp` is repurposed, not replaced.** It exists,
  installs (`src/CMakeLists.txt:21`), and is included nowhere. Its eleven keys
  (`:68-78`) name phases of an `FmmBRSolver` that does not exist on this
  branch. T2 replaces the keys with the phases of `FarFieldSolver::evaluate`
  and keeps the registry, the fencing `ScopedTimer` (`:112-133`) and the macros.
- **Every measurement runs at `ncrit 8`, not production's 64.** `ncrit 8` is
  the milestone members' leaf size and puts the most work through the far field
  (74.6 % of pairs through M2L at level 4, against 15 % at `ncrit 64`). At 64
  much of the far field becomes P2P and would hide the unoptimized code this
  document exists to fix. It costs accuracy above level 4: the Python reference
  holds about 9.3e-4 at L5 and L6, while Beatnik at `ncrit 8` reaches 1.6e-3
  and 2.7e-3 (README Known Issues). V1 re-measures the final speedup at
  `ncrit 64` as well, because README's claim is about production.
- **The milestone tier is not re-run per optimization.** Each O task's accuracy
  evidence is the probe's per-state claim-A metric on the SMS, which covers the
  same states. The tier's 2000-step claim B is re-asserted at V1, once for the
  whole of Phase O. If an O task moves the metric by more than 20 % of its
  baseline at any state, V1 is run before the next O task starts rather than at
  the end of the phase.

## Current state

- **The FMM path works and is accurate.** Production parameters are
  CartesianTaylor, order 3, `mac_theta 0.3`, `max_depth 10`, softening
  `sqrt(blob())`, `near_softening_factor 0`, and `ncrit 8` in the milestone
  members (the `FmmParams` default is 64, `src/Beatnik_Params.hpp:256`). The
  level-4 worst claim-A error is 1.2473e-3. The milestone tier is 16/16 green
  as of T9b. Above level 4 the error grows with N at fixed `ncrit`, which the
  reference's does not; see [Deliberate deviations](#deliberate-deviations).
- **Three FMM evaluations per timestep**, one per RK3 stage (see
  [Why the trajectory is FMM-driven](#why-the-trajectory-is-fmm-driven)). Each
  goes through `FarFieldSolver::evaluate`
  (`src/Beatnik_FarFieldInterface.hpp:1395-1733`, as of O1) in these phases,
  each under its T2 `ff_*` timer:

  | # | Phase | Code | Timer |
  | --- | --- | --- | --- |
  | 1 | pack | `packSources`, `:1443` (body `:1193-1218`) | `ff_pack` |
  | 2 | forward distribute | `buildForwardDistributor`, `:1455` (body `:1252-1324`, its claims migrate at `:1284`), then an `MPI_LAND` reduce, `:1456` | `ff_fwd_distribute` |
  | 3 | forward migrate | `Cabana::migrate`, `:1467` | `ff_fwd_migrate` |
  | 4 | `auto_maintain` | `:1472`, or `setup` on first use or tag mismatch, `:1483-1494` | `ff_maintain` |
  | 5 | solve | `:1499` | `ff_solve` |
  | 6 | contract | `:1503-1559` | `ff_contract` |
  | 7 | reverse distribute | `:1561-1579` | `ff_rev_distribute` |
  | 8 | reverse migrate | `:1585` | `ff_rev_migrate` |
  | 9 | scatter | `:1590-1633` | `ff_scatter` |
  | — | diagnostics | `readDiagnostics` `:1645` (body `:968-1030`), `p2pPairCountLocal` (`:1345-1370`), the `MPI_Allreduce` at `:1676`, the cumulative counters at the end | `ff_diagnostics` |

  `ff_evaluate_total` covers the whole of `evaluate`.
- **Beatnik's FMM path is timed at `beatnik profiling_level=2` (T2).**
  `src/Profiling.hpp` carries the ten `ff_*` phase keys and
  `ff_evaluate_total`, fenced on exit, with a never-reset `timer_totals()`.
  The dev env builds at level 2; production (level 1) compiles them out.
  `Beatnik_Probe_FmmCost` differences the totals around every step. The coarse
  test wall clocks remain: `[m0d1] TIMING` in
  `Beatnik_Test_Milestone0Run.cpp:478`, `[t6] COST` in
  `Beatnik_Test_Milestone0Fmm.cpp:2215`, and `eval_wall` in
  `Beatnik_Probe_FmmKeyDemand.cpp:579-583`.
- **`FarFieldDiagnostics` describes the last evaluation only**
  (`src/Beatnik_FarFieldInterface.hpp:214-399`). With three evaluations per
  step, a caller that reads it after `advanceOneStep` sees the third stage's
  maintenance action and nothing of the first two. Nothing in `src/` prints it.
  The test and probe readers are `Beatnik_Test_FmmVsDirect.cpp:717`,
  `Beatnik_Test_Milestone0Fmm.cpp:1467,1968`, `Beatnik_Test_FmmScan.cpp:826-831`
  and `Beatnik_Probe_FmmKeyDemand.cpp:585`.
- **The driving BR solver is not reachable from outside `Solver`.**
  `Solver::_br_solver` (`src/Beatnik_Solver.hpp:1099`) is private and has no
  accessor. It is created from `_params.fmm` at `:264-265` through
  `createBRSolver`.
- **Canopy prints every timing table on every call in a profiling build** (see
  [The instrument](#the-instrument)). The dev env's level is 1, so the
  operator-build and DTT timers are compiled out.
- **The existing logs already hold per-solve Up/Down/P2P breakdowns** for every
  T9b milestone launch: the eight `beatnik_milestone_Milestone0*_{SERIAL,HIP}.*.log`
  files in this directory. Their `[Canopy Diagnostics] solve() phase breakdown`
  blocks interleave with claim-A `[note]` lines (claim A's direct-driven
  evaluations) and with claim B's FMM-driven steps.
- **No measurement exists above 2562 vertices on this solver.** The
  `/p/lustre5/stewartj/beatnik/fmm/` runs (`16p_*mesh`, `n1536_p256`) came from
  the pre-redesign planar solver removed in `89ec015` and cannot be re-run. The
  only driver at larger N is the icosphere bubble at higher
  `--icosphere-subdivisions`: level 5 is 10 242 vertices and level 6 is
  40 962. `Beatnik_Test_Milestone0Run.cpp:286` accepts any level ≥ 0.
- **add-canopy T8's measurements are owned here.**
  [canopy/add-canopy.md](canopy/add-canopy.md) T8 is marked MOVED, and T3 and
  V1 below carry its exit obligations: the action histogram, the phase split,
  the operator-construction share, `fmm` vs `direct` with the P2P fraction, and
  the README speedup and crossover statement.

## Progress log

The session record is **[canopy_profiling-progress-log.md](canopy_profiling-progress-log.md)**.
Read it before implementing a task, changing a signature, or reopening a
question this document treats as settled. The reasoning, the measured numbers
and the Canopy commit hash behind each statement here are there, cited by task
ID. Append a `## <task ID>` section at the end of every task.

---

# Task sequence

## Phase P — measure (single node)

### T0 — Baseline from the existing T9b logs (no build, no run) — **DONE**

**Depends on:** none.

**Fill in:** `tests/regression_tests/fmm_profile_reduce.py` (new; BSD-3 header,
Python comment style as `fmm_divergence_ladder.py`); the progress log.

**Reference:**
- The four T9b FMM logs in this directory, `beatnik_milestone_Milestone0{Fmm,FmmL4}_{SERIAL,HIP}.*.log`,
  each holding an np1 and an np4 launch: the **eight launches**. The four
  `Frozen*` logs ran the direct sum only and carry no breakdown blocks.
- The block format, from `print_solve_breakdown`
  (`../canopy/src/Canopy_Profiling.hpp:333-373`): a header line naming the rank
  count, then Total, Upward, Downward and P2P rows of mean seconds and percent.
- The claim A and claim B boundaries are `Beatnik_Test_Milestone0Fmm.cpp`'s
  `[note] ... claim {A,B} step N` lines. `[t6]` contributes only the
  `CLAIM A SERIES` table and the `COST` line.

**Do:**
1. Write a reducer that splits each FMM log into (rank count, claim A
   evaluations, claim B evaluations) and reports, per launch, the per-solve
   Total/Upward/Downward/P2P distribution (median, p90, max) and its trend
   over claim B's steps.
   - Attribute blocks by their position relative to the `[note] ... claim A step N`
     and `[note] ... claim B step N` lines.
   - **Fail loudly** on a block the reducer cannot attribute, rather than
     dropping it.
2. Write the same reducer so T2's probe output (`[cost]` rows, defined in T2)
   is a second input format. T3 and every O task then use one tool.
3. Record in the log:
   - the per-launch tables;
   - the ratio of summed `solve()` time to the `[t6] COST` claim-B wall, which
     is the share of a timestep that is inside `solve()` at all;
   - whether the downward share grows with the step count, as the operator
     rebuild predicts.

**Exit criterion:** `python3 tests/regression_tests/fmm_profile_reduce.py beatnik_milestone_Milestone0FmmL4_HIP.f3d7zPLmZqxj.log`
prints a per-phase table for both np1 and np4 with an evaluation count equal to
the number of `solve() phase breakdown` blocks in that log (`grep -c`), and the
log records the eight launches' tables. Failure direction: the same command run
on a log with one breakdown block truncated mid-table exits non-zero, naming the
line.

**Met.** `tests/regression_tests/fmm_profile_reduce.py` reduces all four FMM
logs with exit 0, attributing every block: 6082 per launch, 81 claim A + 6000
claim B + 1 final probe, and 12164 per log, equal to `grep -c 'solve() phase
breakdown'` on each. On the exit-criterion log it prints the np1 and np4 tables.
The np4 block carrying another rank's `[H2SCR]` line at line 344082 reduces
cleanly. A scratch copy with block 3000's Downward and P2P rows deleted exits 1,
naming the block's header line, 126009. A copy cut off mid-trajectory also exits
1, naming the claim-B count shortfall. `--self-test` exercises the `[cost]` /
`[acc]` / `[cost-total]` grammar and the block rules on hand-written fixtures.
The eight launches' tables are in the progress log under `## T0`.

---

### T1 — Canopy: a run-time reporting switch — **DONE**

**Depends on:** none. T1 and T2 share **one** Beatnik `spack install`: implement
T1, commit it in `../canopy`, implement T2, then install once.

**Fill in (all in `../canopy`):**
- `src/Canopy_Profiling.hpp`: a process-global `reporting_enabled()` and
  `set_reporting( bool )`, beside the registry (`:134-164`).
- Every `print_*` function reached by the macros at `:470-540` returns
  immediately when reporting is off. This includes `print_solve_breakdown`
  (`:333-373`) and `print_timing_table` (`:420`).
- The `[Canopy diag] auto_maintain` `fprintf`s (`src/Canopy_Solver.hpp:457-543`)
  and the M2L operator-table line (`src/Canopy_DownwardSweep.hpp:2003-2020`)
  are gated on the same flag.
- `README.md`: document the switch in a "Profiling" subsection under
  "Building and Running Tests".

**Reference:** the macros and print functions above. The `print_*` functions
call `MPI_Reduce` (`:339`, and the one inside `print_timing_table`), so the
switch removes collectives as well as output.

**Do:**
1. Default **on**. Accumulation into `timer_registry()` and `timer_totals()`
   is unaffected by the flag.
2. **The flag must be set identically on every rank, because the print
   functions are collective.** State that precondition on the declaration. The
   switch cannot cheaply detect a mismatch, so the declaration comment is the
   only guard, and a mismatch deadlocks.
3. In a non-profiling build the two functions exist and do nothing, so a
   caller needs no `#ifdef`.
4. Configure `../canopy/build-profiling/` (see [Conventions](#conventions))
   and build `Canopy_Test_LaplaceSolve_MPI_SERIAL` and
   `Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`. Add a gtest case to
   `tests/tstLaplaceSolve.hpp` that calls `set_reporting( false )`, runs one
   solve, asserts that `timer_totals()` gained the `downward_total` key, and
   restores the flag.
5. Commit on `profiling-canopy`.

**Exit criterion:** in one flux job, `ctest --output-on-failure -R '^Canopy_Test_(LaplaceSolve|CartesianTaylorSolve)_MPI_SERIAL_np_[1-6]$'`
in `build-profiling/` passes, and the new case's output contains **zero**
`[Canopy Diagnostics]` lines while the existing cases' output still contains
them. Failure direction: with the gate in `print_solve_breakdown` deleted, the
new case's captured output contains a `solve() phase breakdown` block. Check
this once by hand and record it in the log. It is not left in the code.

**Met.** Canopy `c182348` on `profiling-canopy` (switch in `aa56e64`, two
more prints gated in `c182348`). Flux job `f3diM2F6p3YB`, `build-profiling/` at
level 2:
- **The exit criterion.** LaplaceSolve and CartesianTaylorSolve SERIAL
  np 1–6 all pass.
- **The new case.** `LaplaceSolve.reportingSwitch` captured zero
  `[Canopy Diagnostics]` and zero `[Canopy diag]` lines on all 21 ranks, and
  `downward_total` still accumulated.
- **The existing cases** still printed 8917 `[Canopy Diagnostics]` lines.
- **The failure direction**, run once in the same job: a hand-built binary
  with the `print_solve_breakdown` gate deleted fails the case, with the
  breakdown block in its captured output.

Every `[Canopy diag]` print is gated, not only the ones named above: the
partition, `refresh_ownership` and `p2p_inter_leaf_pairs` lines too.

---

### T2 — Beatnik: phase timers, per-evaluation counters, and `Beatnik_Probe_FmmCost` — **DONE**

**Depends on:** T1 committed in `../canopy`.

**Fill in:**
- `src/Profiling.hpp`: replace the eleven stale keys (`:68-78`) with one key
  per phase in the [Current state](#current-state) table: `ff_pack`,
  `ff_fwd_distribute`, `ff_fwd_migrate`, `ff_maintain`, `ff_solve`,
  `ff_contract`, `ff_rev_distribute`, `ff_rev_migrate`, `ff_scatter`,
  `ff_diagnostics`, plus `ff_evaluate_total`.
  - Rewrite the level comment (`:26-32`) to describe `FarFieldSolver::evaluate`.
  - Add a never-reset `timer_totals()` mirroring Canopy's (`Canopy_Profiling.hpp:145-159`).
- `src/Beatnik_FarFieldInterface.hpp`:
  - Include `Profiling.hpp`.
  - Wrap each phase of `evaluate` (`:1348-1625`) in `BEATNIK_SCOPED_TIMER_DETAILED`
    (a no-op below level 2, so production is unchanged).
  - Add cumulative, always-on counters to `FarFieldDiagnostics` (`:214-399`):
    `evaluations`, and `maintenance_counts[4]` indexed by the existing
    maintenance enum, so a caller can difference them across a timestep.
    Increment them in `evaluate`. They are integers, carry no profiling
    dependency, and add no collective.
- `src/Beatnik_Solver.hpp`: a `const br_solver_type& brSolver() const`
  accessor for `_br_solver` (`:1099`). It is a new signature with no existing
  callers.
- `tests/regression_tests/Beatnik_Probe_FmmCost.cpp` (new), added to
  `BEATNIK_DRIVER_SOURCES` in `tests/CMakeLists.txt` (the list ending at about
  `:640`) with a comment block in the style of its neighbours.
- `scripts/tuolumne/canopy_profiling_smoke.flux` (new, dev env, refuses PROD).
- The dev env `/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml:16` gets
  `profiling_level=2` on the canopy spec. Make the same edit to the snapshot
  `systems/tuolumne/spack.yaml`, and update `systems/tuolumne/claude.md` §1's
  sentence on the canopy variant. Then run `spack concretize -f`.
- README: the probe is a test driver, not an example, so no example arguments
  move. Add a line under "Future Optimizations" → "Higher profiling levels"
  only if the user approves.

**Reference:**
- `Beatnik_Probe_FmmKeyDemand.cpp` is the closest existing driver to copy from:
  its setup, its header-after-first-evaluation ordering (`add-canopy-t6-progress-log.md`
  `## T3`, "Three shape departures") and its `-1` sentinels.
- `Beatnik_Test_Milestone0Fmm.cpp:1447-1490` defines claim A's evaluation and
  error metric.
- `Beatnik_Test_Milestone0Run.cpp:400-480` is the step loop.

**The probe's interface.**
`Beatnik_Probe_FmmCost_MPI_<DEVICE> <level> <steps> <sample_every> [key=value ...]`.
Its rows follow the grammar in
[canopy_profiling-progress-log.md](canopy_profiling-progress-log.md) `## T0`,
"Decisions".
- **Accepted keys:** `ncrit` (default 8), `order` (default 3; only the orders
  the binary instantiates), `mac_theta` (0.3), `m2l_op_count_cap` (65536,
  matching the level-4 member), and `canopy_report` (0 or 1, default 0).
- **An unknown key throws, naming it.** O tasks that add a knob add its key
  here.
- **Fixed configuration:** frozen mesh, `--br-approximation fmm`, no
  checkpoints.
- **Per timestep:**
  - Snapshot both registries' `timer_totals()` and the driving solver's
    cumulative counters (reached through `brSolver()` and a `dynamic_cast` to
    `BRSolverFMM`; a failed cast throws).
  - Call `advanceOneStep`, fence, and difference the snapshots.
  - Per-rank values are reduced to max and mean.
- **Each `sample_every` steps, and at step 1 and the last step:**
  - Emit one `[cost]` row carrying: step, physical time, step wall, the eleven
    `ff_*` phase times, and Canopy's `upward_total`, `downward_total`,
    `dn_build_ilist`, `ilist_s1`…`ilist_s5_*`, `ilist_s4_op_table_build`,
    `m2l_kernel`, `l2p`, `p2p_total`, `migrate_total`, `rebalance_total`,
    `setup_total`, `partition`, `repartition`, `redistribute`, `comm_plan_build`
    and `commplan_dtt_traversal`.
  - Each time value is summed over the window and divided by the evaluations in
    the window.
  - The row also carries:
    - the maintenance counts in the window;
    - `keys_built` delta against `cache_size`;
    - `unique_ops`;
    - fallback pairs by reason;
    - the P2P fraction;
    - occupied depths;
    - the global particle count.
  - At the same state, run a standalone `BRSolverDirect` and a standalone
    `BRSolverFMM` (both constructed once, outside the loop, so the FMM one's
    cache behaves as the T6 probe's did) and emit one `[acc]` row: claim A's
    metric, the direct evaluation wall, the standalone FMM evaluation wall, and
    `3 ×` direct wall `+` (step wall `−` the step's three `ff_evaluate_total`)
    as the estimated per-step `direct` cost. The standalone evaluations run
    **outside** the snapshot window, so they never pollute `[cost]`.
- **At the end**, emit a `[cost-total]` trailer:
  - the action histogram;
  - each phase's share of `ff_evaluate_total`;
  - `ff_evaluate_total`'s share of step wall;
  - the operator-build share of `ff_solve`;
  - the git hash of `../canopy` if the batch script passes it in as an
    environment variable (`BEATNIK_CANOPY_HASH`, else `unknown`).
- **Sentinels:** with `CANOPY_ENABLE_PROFILING` undefined, every Canopy timer
  field prints `-1`. With `BEATNIK_PROFILING_LEVEL < 2`, every `ff_*` field
  prints `-1`.
- **Assertions:** none. The probe exits non-zero only when it cannot run: a
  non-finite state, a bad argument, or a failed cast.

**Do:**
1. Make all edits.
2. Edit the env and snapshot.
3. Install **once**.
4. Write the smoke script. It logs the Canopy hash and porcelain status and
   exports `BEATNIK_CANOPY_HASH`. It runs `Beatnik_Probe_FmmCost_MPI_HIP` at
   level 4 for 60 steps with `sample_every 20`, then repeats with
   `canopy_report=1` to measure the reporting overhead. It then runs one
   10-step sizing launch each at levels 5 and 6 (`sample_every 5`), at np1 and
   np4. A budget guard covers the whole matrix, on `pdebug`.
5. In the same sitting, submit T1's Canopy job if it has not already run.
   Record the sizing launches' s/step: T3's walltimes come from them.

**Exit criterion:** the smoke job's log satisfies all of the following:
- Every `[cost]` row has non-sentinel values in every field.
- The ten `ff_*` phases sum to `ff_evaluate_total` within 5 % at every row.
- `ff_solve` is within 5 % of Canopy's `upward_total + downward_total + p2p_total`.
- Each row's maintenance counts sum to 3 × the steps in the window.
- `[acc]` metric ≤ 1.5e-3 at every level-4 row.
- **Zero** `[Canopy Diagnostics]` lines in the `canopy_report=0` launches.

The log records the `canopy_report=1`/`=0` step-wall ratio and the level-5/6
sizing rates.

Failure direction: launching the probe with `bogus=1` exits non-zero with a
message naming `bogus`, and the `canopy_report=1` launch's log *does* contain
`[Canopy Diagnostics]` lines.

**Met.** Smoke job `f3diM2PdyufM` (`scripts/tuolumne/canopy_profiling_smoke.flux`,
`pdebug`, Canopy `c182348` clean):
- **The `[cost]` rows.** All 20 rows have no sentinel. The ten phases sum to
  `ff_evaluate_total` within 0.28 %, and `ff_solve` is within 0.24 % of
  Canopy's three totals.
- **Maintenance counts** equal 3 × steps in every row.
- **Level-4 `[acc]`** is 5.0e-4 to 5.1e-4.
- **Canopy lines.** The five `canopy_report=0` launches print no Canopy
  diagnostic line.
- **Failure direction.** `bogus=1` exits 1, naming `bogus`, and the
  `canopy_report=1` launch printed 1290 `[Canopy Diagnostics]` lines.

The log records the reporting ratio (1.02) and the L5/L6 sizing rates
(0.83/0.69 s/step at L5, 2.9/1.8 s/step at L6, np1/np4). Took two
`spack install`s, not one: see the log.

---

### T3 — The baseline cost model, the crossover, and the SMS — **DONE**

**Depends on:** T0, T2.

**Fill in:**
- `scripts/tuolumne/canopy_profiling_baseline.flux` (new). The HIP matrix may
  be split into `_np1` and `_np4` scripts submitted together.
- `tests/regression_tests/fmm_profile_reduce.py`: the per-row
  evaluation-count check T0 deferred to T3 (maintenance counts = `evals` =
  3 × `steps`; a failing row is reported and excluded), with `--self-test`
  fixtures for it.
- The progress log.
- This document's [Phase O](#phase-o--optimize-single-node) ordering, and an
  **SMS definition** subsection appended under T3 below.
- README Known Issues: update the "FMM per-step cost grows" entry's pointer to
  this document.

**Do:**
1. **The matrix, in one or two `pdebug` jobs:**
   - HIP np1 and np4 at level 4 to step 2000, with `sample_every 50`. This is
     the trajectory-growth measurement and the full 81-state equivalent.
   - HIP np1 and np4 at level 5 and level 6 to the step count T2's sizing says
     fits, with `sample_every` set to give at least 10 rows.
   - An `ncrit` arm {8, 32, 64} at level 5, HIP np4, to the same step count.
     This shows whether the crossover depends on leaf size. It informs V1's
     production-`ncrit` statement and does **not** change the measurement
     `ncrit`, which stays 8 ([Deliberate deviations](#deliberate-deviations)).
     The `ncrit 64` point at L5 np4 already existed at early states (log, "L5/L6
     accuracy check": 0.330 s/step, `[acc]` 5.4e-4); the arm extends it along
     the trajectory.
   - SERIAL np1 and np4 at level 4 to step 500.

   If level 4 HIP np1 to step 2000 does not fit `pdebug`'s 60 minutes with the
   rest, it gets its own `pdebug` job. Do not move it to `pbatch`, and do not
   shorten it silently: the guard skips loudly and the log says what was
   skipped.
2. **Reduce with `fmm_profile_reduce.py`.** Record:
   - **(a)** the phase split of one evaluation and of one step, per (level, np,
     backend), at early, middle and late states;
   - **(b)** the action histogram;
   - **(c)** the operator-build share of `ff_solve`, and retention
     (`keys_built` delta against `cache_size`) on Beatnik's tree;
   - **(d)** `fmm` against estimated `direct` per step with the P2P fraction at
     each level, and the crossover vertex count by interpolation across levels
     4–6, or the statement that there is none in range;
   - **(e)** the share outside `solve()` (adapter phases and `ff_diagnostics`);
   - **(f)** the `[acc]` metric at every state. **This is the accuracy baseline
     every O task compares against.**
3. **Rank the optimization candidates by measured share.** Reorder the O tasks
   and rewrite their **Depends on** lines to match. Delete any O task whose
   target is measured below 5 % of step wall at every level, with the number
   cited in the log. Add an O task for any phase above 10 % that no O task
   covers.
4. **Define the SMS:** the subset of this matrix every O task re-runs. It must
   fit **one `pdebug` job per backend-np pair**, include level 4 to a state
   past the occupied-depth-7 onset (step ≥ 1000), and include level 5. Write it
   below as exact script arguments.
5. Do not change any code in Canopy or the adapter in this task. Findings
   become O tasks.

**Exit criterion:** the log carries (a)–(f) at stated levels, ranks, backends,
basis and order. This document has an SMS subsection with exact arguments and an
O sequence whose order cites the measured shares. Failure direction: the
reducer's per-row evaluation-count check (maintenance counts = 3 × steps) passes
on every row it reduced. A row that fails it is reported and excluded by name,
never silently averaged in.

**Met.** Nine `pdebug` jobs on Beatnik `d45aeed` (the installed binary is
`86c32ac`'s; nothing compiled changed since) and Canopy `c182348`, clean, all
rc 0 with no row skipped by the budget guard ([log](canopy_profiling-progress-log.md)
`## T3` names them). The log carries (a)–(f) for HIP np1 and np4 at levels 4
(2000 steps), 5 (800) and 6 (250), SERIAL np1 and np4 at level 4 (500), and the
`ncrit` 32/64 arm at level 5, all CartesianTaylor order 3, θ 0.3, `ncrit 8`
unless named. The reducer's count check passed on all 328 `[cost]` rows of
the nine logs; none was excluded. Its `--self-test` carries a passing and a
failing fixture, and a copy with the check neutered fails that self-test. The
run-to-run spread is measured at level 4 and level 5 (HIP np4, the SMS rows
run twice in one job and once more on another node). `[acc]` is bit-identical
across repetitions, nodes, rank counts and backends, so the spread is in step
wall only. The Phase O sequence below is reordered from the measured shares;
the SMS is defined next.

#### SMS definition

The **standard measurement set** is four `pdebug` jobs, one per backend-np
pair, each one set of `scripts/tuolumne/canopy_profiling_baseline.flux`:

```
BEATNIK_CPB_SET=sms_hip_np1    flux batch --job-name=beatnik_cpb_sms_hip_np1    scripts/tuolumne/canopy_profiling_baseline.flux
BEATNIK_CPB_SET=sms_hip_np4    flux batch --job-name=beatnik_cpb_sms_hip_np4    scripts/tuolumne/canopy_profiling_baseline.flux
BEATNIK_CPB_SET=sms_serial_np1 flux batch --job-name=beatnik_cpb_sms_serial_np1 scripts/tuolumne/canopy_profiling_baseline.flux
BEATNIK_CPB_SET=sms_serial_np4 flux batch --job-name=beatnik_cpb_sms_serial_np4 scripts/tuolumne/canopy_profiling_baseline.flux
```

Each set runs `Beatnik_Probe_FmmCost_MPI_<SPACE>` at the probe's defaults
(`ncrit 8`, order 3, θ 0.3, `m2l_op_count_cap 65536`, `canopy_report 0`),
with the binding of `launch()` in that script. Probe arguments are
`<level> <steps> <sample_every> run=<label>`:

| set | runs (probe arguments) | last state | T3 wall | baseline rows |
| --- | --- | --- | --- | --- |
| `sms_hip_np1` | `5 300 25 run=smsl5_L5_HIP_np1`; `4 1200 50 run=smsl4_L4_HIP_np1` | L5 t = 0.80, L4 t = 1.65 (occupied depth 7 from step 700) | 384 + 1038 s | job `f3djNiFw5apw` |
| `sms_hip_np4` | `5 300 25 run=smsl5_L5_HIP_np4`; `4 1200 50 run=smsl4_L4_HIP_np4` | as above | 297 + 744 s | `smsl{5,4}rep{0,1}` in job `f3dj4RSg5NSs` |
| `sms_serial_np1` | `4 500 25 run=smsl4_L4_SERIAL_np1` | L4 t = 1.15 | 885 s | `l4_L4_SERIAL_np1`, job `f3dj4RASfEVu` |
| `sms_serial_np4` | `4 500 25 run=smsl4_L4_SERIAL_np4` | L4 t = 1.15 | 1931 s | `l4_L4_SERIAL_np4`, job `f3dj4RJLmQaT` |

- **Why these rows.** Level 4 to step 1200 passes the occupied-depth-7 onset
  (step 700 at np1, 800 at np4) and the plateau of the per-step cost (R3).
  Level 5 to step 300 gives 13 states. Level 6 is not in the SMS: its spread
  is unmeasured, and 250 steps cost 891 s at np1. A task that changes the tree
  (O1, O2) adds T3's level-6 rows (`hip_np4_l5l6`'s `l6`) to its job.
- **SERIAL** runs only level 4, to step 500, the T3 rows unchanged, so they are
  their own baseline. SERIAL np4 fits one copy per job; a task needing an
  on/off control on SERIAL np4 submits two jobs.
- **Controls fit in one job on HIP.** Two copies of `sms_hip_np4` are about
  2.1 ks and two of `sms_hip_np1` about 2.85 ks, against the 3.3 ks budget.
  Pass the control's rows through `BEATNIK_CPS_SWEEP` with a knob column.
- **Compare by state.** Read the jobs with `fmm_profile_reduce.py --format
  probe --states`. The `[acc]` baseline at every SMS state is the T3 value in
  the log's (f); it is identical at np1 and np4 and on SERIAL and HIP.
- **The gain threshold (R2)** is twice T3's spread: a run-level step-wall gain
  under 1.7 % at level 5 or 0.6 % at level 4 is not claimed, and a per-state
  gain under 9 % (level 4) or 14 % (level 5) is not claimed state by state.
- **The accuracy spread is zero.** `[acc]` repeated to all nine printed digits
  at every state. The guard is therefore exact and aggregate: per level, the
  worst `[acc]` over the SMS states must not rise. A task that changes the tree
  (O1, O2) may move individual states either way. This is the user's ruling
  after T3 (log, "Decision: no accuracy regression in any O task"). T3's
  per-level maxima: L4 1.099e-3 (step 1000), L5 1.634e-3 (step 1).
- **After O1 the baseline is O1's, not T3's.** O1 raised level 4's maximum
  and was adopted by ruling (log, "Decision: adopt O1 despite the level-4
  `[acc]` rise"); the guard itself is unchanged for every later task. O1's
  maxima: L4 1.402e-3 (step 800; 1.134e-3 over SERIAL's 500 steps), L5
  1.212e-3 (step 75), L6 2.248e-3 (step 250).

---

## Phase O — optimize (single node)

**Every O task:**
1. Implement the change in `../canopy`, the adapter, or both, and commit in
   `../canopy` first.
2. Compile-check and run the named Canopy stems in `build-profiling/`.
3. Install Beatnik once.
4. Submit the Canopy test job and the SMS job(s) together.
5. Reduce.

**Every O task's exit criterion includes all of the following:**
- The SMS's targeted phase time and its step wall, before (T3 or the previous O
  task) and after, recorded at every SMS state.
- The `[acc]` metric does not regress in aggregate at any level: the maximum
  over that level's SMS states after the task ≤ before, exactly
  ([Conventions](#conventions)). Its change against the previous task and
  against T3 is recorded per state. A move over 20 % at any
  state triggers V1 before the next O task.
- The task's named Canopy stems passing at SERIAL np 1–6 and HIP np 1–4.

**Failure direction for every O task:** an SMS run with the change disabled (its
knob off, or the previous Canopy commit) reproduces the previous numbers within
the run-to-run spread recorded in T3. A gain that does not vanish when the
change is off is not attributed to the change.

**The order below is T3's, from measured shares** ([log](canopy_profiling-progress-log.md)
`## T3`). Each cell is the target's share of step wall, the median over every
`[cost]` row of the full T3 trajectory (L4 2000 steps, L5 800, L6 250), HIP,
`ncrit 8`:

| target | task | L4 np1 | L4 np4 | L5 np1 | L5 np4 | L6 np1 | L6 np4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| operator-table build, `ilist_s4_op_table_build` (retention ≈ 0 at every row) | O1, then O2/O4 | 53.8 | 39.9 | 41.6 | 35.9 | 24.6 | 30.1 |
| maintenance, `ff_maintain` | O5 | 9.8 | 11.8 | 19.6 | 19.9 | 34.2 | 29.5 |
| operator-table assembly, `ilist_s4_op_table_copy` | O8 (new) | 18.0 | 13.4 | 14.5 | 13.0 | 9.0 | 11.5 |
| pair classification, `ilist_s3_classify_pairs` | O9 (new) | 9.8 | 4.3 | 12.1 | 8.1 | 16.5 | 9.4 |
| downward outside the ilist, M2L and L2P (exchange, per-depth loop) | O10 (new) | 0.8 | 11.1 | 0.8 | 10.2 | 0.7 | 6.3 |
| adapter phases + `ff_diagnostics` | O6 | 1.0 | 3.4 | 1.5 | 2.8 | 2.6 | 2.8 |
| `m2l_kernel`, which contains the range-guard fallback | O3 | 2.0 | 1.5 | 3.4 | 1.9 | 3.9 | 2.4 |

On SERIAL at level 4 the adapter's forward and reverse distribute are 52 % of
the step at np1 and 87 % at np4 (O7).

Sequence: **O1 → O2 → O4 → O5 → O8 → O9 → O10 → O6 → O7 → V1.** O2, O4 and O6
are conditional; O3 is NOT TAKEN.

**After O1 the sequence is O8 → O5 → O9 → O10 → O6 → O7 → V1.** O2 and O4
are NOT TAKEN. O8 moved ahead of O5 by the user's choice, on O1's post-retention
shares (log, "Decision: O2 and O4 not taken; O8 before O5"). The table above is
T3's and predates O1.

### O1 — Turn on `quantize_root_half_width` from Beatnik — **DONE**

**Depends on:** T3.

**Fill in:**
- `src/Beatnik_Params.hpp`: `FmmParams::quantize_root_half_width`, beside
  `m2l_op_count_cap` (`:433`), with a doc comment. The default is decided by
  step 3.
- `src/Beatnik_FarFieldInterface.hpp`:
  - Map the knob in `buildConfig` (`:1124`). Its only caller is `evaluate`
    (`:1393`).
  - Add two cumulative, always-on counters to `FarFieldDiagnostics` (`:215`).
    Like `maintenance_counts`, they are incremented at the end of a completed
    `evaluate` (`:1681`), rank-local.
    - `root_width_changes`: the evaluations at which the downward sweep's root
      half-width differs from the previous evaluation's. `readDiagnostics`
      (`:943`) reads it through `down.root_half_width()`
      (`../canopy/src/Canopy_DownwardSweep.hpp:463`). Every width change
      empties the cache (`set_root_half_width`, `:451-460`). With the knob on,
      a change is an octave crossing.
    - `op_cache_overflow_clears`: the evaluations without a width change at
      which the cache was emptied anyway. That is the column-cap overflow at
      `:2085-2091` (or the cap check at `:2066`), detected exactly. Without a
      clear, `cache_size` after an evaluation equals `cache_size` before it
      plus the keys built in it; after a clear it does not.
- `tests/regression_tests/Beatnik_Probe_FmmCost.cpp`:
  - the key `quantize_root_half_width`;
  - two `[cost]` fields, the window deltas of the two counters
    (`width_changes`, `overflow_clears`, mean and max over ranks like the
    other rank-local fields).
- README "FMM tunables" (`README.md:306`).

**Reference:** `../canopy/src/Canopy_Solver.hpp:118-126`; B2 in
`../canopy/tasks/tree-opt.md` (`:1497`) and its progress log `§B2`, including
its failure-direction check for a column used at the wrong width (B2's R4).

**Retention, as measured here.** Per `[cost]` window:

$$
r = 1 - \frac{\text{keys\_built\_delta}}{\text{evals} \times \text{unique\_ops}}
$$

T3's knob-off baseline is $r \approx 0$ at every window.

**The column cap stays at the SMS's 65536.**
- The effective cap is the smaller of the byte budget's column count and
  `m2l_op_count_cap` (`Canopy_DownwardSweep.hpp:368-374`). Beatnik's 2 GiB
  budget is about 670 000 columns, so the count cap binds.
- A cache that persists across trees accumulates keys. One tree holds up to
  42 000 unique operators at L5 np1 and 47 000 at L6 np1 (T3), so overflow
  clears are expected at level 5.
- They are counted, not avoided: raising the cap would move the SMS off T3's
  configuration.

**Do:**
1. No Canopy change.
2. In each SMS job, run the SMS with the knob on and, as the control, off.
   - The HIP pair fits one job per np (about 2.1 ks at np4, 2.85 ks at np1,
     of 3.3 ks).
   - SERIAL np4 needs a second job.
3. Submit `scripts/tuolumne/t4_fmm_vs_direct.flux` as its own job in the same
   sitting. It needs two nodes for ranks 5 and 6, and the SMS jobs are one
   node each.
4. If the gain is real and `[acc]` holds, set the default to `true`.
   - The quantized root can add one level at the top of the tree
     (`Canopy_Solver.hpp:123-125`).
   - `max_depth 10` is a cap, so check `occupied_depths` against it and
     confirm no `depth_dropped` fallback appears.
5. Record per window:
   - $r$;
   - `width_changes` and `overflow_clears`;
   - the change in `ilist_s4_op_table_build`.

   Split the rebuilt keys into three parts:
   - octave crossings: one full rebuild (`unique_ops`) per width change;
   - overflow clears: one full rebuild per clear;
   - the remainder: new keys in a surviving cache.

   Give the retention and step wall the overflow clears cost.

**Exit criterion:** the common O criteria, plus `FmmVsDirect` passing at
ranks 1–6 on SERIAL and HIP (`t4_fmm_vs_direct.flux`).

**Retention.** With the knob on, in every SMS window without an overflow clear,
retention exceeds 90 % after the crossings are discounted:

$$
1 - \frac{\text{keys\_built\_delta} - \text{width\_changes} \times \text{unique\_ops}}{\text{evals} \times \text{unique\_ops}} > 0.9
$$

Windows with overflow clears are exempt from the 90 %. Their count and cost are
recorded, and the log's `**Affects:**` line names the follow-up (a larger cap,
or eviction instead of a full clear) when they occur.

**Failure direction.**
- The knob-off control reproduces T3's rebuild-every-evaluation pattern:
  `keys_built_delta / evals` within 0.96–1.00 of `unique_ops` as a per-run
  median, and exactly 2/3 of it at step 1.
- Per-row values range over 0.65–1.23 only because `unique_ops` is the
  window's last value.
- In every knob-off window after step 1, `width_changes` is at least
  0.9 × `evals` and `overflow_clears` is 0. The width moves at nearly every
  build, and a cache emptied that often never reaches the cap.

**Met.** Verified 2026-10-12/13, Beatnik `58937d2` installed, Canopy
`c182348`; numbers and job IDs in the [log](canopy_profiling-progress-log.md)
`## O1`.
- **Retention, knob on.** Discounted $r \ge 0.993$ in every SMS window after
  step 1, at every level, backend and rank count. There were **no** octave
  crossings and **no** overflow clears anywhere (L5 np1's cache peaked at
  24 528 keys of 65 536), so no window was exempt. The operator build fell
  from 78–549 s to 0.2–0.3 s per run.
- **Step 1 by interpretation.** The step-1 window reads $r = 0.66$, because
  the first evaluation's cold fill sits inside it and the counter, as
  specified, never counts the first evaluation (Canopy's
  `set_root_half_width` does treat 0 → w as a change). Counting that fill as
  one full build gives $r = 1.000$. The user ruled to keep the first
  evaluation uncounted, so the 90 % criterion is read as excluding the cold
  fill.
- **Failure direction.** The knob-off arm has `[acc]` bit-identical to T3 at
  every state, per-run `kb/ev/uo` medians of 0.955–1.000, exactly 0.667 at
  step 1, and `width_changes` ≥ 0.96 × evals with zero overflow clears after
  step 1. **The step wall reproduces T3 only approximately.** On HIP the
  per-state change is within T3's per-state spread (9 % L4, 14 % L5) at
  every state but L4's step 1 (+12.5 % np1, −13.4 % np4). Run-level, the
  summed sampled step wall is −3.5 % (L4 np4), +1.5 % (L4 np1), −1.6 / +1.4 %
  (L5 np4 / np1) and −0.4 % (L6). L4 np4 is therefore outside T3's 0.6 %
  run-level spread, in the faster direction. SERIAL np4 is −0.9 % run-level
  (per state −10.8 to +6.2 %). **SERIAL np1 is +19.2 %** (13–35 % per state),
  on a different node (`tuolumne1015` against `1016`); T3 measured no SERIAL
  spread. None of this is near the size of the gain, which is therefore
  quoted only from each on/off pair run inside one job.
- **Gain.** The summed SMS step wall falls 53.5 % at L4 HIP np1, 47.5 % at L4
  HIP np4, 22.1 % at L5 np1, 36.1 % at L5 np4, 20.6 % at L6 np4 and 21.4 % at
  L4 SERIAL np1, all far above twice the spread. L4 SERIAL np4's 7.4 % is a
  two-job comparison. Per state the gain grows along the trajectory, to −50 %
  and more late. Step 1 is up to 18 % slower: the cold fill plus one extra
  level.
- **Depth (R4).** Knob on adds one occupied level at the top: L4 starts at 6
  rather than 5, L5 at 7 rather than 6, and L6 is 8 throughout. `max_depth`
  10 never binds, and `fb_dropped` is 0 in every row.
- **Accuracy: the guard FAILS at level 4, and O1 is adopted by ruling.** Per
  level, max knob-on `[acc]` against the knob-off arm (= T3): L4 **1.402e-3
  against 1.099e-3** (SERIAL's 500 steps: 1.134e-3 against 8.13e-4); L5
  1.212e-3 against 1.634e-3; L6 2.248e-3 against 2.771e-3. Knob-on `[acc]`
  is bit-identical across backends and rank counts. It is geometry (R4), not
  R5: the rise is already present at step 1, before any column is reused.
  The user ruled that the speed is worth the level-4 rise, which stays under
  τ_A, and adopted the default `true` (log, "Decision: adopt O1 despite the
  level-4 `[acc]` rise"). Moves over 20 % occur at every level (L4 −42 to
  +130 %, L5 −39 to +18 %, L6 −20.5 to +41.3 %), so **V1 is triggered** before
  the next O task.
- **FmmVsDirect: PASS at ranks 1–6 on HIP** (job `f3dk9yN1mqFM`, installed
  default `true`). **Gap:** the binary instantiates only
  `Kokkos::DefaultExecutionSpace`
  (`tests/unit_tests/Beatnik_Test_FmmVsDirect.cpp:1193-1195`), so there is
  no SERIAL FmmVsDirect to run. SERIAL's evidence is the SERIAL probe's
  `[acc]` rows, identical to HIP's.

### O2 — Keep the operator cache across an octave crossing — **NOT TAKEN (O1)**

**Not taken.** O1 measured **zero** octave crossings (`width_changes` 0 in
every knob-on `[cost]` row): 1200 steps at level 4, 300 at level 5 and 250
at level 6, on HIP np1 and np4 and SERIAL np1 and np4. Crossing rebuilds cost
0 % of SMS step wall against the 5 % trigger (log `## O1`). The design below
is kept for a future run that does cross.

**Depends on:** O1 **DONE**.

**Trigger:** O1's log shows octave-crossing rebuilds costing more than 5 % of
SMS step wall. Otherwise mark this task NOT TAKEN with the number.

**Fill in:** `../canopy/src/Canopy_DownwardSweep.hpp`, in `set_root_half_width`
(`:451-460`) and the cache's key storage.

**Reference:** `../canopy/README.md`, "Keep the M2L operator cache across an
octave crossing of the root width". The column at depth $d$ and root width $W$
equals the column at depth $d+1$ and width $2W$, so shift each cached key's
`max_d` by the change in the quantized exponent. Keys shifted past depth 0 or
`max_depth` are dropped.

**Exit criterion:** the common O criteria, with Canopy stems `DownwardSweep`
and `CartesianTaylorSolve`, plus B2's failure-direction check re-run with a
deliberately wrong shift, which must fail.

### O4 — Parallelize the host operator-table build — **NOT TAKEN (O1)**

**Not taken.** After O1, `ilist_s4_op_table_build` sums to 0.2–0.3 s per SMS
run. That is 0.02–0.11 % of summed step wall (largest at L5 HIP np4,
0.2 s of 179 s), against the 5 % trigger (log `## O1`).

**Depends on:** O1 **DONE**.

**Trigger:** after O1, `ilist_s4_op_table_build` is still more than 5 % of SMS
step wall.

**Fill in:** `../canopy/src/Canopy_CartesianTaylorBasis.hpp:1186-1240`, a
serial host loop over missing keys whose columns are independent.

**Exit criterion:** the common O criteria, with Canopy stems
`CartesianTaylor` (non-MPI) and `CartesianTaylorSolve`. The built columns must
be bit-identical to the serial build's: assert it in a unit case that builds
one key set both ways.

### O5 — Cheaper maintenance between RK stages — **DONE**

**Depends on:** T3, O1 **DONE** (O1 changes what maintenance rebuilds) and
O8 **DONE** (O8's knob-on SMS is the baseline).

**Fill in:**
- `../canopy/src/Canopy_Solver.hpp`: `auto_maintain` (`:453-557`) and the
  paths it selects, `_finish_topology_change` (`:676-711`) and
  `_finish_topology_stable` (`:718-738`).
- `../canopy/src/Canopy_CommunicationPlan.hpp`: `build_all_interaction_lists`
  (`:557`), a serial host dual-tree traversal of the replicated global cell
  list on every rank, called from `build` under `TIMER_COMMPLAN_DTT_TRAVERSAL`
  (`:994-996`).
- `../canopy/src/Canopy_Profiling.hpp`: the new maintenance timer keys.
- The knob: a Canopy `FmmConfig` field, a Beatnik `FmmParams` field mapped in
  `buildConfig`, and a probe key, as O8's `device_op_table_assembly`; README
  "FMM tunables" and Canopy README's `FmmConfig` table.
- `tests/regression_tests/Beatnik_Probe_FmmCost.cpp`: `kCanopyKeys`
  (`:181-205`).

**Baseline (after O8).** Medians of `3 × phase / step_wall` over O8's knob-on
rows, HIP (jobs `f3dvNa7ovPnF` np1, `f3dvNaGqG2Fy` np4; log `## O8`):

| target | L4 np1 | L4 np4 | L5 np1 | L5 np4 |
| --- | --- | --- | --- | --- |
| `ff_maintain` | 38.0 | 39.0 | 44.4 | 46.7 |
| `comm_plan_build` | 20.8 | 16.4 | 34.3 | 30.0 |
| `commplan_dtt_traversal` | 12.8 | 13.6 | 23.3 | 24.1 |
| untimed maintenance | 10.8 | 11.2 | 7.3 | 8.4 |
| `builder_build` | 5.7 | 6.7 | 1.4 | 3.7 |
| `repartition` | 0.8 | 3.4 | 1.3 | 4.6 |

- *Untimed maintenance* is `ff_maintain − builder_build − repartition −
  sort_by_leaf − comm_plan_build`. It is `auto_maintain`'s old-key snapshot,
  comparison `_builder.build` and symmetric difference (`:487-511`), and the
  Migrate path's `_partitioner.redistribute` (`:539-541`).
- `comm_plan_build` outside the DTT is 6–11 % of the step. The probe cannot
  split it: `kCanopyKeys` prints only `commplan_dtt_traversal` of the six
  comm-plan sub-timers. `commplan_cell_map_fill`, `commplan_vertical_plans`,
  `commplan_subtree_relevance`, `commplan_finalize_m2l` and
  `commplan_finalize_p2p` (`Canopy_Profiling.hpp:128-133`, timed at
  `Canopy_CommunicationPlan.hpp:967-1008`) are missing.
- `auto_maintain` chooses Rebalance on 77 % of evaluations at level 4 and
  98 % at levels 5 and 6; Migrate takes the rest, and Rebuild never fires
  (T3). Rebalance fires whenever `k_changed > 0`, even for one cell
  (`:513-535`), and then runs repartition, a full comm-plan build and the DTT.
- Accuracy baseline, O8's maxima: L4 1.4025e-3, SERIAL's 500 steps
  1.1338e-3, L5 1.2115e-3. SERIAL's "before" is O8's `f3dvNaQDiShd` (np1)
  and `f3dvNaYRcUAK` (np4).

**Do:**
1. Time the untimed maintenance: add Canopy timers around the snapshot,
   comparison build and symmetric difference, and around `redistribute`. Add
   those keys and the five missing `commplan_*` keys to `kCanopyKeys`. This
   goes in the same single install as step 2.
2. Make the Rebalance path cheaper, chiefly `comm_plan_build` and its DTT,
   and consider alongside it a change to which `MaintenanceAction`
   `auto_maintain` chooses (for example a cheaper path when `k_changed` is
   small). A policy change is in scope. It changes what the milestone tier
   exercises, which V1 re-runs.
3. Put the change behind the knob, default on, so the off arm runs in the
   same job and on the same node.

- **A gain from a single-rank short-circuit alone is not claimed.** The DTT
  is 13–23 % of the step at np1, where there is nothing to communicate, but
  production is multi-rank.
- **Repartition is out of scope.** It is 1–5 % of the step.

**Exit criterion:** the common O criteria, with Canopy stems `TreeBuilder`,
`TreePartitioner`, `CommunicationPlan` and `MultiSolve`. In addition:
- every new timer key prints a non-sentinel value in every `[cost]` row;
- untimed maintenance, recomputed with the new timers counted as timed, is
  under 2 % of step wall, or the log names what the remainder is.

**Met.** Verified 2026-10-15, Canopy `c9ff5dc` (0 porcelain lines in every
job), Beatnik `c22d346` installed. Numbers and job IDs are in the
[log](canopy_profiling-progress-log.md) `## O5`.
- **The change: a cheaper Rebalance path, not a policy change.**
  `CommunicationPlan::build` gains an indexed build that produces the
  identical plan. The DTT, the vertical plans and the subtree relevance read
  children, parent and owner from per-cell index arrays. The DTT appends to
  flat buffers grouped by target with a counting sort, and first-emission
  marks admit each send, receive and ghost once. Off is the hash-driven
  build, unchanged.
  - **No `MaintenanceAction` change.** Any key change invalidates the
    cell-keyed comm plan, so no policy can skip the rebuild without keeping
    a stale tree. A lazy tree would move `[acc]`, and repartition alone is
    out of scope. The milestone tier therefore exercises the same actions.
- **The knob.** `FmmConfig::indexed_comm_plan` and
  `FmmParams::indexed_comm_plan`, both default `true`; probe key
  `indexed_comm_plan`.
- **Two installs, a departure.** v1 buffered every repeat emission: 452 276
  receives for 1 660 unique on one rank of the np 2 fixture. Its sort would
  have erased the multi-rank gain. v2 added the marks, which needed a second
  install; no SMS job had run on v1.
- **Canopy stems: PASS.** `TreeBuilder`, `TreePartitioner`,
  `CommunicationPlan` and `MultiSolve` at SERIAL np 1–6 and HIP np 1–4 (job
  `f3eHuZoQbRcs`). `CommunicationPlan.testIndexedMatchesLegacy{Deep,Basic}`
  compare every public plan field between the two builds and are `[ OK ]` on
  every rank. Two broken builds fail as predicted: index tables missing
  octant-7 children, and an unsorted receive buffer (passes at np 1, fails
  at np 2).
- **New timers: non-sentinel in every `[cost]` row** (12 keys × 236 rows).
  - Untimed maintenance falls to a median 0.0–0.4 % of the step (7–11 % at
    O8), and ≤ 0.4 % at every state after step 1.
  - Step 1's 0.8–4.4 % is the one-time Setup's `partition`, which the
    reducer does not subtract.
  - The old remainder was mostly `maint_compare_build` (2.4–9.1 %) and
    `maint_sweep_setup` (4.4–6.5 %).
- **Gain (same-job pairs).** The DTT falls 63–84 % and `comm_plan_build`
  61–66 % at np4 (23–25 % at np1, where the indexed finalizers absorb part of
  the saving).
  - Summed SMS step wall: **−19.2 %** at L5 HIP np4 (the headline), −9.3 %
    at L5 np1, −4.3 % at L4 np1 and −1.3 % at L4 np4.
  - At L4 np4, phases the knob does not touch moved 13–25 % between the
    arms. That noise hides most of the 15.8 s `ff_maintain` saving.
  - SERIAL is not claimed: the saving is under 1 % of its step.
- **Accuracy: bit-identical.** `[acc]` equals O8's knob-on value at every
  state of every run, in both arms. The maxima are unchanged: L4 1.4025e-3,
  SERIAL 500-step 1.1338e-3, L5 1.2115e-3.
- **Failure direction.** The off arm has `[acc]` identical to O8, and its
  `comm_plan_build`, DTT and `ff_maintain` are within 3 % of O8 per run.
  - Its step wall is within T3's per-state spread at every HIP state after
    step 1 (L4 −7.0 to +6.2 %, L5 −2.8 to +1.3 %).
  - Run level: L5 +0.2 / −1.0 % (np1 / np4), within 1.7 %. L4 +3.9 / −4.0 %,
    outside 0.6 % in both directions, on different nodes from O8's, as O1's
    and O8's own controls were.

### O8 — Assemble the operator table without a full host copy and upload — **DONE**

**Depends on:** O1 **DONE** and O4 resolved. Both change stage 4's cache.

**Measured (T3, history):** `ilist_s4_op_table_copy` was 9.0–18.0 % of HIP
step wall (medians per run), 13–18 % at levels 4 and 5. O1's retention does
not shrink it: stage 4b runs for every unique operator, retained or not.

**Baseline (after O1):** 43.2 / 28.4 % of HIP step wall at L4 np1 / np4,
19.8 / 20.8 % at L5, 17.8 % at L6 np4 (medians of `3 × phase / step_wall`
over O1's knob-on rows; log, "Decision: O2 and O4 not taken; O8 before O5").
O1 shortened the step, so the same copy is a larger share of it.

**Fill in:** `../canopy/src/Canopy_DownwardSweep.hpp`, stage 4b
(`:2146-2172` at Canopy `c182348`). On every interaction-list build it:
- copies each of `n_unique_ops` columns from the host cache into a host
  mirror, element by element;
- then `deep_copy`s the whole table to the device.

Candidates:
- keep the cache device-resident and gather columns there by index;
- or upload only the newly built columns and gather on the device.

**Exit criterion:** the common O criteria, with Canopy stems `DownwardSweep`
and `CartesianTaylorSolve`, plus a unit case asserting the assembled table is
bitwise equal to the host path's. That is the invariant the comment at
`:2158-2161` (Canopy `c182348`) states.

**Met.** Verified 2026-10-13, Canopy `046fdc6` (jobs ran HEAD `56f3932`,
which adds only the test script, 0 porcelain lines), Beatnik `40fc8c7`
installed. Numbers and job IDs are in the [log](canopy_profiling-progress-log.md)
`## O8`.
- **The change.** The operator cache keeps a device copy, grown and dropped
  with the host cache. Each build uploads only the columns built since the
  last one, and the table is gathered on the device by
  `_m2l_op_cache_index`. A host backend gathers from the host cache and keeps
  no copy.
- **The knob.** `FmmConfig::device_op_table_assembly` and
  `FmmParams::device_op_table_assembly`, both default `true`; probe key
  `device_op_table_assembly`. Off is the old path, unchanged.
- **Canopy stems: PASS.** `DownwardSweep` and `CartesianTaylorSolve` at
  SERIAL np 1–6 and HIP np 1–4 (job `f3dvCN5tgncw`).
- **Bitwise unit case: PASS** on every rank of every entry.
  `DownwardSweepCaching.deviceOpTableAssemblyBitwise` builds one tree with
  the knob off and on and `memcmp`s every column, matched by key, after:
  - a first fill;
  - a growth through `_grow_m2l_op_cache`;
  - a no-new-key rebuild;
  - a forced `clear_m2l_op_cache()`;
  - the cap-overflow clear.
- **Stale-column check, by hand, not committed.** Both broken HIP binaries
  fail the case at np 1 and 2.
  - A device copy whose growth drops its prefix fails the **bitwise
    comparison**.
  - A clear that skips the device copy refills identically on this fixture,
    so it compares equal after the clear itself. It fails one step later as
    a hard HIP copy error, when the next refill is smaller.

  Out-of-step extents also throw by design.
- **Gain (same-job pairs).** The summed SMS step wall falls 49.0 % (L4 HIP
  np1), 38.0 % (L4 np4), 22.2 % (L5 np1) and 30.5 % (L5 np4). Every HIP state
  after step 1 is past the per-state threshold.
  `ilist_s4_op_table_copy` falls 91–98 % per run, to 0.7–2.2 % of the step.
  `ilist_s5_device_upload` did not grow.
- **Accuracy: bit-identical.** `[acc]` equals O1's knob-on value at every
  state of every run, in both arms. The maxima are unchanged: L4 1.4025e-3,
  SERIAL 500-step 1.1338e-3, L5 1.2115e-3.
- **Failure direction.** The knob-off arm has `[acc]` identical to O1 and
  `s4 copy` within ±3 % of O1 per run.
  - Its step wall is within T3's per-state spread at every HIP state after
    step 1 (L4 −5.6 to +4.0 %, L5 −4.1 to +3.5 %).
  - Run level: L5 −0.5 / +1.3 % (np1 / np4), within 1.7 %. L4 +1.3 / −1.8 %,
    outside 0.6 % in both directions, on different nodes from O1's, as O1's
    own control was.
  - Step 1 at np4 is the usual outlier (−29 % L4, −15 % L5).
- **SERIAL** stays correct, and its step wall is unchanged within noise
  (np1 −4.9 % in one job; np4 +7.6 % across two jobs and nodes, not
  claimed).
  - **Open: the host gather is 3–5 % slower than the old host loop** in
    `ilist_s4_op_table_copy` at SERIAL np1 (+11 % summed with outliers,
    about 0.45 % of the step), and flat at np4.
  - A column-per-iteration functor on host backends would recover it; that
    is left out of O8, which allows one install.

### O9 — Cheaper pair classification in the interaction-list build — **NOT STARTED**

**Depends on:** O5 **DONE** (O5's knob-on SMS is the baseline).

**Baseline (after O5).** Medians of `3 × phase / step_wall` over O5's
knob-on rows, HIP (jobs `f3eJ14annaAT` np1, `f3eJ14ijrioh` np4; log `## O5`):

| target | L4 np1 | L4 np4 | L5 np1 | L5 np4 |
| --- | --- | --- | --- | --- |
| `ilist_s3_classify_pairs` | 31.4 | 10.2 | 30.4 | 23.6 |
| `dn_build_ilist` outside its stage timers | 8.2 | 4.8 | 4.9 | 3.6 |

Per evaluation, stage 3 is a median 22.8 ms (L4 np1), 6.1 (L4 np4), 70.9 (L5
np1) and 26.4 (L5 np4). SERIAL's "before" is O5's `f3eJ14rqpo91` (np1) and
`f3eJ14zvJtBy` (np4). Accuracy baseline, O5's maxima (unchanged since O1): L4
1.4025e-3, SERIAL's 500 steps 1.1338e-3, L5 1.2115e-3.

**Fill in:** `../canopy/src/Canopy_DownwardSweep.hpp`, stage 3 of
`build_interaction_list_device`, under `TIMER_ILIST_S3_CLASSIFY_PAIRS`
(`:1672-2100`). It is already threaded on the host execution space, which is
OpenMP at `OMP_NUM_THREADS=24` in every SMS launch
(`scripts/tuolumne/runtime_env.sh:46`). It runs in three parts:
1. **Sharded classify** (`ilist_s3_classify_shard`, `:1755-1946`): the pairs
   are split into one contiguous slice of entries per host thread. Per pair,
   it decodes both Morton keys, forms the canonical `M2LKey`
   (`KernelType::canonicalize_key`), range-checks it against `_max_depth`,
   `M2L_KEY_DD_MAX` and `M2L_KEY_OFFSET_MAX` (`:598-599`), and looks it up in
   the thread's own `std::unordered_map`, appending first sightings to the
   thread's key list.
2. **Serial merge** (`:1948-2032`): the per-thread key lists are merged into
   `key_to_op` and `ops`, with the column-cap test (`effective_op_cap`), giving
   a local-to-global index map per thread. A global operator index is
   therefore the order in which the merge first meets the key, which depends
   on the thread count.
3. **Parallel remap** (`ilist_s3_remap`, `:2056-2079`): each pair's local
   index becomes its global one, or `-1` for a fallback pair.

In a profiling build, two pieces of instrumentation run inside the same
timer: the demanded-key set (`:1971`, filled in the merge) and the per-reason
fallback counters (`:2050`, filled in the remap). The production build
compiles both out.

The knob: a Canopy `FmmConfig` field, a `DownwardSweep` setter, a Beatnik
`FmmParams` field mapped in `buildConfig`, and a probe key, as O8's
`device_op_table_assembly` and O5's `indexed_comm_plan`
(`../canopy/src/Canopy_Solver.hpp:134,141`; `src/Beatnik_Params.hpp:488`;
`src/Beatnik_FarFieldInterface.hpp:1184`). README "FMM tunables" and the
Canopy README's `FmmConfig` table each gain a row.

**Do:**
1. **Split stage 3 first.** Add level-2 timers around the three parts and the
   probe keys for them, in the same install as the change. Choose the
   candidate from where the time is, and record the split in the log.
2. Candidates:
   - **A direct-index key table in place of the per-pair hash lookup.** Every
     in-range canonical key is a small bounded tuple (`max_d ≤ _max_depth`,
     `|dd| ≤ M2L_KEY_DD_MAX`, `|ii|, |jj|, |kk| ≤ M2L_KEY_OFFSET_MAX` = 32),
     so a dense index of
     `(_max_depth + 1) × (2·M2L_KEY_DD_MAX + 1) × 65³` slots can replace the
     hash. At `max_depth` 10 that is about 3.0 M × (2·`M2L_KEY_DD_MAX` + 1)
     slots, so size it and say whether it is per thread or shared.
   - **Classification on the device.** The pair arrays are uploaded in stage 5
     anyway.
   - **A sort-unique merge** in place of the hash maps.
3. Put the change behind the knob, default on, so the off arm (the current
   path, unchanged) runs in the same job and on the same node.
4. Measure the profiling-only instrumentation's cost inside the stage-3 timer
   (the demanded set and the fallback counters) and report it. Do not move or
   change it.

- **Out of scope.** Reusing the interaction list on a Migrate evaluation:
  Migrate is 6.6 % of evaluations after O1 (237 of 3600 at L4, 59 of 900 at
  L5; log `## O5`), so it bounds the gain at about 7 % of stage 3. The
  comm-plan finalizers (`commplan_finalize_m2l`, 2–15 % of the step after O5)
  are not stage 3 either.
- **Operator order may change.** Any candidate that reorders the merge
  reorders the global operator indices. That is safe: stage 4 gathers each
  table column by key through `_m2l_op_cache_index` (`:774`, read at
  `:2295,2332`), never by position (log `## O8`).

**Exit criterion:** the common O criteria, with Canopy stems `DownwardSweep`,
`MultiSolve` and `CartesianTaylorSolve`. In addition:
- a unit case builds one tree's interaction list with the knob off and on,
  and asserts the same pairs (target, source), the same canonical key per
  pair, the same fallback pairs and fallback slices, and an operator table
  equal column for column once columns are matched by key. The raw `op_idx`
  integers need not match;
- the three stage-3 sub-timer keys print a non-sentinel value in every
  `[cost]` row.

Failure direction: a broken build whose classify assigns one wrong key fails
the unit case (checked once by hand, not committed).

### O10 — Downward-sweep communication and per-depth loop at np > 1 — **NOT STARTED**

**Depends on:** T3.

**Measured (T3):** `downward_total` minus `dn_build_ilist`, `m2l_kernel` and
`l2p` is 6.3–11.1 % of HIP step wall at np4 (medians per run), up to 18.2 % at
one level-4 state. It is under 1 % at np1, so it is the multi-rank part of
`execute` (`../canopy/src/Canopy_DownwardSweep.hpp:3029-3104`):
- `exchange_multipoles_for_m2l`;
- the loop over all `max_depth + 1` = 11 depths of snapshot, allreduce, L2L
  and exchange, of which only 5–8 are occupied.

Phase S raises its weight.

**Additional information needed:** the split, which T3 could not take without
a new probe key. Canopy already times the parts: `m2l_comm`, `m2l_allreduce`,
`l2l_kernel`, `l2l_comm`, and at level 2 `dn_zero_locals`, `dn_pre_m2l`,
`dn_m2l_call` and `dn_post_m2l`. The probe prints none of them. Step 1 adds
them to `kCanopyKeys` (`tests/regression_tests/Beatnik_Probe_FmmCost.cpp:168-193`)
in the same install as the change.

Candidates:
- skip depths with no shared cells;
- coalesce the per-depth collectives.

**Exit criterion:** the common O criteria, with Canopy stems `DownwardSweep`
and `MultiSolve` at np 1–6.

### O6 — Adapter overheads in `FarFieldSolver::evaluate` — **NOT STARTED (conditional)**

**Depends on:** O1, O5, O8, O9 and O10 resolved (DONE or NOT TAKEN).

**Trigger:** `ff_pack + ff_fwd_* + ff_rev_* + ff_scatter + ff_diagnostics` is
more than 5 % of the HIP SMS step wall. T3 measured 1.0–3.4 % (medians per
run, HIP), with one level-4 np4 state at 6.7 %: **not met.** It is re-evaluated
on the SMS once the earlier tasks have shrunk the rest of the step, which
raises this share. SERIAL's adapter share is O7's.

**Fill in:** `src/Beatnik_FarFieldInterface.hpp`, the phases `T3` ranks. The
likeliest target is `p2pPairCountLocal` (`:1298-1323`), a host hash computed on
every evaluation for a diagnostic. It is read by `FmmVsDirect`,
`Milestone0Fmm`, `FmmScan` and the probes, so gating it changes those readers.
Enumerate them in the task and give them a way to request it.

**Exit criterion:** the common O criteria, plus `FmmVsDirect` at ranks 1–6 on
SERIAL and HIP, and every `FarFieldDiagnostics` reader still compiling (the
single `spack install` builds them all).

### O7 — SERIAL rank scaling — **NOT STARTED (lowest priority)**

**Depends on:** T3, and every other O task resolved (DONE or NOT TAKEN).

**What T3 measured** (log `## T3`). At level 4, SERIAL np4 is 3.78 s/step
against np1's 1.72 s (2.2x, 500 steps).
- **The adapter's distributors are the step.** `ff_fwd_distribute` +
  `ff_rev_distribute` are 52 % of step wall at np1 (16.5 + 35.2, medians) and
  87 % at np4 (54.2 + 33.0).
- `solve()` is 41 % of the step at np1 and 11 % at np4. On HIP the same two
  phases are at most 2.3 %.

**Fill in:** the forward and reverse distributor scopes in
`src/Beatnik_FarFieldInterface.hpp` (`:1424`, `:1534`), and whatever Cabana
distributor path they take on the host.

**Exit criterion:** the common O criteria on the SERIAL rows of the SMS.

### O3 — Remove the range-guard fallback's cost — **NOT TAKEN (T3)**

Measured below 5 % of step wall at every level on HIP, so T3 step 3 removes it.
- **Fallback pairs do appear along the trajectory.** `fb_range_guard` peaks
  at 6166 at level 4, 16018 at level 5 and 21768 at level 6, against T2's
  early-state "none at L4/L5".
- **But the fallback M2L runs inside `m2l_kernel`'s timer**
  (`run_m2l_at_depth` calls `run_m2l_fallback_at_depth`,
  `../canopy/src/Canopy_DownwardSweep.hpp:2614-2630`). `m2l_kernel`, fused M2L
  and fallback together, is at most 4.3 % of step wall in any HIP row
  (medians 1.5–3.9 %), and `ilist_s5_fallback_table` is at most 0.2 %.
- **SERIAL is not separated.** At SERIAL np1, `m2l_kernel` is 12.6–15.3 %,
  but fallback pairs are at most 1.1 % of M2L pairs there, and the probe
  cannot split the two.

Reopen if an SMS shows `m2l_kernel` above 5 % of HIP step wall. Routes (a)
batching the fallback on the device and (b) widening `M2L_KEY_OFFSET_MAX`, with
their references, are in this file at `d45aeed`.

### V1 — Phase O boundary: accuracy, gate, docs, production env — **NOT STARTED**

**Depends on:** every O task DONE or NOT TAKEN. It may also be pulled forward by
an O task's 20 % rule.

**O1 triggered that rule, and the user deferred V1 anyway** (log, "Decision:
defer V1; a milestone failure from O1 is accepted"). **The milestone FMM
members may fail claim A in step 1, because they now inherit O1's quantized
root (level-4 SMS worst 1.402e-3 against `kTauA` 1.5e-3). That is accepted.
If they fail, raise `kTauA` to cover the measured worst, record the run, step
and value in the log, and re-run; do not turn quantization off.** The exit
criterion's "16/16 green" is then met at the raised bound.

**Do:**
1. Run the milestone tier, split:
   `scripts/tuolumne/submit_milestone_split.sh` from the login node. Its
   per-job `-t` values are about 1.5x T9b's walls. If Phase O cut the FMM cost
   substantially, tighten them in the same change from the new measured walls.
2. Submit the regression gate, `scripts/tuolumne/run_regression_minset.flux`,
   in the same sitting. The adapter is included by every TU, so a Beatnik `src/`
   change must not break the direct path.
3. Re-measure the final speedup at production `ncrit 64` as well as at 8: run
   T3's level-4 and level-5 HIP rows with `ncrit=64` on the T3 binary and on the
   final one, in one job each. README states both.
4. Update README:
   - "FMM accuracy (measured)" with the new claim-A figures;
   - the speedup and crossover statement (add-canopy T8's README obligation);
   - "Known Issues" → "FMM per-step cost grows" with the post-optimization
     growth factor.
5. Update add-canopy.md's Current state numbers that Phase O changed (costs,
   walls).
6. Bring the production env up to date: pull nothing (Canopy is a sibling
   clone, already at the committed hash), then `spack install` in
   `~/spack_envs/tuolumne_beatnik_production` **only when no production job is
   live**.

**Exit criterion:** the milestone tier 16/16 green, the gate 60/60 green, and
README carries the measured speedup and crossover at stated levels, ranks and
backend, at `ncrit 8` and at production `ncrit 64`. Failure direction: the tier's claim-A assertion still fails on its
built-in negative case (a state perturbed past $\tau_A$), as it did at T9b.

---

## Phase S — scale (multi-node)

### S0 — Scale measurement campaign — **NOT STARTED**

**Depends on:** V1.

**Fill in:** `scripts/tuolumne/canopy_profiling_scale.flux` (new).

**Additional information needed, to settle with the user before writing the
script:**
- **The environment.** Profiling needs the dev env (`canopy +profiling`), but
  `CLAUDE.md` puts big jobs on the production env so a dev rebuild cannot
  SIGBUS them. Either:
  - **(a)** run in the dev env and freeze it (no `spack install`) while S0's
    jobs are queued or running; or
  - **(b)** temporarily give production a profiling canopy.
- **The matrix.** Levels 6 and 7 (163 842 vertices), 16, 32 and 64 ranks (4–16
  nodes), HIP. Steps from a level-6 np16 sizing run. `direct` at level 7 is
  affordable only as the probe's sampled standalone evaluation, so set
  `sample_every` to keep it a small share.
- **The queue.** `pbatch`. Size `-t` at 1.5x a sizing measurement, as T9b did.

**Do:**
1. Run the probe at scale.
2. Reduce.
3. Record the phase split at each rank count, the scaling efficiency of each
   phase, and which phases grow with rank count: `ff_fwd_migrate`,
   `ff_rev_migrate`, `repartition`, `comm_plan_build` and
   `commplan_dtt_traversal` (the DTT runs on the replicated global cell list).
   Also record the M2L/P2P communication inside `solve`.
4. Define an SMS-S, the scale counterpart of T3's SMS.

**Exit criterion:** the log carries the per-phase split and scaling efficiency
at every measured (level, ranks), and this document gains S-tasks ranked by
measured share, each with its own exit criterion. Failure direction: the
maintenance-count check (3 × steps per row) passes at every rank count.

### S1… — Scale optimizations — **NOT STARTED (defined by S0)**

**Depends on:** S0.

Candidates already recorded in Canopy, to be confirmed or discarded by S0's
numbers:
- NBX sparse peer discovery in `migrate_particles`, replacing the `MPI_Alltoall`
  of counts (`../canopy/src/Canopy_TreePartitioner.hpp:891`;
  `../canopy/README.md`, "Nonblocking-consensus peer discovery");
- partition cost on small per-rank trees ("Partition cost on small trees");
- the replicated DTT.

Each S task follows the O-task pattern against SMS-S.

### V2 — Phase S boundary — **NOT STARTED**

**Depends on:** every S task DONE or NOT TAKEN.

**Do:** V1's steps 1–6 again, with README's scaling statement added.

**Exit criterion:** as V1's.

## Known risks

**R1 — Instrumentation changes what it measures.** Beatnik's `ScopedTimer`
fences on destruction (`src/Profiling.hpp:112-133`), and Canopy level-2 timers
add host work. Either can hide overlap that a production binary has. **Presents
as:** an optimization that wins under the probe and loses in the milestone tier's
wall. **Distinguishing measurement:** compare the probe's step wall in a
level-2 build with the `[t6] COST` claim-B wall of the same (level, np) at
matched steps. T2's `canopy_report` ratio bounds the reporting part; the fence
part is bounded by V1's tier walls. Record both. A gap above 10 % means O-task
gains are quoted only from V1's walls.

**R2 — Run-to-run spread swamps a small gain.** ParMETIS partitions and HIP
reductions are not bit-reproducible (add-canopy R8; Canopy README "HIP not
bit-reproducible"). **Presents as:** a gain of a few percent that reverses on
re-run. **Mitigation:** T3 runs one level-4 and one level-5 SMS configuration
twice each and records the per-state `[acc]` and step-wall spread at both
levels. No O task claims a gain smaller than twice that spread. T3 measured it ([SMS
definition](#sms-definition)): `[acc]` did not move at all, and the step wall
moved 0.15–0.83 % per run and up to 4.6 % (L4) and 6.9 % (L5) per state.
The accuracy guard therefore needs no spread allowance.

**R3 — A short run is mistaken for a representative one.** Per-step cost grows
2.7x along the level-4 trajectory. **Presents as:** an O task measured only at
early states that "wins" by optimizing a phase that matters only late, or the
reverse. **Mitigation:** the SMS must include level 4 past step 1000 (T3 step
4). Compare at matched states, never at matched wall.

**R4 — Quantizing the root box costs accuracy or depth.** The root cell can be
up to 2x the bounding box, adding a level at the top
(`../canopy/src/Canopy_Solver.hpp:123-125`) and shifting every cell boundary.
**Presents as:** an `[acc]` change at O1, or `depth_dropped` fallback when
`max_depth 10` binds. **Distinguishing measurement:** `occupied_depths` and
`fb_dropped` in O1's rows. A change in `[acc]` with unchanged depth is the MAC
seeing different cells. It is legitimate as long as the level's worst state
does not get worse ([Conventions](#conventions)). Dropped pairs are a defect.

**R5 — Reusing a column at the wrong width.** Any cache-retention change (O1,
O2) can evaluate an operator built for a different width. **Presents as:** a
plausible field with an error that is small but **not** small enough. The
`[acc]` metric moves to near $\tau_A$ rather than exploding. **Distinguishing
measurement:** B2's failure-direction check in Canopy, and the `[acc]` change
against T3's baseline at the same state. A retention gain that coincides with an
`[acc]` jump is this, not geometry.

**R6 — A dev-env install kills a running measurement.** **Presents as:** rc=135
(SIGBUS), or a job that ran a different binary than the log's Canopy hash says.
**Mitigation:** [Build and run economy](#build-and-run-economy-applies-to-every-task)
rule 2. If it happens, the job's numbers are discarded, not averaged in.

**R7 — The probe's direct estimate is not the direct path's cost.** `[acc]`'s
per-step direct estimate is 3 standalone evaluations plus the measured non-BR
step overhead, not a `--br-approximation direct` run. **Presents as:** a
crossover that disagrees with a real direct run. **Distinguishing measurement:**
at level 4 HIP np1, compare against M0-D1's measured 0.0085 s/step
(`[m0d1] TIMING`). T3 records the ratio and applies it if it exceeds 10 %.

**R8 — Level 6 does not fit a node.** Memory for 40 962 sources with a deeper
tree and a larger operator table is unmeasured on one MI300A. **Presents as:**
an allocation failure or a `pdebug` timeout in T2's sizing launch.
**Mitigation:** T2's sizing launch exists to find this before T3 commits a
matrix to it. If level 6 does not fit at np1, T3 measures it at np4 only and
says so.
