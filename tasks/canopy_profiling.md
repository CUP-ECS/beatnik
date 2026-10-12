# Canopy profiling and optimization, driven through Beatnik

**Status:** NOT STARTED

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
are therefore measured at early states only, for scaling with N; level 4 alone
carries the trajectory-growth measurement.** T3 records the physical time each
measured state corresponds to, so the comparison is explicit.

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
  (`src/Beatnik_FarFieldInterface.hpp:1107-1137`) never sets it, and
  `FmmParams` (`src/Beatnik_Params.hpp:167-434`) has no field for it.
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
| Accuracy bound for any optimization | Claim A's metric, `max|u_fmm − u_direct| / max|u_direct|` over owned vertices (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1447-1490`), must stay **≤ `kTauA` = 1.5e-3** (`:320`) at every SMS sampled state. Any change against T3's baseline is reported per state. Results may move within that bound | The FMM's accuracy contract is $\tau_A$, not bit-identity. A quantized root box, for example, changes the tree. |
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
  as of T9b.
- **Three FMM evaluations per timestep**, one per RK3 stage (see
  [Why the trajectory is FMM-driven](#why-the-trajectory-is-fmm-driven)). Each
  goes through `FarFieldSolver::evaluate`
  (`src/Beatnik_FarFieldInterface.hpp:1348-1625`) in these phases:

  | # | Phase | Code |
  | --- | --- | --- |
  | 1 | pack | `packSources`, `:1388` (body `:1146-1170`) |
  | 2 | forward distribute | `buildForwardDistributor`, `:1395` (body `:1204-1276`, its claims migrate at `:1237`), then an `MPI_LAND` reduce, `:1397` |
  | 3 | forward migrate | `Cabana::migrate`, `:1403` |
  | 4 | `auto_maintain` | `:1406`, or `setup` on first use or tag mismatch, `:1417-1424` |
  | 5 | solve | `:1427` |
  | 6 | contract | `:1431-1485` |
  | 7 | reverse distribute | `:1489-1504` |
  | 8 | reverse migrate | `:1508` |
  | 9 | scatter | `:1511-1553` |
  | — | diagnostics | `readDiagnostics` `:1558` (body `:926-984`), `p2pPairCountLocal` (`:1298-1323`), the `MPI_Allreduce` at `:1589` |

  **None of these is timed today.**
- **Beatnik has no FMM timing instrumentation.** `src/Profiling.hpp` is
  included nowhere, so `beatnik profiling_level=2` compiles no timer into the
  FMM path. The only timings are coarse wall clocks in tests: `[m0d1] TIMING`
  in `Beatnik_Test_Milestone0Run.cpp:478`, `[t6] COST` in
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

### T0 — Baseline from the existing T9b logs (no build, no run) — **NOT STARTED**

**Depends on:** none.

**Fill in:** `tests/regression_tests/fmm_profile_reduce.py` (new; BSD-3 header,
Python comment style as `fmm_divergence_ladder.py`); the progress log.

**Reference:**
- The eight T9b logs in this directory: `beatnik_milestone_Milestone0{Fmm,FmmL4,Frozen,FrozenL4}_{SERIAL,HIP}.*.log`.
- The block format, from `print_solve_breakdown`
  (`../canopy/src/Canopy_Profiling.hpp:333-373`): a header line naming the rank
  count, then Total, Upward, Downward and P2P rows of mean seconds and percent.
- The claim A and claim B phase boundaries are in `Beatnik_Test_Milestone0Fmm.cpp`'s
  `[note]` and `[t6]` lines.

**Do:**
1. Write a reducer that splits each FMM log into (rank count, claim A
   evaluations, claim B evaluations) and reports, per launch, the per-solve
   Total/Upward/Downward/P2P distribution (median, p90, max) and its trend
   over claim B's steps.
   - Attribute blocks by their position relative to the `[note] ... claim A step N`
     lines and the claim-B step lines.
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

---

### T1 — Canopy: a run-time reporting switch — **NOT STARTED**

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
- `README.md`: document the switch in its profiling section.

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

---

### T2 — Beatnik: phase timers, per-evaluation counters, and `Beatnik_Probe_FmmCost` — **NOT STARTED**

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
- The eleven `ff_*` phases sum to `ff_evaluate_total` within 5 % at every row.
- `ff_solve` is within 5 % of Canopy's `upward_total + downward_total + p2p_total`.
- Each row's maintenance counts sum to 3 × the steps in the window.
- `[acc]` metric ≤ 1.5e-3 at every level-4 row.
- **Zero** `[Canopy Diagnostics]` lines in the `canopy_report=0` launches.

The log records the `canopy_report=1`/`=0` step-wall ratio and the level-5/6
sizing rates.

Failure direction: launching the probe with `bogus=1` exits non-zero with a
message naming `bogus`, and the `canopy_report=1` launch's log *does* contain
`[Canopy Diagnostics]` lines.

---

### T3 — The baseline cost model, the crossover, and the SMS — **NOT STARTED**

**Depends on:** T0, T2.

**Fill in:**
- `scripts/tuolumne/canopy_profiling_baseline.flux` (new). The HIP matrix may
  be split into `_np1` and `_np4` scripts submitted together.
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
     This shows whether the crossover depends on leaf size.
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
- The `[acc]` metric ≤ 1.5e-3 at every SMS state, with its change from T3's
  baseline. A move over 20 % at any state triggers V1 before the next O task.
- The task's named Canopy stems passing at SERIAL np 1–6 and HIP np 1–4.

**Failure direction for every O task:** an SMS run with the change disabled (its
knob off, or the previous Canopy commit) reproduces the previous numbers within
the run-to-run spread recorded in T3. A gain that does not vanish when the
change is off is not attributed to the change.

The order below is the expected ranking from
[What is already known](#what-is-already-known-and-what-it-predicts). **T3
reorders it.**

### O1 — Turn on `quantize_root_half_width` from Beatnik — **NOT STARTED**

**Depends on:** T3.

**Fill in:**
- `src/Beatnik_Params.hpp`: `FmmParams::quantize_root_half_width`, beside
  `m2l_op_count_cap` (`:433`), with a doc comment. The default is decided by
  step 3.
- `src/Beatnik_FarFieldInterface.hpp`: map it in `buildConfig` (`:1107-1137`).
  `buildConfig`'s only caller is `evaluate` (`:1368-1373`).
- `Beatnik_Probe_FmmCost.cpp`: key `quantize_root_half_width`.
- README "FMM tunables" (`README.md:306`).

**Reference:** `../canopy/src/Canopy_Solver.hpp:118-126`; B2 in
`../canopy/tasks/tree-opt.md` (`:1498`) and its progress log `§B2`, including
its failure-direction check for a column used at the wrong width (B2's R4).

**Do:**
1. No Canopy change.
2. Run the SMS with the knob on, and with it off as the control, in one job.
3. If the gain is real and `[acc]` holds, set the default to `true`. The
   quantized root can add one level at the top of the tree (`:123-125`).
   `max_depth 10` is a cap, so check `occupied_depths` against it and confirm
   no `depth_dropped` fallback appears.
4. Record per evaluation:
   - retention, as `keys_built` delta against `cache_size`;
   - the evaluations at which the cache emptied anyway because of an octave
     crossing;
   - the change in `ilist_s4_op_table_build`.

**Exit criterion:** the common O criteria, plus retention above 90 % on every
non-crossing evaluation in the SMS. `FmmVsDirect` passes at ranks 1–6 on SERIAL
and HIP via `scripts/tuolumne/t4_fmm_vs_direct.flux`; fold that run into the SMS
job. Failure direction: the knob-off control shows
`keys_built_delta == unique_ops` at every row, as T3 did.

### O2 — Keep the operator cache across an octave crossing — **NOT STARTED (conditional)**

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

### O3 — Remove the range-guard fallback's cost — **NOT STARTED**

**Depends on:** T3. It is independent of O1.

**Fill in:** `../canopy/src/Canopy_DownwardSweep.hpp`. Either the fallback
path, `run_m2l_fallback_at_depth` (`:2638`), or the key encoding, the range
guard at `:1705-1709` with `M2L_KEY_DD_MAX` / `M2L_KEY_OFFSET_MAX` at `:570-571`.

**Additional information needed:** which of two routes, decided from T3's
measured fallback time share:
- **(a)** Batch fallback pairs on the device, either grouped by canonical key
  into a transient table evaluated by the fused kernel, or one team per target
  ([`../canopy/README.md`](../../canopy/README.md), "Batch the per-pair M2L
  fallback on the device"). It keeps the key space and speeds up every refusal.
- **(b)** Widen `M2L_KEY_OFFSET_MAX`. For `CartesianTaylorBasis` the
  `m2l_key_dd_max` "merely bounds the key space"
  (`Canopy_CartesianTaylorBasis.hpp:466-470`), and with `dd` already
  canonicalized away (B1) only the offset bound fires. This removes refusals
  but grows the table, so its cost is weighed against O1's retention.

The session decides with the user once T3's numbers exist.

**Exit criterion:** the common O criteria, with Canopy stems `DownwardSweep`,
`MultiSolve` (whose `M2L_BinEdge_Fallback` asserts a non-zero fallback
population, `tests/tstMultiSolve.hpp:1197-1229`) and `CartesianTaylorSolve`.
The SMS's `fb_range_guard` and fallback time are recorded before and after.

### O4 — Parallelize the host operator-table build — **NOT STARTED (conditional)**

**Depends on:** O1 **DONE**.

**Trigger:** after O1, `ilist_s4_op_table_build` is still more than 5 % of SMS
step wall.

**Fill in:** `../canopy/src/Canopy_CartesianTaylorBasis.hpp:1186-1240`, a
serial host loop over missing keys whose columns are independent.

**Exit criterion:** the common O criteria, with Canopy stems
`CartesianTaylor` (non-MPI) and `CartesianTaylorSolve`. The built columns must
be bit-identical to the serial build's: assert it in a unit case that builds
one key set both ways.

### O5 — Cheaper maintenance between RK stages — **NOT STARTED**

**Depends on:** T3, and O1 **DONE** (O1 changes what maintenance rebuilds).

**Fill in:** `../canopy/src/Canopy_Solver.hpp` `auto_maintain` and the paths it
selects (`:442-726`), `../canopy/src/Canopy_CommunicationPlan.hpp`
(`build_all_interaction_lists`, a serial host dual-tree traversal on every rank,
`:556-~800`, called from `build` at `:960-1013`), and
`../canopy/src/Canopy_TreePartitioner.hpp` (`AdaptiveRepart` at `:592-599`).

**Additional information needed:** T3's split of `ff_maintain` into tree build,
partition and repartition, comm-plan build and DTT, and migrate. The fix depends
on which dominates. Candidates:
- skip repartition when imbalance is under tolerance;
- reuse the interaction list when the cell key set is unchanged (the
  `_finish_topology_stable` path, `:706-726`, already reuses the comm plan);
- reduce the DTT's per-rank work.

**Any change to which `MaintenanceAction` is chosen is a policy change.** It
needs the user's agreement before implementation, because it changes what the
milestone tier exercises.

**Exit criterion:** the common O criteria, with Canopy stems `TreeBuilder`,
`TreePartitioner`, `CommunicationPlan` and `MultiSolve`.

### O6 — Adapter overheads in `FarFieldSolver::evaluate` — **NOT STARTED (conditional)**

**Depends on:** T3.

**Trigger:** `ff_pack + ff_fwd_* + ff_rev_* + ff_scatter + ff_diagnostics` is
more than 5 % of SMS step wall.

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

**Additional information needed:** T3's SERIAL np1-versus-np4 phase split.
SERIAL np4 is slower than np1 (2.21x at level 3, 1.37x at level 4), and which
phase scales backwards decides the task.

**Exit criterion:** the common O criteria on the SERIAL rows of the SMS.

### V1 — Phase O boundary: accuracy, gate, docs, production env — **NOT STARTED**

**Depends on:** every O task DONE or NOT TAKEN. It may also be pulled forward by
an O task's 20 % rule.

**Do:**
1. Run the milestone tier, split:
   `scripts/tuolumne/submit_milestone_split.sh` from the login node. Its
   per-job `-t` values are about 1.5x T9b's walls. If Phase O cut the FMM cost
   substantially, tighten them in the same change from the new measured walls.
2. Submit the regression gate, `scripts/tuolumne/run_regression_minset.flux`,
   in the same sitting. The adapter is included by every TU, so a Beatnik `src/`
   change must not break the direct path.
3. Update README:
   - "FMM accuracy (measured)" with the new claim-A figures;
   - the speedup and crossover statement (add-canopy T8's README obligation);
   - "Known Issues" → "FMM per-step cost grows" with the post-optimization
     growth factor.
4. Update add-canopy.md's Current state numbers that Phase O changed (costs,
   walls).
5. Bring the production env up to date: pull nothing (Canopy is a sibling
   clone, already at the committed hash), then `spack install` in
   `~/spack_envs/tuolumne_beatnik_production` **only when no production job is
   live**.

**Exit criterion:** the milestone tier 16/16 green, the gate 60/60 green, and
README carries the measured speedup and crossover at stated levels, ranks and
backend. Failure direction: the tier's claim-A assertion still fails on its
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

**Do:** V1's steps 1–5 again, with README's scaling statement added.

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
re-run. **Mitigation:** T3 runs one SMS configuration twice and records the
spread. No O task claims a gain smaller than twice that spread.

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
seeing different cells, which is legitimate within $\tau_A$. Dropped pairs are a
defect.

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
