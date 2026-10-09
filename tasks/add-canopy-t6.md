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

At `--icosphere-subdivisions 4` the level-4 fallback count is zero through step
225, first becomes non-zero at step 250, and by step 1375 routes about
13 000 pairs to the per-pair fallback **at np1** — the cap is per rank, so np4
routes far fewer at the same step, 5 300 (SERIAL) and 5 392 (HIP). The onset at
step 250 is not itself a cap exceedance: demand there is 18 396 keys, 56 % of
the cap, and does not exceed it until step 900 (T5). Four assertion sites fail
as a result:
the purity precondition `p.m2l_fallback == 0`
(`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:1500`) at 71 of the 81
states, the same check on claim B's final state (`:1967`), and — at four to six
of the 81 states — **both** forms in which the member asserts the accuracy
bound, `p.max_abs <= kTauA * p.scale` (`:1505`) and `p.rel <= kTauA` (`:1506`),
peaking at `1.2513e-3` against `kTauA = 1.0e-3` (defined at `:320`).

**Most of that fallback is not a budget, and no configuration removes it.** T8b
measured the classify pass's range guard carrying **100 %** of level-4 fallback
at the raised cap — 215 302 pairs at np1, 215 742 at np4 — against bounds that
are `static constexpr` in Canopy (`M2L_KEY_DD_MAX`, `M2L_KEY_OFFSET_MAX`) with
no `FmmConfig` or `FmmParams` route. Zero fallback at level 4 is unreachable,
so the `p.m2l_fallback == 0` checks at `:1500` and `:1967` can never pass there.
The observable that matters is zero **cap-driven** refusal.

**The fallback does not explain the τ_A exceedance.** The per-pair fallback
(`m2l_translate`) and the operator-table path evaluate the same mathematics, and
Canopy measures them agreeing to **3.27e-15** of the field, bounded at
`CTS_PATH_DEV_TOL = 6.55e-15`, on `CartesianTaylorBasis` at order 3 — this
member's basis and order (`canopy/tasks/tree-opt.md` C1,
`canopy/tasks/tree-opt-progress-log.md` `## C1`). A 2.5e-4 excess over τ_A
cannot come from a path that moves the field by 1e-14, so the level-4 peak of
`1.2513e-3` is in substance a measurement of the expansion at order 3,
`mac_theta` 0.3. τ_A must not be widened on this or any evidence.

**With no cap-driven refusal anywhere, claim A is still over τ_A, so the
expansion has to change.** At order 3, `mac_theta` 0.3, the worst level-4
relative error is `1.2536745760757648e-3` at HIP np1 and
`1.2473681315787063e-3` at HIP np4. Both are at step 1375, with
`fb_count_cap == 0` at every state (T9a). τ_A was set at `1e-3` from a single
state five steps off the initial condition, where T5 measured `5.0e-4`. The
roll-up is 2.5x worse. **Beatnik's production `order` is chosen to match the
reference treecode's accuracy, not its order number.** An FMM's order-$p$
gradient carries the truncation of a treecode's order-$(p-1)$ velocity, so
Beatnik's order 3 is the counterpart of the reference's order 2
(`README.md:317-328`). They measure `5.0e-4` and `4.8e-4` on comparable
2562-source early states (T5; `tasks/treecode.md` §1). **At the roll-up the
reference does not meet 1e-3 either.** T9r measured it on the member's own 81
states: worst `1.4987010690098229e-3` at step 1550, over 1e-3 at 19 of 80
states. Beatnik's worst errors are under that, so production stays at order 3
and τ_A is re-derived from the reference as `1.5e-3` (T9e). T9c and T9d, which
would have raised the production `order` to 4, are not taken.

**The demand is unmeasurable at the current cap, and that is the first thing to
fix.** The serial merge refuses a key once `ops.size()` reaches
`effective_op_cap` (`src/Canopy_DownwardSweep.hpp:1616-1645`), so
`n_unique_ops` (`:1663`), `_m2l_realized_keys` (`:1697`) and every consumer of
`m2l_n_unique_ops()` (`:934-937`) saturate at the cap by construction. They read
32768 whether the tree wants 33 000 keys or 300 000. The fallback pair count is
not a proxy: it counts *pairs* refused a column, not *keys* refused one, and
many pairs share a key.

So the cap cannot be sized. This document makes the demand observable, measures
it on Beatnik's own level-4 geometry, raises the cap to cover that number, and
then separates out the refusal path that routes pairs to the fallback where the
cap is not reached at all. With the cap ruled out, it measures the reference
treecode at the roll-up and re-derives τ_A from the reference's error there.

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
1432 s, SERIAL np4 523 s, HIP np1 **84.543 s** and HIP np4 **100.154 s**. All
four claim-A halves together are about 35 minutes
(`tasks/canopy/add-canopy-progress-log.md:2377-2386`), and the two HIP ones
about three minutes. **The
measurement fits `-q pdebug -t 60m`.** Only the final validation (T9b) needs
`pbatch`, and T9a exists so that it is not submitted until it is expected to
pass.

### Out of scope

- **Widening `kTauA` to fit Beatnik's own output.** It is a claim about a
  parameter set, and no Beatnik measurement is evidence for loosening it. The
  only route by which it moves is T9e, which re-derives it from the reference
  treecode's measured error on the same states.
- **Changing the gate.** The `regression` tier keeps exactly five members and
  60 launches; nothing here adds or removes one.
- **The `milestone` tier's membership.** It keeps its four members and sixteen
  launches. The probe (T3) is registered in no tier.
- **The unpairable level-4 window at steps 1350–1900.** A known T5 finding,
  neither a pass nor a failure, and independent of the cap.
- **Widening or re-encoding the M2L key bounds.** Canopy keeps
  `M2L_KEY_DD_MAX` and `M2L_KEY_OFFSET_MAX` as sized for a balanced tree and
  treats the per-pair fallback as the correct algorithm for out-of-range pairs
  (`canopy/tasks/tree-opt.md`, "What this is not").
- **Enabling tree balancing** (`FmmConfig::tree_balance_max_level_delta`). It
  raises range-guard refusals 1.6–3.5x on the only measured draw, which is why
  its default stays off (`canopy/tasks/tree-opt.md` A3).
- **Lowering `mac_theta` as the first lever.** T9c measures it as the fallback
  T9d may adopt only under the rule T9c states; see [Approach](#approach).
- **Error-controlled acceptance in Canopy** (a per-interaction error estimate
  or a per-level order) and **`FmmConfig::quantize_root_half_width`**. They are
  the long-term answers to accuracy and rebuild cost under AMR roll-ups of
  millions of points, and they are Canopy projects, not tasks here.
- **Setting `run_milestone.flux`'s `-t` from a measurement.** It stays at
  `-t 1440m` until a tier run passes; re-timing from a failing run whose most
  expensive member does a different amount of work than the fixed version would
  bake in a cost that is about to change.

## Approach

Five moves, in order, each cheap enough to verify before the next commits to
anything:

1. **Make demand observable** (T1, T2). Count the distinct canonical keys the
   merge *sees*, independently of how many it *admits*. The merge loop at
   `src/Canopy_DownwardSweep.hpp:1602-1645` already iterates each thread's
   distinct keys, and those keys are already canonical (`:1367-1370`, call site
   `:1495`), so the counter is one hash insert per key already in hand — not per
   pair. It feeds nothing in the solve.
2. **Validate the counter without an HPC job** (T6's unit test, runnable at T1
   time). `tests/tstLaplaceSolve.hpp:793` already takes
   `m2l_op_table_byte_budget` as its one configuration knob and applies it at
   `:906-909`. Driving it at a budget worth one column makes demand exceed
   realized by a known margin on a tree small enough to run in seconds.
3. **Measure on the real geometry** (T3, T4, T5). A standalone probe that drives
   the direct trajectory and evaluates the FMM once per checkpointed state,
   printing the demand series. Minutes on HIP.
4. **Act on the number** (T6, T7, T8, T8b). Plumb the count cap through
   `FmmConfig` and `FmmParams` with the default unchanged, raise it at the
   level-4 member to cover the measured demand, and then identify the refusal
   path that routes pairs to the fallback where the cap is not reached. That
   path is the range guard (T8b), and it stays: the end state is zero
   cap-driven refusal, not zero fallback.
5. **Settle claim A against the reference** (T9r, then T9e or T9c and T9d).
   Measure the reference treecode's own error on the member's 81 states. If it
   also exceeds 1e-3 at the roll-up, re-derive τ_A from it and keep order 3
   (T9e). Otherwise measure the error against `order` and `mac_theta` at the
   states over τ_A and at level 5, raise the production `order` to 4 at
   `mac_theta` 0.3, and re-derive claim B at the new order (T9c, T9d).

### Why the order, and not `mac_theta`

**Raising `order` changes nothing about the tree.** The MAC, the interaction
lists, the P2P near field, the liveness inequality, the operator-key demand and
the range guard's refusals are all functions of `mac_theta`, `ncrit` and the
geometry. `order` changes only the expansion length: $\binom{p+3}{3}$
coefficients, 20 at $p=3$ and 35 at $p=4$, so `bytes_per_key` goes from 3200 to
9800 and M2L work per pair rises about 3x. Every count T5–T9a measured carries
over unchanged.

**Lowering `mac_theta` gets worse as N grows.** On a 2-manifold the near field
and the interaction lists both scale as $\theta^{-2}$, so going from 0.3 to 0.2
multiplies P2P work, M2L work and key demand by about 2.25x. It also pushes more
cross-depth pairs past the range guard into the per-pair fallback, which costs
52–109x a table pair on HIP (`canopy/tasks/tree-opt-progress-log.md` `## C1`).
That is the cost an AMR roll-up of millions of points pays. At level 4 it also
moves the P2P fraction toward `kP2PFractionBound = 0.75`: T5 measured 0.566 at
$\theta=0.2$ five steps in, and the fraction rises along the trajectory.

**The error is set by the roll-up's geometry, not by N.** At fixed `order` and
`mac_theta` the per-pair truncation is fixed by the MAC ratio, so the relative
error is expected to be flat in N. That is reasoned and not measured; T9c
measures it at level 5. The 2.5x between step 5 and step 1375 is geometry, and
AMR concentrates points in exactly that regime, so the margin is sized at the
worst roll-up state. At T5's state, order 4 bought 8.9x (`5.61e-5`) and
$\theta=0.2$ bought 4.3x (`1.16e-4`) (`tasks/canopy/add-canopy.md` T5 Met).

**Order 4 is gated upstream.** Canopy's derivative ladder at order $p$ reaches
degree $|k|=2p$, and its independent oracle is validated only to $|k|=6$
(`tasks/canopy/add-canopy.md` R11). Order 4 needs $|k|=8$, so
`canopy/tasks/02_oracle_extension.md` must be **DONE** before T9d adopts it.

### The two facts that shape every decision here

**CartesianTaylor keys carry the tree level.**
`src/Canopy_CartesianTaylorBasis.hpp:486` sets `key_needs_level = true`, and
`canonicalize_key` keeps `max_d` and zeroes only `dd` (`:492` declares
`key_needs_dd = false`; `:504-507`), so the canonical key is
$(\texttt{max\_d}, 0, \texttt{ii}, \texttt{jj}, \texttt{kk})$, because the operator is
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
whenever the root half-width changes (`src/Canopy_DownwardSweep.hpp:451`), so
on a drifting bounding box the cache empties on every rebuild and every
admitted column is built again. `FmmConfig::quantize_root_half_width`
(`Canopy_Solver.hpp:126`, default `false`) would let the cache survive drift
within an octave; nothing here sets it. `local_m2l_op_keys_built`
(`src/Beatnik_FarFieldInterface.hpp:279-283`) is the counter that shows it:
"climbing by the full cache size at every build is the signature of a cache that
retains nothing, which is what a level-keyed basis on a drifting bounding box
does." T3, T4 and T5 measured exactly that — `keys_built_delta == unique_ops` in
405 of 405 rows at level 3 and again at level 4, including in the regime where
the cap binds.

**What that costs a cap raise is bounded by demand, not by the cap.** The
rebuilt column count per evaluation is `min(demand, effective_cap)`, because
only admitted keys are built, so raising the cap above the demand peak buys no
extra rebuild work at all and raising it to cover the peak costs the ratio of
peak demand to the old cap — **1.150x** at the measured level-4 peak, and
nothing at the 44 of 81 np1 states or at any np4 rank where demand is already
under 32 768. Memory scales with the cap and rebuild time does not.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Demand counter gating | `#ifdef CANOPY_ENABLE_PROFILING` | Zero cost and zero memory in a production build. Two mechanisms set the define, for two consumers. **Into Beatnik** (T4): Canopy's exported INTERFACE target (`canopy/src/CMakeLists.txt:27-33`), so a `canopy +profiling` spec is sufficient and no Beatnik CMake change is needed. **Into a standalone Canopy build** (T1's own two builds): the cmake option `Canopy_ENABLE_PROFILING` (`canopy/CMakeLists.txt:252-277`), as `-DCanopy_ENABLE_PROFILING=ON` or `=OFF` — `OFF` is an authoritative kill switch there, forcing the level to 0 whatever `Canopy_PROFILING_LEVEL` says. |
| "Unavailable" sentinel | `-1` | Zero demand is legal — a tree with no M2L pairs realizes no keys — so `0` must never mean "not compiled in". Every accessor and diagnostic field returns `-1` in a `~profiling` build. |
| Diagnostic set bound | `M2L_DEMAND_COUNT_CAP = 1048576` | $2^{20}$ keys, about 56 MB of `unordered_set`. Chosen to exceed what the 2 GiB byte budget could ever buy at this basis's 3200 B per key (671 088 columns), so a *saturated* demand counter would already say the cap is the wrong instrument without needing the exact figure. Reported through a separate `demand_saturated` flag, never conflated with the count. |
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
  in every production configuration. Demand is a sizing instrument, not a
  gate. The gate-side observable is zero **cap-driven** refusal, which an
  ungated comparison already shows: the merge refuses a key only once
  `local_m2l_unique_op_count` reaches `local_m2l_op_cap`
  (`Beatnik_FarFieldInterface.hpp:273`, `:303`), so a realized count below the
  effective cap on every rank means no cap refusal. T9b puts that assertion in
  place of `p.m2l_fallback == 0`.
- **`M2L_OP_COUNT_CAP` becomes a default rather than a floor.** Today
  `m2l_effective_op_cap()` (`Canopy_DownwardSweep.hpp:368-374`) is
  `min(M2L_OP_COUNT_CAP, byte_budget / bytes_per_key)` and the constant is a
  hard floor on purpose — the deviation note at
  `canopy/tasks/abstract-solver-backend.md:303-311` keeps it so that a pure byte
  budget cannot move which pairs overflow. T6 preserves that property by a
  different mechanism: the cap remains a count, is still floored by the byte
  budget, and its default is the same constant — so the overflow set moves only
  for a configuration that explicitly asks. The note's cited line for the
  constant (`src/Canopy_DownwardSweep.hpp:343`) was stale; T6 corrected it
  to `:607`, the line the constant lands on after T6's own edit.
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

- **The clone is on branch `investigate-m2l-cap` at commit `38658ad`**, and
  that is what `canopy@=develop` — a `spack develop` spec, compiled in place —
  builds. It carries T1's, T6's and T8b's instrumentation and, beyond the
  commit T1-T8b measured on (`fd89815`), these changes that move level-4
  numbers:
  - a distributed **ParMETIS** cell partitioner
    (`src/Canopy_TreePartitioner.hpp`, which includes `parmetis.h`). ParMETIS
    4.0.3 is in the Beatnik env's view as a trilinos dependency; T9a's
    `spack install` was the first Beatnik build of this partitioner, and it
    compiled cleanly;
  - `mac_satisfied` rejects exact MAC ties;
  - `CartesianTaylorBasis` declares `key_needs_dd = false` and its canonical key
    drops `dd`, so the same tree realizes fewer columns;
  - `TreeBuilder::build()` prints a rank-0
    `[Canopy] WARNING: TreeBuilder::build: … leaves at max_depth …` line when
    the depth limit, not `ncrit`, stops refinement
    (`src/Canopy_TreeBuilder.hpp:994`);
  - two `FmmConfig` knobs, both off by default and set nowhere in Beatnik:
    `quantize_root_half_width` (`src/Canopy_Solver.hpp:126`) and
    `tree_balance_max_level_delta` (`:135`).

  **Every T5, T8 and T8b demand and fallback figure predates these**, so a
  re-measurement on this clone is a new draw, not a reproduction. Canopy line
  citations in the DONE task entries below are against `fd89815` and have
  shifted.
- `M2L_OP_COUNT_CAP = 32768` (`src/Canopy_DownwardSweep.hpp:608`) is the
  default of the configurable `FmmConfig::m2l_op_count_cap`.
  `m2l_effective_op_cap()` (`:368-374`) floors the configured cap by
  `_m2l_op_table_byte_budget / KernelType::bytes_per_key`. At CartesianTaylor
  order 3 a column is 3200 B, so the 2 GiB default budget buys 671 088 columns
  and **the count cap is the only constraint that binds on Beatnik's path**.
- The range guard's bounds are `static constexpr` with no setter:
  `M2L_KEY_OFFSET_MAX = 32` (`src/Canopy_DownwardSweep.hpp:571`) and
  `CartesianTaylorBasis::m2l_key_dd_max = 6`
  (`src/Canopy_CartesianTaylorBasis.hpp:470`).
- Accessors: `m2l_n_unique_ops()` (`:1069`), `m2l_n_demanded_ops()` (`:1093`),
  `m2l_n_fallback_pairs_range_guard()` (`:1114`),
  `m2l_n_fallback_pairs_count_cap()` (`:1121`),
  `m2l_n_fallback_pairs_depth_dropped()` and `m2l_cells_at_depth()` (`:1152`).
  The demand and per-reason counters return `-1` without
  `CANOPY_ENABLE_PROFILING`; the unique count and the per-depth occupancy are
  ungated.
- Canopy's own test builds are **manual** mode — out-of-tree cmake + make under
  `spack env activate ${HOME}/spack_envs/tuolumne_trilinos`
  (`canopy/systems/tuolumne/claude.md` §1 and §3). T1's two trees,
  `build-t1-prof-on` and `build-t1-prof-off`, live in this clone beside spack's
  `build-linux-rhel8-zen4-*` directories.

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
  `m2l_op_table_byte_budget` and `m2l_op_count_cap` (default 32768), both
  routed to `FmmConfig` in `Beatnik_FarFieldInterface.hpp` with no CLI option.
  T8b added three globally reduced per-reason fallback fields to
  `FarFieldDiagnostics`, `-1` when Canopy carries no profiling.
- `m2l_op_table_byte_budget`'s doc comment (`Beatnik_Params.hpp:382-394`) states
  the current doctrine: "the constraint to act on is the count cap, and the
  response to realized overflow is a lower `max_depth` or `order`, not a
  smaller table." T7 made the count cap itself actionable and rewrote that
  paragraph around the measured figures; `max_depth` 10 (`:291`) stays, and the
  lever its own comment at `:272-291` names was considered and rejected in T8.
- **`order` is a compile-time dispatch, and order 4 is already built.**
  `FarFieldInterface` switches on `params.order` over CartesianTaylor 0, 2, 3,
  4 and 5 (`src/Beatnik_FarFieldInterface.hpp:1024-1040`); SolidHarmonic is
  built at 3 only (`:1047`). Changing the default needs no new instantiation.
  Order 3 is the default, stated and reasoned at `src/Beatnik_Params.hpp:196-229`
  and in README (`README.md:210`, `:314-328`). The callers that read the
  *default* rather than setting `order` themselves:
  `Beatnik_Test_Milestone0Fmm.cpp` (`kProductionOrder = 3` at `:345`, asserted
  equal to `fmm.farField().params().order` at `:1421-1422`, and in τ_A's
  qualification list at `:295`), and `Beatnik_Test_Milestone0Run.cpp`
  (`argv[6]` defaults to `FmmParams::order`). `Beatnik_Test_FmmVsDirect.cpp`
  sets `order` per arm (`:465-469`), and `Beatnik_Test_FmmScan.cpp` uses its
  own `kBgOrder`, so a default change moves neither. No `regression`-tier
  member uses the FMM.
- **The FMM measurement drivers already cover what T9c and T9d need.**
  `Beatnik_Test_FmmScan` (`argv[1]` level, `argv[2]` direct spin-up steps)
  spins up one state on the milestone-0 dt controls (`:212-219`) and evaluates
  an `order` axis (0, 2, 3, 4, 5), a `mac_theta` axis (0.2, 0.3, 0.4, 0.5, 0.7)
  and others against `BRSolverDirect` on that same state (`:270-330`). Its
  `grad_rel` is `grad_abs / max|u_direct|`, the member's claim-A quantity. It
  runs at the `FmmParams` default count cap, 32768, and prints `ops` and
  `op_cap` per arm. Its runner is `scripts/tuolumne/t5_fmm_scan.flux`.
  `Beatnik_Test_Milestone0Run` (`argv[4]` `fmm`, `argv[5]` `ncrit`, `argv[6]`
  `order`) writes FMM-driven 2000-step checkpoint series. Its runner is
  `scripts/tuolumne/t5_divergence.flux`, which is how T5 measured claim B's
  envelope and volume-drift bound (`tasks/canopy/add-canopy.md` T5 step 7).
  Neither driver has a `mac_theta` argument.
- **Claim B's tolerances are measurements at order 3.** `kHorizonEnvelope`
  (`:485` level 3, `:591` level 4) and `kFmmVolumeDriftRtol = 5.0e-2`
  (`:377`, from worst deviations `2.137678e-02` at level 4 and `1.540737e-02`
  at level 3, `:364-366`) were derived from FMM-driven runs at order 3,
  `mac_theta` 0.3, `ncrit` 8. A different order invalidates the derivation,
  even if not the numbers.
- The level-4 member runs `kNcrit = 8`
  (`tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp:340`),
  `kProductionOrder = 3` (`:345`), `kVertices = 2562` (`:511`),
  `kP2PFractionBound = 0.75` (`:544`), `max_depth` 10 asserted at `:1385`.
  `ncrit = 8` is near its floor: the liveness inequality
  $N\gg\pi(\sqrt3/\theta)^2\cdot\texttt{ncrit}$ (`Beatnik_Params.hpp:238-251`)
  is about 840 at `ncrit = 8` against 2562 vertices, and 6720 at the default 64.
  **`ncrit = 8` is itself a demand driver** — it deepens the tree to make the
  far field live at all — which is why T8 cannot simply raise it. The column
  cap is the per-level `kM2LOpCountCap` — 32768 in the level-3 arm (`:478`),
  65536 in the level-4 arm (`:583`) — read by `makeFmmParams` (`:1074`) for
  claim A only; claim B's `SolverParams` (`p.fmm`, `:1029`) still takes the
  `FmmParams` default 32768, so the comment at `:1062-1064` saying the two
  claims cannot be at different configurations is false in that one field.
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
- **T6 remains IN PROGRESS in `tasks/canopy/add-canopy.md`**, its failing tier
  run recorded by T0.
- **The installed prefix is full again.** T9a's untrimmed `spack install`
  put back every test binary (45 in `share/Beatnik/tests`) and both manifests,
  replacing the three-probe prefix T8b left. Any later change to Beatnik or
  the Canopy clone still needs its own `spack install` before a member, gate or
  tier run.

**Environment** (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/spack.yaml`, whose
committed snapshot `systems/tuolumne/spack.yaml` is byte-identical to it):
`canopy@develop` at `:16` carries `+profiling` (dev only — the production
snapshot `systems/tuolumne/spack-production.yaml` does not); `beatnik@develop` at
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
budget-parameterized driver at `:793`).
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
   `M2L_OP_COUNT_CAP` (`:607`), commented with the reasoning in the conventions
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

### T5 — Measure the level-4 demand series — **DONE**

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

**Met.** Three `flux batch` submissions of
`scripts/tuolumne/t6b_key_demand.flux` — `f3bZ3aqyro5y`, `f3bZ93jf57sM` and
`f3bZAPsQynnB`, 169.46 s, 171.64 s and 171.47 s, all `COMPLETED` rc 0 — each
ran level 4 on HIP at np1 and np4 plus the level-3 np1 control, 486 rows per
draw, 81 states per rank with none skipped. **Worst-observed demand is 37 678
keys at HIP np1 rank 0, step 1650**, 1.150x the 32 768 cap and 120 569 600 B
(114.99 MiB) of table against a 2 GiB byte budget that buys 671 088 columns —
so the count cap binds by 17.8x and the memory figure is not the constraint.
Run-to-run spread is **0.46 %** at np1 (37 504 / 37 678 / 37 504) and
0.31–1.50 % per rank at np4, whose worst rank demands **17 144 at step 1375**.
`demand_saturated` was **never set** — 0 of 1944 rows across four complete
draws — so the peak is a measurement, not the $2^{20}$ lower bound the failure
direction describes. At the peak `occupied_depths` is 7 (level-4 range 5–8) and
`keys_built_delta` is 32 768, the full admitted table rebuilt in that one
evaluation; `keys_built_delta == unique_ops` in 405 of 405 level-4 rows,
confirming **R4** at level 4 and in the regime where the cap binds. **R3 is
confirmed:** the demand peak is at step **1650**, the fallback peak's step, not
the error peak's 1375 (97.2 % of the peak) and not step 250, where fallback
begins at 56 % of the cap; demand first exceeds the cap at step **900** in all
three draws.

Two things fall outside what the exit criterion anticipated and are recorded
rather than smoothed over. First, **the level-3 control does not stay inside
the stated 5 938 – 6 198 band** — the three draws peak at 5 728, 6 624 and
5 948, a 15.64 % spread — because that band was a two-draw min/max and five
draws of the same binary now span 5 586 to 6 624. Everything the control
actually tests passes identically in all three draws (zero fallback and zero
over-cap rows in all 243, `demand == unique_ops` 81 of 81, the demand minimum
828 in every draw, `occupied_depths` 4–6), so the apparatus did not move; the
numeric band is nevertheless not met as written, and no level-4 figure depends
on it. Second, **fallback at level 4 is not all cap-driven**: at np4 no rank's
demand ever reaches the cap, yet `global_m2l_fallback` is non-zero at 71 of 81
states in all three draws, and at np1 about half the fallback states sit at or
under the cap. **A cap raise alone therefore cannot drive
`p.m2l_fallback == 0`** — a constraint on T8 that this task's own
Do steps did not ask for and T8 must now carry. Full per-rank series, both
findings and the discarded overlapping submissions that forced the scratch path
to become per job: `## T5` in `tasks/add-canopy-t6-progress-log.md`.

---

### T6 — Canopy: make the count cap configurable, default unchanged — **DONE**

**Depends on:** T1 **DONE**. Independent of T5 — the knob is worth having
whichever way the measurement falls, and building it in parallel with T5 is
fine. Choosing its *value* is T8.
**Fill in:** `canopy/src/Canopy_Solver.hpp` (the config struct near `:84-116`,
the constructor route near `:213`);
`canopy/src/Canopy_DownwardSweep.hpp` (`m2l_effective_op_cap()` at `:368-374`, a
new setter beside `:312-316`, the member block near `:841-855`, the constant at
`:607`, the overflow message at `:1817-1826`);
`canopy/tasks/abstract-solver-backend.md:303-311`;
`canopy/tests/tstLaplaceSolve.hpp`.
*(All `Canopy_DownwardSweep.hpp` lines in this entry are post-T6. T1 inserted
about 124 lines and T6 about 100 more, so any citation of this file written
before T6 is low by roughly that much.)*
**Reference:** `m2l_op_table_byte_budget` is the exact template — declared at
`Canopy_Solver.hpp:96`, routed at `:213`, set at
`Canopy_DownwardSweep.hpp:312-316`, read at `:368-374`, stored at `:847`,
defaulted by a constant at `:623-624`.
**Do:**
1. Add `int m2l_op_count_cap = 32768;` to `FmmConfig`, documented as a per-rank
   bound on the operator **column count**, the companion of the byte budget, and
   the thing a level-keyed basis on a deep tree actually runs out of.
2. Add `set_m2l_op_count_cap( int )` beside the byte-budget setter, invalidating
   the interaction list the same way (`:315`), since it sizes a table built
   there. Reject a value below 0 loudly; 0 is legal and means every pair takes
   the overflow path, matching the byte budget's documented "smaller than one
   column is legal" behaviour (`:309-310`).
3. Change `m2l_effective_op_cap()` to floor the configured cap by the byte
   budget's column count, replacing the `M2L_OP_COUNT_CAP` term.
4. Keep `M2L_OP_COUNT_CAP = 32768` as the **default** the config initializer and
   `_m2l_op_count_cap` member take, so a configuration that sets nothing gets
   today's cap and today's overflow set. Rewrite the comment at `:572-606` to
   say the cap is now configurable with that default, and why the default is
   what preserves every existing answer.
5. Route it from the Solver constructor at `:213`.
6. Update the overflow message (`:1817-1826`) **and the profiling printf
   (`:1879-1891`)** to print the configured cap alongside the byte budget, so a
   log says which of the two bound. Both printed the constant before T6, so
   either one left alone would report 32768 for a run configured otherwise.
7. Update the deviation note at `abstract-solver-backend.md:303-311`: the cap is
   still a count and still floored by the byte budget, the default still makes
   today's overflow set unchanged, and the constant's line is `:607`, not
   `:343`. Note that at CartesianTaylor order 3 the per-key cost is 3200 B, not
   the note's 58 KB at $P=8$, so the count cap binds by a factor of 20 on that
   basis and is the only constraint that ever binds there.
8. **Callers of `m2l_effective_op_cap()`**, enumerated: the merge's
   `effective_op_cap` read (`:1478`), the cache-overflow guard (`:1934`), the
   `:1955` bound check, the two message sites (`:1822`, `:1888`), and in Beatnik
   `readDiagnostics` (`Beatnik_FarFieldInterface.hpp:890`). The signature does
   not change, so none needs editing; all are listed because each reads a number
   whose provenance moves from a constant to a config field.
9. Extend the T1 test cases: one driving a small `m2l_op_count_cap` with a
   generous byte budget and asserting realized equals that cap while demand
   exceeds it, and one asserting that at the default the effective cap is still
   32768 and the realized key set is byte-identical to a pre-change run.
   `with_laplace_solve` (`:793`) gains a second optional parameter for the cap,
   mirroring the byte budget exactly; 0 means "leave the config default", so a
   cap of **0** cannot be driven through it and gets a separate no-solve case.

**Exit criterion:** in the Canopy checkout, a `+profiling` build's suite passes;
a case at `m2l_op_count_cap = 4` reports `m2l_n_unique_ops() == 4` with
`m2l_n_demanded_ops() > 4` and non-zero fallback; and a default-configured case
reports `m2l_effective_op_cap() == 32768` with `m2l_realized_keys()` identical
to the same case before this task. In the failure direction: a negative
`m2l_op_count_cap` raises rather than clamping, and `m2l_op_count_cap = 0`
yields zero columns with every pair on the fallback path.

**Met.** Measured by job **`f3bZMnmTkWbZ`**
(`canopy/scripts/tuolumne/run_t6_count_cap.flux`, `pdebug`, one node,
`--time-limit=40`), which ran `ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL`
at ranks 1-6 in **both** of T1's trees. `100% tests passed, 0 tests failed out
of 6` in each, combined rc 0 — so the `+profiling` suite passes, and so does
the `~profiling` one, which matters because both new cases' cap assertions are
ungated and must hold there too.

Every clause of the criterion was observed at all **21 `(nprocs, rank)`
pairs**:

- **The cap at 4.** `m2lOpCountCapConstrained` reports `eff_cap=4 realized=4`
  with `budget=2147483648` — the 2 GB default, worth 97 823 columns at this
  basis's 21 952 B per key, so the **count** is unambiguously what bound.
  `demanded` runs **111 .. 718** (strictly above 4 everywhere) and `fallback`
  **175 .. 1 677** (non-zero everywhere), with `saturated=0`. The demanded
  range reproduces T1's one-column case exactly, which is a free cross-check
  that the two caps refuse from the same demanded set.
- **The default still 32768.** `m2lKeyDemandDefault` reports
  `eff_cap=32768` at every pair, and the generic driver line reads
  `op_budget=2147483648 op_count_cap=32768 op_cap=32768` for every
  default-configured solve.
- **Byte-identity, scoped to the two demand cases.** The full
  `m2l_demand_constrained` and `m2l_demand_default` lines — `eff_cap`,
  `realized`, `demanded`, `saturated`, `fallback`, the verbatim
  `realized_keys` string and `cells_at_depth` — are **byte-identical** to T1's
  job `f3bPfi66qz4X` across all 21 pairs **in both trees**, diffed line for
  line. Nothing in the realized output moved, including at np 3 and np 6 where
  T1 documented pre-existing run-to-run instability in other tests.
- **The failure direction**, by `m2lOpCountCapBounds`, which needs no solve:
  `default_cap=32768`, a cap of 0 giving `zero_cap_eff=0` (no column, so the
  merge's `ops.size() < 0` admits nothing and every pair takes the overflow
  path), the byte budget still flooring a 32768 cap to `floored_eff=1` at a
  one-column budget, and a negative cap **raising** `std::runtime_error`
  rather than clamping, with the rejected value not stored.

No default moved, and no Canopy signature changed — only the term
`m2l_effective_op_cap()` reads, so none of its callers needed editing.
`with_laplace_solve` gained an optional parameter and every existing call site
is unchanged. **Beatnik's gate is untouched**: T6 adds two cases to Canopy's
own `Canopy_Test_LaplaceSolve_MPI_SERIAL` suite and no Beatnik test of any
tier, so the gate is still five `regression` members and 60 launches.

---

### T7 — Beatnik: plumb the count cap through `FmmParams` — **DONE**

**Depends on:** T6 **DONE**.
**Fill in:** `src/Beatnik_Params.hpp` (a new member beside `:395`, and the
doc comment at `:382-394`); `src/Beatnik_FarFieldInterface.hpp:1068` and the
`local_m2l_op_cap` doc comment at `:294-301`;
`tests/regression_tests/Beatnik_Probe_FmmKeyDemand.cpp` (the ARGUMENTS block at
`:103-110`, the argument parse at `:394-400`, the header printf at `:590-601`).
**Reference:** `m2l_op_table_byte_budget` at `Beatnik_Params.hpp:395` routed at
`Beatnik_FarFieldInterface.hpp:1068` is the exact pattern — no CLI option, no
Python counterpart, reaching one `FmmConfig` member. The Canopy end is
`FmmConfig::m2l_op_count_cap` (`Canopy_Solver.hpp:116`), an `int` defaulting to
32768 and routed to `DownwardSweep::set_m2l_op_count_cap()` at `:217`.
**Do:**
1. Add `int m2l_op_count_cap = 32768;` to `FmmParams`, documented as reaching
   `FmmConfig::m2l_op_count_cap`, with no CLI option, and with the reason it
   exists: under `FarFieldBasis::CartesianTaylor` the keys carry the tree level,
   so occupied depth multiplies the key count and this is the cap that binds.
   Cite the measured level-4 peak from T5.
   **The doc comment states the two edges, because `FmmParams` does not police
   them.** A negative value raises `std::runtime_error` from
   `DownwardSweep::set_m2l_op_count_cap()` during the `Canopy::Solver`
   constructor rather than clamping, and Beatnik lets it — the member has no CLI
   option, so the only route to a negative value is a programmer's literal,
   Canopy already rejects it, and `m2l_op_table_byte_budget` beside it is
   likewise unvalidated here. And **0 is legal**: it admits no column and puts
   every pair on the overflow path. Canopy's test-only `with_laplace_solve`
   convention, where 0 means "leave the config default", is therefore **not**
   copied — this member carries 32768 as its own default and 0 means zero.
2. Route it at `Beatnik_FarFieldInterface.hpp:1068` beside the byte budget.
3. Rewrite the doctrine paragraph at `Beatnik_Params.hpp:382-394`. It currently
   says "the response to realized overflow is a lower `max_depth` or `order`,
   not a smaller table" — true about the *byte budget*, and it must now also say
   that the count cap is the constraint that binds, that it is configurable, and
   what the realized level-4 demand is. Keep the statement that lowering the
   byte budget is the wrong lever. **Its arithmetic is measured now and the
   paragraph's own figures are not the ones that apply**: it reasons from "at
   `order` $\le4$ a column costs at most about 9.8 KB, so the full 32768 keys
   occupy roughly 0.3 GiB", which makes the two constraints look close. At
   CartesianTaylor order 3 a column is 3200 B
   (`Canopy_CartesianTaylorBasis.hpp:506-508`), so 2 GiB buys 671 088 columns
   and the count cap binds by a factor of **20** — it is the only constraint
   that ever binds on this path. Carry that figure, and T5's level-4 peak:
   **37 678 keys worst-observed at HIP np1 step 1650**, 1.150x the 32 768 cap
   and 115 MiB of table, with 17 144 at the worst np4 rank.
4. **Give the probe an optional `argv[2]` carrying `m2l_op_count_cap`**, absent
   meaning the `FmmParams` default. The exit criterion's failure direction
   cannot be driven otherwise — the probe takes one required positional and its
   ARGUMENTS block (`:103-110`) forbids an option surface outright. That
   prohibition is about a step-count override, "a knob that can silently shorten
   a 2000-step run", and a cap override shortens nothing; amend the block to
   permit this one argument on that reasoning and restate the step-count refusal
   as the standing rule it is. Add `op_count_cap=` to the `[t6probe] header`
   line (`:590-601`), beside the `byte_budget=` and `op_cap=` it already prints,
   so a run records which cap was *configured* as well as which was *effective*.
   Reject a negative `argv[2]` through `rec.fail` the way the level is rejected
   at `:401-405`, rather than letting it reach the Canopy throw.
5. Update `FarFieldDiagnostics::local_m2l_op_cap`'s doc comment
   (`Beatnik_FarFieldInterface.hpp:294-301`). It describes the effective cap as
   the smaller of what the byte budget buys and "Canopy's own 32768-key count
   cap"; the count cap is configurable since T6 and reaches Canopy from
   `FmmParams::m2l_op_count_cap`, so the comment must name that member rather
   than a constant.
6. Update `README.md` only if an example's accepted arguments change. They do
   not — this member has no CLI option, and the probe is a measurement driver in
   no tier rather than an example — so confirm and record that rather than
   editing.

**Exit criterion:** `spack install` succeeds and the level-3 FMM member still
passes unchanged at HIP np1 (measured 316 s at T2), demonstrating that a default
`FmmParams` produces the same cap and the same answers. In the failure
direction: `beatnik_exe Beatnik_Probe_FmmKeyDemand_MPI_HIP 3 1024` at HIP np1
prints `op_count_cap=1024` and `op_cap=1024` in its header and non-zero
`global_m2l_fallback` in its rows, where the same binary at level 3 with no
`argv[2]` prints `op_cap=32768` and fallback exactly zero at every row — which
is what proves the knob reaches Canopy rather than being accepted and dropped. A
cap of 1024 binds with certainty at level 3: T3 and T4 measured peak
`unique_ops` there at about 6 400 and demand at 5 938 – 6 624.

**Met.** `int m2l_op_count_cap = 32768;` is in `FmmParams`
(`src/Beatnik_Params.hpp`), routed to `FmmConfig::m2l_op_count_cap` at
`src/Beatnik_FarFieldInterface.hpp` beside the byte budget, with **no CLI
option and no Python counterpart**. `spack install` in the dev env succeeded
**rc 0 in 5 m 47 s** — against a tree trimmed to the two exit-criterion
targets, on the user's explicit instruction to skip the full rebuild, so the
claim is scoped to `Beatnik_Test_Milestone0Fmm_MPI_HIP` and
`Beatnik_Probe_FmmKeyDemand_MPI_HIP`, not to the whole project; the trims are
reverted and no `cmake/` file, no `CMakeLists.txt` and no `tests/CMakeLists.txt`
appears in T7's diff.

**The default direction**, job `f3bZYbR41Y31`
(`scripts/tuolumne/t6_l3_member.flux HIP`, rc 0, 615 s): the level-3 FMM member
passes **`[PASS] Beatnik_Test_Milestone0Fmm (3097/3097 checks)`** at HIP np1 in
**308 s**, the same check count T2 recorded, against the 316 s budget. np4 came
free in the same job and also passed (3097/3097 on rank 0, 2919/2919 on the
other three), `SUMMARY: PASS (2/2 launches)`. A default `FmmParams` therefore
produces the same cap and the same answers.

**The failure direction**, job `f3bZYbHfZ7bM`
(`scripts/tuolumne/t7_cap_knob.flux`, rc 0, 37 s total — 21 s and 16 s), both
launches `174/174 checks`, 81 rows each:

| launch | header | fallback over 81 rows | `unique_ops` | `demand` |
| --- | --- | --- | --- | --- |
| no `argv[2]` | `op_count_cap=32768 op_cap=32768` | **0 in 81 of 81**, `first_exceed_step=-1` | 828 – 5 790 | 828 – 5 790 |
| `argv[2]=1024` | `op_count_cap=1024 op_cap=1024` | **non-zero in 72 of 81**, peak 14 451 at step 400, `first_exceed_step=225` | 828 – **1 024** | 828 – 6 178 |

**Both header fields moved together**, which is the claim: `op_count_cap` is
what `FmmParams` was configured with and `op_cap` is
`DownwardSweep::m2l_effective_op_cap()`, so the second moving proves Canopy saw
the value rather than Beatnik storing it and dropping it. `unique_ops` is
clamped at exactly 1 024 while `demand` still reaches 6 178 — keys are refused,
not un-demanded — and the 72 non-zero-fallback rows are **exactly** the 72 rows
with `demand > op_cap`. The 9 zero-fallback rows are the 9 shallow
`occupied_depths=4` states at steps 0–200, whose demand of 828–864 is genuinely
under 1 024; the `[Canopy] M2L op count exceeded cap` warning appears 72 times,
again exactly the over-cap count.

**The cap's value was not chosen here.** T7 ships 32768, byte-for-byte today's
behaviour; T8 is the task that changes it on T5's measurement. One deliberate
departure from Do step 6: `README.md` gained a `m2l_op_count_cap` row in the
CLI-less `FmmParams` table and a correction to the byte-budget row beside it.
No example's accepted arguments moved — the step's own test is satisfied — but
that table enumerates exactly these public `FmmParams` members, and CLAUDE.md's
README-sync rule covers a public API addition. See `## T7` in the progress log.

---

### T8 — Raise the level-4 count cap to 65536 — **DONE**

**Depends on:** T5 **DONE**, T7 **DONE**.
**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp` — a new
per-level constant in the level-4 arm of the `#if BEATNIK_M0_FMM_LEVEL` block
(`:425` opens the level-3 arm, `:508` the level-4 arm, `:585` the `#endif`),
beside `kP2PFractionBound` (`:468` and `:544`), read by `makeFmmParams`
(`:1027-1032`). **`makeFmmParams` sits outside those arms**, so an unguarded
literal there would move the level-3 member too; the level-3 arm keeps 32768.
Also a new `scripts/tuolumne/t8_cap_raise.flux`. The decision and its evidence
go in the progress log.
**Reference:** the knob is `FmmParams::m2l_op_count_cap`
(`src/Beatnik_Params.hpp:433`), routed to `FmmConfig::m2l_op_count_cap` from
`src/Beatnik_FarFieldInterface.hpp`, with its doctrine paragraph at
`Beatnik_Params.hpp:382-404`. The memory arithmetic is `demand × bytes_per_key`
at 3200 B (`Canopy_CartesianTaylorBasis.hpp:506-508`). The before/after script
shape is `scripts/tuolumne/t7_cap_knob.flux`, which is level-3-and-np1 by
construction and should be copied rather than edited; the level-4 matrix and
binding are in `scripts/tuolumne/t6b_key_demand.flux:154` and `:233-243`.
**The probe takes the cap as an optional `argv[2]`**, so a candidate cap is
measurable at level 4 without any rebuild:
`beatnik_exe Beatnik_Probe_FmmKeyDemand_MPI_HIP 4 65536`. An **empty** `argv[2]`
is a trap — `std::atoi("")` is 0, a legal cap meaning "admit no column" — so a
script must omit the argument rather than pass `""`.

**The measurement that sizes the cap.** Worst-observed demand across three
draws is **37 678 keys at HIP np1 rank 0, step 1650**, 1.150x the 32 768 cap,
with 17 144 at the worst np4 rank and a run-to-run spread of 0.46 % at np1.
`demand_saturated` was never set — 0 of 1944 rows — so that figure is a peak
and not the $2^{20}$ lower bound. **65536** covers it with 74 % headroom,
costs 65536 × 3200 B = **200 MiB** of table per rank, and leaves the byte
budget still not binding by 10.2x: 2 GiB buys 671 088 columns.

**Do:**
1. Confirm the candidate fits before changing any source. Run the probe at
   level 4 on HIP np1 and np4 at `argv[2] = 65536` and at `argv[2] = 32768`,
   the second reproducing T5's reading as the control. No rebuild is needed for
   either.
2. Add the per-level cap constant to the level-4 arm and read it from
   `makeFmmParams`. The level-3 arm keeps 32768, so the level-3 member's
   overflow set is unchanged bit for bit.
3. `spack install`, then confirm the level-3 FMM member still passes unchanged
   at HIP np1 — 308 s and `3097/3097` checks is the measured baseline.
4. Record the number, the three conditions below that it was tested against,
   and the realized P2P fraction at the raised cap.

**What raising the cap does not do.** It cannot drive `p.m2l_fallback == 0`
(`Beatnik_Test_Milestone0Fmm.cpp:1456`). At np4 no rank's demand ever reaches
the cap — `first_exceed_step` is `-1` for all four ranks in all three draws —
and fallback is still non-zero at 71 of 81 states, so **zero** np4 fallback
states are cap-driven; at np1 only about 36 of the 71 are. **T8b owns that
path**, and T9a does not run until it is understood.

**Considered and rejected, so neither is reopened:**

- **Reducing the demand by lowering `max_depth`** from 10
  (`Beatnik_Params.hpp:291`, the lever its comment at `:272-291` names). Demand
  at the measured peak is 1.150x the cap, not the 10x that would make the cap
  the wrong instrument; `demand_saturated` was never set; and the rebuild cost
  a raise pays is bounded by demand rather than by the cap. A shallower tree
  would also move work into the near field, which claim A bounds at
  `kP2PFractionBound = 0.75` (`:544`), and it would not touch the np4 fallback
  either, since no np4 rank reaches the cap.
- **Raising `ncrit`.** It would reduce depth and therefore demand, but
  `ncrit = 8` (`:340`) is near its floor: the liveness inequality
  (`Beatnik_Params.hpp:238-251`) puts the far field's existence at $N\gg840$
  against 2562 vertices, and the default 64 would require $N\gg6720$. Raising
  it buys a cheaper solve that is a direct sum wearing an FMM's name, which is
  the one failure mode claim A's P2P-fraction bound exists to catch. Do not
  take it without a new liveness measurement.
- **A level-blind key.** `canonicalize_key` cannot zero `max_d` for this basis
  — the operator is physical, so two pairs with the same integer offset at
  different levels have different operators and would alias
  (`Canopy_CartesianTaylorBasis.hpp:471-486`). That would be a change to the
  basis's normalization, not to a cap.

**Exit criterion:** the level-4 arm carries the 65536 constant and
`makeFmmParams` reads it, the level-3 arm still reads 32768, and the level-3
FMM member passes unchanged at HIP np1 with `3097/3097` checks. The probe's
level-4 launches at `argv[2] = 65536` report `op_count_cap=65536` and
`op_cap=65536` in the header, **`unique_ops == demand` at all 81 states on
every rank** at both np1 and np4, `first_exceed_step == -1` on every rank,
`demand_saturated` never set, and `keys_built_delta` at the np1 peak state
equal to that state's demand rather than to 32 768. In the failure direction:
the same script's `argv[2] = 32768` launches reproduce T5's reading —
`unique_ops` clamped at exactly 32 768 at the np1 over-cap states and
`first_exceed_step = 900` — so the raised-cap launches are known to be
measuring the change and not a flake. **Fallback is recorded, not asserted on:
it does not reach zero at either rank count, and T8b is why.**

**Met.** `kM2LOpCountCap` is a per-level constant in each arm of the
`#if BEATNIK_M0_FMM_LEVEL` block — **65536** in the level-4 arm beside
`kP2PFractionBound`, **32768** in the level-3 arm — and `makeFmmParams` reads
the name rather than a literal, so the level-3 overflow set is unchanged. The
level-3 FMM member passes unchanged at HIP np1, **308 s and `3097/3097`
checks**, identical to T7's measurement (job `f3baoXyQyE7Z`, rc 0, np4 free in
the same job). The probe matrix (job `f3bafaXSEZT5`,
`scripts/tuolumne/t8_cap_raise.flux`, four launches, 260 s, 810 rows, all
`174/174 checks`) meets every condition: the `argv[2] = 65536` launches report
`op_count_cap=65536 op_cap=65536`, **`unique_ops == demand` at all 81 states on
every rank at both np1 and np4**, `first_exceed_step = -1` on every rank,
`demand_saturated` unset in all 810 rows, and `keys_built_delta` at the np1
peak equal to that state's demand (**37 490**) rather than to 32 768. The
failure direction holds: the `argv[2] = 32768` launches reproduce T5 —
`unique_ops` clamped at exactly 32 768 at the 37 np1 over-cap states and
`first_exceed_step = 900`. T8's own control peaked at **37 846** keys, above
T5's 37 678 at the same step 1650 and within its spread, making 65536 a **73 %**
headroom rather than 74 %; the constant's comment carries both.

Three things the exit criterion did not ask for and a later task needs.
**Fallback was recorded, not asserted on**, and the raise removes **44.3 %** of
np1 fallback *pairs* (301 871 → 168 228 over the 37 over-cap states) while
moving the non-zero *state* count by **zero** — still 71 of 81 at both caps and
both rank counts, which is the observable `assertClaimA` uses. **R4's cost is
+1.3 % of per-evaluation wall overall and +4.5 % at the capped states**, not a
timeout. And **claim B's `SolverParams` (`:991`) still runs at the 32768
default**, so `makeFmmParams`'s "the two claims cannot be at different
configurations" comment is now false in that one field — the instructed scope,
but T9a owns the decision to extend the cap to claim B or to correct the
comment. The installed prefix is left **trimmed** (three HIP binaries, an empty
gate manifest); T9a and T9b must `spack install` the reverted tree first.
`tasks/add-canopy-t6-progress-log.md` `## T8` has the series, the table and the
build failure that cost five minutes.

---

### T8b — Identify the non-cap refusal path at level 4 — **DONE**

**Depends on:** T8 **DONE**.
**Fill in:** `canopy/src/Canopy_DownwardSweep.hpp` (the merge and classify
passes, and the fallback accounting beside `total_fallback_pair_count()`);
`src/Beatnik_FarFieldInterface.hpp` (`FarFieldDiagnostics`, the struct and the
single `readDiagnostics` override);
`tests/regression_tests/Beatnik_Probe_FmmKeyDemand.cpp` (the per-state row);
`canopy/tests/tstLaplaceSolve.hpp`, which carries the exit criterion's
`~profiling` direction. That direction is **not observable from any Beatnik
binary** — this env concretizes `canopy +profiling` since T4 — so it is taken
the way T1 and T6 took theirs: a case in that suite, built in both of T1's cmake
trees (`build-t1-prof-on` and `build-t1-prof-off` under
`/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy`), run at ranks 1-6 in one
`pdebug` job shaped like `canopy/scripts/tuolumne/run_t6_count_cap.flux`.
**Reference:** T1's demand counter is the pattern to mirror for a new
profiling-gated counter — a function-local accumulator that feeds nothing in
the solve, an `m2l_n_*` accessor beside `m2l_n_demanded_ops()`, `-1` as the
"not compiled in" sentinel with `0` a legal count, and one field per counter on
`FarFieldDiagnostics`. **There is exactly one cap-driven refusal site.**
`m2l_effective_op_cap()` is called once, at `Canopy_DownwardSweep.hpp:1478`,
into the local `effective_op_cap`, and that single value feeds the merge's
admit test `ops.size() < effective_op_cap` (`:1806-1826`), whose `else` branch
assigns `g = -1` and emits the once-per-build `[Canopy]` warning. T8 already
raised that threshold, so it is the one reason a measurement at 65536 should
no longer see.

The second site is the classify pass's range guard (`:1631-1636`), which leaves
`local_op = -1` when `max_d` falls outside $[0,\texttt{max\_depth}]$ or any of
`dd`, `ii`, `jj`, `kk` exceeds its bound. Those bounds are **hard**:
`dd` over $[-6,6]$ is `KernelType::m2l_key_dd_max`
(`Canopy_CartesianTaylorBasis.hpp:469`) and `ii,jj,kk` over $[-32,32]$ is
`M2L_KEY_OFFSET_MAX` (`Canopy_DownwardSweep.hpp:570`) — representability
limits, not budgets, and the obvious candidate. Both kinds of `-1` meet at the
remap (`:1845-1852`), where a pair with `lo < 0` is a range-guard refusal and a
pair with `lo >= 0` whose `l2g[lo] < 0` is a cap refusal; that single site is
where the two reasons are still distinguishable. T5's per-rank series is the
data the breakdown must reconcile against.
**Do:**
1. Enumerate **every** site that routes a pair to the per-pair fallback rather
   than to an operator column, by reading the classify and merge passes. Name
   each with its file and line and say what condition it tests. Cover the
   fallback-table assembly at `Canopy_DownwardSweep.hpp:2194-2220` as well: a
   refused pair whose `pair_target_depth` falls outside $[0,\texttt{max\_depth}]$
   is `continue`d there and placed in **neither** an operator column nor the
   fallback table, so it is invisible to `total_fallback_pair_count()`. If that
   ever fires it is a dropped pair — a wrong velocity, not a slow one — and
   step 2's sum identity is what catches it.
2. Add one profiling-gated counter per distinct reason, each counting *pairs*
   rather than keys, so the counters sum to `total_fallback_pair_count()`
   exactly. Assert that identity rather than inspecting for it.
3. Surface them through `FarFieldDiagnostics` and print them in the probe's
   per-state row beside `global_m2l_fallback_pair_count`. **The counters join
   the existing reduction**, unlike T1's demand counters, which are rank-local
   and unreduced: `global_m2l_fallback_pair_count` is summed across ranks by the
   five-element `MPI_Allreduce` at `src/Beatnik_FarFieldInterface.hpp:1494-1516`,
   so a breakdown that did not travel with it would be a per-rank number printed
   beside a global one and the sum identity would fail at every rank but one.
   Extend that reduction rather than adding a second. The asymmetry against the
   demand fields is deliberate — the identity the exit criterion checks is
   between two global figures.
4. Measure at level 4 on HIP np1 and np4 at the post-T8 cap, and record which
   reason accounts for the np4 fallback at all 71 states and for the np1
   states whose demand is under the cap.
5. **Do not change the key encoding or any bound.** If the dominant reason is a
   representability limit rather than a budget, that is a finding and a new
   task at the basis's key encoding, not an edit here.

**Additional information needed:** whether the dominant reason is reachable
from a Beatnik-side configuration at all. Step 1 answers it; until then the
remedy cannot be designed and this task does not attempt one.

**Exit criterion:** the probe's level-4 rows at HIP np1 and np4 carry a
per-reason fallback breakdown whose counters **sum exactly** to
`global_m2l_fallback_pair_count` at every one of the 81 states, and the
progress log names the reason that accounts for the np4 fallback with its
per-state counts. In the failure direction: in a `~profiling` build every
per-reason counter reads `-1` and not `0`, and the identity check is skipped
rather than passing vacuously on a row of sentinels (**R7**).

**Met.** Job **`f3bbMmxgc4ZD`** (`scripts/tuolumne/t8b_fallback_reasons.flux`,
`-q pdebug -t 15m`, 132 s of job wall, `SUMMARY: PASS (2/2 launches)`) ran the
probe at level 4 on HIP at np1 and np4, both at the post-T8 cap **65536**,
against `canopy@develop+profiling`. `fallback_breakdown_available=1` in both
headers and **0 of 405 rows carry a sentinel**, so every figure is a
measurement. **The identity holds at all 405 rows** — 81 np1 states and 324 np4
(state, rank) pairs — checked both by the probe's own assertion
(`337/337 checks` on all five rank reports, up from T8's 174 by the 162
per-state identity checks and the one availability check) and independently by
re-deriving it from the logged rows.

**The reason is the classify pass's RANGE GUARD, and it accounts for 100 % of
the fallback at both rank counts.** `fb_count_cap_total = 0` and
`fb_range_guard_total = fb_total` at np1 (**215 302** pairs) and at np4
(**215 742** pairs), with `fb_count_cap > 0` in **0 of 405 rows** and
`fb_dropped = 0` everywhere — no pair is placed in neither table, so nothing is
silently dropped. Every reconciliation T5 and T8 set holds: **71 of 81** states
non-zero at both rank counts, the ten zero-fallback states exactly steps 0
through 225 at `occupied_depths` 5 or 6, first fallback at step 250, and the
np1 total within 0.5 % of T8's 216 288 at the same cap. **The dominant reason
is a representability limit of the basis's key encoding, not a budget**:
`M2L_KEY_DD_MAX` and `M2L_KEY_OFFSET_MAX` were not changed, and no cap value
reaches this path. The fallback peak is 6 884 pairs at step 1550 at np1 against
6 832 at np4 — essentially rank-count-independent, as a geometric limit should
be and a per-rank budget should not.

The `~profiling` direction was taken in **Canopy**, job **`f3bbHk4tFmkK`**
(`canopy/scripts/tuolumne/run_t8b_fallback_reasons.flux`), in both of T1's
cmake trees at ranks 1-6: `100% tests passed, 0 tests failed out of 6` in each.
The new `m2lFallbackReasonBreakdown` case reads all three counters as **`-1`
and not `0`** at 21 of 21 `(nprocs, rank)` pairs in the `OFF` tree with the
identity **skipped**, and `fb_range_guard + fb_count_cap == fallback` with
`fb_dropped = 0` at 21 of 21 pairs in the `ON` tree (**R7**).

---

### T9a — Measure level-4 claim A at zero cap-driven refusal, before any long job — **DONE**

**Depends on:** T8 **DONE**, T8b **DONE**.
**Fill in:** no source changes. A new `scripts/tuolumne/t9a_l4_member.flux`
copied from `scripts/tuolumne/t6_l3_member.flux`, running
`Beatnik_Test_Milestone0FmmL4_MPI_HIP`; the progress log.
**Reference:** `kTauA = 1.0e-3` (`Beatnik_Test_Milestone0Fmm.cpp:320`), checked
at `:1505-1506`; the contaminated peaks T0 recorded; the fallback/table path
agreement of 3.27e-15 (`canopy/tasks/tree-opt-progress-log.md` `## C1`);
`scripts/tuolumne/t8b_fallback_reasons.flux` (untracked, present) for the
per-reason probe matrix. Measured member costs at HIP: **2 450 s at np1 and
1 598 s at np4** (`tasks/canopy/add-canopy-progress-log.md`, T6 cost table) —
67 minutes together, over `pdebug`'s 60-minute cap.

**The pure-path condition is zero cap-driven refusal, not zero fallback.** The
range guard's fallback is present at 71 of 81 level-4 states and stays (see
[Problem](#problem)); it evaluates the same mathematics as the table to
3.3e-15 and cannot move a 1e-3 error. A pair refused by the **cap** is likewise
the same mathematics, but a cap-driven refusal means the configuration is not
the post-T8 one, so it is the contamination this task rules out.

**Do:**
1. `spack install` the dev env (the prefix is trimmed; see [Current
   state](#current-state)). This is the first Beatnik build against the
   ParMETIS partitioner. Record the canopy commit and branch, and the beatnik
   commit, in the log.
2. Run the probe at level 4, HIP np1 and np4, cap 65536, with
   `t8b_fallback_reasons.flux` unchanged. Confirm `fb_count_cap == 0` at all
   81 states at both rank counts and that the per-reason identity holds.
   Record `fb_range_guard`, demand and `unique_ops` against T8b's, as a new
   draw on the changed Canopy, not a reproduction.
3. Run `Beatnik_Test_Milestone0FmmL4_MPI_HIP` at np1 and at np4 as **two
   separate `pdebug` submissions** of the new runner. The member reports
   **FAIL by construction** on `p.m2l_fallback == 0` (`:1500`, 71 states) and
   claim B's final-state check (`:1967`); read the per-state claim-A lines,
   never the summary alone.
4. Record the worst claim-A relative error at 17 digits with its step and the
   realized P2P fraction, at both rank counts, and state whether it is under
   `kTauA`. Record claim B's cost too: this is the first claim-B run since T8,
   and it still runs at cap 32768.
5. Record that T9b, not this task, replaces the member's zero-fallback checks
   and gives claim B `kM2LOpCountCap`.
6. **If the error exceeds τ_A, stop here and do not submit T9b.** That is
   **R9**: a finding about the expansion at level 4 and a new task, at `order`,
   `mac_theta` or the bound's derivation — never a wider τ_A.

**Exit criterion:** the log carries the level-4 worst relative error at 17
digits, with step and P2P fraction, from the member's HIP np1 and np4 launches,
and states whether each is under `kTauA`; and the probe's rows at HIP np1 and
np4 at cap 65536 show `fb_count_cap == 0` at all 81 states with the per-reason
identity holding. In the failure direction: a non-zero `fb_count_cap` at any
state means the run was cap-contaminated, and its error is not the number.

**Met — and the error is OVER τ_A, so T9b is blocked by R9.** Built with a
full, untrimmed `spack install` of the dev env (rc 0, 660 s) against canopy
branch `investigate-m2l-cap` at `38658ad` and beatnik `7b27cb0` (one untracked
runner, no source change). It is the first Beatnik build of Canopy's ParMETIS
partitioner, and it compiled cleanly on the first attempt. Probe job
`f3cx8dG3H7MZ`
(`t8b_fallback_reasons.flux`, unchanged): `fb_count_cap == 0` and
`fb_range_guard + fb_count_cap == global_m2l_fallback` in **405 of 405** rows
(81 at np1, 4×81 at np4), re-derived from the rows as well as asserted
(`337/337` on all five rank reports), `fb_dropped = 0` everywhere. Member jobs
`f3cx8dQGf86b` (HIP np1) and `f3cx8dYQ7BiF` (HIP np4), each a separate `pdebug`
submission of the new `scripts/tuolumne/t9a_l4_member.flux`. The member's own
`[Canopy Diagnostics]` lines also read `fb_count_cap=0` at `effective_cap=65536`
in every claim-A evaluation. **Worst claim-A relative error:
`1.2536745760757648e-3` at np1 and `1.2473681315787063e-3` at np4, both at step
1375**, with realized P2P fraction `0.337366` and `0.339265` there. Both are
over `kTauA = 1.0e-3`, by 1.254x and 1.247x. Every failed check is at `:1500`,
`:1505`, `:1506` or `:1967` (np1: 71+5+5+1 = 82; np4: 71+4+4+1 = 80 per rank).
There were no walltime kills, and all 81 claim-A states are present at both
rank counts. Per Do step 6, T9b was not submitted and τ_A was not touched.

---

### T9r — Measure the reference treecode's own error on the member's 81 states — **DONE**

**Depends on:** T9a **DONE**.
**Fill in:**
- a new `tests/regression_tests/reference_treecode_error.py`, carrying the
  project's BSD-3-Clause header in `#` style as
  `tests/regression_tests/fmm_divergence_ladder.py` does;
- a new `scripts/tuolumne/t9r_reference_treecode.flux`, its `pdebug` runner,
  with the preamble and provenance echo copied from
  `scripts/tuolumne/t9a_l4_member.flux` (one node, one task, no GPU);
- the progress log.

No C++ changes, and nothing in the reference repository changes.
**Reference:**
- The reference package `zmodel3d` at
  `~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/`, which this task
  reads and never edits. Record its commit in the log (`ec7d7bf` when this
  task was written).
- `potential_mesh_birkhoff_rott_velocity(state, params)`
  (`zmodel3d/mesh_solver.py:804-842`) on a `MeshPotentialZModelState(vertices,
  faces, potential)` (`:122-200`). It dispatches on `params.br_approximation`
  to `_source_velocity_direct_unsigned` (`:437-455`) or
  `treecode_velocity_unsigned` (`zmodel3d/treecode.py:96-130`) through
  `_mesh_birkhoff_rott_velocity_from_sources` (`mesh_solver.py:388-435`).
- The precedent measurement in `tasks/treecode.md` §1: the same function, the
  same `relmax`. The member's gold set,
  `tests/regression_tests/milestone0-sub4-2000-steps/gold/`, holds 81 step
  files whose `vertices`, `faces` and `potential` arrays are exactly that
  state.

**Do:**
1. Write the script. For each of the 81 `checkpoint_*_stepNNNNNNN.npz` files,
   build `MeshPotentialZModelState` from `vertices`, `faces` and `potential`.
   Evaluate the velocity twice with `MeshZModelParams(eps=0.025,
   use_matlab_blob=False, source_quadrature="vertex", br_approximation=...)`:
   once at `"direct"`, and once at `"treecode"` with the reference's own
   defaults (`br_treecode_theta=0.3`, `br_treecode_order=2`,
   `br_treecode_ncrit=64`, `mesh_solver.py:52-54`).
   - **Set those three explicitly** and print them, so a later change to the
     reference's defaults cannot silently move the measurement.
   - `use_matlab_blob=False` is the gold set's `--kernel-blob-mode length`,
     with blob $=\varepsilon^2$ (`mesh_solver.py:394`), which is the member's
     softening of 0.025.
   - Error per state: `max_i |u_tree[i] - u_direct[i]| / max_i |u_direct[i]|`,
     with row 2-norms. That is claim A's quantity exactly
     (`Beatnik_Test_Milestone0Fmm.cpp` `fieldDifference` `:924-958` and
     `fieldScale` `:898-912`).
   - Step 0 has a zero field (`potential` is identically 0). Report it in
     absolute form and exclude it from the worst, as the member does.
   - Import `zmodel3d` by `PYTHONPATH` from the runner and never by copying;
     the script fails loudly if the import or any of the 81 files is missing.
   - Print one `[t9r] row step=... time=... rel=... max_direct=...` line per
     state at 17 digits, then a `[t9r] worst` line.
2. Run it in `pdebug` through the runner, with `/usr/tce/bin/python3` (NumPy
   2.1.2) and `PYTHONPATH` set to the reference repository root. The direct sum
   at 2562 sources is seconds per state. Budget `-t 30m`, and record the
   measured wall.
3. **Calibrate before reading.** At step 25, the reference's error must be the
   same order as `tasks/treecode.md`'s 2562-source, order-2, θ 0.3 figure of
   `4.8e-4`, which was measured on a smooth initial state. A figure off by
   10x or more means the parameters or the error definition do not match.
   Stop, and record no verdict.
4. Tabulate the reference's error beside T9a's HIP np1 and np4 series, state
   by state, and record the worst of each with its step.
5. **Apply the decision rule and record the verdict.** Let `E_ref` be the
   reference's worst error over the 80 non-zero-field states, and let `τ_ref`
   be `E_ref` rounded **up** to two significant digits.
   - **Verdict "order 3 matches the reference"** if `τ_ref > 1.0e-3` and T9a's
     worst errors (`1.2536745760757648e-3` at np1, `1.2473681315787063e-3` at
     np4) are both `<= τ_ref`. Then τ_A's stated basis, "the reference
     implementation's own fidelity", was measured only at smooth states, and the
     reference itself exceeds 1e-3 at the roll-up. Production stays at order 3,
     matching the reference. **T9e** is the next task, and T9c and T9d are not
     done.
   - **Verdict "Beatnik is less accurate than the reference"** otherwise: either
     the reference meets 1e-3 at every state, or Beatnik at order 3 exceeds
     `τ_ref`. **T9c** is the next task, then T9d. T9e is not done.

**Exit criterion:**
- The log carries the reference's error at all 81 states at 17 digits, with
  each state's simulation time.
- The step-25 calibration is within 10x of `4.8e-4`.
- `E_ref` is recorded with its step, along with `τ_ref`.
- Exactly one verdict is recorded under step 5's rule, naming the next task.

In the failure direction: a calibration outside 10x, fewer than 81 rows, or a
missing printed parameter means no verdict.

**Met — verdict "order 3 matches the reference"; T9e is next.**
- **Job:** `f3d7BbCHyPu1` (`pdebug`, 76 s of measurement), rc 0, 81 `[t9r]
  row` lines.
- **Provenance:** reference `ec7d7bf` with a clean `zmodel3d/`;
  `/usr/tce/bin/python3` 3.13.2 and NumPy 2.1.2.
- **Parameters:** all printed — `eps` 0.025, `use_matlab_blob` False, `vertex`
  quadrature, θ 0.3, order 2, `ncrit` 64.
- **Calibration:** step 25 reads `5.8288877682024988e-4`, 1.21x of `4.8e-4`.
- **Worst error:** `E_ref = 1.4987010690098229e-3` at step 1550 (`t =
  1.7873730821831051`), so `τ_ref = 1.5e-3`. The reference is over 1e-3 at 19
  of 80 states.
- **Rule:** `τ_ref > 1.0e-3`, and T9a's `1.2536745760757648e-3` (np1) and
  `1.2473681315787063e-3` (np4) are both `<= τ_ref`.
- **Note:** the reference's worst state (1550) is not Beatnik's (1375). The
  per-state table is in the progress log under `## T9r`.

### T9e — Re-derive τ_A from the reference's measured fidelity; production stays at order 3 — **NOT STARTED**

**Depends on:** T9r **DONE** with verdict "order 3 matches the reference".
**Fill in:**
- `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp`: `kTauA` (`:320`)
  and its provenance comment (`:280-319`). The literal is shared by both
  levels.
- `README.md` "FMM accuracy (measured)" (`:387`) and the order paragraph
  (`:317-328`).
- `CLAUDE.md:107` and `docs/testing.md:33`, which state the bound as `1e-3`.
- `src/Beatnik_Params.hpp`, the `order` comment (`:196-228`): order 3 is the
  reference's accuracy class, now measured at the roll-up and not only at a
  smooth state.
- The progress log.

**Reference:** T9r's per-state table and `τ_ref`.
**Additional information needed:** none beyond T9r's verdict. If T9r shows
the reference's worst state differs from Beatnik's (step 1375), record that,
because it bears on R11.
**Do:**
1. Set `kTauA = τ_ref`. Rewrite its comment:
   - the bound is the reference treecode's own worst error (order 2, θ 0.3,
     `ncrit` 64) over the member's 81 level-4 states, measured by T9r;
   - it is not a figure fitted to Beatnik's output;
   - give Beatnik's own worst error beside it, and its margin;
   - keep the qualification list.
2. Keep `kProductionOrder = 3` and every other tolerance unchanged. Negative
   case 2 (`kPerturbationFactor`) scales with `kTauA` and needs no edit;
   confirm that it still fails as intended.
3. Update the README, `CLAUDE.md` and `docs/testing.md` statements of the bound.
4. `spack install` the dev env, and edit nothing while it runs. Then run
   `scripts/tuolumne/t6_l3_member.flux HIP`, and
   `scripts/tuolumne/t9a_l4_member.flux` at np1 and at np4.

**Exit criterion:**
- The level-3 member reports `[PASS]` at HIP np1 and np4.
- The level-4 member at HIP np1 and np4 has **zero** failed checks at `:1505`
  and `:1506`, and every failed check is at `:1500` or `:1967`, which T9b step
  0 replaces.
- Negative case 2 still reports its perturbed state rejected.

In the failure direction: a level-4 state over the new `kTauA` means T9r's
verdict rested on a draw that did not reproduce. Stop and record it; do not
round `τ_ref` up further.

---

### T9c — Measure the claim-A error against `order` and `mac_theta` at the worst states, and at level 5 — **NOT STARTED**

**Depends on:** T9r **DONE** with verdict "Beatnik is less accurate than the
reference".
**Fill in:** no source change. A new `scripts/tuolumne/t9c_fmm_scan.flux`,
copied from `scripts/tuolumne/t5_fmm_scan.flux` (preamble, provenance echo and
rank-to-GPU binding unchanged), running `Beatnik_Test_FmmScan_MPI_HIP` once per
`(level, spin-up)` entry below; the progress log.
**Reference:** `Beatnik_Test_FmmScan.cpp` (arguments `:91-97`, scan axes
`:270-330`, `[t5arm]` line format `:106-110`); T9a's claim-A series in the log;
the physical times of the level-4 states over τ_A, read from the gold set's
filenames in `tests/regression_tests/milestone0-sub4-2000-steps/gold/`: step
1000 at `t = 1.574929`, 1350 at `1.703397`, 1375 at `1.713585`, 1475 at
`1.755355`.
**Do:**
1. Confirm the installed prefix has `Beatnik_Test_FmmScan_MPI_HIP`
   (`beatnik_exe` resolves it). If any Beatnik or Canopy source has changed
   since T9a's install, `spack install` the dev env first.
2. Level 4, HIP np1, one launch per spin-up of 1000, 1350, 1375 and 1475 steps,
   the four steps over τ_A at both rank counts in T9a; and spin-up 1375 at HIP
   np4. Each launch prints `simulation time`. It must match the gold time above
   to 1e-6, or the scan is not on the member's trajectory.
3. **Cross-check before reading any arm.** At spin-up 1375 the background arm
   (`order` 3, `ncrit` 8, θ 0.3) must reproduce T9a's
   `1.2536745760757648e-3` (np1) and `1.2473681315787063e-3` (np4) within 1 %,
   the size of the draw spread across T0's and T9a's six level-4 readings. A
   larger gap means the scan is not measuring the member's state. Stop, and
   record no verdict.
4. Level 5, HIP np1. Level 5's `initial_min_edge` is about half level 4's and
   the adaptive dt scales with it, so the same physical time takes about twice
   the steps. Run at spin-up 2750, read the printed time, and if it is outside
   `[1.703397, 1.755355]` rescale the step count once by the time ratio and run
   again. Direct spin-up at level 5 costs about 16x level 4 per step, roughly
   7 minutes at HIP np1, inside `pdebug`. Record the arms at that state.
5. For every launch, record each arm of the `order` axis (0, 2, 3, 4, 5) and
   the `theta` axis (0.2, 0.3, 0.4, 0.5, 0.7): `grad_rel` at 17 digits,
   `p2p_frac`, `ops` and `op_cap`. An arm with `ops == op_cap` is cap-saturated.
   Flag it, and still read its error, because a refused pair evaluates the same
   mathematics to 3.27e-15 (`canopy/tasks/tree-opt-progress-log.md` `## C1`).
6. Record the N-scaling point: the background arm's `grad_rel` at level 5
   divided by level 4's at spin-up 1375.
7. **Apply the decision rule and record the verdict:**
   - **order 4** if the `order` 4, θ 0.3 arm's `grad_rel` is at most
     `kTauA / 2 = 5.0e-4` at every level-4 state **and** at the level-5 state;
   - else **θ** if, at `order` 3, the largest scanned θ below 0.3 has
     `grad_rel <= 5.0e-4` and `p2p_frac < kP2PFractionBound = 0.75` at every
     one of those states;
   - else **neither**: R9 stands with both levers measured.

   The 2x margin is set at the worst measured state, because T5's 2.0x margin
   at a single early state did not cover the trajectory.
**Exit criterion:** the log carries, for the five level-4 launches and the
level-5 launch, the printed simulation time, every `order`- and `theta`-axis
arm's `grad_rel` at 17 digits with `p2p_frac` and `ops`/`op_cap`, the step-3
cross-check within 1 %, the level-5/level-4 ratio, and exactly one verdict
under step 7's rule. In the failure direction: a cross-check outside 1 % or a
level-5 time outside the window means the corresponding figure is not reported
as a measurement of the member's state, and no verdict is given.

### T9d — Make `order` 4 Beatnik's production default, and re-derive claim B at order 4 — **NOT STARTED**

**Depends on:** T9c **DONE** with verdict **order 4**; every task in
`canopy/tasks/02_oracle_extension.md` (O1, O2, O3) **DONE**, on a Canopy commit
the clone is moved to before this task's `spack install`.
**Fill in:**
- `src/Beatnik_Params.hpp`: `order = 4` (`:229`), with its doc comment
  (`:196-228`) rewritten around T9c's worst-state measurement and Canopy's
  validation. The `mac_theta` comment (`:177-193`) already says 0.5 "would need
  order 4"; check that it still reads true.
- `README.md`: the order default at `:210` and `:314-328`, the production-order
  statement and curve at `:425-442`, the byte-budget row at `:350` (3200 →
  9800 B per column), and the validated parameter set.
- `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp`:
  `kProductionOrder = 4` (`:345`); τ_A's qualification list and "Where the value
  comes from" (`:295`, `:301-316`) rewritten to cite T9c, with `kTauA` itself
  unchanged; `kHorizonEnvelope` in both arms (`:485`, `:591`); and
  `kFmmVolumeDriftRtol` (`:377`) with its derivation comment (`:351-376`), set
  from this task's runs.
- `CLAUDE.md:106` and `docs/testing.md:31`, which say the FMM members run at
  `order = 3`.
- a new `scripts/tuolumne/t9d_divergence.flux`, copied from
  `scripts/tuolumne/t5_divergence.flux`, driving `Beatnik_Test_Milestone0Run`
  at `argv[4] = fmm`, `argv[5] = 8`, `argv[6] = 4`; and the progress log.

**Reference:** `tasks/canopy/add-canopy.md` T5 step 7, the procedure that
produced claim B's current envelope and drift bound; R8 there: more than one
FMM-driven run, the envelope from the earliest observed horizon.
**Do:**
1. Move the Canopy clone to the commit carrying O1–O3 and record it.
   `spack install` the dev env, with `HIPCC_*_FLAGS_APPEND` cleared, and edit
   nothing while it runs.
2. Change the default and its documentation (Params, README, `CLAUDE.md`,
   `docs/testing.md`). No other caller reads the default; see [Current
   state](#current-state).
3. **Budget before sweeping.** Run `t9d_divergence.flux` at a reduced step
   count at level 4 HIP np1 to measure the order-4 FMM-driven per-step cost.
   Claim B at order 3 is 1.14042 s/step there (T9a). Project each 2000-step
   run, and the whole milestone tier, from that measurement; see R12. A run
   that does not fit `pdebug` goes to `pbatch` at a `-t` set from the
   projection, and the log says so.
4. Run 2000-step FMM-driven trajectories at order 4: at level 4, two at HIP np4
   (which vary the partition) and one at HIP np1; at level 3, two at HIP np1.
   Measure each against its level's gold set through `fmm_divergence_ladder.py`
   and `milestone0_ladder.py`, as T5 step 7 did, and record per-rung horizons
   and the worst volume-drift deviation from `kRefVolumeDrift`.
5. Set `kHorizonEnvelope` per level from the earliest observed horizon per rung.
   Set `kFmmVolumeDriftRtol` as the smallest one-significant-digit value giving
   at least 2x margin over the worst deviation at both levels, the margin T6
   took. Write the runs and jobs into each comment. `kTauA`, claim A's gold rung
   and `kVolumeDriftRtol` do not move.
6. Set `kProductionOrder = 4` and rewrite τ_A's provenance.
7. Check the table size: at 9800 B per column, the level-4 cap of 65536 is
   642 MB per rank and level 3's 32768 is 321 MB, both under the 2 GiB budget,
   so the count cap still binds. Read `bytes_per_key=9800` and
   `fb_count_cap=0` off the `[Canopy Diagnostics]` lines in step 8's logs.
8. `spack install` again, and edit nothing while it runs. Then run
   `t6_l3_member.flux HIP`, and `t9a_l4_member.flux` at np1 and at np4.
9. Confirm the gate is untouched: no source in `BEATNIK_REGRESSION_TEST_SOURCES`
   (`tests/CMakeLists.txt:256-266`) selects the FMM. Grep for
   `BRApproximation::Fmm` and record the empty result. The gate stays five
   members and 60 launches, and is not re-run.

**Exit criterion:** `FmmParams{}.order == 4`. The level-3 member reports
`[PASS] Beatnik_Test_Milestone0Fmm` at HIP np1 and np4. The level-4 member at
HIP np1 and np4 has **zero** failed checks at `:1505`, `:1506` and the
`kProductionOrder` check, and every failed check is at `:1500` or `:1967`,
which T9b step 0 replaces. Its worst claim-A error is recorded at 17 digits
with step and P2P fraction, under `kTauA`, at both rank counts. The log carries
the order-4 per-step cost and the projected tier total. In the failure
direction: a claim-A state over τ_A at order 4 is R10. Stop and record it; do
not widen τ_A. A level-3 horizon failure means step 5's envelope was set from
too few runs.

---

### T9b — Re-run the milestone tier; close T6 — **NOT STARTED**

**Depends on:** T9d **DONE** or T9e **DONE**, whichever T9r's verdict selects,
with the level-4 member's claim-A error under `kTauA`.
**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone0Fmm.cpp` (the
purity checks at `:1500` and `:1967`, claim B's `p.fmm` near `:1029`, and the
`makeFmmParams` comment at `:1062-1064`);
`scripts/tuolumne/run_milestone.flux` (`-t` only, and only after the run);
`tasks/canopy/add-canopy.md` (T6 status, and every statement of the τ_A bound
— see step 4a); both progress logs.
**Reference:** the runner's own header comment says how to set `-t` from a tier
run. The gate is untouched: five `regression` members, 60 launches on tuolumne.
**Do:**
0. Make the member assert what level 4 can satisfy. Replace
   `p.m2l_fallback == 0` (`:1500`) and
   `diag.global_m2l_fallback_pair_count == 0LL` (`:1967`) with an ungated
   no-cap-refusal check — `local_m2l_unique_op_count < local_m2l_op_cap` on
   every rank — since the per-reason counters read `-1` in a `~profiling`
   build. Give claim B's `p.fmm` `kM2LOpCountCap` so both claims run one
   configuration, and correct the `makeFmmParams` comment to match. Check
   counts may re-baseline; the level-3 member must pass again at HIP np1
   before the tier is submitted.
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

   a. On the T9e path, restate τ_A in `tasks/canopy/add-canopy.md` to match the
      compiled `kTauA = 1.5e-3`. Its basis is the reference treecode's worst
      claim-A error over the level-4 member's 80 non-zero-field states,
      `E_ref = 1.4987010690098229e-3` at step 1550 (T9r). That doc states the
      bound as $10^{-3}$, or argues from that value, at `:84`, `:92`, `:247`
      (the "Why not tighter than $10^{-3}$" heading and its section), `:425`,
      `:2089`, `:2123-2127` and `:2200`. Read each, and also its other τ_A
      mentions (`:8`, `:182`, `:419-421`, `:465`, `:1412`, `:1456`, `:1668`).
      A statement of the bound or of its basis changes. Historical
      measurements stay exactly as recorded: T5's `5.0e-4` figures, the
      reference's `4.8e-4` and `1.6e-3` smooth-state readings, and the order
      curve. `:84-92` justifies `1e-3` as the reference's smooth-state
      fidelity. Keep that measurement and add that T9r measured the reference
      at the roll-up, where it reaches `1.4987e-3`. On the T9d path `kTauA`
      stays `1e-3` and this step is a no-op.
5. Confirm the gate is unchanged and say so: `regression` still has five members
   and 60 launches on tuolumne.

**Exit criterion:** `scripts/tuolumne/run_milestone.flux` reports
`SUMMARY: PASS (16/16 launches)`; `run_milestone.flux`'s `-t` is set from that
run's measured total; `tasks/canopy/add-canopy.md` shows T6 **DONE** with the
measured cost, and — on the T9e path — states τ_A as `1.5e-3` with its T9r
basis. That doc's state is checked with
`grep -nE 'tau_A\$? *(\\le|\\approx|is) *\$?10\^\{-3\}' tasks/canopy/add-canopy.md`
(before this task it hits `:84`, `:92`, `:425`, `:2089`, `:2123` and `:2200`, all in
current-design text: the fidelity target, the milestone members, X1 and Known
risks). Afterwards it returns nothing. Finally, the gate runner
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
$[-32,32]$ (`Canopy_DownwardSweep.hpp:570`) — but that product is about 39
million keys at `max_depth` 10, far more than fits. `M2L_DEMAND_COUNT_CAP` at
$2^{20}$ bounds the set at roughly 56 MB against the level-3 member's measured
peak RSS of 1 060 488 kB. Presentation if the bound is hit:
`demand_saturated` set, and the reported count is a lower bound, not a peak.
T5 measured the flag unset in 0 of 1944 rows across four complete draws, so the
bound never bound and the figure T8 sizes from is a peak. A later measurement
that does set it is reporting a lower bound and must not be read as a peak.

**R3 — Demand is measured but the peak is not where the error peaks.** Four
steps are distinct and T5 measured all four: fallback first becomes non-zero at
step 250, demand first exceeds the cap at step 900, claim A's error peaks at
step 1375, and demand and the fallback pair count both peak at step 1650.
Demand at step 1375 is 97.2 % of the peak, so a cap sized from the error peak
would be short by about 1 050 keys at step 1650. T8 sizes from the demand peak
at step 1650 for that reason; any later resizing must do the same.

**R4 — The raised cap's rebuild cost destroys the long run.** On a
`key_needs_level` basis with a drifting bounding box the operator cache retains
nothing between evaluations (`Canopy_DownwardSweep.hpp:406-416`), measured at
`keys_built_delta == unique_ops` in 405 of 405 rows at both levels. The raise
is therefore bounded — rebuilt columns are `min(demand, cap)`, so at most
1.150x at the level-4 demand peak and nothing below it — but bounded is not
free, and the cost presents as a *timeout*, not as a wrong answer. A timeout in
a `pbatch` job is expensive to diagnose. Distinguishing measurement:
`local_m2l_op_keys_built`'s increment per evaluation, printed by the probe at
every state, read beside the per-evaluation wall time. T8's exit criterion
pins the increment to demand rather than to the cap, and T9a re-measures at
`pdebug` scale before T9b is submitted.

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

**R9 — τ_A fails on the expansion at level 4.** Confirmed by T9a: with
`fb_count_cap == 0` at every state, the worst level-4 claim-A error is
`1.2537e-3` (np1) and `1.2474e-3` (np4) at step 1375. The fallback cannot
account for it, since the fallback and table paths agree to 3.3e-15 (Canopy
C1). **Response:** T9r measures the reference treecode on the same states. Its
verdict selects T9e (τ_A re-derived from the reference, order 3 kept) or T9c
and T9d (production `order` raised). τ_A is never widened to fit Beatnik's own
number.

**R10 — Order 4 does not clear τ_A at the roll-up.** Unlikely: order 4 bought
8.9x at T5's early state, and 1.25x is needed. **Presents as:** T9c's `order` 4
arm above `5.0e-4` at a level-4 state, or a T9d claim-A state over τ_A.
Without Canopy's oracle at $|k|=8$, a recurrence defect at degrees 7–8 and a
genuine truncation plateau at the roll-up look identical. That is why
`canopy/tasks/02_oracle_extension.md` gates T9d. **Distinguishing
measurement:** O1's ladder deviation at degrees 7 and 8. If O1 passes and the
gain is still small, the cause is geometry. T9c's rule then selects θ, and
T9d's text no longer describes the work and must be rewritten before it
starts.

**R11 — The error grows with N at fixed `order` and `mac_theta`.** The design
assumes it is flat in N, set by the MAC ratio and the roll-up geometry. That is
reasoned, not measured. **Presents as:** T9c's level-5/level-4 ratio well above
1 at matched physical time. A production default sized at level 4 would then
not be sized for AMR. The T9c rule already demands margin at level 5, so this
risk fails T9c's verdict rather than passing silently. Record the ratio either
way.

**R12 — Order 4's cost breaks the milestone tier's walltime.** Claim B is 76–98
% of every FMM launch, and at level 4 the downward sweep (M2L) is about 89 % of
an evaluation; order 4 makes each M2L about 3x dearer. T6's tier ran 31 266 s
at order 3 against `pbatch`'s 86 400 s ceiling (`-t 1440m`). A uniform 2.5x on
the FMM members' 28 832 s would take the tier to roughly 75 000 s, near the
ceiling, and SERIAL np4's level-4 claim B (8 059 s at order 3) is the largest
single term. **Presents as:** a T9b tier killed by walltime, which costs a
day. **Distinguishing measurement:** T9d step 3's measured per-step cost and
projected tier total. If the projection exceeds about 80 % of 1440 minutes,
T9b is not submitted until the tier is split or re-timed, and that becomes a
new task.
