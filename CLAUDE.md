# Beatnik

The Canopy FMM integration (v1) is complete. See
[tasks/integrate_canopy.md](tasks/integrate_canopy.md) for the
checkpoint-by-checkpoint history and the user-driven TODO list
(smoke scaling run remains). Open optimization opportunities are
listed under [Future optimization opportunities](#future-optimization-opportunities)
below.

## System detection

Build and run commands differ by system. Before building or running anything,
run `hostname` and match the result against the table below. Then Read the
matching system-specific instructions file and follow it for the rest of the
session.

| Hostname pattern | Instructions file              |
| ---------------- | ------------------------------ |
| `tuolumne*`      | `docs/claude-tuolumne.md`      |

The pattern is the alphabetic prefix of the host (e.g. `dane1234` matches
`dane*`, `lassen708` matches `lassen*`). To add support for a new system,
create `docs/claude-<system>.md` and add a row above.

If the hostname does not match any row, or the matching file is missing one of
the required sections below, ask the user to fill in the gap and update (or
create) the doc before proceeding.

### Required sections in every `docs/claude-<system>.md`

1. **Spack environment** — the `spack env activate ...` command that must be
   run before compiling or running any binary from this library. When
   setting up a new system, ask the user for **both** a development spack
   environment (used for builds and iterative work) and a production
   spack environment (used for large-scale runs that may sit in queue
   for a long time, so ongoing development rebuilds cannot break a
   queued job). Record both in the system doc, and note that batch
   scripts under `scripts/<hostname>/` should ask the user which env to
   activate before being written.
2. **CMake args** — system-specific args that must be passed to `cmake` (or to
   any helper bash script that wraps `cmake`).
3. **Build command** — how to build a target on this system. Default:
   `make [EXECUTABLE]` (the user specifies the target when appropriate). If
   the system installs via spack, the build command is `spack install`
   instead. Every `docs/claude-<system>.md` must state which of the two
   applies.
4. **Run command for binaries** — the command template for running a built
   binary. Default starting point:
   `mpirun --oversubscribe -n [num_procs] [EXECUTABLE] [EXTRA_ARGS]`. Replace
   `mpirun` with `flux run`, `srun`, or whatever the system uses.
5. **Job-scheduler batch template** — if the system has a scheduler (flux,
   slurm, …), include a template batch script that can be filled in and
   submitted (e.g. `flux batch <script>`) to run binaries when the user is
   not inside an interactive allocation. Save concrete scripts to
   `scripts/<hostname>/` (create the directory if it does not exist).
6. **Running non-test binaries** — when asked to run something other than a
   test (e.g. an `examples/` problem), ask the user for the example name and
   args, then plug them into sections 4 and 5.

The required tests themselves (names + MPI rank counts) are project-wide and
live in [Minimum test set](#minimum-test-set) below, not in the per-system
doc. The per-system doc only describes *how* to run any given test on that
machine.

## Minimum test set

These tests must pass before any code change ships. Each entry lists the
test name and the MPI rank counts it must be run at (e.g.
`Beatnik_Test_Particle` at 1, 2, 3, 4, 5, 6 ranks). Use the run command and
batch template from the active system's `docs/claude-<system>.md` to execute
them.

The minimum test set:

- `Beatnik_Test_FmmVsExact_MPI_<DEVICE>` — at 1, 4 ranks.
  Built only when `Beatnik_ENABLE_TESTING=ON` and
  `Beatnik_ENABLE_CANOPY=ON` (spack: `beatnik +testing +canopy`).
  Verifies the FMM BR solver agrees with the Exact BR solver on a
  32×32 free-boundary case. `<DEVICE>` is the active Kokkos backend
  (e.g. `HIP`, `OPENMP`, `SERIAL`).

## Scaling single-mode runs

When creating a new single-mode `sech2` rollup deck or flux script, scale every
geometry/IC parameter from the **64-node baseline** (`bounding_box = 1.0`,
`magnitude = 0.1`, `period = 9.0`, `atwood = 0.15`). The shipped
`examples/01_rocketrig/single_mode_{1024,4000,16000}.in` are worked examples,
and README's "Scaling to large meshes" section has the long-form rationale.

Let `B = bounding_box`. Scale by mesh resolution:

- **`bounding_box = nodes / 64`** (linear in mesh). This holds the physical grid
  spacing `dx = 2B/(nodes − 1) ≈ 0.031` constant, which is what keeps interface
  points from getting too close together — the regime where the BR/FMM solve
  collapses. At a *fixed* bbox the points crowd as `nodes` grows (empirically
  64→1.0 is fine but 256 needs bbox ≥ 3.0); the `nodes/64` rule avoids that.
- **`magnitude = 0.1 · B`** (linear). `magnitude` is the vertical amplitude.
- **`period = 9.0 / B²`** (**inverse square**). The sech² IC is
  `z = magnitude · sech²(period · r²)` with `r` the physical distance from the
  domain centre (domain is `(−B..B)`), so the bump half-width is `~1/√period`.
  To keep the bump the same *fraction* of a domain whose half-width grows
  linearly, `period ∝ 1/B²`. Scaling `period` the wrong way collapses the bump
  to a sub-cell spike → a flat, unevolving surface. (This inverse-square rule is
  specific to `sech2`; the `cos` IC's `period` is a wavelength and scales
  *linearly*.)

| nodes | B = bbox | magnitude | period      |
| ----- | -------- | --------- | ----------- |
| 64    | 1        | 0.1       | 9.0         |
| 256   | 4        | 0.4       | 0.5625      |
| 512   | 8        | 0.8       | 0.140625    |
| 1024  | 16       | 1.6       | 0.03515625  |
| 4000  | 62.5     | 6.25      | 0.0023040   |
| 16000 | 250      | 25.0      | 0.000144    |

**Time integration.** `delta_t` is *not* the cause of the rollup failures (those
were FMM accuracy + the NIC crash, both resolved under Known issues), but it has
its own scaling:

- Keep `delta_t` at or below the auto value `tau/50` (high order),
  `tau = 1/√(atwood·gravity)` — independent of `B`. Do **not** inflate it to
  "evolve faster": interface velocity grows as `v ~ √(atwood·gravity·λ)` with
  `λ ∝ B`, so the CFL-stable step *shrinks* as `~1/√B` (≈ `auto/√B` at peak
  velocity). An oversized `delta_t` triggers a separate CFL blowup.
- **Timesteps to reach a full rollup grow ~linearly with `B`** (physical time
  `∝ √B`, stable step `∝ 1/√B`). A large mesh with a small fixed step count
  shows only early motion, not a developed rollup.

**FMM solver settings for rollup runs** (validated configuration from the
rollup debugging — see [Known issues](#known-issues)):

- `P_ORDER = 10` (compile-time, [src/FmmBRSolver.hpp](src/FmmBRSolver.hpp)),
  `fmm_mac_theta = 0.4`, `fmm_max_depth = 19`, `fmm_imbalance_tol = 0.20`,
  `epsilon = 2`. This is the config that cleared full rollup (256², 16 ranks).
- Keep `fmm_near_softening_factor` at its default (`4.0`) or higher — it is the
  M2L-vs-P2P softening floor that fixed the premature far-field NaN.
- **`fmm_ncrit = 64` beats `128` at scale (GPU).** On a 1536² (B=24) full-rollup
  run at 256 ranks / 64 nodes, calibration measured **3.80 s/step at `ncrit=64`
  vs 4.11 s/step at `ncrit=128`** (early-regime, 50 steps) — keep `ncrit=64`; the
  larger-leaf GPU-throughput win does not materialize here. `fmm_max_depth = 19`
  is the hard maximum (Canopy's `TreeBuilder` throws above 19 — Morton-key
  storage). That run is staged in
  [scripts/tuolumne/rocketrig1536.flux](scripts/tuolumne/rocketrig1536.flux)
  (24 h wall sized for ~20 h full rollup; deck
  `/p/lustre5/stewartj/beatnik/fmm/n1536_p256/single_mode.in`), with calibration
  variants `rocketrig1536_calib_ncrit{64,128}.flux`.
- **Watch FMM cost at production sizes (e.g. 16000):** Canopy `solve()` is ~92%
  of per-call cost, and the softening floor widens the near-field P2P as the
  core densifies, so cost grows through roll-up. `fmm_near_softening_factor`
  trades far-field accuracy against that cost.

## Plans

When creating plans via plan mode, save plan files to `./plans/` in this
repository, not the default plan location.

## Future optimization opportunities

Carried forward from the v1 Canopy FMM integration. Each is
self-contained — pick them up in any order. See
[tasks/integrate_canopy.md](tasks/integrate_canopy.md) for the
detailed context behind each.

- **Cache the forward `Cabana::Distributor` across Migrate-action
  steps in `FmmBRSolver::computeInterfaceVelocity`.** Today
  `buildForwardDistributor()` runs every call. When
  `auto_maintain()` returns `Migrate` (no topology change), the
  forward distributor's comm pattern is reusable. Hook the
  returned `MaintenanceAction` into a cache-invalidation check
  and only rebuild on `Rebalance`/`Rebuild`. Profile first: at
  small drift (the typical RK3 step), the build is already a
  small share of the per-call cost.

- **Expose every Canopy tunable as a runtime rocketrig CLI flag.**
  Today `ncrit`, `max_depth`, `bbox_tol`, `ncrit_tol`,
  `replication_depth`, `imbalance_tolerance`, `mac_theta`, and
  `softening` are baked into `Beatnik::Params` defaults. `P_ORDER`
  is a compile-time constant (`6` in
  [src/FmmBRSolver.hpp](src/FmmBRSolver.hpp)). Mirror the flag
  names from
  [Canopy/examples/03_gravity_solve/gravity_solve.cpp](../canopy/examples/03_gravity_solve/gravity_solve.cpp).
  Surfacing `P_ORDER` at runtime requires either compile-time
  enumeration of common values or a runtime template dispatch.

- **Track header dependencies in the Beatnik INTERFACE target.**
  Today `add_library(Beatnik INTERFACE)` in
  [src/CMakeLists.txt](src/CMakeLists.txt) means consumer .o files
  don't depend on `HEADERS_PUBLIC`. When only a header changes,
  `make` sees nothing to rebuild and `spack install` reports a
  sub-second build — the workaround is to `touch
  examples/01_rocketrig/rocketrig.cpp` before `spack install`.
  Fix: use `target_sources(Beatnik INTERFACE FILE_SET HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS_PUBLIC})`
  or attach a `PUBLIC_HEADER` property so consumers track them
  as real dependencies.

- **Higher profiling levels (2, 3) in FmmBRSolver.** Levels 2 and
  3 are reserved but unused today. Natural additions: at level 2,
  per-phase timing inside `computeInterfaceVelocity`
  (`packGridParticles`, `buildForwardDistributor`, the migrate,
  `auto_maintain`, `solve`, the reverse distribute, `writeZdot`);
  at level 3, per-rank particle counts and Canopy's `Migrate`-vs-rebuild
  cost ratio. Mirror Canopy's existing `[Canopy Diagnostics] solve()
  phase breakdown` format.

- **Switch from `setup()` every step to a more aggressive cache
  pattern.** Today the AoSoA itself is rebuilt every call via
  `Cabana::migrate`. If the local-grid layout never changes (the
  common case), the destination AoSoA could be a persistent member
  resized only on rebuild. Profile first.

- **Real assertion on the multi-step trajectory in the test suite.**
  `OneRK3StepComparison` and `FiveRK3StepsComparison` both compare
  `z` after stepping; extending to e.g. 20 steps would catch slow
  drift. Tolerance accumulates roughly linearly with step count;
  raise to ~1e-2 or compute a per-step bound.

## Known issues

- **FMM large-scale crash: GTL dreg-cache exhaustion in the Canopy `Rebalance`
  particle migrate (MITIGATED 2026-06-22; durable fix tracked in Canopy #22).**
  The 256-rank / 64-node 1536² full-rollup run
  ([scripts/tuolumne/rocketrig1536.flux](scripts/tuolumne/rocketrig1536.flux))
  deadlocked at step 307 of 8400, during the first significant `Rebalance`, with
  `(GTL DEBUG) dreg_evict returned NO_SPACE ... more than 10000 active memory
  regions`. One rank's GPU comm died and the other 255 blocked in the next
  collective, so the job hung (flux still showed it `R` for ~19 h until
  cancelled). **Root cause:** `dreg_evict NO_SPACE` is a peak-*simultaneity*
  failure (all 10000 GTL registration-cache entries active at once), not a leak.
  Canopy's `RegisteredBufferPool` bounds the `solve()` exchanges (coalesced M2L
  + P2P) to one registered region per direction, but the `Rebalance` particle
  migration (`TreePartitioner::migrate_particles` → `Cabana::Distributor` /
  `Cabana::migrate`) is **not** pooled — at 256 ranks the many-way scatter hands
  O(peers) device buffers to GPU-aware MPI at once, blowing the 10000 default.
  **Stopgap (deployed):** `export GTL_DREG_CACHE_SIZE=262144` in the
  `rocketrig1536*.flux` scripts (≈26× headroom). Out-provisioning only; the peak
  likely grows as the core densifies, so this may not survive to full rollup.
  **Durable fix:** route the rebalance migrate through a pool-backed coalesced
  exchange (one registered region per direction), mirroring the `solve()` fix —
  **Canopy #22**. When that lands and Beatnik picks up the new Canopy, the env
  bump becomes belt-and-suspenders.

- **FMM full-rollup crash: Slingshot NIC registration exhaustion during
  Canopy `Rebalance` (RESOLVED 2026-06-17).** In a single-mode rollup
  driven by the FMM BR solver, once the sheet fully rolled up, Canopy's
  `auto_maintain` went near-continuous `Rebalance` and the run aborted with
  `cxil_map: write error` → `MPIDI_OFI_send_normal: Invalid argument`.
  **Root cause:** GPU-aware-MPI memory *registration churn* — both halo
  exchanges in Canopy `solve()` (`coalesced_view_exchange` for M2M/M2L/L2L,
  and `P2P::gather_ghost_particles`) allocated a fresh device `Kokkos::View`
  per peer every call, and each fresh device allocation is a fresh CXI NIC
  registration; the cache accumulated until the NIC was exhausted.
  **Fix:** `Canopy_RegisteredBufferPool.hpp` — persistent, grow-only device
  buffers with stable base addresses; each exchange carves per-peer
  unmanaged subviews from one registered region per direction (Canopy
  branch `redesign`: `fac4519`, `d834034`). Validated: `FmmVsExact` green on
  HIP/OPENMP/SERIAL at 1,4 ranks; full crash-deck run `f3F1T7e6F24F` cleared
  the entire Rebalance-heavy tail with **0 `cxil_map` errors**. See
  [tasks/fmm_fullrollup_crash.md](tasks/fmm_fullrollup_crash.md) for the
  full record.

- **Premature FMM NaN at full rollup (RESOLVED 2026-06-18).** Live record:
  [tasks/fmm_premature_nan.md](tasks/fmm_premature_nan.md).
  **Root cause:** the FMM **far-field (M2L) used the unsoftened `1/r` Laplace
  kernel** — the Plummer softening (`eps = sqrt(epsilon) ≈ 1.414`) was applied
  only in the near-field P2P. At full roll-up the core collapses so cells shrink
  to `~3e-3` and "geometrically far" M2L pairs sit at `rho ≈ 0.03–0.4 ≪ eps`,
  where the softened kernel is nearly flat but the unsoftened multipole `1/r` is
  ~35× larger → a spurious single-node velocity that seeds a runaway (bbox
  escape → `Rebuild` on an exploding box → NaN at step 1364). The exact solver
  sums the softened kernel directly, so it stays bounded. Confirmed: at
  `softening=0` the FMM matches an unsoftened all-pairs reference to machine
  precision; the error is P-independent (`P_ORDER` 10→16 byte-identical, ruling
  out truncation), depth-dependent, and far-field-only.
  **Fix:** Canopy `CommunicationPlan::mac_satisfied` now forces any pair closer
  than `near_softening_factor · eps` to the softened P2P path instead of M2L —
  a runtime knob `FmmConfig::near_softening_factor` (default `4.0`), settable
  from a rocketrig `.in` deck as `fmm_near_softening_factor` — the parser wiring
  in `InputFile.hpp` was completed in `db6a956`; before that commit the key was
  rejected as "unknown" and only the compile-time `4.0` default applied. Plus a
  secondary fix to the L2P
  gradient finite-difference step (fixed `h=1e-5` → `h=1e-5·w_self`). Canopy
  branch `debug-nan`: `f484cbc` + knob/README follow-ups.
  **Validation:** `FmmVsExact` green on HIP/OPENMP/SERIAL at 1,4 ranks (job
  `f3FZycxsexw1`); full crash-deck run (16 ranks/4 nodes, job `f3Fa1vABnaHu`)
  completed **all 1400 steps, 0 NaN, 0 bbox-escape Rebuilds**
  (`Migrate=1180 Rebalance=3019 Rebuild=0`).
  - Repro config: single-mode `sech2`, 256×256 (B=4), `P_ORDER=10`,
    `fmm_max_depth=19`, `fmm_mac_theta=0.4`, `fmm_imbalance_tol=0.20`,
    `epsilon=2`, `delta_t=0.0006`, 16 ranks / 4 nodes.
  - Temporary `CANOPY_NAN_DEBUG` / snapshot-dump diagnostics removed (the
    `04_nan_replay` FMM-vs-exact harness is kept; the snapshot dump is now the
    OFF-by-default `Beatnik_ENABLE_FMM_SNAPSHOT` option). `FmmVsExact` re-run
    green on the clean build (job `f3FnQ2sgDZUK`); `debug-nan` merged into
    `develop-canopy` (Beatnik) and `redesign` (Canopy).
  - **Watch:** the softening floor widens P2P at full roll-up (more near-field
    pairs in the dense core), so keep an eye on FMM cost at production mesh
    sizes (e.g. 16000); `fmm_near_softening_factor` trades far-field accuracy
    against that cost.
  - Earlier, already FIXED: the *premature* FMM NaN at rollup *onset* was a
    separate accuracy problem, resolved by raising `P_ORDER` 6→10 and
    tightening `fmm_mac_theta` 0.6→0.4.

## General guidelines

- **Checkpoint commits in plans.** When planning a large code change, include
  explicit checkpoints in the plan file where progress should be committed.
  If a later step fails (test failure, performance regression), we can roll
  back to the nearest checkpoint and retry.
- **Follow `.clang-format`.** If `.clang-format` exists at the repo root,
  follow its formatting rules for any C/C++ code you write or edit. If it
  does not exist, ignore this rule.
- **Keep `README.md` in sync.** When a public-facing API changes, or when the
  arguments accepted by an example problem change, update `README.md` in the
  same change so its documentation stays accurate.
