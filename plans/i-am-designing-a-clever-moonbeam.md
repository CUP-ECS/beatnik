# Integrate Canopy FMM as a far-field BR solver in Beatnik

## Context

Beatnik currently offers two far-field solvers for the Birkhoff-Rott integral on a logically uniform 2D surface mesh: `ExactBRSolver` (all-pairs, O(N²) per step via ring-pass) and `CutoffBRSolver`. The user wants to add a third option, `FmmBRSolver`, backed by the Canopy fast multipole library at `~/research-Bridges/Canopy`. Target hardware is 4th-gen AMD EPYC CPUs and MI300A APUs, with up to 1e8 particles per MPI rank in production. The integration must be optional (compile-time `Beatnik_ENABLE_CANOPY=ON`) and CLI-selectable via `-S fmm`.

Beatnik already has scaffolding: [src/CreateBRSolver.hpp](src/CreateBRSolver.hpp) declares `BR_FMM` and has commented-out dispatch; [src/FmmBRSolver.hpp](src/FmmBRSolver.hpp) is a near-verbatim copy of `ExactBRSolver` to be rewritten; and the top-level [CMakeLists.txt](CMakeLists.txt) at line 78 already runs `Beatnik_add_dependency(PACKAGE Canopy)`, which sets `Beatnik_ENABLE_CANOPY` based on whether `find_package(Canopy)` succeeded.

## Canopy API in one paragraph

`Canopy::Solver<MemSpace, ExecSpace, double, P_ORDER, NComps>` (in [Canopy_Solver.hpp](../Canopy/src/Canopy_Solver.hpp)) is constructed with `(MPI_Comm, ncrit, max_depth, bbox_tol[3], ncrit_tol, replication_depth, imbalance_tol, mac_theta, softening)`. The user provides a `Cabana::AoSoA` whose tuple holds at minimum `double[3]` position and `double[NComps]` "charge". `setup<PositionIdx, ChargeIdx>(aosoa, num_local_before)` builds the tree, partitions, sorts the AoSoA in-place by leaf, and migrates particles across MPI ranks — after this the caller's AoSoA is in FMM order. `solve<PositionIdx, ChargeIdx>(aosoa, compute_gradient=true)` runs P2M→M2M→M2L→L2L→L2P→P2P; the gradient view is `Kokkos::View<double*[NComps][3], MemSpace>` indexed `gradient(particle, comp, dim)`. `auto_maintain<PositionIdx, ChargeIdx>(aosoa)` picks Migrate/Rebalance/Rebuild and is the per-step entry point we want to hit. The Laplace kernel computes `φ(x) = Σⱼ qⱼ / |x − xⱼ|` with **Plummer softening** `(r² + softening²)^{−3/2}` for the gradient. No built-in periodic handling.

## Beatnik's BR kernel (from [src/Operators.hpp:111-131](src/Operators.hpp#L111))

```
u_pair = −(dx·dy·w)/(4π) · ω × (x_i − x_j) / (|x_i − x_j|² + ε)^{3/2}
```

Two critical observations:

1. **Softening is linear in ε, not squared.** The kernel uses `(r² + ε)^{−3/2}`, not Plummer's `(r² + ε²)^{−3/2}`. The comment on Operators.hpp:123 makes this intentional: "matlab code doesn't square epsilon". To reproduce this in Canopy, **pass `softening = sqrt(_epsilon)`** to the Canopy constructor so Canopy's `(r² + softening²)^{−3/2}` matches.
2. **Per-source prefactor is `−(dx·dy·w_simpson)/(4π)`**, where `w_simpson = simpsonWeight(global_i, N) · simpsonWeight(global_j, N)`.

## Math: from Canopy gradients to zdot

Pack charges as `q_c^(j) = w_simpson(j) · ω_c^(j)` so the Simpson weight is folded into the FMM source. Canopy then returns:

> `gradient(i, c, d) = ∂_d Σⱼ q_c^(j) / |x_i − x_j|_softened = −Σⱼ q_c^(j) (x_i − x_j)_d / r³`

Compute the standard `ω × ∇G` cross product from three component-gradients (user confirmed eq 6 in the image is a typo; use standard sign):

```
u_cross[0] = gradient(i,1,2) − gradient(i,2,1)
u_cross[1] = gradient(i,2,0) − gradient(i,0,2)
u_cross[2] = gradient(i,0,1) − gradient(i,1,0)
```

Tracing the signs: Beatnik wants `u = −(dx·dy)/(4π) · Σⱼ w_j ω_j × (x_i − x_j)/r³`. The cross-of-gradients above evaluates to `u_cross = −Σⱼ w_j ω_j × (x_i − x_j)/r³`. Therefore the final scaling that lands in `zdot` is:

> `zdot(i, j, d) = (dx · dy) / (4π) · u_cross[d]`

(positive sign, no extra negation).

## Periodic boundaries — v1 scope

User confirmed: **v1 supports non-periodic boundaries only.** If `_bc.isPeriodicBoundary({0,1})` or `{1,1}` is true, `FmmBRSolver`'s constructor prints an error and aborts. Periodic FMM via image replication is a follow-up. The check belongs in the constructor so the failure surfaces at solver creation, not mid-step.

## Data flow per `computeInterfaceVelocity(zdot, z, omega)` call

Beatnik calls this 3× per RK3 step. Positions and 3-component omega arrive as `Kokkos::View<double***, MemSpace>` of shape `[Ni+halo][Nj+halo][3]`. The flow:

1. **Pack grid-ordered AoSoA** `_grid_particles` from the owned index space of the local grid: one tuple per owned node `(i, j)` with fields `(position = z(i,j,:), charge = simpson(i,j) · omega(i,j,:), tag = (origin_rank, i, j))`.
2. **Forward distribute** `_grid_particles` → `_canopy_particles` via a persistent `Cabana::Distributor`. Initial population (first call) goes straight into `_canopy_particles` and Canopy's `setup<Position, Charge>` runs; subsequent calls reuse the AoSoA in place.
3. **`_canopy.auto_maintain<Position, Charge>(_canopy_particles)`** — refreshes Canopy's internal tree/partition/comm plan for the new positions. The maintenance return value (Migrate/Rebalance/Rebuild) is recorded but not branched on (see "mapping cheaply" below).
4. **`_canopy.solve<Position, Charge>(_canopy_particles, /*compute_gradient=*/true)`**.
5. **Compute u_cross** with a `Kokkos::parallel_for` over Canopy's local particle count using the three differences above; write into a `u_out` slice of the AoSoA (or a parallel view sized to `_canopy.num_local_particles()`).
6. **Reverse distribute** `u_out` (keyed by the `tag` field) back to its origin grid rank.
7. **Write into `zdot`**: parallel over received tuples, write `zdot(tag.i, tag.j, d) = (dx·dy)/(4π) · u_cross[d]` for d=0..2.

## Maintaining the grid↔FMM mapping cheaply

Each AoSoA tuple carries `tag = (origin_rank, i, j)` so its grid origin travels with the particle through any Canopy migration. After every `auto_maintain` call we rebuild the `Cabana::Distributor` from the current `tag.origin_rank` field — this is O(local_N), no all-to-all comparison, no MPI handshake beyond what `Cabana::Distributor` does internally. We rebuild every call rather than caching; if profiling later flags this as a bottleneck, we can gate the rebuild on `auto_maintain`'s returned action (Migrate keeps the comm plan stable, so we *could* keep the distributor too).

## Critical files to modify

- [CMakeLists.txt](CMakeLists.txt) — `find_package(Canopy)` already runs at L77-82. When `Beatnik_ENABLE_CANOPY` is true, link `Canopy::Canopy` (or equivalent exported target — verify the actual name in Canopy's installed CMake config) into the `DEPENDS_ON` list and propagate the compile definition.
- [src/CMakeLists.txt](src/CMakeLists.txt) — conditionally append `FmmBRSolver.hpp` to `HEADERS_PUBLIC` when `Beatnik_ENABLE_CANOPY`. Add `Canopy::Canopy` to `DEPENDS_ON` in the same guard.
- [src/Beatnik_Config.hpp.in](src/Beatnik_Config.hpp.in) — add `#cmakedefine BEATNIK_ENABLE_CANOPY` so the header guards work in `CreateBRSolver.hpp` and `rocketrig.cpp`.
- [src/CreateBRSolver.hpp](src/CreateBRSolver.hpp) — uncomment the `BR_FMM` branch and wrap it (plus the `#include <FmmBRSolver.hpp>`) in `#ifdef BEATNIK_ENABLE_CANOPY`. The fallthrough case prints an error and aborts (existing pattern at L50-53 reused).
- [src/FmmBRSolver.hpp](src/FmmBRSolver.hpp) — full rewrite per the data-flow section. Holds:
  - `Canopy::Solver<MemSpace, ExecSpace, double, P_ORDER, /*NComps=*/3>` instance
  - persistent `Cabana::AoSoA<MemberTypes<double[3], double[3], double[3], int[3]>, MemSpace>` with slots: position, charge (simpson·omega), u_out, tag
  - `Cabana::Distributor` for grid↔FMM exchange, rebuilt each call
  - boolean `_first_call` to dispatch `setup` vs `auto_maintain`
  - periodic-boundary check in constructor (abort if either dim is periodic)
- [src/Solver.hpp](src/Solver.hpp) — extend `struct Params` at L41 with FMM-specific fields:
  ```
  int    fmm_p_order            = 6;
  int    fmm_ncrit              = 32;
  int    fmm_max_depth          = 15;
  double fmm_mac_theta          = 0.5;
  int    fmm_replication_depth  = 3;
  double fmm_imbalance_tol      = 0.10;
  double fmm_ncrit_tol          = 0.10;
  double fmm_bbox_tol           = 0.10;
  ```
  P_ORDER is a Canopy template parameter, so `fmm_p_order` is consumed at FmmBRSolver template-instantiation time (or we pick a fixed default like 6 for the first cut and expose it later).
- [examples/01_rocketrig/rocketrig.cpp](examples/01_rocketrig/rocketrig.cpp) — extend the `-S` parser at L247-265 with an `"fmm"` case. Guarded so that when `BEATNIK_ENABLE_CANOPY` is undefined, the case still parses but prints an error and exits. CLI flags for the FMM-specific Params fields can wait for a follow-up — for v1, the defaults above are baked in.

## Implementation order and commit checkpoints

Each checkpoint below should land as its own commit so we can `git revert` cleanly if a later step uncovers a problem. After every checkpoint, append a one-line entry to `tasks/integrate_canopy.md` (see next section) noting the commit hash and what was verified.

1. **Build-system plumbing.** Add `BEATNIK_ENABLE_CANOPY` to [Beatnik_Config.hpp.in](src/Beatnik_Config.hpp.in); wire the option through [CMakeLists.txt](CMakeLists.txt) and [src/CMakeLists.txt](src/CMakeLists.txt) (link target + conditional header). Build with the option both OFF and ON. **Checkpoint commit:** "Wire Beatnik_ENABLE_CANOPY through CMake; no behavior change."
2. **CLI `-S fmm` parsing.** Extend [rocketrig.cpp](examples/01_rocketrig/rocketrig.cpp) to accept `"fmm"` and produce a clean error when `BEATNIK_ENABLE_CANOPY` is undefined. Confirm `-S fmm` errors out with the option OFF and is accepted with it ON (no solver constructed yet — `createBRSolver` still fails). **Checkpoint commit:** "Accept -S fmm in rocketrig with ENABLE_CANOPY guard."
3. **CreateBRSolver dispatch + Params extension.** Add FMM-specific fields to [Solver.hpp's Params](src/Solver.hpp#L41); uncomment the FMM branch in [CreateBRSolver.hpp](src/CreateBRSolver.hpp) under `#ifdef BEATNIK_ENABLE_CANOPY`. At this point `FmmBRSolver` still has the copied ExactBRSolver body — it should compile and produce correct (exact-quality) output, just slowly. Run the existing rocketrig example with `-S fmm` and confirm matching output vs `-S exact`. **Checkpoint commit:** "Dispatch FMM branch through CreateBRSolver; FmmBRSolver still all-pairs."
4. **Lift `simpsonWeight` into Operators.hpp.** Pure refactor; verify nothing changes. **Checkpoint commit:** "Move simpsonWeight to Operators.hpp."
5. **FmmBRSolver constructor + periodic guard + Canopy::Solver member.** Add the persistent `Canopy::Solver` and AoSoA members, constructor wiring (softening = `sqrt(_epsilon)`), and the periodic-boundary error path. `computeInterfaceVelocity` still falls back to the all-pairs code. **Checkpoint commit:** "FmmBRSolver: hold Canopy::Solver instance; guard periodic; legacy compute path."
6. **First-call setup + grid→FMM pack + forward distribute.** Implement the AoSoA pack (single-rank first), `_canopy.setup<...>(...)` on first call, and the forward `Cabana::Distributor`. No solve yet — just verify particle counts match across the round-trip. **Checkpoint commit:** "FmmBRSolver: first-call setup and forward distribute (no solve)."
7. **Canopy solve + cross-product + reverse distribute + zdot write.** Wire `auto_maintain` → `solve` → cross-product → reverse distribute → write `zdot`. This is the load-bearing checkpoint. Compare to `-S exact` on a 32×32 free-boundary single-rank case. **Checkpoint commit:** "FmmBRSolver: full FMM compute path; matches Exact on small case."
8. **Multi-rank validation + fixes.** Run the 4-rank case from the Verification section; address any bugs uncovered. **Checkpoint commit:** "FmmBRSolver: multi-rank correctness."
9. **Smoke scaling run + tasks/integrate_canopy.md final notes.** Capture timing and `auto_maintain` action histogram for the 256² × 4-rank case in the ledger. **Checkpoint commit:** "FmmBRSolver: scaling smoke test + ledger update."

If verification at any checkpoint fails and the cause isn't clear in a single sitting, `git revert` the offending commit, note the failure mode in `tasks/integrate_canopy.md`, and re-plan that checkpoint before retrying.

## Implementation ledger

Create a new directory `tasks/` at the repo root and a running ledger `tasks/integrate_canopy.md`. As implementation proceeds, append to this file:

- **Progress** — what landed in each commit, with file:line references for non-obvious decisions.
- **Bugs** — anything discovered that needs a follow-up (kernel-form discrepancies, distributor edge cases, Canopy assumptions that didn't hold, scaling issues observed during verification).
- **Future optimization areas** — places where a measurement showed a cost worth revisiting (distributor rebuild cadence, AoSoA layout, alternative packing strategies, kernel-fusion opportunities, MI300A-specific tuning).

Update [CLAUDE.md](CLAUDE.md) with a short pointer near the top: e.g. *"For Canopy integration progress, open questions, and known issues, see [tasks/integrate_canopy.md](tasks/integrate_canopy.md)."* This makes the ledger discoverable to future agent sessions without re-deriving context.

## Reuse from existing code

- The Simpson weight helper (`simpsonWeight(int, int)` static in [ExactBRSolver.hpp:87-92](src/ExactBRSolver.hpp#L87) and the copy in FmmBRSolver.hpp) — lift it into [Operators.hpp](src/Operators.hpp) once so all BR solvers share it.
- `prepareOmega` in [ZModel.hpp:304-319](src/ZModel.hpp#L304) already produces the 3-component omega we'll pack; FmmBRSolver receives it via `computeInterfaceVelocity`'s `o` argument and never recomputes.
- `Cabana::Distributor` + `Cabana::migrate` for the grid↔FMM exchange; no hand-rolled MPI.
- The local index space pattern from ExactBRSolver:
  ```
  auto local_grid  = _pm.mesh().localGrid();
  auto local_space = local_grid->indexSpace(Own(), Node(), Local());
  ```
  is reused verbatim for packing/unpacking.

## Verification

1. Build with `Beatnik_ENABLE_CANOPY=OFF` (default). Confirm `rocketrig -S fmm` errors out cleanly and `-S exact` / `-S cutoff` still work.
2. Build with `Beatnik_ENABLE_CANOPY=ON` after `spack env activate ~/spack_envs/beatnik-canopy` per [CLAUDE.md](CLAUDE.md).
3. **Single-rank correctness**: 32×32 mesh, `-b free`, run one RK3 step with `-S exact` and `-S fmm`. Compare `zdot` per node; expect relative agreement ≲ 1e-3 at P=6, tighter at P=8.
4. **Multi-rank correctness**: same test at 4 ranks. After reverse-distribute back to grid order, per-node values should match the single-rank FMM run to floating-point rounding.
5. **Periodic guard**: run with `-b periodic -S fmm`; expect a clean error and exit at solver construction.
6. **Smoke scaling**: 256² nodes, 4 ranks, 20 steps. Confirm FMM runtime is well below Exact's, and that `auto_maintain` lands on `Migrate` for the bulk of steps (visible via `CANOPY_ENABLE_PROFILING` if compiled in).

## Open follow-ups (after v1 lands)

- Periodic-boundary support via 9× image replication of sources.
- **Expose every tunable Canopy option as a runtime CLI argument in rocketrig** — currently the v1 plan bakes defaults into `Params`. The full set to surface: `ncrit`, `max_depth`, `bbox_tol`, `ncrit_tol`, `replication_depth`, `imbalance_tolerance`, `mac_theta`, `softening` (overriding the `sqrt(_epsilon)` default), and `P_ORDER` (which is a template parameter — surfacing this requires either compile-time enumeration of common P values or a runtime dispatch). Mirror the flag names used in [Canopy/examples/03_gravity_solve/gravity_solve.cpp](../Canopy/examples/03_gravity_solve/gravity_solve.cpp) where possible.
- Test target under [tests/](tests/) that runs the FMM↔Exact comparison in CI when `Beatnik_ENABLE_CANOPY=ON`.
- Cache the `Cabana::Distributor` across `Migrate`-action steps (skip rebuild) if profiling shows the rebuild is a measurable cost.
