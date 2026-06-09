# Canopy FMM integration — running ledger

Tracks per-checkpoint progress, follow-up bugs, and future-optimization
notes for the Canopy fast-multipole BR solver integration.
See [../plans/i-am-designing-a-clever-moonbeam.md](../plans/i-am-designing-a-clever-moonbeam.md)
and the review at
`~/.claude/plans/please-review-the-plan-declarative-creek.md`.

## Progress

| # | Checkpoint | Commit | Verified |
|---|-----------|--------|----------|
| 1 | Wire `BEATNIK_ENABLE_CANOPY` through CMake | fb123c6 | `spack install` succeeds with Canopy ON (env has Canopy installed). OFF path is structurally guarded via `if(Beatnik_ENABLE_CANOPY)`; testing OFF would require an env without Canopy and is deferred. |
| 2 | Accept `-S fmm` in rocketrig + README sync | 4209e51 | `spack install` succeeds; rocketrig recompiled (1m 3s). Actual `-S fmm` runtime accept/reject behavior verified at the next checkpoint when the dispatch lands. |
| 3 | CreateBRSolver dispatch + Params extension | 0ae01cd | `spack install` succeeds (1m 4s). `FmmBRSolver` is currently the all-pairs ExactBRSolver body — runtime equivalence test is part of checkpoint 7. |
| 4 | Lift `simpsonWeight` into `Operators.hpp` | bfcd383 | Pure refactor; both BR solvers now call `Operators::simpsonWeight`. `spack install` succeeds (1m 3s). |
| 5 | FmmBRSolver: hold Canopy::Solver instance; guard periodic; legacy compute path | ca2d8c7 | `spack install` succeeds (1m 3s after `touch rocketrig.cpp` — see Bugs/follow-ups). Compute path is still ring-pass all-pairs; the persistent Canopy solver is held but not yet invoked. Periodic-boundary guard added in the constructor. |
| 6 | FmmBRSolver: first-call setup + grid pack (no solve, no Distributor) | 227a61e | `spack install` succeeds (1m 12s) after installing Canopy first via `spack install canopy@develop … %cce` and fixing Canopy's `CanopyConfig.cmakein` to `find_dependency(Trilinos)` (Canopy commit `ab1e8fc`). Compute path still falls through to ring-pass all-pairs after first-call setup. |
| 7 | FmmBRSolver: full FMM compute path | c2b5250 | `spack install` succeeds (1m 2s). Pipeline: pack → setup → solve → cross-product → reverse-distribute → write zdot. Runtime equivalence against `-S exact` is gated on the test target in checkpoint 7a. v1 simplification: `setup()` runs every step; switching to `auto_maintain` is a follow-up. |
| 7a | Add `Beatnik_Test_FmmVsExact` CTest + spack `+testing`/`+canopy` variants | 0d66fa7 + `e5ec649` | Two TEST cases in `tests/tstFmmVsExact.hpp` (`.BRDirectComparison`, `.OneRK3StepComparison`); both pass at `ntasks=1` on tuolumne after the `e5ec649` cmakedefine fix. spack: `+testing`, `+canopy`, `+examples` variants on the compass-repo beatnik package; `setup_run_environment` prepends `share/Beatnik/tests/` to `PATH`. CLAUDE.md "Minimum test set" registered `Beatnik_Test_FmmVsExact_MPI_<DEVICE>` at 1, 4 ranks. |
| 8 | Multi-rank correctness | ea70c9c | Both TEST cases pass at `ntasks=4` interactively on tuolumne. Batch script at [scripts/tuolumne/fmm_vs_exact.flux](../scripts/tuolumne/fmm_vs_exact.flux) covers HIP/OPENMP/SERIAL at 1 and 4 ranks for reproducibility. |

### Notes on checkpoint 1

- `Beatnik_add_dependency(PACKAGE Canopy)` at
  [../CMakeLists.txt:78](../CMakeLists.txt#L78) already sets the
  `Beatnik_ENABLE_CANOPY` CMake variable based on `find_package(Canopy)`.
  This checkpoint only had to:
  - Add `#cmakedefine BEATNIK_ENABLE_CANOPY` to
    [../src/Beatnik_Config.hpp.in](../src/Beatnik_Config.hpp.in).
  - Conditionally append `FmmBRSolver.hpp` to `HEADERS_PUBLIC` and
    `Canopy::Canopy` to `DEPENDS_ON` in
    [../src/CMakeLists.txt](../src/CMakeLists.txt).
- Canopy's exported target is `Canopy::Canopy` (interface library,
  confirmed via [canopy/src/CMakeLists.txt](../../canopy/src/CMakeLists.txt)).

## Explicitly out of scope (deferred)

- **Periodic boundary support.** Not needed in the foreseeable future.
  FmmBRSolver still `MPI_Abort`s at construction if either dim is
  periodic. If this changes, the path is 9× image replication of
  sources around the central tile.

## Bugs / follow-ups
- **Canopy's exported CMake config didn't `find_dependency(Trilinos)`.**
  `Canopy_Targets.cmake` lists Trilinos sub-targets
  (`Zoltan2::all_libs`, `Tpetra::all_libs`, `Teuchos*::all_libs`, ...)
  in `INTERFACE_LINK_LIBRARIES` for `Canopy::Canopy`, but
  `CanopyConfig.cmake` only `find_dependency`'d Kokkos and MPI.
  Consumers of `Canopy::Canopy` hit "target Zoltan2::all_libs not
  found" at cmake generate time. Fixed in Canopy as
  `ab1e8fc` ("CanopyConfig: find_dependency(Trilinos) before targets
  include") on the `redesign` branch.
- **`spack install` does not always pick up header-only changes.**
  `add_library(Beatnik INTERFACE)` in [../src/CMakeLists.txt](../src/CMakeLists.txt)
  means consumer .o files don't depend on `HEADERS_PUBLIC`. When only
  a header changes, `make` sees nothing to rebuild and `spack install`
  reports a sub-second build. Workaround: `touch
  examples/01_rocketrig/rocketrig.cpp` before `spack install` to force
  rocketrig to be re-compiled. Long-term fix: list the public headers
  as `target_sources(... INTERFACE FILE_SET HEADERS ...)` or attach a
  PUBLIC_HEADER property so consumers track them as real
  dependencies.

## Future optimization areas

- **Switch from `setup()` every step to `auto_maintain()`.** Currently
  FmmBRSolver rebuilds Canopy's tree on every `computeInterfaceVelocity`
  call — see the v1 simplification comment in
  [../src/FmmBRSolver.hpp](../src/FmmBRSolver.hpp). The plan target was
  to call `auto_maintain` on subsequent calls so Canopy picks
  Migrate/Rebalance/Rebuild based on actual particle drift. Requires
  a persistent forward distributor keyed on `tag.origin_rank`; design
  is sketched in [../plans/i-am-designing-a-clever-moonbeam.md](../plans/i-am-designing-a-clever-moonbeam.md).
