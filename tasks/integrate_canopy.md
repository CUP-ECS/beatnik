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

## Bugs / follow-ups

- **Update Beatnik's spack `package.py` to add a `+canopy` variant.**
  Today the spack-side `beatnik` package has no Canopy variant — see
  `spack info beatnik` (variants are `cuda`, `openmp`, `rocm` only).
  Whether Beatnik builds with Canopy is decided purely by whether
  `find_package(Canopy)` happens to succeed at configure time, which
  in turn depends on Canopy being installed in the env *before*
  beatnik. That's fragile: any spack reconcretization that schedules
  beatnik before canopy silently disables FMM support. Add a
  `+canopy` variant that explicitly `depends_on('canopy', when='+canopy')`
  and passes `-DBeatnik_REQUIRE_CANOPY=ON` to cmake. Until then,
  build Canopy first by spec, e.g.
  `spack install canopy@develop amdgpu_target=gfx942 +profiling ldflags=... ldlibs=... %cce`,
  then `spack install` the rest of the env.
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

_(none yet)_
