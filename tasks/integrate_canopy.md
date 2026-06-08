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
| 3 | CreateBRSolver dispatch + Params extension | _pending_ | `spack install` succeeds (1m 4s). `FmmBRSolver` is currently the all-pairs ExactBRSolver body — runtime equivalence test is part of checkpoint 7. |

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

_(none yet)_

## Future optimization areas

_(none yet)_
