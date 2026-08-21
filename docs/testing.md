# Testing reference

Read this when running, adding, or changing tests. The gate definition itself is
single-sourced in [CLAUDE.md](../CLAUDE.md#minimum-test-set) — this document
explains the tiers, how the installed path runs them, and what the gate does and
does not cover.

## The tiers

- **`regression`** — full end-to-end runs composing the whole pipeline. **This is
  the gate.** Everything here must pass before a code change ships.
- **`milestone`** — long end-to-end runs against a multi-thousand-step reference
  gold set, at ranks **1 and 4** on SERIAL and HIP. **Not the gate**, and
  deliberately so: a 2000-step run in front of every change is a stall, not a
  gate. Run it on demand with `ctest -L milestone`, or through the wrapper
  `scripts/<system>/run_milestone.<scheduler>` — on tuolumne
  [scripts/tuolumne/run_milestone.flux](../scripts/tuolumne/run_milestone.flux).
  Created by task M0-T1 and filled by M0-T3, which registered its **two**
  members: `Beatnik_Test_Milestone0Frozen` (2000 frozen-mesh timesteps at
  `--icosphere-subdivisions 3` against the M0-G1 gold set, all 81 checkpointed
  steps at `--rtol 1e-10 --atol 1e-12`) and `Beatnik_Test_Milestone0FrozenL4`
  (the same at subdivisions 4 against M0-G2). Two members x two backends x two
  rank counts is **eight launches**, about 35 minutes on tuolumne, which is what
  the wrapper's `-t 40m` is sized for. The wrapper still exits non-zero if the
  manifest names nothing runnable, exactly as the gate wrapper does for an empty
  gate. Its rank sweep comes from `BEATNIK_MILESTONE_MPI_RANKS`
  (default `1;4`) for ctest and `BEATNIK_MILESTONE_RANKS` in the wrapper.
  A `milestone` failure is a real failure: fix it or record it in README
  "Known Issues" — it is never a reason to change the gate.
- **`unit`** — utilities, kernels, single-component and single-phase tests.
  Diagnostic: it tells you *where* a fault is, but does not gate. Run the whole
  tier with `ctest -L unit`, or through the wrapper
  `scripts/<system>/unit_tests.<scheduler>` — on tuolumne
  [scripts/tuolumne/unit_tests.flux](../scripts/tuolumne/unit_tests.flux).

Ranks come from the `BEATNIK_TEST_MPI_RANKS` cache variable (default
`1;2;3;4;5;6`) — `BEATNIK_MILESTONE_MPI_RANKS` (default `1;4`) for the
`milestone` tier — which registers one ctest case per rank count, so a single
`ctest -L <tier> -R <backend>` covers the whole sweep.

Registration and the tier lists are in
[tests/CMakeLists.txt](../tests/CMakeLists.txt), which carries the same gate
definition as a comment block.

**Promoting a test into the `regression` tier changes what must pass before
anything ships — confirm with the user first.** Conversely, a failing or flaky
test must never be quietly removed from the lists to green the gate: label it
`unit`, record it in README "Known Issues", and tell the user.

## Running tests in `spack` mode: there is no build tree, so there is no ctest

Both wrappers exist because of one fact worth stating separately from the gate
definition: **in `spack` mode the build tree is discarded, so `ctest` cannot be
run at all.** A test that exists only in that tree cannot be run either. Two
consequences, and both are load-bearing:

- **Test artifacts must install.** Under `+testing` the package sets
  `Beatnik_ENABLE_TESTING` *and* `Beatnik_INSTALL_TEST_EXECUTABLES` and prepends
  `share/Beatnik/tests` to `PATH`, and `tests/CMakeLists.txt` installs the
  regression tier's script and gold fixtures alongside the binaries, mirroring
  the repo's `tests/` layout. A regression test installed without its gold file
  is not installed.
- **Every test must return non-zero on failure**, because a directly-launched
  binary is judged by its exit code and nothing else. That is also all ctest
  needs, so it costs nothing in tree mode — see
  [tests/unit_tests/Beatnik_TestAssert.hpp](../tests/unit_tests/Beatnik_TestAssert.hpp).

The three wrappers find their work through generated manifests
(`beatnik_gate_manifest.txt`, `beatnik_milestone_manifest.txt`,
`beatnik_unit_manifest.txt`) emitted by the same
registrations that apply the tier labels, so the installed path and `ctest`
cannot drift. An empty tier still gets its manifest — with zero non-comment
lines — because a missing manifest and an empty one are different failures and
the wrappers report them differently. **A `WILL_FAIL` test needs explicit handling on the installed
path** — the unit manifest spells it `py-fail` — because a runner that treated
the comparator's negative case as an ordinary test would report the tier red
exactly when the comparator is working, and a *missing* fixture would make it
pass for the wrong reason.

## What the gate currently covers

> **The `regression` tier has FIVE members** as of T4b, so the gate is
> **60 launches** on tuolumne, and all five are green:
> `Beatnik_Test_InitialConditions` (regression test 1, T1c — the whole driver
> path at 0 timesteps against a Python gold checkpoint),
> `Beatnik_Test_BirkhoffRott` (T2c — the vertex quadrature and the direct BR
> sum), `Beatnik_Test_DirectSolve10Steps` (regression test 2, T2d — ten TVD-RK3
> timesteps against the T2a gold set), `Beatnik_Test_RefineSplitEdges`
> (regression test 4, T4a — twenty timesteps with indicator-driven refinement
> through `Tessera::splitEdges()`) and `Beatnik_Test_DynamicRemeshSplit`
> (regression test 5, T4b — twenty timesteps of metric-driven dynamic remeshing
> with only the split third live). The tier was empty from `89ec015` (which
> removed the pre-redesign solver and its only end-to-end test) until T1c, and
> the gate was vacuous; it is not any more. **But it covers only what exists** —
> both adaptivity paths are covered for their *refinement* halves only, and only
> for the structural and shape claims those two tests make. Nothing in the gate
> coarsens: collapse, quality flips and the isotropic cleanup are T4d, blocked
> upstream, and any configuration reaching one is rejected before the first
> step. See `tasks/framework.md`.

## CI

**The gate has no CI and is operator-run.** Beatnik's dependency stack (Kokkos,
Cabana, HeFFTe, Tessera, optional Canopy, all via spack) plus GPU-aware MPI and
a flux scheduler is not reproducible in hosted CI in reasonable time, and the
target machines are behind a lab fence. Run it with the
`run_regression_minset.*` wrapper.

There is exactly **one** GitHub Actions workflow,
[.github/workflows/regression-compare.yml](../.github/workflows/regression-compare.yml),
and it is deliberately narrower than the gate:

| | CI workflow | The gate |
| --- | --- | --- |
| What runs | `tests/regression_tests/compare_output.py` on the committed fixtures | `ctest -L regression -R SERIAL` (+ `HIP`) at ranks 1-6 |
| What it proves | the gold-file **comparator** works | **Beatnik** works |
| Builds C++ | no | yes |

The exception is justified by the comparator sharing none of the reasons the
rest cannot be built in CI: it is pure Python over numpy and h5py, runs against
committed fixtures, and is the piece most likely to be edited by someone who
cannot run the full stack. It invokes the **same command on the same fixtures**
as the `Beatnik_Test_PythonCompare[_Negative]` ctest cases in
[tests/CMakeLists.txt](../tests/CMakeLists.txt); change one and change the other.

**A green check on that workflow says nothing about the solver.** Do not read it
as gate coverage, and do not extend it to build Beatnik without revisiting this
section.
