# Beatnik

Control document for this repository's build/run/test framework. It covers both
how to build, run and test Beatnik, and how to **extend the framework without
letting its pieces drift apart** (see [Maintaining this framework](#maintaining-this-framework)).

Project documentation lives elsewhere and is not duplicated here:
[README.md](README.md) for usage and API, [docs/design.md](docs/design.md) for
algorithms and design decisions, `systems/<system>/claude.md` for machine facts.

## New-checkout quickstart

1. `hostname`, then Read the matching `systems/<system>/claude.md` — every build
   and run command below is system-specific ([System detection](#system-detection)).
2. Only if you need to override the committed defaults, create
   `scripts/<system>/profile.local.sh` (gitignored). Otherwise skip it: the
   defaults work as-is ([Build & run profile](#build--run-profile)).
3. Confirm this checkout's build mode:
   `BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh`. If it reports the
   mode came from `profile.defaults.sh`, ask before building — that is the
   machine fallback, not this instance's recorded choice.
4. Build with that system doc's command **for the active mode** — `spack install`
   in spack mode, `cmake --build` in manual mode. Never both.
5. Run the gate: `flux batch scripts/<system>/run_regression_minset.flux`, or in
   manual mode `ctest -L regression -R SERIAL` in the build dir
   ([Minimum test set](#minimum-test-set)).

On tuolumne, steps 3–5 with zero configuration are: `spack env activate
~/spack_envs/tuolumne_beatnik && spack install`, then `flux batch
scripts/tuolumne/run_regression_minset.flux`.

## Background / task logs

Ongoing multi-phase problems live in `tasks/`, one file per topic, each recording
*why* a problem is being worked and *how* it is being attacked, with a dated
progress log so a later session can resume. **At the start of a session that
touches one of these topics, Read `tasks/<topic>.md` first**, and append progress
to it as work lands — not at the end, when the reasoning has been forgotten.

## System detection

Build and run commands differ by system. **Before building or running anything,
run `hostname`**, match it against the table below, then Read the matching
instructions file and follow it for the rest of the session.

| Hostname pattern | Instructions file |
| ---------------- | ------------------------------- |
| `tuolumne*`      | `systems/tuolumne/claude.md`    |

The pattern is the alphabetic prefix of the host: `tuolumne2152` matches
`tuolumne*`, `dane1234` would match `dane*`. This table is mirrored by the
`case` statement in [scripts/lib/beatnik_env.sh](scripts/lib/beatnik_env.sh);
the two must always agree.

Each `systems/<system>/` also holds a committed snapshot of that system's spack
environment (`spack.yaml`, and `spack-production.yaml` where a production env
exists), kept in sync with the live environment.

**Fallback:** if the hostname matches no row, or the matching doc is missing one
of the required sections below, **stop and ask the user** to fill the gap; do not
guess. To add a system, follow [Maintaining this framework](#maintaining-this-framework).

## Build & run profile

Orthogonal to *which system* you are on is *how the work is being done in this
checkout*. That choice changes which environment to activate, where binaries
land, and how the gate runs.

### Three scopes — put each fact in the right one

The most common way this framework goes wrong is recording a fact at the wrong
scope: a committed file then asserts something that is only true of one clone.

| Scope | Lives in | Committed? | Examples |
| --- | --- | --- | --- |
| **Project-wide** | this file, `tests/CMakeLists.txt` | yes | the gate definition; tier meanings; the license convention |
| **Per-system** | `systems/<system>/claude.md`, `scripts/<system>/profile.defaults.sh`, `scripts/<system>/runtime_env.sh` | yes | scheduler is flux; 4 ranks per node; HIP and SERIAL build here; the launch-time env |
| **Per-instance** | `scripts/<system>/profile.local.sh` | **no — gitignored** | *this checkout's* build mode, the env it is developed into, its build directory |

**Build mode is per-instance, not per-system and not per-repository.** Two clones
side by side on the same machine can legitimately differ — one `spack
develop`ed into an environment, one hand-built out-of-tree — and a clone on a
different system almost certainly does. `profile.defaults.sh` therefore holds
only a **fallback** representing the most common setup on that machine, so a
fresh clone runs with zero configuration. It is not a policy, and a checkout's
own choice must never be written back into it.

### The two build modes

- **`spack`** — `spack develop beatnik` + `spack install`. The environment
  provides dependencies *and* builds Beatnik itself; binaries land on `PATH`.
  There is **no build tree**, so the gate runs the installed test binaries
  through the scheduler. Each system may define a **dev** environment and
  optionally a **prod** environment (see the guideline on finalizing prod before
  big jobs).
- **`manual`** — the package manager provides only *dependencies*. Binaries are
  hand-compiled out-of-tree into `build-<system>/`, and the gate is the in-tree
  runner: `ctest` in the build dir.

> **Build rules are scoped to the ACTIVE MODE, not to the repository.** Check
> `BEATNIK_BUILD_MODE` before you build anything, then:
>
> - **In `spack` mode: build and install ONLY via `spack install`** after
>   activating the environment. Do **not** hand-run `cmake` / `make` /
>   `cmake --build`, and do not create an out-of-tree build directory even to
>   "just check that it configures" — the dependency graph is resolved by the
>   spack environment, so a configure outside it does not reflect a real build
>   and its failures are misleading rather than informative.
> - **In `manual` mode: `cmake` and `cmake --build "${BEATNIK_BUILD_DIR}"` are
>   the correct commands**, and `spack install` is not — the environment is
>   supplying dependencies only.
>
> What is wrong in either mode is **mixing** them: a stale hand-built tree
> alongside an installed binary is how you end up testing something other than
> what you changed. Each system doc gives the concrete command per mode.

**This checkout (`/g/g20/stewartj/spack_envs/tuolumne_beatnik/beatnik`) is in
`spack` mode**, recorded in its own `scripts/tuolumne/profile.local.sh`, so the
`spack` rule above is the one that applies here. Confirm with
`BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh` rather than assuming —
the summary reports the mode *and which file it came from*.

### The profile mechanism

Per-checkout choices live in `scripts/<system>/profile.local.sh`, which is
**gitignored** and overrides the committed
`scripts/<system>/profile.defaults.sh`. Both are sourced, in that order, by the
resolver [scripts/lib/beatnik_env.sh](scripts/lib/beatnik_env.sh). A missing
local file simply means the committed defaults apply — a fresh checkout needs
**zero configuration**. Precedence is `environment > profile.local.sh >
profile.defaults.sh`, implemented with `${VAR:=default}`.

### Session-start flow

1. Run `BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh`. It prints the
   resolved profile **and** `BEATNIK_PROFILE_SOURCE` — which of
   `environment` / `profile.local.sh` / `profile.defaults.sh` supplied the build
   mode.
2. If the source is **`profile.local.sh`**, this checkout has recorded its
   choice. Read the file and use it; **do not re-ask.**
3. If the source is **`profile.defaults.sh`**, this checkout has *never* recorded
   a choice and is running on the machine-wide fallback. That is a guess about
   this instance, so **do not silently rely on it for a build.** Use
   **AskUserQuestion** to confirm the build mode, the environment(s) to use, and
   (in manual mode) the build directory.
4. Write the answers to `scripts/<system>/profile.local.sh` — gitignored, so it
   stays with this instance — recording only values that differ from the
   fallbacks. The next session then takes path 2.

Reading is always safe on the fallbacks; it is *building* and *submitting jobs*
that need the mode confirmed. Scripts deliberately do **not** follow this flow:
they fall straight through to the fallbacks so a batch job never blocks waiting
on a prompt.

A minimal `profile.local.sh` looks like
[scripts/profile.local.sh.example](scripts/profile.local.sh.example).

### Resolver knobs

Every batch script and run wrapper must `source scripts/lib/beatnik_env.sh`
**first**, before anything else. It resolves the repo root (`BEATNIK_REPO`),
detects the system, applies the profile, activates the environment, sources the
per-system runtime env, and exposes `beatnik_exe`.

| Knob | Meaning |
| --- | --- |
| `BEATNIK_SYSTEM` | Override hostname-based system detection. |
| `BEATNIK_BUILD_MODE` | `spack` or `manual`. |
| `BEATNIK_BIN_MODE` | Derived from build mode: `installed` or `tree`. Set only to override. |
| `BEATNIK_SPACK_ENV` | Development spack environment path. |
| `BEATNIK_SPACK_PROD_ENV` | Production spack environment path. |
| `BEATNIK_BUILD_DIR` | Out-of-tree cmake build dir (manual mode). |
| `BEATNIK_USE_PROD=1` | Activate the production env instead of dev. |
| `BEATNIK_NO_SPACK_ACTIVATE=1` | Skip env activation *and* `runtime_env.sh`; the caller manages the shell. |
| `BEATNIK_ENV_DRY_RUN=1` | Resolve and print the profile, change nothing. |
| `BEATNIK_PROFILE_SOURCE` | *Output, not input.* Which source supplied the build mode: `environment`, `profile.local.sh`, or `profile.defaults.sh`. The last means this checkout never recorded a choice. |
| `BEATNIK_TEST_MPI_RANKS` | CMake cache var: rank counts MPI tests register at. Default `1;2;3;4;5;6`. |

`beatnik_exe <relpath|name>` resolves a binary for the active bin mode —
`command -v <name>` on the env's PATH in `installed` mode, a path under
`BEATNIK_BUILD_DIR` in `tree` mode. **Use it instead of hardcoding binary
paths**; that is what lets one script work in both modes.

Quick check of what the resolver thinks:

```
BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh
```

## Per-system runtime environment

Launch-time environment variables that must reach scheduler-launched tasks
(GPU-aware MPI toggles, fabric settings, OpenMP placement) live in **one file per
system**, `scripts/<system>/runtime_env.sh`, which the resolver sources
automatically. Omit the file on a system that needs none. It is skipped under
`BEATNIK_NO_SPACK_ACTIVATE=1`.

**Batch scripts must not re-export these inline.** Two copies drift, and a stale
copy in one script makes that script's runs quietly unreproducible.

## Compute backends

Beatnik builds for multiple Kokkos backends — the CMake-supported set is
`SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET`. **Which of them actually
build and run is a per-system fact**, declared in each system's
`systems/<system>/claude.md` "Backends" section, because it follows from that
system's hardware and spack spec.

Gate policy:

- **`SERIAL` everywhere** — the project-wide gate backend.
- Systems add GPU or threading gates where supported. On tuolumne the gate is
  **SERIAL + HIP**; `OPENMP` builds and runs but is not gated.

Tests carry the backend as a name suffix (e.g. `Beatnik_Test_Foo_MPI_SERIAL`),
so the gate selects a backend with `-R <backend>` layered on top of the tier
label `-L`.

## Required sections in every `systems/<system>/claude.md`

1. **Environment** — the env path(s) *available on this machine*, plus any
   pre-activation step (e.g. `module load`). Where both a dev and a prod env
   exist, record both and note that new batch scripts must ask the user which one
   to target. Which env a given checkout is developed into is per-instance and
   belongs in that checkout's `profile.local.sh`.
2. **Build-config args** — system-specific args passed to `cmake` (or to the
   spack spec).
3. **Build command** — **one entry per build mode**, since the mode is a property
   of the checkout, not the machine: `spack install` for spack mode,
   `cmake --build` for manual mode. Describe both even if one is rare here, and
   scope any prohibition to a mode rather than to the machine or the repository.
4. **Run command for binaries** — the scheduler launch template, with the binary
   resolved via `beatnik_exe`.
5. **Job-scheduler batch template** — a concrete template under
   `scripts/<system>/`, sourcing the resolver first and branching on
   `BEATNIK_BIN_MODE` where behavior differs.
6. **Running non-test binaries** — when asked to run an `examples/` problem, ask
   the user for the example name and its arguments, then plug them into
   sections 4–5.
7. **Backends** — which compute backends build and run here, and which the gate
   runs here.

The tests themselves are project-wide and live in
[Minimum test set](#minimum-test-set), not in a per-system doc. A system doc only
says *how* to run a test on that machine.

## Minimum test set

The gate is defined by **tier label + backend(s) + rank counts** — never by
enumerating test names, which is how a gate silently shrinks.

> **The gate:** every test labeled **`regression`**, on the **SERIAL** backend
> (plus **HIP** on tuolumne), at **ranks 1, 2, 3, 4, 5, 6**.

```
ctest -L regression -R SERIAL      # project-wide gate
ctest -L regression -R HIP         # additional gate on tuolumne
```

Ranks come from the `BEATNIK_TEST_MPI_RANKS` cache variable (default
`1;2;3;4;5;6`), which registers one ctest case per rank count, so the two
commands above cover the whole sweep.

The tiers:

- **`regression`** — full end-to-end runs composing the whole pipeline. **This is
  the gate.** Everything here must pass before a code change ships.
- **`unit`** — utilities, kernels, single-component and single-phase tests.
  Diagnostic: it tells you *where* a fault is, but does not gate. Run the whole
  tier with `ctest -L unit`, or through the wrapper
  `scripts/<system>/unit_tests.<scheduler>` — on tuolumne
  [scripts/tuolumne/unit_tests.flux](scripts/tuolumne/unit_tests.flux).

### Running tests in `spack` mode: there is no build tree, so there is no ctest

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
  [tests/unit_tests/Beatnik_TestAssert.hpp](tests/unit_tests/Beatnik_TestAssert.hpp).

The two wrappers find their work through generated manifests
(`beatnik_gate_manifest.txt`, `beatnik_unit_manifest.txt`) emitted by the same
registrations that apply the tier labels, so the installed path and `ctest`
cannot drift. **A `WILL_FAIL` test needs explicit handling on the installed
path** — the unit manifest spells it `py-fail` — because a runner that treated
the comparator's negative case as an ordinary test would report the tier red
exactly when the comparator is working, and a *missing* fixture would make it
pass for the wrong reason.

Registration and the tier lists are in [tests/CMakeLists.txt](tests/CMakeLists.txt),
which carries the same definition as a comment block. Run it via
`scripts/<system>/run_regression_minset.<scheduler>` — on tuolumne
[scripts/tuolumne/run_regression_minset.flux](scripts/tuolumne/run_regression_minset.flux).

**Promoting a test into the `regression` tier changes what must pass before
anything ships — confirm with the user first.** Conversely, a failing or flaky
test must never be quietly removed from the lists to green the gate: label it
`unit`, record it in README "Known Issues", and tell the user.

> **The `regression` tier has ONE member** as of 2026-08-12 (task T1c):
> `Beatnik_Test_InitialConditions`, regression test 1 — the whole driver path at
> 0 timesteps against a Python gold checkpoint. The tier was empty from `89ec015`
> (which removed the pre-redesign solver and its only end-to-end test) until then,
> and the gate was vacuous; it is not any more. **But it covers only what
> exists** — mesh generation, the initial condition and the checkpoint write.
> There is no timestep and no adaptivity yet, so a green gate does not say the
> solver integrates anything. See `tasks/framework.md`.

### CI

**The gate has no CI and is operator-run.** Beatnik's dependency stack (Kokkos,
Cabana, HeFFTe, Tessera, optional Canopy, all via spack) plus GPU-aware MPI and
a flux scheduler is not reproducible in hosted CI in reasonable time, and the
target machines are behind a lab fence. Run it with the
`run_regression_minset.*` wrapper.

There is exactly **one** GitHub Actions workflow,
[.github/workflows/regression-compare.yml](.github/workflows/regression-compare.yml),
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
[tests/CMakeLists.txt](tests/CMakeLists.txt); change one and change the other.

**A green check on that workflow says nothing about the solver.** Do not read it
as gate coverage, and do not extend it to build Beatnik without revisiting this
section.

## General guidelines

- **Build the way the active mode says, and never mix modes.** In `spack` mode
  (this checkout) that means activating the environment and running
  `spack install`, never a hand-rolled `cmake`/`make`; in `manual` mode it means
  `cmake --build`, never `spack install`. Check the mode first — see
  [Build & run profile](#build--run-profile).
- **Never launch a job interactively from a login node — submit a batch script.**
  On a login node there is no allocation to run in, so an interactive launch
  (`flux run`, `srun`, …) does not fail: it **blocks forever** waiting for
  resources that will never be granted, and the session hangs until it is killed.
  This includes "quick" one-off invocations and `--help`-adjacent smoke tests.
  Write the invocation into a batch script under `scripts/<system>/`, submit it
  (`flux batch …` on tuolumne), and read the `.log` file it writes. The
  interactive launch template in a system doc is only valid **inside** an
  allocation you already hold.
- **Checkpoint commits in tasks.** When planning a large change, put explicit
  checkpoints in the task log where progress should be committed, so a later
  failure can roll back to the nearest one instead of unwinding everything.
- **Follow the formatter config.** [.clang-format](.clang-format) governs all
  C/C++ written or edited here. Fast helper: [clangformat.sh](clangformat.sh)
  (formats every `.cpp`/`.hpp` outside build dirs); the CMake target
  `cabana-format` does the same via `CLANG_FORMAT_EXECUTABLE`.
- **Match the license/header convention.** Every new source and script file
  carries the project's BSD-3-Clause header block with
  `SPDX-License-Identifier: BSD-3-Clause`, in the comment style of that file
  type, exactly as existing files do.
- **Keep `README.md` in sync.** When a public API changes, **or when an example's
  accepted arguments change**, update the README in the same change.
- **Track optimization opportunities.** When you notice one, ask the user before
  adding it to README "Future Optimizations".
- **Record known issues** in README "Known Issues": what fails, how it
  reproduces, and whether it is a regression from current work or pre-existing.
- **Finalize the production env before submitting big jobs (any HPC system).**
  On any system with a separate production env, make "prod env up to date +
  built" a gate *before* `flux batch`/`sbatch`/`srun` of a large or long-running
  job: pull the dependency source clones (e.g. canopy and beatnik) to the
  intended commits and run `spack install` **first**, so the binary reflects
  them. **Never run the install against the production env while a production job
  is live** — `spack install` overwrites the binary in place, and a running job
  whose executable pages change underneath it takes a SIGBUS (rc=135) and dies.
  This bit us on 2026-06-24: a reinstall during a live 64-node run SIGBUS-killed
  it.

## Maintaining this framework

Meta-rules that keep the pieces consistent. Breaking one of these does not fail
loudly — it just makes the framework quietly wrong later.

- **Adding a system.** Do all of it in one change: create
  `systems/<system>/claude.md` with all seven required sections; add a row to the
  [System detection](#system-detection) hostname table; add a matching `case`
  branch in [scripts/lib/beatnik_env.sh](scripts/lib/beatnik_env.sh); add
  `scripts/<system>/profile.defaults.sh` (and `runtime_env.sh` if the system
  needs launch-time exports); add the gate wrapper
  `scripts/<system>/run_regression_minset.<scheduler>`; declare the system's
  backends and which of them the gate runs there; and commit an env snapshot
  under `systems/<system>/`.
- **Never record a per-instance fact in a committed file.** A committed file
  asserts something about *every* clone. This checkout's build mode, the specific
  environment it is developed into, and its build directory go in the gitignored
  `scripts/<system>/profile.local.sh` — not in `profile.defaults.sh`, not in
  `systems/<system>/claude.md`, and not in this file. When adding guidance, ask
  which of the three scopes it belongs to; if the answer is "it depends on the
  checkout", it is per-instance. Corollary: policy statements in committed files
  must be scoped to the **active mode** (`BEATNIK_BUILD_MODE`) rather than
  written as blanket repository rules, because a blanket rule is a per-instance
  fact in disguise.
- **Env snapshots stay in sync.** Changing a live spack environment means
  updating `systems/<system>/spack*.yaml` in the same change. A stale snapshot is
  worse than none, because it looks authoritative.
- **No inline runtime-env in batch scripts.** Source the resolver; edit
  `scripts/<system>/runtime_env.sh` when a launch-time variable changes.
- **The gate definition is single-sourced.** The tier label, backend(s) and rank
  set must read identically in this file, in
  [tests/CMakeLists.txt](tests/CMakeLists.txt), in the `run_regression_minset.*`
  wrapper(s), in each system doc's Backends section, and in CI if one is ever
  added. Changing it is deliberate and confirmed with the user — never a side
  effect.
- **Example argument changes mirror into README.**
- **New files carry the license/SPDX header.**
- **Keep the two directory trees apart.** `systems/<system>/` holds
  hostname-keyed machine instructions and env snapshots and nothing else;
  `docs/` holds human-facing project documentation and no machine instructions.
