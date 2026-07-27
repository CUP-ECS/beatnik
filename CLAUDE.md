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
3. Build mode is `spack` — **build only via `spack install`, never manually.**
4. Build with that system doc's build command.
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

Use [tasks/TEMPLATE.md](tasks/TEMPLATE.md) as the skeleton so logs stay uniform:

```markdown
# <Topic>

## Problem
What is wrong / what we want, and why it matters.

## Approach
The strategy, the phases, and the key decisions/constraints.

## Progress log
- YYYY-MM-DD — <what was done, what was learned, what's next>
```

| Task log | Topic |
| --- | --- |
| _(none active)_ | — |

The three pre-redesign logs (`integrate_canopy.md`, `fmm_premature_nan.md`,
`fmm_fullrollup_crash.md`) were removed in `89ec015` with the solver they
tracked; they remain readable at `89ec015^:tasks/`. The durable outcome of the
still-open one is in README [Known Issues](README.md#known-issues). Design and
progress documents that used to live in `plans/` and `DESIGN.md` now belong in
`tasks/` and [docs/design.md](docs/design.md) respectively.

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

Orthogonal to *which system* you are on is *how the work is being done this
session*. That choice changes which environment to activate, where binaries land,
and how the gate runs. Beatnik supports two build modes:

- **`spack`** (default, **and the only mode sanctioned in this repository**) —
  `spack develop beatnik` + `spack install`. Binaries land on `PATH`; there is
  **no build tree**, so the gate instead runs the installed test binaries through
  the scheduler. Each system may define a **dev** environment and optionally a
  **prod** environment (see the general guideline on finalizing prod before big
  jobs).
- **`manual`** — the package manager provides only *dependencies*. Binaries are
  hand-compiled out-of-tree into `build-<system>/`, and the gate is the in-tree
  runner (`ctest` in the build dir). The resolver and the gate wrapper support
  this mode for portability to a system that needs it, but **do not use it here**
  — see the guideline below.

> **Build and install in this repository ONLY via `spack install`, after
> activating the spack environment.** On tuolumne that is:
>
> ```
> spack env activate ~/spack_envs/tuolumne_beatnik
> spack install
> ```
>
> Do **not** hand-run `cmake` / `make` / `cmake --build` against this checkout,
> and do not create an out-of-tree build directory to "just check that it
> configures". The dependency graph is resolved by the spack environment, so a
> manual configure outside it does not reflect a real build and can produce
> misleading failures.

### The profile mechanism

Per-checkout choices live in `scripts/<system>/profile.local.sh`, which is
**gitignored** and overrides the committed
`scripts/<system>/profile.defaults.sh`. Both are sourced, in that order, by the
resolver [scripts/lib/beatnik_env.sh](scripts/lib/beatnik_env.sh). A missing
local file simply means the committed defaults apply — a fresh checkout needs
**zero configuration**. Precedence is `environment > profile.local.sh >
profile.defaults.sh`, implemented with `${VAR:=default}`.

### Session-start flow

1. If `scripts/<system>/profile.local.sh` exists, Read it and use it. **Do not
   re-ask** — the choice has already been recorded.
2. Otherwise use **AskUserQuestion** to get the build mode, the spack env(s) to
   use, and (for manual mode) the build directory.
3. Write the answers to `scripts/<system>/profile.local.sh` so the next session
   does not ask again. Only write the values that differ from the defaults.

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

1. **Environment** — the concrete env path(s), plus any pre-activation step
   (e.g. `module load`). Where both a dev and a prod env exist, record both and
   note that new batch scripts must ask the user which one to target.
2. **Build-config args** — system-specific args passed to `cmake` (or to the
   spack spec).
3. **Build command** — per build mode. `spack install` for spack mode;
   `cmake --build` for manual mode.
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
  Diagnostic: it tells you *where* a fault is, but does not gate.

Registration and the tier lists are in [tests/CMakeLists.txt](tests/CMakeLists.txt),
which carries the same definition as a comment block. Run it via
`scripts/<system>/run_regression_minset.<scheduler>` — on tuolumne
[scripts/tuolumne/run_regression_minset.flux](scripts/tuolumne/run_regression_minset.flux).

**Promoting a test into the `regression` tier changes what must pass before
anything ships — confirm with the user first.** Conversely, a failing or flaky
test must never be quietly removed from the lists to green the gate: label it
`unit`, record it in README "Known Issues", and tell the user.

> **The `regression` tier is currently EMPTY.** `89ec015` removed the
> pre-redesign solver and its only end-to-end test to build a new solver from
> scratch. The gate is structurally correct but vacuous until the new solver
> lands its first end-to-end test — **a green gate right now proves nothing.**
> Recorded in README "Known Issues".

### CI

There is deliberately **no CI**. Beatnik's dependency stack (Kokkos, Cabana,
HeFFTe, Silo, optional Canopy, all via spack) plus GPU-aware MPI and a flux
scheduler is not reproducible in hosted CI in reasonable time, and the target
machines are behind a lab fence. The gate is **operator-run** via the
`run_regression_minset.*` wrapper. Since there is no CI job there is no
CI-vs-gate divergence to track; if CI is ever added it must invoke the same
wrapper or the same `ctest` command, and any difference must be stated here.

## General guidelines

- **Build and install only via `spack install`.** Activate the environment
  first (`spack env activate ~/spack_envs/tuolumne_beatnik` on tuolumne), then
  `spack install`. **Never build this repository manually** with `cmake`/`make`
  — see [Build & run profile](#build--run-profile).
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
