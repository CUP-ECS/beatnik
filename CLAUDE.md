# Beatnik

Control document for this repository. It carries only what applies to **every**
session; everything task-specific is in a reference doc below, to be read when
the task calls for it.

| Read this | When |
| --- | --- |
| `systems/<system>/claude.md` | **Before any build or run** — machine facts, per [System detection](#system-detection). |
| [docs/environment-and-build.md](docs/environment-and-build.md) | New checkout; build mode in doubt; editing the resolver, a profile or a batch script. |
| [docs/testing.md](docs/testing.md) | Running, adding or changing tests; tiers, the installed-path runners, CI. |
| [docs/framework-maintenance.md](docs/framework-maintenance.md) | Adding a system; writing a system doc; changing the gate; other framework edits. |
| [README.md](README.md) | Usage, API, examples, Known Issues, Future Optimizations. |
| [docs/design.md](docs/design.md) | Algorithms and design decisions. |
| `tasks/<topic>.md` | Working an ongoing multi-phase topic — read the topic file *first*, and log progress as work lands. |

## System detection

Build and run commands differ by system. **Before building or running anything,
run `hostname`**, match it against the table below, then Read the matching
instructions file and follow it for the rest of the session.

| Hostname pattern | Instructions file |
| ---------------- | ------------------------------- |
| `tuolumne*`      | `systems/tuolumne/claude.md`    |

The pattern is the alphabetic prefix of the host: `tuolumne2152` matches
`tuolumne*`, `dane1234` would match `dane*`. This table is mirrored by the `case`
statement in [scripts/lib/beatnik_env.sh](scripts/lib/beatnik_env.sh); the two
must always agree. If the hostname matches no row, **stop and ask the user**; do
not guess. See [docs/framework-maintenance.md](docs/framework-maintenance.md) to
add a system.

## Build mode

*How* the work is done in this checkout is per-instance, orthogonal to which
system it is on, and it decides which environment to activate, where binaries
land and how the gate runs. There are two modes: **`spack`** (the environment
builds Beatnik itself; no build tree) and **`manual`** (the environment supplies
dependencies only; binaries built out-of-tree into `BEATNIK_BUILD_DIR`).

Before building or submitting a job, resolve the mode rather than assuming it:

```
BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh
```

It prints the profile and `BEATNIK_PROFILE_SOURCE`. If that is
`profile.local.sh`, this checkout recorded its choice — use it, do not re-ask. If
it is `profile.defaults.sh`, the checkout is on the machine-wide fallback and has
never recorded a choice: **confirm with the user before building**, then write the
answers to the gitignored `scripts/<system>/profile.local.sh`. Reading code is
always safe on the fallback. Every batch script and run wrapper must source the
resolver **first**, and resolve binaries through `beatnik_exe` rather than
hardcoding paths.

Full mechanism, resolver knobs, scopes and the runtime-env convention:
[docs/environment-and-build.md](docs/environment-and-build.md).

## Minimum test set

The gate is defined by **tier label + backend(s) + rank counts** — never by
enumerating test names, which is how a gate silently shrinks.

> **The gate:** every test labeled **`regression`**, on the **SERIAL** backend
> (plus **HIP** on tuolumne), at **ranks 1, 2, 3, 4, 5, 6**.

```
ctest -L regression -R SERIAL      # project-wide gate
ctest -L regression -R HIP         # additional gate on tuolumne
```

In `spack` mode there is no build tree and therefore no `ctest`: run the gate
through `scripts/<system>/run_regression_minset.<scheduler>` — on tuolumne
[scripts/tuolumne/run_regression_minset.flux](scripts/tuolumne/run_regression_minset.flux).
The `regression` tier currently has **three** members — the initial condition at
0 timesteps (T1c), the direct Birkhoff-Rott sum (T2c), and ten timesteps against
the Python gold set (T2d) — so on tuolumne the gate is **36 launches** and takes
correspondingly longer to run. All 36 are green as of T2d. **`BEATNIK_TEST_SCRATCH`
must name a path on a parallel filesystem**, not a node-local one: the
checkpoints go through MPI-IO, so a node-local scratch fails every launch that
spans more than one node. Tiers, the installed-path runners and CI:
[docs/testing.md](docs/testing.md).

## General guidelines

- **Build the way the active mode says, and never mix modes.** In `spack` mode
  that means activating the environment and running `spack install`, never a
  hand-rolled `cmake`/`make`; in `manual` mode it means `cmake --build`, never
  `spack install`. A stale hand-built tree alongside an installed binary is how
  you end up testing something other than what you changed. Check the mode first.
- **Never launch a job interactively from a login node — submit a batch script.**
  On a login node there is no allocation to run in, so an interactive launch
  (`flux run`, `srun`, …) does not fail: it **blocks forever** waiting for
  resources that will never be granted, and the session hangs until it is killed.
  This includes "quick" one-off invocations and `--help`-adjacent smoke tests.
  Write the invocation into a batch script under `scripts/<system>/`, submit it
  (`flux batch …` on tuolumne), and read the `.log` file it writes. The
  interactive launch template in a system doc is only valid **inside** an
  allocation you already hold.
- **Never run the formatter — that is the user's job.** [.clang-format](.clang-format),
  [clangformat.sh](clangformat.sh) and the `cabana-format` CMake target all stay,
  and the user runs them by hand when they choose to. Do **not** invoke
  clang-format, `clangformat.sh` or `cabana-format` in a session, and do not
  reformat a file as part of another change; write and edit C/C++ in the style of
  the surrounding code and leave formatting alone. Running it unasked has mangled
  files here before, and it buries the real diff.
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
  them. **Never run `spack install` against the production env while a production
  job is live** — it overwrites the binary in place, and a running job whose
  executable pages change underneath it takes a SIGBUS (rc=135) and dies. This
  bit us on 2026-06-24: a reinstall during a live 64-node run SIGBUS-killed it.
  Reinstalling the *dev* env while a prod job runs is fine — they are separate
  install prefixes.
