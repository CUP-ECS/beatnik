# Build & run reference

How a checkout decides *where* it builds and *what* it runs. Read this when
setting up a new checkout, editing the resolver or a profile file, writing a
batch script, or when the build mode is in doubt.

The project-wide rules that apply every session are in [CLAUDE.md](../CLAUDE.md);
machine facts are in `systems/<system>/claude.md`.

## New-checkout quickstart

1. `hostname`, then Read the matching `systems/<system>/claude.md` — every build
   and run command is system-specific.
2. Only if you need to override the committed defaults, create
   `scripts/<system>/profile.local.sh` (gitignored). Otherwise skip it: the
   defaults work as-is.
3. Confirm this checkout's build mode:
   `BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh`. If it reports the
   mode came from `profile.defaults.sh`, ask before building — that is the
   machine fallback, not this instance's recorded choice.
4. Build with that system doc's command **for the active mode** — `spack install`
   in spack mode, `cmake --build` in manual mode. Never both.
5. Run the gate: `flux batch scripts/<system>/run_regression_minset.flux`, or in
   manual mode `ctest -L regression -R SERIAL` in the build dir.

On tuolumne, steps 3–5 with zero configuration are: `spack env activate
~/spack_envs/tuolumne_beatnik && spack install`, then `flux batch
scripts/tuolumne/run_regression_minset.flux`.

## Three scopes — put each fact in the right one

The most common way this framework goes wrong is recording a fact at the wrong
scope: a committed file then asserts something that is only true of one clone.

| Scope | Lives in | Committed? | Examples |
| --- | --- | --- | --- |
| **Project-wide** | `CLAUDE.md`, `docs/`, `tests/CMakeLists.txt` | yes | the gate definition; tier meanings; the license convention |
| **Per-system** | `systems/<system>/claude.md`, `scripts/<system>/profile.defaults.sh`, `scripts/<system>/runtime_env.sh` | yes | scheduler is flux; 4 ranks per node; HIP and SERIAL build here; the launch-time env |
| **Per-instance** | `scripts/<system>/profile.local.sh` | **no — gitignored** | *this checkout's* build mode, the env it is developed into, its build directory |

**Build mode is per-instance, not per-system and not per-repository.** Two clones
side by side on the same machine can legitimately differ — one `spack
develop`ed into an environment, one hand-built out-of-tree — and a clone on a
different system almost certainly does. `profile.defaults.sh` therefore holds
only a **fallback** representing the most common setup on that machine, so a
fresh clone runs with zero configuration. It is not a policy, and a checkout's
own choice must never be written back into it.

## The two build modes

- **`spack`** — `spack develop beatnik` + `spack install`. The environment
  provides dependencies *and* builds Beatnik itself; binaries land on `PATH`.
  There is **no build tree**, so the gate runs the installed test binaries
  through the scheduler. Each system may define a **dev** environment and
  optionally a **prod** environment.
- **`manual`** — the package manager provides only *dependencies*. Binaries are
  hand-compiled out-of-tree into `build-<system>/`, and the gate is the in-tree
  runner: `ctest` in the build dir.

Build rules are scoped to the **active mode**, not to the repository. Check
`BEATNIK_BUILD_MODE` before you build anything, then:

- **In `spack` mode: build and install ONLY via `spack install`** after
  activating the environment. Do **not** hand-run `cmake` / `make` /
  `cmake --build`, and do not create an out-of-tree build directory even to
  "just check that it configures" — the dependency graph is resolved by the
  spack environment, so a configure outside it does not reflect a real build
  and its failures are misleading rather than informative.
- **In `manual` mode: `cmake` and `cmake --build "${BEATNIK_BUILD_DIR}"` are
  the correct commands**, and `spack install` is not — the environment is
  supplying dependencies only.

What is wrong in either mode is **mixing** them: a stale hand-built tree
alongside an installed binary is how you end up testing something other than
what you changed. Each system doc gives the concrete command per mode.

## The profile mechanism

Per-checkout choices live in `scripts/<system>/profile.local.sh`, which is
**gitignored** and overrides the committed
`scripts/<system>/profile.defaults.sh`. Both are sourced, in that order, by the
resolver [scripts/lib/beatnik_env.sh](../scripts/lib/beatnik_env.sh). A missing
local file simply means the committed defaults apply — a fresh checkout needs
**zero configuration**. Precedence is `environment > profile.local.sh >
profile.defaults.sh`, implemented with `${VAR:=default}`.

A minimal `profile.local.sh` looks like
[scripts/profile.local.sh.example](../scripts/profile.local.sh.example).

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

## Resolver knobs

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
