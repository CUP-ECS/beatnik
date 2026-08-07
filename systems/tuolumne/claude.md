# Tuolumne

LLNL AMD MI300A system. Hostnames match `tuolumne*`. Scheduler: **flux**.
**4 ranks per node**, 1 GPU and 24 cores per rank, GPU arch `gfx942`.

This file is the *only* place tuolumne-specific build/run facts live. Project
design prose belongs in [docs/design.md](../../docs/design.md), not here.

## 1. Environment

Tuolumne has two spack environments for this project:

- **Development** (`~/spack_envs/tuolumne_beatnik`) — builds and iterative
  work. Rebuilt frequently as code changes.
- **Production** (`~/spack_envs/tuolumne_beatnik_production`) — large-scale runs
  that may sit in the queue for hours. Pinning queued jobs here means ongoing
  development rebuilds cannot break a job that is already queued or running.

No pre-activation `module load` is required; the default module set is enough.

Both paths are the committed **fallbacks** in
[scripts/tuolumne/profile.defaults.sh](../../scripts/tuolumne/profile.defaults.sh)
as `BEATNIK_SPACK_ENV` and `BEATNIK_SPACK_PROD_ENV`. They say "these envs exist on
tuolumne", not "your checkout uses them" — a checkout developed into a different
env overrides them in its own `profile.local.sh`. **Do not hardcode
`spack env activate` in a script** — source the resolver, which activates the
dev env by default and the production env under `BEATNIK_USE_PROD=1`:

```
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh"
```

For interactive work, activating by hand is fine:

```
spack env activate ~/spack_envs/tuolumne_beatnik
```

When generating a new batch script under `scripts/tuolumne/`, **ask the user
whether it should target the production or the development env** before writing
the `BEATNIK_USE_PROD` line.

Committed env snapshots, kept in sync with the live environments in the same
change that alters them:

| Snapshot | Live environment |
| --- | --- |
| [spack.yaml](spack.yaml) | `~/spack_envs/tuolumne_beatnik` |
| [spack-production.yaml](spack-production.yaml) | `~/spack_envs/tuolumne_beatnik_production` |

The two differ only in `profiling_level` (dev 2, prod 1).

## 2. Build-config args

No tuolumne-specific CMake args are needed beyond the project defaults — a
plain `cmake ..` inside the activated env is enough. The spack spec carries
everything that matters:

```
beatnik@develop +testing +canopy +examples +profiling profiling_level=2 \
    +rocm amdgpu_target=gfx942 build_type=RelWithDebInfo %cce
```

To change what is built, edit the spec in the environment's `spack.yaml` and
update the [spack.yaml](spack.yaml) snapshot here in the same change — do not
reach for a hand-rolled `cmake` invocation (see section 3).

Update this section if that changes.

## 3. Build command

**Which of the two applies is a property of your checkout, not of tuolumne.**
Check it before building — the mode and the file it came from are both reported:

```
BEATNIK_ENV_DRY_RUN=1 source scripts/lib/beatnik_env.sh
```

If that says the mode came from `profile.defaults.sh`, the checkout has never
recorded a choice and you are looking at the machine fallback — confirm with the
user before building, then write `scripts/tuolumne/profile.local.sh`.

### spack mode

The fallback on tuolumne, and the mode most checkouts here use. The env already
has `spack develop beatnik`, so this builds the working tree in place:

```
spack env activate ~/spack_envs/tuolumne_beatnik
spack install
```

In this mode, **do not hand-build**: no `cmake`, no `make`, no out-of-tree build
directory, not even to check that the project configures. The dependency graph
comes from the spack environment, so a configure outside it does not represent a
real build and its failures are misleading rather than informative.

### manual mode

For a checkout that hand-compiles out-of-tree (`BEATNIK_BUILD_MODE=manual` in its
`profile.local.sh`). The env supplies dependencies only, so `spack install` is
*not* the build command here:

```
spack env activate ~/spack_envs/tuolumne_beatnik   # dependencies only
cmake -S "${BEATNIK_REPO}" -B "${BEATNIK_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBeatnik_ENABLE_TESTING=ON \
    -DBeatnik_ENABLE_EXAMPLES=ON
cmake --build "${BEATNIK_BUILD_DIR}" -j
```

`BEATNIK_BUILD_DIR` defaults to `${BEATNIK_REPO}/build-tuolumne`, derived from the
repo root so two clones cannot collide on one build tree.

**Do not mix the modes in one checkout.** A stale hand-built tree sitting beside
an installed binary is how you end up running something other than what you
changed.

### Required: clear stray `HIPCC_*_FLAGS_APPEND` before installing

With the default modules loaded, the interactive shell can carry a contaminated
`HIPCC_LINK_FLAGS_APPEND` (a stray leading `.`). Spack's `hip` package *appends*
to whatever value is already exported, so the `.` survives into the cabana build
and hipcc passes it to the linker as `--hip-link .`, which fails with:

```
ld.lld: error: cannot open .: Is a directory
... The C++ compiler "/opt/rocm-.../bin/hipcc" is not able to compile a simple test program.
```

This only triggers when a hip-using package (e.g. cabana) actually builds — if
it is already cached in the spack store, `spack install` skips it and the bug
stays hidden. Before installing (especially in a freshly concretized env where
cabana must build), clear the variables and re-concretize:

```
unset HIPCC_LINK_FLAGS_APPEND HIPCC_COMPILE_FLAGS_APPEND
spack concretize -f
spack install
```

A fresh login shell with only the standard modules also works, since nothing in
the spack config or any modulefile sets these variables — the `.` comes from
leftover interactive shell state.

### Header-only rebuild caveat

`Beatnik` is a CMake INTERFACE library that does not track `HEADERS_PUBLIC` as
dependencies, so when only a header changes `spack install` can report a
sub-second no-op build. Workaround: `touch` a consumer `.cpp` (e.g. the example
driver) first. See README "Future Optimizations" for the real fix.

## 4. Run command for binaries

Resolve the binary with `beatnik_exe` rather than hardcoding a path — that is
what makes the same script work in both build modes:

```
EXE="$(beatnik_exe [EXECUTABLE_NAME])" || exit 1
```

The launch-time environment (`MPICH_GPU_SUPPORT_ENABLED`, `GTL_*`, `FI_CXI_ATS`,
`HSA_XNACK`, `OMP_*`) is exported by
[scripts/tuolumne/runtime_env.sh](../../scripts/tuolumne/runtime_env.sh), which
the resolver sources automatically. **Do not re-export those inline.**

Then launch. **`flux run` is only valid inside an allocation you already hold.**
On a login node there is nothing to run in, so it does not error — it blocks
forever waiting for resources and hangs the session. From a login node, always go
through section 5 and `flux batch` a script instead, even for a one-off smoke
test.

```
flux run \
    --ntasks=[NUM_PROCS] \
    --nodes=[NUM_PROCS / 4] \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${EXE}" [EXTRA_ARGS]
```

Tuolumne runs 4 ranks per node, so `--nodes` is derived from `--ntasks` as
`ntasks / 4`, rounded up. `[NUM_PROCS]` should be a multiple of 4 to fill a
node (round up if needed).

## 5. Job-scheduler batch template

Tuolumne uses the **flux** scheduler. When not inside an interactive
allocation, generate a batch script from
[scripts/tuolumne/test_template.flux](../../scripts/tuolumne/test_template.flux),
save it under `scripts/tuolumne/`, and submit it:

```
flux batch scripts/tuolumne/<your_script>.flux
```

Every such script must, in this order:

1. Put any `module load` first (tuolumne needs none today).
2. Pin `BEATNIK_REPO`.
3. `source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh"` — **before anything
   else**, and never re-export the runtime env inline.
4. Branch on `BEATNIK_BIN_MODE` where behavior differs between a build tree and
   installed binaries (see the gate wrapper for the canonical example).

The `# flux: --output={{name}}.{{jobid}}.log` line writes stdout/stderr to a
`.log` file in the submitting directory. Read that log when the job finishes.

Inside a batch script, pick `--ntasks` (a multiple of 4) and set
`--nodes = ntasks / 4` in **both** the flux header and the `flux run` line.

The ship gate on this system is
[scripts/tuolumne/run_regression_minset.flux](../../scripts/tuolumne/run_regression_minset.flux).

## 6. Running non-test binaries

When asked to run something other than a test (e.g. an `examples/` problem),
**ask the user for the example name and the arguments to pass**, then plug them
into the run command in section 4 or a batch script from section 5. Accepted
arguments per example are documented in [README.md](../../README.md).

Defaults for batch runs of non-test binaries on tuolumne, unless the user says
otherwise:

- `--time=15` (15 minutes)
- `-q pdebug`
- `--nodes=1` (so `--ntasks=4`)

For anything large or long-running, set `BEATNIK_USE_PROD=1` and confirm the
production env is up to date and installed **before** submitting — see the
CLAUDE.md general guidelines.

## 7. Backends

Which Kokkos backends build and run is a per-system fact. On tuolumne the spack
spec is `+rocm +openmp +serial amdgpu_target=gfx942`, so:

| Backend | Builds | Runs | In the gate here |
| --- | --- | --- | --- |
| `SERIAL` | yes | yes | **yes** (project-wide gate) |
| `HIP` | yes | yes | **yes** (tuolumne-specific gate) |
| `OPENMP` | yes | yes | no — verify manually when threading changes |
| `CUDA`, `SYCL`, `OPENMPTARGET`, `THREADS` | no | no | no |

The gate on tuolumne is therefore the `regression` label on **SERIAL and HIP**
at **ranks 1–6**. `OPENMP` was in the pre-redesign gate and was dropped: MI300A
production work is HIP, and OpenMP is a fallback path. Re-add it by extending
`BEATNIK_GATE_BACKENDS` in the wrapper *and* the gate definition in CLAUDE.md
together — never one without the other.

Test names carry the backend as a suffix (e.g. `Beatnik_Test_Foo_MPI_SERIAL`),
which is what lets `ctest -R <backend>` select one.
