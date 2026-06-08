# Tuolumne

## 1. Spack environment

Before compiling or running any binary from this library, run:

```
spack env activate ~/spack_envs/beatnik-canopy
```

## 2. CMake args

No tuolumne-specific CMake args are required beyond the project defaults. A
plain `cmake ..` (inside the spack env) is enough. Update this section if that
changes.

## 3. Build command

Tuolumne builds via spack. The build command is:

```
spack install
```

## 4. Run command for binaries

The following environment variables must be exported before `flux run`:

```
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
```

Then run the binary with:

```
flux run \
    --ntasks=[NUM_PROCS] \
    --nodes=[NUM_PROCS / 4] \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    [EXECUTABLE] [EXTRA_ARGS]
```

Tuolumne runs 4 ranks per node, so `--nodes` is derived from `--ntasks` as
`ntasks / 4`. `[NUM_PROCS]` must be a multiple of 4 (round up if needed).

## 5. Flux batch template

Tuolumne uses the **flux** scheduler. When the user is not inside an
interactive allocation, generate a batch script from the template at
[scripts/tuolumne/test_template.flux](../scripts/tuolumne/test_template.flux),
save it under `scripts/tuolumne/` (create the directory if missing), and
submit it with:

```
flux batch scripts/tuolumne/<your_script>.flux
```

The template's `# flux: --output={{name}}.{{jobid}}.log` line writes the job's
stdout/stderr to a `.log` file in the current directory. Read that log when
the job finishes to check whether tests passed or to harvest output from
other binaries.

Inside a batch script, pick `--ntasks` (the desired rank count, a multiple of
4) and set `--nodes = ntasks / 4` in both the flux header and the `flux run`
line.

## 6. Running non-test binaries

When asked to run something other than a test (e.g. an `examples/` problem),
ask the user for the example name and the args to pass, then plug them into
the run command in section 4 or a batch script based on section 5.

Defaults for batch runs of non-test binaries on tuolumne, unless the user
says otherwise:

- `--time=15` (15 minutes)
- `-q pdebug`
- `--nodes=1` (so `--ntasks=4`)
