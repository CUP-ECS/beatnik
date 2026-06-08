# Beatnik

## System detection

Build and run commands differ by system. Before building or running anything,
run `hostname` and match the result against the table below. Then Read the
matching system-specific instructions file and follow it for the rest of the
session.

| Hostname pattern | Instructions file              |
| ---------------- | ------------------------------ |
| `tuolumne*`      | `docs/claude-tuolumne.md`      |

The pattern is the alphabetic prefix of the host (e.g. `dane1234` matches
`dane*`, `lassen708` matches `lassen*`). To add support for a new system,
create `docs/claude-<system>.md` and add a row above.

If the hostname does not match any row, or the matching file is missing one of
the required sections below, ask the user to fill in the gap and update (or
create) the doc before proceeding.

### Required sections in every `docs/claude-<system>.md`

1. **Spack environment** — the `spack env activate ...` command that must be
   run before compiling or running any binary from this library.
2. **CMake args** — system-specific args that must be passed to `cmake` (or to
   any helper bash script that wraps `cmake`).
3. **Run command for binaries** — the command template for running a built
   binary. Default starting point:
   `mpirun --oversubscribe -n [num_procs] [EXECUTABLE] [EXTRA_ARGS]`. Replace
   `mpirun` with `flux run`, `srun`, or whatever the system uses.
4. **Job-scheduler batch template** — if the system has a scheduler (flux,
   slurm, …), include a template batch script that can be filled in and
   submitted (e.g. `flux batch <script>`) to run binaries when the user is
   not inside an interactive allocation. Save concrete scripts to
   `scripts/<hostname>/` (create the directory if it does not exist).
5. **Running non-test binaries** — when asked to run something other than a
   test (e.g. an `examples/` problem), ask the user for the example name and
   args, then plug them into sections 3 and 4.

The required tests themselves (names + MPI rank counts) are project-wide and
live in [Minimum test set](#minimum-test-set) below, not in the per-system
doc. The per-system doc only describes *how* to run any given test on that
machine.

## Minimum test set

These tests must pass before any code change ships. Each entry lists the
test name and the MPI rank counts it must be run at (e.g.
`Beatnik_Test_Particle` at 1, 2, 3, 4, 5, 6 ranks). Use the run command and
batch template from the active system's `docs/claude-<system>.md` to execute
them.

**There are no tests in the project yet.** Update this section as soon as
the first test lands — add the test name and required rank counts here.

## Plans

Save plan files to `./plans/` in this repository, not the default plan
location.

## General guidelines

- **Checkpoint commits in plans.** When planning a large code change, include
  explicit checkpoints in the plan file where progress should be committed.
  If a later step fails (test failure, performance regression), we can roll
  back to the nearest checkpoint and retry.
- **Follow `.clang-format`.** If `.clang-format` exists at the repo root,
  follow its formatting rules for any C/C++ code you write or edit. If it
  does not exist, ignore this rule.
- **Keep `README.md` in sync.** When a public-facing API changes, or when the
  arguments accepted by an example problem change, update `README.md` in the
  same change so its documentation stays accurate.
