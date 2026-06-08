# Plan review: `plans/i-am-designing-a-clever-moonbeam.md` vs. `CLAUDE.md`

## Context

The user asked me to review the existing Canopy-FMM integration plan at
[plans/i-am-designing-a-clever-moonbeam.md](plans/i-am-designing-a-clever-moonbeam.md)
against the project's CLAUDE.md requirements, and to prepare for
execution. The plan itself is comprehensive on the technical integration
(API mapping, math, data flow, checkpoint commits, ledger). This review
covers only the gaps relative to CLAUDE.md and the per-system doc
[docs/claude-tuolumne.md](docs/claude-tuolumne.md), plus a short
pre-execution readiness checklist.

## What CLAUDE.md requires that the plan does not cover

1. **README.md sync (CLAUDE.md "General guidelines").** The plan adds
   `-S fmm` as a new accepted value of the example's `-S` flag.
   [README.md:31](README.md#L31) currently documents only `exact` and
   `cutoff`. CLAUDE.md mandates README.md be updated in the same change
   when an example's accepted arguments change.
   - **Addendum:** in checkpoint 2 ("Accept -S fmm in rocketrig…"),
     also update README.md:31 to list `fmm` as a third accepted value
     (note: requires `Beatnik_ENABLE_CANOPY=ON`).

2. **System-specific build/run commands (CLAUDE.md "System detection").**
   The plan's Verification section did not specify how to build or
   launch on tuolumne. Per [docs/claude-tuolumne.md](docs/claude-tuolumne.md):
   - **Spack env:** `~/spack_envs/tuolumne_beatnik` must be activated
     before any build or run. **Note:** the docs file currently says
     `~/spack_envs/beatnik-canopy` at line 8 — this is incorrect per
     the user. As the first action during execution (before
     checkpoint 1), edit `docs/claude-tuolumne.md:8` to read
     `~/spack_envs/tuolumne_beatnik`.
   - **Build:** tuolumne builds via spack, per
     [docs/claude-tuolumne.md §3](docs/claude-tuolumne.md#L17):
     ```
     spack env activate ~/spack_envs/tuolumne_beatnik
     spack install
     ```
     Do **not** invoke `cmake ..` / `make` directly. The spack
     package handles `Beatnik_ENABLE_CANOPY` via a variant — check
     `spack info beatnik` and `spack spec` inside the env to confirm
     the variant name. If the variant doesn't exist yet, the spack
     env's `spack.yaml` (under `~/spack_envs/tuolumne_beatnik/`) may
     need an `'+canopy'` / `'~canopy'` toggle added; flag this
     during checkpoint 1 if encountered.
   - **Run:** `flux run` per
     [docs/claude-tuolumne.md §4](docs/claude-tuolumne.md#L25)
     with `--ntasks=N`, `--nodes=ceil(N/4)`, `--exclusive`,
     `--gpus-per-task=1`, `--cores-per-task=24`,
     `--setopt=mpibind=verbose:1`, and the env vars exported in
     advance (`MPICH_GPU_SUPPORT_ENABLED=1`, `HSA_XNACK=1`, etc.).
     The installed binary path is whatever `spack install`
     deposits — use `spack location -i beatnik` to resolve it (or
     run via `spack load beatnik && which rocketrig`).
   - **Batch:** non-interactive runs use `flux batch` against a
     script under `scripts/tuolumne/` based on the template at
     [scripts/tuolumne/test_template.flux](scripts/tuolumne/test_template.flux);
     defaults `--time=15`, `-q pdebug`, `--nodes=1` (so
     `--ntasks=4`) per
     [docs/claude-tuolumne.md §6](docs/claude-tuolumne.md#L78).
   - **Single-rank verification (user decision):** attempt `flux run
     --ntasks=1 --nodes=1 --gpus-per-task=1 …` for checkpoint 7's
     1-rank Exact-vs-FMM comparison; if flux rejects it, fall back
     to 4-rank as the first comparison point.

3. **Minimum test set (CLAUDE.md "Minimum test set").** The project
   has no tests yet, and CLAUDE.md:46-55 must be updated as soon as
   the first test lands. **User decision:** v1 will land a minimal
   CTest target `Beatnik_Test_FMM_vs_Exact` as part of checkpoint 7
   or 8.
   - **Addendum (new checkpoint 7a, between 7 and 8):** add a CTest
     target under [tests/](tests/) (create the directory) that runs
     a 32² free-boundary case with `-S exact` and `-S fmm` and
     diffs `zdot` to a tolerance (≲1e-3 at P=6). The test must be
     guarded by `Beatnik_ENABLE_CANOPY=ON`. Register it in
     [CMakeLists.txt](CMakeLists.txt) via `add_test`, and update
     CLAUDE.md:46-55 with the test name and the rank counts it must
     pass at (proposed: 1, 4 — multi-rank coverage gained at
     checkpoint 8).

4. **`.clang-format`** — does not exist at the repo root, so the
   "Follow `.clang-format`" guideline is moot. Match surrounding code
   style instead. No plan change needed; noting for completeness.

5. **`tasks/` directory.** Plan says to create `tasks/` and append a
   running ledger. Confirmed it does not yet exist. The first
   checkpoint commit should `mkdir tasks` and add the initial
   `tasks/integrate_canopy.md` skeleton plus the CLAUDE.md pointer
   from plan §"Implementation ledger" — the plan already specifies
   both; just flagging that nothing exists yet so checkpoint 1 owns
   the directory creation, not a later step.

## Pre-execution readiness check

- Spack env `~/spack_envs/tuolumne_beatnik` — must be activated
  before any cmake/build/run.
- Canopy source at `~/research-Bridges/Canopy` — plan references
  [Canopy_Solver.hpp](../Canopy/src/Canopy_Solver.hpp); we'll need to
  verify the exported CMake target name (`Canopy::Canopy` vs other)
  during checkpoint 1.
- Branch is `develop-canopy`, already off `main`. Checkpoint commits
  land here.
- Untracked `build-linux-rhel8-zen4-jvoqjzn` at the repo root is
  spack's build directory — leave it alone, do not delete or commit.

## Recommended additions to fold into the plan (summary)

1. README.md:31 update bundled into checkpoint 2.
2. Replace verification command lines with the tuolumne `flux run`
   template; resolve the "single-rank" question (see below).
3. Note that v1 verification is manual; landing a real test later
   must also update CLAUDE.md's Minimum test set.
4. Make checkpoint 1 explicitly own `mkdir tasks` and the CLAUDE.md
   pointer add.

## Verification

After incorporating the addenda above, execution can begin per the
existing plan's checkpoint order. At each checkpoint:

1. `spack env activate ~/spack_envs/tuolumne_beatnik` then build via
   `spack install` (per [docs/claude-tuolumne.md §3](docs/claude-tuolumne.md#L17)).
   Confirm the Canopy variant is enabled on the spec before installing.
2. Resolve the installed `rocketrig` path with
   `spack location -i beatnik` (or `spack load beatnik`) and launch
   it via `flux run` per [docs/claude-tuolumne.md §4](docs/claude-tuolumne.md#L25),
   after exporting the env vars from lines 30-38 of that doc. For
   batch verification at checkpoints 7-9 (and the new 7a CTest
   target), submit a script under `scripts/tuolumne/` derived from
   [scripts/tuolumne/test_template.flux](scripts/tuolumne/test_template.flux).
3. Append the checkpoint's commit hash and observed result (pass /
   diff vs Exact / wallclock) to `tasks/integrate_canopy.md`.
