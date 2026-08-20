# Milestone 0 — progress log

Session record for milestone0. Companion to `milestone0.md`, which holds the
design, the task sequence and the risks; this file holds what actually happened,
in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `milestone0.md` can cite it by
ID. No dates: the order of the sections is the chronology. If a session covers
more than one task, name them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that invalidates
a later task is worthless if the session starting that task has to read the whole
log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could not
stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation departed
from the task's stated **Do** steps, and why — a task marked `**DONE**` that was
done differently than it was written is the quietest way for a design to stop
describing the code.

## M0-G1, M0-G2, M0-T1

Both gold sets were already generated and committed when this session started;
what it did was the analysis (**Do** steps 2 and 3 of both), the README
recording, and the whole of M0-T1.

### Decisions

- **M0-T1 installs both milestone-0 gold directories, not M0-T3.** The gold is
  committed (9.5 MB in the repo, 2.3 MB + 7.2 MB), installing it here is what
  makes the `FATAL_ERROR` glob guard mean anything, and it leaves M0-T3 a
  test-source task only.
- **`checkpoint_latest.npz` stays in both gold directories and is excluded from
  the install glob** (`*_step*.npz`, not `*.npz`). It is inert to `goldForStep`
  — no `_step%07d.npz` suffix — and it duplicates the largest file in the set;
  verified bit-identical to the last numbered file in both sets. Each
  `gold/README.md` says so.
- **The `gold/README.md` files stay inside `gold/`**, beside the `.npz`, unlike
  `direct-solve-10-steps/README.md` which sits one level up. The install rules
  handle both layouts rather than unifying them, because the installed tree
  mirrors the repo's `tests/` layout exactly and that is what makes a
  manifest-relative argument name the same file in both places.

### Measured

Pure numpy over the committed `.npz` on the login node — no MPI, no binary.
Quality is `4*sqrt(3)*A / sum(l^2)` (the project convention,
`src/Beatnik_Params.hpp:220`); volume is `(1/6) sum_f a.(b x c)` over `faces`,
as T2d computes it, and the step-0 value matches each file's own
`initial_volume` bit for bit. Full 81-row tables are in the two
`gold/README.md`.

| | level 3 (642 v) | level 4 (2562 v) |
| --- | --- | --- |
| final `time` @ step 2000 | `1.998284` | `1.964304` |
| final `V/V0 - 1` | `+3.352894e-09` | `+4.741414e-09` |
| final min quality | `6.303626e-02` | `1.242421e-01` |
| global min quality | `3.826563e-02` @ step 1700 | `1.242421e-01` @ step 2000 |
| first below `0.18` | step 1050 | step 1800 |
| first below `0.1` | step 1475 | never |

**The frozen LEVEL-3 mesh degrades faster** — the opposite of what M0-G2's task
entry expected to have to rule out. At step 2000 the level-4 minimum is ~2x the
level-3 one, and level 4 never reaches `0.1`. Neither run stopped early and
every field is finite everywhere, so **M0-R2 did not fire at either level and
M0-R6 did not fire at level 4**: the level-4 set is not a `--dt` decision. Both
last-file self-compares at `--rtol 1e-12 --atol 1e-14` exit 0 with full
unambiguous pairing (`642/642`, `2562/2562`, `ambiguous cpp=0 gold=0`), so
**M0-R4 has not bitten the gold sets**. Both sets carry the same nine keys as
`initial_conditions/gold.npz` in all 81 files: `FIELD_MAP` needs no edit.
Volume drift at `~3-5e-9` confirms **M0-R3**: T2d's `kVolumeDriftAbsCap = 1e-9`
must not be reused.

The two runs do **not** cover the same physical time after 2000 steps
(`1.998284` vs `1.964304`), because the reference's adaptive dt is relative to
each run's own initial minimum edge (`3.457079e-02` vs `1.729575e-02`). Any
cross-level comparison M0-D1 or M0-A1 makes must be by *step*, not by time.

### Departures from the stated **Do** steps, and what the tooling forced

- **The gate manifest has fifteen non-comment lines, not the ten M0-T1's exit
  criterion names.** Pre-existing and not a gate change: tuolumne's spec is
  `+rocm +openmp +serial`, so `BEATNIK_TEST_DEVICES` is `SERIAL;OPENMP;HIP` and
  the regression loop — untouched here; the diff to `tests/CMakeLists.txt` is
  pure addition — emits five targets on each of *three* backends. The runner's
  `_<BACKEND>` filter selects only SERIAL and HIP, and the gate re-run confirmed
  60 launches and `[gate] PASS`. The criterion's "ten" should be read as "ten
  that the gate runner selects".
- **The milestone tier registers its ctest cases at `BEATNIK_MILESTONE_MPI_RANKS`
  (default `1;4`)**, a new cache variable, rather than reusing
  `BEATNIK_TEST_MPI_RANKS`. The tier's rank set is a property of the tier, so it
  is single-sourced in exactly two places — that variable and the runner's
  `BEATNIK_MILESTONE_RANKS` — and never as a per-test list on a manifest line.
- **The milestone manifest is written unconditionally**, so an empty tier
  produces a file with zero non-comment lines rather than no file. A missing
  manifest and an empty one are different failures and `run_milestone.flux`
  reports them differently (`not found on PATH` vs `named no runnable`).
- **`run_milestone.flux` is a copy of `run_regression_minset.flux`'s structure**,
  not a generalization of it, per the task: the gate script is single-sourced
  against CLAUDE.md's gate definition. Copied verbatim in shape: the
  `beatnik_find_repo` walk-up, the PATH manifest scan, the FD-3 read loop, the
  rank-to-GPU binding, and the vacuous-pass guard. It differs in giving each
  target its **own** scratch directory, `rm -rf`'d and recreated immediately
  before that target runs, under
  `/p/lustre5/stewartj/beatnik/milestone0/<target>` (lustre, because the
  checkpoints go through MPI-IO).
- `flux batch --flags=waitable` is **rejected on tuolumne** — "only the instance
  owner can submit with FLUX_JOB_WAITABLE". `flux job attach <jobid>` works and
  exits with the job's own status; use that.

### Verified

`spack install` clean. `beatnik_milestone_manifest.txt` installed beside
`beatnik_gate_manifest.txt` with zero non-comment lines; both gold sets
installed as 81 `.npz` + `README.md` with no `checkpoint_latest.npz`;
`run_milestone.flux` (job `f3TRgG5aCNsq`) exited **1** with the ported "named no
runnable milestone tests" message; `run_regression_minset.flux` (job
`f3TRgYcw5dfu`) reported `[gate] PASS` with exactly **60** `[gate] ===` lines.

**Affects:** **M0-D1** — compare by step, not by time, since the two levels
reach different `t` after 2000 steps; the level-3 minimum quality below `0.1`
after step 1475 is the M0-R2-adjacent regime its attribution step must account
for, and its volume-drift step must expect `~3-5e-9`, not T2d's `1e-9`.
**M0-A1** — the mesh-health answer is in: level 4 is the healthier frozen mesh
over this horizon, so a "level 3 primary" choice needs a reason other than mesh
degradation; `kVolumeDriftAbsCap` must be re-derived at `~1e-8`, and the
step-0-per-level `max|e|` (M0-R5) is still unmeasured. **M0-T3** — the tier, its
manifest, its runner and **both gold sets' install rules already exist**; M0-T3
adds `Beatnik_Test_Milestone0Frozen.cpp` to `BEATNIK_MILESTONE_TEST_SOURCES`
plus its `_beatnik_args_<stem>_abs`/`_rel` pair (the `FATAL_ERROR` fires without
it) and nothing else in CMake. **M1-T1** in `milestone1.md` — unaffected; the
tier is as its entry expects (label `milestone`, ranks 1 and 4, SERIAL and HIP,
gate untouched), so it still only has to register its member.
