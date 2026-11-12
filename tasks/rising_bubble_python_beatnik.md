# Rising bubble — Beatnik against the Python reference, accuracy and wall time

**Status:** NOT STARTED

## Problem

`examples/02_adaptive_mesh_bubble` is a port of the reference driver
`examples/run_adaptive_mesh_bubble.py` in `~/research-bridges/zmodel-steve/zmodel3d-amr`
(HEAD `ec7d7bf`), and one command line drives both
([examples/02_adaptive_mesh_bubble/InputFile.hpp](../examples/02_adaptive_mesh_bubble/InputFile.hpp),
file header). No run has yet compared the two on the reference author's
benchmark: a latitude–longitude sphere of 1,514 vertices rising at Atwood
number 0.2 with split/collapse remeshing, to `t = 3.45`, where the reference's
own run (at `--remesh-h-min 0.008`, treecode, face quadrature) reaches about
6,500 vertices in 1.1 h on a laptop and stays clean (`max|S|` about 0.65;
`max|S|` spikes past `t ≈ 3.5`).

**End state.** A report, under `/p/lustre5/stewartj/beatnik/rb_report/`, that
states for the benchmark at `--remesh-h-min 0.006`:

- **accuracy:** matched-time differences in the bubble's observables between
  Beatnik (1 and 4 ranks, direct sum) and the Python reference (direct sum to
  as far as a 5-hour job reaches; treecode θ = 0.3 to `t = 3.45`), set against
  the two reference floors this run measures (Python direct vs Python treecode,
  Beatnik 1 rank vs 4 ranks);
- **wall time:** each run's wall to reach common physical times, its s/step,
  and the Beatnik-over-Python speedup at each common time;

for each of two mesh-operation variants (`noflip`, `flip`, Approach).

**Out of scope:**

- Any change to `src/`, `examples/` or the reference code. This is a
  measurement; a defect it finds becomes its own task.
- Adding a test or changing any ctest tier or the gate.
- The Canopy FMM (`--br-approximation fmm` / `treecode` on the Beatnik side):
  every Beatnik run uses `--br-approximation direct`.
- A Python run long enough to bring the direct-sum reference to `t = 3.45`
  (Approach, "The Python direct run is time-boxed").
- The laptop timing quoted above: a different configuration
  (`--remesh-h-min 0.008`, face quadrature, treecode, different hardware), so
  it is cited in the report as context and never paired.

## Approach

**Eight jobs, one staged copy, one comparison script.** Two variants × four
runs. Every run reads its solver arguments from one shared arguments file per
variant, so the codes cannot be handed different physics; each script appends
only the BR mode, the output paths and `--t-end`.

| Run | Code | Ranks / cores | BR | Queue, `-t` | Output |
| --- | --- | --- | --- | --- | --- |
| `b1` | Beatnik `adaptive_mesh_bubble` | 1 rank, 1 GPU | direct | pdebug, 60m | `/p/lustre5/stewartj/beatnik/rb1/<variant>` |
| `b4` | Beatnik `adaptive_mesh_bubble` | 4 ranks, 4 GPUs, 1 node | direct | pdebug, 60m | `/p/lustre5/stewartj/beatnik/rb4/<variant>` |
| `pd` | Python reference | 1 core | direct | pbatch, 300m | `/p/lustre5/stewartj/beatnik/rb_python/<variant>/direct` |
| `pt` | Python reference | 1 core | treecode θ 0.3, order 2, ncrit 64 | pbatch, 240m | `/p/lustre5/stewartj/beatnik/rb_python/<variant>/treecode` |

`<variant>` is `noflip` or `flip`.

### The configuration

The `noflip` arguments file, verbatim (one option per line is fine; the
scripts join lines with spaces):

```
--mesh-kind latlon --n-theta 28 --n-phi 56 --radius 0.25 --center-z 0.25 --initial-shape sphere
--A 0.2 --g 1.0 --eps 0.04 --kernel-blob-mode length
--mu 0.002 --viscosity-mode laplace-beltrami
--source-quadrature vertex
--dt 1.5e-3 --adaptive-dt --min-dt 5e-6 --max-sheet-dt-product 0.006
--dynamic-remesh --remesh-every 3 --remesh-h-min 0.006 --remesh-smooth-iters 0
--remesh-flip-min-gain 1e9 --remesh-collapse-factor 0.15 --remesh-split-factor 1.2
--remesh-max-splits 800 --isotropic-cleanup
--steps 1000000 --stride 1000000 --progress-time-interval 0.03
--checkpoint-every-time 0.03 --no-video
```

The `flip` file is the same with `--remesh-flip-min-gain 1e9` deleted, so the
quality-flip pass runs at its default gain `1.0e-3` (reference
`run_adaptive_mesh_bubble.py:427`; Beatnik shares the default by design).
Tangential smoothing stays off in both (`--remesh-smooth-iters 0`), and edge
collapse stays on at 0.15 in both; Tessera performs both collapse and flips.
`--isotropic-cleanup` is each code's default and is written out so the log
shows it.

Every option above is parsed by both codes (`InputFile.hpp` setter table;
reference `parse_args`, `run_adaptive_mesh_bubble.py:66-450`). `--eps 0.04`
with `--kernel-blob-mode length` gives blob `b = eps² = 1.6e-3`.

**`--remesh-max-splits 800`.** The reference author's value at
`--remesh-h-min 0.008` is 450. The population of edges held at the sizing
floor scales with vertex density, `(0.008/0.006)² ≈ 1.78`, and
`450 × 1.78 ≈ 800`. A cap that never binds matters to the comparison: when it
binds, Beatnik keeps edges above a global ratio threshold
([src/Beatnik_DynamicRemesh.hpp:907-921](../src/Beatnik_DynamicRemesh.hpp#L907-L921);
`split_capped`, `:181-185`) and the reference sorts and slices
(`zmodel3d/dynamic_remesh.py:253-254`, `:349-350`), so the two accept
different edge sets. Neither code prints whether the cap bound (R4).
`--remesh-max-collapses` stays at its default, 120.

### The Python direct run is time-boxed

The reference's direct velocity kernel (`_source_velocity_direct_unsigned`,
`zmodel3d/mesh_solver.py:437-454`) costs **5.9 s per evaluation at 6,500
vertices** on one tuolumne core, and the treecode
(`zmodel3d/treecode.py:96`, θ 0.3) **1.1 s**; at 13,000 vertices the treecode
costs 2.6 s (log `## Design measurements`). The RHS evaluates only the
velocity under the default `--bernoulli-scalar-mode normal-speed`
(`mesh_solver.py:1236-1269`), three times per TVD-RK3 step, so direct costs
about 18 s/step at 6,500 vertices before remeshing overhead, and reaching
`t = 3.45` would take roughly 10–25 h. So `pd` runs for one 5-hour pbatch job
and is compared up to wherever it stops (`t_pd`). `pt` runs for one 4-hour
pbatch job and is expected to reach `t = 3.45`. `pd` against `pt` over
`[0, t_pd]` is the reference's own treecode error, one of the two floors.

A Python run is stopped gracefully: the job launches the driver under
`timeout -s INT <cap>`, with the cap 10 minutes under `flux job timeleft`. The
driver catches `KeyboardInterrupt` and writes its last finite state as a final
checkpoint (`run_adaptive_mesh_bubble.py:1637-1652`).

### Beatnik runs fit pdebug

Estimated 3–15 min per Beatnik run: the large_rollup P1 variant A ran 8,457
steps in 476 s, 0.056 s/step at 2,000–10,000 vertices on np4 HIP
(`tasks/large_rollup.md`, P1 findings), and this run takes about 2,300 steps
at the initial `dt` 1.5e-3, more as the adaptive `dt` shrinks; allow 2,300–7,000.
`max|S| ≤ 0.65` keeps `--max-sheet-dt-product 0.006` above about `9e-3`, so it
does not bind. Under 20 minutes, so both Beatnik runs go to **pdebug** with
`-t 60m`. A Beatnik launch is capped at `flux job timeleft − 600 s` with
`flux run -t`, as in
[scripts/tuolumne/large_rollup_segment.flux](../scripts/tuolumne/large_rollup_segment.flux);
a capped run restarts from its `checkpoint_latest.h5` (R3).

### Production env, staged copy

Beatnik runs use the **production** env (`BEATNIK_USE_PROD=1`):
`profiling_level=1` and canopy `~profiling`, so the walls are not inflated by
level-2 timers ([systems/tuolumne/claude.md](../systems/tuolumne/claude.md),
§1). The production checkout `~/spack_envs/tuolumne_beatnik_production/beatnik`
is on `large-rollup` at `de299e2`, and its install is spec hash `inrvs7n`
(`beatnik@develop … profiling_level=1`, canopy `~profiling`), matching
[systems/tuolumne/spack-production.yaml](../systems/tuolumne/spack-production.yaml).

Everything a job executes or reads is staged into
`/p/lustre5/stewartj/beatnik/rb_stage/`:

```
rb_stage/
  COMMIT                 Beatnik dev HEAD, production-checkout HEAD, `spack find -lv beatnik`
                         (prod env), reference HEAD, reference `git diff`,
                         reference `git status --porcelain -- zmodel3d examples`
  bin/adaptive_mesh_bubble   copied from the production view
  noflip.args  flip.args     the two arguments files
  reference/examples/  reference/zmodel3d/   copied from the reference checkout
  jobs/                  the job files actually submitted (below)
```

**Jobs are submitted from `rb_stage/jobs/`.** Every submission goes through
the task loop's wrapper, `"$TASKLOOP_SUBMIT" --purpose "<why>" <job file> --
<script args>`, which reads nodes, walltime, queue and bank **only** from the
file's `# flux:` directives and takes no scheduler option on the command line.
So each queue/walltime combination is its own copy of
`rb_compare_beatnik.flux` or `rb_compare_python.flux` in `rb_stage/jobs/`,
identical but for its `# flux: -q` and `# flux: -t` lines, named for them,
for example `rb_compare_python.pbatch-240m.flux`. Only the directive lines
differ (`diff` against the committed script shows nothing else). The scripts
find the checkout from `PWD` (as `large_rollup_segment.flux` does), so submit
from the repository root.

The reference checkout carries one uncommitted line: `--source-quadrature`
defaults to `vertex` instead of `face`. Every run passes
`--source-quadrature vertex`, so it does not change behaviour; `COMMIT` records
it. The driver puts `parents[1]` on `sys.path`, so running
`reference/examples/run_adaptive_mesh_bubble.py` imports the staged
`reference/zmodel3d/` ([scripts/tuolumne/m1g3_gold.flux](../scripts/tuolumne/m1g3_gold.flux)
relies on the same).

### The comparison

`scripts/tuolumne/rb_compare.py` reads the eight output directories and writes
`rb_report/rb_compare.json` and `rb_report/rb_compare.md`. Pairs are matched in
physical time, never by index: the meshes differ by remesher selection.

- **Scalar series** from the timestamped progress lines, written every 0.03 of
  physical time by both codes in the same format
  ([src/Beatnik_Diagnostics.hpp:348-379](../src/Beatnik_Diagnostics.hpp#L348-L379);
  reference `run_adaptive_mesh_bubble.py:1633`): `centroid_z`, `zmax`,
  `volRel`, `max|S|`, `V`, `F`, `minQ`, `minEdge`, `dt`. Each run's series is
  linearly interpolated onto the other's progress times.
- **Shape and roll-up** from the checkpoints, also every 0.03: total area,
  enclosed volume, and the reference's mesh-independent roll-up observables
  (`roll_up_center`, `roll_up_radius`, `max_height`, `total_circ`,
  `zmodel3d/rollup_observables.py:32`), plus the radial-function difference
  `dr` (max, rms) from the casting in
  [scripts/tuolumne/m1d3_drift.py](../scripts/tuolumne/m1d3_drift.py)
  (`raycast` `:98`, `centroid` `:132`). Checkpoints are paired by nearest time
  and `|Δt|` is reported beside each row. Both codes load through
  `load_any` in
  [tests/regression_tests/compare_output.py:285](../tests/regression_tests/compare_output.py#L285),
  which maps Beatnik's `.h5` to the `.npz` key set.
- **Pairs:** `b1–pd`, `b4–pd`, `b1–pt`, `b4–pt` (accuracy); `pd–pt` and
  `b1–b4` (the floors); `noflip–flip` within each run (mesh-operation
  sensitivity).
- **Wall time** from the epoch prefix on each progress line: wall since launch
  at `t` = 0.3, 0.6, …, `t_pd`, 3.45 (where reached), s/step per interval and
  overall, and `wall_python / wall_beatnik` at each common `t`. The report
  states the hardware: Beatnik on MI300A GPUs, Python on one CPU core of the
  same node type.

The report draws no pass/fail line. Its reading is the Beatnik–Python gap
against the two floors and against the mesh-realisation gap already measured
for configuration (a) (`tasks/milestone1.md`, M1-D3: area `1.06e-2`, rise
`2.5e-3` at `t = 1.24`), R5.

### Conventions

| Convention | Why | Check |
| --- | --- | --- |
| Every new script carries the BSD-3-Clause block with `SPDX-License-Identifier: BSD-3-Clause` in its comment style | Project header rule (CLAUDE.md) | `grep -L SPDX-License-Identifier` over the new files prints nothing |
| Each batch script sources `scripts/lib/beatnik_env.sh` before any other work, with `BEATNIK_USE_PROD=1` exported first | Resolver-first rule; the Beatnik runs are on the production env | by reading: the `source` line precedes every command but `beatnik_find_repo` |
| Jobs run only from `rb_stage/`: the staged binary, never `beatnik_exe` or the view path; the staged reference, never `~/research-bridges/...`; provenance from `rb_stage/COMMIT`, never `git rev-parse` | The checkout and env may move between submission and start | `grep -nE 'beatnik_exe|research-bridges|rev-parse' scripts/tuolumne/rb_compare_*.flux` prints nothing |
| Solver arguments come only from `<variant>.args`; a script appends only the BR mode and its treecode flags, `--checkpoint-dir`, `--output`, `--t-end`, and `--restart-from` | The two codes must see identical physics | by reading; `rb_compare.py` re-checks it from the launch logs (T2) |
| Every launch-log line is prefixed with `date +%s.%N` | Wall time at matched physical time comes from these stamps | `head -1` of any `*.launch.log` starts with an epoch |
| A script refuses an output directory that already holds checkpoints, unless restarting into it | Never mix two runs' files | by reading; T1's smoke shows the refusal |
| Python runs with `/usr/tce/bin/python3 -u`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, one core (`flux run -N 1 -n 1`) | One defined core count for the wall comparison; no `__pycache__` in the stage | by reading; the log echoes `nproc` inside the launch |
| `rb_compare.py` runs as `/usr/tce/bin/python3 -I`, adds only `rb_stage/reference` and `tests/regression_tests` to `sys.path` | `-I` keeps the cwd off the import path | by reading |
| Rank-count dependence: `b1` and `b4` differ only by reduction order and are reported as a floor, not judged | Results legitimately differ across rank counts (BRSolverDirect ring accumulation) | the report carries a `b1–b4` row |
| No production `spack install` while any production job is live, and none while this design's Beatnik jobs are queued or running | A reinstall under a live job SIGBUSes it (CLAUDE.md) | `flux jobs` output is pasted into the log before any install |
| Formatter never run; no file under `src/`, `examples/`, `tests/` changes | Measurement only | `git diff --stat` of each task touches only `scripts/tuolumne/` and `tasks/` |

### Deliberate deviations

- **The Python direct run stops at a 5-hour cap**, short of `t = 3.45`, and
  the treecode run is the reference to `t = 3.45`. The reference author
  reports treecode at θ 0.3 within about `1e-3` of the direct sum;
  `pd–pt` measures it for this configuration.
- **Beatnik always uses the direct sum**, including where it is compared with
  the Python treecode.
- **`--isotropic-cleanup` is written out** although it is each code's default.

## Current state

- The two codes parse the whole configuration; nothing has run with it.
- No `rb_compare_*` script, arguments file, stage or output directory exists.
  `/p/lustre5/stewartj/beatnik/` is empty.
- The production install's `adaptive_mesh_bubble` (`inrvs7n`,
  `profiling_level=1`) is built from `de299e2`, the dev checkout's HEAD.
- Neither code reports whether `--remesh-max-splits` bound in a pass.
- Python progress lines carry no wall time; the job's timestamp prefix
  provides it.

## Progress log

Consult [rising_bubble_python_beatnik-progress-log.md](rising_bubble_python_beatnik-progress-log.md)
before implementing a task, changing a script's interface, or reopening a
question this design treats as settled. It holds the measurements behind the
design's estimates (`## Design measurements`).

> Each task is reviewed with `review-task` before any task that depends on it
> starts. A task is ready to build on when it is marked **DONE** and the log has
> a `## <ID> review` section whose verdict is `Passed`. The document's status
> becomes COMPLETE only when a `review-task` sweep of the whole feature passes.

## Task sequence

### T1 — scripts, arguments files and stage, proven by a smoke job — **NOT STARTED**

**Depends on:** none.

**Fill in:**
- `scripts/tuolumne/rb_compare/noflip.args`, `scripts/tuolumne/rb_compare/flip.args`:
  the two files of "The configuration".
- `scripts/tuolumne/rb_compare_beatnik.flux` — directives `--nodes=1`,
  `--exclusive`, `-q pdebug`, `-t 60m`, `--bank=unm`,
  `--output=/p/lustre5/stewartj/beatnik/rb_logs/{{name}}.{{jobid}}.log`.
  Arguments: `<args file> <np: 1|4> <out> <t_end> <tag> [restart path]`. The
  stage is the args file's directory. Body follows
  `large_rollup_segment.flux`: repo discovery, `BEATNIK_USE_PROD=1`, resolver,
  staged-file checks, provenance (env summary, `spack find -lv beatnik`,
  `COMMIT`, jobid, host), launch cap `timeleft − 600 s`, RSS sampler,
  `flux run -t <cap>s --ntasks=<np> --nodes=1 --exclusive --gpus-per-task=1
  --cores-per-task=24 /usr/bin/time -f rss_kb=%M <exe> <args>
  --br-approximation direct --checkpoint-dir <out> --output <out>/unused.mp4
  --t-end <t_end> [--restart-from <path>]`, epoch-prefixed `<out>/<tag>.launch.log`,
  summary `[rb_compare] <tag> rc=<rc> wall=<s> end=<t_end|cap|solver_stop|failure>`.
- `scripts/tuolumne/rb_compare_python.flux` — directives as above but
  `-q pbatch -t 300m`. Arguments: `<args file> <br: direct|treecode> <out>
  <t_end> <tag> [restart path]`. `treecode` appends `--br-approximation
  treecode --br-treecode-theta 0.3 --br-treecode-order 2 --br-treecode-ncrit
  64`; `direct` appends `--br-approximation direct`. Launch: `cd
  <stage>/reference && flux run -N 1 -n 1 timeout -s INT <cap>
  /usr/tce/bin/python3 -u examples/run_adaptive_mesh_bubble.py <args> …`, with
  the same cap, log prefix and summary line; `end=cap` when `timeout` fired
  (`saved final finite checkpoint` is then in the log). Echo Python, numpy and
  scipy versions and `nproc` inside the launch.
- Stage `/p/lustre5/stewartj/beatnik/rb_stage/` per "Production env, staged
  copy", and `mkdir -p /p/lustre5/stewartj/beatnik/rb_logs`.

**Removes:** nothing.

**Reference:** `scripts/tuolumne/large_rollup_segment.flux` (cap, sampler,
prefix, summary), `scripts/tuolumne/m1g3_gold.flux` (Python staging and
provenance), `run_adaptive_mesh_bubble.py:1637-1652` (interrupt path).

**Do:**
1. Compare the production checkout's HEAD with the dev checkout's HEAD over
   `src/` and `examples/`. If they differ, run `flux jobs` (your active jobs),
   paste it into the log and, only if no production job is live, fast-forward
   the production checkout and run the production `spack install` (HIPCC flags
   cleared, systems doc §3). If they do not differ, the install (`inrvs7n`)
   is current; record `spack find -lv beatnik` and stage it.
2. Write the files and the stage.
3. Smoke: four jobs, each from a `rb_stage/jobs/*.pdebug-15m.flux` copy
   (directives `-q pdebug`, `-t 15m`) with `t_end 0.01`, into
   `/p/lustre5/stewartj/beatnik/rb_smoke/<tag>`: `b4` `noflip`, `b1` `flip`,
   `pd` `noflip`, `pt` `flip`. Then resubmit the `b1 flip` job into its
   now-populated directory with no restart path.

**Exit criterion:** the four smoke jobs each print `[rb_compare] <tag> rc=0
… end=t_end`, each output directory holds a step-0 and a final checkpoint,
each launch log's first progress line starts with an epoch stamp and shows
`V=1514` (`(28 − 1) × 56 + 2`), and no Python launch log contains
`interrupted at step=`; **and**
the resubmitted `b1 flip` job exits non-zero with its refusal naming the
populated output directory, having launched nothing.

### T2 — the comparison script — **NOT STARTED**

**Depends on:** T1.

**Fill in:** `scripts/tuolumne/rb_compare.py`, invoked as

```
/usr/tce/bin/python3 -I scripts/tuolumne/rb_compare.py \
    --stage /p/lustre5/stewartj/beatnik/rb_stage \
    --run b1:noflip=<dir> --run b4:noflip=<dir> --run pd:noflip=<dir> --run pt:noflip=<dir> \
    --run b1:flip=<dir> ... \
    --out /p/lustre5/stewartj/beatnik/rb_report
```

It computes everything in "The comparison" for whatever runs it is given
(pairs whose members are absent are skipped and listed), writes
`rb_compare.json` and `rb_compare.md`, and refuses a non-empty `--out`.

Loud failures, each naming the run: a directory with no `*.launch.log` or no
checkpoints; a progress line that does not parse; and **argument parity** — the
solver arguments recorded on each launch log's first line, with the BR mode and
its treecode flags, `--checkpoint-dir`, `--output`, `--t-end` and
`--restart-from` removed, must be identical within a variant and differ between
variants only by `--remesh-flip-min-gain 1e9`.

**Removes:** nothing.

**Reference:** `scripts/tuolumne/m1d3_drift.py` (`raycast`, `centroid`,
matched-time interpolation), `tests/regression_tests/compare_output.py:285`
(`load_any`), `zmodel3d/rollup_observables.py:32`.

**Exit criterion:** run on T1's four smoke directories, the script exits 0 and
`rb_compare.md` has a row for every pair those runs allow, with every number
finite and the `pd–pt` row present; **and** run again with one smoke launch
log copied to a scratch directory and its `--eps 0.04` edited to `--eps 0.05`,
it exits non-zero with a parity message naming that run and `--eps`.

### T3j — submit the eight runs — **NOT STARTED**

**Depends on:** T1.

**Fill in:** nothing in the tree; the log records the submissions.

**Submits:**
- `rb_compare_beatnik.flux` × 4: `<stage>/<variant>.args <np> <out> 3.45 b<np>`
  for np ∈ {1, 4}, variant ∈ {noflip, flip}; 1 node, pdebug, `-t 60m`;
  expected 3–15 min each (Beatnik estimate in Approach); outputs
  `/p/lustre5/stewartj/beatnik/rb{1,4}/{noflip,flip}`.
- `rb_compare_python.flux … direct … 3.45 pd` × 2: 1 node, pbatch, `-t 300m`;
  runs to its cap, about 290 min (Python estimate in Approach); outputs
  `/p/lustre5/stewartj/beatnik/rb_python/{noflip,flip}/direct`.
- `rb_stage/jobs/rb_compare_python.pbatch-240m.flux … treecode … 3.45 pt` × 2:
  1 node, pbatch, `-t 240m`; expected 2–5 h, so it may end on its cap; outputs
  `/p/lustre5/stewartj/beatnik/rb_python/{noflip,flip}/treecode`.
- Staged directory: `/p/lustre5/stewartj/beatnik/rb_stage/` from T1. Job logs:
  `/p/lustre5/stewartj/beatnik/rb_logs/`.

**Removes:** nothing.

**Do:** confirm `rb_stage/COMMIT` still names the production install that is
current (`spack find -lv beatnik` in the production env matches it) and that no
production `spack install` is pending. Submit all eight through
`"$TASKLOOP_SUBMIT"` from the repo root: the Beatnik and `pd` jobs from
`rb_stage/jobs/` copies identical to the committed scripts, the `pt` jobs from
the `pbatch-240m` copy. The eight jobs total 22 node-h requested.

**Exit criterion:** `flux jobs` lists eight accepted jobs with the queues and
time limits above, and the log's `**Jobs:**` line carries all eight IDs with
their run tag and variant.

### T3 — audit the runs, compare, report — **NOT STARTED**

**Depends on:** T3j, T2.

**Do:**
1. For each job, record from its summary line `rc`, `wall`, `end`, the last
   progress line (step, `t`, `V`, `F`, `max|S|`), and peak `rss_kb`.
2. A Beatnik run that ended on `cap` restarts from its
   `checkpoint_latest.h5` from a `rb_stage/jobs/` copy with directives
   `-q pbatch` and `-t` 1.5× the remaining estimate; its wall is the sum of the
   segments (R3). A `pt` run that ended on `cap` short of 3.45 is reported as
   is, with its `t`. A `solver_stop` in any run is reported with its last finite
   `t` (R2).
3. Run `rb_compare.py` on all eight directories into `rb_report/`.
4. Append `## T3` to the log: the job table, the report's accuracy and wall
   tables, `t_pd` for each variant, and a reading of the Beatnik–Python gap
   against the floors (R5).

**Removes:** nothing.

**Exit criterion:** `rb_report/rb_compare.md` exists with every pair of
"The comparison" for both variants, and the log's `## T3` gives, for each
variant, the speedup `wall_pt / wall_b1` and `wall_pt / wall_b4` at the largest
`t` both reached and `wall_pd / wall_b1` at `t_pd`; **and** every run whose
`end` is not `t_end` is named in the log with its stop reason and last `t`.

## Known risks

**R1 — `pd` stops very early.** If `t_pd` is below about 1.0 the `pd`
comparison sees only the near-spherical rise. Presents as `end=cap` with a
small `t`. Report it as measured; the `pt` comparison carries the roll-up. Do
not extend `pd` within this design.

**R2 — the sheet-strength spike arrives before `t = 3.45` at `h_min` 0.006.**
At `h_min` 0.008 `max|S|` stays near 0.65 to 3.45 and reaches 70 by 3.94.
A finer floor may bring it earlier. Presents as `max|S|` rising more than
tenfold over a few progress lines, `dt` falling to `--min-dt`, or a
`nonfinite` stop. Compare only up to the earliest such time across the paired
runs and state it; do not tune parameters within this design.

**R3 — a Beatnik run exceeds pdebug's 60 minutes.** Presents as
`end=cap` with a checkpoint. Restart into pbatch (T3 Do 2). If `b1` runs much
longer than `b4`, the direct sum dominates at this size; record both walls.

**R4 — the split cap binds.** Neither code reports it, and progress lines
(every 0.03 of time, many remesh calls apart) cannot show it. Presents as the
codes' `V` parting abruptly rather than drifting, by more than the `b1–b4` and
`noflip–flip` differences in `V`. It is indistinguishable from a remesher
selection difference without per-pass split counts; if T3 sees such a parting,
record it and propose an instrumentation task rather than raising the cap
inside this design.

**R5 — the Beatnik–Python gap is a mesh-realisation gap.** For configuration
(a) the reference's own solver, restarted from Beatnik's step-10 mesh, misses
the reference by area `1.05e-2` and rise `2.8e-3` at `t = 1.24`
(`tasks/milestone1.md`, M1-D3). A gap of that order is not a defect. A defect
looks different: the gap exceeds both floors (`pd–pt`, `b1–b4`) already
before the first remesh changes `V`, or grows while `V` and `F` still agree.

**R6 — a production reinstall during these jobs.** Beatnik jobs load the
production env's libraries at run time. A production `spack install` (for
example by large_rollup) while a `b1`/`b4` job is queued or running can SIGBUS
it (rc 135). Presents as `rc=135` with no solver message. Rerun that job; do
not install while T3j's Beatnik jobs are live.

**R7 — the Python interrupt lands inside a long numpy call.** `SIGINT` raises
only between bytecodes, so the final checkpoint follows the current velocity
evaluation (seconds at these sizes); the 10-minute margin covers it. If the
log lacks `saved final finite checkpoint`, the last periodic checkpoint is the
run's end; record it.

**R8 — checkpoint times differ between codes.** Both write at the first step
past each multiple of 0.03, so paired checkpoints differ by up to one `dt`.
Scalars are interpolated; shape rows carry `|Δt|`. A shape difference that
scales with `|Δt|` is a timing artefact, not a solver difference.
