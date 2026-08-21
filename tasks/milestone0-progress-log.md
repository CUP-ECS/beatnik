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

## M0-D1

The measurement, in three artifacts and two jobs. **The milestone's premise
holds, with room to spare: at `--rtol 1e-10 --atol 1e-12` — the tolerance
regression test 2 already passes at — Beatnik matches the reference at every one
of the 81 checkpointed steps out of 2000, at both levels, on both backends, at
both rank counts. All twelve comparisons. M0-R1 did not fire.** What the 2000
steps do bound is the `1e-12/1e-14` rung, and that is where the ladder has
content.

### Decisions (as given to this task, and honored)

- **The driver is `tests/regression_tests/Beatnik_Test_Milestone0Run.cpp`,
  registered in NO tier**: no `LABELS`, no `add_test`, no manifest line. Its
  CMake loop (`tests/CMakeLists.txt`, "Measurement drivers -- IN NO TIER") is the
  milestone tier's loop stopped short of the point where it labels and emits a
  manifest line, and it deliberately has no argument-list table or `FATAL_ERROR`
  guard: a driver's arguments come from the batch script, because there is no
  manifest line for them to live on. Verified after `spack install`: the gate
  manifest still has **15** non-comment lines, the milestone manifest still
  **0**, and `grep Milestone0Run` over both finds nothing. `examples/
  02_adaptive_mesh_bubble` is unusable here as the task states — `adaptive_mesh_
  bubble.cpp:209` fixes the space to `Kokkos::DefaultExecutionSpace` and the
  installed binary is `+rocm`, so it is HIP-only and the SERIAL half of the
  matrix has no other driver.
- **The driver runs and checkpoints only; every comparison is offline in Python**
  (`tests/regression_tests/milestone0_ladder.py`). It knows no tolerance and
  loads no gold file. What it does assert, because it is cheap there and
  unrecoverable afterwards, is the global vertex and face count **every step**,
  the two carried scalars at 17 digits, and that the step budget was reached.
- **Dev spack env** (`BEATNIK_USE_PROD` unset, and the script *refuses* to run
  with it set), `-q pdebug -t 1h`. The sweep took **1939 s of the 3600 s cap** —
  it fits, comfortably, so there is no performance finding to write up here.
- **The ladder is derived from the `max|e|` / `max|e|/|g|` pair the comparator
  already prints, then confirmed with real invocations.** Every number below is
  the confirmed one; the derived one is carried beside it because the gap between
  them is large and is the interesting part (see "What only running revealed").
- **Peak resident memory from `/usr/bin/time -v` inside `flux run`**, so each
  rank reports its own; wall time from the script clocking each launch, with the
  driver's own `MPI_Wtime` solve time beside it. **GPU-side memory is out of
  scope** — there is no mechanism for it here.

### Provenance

`beatnik@develop` in `~/spack_envs/tuolumne_beatnik` (dev), spec `+rocm +openmp
+serial amdgpu_target=gfx942 build_type=RelWithDebInfo %cce`, install hash
`4bhhtbd`, at commit `66293f5` plus this task's additions. Sweep job
**`f3TT4psJ8it7`** (`beatnik_m0div.f3TT4psJ8it7.log`), timing probe job
`f3TSuF7DFxAB`. Binding on every launch, exactly `run_milestone.flux:195-203`:
`--ntasks=<np> --nodes=$(( (np+3)/4 )) --exclusive --gpus-per-task=1
--cores-per-task=24 --setopt=mpibind=verbose:1`. Ranks 1 and 4 are both one
node at tuolumne's 4-ranks-per-node. Offline tabulation artifacts (JSON + text
for all 12 ladders, 8 growth series and 10 drift/quality series) are under
`/p/lustre5/stewartj/beatnik/milestone0/analysis`.

### Step 1 — the step-0 generator gate. **PASSES at all three levels; M0-R5 did not fire.**

At `--rtol 1e-12 --atol 1e-14`, level 2 against `initial_conditions/gold.npz` and
levels 3 and 4 against their own `_step0000000.npz`. `RESULT: match`, exit 0,
full unambiguous pairing at every level, and identical on SERIAL and HIP.

| level | `vertices` `max|e|` | `max|e|/|g|` | `potential` `max|e|` |
| --- | --- | --- | --- |
| 2 (162 v) | `5.55111512312578270e-17` | `2.916908e-15` | `0` |
| 3 (642 v) | `5.55111512312578270e-17` | `9.017655e-15` | `0` |
| 4 (2562 v) | `5.55111512312578270e-17` | `4.639184e-14` | `0` |

**The absolute disagreement does not grow with subdivision level at all** — it is
the same single ulp of 0.5 at every level. What grows is only the *relative* max,
and only because deeper subdivision puts coordinates closer to zero. M0-R5's
premise — that midpoint-normalize ulps compound across levels — is therefore
**not** what happens: `detail::normalize3`'s reciprocal-multiply and NumPy's
divide differ by at most one ulp per coordinate and that does not accumulate
through the 1->4 splits. No per-level step-0 tolerance is needed; `1e-12` covers
all three.

### Steps 2 and 3 — the tolerance ladder against Python

81 checkpointed steps per run, `derived / confirmed` first-failing step. `--atol`
is two decades below `--rtol` on every rung.

| level, backend, ranks | `1e-12` | `1e-10` | `1e-8` | `1e-6` | `1e-4` |
| --- | --- | --- | --- | --- | --- |
| 3, SERIAL, 1 | 475 / **1325** | — / **none** | — / none | — / none | — / none |
| 3, SERIAL, 4 | 475 / **1350** | — / **none** | — / none | — / none | — / none |
| 3, HIP, 1 | 450 / **1325** | — / **none** | — / none | — / none | — / none |
| 3, HIP, 4 | 450 / **1325** | — / **none** | — / none | — / none | — / none |
| 4, SERIAL, 1 | 450 / **775** | — / **none** | — / none | — / none | — / none |
| 4, SERIAL, 4 | 450 / **775** | — / **none** | — / none | — / none | — / none |
| 4, HIP, 1 | 450 / **775** | — / **none** | — / none | — / none | — / none |
| 4, HIP, 4 | 450 / **800** | — / **none** | — / none | — / none | — / none |

"none" means **no checkpointed step through 2000 fails**, and at those four rungs
it is not merely unconfirmed: no step even *permits* a failure, so the bound is
proved rather than sampled. A dash means there was no candidate to confirm.

Two things to read off this. **The rank count barely moves the horizon** (1325 vs
1350, 775 vs 800 — one checkpoint interval), and **the backend does not move it
at all** at three of the four (level, ranks) pairs. And **level 4 diverges
earlier than level 3** at the tight rung, 775 against 1325, which is the opposite
of the mesh-health ordering M0-G1/M0-G2 found: the finer mesh is the healthier
one and the faster-diverging one.

### Step 4 — the attribution. **Decomposition dependence is NOT what sets the horizon.**

The single most valuable number this task produces, and it needed a correction
before it meant anything (see below). Beatnik rank 1 against Beatnik rank 4, same
level, same backend:

| level, backend | `1e-12` | `1e-10` | field that binds at `1e-12` |
| --- | --- | --- | --- |
| 3, SERIAL | 375 / **475** | 1025 / **none** | `sheet_vector` (375) |
| 3, HIP | 400 / **425** | — / **none** | `sheet_vector` (400) |
| 4, SERIAL | 350 / **350** | 1025 / **none** | `sheet_vector` (350) |
| 4, HIP | 325 / **350** | 1100 / **none** | `sheet_vector` (325) |

Read naively this says Beatnik-vs-Beatnik gives out *before* Beatnik-vs-Python
(475 against 1325), which would make decomposition the binding constraint and
M0-T2 mandatory. **That reading is wrong, and the reason is a field-set
asymmetry nothing in the design had noticed:** the reference's `.npz` carries
nine keys and **`sheet_vector` is not one of them**. So a Beatnik-vs-Python
comparison compares `{vertices, potential, remesh_material_position, time,
initial_volume, initial_min_edge, faces}` and a Beatnik-vs-Beatnik comparison of
two `.h5` files compares **all of those plus `sheet_vector`**.
`compare_output.py` compares whatever is present in both, which is correct — but
it means the two horizons are over different field sets, and `sheet_vector` is
the field that binds every Beatnik-vs-Beatnik rung above.

On the **shared** field set the ordering reverses, and decisively. Derived
first-failing step at `1e-12/1e-14`, per field:

| comparison | `vertices` | `potential` |
| --- | --- | --- |
| L3 vs Python (SERIAL / HIP) | 475 / 450 | 650 / 650 |
| L3 rank1-vs-rank4 (SERIAL / HIP) | **675 / 900** | **1400 / 1350** |
| L4 vs Python (SERIAL / HIP) | 450 / 450 | 525 / 525 |
| L4 rank1-vs-rank4 (SERIAL / HIP) | **725 / 600** | **875 / 1275** |

and in magnitude, peak `max|e|` over the 81 steps:

| level, backend | field | vs Python | rank1-vs-rank4 | ratio |
| --- | --- | --- | --- | --- |
| 3, SERIAL | `vertices` | `8.53317416726895317e-13` | `3.88578058618804789e-14` | **21.96x** |
| 3, SERIAL | `potential` | `1.93782490054417167e-13` | `1.31838984174237339e-14` | **14.70x** |
| 3, HIP | `vertices` | `9.26592136352155649e-13` | `5.77315972805081401e-14` | **16.05x** |
| 3, HIP | `potential` | `2.10338690909139814e-13` | `2.11775041947248610e-14` | **9.93x** |
| 4, SERIAL | `vertices` | `3.17634807345257286e-13` | `1.23678844943242439e-13` | **2.57x** |
| 4, SERIAL | `potential` | `4.51583215266282423e-14` | `2.38697950294408656e-14` | **1.89x** |
| 4, HIP | `vertices` | `3.28848059893971367e-13` | `1.22235555011229735e-13` | **2.69x** |
| 4, HIP | `potential` | `4.60464999463283675e-14` | `1.82076576038525673e-14` | **2.53x** |

**So: at level 3 Beatnik's own decomposition dependence is an order of magnitude
below the Python drift — M0-D1 step 4's "orders of magnitude smaller" branch, and
M0-T2 buys nothing. At level 4 the two are within a factor of 2.6 — "comparable"
— so there M0-T2 would buy at most a factor of ~2.6 in `max|e|`.** At the
observed growth rate that is worth on the order of one to two hundred extra
steps at the `1e-12` rung and *nothing at all* at `1e-10`, where there is no
failure to push out. The recommendation to M0-A1 is therefore **do not take
M0-T2**; the decision is M0-A1's.

**SERIAL is the clean instrument and it says the same thing HIP does.** Peak
rank-1-vs-rank-4 `vertices` `max|e|` is `3.89e-14` (SERIAL) against `5.77e-14`
(HIP) at level 3, and `1.237e-13` against `1.222e-13` at level 4 — HIP is
*within 1.5x of, and at level 4 marginally below,* SERIAL. Within-rank GPU
nondeterminism therefore adds essentially nothing on top of cross-rank summation
order, which was not a given and is worth having as a number.

### Step 5 — the growth series, the volume drift, the minimum quality

`max|e|` on `vertices` and `potential`, at **full double precision** rather than
the comparator's printed `%.6e`: `milestone0_ladder.py growth` recomputes them
through `compare_output.quantized_lexsort` — imported, so the pairing is the
comparator's own. Beatnik np1 against Python:

| step | L3 SERIAL `vertices` | L3 SERIAL `potential` | L4 SERIAL `vertices` | L4 SERIAL `potential` |
| --- | --- | --- | --- | --- |
| 0 | `5.55111512312578270e-17` | `0` | `5.55111512312578270e-17` | `0` |
| 25 | `6.66133814775093924e-16` | `1.90819582357448780e-17` | `6.66133814775093924e-16` | `2.08166817117216851e-17` |
| 100 | `1.55431223447521916e-15` | `1.94289029309402395e-16` | `1.99840144432528177e-15` | `1.87350135405495166e-16` |
| 200 | `2.66453525910037570e-15` | `5.13478148889134900e-16` | `3.05311331771918049e-15` | `6.24500451351650554e-16` |
| 400 | `6.88338275267597055e-15` | `3.49720252756924310e-15` | `7.66053886991358013e-15` | `3.77475828372553224e-15` |
| 600 | `1.45439216225895507e-14` | `8.32667268468867405e-15` | `6.15063555642336723e-14` | `1.77080572427712468e-14` |
| 800 | `2.83106871279414918e-14` | `1.27675647831893002e-14` | `1.52988732793346571e-13` | `2.85327317328665231e-14` |
| 1000 | `5.33462163332387718e-14` | `1.59525170850827180e-14` | `2.01505478969465912e-13` | `3.55618312575245454e-14` |
| 1325 | `8.82627304576999450e-14` | `2.58335020042466112e-14` | `2.88213897192690638e-13` | `4.11337630623620498e-14` |
| 1600 | `2.79221090693226870e-13` | `5.68694397129476670e-14` | `2.87436741075453028e-13` | `3.79141162909490959e-14` |
| 2000 | `8.53317416726895317e-13` | `1.93782490054417167e-13` | `1.31783473023006081e-13` | `3.05588887528074338e-14` |

**The growth is not exponential** — over 2000 steps `vertices` goes from one ulp
to `8.5e-13`, four decades in 2000 steps, which is power-law-like, not the
exponential amplification "Read this first" #2 assumed. **And at level 4 it is
not even monotone:** it peaks at `3.17634807345257286e-13` at **step 1400** and
falls back to `1.3e-13` by step 2000. A first-failing step is therefore a real
statement about a rung but *not* a proxy for "how far apart the codes are"; the
peak is. `sheet_vector` (rank-1-vs-rank-4 only) peaks at
`9.53459533548084437e-13` @1750 (L3 SERIAL) and `3.18478576843972405e-12` @1650
(L4 HIP) — roughly an order of magnitude above `vertices`, which is why it binds.

`time` at step 2000: `|e|` = `1.334043986389588e-12` (L3 SERIAL vs Python),
`1.4492851363456793e-12` (L3 HIP), `3.597122599785507e-14` (L4 SERIAL),
`2.8199664825478976e-14` (L4 HIP). The adaptive dt tracks as "Read this first"
#2 predicted it would: `MPI_MIN` is order-independent.

**Volume drift `V/V0 - 1`, both codes.** Level 3, step 2000: reference
`3.35289418451623078e-09`, Beatnik `3.35298433462583034e-09` (SERIAL np1),
`3.35298611098266974e-09` (SERIAL np4), `3.35298566689345989e-09` (HIP np1),
`3.35298167009057124e-09` (HIP np4). Level 4, step 2000: reference
`4.74141392814431128e-09`, Beatnik `4.74149830509418280e-09` (SERIAL np1),
`4.74150074758483697e-09` (SERIAL np4), `4.74149941531720742e-09` (HIP np1),
`4.74150074758483697e-09` (HIP np4). Intermediate points, level 3 / level 4
reference: step 25 `1.54374513172683692e-10` / `1.59270374666675707e-10`, step 500
`2.40471798029773254e-09` / `2.88076495991163029e-09`, step 1000
`3.05862224436737051e-09` / `4.64355776053082536e-09`, step 1500
`3.27790927734383786e-09` / `4.69385197376936958e-09`.

**Beatnik's drift tracks the reference's to `2.758331e-05` relative, worst case
over all 81 steps and all eight runs** (`sub3_Serial_np4` @ step 1975; the level-4
worst is `1.831088e-05`). That is a factor of **36 tighter than T2d's
`kVolumeDriftRtol = 1e-3`**, so M0-T3 can reuse that relative tolerance
unchanged with a wide margin — while `kVolumeDriftAbsCap = 1e-9` must **not** be
reused, exactly as M0-R3 says: the drift reaches `4.74e-09` at level 4, and the
re-derived cap is `~1e-8`.

**Minimum quality tracks to `6.643130e-12` relative, worst case over all steps
and runs.** Level 3 global minimum `3.826562959465268e-02` at step 1700 (Beatnik
SERIAL np1) against the reference's `3.826562959474448e-02` at the same step;
level 4's minimum is its step-2000 value, `1.24242126647961221e-01` against
`1.24242126647510803e-01`. **No Beatnik run stopped early and every entity count
held**, so **M0-R2 and M0-R6 did not fire on Beatnik's side either** and the
compare-depth ceiling is 2000 steps on both sides at both levels.

**M0-R4 did not bite.** Every one of the 972 cross-code and cross-rank
comparisons paired every vertex unambiguously — `ambiguous cpp=0 gold=0`
throughout, at the default `--match-eps 1e-9`, at both levels and at step 2000.
Cross-code pairing, which M0-G1/M0-G2 left untested, is therefore now tested.

**The counts never changed.** All 81 steps of all eight runs report
`(642, 1280)` or `(2562, 5120)` and nothing else, both in the driver's per-step
integer check and in the comparator's structural line. Adaptivity did not leak
into the configuration that exists to exclude it.

### Step 6 — cost

Level 4, 2000 steps, from job `f3TT4psJ8it7`. "solve" is the driver's own
`MPI_Wtime` around the step loop; "launch" is the script's clock around
`flux run` and carries startup and I/O.

| backend, ranks | solve (s) | s/step | launch (s) | peak RSS per rank (kB) |
| --- | --- | --- | --- | --- |
| HIP, 1 | `18.626937` | `0.009313` | 22 | 851068 |
| HIP, 4 | `38.864910` | `0.019432` | 45 | 1015732 / 1016188 / 1017676 / 1019096 |
| SERIAL, 4 | `375.156870` | `0.187578` | 382 | 708172 / 708404 / 709276 / 710792 |
| SERIAL, 1 | `1289.417426` | `0.644709` | 1293 | 705000 |

Level 3 for scale: HIP np1 `8.198673` s, HIP np4 `29.301308` s, SERIAL np4
`43.165985` s, SERIAL np1 `86.176564` s; peak RSS 850088 kB (HIP np1), ~707 MB
per rank on SERIAL. **The whole ten-launch sweep took 1939 s of the 3600 s
pdebug cap** against the reference's ~110 minutes of single-core Python for the
two gold sets, so the port is roughly 3x faster than the reference on the
worst-case configuration and ~350x faster on the best.

Two cost facts worth carrying to M0-A1. **HIP gets *slower* with more ranks**
(0.0093 -> 0.0194 s/step at level 4) — 2562 vertices does not fill one MI300A, so
four ranks add ring communication to a device that was already idle; the
crossover is above level 4. **SERIAL scales the other way and nearly ideally**
(0.645 -> 0.188, 3.4x on 4 ranks), which is what an O(N^2/P) direct sum with no
device to saturate should do. Memory is flat in the rank count on SERIAL
(~705 MB/rank at 1 and 4 ranks), so the direct BR ring's `O(N/P)` source
storage is nowhere near the footprint at this size — the footprint is Kokkos and
the runtime, not the problem. Any depth M0-A1 asserts is affordable at every
point of this matrix; the binding cost is SERIAL np1 at level 4 at 21.5 minutes,
which is a *milestone*-tier cost and not a gate cost.

### Departures from the stated **Do** steps, and what running forced

- **`milestone0_ladder.py` grew a third subcommand, `growth`, that the task did
  not ask for.** The comparator prints `max|e|` as `%.6e`, six significant
  digits — enough to place a rung, but it throws away most of a growth series
  that the exit criterion wants at 17 digits. `growth` recomputes `max|e|` in
  NumPy at full precision **through `compare_output.quantized_lexsort`**, imported
  rather than reimplemented, so the vertex pairing is the comparator's own and
  the two cannot disagree about which vertex is which. Without it the `vertices`
  and `potential` series in step 5 would be a 6-digit table.
- **`pair` also grew a "derived first-failing step BY FIELD" table**, for the
  field-set asymmetry above. The ladder is a minimum over fields and therefore
  cannot say *what* gave out; without the per-field breakdown the
  Beatnik-vs-Beatnik horizon reads as decomposition dependence when it is
  `sheet_vector`, a field the Python cannot see. That mistake was made and
  corrected inside this task, which is why the tool now reports the compared
  field set unprompted in all three subcommands.
- **The restricted Beatnik-vs-Beatnik ladder is DERIVED ONLY, never confirmed,
  and cannot be.** `compare_output.py` compares every field present in both files
  and the CLI surface is closed (Conventions), so there is no invocation that
  compares two `.h5` files while ignoring `sheet_vector`. The confirmed
  Beatnik-vs-Beatnik numbers in step 4's first table are over the full shared
  field set; the per-field table beside them is the apples-to-apples comparison
  and is derived. Both are reported as what they are.
- **A timing probe job came first** (`f3TSuF7DFxAB`, the same script under
  `BEATNIK_M0_MODE=probe`, 25 steps per row). The sweep script's deadline guard
  needs a per-row wall-time estimate, and inventing one risks the outcome the
  task names as the worst possible: a job killed at the 1h wall that leaves a
  truncated checkpoint series a tabulator would happily consume. The probe cost
  76 s and its measured per-step costs are the estimates now in the script.
- **`flux job attach` forwards a SIGTERM to the job, and that killed the first
  full sweep** (`f3TSvQtWhW8f`) two rows from the end — the attaching client hit
  a 10-minute cap in the calling harness, took a SIGTERM, and the job died with
  it at 1447 s. The log looked exactly like a completed sweep up to the cut. The
  sweep was re-run from scratch as `f3TT4psJ8it7` rather than patched up with the
  two missing rows, because a measurement assembled from two jobs is not the
  measurement the provenance block claims. **Attach only under a timeout longer
  than the job, or poll `flux jobs --filter=active`.** M0-T1's log entry says
  attach "works and exits with the job's own status", which is true and
  incomplete; this is the missing half.
- **Offline artifacts live on lustre, not `/tmp`.** `/tmp` on tuolumne is
  node-local tmpfs and successive shells land on different login nodes, so an
  analysis directory under `/tmp` silently vanishes mid-task. It did. Everything
  is now under `/p/lustre5/stewartj/beatnik/milestone0/analysis`.
- **`grep` is `ugrep` in this environment and does not preserve argument order
  across multiple files.** A two-file `grep -h` gave interleaved output that read
  as evidence for the exact opposite of the field-set finding above. Read one
  file at a time when the order carries meaning.

### Verified

`spack install` clean; three per-backend driver binaries installed
(`_MPI_SERIAL`, `_MPI_OPENMP`, `_MPI_HIP`) and named by no manifest. Sweep job
`f3TT4psJ8it7` exited **0** with `launched=10 skipped=0`, 81 checkpoints from
each of the eight 2000-step runs and 1 from each level-2 step-0 run, and **22**
`[PASS] Beatnik_Test_Milestone0Run` tallies (1+1+1+4+1+4+4+1+4+1) with zero
`FAIL`, zero `STOPPED EARLY` and zero `ENTITY COUNTS CHANGED`. All twelve
`milestone0_ladder.py pair` runs and all ten `series` runs exited 0.
`milestone0_ladder.py series` over the committed level-3 gold reproduces
M0-G1's own table exactly — final drift `3.3528941845162308e-09`, final minimum
quality `6.303625774911138e-02`, global minimum `3.826562959474448e-02` at step
1700 — which is what validates the tool against a number measured independently
before it existed.

**Affects:** **M0-A1** — every input it was waiting on is now measured. The
compare depth can be **2000 steps, the full gold set, at `--rtol 1e-10 --atol
1e-12`**, at both levels and every point of the matrix, with the margin proved
rather than sampled; a `1e-12` assertion would have to stop at 775 (level 4) or
1325 (level 3) and buys nothing. **The determinism decision is: do not take
M0-T2** — decomposition dependence is 10-22x below the Python drift at level 3
and within 2.6x at level 4, so eliminating it moves no rung that matters, and
M0-R7's `Allgatherv` cost would be paid for nothing. M0-A1 should also record
that `kVolumeDriftRtol = 1e-3` survives re-derivation with 36x margin (worst
`2.758331e-05`) while `kVolumeDriftAbsCap` must move to `~1e-8`, and that
**level 4 diverges *earlier* than level 3** at the tight rung — so with mesh
health favouring level 4 (M0-G1/M0-G2) and agreement favouring level 3, "level 3
primary" now has an agreement-based reason it previously lacked.
**M0-T2** — **conditional, and the condition is not met.** Do not start it.
**M0-T3** — four things it must not get wrong. (1) The gold `.npz` has **no
`sheet_vector`**, so a Beatnik-vs-Python test cannot assert on that field however
much it would like to; the fields available are `vertices`, `potential`,
`remesh_material_position`, `faces`, `time`, `initial_volume`,
`initial_min_edge`. (2) Assert at `1e-10`/`1e-12` to step 2000, with these
measured `max|e|` values in the comment beside the literal. (3) Reuse
`kVolumeDriftRtol = 1e-3`, re-derive `kVolumeDriftAbsCap` to `~1e-8`, and take
the reference drift series from the two `gold/README.md` tables. (4) Its cost is
the step-6 table: at level 4 the tier's four launches are ~22 s, ~45 s, ~382 s
and ~1293 s, so a level-4 member is a ~30-minute tier run and comfortably inside
`run_milestone.flux`'s `-t 30m` only if level 3 is not also in the same job —
budget for that. `makeParams()` in `Beatnik_Test_Milestone0Run.cpp` is the
`SolverParams` M0-T3 needs, already written and already exercised for 2000 steps
at both levels.

## M0-A1

Documentation only: no build, no run, no comparator invocation, no source file
touched. Every number below comes from `## M0-D1` (sweep job `f3TT4psJ8it7`) or
from a `gold/README.md`; nothing was re-measured, and nothing could have been —
a fresh measurement outside that job's provenance would not be comparable to the
table it landed in.

### The four decisions, and the alternative each one rejected

**1. Depth and tolerance: flat, all 81 checkpointed steps through step 2000, at
`--rtol 1e-10 --atol 1e-12`, at both levels.**

*Rejected: a per-step tolerance table.* M0-D1's ladder has **no failing step** at
`1e-10/1e-12` in any of the twelve comparisons — two levels x two backends x two
rank counts against Python, plus the four rank-1-vs-rank-4 pairs — and at that
rung no step even *permits* a failure, so the bound is proved from the
comparator's own printed `max|e|` / `max|e|/|g|` pair rather than sampled. A
table whose every row carries the same number is not a table.

*Rejected: a per-level depth.* The ladder's "none" is identical at level 3 and
level 4, so there is nothing for a per-level depth to express. (The levels do
*not* reach the same physical time at step 2000 — `1.998284` against `1.964304` —
which is why the depth is in steps and never in time; see `## M0-G1, M0-G2,
M0-T1`.)

*Rejected: asserting at `1e-12/1e-14`.* It would stop at step **1325** (level 3
SERIAL np1; 1350 at np4, 1325 on HIP at both rank counts) or step **775** (level
4 SERIAL at both rank counts and HIP np1; 800 at HIP np4). Since `1e-10` has no
failing step through 2000, the tight rung buys nothing the loose one does not
already have and costs 675-1225 steps of asserted depth. The headroom at the
asserted rung is about two decades: peak `vertices` `max|e|` is
`8.53317416726895317e-13` (level 3 SERIAL) and `3.17634807345257286e-13` (level
4 SERIAL, peaking at step **1400** and falling to `1.31783473023006081e-13` by
step 2000).

**2. Level 3 is the primary member; level 4 is a second member.**

*Rejected: level 4 as a recorded measurement only, asserted by nothing.* Level 4
is the set that resolves `eps = 0.025` (initial minimum edge `1.729575e-02`
against `3.457079e-02`) and the healthier frozen mesh — final minimum quality
`1.242421e-01` against level 3's `6.303626e-02`, and level 3 spends its last ~500
steps below `0.1` where level 4 never reaches it. Leaving the trustworthy
resolution regime unasserted while asserting the marginal one was the wrong way
round.

*Rejected: level 4 as primary.* Two measurements put level 3 first. **Agreement:**
level 4 diverges *earlier* at the tight rung, step 775 against 1325 — the
opposite of the mesh-health ordering, and the agreement-based reason "level 3
primary" previously lacked. **Cost:** level 3's four tier launches are
`166.842830` s of solve (`8.198673` HIP np1 + `29.301308` HIP np4 + `43.165985`
SERIAL np4 + `86.176564` SERIAL np1) against level 4's `1722.066143` s
(`18.626937` + `38.864910` + `375.156870` + `1289.417426`), a factor of ~10.3.
Both are affordable — that is why level 4 is still a member — but the cheaper and
later-diverging one is the primary.

*Consequence handed to M0-T3, and not acted on here:* two members do **not** fit
`run_milestone.flux`'s `# flux: -t 30m`
([scripts/tuolumne/run_milestone.flux:5](../scripts/tuolumne/run_milestone.flux#L5)).
Level 4's four launches alone are `22` + `45` + `382` + `1293` = **1742 s of
launch wall, 29.0 minutes**, before level 3's ~167 s of solve and its own startup
and I/O. **M0-T3 raises that walltime.** This task did not edit the script:
`scripts/` is out of its scope, and the raise belongs in the change that adds the
second member.

**3. Beyond-depth comparison is moot, not declined.**

The design asked for a reason if the answer was "not at all", because
beyond-depth structural comparison is nearly free here — connectivity is frozen,
so counts, `time`, volume drift and minimum quality cost almost nothing. The
answer is that **the question does not arise**: decision 1's depth is the entire
gold set (81 of 81 checkpointed steps, step 0 through 2000), so there is no step
beyond the depth to compare. It is recorded as moot rather than as "none" so a
later reader does not read a declined-but-cheap check where there is no check to
decline.

**4. M0-T2 is DECLINED.**

*Rejected: taking M0-T2.* M0-D1 step 4 is the criterion, and it answers on the
shared field set. Peak `vertices` `max|e|` over the 81 steps,
Beatnik-rank-1-vs-rank-4 against Beatnik-vs-Python:

| level, backend | vs Python | rank1-vs-rank4 | ratio |
| --- | --- | --- | --- |
| 3, SERIAL | `8.53317416726895317e-13` | `3.88578058618804789e-14` | **21.96x** |
| 4, SERIAL | `3.17634807345257286e-13` | `1.23678844943242439e-13` | **2.57x** |

Beatnik's own decomposition dependence is **21.96x** below the Python drift at
level 3 SERIAL and **2.57x** below it at level 4 SERIAL. Removing it buys at most
a factor of ~2.6 in `max|e|` — one to two hundred extra steps at the `1e-12` rung
— and **nothing at the asserted `1e-10` rung, where there is no failing step for
determinism to push out.** So M0-R7 is **not paid**: no unconditional
`Allgatherv` of `7N` doubles per RK stage and no `O(N)` per-rank source storage
on the production path, on a path whose scaling future is the FMM (T3a), which is
not bit-compared anyway.

*Rejected: reading the naive Beatnik-vs-Beatnik ladder as the criterion.* Read
over the full shared field set it gives step **475** (level 3 SERIAL) against
1325 vs Python, which would make decomposition the binding constraint and M0-T2
mandatory. That reading is wrong: the binding field at every such rung is
`sheet_vector`, which the reference's `.npz` does not carry, so the two horizons
are over different field sets. M0-D1 found and corrected this inside itself; the
apples-to-apples per-field comparison is the table above.

*Rejected: declining on SERIAL evidence while leaving HIP a doubt.* Peak
rank-1-vs-rank-4 `vertices` `max|e|` is `5.77315972805081401e-14` on HIP against
`3.88578058618804789e-14` on SERIAL at level 3, and `1.22235555011229735e-13`
against `1.23678844943242439e-13` at level 4 — **HIP is within 1.5x of SERIAL and
at level 4 marginally below it.** Within-rank GPU nondeterminism therefore adds
essentially nothing on top of cross-rank summation order, and is **not** what is
being declined.

**Scope of the decline.** It is on **milestone-0 evidence only** — this
configuration, these two levels, these rank counts, this 2000-step horizon, and
the `1e-10` rung M0-A1 asserts. It is **not** a finding that determinism has no
value: `doc/PARALLELIZATION.tex` §"What must NOT change" item 3 still says what it
says, and a future task may retake M0-T2 on other grounds — a deeper horizon, a
tighter rung, higher rank counts, or a bit-reproducibility requirement that is
not about agreeing with the reference. Retaking it is a new decision entry with
its own measurement, not a reversal of this one.

### Volume drift, restated for M0-T3 because it is a literal it must not get wrong

`kGoldVolumeDriftRtol = 1e-3` is **reused** unchanged: Beatnik's drift tracks the
reference's to `2.758331e-05` relative, worst case over all 81 steps of all eight
runs (`sub3_Serial_np4` @ step 1975; the level-4 worst is `1.831088e-05`), a
**36x** margin. `kVolumeDriftAbsCap` is **re-derived to `~1e-8`** and T2d's `1e-9`
must not be reused: the reference's own drift reaches
`3.35289418451623078e-09` at level 3 and `4.74141392814431128e-09` at level 4
(M0-R3). T2d's `kGoldVolumeDrift` is not imported at all — the reference series
for this configuration comes from the two `gold/README.md` tables.

### Verified

Nothing was built or run — correctly, for a decision task. What was checked is
that the four decisions now appear as numbers in `milestone0.md`: `## Problem`
carries the depth, the flat tolerance, the primary level, the moot beyond-depth
treatment and the M0-T2 verdict with its 21.96x / 2.57x; `## Read this first`
item 2's exponential-growth and "where `1e-10` breaks is unknown" claims are
**corrected in place** rather than annotated, since M0-D1 measured the growth as
power-law-like and non-monotone at level 4; M0-T2 is marked **DECLINED** with its
numbers and its entry intact; and M0-T3 carries the depth, the tolerance, both
members, the walltime raise, the volume-drift literals and the absent
`sheet_vector`.

**Affects:** **M0-T2** — **DECLINED, do not start it.** The entry stays in
`milestone0.md` for its design and its exit criterion; it is not pending work,
and no task depends on it. **M0-T3** — its specification is now fixed, and it is
the only task left. Depth **2000 steps / all 81 checkpoints**, tolerance
**`--rtol 1e-10 --atol 1e-12` flat** (no per-step table, so its Do step 2's
non-flat branch is dead), **two members** with level 3 primary and level 4
second, **no beyond-depth comparison** (its Do step 5 has nothing to do), it must
**raise `run_milestone.flux`'s `-t 30m`** on the 1742 s + ~167 s numbers above,
it **reuses `kGoldVolumeDriftRtol = 1e-3`** and **re-derives
`kVolumeDriftAbsCap` to `~1e-8`** while importing neither of T2d's literals, and
it can assert only the seven fields the gold `.npz` carries — `vertices`,
`potential`, `remesh_material_position`, `faces`, `time`, `initial_volume`,
`initial_min_edge` — because there is **no `sheet_vector`** in it.
**M0-R8** — satisfied on the design side: the depth is a number in
`milestone0.md` in three places and in this entry, so lowering it in a header
later is visible as a contradiction rather than a silent green. **M1-T1** in
`milestone1.md` — unaffected.

## M0-T3

The milestone-0 comparison test, in two source stems and one assertion body.
**This section was written with the tier's first full run still in flight** and
is completed at the bottom under "The tier run"; the sub-sections above it were
all settled before that job was submitted.

### Decisions (as given to this task, and honored)

- **Two members, two source stems, one body.**
  `Beatnik_Test_Milestone0Frozen.cpp` carries the whole test with the subdivision
  level in `BEATNIK_M0_LEVEL`, defaulting to **3** (M0-A1's primary member);
  `Beatnik_Test_Milestone0FrozenL4.cpp` is a documented header plus
  `#define BEATNIK_M0_LEVEL 4` and `#include` of the first. Two stems is what
  keeps `_beatnik_args_<stem>_abs`/`_rel` honest — each member names its **own**
  gold directory — and it left the milestone registration loop at
  `tests/CMakeLists.txt` **completely untouched**, which one stem with two
  argument lists could not have done.
- **Every per-level literal is re-derived, never transferred.** Entity counts
  `642`/`1920`/`1280` (level 3) and `2562`/`7680`/`5120` (level 4); the two
  carried scalars; the polyhedral deficit `kVolumeOverSphere`
  (`9.91393842629754940e-01` at level 3, `9.97839171610598097e-01` at level 4 —
  T2d's `0.96616074859858714` is the *subdivision-2* value and is at
  `Beatnik_Test_DirectSolve10Steps.cpp:182`, not `:180` as milestone0.md said,
  now corrected); the final `time`; and the 81-entry reference volume-drift
  series.
- **One flat pair of tolerance literals**, `kRtol = "1e-10"` / `kAtol = "1e-12"`,
  with M0-D1's peak `max|e|` beside them (`8.53317416726895317e-13` level 3
  SERIAL at step 2000; `3.17634807345257286e-13` level 4 SERIAL at step **1400**,
  falling to `1.31783473023006081e-13` by 2000). No table: M0-A1's ladder is
  flat.
- **`kVolumeDriftRtol = 1e-3` reused, `kVolumeDriftAbsCap` re-derived to `1e-8`.**
  Neither of T2d's `kGoldVolumeDrift` nor its `1e-9` cap is imported. Note the
  symbol name: milestone0.md's M0-T3 entry said to reuse `kGoldVolumeDriftRtol`
  and **there is no such symbol** — T2d's is `kVolumeDriftRtol` at
  `Beatnik_Test_DirectSolve10Steps.cpp:244`. The value and the instruction were
  right, the name was not; milestone0.md is corrected.
- **No CLI option added**, and the gold-set `install()` rules and the tier's
  manifest were left exactly as M0-T1 wrote them.

### The reference volume-drift series: 17 digits, from the `.npz` rather than the table

milestone0.md said to take the reference drift series from each `gold/README.md`
table, and those tables print `V/V0 - 1` at **seven** significant digits, which
the Conventions' "17-digit literals" rule cannot be satisfied from. **Departure:**
the 81 literals per level were regenerated at full precision with
`milestone0_ladder.py series` over the same committed `.npz` the tables were
computed from — the tool M0-D1 wrote and validated against M0-G1's independently
measured table — and they reproduce each README's 7-digit column at every one of
the 81 steps. Spot values, which are also the ones `## M0-D1` and `## M0-A1`
quote independently: level 3 step 25 `1.54374513172683692e-10`, step 500
`2.40471798029773254e-09`, step 2000 `3.35289418451623078e-09`; level 4 step 500
`2.88076495991163029e-09`, step 1000 `4.64355776053082536e-09`, step 1500
`4.69385197376936958e-09`, step 2000 `4.74141392814431128e-09`. The provenance
comment in the source says exactly this.

### What the entity-count assertion actually is

milestone0.md's Do step 4 asks for an `MPI_Allreduce` over **owned** counts
rather than a number read from Tessera. Both are in, and they are different
checks rather than a belt-and-braces duplicate:

- **Every step**, Tessera's own `globalVertexCount()` / `globalFaceCount()`
  against the generator's — cheap, integer, and the thing that catches
  adaptivity leaking in at the step it leaks.
- **At every compared step**, `MPI_Allreduce(MPI_SUM)` over
  `ownedVertexCount()` / `ownedEdgeCount()` / `ownedFaceCount()` against
  `V`/`E`/`F` — R9 discriminator 1, a second independent path to the same number,
  and the one that would catch owned sets that stopped partitioning the global
  ones while the global counts stayed right.

### Step 0 is a compared step

M0-A1's depth is 81 checkpoints, steps 0 **through** 2000. `Solver::setup()`
writes step 0 unconditionally (`src/Beatnik_Solver.hpp`, the `writeCheckpoint()`
at the end of `setup()`), so the test compares `lastCheckpointPath()` against the
step-0 gold *before* the step loop. Without that the depth would be 80 files, not
81, and the M0-D1 step-1 generator gate — the check that the two icospheres agree
at this subdivision level at all — would not be in the test.

### The two failure directions, and how each is demonstrated

- **A detected mismatch, not a load error.** The negative case is inside the test:
  the final state is compared against the **step-0** gold and must exit **exactly
  1**. Exit 2 (could not load) is a vacuous pass and is rejected as such. It
  therefore runs on every one of the tier's eight launches rather than being a
  one-off demonstration.
- **A build with `--dynamic-remesh` forced fails on the constant-entity-count
  assertion of step 4.** `BEATNIK_M0_FORCE_DYNAMIC_REMESH`, a build-time define
  defaulting to **0**, switches `makeParams()` to T4b's accepted split-only
  remesh set at `--remesh-every 4` with the sizing field tightened to
  `h_max 1e-3 / h_min 1e-4` so splits are certain at 642 and 2562 vertices. It is
  a define and not an option because milestone0.md's conventions close the CLI
  surface, and it is kept in the source so the demonstration is reproducible from
  one line rather than from a reconstruction of this session.

  **Demonstrated**, job **`f3Td5AJiqAfq`** (built with the define at 1, then
  reverted and rebuilt): the runner reported `[milestone] FAIL` and every one of
  the eight launches failed with
  `ENTITY COUNTS CHANGED at step 4: vertices 942 (expected 642), faces 1880
  (expected 1280)` at level 3 and `vertices 2862 (expected 2562), faces 5720
  (expected 5120)` at level 4, on SERIAL and HIP at ranks 1 and 4. Crucially it
  failed **there** and not earlier: `step 0 comparator exit 0` on all eight, so
  the gold-directory resolution, the manifest-relative argument pair, the lustre
  scratch and the comparator plumbing were all proved by the same job — and the
  negative case still returned `exit 1`, so the run had moved the surface by step
  4.

### The walltime, and the number M0-A1's estimate did not carry

M0-A1 handed this task a raise from `-t 30m` justified by level 4's four launches
at `22 + 45 + 382 + 1293 = 1742` s (29.0 min) plus level 3's `166.842830` s of
solve — about 32 minutes of measured run wall, for which the task specified 40m.
**The forced-remesh job measured what that estimate omitted:** each launch spawns
**83** `compare_output.py` invocations (81 compared steps plus the negative case),
and on-node they cost ~`0.65` s each (`comparator=1.10`-`1.55` s for the two
invocations that build reaches, on every backend and both levels). That is
~54 s per launch and **~7 minutes over the eight launches**, which puts the tier
at ~39.5 min against a 40m cap — about 1% of margin, and M0-R8 is precisely the
failure where a run killed at the wall leaves the queue looking like a shorter
pass. **Raised to `-t 60m` with the user's decision**, the same cap M0-D1's own
sweep ran under in `pdebug`. The script comment carries all three numbers.

The test reports the split itself, so a later session sizing a third member does
not have to re-derive it: `[m0t3] COST level=.. space=.. np=.. steps=..
wall=.. comparator=.. peak_rss_kb=..`, one greppable line per launch, with peak
resident memory from `getrusage(RUSAGE_SELF).ru_maxrss` reduced by `MPI_MAX` so
the number is the worst rank's. **GPU-side memory stays out of scope**, as it was
for M0-D1 — there is no mechanism for it here.

### Verified before the tier run

`spack install` clean in the **dev** env (`BEATNIK_USE_PROD` unset), twice: once
with `BEATNIK_M0_FORCE_DYNAMIC_REMESH` at 1 for the failure-direction
demonstration and once with it back at 0, and the installed binary was checked to
no longer carry the forced build's banner string before the real job was
submitted. `beatnik_milestone_manifest.txt` carries **six** non-comment lines —
two members x `SERIAL;OPENMP;HIP`, of which the runner's `_<BACKEND>` filter
selects the four that are the tier — each naming its own level's gold directory
and the comparator, manifest-relative. `beatnik_gate_manifest.txt` still carries
**fifteen**: the diff to `tests/CMakeLists.txt` is confined to the milestone
tier's source list, its argument-list table and its comment block, and the
regression loop is untouched.

### The tier run

Two jobs, submitted back to back, and they are the two halves of M0-T3's exit
criterion.

**The gate half is MET.** Job **`f3Td82RoqczB`**
(`beatnik_regression_minset.f3Td82RoqczB.log`) exited **0** with exactly
**sixty** `[gate] ===` launch lines and `[gate] PASS (label=regression)`. That is
the `spack`-mode form of "`ctest -N -L regression` lists 60 cases", exactly as
M0-T1's exit criterion was met, and it is the check that the milestone tier's two
new members took nothing out of the gate.

**The milestone half is MET.** Job **`f3Td7rshE3y1`** came back `COMPLETED` after
**37.25 min** of the 60m cap, with exactly **eight** `[milestone] ===` launch
lines — both members x {SERIAL, HIP} x ranks {1, 4} — **zero** `[FAIL]` lines and
`[milestone] PASS (label=milestone)`. It is not M0-R8: each of the eight launches
carries **81** `comparator exit 0` lines (the whole gold set, step 0 through step
2000), `step 0 comparator exit 0`, and the in-test negative case
`NEGATIVE case, final state vs the step-0 gold: exit 1` — the detected-mismatch
direction, not exit 2. Per-rank tallies are `2337/2337` checks on rank 0 and
`2172/2172` on the others (the difference is the comparator invocations, which
only rank 0 makes).

**The cost, from the eight `[m0t3] COST` lines** — the numbers M1-T1 needs to size
the tier's walltime again:

| level | space | np | wall (s) | comparator (s) | s/step | worst-rank peak RSS (kB) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | Serial | 1 | `120.939298` | `35.974838` | `0.060470` | `704272` |
| 3 | Serial | 4 | `77.767782` | `35.462679` | `0.038884` | `708096` |
| 4 | Serial | 1 | `1324.510988` | `36.285307` | `0.662255` | `708644` |
| 4 | Serial | 4 | `409.246860` | `36.132166` | `0.204623` | `710896` |
| 3 | HIP | 1 | `41.848757` | `34.916752` | `0.020924` | `850392` |
| 3 | HIP | 4 | `63.463393` | `35.618029` | `0.031732` | `1017004` |
| 4 | HIP | 1 | `51.270562` | `34.837122` | `0.025635` | `851148` |
| 4 | HIP | 4 | `73.493593` | `35.875890` | `0.036747` | `1018896` |

Total in-test wall **`2162.542`** s (36.0 min) against the job's 37.25 min, so
startup, the tier wrapper and the scratch teardown cost ~1.2 min in total. Three
things in that table are worth carrying forward:

- **The binding launch is still level 4 SERIAL at 1 rank**, `1324.510988` s —
  61% of the tier on its own, and `0.662255` s/step against M0-D1's
  `0.644709`, so the estimate held to 2.7%.
- **The comparator is a fixed ~`35`-`36` s per launch**, not the ~54 s the
  forced-remesh job's on-node ~`0.65` s x 83 predicted; the real cost is
  ~`0.43` s per invocation. It is essentially independent of level, backend and
  rank count, which makes it a flat **~4.7 min** over the eight launches and the
  term that dominates the two cheap HIP level-3 launches (`34.9` of `41.8` s).
- **Peak RSS is `704272`-`1018896` kB per rank**, within M0-D1's
  `705000`-`1019096` kB, and HIP at 4 ranks is the high-water mark on both
  levels. GPU-side memory stays out of scope.

One cosmetic wart, recorded so a later session does not read it as a bug: the
per-rank `FINAL` note prints the `MPI_MAX`-reduced comparator seconds beside the
*local* invocation count, so non-zero ranks say `comparator 35.8759 s in 0
invocation(s)`. The `[m0t3] COST` line, which is what this table is built from, is
rank 0's and is consistent.

**Superseded, and fixed in the same change:** CLAUDE.md's "Minimum test set" and
`docs/testing.md` both said the tier was "about 35 minutes under `-t 40m`" —
written before the walltime was raised to 60m. Both now carry the measured 37.25
min against `-t 60m` and this job's ID.

Job **`f3Td7rshE3y1`**'s pre-run description follows; it is what the run above
confirmed. Job
**`f3Td7rshE3y1`** (`beatnik_milestone.f3Td7rshE3y1.log`), submitted from the
repo root against the reverted (`BEATNIK_M0_FORCE_DYNAMIC_REMESH 0`) install.
What it must show: **eight** `[milestone] ===` launch lines — two members x
{SERIAL, HIP} x ranks {1, 4} — with **zero** `[FAIL]` lines and
`[milestone] PASS (label=milestone)`, and eight `[PASS]
Beatnik_Test_Milestone0Frozen` tallies.

**How it was closed**, kept because the waiting protocol is the part worth
reusing: `flux job status f3Td7rshE3y1` — it blocks until the job completes,
exits with the job's own status, works on an already-finished job and can be
re-run as often as you like when the harness's ten-minute command timeout cuts
the wait; the job is unaffected. **Never `flux job attach`** — it forwards
SIGTERM and that is what destroyed M0-D1's first sweep (`f3TSvQtWhW8f`), leaving
a log that read exactly like a completed run. Then the log was read for the
launch-line count, the `[FAIL]` count and the final `[milestone] PASS` line
rather than the exit status alone, because a job killed at the wall leaves the
queue looking like one that passed (M0-R8) — the 81-per-launch comparator count
above is what rules that out here. M0-T3 is then **DONE** in `milestone0.md`, and
milestone 0 with it. Nothing went red, so no README "Known Issues" entry was
needed; had it, the response would have been to report the failure verbatim
rather than substitute a shallower depth, a looser tolerance or one member.

**Affects:** **M1-T1** in `milestone1.md`, which registers the tier's next
member: the tier is no longer empty, so M1-T1 adds a **third** source stem and a
third `_beatnik_args_<stem>_abs`/`_rel` pair, and it must **raise
`run_milestone.flux`'s walltime again** — the tier is already ~40 minutes of the
60m cap, and the per-launch comparator cost (~0.65 s x one invocation per
compared step) is the term that estimate has to include. The pattern for a
member that differs from an existing one only in a compile-time constant is
`Beatnik_Test_Milestone0FrozenL4.cpp`: a stem that `#define`s and `#include`s,
never a copy of the assertion body.
