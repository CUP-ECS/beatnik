# Milestone 0 — the reference's default physics on a frozen mesh, run long

**Status:** IN PROGRESS — M0-T1, M0-G1, M0-G2 and M0-D1 are **DONE**; M0-A1 is
next, and every measurement it depends on is in
`milestone0-progress-log.md` under `## M0-D1`.

## Problem

Milestone 0 is [`milestone1.md`](milestone1.md) **minus adaptivity**: the
reference's full default *physics* — README configuration (a) of
`~/research-bridges/zmodel-steve/zmodel3d-amr` — run with connectivity frozen for
the whole run, on a mesh fine enough that the frozen connectivity still resolves
the problem, for as many timesteps as the two codes actually agree for. It exists
because milestone 1 bundles two independent questions into one comparison — *does
Beatnik reproduce the reference's evolution?* and *does Beatnik reproduce the
reference's mesh edits?* — and only the second one needs T4d. Milestone 0 answers
the first, alone, and it needs **no new solver code to run**.

On the Beatnik side, `examples/02_adaptive_mesh_bubble`'s command line **is** the
configuration — it is the argument surface these flags name — but **no
milestone-0 task runs that binary.** Both the tasks that execute this
configuration build it as a `SolverParams` instead: M0-D1 in its own measurement
driver (`Beatnik_Test_Milestone0Run.cpp`, which also explains why the example
cannot be used — it is HIP-only) and M0-T3 in its test. The reference side is

```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --no-dynamic-remesh --refine-every 0 \
  --source-quadrature vertex \
  --icosphere-subdivisions <L> --steps 2000 \
  --checkpoint-every-steps 25 --no-video --checkpoint-dir results<L>
```

at `<L>` = 3 and 4 (M0-G1, M0-G2). Every unlisted option is at its `parse_args`
default. Acceptance is `tests/regression_tests/compare_output.py` over the
per-step checkpoints, at a depth and a tolerance ladder that M0-D1 **measures**
and M0-A1 **records**.

What exists now is the same comparison at **10 steps** and **162 vertices**:
`tests/regression_tests/direct-solve-10-steps` and
[tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp),
which passes at `--rtol 1e-10` at ranks 1-6 on SERIAL and HIP. Milestone 0 is
that test, deeper and finer, in its own test tier, with the depth and the
tolerances derived from measurement instead of chosen.

The end state: a `milestone`-tier test comparing Beatnik against a 2000-step
reference gold set at two mesh resolutions, to a measured and documented depth,
green at ranks 1 and 4 on SERIAL and HIP.

**Out of scope.** Everything in milestone 1's out-of-scope table, and in addition
every adaptivity path — T4a's indicator-driven refinement, T4b's split pass,
T4c's tangential relaxation and all of T4d. Those are reachable code in this
checkout; milestone 0 does not run them, because `--no-dynamic-remesh
--refine-every 0` is the whole point. `projectToVolume` is out of scope for the
same reason: all three call sites in the reference sit inside a branch this
configuration switches off — `run_adaptive_mesh_bubble.py:1468` (refine), `:1516`
(remesh) and `:1565` (`--redistribute-every`, which defaults to 0) — so no
configuration milestone 0 runs executes it, in either code.

## Read this first

Four places where inspection contradicted the brief or milestone 1's text, and
the answer to the question the brief asked.

### 1. Milestone 0 needs no new solver code to *run*

The brief asked what must be done to reach milestone 0. The answer is: generate
gold sets, measure, decide, and write a test — not implement anything. Every
rejection in `Solver::requireSupportedConfiguration`
([src/Beatnik_Solver.hpp:729-870](../src/Beatnik_Solver.hpp#L729)) is guarded on

```
refining  = !dynamic_remesh && amr.refine_every > 0      (:747-748)
remeshing =  dynamic_remesh && remesh_every > 0          (:749-750)
```

and this configuration makes **both false**, so all seven adaptivity rejections
are skipped. That includes the `--isotropic-cleanup` one
([:769-775](../src/Beatnik_Solver.hpp#L769)), which is milestone 1's first
blocker: it fires only under `refining || remeshing`. The eighth rejection,
`--field-filter-every > 0` ([:867](../src/Beatnik_Solver.hpp#L867)), is off by
default. So the command above is accepted at HEAD.

The reference agrees that `--isotropic-cleanup` is moot here rather than merely
tolerated: its two cleanup call sites are `run_adaptive_mesh_bubble.py:1452-1464`
(inside `if refine_diag.marked_faces > 0` at `:1438`, itself inside the
`refine_every > 0` branch at `:1424`) and `:1491-1504` (gated on
`remesh_diag is not None`, inside the `args.dynamic_remesh` branch at
`:1469-1471`). With both branches dead, the flag
is unreachable in **both** codes, which is why the command neither passes it nor
negates it — matching `direct-solve-10-steps/README.md`, whose command does the
same.

### 2. Answer to the brief's question: yes, much deeper than 500 steps — but the depth is still a measurement, and for a different reason than milestone 1's

Removing AMR removes **three** of milestone 1's four divergence mechanisms
outright, and it changes the *character* of the fourth's failure:

| Mechanism | Status with AMR off |
| --- | --- |
| R7 — the reference's serial, order-dependent edit sets vs Tessera's independent sets | **Gone.** No edit is applied. |
| R13 — `Tessera::splitEdges` picks the shorter quad diagonal, `mesh.py::refine_marked_faces` a fixed one | **Gone.** No split. |
| R4 — greedy per-pass caps truncating different candidate orderings | **Gone.** No pass. |
| R2 — cross-rank summation and reduction order | **Present, and now the only mechanism.** |

The qualitative consequence is the important one: `compare_output.py` is
structural before it is numeric — it fails outright on a differing vertex or face
count ([:555-566](../tests/regression_tests/compare_output.py#L555-L566)) and
requires the canonicalized face lists to be equal
([:657-675](../tests/regression_tests/compare_output.py#L657-L675)). With
connectivity frozen, the counts are the generator's for the whole run
(`10*4^L+2` vertices, `20*4^L` faces) and the face list never changes, so **the
structural comparison cannot fail at any step**. Failure becomes smooth and
graded — field errors that grow — instead of catastrophic and uninterpretable.
That alone is worth far more than 500 steps.

What bounds the depth is round-off amplification, and Beatnik is **not**
decomposition-independent today:

- The direct BR sum circulates source blocks around a rank ring, **starting with
  the rank's own block** ([src/Beatnik_BRSolverDirect.hpp:285-286](../src/Beatnik_BRSolverDirect.hpp#L285)),
  so the summation order for a given target depends on the rank count and on
  which rank owns it.
- Three global sums per RK stage are plain `MPI_Allreduce(MPI_SUM)`, whose
  partial-sum order is the partition's: the volume-flux inner products
  ([src/Beatnik_VolumeProjection.hpp:150](../src/Beatnik_VolumeProjection.hpp#L150)),
  the area-weighted potential re-centring
  ([src/Beatnik_SurfaceState.hpp:388](../src/Beatnik_SurfaceState.hpp#L388)) and
  its `ZModelSolver` twin
  ([src/Beatnik_ZModelSolver.hpp:634](../src/Beatnik_ZModelSolver.hpp#L634)).

The reference's own handoff document names this exact hazard:
`doc/PARALLELIZATION.tex` §"What must NOT change" item 3 — *"Summation order
changes results at round-off level. That is normally fine, but if you are
bit-comparing against a serial baseline, use a deterministic reduction
(fixed-order or pairwise) or you will chase phantom regressions. Prefer
deterministic reductions by default — the sheet is sensitive and the runs are
long."* Milestone 0 is precisely the long run bit-compared against a serial
baseline.

Three findings that make the outlook better than that sounds, and one that makes
it worse:

- **The adaptive dt is *not* a divergence amplifier here.** `chooseStepSize`
  reduces `h_min` with `MPI_MIN`
  ([src/Beatnik_TimeIntegrator.hpp:262](../src/Beatnik_TimeIntegrator.hpp#L262)),
  and a min reduction is order-independent — so `dt` is bit-identical across rank
  counts until `h_min` itself diverges, rather than injecting fresh noise every
  step. Milestone 1's `time`-series trap does not compound here.
- **`--eps 0.025` bounds the amplification rate.** The regularized kernel caps
  the smallest dynamically active scale, so the roll-up's growth rate is bounded
  by `eps`, not by the mesh.
- **`dt` does not shrink when the mesh is refined at `t=0`.** The formula is
  `dt = max(dt_min, dt0 * min(1, h_min/h_min^0)^p)` — *relative* to the run's own
  initial minimum edge — so a level-4 run starts at the same `dt0` as a level-2
  one and level 4 is not 4× the steps for the same physics. It does **not** make
  the two levels land on the same physical time: `h_min` diverges from `h_min^0`
  at a level-dependent rate, and 2000 steps reaches `time = 1.998284` at level 3
  against `1.964304` at level 4 (initial minimum edges `3.457079e-02` and
  `1.729575e-02`). **Every cross-level comparison is therefore by step, never by
  time** — and `time` is under test besides.
- **Against that:** the growth is exponential in principle, so each decade of
  loosened tolerance buys only a roughly *constant* number of extra steps. The
  deepest field agreement ever demonstrated in this port is 10 steps at
  `--rtol 1e-10` (T2d, level 2), with the volume drift agreeing to two ulps of
  the drift itself. Where `1e-10` breaks is unknown.

**Therefore this document asserts no step count.** The gold sets run to 2000
steps, M0-D1 measures the first failing step at each of five tolerances *and*
separately measures how much of the disagreement is Beatnik-vs-Beatnik across
rank counts, and M0-A1 records the decision. If M0-D1 shows the horizon is set by
Beatnik's own decomposition dependence rather than by Python-vs-Beatnik drift,
M0-T2 fixes that — see item 4.

### 3. Why more faces, quantitatively — and why level 4 and not level 5

The brief's instinct is right, and `--icosphere-subdivisions` is fully plumbed:
[src/Beatnik_Params.hpp:517](../src/Beatnik_Params.hpp#L517) →
[src/Beatnik_InitialCondition.hpp:114](../src/Beatnik_InitialCondition.hpp#L114) →
`Tessera::buildIcosphere`, and `examples/02_adaptive_mesh_bubble/InputFile.hpp:303`
parses it. Entity counts are `V = 10*4^L + 2`, `E = 30*4^L`, `F = 20*4^L`.

The resolution argument is about `--eps 0.025`, the kernel's regularization
length, against the mesh's minimum edge. The level-2 gold set records
`initial_min_edge = 0.06897612106381684`, and each level halves it:

| L | V | F | min edge | vs `eps = 0.025` | Python BR, 1 core |
| --- | --- | --- | --- | --- | --- |
| 2 | 162 | 320 | `0.069` | 2.8× **coarser** than the blob | 0.0022 s |
| 3 | 642 | 1280 | `≈0.034` | marginal | 0.0348 s |
| 4 | 2562 | 5120 | `≈0.017` | resolves the blob | 0.4849 s |
| 5 | 10242 | 20480 | `≈0.0086` | resolves it well | 8.3107 s |

The BR column is measured, from `doc/PARALLELIZATION.tex` §"Measured"
(`scripts/benchmark_br.py`, one evaluation). At
`--bernoulli-scalar-mode normal-speed` — the default, and the only mode either
gold set uses — the Bernoulli input is `u·n̂` and **not** a second all-pairs sum
([src/Beatnik_ZModelSolver.hpp:250-255](../src/Beatnik_ZModelSolver.hpp#L250)),
so a step is three BR evaluations plus sparse work. That gives ~0.10 s/step at
level 3 and ~1.5 s/step at level 4: **2000 steps is ~4 minutes and ~50 minutes**
of single-core Python respectively. Level 5 would be ~7 hours per gold set, for
no new resolution regime, which is why the ladder stops at 4.

So: **level 3 is the affordable primary set; level 4 is the one that actually
resolves `eps` and is where the roll-up's mathematics is trustworthy.** Both are
generated (M0-G1, M0-G2) because a failure at level 4 alone could not be
attributed between resolution and agreement.

Tessera's generator and the reference's agree structurally at every level, which
is what makes the face-list comparison safe: same golden-ratio base table, same
20-face order, same `std::map`-ordered midpoint cache, same 1→4 split order
(`Tessera_Icosphere.hpp:80-150` against `mesh.py::icosphere_mesh` 362-440). The
*positions* differ in the last bits and always have — Tessera normalizes by
reciprocal-multiply (`detail::normalize3`, `Tessera_Icosphere.hpp:65-73`) where
NumPy divides — which is why regression test 1 compares at `1e-12` rather than
bitwise. Those ulps compound across subdivision levels, so M0-D1 checks step 0 at
each level explicitly rather than assuming T1c's level-2 result transfers.

### 4. The `milestone` tier moves here from milestone 1

`milestone1.md`'s M1-T1 owned creating the tier. Milestone 0 lands first and needs
it, so **M0-T1 owns it** and milestone1.md's M1-T1 has been reduced to registering
its member in the tier M0-T1 creates. Milestone 1's other IDs are untouched.

## Approach

Four strands, in order of *risk*:

1. **Generate the reference data** (M0-G1, M0-G2) — two human steps, no code,
   independent of everything else, and the long pole in wall-clock terms.
2. **Build the tier** (M0-T1) — independent of the data, so it can proceed in
   parallel.
3. **Measure** (M0-D1) and **decide** (M0-A1). This is where the milestone's
   premise is confirmed or replaced, and where the determinism question is
   settled with numbers instead of argument.
4. **Fix determinism if the measurement says to** (M0-T2, conditional), then
   **assert it** (M0-T3).

### Conventions

Framework.md's conventions table governs, and milestone1.md's additions apply
where they are not adaptivity-specific. These are milestone 0's own:

| Convention | Choice |
| --- | --- |
| Task IDs | `M0-*`. `M0-G*` = human gold-file generation, `M0-D*` = measurement, `M0-A*` = a decision to be recorded, `M0-T*` = code. No `M0-` ID appears in any source string, so unlike milestone 1's `T4d*` these are free to renumber. |
| Test tier | the **`milestone`** label, at ranks **1 and 4** on SERIAL and HIP, outside the 60-launch ship gate. The `regression` tier keeps exactly its five members; promoting anything into it needs the user's confirmation (CLAUDE.md "Minimum test set"). |
| Gold-set layout | `tests/regression_tests/<name>/gold/*.npz` plus that directory's `README.md` carrying the generating command **verbatim**, per [tests/regression_tests/direct-solve-10-steps/README.md](../tests/regression_tests/direct-solve-10-steps/README.md). Names: `milestone0-sub3-2000-steps`, `milestone0-sub4-2000-steps`. |
| Finding a step's gold file | by its `_step%07d.npz` suffix, never by rebuilding the name from a time — the time is under test. `goldForStep` ([Beatnik_Test_DirectSolve10Steps.cpp:263-286](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp#L263)) already does this and is reused. |
| Comparator exit status | a mismatch is **exactly 1**; 2 is a load error and 127/-1 a plumbing failure, and the three are never conflated ([Beatnik_Test_DirectSolve10Steps.cpp:289-306](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp#L289)). |
| Reference numbers in tests | 17-digit literals with the measurement's provenance in a comment, as `kGoldTime` and `kGoldVolumeDrift` already are. A tolerance that was loosened without a recorded `max|e|` beside it is not a tolerance. |
| Failure behavior | loud. A gold file missing for a compared step is a named failure, not a skipped step; a run that stops early is a reported stop step, not a shorter pass. |
| CLI surface | unchanged. No option is added, including for determinism — framework.md's rule is that Beatnik defines no option `parse_args` does not, so M0-T2 (if taken) changes behavior unconditionally rather than behind a switch. |

### Deliberate deviations

- **The two gold sets are 2000 steps even though the asserted depth will be
  less.** Generating to the horizon and no further would mean regenerating the
  moment M0-T2 pushes the horizon out. Python cost makes this cheap (item 3).
- **`--icosphere-subdivisions 2` gets no milestone-0 member.** It is already
  covered at 10 steps by regression test 2, and item 3 shows it does not resolve
  `eps`, so a long level-2 run would measure the amplification of an
  under-resolved discretization — a real number about the wrong problem.
- **No `projectToVolume`.** Both codes' volume drifts linearly on this path
  (T2d measured `5.17e-11` at step 10, level 2), and that is the reference's own
  behavior. Milestone 0 compares the drift against a **measured reference
  series** for its own configuration, and must not reuse T2d's `kGoldVolumeDrift`
  or its `kVolumeDriftAbsCap = 1e-9`. The gold sets measure the drift at step
  2000 as `+3.352894e-09` (level 3) and `+4.741414e-09` (level 4) — both above
  that cap, which would therefore fail for the right reason at the wrong scale.
  The re-derived blow-up cap is `~1e-8`.
- **Determinism, if taken, is not a CLI mode.** See the conventions table.

## Current state

True at HEAD:

- **The configuration runs.** Nothing on this path throws (Read this first #1).
  `--no-dynamic-remesh --refine-every 0` is the configuration regression test 2
  already exercises at ranks 1-6 on both backends.
- **Every option in the milestone-0 command is implemented and validated at
  level 2 for 10 steps:** the initial condition's fast path (T1c), the surface
  operators (T2b), the vertex quadrature and direct BR sum (T2c), the RHS, the
  rate-only volume projection, the TVD-RK3 integrator with adaptive dt, the step
  loop, the checkpoint writer and the diagnostics (T2d).
- **`--icosphere-subdivisions` is plumbed but Beatnik has never been run at any
  value other than 2** by anything in this tree: every test in
  `tests/regression_tests/` sets `kSubdivisions = 2`. The *reference* has now
  been run at 3 and 4 (M0-G1, M0-G2), so levels 3 and 4 exist on the Python side
  only, and Beatnik's step-0 agreement at those levels is unmeasured — which is
  what M0-D1 step 1 gates on.
- **The `milestone` tier exists and is empty** (M0-T1): the label, the
  `BEATNIK_MILESTONE_MPI_RANKS` registration loop
  ([tests/CMakeLists.txt:437-495](../tests/CMakeLists.txt#L437)),
  `beatnik_milestone_manifest.txt` with zero non-comment lines,
  [scripts/tuolumne/run_milestone.flux](../scripts/tuolumne/run_milestone.flux)
  (which exits non-zero on the empty tier, as intended), and the install rules
  for **both** milestone-0 gold sets. M0-T3 therefore adds a test source and its
  `_beatnik_args_<stem>_abs`/`_rel` pair, and nothing else in CMake.
- **Beatnik's trajectory is decomposition-dependent** (Read this first #2), and
  no test measures by how much: regression test 2's rank sweep asserts agreement
  with *Python* at `1e-10` at each rank count, never Beatnik-vs-Beatnik.
- **Both 2000-step gold sets exist**, committed and installed: 81 numbered
  `.npz` each under `tests/regression_tests/milestone0-sub{3,4}-2000-steps/gold`
  with the generating command and the measured series in each `gold/README.md`
  (M0-G1, M0-G2). **M0-D1 has compared Beatnik against both**, over the whole
  2000 steps at ranks 1 and 4 on SERIAL and HIP; no test asserts it yet — that is
  M0-T3.
- `CheckpointIO::read`/`RestartReader::load` still throw (T5b). Not needed:
  milestone 0 compares files Beatnik writes against `.npz`, and reads neither.

## Progress log

Session-by-session record: **[`milestone0-progress-log.md`](milestone0-progress-log.md)**.

Read it before implementing a task, before changing any signature this document
names, and before reopening a question this document states flatly — a completed
task may have changed the plan for a later one, and each entry's `**Affects:**`
line is the index of exactly that.

## Task sequence

### M0-T1 — the `milestone` test tier — **DONE**

**Depends on:** none. Do it first or in parallel with the gold sets; it is
independent of both.

**Fill in:** [tests/CMakeLists.txt](../tests/CMakeLists.txt) — a third tier
alongside `regression` and `unit` — a new `scripts/tuolumne/run_milestone.flux`,
[docs/testing.md](../docs/testing.md) and CLAUDE.md's "Minimum test set".

**Reference:** the tier comment at [tests/CMakeLists.txt:11-48](../tests/CMakeLists.txt#L11);
the standalone regression registration loop at
[:309-372](../tests/CMakeLists.txt#L309), which is the shape to copy (per-backend
generated translation unit pinning `BEATNIK_TEST_EXEC_SPACE`, one ctest case per
rank count, the `_beatnik_args_<stem>_abs`/`_rel` pair with its `FATAL_ERROR` for
a forgotten entry, and the manifest line emitted from the same loop that applied
the label); the manifest generation at [:520-575](../tests/CMakeLists.txt#L520);
the gold-set `install()` rules with their `FATAL_ERROR` at
[:500-517](../tests/CMakeLists.txt#L500); and the installed-path runner
[scripts/tuolumne/run_regression_minset.flux](../scripts/tuolumne/run_regression_minset.flux),
already parameterized by `BEATNIK_GATE_LABEL`/`BEATNIK_GATE_BACKENDS`/`BEATNIK_GATE_RANKS`
(`:73-75`) but hardcoding `beatnik_gate_manifest.txt` (`:112-120`) and carrying
the vacuous-pass guard at `:201-208`.

**Do:**

1. Add `BEATNIK_MILESTONE_TEST_SOURCES`, a `BEATNIK_MILESTONE_TARGETS` global
   property, `LABELS milestone` on the ctest cases, and
   `beatnik_milestone_manifest.txt` with the same line format and the same
   "paths are relative to this file's directory" convention.
2. Install the manifest and the tier's gold data under `share/Beatnik/tests`,
   preserving the repo layout, exactly as the regression tier's rules do, and
   keep the `FATAL_ERROR` pattern: a milestone test installed without its gold
   set is not installed. The gold data is both milestone-0 sets — the `.npz`
   under `tests/regression_tests/milestone0-sub3-2000-steps/gold` and
   `.../milestone0-sub4-2000-steps/gold`, and each directory's `README.md` —
   globbed with the same guard as the T2a rule at
   [tests/CMakeLists.txt:505-517](../tests/CMakeLists.txt#L505). They are
   installed here rather than by M0-T3, which leaves M0-T3 a test-source task
   only. The glob excludes `checkpoint_latest.npz` (see M0-G1's exit
   criterion): it duplicates the final step and shipping it would put a second
   copy of the largest file in every install.
3. `run_milestone.flux` runs the tier at ranks **1 and 4** on SERIAL and HIP. Do
   not generalize `run_regression_minset.flux` in place; the gate script is
   single-sourced against CLAUDE.md's gate definition and must keep saying
   `regression` × ranks 1-6. It must source the resolver first and resolve
   binaries through `beatnik_exe`, and it must honour `BEATNIK_TEST_SCRATCH`
   (**a parallel filesystem** — the checkpoints go through MPI-IO and a
   node-local scratch fails every multi-node launch, CLAUDE.md. Use
   `/p/lustre5/stewartj/beatnik/milestone0/<test_name>` as the I/O directory.
   Delete and recreate this directory before each run of <test_name>).
4. Extend the tier comment block, `docs/testing.md` and CLAUDE.md's "Minimum test
   set" to name the third tier and state that it is **not** part of the gate. The
   gate stays at five members / 60 launches.

**Exit criterion**, in its `spack`-mode form because this checkout has no build
tree and therefore no `ctest`: with an empty tier, after `spack install`,

- `beatnik_milestone_manifest.txt` is installed beside
  `beatnik_gate_manifest.txt` under `share/Beatnik/tests` and carries **zero**
  non-comment lines;
- `beatnik_gate_manifest.txt` still carries the regression tier's five targets on
  each backend in `BEATNIK_TEST_DEVICES` — **fifteen** lines on tuolumne, where
  that list is `SERIAL;OPENMP;HIP`, of which the runner's `_<BACKEND>` filter
  selects the ten that are the gate;
- `flux batch scripts/tuolumne/run_milestone.flux` exits **non-zero** with the
  "named no runnable tests" message (the guard at
  `run_regression_minset.flux:201-208`, ported);
- `flux batch scripts/tuolumne/run_regression_minset.flux` still reports PASS
  with exactly **sixty** `[gate] ===` launch lines — the check that the third
  tier took nothing out of the gate.

A `manual`-mode checkout checks the same two things with `ctest -N -L milestone`
(zero cases, no error) and `ctest -N -L regression` (exactly 60).

**Met.** After `spack install` in this `spack`-mode checkout:
`share/Beatnik/tests/beatnik_milestone_manifest.txt` is installed beside
`beatnik_gate_manifest.txt` and carries **zero** non-comment lines.
`beatnik_gate_manifest.txt` carries the expected **fifteen** non-comment lines
from the regression loop, which this task did not touch — its diff to
`tests/CMakeLists.txt` is pure addition — and the gate run below confirms the ten
the runner selects: 5 x 2 x 6 = 60 launches. `flux batch
scripts/tuolumne/run_milestone.flux` exited **1** (job `f3TRgG5aCNsq`) with
`[milestone] FAIL: the manifest named no runnable milestone tests for backends
'SERIAL HIP'`, the ported vacuous-pass guard, preceded by `no milestone binaries
for SERIAL` / `... for HIP`. `flux batch
scripts/tuolumne/run_regression_minset.flux` (job `f3TRgYcw5dfu`) reported
`[gate] PASS (label=regression)` with exactly **sixty** `[gate] ===` launch
lines. Both milestone-0 gold sets install: 81 numbered `.npz` plus the `gold/`
`README.md` under each of
`share/Beatnik/tests/regression_tests/milestone0-sub{3,4}-2000-steps/gold`, with
`checkpoint_latest.npz` absent from both, as the glob intends.

---

### M0-G1 — the level-3 2000-step gold set *(human step, no code)* — **DONE**

**Depends on:** none.

**Fill in:** `tests/regression_tests/milestone0-sub3-2000-steps/gold/*.npz` plus
that directory's `README.md`.

**Reference:** the command in `## Problem` at `<L> = 3`, and
[tests/regression_tests/direct-solve-10-steps/README.md](../tests/regression_tests/direct-solve-10-steps/README.md)
for the convention — the generating command recorded verbatim beside the data.
Expect ~4 minutes of single-core Python (Read this first #3).

**Do:**

1. Generate it. `--checkpoint-every-steps 25` gives 81 numbered files (steps 0,
   25, …, 2000) if the run completes, plus a `checkpoint_latest.npz` duplicating
   the last of them.
2. Record in `gold/README.md`, beside the `.npz` files, because M0-D1 and M0-A1
   reason about all of them and re-deriving them from 81 files is wasted work:
   the vertex and face counts
   (constant, `642`/`1280` — any change means adaptivity leaked in), the `time`
   series, the per-step minimum triangle quality, the enclosed-volume drift
   series `V/V0 - 1`, and **whether the run stopped early**
   (`stopping at step=… nonfinite …`) with the stop step if so.
3. Confirm the key set matches `initial_conditions/gold.npz`'s nine keys, so
   `compare_output.py`'s `FIELD_MAP` ([:111-135](../tests/regression_tests/compare_output.py#L111))
   needs no edit.

**Exit criterion:** 81 numbered `.npz` checkpoints present — fewer only if the
reference itself stopped early, in which case the stop step is in the
`README.md` and becomes the compare-depth ceiling. The `checkpoint_latest.npz`
beside them does not count toward the 81: it duplicates the last numbered step,
it is inert to `goldForStep` because it carries no `_step%07d.npz` suffix, and
M0-T1's install glob excludes it. Also: a self-compare of the **last** numbered
file against itself at
`--rtol 1e-12 --atol 1e-14` exits 0 reporting `642/642` unambiguous vertices (the
check that `--match-eps 1e-9` still resolves this mesh after 2000 steps of
roll-up, M0-R4); and `vertices.shape == (642, 3)`, `faces.shape == (1280, 3)` in
every file.

**Met.** 81 numbered checkpoints, `_step0000000` through `_step0002000` every
25, plus the `checkpoint_latest.npz` that does not count toward them (verified
bit-identical to the last numbered file, and excluded from M0-T1's install
glob). No early stop: step 2000 is present and every field is finite in all 81
files, so **M0-R2 did not fire**. `vertices (642, 3)` / `faces (1280, 3)` in
every file, and the same nine keys as `initial_conditions/gold.npz` in every
file, so `FIELD_MAP` needs no edit. The self-compare of
`checkpoint_t00001p998284_step0002000.npz` against itself at `--rtol 1e-12
--atol 1e-14` exits 0 with `matching (eps=1e-09): 642/642 unambiguous, ambiguous
cpp=0 gold=0` and `max|e| = 0` on every field — **M0-R4 has not bitten this
set**. The 81-row `step / time / min quality / V/V0 - 1` table is in
[the gold README](../tests/regression_tests/milestone0-sub3-2000-steps/gold/README.md);
headline numbers: final `time = 1.998284`, min quality falls from `9.749529e-01`
to a global minimum `3.826563e-02` at step 1700 and recovers to `6.303626e-02`
at step 2000, and volume drift reaches `+3.352894e-09` — above T2d's
`kVolumeDriftAbsCap = 1e-9`, as M0-R3 predicted.

---

### M0-G2 — the level-4 2000-step gold set *(human step, no code)* — **DONE**

**Depends on:** none. Independent of M0-G1 — generate them in parallel.

**Fill in:** `tests/regression_tests/milestone0-sub4-2000-steps/gold/*.npz` plus
that directory's `README.md`.

**Reference:** as M0-G1, at `<L> = 4`. Expect ~50 minutes of single-core Python.

**Do:** as M0-G1, with `2562`/`5120` as the constant counts. Additionally record
in the `README.md` how the minimum-quality series compares with M0-G1's — this
set is the one that resolves `eps`, and M0-A1 cannot choose a primary level
without knowing which frozen mesh survives the roll-up better.

**Exit criterion:** as M0-G1, with `2562/2562` unambiguous vertices in the
last-file self-compare and `(2562, 3)` / `(5120, 3)` shapes.

**Met.** 81 numbered checkpoints through `_step0002000`, `checkpoint_latest.npz`
bit-identical to the last of them and not counted, no early stop, all fields
finite, `vertices (2562, 3)` / `faces (5120, 3)` and the same nine keys in every
file. The last-file self-compare at `--rtol 1e-12 --atol 1e-14` exits 0 with
`2562/2562 unambiguous, ambiguous cpp=0 gold=0`. The 81-row table is in
[the gold README](../tests/regression_tests/milestone0-sub4-2000-steps/gold/README.md);
final `time = 1.964304`, final min quality `1.242421e-01` (which is also this
series' global minimum — still falling at step 2000), final volume drift
`+4.741414e-09`.

**The comparison M0-A1 needs: the frozen level-3 mesh degrades FASTER, not
slower.** At step 2000 the level-4 minimum quality is ~2x the level-3 one
(`1.24e-01` vs `6.30e-02`); level 3 first crosses `0.18` at step 1050 against
level 4's step 1800, and level 3 spends its last ~500 steps below `0.1`, which
level 4 never reaches. Neither **M0-R2** nor **M0-R6** fired at level 4. What
this settles for M0-A1 is under its own entry.

---

### M0-D1 — measure the divergence horizon, and attribute it — **DONE**

**Met.** The full matrix was measured in one `-q pdebug -t 1h` sweep (job
`f3TT4psJ8it7`, 1939 s of the 3600 s cap) with a new no-tier driver
[tests/regression_tests/Beatnik_Test_Milestone0Run.cpp](../tests/regression_tests/Beatnik_Test_Milestone0Run.cpp),
a new offline tabulator
[tests/regression_tests/milestone0_ladder.py](../tests/regression_tests/milestone0_ladder.py)
and
[scripts/tuolumne/milestone0_divergence.flux](../scripts/tuolumne/milestone0_divergence.flux);
all numbers are in `milestone0-progress-log.md` under `## M0-D1`. What was
verified: **step 0 matches at `1e-12` at levels 2, 3 and 4** with the same single
ulp of absolute error at every level, so **M0-R5 did not fire and the generators
do not diverge with subdivision depth**; the vertex and face counts held at every
one of the 81 checkpointed steps of all eight 2000-step runs, in both the
driver's own per-step integer check and the comparator's structural line, so
**adaptivity did not leak in**; and **at `--rtol 1e-10 --atol 1e-12` no
checkpointed step fails through 2000 in any of the twelve comparisons** — two
levels x two backends x two rank counts against Python, plus the four
rank-1-vs-rank-4 pairs — with that bound *proved* by the comparator's own printed
bounds rather than sampled. The `1e-12/1e-14` rung is where the ladder has
content: first failing step **1325-1350** at level 3 and **775-800** at level 4,
confirmed by real invocations, and level 4 therefore diverges *earlier* than level
3 even though M0-G1/M0-G2 found its mesh the healthier one. The attribution
needed a correction to mean anything — the reference's `.npz` carries no
`sheet_vector`, so a Beatnik-vs-Beatnik comparison is over a strictly larger field
set than a Beatnik-vs-Python one — and on the shared fields **Beatnik's own
decomposition dependence is 10-22x below the Python drift at level 3 and within
2.6x at level 4**, with SERIAL and HIP agreeing to within 1.5x, so within-rank GPU
nondeterminism adds essentially nothing to cross-rank summation order. Both codes'
volume-drift and minimum-quality series agree to `2.758331e-05` and
`6.643130e-12` relative respectively, worst case over every step and run; no run
stopped early, so **M0-R2 and M0-R6 did not fire on Beatnik's side either** and
every vertex paired unambiguously at every step, so **M0-R4 did not bite** the
cross-code pairing that M0-G1/M0-G2 left untested. Level-4 cost: `18.6` s (HIP,
1 rank) to `1289.4` s (SERIAL, 1 rank) of solve wall, peak resident set
`705000`-`1019096` kB per rank; GPU-side memory is out of scope. The measurement
recommends **not** taking M0-T2 — M0-A1 decides.

**Depends on:** M0-G1, M0-G2.

The task that decides whether the milestone's premise holds, and the one that
answers the determinism question with numbers. It writes **no library code** —
its only C++ is a test-source measurement driver, and every comparison it makes
is offline in Python.

**Fill in:** three artifacts, plus a table of measured numbers in
`milestone0-progress-log.md`.

1. `tests/regression_tests/Beatnik_Test_Milestone0Run.cpp` and its registration
   in [tests/CMakeLists.txt](../tests/CMakeLists.txt): a per-backend driver that
   builds the params, runs, and writes checkpoints, and compares nothing.
   Registered in **no tier** — no `LABELS`, no ctest case, no manifest line — so
   neither the gate nor `run_milestone.flux` can pick it up, and installed under
   `share/Beatnik/tests` so `beatnik_exe` resolves it by name.
2. `tests/regression_tests/milestone0_ladder.py`: the tabulation helper. It
   drives `compare_output.py` over a run directory and a gold directory (or two
   run directories) and emits the per-step, per-field table the progress log
   carries. A measurement tool, not a test — it is registered nowhere.
3. `scripts/tuolumne/milestone0_divergence.flux`: the batch script — never launch
   interactively from a login node, CLAUDE.md.

**Why not `examples/02_adaptive_mesh_bubble`.** It cannot produce the SERIAL half
of the matrix.
[examples/02_adaptive_mesh_bubble/adaptive_mesh_bubble.cpp:209](../examples/02_adaptive_mesh_bubble/adaptive_mesh_bubble.cpp#L209)
fixes the execution space to `Kokkos::DefaultExecutionSpace`, and the installed
binary is built from a `+rocm` spec, so `adaptive_mesh_bubble` is HIP-only — and
no runtime option changes that, both because the space is a compile-time choice
and because the CLI surface is closed (Conventions). Per-backend binaries exist
only for **test sources**, through the generated translation unit that pins
`BEATNIK_TEST_EXEC_SPACE`
([tests/CMakeLists.txt:437-495](../tests/CMakeLists.txt#L437) for the milestone
tier, [:323-386](../tests/CMakeLists.txt#L323) for the regression tier). Copy
that loop's shape for the driver and stop short of the point where it applies a
label and emits a manifest line.

**Reference:** [scripts/tuolumne/run_regression_minset.flux](../scripts/tuolumne/run_regression_minset.flux)
for the resolver-sourcing, `beatnik_exe` and scratch conventions;
[Beatnik_Test_DirectSolve10Steps.cpp:263-306](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp#L263)
for the `goldForStep` + `compare_output.py` subprocess pattern; `makeParams()`
([:308-378](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp#L308))
for the mapping from the Python command line to a `SolverParams`, which the
driver's params must match field for field except
`initial.icosphere_subdivisions` (the level under test), `time.steps = 2000` and
`checkpoint.every_steps = 25`.

**Environment and queue:** the **dev** spack env — do not set `BEATNIK_USE_PROD`
— submitted `-q pdebug -t 1h`. The reference's own single-core Python produced
both gold sets in about 110 minutes total, and an MPI/HIP port slower than that
would defeat the purpose of the port, so one hour is the budget for the whole
sweep. A sweep that does not fit it is a **performance finding for the progress
log**, not a reason to move queues or lengthen the walltime.

**Do, in this order:**

1. **Step 0 first, and treat it as a gate.** For each level, compare Beatnik's
   step-0 checkpoint against the gold at `--rtol 1e-12 --atol 1e-14`: level 2
   against [tests/regression_tests/initial_conditions/gold.npz](../tests/regression_tests/initial_conditions/gold.npz)
   (step 0 at level 2, `162`/`320`, carrying the same nine keys as the two
   milestone-0 sets), levels 3 and 4 against the `_step0000000.npz` of their own
   gold directories. A failure here is a *generator* disagreement at that
   subdivision level (Read this first #3), not a divergence measurement, and must
   be reported as such rather than absorbed into the trend. Record `max|e|` on
   `vertices` at levels 2, 3 and 4 so the ulp growth with level is a number.
2. Run both levels for 2000 steps, checkpointing every 25, at ranks 1 and 4 on
   SERIAL and HIP, into `BEATNIK_TEST_SCRATCH`.
3. **Build the tolerance ladder by derivation, then confirm it.** For each
   checkpointed step run `compare_output.py` **once**, at `--rtol 1e-12 --atol
   1e-14` and not `--quiet`, and harvest the per-field `max|e|` and `max|e|/|g|`
   it prints
   ([compare_output.py:445-448](../tests/regression_tests/compare_output.py#L445)).
   Those two numbers give a candidate first-failing step at each of
   `--rtol 1e-12/1e-10/1e-8/1e-6/1e-4` (`--atol` two decades below) without
   re-running the comparator 5× per step. The candidate is **not** the answer:
   the pass criterion is elementwise `|e| <= atol + rtol*|g|`
   ([:442](../tests/regression_tests/compare_output.py#L442)) and the worst-`|e|`
   entry need not be the worst-relative one, so each candidate must be confirmed
   by two real invocations at that tolerance — the candidate step, which must
   fail, and the previous checkpointed step, which must pass. Record the
   confirmed step, not the derived one. This ladder is what M0-A1 consumes.
4. **Attribute the divergence.** Compare Beatnik at ranks 1 against Beatnik at
   ranks 4, same backend, step by step, at the same ladder. If the
   Beatnik-vs-Beatnik divergence is comparable to the Beatnik-vs-Python
   divergence, the horizon is set by Beatnik's own decomposition dependence
   (Read this first #2) and M0-T2 is worth doing; if it is orders of magnitude
   smaller, the horizon is Python-vs-Beatnik drift and M0-T2 buys nothing.
   **This comparison is the single most valuable number this task produces** —
   record it as such, per level and per backend. **SERIAL is the clean
   instrument** here: on HIP a rank-1-vs-rank-4 difference conflates the
   decomposition order M0-T2 would fix with within-rank GPU nondeterminism it
   would not, so the HIP number bounds the effect and the SERIAL number isolates
   it.
5. Record the volume-drift series `V/V0 - 1` per step for both codes at both
   levels, computed the same way T2d's was
   (`V = (1/6) Σ_f a·(b×c)` over `faces`, offline in NumPy from the `.npz` and
   the `.h5`), and the minimum-quality series. These are what M0-T3 asserts and
   what distinguishes M0-R2 from M0-R3.
6. Record the wall time per step and the peak resident memory per rank at level 4
   — what tells M0-A1 whether the asserted depth is affordable at both rank
   counts on both backends. Neither is instrumented: nothing in `src/` or in any
   driver reports a timing or a high-water mark, so the batch script supplies
   both. Peak resident memory per rank comes from `/usr/bin/time -v` wrapped
   around the task *inside* `flux run`, so each rank reports its own; wall time
   comes from the script clocking each launch, with the per-step cost derived
   from it and the step count. **GPU-side memory is out of scope** — there is no
   mechanism for it here, and saying so is what stops a later session reading its
   absence as an oversight.

**Exit criterion:** the progress log carries, as 17-digit literals: the
per-tolerance first-failing step — **derived and then confirmed, per step 3** —
for each of (level 3, level 4) × (SERIAL, HIP) × (ranks 1, 4) against Python; the
same ladder for Beatnik-rank-1 vs Beatnik-rank-4; the `max|e|` growth series for
`vertices` and `potential`; both volume-drift series; and the level-4 wall time
and peak resident memory per rank. The measurement **fails** if step 0 does not
match at `1e-12` at any level (step 1 above), or if the vertex or face count
changes at any step in any run — that would mean adaptivity leaked into the
configuration that exists to exclude it.

---

### M0-A1 — fix the compare depth, the tolerance ladder, and the determinism decision — **NOT STARTED**

**Depends on:** M0-D1.

Not a coding task: the three decisions that must be *recorded* rather than left
implicit in a test's literals.

**Do:** bring together M0-D1's ladder, its rank-1-vs-rank-4 attribution, both
gold sets' `README.md` series, and the level-4 cost numbers. Decide, with the
user:

1. **The compare depth and the tolerance at each compared step** — flat if the
   measurement allows it, otherwise a per-step table. State the depth for each
   level separately; there is no reason they should be equal.
2. **Which level is the primary member** and whether the other is a second
   member or a recorded measurement only.
3. **Whether steps beyond the depth are compared structurally/statistically**
   instead of field-by-field (counts, `time` series, volume drift, minimum
   quality, valence histogram) or not at all. Beyond-depth structural comparison
   is nearly free here — connectivity is frozen — so "not at all" needs a reason.
4. **Whether M0-T2 is taken.** The criterion is M0-D1 step 4: if
   Beatnik-vs-Beatnik divergence is a significant fraction of
   Beatnik-vs-Python divergence, determinism raises the horizon and is worth its
   cost; otherwise it is not. Record the number the decision rests on, and if
   M0-T2 is declined, mark it **DECLINED** in this document with that number
   beside it rather than deleting it.

**Additional information needed, and which task answers it:** how fast the two
codes diverge and at what tolerance (**M0-D1** step 3); how much of that is
Beatnik's own decomposition dependence (**M0-D1** step 4); and whether a
2000-step level-4 run is affordable at both rank counts on both backends
(**M0-D1** step 6).

The fourth question is **answered**: the frozen **level-3** mesh is the one that
degrades faster. Final minimum quality is `6.303626e-02` at level 3 against
`1.242421e-01` at level 4, and level 3 spends its last ~500 steps below `0.1`
where level 4 never reaches it (M0-G1, M0-G2). So decision 2 cannot pick level 3
as primary on mesh-health grounds; if it picks level 3, the reason is cost.

**Exit criterion:** this document's `## Problem`, M0-T2 and M0-T3 entries state
the decided depth, tolerances, primary level, beyond-depth treatment and the
M0-T2 verdict **as numbers**; the progress log records each alternative that was
rejected and the measurement that rejected it.

---

### M0-T2 — a decomposition-independent trajectory — **NOT STARTED (CONDITIONAL)**

**Depends on:** M0-D1, M0-A1. **Do not start this task until M0-A1 has decided to
take it** — the user's instruction is measure first, then decide, and the cost is
real.

**Fill in:** `BRSolverDirect::ringAccumulate` and its two callers
([src/Beatnik_BRSolverDirect.hpp:240-300](../src/Beatnik_BRSolverDirect.hpp#L240)),
and the three `MPI_SUM` reductions named in Read this first #2:
[src/Beatnik_VolumeProjection.hpp:150](../src/Beatnik_VolumeProjection.hpp#L150)
and [:257](../src/Beatnik_VolumeProjection.hpp#L257),
[src/Beatnik_SurfaceState.hpp:388](../src/Beatnik_SurfaceState.hpp#L388) and
[:409](../src/Beatnik_SurfaceState.hpp#L409),
[src/Beatnik_ZModelSolver.hpp:634](../src/Beatnik_ZModelSolver.hpp#L634) and
[:652](../src/Beatnik_ZModelSolver.hpp#L652).

**Reference:** `doc/PARALLELIZATION.tex` §"What must NOT change" item 3 (the
requirement) and §"The BR sum is the ideal target" (the recommended
decomposition: *"a row block of targets per rank, with the full source set
replicated"* — sources are ~`7N` doubles, "replicate it and stop worrying"), and
§"What must NOT change" item 5 (the volume projection is genuinely global and
must stay a global reduction).

**Callers of the signatures this changes:** `ringAccumulate` is `private` with
exactly two call sites, both in `BRSolverDirect` — `computeInterfaceVelocity`
([:274](../src/Beatnik_BRSolverDirect.hpp#L274)) and
`computeSurfaceRieszScalar`. `Comm::allReduceSum`'s signature does not change;
only its use at the five sites above does. Enumerate them again by grep before
starting: this document's list is the state at the time it was written and
M0-T2 is a late task.

**Do:**

1. Replace the rank ring with an ordered gather: every rank holds the full source
   list **in ascending gid order**, and every target sums over it **sequentially
   in that order**. Sequential per-target accumulation, not a nested team
   reduction — a tree reduction's shape depends on the source count and the
   backend, so it is deterministic run to run but not across either. Parallelism
   comes from the target loop, which is what the reference recommends and what
   there is plenty of.
2. State the cost on the declaration: one `Allgatherv` of `7N` doubles per RK
   stage and `O(N)` per-rank storage, in exchange for a trajectory that does not
   depend on the rank count. This replaces the ring's `O(N/P)` storage, so say
   so — the ring's doc comment argues for itself and would otherwise be left
   contradicting the code.
3. Make the three global sums reproducible. A plain `MPI_Allreduce(MPI_SUM)` of
   per-rank partials is decomposition-dependent because the partials are; the
   two admissible routes are a fixed-order sum over per-entity contributions
   gathered in gid order (simple, `O(N)` communication, fine at milestone-0
   sizes) and a binned/pre-rounded reproducible summation (decomposition-
   independent without a gather, more code). **Choose one, record why, and state
   the scaling consequence** — this is a decision, not an implementation detail,
   and the design does not make it for you.
4. Add nothing to the CLI (conventions table). The change is unconditional.
5. Do **not** touch `chooseStepSize`, `allFinite` or the diagnostics' min/max
   reductions: `MPI_MIN`/`MPI_MAX` are order-independent already (Read this
   first #2), and rewriting them would be a drive-by refactor.

**Exit criterion:** a `unit`-tier member run at ranks 1, 2, 3, 4, 5, 6 on both
backends shows a **bitwise identical** trajectory across all six rank counts for
20 steps at level 3 — every vertex position, the potential, and `time`, compared
as raw bits and not at a tolerance — and the same at level 4 for 5 steps. Failure
direction: the same test built against the pre-M0-T2 ring **fails** at ranks 1 vs
4, reported as a bit difference with its magnitude, so the test is demonstrably
sensitive to the thing it claims to pin. Cross-*backend* bitwise identity is
**not** claimed and must not be asserted — `sqrt`, `pow` and the fused-multiply
form differ between SERIAL and HIP — and the progress log must record the
measured SERIAL-vs-HIP deviation so the next reader does not mistake the omission
for an oversight.

---

### M0-T3 — the milestone-0 comparison test — **NOT STARTED**

**Depends on:** M0-T1, M0-G1, M0-G2, M0-A1, and M0-T2 if M0-A1 took it.

**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp`,
registered in the `milestone` tier with the gold **directory** and the comparator
as its arguments — one registration per level, or one target parameterized by its
argument list, whichever keeps `_beatnik_args_<stem>_abs`/`_rel` honest.

**Reference:** [tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp)
end to end. It is this test at 10 steps and level 2, and most of it transfers
verbatim: `makeParams()` (`:308-378`), `goldForStep()` (`:263-286`),
`runComparator()` (`:289-306`), the three R9 partition discriminators (`:92-104`
for what they are and why the third one stops working once positions evolve —
note the polyhedral-deficit literal at `:180` is **level-2 specific** and must be
re-derived per level), and the per-step `time` assertion against 17-digit
literals.

**Do:**

1. `makeParams()` is the milestone-0 command line: T2a's params with
   `icosphere_subdivisions` from the level under test, `time.steps` from M0-A1's
   decided depth, and `checkpoint.every_steps = 25`.
2. Compare every checkpointed step up to the decided depth at the decided
   tolerances. Where the ladder is not flat, put the per-step tolerance in a
   compiled-in table **with the measured `max|e|` beside it as a comment**, so a
   later session can see the headroom rather than re-measure it.
3. Replace T2d's volume-drift literals with this configuration's own measured
   series from M0-D1 step 5, at a stated relative tolerance, and re-derive the
   absolute blow-up cap for a 2000-step drift (deliberate deviations). Do not
   import `kGoldVolumeDrift` or `kVolumeDriftAbsCap`.
4. Assert the entity counts are **constant and equal to the generator's** at
   every compared step, with an `MPI_Allreduce` over owned counts rather than a
   number read from Tessera — the check that adaptivity did not leak in, and the
   thing that makes the frozen-connectivity premise a test rather than a claim.
5. If M0-A1 chose beyond-depth statistical comparison, assert those quantities
   for the remaining steps: counts, `time`, volume drift, minimum quality.
6. Report the wall time and the peak memory in the test's output; they are what
   tells the next session whether a deeper depth is affordable.

**Exit criterion:** `flux batch scripts/tuolumne/run_milestone.flux` reports PASS
at ranks 1 and 4 on SERIAL and HIP with zero `[FAIL]` lines, for both levels, and
`flux batch scripts/tuolumne/run_regression_minset.flux` still reports PASS at 60
launches with `ctest -N -L regression` listing 60 cases. Failure direction, both
of which must be demonstrated: the same test invoked against the **step-0** gold
for a later step exits **exactly 1** from the comparator — a detected mismatch,
not a load error — and a build with `--dynamic-remesh` forced fails on the
constant-entity-count assertion of step 4 rather than passing against a different
mesh.

## Known risks

**M0-R1 — the divergence horizon is shorter than the milestone's premise.** The
two codes stop agreeing at `1e-10` well before the mesh or the physics gives out.
*Presents as:* M0-D1 reporting a first-failing step in the tens or low hundreds
on a run with no adaptivity at all. *Do:* this is information, not a bug — take it
to M0-A1, and use M0-D1 step 4 to say whether M0-T2 would move it. Do **not**
loosen `--rtol` to make a step pass without recording the measured `max|e|`
beside it; an unrecorded loosening is how a comparison stops being a test.

**M0-R2 — the frozen mesh gives out before the codes disagree.** With no collapse
and no split, the roll-up stretches triangles until the cotangent Laplacian
acquires negative weights and the viscosity term destabilizes. Both codes suffer
it identically, so the *comparison* stays valid, but the run stops. *Presents
as:* the Python printing `stopping at step=… nonfinite …` during M0-G1/M0-G2, or
the minimum-quality series collapsing toward zero. *Distinguished from M0-R1 by:*
the minimum-quality series, which M0-G1 and M0-G2 both record for exactly this
reason — a quality collapse is M0-R2, a healthy quality series with growing field
errors is M0-R1. *Do:* the stop step becomes the compare-depth ceiling and is
recorded in the gold `README.md`; it is not a failure of either code. **It did
not fire in either gold set** — neither run stopped early and every field is
finite in all 162 files — so the ceiling is 2000 steps at both levels on the
reference side. **M0-D1 measured Beatnik's side and it did not fire there
either:** no run stopped early, and the minimum-quality series tracks the
reference's to `6.643130e-12` relative at every step of all eight runs.

**M0-R3 — the volume drift is not the T2d series and looks like a regression.**
No `projectToVolume` runs here, so drift accumulates linearly, and **M0-G1 and
M0-G2 confirm it**: at step 2000 the reference's drift is several times T2d's
`kVolumeDriftAbsCap = 1e-9` (Deliberate deviations carries the two numbers and
the re-derived cap). *Presents as:* the
new test failing on volume drift at a few hundred steps if T2d's literals were
reused. *Distinguished from a real conservation bug by:* whether Beatnik's series
tracks the *Python's own* series, which is what M0-D1 step 5 measures for both
codes. *Do:* measure both, assert agreement with the reference's series, keep an
absolute cap re-derived for 2000 steps as the blow-up detector.

**M0-R4 — the quantized vertex pairing degrades silently.**
`compare_output.py` sorts each file's vertices *independently* at
`--match-eps 1e-9` ([:310-343](../tests/regression_tests/compare_output.py#L310)).
Two hazards compound at milestone 0: a frozen mesh cannot collapse, so roll-up
can crowd vertices together, and higher subdivision levels start them closer.
*Presents as:* huge, uniform `max|e|` across all fields at once, often with
`ambiguous cpp=0 gold=0`. *Distinguished from a real field error by:* `vertices`
failing first and by the same magnitude as every other field. *Do:* M0-G1 and
M0-G2 both check the last file's self-compare for unambiguous pairing, which
catches it in the gold set rather than in the test; report it as a pairing
failure with the step index, and do not raise `--match-eps` or `--max-ambiguous`
without stating what that then cannot detect. **It has not bitten the gold
sets**: both last-file self-compares pair every vertex unambiguously (`642/642`,
`2562/2562`, `ambiguous cpp=0 gold=0`). **M0-D1 tested cross-code pairing too,
where the two files' positions differ, and it holds**: every step of all twelve
comparisons reports `ambiguous cpp=0 gold=0` at the default `--match-eps 1e-9`,
including step 2000 at level 4.

**M0-R5 — the icosphere generators disagree more at higher subdivision.** The
midpoint-normalize ulp differences of Read this first #3 compound across levels,
and level 4 is two levels further than anything ever tested. *Presents as:* step 0
failing at `1e-12` at level 3 or 4 while passing at level 2. *Do:* M0-D1 step 1
makes this a gate before any trajectory number is recorded, and records `max|e|`
per level so the growth is a number rather than a suspicion. If it exceeds
`1e-12`, the fix is a stated step-0 tolerance per level with the measurement
beside it — not a looser tolerance everywhere.

**M0-R6 — `dt0 = 0.003` is not stable on a finer mesh.** The reference's adaptive
dt is relative to the run's *own* initial minimum edge (Read this first #2), so a
level-4 run starts at the same `0.003` as a level-2 one and the formula offers no
protection against a finer mesh needing less. *Presents as:* both codes going
nonfinite early at level 4 while level 3 completes — and, because both codes
share the formula, as a *matching* pair of early stops rather than a
disagreement. *Do:* if M0-G2 stops early where M0-G1 does not, the level-4 set is
a `--dt` decision for M0-A1, not a Beatnik bug. Changing `--dt` means
regenerating that gold set, so decide before generating a third one. **It did not
fire on the reference side:** M0-G2 completed all 2000 steps, so `--dt` is not a
decision M0-A1 has to make unless Beatnik itself goes nonfinite at level 4 where
the reference did not.

**M0-R7 — M0-T2's cost is paid on the production path.** The framework forbids a
Beatnik-only CLI switch, so a deterministic BR sum is unconditional: an
`Allgatherv` of `7N` doubles per RK stage and `O(N)` per-rank source storage,
replacing the ring's `O(N/P)`. *Presents as:* no failure at all — a milestone-0
run that passes and a future FMM-scale run that does not fit in memory.
*Do:* M0-T2 step 2 requires the cost on the declaration, and M0-A1 must record
the number that justified taking the task. The scaling path is the FMM (T3a),
which has its own error structure and is not bit-compared, so the two do not
conflict — but say that in the log rather than leaving a reader to work it out.

**M0-R8 — a milestone test that passes because the run was quietly truncated.**
The generic form of R15 in this milestone's shape: the asserted depth is a number
in a header, and lowering it makes a red test green with no signal anywhere.
*Presents as:* it does not. *Distinguished by:* M0-A1's exit criterion, which
requires the depth to appear as a number in this document *and* in the test, and
by the progress log entry recording what it was and why. Any later change to the
depth is a new M0-A entry with its own measurement, never an edit to a literal.
