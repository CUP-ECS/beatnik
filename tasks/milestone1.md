# Milestone 1 — the reference's default rising-bubble configuration

**Status:** IN PROGRESS — M1-U1, M1-U2, M1-U3, M1-G1, M1-G2, M1-D1, T4d1,
T4d2, T4d3, T4d4, T4d5, T4d6j, T4d6, M1-T2j, M1-T2, M1-D2 and M1-G3j are
**DONE** and reviewed, Passed. The codes part at step 2, where Tessera's one-round
collapse independent set accepts 8 of the 44 collapse candidates the reference's
serial walk accepts 31 of, and Beatnik's mesh obeys the reference's rules at every
probed step after (log `## M1-D2`). M1-G3 is **DONE**: the reference job
`f3iZLsmfWYw1` stopped at step 1625, and the gold set is steps 0–1610, 162
checkpoints ending at `time` 1.238122, recorded in
`tests/regression_tests/milestone1-bubble-1610/README.md` (Read this first #1).
M1-A1j is **DONE**: the test asserts step 0 field-wise and time-matched
statistics through step 1610 (code `f3a6c80`). Its confirming job
`f3j79G2VDX4T` ran clean (no defect, 1610/1610 steps, every liveness counter
non-zero, both failure directions pass) but **missed two bands** identically on
all four compare launches: total area from gold step 1060 (worst `1.063e-2`
against `1e-3`, step 1610) and rise from gold step 1340 (worst `2.545e-3`
against `1e-3`, step 1610); log `## M1-A1`. The user ruled to keep every band
as decided and diagnose the drift first. **M1-D3** is **DONE**. The drift is
a discretisation gap, not a defect. The reference's own solver, restarted from
Beatnik's step-10 mesh, misses the gold by the same amount (area `1.047e-2`,
rise `2.808e-3` at step 1610). Step size, the gold's late behaviour and face
count are ruled out (log `## M1-D3`). M1-A1 (NOT STARTED) awaits the user's
band choice against those measured values.

## Problem

Milestone 1 is the first configuration in which Beatnik runs the **reference's own
documented working default** — README configuration (a) of
`~/research-bridges/zmodel-steve/zmodel3d-amr` — and is compared against it
checkpoint by checkpoint. The target command is

```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --dynamic-remesh --isotropic-cleanup \
  --source-quadrature vertex
```

with **every unlisted option at its `parse_args` default**, and its Beatnik
equivalent through `examples/02_adaptive_mesh_bubble`. Acceptance compares the
Python `.npz` checkpoints with Beatnik's run, as decided at M1-A1j:

- **Field-wise depth: step 0**, Beatnik's `.h5` against the M1-G2 `.npz` by
  `tests/regression_tests/compare_output.py` at its defaults `--rtol 1e-10
  --atol 1e-12` (measured `max|e|` `5.55e-17` on `vertices`). From step 1 the
  meshes differ by edit selection, not floating point (log `## M1-D2`).
- **Beyond step 0, mesh statistics matched on physical time**: Beatnik's
  statistics, computed every step and linearly interpolated to each gold
  checkpoint's `time`, at every M1-G3 checkpoint from step 10 to 1610 (161
  checkpoints, `--steps 1610`):
  - rise (area-weighted centroid `z`): `|Beatnik − gold| ≤ 1e-3` absolute;
  - total area: `|Beatnik/gold − 1| ≤ 1e-3`;
  - Beatnik's `|volume drift| ≤ 1e-12`;
  - face count: `|F_Beatnik/F_gold − 1| ≤ 0.10`;
  - Beatnik's minimum triangle quality `≥ 0.18`.

  Valence 5/6/7 counts and minimum and mean edge length are printed, not
  asserted. A gold checkpoint later than Beatnik's final `time` is a failure,
  not a skip.

What exists now is a Beatnik that **rejects this command at setup**, by name and
by task ID, from `Solver::requireSupportedConfiguration`
([src/Beatnik_Solver.hpp:759-892](../src/Beatnik_Solver.hpp#L759-L892)). Four
rejections fire and every one of them is **T4d**:

| Option, at its default | Rejection | Needs |
| --- | --- | --- |
| `--isotropic-cleanup` (default **on**) | [:791-797](../src/Beatnik_Solver.hpp#L791-L797) | `MeshQuality::isotropicCleanup` |
| `--remesh-collapse-factor 0.45` | [:830-850](../src/Beatnik_Solver.hpp#L830-L850) | `DynamicRemesh::collapseShortEdges` |
| `--remesh-smooth-iters 1` | [:852-859](../src/Beatnik_Solver.hpp#L852-L859) | `DynamicRemesh::tangentialSmooth` |
| `--remesh-flip-min-gain 1e-3` | [:861-870](../src/Beatnik_Solver.hpp#L861-L870) | `DynamicRemesh::flipEdgesForQuality` |

Nothing else on the command line is stubbed. The end state is: the command runs,
and the depth, tolerances and statistics bands above are asserted by
`Beatnik_Test_Milestone1Bubble` in the `milestone` tier.

**Out of scope**, because every option that reaches them is at a default that
switches them off — this is the whole list, and it is what makes the milestone
T4d and nothing else:

| Task | Kept out of scope by |
| --- | --- |
| T3a, T3b (Canopy FMM) | `--br-approximation direct` |
| T4e (nonlocal proximity) | `--remesh-proximity` / `--remesh-surgical-proximity` off (`dynamic_remesh.py:33,41`) |
| T4f (tight remesh profile) | `--remesh-tight-after -1.0` |
| T5a (shape deformation, initial vorticity) | `--initial-shape sphere`, `--polar-amp 0.0`, `--initial-potential-strength 0.0`, `--mesh-kind icosphere` |
| T5b (restart) | `--restart-from ""` |
| T5c (field filter, redistribution, sheet-vector model) | `--field-filter-every 0`, `--redistribute-every 0`, `--state-model potential` |
| T5d (load balancing) | never on a reference path |
| T5e, T5f (the three inert dt controls) | `--max-sheet-dt-product 0.0`, `--dt-switch-time -1.0`, `--t-end None` |
| The indicator-driven AMR path (T4a) and `--flip-passes`, `--smooth-iters`, `--max-faces` | `--dynamic-remesh` makes the two adaptivity branches mutually exclusive (`run_adaptive_mesh_bubble.py:1424` vs `:1469`); `--flip-passes 0` |

Also out of scope: plotting, video and the plane-section diagnostic, per
framework.md's standing deviations.

## Read this first

Six facts about the reference, the comparator and Tessera that every task below
builds on.

### 1. The target command runs 140 steps and writes no checkpoints

`--steps` defaults to **140**, `--checkpoint-dir` to `""` and
`--checkpoint-every-steps` to **0** (`run_adaptive_mesh_bubble.py:133,316,329`),
and both checkpoint criteria are guarded on the directory being set (`:1571`,
`:1573`). The command as written therefore produces no `.npz` at all, and cannot
reach step 500.

**Resolved with the user:** the milestone gold set is a **600-step** run
checkpointing **every 10 steps** — see M1-G2 for the exact command — and a
long run of the same command at the same cadence (M1-G3j, `--steps 5000`),
whose statistics the milestone test asserts. The reference itself stops at
step 1625 (`t = 1.254`, nonfinite mesh-RHS state) after its vertex count jumps
from 610 at step 1610 to 1993 at step 1624, so that gold set is **steps 0–1610
(162 checkpoints, final `time` 1.238122)**; the step-1620 and step-1624 files
are excluded.

### 2. Steps 500 / 1000 / 2000 / 5000 are not a criterion this port can be assumed to meet

`compare_output.py` is **structural before it is numeric**. It fails outright on a
differing vertex or face count ([:555-566](../tests/regression_tests/compare_output.py#L555-L566)),
it pairs vertices by an *independent* quantized lexicographic sort of each file at
`--match-eps 1e-9` ([:310-343](../tests/regression_tests/compare_output.py#L310-L343)),
and it requires the remapped, canonicalized face lists to be **equal**
([:657-675](../tests/regression_tests/compare_output.py#L657-L675)). Its default
tolerances are `--rtol 1e-10 --atol 1e-12`
([:695-706](../tests/regression_tests/compare_output.py#L695-L706)).

Three recorded facts stand against that at 500 steps of a *remeshing* run:

- **R7 (framework.md).** The reference's edit sets are serial and
  order-dependent: `collapse_short_edges` walks candidates in ascending
  normalized length and each accepted collapse changes which later ones are safe
  (`dynamic_remesh.py:361-405`); `flip_edges_for_quality` sweeps one edge map with
  a `touched` *face* set (`:408-457`); `_valence_equalizing_flips` sweeps with a
  `touched` *vertex* set (`mesh_quality.py:44-87`). Tessera's operations accept a
  deterministic **independent set** instead, which is a different set by
  construction, and Tessera's own header says consumers must compare face count,
  quality distribution and edge-length histogram — never edit sets.
- **R13.** `Tessera::splitEdges` cuts a two-split-edge quad on the shorter
  diagonal; `mesh.py::refine_marked_faces` uses a fixed one. Measured at T4a: the
  two codes' face counts diverged by pass 2 (788 vs 796). Measured at T4b: they
  did *not* diverge over 20 steps. In this milestone's configuration it acts at
  step 1: the reference's fixed diagonal is the longer one in 41 of 82
  two-split quads, which its flip sweep then undoes (M1-D2).
- **R2.** Cross-rank reduction order gives ulp-level differences in the reduced
  scalars, and a rolling-up vortex sheet amplifies them.

The deepest **field** agreement ever demonstrated in this port is **10 steps** at
`--rtol 1e-10` (T2d), on a fixed-connectivity mesh.

**Resolved with the user: measure first, then decide.** M1-G1/M1-D1 measure the
divergence horizon of the *fixed-connectivity* configuration — the best case,
isolating R2 from R7 and R13 — before any T4d work is spent. M1-A1j then sets the
milestone's compare depth and tolerance ladder from the measured numbers, with
the 600-step gold set in hand. **This document does not assert that step 500,
1000, 2000 or 5000 is reachable at `1e-10`; it makes finding out a task.**

### 3. `--max-faces` does not bound a `--dynamic-remesh` run

`max_faces` is a parameter of the *indicator-driven* refiner only: it is absent
from `DynamicRemeshParams` (`run_adaptive_mesh_bubble.py:1326-1357` constructs it
without one) and in Beatnik it is read only by `AdaptiveMesh`
([src/Beatnik_AdaptiveMesh.hpp:1029-1032](../src/Beatnik_AdaptiveMesh.hpp#L1029-L1032)).
So T4b's plateau at 1400 faces was the sizing field reaching equilibrium, **not**
a cap, and a 600-step milestone run has no face-count ceiling other than
`--remesh-h-min 0.0015` and `--remesh-max-splits 300` per pass. R4's greedy-cap
divergence is therefore *not* in play on this path, and the cost of the
`O(N^2)` direct Birkhoff-Rott sum is (M1-D1 measures the trajectory).

### 4. T4d1–T4d6 have moved here, and keep their IDs

framework.md carried the T4d design; it now points here. **The IDs are
unchanged** — `T4d`, `T4d1`…`T4d6` — because four `requireSupportedConfiguration`
messages, several header comments and
[src/Beatnik_DynamicRemesh.hpp:1020-1047](../src/Beatnik_DynamicRemesh.hpp#L1020-L1047)
all name "task T4d" in text a user will read at a failed run. Renaming them to
`M1-*` would make every one of those strings wrong. Tasks that are new with this
milestone carry an `M1-` prefix.

Everything framework.md says about **Tessera's** collapse/flip/compact contract
still holds and is not restated here beyond what each task needs; the three
Tessera headers — `Tessera_EdgeCollapse.hpp`, `Tessera_EdgeFlip.hpp`,
`Tessera_Compact.hpp` — are the design documents for those operations and must be
read before writing against them.

### 5. Three reference details the earlier T4d text stated imprecisely

Each changes an implementation, not just a comment:

- **The collapse priority is a *normalized* length, not a length.** Candidates are
  ranked by `length / max(local_target, 1e-300)` with
  `local_target = min(target[i], target[j])`, and the per-pass cap truncates
  *that* ordering (`dynamic_remesh.py:368-378`). Tessera's priority is
  `(squared length, EdgeKey)`. Beatnik's mask must be built and truncated on the
  reference's key; Tessera then orders within the mask by its own. Recorded as
  M1-R3.
- **`flip_edges_for_quality` does not rebuild its edge map.** It iterates one
  snapshot of `edges_from_faces` and skips any edge whose incident faces are
  already `touched` (`:415-421`). R7's claim that it "rebuilds the edge map after
  each accepted flip" describes `mesh_solver.py::improve_mesh_connectivity_by_edge_flips`
  (the `--flip-passes` entry point, off in this configuration), not this one.
  The pass is still order-dependent, so R7's conclusion is unaffected.
- **`tangential_smooth_vertices` carries an all-or-nothing quality guard that
  `MeshQuality::relaxTangential` does not.** Each iteration builds a *trial*
  position set, recomputes the global minimum triangle quality, and **discards
  the whole iteration and stops** if
  `new + 1e-14 < max(min_quality, 0.85 * old)` (`:459-490`). It also skips
  vertices with fewer than three neighbours. So T4d4 is not "call
  `relaxTangential` with different parameters" — it is that kernel plus a global
  reduction and an accept/reject decision, and the returned `smooth_steps` counts
  *accepted* iterations.

### 6. Tessera's admissibility tests are not the reference's

Tessera's collapse and flip each run a geometric test of their own, configured by
a policy struct. Neither test is the reference's at its default settings, and one
of them cannot be made the reference's without an upstream change. Tessera line
references in this section are against `33dc6b9` (M1-U3):

- **The quality metric differs.** The reference's `triangle_quality` is
  `4√3·A / Σℓ²` (`dynamic_remesh.py:626-636`), which Beatnik already has as
  `SurfaceOperators::triangleQuality`
  ([src/Beatnik_MeshGeometry.hpp:305](../src/Beatnik_MeshGeometry.hpp#L305)).
  Tessera's `minQuality` thresholds the policy's `quality()`, which by default
  is the **radius ratio** `2r/R`
  (`detail::flipRadiusRatio`, `Tessera_EdgeFlip.hpp:340-366`), which Beatnik has
  as `SurfaceOperators::radiusRatioStats` (`:384`) and reports only as R12's
  diagnostic. The two are not monotone in one another, so no threshold on one
  reproduces a threshold on the other.
- **For collapse this has to be fixed in Tessera.** The reference rejects a
  collapse if any surviving incident face's quality falls below `min_quality`
  (`dynamic_remesh.py:519-549`). That test needs the full one-rings of both
  endpoints, which only Tessera's vertex coordinator assembles
  (`Tessera_EdgeCollapse.hpp:230-236`); the rank that owns an edge may own neither
  endpoint, so Beatnik cannot evaluate it while building the mask. Tessera's
  policies carry a `quality( p0, p1, p2 )` hook (**M1-U1**, **DONE**) that
  `minQuality` thresholds, so Beatnik passes the reference metric through it.
- **The collapse fold guard maps exactly.** The reference accepts iff every
  surviving incident face has `n_old·n_new > 0.1` on unit normals
  (`dynamic_remesh.py:546-549`). Tessera rejects iff
  `angle(n_old, n_new) > maxNormalRotation` over the same faces, with the two
  dying faces skipped (`Tessera_EdgeCollapse.hpp:991-995`). So
  `maxNormalRotation = acos(0.1) = 1.4706289056333368` reproduces it; the two
  differ only at `n_old·n_new == 0.1` exactly, which the reference rejects and
  Tessera accepts.
- **The flip tests need only the quad, so they belong in the mask.**
  `DefaultFlipPolicy` rejects a new face whose normal deviates more than
  0.35 rad from the area-weighted mean of the two old normals, or whose radius
  ratio is below 0.05 (`Tessera_EdgeFlip.hpp:387-406`, `:744-777`). The
  reference's quality flips have **no** normal test (`dynamic_remesh.py:408-457`);
  its valence flips require `dot ≥ 0.2` against each face's *own* pre-flip normal
  (`mesh_quality.py:78`). The defaults would therefore reject flips the
  reference accepts. Both flip call sites pass
  `{maxNormalDeviation = π, minQuality = 0}`, which disables both tests (a
  degenerate normal is still rejected), and put the reference's tests in the
  mask.
- **Tessera checks that the new diagonal does not already exist, exactly.** A
  flip whose diagonal `(c,d)` is an existing edge is rejected and counted in
  `rejectedDuplicateEdge`. The answer comes from the `(c,d)` coordinator,
  which has seen every owned face's edges (`Tessera_EdgeFlip.hpp:142-178`).
  A Beatnik mask cannot answer it reliably, so the mask does not try. The same
  coordinator also sees every candidate in one call that would create `(c,d)`,
  and it accepts at most one of them: the best geometric survivor by the
  priority `(squared length DESCENDING, EdgeKey ASCENDING)`. The others count
  in `rejectedConflict`, under either `FlipConflict` relation (**M1-U3**,
  **DONE**, Tessera `33dc6b9`). T4d3 and T4d5 rely on this.
- **Tessera's default flip conflict relation is per face; the valence pass
  needs a different one.** Under the default `FlipConflict::SharedFace`,
  Tessera's independent set forbids two accepted flips sharing a *face* and
  allows two sharing only a vertex (`Tessera_EdgeFlip.hpp:187-239`). That is the
  reference's rule for its
  quality flips, whose `touched` set holds faces (`dynamic_remesh.py:416-421`,
  `:454`). The valence pass works differently (`mesh_quality.py:57-86`).
  - It counts valences, normals and neighbour sets **once, at the start of
    each pass**, and never updates them within the pass.
  - After accepting a flip it marks all four quad corners touched, and it skips
    a later edge iff one of **its own endpoints** is touched. A later edge whose
    *opposite* corners are touched may still flip.

  The symmetric form of that rule is: *two flips conflict iff a vertex is an
  endpoint of one and any corner of the other.* It implies the face rule, since
  a second flip sharing a face has both its endpoints in the first flip's quad.
  `flipEdges` has it as the opt-in mode `FlipConflict::EndpointInQuad`, set
  through `DefaultFlipPolicy::conflict` (**M1-U2**, **DONE**, Tessera `05b5942`),
  and T4d5 runs several rounds per pass to reproduce the reference's pass
  structure.

## Approach

Three strands, run in this order of *risk*, not of code size:

1. **Measure the achievable comparison first** (M1-G1, M1-D1). A fixed-mesh
   600-step Python/Beatnik comparison needs no new solver code and answers the
   only question that can invalidate the milestone's premise.
2. **Land T4d** (M1-U1, M1-U2, M1-U3, T4d1–T4d5, T4d6j, T4d6). M1-U1, M1-U2 and M1-U3 are
   small upstream Tessera changes that T4d2, T4d3 and T4d5 need (Read this first
   #6); T4d1 is an
   adapter layer, T4d2–T4d5 are one reference function each, T4d6j and T4d6 are an
   operational decision.
3. **Wire the milestone test and measure it** (M1-T2j, M1-T2, M1-D2), generate
   the long gold set, steps 0–1610 (M1-G3j, M1-G3), then **fix the acceptance and
   confirm it** (M1-A1j, M1-A1).

### Conventions

Framework.md's conventions table governs; these are the additions this milestone
introduces or sharpens.

| Convention | Choice |
| --- | --- |
| Task IDs | `T4d*` for work moved from framework.md (IDs are referenced from source strings); `M1-*` for work new here. `M1-G*` = reference-run gold-file generation, `M1-D*` = measurement, `M1-T*` = code, `M1-U*` = a change made upstream in `../tessera`, `M1-A*` = a decision to be recorded. A trailing `j` (`T4d6j`, `M1-T2j`, `M1-A1j`) marks the submit half of a long-job pair; the ID without it is the post-job half. |
| Edge-mask type | host `const std::vector<char>&`, sized `ownedEdgeCount()`, one entry per **owned** edge — T4a's convention for `splitEdges`, and exactly what Tessera's collapse and flip take. De-template the two stale `EdgeListView` signatures rather than adding an overload. |
| Where a reference acceptance rule lives | **In the mask whenever it needs only the quad; in the Tessera policy only when it needs both endpoints' one-rings.** Flip tests (the gain test, the valence test, the valence path's normal and quality tests) and the collapse candidate predicate and normalized-length ranking are evaluated by Beatnik while building the mask, and both flip call sites pass the permissive `DefaultFlipPolicy{ maxNormalDeviation = π, minQuality = 0 }`, the valence call site with `conflict = FlipConflict::EndpointInQuad` (M1-U2). Collapse's geometric test needs the full one-rings, so it is a Tessera policy: `maxNormalRotation = acos(0.1)`, `minQuality = --remesh-min-quality`, and the reference quality metric through M1-U1's hook. Read this first #6 has the evidence. A marked edge may still be rejected; that shows up in the `rejected*` counters, never as a silent drop. |
| Reference-state re-basing | every pass that moves a vertex or changes connectivity states, in its progress-log entry, whether it calls `AdaptiveMesh::resetReferenceState`, and why. A silent choice here changes every later refinement decision. |
| Failure behavior | loud. A configuration Beatnik cannot reproduce is rejected before the first step, by method name and task ID, exactly as the four rejections this milestone deletes do. No pass may silently no-op. |
| `Solver` rejections during T4d | **Every T4d rejection in `Solver::requireSupportedConfiguration` stays until T4d6j**, message text included: `--flip-passes > 0` under `--refine-every` ([src/Beatnik_Solver.hpp:778-783](../src/Beatnik_Solver.hpp#L778-L783)), `--isotropic-cleanup` ([:791-797](../src/Beatnik_Solver.hpp#L791-L797)), `--remesh-collapse-factor > 0` ([:830-850](../src/Beatnik_Solver.hpp#L830-L850)), `--remesh-smooth-iters > 0` ([:852-859](../src/Beatnik_Solver.hpp#L852-L859)) and `--remesh-flip-min-gain < 1e12` ([:861-870](../src/Beatnik_Solver.hpp#L861-L870)). So do the `regression`-tier cases that assert them: the four T4d `checkRejected` calls in [tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:696-707](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L696-L707) and `checkFailureDirection` in [tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp:379-417](../tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp#L379-L417). T4d2–T4d5 each **wire their pass into its call site anyway** and meet their exit criterion at the `unit` tier, so after each of them a `Solver` run reaches the new pass only at the knob value that makes it a no-op, and the gate trajectory is unchanged. T4d6j deletes all five rejections and their test cases in one change. Deleting one earlier is out of scope for the task at hand, even when its pass is finished. |
| Diagnostics that must be numbers | per remesh pass: `splits`, `collapses`, `flips`, `smooth_steps`, `min_quality` before/after, R12's two shape signals, and (T4d6j) the gid-space high-water mark. A milestone run that reports zero edits everywhere is M1-R7, not a pass. |
| Unit-tier rank and backend coverage | Unit members register at **one** rank on the build's default execution space ([tests/unit_tests/CMakeLists.txt:68-80](../tests/unit_tests/CMakeLists.txt#L68-L80)). "Ranks 1 and 4" in a T4d exit criterion means two submissions, `BEATNIK_UNIT_RANKS=1` and `BEATNIK_UNIT_RANKS=4 flux batch scripts/tuolumne/unit_tests.flux`, and the member's checks must be global reductions so one set of literals holds at both, as `Beatnik_Test_TangentialRelaxation` does. The unit tier has no SERIAL/HIP pair; the `regression` gate and the T4d umbrella launches in T4d6j's job supply backend coverage. |
| Long jobs | A task whose exit criterion needs a job expected to run longer than 20 minutes is a pair. `<ID>j` builds, runs anything shorter (the gate, the unit tier) in the session, stages, and submits; it ends when the scheduler accepts the job and never waits for it. `<ID>` depends only on `<ID>j`, reads the job's output and judges the original criterion; every later task depends on `<ID>`, never on `<ID>j`. The job reads only from a staged copy, `/p/lustre5/stewartj/beatnik/<ID>j/stage/`. It holds the binaries the job runs, copied from the dev view's `share/Beatnik/tests/` (where `beatnik_exe` resolves them) in their manifest-relative layout, plus the gold directories and `compare_output.py` they name, plus `COMMIT`: `git rev-parse HEAD` and the `git status --porcelain` count, taken at staging time. Beatnik, Tessera and Canopy are header-only in these binaries (`ldd` lists only Kokkos, Kokkos Kernels, heFFTe and Trilinos from the spack store), so the copied executable freezes the code under test; a dev-env `spack install` during the job cannot change it. The job echoes `stage/COMMIT`, never `git rev-parse HEAD`, and writes under `/p/lustre5/stewartj/beatnik/<ID>j/out/`. A post-job task that must resubmit restages under its own ID. |
| Test tier for the milestone comparison | the **`milestone`** label, at ranks **1 and 4** on SERIAL and HIP, outside the 60-launch ship gate. The tier is created by **M0-T1** in [`milestone0.md`](completed/milestone0.md); M1-T2j registers a member in it. The `regression` tier keeps exactly its five members; promoting anything into it needs the user's confirmation (CLAUDE.md "Minimum test set"). |
| Provenance comments | `// Port of <file>::<fn> (lines N-M)` against the **real** origin, per framework.md. The functions this milestone ports live in `dynamic_remesh.py` and `mesh_quality.py`; cite line ranges, not function names alone. |

### Deliberate deviations

- **Beatnik's accepted edit sets will not equal the reference's**, and no attempt
  is made to make them. The alternative — a sequential-equivalent scheduler in
  Tessera plus reduction-order pinning — was considered and rejected with the
  user: it is large, it reaches upstream, and it still cannot make a NumPy and a
  Kokkos trajectory agree over hundreds of steps. The consequence is that the
  milestone's compare depth is an empirical quantity (M1-A1j), not an assumption.
- **`--remesh-max-collapses 0` is not a lever.** A non-positive value maps to
  `None` = *unlimited* (`run_adaptive_mesh_bubble.py:1338`), reproduced by
  `RemeshParams::max_collapses_per_pass`. Only `--remesh-collapse-factor 0`
  disables collapse. The deleted rejection text says so and the replacement
  diagnostics must not re-introduce the confusion.
- **The valence flips' normal test uses the new faces as Tessera winds them**
  (T4d5, the user's ruling, 2026-11-06). The reference writes the children as
  `(c,d,a)` and `(d,c,b)` whatever the winding of `f0`, its first-indexed face
  of the edge (`mesh_quality.py:76-79`). On every edge where `f0` runs `a → b`
  both children are inverted, `n·n_f ≈ −1`, and the `≥ 0.2` test rejects the
  flip: 206 of the default icosphere's 480 edges can never flip, by array
  index rather than geometry. Beatnik's two faces per edge come in a rank-local
  order, so a literal copy would reject a rank-count-dependent set. Beatnik
  winds the child holding `a` and the child holding `b` consistently with the
  mesh and tests each against the old face it replaces, which rejects only
  genuine inversions and accepts flips the reference does not. On a scrambled
  icosphere the Python model gave 7, 1, 0 flips per pass for the reference
  (7 of 15 valence candidates rejected by the normal test) against 7, 2, 0
  (3 of 12) with consistent winding. Compare valence statistics accordingly.
- **`SurfaceMesh::compact()` is not called from the remesh cycle.**
  `Tessera::collapseEdges` runs `compact()` as its own last step; the standalone
  adapter is called only by tests. T4d6j's renumbering cadence calls
  `SurfaceMesh::compactAndRenumberGids()`, which compacts first. Its doc comment
  says so, rather than leave a reader to infer it.

## Current state

True at HEAD, and specific about what is *defined but wrong* versus what
*throws*:

- **Implemented and reachable from a `Solver` run** (T4d3, **DONE**; the
  rejections deleted by T4d6j): `DynamicRemesh::flipEdgesForQuality`
  ([src/Beatnik_DynamicRemesh.hpp:1160](../src/Beatnik_DynamicRemesh.hpp#L1160)),
  called from the pass loop's repair gate, and
  `MeshQuality::improveConnectivityByFlips`
  ([src/Beatnik_MeshQuality.hpp:986](../src/Beatnik_MeshQuality.hpp#L986)),
  called from the refine branch
  ([src/Beatnik_Solver.hpp:478](../src/Beatnik_Solver.hpp#L478)), which
  re-bases the AMR reference state under `flips > 0 && smooth_iters == 0`
  (`:481`). Both run `qualityFlipSweep` (`:417`): several `flipEdges` rounds
  with a sweep-wide touched set of new diagonals, gathered from
  `MeshEditReport::flipped`. At `--remesh-flip-min-gain 1e12` and
  `--flip-passes 0` the passes return before any mutation. Covered by case 4 of
  `Beatnik_Test_MeshEdits`; the refine branch's flips are asserted live at
  `--flip-passes 2` by check 6 of the `regression` member
  `Beatnik_Test_RefineSplitEdges`.
- **Implemented and reachable from a `Solver` run** (T4d2, **DONE**; the
  rejection deleted by T4d6j): `DynamicRemesh::collapseShortEdges`, called once
  per pass after a second sizing-field recompute, through
  `ReferenceCollapsePolicy` (the reference metric on M1-U1's hook). At
  `--remesh-collapse-factor 0` it returns before any mutation. Covered by case 3
  of `Beatnik_Test_MeshEdits`.
- **Implemented** (T4d1, **DONE**), with `collapseEdges` called by T4d2 and
  `flipEdges` called by T4d3 and `compact` called only by the test; T4d6j added
  `SurfaceMesh::compactAndRenumberGids` (called by `DynamicRemesh::remesh` every
  `RemeshParams::renumber_gids_every` calls, default 50) and `gidSpace()`
  (reported as `RemeshDiagnostics::gid_space_high_water`):
  `SurfaceMesh::{collapseEdges, flipEdges, compact}` take a host
  `std::vector<char>` mask sized `ownedEdgeCount()` and a trailing policy
  defaulted to `collapse_policy_type` / `flip_policy_type`, and return
  `MeshEditReport`, which carries Tessera's global collapse, flip and compact
  counters beside the split ones. `collapseEdges` exchanges the halo first and
  throws if the V/E/F identities fail. Covered by the `unit` member
  `Beatnik_Test_MeshEdits`.
- **Present upstream, unused by Beatnik:** Tessera's collapse and flip policies
  carry the `quality()` hook (M1-U1, **DONE**, Tessera `279287e`), defaulting to
  the radius ratio; T4d2's `ReferenceCollapsePolicy` shadows it. `flipEdges` has the
  opt-in `FlipConflict::EndpointInQuad` relation beside the default per-face one
  (M1-U2, **DONE**, Tessera `05b5942`); T4d5's valence flips set it.
  `flipEdges` accepts at most one of several flips in one call that would create
  the same new diagonal, under either relation (M1-U3, **DONE**, Tessera
  `33dc6b9`); T4d3's quality flips rely on it.
- **Implemented and reachable from a `Solver` run** (T4d4; the rejection
  deleted by T4d6j): `DynamicRemesh::tangentialSmooth`
  ([src/Beatnik_DynamicRemesh.hpp:1247](../src/Beatnik_DynamicRemesh.hpp#L1247)),
  called from the pass loop's repair gate after the flips
  (`diag.smooth_steps += tangentialSmooth( mesh )`,
  [:1416](../src/Beatnik_DynamicRemesh.hpp#L1416)). It is T4c's stencil
  (`MeshGeometry` normals, `graphLaplacianVector`, `projectTangent`) plus the
  reference's all-or-nothing quality guard per iteration, and returns the
  accepted-iteration count. At `--remesh-smooth-iters 0` the pass returns
  before any halo exchange. Covered by case 5 of `Beatnik_Test_MeshEdits`.
- **Implemented and reachable from a `Solver` run** (T4d5; the rejection
  deleted by T4d6j): `MeshQuality::valenceEqualizingFlips`
  ([src/Beatnik_MeshQuality.hpp:652](../src/Beatnik_MeshQuality.hpp#L652))
  and `MeshQuality::isotropicCleanup` ([:937](../src/Beatnik_MeshQuality.hpp#L937)),
  called from both branches of `Solver::advanceOneStep` under
  `_params.cleanup.enabled`
  ([src/Beatnik_Solver.hpp:511](../src/Beatnik_Solver.hpp#L511),
  [:568](../src/Beatnik_Solver.hpp#L568)), each call followed by
  `Solver::rebuildState` ([:839](../src/Beatnik_Solver.hpp#L839):
  `resetReferenceState` + `centerPotential`); the remesh branch also rebuilds
  before the cleanup. The valence flips run several `EndpointInQuad` rounds per
  pass with a pass-wide touched set of vertex gids. The cleanup is on by
  default; under `--no-isotropic-cleanup` neither the cleanup nor its rebuild
  runs. Covered by case 6 of `Beatnik_Test_MeshEdits`.
- Implemented and validated, and this milestone must not re-invent them: the
  sizing field, the gradation sweeps, the split pass and the edge-mask plumbing
  (T4a/T4b); `MeshQuality::relaxTangential` and `improveQualityTangential` (T4c);
  the RHS, the TVD-RK3 integrator with adaptive dt, the volume projection, the
  checkpoint writer and the step loop (T2b–T2d).
- `CheckpointIO::read`/`RestartReader::load` still throw (T5b). Not needed here:
  the milestone compares files written by Beatnik against `.npz`, and reads
  neither.

## Progress log

Session-by-session record: **[`milestone1-progress-log.md`](milestone1-progress-log.md)**.

Read it before implementing a task, before changing any signature this document
names, and before reopening a question this document states flatly — a completed
task may have changed the plan for a later one, and each entry's `**Affects:**`
line is the index of exactly that.

> Each task is reviewed with `review-task` before any task that depends on it
> starts. A task is ready to build on when it is marked **DONE** and the log has
> a `## <ID> review` section whose verdict is `Passed`. The document's status
> becomes COMPLETE only when a `review-task` sweep of the whole feature passes.

## Task sequence

### M1-G1 — the fixed-mesh 600-step gold set *(reference run, no code)* — **DONE**

**Depends on:** none.

**Fill in:** `tests/regression_tests/direct-solve-600-steps/gold/*.npz` plus that
directory's `README.md`, following the convention of
[tests/regression_tests/direct-solve-10-steps/README.md](../tests/regression_tests/direct-solve-10-steps/README.md)
— the generating command recorded verbatim beside the data.

**Reference:** T2a's command, extended. The point of this set is to hold
connectivity **fixed** so the only source of divergence is floating-point
(R2) — no R7, no R13:

```
python examples/run_adaptive_mesh_bubble.py --steps 600 \
  --source-quadrature vertex --br-approximation direct \
  --no-dynamic-remesh --refine-every 0 \
  --checkpoint-every-steps 10 --no-video --checkpoint-dir results
```

**Do:** generate it; keep all 61 files (steps 0, 10, …, 600); confirm step 0 is
bitwise identical to `initial_conditions/gold.npz` as T2a's was, and that the key
set matches so `FIELD_MAP` needs no edit.

**Exit criterion:** 61 numbered `_step%07d.npz` checkpoints present (steps 0,
10, …, 600). The `checkpoint_latest.npz` the reference writes beside them does not
count: it duplicates the last numbered step and is inert to `goldForStep`, which
matches only the `_step%07d.npz` suffix. Also:
`python tests/regression_tests/compare_output.py <gold>/…_step0000000.npz tests/regression_tests/initial_conditions/gold.npz --rtol 1e-12 --atol 1e-14`
exits 0, and a self-compare of the step-600 file against itself at the same
tolerances exits 0 with zero ambiguous vertices (the check that
`--match-eps 1e-9` still resolves this mesh at step 600).

**Met.** Job `f3hyB1FBHAym` (`scripts/tuolumne/m1g1_gold.flux`, `pdebug`) ran
the command above verbatim except `--checkpoint-dir`, in 25 s. 61 numbered
checkpoints, `_step0000000` through `_step0000600` every 10, plus the
`checkpoint_latest.npz` that does not count (byte-identical to the step-600 file
by `cmp`), copied into `gold/` and verified by `cmp` and `sha256sum`. No early
stop: the last progress line is `step=  600`, `saved final finite checkpoint` is
logged, there is no `stopping at step=`, and every field is finite in all 61
files. `vertices (162, 3)` / `faces (320, 3)` and the same nine keys as
`initial_conditions/gold.npz` in every file, so `FIELD_MAP` needs no edit. Step 0
is `np.array_equal` to `initial_conditions/gold.npz` on all nine keys, and the
step-0 `compare_output.py` at `--rtol 1e-12 --atol 1e-14` exits 0. The
self-compare of `checkpoint_t00001p314132_step0000600.npz` at the same
tolerances exits 0 with `matching (eps=1e-09): 162/162 unambiguous, ambiguous
cpp=0 gold=0`. Final `time = 1.3141321331131375`. The command and provenance are
in [the README](../tests/regression_tests/direct-solve-600-steps/README.md).

---

### M1-D1 — measure the fixed-mesh divergence horizon — **DONE**

**Depends on:** M1-G1.

This is the task that decides whether the milestone's premise holds. It writes no
library code.

**Fill in:** `scripts/tuolumne/milestone1_divergence.flux` (a batch script; never
launch interactively from a login node), and a table of measured numbers in
`milestone1-progress-log.md`.

**Reference:** M0-D1 made the same measurement for milestone 0, and its two
artifacts are this task's tools, used unchanged:

- The driver
  [tests/regression_tests/Beatnik_Test_Milestone0Run.cpp](../tests/regression_tests/Beatnik_Test_Milestone0Run.cpp),
  installed as `Beatnik_Test_Milestone0Run_MPI_{SERIAL,HIP}` in no test tier and
  resolved through `beatnik_exe`. Its `makeParams()` (`:174-260`) is the M1-G1
  command field for field once the positionals are `2 600 10`: subdivision level
  2, 600 steps, a checkpoint every 10. It asserts the global vertex and face
  counts every step and that the step budget was reached.
  `examples/02_adaptive_mesh_bubble` cannot serve: it fixes
  `Kokkos::DefaultExecutionSpace` (`adaptive_mesh_bubble.cpp:212`) and the
  installed binary is `+rocm`, so it runs HIP only.
- The offline tabulator
  [tests/regression_tests/milestone0_ladder.py](../tests/regression_tests/milestone0_ladder.py):
  `pair` for the confirmed per-rung first-failing step and the per-field table,
  `growth` for full-precision `max|e|` series, `series` for volume drift and
  minimum quality. Its `RUNGS` (`:105`) are step 2's five tolerances.
- [scripts/tuolumne/milestone0_divergence.flux](../scripts/tuolumne/milestone0_divergence.flux)
  is the batch-script shape to follow: resolver sourcing, provenance block, the
  rank-to-GPU binding, one freshly-created scratch directory per run, the
  FD-3 row loop, and the refusal to run under `BEATNIK_USE_PROD=1`. This task's
  script uses the **dev** spack env for the same reason.

**Do:**

1. Run `Beatnik_Test_Milestone0Run_MPI_<BACKEND> 2 600 10` at ranks 1 and 4 on
   SERIAL and HIP, into `BEATNIK_TEST_SCRATCH` (**a parallel filesystem** — a
   node-local scratch fails every multi-node launch, CLAUDE.md).
2. For each of the 61 steps, run `compare_output.py` at each of
   `--rtol 1e-12/1e-10/1e-8/1e-6/1e-4` (with `--atol` two decades below) and
   record the first step at which each fails, plus `max|e|` per field per step.
3. Record separately the first step at which the **quantized matching** reports
   any ambiguity at `--match-eps 1e-9`, and the first step at which two Beatnik
   runs at different rank counts disagree by more than the Python comparison does
   — that separates R2's cross-rank noise from genuine Python/Beatnik drift.

**Exit criterion:** the progress log carries, as 17-digit literals, the
per-tolerance first-failing step for the four (backend, rank) combinations, and
the `max|e|` growth series for `vertices` and `potential`. The measurement
**fails** if step 0 does not match at `1e-12` — that would be a regression in an
already-validated path, not a divergence measurement, and must be reported as
such rather than absorbed into the trend.

**Met.** Job `f3hypK87UYCo` (`scripts/tuolumne/milestone1_divergence.flux`,
`pdebug`, dev env, beatnik `ckhvawf` built from `70fda7c`) ran
`Beatnik_Test_Milestone0Run_MPI_<B> 2 600 10` at HIP np1/np4 and SERIAL np4/np1.
It completed with rc 0 and `launched=4 skipped=0`. Each launch wrote 61
checkpoints, and every rank printed `[PASS] Beatnik_Test_Milestone0Run (6/6
checks)`. The log has no `FAIL`, `STOPPED EARLY` or `ENTITY COUNTS CHANGED`.
**Step-0 gate passed** for all four runs at `--rtol 1e-12 --atol 1e-14`
(`RESULT: match`, rc 0). **Against Python, no rung fails at any checkpointed step
through 600**, for all four (backend, rank) combinations, not even
`1e-12`/`1e-14`. Every rung reads derived `None` / confirmed `None`. Peak
`vertices` `max|e|` is `9.88098491916389321e-15` (HIP np4, step 580), within 1.2 %
of the tightest rung's `atol`. **Rank 1 against rank 4** never fails on the
shared fields (`vertices`, `potential`) at any rung, derived or confirmed. Over
the full field set, `sheet_vector` permits a `1e-12` failure from step 420 (HIP)
or 530 (SERIAL), and confirmation finds no failing step. Cross-rank `max|e|` stays
below the Python comparison's at every step on both shared fields (peak ratio
Python/cross-rank 2.21x–4.73x), so the drift is Python-versus-Beatnik, not
decomposition noise. The quantized pairing reported **no ambiguity** at
`--match-eps 1e-9` in any of the 366 scan invocations or the 366 growth rows.
Counts stay at 162 vertices / 320 faces throughout. Solve walls are 2.56–6.35 s,
and peak RSS is at most 1003816 kB per rank. The 17-digit tables are in
[`milestone1-progress-log.md`](milestone1-progress-log.md) `## M1-D1`.

---

### M1-G2 — the milestone gold set *(reference run, no code)* — **DONE**

**Depends on:** none. Independent of M1-D1 — generate it in parallel.

**Fill in:** `tests/regression_tests/milestone1-bubble/gold/*.npz` plus that
directory's `README.md` with the command recorded verbatim.

**Reference:** the target command (`## Problem`), plus the three options that make it produce
data (Read this first #1):

```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --dynamic-remesh --isotropic-cleanup \
  --source-quadrature vertex \
  --steps 600 --checkpoint-every-steps 10 --no-video --checkpoint-dir results
```

Every other option stays at its `parse_args` default. The four explicit switches
and `--source-quadrature vertex` are *already* the defaults
(`run_adaptive_mesh_bubble.py:64-533`); they are written out because the
reference README's configuration (a) does, and because a reader comparing this set against the T2a set will otherwise
suspect the two describe different physics — they do not (see the T2a entry in
[`framework-progress-log.md`](framework-progress-log.md)).

**Do:** generate it; record in the directory's `README.md` the per-step face and
vertex counts, the `time` series, and whether the run stopped early
(`stopping at step=… nonfinite …`). Those three are what M1-T2j asserts against
and what M1-A1j reasons about; a gold set whose face count is not recorded forces
the next session to re-derive it from 61 files.

**Exit criterion:** 61 numbered `_step%07d.npz` checkpoints present (fewer only if
the reference itself stopped early, in which case the stop step is recorded in the
`README.md` and becomes the compare depth ceiling). The `checkpoint_latest.npz`
beside them does not count, as in M1-G1. Also: a self-compare of the last file at
`--rtol 1e-12 --atol 1e-14` exits 0; step 0 is bitwise identical to
`initial_conditions/gold.npz`.

**Met.** Job `f3hyB1NchZz7` (`scripts/tuolumne/m1g2_gold.flux`, `pdebug`) ran
the command above verbatim except `--checkpoint-dir`, in 707 s (1.18 s/step).
61 numbered checkpoints through `_step0000600`, plus the `checkpoint_latest.npz`
that does not count (byte-identical to the step-600 file by `cmp`), copied into
`gold/` and verified by `cmp` and `sha256sum`. **No early stop**: the last
progress line is `step=  600`, `saved final finite checkpoint` is logged, there
is no `stopping at step=`, and every field is finite in all 61 files, so the
compare-depth ceiling is the full 600 steps. The same nine keys in every file.
Step 0 is `np.array_equal` to `initial_conditions/gold.npz` on all nine keys. The
self-compare of `checkpoint_t00000p557934_step0000600.npz` at `--rtol 1e-12
--atol 1e-14` exits 0 with `matching (eps=1e-09): 538/538 unambiguous,
ambiguous cpp=0 gold=0`. Face count: `320` at step 0, a peak of `1090` at step
10, then `1072`–`1076` from step 60 to the end, finishing at `1072` (`538`
vertices). Final `time = 0.55793412422180344`. The 61-row step / `time` / V / F
table is in [the README](../tests/regression_tests/milestone1-bubble/README.md).

---

### T4d — Coarsening, flips, and isotropic cleanup

The umbrella ID that four `requireSupportedConfiguration` messages and several
header comments name. It is **not** a task: the work is T4d1–T4d6 below, and the
umbrella exit criterion is judged in T4d6 from T4d6j's job.

← *Python:* `dynamic_remesh.py::{collapse_short_edges (361-405),
flip_edges_for_quality (408-457), tangential_smooth_vertices (459-490),
compact_mesh (492-508), _edge_collapse_is_topologically_safe (509-516),
_edge_collapse_is_geometrically_safe (519-549)}`,
`mesh_quality.py::{_valence_equalizing_flips (44-87), _tangential_relaxation
(90-116), isotropic_cleanup (146-167)}`, and the cycle that calls them,
`dynamic_remesh.py::dynamic_remesh_arrays (118-194)`.

**Read the three Tessera header comment blocks before writing anything** —
`Tessera_EdgeCollapse.hpp`, `Tessera_EdgeFlip.hpp`, `Tessera_Compact.hpp`. Four
caller obligations from them apply to every sub-task and are not restated in
each:

1. **Halo depth ≥ 2** or `collapseEdges` throws. `SurfaceMesh::halo_depth` is
   already 2 ([src/Beatnik_MeshInterface.hpp:423](../src/Beatnik_MeshInterface.hpp#L423));
   do not lower it.
2. **`haloExchange()` *before* collapsing** — the surviving vertex's owner blends
   positions and every vertex user field from local copies of both endpoints, one
   of which is generally a ghost. Note the asymmetry with `splitEdges()`, whose
   wrapper exchanges *after*
   ([src/Beatnik_DynamicRemesh.hpp:873-874](../src/Beatnik_DynamicRemesh.hpp#L873-L874)).
3. **One independent set per call.** More progress means calling again:
   `while ( collapseEdges( … ).accepted > 0 ) {}`.
4. **All three invalidate every slice, CSR and key View.** Re-slice afterwards.

Two facts that remove work the earlier design assumed: `compact()` is a halo
rebuild that whole-tuple copies every AoSoA, so **face and vertex user fields
survive collapse, flip and compaction verbatim** — T4a's `/faces/u0..u2` are not
invalidated; and `DefaultCollapsePolicy{t=0.5}` is the same linear average
Beatnik already uses for refinement
([src/Beatnik_MeshInterface.hpp:75-104](../src/Beatnik_MeshInterface.hpp#L75-L104)),
which is what the reference does (`dynamic_remesh.py:395-398`), so **no custom
field-blending rule is needed** — the collapse policy T4d2 passes derives from
`DefaultCollapsePolicy` for M1-U1's quality hook and inherits its blend
unchanged. The real hazard is *staleness*: the face user
fields are the AMR reference state and now describe a geometry that no longer
exists — see the re-basing convention.

---

### M1-U1 — a quality hook on Tessera's collapse and flip policies *(upstream, in `../tessera`)* — **DONE**

**Depends on:** none.

**Fill in:** in `../tessera/src/`: `DefaultCollapsePolicy`
(`Tessera_EdgeCollapse.hpp:378-407`) and its one quality call
(`:995-996`); `DefaultFlipPolicy` (`Tessera_EdgeFlip.hpp:262-272`) and its
two quality calls (`:662-663`); the doc comments that name the radius ratio
(`DefaultCollapsePolicy::minQuality` `:387-390`, `CollapseResult::rejectedQuality`
`:329-331`, `DefaultFlipPolicy::minQuality` `Tessera_EdgeFlip.hpp:268-269`); one new case each in
`tests/test_collapse_edges.cpp` and `tests/test_flip_edges.cpp`; Tessera's
`README.md` and `docs/design.md` wherever they describe the policies.

**Reference:** Read this first #6 for why it exists. The per-field override
pattern `DefaultCollapsePolicy` already documents (`Tessera_EdgeCollapse.hpp:351-377`) is the shape
to match: a policy is duck-typed and a consumer derives from the default and
shadows one member.

**Do:**

1. Add to both default policies a host member
   `double quality( const double* p0, const double* p1, const double* p2 ) const`
   returning `detail::flipRadiusRatio( p0, p1, p2 )`, documented with the
   metric's range (`[0,1]`, 1 for an equilateral triangle) and that
   `minQuality` thresholds whatever it returns. `flipRadiusRatio` is defined
   after `DefaultFlipPolicy` (`Tessera_EdgeFlip.hpp:391`), so move it, or the
   `detail` helpers it needs, above the struct; do not duplicate it.
2. Replace the three `flipRadiusRatio( … ) < / >= policy.minQuality`
   comparisons with `policy.quality( … )`. Keep the comparison directions
   unchanged: collapse rejects `quality < minQuality`, which is exactly the
   reference's `min(quality) < min_quality` reject
   (`dynamic_remesh.py:544`).
3. Default behavior must be bit-identical. **Callers:** no policy other than
   the two defaults exists in `src/`, `tests/` or `examples/` (grep for
   `CollapsePolicy`/`FlipPolicy`); the only explicit constructions are
   `test_collapse_edges.cpp:1209` and `test_flip_edges.cpp:1061`, both of the
   defaults. Beatnik passes no policy today.
4. Follow Tessera's own CLAUDE.md for build, header and test conventions. It is
   a separate repository with its own gate; this task commits there. The Beatnik
   environment builds Tessera as a `develop` spec
   (`../spack.yaml`, `develop: tessera: spec: tessera@=develop`), so a Beatnik
   `spack install` picks up the change with no version bump.

**Exit criterion:** the two affected Tessera tests pass at every rank and backend
they are registered at (SERIAL and HIP, ranks 1–5). From the `../tessera` root,
with `TESSERA_REPO` exported:
`flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`.
The full regression gate is **not** required, and Tessera's `docs/testing.md`
("When to run the full gate") makes a targeted run the default. The changed
symbols — the two default policies and `detail::flipRadiusRatio` — are used
nowhere in `src/` outside `Tessera_EdgeCollapse.hpp` and `Tessera_EdgeFlip.hpp`,
and only `test_collapse_edges.cpp` and `test_flip_edges.cpp` call
`collapseEdges`/`flipEdges`. Build only the four executables that run needs,
not the whole tree:
`make -j tessera_test_collapse_edges_SERIAL tessera_test_collapse_edges_HIP tessera_test_flip_edges_SERIAL tessera_test_flip_edges_HIP`
in the build directory. Tessera is header-only (`add_library(Tessera INTERFACE)`,
`../tessera/CMakeLists.txt:74`), so these four targets
(`tessera_test_<NAME>_<BACKEND>`, `../tessera/tests/CMakeLists.txt:75`) compile
every line the change touches. Other test executables, which include the
headers through `Tessera.hpp`, are not built. The task's report states this
subset and the reason. The two new cases show the hook is live in both
directions: on a fixture where the radius ratio and `4√3·A/Σℓ²` fall on
opposite sides of one threshold, a policy deriving from the default and
overriding `quality` with `4√3·A/Σℓ²` changes the verdict compared with the
default. The changed candidate moves between `accepted` and
`rejectedQuality` (collapse) or `rejectedGeometric` (flip). The
default-policy result is unchanged.

**Met.** Tessera `279287e` on `mesh-quality`, job `f3hmiTasm3Ef` (`pdebug`,
182 s, rc 0): `100% tests passed, 0 tests failed out of 20` — all twenty cases
ran, `flip_edges_{SERIAL,HIP}_np1`–`np5` and
`collapse_edges_{SERIAL,HIP}_np1`–`np5`. Only the four test executables were
built, for the reason above. Every existing case passes at its unchanged
measured literals, which is the bit-identity check on the default path. The new
cases, `test_collapse_edges` case 15 and `test_flip_edges` case 13, printed the
same line in all ten launches of their test and in both execution spaces:

- **Collapse** (case 7's spike bipyramid, ring edge `(2,3)`, `minQuality = 0.47`):
  the surviving faces' worst radius ratio is `0.48306530887389387` and their worst
  mean ratio is `0.46367649590853621`. The default policy gives `accepted=1`; a
  policy deriving from it with `quality` = `4√3·A/Σℓ²` gives `rejectedQuality=1`.
- **Flip** (a hexagonal bipyramid with top apex at `z = 0.1`, spoke `(0,2)`,
  `minQuality = 0.55`): the new faces' worst radius ratio is
  `0.46410161513775444` and their worst mean ratio is `0.59999999999999987`. The
  default policy gives `rejectedGeometric=1`; the mean-ratio policy gives
  `accepted=1`.

So the hook moves the verdict in both directions. `flipTriNormal` and
`flipRadiusRatio` now sit in a `detail` block directly above `DefaultFlipPolicy`
(`Tessera_EdgeFlip.hpp:260-302`). The line references in this entry's **Fill
in** and **Do** are against the pre-change file. Log: `## M1-U1`.

---

### M1-U2 — an endpoint-aware conflict mode for `flipEdges` *(upstream, in `../tessera`)* — **DONE**

**Depends on:** none. M1-U1 (**DONE**, Tessera `279287e`) also edited
`DefaultFlipPolicy`, adding `quality()`, so this task builds on it. The line
references below are against `279287e`.

**Fill in:** in `../tessera/src/Tessera_EdgeFlip.hpp`: a new
`enum class FlipConflict { SharedFace, EndpointInQuad }` and a member
`FlipConflict conflict = FlipConflict::SharedFace;` on `DefaultFlipPolicy`
(`:308-325`); the step-3 independent-set round (`:687-736`); the header's step-3
and Decision 2 text (`:154-177`) and the `flipEdges` doc comment
(`:896-900`). Move `detail::vertexCoordRank` from `Tessera_EdgeCollapse.hpp:422-429`
into `Tessera_EdgeFlip.hpp`'s `detail` namespace, which `Tessera_EdgeCollapse.hpp`
already includes (`:18`), rather than defining it twice. Add one new case to
`tests/test_flip_edges.cpp`, and update Tessera's `README.md` and
`docs/design.md` where they state the flip conflict rule.

**Reference:** Read this first #6 for the rule and its derivation from
`mesh_quality.py:57-86`. `collapseEdges`' vertex-coordinator independent-set
round (`Tessera_EdgeCollapse.hpp:1156-1203`, message types `CollapseNbr`/
`CollapseWin` at `:552-567`) is the pattern to copy.

**Do:**

1. Under `EndpointInQuad`, step 3 sends each surviving candidate to the
   `vertexCoordRank` coordinator of each of its four quad corners, tagged with
   the corner's **role**: *endpoint* (`lo`, `hi`) or *opposite* (`w`, `x`). Use
   the existing priority `(len2 DESCENDING, EdgeKey ASCENDING)`. A candidate's
   four corners are distinct, so it names each vertex once.
2. At the coordinator of vertex `v`, let `B` be the best candidate naming `v`.
   - If `B` names `v` as an **endpoint**, endorse only `B`.
   - Otherwise, endorse **every** candidate naming `v` as *opposite*, and none
     that names it as an endpoint.

   A candidate is accepted iff it collects all four endorsements. This
   accepts no conflicting pair. If accepted flips `E1` and `E2` conflicted at
   `v`, with `v` an endpoint of `E1`, then `E1` being endorsed at `v` means
   `B = E1` and `E1` was the only flip endorsed there, which contradicts `E2`
   being accepted. State that argument in the header.
3. Under `EndpointInQuad` the per-face round does not run; the new rule
   implies it. Losers count into `rejectedConflict`, as now.
4. `SharedFace` is the default, and its path must stay **bit-identical**.
   **Callers:** no policy other than `DefaultFlipPolicy` exists in `src/` or
   `examples/`. In `tests/` the explicit constructions are
   `test_flip_edges.cpp:1067` and case 13's `caseQualityHook`, which builds a
   `DefaultFlipPolicy` and a `MeanRatioFlipPolicy` deriving from it. A derived
   policy inherits `conflict`'s `SharedFace` default, so neither changes.
   `flipEdges`' signature does not change.
5. Follow Tessera's CLAUDE.md, as M1-U1 does.

**Exit criterion:** the same targeted run as M1-U1, and for the same reason. Run
`flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`
from `../tessera` with `TESSERA_REPO` exported: SERIAL and HIP, ranks 1–5,
after building the same four executables. The full regression gate is **not**
required.
`collapse_edges` stays in the set because `detail::vertexCoordRank` moves out
of its header, and `vertexCoordRank`'s only callers are in the two headers. The
new `flip_edges` case checks the rule from
`FlipResult::flipped`, whose `(old key, new key)` pairs give each accepted flip's
endpoints and opposite corners, gathered across ranks:
- **Success, `EndpointInQuad`:** no accepted flip's endpoint is a corner of
  another accepted flip.
- **Success, `EndpointInQuad`:** a fixture with two flips that share only an
  opposite corner accepts both. This shows the mode is not plain vertex
  disjointness.
- **Failure direction:** on a fixture with two marked flips where one's
  endpoint is a corner of the other but they share no face, `SharedFace`
  accepts both and `EndpointInQuad` accepts exactly one, with the other in
  `rejectedConflict`.

The accepted sets are identical at ranks 1–5.

**Met.** Tessera `05b5942` on `mesh-quality`, job `f3hn6UjNexxw` (`pdebug`,
193 s, rc 0): `100% tests passed, 0 tests failed out of 20`. All twenty cases
ran: `flip_edges_{SERIAL,HIP}_np1`–`np5` and
`collapse_edges_{SERIAL,HIP}_np1`–`np5`. Only the four test executables were
built, for the reason above. Every existing case passes at its unchanged
literals, including both M1-U1 hook cases. The `SharedFace` code is textually
unchanged; the diff removes only comment lines from `Tessera_EdgeFlip.hpp`. The
new case is `test_flip_edges` case 14, with the permissive policy
`{maxNormalDeviation = π, minQuality = 0}`. It printed the same four lines in all
twenty launches of `flip_edges`, Serial and Default spaces alike:

- **Two-ring bipyramid** (V 18, E 48, F 32), three marked pairs, no pair sharing
  a face:
  - **Shared opposite corner**, `(2,3)+(6,7)`, sharing only the apex `T = 0`.
    `SharedFace` and `EndpointInQuad` both accept 2.
  - **Endpoint wins**, `(2,3)+(0,6)`. `SharedFace` accepts 2, and that set has
    1 endpoint-in-quad violation. `EndpointInQuad` accepts `{(0,6)}` with
    `rejectedConflict=1`.
  - **Opposite wins**, `(2,3)+(1,10)`. `SharedFace` accepts 2, with 1
    violation. `EndpointInQuad` accepts `{(2,3)}` with `rejectedConflict=1`.
- **Subdivision-2 icosphere, all 480 edges marked.** `SharedFace` accepts 80 and
  its set has 160 ordered violating pairs. `EndpointInQuad` accepts 28, with
  `rejectedConflict=452` and 0 violations. The `MPI_COMM_SELF` reference accepts
  28, and its flip set is identical.

The violation check runs on `FlipResult::flipped`, gathered across ranks. The
two "wins" pairs exercise the two branches of the endorsement rule. The line
references in this entry's **Fill in**, **Reference** and **Do** are against
`279287e`. At `05b5942`, `vertexCoordRank` is `Tessera_EdgeFlip.hpp:463-466`,
and the step-3 round is `:764-872`. Its `EndpointInQuad` branch is `:775-827`,
and the `SharedFace` branch is the `else` at `:828`. Log: `## M1-U2`.

---

### M1-U3 — no two accepted flips create the same diagonal *(upstream, in `../tessera`)* — **DONE**

**Depends on:** M1-U2 (**DONE**, Tessera `05b5942`). The line references below
are against `05b5942`.

**The defect.** Two marked edges `(a,b)` and `(c,d)` can have the same
opposite corners `w, x`, so both flips would create the same diagonal `(w,x)`.
`flipEdges` accepts both, under either `FlipConflict`, and the mesh ends with
two edges keyed `(w,x)`:

- The duplicate-edge round (step 2b, `Tessera_EdgeFlip.hpp:695-723`) asks the
  `(w,x)` coordinator only whether `(w,x)` is in the **pre-flip** edge set.
- The flips share no face, so `SharedFace` does not separate them.
- Neither flip has an endpoint among the other's corners, so `EndpointInQuad`
  does not separate them either.

Reproduction: M1-U1's `flatBipyramidSoup` (`tests/test_flip_edges.cpp:1129`)
with ring edges `(2,3)` and `(5,6)` marked, under the permissive policy. Both
edges have opposite corners `0, 1`. This has been found by reading, not yet by
running. A Beatnik mask cannot prevent it reliably: the owner of `(a,b)` does
not see `(c,d)` in general. The fix therefore belongs in `flipEdges`, for every
caller and under both relations. T4d3 and T4d5 call `flipEdges` without
testing the new diagonal in their masks (Read this first #6).

**Fill in:** in `../tessera/src/Tessera_EdgeFlip.hpp`:

- step 2b and 2c (`:695-762`);
- `detail::DupQuery`/`DupReply` (`:411-421`), whose only users are step 2b;
- the header's step-2 `DUPLICATE` text (`:142-150`);
- the `FlipResult::rejectedConflict` comment (`:277-281`).

Add one new case to `tests/test_flip_edges.cpp`. Update the duplicate-edge
passage in `docs/design.md` ("The duplicate-edge round") and the `rejected*`
comments in `README.md`'s flip section. `flipEdges`' signature and
`FlipResult`'s members do not change.

**Reference:** the `(w,x)` coordinator in step 2b already receives one
`DupQuery` from every candidate that would create `(w,x)`, so it is the one rank
that sees every claimant. Resolve the claims there, in the existing round, with
no new collective.

**Do:**

1. **Evaluate the geometric test before the duplicate round, and change no
   verdict precedence.** Compute each candidate's pass/fail in 2c's code, but
   attribute verdicts in the existing order: boundary, then duplicate, then
   geometric. A candidate whose diagonal exists still counts as
   `rejectedDuplicateEdge` even if it also fails geometry. Case 3
   (`caseDuplicateEdge`, `:901`) asserts `rejectedDuplicateEdge == 1`, and it
   must not move.
2. **Each `DupQuery` carries the asker's `len2` and its geometric pass/fail.**
   At the `(w,x)` coordinator:
   - If `(w,x)` exists, every asker is answered *exists*.
   - Otherwise, among askers that **passed geometry**, the best by the existing
     priority `(len2 DESCENDING, EdgeKey ASCENDING)` is answered *free*. Every
     other geometric survivor is answered *claimed*.
   - An asker that failed geometry is answered *free*, because its own
     geometric verdict rejects it next.

   `DupReply::exists` becomes a three-valued reply. A *claimed* candidate counts
   into `rejectedConflict`: it lost to a higher-priority candidate, and no
   pre-flip edge is in the way. Filtering on geometry is what stops a winner
   that is about to fail its own geometric test from vetoing a valid flip.
3. **The claim winner can still lose step 3.** If it does, the claimant it shut
   out is not reconsidered. That is the same non-maximality as Decision 3 (one
   round, one bounded cost), and the header says so.
4. **Bit-identity.** On every input where no two geometric survivors share a new
   diagonal, every verdict and counter equals `05b5942`'s, under both relations.
   Every existing `flip_edges` and `collapse_edges` case must pass at its
   unchanged literals.
   - Most existing calls cannot contain such a pair: they mark one edge, mark
     pairs whose diagonals differ, or mark on the unedited subdivision-2
     icosphere, where two non-adjacent vertices never share four neighbours.
   - Three places mark on an already-edited mesh, so their counts may change:
     `test_flip_edges` case 9's second all-marked call (`caseEmptyAndHalo`,
     `mask2`), case 12, and the flip passes in `test_collapse_edges` case 14.
     Whatever they assert must still pass.
   - If a literal fails, stop and report it. Do not change it.
5. **The new case**, in the style of case 14 (`conflictRun`, `:1393`), runs
   under both relations with the permissive policy:
   - **Success:** `flatBipyramidSoup` with `(2,3)` and `(5,6)` marked gives
     `accepted == 1` and `rejectedConflict == 1` under both relations, and the
     mesh passes `checkAll` and `checkKeyTables`. The winner is whichever edge
     the priority picks; assert it is the same at every rank count by comparing
     the gathered flip set against an `MPI_COMM_SELF` run.
   - **Failure direction:** a same-diagonal pair whose higher-priority member
     fails geometry. Then the other member is accepted and the loser counts
     `rejectedGeometric`, not `rejectedConflict`. Build it by perturbing the
     flat bipyramid so the two ring edges differ in length and in new-face
     quality, and set `minQuality` between the two. Check the fixture with a
     Python model of both metrics before the build, as M1-U2 did, and have the
     case assert its own premise from the soup coordinates, as case 13 does.
   - **Unchanged:** case 3's existing-diagonal rejection still gives
     `rejectedDuplicateEdge == 1`.
6. Build only the four test executables and run the targeted set, as M1-U1 and
   M1-U2 did (Log: `## M1-U1`, `## M1-U2`). Follow Tessera's `CLAUDE.md`.

**Exit criterion:** the same targeted run as M1-U2, after building the same four
executables. Run
`flux batch scripts/tuolumne/run_unit_tests.flux regression -R "^(collapse_edges|flip_edges)_"`
from `../tessera` with `TESSERA_REPO` exported. It must report
`100% tests passed, 0 tests failed out of 20`: `flip_edges` and
`collapse_edges`, SERIAL and HIP, np1–5. The new case must print identical lines
in every launch, showing all three of:

- the same-diagonal pair accepts one flip under each relation;
- the higher-priority member that fails geometry does not veto the other;
- case 3 still reports `rejectedDuplicateEdge=1`.

**Met.** Tessera `33dc6b9` on `mesh-quality-auto`, committed before the submit.
Job `f3hxYnUALpnB` (`pdebug`, COMPLETED rc 0, 176 s) reported
`100% tests passed, 0 tests failed out of 20`: all twenty cases ran,
`flip_edges_{SERIAL,HIP}_np1`–`np5` and `collapse_edges_{SERIAL,HIP}_np1`–`np5`.
Only the four test executables were built (rc 0, first try), and the only
warnings are the 32 accepted `-Wnontrivial-memcall` ones from
`Tessera_HaloRebuild.hpp:99`. Every existing case passes at its unchanged
literals. Among them, case 3 printed `requested=1 rejectedDuplicateEdge=1
accepted=0` in all twenty `flip_edges` outputs. The three calls on an
already-edited mesh (case 9's `mask2`, case 12, `collapse_edges` case 14) pass,
and case 14 printed the same `F=1244`/`1640` and `F=1236`/`1664` lines as at
`05b5942`. The new case is `test_flip_edges` case 15, with
`maxNormalDeviation = π`. It printed the same four lines in all twenty
`flip_edges` outputs, Serial and Default spaces alike:

- **Same-diagonal pair**: `flatBipyramidSoup` (V 8, E 18, F 12) with `(2,3)` and
  `(5,6)` marked, `minQuality = 0`. Both would create the axis `(0,1)`. Under
  `SharedFace` and `EndpointInQuad` alike: `accepted=1`, `rejectedConflict=1`,
  flip set `{(2,3)→(0,1)}`, identical to the `MPI_COMM_SELF` run. `(2,3)` wins
  on squared length `0.99999999999999978` against `0.99999999999999922`.
  `checkAll` and `checkKeyTables` pass.
- **Pushed pair**: a copy with ring vertex 5 moved to radius 2, the same two
  edges marked, `minQuality = 0.8`. `(5,6)` has the higher priority (squared
  length `2.9999999999999987`) but its worst new-face radius ratio is
  `0.73625936037355844`, against `0.87969658092961811` for `(2,3)`. Under both
  relations: `accepted=1` `{(2,3)}`, `rejectedGeometric=1`, `rejectedConflict=0`.
  The case asserts these premises from the soup coordinates.

The line references in this entry's **Fill in**, **Reference** and **Do** stay
against `05b5942`. At `33dc6b9`, `DupQuery`, `DupStatus` and `DupReply` are
`Tessera_EdgeFlip.hpp:444-469`, the geometric evaluation (2a) is `:744-777`, the
duplicate and claim round (2b) is `:779-838`, the verdict attribution (2c) is
`:840-857`, and the header's step-2 `DUPLICATE` text is `:142-178`. Log:
`## M1-U3`.

---

### T4d1 — the three `SurfaceMesh` adapters — **DONE**

**Depends on:** none.

**Fill in:** `SurfaceMesh::{collapseEdges, flipEdges, compact}` in
[src/Beatnik_MeshInterface.hpp](../src/Beatnik_MeshInterface.hpp) at
[:1518-1543](../src/Beatnik_MeshInterface.hpp#L1518-L1543),
[:1545-1567](../src/Beatnik_MeshInterface.hpp#L1545-L1567),
[:1569-1584](../src/Beatnik_MeshInterface.hpp#L1569-L1584), and
`MeshEditReport` ([:373-391](../src/Beatnik_MeshInterface.hpp#L373-L391));
the new unit member `tests/unit_tests/Beatnik_Test_MeshEdits.cpp`; and
`scripts/tuolumne/t4d1_mesh_edits.flux`, a two-node batch script modeled on
[scripts/tuolumne/t4_fmm_vs_direct.flux](../scripts/tuolumne/t4_fmm_vs_direct.flux)
and carrying `# flux: --bank=unm`, that runs `Beatnik_Test_MeshEdits` at ranks
1, 2, 3, 4, 5 and 6 in one job. `scripts/tuolumne/unit_tests.flux` cannot serve:
it allocates one node (`:3`) while each launch asks for `ceil(n/4)` nodes, so 5
and 6 ranks cannot launch (`t4_fmm_vs_direct.flux:22-38`), and it carries no
`--bank` directive.

**Reference:** `Tessera::{collapseEdges, flipEdges, compact,
compactAndRenumberGids}` and their policies and result structs, in the three
headers named under T4d. `SurfaceMesh::splitEdges`
([:1502-1516](../src/Beatnik_MeshInterface.hpp#L1502-L1516)) is the adapter shape to match.

**Callers of the signatures this changes:** **none** — both edge entry points
carry a placeholder `template <class EdgeListView>`, `flipEdges` returns `int`,
and no call site of any of the three exists in `src/`, `examples/` or `tests/`.
`MeshEditReport`'s existing fields are read by the split path only, and this
task adds fields without changing those. Changing the signatures is therefore
free, and doing it now is what keeps T4d2/T4d3/T4d5 from each inventing a mask
type.

**Do:**

1. Replace `const EdgeListView&` with `const std::vector<char>&` sized
   `ownedEdgeCount()` on both, per the conventions table, and make both return
   `MeshEditReport`.
2. Give both edge adapters a trailing policy parameter, templated and defaulted
   to Tessera's default policy, forwarded verbatim — the policy is chosen by the
   caller (T4d2, T4d3, T4d5), never inside the adapter.
3. Extend `MeshEditReport` with the counters `CollapseResult` and `FlipResult`
   report (`requested`, `accepted`, each `rejected*`, and for collapse the three
   `*Removed`) — the global counts, which Tessera already reduces. Then **assert
   the closed-manifold identities** Tessera's header states —
   `verticesRemoved == accepted`, `edgesRemoved == 3*accepted`,
   `facesRemoved == 2*accepted` (`Tessera_EdgeCollapse.hpp:337-341` at `05b5942`) — as the
   cheapest end-to-end check that the connectivity rewrite closed. Fail loudly;
   do not log and continue.
4. Rewrite the three doc comments as the adapters they now are. State in
   `compact()`'s that the remesh cycle does **not** call it (`collapseEdges`
   compacts internally) and that it exists for T4d6j and for tests. The
   `flipEdges` comment describes the reference's flip
   (`dynamic_remesh.py:408-458`), not what Tessera checks; replace it with
   Tessera's tests and point at Read this first #6.

**Exit criterion:** a new `unit`-tier member
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`, registered in
`BEATNIK_UNIT_TEST_SOURCES`
([tests/unit_tests/CMakeLists.txt:42](../tests/unit_tests/CMakeLists.txt#L42)), passes at
ranks 1, 2, 3, 4, 5 and 6 in one job of `scripts/tuolumne/t4d1_mesh_edits.flux`
(this task sweeps 1–6 rather than the Conventions table's 1-and-4 default): on the default
icosphere it collapses a named edge set, flips another, and compacts, and checks
(a) the three V/E/F identities, (b) `V - E + F == 2` globally afterwards,
(c) gid preservation across `compact()`, and (d) the **failure direction** — a
mask marking a boundary-of-nothing/link-condition-violating edge is *rejected*
and shows up in `rejectedLinkCondition`, with `accepted` unchanged, rather than
producing a torn mesh.

**Met.** Commit `14764a5`, job `f3iArVXa12CF` (`pdebug`, two nodes, COMPLETED
rc 0, 68 s): `[t4d1] SUMMARY: PASS (6/6 rank counts)`, and every one of the 21
launched ranks across np 1–6 printed `[PASS] Beatnik_Test_MeshEdits (86/86
checks)` on the HIP execution space, with identical counts. On the
subdivision-2 icosphere (V 162, E 480, F 320), six named collapses are first
all rejected into `rejected_quality` under a `minQuality = 1.1` policy (the
policy is forwarded), then all accepted under the permissive one: removed
V/E/F `6/18/12`, mesh V 156 E 462 F 308 — (a) holds both on the report and on
the global counts, (b) `V - E + F = 2`. Eight named flips are all accepted with
V/E/F and owned face gids unchanged. (c) `compact()` removes `0/0/0` with the
gid space `962` before and after against 926 live entities, and owned face
gids are identical. (d) After two spoke flips bring a vertex to valence 3, the
ring edge `(x,y)` whose endpoints share 3 neighbours (asserted) is rejected
alone with `rejectedLinkCondition=1`, `accepted=0` and the mesh unchanged; with
a valid antipodal edge beside it, `accepted=1`, `rejectedLinkCondition=1`, and
the mesh goes to V 161 E 477 F 318. Log: `## T4d1`.

---

### T4d2 — `DynamicRemesh::collapseShortEdges` — **DONE**

**Depends on:** T4d1, M1-U1.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:892](../src/Beatnik_DynamicRemesh.hpp#L892);
the sizing-field recompute and the `const int collapses = 0` placeholder inside
the pass loop at [:1020-1032](../src/Beatnik_DynamicRemesh.hpp#L1020-L1032).

**The `Solver` rejection stays.** The collapse rejection at
[src/Beatnik_Solver.hpp:797-817](../src/Beatnik_Solver.hpp#L797-L817) is deleted
by T4d6j, not here. So after this task `collapseShortEdges` is reachable from
`DynamicRemesh::remesh` but not from a `Solver` run with
`collapse_factor > 0`. The only `Solver` configuration that reaches it has
`collapse_factor <= 0`, which the T4b regression member
`Beatnik_Test_DynamicRemeshSplit` runs (`makeParams`,
[tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:350](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L350)).
On that path the pass must return 0 before any mutation, so the gate trajectory
is unchanged.

**Reference:** `dynamic_remesh.py::collapse_short_edges` (361-405), its two
safety predicates (509-516, 519-549), and the cycle's *second* sizing-field
recompute (`dynamic_remesh.py:158`).

**Do:**

1. Recompute the sizing field **before** the collapse pass — `vertexTargetEdgeLength`
   again, on the post-split mesh. Without it the collapse pass undoes the split
   pass.
2. Build the mask: an owned edge is a candidate iff
   `length < collapse_factor * max(min(target[i], target[j]), h_min)`.
3. Rank candidates by the reference's key — `length / max(local_target, 1e-300)`,
   ascending — and truncate to `max_collapses_per_pass`
   ([src/Beatnik_Params.hpp:678](../src/Beatnik_Params.hpp#L678), 120). The cap is
   mask construction; there is no per-call cap argument (M1-R3).
4. `haloExchange()` first. Then call `SurfaceMesh::collapseEdges` with a policy
   deriving from `Tessera::DefaultCollapsePolicy` that overrides M1-U1's
   `quality` hook with the reference metric `4√3·A/Σℓ²` — the same formula as
   `SurfaceOperators::triangleQuality`
   ([src/Beatnik_MeshGeometry.hpp:305](../src/Beatnik_MeshGeometry.hpp#L305)),
   on host pointers — and sets `minQuality = --remesh-min-quality`
   ([src/Beatnik_Params.hpp:654](../src/Beatnik_Params.hpp#L654), 0.18),
   `maxNormalRotation = std::acos( 0.1 )`, `t = 0.5`. Read this first #6 derives
   each value; do not leave any at Tessera's default.
5. Report `collapses` into `RemeshDiagnostics` and decide, explicitly, whether the
   pass re-bases the AMR reference state (it deletes faces and moves a vertex, so
   the convention applies).

**Exit criterion:** a `unit`-tier sub-case in `Beatnik_Test_MeshEdits.cpp` (or a
new member) shows, at `BEATNIK_UNIT_RANKS` 1 and 4 (conventions table): the global face count
**falls** on a mesh given an artificially large `collapse_factor`, by the **same
amount at both rank counts**; `V - E + F == 2` after the pass; and the failure
direction — `--remesh-collapse-factor 0` produces `collapses == 0` with no
mutation, and `--remesh-max-collapses 0` does **not** disable the pass (it means
unlimited), asserted so the rejection's warning about that lever is carried by a
test of the pass itself.

**Met.** Commit `848e009`, job `f3iBGQsreZPm` (`pdebug`, two nodes, COMPLETED
rc 0, 66 s, `scripts/tuolumne/t4d1_mesh_edits.flux` at its default ranks 1–6,
which covers 1 and 4): `SUMMARY: PASS (6/6 rank counts)`, and all 21 ranks
printed `[PASS] Beatnik_Test_MeshEdits (122/122 checks)` on HIP with identical
counts. Case 3 works on a radially perturbed subdivision-2 icosphere (V 162,
E 480, F 320), which breaks length ties. At `collapse_factor = 0` it finds no
candidates and changes nothing: V/E/F, face gids, every local position and the
mesh generation are the same afterwards. At factor 6 with
`max_collapses_per_pass = 0` (unlimited) it finds 240 candidates and requests
all 240. It accepts 4 (236 `rejected_conflict`), F falls 320 → 312, V − E + F =
2. With the cap at 25 it requests exactly 25, reports capped, and accepts 4.
`remesh()` with splitting switched off collapses the same 4. At factor 0 it
leaves 320 faces. All of these are literals asserted in the test. The
regression gate was not re-run; the factor-0 no-mutation check is the evidence
that it is unaffected. Log: `## T4d2`.

---

### T4d3 — the two quality-flip entry points — **DONE**

**Depends on:** T4d1, M1-U3 (two quality flips in one round must not create the same diagonal).

**Fill in:** `DynamicRemesh::flipEdgesForQuality`
([src/Beatnik_DynamicRemesh.hpp:1112](../src/Beatnik_DynamicRemesh.hpp#L1112)),
`MeshQuality::improveConnectivityByFlips`
([src/Beatnik_MeshQuality.hpp:210](../src/Beatnik_MeshQuality.hpp#L210)), the
`diag.flips += 0` placeholder ([src/Beatnik_DynamicRemesh.hpp:1244-1246](../src/Beatnik_DynamicRemesh.hpp#L1244-L1246)),
and the `const int flips = 0` on the refine branch
([src/Beatnik_Solver.hpp:456-465](../src/Beatnik_Solver.hpp#L456-L465)).

**The two `Solver` flip rejections stay.** The rejections at
[src/Beatnik_Solver.hpp:746-751](../src/Beatnik_Solver.hpp#L746-L751)
(`--flip-passes > 0` under `--refine-every`) and
[:828-837](../src/Beatnik_Solver.hpp#L828-L837)
(`--remesh-flip-min-gain < 1e12` under `--dynamic-remesh`) are deleted by T4d6j,
together with the two `regression`-tier cases that assert them. So after this
task both passes are wired into their call sites but a `Solver` run reaches them
only at their disabling values: the refine branch at `--flip-passes 0`, where
`improveConnectivityByFlips` returns 0 before any mutation
(`mesh_solver.py:1713-1715`), and the remesh pass loop at
`flip_min_gain = kFlipsDisabledMinGain`
([src/Beatnik_Params.hpp:615](../src/Beatnik_Params.hpp#L615)), which the T4b
member `Beatnik_Test_DynamicRemeshSplit` runs
([tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:354](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L354)).
On those paths the gate trajectory must be unchanged.

**Reference:** `dynamic_remesh.py::flip_edges_for_quality` (408-457) for the
remesh criterion and its single sweep; `mesh_solver.py::improve_mesh_connectivity_by_edge_flips`
(1704-1772) for the second entry point's criterion, its `max_passes` sweeps and
its `reset_reference` argument.

**Do:**

1. Write the **gain predicate once**, with the quality floor as a parameter, and
   give both call sites the mask it produces: mark an edge iff its quad
   `(a,b,c,d)` yields finite `q_new`, `min(q_new) >= floor` where a floor is
   given, **and** `min(q_new) > min(q_old) * (1 + gain)`, with `q` the reference
   metric (`SurfaceOperators::triangleQuality`, not the radius ratio).
   - `flipEdgesForQuality` passes `floor = min_quality`
     ([src/Beatnik_Params.hpp:654](../src/Beatnik_Params.hpp#L654), 0.18) and
     `gain = --remesh-flip-min-gain` (`:658`), as `dynamic_remesh.py:446-451`
     does.
   - `improveConnectivityByFlips` passes **no** floor: the reference's second
     entry point has only the gain test (`mesh_solver.py:1755-1758`). Its gain
     is the `min_gain` argument. The refine-branch call site passes `1e-3` as a
     named constant citing `mesh_solver.py:1708`'s default, because the
     reference's call site (`run_adaptive_mesh_bubble.py:1440-1444`) passes no
     gain and no CLI option sets one.

   The reference has no normal test here, only a re-orientation of the children
   against the old faces' summed normal (`dynamic_remesh.py:434-443`), which
   Tessera's orientation-preserving rewrite already guarantees. The quad comes
   from `SurfaceMesh::faceAdjacency()`
   ([src/Beatnik_MeshInterface.hpp:1349](../src/Beatnik_MeshInterface.hpp#L1349)).
   Reading the neighbour face's positions makes this a *geometric* consumer, so
   it must check `non_resident == 0` rather than assume halo depth 2 supplies
   the quad ([:656-670](../src/Beatnik_MeshInterface.hpp#L656-L670)), and throw
   if it does not hold.
2. **Do not test whether `(c,d)` already exists** — Tessera rejects that flip
   exactly, into `rejectedDuplicateEdge` (Read this first #6).
3. Call `SurfaceMesh::flipEdges` with
   `Tessera::DefaultFlipPolicy{ maxNormalDeviation = π, minQuality = 0 }`. Do
   **not** express the gain through the policy — the policy offers absolute
   admissibility only, and its priority is longest-edge-first, not gain-ordered.
   Tessera's default conflict relation (two flips may not share a face) is the
   reference's here (`dynamic_remesh.py:416-421`).
4. **One reference sweep is several `flipEdges` rounds**, because one Tessera
   round accepts an independent set that is not maximal, where a reference sweep
   tests every edge of one edge-map snapshot and never tests a face it already
   rewrote. Within a sweep:
   - Keep a sweep-wide set of **touched faces**, and leave out of the mask every
     owned edge with a touched incident face.
   - After each round, add the two faces of every accepted flip to the touched
     set. Take the flips from `FlipResult::flipped`, gathered across ranks with
     `MPI_Allgatherv` and deduplicated by old key, because `flipped` lists only
     the flips the calling rank touches.
   - End the sweep when a round accepts nothing.

   A new diagonal's two faces are both touched, so no edge created in the sweep
   is tested in it. `flipEdgesForQuality` runs **one** sweep
   (`dynamic_remesh.py:408-457`). `improveConnectivityByFlips` runs up to
   `max_passes` sweeps, each with a fresh touched set, and stops after a sweep
   that flips nothing (`mesh_solver.py:1720-1768`). Do **not** loop rounds with
   a fresh mask until `accepted == 0`: that re-tests faces flipped earlier in the
   sweep and flips more than the reference.
5. Answer T4c's deferred question — **does the flip path re-base the AMR
   reference state?** The reference passes `reset_reference=(smooth_iters == 0)`
   at the refine call site (`run_adaptive_mesh_bubble.py:1440-1445`) and nothing
   at the remesh one. Record the decision and its reasoning in the log; the
   comment at [src/Beatnik_MeshQuality.hpp:200-209](../src/Beatnik_MeshQuality.hpp#L200-L209)
   is where T4c parked it.
6. `--flip-passes` is **0** in the milestone configuration, so
   `improveConnectivityByFlips` has no milestone caller. Implement it anyway —
   T4d6j deletes the rejection at [src/Beatnik_Solver.hpp:746](../src/Beatnik_Solver.hpp#L746), and the
   path then becomes reachable — and say in the log that it is unexercised by any
   milestone run (the standing `relaxTangential` situation from T4c).

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4 shows the
global minimum triangle quality **rises or holds** across a sweep and never falls,
and the count of faces below `--remesh-min-quality` does not increase.
**Statistics are compared, never flip sets** (R7). Failure direction:
`--remesh-flip-min-gain 1e12` yields `flips == 0` and a byte-identical mesh,
which is what the T4b regression member already depends on.

**Met.** Commit `5fa0488`, job `f3iL4TUfQCCX` (`pdebug`, two nodes, COMPLETED
rc 0, 63 s, `scripts/tuolumne/t4d1_mesh_edits.flux` at its default ranks 1–6,
which covers 1 and 4): `SUMMARY: PASS (6/6 rank counts)`, and all 21 ranks
printed `[PASS] Beatnik_Test_MeshEdits (177/177 checks)` on HIP with identical
counts. Case 4 works on a subdivision-2 icosphere whose vertices are moved
tangentially and put back on the sphere (V 162, E 480, F 320; worst quality
0.0775, 3 faces below 0.18). One `flipEdgesForQuality` sweep at gain 1e-3 and
floor 0.18 runs 2 rounds, requests 59 and flips 44 (15 `rejected_conflict`);
the minimum quality rises to 0.2128 and the count below 0.18 falls to 0, with
V/E/F, χ and every face gid unchanged. `remesh()` with split and collapse off
reaches it through the repair gate and flips the same 44 in 2 rounds.
`improveConnectivityByFlips` at three passes runs 3 sweeps flipping 44, 10 and
2 (56), the minimum rising to 0.2963. The failure direction: at gain 1e12,
`flipEdgesForQuality` and `remesh()` make no `flipEdges` call, and V/E/F, face
gids, every local position and the generation are unchanged; `max_passes = 0`
returns 0 with the same checks. All counts are literals in the test. The
regression gate was not re-run; the 1e12 and `max_passes = 0` no-mutation
checks are the evidence it is unaffected. Log: `## T4d3`.

---

### T4d4 — `DynamicRemesh::tangentialSmooth` — **DONE**

**Depends on:** none. It moves vertices and changes no connectivity; it was
parked under T4d only because it was unreachable while the flips were.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:1187](../src/Beatnik_DynamicRemesh.hpp#L1187)
and the `diag.smooth_steps += 0` placeholder
([:1299](../src/Beatnik_DynamicRemesh.hpp#L1299)), which becomes the call and
its count.

**The `Solver` rejection stays** (Conventions, "`Solver` rejections during
T4d"). Leave the `--remesh-smooth-iters > 0` rejection at
[src/Beatnik_Solver.hpp:838-845](../src/Beatnik_Solver.hpp#L838-L845), message
text included, and the `tangentialSmooth` `checkRejected` case in
[tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:699-701](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L699-L701)
that asserts it. T4d6j deletes both. So after this task a `Solver` run reaches
`tangentialSmooth` only at `smoothing_iterations = 0`, which the T4b member
`Beatnik_Test_DynamicRemeshSplit` runs, and where the pass must return 0 before
moving any vertex (`dynamic_remesh.py:463-465`). On that path the gate
trajectory must be unchanged.

**Reference:** `dynamic_remesh.py::tangential_smooth_vertices` (459-490) — **not**
`mesh_solver.py::improve_mesh_quality_tangential`, which is T4c's landed
`MeshQuality::relaxTangential`.

**Do:**

1. Reuse `relaxTangential`'s kernel; do not write a second umbrella operator.
2. Add what the reference has and `relaxTangential` does not (Read this first
   #5): per iteration, compute *trial* positions, reduce the global minimum
   triangle quality over the trial mesh, and **reject the whole iteration and
   stop** if `new + 1e-14 < max(min_quality, 0.85 * old)`, with quality the
   reference metric (`SurfaceOperators::triangleQuality`). Return the count of
   **accepted** iterations as `smooth_steps`. The reduction is global, so every
   rank must take the same decision — reduce, then branch on the reduced value.
3. Skip vertices with fewer than three neighbours, as the reference does.
4. Keep the normal projection. Without it this is Laplacian smoothing of the
   interface, which shrinks the bubble — a failure that reads as excessive
   numerical dissipation rather than as the geometry bug it is.

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4, on the
default icosphere unless stated:

- **Shape preserved, in T4c's form.** Each accepted iteration's displacement is
  tangent to the vertex normals that iteration used:
  `max|Δx·n| ≤ 1e-13·max|Δx|`, both maxima global over owned vertices. The
  relative volume change of the pass is printed, and asserted to be at least
  `1e3` times smaller in magnitude than that of one *un-projected* iteration at
  the same relaxation, built in the test from the same public operators
  (`graphLaplacianVector` without `projectTangent`), as
  `Beatnik_Test_TangentialRelaxation` group 5 does. The volume is not bounded
  absolutely: the projection cancels it to first order only, and the reference
  itself changes it by `-4.338e-07` relative in one iteration at the default
  relaxation 0.04 on this mesh.
- **Quality.** The mean triangle quality rises, and each accepted iteration
  satisfies the guard's own bound,
  `min_after + 1e-14 ≥ max(min_quality, 0.85·min_before)`. The minimum is
  printed and not required to hold: the reference lowers it on this mesh, from
  `0.97727413140883002` to `0.97725307555529162`.
- **The guard in both directions.** A mesh constructed so the trial worsens
  quality past the `0.85` bound returns `smooth_steps == 0` with positions
  **unchanged**, and a healthy mesh returns
  `smooth_steps == smoothing_iterations`.

**Met.** Job `f3iLqpsgPez3` (`scripts/tuolumne/t4d1_mesh_edits.flux`, ranks
1–6 on HIP, commit `4041aec`): `SUMMARY: PASS (6/6 rank counts)` and 21
`[PASS] Beatnik_Test_MeshEdits (232/232 checks)`, with case 5's integer and
bound results identical on every rank. Shape: over three accepted iterations
at relaxation 0.04 the per-iteration tangency ratio `max|Δx·n|/max|Δx|` is
`2.06e-14`, `6.73e-14` and `4.50e-14`, all under `1e-13`; the three-iteration
pass changes the volume by `-1.16637e-06` relative, against `-5.37065e-03` for
one un-projected iteration, a factor of `4605`. Quality: the mean rises
0.988529 → 0.989143 → 0.989697 → 0.990197; the minimum falls 0.97727413 →
0.97725308 → 0.97723276 → 0.97721316, inside each iteration's guard bound
(about 0.8307), and agrees with the reference's one- and three-iteration
minimum and mean to `1e-12` relative and its volume change to `1e-8`. One call at
three iterations returns 3 and lands within `5e-20` of the three single
calls. The guard in both directions: at `min_quality = 0` and relaxation 5.0
the test's own trial takes the minimum to 0.62203 (ratio 0.6365 < 0.85), and
`tangentialSmooth` returns 0 with every local position bit-identical and the
generation unchanged; the healthy mesh returns `smooth_steps == 3`.
`smoothing_iterations = 0` (directly, and through `remesh()` with the repair
gate firing) and relaxation 0 or negative return 0 with positions
bit-identical; through `remesh()` with flips and one smoothing iteration on
case 4's fixture, `flips = 44` and `smooth_steps = 1`.

---

### T4d5 — `MeshQuality::{valenceEqualizingFlips, isotropicCleanup}` and the cleanup call site — **DONE**

**Depends on:** T4d1, T4d3 (shares the flip adapter and the mask machinery),
M1-U2, M1-U3, and T4c (landed).

**Fill in:** the stubs `valenceEqualizingFlips` and `isotropicCleanup` in
`src/Beatnik_MeshQuality.hpp` (at `:536` and `:607` before this task); the two
call sites in `Solver::advanceOneStep` — the remesh branch
(`src/Beatnik_Solver.hpp:533-560` before this task) and the refine branch
(`:457-517` before this task).

**The `Solver` rejection stays** (Conventions, "`Solver` rejections during
T4d"). Leave the `--isotropic-cleanup` rejection at
[src/Beatnik_Solver.hpp:791-797](../src/Beatnik_Solver.hpp#L791-L797), message
text included, and the `isotropicCleanup` `checkRejected` case in
[tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:705-707](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L705-L707)
that asserts it. T4d6j deletes both. Wire both call sites anyway, so that
deleting the rejection is all T4d6j has to do to turn cleanup on. After this
task a `Solver` run reaches the call sites only with `--no-isotropic-cleanup`,
which every `regression` member already passes, so the gate trajectory must be
unchanged.

**Reference:** `mesh_quality.py::_valence_equalizing_flips` (44-87) and
`::isotropic_cleanup` (146-167); the call sites at
`run_adaptive_mesh_bubble.py:1452-1464` (refine) and `:1491-1504` (remesh).

**Do:**

1. `valenceEqualizingFlips`: a **valence-based** mask, not the gain mask. Mark the
   shared edge `(a,b)` of faces `f0` (opposite vertex `c`) and `f1` (opposite
   `d`) iff
   `|v_a-7| + |v_b-7| + |v_c-5| + |v_d-5| < |v_a-6| + |v_b-6| + |v_c-6| + |v_d-6|`,
   the child holding `a` and the child holding `b` satisfy `n·n_f0 ≥ 0.2` and
   `n·n_f1 ≥ 0.2` respectively on unit normals (`mesh_quality.py:76-79`), each
   child wound as Tessera writes it rather than as the reference's literal
   `(c,d,a)`, `(d,c,b)` (Deliberate deviations), and
   `min(q_new) >= 0.05` in the reference metric. Valence is the face-incidence
   count (`np.bincount(F.ravel())`), which on a closed manifold is the edge
   count Tessera's header says is local at a vertex's owner. Do **not** test
   whether `(c,d)` exists (Read this first #6). Call `SurfaceMesh::flipEdges`
   with T4d3's permissive policy plus `conflict = FlipConflict::EndpointInQuad`
   (M1-U2).
2. **One reference pass is several `flipEdges` rounds**, because one Tessera
   round accepts an independent set that is not maximal, where a reference pass
   sweeps every edge. Within a pass:
   - Take each vertex's valence and each face's normal **once, at the start of
     the pass**, keyed by gid. Flips preserve vertex and face gids.
   - Keep a pass-wide set of **touched vertex gids**, and leave out of the mask
     every owned edge with a touched endpoint.
   - After each round, add the four corners of every accepted flip to the touched
     set. Take them from `FlipResult::flipped`, gathered across ranks with
     `MPI_Allgatherv` and deduplicated by old key, because `flipped` lists only
     the flips the calling rank touches.
   - End the pass when a round accepts nothing.

   A face rewritten earlier in the pass has all its corners touched, so no later
   edge of it is tested: the pass-start normals are never stale where they are
   read. Run `--isotropic-cleanup-flips` (3) passes and stop early after a pass
   that flips nothing, as `mesh_quality.py:85-86` does.
3. `isotropicCleanup` = `--isotropic-cleanup-flips` (3) valence passes, then
   `--isotropic-cleanup-relax` (2) passes of the landed `tangentialRelaxation` at
   `--isotropic-cleanup-weight` (0.4). Vertex count is unchanged and the shape is
   preserved, so no field is touched.
4. **The call site is two state rebuilds, not one.** The reference rebuilds the
   state inside `dynamic_remesh_state_with_material` (which re-bases the
   reference area/curvature and re-centres the potential against the new area
   weights) and **then rebuilds it again after the cleanup**
   (`run_adaptive_mesh_bubble.py:1502-1504` constructs a fresh
   `MeshPotentialZModelState`, whose `__post_init__` re-seeds both). So the
   Beatnik order is: `remesh` → `resetReferenceState` + `centerPotential` →
   `isotropicCleanup` → `resetReferenceState` + `centerPotential` →
   `projectToVolume`. Getting this wrong is invisible in a 20-step run and
   compounds over 600.
5. The cleanup fires whenever a remesh **ran**, not whenever it changed something
   (`:1491`, gated on `remesh_diag is not None`) — the same gate shape the volume
   projection already uses on that branch.
6. On the refine branch, the reference runs cleanup only when
   `refine_diag.marked_faces > 0` (`:1438`), inside the same block as the flips
   and the tangential pass, and it feeds the `repaired` gate through
   `args.isotropic_cleanup` — which
   [src/Beatnik_Solver.hpp:491-493](../src/Beatnik_Solver.hpp#L491-L493) already
   transcribes as `_params.cleanup.enabled`. Add the call; do not touch the gate.
   Follow it with the same state rebuild as the remesh branch,
   `resetReferenceState` + `centerPotential`, under `_params.cleanup.enabled`:
   the reference replaces the state after the cleanup with a fresh
   `MeshPotentialZModelState` (`run_adaptive_mesh_bubble.py:1462-1464`), whose
   `__post_init__` (`mesh_solver.py:138-163`) re-seeds the reference area and
   curvature and re-centres the potential. Rewrite the refine-branch comment
   ([src/Beatnik_Solver.hpp:491-498](../src/Beatnik_Solver.hpp#L491-L498)) that
   says nothing re-bases the reference state after the tangential pass: with
   cleanup enabled, the rebuild after the cleanup does.

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4: on a mesh
seeded with a high-valence patch, every accepted flip of a pass has no endpoint
touched earlier in that pass (checked from the gathered `flipped` lists); the
**valence histogram concentrates toward 6**
(the count of vertices with valence ≠ 6 falls); the volume is handled in T4d4's
form:

- **Flips:** the relative volume change of the valence passes is printed and
  not bounded. A flip on a curved surface changes the enclosed volume by the
  signed volume of its quad's tetrahedron, and the reference has no volume
  control there; its call sites restore volume with `project_state_to_volume`.
- **Relaxation:** each relaxation pass's displacement is tangent to the vertex
  normals that pass used, `max|Δx·n| ≤ 1e-13·max|Δx|`, both maxima global over
  owned vertices. The relative volume change of the relaxation passes is
  printed, and asserted at least `1e3` times smaller in magnitude than that of
  one *un-projected* relaxation pass at weight 0.4 on the same mesh, built in
  the test from the same public operators (`graphLaplacianVector` without
  `projectTangent`). The projection cancels the volume change to first order
  only, so it is not bounded absolutely.

And, in the failure direction,
`--no-isotropic-cleanup` leaves the mesh byte-identical while
`--isotropic-cleanup` does not, so the switch is demonstrably live. The
`Solver` still rejects `--isotropic-cleanup` after this task, so show this at
the unit tier on the call site's `_params.cleanup.enabled` gate, not through
`Solver::solve`. Statistics only, never flip sets (R7).

**Met.** Job `f3iWgRGrp42K` (`scripts/tuolumne/t4d1_mesh_edits.flux`, ranks
1–6 in one job, HIP): `SUMMARY: PASS (6/6 rank counts)`, every one of the 21
ranks `[PASS] Beatnik_Test_MeshEdits (282/282 checks)`, so ranks 1 and 4 are
both covered with one set of literals. Case 6, on case 3's perturbed icosphere
after one `SharedFace` scramble round in a 45° cap (35 vertices of valence ≠ 6,
maximum 10): three valence passes flip 6, 3 and 1 over 3, 1 and 1 rounds; no
gathered flip has an endpoint touched earlier in its pass or among another
same-round flip's corners (0 violations); the valence ≠ 6 count falls 35 → 20
(maximum 10 → 7); V/E/F and χ unchanged. Flips' relative volume change
`+1.2007e-3`, printed. Two relaxation passes at 0.4: tangency ratios `2.12e-15`
and `2.53e-15` (bound `1e-13`); volume change `3.1015e-05` against
`-5.3809e-02` for one un-projected pass, factor 1735 (asserted ≥ 1e3).
`enabled = false` leaves every local position, the face gids and the generation
bit-identical; `enabled = true` changes positions and generation and lands
within `1.1e-16` of the separate calls. The early stop ran 2 of 3 passes on a
fixture whose second pass flips nothing. The switch was shown through
`MeshQuality`'s own `CleanupParams::enabled` check, the struct the `Solver`
gates both call sites on; `Solver::solve` still rejects `--isotropic-cleanup`.

---

### T4d6j — the gid-renumbering cadence, built and submitted — **DONE**

**Depends on:** T4d2, T4d3, T4d4, T4d5. It is the last T4d task because it
deletes the `Solver` rejections that T4d2–T4d5 all leave in place (Conventions,
"`Solver` rejections during T4d").

**Fill in:** the remesh cycle in
[src/Beatnik_DynamicRemesh.hpp:1367-1448](../src/Beatnik_DynamicRemesh.hpp#L1367-L1448)
and `RemeshDiagnostics` ([:149-259](../src/Beatnik_DynamicRemesh.hpp#L149-L259));
the `SurfaceMesh` adapter `compactAndRenumberGids()` and the query
`gidSpace()` in
[src/Beatnik_MeshInterface.hpp:1914-1947](../src/Beatnik_MeshInterface.hpp#L1914-L1947),
modelled on `compact()` and returning a `MeshEditReport` with
`gid_space_{before,after}`; the cadence field
`RemeshParams::renumber_gids_every`
([src/Beatnik_Params.hpp:707](../src/Beatnik_Params.hpp#L707), `0` = never);
the new driver `tests/regression_tests/Beatnik_Test_Milestone1Run.cpp`, its
header `tests/regression_tests/Beatnik_Milestone1Params.hpp`, and its
entry in `BEATNIK_DRIVER_SOURCES`
([tests/CMakeLists.txt:608-634](../tests/CMakeLists.txt#L608-L634)); the new job
script `scripts/tuolumne/t4d6_gid_cadence.flux`; the replacement of
`RefineSplitEdges`' check 6 (see **Removes**); and the deletion of
**all five** T4d rejections in `Solver::requireSupportedConfiguration`, comments
included, in one change:

| Rejection | Lines | Left in place by |
| --- | --- | --- |
| `--flip-passes > 0` under `--refine-every`, naming `improveConnectivityByFlips` | [src/Beatnik_Solver.hpp:778-783](../src/Beatnik_Solver.hpp#L778-L783) | T4d3 |
| `--isotropic-cleanup`, naming `isotropicCleanup` | [:791-797](../src/Beatnik_Solver.hpp#L791-L797) | T4d5 |
| `--remesh-collapse-factor > 0`, naming `collapseShortEdges` | [:830-850](../src/Beatnik_Solver.hpp#L830-L850) | T4d2 |
| `--remesh-smooth-iters > 0`, naming `tangentialSmooth` | [:852-859](../src/Beatnik_Solver.hpp#L852-L859) | T4d4 |
| `--remesh-flip-min-gain < 1e12`, naming `flipEdgesForQuality` | [:861-870](../src/Beatnik_Solver.hpp#L861-L870) | T4d3 |

Also rewrite the comment block above the metric remesher's rejections
([:799-810](../src/Beatnik_Solver.hpp#L799-L810)), which says which thirds are
still rejected. Line numbers are as of T4d5.

**Submits:** `scripts/tuolumne/t4d6_gid_cadence.flux`, `pdebug`, 1 node
`--exclusive`, `-t 60m`, `--bank=unm`, the dev env. Every launch runs
`Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP}`. Three groups of launches, in
this order:
1. **The cadence run:** 600 steps at the milestone defaults (no knob turned off),
   at ranks 1 and 4 on HIP, reporting the gid-space high-water mark, the global
   vertex and face counts and each rank's current RSS every step, and each
   rank's peak RSS (`/usr/bin/time -v` inside `flux run`, as
   `milestone1_divergence.flux` does).
2. **The control:** the same run at rank 1 on HIP with renumbering switched off
   (cadence `0`), so the high-water ratio *must* grow. The cadence is therefore
   a `RemeshParams` field with `0` meaning "never", set from the driver's third
   positional.
3. **The T4d umbrella:** 50 steps at the milestone defaults at ranks 1 and 4 on
   SERIAL and HIP, reporting volume drift, the per-step minimum triangle quality,
   and cumulative `splits`, `collapses`, `flips` and `smooth_steps`.

Expected run time **about 40 minutes, an estimate**: no Beatnik dynamic-remesh
run has been timed. M1-G2's Python run took 707 s for 600 steps at about 1074
faces (log `## M1-G2`), so that figure is taken as the bound for each 600-step
launch (three launches, about 36 min), plus four 50-step launches. Beatnik's
frozen-mesh runs are 4–10x faster than Python's (log `## M1-D1`). Carry
`milestone1_divergence.flux`'s deadline guard so that a launch which cannot fit
the remaining budget is skipped loudly, not killed at the wall. Staged directory
`/p/lustre5/stewartj/beatnik/T4d6j/stage/` (Conventions, "Long jobs"). Output
goes to `/p/lustre5/stewartj/beatnik/T4d6j/out/` and to
`beatnik_t4d6.<jobid>.log` in the repo root.

**Removes:** the five rejections above, with their comments, and the
`regression`-tier cases that expect `solve()` to throw because of them:
- the four T4d `checkRejected` cases in
  [tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:696-707](../tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp#L696-L707)
  (collapse, smooth, flip, cleanup). The T4e proximity case at `:693-695` stays,
  so the member keeps a failure direction. Correct the file's header comment at
  `:21-40`, which says collapse and the flip third are unimplemented and that
  any other collapse factor is rejected. The recorded reference command at
  `:183-186` describes the gold run and stays as it is;
- `checkFailureDirection` in
  [tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp:379-417](../tests/regression_tests/Beatnik_Test_RefineSplitEdges.cpp#L379-L417),
  which asserts that `--flip-passes 2` is rejected, and the header's check 6
  (`:72-77`) that describes it. **Replace** both with a flips liveness check, so
  the member still guards the refine branch's flips against doing nothing
  (R15). A second 20-step solver, built from the same `makeParams()` with
  `flip_passes = 2`, must complete all four refinement passes with cumulative
  `flips > 0` summed globally, and Euler and conformity
  (`checkConformity`) must hold after every pass. It compares nothing
  against the Python, because `--flip-passes 2` changes the mesh the `kPy*`
  counts describe. The member's loud failure modes are then the exact integer
  identities of checks 2 and 3. No refine-branch rejection is left to target:
  the refiner's own throws (`balanceRedGreen`'s runaway,
  [src/Beatnik_AdaptiveMesh.hpp:930](../src/Beatnik_AdaptiveMesh.hpp#L930);
  the halo-residency check, `:1754`) cannot be reached from `SolverParams`.
  If the flips count is 0 at this configuration, stop and record it: that is
  a finding about `improveConnectivityByFlips`, not a literal to relax.

**The driver.** `Beatnik_Test_Milestone1Run` is a measurement driver in
`BEATNIK_DRIVER_SOURCES`, beside `Beatnik_Test_Milestone0Run`. It has no tier
label and no manifest line, and it gets per-backend binaries the way every test
source does. `adaptive_mesh_bubble` cannot serve, because
`adaptive_mesh_bubble.cpp:209` pins `Kokkos::DefaultExecutionSpace` and the
installed binary is HIP-only. `Milestone0Run` cannot serve either: it asserts a
constant mesh every step, and its header forbids new options. The new driver:
- its `makeParams()` is M1-G2's command line field for field (the milestone
  defaults, `--checkpoint-every-steps 10`). It lives in a new header,
  `tests/regression_tests/Beatnik_Milestone1Params.hpp`, which both this driver
  and M1-T2j's test include with a quoted `#include`. The generated per-backend
  translation unit includes each source by absolute path
  ([tests/CMakeLists.txt:349-355](../tests/CMakeLists.txt#L349-L355)), so a
  quoted include resolves next to the source and needs no CMake change. The
  header carries the BSD-3-Clause block;
- takes three positionals, documented in its file header as `Milestone0Run`
  documents its own: `argv[1]` `--steps`, `argv[2]` `--checkpoint-every-steps`
  (`0` = no checkpoints), `argv[3]` the gid-renumbering cadence (`0` = never);
- prints one line per step with `step`, `time`, global `V` and `F`, the
  gid-space high-water mark, volume drift against the initial volume, the
  global minimum triangle quality, the cumulative `splits`, `collapses`,
  `flips` and `smooth_steps`, and this rank's current RSS (`VmRSS` from
  `/proc/self/status`);
- asserts only that it reached its step budget, reporting an early non-finite
  stop as a failure, as `Milestone0Run` does. Every judgment is T4d6's, made
  from these lines.

**Reference:** `Tessera_Compact.hpp` on `compact()` versus
`compactAndRenumberGids()` (`:541-570`), and its statement that several Tessera
paths index a dense host array sized to the **max gid** (`:77-94`).
`scripts/tuolumne/milestone1_divergence.flux` for the deadline guard, the
per-rank RSS measurement and the provenance block.
[tests/regression_tests/Beatnik_Test_Milestone0Run.cpp](../tests/regression_tests/Beatnik_Test_Milestone0Run.cpp)
for the driver's shape: its header, positionals, step loop and early-stop
report.

**Do:** gids come from an `MPI_Exscan` onto a monotonically rising global count,
so a long split-and-collapse run grows the gid *space* without bound even at
constant mesh size, and `collapseEdges`'s internal `compact()` preserves gids.
Choose a renumbering cadence, document **why that cadence** (it costs a second
halo rebuild), and put the gid-space high-water mark into the remesh diagnostics
so the leak is observable rather than inferred. Then build (`spack install`),
run the gate, stage and submit.

**Exit criterion:** the dev-env `spack install` is rc 0 with no new warnings in
the changed files; `BEATNIK_UNIT_RANKS=1` and `BEATNIK_UNIT_RANKS=4 flux batch
scripts/tuolumne/unit_tests.flux` both pass; `flux batch
scripts/tuolumne/run_regression_minset.flux`, which takes about 11 minutes and
is waited on in the session, ends `[gate] PASS (label=regression)` over 60
launches with zero `[FAIL]` lines, with the five rejection cases deleted; the
staged directory exists and its `COMMIT` names the commit holding this task's
code; and the scheduler accepted `t4d6_gid_cadence.flux`, its job ID on the log's
`**Jobs:**` line.

**Met.** Commit `092e2e5`. The dev-env `spack install` (beatnik `ckhvawf`) is
rc 0 in 12m28s; its 75 warnings are all Canopy `-Wpass-failed` occupancy notes
(70 before; the new HIP driver translation unit instantiates the same
kernels), none in a changed file. `BEATNIK_UNIT_RANKS=1` (`f3iXQgeQwhKD`):
`[unit] SUMMARY: PASS (8/8 tests)`. `BEATNIK_UNIT_RANKS=4` (`f3iXQgvioo7D`):
7/8, every member green including `Beatnik_Test_MeshEdits` (282/282 on all
four ranks), the one failure being `Beatnik_Test_T2bOperators`'s deliberate
`comm_size == 1` guard (`expected 1, got 4`), pre-existing and recorded in
`framework.md` T4c. The gate (`f3iXR6G8cAF9`) ends `[gate] PASS
(label=regression)` over 60 launches with zero `[FAIL]` lines, with the five
rejections and their four `checkRejected` cases and `checkFailureDirection`
deleted; the new `RefineSplitEdges` check 6 reports cumulative flips 58 or 60
(rank-count dependent) over four passes. Stage
`/p/lustre5/stewartj/beatnik/T4d6j/stage/` holds both driver binaries and a
`COMMIT` naming `092e2e5`. The scheduler accepted `t4d6_gid_cadence.flux` as
`f3iXs3sWtvv3`; its output is T4d6's to judge.

---

### T4d6 — the gid space and resident memory are bounded — **DONE**

**Depends on:** T4d6j.

**Fill in:** the progress log's `## T4d6` section; a launch selector in
`scripts/tuolumne/t4d6_gid_cadence.flux` and its `diag` row (below); a README
"Known Issues" entry if the diagnostic attributes the growth to checkpoints. If
the output shows a defect, fix it inside T4d6j's **Fill in** and resubmit:
restage into `/p/lustre5/stewartj/beatnik/T4d6/stage/` and submit
`t4d6_gid_cadence.flux` again.

**Submits:** one diagnostic job, `t4d6_gid_cadence.flux` restricted to a single
`diag` row: 600 steps, 4 ranks, HIP, checkpoint cadence `0`, renumber cadence
`50`, `pdebug`, 1 node `--exclusive`, `--bank=unm`. The launch is selected by a
script argument, since the submit wrapper takes arguments and no options. It
runs from `/p/lustre5/stewartj/beatnik/T4d6/stage/` (the `092e2e5` binaries,
restaged) and writes under `/p/lustre5/stewartj/beatnik/T4d6/out/`. Expected
run time under a minute: the T4d6j job's 600-step 4-rank HIP launch took 17 s.

**Removes:** nothing.

**Reference:** the T4d6j job's log and its `out/` directory.

**Do:** confirm that every launch ran to its step budget (no deadline-guard skip,
no wall kill), then judge the criterion. Write the high-water ratio series, the
per-rank peak RSS and the umbrella's four counter totals into the log. Its
`**Affects:**` line names M1-T2j's walltime when the measured 600-step wall
differs from T4d6j's estimate.

The T4d6j job's 4-rank cadence run grows every rank's RSS by about 11–19 MB
over 600 steps at constant face count, only on checkpoint steps (every 10th),
while both 1-rank runs, which write the same checkpoints, stay flat between
face-count changes. Submit the `diag` launch to attribute it. If per-rank RSS
is flat at constant face count without checkpoints, the gid space is not the
cause: record the checkpoint-step growth as a README "Known Issues" entry
(pre-existing, 4-rank checkpoint writes) and as a follow-up in the log's
`**Affects:**` line. If RSS still grows without checkpoints, the per-rank RSS
clause fails: stop and report.

**Exit criterion,** judged from the T4d6j job's output:
- **The cadence run:** at ranks 1 and 4, the ratio of the gid-space high-water
  mark to the global vertex count does **not** grow monotonically with step
  index across the last third of the run (steps 400–600). Per-rank resident
  memory does not grow with step count at constant face count, judged on the
  `diag` launch (no checkpoints) at 4 ranks and on the cadence run at 1 rank.
- **The control:** the ratio **does** grow monotonically across the same third,
  which shows the measure detects the leak this task bounds. Monotone means
  non-decreasing at every step with a net rise over the window; the mesh is
  nearly static at the milestone defaults, so the ratio rises only when a
  split adds gids.
- **The T4d umbrella:** all four 50-step launches complete; volume drift is below
  `1e-10`; the minimum triangle quality stays above `--remesh-min-quality` at
  every step; `Solver::requireSupportedConfiguration` rejects **nothing** for that
  command line; and the cumulative `splits`, `collapses`, `flips` and
  `smooth_steps` are each non-zero. A run that completes with all four at zero has
  not exercised T4d and is M1-R7, not a pass. Compare **statistics** against the
  reference, never the flip or collapse set (R7).

Record both the ratio series and the RSS series in the progress log. This is the
first thing to suspect if a long run's memory grows with step count rather than
with face count.

**Met.** Judged from `f3iXs3sWtvv3` (stage `T4d6j/stage`, `COMMIT` `092e2e5`;
COMPLETED rc 0, `launched=7 skipped=0`, every launch `completed=` its budget) and
two diag jobs from `T4d6/stage` (the same binaries, sha256-identical):
`f3iYLanaSJHd` (`diag`) and `f3iYMv1J4fSw` (`diag` and `diag0`), both rc 0.
**Cadence:** at ranks 1 and 4 the ratio over steps 400–600 goes 5.984645 →
5.980843, with two drops (renumberings at 400 and 550) and one rise (the split
at 516), so it does not grow monotonically; the gid space falls back to
`3104–3122` at every renumbering. **Control:** non-decreasing at every step from
6.806142 (step 398) to 6.808429 (step 516), a net rise, which meets the clause
as the user reads it (the same series at 4 ranks in `diag0`). **Per-rank RSS:** at 1
rank it is flat between face-count changes except two 264 kB steps (+1.26 MB
over 600 steps). At 4 ranks without checkpoints it still grows 6.4–9.5 MB
per rank, 3.7–7.7 MB of it at constant face count, by the same amount with
renumbering every 50 (`diag`) and never (`diag0`). The gid space is flat
throughout. So, by the user's ruling, the clause is met with respect to the gid leak, and the
growth is recorded as a pre-existing README Known Issue (follow-up on the log's
**Affects:** line). The second diag row (`diag0`, renumber `0`) is the
user's addition to **Submits**. **Umbrella:** all four 50-step launches complete,
`requireSupportedConfiguration` rejects nothing, cumulative splits / collapses /
flips / smooth_steps are 364 / 8 / 76 / 3 in every launch, the minimum quality is
never below 0.6939 (`--remesh-min-quality` 0.18), and the largest |volume drift| is
3.62e-12 (step 1; 3.1e-14 after it), below `1e-10`. Measured 600-step walls:
13 s (np1), 17 s (np4), 10 s (control), 16–20 s (4-rank diag).

---

### M1-T2j — the milestone comparison test, registered and submitted at 600 steps — **DONE**

**Depends on:** M1-U1, M1-U2, M1-U3, T4d1, T4d2, T4d3, T4d4, T4d5, T4d6, M1-G2,
M1-D1, and **M0-T1** (**DONE**, [`milestone0.md`](completed/milestone0.md)). M0-T1
**created** the `milestone` tier: the CMake wiring, `beatnik_milestone_manifest.txt`,
the installed-path gold-data rules, `scripts/tuolumne/run_milestone.flux` at
ranks 1 and 4 on SERIAL and HIP, and the `docs/testing.md` / CLAUDE.md text saying
the tier does not gate. This task adds a member and changes none of that.

**Fill in:**
- `tests/regression_tests/Beatnik_Test_Milestone1Bubble.cpp`, new.
- The cleanup flip count, which no diagnostic carries:
  `MeshQuality::isotropicCleanup`
  ([src/Beatnik_MeshQuality.hpp:937-943](../src/Beatnik_MeshQuality.hpp#L937-L943))
  returns `void` and discards the global count `valenceEqualizingFlips`
  returns (`:652`), and the remesh `flips` diagnostic does not include the
  valence flips. Change `isotropicCleanup` to return that count as `int`
  (`0` when disabled). In `Solver`, add the cumulative global sum over both
  call sites
  ([src/Beatnik_Solver.hpp:511](../src/Beatnik_Solver.hpp#L511),
  [:568](../src/Beatnik_Solver.hpp#L568)) and the accessor
  `long long cleanupFlips() const`, beside `_edge_flips` (`:1113`) and
  `edgeFlips()` (`:739`). **Callers of `isotropicCleanup`:** those two call
  sites and `tests/unit_tests/Beatnik_Test_MeshEdits.cpp:1888` and `:2041`,
  which discard the result and need no change.
- In [tests/CMakeLists.txt](../tests/CMakeLists.txt): the source in
  `BEATNIK_MILESTONE_TEST_SOURCES` (`:429`); the
  `_beatnik_args_Beatnik_Test_Milestone1Bubble_abs`/`_rel` pair beside milestone
  0's (`:475-513`), naming the gold **directory** and the comparator; and the
  `install()` rule for the M1-G2 gold directory, with its `FATAL_ERROR`, beside
  milestone 0's (`:828-858`). A milestone test installed without its gold set is
  not installed.
- `scripts/tuolumne/m1_milestone_run.flux`, new. It runs only
  `Beatnik_Test_Milestone1Bubble_MPI_{SERIAL,HIP}`, from a staged directory
  named by its first argument, at ranks 1 and 4, plus the two failure-direction
  launches below. It is modelled on `run_milestone.flux`, including the
  `[milestone] PASS|FAIL … np=<N> in <S>s` lines and the `SUMMARY` line.
  `run_milestone.flux` resolves binaries from the live view, so it is not used
  for this job.
- The `milestone` entry in [docs/testing.md](../docs/testing.md#the-tiers): the
  new member, what it asserts, the member count and the launch count. CLAUDE.md
  defers the member list to that file, so it needs no edit.
  `scripts/tuolumne/submit_milestone_split.sh` already carries the member's
  SERIAL and HIP rows, marked `pending=M1-T2j` with an estimated `-t 40m`. They
  are submitted as soon as `spack install` puts the member in the installed
  manifest, so the script needs no edit either.

**Submits:** `scripts/tuolumne/m1_milestone_run.flux
/p/lustre5/stewartj/beatnik/M1-T2j/stage`, `pdebug`, 1 node `--exclusive`,
`-t 12m`, `--bank=unm`, the dev env. The job makes four comparison launches
(SERIAL and HIP, at ranks 1 and 4), then the two failure-direction launches at
rank 1 on HIP. Expected run time **about 6 minutes**, sized from T4d6's
measured 600-step walls at the milestone defaults (10–20 s per launch at 1 or
4 ranks on HIP, log `## T4d6`) plus 61 `compare_output.py` calls of about
0.5 s per comparison launch; `-t 12m` is about 2x that, with per-launch
deadline-guard estimates summing to 540 s of a 600 s budget (log
`## M1-T2j`). Staged directory
`/p/lustre5/stewartj/beatnik/M1-T2j/stage/`, holding the manifest-relative
layout of the member's binaries, `regression_tests/<M1-G2 gold dir>/` and
`regression_tests/compare_output.py`. Output goes to
`/p/lustre5/stewartj/beatnik/M1-T2j/out/<target>/` and to
`beatnik_m1_milestone.<jobid>.log` in the repo root.

**Removes:** nothing.

**Reference:** what M0-T1 built: the milestone registration loop
([tests/CMakeLists.txt:515](../tests/CMakeLists.txt#L515)), plus the regression
tier's registration loop (`:323-385`) and gold-set `install()` rules
(`:810-826`) that it copied. Read the M0-T1 entry in
[`milestone0-progress-log.md`](completed/milestone0-progress-log.md) first: if
M0-T1 departed from its stated plan, this task is where that surfaces. For the
test, read
[tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp)
end to end:
- `makeParams()` (`:308-378`) expresses a Python command line as a
  `SolverParams`. For this test, include
  `Beatnik_Milestone1Params.hpp` (T4d6j), whose `makeParams()` already is
  M1-G2's command line.
- `goldForStep()` (`:263-286`) finds a step's gold file by its `_step%07d.npz`
  suffix rather than by rebuilding a filename from a time, because the time is
  under test.
- The `compare_output.py` subprocess wrapper `runComparator()` (`:291-304`)
  demands exit status **exactly 1** for a mismatch, never 2, so a mis-resolved
  path cannot masquerade as a pass.

**Do:**

1. `makeParams()` is the milestone command line: the four switches at their
   defaults, `--checkpoint-every-steps 10`, and `--steps 600`, M1-G2's full
   depth. This run *measures* where the codes part; M1-A1j fixes the depth the
   test asserts.
2. Compare every checkpointed step through step 600 against a compiled-in
   per-step tolerance table, initialized to the comparator's defaults
   (`--rtol 1e-10 --atol 1e-12`) at every step. **Continue past a failing step.**
   Print one line per step with the comparator's verdict and each field's
   `max|e|`, plus the vertex and face counts of both files, and exit failing if
   any step failed. M1-A1j edits the table, not the structure.
3. Assert the trajectory is *live*: cumulative `splits`, `collapses`, `flips`,
   `smooth_steps` and `Solver::cleanupFlips()` all non-zero by the end (M1-R7), and the
   per-step `time` series matches the gold's. A fixed-dt run reproduces neither.
4. Report the wall time and the face-count trajectory in the test's output.
   They tell the next session whether a deeper compare depth is affordable.
5. Failure direction, selected by an argument the test accepts for this purpose:
   (a) a later step compared against the **step-0** gold exits exactly 1 from the
   comparator, a detected mismatch rather than a load error; (b) a run with
   `--no-isotropic-cleanup` forced fails the comparison rather than passing with
   a different mesh.
6. Build (`spack install`), run the gate, stage and submit.

**Exit criterion:** the dev-env `spack install` is rc 0 with no new warnings in
the changed files. The installed `beatnik_milestone_manifest.txt`
(`share/Beatnik/tests/` in the dev view) lists the member's three backend lines
beside milestone 0's twelve, and the installed `beatnik_gate_manifest.txt` still
lists 15. `flux batch scripts/tuolumne/run_regression_minset.flux`, waited on in
the session, ends `[gate] PASS (label=regression)` over 60 launches with zero
`[FAIL]` lines. The staged directory exists and its `COMMIT` names the commit
holding this task's code. The scheduler accepted `m1_milestone_run.flux`, its job
ID on the log's `**Jobs:**` line.

**Met.** Dev-env `spack install` rc 0 (13m42s); every one of the build's 80
warnings is the pre-existing Canopy HIP `-Wpass-failed` occupancy note, none in
a changed file. The installed `beatnik_milestone_manifest.txt` lists 15 lines,
milestone 0's twelve plus `Beatnik_Test_Milestone1Bubble_MPI_{SERIAL,OPENMP,HIP}`;
`beatnik_gate_manifest.txt` lists 15. The installed `Beatnik_Solver.hpp` and
`Beatnik_MeshQuality.hpp` are byte-identical to commit `347c2ef`. Gate
`f3iYat4nhyXd` COMPLETED rc 0 in 7.9 min: `[gate] PASS (label=regression)`, 60
`[gate] ===` launches, zero `[FAIL]` lines, 42 `[PASS]` per member. Stage
`/p/lustre5/stewartj/beatnik/M1-T2j/stage/` holds both binaries, the comparator
and the 61-file gold set, each byte-identical to the dev view's, and `COMMIT`
names `347c2ef…` (porcelain count 4: none compiled). The scheduler accepted
`m1_milestone_run.flux` as `f3iYnNt9YPJP` (`pdebug`, `-t 12m`); the user
cancelled the first submission, `f3iYjQ4E4J8X` in `pbatch`, before it started.
M1-T2 judges the job's output.

---

### M1-T2 — read the 600-step measurement — **DONE**

**Depends on:** M1-T2j.

**Fill in:** the progress log's `## M1-T2` section. If the output shows a defect,
fix it inside M1-T2j's **Fill in** and resubmit from
`/p/lustre5/stewartj/beatnik/M1-T2/stage/`. Defects include a launch that did not
run its 600 steps, a liveness counter that reads zero, and a comparator exit
other than 0 or 1. A comparison mismatch is a measurement, not a defect.

**Removes:** nothing.

**Reference:** the M1-T2j job's log and its `out/` directory; M1-D1's
`vertices` `max|e|` series (log `## M1-D1`), which is the floating-point floor
on a frozen mesh; M1-G2's face-count table (log `## M1-G2`).

**Do:** for each of the four launches, record:
- the first failing step, or "none through 600";
- the field that failed first and its `max|e|`, set against M1-D1's floor at the
  same step;
- whether the vertex and face counts agree at that step, which is M1-R2's
  distinguishing measurement;
- whether `vertices` failed first and by the same magnitude as every other
  field, which is M1-R8's signature;
- the wall time and the face-count trajectory against M1-G2's.

**Exit criterion,** judged from the M1-T2j job's output:
- all four comparison launches ran to step 600, each with its `[milestone]` line
  and the job's `SUMMARY` line present;
- the liveness assertions hold in every launch;
- failure direction (a) exits exactly 1 from the comparator, and (b) fails the
  comparison;
- the log's `## M1-T2` section holds the per-launch record from **Do**, with an
  `**Affects:**` line naming M1-A1j.

**Met.** Job `f3iYnNt9YPJP` is INACTIVE, FAILED, rc 1, 230 s of runtime; the
stage's `COMMIT` is `347c2ef78b70…`, M1-T2j's code commit. The log ends
`launched=6 skipped=0 total=204s` and `[milestone] SUMMARY: FAIL (2/6
launches)`, with a `[milestone]` line for each launch. All four compare launches
(SERIAL and HIP, ranks 1 and 4) read `steps=600/600`, `defects=0`,
`time_liveness_failures=0`, `matched=1 mismatched=60`, `first_failing_step=10`,
and splits / collapses / flips / smooth_steps / cleanup_flips = 368 / 8 / 76 /
7 / 84, all non-zero. Every comparator exit is 0 (step 0, five times) or 1; the
244 `[FAIL]` lines are the 60 step mismatches and the summary line per compare
launch, none naming a counter or `time`. Failure direction (a): `fail-step0`
compares step 10 against the step-0 gold, comparator exit 1, PASS. Direction
(b): `fail-no-cleanup` (`cleanup_flips=0`) mismatches at step 10 with no defect,
PASS. So no listed defect occurred and nothing was resubmitted. The four launches
agree exactly on every count, counter and `time_rel`; `time` differs between them
by at most 1.8e-15 relative. At step 10 and every compared step after it the
counts differ (step 10: Beatnik 518/1032, gold 547/1090), the comparator stops at
the count check, and so no field after step 0 was compared: the first failing
field, its `max|e|` against M1-D1's floor and M1-R8's signature are "not
reached". This is M1-R2's differing-counts case. The per-launch record, the face
count and `time` series and the walls are in the log's `## M1-T2`, with an
`**Affects:**` line naming M1-A1j.

---

### M1-D2 — where in steps 1–10 the codes part, and why — **DONE**

**Depends on:** M1-T2.

**Fill in:** `scripts/tuolumne/m1d2_split_ab.flux` (the job: part `all` runs the
reference, the Beatnik launches, the per-step comparison and the probe; part
`diag` reruns only the Beatnik launches into `<root>/diag`),
`scripts/tuolumne/m1d2_refhooks.py` (wraps the reference's pass functions to
print per-step counts without editing it; `probe` applies one reference remesh
step to any checkpoint), and the rank-0 `[m1rem]` line in
[tests/regression_tests/Beatnik_Test_Milestone1Run.cpp](../tests/regression_tests/Beatnik_Test_Milestone1Run.cpp)
(that step's `RemeshDiagnostics` selection counts). Output
`/p/lustre5/stewartj/beatnik/M1-D2/`.

**Removes:** nothing.

**Reference:** M1-G2's command at `--steps 10 --checkpoint-every-steps 1`;
`Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP} 10 1 50` at ranks 1 and 4; the
reference's `split_long_edges`, `collapse_short_edges`,
`flip_edges_for_quality` and `_children_for_split_face`
(`dynamic_remesh.py:234-258`, `:361-405`, `:408-457`, `:552-581`).

**Do:** checkpoint both codes every step through step 10; record each step's
split and collapse candidates before the cap and the accepted edits on both
sides; compare each step's fields; apply the reference's rules to each code's
checkpoints. A Beatnik mesh on which the reference's rules want edits that
Beatnik does not make is a rule difference (b); identical candidate sets
resolved to different accepted sets is a selection difference (a).

**Exit criterion:** both jobs COMPLETED rc 0 with 11 checkpoints per launch; the
instrumented reference's step-10 file is `np.array_equal` to M1-G2's; the log's
`## M1-D2` section names the first step at which the counts differ, the pass
that makes them differ, and (a) or (b) with the counts that decide it.

**Met.** Jobs `f3iZ2TAcXCc7` (part `all`, 117 s) and `f3iZ84VbzTGj` (part `diag`,
47 s) COMPLETED rc 0, 11 checkpoints per launch, and the instrumented step 10 is
`np.array_equal` to M1-G2's on all nine keys. The four Beatnik launches agree
exactly. **Reading (a).**
- **Step 1.** Counts equal (462/920) and the same 300 split edges, the cap
  binding inside a tie class. The reference's fixed quad diagonal (R13) is the
  longer one in 41 of the 82 two-split faces, and 41 of its 90 quality flips
  undo exactly those. The meshes end 26 of 920 faces apart, with positions
  1.7e-5 apart (median).
- **Step 2.** The counts part (reference 498/992, Beatnik 514/1024) on the
  collapse pass alone. Both see the same 44 candidates, each admissible under
  the reference's tests. The reference's serial walk accepts 31. Tessera's
  one-round independent set accepts 8 and declines 36 for conflict, none for
  geometry.
- **Steps 3–10.** The reference's step-2 mesh falls to minimum quality 0.0995,
  under its own 0.18, and it re-refines to 1126 faces and recoarsens to 1090.
  Beatnik settles at 1032 faces at step 3.
- **Beatnik's mesh obeys the reference's rules.** One reference remesh plus
  cleanup applied to Beatnik's checkpoints wants exactly Beatnik's edits at
  step 2 and none at any probed step from 3 to 600.

The numbers are in the log's `## M1-D2`.

---

### M1-G3j — the 5000-step gold set, staged and submitted *(reference run, no code)* — **DONE**

**Depends on:** M1-G2.

**Fill in:** `scripts/tuolumne/m1g3_gold.flux`, new, modelled on
[scripts/tuolumne/m1g2_gold.flux](../scripts/tuolumne/m1g2_gold.flux): the same
provenance block, clean-`zmodel3d/` refusal, refuse-to-mix-runs check and
`flux run -N 1 -n 1` single-core launch, running the reference from the staged
copy rather than from the live checkout.

**Submits:** `scripts/tuolumne/m1g3_gold.flux`, `pbatch`, 1 node `--exclusive`,
`-t 150m`, `--bank=unm`. It runs M1-G2's command with `--steps 5000` in place of
`--steps 600`, every other option unchanged (`--checkpoint-every-steps 10`).
Expected run time **about 100 minutes**: M1-G2 measured 1.18 s/step at
1072–1090 faces (log `## M1-G2`), and the reference's face count is flat at
1072–1076 from step 60 to 600, so 5000 steps is about 98 minutes. Staged
directory `/p/lustre5/stewartj/beatnik/M1-G3j/stage/`: a copy of the reference
checkout's `examples/` and `zmodel3d/`, plus `COMMIT` holding the reference's
`git rev-parse HEAD` and `git diff`, and Beatnik's `git rev-parse HEAD` and
`git status --porcelain` count, taken at staging time. Output goes to
`/p/lustre5/stewartj/beatnik/M1-G3j/out/` (501 numbered checkpoints) and to
`beatnik_m1g3_gold.<jobid>.log` in the repo root.

**Removes:** nothing.

**Reference:** M1-G2's command and its job script; M1-G2's step-600 checkpoint,
`tests/regression_tests/milestone1-bubble/gold/checkpoint_t00000p557934_step0000600.npz`.

**Do:** write the script, stage, submit. The 5000-step `.npz` files stay on
Lustre and are not committed: the milestone test consumes only their statistics,
compiled in by M1-A1j, and its one field-wise comparison (step 0) uses M1-G2's
step-0 file.

**Exit criterion:** the staged directory exists with its `COMMIT`; the scheduler
accepted `m1g3_gold.flux`, its job ID on the log's `**Jobs:**` line.

**Met.** `scripts/tuolumne/m1g3_gold.flux` (commit `2a38355`) is M1-G2's command
with `--steps 5000`, run with `cd` into the stage. It keeps M1-G2's in-job checks
for 501 files and adds the step-600 `np.array_equal` check against M1-G2's file.
The stage `/p/lustre5/stewartj/beatnik/M1-G3j/stage/` holds the reference's
`examples/` and `zmodel3d/` working trees (no `__pycache__`), equal to the live
trees by `diff -r -x __pycache__`, no output. It also holds
`beatnik/compare_output.py`, `beatnik/initial_conditions/gold.npz` and
`beatnik/milestone1-bubble/checkpoint_t00000p557934_step0000600.npz`, each
identical to the repo copy by `cmp`. `COMMIT` records reference HEAD `ec7d7bf`, an
empty `zmodel3d/` status, the one-line `--source-quadrature` default diff in
`examples/`, Beatnik HEAD `2a38355` and a porcelain count of 2. The loop's wrapper
needed approval for `pbatch`, 150 minutes and 2.5 node-hours. The user approved it,
and the wrapper submitted job `f3iZLsmfWYw1` (rc 0). `flux jobs` shows it in state
`S` in `pbatch`, 1 node, 2.5 h limit.

---

### M1-G3 — the steps 0–1610 gold set is complete and recorded *(no code)* — **DONE**

**Depends on:** M1-G3j.

**Fill in:** `tests/regression_tests/milestone1-bubble-1610/README.md`, new: the
command verbatim, the job ID and log, the Lustre path of the files, the stop
(`stopping at step=1625 t=1.2540: nonfinite mesh-RHS state`), the cap at step
1610 and why steps 1620 and 1624 are excluded, and the per-checkpoint step /
`time` / V / F table for steps 0–1610, as
[tests/regression_tests/milestone1-bubble/README.md](../tests/regression_tests/milestone1-bubble/README.md)
does for M1-G2. There is no resubmission: the reference's own stop sets the cap.

**Removes:** nothing.

**Reference:** the M1-G3j job's log `beatnik_m1g3_gold.f3iZLsmfWYw1.log` and its
`out/` directory, `/p/lustre5/stewartj/beatnik/M1-G3j/out/`.

**Do:** check the run, write the README. The job's in-job checks report one
failure, the self-compare of the degenerate step-1624 file (1992 of 1993
vertices ambiguous at `--match-eps 1e-9`); every other in-job check passed. The
gold set is steps 0–1610 (Read this first #1). The step-1620 and step-1624 files
stay on Lustre but are not part of it: V/F go 610/1216 at step 1610, 793/1590 at
1620 and 1993/4002 at 1624, against a smooth rise from 538/1072 before.

**Exit criterion,** judged from the M1-G3j job's output: 162 numbered
`_step%07d.npz` checkpoints (steps 0, 10, …, 1610), every field finite; the
step-600 file is `np.array_equal` on every key to M1-G2's step-600 file (the
reference is serial and deterministic, so a difference means the staged
reference is not M1-G2's); a self-compare of the step-1610 file at
`--rtol 1e-12 --atol 1e-14` exits 0, where the step-1624 file's fails on
ambiguous pairing; and the README exists with the table.

**Met.** Job `f3iZLsmfWYw1` (`beatnik_m1g3_gold.f3iZLsmfWYw1.log`, stage `COMMIT`
naming Beatnik `2a38355` and reference `ec7d7bf`) stopped by itself with
`stopping at step=1625 t=1.2540: nonfinite mesh-RHS state` after a 1782 s
generation, leaving 164 numbered checkpoints (0..1620 every 10, plus 1624). Its
in-job checks show step 0 `np.array_equal` to the initial-condition gold and step
600 `np.array_equal` to M1-G2's step-600 file on all nine keys, every numeric
field of every file finite, and the step-1624 self-compare failing with 1992 of
1993 vertices ambiguous. The verification job `f3j6vKt4EQS3`
(`scripts/tuolumne/m1g3_check.flux`, `beatnik_m1g3_check.f3j6vKt4EQS3.log`,
COMPLETED rc 0) self-compared the step-1610 file with the staged
`compare_output.py` at `--rtol 1e-12 --atol 1e-14`: rc 0, `610/610 unambiguous,
ambiguous cpp=0 gold=0`. It found exactly 162 files for steps 0, 10, …, 1610,
each with the nine keys, the matching stored `step`, and every numeric field
finite; step 1610's `time` is 1.2381221993625053. The README
`tests/regression_tests/milestone1-bubble-1610/README.md` carries the command,
provenance, stop, cap, checks and the 162-row table, identical to the generation
job's table for those steps.

---

### M1-A1j — encode the decided acceptance, and submit the confirming run — **DONE**

**Depends on:** M1-T2, M1-D2, M1-G3.

**Fill in:**
- this document's `## Problem` section, which states the decided acceptance
  below as numbers;
- [tests/regression_tests/Beatnik_Test_Milestone1Bubble.cpp](../tests/regression_tests/Beatnik_Test_Milestone1Bubble.cpp),
  restructured: field-wise comparison at step 0 only; the time-matched
  statistics checks below at every gold checkpoint from step 10 to 1610; the
  `time` liveness check against the steps 0–1610 gold times; the failure
  directions kept;
- `kSteps` in
  [tests/regression_tests/Beatnik_Milestone1Params.hpp](../tests/regression_tests/Beatnik_Milestone1Params.hpp)
  (`:63`), set to 1610, the gold set's last step. At equal step index Beatnik
  leads the gold in `time` (`time_rel` 0.116–0.336 through step 600, log
  `## M1-D2`), so 1610 steps carry it past the gold's final `time` 1.238122.
  A shortfall is the loud failure under **Decided**, not a skip.
  `Beatnik_Test_Milestone1Run` takes its step count from `argv[1]` and does
  not read it;
- a compiled-in gold-statistics table, one row per M1-G3 checkpoint of steps
  0–1610, 162 rows, never the step-1620 or step-1624 file (step, `time`, V, F,
  total area, area-weighted centroid `z`, volume drift, minimum triangle
  quality, valence 5/6/7 counts, minimum and mean edge length), in a
  new header beside the test, generated at 17 digits by a new committed script
  under `tests/regression_tests/` that reads M1-G3's `.npz` files. The
  definitions are those of M1-D2's
  `/p/lustre5/stewartj/beatnik/M1-D2/analysis/structural_stats.py`: area
  `Σ ½|(b−a)×(c−a)|`, centroid `z` area-weighted over face centroids, volume
  drift `vol/initial_volume − 1`, quality `4√3·A / Σℓ²`;
- [scripts/tuolumne/m1_milestone_run.flux](../scripts/tuolumne/m1_milestone_run.flux):
  `-t` and the per-row deadline-guard estimates, for 1610 steps;
- the `milestone` entry in [docs/testing.md](../docs/testing.md#the-tiers): what
  the member asserts. Reword its `time` sentence to "at most twice the gold's"
  (M1-T2j review, nit 1);
- README "Known Issues": Beatnik's collapse pass runs one Tessera
  independent-set round per pass and accepts fewer collapses than the
  reference's serial walk (8 against 31 of 44 at step 2, log `## M1-D2`),
  pre-existing, by design.

**Submits:** `scripts/tuolumne/m1_milestone_run.flux
/p/lustre5/stewartj/beatnik/M1-A1j/stage`, `pdebug`, 1 node `--exclusive`,
`-t 10m`, `--bank=unm`, the dev env, the same six launches as M1-T2j. Expected
run time **about 5 minutes**, scaled from M1-T2's measured walls (log
`## M1-T2`): `run_wall` per 600 steps of 18.9 s (SERIAL), 6.8 s (HIP np1) and
9.8 s (HIP np4) times 1610/600 is about 51 + 51 + 18 + 26 s; the comparator,
which was 26–30 s of each compare launch, runs once per launch; plus the two
short failure-direction launches and about 30 s of setup. Staged directory
`/p/lustre5/stewartj/beatnik/M1-A1j/stage/`, the layout M1-T2j staged (the
member's binaries, `regression_tests/compare_output.py` and M1-G2's gold
directory). Output goes to `/p/lustre5/stewartj/beatnik/M1-A1j/out/<target>/`
and to `beatnik_m1_milestone.<jobid>.log` in the repo root.

**Removes:** the field-wise comparison of steps 10–600 in
`Beatnik_Test_Milestone1Bubble.cpp`, and `toleranceForStep`'s every-step table
with it, replaced by the step-0 tolerance; the 61-row `kGold` table, replaced
by the generated 162-row table. Any helper left without a caller goes too.

**Reference:** the measured inputs below, and M1-T2j's output contract (log
`## M1-T2j`), which the restructured test keeps for its `[m1traj]`, `[m1fail]`
and `[m1final]` lines.

**Inputs, all measured:**
- **Frozen connectivity (M1-D1).** No compared step fails at any rung through
  step 600, on either backend or rank count. Peak `vertices` `max|e|` is
  `9.88e-15`.
- **With remeshing, the codes part on selection, not floating point (M1-T2,
  M1-D2).** Step 1 is count-equal but 26 of 920 faces apart (R13), and the
  comparator's pairing fails there (M1-R8); from step 2 on the counts differ.
  Field-wise comparison therefore reaches step 0 only, where every launch
  matches with `max|e|` `5.55e-17` on `vertices`. Beatnik's mesh obeys the
  reference's rules at every probed step from 2 to 600 (reading (a)).
- **Statistics at matched physical time, through step 600 (log `## M1-D2`).**
  Rise differs by at most `2.683e-4`; area by at most `1.805e-4` relative;
  `|volume drift|` is at most `4.4e-16` on both sides; Beatnik has 3–5 % fewer
  faces; its minimum quality is 0.840–0.870. At equal step index Beatnik leads
  in time by `time_rel` 0.116–0.336. With `--no-isotropic-cleanup` Beatnik holds
  1280 faces against the gold's 1090 at step 10, +17 % (log `## M1-T2`).

**Decided:**
- **Field-wise compare depth: step 0**, at the comparator's defaults
  `--rtol 1e-10 --atol 1e-12`, with the measured `5.55e-17` written beside the
  entry as a comment.
- **Beyond step 0, matched on physical time.** Beatnik's statistics are
  computed every step and linearly interpolated to each gold checkpoint's
  `time`; a gold checkpoint later than Beatnik's final `time` is a loud
  failure, not a skip. At every gold checkpoint from step 10 to 1610 the test
  asserts:
  - rise, the area-weighted centroid `z`: `|Beatnik − gold| ≤ 1e-3` absolute;
  - total area: `|Beatnik/gold − 1| ≤ 1e-3`;
  - Beatnik's `|volume drift| ≤ 1e-12`;
  - face count: `|F_Beatnik/F_gold − 1| ≤ 0.10`;
  - Beatnik's minimum triangle quality `≥ 0.18` (`--remesh-min-quality`).

  Valence 5/6/7 counts and the minimum and mean edge length are printed beside
  the gold's at each such checkpoint, not asserted.
- **Collapse stays one independent-set round per pass.** The accepted-collapse
  difference is recorded in README "Known Issues", not changed.
- **`--steps 1610`**, against M1-G3's steps 0–1610 gold set.
- Failure direction (a) is unchanged. Failure direction (b),
  `--no-isotropic-cleanup`, must fail on a statistics check, which the face
  count band catches at step 10.

**Exit criterion:** this document's `## Problem` section states the decided
depth, tolerances and beyond-depth checks as numbers, equal to **Decided**; the
test's `kSteps`, step-0 tolerance and statistics tolerances equal them, checked
by reading; the progress log records the rejected alternatives (field-wise
comparison past step 0, step-indexed matching, a multi-round collapse, stopping
at 600 steps) and why. The dev-env `spack install` is rc 0. The staged directory exists with its
`COMMIT`. The scheduler accepted `m1_milestone_run.flux`, its job ID on the
log's `**Jobs:**` line.

**Met.** The `## Problem` section states the decided depth (step 0 at
`--rtol 1e-10 --atol 1e-12`), the five statistics bands and the matching
rule, equal to **Decided**; the test's `kSteps` (1610, from
`Beatnik_Milestone1Params.hpp`), `kStep0Tolerance` and `kRiseAtol`,
`kAreaRtol`, `kVolumeDriftMax`, `kFaceCountRtol`, `kMinQuality` equal them,
checked by reading. The 162-row table is generated by
`tests/regression_tests/milestone1_gold_stats.py`, whose step / `time` / V / F
columns agree exactly with the M1-G3 README. The log's `## M1-A1j` records the
four rejected alternatives and why. The dev-env `spack install` is rc 0
(13m09s, 80 warnings, all `-Wpass-failed` in `Kokkos_HIP_KernelLaunch.hpp`);
the dev view's `Beatnik_Test_Milestone1Bubble_MPI_{SERIAL,HIP}` postdate the
edit and contain the new `[m1statworst]` string. The stage
`/p/lustre5/stewartj/beatnik/M1-A1j/stage/` holds those binaries
byte-identical to the dev view, the comparator, M1-G2's 61-file gold
directory and README, and `COMMIT` naming `f3a6c80` (porcelain count 3, none
compiled). Job `f3j79G2VDX4T` (`m1_milestone_run.flux`, `pdebug`, `-t 10m`)
was accepted and was in state RUN at submission; its output is M1-A1's.

---

### M1-D3 — why the trajectory drifts from the gold after gold step 1000 — **DONE**

**Depends on:** M1-A1j (its job `f3j79G2VDX4T`, read in the log's `## M1-A1`).

**Why this task exists.** M1-A1's confirming run has no defect, but the area
band misses from gold step 1060 (worst `1.063e-2` at 1610) and the rise band
from 1340 (worst `2.545e-3` at 1610), identically on SERIAL and HIP at ranks 1
and 4. Through gold step 1050 every band holds. The user ruled (2026-11-09):
keep every band, `kSteps` and the asserted depth as decided, and find out why
the trajectory drifts **before** any tolerance is chosen. This task measures;
it changes no tolerance and no remesher behaviour.

**Fill in:** a diagnostic script or job under `scripts/tuolumne/` (BSD header),
output under `/p/lustre5/stewartj/beatnik/M1-D3/`, and the log's `## M1-D3`
section. Beatnik's M1-A1j checkpoints
(`/p/lustre5/stewartj/beatnik/M1-A1j/out/Beatnik_Test_Milestone1Bubble_MPI_<BACKEND>_np<N>/beatnik_milestone1/<Backend>_np<N>_compare/`,
e.g. `HIP_np1_compare/`, every 10 steps) and M1-G3's gold `.npz` (`/p/lustre5/stewartj/beatnik/M1-G3j/out/`) are
the first inputs; read them before running anything new.

**Removes:** nothing.

**Reference:** the log's `## M1-A1` (the growth tables), `## M1-D2` (the codes
part on collapse selection from step 2; Beatnik's mesh obeys the reference's
rules through step 600), `## M1-G3` (the gold's late mesh growth and its
nonfinite stop at step 1625), and
`tests/regression_tests/milestone1_gold_stats.py` (the statistic definitions).

**Do:** at matched physical time, separate the candidate causes and say which
the numbers support:
- **resolution** — Beatnik carries 0.7–3.7 % fewer faces late (1.3 % at gold
  step 1200, 2.9 % at 1300, 3.7 % at 1400, 0.7 % at 1500, 2.5 % at 1610:
  1186 against 1216), yet has the *larger* area; compare the two meshes'
  shapes (e.g. per-region or by polar angle) rather than totals alone;
- **a physical-trajectory difference** — the rise lags while the area grows,
  i.e. the bubble deforms more and rises less; locate where on the surface the
  shapes part and from what time;
- **the gold's own late behaviour** — whether the reference's trajectory
  approaching its step-1625 blow-up is itself the outlier (its mesh grows
  from step 1100, 1078 faces at 1150 to 1216 at 1610, and surges only at
  steps 1620–1624);
- **a step-size effect** — Beatnik reaches gold step 1610's time at its own
  step 1447–1448, so it takes longer steps late; whether the drift tracks the
  dt difference.

**Exit criterion:** the log's `## M1-D3` names the gold step (time) from which
the shapes measurably part, which of the candidate causes the measurements
support and which they rule out, with the numbers that decide it, and states
what it implies for M1-A1's two missed bands (a defect to fix, or a
measured physical/resolution gap for the user to set a band against). No
tolerance, `kSteps` or remesher change is made here.

**Met.** Measured on M1-A1j's HIP np1 checkpoints, M1-G3's gold, and three
reference runs (job `f3j7Pykw2A6s`), each changing one variable; log
`## M1-D3`. The shapes measurably part from gold step 600 (`time` 0.558).
**The drift is a mesh-realisation (discretisation) gap and not a defect.** The
reference's own solver, rules and step rule, restarted from Beatnik's step-10
state (`bk10`), miss the gold at gold step 1610 by area `+1.047e-2` and rise
`-2.808e-3`, against Beatnik's `+1.063e-2` and `-2.545e-3`. `bk10` equals
Beatnik to `7e-16` through its step 240 and stays within `4.4e-4` (area) and
`5.7e-4` (rise) of it to `time` 1.2382. Four checks rule the other causes out:
- **Step size:** the reference run with Beatnik's step sizes misses by at most
  `9.9e-4` and `2.3e-4`, inside both bands.
- **The gold's late behaviour:** a restart from the gold's own step 10
  reproduces it to `3e-15`.
- **Face count:** `bk10` carries *more* faces than the gold late (1240 against
  1216) and the same excess area.
- **Remesher:** both codes' late meshes are fixed points of the reference's
  rules (probe job `f3j7R4vqZF67`).

No compiled file changed.

---

### M1-A1 — the milestone test is green at the decided numbers — **NOT STARTED**

**Depends on:** M1-A1j, M1-D3 (the user's ruling of 2026-11-09: no tolerance
is chosen for the area and rise misses until M1-D3 explains the drift).

**Fill in:** the progress log's `## M1-A1` section; the `Milestone1Bubble`
row of the milestone walls table in [docs/testing.md](../docs/testing.md#the-tiers).
If the output shows a defect,
fix it inside M1-T2j's or M1-A1j's **Fill in** and resubmit from
`/p/lustre5/stewartj/beatnik/M1-A1/stage/`.

**Removes:** nothing.

**Reference:** the M1-A1j job's log and its `out/` directory.

**Do:** judge the criterion, and record each launch pair's job wall in the
log. Its `**Affects:**` line asks the user to set the two
`Beatnik_Test_Milestone1Bubble` rows' `-t` in `submit_milestone_split.sh` to
about 1.5x those walls and drop their `pending=M1-T2j` field. The task loop
protects that file. A step that misses its decided tolerance is **not**
a defect to fix by loosening the table. Take it back to the user with the
measured `max|e|`, as M1-A1j's decision (M1-R1).

**Exit criterion,** judged from the M1-A1j job's output: PASS at ranks 1 and 4
on SERIAL and HIP, zero `[FAIL]` lines and the `SUMMARY` line present; failure
direction (a) exits exactly 1 from the comparator; and (b) fails the comparison.
The milestone is **accomplished** when this holds.

## Known risks

**M1-R1 — the divergence horizon is shorter than the milestone's premise.** The
two codes stop agreeing at `1e-10` well before step 500. *Presents as:* M1-D1
reporting a first-failing step in the tens, on a run with no adaptivity at all.
*Do:* this is information, not a bug — take it to M1-A1j. Do **not** loosen
`--rtol` to make a step pass without recording the measured `max|e|` beside it;
an unrecorded loosening is how a comparison stops being a test.

**M1-R2 — edit-set divergence (R7) makes the comparison fail structurally.** The
first flip or collapse Beatnik accepts differently from the reference changes
connectivity, and `compare_output.py` compares connectivity exactly. *Presents
as:* `faces: N of M triangles differ after remapping` with **equal** vertex and
face counts. *Distinguishing measurement:* equal counts + differing connectivity
= R7 or R13 (an edit-set or diagonal difference); **differing counts** = the two
codes selected different mark sets, which is a step further along and cannot be
fixed by an ordering change. Report which, with the step index, before proposing
anything. *Measured* (M1-D2): equal counts with 26 of 920 faces different at
step 1 (R13); differing counts from step 2, from collapse selection (Tessera's
one-round independent set against the reference's serial walk, on an identical
44-candidate set).

**M1-R3 — the collapse priority key differs from Tessera's.** The reference ranks
by `length / local_target`; Tessera by `(squared length, EdgeKey)`. Beatnik's mask
is built and truncated on the reference's key, but *within* the mask Tessera's
independent-set round uses its own priority, so the accepted subsets differ
whenever the cap binds or two candidates conflict. *Presents as:* a collapse count
that agrees with the reference when the cap does not bind and diverges when it
does. *Do:* log `requested` vs `accepted` vs the cap per pass, so the three cases
are distinguishable.

**M1-R4 — a `tangentialSmooth` without the quality guard is plausible and
wrong.** Omitting the trial/accept/reject step (Read this first #5) produces a
smoother mesh and a slightly different trajectory, with no error anywhere.
*Presents as:* a comparison that fails a few steps later than expected with small,
diffuse position errors and a *better* minimum quality than the gold. *Do:* T4d4's
exit criterion asserts both directions of the guard precisely because nothing
downstream would notice.

**M1-R5 — the gid-space leak.** *Presents as:* resident memory growing with step
count at constant face count, or an allocation failure deep inside Tessera late in
a long run. Owned by T4d6j and T4d6; suspect it first for any late-run memory symptom.

**M1-R6 — cost.** The direct Birkhoff-Rott sum is `O(N^2)` per RK stage, three
stages per step, and nothing caps the face count on this path (Read this first
#3). *Presents as:* a milestone launch that times out rather than fails. *Do:*
M1-D1 and M1-T2 both report the face-count trajectory and wall time; if the
trajectory is superlinear in step index, raise it at M1-A1j rather than silently
lowering the step count — a quietly truncated milestone run is a gate that
shrank.

**M1-R7 — a milestone test that passes because nothing happened.** R15's trap in
this milestone's shape: if the sizing field never triggers a split, or
`isotropicCleanup` never flips, the run reproduces a simpler trajectory and the
comparison passes while testing none of T4d. *Presents as:* it does not — a green
test and no signal. *Distinguished by:* the four cumulative edit counters and the
cleanup flip count, asserted non-zero (T4d6's umbrella clause, M1-T2j step 3).

**M1-R8 — the quantized pairing degrades silently.** `compare_output.py` sorts
each file's vertices *independently* at `--match-eps 1e-9`. Once positions
disagree by more than the cell size near a cell boundary, the two sorts pair
different vertices and every field reports large errors that have nothing to do
with the field. *Presents as:* huge, uniform `max|e|` across all fields at once,
often with `ambiguous cpp=0 gold=0`. *Distinguished from a real field error by:*
`vertices` itself failing first and by the same magnitude as the other fields.
*Do:* report it as a pairing failure and record the step; do not raise
`--match-eps` without stating what it now cannot detect.

