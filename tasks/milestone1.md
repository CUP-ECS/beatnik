# Milestone 1 — the reference's default rising-bubble configuration

**Status:** NOT STARTED

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
equivalent through `examples/02_adaptive_mesh_bubble`. Acceptance is a comparison
of the Python `.npz` checkpoints against the Beatnik `.h5` checkpoints by
`tests/regression_tests/compare_output.py`.

What exists now is a Beatnik that **rejects this command at setup**, by name and
by task ID, from `Solver::requireSupportedConfiguration`
([src/Beatnik_Solver.hpp:729-872](../src/Beatnik_Solver.hpp#L729-L872)). Four
rejections fire and every one of them is **T4d**:

| Option, at its default | Rejection | Needs |
| --- | --- | --- |
| `--isotropic-cleanup` (default **on**) | [:769-775](../src/Beatnik_Solver.hpp#L769) | `MeshQuality::isotropicCleanup` |
| `--remesh-collapse-factor 0.45` | [:818-828](../src/Beatnik_Solver.hpp#L818) | `DynamicRemesh::collapseShortEdges` |
| `--remesh-smooth-iters 1` | [:830-837](../src/Beatnik_Solver.hpp#L830) | `DynamicRemesh::tangentialSmooth` |
| `--remesh-flip-min-gain 1e-3` | [:839-850](../src/Beatnik_Solver.hpp#L839) | `DynamicRemesh::flipEdgesForQuality` |

Nothing else on the command line is stubbed. The end state is: the command runs,
the two codes' checkpoints are compared to a **measured, documented** depth, and
that depth plus its tolerances are asserted by a test in a new `milestone` tier.

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

Five places where inspection contradicted the brief or the prior design text.

### 1. The brief's command runs 140 steps and writes no checkpoints

`--steps` defaults to **140**, `--checkpoint-dir` to `""` and
`--checkpoint-every-steps` to **0** (`run_adaptive_mesh_bubble.py:133,318,325`),
and both checkpoint criteria are guarded on the directory being set (`:1571`,
`:1573`). The command as written therefore produces no `.npz` at all, and cannot
reach step 500.

**Resolved with the user:** the milestone gold set is a **600-step** run
checkpointing **every 10 steps** — see M1-G2 for the exact command. The
`--checkpoint-every-steps` cadence is expected to be revisited once M1-D1 and
M1-A1 have measured where the two codes part company.

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
  did *not* diverge over 20 steps. The honest reading is that it is
  configuration-dependent, not retired.
- **R2.** Cross-rank reduction order gives ulp-level differences in the reduced
  scalars, and a rolling-up vortex sheet amplifies them.

The deepest **field** agreement ever demonstrated in this port is **10 steps** at
`--rtol 1e-10` (T2d), on a fixed-connectivity mesh.

**Resolved with the user: measure first, then decide.** M1-G1/M1-D1 measure the
divergence horizon of the *fixed-connectivity* configuration — the best case,
isolating R2 from R7 and R13 — before any T4d work is spent. M1-A1 then sets the
milestone's compare depth and tolerance ladder from the measured numbers, with
the 600-step gold set in hand. **This document does not assert that step 500,
1000, 2000 or 5000 is reachable at `1e-10`; it makes finding out a task.**

### 3. `--max-faces` does not bound a `--dynamic-remesh` run

`max_faces` is a parameter of the *indicator-driven* refiner only: it is absent
from `DynamicRemeshParams` (`run_adaptive_mesh_bubble.py:1310-1357` constructs it
without one) and in Beatnik it is read only by `AdaptiveMesh`
([src/Beatnik_AdaptiveMesh.hpp:1027-1030](../src/Beatnik_AdaptiveMesh.hpp#L1027-L1030)).
So T4b's plateau at 1400 faces was the sizing field reaching equilibrium, **not**
a cap, and a 600-step milestone run has no face-count ceiling other than
`--remesh-h-min 0.0015` and `--remesh-max-splits 300` per pass. R4's greedy-cap
divergence is therefore *not* in play on this path, and the cost of the
`O(N^2)` direct Birkhoff-Rott sum is (M1-D1 measures the trajectory).

### 4. T4d1–T4d6 have moved here, and keep their IDs

framework.md carried the T4d design; it now points here. **The IDs are
unchanged** — `T4d`, `T4d1`…`T4d6` — because four `requireSupportedConfiguration`
messages, several header comments and
[src/Beatnik_DynamicRemesh.hpp:1018-1045](../src/Beatnik_DynamicRemesh.hpp#L1018-L1045)
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

## Approach

Three strands, run in this order of *risk*, not of code size:

1. **Measure the achievable comparison first** (M1-G1, M1-D1). A fixed-mesh
   600-step Python/Beatnik comparison needs no new solver code and answers the
   only question that can invalidate the milestone's premise.
2. **Land T4d** (T4d1–T4d6). Six sub-tasks; T4d1 is an adapter layer, T4d2–T4d5
   are one reference function each, T4d6 is an operational decision.
3. **Wire the milestone test** (M1-T1, M1-T2) and **fix the acceptance**
   (M1-A1).

### Conventions

Framework.md's conventions table governs; these are the additions this milestone
introduces or sharpens.

| Convention | Choice |
| --- | --- |
| Task IDs | `T4d*` for work moved from framework.md (IDs are referenced from source strings); `M1-*` for work new here. `M1-G*` = human gold-file generation, `M1-D*` = measurement, `M1-T*` = code, `M1-A*` = a decision to be recorded. |
| Edge-mask type | host `const std::vector<char>&`, sized `ownedEdgeCount()`, one entry per **owned** edge — T4a's convention for `splitEdges`, and exactly what Tessera's collapse and flip take. De-template the two stale `EdgeListView` signatures rather than adding an overload. |
| Where a Beatnik-only acceptance rule lives | **mask construction, never a Tessera policy.** The gain test, the valence test and the normalized-length ranking are Beatnik's; `maxNormalRotation`/`minQuality`/`maxNormalDeviation` are Tessera's absolute admissibility on top. A marked edge may still be rejected, and that shows up in the `rejected*` counters. |
| Reference-state re-basing | every pass that moves a vertex or changes connectivity states, in its progress-log entry, whether it calls `AdaptiveMesh::resetReferenceState`, and why. A silent choice here changes every later refinement decision. |
| Failure behavior | loud. A configuration Beatnik cannot reproduce is rejected before the first step, by method name and task ID, exactly as the four rejections this milestone deletes do. No pass may silently no-op. |
| Diagnostics that must be numbers | per remesh pass: `splits`, `collapses`, `flips`, `smooth_steps`, `min_quality` before/after, R12's two shape signals, and (T4d6) the gid-space high-water mark. A milestone run that reports zero edits everywhere is M1-R7, not a pass. |
| Test tier for the milestone comparison | a new **`milestone`** label, at ranks **1 and 4** on SERIAL and HIP, outside the 60-launch ship gate. The `regression` tier keeps exactly its five members; promoting anything into it needs the user's confirmation (CLAUDE.md "Minimum test set"). |
| Provenance comments | `// Port of <file>::<fn> (lines N-M)` against the **real** origin, per framework.md. The functions this milestone ports live in `dynamic_remesh.py` and `mesh_quality.py`; cite line ranges, not function names alone. |

### Deliberate deviations

- **Beatnik's accepted edit sets will not equal the reference's**, and no attempt
  is made to make them. The alternative — a sequential-equivalent scheduler in
  Tessera plus reduction-order pinning — was considered and rejected with the
  user: it is large, it reaches upstream, and it still cannot make a NumPy and a
  Kokkos trajectory agree over hundreds of steps. The consequence is that the
  milestone's compare depth is an empirical quantity (M1-A1), not an assumption.
- **`--remesh-max-collapses 0` is not a lever.** A non-positive value maps to
  `None` = *unlimited* (`run_adaptive_mesh_bubble.py:1338-1340`), reproduced by
  `RemeshParams::max_collapses_per_pass`. Only `--remesh-collapse-factor 0`
  disables collapse. The deleted rejection text says so and the replacement
  diagnostics must not re-introduce the confusion.
- **`SurfaceMesh::compact()` is not called from the remesh cycle.**
  `Tessera::collapseEdges` runs `compact()` as its own last step; the standalone
  adapter exists for T4d6's renumbering cadence and for tests. Its doc comment
  must say that, rather than leave a reader to infer it.

## Current state

True at HEAD, and specific about what is *defined but wrong* versus what
*throws*:

- **Throws** (`BEATNIK_NOT_IMPLEMENTED`), so a run cannot reach them silently:
  `SurfaceMesh::{collapseEdges, flipEdges, compact}`
  ([src/Beatnik_MeshInterface.hpp:1578](../src/Beatnik_MeshInterface.hpp#L1578),
  [:1604](../src/Beatnik_MeshInterface.hpp#L1604),
  [:1622](../src/Beatnik_MeshInterface.hpp#L1622));
  `DynamicRemesh::{collapseShortEdges, flipEdgesForQuality, tangentialSmooth}`
  ([src/Beatnik_DynamicRemesh.hpp:889](../src/Beatnik_DynamicRemesh.hpp#L889),
  [:911](../src/Beatnik_DynamicRemesh.hpp#L911),
  [:936](../src/Beatnik_DynamicRemesh.hpp#L936));
  `MeshQuality::{valenceEqualizingFlips, isotropicCleanup, improveConnectivityByFlips}`
  ([src/Beatnik_MeshQuality.hpp:108](../src/Beatnik_MeshQuality.hpp#L108),
  [:177](../src/Beatnik_MeshQuality.hpp#L177),
  [:205](../src/Beatnik_MeshQuality.hpp#L205)).
- **Defined and deliberately inert**, transcribed rather than folded away, so
  landing T4d turns them on by deleting a rejection instead of by remembering to
  add a call: `const int collapses = 0` and `diag.flips += 0; diag.smooth_steps += 0`
  inside the pass loop
  ([src/Beatnik_DynamicRemesh.hpp:1006-1047](../src/Beatnik_DynamicRemesh.hpp#L1006-L1047)),
  and `const int flips = 0` on the refine branch
  ([src/Beatnik_Solver.hpp:467](../src/Beatnik_Solver.hpp#L467)). Note the
  **sizing-field recompute before the collapse pass is absent**, not merely
  unused — the comment at
  [:1018-1028](../src/Beatnik_DynamicRemesh.hpp#L1018-L1028) records that it is
  load-bearing the moment collapse lands, because without it the collapse pass
  undoes the split pass.
- **Missing entirely from the remesh branch:** the `isotropic_cleanup` call site
  and the *second* state rebuild that follows it
  ([src/Beatnik_Solver.hpp:527-547](../src/Beatnik_Solver.hpp#L527-L547) has the
  first rebuild and a comment where the cleanup goes).
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

## Task sequence

### M1-G1 — the fixed-mesh 600-step gold set *(human step, no code)* — **NOT STARTED**

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

**Exit criterion:** 61 `.npz` files present;
`python tests/regression_tests/compare_output.py <gold>/…_step0000000.npz tests/regression_tests/initial_conditions/gold.npz --rtol 1e-12 --atol 1e-14`
exits 0, and a self-compare of the step-600 file against itself at the same
tolerances exits 0 with zero ambiguous vertices (the check that
`--match-eps 1e-9` still resolves this mesh at step 600).

---

### M1-D1 — measure the fixed-mesh divergence horizon — **NOT STARTED**

**Depends on:** M1-G1.

This is the task that decides whether the milestone's premise holds. It writes no
library code.

**Fill in:** `scripts/tuolumne/milestone1_divergence.flux` (a batch script; never
launch interactively from a login node), and a table of measured numbers in
`milestone1-progress-log.md`.

**Reference:** the existing runner
[scripts/tuolumne/run_regression_minset.flux](../scripts/tuolumne/run_regression_minset.flux)
for the resolver-sourcing, `beatnik_exe` and scratch conventions;
`Beatnik_Test_DirectSolve10Steps.cpp:257-306` for the `goldForStep` +
`compare_output.py` subprocess pattern.

**Do:**

1. Run `examples/02_adaptive_mesh_bubble` with the M1-G1 command's Beatnik
   equivalent for 600 steps, checkpointing every 10, at ranks 1 and 4 on SERIAL
   and HIP, into `BEATNIK_TEST_SCRATCH` (**a parallel filesystem** — a node-local
   scratch fails every multi-node launch, CLAUDE.md).
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

---

### M1-G2 — the milestone gold set *(human step, no code)* — **NOT STARTED**

**Depends on:** none. Independent of M1-D1 — generate it in parallel.

**Fill in:** `tests/regression_tests/milestone1-bubble/gold/*.npz` plus that
directory's `README.md` with the command recorded verbatim.

**Reference:** the brief's command, plus the three options that make it produce
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
(`run_adaptive_mesh_bubble.py:68-527`); they are written out because the brief
does, and because a reader comparing this set against the T2a set will otherwise
suspect the two describe different physics — they do not (see the T2a entry in
[`framework-progress-log.md`](framework-progress-log.md)).

**Do:** generate it; record in the directory's `README.md` the per-step face and
vertex counts, the `time` series, and whether the run stopped early
(`stopping at step=… nonfinite …`). Those three are what M1-T2 asserts against
and what M1-A1 reasons about; a gold set whose face count is not recorded forces
the next session to re-derive it from 61 files.

**Exit criterion:** 61 `.npz` files present (fewer only if the reference itself
stopped early, in which case the stop step is recorded in the `README.md` and
becomes the compare depth ceiling); a self-compare of the last file at
`--rtol 1e-12 --atol 1e-14` exits 0; step 0 is bitwise identical to
`initial_conditions/gold.npz`.

---

### T4d — Coarsening, flips, and isotropic cleanup

The umbrella ID that four `requireSupportedConfiguration` messages and several
header comments name. It is **not** a task: the work is T4d1–T4d6 below, and the
umbrella exit criterion is at the end.

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
   already 2 ([src/Beatnik_MeshInterface.hpp:441](../src/Beatnik_MeshInterface.hpp#L441));
   do not lower it.
2. **`haloExchange()` *before* collapsing** — the surviving vertex's owner blends
   positions and every vertex user field from local copies of both endpoints, one
   of which is generally a ghost. Note the asymmetry with `splitEdges()`, whose
   wrapper exchanges *after*
   ([src/Beatnik_DynamicRemesh.hpp:873](../src/Beatnik_DynamicRemesh.hpp#L873)).
3. **One independent set per call.** More progress means calling again:
   `while ( collapseEdges( … ).accepted > 0 ) {}`.
4. **All three invalidate every slice, CSR and key View.** Re-slice afterwards.

Two facts that remove work the earlier design assumed: `compact()` is a halo
rebuild that whole-tuple copies every AoSoA, so **face and vertex user fields
survive collapse, flip and compaction verbatim** — T4a's `/faces/u0..u2` are not
invalidated; and `DefaultCollapsePolicy{t=0.5}` is the same linear average
Beatnik already uses for refinement
([src/Beatnik_MeshInterface.hpp:105-112](../src/Beatnik_MeshInterface.hpp#L105-L112)),
which is what the reference does (`dynamic_remesh.py:395-398`), so **no custom
field-blending policy is needed**. The real hazard is *staleness*: the face user
fields are the AMR reference state and now describe a geometry that no longer
exists — see the re-basing convention.

---

### T4d1 — the three `SurfaceMesh` adapters — **NOT STARTED**

**Depends on:** none.

**Fill in:** `SurfaceMesh::{collapseEdges, flipEdges, compact}` in
[src/Beatnik_MeshInterface.hpp](../src/Beatnik_MeshInterface.hpp) at
[:1578](../src/Beatnik_MeshInterface.hpp#L1578),
[:1604](../src/Beatnik_MeshInterface.hpp#L1604),
[:1622](../src/Beatnik_MeshInterface.hpp#L1622).

**Reference:** `Tessera::{collapseEdges, flipEdges, compact,
compactAndRenumberGids}` and their policies and result structs, in the three
headers named under T4d. `SurfaceMesh::splitEdges`
([:1535](../src/Beatnik_MeshInterface.hpp#L1535)) is the adapter shape to match.

**Callers of the signatures this changes:** **none today** — both edge entry
points still carry the M1-era `template <class EdgeListView>` and no call site
exists (verified by grep over `src/`, `examples/` and `tests/`). De-templating
them is therefore free, and doing it now is what keeps T4d2/T4d3/T4d5 from each
inventing a mask type.

**Do:**

1. Replace `const EdgeListView&` with `const std::vector<char>&` sized
   `ownedEdgeCount()` on both, per the conventions table.
2. Map `CollapseResult` and `FlipResult` into `MeshEditReport`, and **assert the
   closed-manifold identities** Tessera's header states —
   `verticesRemoved == accepted`, `edgesRemoved == 3*accepted`,
   `facesRemoved == 2*accepted` — as the cheapest end-to-end check that the
   connectivity rewrite closed. Fail loudly; do not log and continue.
3. State in `compact()`'s doc comment that the remesh cycle does **not** call it
   (`collapseEdges` compacts internally) and that it exists for T4d6 and for
   tests.
4. Delete the three "M1 GAP (G5b/G5c/G5d) — STILL OPEN" comment blocks, which are
   false, and the "when it lands" phrasing around them.

**Exit criterion:** a new `unit`-tier member
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`, registered in
[tests/unit_tests/CMakeLists.txt](../tests/unit_tests/CMakeLists.txt), passes at
ranks 1–6 via `flux batch scripts/tuolumne/unit_tests.flux`: on the default
icosphere it collapses a named edge set, flips another, and compacts, and checks
(a) the three V/E/F identities, (b) `V - E + F == 2` globally afterwards,
(c) gid preservation across `compact()`, and (d) the **failure direction** — a
mask marking a boundary-of-nothing/link-condition-violating edge is *rejected*
and shows up in `rejectedLinkCondition`, with `accepted` unchanged, rather than
producing a torn mesh.

---

### T4d2 — `DynamicRemesh::collapseShortEdges` — **NOT STARTED**

**Depends on:** T4d1.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:889](../src/Beatnik_DynamicRemesh.hpp#L889);
the sizing-field recompute and the `const int collapses = 0` placeholder inside
the pass loop at [:1018-1029](../src/Beatnik_DynamicRemesh.hpp#L1018-L1029);
and the deletion of the rejection at
[src/Beatnik_Solver.hpp:818-828](../src/Beatnik_Solver.hpp#L818-L828).

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
   ([src/Beatnik_Params.hpp:293](../src/Beatnik_Params.hpp#L293), 120). The cap is
   mask construction; there is no per-call cap argument (M1-R3).
4. `haloExchange()` first. Then call `SurfaceMesh::collapseEdges` with
   `policy.minQuality` from `--remesh-min-quality` and `maxNormalRotation` chosen
   to reproduce the reference's fold guard — the reference accepts iff every
   incident face's `old·new` normal dot exceeds **0.1** (`dynamic_remesh.py:546-549`);
   read Tessera's header for what `maxNormalRotation` measures before picking a
   number, and record the mapping in the log.
5. Report `collapses` into `RemeshDiagnostics` and decide, explicitly, whether the
   pass re-bases the AMR reference state (it deletes faces and moves a vertex, so
   the convention applies).

**Exit criterion:** a `unit`-tier sub-case in `Beatnik_Test_MeshEdits.cpp` (or a
new member) shows, at ranks 1 and 4 on SERIAL and HIP: the global face count
**falls** on a mesh given an artificially large `collapse_factor`, by the **same
amount at both rank counts**; `V - E + F == 2` after the pass; and the failure
direction — `--remesh-collapse-factor 0` produces `collapses == 0` with no
mutation, and `--remesh-max-collapses 0` does **not** disable the pass (it means
unlimited), asserted so the deleted rejection's warning cannot be re-lost.

---

### T4d3 — the two quality-flip entry points — **NOT STARTED**

**Depends on:** T4d1.

**Fill in:** `DynamicRemesh::flipEdgesForQuality`
([:911](../src/Beatnik_DynamicRemesh.hpp#L911)),
`MeshQuality::improveConnectivityByFlips`
([src/Beatnik_MeshQuality.hpp:205](../src/Beatnik_MeshQuality.hpp#L205)), the
`diag.flips += 0` placeholder ([:1042-1044](../src/Beatnik_DynamicRemesh.hpp#L1042-L1044)),
the `const int flips = 0` on the refine branch
([src/Beatnik_Solver.hpp:467](../src/Beatnik_Solver.hpp#L467)), and the deletion
of two rejections ([:752-759](../src/Beatnik_Solver.hpp#L752-L759),
[:839-850](../src/Beatnik_Solver.hpp#L839-L850)).

**Reference:** `dynamic_remesh.py::flip_edges_for_quality` (408-457) for the
criterion; `mesh_solver.py::improve_mesh_connectivity_by_edge_flips` (1704-1772)
for the second entry point's pass structure and its `reset_reference` argument.

**Do:**

1. Write the **gain predicate once** and give both call sites the mask it
   produces: mark an edge iff its quad `(a,b,c,d)` yields
   `min(q_new) >= min_quality` **and** `min(q_new) > min(q_old) * (1 + gain)`,
   with `gain` = `--remesh-flip-min-gain`, and the opposite diagonal `(c,d)` does
   not already exist. The quad comes from `buildFaceAdjacency` and is resident at
   halo depth 2.
2. Do **not** express the gain through `DefaultFlipPolicy` — it offers absolute
   admissibility only, and its priority is longest-edge-first, not gain-ordered.
   A marked edge Tessera rejects lands in `rejectedGeometric`; that is correct.
3. Answer T4c's deferred question — **does the flip path re-base the AMR
   reference state?** The reference passes `reset_reference=(smooth_iters == 0)`
   at the refine call site (`run_adaptive_mesh_bubble.py:1440-1445`) and nothing
   at the remesh one. Record the decision and its reasoning in the log; the
   comment at [src/Beatnik_MeshQuality.hpp:195-203](../src/Beatnik_MeshQuality.hpp#L195-L203)
   is where T4c parked it.
4. `--flip-passes` is **0** in the milestone configuration, so
   `improveConnectivityByFlips` has no milestone caller. Implement it anyway —
   the rejection at [:752](../src/Beatnik_Solver.hpp#L752) is being deleted, so
   the path becomes reachable — and say in the log that it is unexercised by any
   milestone run (the standing `relaxTangential` situation from T4c).

**Exit criterion:** a `unit`-tier case at ranks 1 and 4, both backends, shows the
global minimum triangle quality **rises or holds** across a pass and never falls,
and the count of faces below `--remesh-min-quality` does not increase.
**Statistics are compared, never flip sets** (R7). Failure direction:
`--remesh-flip-min-gain 1e12` yields `flips == 0` and a byte-identical mesh,
which is what the T4b regression member already depends on.

---

### T4d4 — `DynamicRemesh::tangentialSmooth` — **NOT STARTED**

**Depends on:** none. It moves vertices and changes no connectivity; it was
parked under T4d only because it was unreachable while the flips were.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:936](../src/Beatnik_DynamicRemesh.hpp#L936),
the `diag.smooth_steps += 0` placeholder
([:1045](../src/Beatnik_DynamicRemesh.hpp#L1045)), and the deletion of the
rejection at [src/Beatnik_Solver.hpp:830-837](../src/Beatnik_Solver.hpp#L830-L837).

**Reference:** `dynamic_remesh.py::tangential_smooth_vertices` (459-490) — **not**
`mesh_solver.py::improve_mesh_quality_tangential`, which is T4c's landed
`MeshQuality::relaxTangential`.

**Do:**

1. Reuse `relaxTangential`'s kernel; do not write a second umbrella operator.
2. Add what the reference has and `relaxTangential` does not (Read this first
   #5): per iteration, compute *trial* positions, reduce the global minimum
   triangle quality over the trial mesh, and **reject the whole iteration and
   stop** if `new + 1e-14 < max(min_quality, 0.85 * old)`. Return the count of
   **accepted** iterations as `smooth_steps`. The reduction is global, so every
   rank must take the same decision — reduce, then branch on the reduced value.
3. Skip vertices with fewer than three neighbours, as the reference does.
4. Keep the normal projection. Without it this is Laplacian smoothing of the
   interface, which shrinks the bubble — a failure that reads as excessive
   numerical dissipation rather than as the geometry bug it is.

**Exit criterion:** a `unit`-tier case at ranks 1 and 4, both backends: on the
default icosphere the enclosed volume changes by less than `1e-12` relative
across a pass (shape preserved), the minimum quality does not fall, and the
accept/reject guard is exercised in **both** directions — a mesh constructed so
the trial worsens quality past the `0.85` bound returns `smooth_steps == 0` with
positions **unchanged**, and a healthy mesh returns
`smooth_steps == smoothing_iterations`.

---

### T4d5 — `MeshQuality::{valenceEqualizingFlips, isotropicCleanup}` and the cleanup call site — **NOT STARTED**

**Depends on:** T4d1, T4d3 (shares the flip adapter and the mask machinery), and
T4c (landed).

**Fill in:** [src/Beatnik_MeshQuality.hpp:108](../src/Beatnik_MeshQuality.hpp#L108)
and [:177](../src/Beatnik_MeshQuality.hpp#L177); the two call sites in
`Solver::advanceOneStep` — the remesh branch
([src/Beatnik_Solver.hpp:527-547](../src/Beatnik_Solver.hpp#L527-L547)) and the
refine branch ([:459-466](../src/Beatnik_Solver.hpp#L459-L466)); and the deletion
of the rejection at [:769-775](../src/Beatnik_Solver.hpp#L769-L775).

**Reference:** `mesh_quality.py::_valence_equalizing_flips` (44-87) and
`::isotropic_cleanup` (146-167); the call sites at
`run_adaptive_mesh_bubble.py:1452-1464` (refine) and `:1491-1504` (remesh).

**Do:**

1. `valenceEqualizingFlips`: a **valence-based** mask, not the gain mask. Mark the
   shared edge of two faces iff
   `|v_a-7| + |v_b-7| + |v_c-5| + |v_d-5| < |v_a-6| + |v_b-6| + |v_c-6| + |v_d-6|`,
   the opposite diagonal does not exist, both child normals satisfy `new·old > 0.2`
   against the pre-flip face normals, and `min(q_new) >= 0.05`. Valence is the
   face-incidence count (`np.bincount(F.ravel())`). Exit early when a pass flips
   nothing.
2. `isotropicCleanup` = `--isotropic-cleanup-flips` (3) valence passes, then
   `--isotropic-cleanup-relax` (2) passes of the landed `tangentialRelaxation` at
   `--isotropic-cleanup-weight` (0.4). Vertex count is unchanged and the shape is
   preserved, so no field is touched.
3. **The call site is two state rebuilds, not one.** The reference rebuilds the
   state inside `dynamic_remesh_state_with_material` (which re-bases the
   reference area/curvature and re-centres the potential against the new area
   weights) and **then rebuilds it again after the cleanup**
   (`run_adaptive_mesh_bubble.py:1501-1504` constructs a fresh
   `MeshPotentialZModelState`, whose `__post_init__` re-seeds both). So the
   Beatnik order is: `remesh` → `resetReferenceState` + `centerPotential` →
   `isotropicCleanup` → `resetReferenceState` + `centerPotential` →
   `projectToVolume`. Getting this wrong is invisible in a 20-step run and
   compounds over 600.
4. The cleanup fires whenever a remesh **ran**, not whenever it changed something
   (`:1491`, gated on `remesh_diag is not None`) — the same gate shape the volume
   projection already uses on that branch.
5. On the refine branch, the reference runs cleanup only when
   `refine_diag.marked_faces > 0` (`:1439`), inside the same block as the flips
   and the tangential pass, and it feeds the `repaired` gate through
   `args.isotropic_cleanup` — which
   [src/Beatnik_Solver.hpp:497-499](../src/Beatnik_Solver.hpp#L497-L499) already
   transcribes as `_params.cleanup.enabled`. Add the call; do not touch the gate.

**Exit criterion:** a `unit`-tier case at ranks 1 and 4, both backends: on a mesh
seeded with a high-valence patch the **valence histogram concentrates toward 6**
(the count of vertices with valence ≠ 6 falls) and the enclosed volume changes by
less than `1e-12` relative; and, in the failure direction,
`--no-isotropic-cleanup` leaves the mesh byte-identical while
`--isotropic-cleanup` does not, so the switch is demonstrably live. Statistics
only, never flip sets (R7).

---

### T4d6 — the gid-renumbering cadence — **NOT STARTED**

**Depends on:** T4d2.

**Fill in:** the remesh cycle in
[src/Beatnik_DynamicRemesh.hpp:995-1056](../src/Beatnik_DynamicRemesh.hpp#L995-L1056)
and `RemeshDiagnostics` ([:141-200](../src/Beatnik_DynamicRemesh.hpp#L141-L200)).

**Reference:** `Tessera_Compact.hpp` on `compact()` versus
`compactAndRenumberGids()`, and its statement that several Tessera paths index a
dense host array sized to the **max gid**.

**Do:** gids come from an `MPI_Exscan` onto a monotonically rising global count,
so a long split-and-collapse run grows the gid *space* without bound even at
constant mesh size, and `collapseEdges`'s internal `compact()` preserves gids.
Choose a renumbering cadence, document **why that cadence** (it costs a second
halo rebuild), and put the gid-space high-water mark into the remesh diagnostics
so the leak is observable rather than inferred.

**Exit criterion:** a 600-step run at ranks 1 and 4 reports a gid-space
high-water mark that is **bounded** — its ratio to the global vertex count does
not grow monotonically with step index across the last third of the run — and the
per-step resident memory does not grow with step count at constant face count.
Both numbers written into the progress log. This is the first thing to suspect if
a long run's memory grows with step count rather than with face count.

---

### T4d exit criterion

A `--dynamic-remesh` run at the **milestone defaults** (no knob turned off)
completes 50 steps at ranks 1 and 4 on both backends; volume drift below
`1e-10`; the minimum triangle quality stays above `--remesh-min-quality` for the
whole run; `Solver::requireSupportedConfiguration` rejects **nothing** for that
command line; and the run reports non-zero `splits`, `collapses`, `flips` and
`smooth_steps` totals — a run that completes with all four at zero has not
exercised T4d and is M1-R7, not a pass. Compare **statistics** against the
reference, never the flip or collapse set (R7).

---

### M1-T1 — the `milestone` test tier — **NOT STARTED**

**Depends on:** none. Do it early; it is independent of T4d and of both gold sets.

**Fill in:** [tests/CMakeLists.txt](../tests/CMakeLists.txt) — a third tier
alongside `regression` and `unit` — and a new
`scripts/tuolumne/run_milestone.flux`.

**Reference:** the tier comment at the top of `tests/CMakeLists.txt` (lines
13-45); the standalone regression registration loop (`:309-372`) which is the
shape to copy; the manifest generation (`:520-575`); the installed-path runner
[scripts/tuolumne/run_regression_minset.flux](../scripts/tuolumne/run_regression_minset.flux),
which is already parameterized by `BEATNIK_GATE_LABEL`, `BEATNIK_GATE_BACKENDS`
and `BEATNIK_GATE_RANKS` but hardcodes `beatnik_gate_manifest.txt`.

**Do:**

1. Add `BEATNIK_MILESTONE_TEST_SOURCES`, a `BEATNIK_MILESTONE_TARGETS` global
   property, `LABELS milestone` on the ctest cases, and
   `beatnik_milestone_manifest.txt` with the same line format and the same
   "paths are relative to this file's directory" convention.
2. Install the manifest and the tier's gold data under
   `share/Beatnik/tests`, preserving the repo layout, exactly as the regression
   tier's `install()` rules do — and keep the existing `FATAL_ERROR` pattern: a
   milestone test installed without its gold set is not installed.
3. `run_milestone.flux` runs the tier at ranks **1 and 4** on SERIAL and HIP. Do
   not generalize `run_regression_minset.flux` in place; the gate script is
   single-sourced against CLAUDE.md's gate definition and must keep saying
   `regression` × ranks 1-6.
4. Update [docs/testing.md](../docs/testing.md) and CLAUDE.md's "Minimum test
   set" to name the third tier and to state that it is **not** part of the gate.
   The gate stays at five members / 60 launches.

**Exit criterion:** with an empty tier,
`flux batch scripts/tuolumne/run_milestone.flux` exits **non-zero** with the
"named no runnable tests" message — the vacuous-pass guard the gate runner
already has (`run_regression_minset.flux:200-208`) — and
`ctest -N -L milestone` in a `manual`-mode build tree lists zero tests without
error. After M1-T2 registers a member, both list exactly it, at two rank counts
per backend, and `ctest -N -L regression` still lists **60** cases.

---

### M1-T2 — the milestone comparison test — **NOT STARTED**

**Depends on:** T4d1, T4d2, T4d3, T4d4, T4d5, T4d6, M1-G2, M1-T1, M1-D1.

**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone1Bubble.cpp`,
registered in the `milestone` tier with the gold **directory** and the comparator
as its two arguments.

**Reference:** [tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp)
end to end — `makeParams()` (`:307-379`) for expressing a Python command line as
a `SolverParams`, `goldForStep()` (`:263-286`) for finding a step's gold file by
its `_step%07d.npz` suffix rather than by rebuilding a filename from a time (the
time is under test), and the `compare_output.py` subprocess wrapper (`:296-306`)
which demands exit status **exactly 1** for a mismatch and not 2, so a
mis-resolved path cannot masquerade as a pass.

**Do:**

1. `makeParams()` is the milestone command line — the four switches at their
   defaults, `--checkpoint-every-steps 10`, `--steps` from M1-A1's decided depth
   (not 600 unless M1-A1 says so).
2. Compare every checkpointed step up to the decided depth at the decided
   tolerances. Where the ladder is not flat, put the per-step tolerance in a
   compiled-in table with the measured `max|e|` beside it as a comment, so a
   later session can see the headroom rather than re-measure it.
3. Assert the trajectory is *live*: cumulative `splits`, `collapses`, `flips`,
   `smooth_steps` and cleanup flips all non-zero by the end (M1-R7), and the
   per-step `time` series matches the gold's — a fixed-dt run reproduces neither.
4. Report the wall time and the face-count trajectory in the test's output; they
   are what tells the next session whether a deeper compare depth is affordable.

**Exit criterion:** `flux batch scripts/tuolumne/run_milestone.flux` reports PASS
at ranks 1 and 4 on SERIAL and HIP with zero `[FAIL]` lines, and
`ctest -L regression -R SERIAL` / `-R HIP` still pass unchanged at 60 launches
(`flux batch scripts/tuolumne/run_regression_minset.flux`). Failure direction: the
same test invoked against the **step-0** gold for a later step exits exactly 1
from the comparator — a detected mismatch, not a load error — and a build with
`--no-isotropic-cleanup` forced fails the comparison rather than passing with a
different mesh.

---

### M1-A1 — fix the compare depth and the tolerance ladder — **NOT STARTED**

**Depends on:** M1-D1, M1-T2.

Not a coding task: the decision the user deferred until the measurements exist,
and it must be *recorded* rather than left implicit in a test's literals.

**Do:** bring together M1-D1's fixed-mesh horizon, M1-G2's face-count and `time`
series, and the first failing step of the full-configuration comparison. Decide,
with the user: the milestone's compare depth; the tolerance at each compared
step; whether steps beyond the depth are compared **structurally/statistically**
instead (counts, valence histogram, quality distribution, volume drift, `time`
series) or not at all; and whether the 5000-step goal in the original brief is
pursued, and under what criterion.

**Additional information needed, and which task answers it:** how fast the two
codes diverge with connectivity frozen (**M1-D1**); how much of the remaining
divergence is edit-set rather than floating-point, distinguished by whether the
face **counts** agree at the first failing step (**M1-T2**'s output); and whether
a 600-step milestone run is affordable at all rank counts (**M1-T2**'s wall-time
report, **T4d6**'s memory numbers).

**Exit criterion:** this document's M1-T2 entry and its `## Problem` section state
the decided depth, tolerances and beyond-depth treatment as numbers; the test
asserts exactly those numbers; and the progress log records the alternative that
was rejected and why. The milestone is **accomplished** when that test is green
at ranks 1 and 4 on both backends.

## Known risks

**M1-R1 — the divergence horizon is shorter than the milestone's premise.** The
two codes stop agreeing at `1e-10` well before step 500. *Presents as:* M1-D1
reporting a first-failing step in the tens, on a run with no adaptivity at all.
*Do:* this is information, not a bug — take it to M1-A1. Do **not** loosen
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
anything.

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
a long run. Owned by T4d6; suspect it first for any late-run memory symptom.

**M1-R6 — cost.** The direct Birkhoff-Rott sum is `O(N^2)` per RK stage, three
stages per step, and nothing caps the face count on this path (Read this first
#3). *Presents as:* a milestone launch that times out rather than fails. *Do:*
M1-D1 and M1-T2 both report the face-count trajectory and wall time; if the
trajectory is superlinear in step index, raise it at M1-A1 rather than silently
lowering the step count — a quietly truncated milestone run is a gate that
shrank.

**M1-R7 — a milestone test that passes because nothing happened.** R15's trap in
this milestone's shape: if the sizing field never triggers a split, or
`isotropicCleanup` never flips, the run reproduces a simpler trajectory and the
comparison passes while testing none of T4d. *Presents as:* it does not — a green
test and no signal. *Distinguished by:* the four cumulative edit counters and the
cleanup flip count, asserted non-zero (T4d exit criterion, M1-T2 step 3).

**M1-R8 — the quantized pairing degrades silently.** `compare_output.py` sorts
each file's vertices *independently* at `--match-eps 1e-9`. Once positions
disagree by more than the cell size near a cell boundary, the two sorts pair
different vertices and every field reports large errors that have nothing to do
with the field. *Presents as:* huge, uniform `max|e|` across all fields at once,
often with `ambiguous cpp=0 gold=0`. *Distinguished from a real field error by:*
`vertices` itself failing first and by the same magnitude as the other fields.
*Do:* report it as a pairing failure and record the step; do not raise
`--match-eps` without stating what it now cannot detect.
