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
([src/Beatnik_Solver.hpp:727-859](../src/Beatnik_Solver.hpp#L727-L859)). Four
rejections fire and every one of them is **T4d**:

| Option, at its default | Rejection | Needs |
| --- | --- | --- |
| `--isotropic-cleanup` (default **on**) | [:759-765](../src/Beatnik_Solver.hpp#L759-L765) | `MeshQuality::isotropicCleanup` |
| `--remesh-collapse-factor 0.45` | [:809-817](../src/Beatnik_Solver.hpp#L809-L817) | `DynamicRemesh::collapseShortEdges` |
| `--remesh-smooth-iters 1` | [:819-826](../src/Beatnik_Solver.hpp#L819-L826) | `DynamicRemesh::tangentialSmooth` |
| `--remesh-flip-min-gain 1e-3` | [:828-837](../src/Beatnik_Solver.hpp#L828-L837) | `DynamicRemesh::flipEdgesForQuality` |

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

Six facts about the reference, the comparator and Tessera that every task below
builds on.

### 1. The target command runs 140 steps and writes no checkpoints

`--steps` defaults to **140**, `--checkpoint-dir` to `""` and
`--checkpoint-every-steps` to **0** (`run_adaptive_mesh_bubble.py:133,316,329`),
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
2. **Land T4d** (M1-U1, M1-U2, M1-U3, T4d1–T4d6). M1-U1, M1-U2 and M1-U3 are
   small upstream Tessera changes that T4d2, T4d3 and T4d5 need (Read this first
   #6); T4d1 is an
   adapter layer, T4d2–T4d5 are one reference function each, T4d6 is an
   operational decision.
3. **Wire the milestone test** (M1-T1, M1-T2) and **fix the acceptance**
   (M1-A1).

### Conventions

Framework.md's conventions table governs; these are the additions this milestone
introduces or sharpens.

| Convention | Choice |
| --- | --- |
| Task IDs | `T4d*` for work moved from framework.md (IDs are referenced from source strings); `M1-*` for work new here. `M1-G*` = human gold-file generation, `M1-D*` = measurement, `M1-T*` = code, `M1-U*` = a change made upstream in `../tessera`, `M1-A*` = a decision to be recorded. |
| Edge-mask type | host `const std::vector<char>&`, sized `ownedEdgeCount()`, one entry per **owned** edge — T4a's convention for `splitEdges`, and exactly what Tessera's collapse and flip take. De-template the two stale `EdgeListView` signatures rather than adding an overload. |
| Where a reference acceptance rule lives | **In the mask whenever it needs only the quad; in the Tessera policy only when it needs both endpoints' one-rings.** Flip tests (the gain test, the valence test, the valence path's normal and quality tests) and the collapse candidate predicate and normalized-length ranking are evaluated by Beatnik while building the mask, and both flip call sites pass the permissive `DefaultFlipPolicy{ maxNormalDeviation = π, minQuality = 0 }`, the valence call site with `conflict = FlipConflict::EndpointInQuad` (M1-U2). Collapse's geometric test needs the full one-rings, so it is a Tessera policy: `maxNormalRotation = acos(0.1)`, `minQuality = --remesh-min-quality`, and the reference quality metric through M1-U1's hook. Read this first #6 has the evidence. A marked edge may still be rejected; that shows up in the `rejected*` counters, never as a silent drop. |
| Reference-state re-basing | every pass that moves a vertex or changes connectivity states, in its progress-log entry, whether it calls `AdaptiveMesh::resetReferenceState`, and why. A silent choice here changes every later refinement decision. |
| Failure behavior | loud. A configuration Beatnik cannot reproduce is rejected before the first step, by method name and task ID, exactly as the four rejections this milestone deletes do. No pass may silently no-op. |
| Diagnostics that must be numbers | per remesh pass: `splits`, `collapses`, `flips`, `smooth_steps`, `min_quality` before/after, R12's two shape signals, and (T4d6) the gid-space high-water mark. A milestone run that reports zero edits everywhere is M1-R7, not a pass. |
| Unit-tier rank and backend coverage | Unit members register at **one** rank on the build's default execution space ([tests/unit_tests/CMakeLists.txt:68-80](../tests/unit_tests/CMakeLists.txt#L68-L80)). "Ranks 1 and 4" in a T4d exit criterion means two submissions, `BEATNIK_UNIT_RANKS=1` and `BEATNIK_UNIT_RANKS=4 flux batch scripts/tuolumne/unit_tests.flux`, and the member's checks must be global reductions so one set of literals holds at both, as `Beatnik_Test_TangentialRelaxation` does. The unit tier has no SERIAL/HIP pair; the `regression` gate and the T4d umbrella run supply backend coverage. |
| Test tier for the milestone comparison | the **`milestone`** label, at ranks **1 and 4** on SERIAL and HIP, outside the 60-launch ship gate. The tier is created by **M0-T1** in [`milestone0.md`](completed/milestone0.md); M1-T1 registers a member in it. The `regression` tier keeps exactly its five members; promoting anything into it needs the user's confirmation (CLAUDE.md "Minimum test set"). |
| Provenance comments | `// Port of <file>::<fn> (lines N-M)` against the **real** origin, per framework.md. The functions this milestone ports live in `dynamic_remesh.py` and `mesh_quality.py`; cite line ranges, not function names alone. |

### Deliberate deviations

- **Beatnik's accepted edit sets will not equal the reference's**, and no attempt
  is made to make them. The alternative — a sequential-equivalent scheduler in
  Tessera plus reduction-order pinning — was considered and rejected with the
  user: it is large, it reaches upstream, and it still cannot make a NumPy and a
  Kokkos trajectory agree over hundreds of steps. The consequence is that the
  milestone's compare depth is an empirical quantity (M1-A1), not an assumption.
- **`--remesh-max-collapses 0` is not a lever.** A non-positive value maps to
  `None` = *unlimited* (`run_adaptive_mesh_bubble.py:1338`), reproduced by
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
  ([src/Beatnik_MeshInterface.hpp:1538](../src/Beatnik_MeshInterface.hpp#L1538),
  [:1562](../src/Beatnik_MeshInterface.hpp#L1562),
  [:1581](../src/Beatnik_MeshInterface.hpp#L1581));
  `DynamicRemesh::{collapseShortEdges, flipEdgesForQuality, tangentialSmooth}`
  ([src/Beatnik_DynamicRemesh.hpp:892](../src/Beatnik_DynamicRemesh.hpp#L892),
  [:914](../src/Beatnik_DynamicRemesh.hpp#L914),
  [:940](../src/Beatnik_DynamicRemesh.hpp#L940));
  `MeshQuality::{valenceEqualizingFlips, isotropicCleanup, improveConnectivityByFlips}`
  ([src/Beatnik_MeshQuality.hpp:111](../src/Beatnik_MeshQuality.hpp#L111),
  [:182](../src/Beatnik_MeshQuality.hpp#L182),
  [:210](../src/Beatnik_MeshQuality.hpp#L210)).
- **Shaped for the wrong result:** `SurfaceMesh::flipEdges` returns `int`, and
  `MeshEditReport`
  ([src/Beatnik_MeshInterface.hpp:373-391](../src/Beatnik_MeshInterface.hpp#L373-L391))
  carries only split counters — nothing for `CollapseResult`'s or
  `FlipResult`'s `requested`/`accepted`/`rejected*`/`*Removed` fields. Neither
  adapter takes a Tessera policy.
- **Present upstream, unused by Beatnik:** Tessera's collapse and flip policies
  carry the `quality()` hook (M1-U1, **DONE**, Tessera `279287e`), defaulting to
  the radius ratio; no Beatnik policy uses it yet (T4d2). `flipEdges` has the
  opt-in `FlipConflict::EndpointInQuad` relation beside the default per-face one
  (M1-U2, **DONE**, Tessera `05b5942`); no Beatnik call site sets it yet (T4d5).
  `flipEdges` accepts at most one of several flips in one call that would create
  the same new diagonal, under either relation (M1-U3, **DONE**, Tessera
  `33dc6b9`); no Beatnik call site calls `flipEdges` yet (T4d3, T4d5).
- **Defined and deliberately inert**, transcribed rather than folded away, so
  landing T4d turns them on by deleting a rejection instead of by remembering to
  add a call: `const int collapses = 0` and `diag.flips += 0; diag.smooth_steps += 0`
  inside the pass loop
  ([src/Beatnik_DynamicRemesh.hpp:1009-1048](../src/Beatnik_DynamicRemesh.hpp#L1009-L1048)),
  and `const int flips = 0` on the refine branch
  ([src/Beatnik_Solver.hpp:465](../src/Beatnik_Solver.hpp#L465)). Note the
  **sizing-field recompute before the collapse pass is absent**, not merely
  unused — the comment at
  [:1020-1031](../src/Beatnik_DynamicRemesh.hpp#L1020-L1031) records that it is
  load-bearing the moment collapse lands, because without it the collapse pass
  undoes the split pass.
- **Missing entirely from the remesh branch:** the `isotropic_cleanup` call site
  and the *second* state rebuild that follows it
  ([src/Beatnik_Solver.hpp:521-541](../src/Beatnik_Solver.hpp#L521-L541) has a
  comment where the cleanup goes, then the first rebuild).
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
`Beatnik_Test_DirectSolve10Steps.cpp:263-304` for the `goldForStep` +
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

### T4d1 — the three `SurfaceMesh` adapters — **NOT STARTED**

**Depends on:** none.

**Fill in:** `SurfaceMesh::{collapseEdges, flipEdges, compact}` in
[src/Beatnik_MeshInterface.hpp](../src/Beatnik_MeshInterface.hpp) at
[:1518-1543](../src/Beatnik_MeshInterface.hpp#L1518-L1543),
[:1545-1567](../src/Beatnik_MeshInterface.hpp#L1545-L1567),
[:1569-1584](../src/Beatnik_MeshInterface.hpp#L1569-L1584), and
`MeshEditReport` ([:373-391](../src/Beatnik_MeshInterface.hpp#L373-L391)).

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
   compacts internally) and that it exists for T4d6 and for tests. The
   `flipEdges` comment describes the reference's flip
   (`dynamic_remesh.py:408-458`), not what Tessera checks; replace it with
   Tessera's tests and point at Read this first #6.

**Exit criterion:** a new `unit`-tier member
`tests/unit_tests/Beatnik_Test_MeshEdits.cpp`, registered in
`BEATNIK_UNIT_TEST_SOURCES`
([tests/unit_tests/CMakeLists.txt:42](../tests/unit_tests/CMakeLists.txt#L42)), passes at
each of `BEATNIK_UNIT_RANKS=1` through `6` via
`flux batch scripts/tuolumne/unit_tests.flux`: on the default
icosphere it collapses a named edge set, flips another, and compacts, and checks
(a) the three V/E/F identities, (b) `V - E + F == 2` globally afterwards,
(c) gid preservation across `compact()`, and (d) the **failure direction** — a
mask marking a boundary-of-nothing/link-condition-violating edge is *rejected*
and shows up in `rejectedLinkCondition`, with `accepted` unchanged, rather than
producing a torn mesh.

---

### T4d2 — `DynamicRemesh::collapseShortEdges` — **NOT STARTED**

**Depends on:** T4d1, M1-U1.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:892](../src/Beatnik_DynamicRemesh.hpp#L892);
the sizing-field recompute and the `const int collapses = 0` placeholder inside
the pass loop at [:1020-1032](../src/Beatnik_DynamicRemesh.hpp#L1020-L1032);
and the deletion of the rejection at
[src/Beatnik_Solver.hpp:797-817](../src/Beatnik_Solver.hpp#L797-L817), comment
included.

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
unlimited), asserted so the deleted rejection's warning cannot be re-lost.

---

### T4d3 — the two quality-flip entry points — **NOT STARTED**

**Depends on:** T4d1, M1-U3 (two quality flips in one round must not create the same diagonal).

**Fill in:** `DynamicRemesh::flipEdgesForQuality`
([src/Beatnik_DynamicRemesh.hpp:914](../src/Beatnik_DynamicRemesh.hpp#L914)),
`MeshQuality::improveConnectivityByFlips`
([src/Beatnik_MeshQuality.hpp:210](../src/Beatnik_MeshQuality.hpp#L210)), the
`diag.flips += 0` placeholder ([src/Beatnik_DynamicRemesh.hpp:1044-1046](../src/Beatnik_DynamicRemesh.hpp#L1044-L1046)),
the `const int flips = 0` on the refine branch
([src/Beatnik_Solver.hpp:456-465](../src/Beatnik_Solver.hpp#L456-L465)), and the deletion
of two rejections ([:746-751](../src/Beatnik_Solver.hpp#L746-L751),
[:828-837](../src/Beatnik_Solver.hpp#L828-L837)).

**Reference:** `dynamic_remesh.py::flip_edges_for_quality` (408-457) for the
criterion; `mesh_solver.py::improve_mesh_connectivity_by_edge_flips` (1704-1772)
for the second entry point's pass structure and its `reset_reference` argument.

**Do:**

1. Write the **gain predicate once** and give both call sites the mask it
   produces: mark an edge iff its quad `(a,b,c,d)` yields
   `min(q_new) >= min_quality` **and** `min(q_new) > min(q_old) * (1 + gain)`,
   with `q` the reference metric (`SurfaceOperators::triangleQuality`, not the
   radius ratio) and `gain` = `--remesh-flip-min-gain`. The reference has no
   normal test here, only a re-orientation of the children against the old
   faces' summed normal (`dynamic_remesh.py:434-443`), which Tessera's
   orientation-preserving rewrite already guarantees. The quad comes from
   `SurfaceMesh::faceAdjacency()`
   ([src/Beatnik_MeshInterface.hpp:1278](../src/Beatnik_MeshInterface.hpp#L1278)).
   Reading the neighbour face's positions makes this a *geometric* consumer, so
   it must check `non_resident == 0` rather than assume halo depth 2 supplies
   the quad ([:586-596](../src/Beatnik_MeshInterface.hpp#L586-L596)), and throw
   if it does not hold.
2. **Do not test whether `(c,d)` already exists** — Tessera rejects that flip
   exactly, into `rejectedDuplicateEdge` (Read this first #6).
3. Call `SurfaceMesh::flipEdges` with
   `Tessera::DefaultFlipPolicy{ maxNormalDeviation = π, minQuality = 0 }`. Do
   **not** express the gain through the policy — the policy offers absolute
   admissibility only, and its priority is longest-edge-first, not gain-ordered.
   Tessera's conflict relation (two flips may not share a face) is the
   reference's here (`dynamic_remesh.py:416-421`), so loop
   `flipEdges` while `accepted > 0`, rebuilding the mask each round.
4. Answer T4c's deferred question — **does the flip path re-base the AMR
   reference state?** The reference passes `reset_reference=(smooth_iters == 0)`
   at the refine call site (`run_adaptive_mesh_bubble.py:1440-1445`) and nothing
   at the remesh one. Record the decision and its reasoning in the log; the
   comment at [src/Beatnik_MeshQuality.hpp:200-209](../src/Beatnik_MeshQuality.hpp#L200-L209)
   is where T4c parked it.
5. `--flip-passes` is **0** in the milestone configuration, so
   `improveConnectivityByFlips` has no milestone caller. Implement it anyway —
   the rejection at [src/Beatnik_Solver.hpp:746](../src/Beatnik_Solver.hpp#L746) is being deleted, so
   the path becomes reachable — and say in the log that it is unexercised by any
   milestone run (the standing `relaxTangential` situation from T4c).

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4 shows the
global minimum triangle quality **rises or holds** across a pass and never falls,
and the count of faces below `--remesh-min-quality` does not increase.
**Statistics are compared, never flip sets** (R7). Failure direction:
`--remesh-flip-min-gain 1e12` yields `flips == 0` and a byte-identical mesh,
which is what the T4b regression member already depends on.

---

### T4d4 — `DynamicRemesh::tangentialSmooth` — **NOT STARTED**

**Depends on:** none. It moves vertices and changes no connectivity; it was
parked under T4d only because it was unreachable while the flips were.

**Fill in:** [src/Beatnik_DynamicRemesh.hpp:940](../src/Beatnik_DynamicRemesh.hpp#L940),
the `diag.smooth_steps += 0` placeholder
([:1047](../src/Beatnik_DynamicRemesh.hpp#L1047)), and the deletion of the
rejection at [src/Beatnik_Solver.hpp:819-826](../src/Beatnik_Solver.hpp#L819-L826).

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

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4: on the
default icosphere the enclosed volume changes by less than `1e-12` relative
across a pass (shape preserved), the minimum quality does not fall, and the
accept/reject guard is exercised in **both** directions — a mesh constructed so
the trial worsens quality past the `0.85` bound returns `smooth_steps == 0` with
positions **unchanged**, and a healthy mesh returns
`smooth_steps == smoothing_iterations`.

---

### T4d5 — `MeshQuality::{valenceEqualizingFlips, isotropicCleanup}` and the cleanup call site — **NOT STARTED**

**Depends on:** T4d1, T4d3 (shares the flip adapter and the mask machinery),
M1-U2, M1-U3, and T4c (landed).

**Fill in:** [src/Beatnik_MeshQuality.hpp:111](../src/Beatnik_MeshQuality.hpp#L111)
and [:182](../src/Beatnik_MeshQuality.hpp#L182); the two call sites in
`Solver::advanceOneStep` — the remesh branch
([src/Beatnik_Solver.hpp:521-541](../src/Beatnik_Solver.hpp#L521-L541)) and the
refine branch ([:456-465](../src/Beatnik_Solver.hpp#L456-L465)); and the deletion
of the rejection at [:759-765](../src/Beatnik_Solver.hpp#L759-L765).

**Reference:** `mesh_quality.py::_valence_equalizing_flips` (44-87) and
`::isotropic_cleanup` (146-167); the call sites at
`run_adaptive_mesh_bubble.py:1452-1464` (refine) and `:1491-1504` (remesh).

**Do:**

1. `valenceEqualizingFlips`: a **valence-based** mask, not the gain mask. Mark the
   shared edge `(a,b)` of faces `f0` (opposite vertex `c`) and `f1` (opposite
   `d`) iff
   `|v_a-7| + |v_b-7| + |v_c-5| + |v_d-5| < |v_a-6| + |v_b-6| + |v_c-6| + |v_d-6|`,
   the children `(c,d,a)` and `(d,c,b)` satisfy `n·n_f0 ≥ 0.2` and `n·n_f1 ≥ 0.2`
   respectively on unit normals (`mesh_quality.py:76-79`), and
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

**Exit criterion:** a `unit`-tier case at `BEATNIK_UNIT_RANKS` 1 and 4: on a mesh
seeded with a high-valence patch, every accepted flip of a pass has no endpoint
touched earlier in that pass (checked from the gathered `flipped` lists); the
**valence histogram concentrates toward 6**
(the count of vertices with valence ≠ 6 falls) and the enclosed volume changes by
less than `1e-12` relative; and, in the failure direction,
`--no-isotropic-cleanup` leaves the mesh byte-identical while
`--isotropic-cleanup` does not, so the switch is demonstrably live. Statistics
only, never flip sets (R7).

---

### T4d6 — the gid-renumbering cadence — **NOT STARTED**

**Depends on:** T4d2.

**Fill in:** the remesh cycle in
[src/Beatnik_DynamicRemesh.hpp:998-1056](../src/Beatnik_DynamicRemesh.hpp#L998-L1056)
and `RemeshDiagnostics` ([:142-196](../src/Beatnik_DynamicRemesh.hpp#L142-L196)).

**Reference:** `Tessera_Compact.hpp` on `compact()` versus
`compactAndRenumberGids()` (`:541-570`), and its statement that several Tessera
paths index a dense host array sized to the **max gid** (`:77-94`).

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

### M1-T1 — register milestone 1 in the `milestone` test tier — **NOT STARTED**

**Depends on:** **M0-T1** (**DONE**, [`milestone0.md`](completed/milestone0.md)), which **creates**
the tier — the CMake wiring, `beatnik_milestone_manifest.txt`, the installed-path
gold-data rules, `scripts/tuolumne/run_milestone.flux` at ranks 1 and 4 on SERIAL
and HIP, and the `docs/testing.md` / CLAUDE.md text saying the tier does not gate.
Milestone 0 lands first and needs the same tier, so it owns it and this task no
longer builds it. Nothing else here depends on M0.

**Fill in:** `BEATNIK_MILESTONE_TEST_SOURCES`
([tests/CMakeLists.txt:429](../tests/CMakeLists.txt#L429)) and the
`_beatnik_args_Beatnik_Test_Milestone1Bubble_abs`/`_rel` pair beside milestone
0's (`:475-513`), plus the `install()` rule for the M1-G2 gold directory beside
milestone 0's (`:828-858`).

**Reference:** what M0-T1 built — the milestone registration loop (`:515`) — and
the regression tier's registration loop (`:323-385`) and gold-set `install()`
rules (`:810-826`) that it copied.
Read the M0-T1 entry in [`milestone0-progress-log.md`](completed/milestone0-progress-log.md)
before starting: if M0-T1 departed from its stated plan, this is the task that
finds out.

**Do:** add the source, its two argument lists (the gold **directory** and the
comparator) and the gold-data `install()` rule with its `FATAL_ERROR` — a
milestone test installed without its gold set is not installed. Change nothing
about the tier's ranks, backends or manifest format; those are M0-T1's, and
milestone 1 has no reason to differ.

**Exit criterion:** `ctest -N -L milestone` lists the M1-T2 member at two rank
counts per backend alongside milestone 0's, `flux batch
scripts/tuolumne/run_milestone.flux` runs it, and `ctest -N -L regression` still
lists **60** cases.

---

### M1-T2 — the milestone comparison test — **NOT STARTED**

**Depends on:** M1-U1, M1-U2, M1-U3, T4d1, T4d2, T4d3, T4d4, T4d5, T4d6, M1-G2, M1-T1, M1-D1.

**Fill in:** `tests/regression_tests/Beatnik_Test_Milestone1Bubble.cpp`,
registered in the `milestone` tier with the gold **directory** and the comparator
as its two arguments.

**Reference:** [tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp](../tests/regression_tests/Beatnik_Test_DirectSolve10Steps.cpp)
end to end — `makeParams()` (`:308-378`) for expressing a Python command line as
a `SolverParams`, `goldForStep()` (`:263-286`) for finding a step's gold file by
its `_step%07d.npz` suffix rather than by rebuilding a filename from a time (the
time is under test), and the `compare_output.py` subprocess wrapper
`runComparator()` (`:291-304`)
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
series) or not at all; and whether a 5000-step comparison is
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

