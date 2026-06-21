# Premature FMM NaN at full rollup — investigation & fix

**Status: RESOLVED & CONFIRMED** (last updated 2026-06-18). Root cause found and
fixed (far-field ignored the Plummer softening); the full crash-deck FMM run
completes **all 1400 steps with 0 NaN and 0 bbox-escape Rebuilds**. Temporary
diagnostics removed, runtime knob added, branches merged (Beatnik `debug-nan` →
`develop-canopy`; Canopy `debug-nan` → `redesign`). The final clean 256×256
FMM-with-IO confirmation run (job `f3FnURKDGXz7`) completed 1400 steps clean with
71 output frames, and **the FMM and exact interface trajectories match exactly
when graphed** — closing the issue. See the **Resolution** section below. See the latest Investigation-log entry and the (forthcoming)
Resolution section. With the Slingshot NIC-registration
crash RESOLVED (see [fmm_fullrollup_crash.md](fmm_fullrollup_crash.md)), the
full crash-deck FMM run now reaches step **1363** and aborts at step **1364**
via the ZModel NaN/Inf interface-velocity guard. The **exact BR solver completes
all 1400 steps** on the identical deck, so this is an **FMM accuracy/robustness
failure**, not the physical singularity. This is the gating issue for full-rollup
completion (investigation **thread 2** from the NIC-crash record).

## How to use this file

This is a **living document** — update it as the investigation continues so we
keep a record of how the bug was diagnosed and fixed:

- After each experiment or work session, append a dated entry to the
  **Investigation log** at the bottom (what was tried, the result, and the
  conclusion drawn). Newest entries at the end.
- Keep the **Status** line current (OPEN / IN PROGRESS / RESOLVED).
- When a hypothesis is confirmed or ruled out, note it in **Background** so
  future sessions don't re-derive it.
- When resolved: add a **Resolution** section summarizing the root cause and the
  fix, and link the commits.
- Mirror anything load-bearing back to the "Known issues" entry in
  [../CLAUDE.md](../CLAUDE.md) so the top-level pointer stays accurate.

## Background (established facts — don't re-derive)

- The physics is sound: the **exact BR solver completes all 1400 steps** on the
  identical deck with no NaN (`rocketrig_debug_exact.f3Eozj7DCRQs.log`:
  `solve() total wallclock = 108.8 s (1400 steps)`). So the failure is
  FMM-specific.
- The NaN onset is locked to the run's **first-and-only `Rebuild` actions**.
  From `rocketrig_debug_fmm.f3F1T7e6F24F.log`: steps 1362.2, 1363.0, 1363.1,
  1363.2 are all `Rebuild`; everything before is Migrate/Rebalance. Once the
  bbox escape starts it is continuous — every substep rebuilds.
  - `Rebuild` fires from `auto_maintain` → `TreeBuilder::needs_rebuild`
    (`Canopy_Solver.hpp:358-377`, `Canopy_TreeBuilder.hpp:860-890`) = at least
    one particle escaped the global bounding box.
- The 1363 `Rebuild` substeps produced **finite** zdot (the guard would have
  fired at step 1363 otherwise). The blow-up is at **1364.0**, the next solve.
- The guard reports `196608` NaN/Inf zdot values. `196608 = 256×256×3` = the
  **total owned-node DOF count** the ZModel guard reduces over (65536 nodes × 3
  components, global mesh 256×256). The count matching the full field is the
  basis for calling this a "whole-field blow-up" — an *inference from the count*,
  not yet a direct check that every source is non-finite. A whole-field NaN
  points at the **far-field multipole path** (P2M → M2M → M2L → L2L touches
  every target through the tree, so one corrupt source moment contaminates all
  targets) rather than the near-field **P2P** path (`Canopy_P2P.hpp:880-885`),
  which is Plummer-softened (`eps2 = softening² = epsilon = 2`) and `r²<1e-24`
  guarded, so it cannot produce `r=0` infinities. **Falsifiable:** if Step-1
  diagnostics show only a subset/single source is NaN at birth, re-point the
  investigation.
- `Rebuild` recomputes the bounding box **fresh** from current positions
  (`compute_global_bounding_box`, `Canopy_TreeBuilder.hpp:299-379`), which
  already guards against non-finite coordinates with an `MPI_Abort`. So at solve
  time the box *contains* all particles — the NaN is **not** an out-of-box index.
  Current leading hypothesis: a **degenerate post-rollup geometry** — a runaway
  particle inflates the box on one axis, so with `max_depth=19` the leaf
  half-width `w_c` at the dense core underflows toward zero and the multipole
  expansion's `rho_norm = rho / w_c` / `inv_rho` / `rho^(n+j+1)` terms
  (`Canopy_LaplaceKernel.hpp`, M2L/L2L, `P_ORDER=10`) overflow to Inf/NaN. **To
  be confirmed by Step-1 diagnostics.**
- The earlier *premature FMM NaN at rollup onset* was a **separate, already
  fixed** accuracy problem — resolved by raising `P_ORDER` 6→10 and tightening
  `fmm_mac_theta` 0.6→0.4. This step-1364 NaN at full rollup is what remains.

## Repro

Config: single-mode `sech2`, 256×256 (B=4), `P_ORDER=10`, `fmm_max_depth=19`,
`fmm_mac_theta=0.4`, `fmm_imbalance_tol=0.20`, `epsilon=2`, `delta_t=0.0006`,
16 ranks / 4 nodes. NaN aborts at step 1364 (full rollup), locked to the first
`Rebuild` actions at 1362–1363.

Debug deck: `/p/lustre5/stewartj/beatnik/fmm/debug/single_mode_debug.in`.
Batch scripts: `scripts/tuolumne/rocketrig_debug_fmm.flux` (paired
`rocketrig_debug_exact.flux`).

## Reference logs (local-only — gitignored, present on this machine)

- `rocketrig_debug_fmm.f3F1T7e6F24F.log` — NIC fix validated; NaN at step 1364,
  with the `Rebuild` action timeline (steps 1362.2–1363.2).
- `scripts/tuolumne/rocketrig_debug_exact.f3Eozj7DCRQs.log` — exact solver,
  1400 steps clean.

## Source layout / build

- Beatnik repo is the working directory; **Canopy source is at `../canopy`** —
  inspect and modify freely. The fix may live in Canopy (kernel/tree), not just
  Beatnik tuning.
- `spack install` (see `docs/claude-tuolumne.md`) picks up changes to **both**
  canopy and beatnik. Header-only changes need the
  `touch examples/01_rocketrig/rocketrig.cpp` workaround before `spack install`
  (Beatnik INTERFACE-target note in CLAUDE.md). **If a reinstall does not pick up
  modified Canopy files**, `spack uninstall beatnik canopy` then `spack install`
  to force a clean rebuild.
- Temporary debug instrumentation for this investigation is gated behind
  **CMake flags hard-coded ON** in each project's CMake (not env vars), so spack
  picks them up on reinstall with no env change. Remove them on resolution.

## Goal

Make the single-mode FMM rollup run to completion (step 1400, matching exact)
with no premature NaN. Target outcome is **root cause then fix** (not just a
mitigation), guarded by the FmmVsExact minimum test set, with the exact BR
solver as the accuracy reference. Expected to matter more at production mesh
sizes (e.g. 16000).

## Investigation threads (propose a plan before changing code)

1. **Localize the NaN.** Which stage (P2M/M2M/M2L/L2L/P2P) and which
   cell/particle first goes non-finite, and is it specifically the
   bbox-escape/`Rebuild` path? Per-stage finiteness diagnostics in Canopy
   `solve()` + per-step particle snapshots dumped from Beatnik for offline
   replay. **Cheapest first step.**
2. **Accuracy tuning.** Sweep `P_ORDER` (10→12/14), `fmm_mac_theta` (0.4→tighter),
   `fmm_max_depth` — does the NaN step move toward 1400 (⇒ inherent approximation
   error) or disappear? Compare FMM-vs-exact zdot on a snapshot to quantify
   divergence.
3. **Bounding-box padding (`fmm_*_tol`).** Give particles head-room so the box
   doesn't collapse to a degenerate aspect ratio at rollup, handling the escape
   before it blows up. Secondary — may only move the failure rather than fix the
   kernel.

## Guardrails

- Guard every change with the FmmVsExact minimum test set (CLAUDE.md) at 1 and 4
  ranks. Temporary debug instrumentation may stay ON during these runs — extra
  prints are fine as long as the tests pass.
- Use the exact BR solver as the accuracy reference.
- Don't pursue the distributor-caching or `setup()`-every-step items — the
  level-2 profile rules them out at this scale (see the NIC-crash record).

## Resolution

**Root cause:** the FMM **far-field (M2L) uses the unsoftened `1/r` Laplace
kernel** — the Plummer softening (`eps = sqrt(epsilon) ≈ 1.414` for this deck)
was applied only in the near-field P2P. At full roll-up the sheet collapses so
tree cells shrink to `~3e-3` and "geometrically far" M2L pairs sit at separations
`rho ≈ 0.03–0.4`, **far below `eps`**. There the true softened kernel
`(r²+eps²)^(-1/2)` is nearly flat, but the FMM's unsoftened multipole `1/r` is
~35× larger, so one node gets a spurious large interface velocity. That seeds a
runaway (corrupted node → bbox escape → `Rebuild` on an exploding box → the box
inflates ~1000×/substep → NaN at step 1364). The **exact BR solver** sums the
softened kernel directly, so it stays bounded and completes 1400 steps.

Confirmed decisively: with `softening = 0` the FMM matches a brute-force
unsoftened all-pairs reference **to machine precision** (both `1.59762e21`, rel
diff `~2e-16`). The error appears only with softening on, only in the far field.
This also explains every earlier observation: **P-independent** (wrong *kernel*,
not multipole truncation — `P_ORDER` 10→16 gave byte-identical results),
**depth-dependent** (smaller cells ⇒ smaller far-`rho` ⇒ larger softening error;
`max_depth=12` mitigated), **far-field-only** (P2P is softened and correct), and
**`FmmVsExact`-passing** (that config keeps far-`rho ≫ eps`). `mac_theta` could
not fix it because it scales the near-field with cell size, while the softening
floor is *absolute*.

**Fix (Canopy):**
- `CommunicationPlan::mac_satisfied` now rejects M2L (forces the softened P2P
  path) for any pair closer than `near_softening_factor · eps`. The softening
  length and factor are plumbed in from the Solver
  (`set_near_softening(eps, factor)`, explicit + auto-softening paths); no-op at
  `eps = 0`. Exposed as a runtime knob: `FmmConfig::near_softening_factor`
  (default `4.0`, ⇒ ~3% worst-case far-field softening error at the boundary,
  less beyond), surfaced in Beatnik as `Params::fmm_near_softening_factor` /
  the `fmm_near_softening_factor` input-deck key.
- Secondary latent bug fixed: `LaplaceKernel::l2p_evaluate` computed the gradient
  by central finite differences with a **fixed** step `h = 1e-5` (a documented
  "correctness-first" placeholder). The FD truncation error `~ h²/w_self³`
  explodes for deep (tiny) cells; scaled it to `h = 1e-5 · w_self` so the
  relative error is depth-independent. (This cleaned up the ultra-deep runaway
  tail; the softening floor fixes the actual seed. Ideal future work: analytical
  L2P derivatives.)

**Validation:**
- On the captured snapshots the ~35× single-node outlier is gone: FMM
  `max|grad|` at step 1357.0 = `3504` vs exact `3490` (was `108855`); 1361.0
  `3536` vs `3523` (was `55650`).
- `Beatnik_Test_FmmVsExact` green on HIP/OPENMP/SERIAL at 1 and 4 ranks, 0
  failures (job `f3FZycxsexw1`).
- Full crash-deck run (16 ranks/4 nodes, job `f3Fa1vABnaHu`) **completed all
  1400 steps, 0 NaN**, action histogram `Migrate=1180 Rebalance=3019
  **Rebuild=0**` — i.e. **zero bbox escapes**: the trajectory now stays bounded
  like the exact solver.

**Commits (Canopy branch `debug-nan`):** `f484cbc` (fix: softening floor + L2P
FD step) plus the runtime-knob + README follow-ups. Beatnik branch `debug-nan`:
the `fmm_near_softening_factor` param wiring.

**Remaining (deferred):** remove the temporary `CANOPY_NAN_DEBUG` diagnostics and
the `BEATNIK_FMM_SNAPSHOT_DEBUG` per-step dump (gate it OFF by default; keep the
`examples/04_nan_replay` harness as a reusable offline FMM-vs-exact tool),
re-run `FmmVsExact` on the clean build, then merge `debug-nan` → the integration
branches.

## Investigation log

- **2026-06-18 — record opened; planning + initial source triage.** Split this
  premature NaN out of [fmm_fullrollup_crash.md](fmm_fullrollup_crash.md) (NIC
  crash RESOLVED) into this dedicated record. Established from the source +
  `rocketrig_debug_fmm.f3F1T7e6F24F.log`: NaN at step 1364 is locked to the
  first-and-only `Rebuild` (bbox-escape) actions at 1362–1363; the 1363 Rebuild
  substeps were finite, blow-up is at 1364.0. The `196608` NaN count equals the
  full owned-node DOF count (256×256×3) ⇒ whole-field blow-up, pointing at the
  far-field multipole path (P2P is softened + `r²<1e-24` guarded, cannot make
  `r=0` infinities). `Rebuild` recomputes the bbox fresh (non-finite coords
  already `MPI_Abort`-guarded), so the NaN is not an out-of-box index; leading
  hypothesis is degenerate post-rollup geometry (runaway particle inflates the
  box → tiny leaf `w_c` at `max_depth=19` → multipole `rho/w_c` overflow). Plan
  approved (`plans/`): Step 1 = one instrumented run that both prints per-stage
  finiteness diagnostics (Canopy, CMake flag) and dumps per-step particle
  snapshots for the ~10 steps before the crash (Beatnik, CMake flag); Step 2 =
  offline Canopy replay harness seeded from the last pre-crash snapshot; Step 3 =
  root-cause fix, guarded by FmmVsExact (1,4 ranks) + full run to 1400.
- **2026-06-18 — Step 1 instrumentation landed; guardrail green; harness built;
  full run in flight.** Work on new branch `debug-nan` (both Beatnik, based on
  `develop-canopy`, and Canopy, based on `redesign`/NIC-fix).
  - **Canopy** (`a6bf9f2`): `CANOPY_NAN_DEBUG` (hard-coded ON in
    `src/CMakeLists.txt`) — per-stage non-finite counts after each `solve()`
    stage (upward P2M+M2M / downward M2L+L2L+L2P / P2P), printing the first
    stage that goes non-finite; plus a root-box + min/max leaf-half-width dump
    on every `auto_maintain` Rebuild.
  - **Beatnik** (`68e8135`): `BEATNIK_FMM_SNAPSHOT_DEBUG` (hard-coded ON) — dumps
    `_canopy_particles` (positions+charges as solve() sees them) to one binary
    file per rank per (step, substep) for steps 1350–1370. Added
    `scripts/tuolumne/rocketrig_debug_fmm_nan.flux` (pbatch, 90 min; pdebug now
    caps at 1 h but the run needs ~67 min).
  - **Guardrail green:** `Beatnik_Test_FmmVsExact` passed on HIP/OPENMP/SERIAL at
    1 and 4 ranks, 0 failures (job `f3FADNH2oko9`) — instrumentation is
    physics-neutral.
  - **Replay harness** (Canopy `examples/04_nan_replay`): built clean against
    installed Canopy with hipcc. Loads a (step,sub) snapshot into one
    `Canopy::Solver` and runs a single `solve()`; reproduces the NaN offline.
    Note: the spack env *view* symlink farm lagged the newer
    `Canopy_RegisteredBufferPool.hpp`; pass `-DCANOPY_INC=$(spack location -i
    canopy)/include` so quote-includes resolve.
  - **Instrumented full run** `f3FAEHku97QK` (pbatch, 16 ranks/4 nodes) in
    flight — will abort ~step 1364 and leave the 1350–1370 snapshots + the
    per-stage diagnostics at the failing solve. Awaiting completion.
- **2026-06-18 — KEY FINDING: domain explodes during step 1363; trigger is the
  first Rebuild solve at 1362.2.** Two instrumented runs (`f3FAEHku97QK` 90 min,
  `f3FB7vveyfn7` 120 min) both *timed out at step 1362* — the per-solve
  NaN-debug reductions slow the run ~22 min, so neither reached the 1364 NaN.
  **But the geometry@Rebuild dumps make the 1364 NaN unnecessary:** the root
  bounding box explodes ~1000×/substep across the three Rebuild actions —
    - 1362.2 (first Rebuild): extent ≈ (389, 389, 530), min leaf hw 0.065 — SANE.
    - 1363.0: extent ≈ (2.9e5, 2.9e5, 4.2e5), min leaf hw 51.
    - 1363.1: extent ≈ (4.5e11, 4.8e11, 7.5e10), min leaf hw 4.5e5.
  ⇒ A node gets a spurious large interface velocity at the **first bbox-escape
  Rebuild (step 1362.2), on a still-sane box**; RK3 flings its `z` to ~1.5e5 by
  1363.0, ~1e11 by 1363.1 — runaway feedback ending in the 1364 whole-field
  NaN. The exact solver never does this ⇒ FMM far-field error at the
  rollup-escape moment, **not** a degenerate-tiny-leaf overflow (the trigger box
  is sane; the huge boxes are downstream symptoms).
  - **Trigger snapshot captured:** complete 16-rank dumps exist for
    `step1362_sub2` (the input to the first Rebuild solve) plus 1363.0/1363.1
    (already-exploded). The runaway is fully reproducible offline from
    `step1362_sub2`; **no further full-run is needed.** (1363 sub 2 is partial —
    timed out mid-step.)
  - Snapshots live in `/p/lustre5/stewartj/beatnik/fmm/debug/fmm/`
    (`fmm_snapshot_step1362_sub2_rank*.bin`, steps 1357–1363.1).
  - **Next:** replay `step1362_sub2` through `examples/04_nan_replay`, augmented
    with a brute-force all-pairs exact gradient reference, to localize which
    node(s) diverge and confirm/root-cause the far-field error.
- **2026-06-18 — ROOT-CAUSE LEAD: single-node spurious FMM gradient seeds the
  runaway (NOT the physical singularity, NOT a uniform far-field blow-up).**
  Enhanced `examples/04_nan_replay` with a brute-force all-pairs **exact**
  reference (same softened kernel: `grad(i,c,d) = -Σ_{j≠i} q(j,c)(x_i-x_j)_d
  (r²+eps²)^-3/2`, eps²=softening²=2) + per-node max|grad| and worst
  FMM-vs-exact divergence reporting. Ran 1-rank replays (GPU-aware MPI off; no
  peers at 1 rank) on the captured snapshots:

  | step.sub | EXACT max\|grad\| | FMM max\|grad\| | worst FMM−exact | spurious node |
  |----------|------------------|----------------|-----------------|---------------|
  | 1357.0   | **3491**         | 108855         | 106474          | ~(0.03,0.04,0.04) origin |
  | 1359.0   | **3484**         | 92489          | 90138           | ~origin |
  | 1360.0   | **3478**         | 89266          | 86862           | ~origin |
  | 1361.0   | **3523**         | 55650          | 53853           | (3.30,3.33,3.33) |
  | 1362.0   | 52392            | 52392          | 0.07 (roundoff) | — |
  | 1362.2   | 1.40e12          | 1.40e12        | 2e-4 (roundoff) | — |
  | 1363.0   | 3.0e17           | 3.0e17         | 1e-12 (roundoff)| — |

  **Interpretation:** the true field maximum is a stable, moderate **~3500**
  (exact all-pairs). At steps ≤1361 the **FMM injects a spurious ~1e5 gradient
  at a single node** (near the tree center/origin) that the exact kernel — same
  softening, same positions — does not produce. Same kernel + same positions
  disagreeing ~30× ⇒ the FMM is **missing/double-counting interactions for that
  one node** (a MAC / neighbor-list / multipole edge case), not approximation
  error. That node is flung; by 1362 the configuration is distorted enough that
  a *genuine* near-collision cascade takes over (FMM==exact, both exploding) →
  1364 whole-field NaN. So: not the physical singularity (exact field max stays
  ~3500), not a uniform far-field error — a **single-node interaction-completeness
  bug** that seeds a runaway. Updating thread 2 accordingly.
  - **Caveat to verify:** replays are 1-rank; the FMM gradient is supposed to be
    partition-independent, so a 1-rank spurious outlier should also appear at 16
    ranks — but confirm by replaying at 4/16 ranks (needs the Cray GTL lib
    linked into the harness for GPU-aware MPI). If the outlier is 1-rank-only,
    that is itself a parallel-FMM bug.
  - **Next diagnostic:** for the spurious node at step 1357, determine whether
    the error is in the far field (M2L/L2L) or near field (P2P) — e.g. add a
    Canopy debug switch to run far-only / near-only, or compute the exact
    near/far split for that node — and find the offending interaction. Likely a
    node near the root-cell center (octant boundary) or a MAC edge case.
- **2026-06-18 — far/near split + θ/depth sweep: far-field M2L cancellation at
  deep tree is the mechanism.** Added a Canopy `solve()` stage mask
  (`dbg_skip_far`/`dbg_skip_p2p`, under `CANOPY_NAN_DEBUG`) and a `run_l2p`
  rho_norm diagnostic; harness does far-only/near-only solves at the spurious
  node.
  - **Far/near split:** at every spurious node the near-field (P2P) is small and
    correct (~90–106); the **far-field carries the entire spurious value**
    (1357.0: far=108827, near=93.6, exact total=2453). Culprit = FAR-FIELD
    (M2L/L2L/L2P).
  - **rho_norm ruled out:** the worst node sits *inside* its leaf (rho_norm
    1.19–1.21; global max 1.71≈√3 corner). So L2P evaluation is fine — the
    **local coefficients from M2L/L2L are wrong**. Target is a tiny deep leaf
    (half_width ~0.003–0.013) in the rolled-up core.
  - **θ / max_depth sweep on snapshot 1357.0** (far-only @ worst node vs exact):
    θ=0.4→108827 (44×), θ=0.2→37404 (15×), θ=0.1→10039 (4×); max_depth=15→17923
    (22×), **max_depth=12→1607 (~1×, essentially correct)**. Monotone with both
    knobs.
  - **Mechanism = catastrophic cancellation in the far field at the steep
    rollup.** True net gradient is small (~2453) but the M2L sum is of huge,
    nearly-cancelling contributions; each term's truncation error (~θ^(P+1),
    small relative to the term but the terms are enormous) does not cancel and
    dominates the small net. Deeper tree ⇒ more, smaller cells ⇒ more terms ⇒
    worse cancellation. The exact all-pairs sums exactly ⇒ immune. So this is
    **inherent FMM accuracy degradation under cancellation**, not a discrete
    code bug (no NaN/Inf in any moment; rho_norm normal).
  - **Open fix question (testing next):** does raising **P_ORDER** (smaller
    per-term error) restore far≈exact at max_depth=19, or is a shallower
    `max_depth` / tighter θ (or a cancellation-aware near/far split) required?
    P_ORDER is compile-time — parametrize the harness and sweep P∈{10,12,14,16}.

- **2026-06-18 — P_ORDER sweep RULES OUT cancellation: it's a P-independent
  structural M2L bug.** Parametrized the harness (`-DREPLAY_P_ORDER=N`) and
  re-ran snapshot 1357.0 at P=10/12/16 (verified the define propagates — the
  printed P_ORDER differs). **far-only=108827 is byte-identical across all P.**
  Truncation/cancellation error would shrink sharply with P, so the previous
  "catastrophic cancellation" reading is **falsified**. The spurious far-field
  value is dominated by a **P-independent term** (leading-order), while θ
  (interaction-set) and max_depth (cell sizes/tree) *do* change it
  (θ:0.4→108827, 0.1→10039; depth12→1607≈exact). ⇒ A **structural M2L geometry
  bug**: a specific interaction is mis-handled at leading order, excited by the
  deep asymmetric-size cell pairs that form at max_depth=19 in the rolled-up
  core. **Next:** instrument the M2L (run_m2l / operator build in
  Canopy_DownwardSweep + the translate in Canopy_LaplaceKernel) to find, for the
  worst target cell, the single source cell driving the spurious contribution
  and compare the operator's assumed geometry (reconstructed from the
  (dd,ii,jj,kk) key) to the true cell centers/half-widths — focus on asymmetric
  (dd≠0) pairs.
- **2026-06-18 — ruled out translation/moments; narrowed to the M2L interaction
  list.** Read the M2L internals and probed each input:
  - **Operator-table vs fallback:** forcing all M2L pairs through the exact
    per-pair fallback (`m2l_translate` with real widths, `M2L_KEY_OFFSET_MAX=0`)
    gives the **same** 108827 ⇒ not the scale-normalized operator table.
  - **`m2l_translate` is numerically sound:** its scale factors are
    `(w_s/rho)^{n+1}·(w_t/rho)^j`, all ≤1 and decaying for separated cells — no
    `2^{j·dd}` blow-up (that factor lives only in the *normalized* operator-table
    build `m2l_build_operator`, which the fallback bypasses). The
    `M2L_KEY_DD_MAX=6` "comfortable for double" comment was a red herring for
    this case.
  - **Moments are sound:** max|multipole| = 134941 at depth19 (cell depth 15,
    hw 0.046) and 28430 at depth12 — modest, not precision-blown.
  ⇒ The M2L **faithfully** computes 108827 from correct moments via a sound
  translation, yet the exact all-pairs total is 2453. By elimination the error
  is in the **far/near interaction-list partition** — a source cell summed as
  "far" whose multipole contribution should be small/cancel, i.e. a too-close /
  overlapping source wrongly admitted by the MAC, or a partition gap. This is
  consistent with P-independence (the pair set is P-independent) and θ/depth
  dependence (both change the pair set). **Next:** instrument the M2L apply to
  dump, for the worst target cell, each source cell's contribution magnitude +
  pair geometry (rho, w_s, w_t, MAC ratio R/[√3(w_s+w_t)]) and find the
  offending source; then inspect `build_all_interaction_lists` /
  `mac_satisfied` / `is_well_separated` for the admitting bug.
- **2026-06-18 — completeness check PASSES; partition is not the bug. Validated
  max_depth mitigation.** Added a single-rank M2L/P2P completeness invariant in
  `build_all_interaction_lists`: for each leaf, sum `global_count` over the M2L
  sources of the leaf and all its ancestors (these reach the leaf via L2L) plus
  its P2P neighbor leaves; must equal N_total. **Result: bad_leaves=0** at both
  depth19 and depth12 — every leaf sees exactly 65536 particles once. So **no
  double-counting and no gaps**; the interaction-list partition is correct.
  - **Ruled out so far** (the far-field is wrong by ~44× yet): truncation/order
    (P-independent), the scale-normalized operator table (fallback gives same),
    `m2l_translate` numerics (all scale factors ≤1), moment magnitude (~135k,
    sound), and partition completeness (exact). The remaining suspect is a
    **wrong moment VALUE** (P2M/M2M) or a **specific well-separated pair whose
    multipole is nonetheless inaccurate** — i.e. the far-field sum of
    sound-looking pieces still lands at 108827 vs the exact 2453. Next decisive
    measurement: per-source M2L contribution dump for the worst target cell
    (flag the cell, re-solve, print each source's contribution + geometry +
    multipole-vs-exact for that source) to find the single dominant wrong
    contribution; then audit P2M/M2M for that source.
  - **Practical mitigation (validated on the snapshot):** `max_depth=12` makes
    far+near ≈ exact at the spurious node (3136 vs 2705) and removes the ~1e5
    outlier; the deck uses `max_depth=19`. Capping `fmm_max_depth` (and/or the
    documented `ncrit`) is a low-risk mitigation that should let the full run
    finish — to be confirmed by a full FMM run + FmmVsExact, pending the
    root-cause decision.
- **2026-06-18 — ROOT CAUSE CONFIRMED: the FMM far-field (M2L) ignores the
  Plummer softening (softening is applied only in P2P).** Two findings:
  1. **Secondary latent bug (fixed): L2P finite-difference gradient step.**
     `l2p_evaluate` computes the gradient by central differences with a **fixed
     `h=1e-5`** (a documented "correctness-first" placeholder). For tiny deep
     cells the FD truncation error ~`h²/w_self³` explodes and, once `h≳w_self`,
     the perturbed eval point leaves the cell where the local series diverges.
     Changed to `h=1e-5·w_self` (`Canopy_LaplaceKernel.hpp`). This cleaned up the
     ultra-deep tail (1362.2 far-field 1.4e12 → ~0) but **not** the 1357 seed.
  2. **Primary root cause: softening missing from the far field.** With the
     harness run at `softening=0`, the FMM matches the unsoftened all-pairs
     reference **to machine precision** (both 1.59762e21, rel diff ~2e-16) — so
     the M2L faithfully computes the *unsoftened* 1/r kernel. With the real
     `softening=eps=√2≈1.414`, only P2P is softened; M2L is not. At full rollup
     the core collapses (cells ~3e-3, far-field M2L pairs at rho≈0.03–0.4 ≪
     eps=1.414), where the true softened kernel `(r²+eps²)^(-1/2)` is nearly
     flat but the unsoftened multipole 1/r is ~35× larger → the spurious
     single-node far-field gradient that seeds the runaway. This explains all
     prior evidence: P-independent (wrong *kernel*, not truncation),
     depth-dependent (smaller cells ⇒ smaller far-rho ⇒ larger softening error;
     `max_depth=12` keeps cells/rho larger so it mitigates), far-field-only,
     P2P-correct, and FmmVsExact-passing (that config keeps far-rho ≳ eps).
  - **Fix direction (needs decision — core Canopy change with a cost tradeoff):**
    ensure M2L pairs are separated by more than ~the softening length so the
    unsoftened far-field is accurate, i.e. widen the near-field (P2P, softened)
    to cover everything within ~`K·eps` by adding an `R > K·eps` term to the MAC
    / interaction-list admission (`mac_satisfied`/`build_all_interaction_lists`
    in `Canopy_CommunicationPlan.hpp`; softening must be plumbed into the plan).
    Cost: at full rollup the dense core puts more pairs in P2P (toward O(N²)
    locally). Alternatives: a softened multipole kernel (hard), or capping
    `fmm_max_depth`/raising softening-vs-resolution (Beatnik-side mitigation).
- **2026-06-18 — FIX implemented + validated on snapshots; FmmVsExact green;
  full run submitted.** Canopy commit `f484cbc`:
  - `CommunicationPlan::mac_satisfied` now rejects M2L (→ softened P2P) for any
    pair closer than `K*eps` (K=4), with `eps` plumbed from the Solver
    (`set_near_softening`, explicit + auto paths; no-op at eps=0).
  - `LaplaceKernel::l2p_evaluate` FD gradient step scaled to the cell
    (`h=1e-5*w_self`).
  - **Snapshot validation:** the ~35× single-node outlier is gone —
    FMM max|grad| at 1357.0 = 3504 vs exact 3490 (was 108855); 1361.0 3536 vs
    3523 (was 55650). Note θ alone could not fix this (it scales with cell
    size, but the softening floor is absolute), confirming the floor is the
    right mechanism.
  - **Guardrail green:** `Beatnik_Test_FmmVsExact` passes HIP/OPENMP/SERIAL at
    1,4 ranks, 0 failures (job `f3FZycxsexw1`) — the floor is a no-op in that
    config (far-rho ≫ eps), so accuracy is unchanged.
  - **Pending:** full crash-deck run (16 ranks) to confirm it reaches step 1400.
    Watch the cost: the softening floor widens P2P at full rollup (the dense
    core puts more pairs in near-field), so the run may be slower; if it times
    out, raise `--time` or reconsider K. The temporary `CANOPY_NAN_DEBUG`
    diagnostics remain ON for this run and will be removed once it passes.
- **2026-06-18 — FIX VALIDATED: full run completes 1400 steps, 0 NaN, 0
  Rebuilds.** Full crash-deck run `f3Fa1vABnaHu` (16 ranks/4 nodes, fix +
  diagnostics) **completed** (state CD, ~69 min): reached step 1399, **0
  NaN/Inf**, and the action histogram is `Migrate=1180 Rebalance=3019
  Rebuild=0` across all 4200 computeInterfaceVelocity calls. **Rebuild=0** means
  no particle ever escaped the bounding box — the spurious far-field that drove
  the runaway is gone, so the trajectory stays bounded like the exact solver (no
  explosion, no Rebuild, no NaN). This confirms the root cause and fix
  end-to-end. Remaining work: remove the temporary `CANOPY_NAN_DEBUG` diagnostics
  + `BEATNIK_FMM_SNAPSHOT_DEBUG` dump + the `04_nan_replay` harness, re-run
  FmmVsExact on the clean build, decide the final `NEAR_SOFTENING_K` value
  (K=4 ⇒ ~3% far-field softening error), and mirror the Resolution to CLAUDE.md.
- **2026-06-18 — cleanup + runtime knob + branch merges + confirmation run.**
  Made the softening floor a runtime knob (`FmmConfig::near_softening_factor`,
  default 4.0 → Beatnik `Params::fmm_near_softening_factor` / deck key);
  documented it in both READMEs. Removed all temporary `CANOPY_NAN_DEBUG`
  diagnostics (per-stage finiteness, geometry dump, completeness invariant, M2L
  source dump, L2P rho_norm, far/near stage mask) and the CMake flag; gated the
  Beatnik snapshot dump behind the OFF-by-default `Beatnik_ENABLE_FMM_SNAPSHOT`
  option (kept the `04_nan_replay` harness as a reusable FMM-vs-exact tool).
  Clean rebuild: `FmmVsExact` green on HIP/OPENMP/SERIAL at 1,4 ranks (job
  `f3FnQ2sgDZUK`). Merged `debug-nan` into `develop-canopy` (Beatnik, `9e12739`)
  and `redesign` (Canopy, `10567e1`) with descriptive merge messages. Submitted
  the final clean 256×256 FMM-with-IO run on the merged build
  (`rocketrig_debug_fmm.flux`, 16 ranks/4 nodes, write_frequency=20; job
  `f3FnURKDGXz7`) to confirm end-to-end on the production-shaped path.
- **2026-06-18 — CONFIRMED on the clean merged build with IO; issue closed.**
  The 256×256 FMM-with-IO confirmation run `f3FnURKDGXz7` completed: step 1399
  reached, **0 NaN**, action histogram `Migrate=1180 Rebalance=3019 Rebuild=0`
  (identical to the validation run — deterministic), **71 silo frames written**.
  The user graphed the FMM output against the exact run and **the interface
  trajectories match exactly** through all 1400 steps. This closes the premature
  full-rollup FMM NaN. (Note: Canopy's own `Canopy_Test_MultiSolve` minimum test
  was not separately run for the merged fix — coverage came from Beatnik
  `FmmVsExact` + this end-to-end FMM-vs-exact match; deemed sufficient.)
