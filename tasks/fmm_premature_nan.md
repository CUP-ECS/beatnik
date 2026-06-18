# Premature FMM NaN at full rollup — investigation & fix

**Status: OPEN** (last updated 2026-06-18). With the Slingshot NIC-registration
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
- _(append next entry here)_
