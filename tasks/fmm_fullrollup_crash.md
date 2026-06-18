# FMM full-rollup crash — investigation & fix

**Status: NIC CRASH RESOLVED** (last updated 2026-06-17). The Slingshot
`cxil_map` registration crash is fixed (see Resolution). Full-rollup
*completion* is now blocked by a **separate, pre-existing premature FMM NaN
at step 1364** that the NIC crash had been masking — tracked as a new open
item below ("Premature FMM NaN at full rollup").

Single-mode FMM rollups abort at full rollup with a Slingshot CXI NIC
memory-registration failure (`cxil_map: write error` → `MPI_Isend` abort).
Not a NaN — the physics is stable through rollup. This file is the living
record of how the bug is identified and addressed.

## How to use this file

This is a **living document** — update it as the investigation continues so we
keep a record of how the bug was diagnosed and fixed:

- After each experiment or work session, append a dated entry to the
  **Investigation log** at the bottom (what was tried, the result, and the
  conclusion drawn). Newest entries at the end.
- Keep the **Status** line current (OPEN / IN PROGRESS / RESOLVED).
- When a hypothesis is confirmed or ruled out, note it in **Background** so
  future sessions don't re-derive it.
- When resolved: add a **Resolution** section summarizing the root cause and
  the fix, and link the commits.
- Mirror anything load-bearing back to the "Known issues" entry in
  [../CLAUDE.md](../CLAUDE.md) so the top-level pointer stays accurate.

## Background (established facts — don't re-derive)

- A single-mode `sech2` rollup with the FMM solver is **numerically stable
  through full rollup**: the ZModel NaN/Inf interface-velocity guard never
  fires, and the **exact BR solver completes the same case**. So the physics
  is sound; the failure is FMM-/infrastructure-specific.
- The earlier *premature* FMM NaN at rollup onset was a **separate, already
  fixed** accuracy problem — resolved by raising `P_ORDER` 6→10 and tightening
  `fmm_mac_theta` 0.6→0.4 (committed on `develop-canopy`).
- The remaining failure: at full rollup, particles concentrate in the rolled-up
  core, Canopy's `auto_maintain` switches to near-continuous **Rebalance**, and
  the all-to-all migration traffic exhausts the Slingshot CXI NIC's memory
  registration → `cxil_map: write error` → `MPI_Isend ... MPIDI_OFI_send_normal:
  Invalid argument` abort.

## Repro

Config: single-mode `sech2`, 256×256 (B=4), `P_ORDER=10`, `fmm_max_depth=19`,
`fmm_mac_theta=0.4`, `fmm_imbalance_tol=0.20`, `epsilon=2`, `delta_t=0.0006`,
16 ranks / 4 nodes. Crash at ~step 1272 (full rollup); by ~step 600
`auto_maintain` is already a Migrate/Rebalance mix, trending to all-Rebalance.

Debug deck: `/p/lustre5/stewartj/beatnik/fmm/debug/single_mode_debug.in`.
Batch scripts: `scripts/tuolumne/rocketrig_debug_fmm.flux` (and the paired
`rocketrig_debug_exact.flux`, `rocketrig_testprof.flux`).

## Profiling findings (level 2, 16 ranks)

Per-call `computeInterfaceVelocity` mean 1127 s splits as:

| Phase                       | Mean (s) | % of call |
| --------------------------- | -------- | --------- |
| **Canopy `solve()`**        | **1044** | **92.6%** |
| `auto_maintain`             | 56       | ~5%       |
| build fwd+rev distributor   | 22       | ~2%       |
| fwd+rev migrate             | 4.5      | <1%       |

⇒ The bottleneck is Canopy `solve()`. The "Cache the forward
`Cabana::Distributor`" and "`setup()` every step" items under *Future
optimization opportunities* in CLAUDE.md are **not worth pursuing at this
scale**. Revisit only if higher rank counts shift the balance.

## Reference logs (local-only — gitignored, present on this machine)

- `scripts/tuolumne/rocketrig_debug_fmm.f3ExhEXgrG4X.log` — the crash
- `scripts/tuolumne/rocketrig_testprof.f3Eyjvuf1wFV.log` — level-2 profile

## Source layout / build

- Beatnik repo root is the working directory; **Canopy source is at `../canopy`**.
  You may **inspect and modify** the Canopy source as needed — the fix may live
  in Canopy's `solve()`/rebalance, not just in Beatnik tuning.
- The standard `spack install` (see `docs/claude-tuolumne.md`) picks up changes
  to **both** canopy and beatnik. Header-only changes need the
  `touch examples/01_rocketrig/rocketrig.cpp` workaround before `spack install`
  (see the Beatnik INTERFACE-target note in CLAUDE.md).

## Goal

Make a single-mode FMM rollup run to completion without the NIC crash, and
understand/reduce the Canopy `solve()` cost so it is viable at production mesh
sizes (target 16000×16000).

## Investigation threads (propose a plan before changing code)

1. **The NIC crash.** Is it driven by Rebalance message *volume* (reduce via
   `fmm_imbalance_tol`, `fmm_max_depth`, `replication_depth`) or by
   registration-cache *accumulation* (a libfabric/CXI env workaround like
   `FI_MR_CACHE_*` / `FI_CXI_*`)? First confirm determinism (does it die at
   ~step 1272 every run?), then single-variable experiments. Get data at higher
   rank counts — the 92.6%/3% split is from 16 ranks only and may shift.
2. **Canopy `solve()` cost and its growth during rollup.** Profile `solve()`
   internals in `../canopy`; weigh the `P_ORDER` / `mac_theta` / `max_depth`
   accuracy-vs-cost tradeoff (find the cheapest config that still clears the
   rollup-onset accuracy bar — exact is the reference); determine whether the
   cost growth is inherent to the steepening geometry or a Canopy-side
   inefficiency fixable in `../canopy`.

## Guardrails

- Guard every change with the FmmVsExact minimum test set (CLAUDE.md).
- Use the exact BR solver as the accuracy reference.
- Don't pursue the distributor-caching or `setup()`-every-step items — the
  profile rules them out at this scale.

## Resolution (NIC crash)

**Root cause:** GPU-aware-MPI memory **registration churn**. Both halo
exchanges inside Canopy `solve()` allocated a fresh device `Kokkos::View`
per peer on every call and passed its `.data()` device pointer to
`MPI_Isend`/`MPI_Irecv` — `detail::coalesced_view_exchange` (M2M in
UpwardSweep; M2L + L2L in DownwardSweep) and `P2P::gather_ghost_particles`.
On Slingshot/CXI each fresh device allocation is a fresh NIC registration;
at full rollup (near-continuous Rebalance + growing M2L working set) the
registration cache accumulates until the NIC is exhausted →
`cxil_map: write error` → `MPIDI_OFI_send_normal: Invalid argument`.

**Fix:** `Canopy_RegisteredBufferPool.hpp` — a persistent, grow-only (1.5×
headroom, never shrinks) device buffer with a stable base address. Each
exchange now carves per-peer **unmanaged subviews** from one registered
region per direction, so the CXI cache registers a small bounded set once
and reuses it. Pure buffer management: pack/unpack, peer ordering, message
layout, and the accumulate-on-recv (L2L) path are unchanged.

**Commits (Canopy branch `redesign`):** `fac4519` (pool + P2P),
`d834034` (coalesced M2M/M2L/L2L). Beatnik unchanged.

**Validation:** `Beatnik_Test_FmmVsExact` green on HIP/OPENMP/SERIAL at 1
and 4 ranks (agreement vs exact unchanged: BRDirect rel_diff ~1e-20,
OneRK3 ~7e-10; job `f3F1Peev48XH`). Full crash-deck run `f3F1T7e6F24F`
(pbatch, 16 ranks/4 nodes) ran with **0 `cxil_map` errors** through the
entire Rebalance-heavy tail (1213 Migrate + 2874 Rebalance actions),
proving the registration crash is gone. The run did not reach step 1400 —
it hit the separate premature NaN below.

## Premature FMM NaN at full rollup (OPEN — new, was masked by the NIC crash)

With the NIC crash fixed, run `f3F1T7e6F24F` advanced to step 1363 and then
aborted via the ZModel guard: `196608 NaN/Inf value(s) detected in
interface velocity (zdot) during timestep 1364`. This is **premature**: the
**exact BR solver completes all 1400 steps** on the identical deck
(`rocketrig_debug_exact.f3Eozj7DCRQs.log`: `solve() total wallclock =
108.8 s (1400 steps)`, no NaN). So it is an FMM accuracy/robustness failure,
not the physical singularity.

**Diagnostic lead:** the NaN onset coincides exactly with the **first and
only** `auto_maintain` → `Rebuild` actions of the whole run (steps 1362.2,
1363.0–1363.2; 4 total; everything before is Migrate/Rebalance). `Rebuild`
fires on `needs_rebuild` = a particle escaping the global bounding box. So
the blow-up is triggered the moment particle(s) leave the bbox at maximal
rollup, and the FMM produces a spurious large velocity the exact all-pairs
solver does not. This is investigation **thread 2** (accuracy/cost), now the
gating issue for full-rollup completion. Candidate directions: (a) far-field
accuracy (P_ORDER 10→higher, mac_theta 0.4→tighter, max_depth) to see if the
NaN step moves toward 1400 — if it does, it is inherent approximation error;
(b) localize which particles/region produce the NaN at step 1363 (compare
FMM vs exact zdot) to test whether it is the bbox-escape/Rebuild path
specifically rather than gradual error; (c) bbox padding tolerances
(`fmm_*_tol`) so the escape/Rebuild is handled before it blows up.

## Investigation log

- **2026-06-17 — initial diagnosis.** Single-mode FMM rollups were dying; built
  the scaled debug decks (B = nodes/64, magnitude ∝ B, period ∝ 1/B²) and an
  inverse-square `period` fix for the `sech2` IC. Established `delta_t` is
  irrelevant to the blowup (halving it kept the blowup at the same physical
  time). Found two distinct failures: (a) a *premature* FMM NaN at rollup onset
  — fixed via `P_ORDER` 6→10 + `mac_theta` 0.6→0.4; (b) this NIC-registration
  crash at full rollup. Added a profiling-gated NaN/Inf guard on the interface
  velocity in ZModel (covers all BR backends). Level-2 profiling pinned the
  cost to Canopy `solve()` (92.6%), ruling out distributor optimizations.
  Exact BR solver completes the same case, confirming the physics is fine.
  Crash signature: `cxil_map: write error` + `MPI_Isend` abort at ~step 1272,
  during continuous Canopy Rebalance.
- **2026-06-17 — root cause found + structural fix landed in Canopy (validation
  run in flight).** Traced the `cxil_map: write error` to GPU-aware-MPI memory
  *registration churn*, not message volume. Both halo exchanges inside Canopy
  `solve()` allocated a brand-new device `Kokkos::View` per peer **every call**
  and passed its `.data()` device pointer to `MPI_Isend`/`MPI_Irecv`:
  `detail::coalesced_view_exchange` (M2M in UpwardSweep; M2L + L2L in
  DownwardSweep) and `P2P::gather_ghost_particles`. On Slingshot/CXI each fresh
  device allocation is a fresh NIC registration; at full rollup (near-continuous
  Rebalance + growing M2L working set) the registration cache accumulates until
  the NIC is exhausted → the failing `MPI_Isend` (count=210672 B is a solve()
  buffer, consistent with this). Fix: added
  `Canopy_RegisteredBufferPool.hpp` — a persistent, grow-only (1.5× headroom,
  never shrinks) device buffer with a stable base address. Each exchange now
  carves per-peer **unmanaged subviews** from one registered region per
  direction, so the CXI cache registers a small bounded set once and reuses it.
  Pack/unpack, peer ordering, message layout, and the accumulate-on-recv (L2L)
  path are byte-for-byte unchanged — pure buffer management. Committed on Canopy
  branch `redesign`: `fac4519` (pool + P2P), `d834034` (coalesced M2M/M2L/L2L).
  **Guardrail green:** `Beatnik_Test_FmmVsExact` passes on HIP, OPENMP, SERIAL
  at 1 and 4 ranks (agreement vs exact unchanged: BRDirect rel_diff ~1e-20,
  OneRK3 ~7e-10; job `f3F1Peev48XH`). The decisive full crash-deck run
  (single_mode_debug.in, 1400 steps, 16 ranks/4 nodes; job `f3F1T7e6F24F`,
  pbatch) is queued — success = it passes step ~1272 and reaches 1400 with no
  `cxil_map` error. Will flip Status → RESOLVED and add a Resolution section
  once that run confirms.
- **2026-06-17 — NIC crash fix validated; premature NaN exposed.** Full
  crash-deck run `f3F1T7e6F24F` (log
  `rocketrig_debug_fmm.f3F1T7e6F24F.log`) confirmed **0 `cxil_map` errors** —
  the persistent-buffer-pool fix eliminated the Slingshot registration crash.
  The run cleared the entire Rebalance-heavy tail (1213 Migrate + 2874
  Rebalance) that previously aborted at ~1272, reaching step 1363. It then hit
  a **premature FMM NaN at step 1364** (ZModel guard: 196608 NaN zdot values).
  Exact completes 1400 on the same deck → premature, not the singularity. NaN
  onset is locked to the **first-ever `Rebuild` actions** (bbox escape) at
  1362–1363. Marked the NIC crash RESOLVED (Resolution section added) and
  opened "Premature FMM NaN at full rollup" as the new gating item (thread 2).
  Pending user decision on whether to pursue the NaN now or hand it off.
- _(append next entry here)_
