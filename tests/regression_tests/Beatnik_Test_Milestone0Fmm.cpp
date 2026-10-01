/****************************************************************************
 * Copyright (c) 2025 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file Beatnik_Test_Milestone0Fmm.cpp
 * @brief **THE FMM MILESTONE MEMBER** (T6) — two claims about Canopy's
 *        far-field Birkhoff-Rott path, in one binary and one launch: a
 *        per-evaluation velocity bound on 81 direct-driven states, and the
 *        stability and divergence horizon of a 2000-step FMM-driven
 *        trajectory.
 *
 * THIS IS A `milestone`-TIER MEMBER AND IT IS **NOT** THE SHIP GATE.
 * ------------------------------------------------------------------
 * The gate stays at five `regression` members and 60 launches (CLAUDE.md
 * "Minimum test set"). This tier is ranks **1 and 4** on SERIAL and HIP, run on
 * demand through `scripts/<system>/run_milestone.<scheduler>`. T6 takes the
 * tier from two members and eight launches to **four and sixteen**, and moves
 * the runner from `-q pdebug` to `-q pbatch`: one 2000-step FMM-driven level-4
 * trajectory alone is 2377 s at HIP np1, where the whole existing level-4
 * member is 22 s.
 *
 * TWO MEMBERS, ONE BODY, exactly as the frozen pair. This file is the whole
 * test at `--icosphere-subdivisions BEATNIK_M0_FMM_LEVEL`, which defaults to
 * **3**. The second member is `Beatnik_Test_Milestone0FmmL4.cpp`: three lines
 * that `#define BEATNIK_M0_FMM_LEVEL 4` and `#include` this file. Every
 * per-level literal below is selected by that macro and is **that level's** —
 * the entity counts, the two carried scalars, the polyhedral deficit, the final
 * `time`, the 81-entry reference volume-drift series, the realized P2P pair
 * fraction and the divergence-horizon envelope all differ between the levels
 * and **none is transferred**.
 *
 * WHY TWO CLAIMS IN ONE BINARY
 * ----------------------------
 * They need different trajectories and only one of them can be a gold
 * comparison (`tasks/canopy/add-canopy.md` §"The two milestone members").
 *
 * **CLAIM A — the per-evaluation bound.** The trajectory is driven by
 * `BRSolverDirect`, so the run is the frozen member's and all 81 gold
 * comparisons still run at `--rtol 1e-10 --atol 1e-12`; that is what proves it
 * is the right trajectory. At each of those 81 states the FMM velocity is
 * additionally evaluated **on the same state** and compared against the direct
 * velocity, asserting a max relative error `<= kTauA`. Same input, same state,
 * no chaotic amplification in the way — the only comparison that isolates the
 * far field and the only one that can carry a number as tight as `1e-3`.
 *
 * **CLAIM B — the trajectory stays physical, and decorrelates no sooner than
 * measured.** The same binary then runs 2000 steps FMM-driven. It **cannot**
 * assert a gold rung: at `tau_A ~ 1e-3` per evaluation the trajectories
 * separate well before step 2000 and a passing rung would only mean the rung
 * was loose (T5 measured the separation at `1.5e-7` by step 25 and `5.1e-3` by
 * step 2000). It asserts instead the four properties that remain meaningful
 * after decorrelation, and that are exactly the ones develop-canopy's
 * full-roll-up blow-up violated: the run reaches step 2000 and every velocity
 * is finite; the entity counts never change; the volume drift tracks the
 * reference's own `kRefVolumeDrift` series within `kFmmVolumeDriftRtol`; and
 * the **divergence horizon** is no earlier than T5's measured envelope.
 *
 * Risk **R7** is why both are here rather than claim A alone: claim A measures
 * the far field on states the *direct* solver produced, and develop-canopy's
 * failure was not there — the FMM tracked the exact solver acceptably for 1362
 * steps and then a single corrupted node seeded a runaway to whole-field NaN.
 * Claim B's finiteness, entity-count and volume-drift assertions are the parts
 * that fire in that scenario, and **no gold comparison at any rung would**.
 *
 * NOTHING HERE IS ASSERTED BITWISE, AND THAT IS MEASURED RATHER THAN ASSUMED.
 * Two runs of an identical binary on an identical command line differ by
 * **1.15e-15 (np1)** and **1.88e-15 (np4)** of field RMS, localized to the
 * timestep rather than to I/O (T3). Every tolerance below clears that floor by
 * at least eleven decades.
 *
 * THE LADDER IS NOT `compare_output.py`, AND THAT IS A MEASURED FINDING
 * --------------------------------------------------------------------
 * Claim A's 81 gold comparisons go through `compare_output.py` **unchanged**: a
 * direct-driven run satisfies its quantized-lexsort pairing precondition by
 * nine decades. Claim B's do not and cannot. At `tau_A` per evaluation the
 * FMM-driven trajectory separates from the gold set by 100x the default
 * `--match-eps` cell by step 25; the two files then quantize into different
 * cells, vertices pair with roughly antipodal partners, and the comparator
 * reports `max|e| ~ 0.5` on a radius-0.25 sphere where the true disagreement is
 * `1.5e-7`. **It fails silently** — `n_ambiguous` counts within-file collisions
 * and stays 0, and the exit status is an ordinary "compared and disagreed".
 * Raising `--match-eps` is not a fix; by step 1000 no value works.
 *
 * So claim B's horizon is measured by
 * `tests/regression_tests/fmm_divergence_ladder.py`, which pairs by **bijective
 * nearest neighbour** and refuses any step whose pairing is not a bijection.
 * **Its exit status reports whether the MEASUREMENT completed, never whether
 * Beatnik agreed with anything** — it exits non-zero when some step was
 * unpairable — so this file reads the verdict out of its `--json` and treats
 * the exit status as plumbing only. A member that read the exit status alone
 * would be reading the wrong thing.
 *
 * WHAT THE HORIZON ASSERTION CAN AND CANNOT CATCH
 * -----------------------------------------------
 * Each rung's horizon is asserted **no earlier than** T5's envelope, and
 * nothing more. At the `1e-12`, `1e-10`, `1e-8` and `1e-6` rungs the envelope is
 * step **25**, which is the first checkpoint, so those four cannot fail except
 * at step 0: the load-bearing rung is `1e-4` (50 at level 4, 75 at level 3), and
 * that is the rung the negative case below is built on. **No rung is asserted
 * to PASS at step 2000** — at `tau_A` none will, and a rung that did would be
 * loose enough to be meaningless.
 *
 * `--checkpoint-every-steps` stays at **25**. The envelope was measured at 25,
 * and a horizon measured at a different interval is not comparable to it, so
 * narrowing it to sharpen the tight rungs would make the assertion compare two
 * different quantities. There is **no step-count or checkpoint-interval
 * override in this file** (risk **R9**): a knob that can silently shorten a
 * 2000-step run is how a truncated run reads as a shorter pass.
 *
 * Steps **1350-1900 at level 4** are unpairable by any position-based scheme,
 * identically across four independent runs, and the tool refuses them rather
 * than mis-pairing. A refused step is neither a pass nor a failure, the
 * assertion does not depend on one, and every refused step is named in the log.
 *
 * THE FAR FIELD HAS TO BE LIVE TO BE MEASURED, AND IT IS LIVE AT ONE LEVEL
 * ------------------------------------------------------------------------
 * Both levels run at **`ncrit = 8`** — the configuration every number this
 * member compiles was measured at, `tau_A`, the horizon envelope and the
 * volume-drift bound alike. `FmmParams::ncrit` keeps the reference's 64, which
 * is right at production vertex counts and wrong at milestone-0's. T5 measured
 * the realized share at theta = 0.3:
 *
 *   * **2562 vertices at `ncrit` 8 puts 74.6% of pairs through M2L**
 *     (`p2p_pair_fraction` 0.253650). That is a genuine far-field bound and it
 *     is where **the far-field accuracy claim rests**.
 *   * **642 vertices at `ncrit` 8 puts 14.5% through M2L**
 *     (`p2p_pair_fraction` 0.854631), so **the level-3 member's claim A is
 *     mostly a P2P comparison** and says so on its assertion. It is still worth
 *     asserting: it is the round trip, the tag handshake and the contraction
 *     under test, all of which are rank-count-dependent and none of which the
 *     level-4 member covers at level 3's decomposition.
 *   * At level 3 with `ncrit >= 32` the M2L cell-pair count is **exactly 0** and
 *     the solve agrees with `BRSolverDirect` to `2e-15` — risk **R1**'s cheapest
 *     misreading with a number on it, and why the realized fraction is
 *     **asserted** at every one of the 81 states rather than assumed once.
 *
 * `ncrit = 4` is **not** used. It would make level 3 live at 58.5% M2L, but it
 * would put claim A at a configuration claim B's envelope was never measured
 * at.
 *
 * READ THE LIVENESS ORDERING THE RIGHT WAY ROUND: **the error rises as the far
 * field takes over**, because a larger M2L share means more of the field is
 * approximated. A low P2P fraction is not better code and a low error is not
 * better accuracy — it may be less measurement, and the two are
 * indistinguishable without the fraction beside them.
 *
 * FAILURE BEHAVIOR IS LOUD. A gold file missing for a compared step is a named
 * failure, not a skipped step. A run that stops early is a **reported stop
 * step, never a shorter pass**. A `compare_output.py` exit of 2 ("could not
 * load") is never conflated with 1 ("compared and disagreed"), and the ladder's
 * third outcome — "could not pair" — is conflated with neither.
 *
 * THE THREE NEGATIVE CASES, all of which must fire:
 *   1. claim A's final state against the **step-0 gold**, which must exit
 *      exactly **1** and not 2 — accepting 2 is how a negative case passes
 *      vacuously, and it also proves 2000 steps moved the surface;
 *   2. a state **perturbed by more than `kTauA`** must fail claim A's
 *      same-state comparison, naming `kTauA`;
 *   3. a **fabricated horizon** one checkpoint earlier than the envelope must
 *      fail claim B's horizon assertion, on the `1e-4` rung — at the four
 *      tighter rungs the envelope is the first checkpoint and nothing but step 0
 *      is earlier, so a negative case there would prove only that the
 *      comparison runs.
 *
 * ARGUMENTS. Both paths; see tests/CMakeLists.txt for the two call sites, which
 * pass them absolute (ctest) and manifest-relative (the installed runner).
 *
 *   argv[1]  the gold DIRECTORY for THIS LEVEL (81 .npz, steps 0-2000 by 25).
 *            **The existing frozen-member gold set, unchanged** — the
 *            trajectory claim A drives is the same one those members compare
 *            against, so no new gold set exists or is needed.
 *              regression_tests/milestone0-sub3-2000-steps/gold   (level 3)
 *              regression_tests/milestone0-sub4-2000-steps/gold   (level 4)
 *   argv[2]  the comparator, for claim A's 81 gold comparisons
 *              regression_tests/compare_output.py
 *   argv[3]  the divergence ladder, for claim B's horizon
 *              regression_tests/fmm_divergence_ladder.py
 *
 * `BEATNIK_PYTHON` overrides the interpreter (default `python3`).
 * `BEATNIK_TEST_SCRATCH` must name a path on a **parallel** filesystem: two
 * 2000-step runs of checkpoints go through MPI-IO and a node-local scratch
 * fails every launch that spans more than one node. There is no option surface
 * here and none may be added.
 */

#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_BRSolverFMM.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------------------//
// The subdivision level, and the ONLY thing that differs between this file and
// Beatnik_Test_Milestone0FmmL4.cpp. Defaulting it here keeps the file
// compilable on its own and makes the cheaper member the one you get by
// accident, not the one you have to remember to ask for.
//---------------------------------------------------------------------------//
#ifndef BEATNIK_M0_FMM_LEVEL
#define BEATNIK_M0_FMM_LEVEL 3
#endif

#if BEATNIK_M0_FMM_LEVEL != 3 && BEATNIK_M0_FMM_LEVEL != 4
#error "BEATNIK_M0_FMM_LEVEL must be 3 or 4: no other gold set exists (M0-G1, M0-G2)"
#endif

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The M0-G1 / M0-G2 configuration, verbatim from each gold/README.md and field
// for field `Beatnik_Test_Milestone0Frozen.cpp::makeParams` — claim A's
// trajectory must BE that member's, or its 81 gold comparisons prove nothing
// about which trajectory was driven. Claim B's differs in exactly one field,
// `zmodel.br_approximation`, plus the `fmm.ncrit` this level runs at.
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = BEATNIK_M0_FMM_LEVEL;
constexpr int kSteps = 2000;
constexpr int kCheckpointEvery = 25;
/// Steps 0, 25, …, 2000 — the entire gold set. **There is no override for
/// either of these two constants, deliberately** (risk R9): a knob that can
/// shorten a 2000-step run is how a truncated run reads as a shorter pass, and
/// a checkpoint interval other than 25 makes the measured horizon
/// incomparable to the envelope it is asserted against.
constexpr int kComparedSteps = kSteps / kCheckpointEvery + 1;

constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

//---------------------------------------------------------------------------//
// CLAIM A's tolerances.
//---------------------------------------------------------------------------//

/// The gold rung claim A's trajectory is compared at, unchanged from the frozen
/// member: it is the same trajectory and the same comparison, and loosening it
/// here would break the one thing that proves claim A is evaluating on the
/// right states.
constexpr const char* kRtol = "1e-10";
constexpr const char* kAtol = "1e-12";

/// The same relative tolerance, for this test's own `time` check.
constexpr double kTimeRtol = 1.0e-10;

/// For the two carried scalars, at the tolerance regression test 1 pins them at.
constexpr double kScalarRtol = 1.0e-12;

/**
 * **`tau_A` — THE PER-EVALUATION VELOCITY BOUND, AND THE NUMBER THIS WHOLE
 * MEMBER EXISTS TO ASSERT.** Max relative error of the FMM velocity against the
 * direct velocity **on the same state**, over owned rows, relative to the
 * direct field's own max magnitude.
 *
 * **Qualification list, which the conventions table requires of every stated
 * tolerance and without which this number is unreadable.** It is on the
 * **velocity**, i.e. the *gradient* of Canopy's potential and **not** the
 * potential — the two differ by a full order at fixed `order`, and T5 measured
 * `3.73e-5` on the potential at the same point, which is a different column and
 * must not be read as this one. Source distribution: the milestone-0 icosphere
 * at subdivision level `BEATNIK_M0_FMM_LEVEL`, at each of the 81 checkpointed
 * states of the 2000-step **direct** trajectory. Rank counts **1 and 4** (the
 * tier's), on SERIAL and HIP. Basis `FarFieldBasis::CartesianTaylor`;
 * `order = 3`; `ncrit = 8`; `max_depth = 10`; `mac_theta = 0.3`;
 * `softening = 0.025` (= sqrt(blob) at `--eps 0.025` under
 * `--kernel-blob-mode length`); `near_softening_factor = 0`. **The realized P2P
 * pair fraction is asserted and reported at every one of the 81 states** — a
 * figure without it may be a direct sum wearing an FMM's name (**R1**).
 *
 * **Where the value comes from.** T5 measured `tau_A = 5.007746e-04` at level 4
 * (`1.703480e-04` at level 3) at exactly that configuration, across three
 * launches agreeing to `5.0073e-4 .. 5.0088e-4`. **`1e-3` is the reference
 * implementation's own fidelity** and is what is compiled here rather than
 * T4's looser `2.0e-3`: T5's figure clears it by **2.0x** at level 4 and by
 * **5.9x** at level 3.
 *
 * **What has NOT been measured, and it is most of what this asserts.** T5's
 * figure is from **one** state, five steps off the initial condition. Claim A
 * evaluates at 81 states running out to a deformed sheet at step 2000, and
 * **no state past step 5 has ever been measured**. The per-step series is
 * printed at 17 digits precisely so that the next reader has it without
 * re-running. **If a late state exceeds this bound, that is the finding** — it
 * is to be recorded with the step and the realized P2P fraction, not
 * accommodated by loosening this literal.
 *
 * **Floor.** Twelve decades above the `1.15e-15` / `1.88e-15` run-to-run noise
 * T3 measured on the same binary, so this is not a bitwise claim in disguise.
 */
constexpr double kTauA = 1.0e-3;

/// Below this field scale the direct velocity is not a usable denominator and
/// claim A's error is compared **absolutely** instead. It is reached at
/// exactly one state and the reason is structural rather than numerical:
/// `--initial-potential-strength 0` makes phi, and therefore the sheet vector
/// `S = -n x grad_s(phi)`, identically zero at **step 0**, so both velocities
/// are identically zero there and a relative error would be `0/0`. Reported as
/// such rather than skipped, because 81 states is the claim and 80 is a
/// different one.
constexpr double kFieldScaleFloor = 1.0e-12;

/// How far past `kTauA` negative case 2 perturbs one component of one owned
/// row. A factor of two, so the case cannot be satisfied by round-off and
/// cannot be missed by a bound that is slightly loose.
constexpr double kPerturbationFactor = 2.0;

/// Leaf occupancy for **both levels and both claims**. See the header: this is
/// the configuration `tau_A`, the horizon envelope and the volume-drift bound
/// were all measured at. `FmmParams::ncrit` keeps the reference's 64.
constexpr int kNcrit = 8;

/// The production order, asserted rather than assumed — T5 measured the curve
/// and `FmmParams::order` already carries this value, so a change to the
/// default would silently move `tau_A`'s meaning.
constexpr int kProductionOrder = 3;

//---------------------------------------------------------------------------//
// CLAIM B's tolerances.
//---------------------------------------------------------------------------//

/**
 * How closely the **FMM-driven** run's per-step volume drift must track the
 * reference's own `kRefVolumeDrift` series.
 *
 * **The frozen member's `kVolumeDriftRtol = 1e-3` DOES NOT CARRY and must not
 * be reused here.** That literal was derived from direct-driven runs, whose
 * worst per-step deviation is `1.80e-5`; an FMM-driven run's is three decades
 * larger. Derived for T6 **offline from T5's surviving 2000-step FMM-driven
 * checkpoint directories** (`/p/lustre5/stewartj/beatnik/t5_divergence/`, 81
 * numbered `.h5` each) rather than from a fresh run, by recomputing
 * `enclosed_volume / initial_volume - 1` at every one of the 81 steps through
 * `milestone0_ladder.py`'s own convention and taking `|drift/reference - 1|`:
 *
 *   level 4, worst **2.137678e-02 at step 350**, over FOUR independent runs
 *           (np1 A, np1 B, np4 A, np4 B) agreeing to `1.3e-7` of the deviation;
 *   level 3, worst **1.540737e-02 at step 300**, over ONE run.
 *
 * The profile peaks mid-trajectory rather than at the end — level 4 falls back
 * to `8.05e-03` by step 2000 — so the **peak** and not the final value is what a
 * bound is set from. `5.0e-2` is a **2.34x** margin at level 4 and **3.25x** at
 * level 3, the larger margin at level 3 covering the fact that its figure rests
 * on a single run. The SERIAL backend is unmeasured here; the four measured runs
 * span two rank counts and agree to seven digits, which is what makes that
 * extrapolation reasonable rather than a hope. Do not loosen it without a new
 * measurement in `tasks/canopy/add-canopy-progress-log.md`.
 */
constexpr double kFmmVolumeDriftRtol = 5.0e-2;

/// The same bound for claim A's direct-driven trajectory, where the frozen
/// member's literal DOES carry: it is the same trajectory, and M0-D1 measured
/// a worst case of `2.758e-05` relative over eight 2000-step runs, a 36x
/// margin. Kept separate from `kFmmVolumeDriftRtol` on purpose — one literal
/// covering both would silently apply the FMM's three-decade-looser bound to
/// the direct half and stop it from catching anything.
constexpr double kVolumeDriftRtol = 1.0e-3;

/// The blow-up detector, kept absolute so a drift that tracks the reference
/// *proportionally* while both explode still fails. Unchanged from the frozen
/// member and it still carries: T5 measured `4.703e-09` FMM-driven against
/// `4.741e-09` direct-driven at step 2000, and both clear `1e-8`.
constexpr double kVolumeDriftAbsCap = 1.0e-8;

/// The ladder's rungs, which must be `milestone0_ladder.RUNGS` in order. They
/// are duplicated here ONLY so the JSON can be checked against them: a rung
/// silently reordered or retuned on the Python side would otherwise shift every
/// horizon in `kHorizonEnvelope` by one column with nothing reporting it.
constexpr int kRungCount = 5;
constexpr double kRungRtol[kRungCount] = { 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6,
                                           1.0e-4 };
constexpr double kRungAtol[kRungCount] = { 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8,
                                           1.0e-6 };
/// The index of the `1e-4` rung — the **load-bearing** one, and the only one
/// whose envelope is not the first checkpoint. Negative case 3 is built here.
constexpr int kLoadBearingRung = 4;

//---------------------------------------------------------------------------//
// PER-LEVEL REFERENCE NUMBERS. Every one is THIS LEVEL's and none is
// transferred from the other, exactly as the frozen pair's blocks are.
//
// The entity counts, the two carried scalars, the polyhedral deficit, the final
// `time` and the 81-entry `kRefVolumeDrift` series are reused VERBATIM from
// `Beatnik_Test_Milestone0Frozen.cpp`'s block for this level. That is a reuse
// and not a transfer: claim A drives the identical trajectory those members
// drive, against the identical gold set, so re-deriving them here would be a
// second source of truth for numbers already validated against M0-G1/M0-G2's
// own independently measured tables at every one of the 81 steps.
//
// `kRefVolumeDrift[i]` is step `25*i`; the row comments carry the first step of
// each row so an index can be checked by eye.
//
// The two numbers that are NOT the frozen member's, because they are about the
// FMM and it has none: `kP2PFractionReference` and `kHorizonEnvelope`, both
// measured by T5 at `ncrit = 8`.
//---------------------------------------------------------------------------//
#if BEATNIK_M0_FMM_LEVEL == 3

/// `V = 10*4^3 + 2`, `E = 30*4^3`, `F = 20*4^3`. Constant for the whole run,
/// and asserted constant at every step of BOTH trajectories.
constexpr long long kVertices = 642;
constexpr long long kEdges = 1920;
constexpr long long kFaces = 1280;

/// The level-3 gold set's own two carried scalars (its `_step0000000.npz`).
constexpr double kInitialVolume = 6.48865752670790275e-02;
constexpr double kInitialMinEdge = 3.45707933867918649e-02;

/// R9 discriminator 2 — the polyhedral deficit of the subdivision-3 icosphere,
/// `initial_volume / (4 pi R^3 / 3)` at `R = 0.25`. Partition-independent, so
/// double-counting even a handful of ghost faces moves it in the second or
/// third digit while a summation-order difference does not move it at all.
constexpr double kVolumeOverSphere = 9.91393842629754940e-01;

/// `time` at step 2000 of the **direct** trajectory, from the gold set's own
/// `/time` scalar. Claim A asserts it. **Claim B does not and must not**: the
/// adaptive dt is a function of the state, so an FMM-driven run is at a
/// slightly different physical time at the same step, and asserting this there
/// would be reporting the dt rather than the trajectory.
constexpr double kFinalTime = 1.99828394714319368e+00;

/// **The realized P2P pair fraction T5 measured at this level and `ncrit`**,
/// on a state five steps off the initial condition. REPORTED beside every one
/// of the 81 realized values and **not asserted equal to them**: the tree is
/// rebuilt against a deforming sheet at every evaluation, so the fraction moves
/// along the trajectory and pinning it would assert the geometry rather than
/// the far field.
constexpr double kP2PFractionReference = 0.854631;

/// **THE LEVEL-3 BOUND IS "A FAR FIELD EXISTS AT ALL", NOT "IT IS LIVE".** At
/// 642 vertices and `ncrit` 8 the M2L share is **14.5%**, so seven eighths of
/// the field is evaluated exactly and **this level's claim A is mostly a P2P
/// comparison**. It is asserted anyway, and what it asserts is the round trip,
/// the tag handshake and the contraction at level 3's decomposition — none of
/// which the level-4 member covers. The far-field accuracy claim rests on
/// level 4. The bound is still load-bearing: at `ncrit >= 32` this level's M2L
/// cell-pair count is **exactly 0** and the solve agrees with `BRSolverDirect`
/// to `2e-15`, which reads as a spectacular success and is **R1**'s cheapest
/// misreading.
constexpr double kP2PFractionBound = 1.0;
constexpr bool kFarFieldIsLive = false;

/// **THE M2L OPERATOR COLUMN-COUNT CAP, AND AT THIS LEVEL IT IS UNCHANGED.**
/// `FmmParams::m2l_op_count_cap`'s own default, restated here so the two arms
/// are read side by side. T5 measured level-3 demand at 828 - 6 624 keys over
/// five draws against this 32768 cap and `global_m2l_fallback` exactly zero in
/// all 243 rows, so the cap does not bind at this level and raising it would
/// change nothing except to move this member's overflow set off the value
/// every number it asserts was measured at.
constexpr int kM2LOpCountCap = 32768;

/// **THE DIVERGENCE HORIZON ENVELOPE**, T5 step 7, first failing checkpointed
/// step per rung, FMM-driven against this level's Python gold set under the
/// bijective nearest-neighbour pairing. Level 3's loosest rung is **75** rather
/// than level 4's 50, and that ordering is the expected one: at a 14.5% M2L
/// share this trajectory is mostly a direct sum, so it decorrelates later.
constexpr long long kHorizonEnvelope[kRungCount] = { 25, 25, 25, 25, 75 };

constexpr double kRefVolumeDrift[kComparedSteps] = {
    /*    0 */ 0.00000000000000000e+00, 1.54374513172683692e-10, 3.02764036064218089e-10,
    /*   75 */ 4.42957670543364657e-10, 5.73999958675130983e-10, 6.96251944987125171e-10,
    /*  150 */ 8.11173572756729300e-10, 9.20958198591392829e-10, 1.02816155589380287e-09,
    /*  225 */ 1.13540954416180284e-09, 1.24523569233758735e-09, 1.36004452144788957e-09,
    /*  300 */ 1.48177004000160650e-09, 1.61008184562660972e-09, 1.74555281340360580e-09,
    /*  375 */ 1.87985516042488143e-09, 2.00272176620330811e-09, 2.11420037032894470e-09,
    /*  450 */ 2.21661822230601047e-09, 2.31281704898833596e-09, 2.40471798029773254e-09,
    /*  525 */ 2.49293363729918838e-09, 2.57733656638947650e-09, 2.65806043842076178e-09,
    /*  600 */ 2.73649192195080104e-09, 2.80525802587305861e-09, 2.85643042552408133e-09,
    /*  675 */ 2.89413737419863537e-09, 2.92228863330024069e-09, 2.94404212120014108e-09,
    /*  750 */ 2.96177860015234273e-09, 2.97728108833439364e-09, 2.99002667070169537e-09,
    /*  825 */ 2.99986435692289888e-09, 3.00796343388753940e-09, 3.01514435641081491e-09,
    /*  900 */ 3.02203484459084848e-09, 3.02919178629679209e-09, 3.03719227545684589e-09,
    /*  975 */ 3.04671954332036421e-09, 3.05862224436737051e-09, 3.07397285403965270e-09,
    /* 1050 */ 3.09411385401858752e-09, 3.11843795230970500e-09, 3.13695780462808216e-09,
    /* 1125 */ 3.15072901102553260e-09, 3.16165471581086877e-09, 3.17093551416292030e-09,
    /* 1200 */ 3.17938542160334237e-09, 3.18761661510791328e-09, 3.19615489630109550e-09,
    /* 1275 */ 3.20551074572961170e-09, 3.21622550814026908e-09, 3.22808535457852486e-09,
    /* 1350 */ 3.23794124845733222e-09, 3.24613469437906588e-09, 3.25325721917124611e-09,
    /* 1425 */ 3.25972115966521869e-09, 3.26583138310354570e-09, 3.27182836379336095e-09,
    /* 1500 */ 3.27790927734383786e-09, 3.28423932494104065e-09, 3.29016880407095869e-09,
    /* 1575 */ 3.29536753440606844e-09, 3.29996518999564614e-09, 3.30408123083714145e-09,
    /* 1650 */ 3.30783045399130060e-09, 3.31131011499508077e-09, 3.31457949975799693e-09,
    /* 1725 */ 3.31768146288879962e-09, 3.32066241170991816e-09, 3.32356386856247354e-09,
    /* 1800 */ 3.32642602351995720e-09, 3.32928928870046548e-09, 3.33219118964223071e-09,
    /* 1875 */ 3.33517125028492956e-09, 3.33827210319270762e-09, 3.34153860137575975e-09,
    /* 1950 */ 3.34502292531624335e-09, 3.34878480501288323e-09, 3.35289418451623078e-09,
};

#else // BEATNIK_M0_FMM_LEVEL == 4

/// `V = 10*4^4 + 2`, `E = 30*4^4`, `F = 20*4^4`. Constant for the whole run.
constexpr long long kVertices = 2562;
constexpr long long kEdges = 7680;
constexpr long long kFaces = 5120;

/// The level-4 gold set's own two carried scalars. `initial_min_edge` is half
/// the level-3 one, which is what makes this the level that resolves
/// `--eps 0.025`.
constexpr double kInitialVolume = 6.53084210624162442e-02;
constexpr double kInitialMinEdge = 1.72957475903747181e-02;

/// R9 discriminator 2 at subdivision 4. Closer to 1 than level 3's, as a finer
/// triangulation of the same sphere must be.
constexpr double kVolumeOverSphere = 9.97839171610598097e-01;

/// `time` at step 2000 of the **direct** trajectory. **Not level 3's** — the
/// adaptive dt is relative to each run's own `initial_min_edge`. Claim A
/// asserts it; claim B must not, for the reason on level 3's copy.
constexpr double kFinalTime = 1.96430414465685987e+00;

/// **The realized P2P pair fraction T5 measured at this level and `ncrit`.**
/// Reported beside every realized value, not asserted equal to them.
constexpr double kP2PFractionReference = 0.253650;

/// **THE LIVENESS BOUND, AND AT THIS LEVEL IT IS A REAL ONE.** At 2562
/// vertices and `ncrit` 8, **74.6% of pairs go through M2L** — the far field
/// dominates and **this is where the far-field accuracy claim rests**. 0.75 is
/// T4's compiled bound, chosen from the a-priori arithmetic (320 occupied
/// leaves against a 105-leaf near field predicts a P2P fraction near 1/3) and
/// not from the measurement, so it is not a measurement in disguise; the
/// realized 0.2537 uses two thirds of the margin. **Asserted at every one of
/// the 81 states, not once**: the tree is rebuilt against a deforming sheet,
/// and a state at which the far field stopped carrying the majority of the
/// field would make `tau_A` a claim about something else.
constexpr double kP2PFractionBound = 0.75;
constexpr bool kFarFieldIsLive = true;

/// **THE M2L OPERATOR COLUMN-COUNT CAP, RAISED ABOVE `FmmParams`' 32768
/// DEFAULT FOR THIS LEVEL ONLY** (T8). At 2562 vertices, `ncrit` 8 and a
/// `key_needs_level` basis, a self-contacting roll-up occupies up to 8 tree
/// depths and every occupied depth multiplies the realized key count: T5
/// measured a worst-observed **37 678** demanded operator keys at HIP np1
/// rank 0, step 1650 -- **1.150x** the 32768 default -- over three draws
/// whose spread is 0.46 %, with `demand_saturated` never set, so that figure
/// is a peak and not a lower bound. T8's own control draw at the 32768 cap
/// peaked slightly higher still, at **37 846** keys at the same step -- within
/// T5's spread and the worst of four draws. 65536 covers that with **73 %**
/// headroom, and T8 measured `unique_ops == demand` at all 81 states on every
/// rank at np1 and np4 with it in force.
///
/// **WHAT IT COSTS, AND WHAT IT DOES NOT BUY.** The table is
/// `cap x bytes_per_key` = 65536 x 3200 B = **200 MiB** per rank, against a
/// 2 GiB byte budget that buys 671 088 columns -- so the count cap is still
/// the binding constraint, by 10.2x, and the byte budget is nowhere near it.
/// The per-evaluation REBUILD cost is bounded by demand and not by the cap
/// (only admitted keys are built, and `keys_built_delta == unique_ops` in 405
/// of 405 measured rows, so the cache retains nothing on a drifting bounding
/// box), so the raise costs at most 1.150x at the demand peak and nothing at
/// the 44 of 81 np1 states already under 32768.
///
/// **IT CANNOT MAKE `p.m2l_fallback == 0` BY ITSELF.** At np4 no rank's demand
/// ever reaches even the old cap, and fallback is still non-zero at 71 of 81
/// states, so those refusals are not cap-driven. Raising the cap removes the
/// cap-driven ones; the rest are a different refusal path.
constexpr int kM2LOpCountCap = 65536;

/// **THE DIVERGENCE HORIZON ENVELOPE**, T5 step 7. Identical across **four**
/// independent runs — two at np1 (reduction order only) and two at np4 (which
/// is the strict test, since it varies Zoltan2's partition) — whose worst
/// spread is `1.4e-11` of the error. **R8 does not bind at this
/// configuration**, so the envelope may be asserted at the observed horizon;
/// the margin comes from the checkpoint interval and not from run-to-run noise.
constexpr long long kHorizonEnvelope[kRungCount] = { 25, 25, 25, 25, 50 };

constexpr double kRefVolumeDrift[kComparedSteps] = {
    /*    0 */ 0.00000000000000000e+00, 1.59270374666675707e-10, 3.12319503592561887e-10,
    /*   75 */ 4.56835680395784038e-10, 5.91826143647722347e-10, 7.17669923488983841e-10,
    /*  150 */ 8.35882252303576934e-10, 9.48717771009910393e-10, 1.05876640787982979e-09,
    /*  225 */ 1.16864451449316675e-09, 1.28081789618761377e-09, 1.39755607087010958e-09,
    /*  300 */ 1.52100820827172356e-09, 1.65344338221018461e-09, 1.79660974986006750e-09,
    /*  375 */ 1.94750837678725475e-09, 2.10779060871857382e-09, 2.28425900417050798e-09,
    /*  450 */ 2.47268627795449447e-09, 2.67017230548560747e-09, 2.88076495991163029e-09,
    /*  525 */ 3.10373327039314972e-09, 3.33270788743789126e-09, 3.55701779142236774e-09,
    /*  600 */ 3.76518038969209101e-09, 3.94856236596297094e-09, 4.10311606913182914e-09,
    /*  675 */ 4.22897050711412703e-09, 4.32896185564857205e-09, 4.40714531535491005e-09,
    /*  750 */ 4.46774239826197572e-09, 4.51457560224355348e-09, 4.55084148143214406e-09,
    /*  825 */ 4.57820026333877195e-09, 4.59779259109893701e-09, 4.61190952094625572e-09,
    /*  900 */ 4.62219973407229645e-09, 4.62981897264569398e-09, 4.63557303653772124e-09,
    /*  975 */ 4.64002281042041886e-09, 4.64355776053082536e-09, 4.64645344422365270e-09,
    /* 1050 */ 4.64890392848360534e-09, 4.65105087776862547e-09, 4.65300065144447217e-09,
    /* 1125 */ 4.65483451783654800e-09, 4.65658089865428337e-09, 4.65826843765171361e-09,
    /* 1200 */ 4.65996508047794578e-09, 4.66173166735472932e-09, 4.66362726214697432e-09,
    /* 1275 */ 4.66570737600591201e-09, 4.66802974052882291e-09, 4.67064475984102501e-09,
    /* 1350 */ 4.67357774702747975e-09, 4.67673100246202011e-09, 4.68008298781796839e-09,
    /* 1425 */ 4.68351801785615862e-09, 4.68695326993895378e-09, 4.69039007633398342e-09,
    /* 1500 */ 4.69385197376936958e-09, 4.69735139674298807e-09, 4.70088168391669114e-09,
    /* 1575 */ 4.70442063082998629e-09, 4.70782945960479537e-09, 4.71105154886686250e-09,
    /* 1650 */ 4.71403471813403030e-09, 4.71676409041776878e-09, 4.71927097400737239e-09,
    /* 1725 */ 4.72159022990581434e-09, 4.72375827342830235e-09, 4.72581085375622933e-09,
    /* 1800 */ 4.72778172166954391e-09, 4.72969907683307156e-09, 4.73159111891163775e-09,
    /* 1875 */ 4.73344585749657654e-09, 4.73519179422510206e-09, 4.73684513835337384e-09,
    /* 1950 */ 4.73842098891452679e-09, 4.73993821969997953e-09, 4.74141392814431128e-09,
};

#endif // BEATNIK_M0_FMM_LEVEL

//---------------------------------------------------------------------------//
// Small utilities. `fileExists`, `goldForStep` and `runComparator` are
// `Beatnik_Test_Milestone0Frozen.cpp`'s, unchanged in behaviour and repeated
// rather than shared because that file is a test body included by its own L4
// stem, not a header: including it here would drag its whole `runChecks` in.
//---------------------------------------------------------------------------//
bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// The gold file for `step`, found by its `_step%07d.npz` suffix rather than by
/// rebuilding the name from a time — the time is exactly what is under test.
/// Empty if the directory holds no such file, which the caller reports as a
/// named failure.
std::string goldForStep( const std::string& directory, long long step )
{
    char suffix[32];
    std::snprintf( suffix, sizeof( suffix ), "_step%07lld.npz", step );
    const std::string want( suffix );

    DIR* dir = ::opendir( directory.c_str() );
    if ( !dir )
        return std::string();

    std::string found;
    while ( struct dirent* entry = ::readdir( dir ) )
    {
        const std::string name( entry->d_name );
        if ( name.size() >= want.size() &&
             name.compare( name.size() - want.size(), want.size(), want ) == 0 )
        {
            found = directory + "/" + name;
            break;
        }
    }
    ::closedir( dir );
    return found;
}

/// Wall seconds spent inside `runComparator`, accumulated on rank 0. Reported
/// beside the two solve times: 83 Python invocations are a real share of a
/// level-3 launch and a later session sizing the tier's walltime needs the
/// split. The ladder is clocked separately — it is one invocation and a
/// different order of cost.
double g_comparator_seconds = 0.0;
long long g_comparator_calls = 0;
double g_ladder_seconds = 0.0;

/// Run `python <script> <a> <b> --rtol .. --atol ..` and return its exit
/// status, or -1 if it could not be run at all. The three outcomes are never
/// conflated: 0 match, 1 compared and disagreed, 2 could not load, -1 plumbing.
int runComparator( const std::string& python, const std::string& script,
                   const std::string& lhs, const std::string& rhs )
{
    std::ostringstream cmd;
    cmd << "'" << python << "' '" << script << "' '" << lhs << "' '" << rhs
        << "' --rtol " << kRtol << " --atol " << kAtol << " --quiet";
    std::printf( "[cmd] %s\n", cmd.str().c_str() );
    std::fflush( stdout );

    const double t0 = MPI_Wtime();
    const int raw = std::system( cmd.str().c_str() );
    g_comparator_seconds += MPI_Wtime() - t0;
    ++g_comparator_calls;

    if ( raw == -1 || !WIFEXITED( raw ) )
        return -1;
    return WEXITSTATUS( raw );
}

//---------------------------------------------------------------------------//
// CLAIM B's LADDER, AND WHY ITS EXIT STATUS IS NOT ITS VERDICT.
//
// `fmm_divergence_ladder.py`'s status reports whether the MEASUREMENT
// completed — it returns 1 when some step was unpairable — so 0 and 1 are both
// "it ran" and the verdict is in the JSON. Anything else is plumbing. A member
// that asserted on the status alone would fail on level 4's known
// steps 1350-1900 refusal, which is neither a pass nor a failure.
//
// The JSON is parsed by hand rather than with a dependency. That is a cost, and
// what buys it back is that the shape being parsed is two scalars per rung out
// of a document this repository writes: `{"rtol": .., "atol": .., "
// first_failing_step": <int|null>, "per_field": {..}}`, five times, under a
// top-level `"ladder"` key, plus a flat `"unpairable_steps"` array. Every
// failure to find what it expects is a named `rec.fail`, never a default.
//---------------------------------------------------------------------------//

/// One rung as the JSON carries it. `null_step` distinguishes "never failed"
/// (JSON `null`) from "failed at step 0", which are opposite verdicts.
struct LadderRung
{
    double rtol = 0.0;
    double atol = 0.0;
    long long first_failing_step = -1;
    bool null_step = true;
};

/// Position just past `"<key>":` at or after `from`, or `npos`.
std::size_t jsonValueAt( const std::string& s, const char* key,
                         std::size_t from )
{
    const std::string k = std::string( "\"" ) + key + "\":";
    const std::size_t at = s.find( k, from );
    if ( at == std::string::npos )
        return std::string::npos;
    std::size_t v = at + k.size();
    while ( v < s.size() && std::isspace( static_cast<unsigned char>( s[v] ) ) )
        ++v;
    return v;
}

/// THE HORIZON PREDICATE, and the ONE place the comparison is made — negative
/// case 3 calls this with a fabricated step rather than reimplementing it, so
/// what the negative case proves live is the code the assertion runs.
///
/// A rung that never failed (`null`) is not earlier than anything and passes.
/// The assertion is deliberately one-sided: a horizon LATER than the envelope
/// is not a failure, because the envelope is a floor on when decorrelation may
/// begin and a change that delays it is not a regression.
bool horizonNoEarlierThan( bool null_step, long long measured,
                           long long envelope )
{
    if ( null_step )
        return true;
    return measured >= envelope;
}

/// Run the ladder and fill `rungs` and `unpairable` from its `--json`. Returns
/// false, having already called `rec.fail`, if the measurement could not be
/// made or the document could not be read. Rank 0 only.
bool runLadder( Beatnik::Test::Recorder& rec, const std::string& python,
                const std::string& script, const std::string& run_dir,
                const std::string& ref_dir, const std::string& json_path,
                const std::string& label, std::vector<LadderRung>& rungs,
                std::vector<long long>& unpairable )
{
    std::ostringstream cmd;
    cmd << "'" << python << "' '" << script << "' ladder --run '" << run_dir
        << "' --ref '" << ref_dir << "' --label '" << label << "' --json '"
        << json_path << "'";
    std::printf( "[cmd] %s\n", cmd.str().c_str() );
    std::fflush( stdout );

    const double t0 = MPI_Wtime();
    const int raw = std::system( cmd.str().c_str() );
    g_ladder_seconds += MPI_Wtime() - t0;
    const int status = ( raw == -1 || !WIFEXITED( raw ) ) ? -1
                                                          : WEXITSTATUS( raw );

    {
        std::ostringstream os;
        os << "claim B ladder exit " << status
           << " (0 = measured every step, 1 = measured with at least one "
              "UNPAIRABLE step, which is neither a pass nor a failure; "
              "anything else is plumbing). The verdict is read from the JSON, "
              "never from this status.";
        rec.note( os.str() );
    }
    if ( status != 0 && status != 1 )
    {
        rec.fail( "claim B: the divergence ladder could not be run at all "
                  "(exit " + std::to_string( status ) + "); its verdict is "
                  "unmeasured, which is not a pass" );
        return false;
    }

    std::ifstream in( json_path.c_str() );
    if ( !in )
    {
        rec.fail( "claim B: the ladder reported exit " +
                  std::to_string( status ) + " but wrote no JSON at " +
                  json_path );
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string doc = buffer.str();

    // `unpairable_steps` is a flat array of ints and sits before `rows`.
    unpairable.clear();
    {
        const std::size_t v = jsonValueAt( doc, "unpairable_steps", 0 );
        if ( v == std::string::npos || doc[v] != '[' )
        {
            rec.fail( "claim B: the ladder JSON carries no "
                      "\"unpairable_steps\" array" );
            return false;
        }
        const std::size_t end = doc.find( ']', v );
        if ( end == std::string::npos )
        {
            rec.fail( "claim B: the ladder JSON's \"unpairable_steps\" array "
                      "is unterminated" );
            return false;
        }
        std::string body = doc.substr( v + 1, end - v - 1 );
        for ( char& c : body )
            if ( c == ',' )
                c = ' ';
        std::istringstream steps( body );
        long long step = 0;
        while ( steps >> step )
            unpairable.push_back( step );
    }

    // `ladder` is the LAST top-level key the tool writes. `rfind` rather than
    // `find`, because `label` is free text and could in principle carry the
    // word; the per-row `"first_failing_rung"` arrays above it are a different
    // key and cannot be confused with `"first_failing_step"`.
    const std::size_t ladder_at = doc.rfind( "\"ladder\":" );
    if ( ladder_at == std::string::npos )
    {
        rec.fail( "claim B: the ladder JSON carries no \"ladder\" array" );
        return false;
    }

    rungs.clear();
    std::size_t cursor = ladder_at;
    for ( int i = 0; i < kRungCount; ++i )
    {
        LadderRung r;
        const std::size_t vr = jsonValueAt( doc, "rtol", cursor );
        const std::size_t va = jsonValueAt( doc, "atol", cursor );
        const std::size_t vs = jsonValueAt( doc, "first_failing_step", cursor );
        if ( vr == std::string::npos || va == std::string::npos ||
             vs == std::string::npos )
        {
            rec.fail( "claim B: the ladder JSON has fewer than " +
                      std::to_string( kRungCount ) + " rungs; it stopped at " +
                      std::to_string( i ) );
            return false;
        }
        r.rtol = std::strtod( doc.c_str() + vr, nullptr );
        r.atol = std::strtod( doc.c_str() + va, nullptr );
        if ( doc.compare( vs, 4, "null" ) == 0 )
        {
            r.null_step = true;
            r.first_failing_step = -1;
        }
        else
        {
            r.null_step = false;
            r.first_failing_step = std::strtoll( doc.c_str() + vs, nullptr, 10 );
        }
        rungs.push_back( r );
        cursor = vs + 1;
    }
    return true;
}

//---------------------------------------------------------------------------//
// Global reductions. Every reported scalar goes through one of these over
// OWNED rows only (risk R9), so a four-rank run and a one-rank run compare the
// same quantity against the same literal.
//---------------------------------------------------------------------------//
double globalMax( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MAX, comm );
    return out;
}

long long globalCount( MPI_Comm comm, long long local )
{
    long long out = local;
    MPI_Allreduce( &local, &out, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

/// Max magnitude of an `(N,3)` field over owned rows, reduced globally. The
/// field scale claim A's relative error is expressed against.
template <class ExecSpace, class ViewType>
double fieldScale( MPI_Comm comm, const ViewType& v, int n )
{
    Real worst = 0;
    Kokkos::parallel_reduce(
        "beatnik_t6_field_scale", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m ) {
            const Real mag = Kokkos::sqrt( v( i, 0 ) * v( i, 0 ) +
                                           v( i, 1 ) * v( i, 1 ) +
                                           v( i, 2 ) * v( i, 2 ) );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ) );
    return globalMax( comm, static_cast<double>( worst ) );
}

/// Max `|a_i - b_i|` over owned rows, reduced globally, together with the count
/// of non-finite rows in `a`. One pass, because the finiteness check and the
/// error are the same walk.
struct FieldDifference
{
    double max_abs = 0.0;
    long long nonfinite = 0;
};

template <class ExecSpace, class ViewA, class ViewB>
FieldDifference fieldDifference( MPI_Comm comm, const ViewA& a, const ViewB& b,
                                 int n )
{
    Real worst = 0;
    long long bad = 0;
    Kokkos::parallel_reduce(
        "beatnik_t6_field_difference", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m, long long& nb ) {
            Real d[3];
            bool finite = true;
            for ( int c = 0; c < 3; ++c )
            {
                d[c] = a( i, c ) - b( i, c );
                if ( !Kokkos::isfinite( a( i, c ) ) )
                    finite = false;
            }
            if ( !finite )
            {
                ++nb;
                return;
            }
            const Real mag =
                Kokkos::sqrt( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ), bad );

    FieldDifference out;
    out.max_abs = globalMax( comm, static_cast<double>( worst ) );
    out.nonfinite = globalCount( comm, bad );
    return out;
}

//---------------------------------------------------------------------------//
/// One claim-A evaluation, everything it produced. Kept per state so the whole
/// 81-entry series can be printed at 17 digits at the end, in one block, rather
/// than interleaved with 83 comparator invocations.
struct ClaimAPoint
{
    long long step = 0;
    double scale = 0.0;      ///< max |u_direct|, the denominator.
    double max_abs = 0.0;    ///< max |u_fmm - u_direct| over owned rows.
    double rel = 0.0;        ///< the quantity `kTauA` bounds.
    bool relative = true;    ///< false at a state whose field scale is zero.
    long long nonfinite = 0;
    double p2p_fraction = 0.0;
    long long m2l_pairs = 0;
    long long m2l_fallback = 0;
    long long particles = 0;
};

//---------------------------------------------------------------------------//
/// The milestone-0 command line as a `SolverParams`. Field for field
/// `Beatnik_Test_Milestone0Frozen.cpp::makeParams`, with the BR approximation
/// and the checkpoint directory as parameters: claim A passes `Direct` and
/// claim B passes `Fmm`, and **that single field is the whole difference
/// between the two trajectories**.
Beatnik::SolverParams makeParams( const std::string& checkpoint_dir,
                                  Beatnik::BRApproximation approximation )
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere, --radius 0.25,
    // --center-z 0.25, --icosphere-subdivisions <L>.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = kSubdivisions;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0.
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --sigma 0
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
    p.zmodel.sigma = 0.0;
    // --forcing-sign 1 --br-sign 1 --kernel-blob-mode length
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = Beatnik::KernelBlobMode::Length;
    // --viscosity-mode laplace-beltrami, --velocity-mode full,
    // --bernoulli-scalar-mode normal-speed, preserve_volume on.
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = Beatnik::VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = Beatnik::BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    // THE ONE FIELD THAT DIFFERS BETWEEN THE TWO CLAIMS.
    p.zmodel.br_approximation = approximation;
    // --source-quadrature vertex. MANDATORY, not decorative: ZModelParams
    // defaults to Face, whose generate() throws, and the far-field adapter
    // rejects anything but Vertex regardless.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // The far field, for claim B's trajectory. `ncrit` is the ONLY departure
    // from T1's compiled defaults, and it is the configuration every number
    // this member asserts was measured at (see kNcrit). `basis`, `order`,
    // `mac_theta`, `max_depth` and `near_softening_factor` are left at their
    // defaults deliberately and then ASSERTED below, so a change to a default
    // is a failure here rather than a silent move of tau_A's meaning.
    p.fmm.ncrit = kNcrit;

    // --steps 2000, --adaptive-dt, and the dt controls both gold sets were
    // generated under. Every one is a Python default and every one changes the
    // trajectory.
    p.time.steps = kSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --no-dynamic-remesh --refine-every 0. Connectivity is frozen for both
    // trajectories, which is what makes the entity-count assertion a statement
    // about the FMM rather than about the remesher.
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;
    p.cleanup.enabled = true;

    // --checkpoint-every-steps 25. `setup()` writes step 0 unconditionally, so
    // 2000 steps every 25 gives the gold sets' own 81 files.
    p.checkpoint.every_steps = kCheckpointEvery;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

    return p;
}

/// `FmmParams` for claim A's standalone evaluations — the same struct claim B's
/// `SolverParams` carries, built the same way, so the two claims cannot be at
/// different configurations.
Beatnik::FmmParams makeFmmParams()
{
    Beatnik::FmmParams f;
    f.ncrit = kNcrit;
    // The M2L operator column-count cap, per level (`kM2LOpCountCap`, set in
    // the `#if BEATNIK_M0_FMM_LEVEL` block above). It is read here rather than
    // written as a literal because this function sits OUTSIDE those arms: a
    // literal would move level 3's overflow set too, and level 3 keeps the
    // 32768 default byte for byte.
    f.m2l_op_count_cap = kM2LOpCountCap;
    return f;
}

#ifndef BEATNIK_ENABLE_CANOPY
//---------------------------------------------------------------------------//
/// **THE `~canopy` BUILD.** There is no far field at all in that build:
/// `Beatnik_CreateBRSolver.hpp` preprocesses out the `new BRSolverFMM<...>`
/// line and `BRSolverFMM`'s own constructor throws. Neither claim is runnable,
/// and running the body anyway would report a stack of failures that say
/// nothing about the code under test. What IS true there is asserted instead —
/// both routes to the FMM throw a `std::runtime_error` naming the missing
/// build option — so the member is meaningful in both builds rather than
/// vacuous in one. Following T4's precedent (`Beatnik_Test_FmmVsDirect.cpp`),
/// and **compiled but unexecuted here**: this machine builds `+canopy`.
template <class ExecSpace, class MemSpace>
void runChecksNoCanopy( Beatnik::Test::Recorder& rec )
{
    rec.note( "BUILT WITHOUT BEATNIK_ENABLE_CANOPY: neither claim is runnable. "
              "Asserting the two documented throws instead." );
    bool threw = false;
    std::string what;
    try
    {
        Beatnik::BRSolverFMM<ExecSpace, MemSpace> fmm( MPI_COMM_WORLD,
                                                       makeFmmParams() );
        (void)fmm;
    }
    catch ( const std::runtime_error& e )
    {
        threw = true;
        what = e.what();
    }
    rec.note( "BRSolverFMM construction threw: " + what );
    BEATNIK_CHECK_TRUE( rec, threw );
    BEATNIK_CHECK_TRUE(
        rec, what.find( "BEATNIK_ENABLE_CANOPY" ) != std::string::npos );
}
#endif

//---------------------------------------------------------------------------//
// The body.
//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using solver_type = Beatnik::Solver<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using direct_type = Beatnik::BRSolverDirect<ExecSpace, MemSpace>;
    using fmm_type = Beatnik::BRSolverFMM<ExecSpace, MemSpace>;
    using vector_view =
        Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );

    {
        std::ostringstream os;
        os << "execution space " << ExecSpace::name() << ", ranks " << comm_size
           << ", icosphere subdivisions " << kSubdivisions << " (" << kVertices
           << " vertices), steps " << kSteps << ", ncrit " << kNcrit
           << ", order " << kProductionOrder
           << "; claim A compares " << kComparedSteps
           << " checkpointed steps at rtol " << kRtol << " atol " << kAtol
           << " and bounds the same-state FMM velocity error by " << kTauA
           << "; claim B runs " << kSteps << " FMM-driven steps";
        rec.note( os.str() );
    }

#ifndef BEATNIK_ENABLE_CANOPY
    runChecksNoCanopy<ExecSpace, MemSpace>( rec );
    (void)argc;
    (void)argv;
    return;
#endif

    if ( argc < 4 )
    {
        rec.fail( "usage: <gold-dir> <compare_output.py> "
                  "<fmm_divergence_ladder.py>; see the ARGUMENTS block in "
                  "this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const std::string gold_dir = argv[1];
    const std::string script = argv[2];
    const std::string ladder_script = argv[3];
    const char* python_env = std::getenv( "BEATNIK_PYTHON" );
    const std::string python = python_env ? python_env : "python3";

    // Every input path is checked BEFORE it is used, so a mis-plumbed path is
    // reported as itself rather than as a comparison failure -- and all 81 gold
    // files are checked up front rather than at the step that needs them,
    // because discovering a missing step-1975 file after an hour of solving
    // wastes the run, and this member's run is HOURS.
    if ( rank == 0 )
    {
        BEATNIK_CHECK_TRUE( rec, fileExists( gold_dir ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( script ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( ladder_script ) );
        int missing = 0;
        for ( int i = 0; i < kComparedSteps; ++i )
        {
            const long long s = static_cast<long long>( i ) * kCheckpointEvery;
            if ( goldForStep( gold_dir, s ).empty() )
            {
                rec.fail( "no gold file for step " + std::to_string( s ) +
                          " in " + gold_dir );
                ++missing;
            }
        }
        std::ostringstream os;
        os << "gold set " << gold_dir << ": " << ( kComparedSteps - missing )
           << " of " << kComparedSteps << " compared steps present";
        rec.note( os.str() );
    }

    // Resolution order, and why there are three levels: the installed runner
    // path runs from the manifest's directory, which is inside a spack install
    // prefix and is READ-ONLY. `BEATNIK_TEST_SCRATCH` is what the runner sets
    // (absolute, and on a PARALLEL filesystem -- the checkpoints go through
    // MPI-IO); TMPDIR covers a hand-run from an install prefix; "." covers
    // ctest, which runs in the build tree.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream root;
    root << ( scratch_env ? scratch_env : "." ) << "/beatnik_milestone0_fmm_sub"
         << kSubdivisions << "/" << ExecSpace::name() << "_np" << comm_size;
    // TWO trajectories, TWO directories, and they must not share one: the
    // ladder refuses a directory holding two files for one step, and a shared
    // directory would silently make claim B's measurement a mixture of the two
    // runs. Named for the claim rather than for the approximation so the log
    // says which assertion a stray file belongs to.
    const std::string claim_a_dir = root.str() + "/claimA_direct";
    const std::string claim_b_dir = root.str() + "/claimB_fmm";
    const std::string ladder_json = root.str() + "/claimB_ladder.json";
    rec.note( "claim A checkpoints " + claim_a_dir );
    rec.note( "claim B checkpoints " + claim_b_dir );

    //=======================================================================//
    // CLAIM A -- the per-evaluation bound, on 81 states of the DIRECT
    // trajectory.
    //=======================================================================//
    std::vector<ClaimAPoint> series;
    std::string claim_a_last_checkpoint;
    double t_claim_a = 0.0;
    bool claim_a_stopped = false;
    double claim_a_final_time = 0.0;
    long long claim_a_completed = 0;

    {
        solver_type solver( comm,
                            makeParams( claim_a_dir,
                                        Beatnik::BRApproximation::Direct ) );
        solver.setup();

        auto& mesh = solver.mesh();
        const auto& state = solver.state();

        //-------------------------------------------------------------------//
        // Structure, before anything evolves. Reduced as integers, so exact at
        // every rank count.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), ( mesh_type::halo_depth ) );

        //-------------------------------------------------------------------//
        // R9 DISCRIMINATOR 1 -- do the owned sets PARTITION the global sets?
        // Summed with a plain MPI_Allreduce over `ownedXCount()` rather than
        // read from Tessera: two independent paths to the same number, and
        // owned-versus-local is exactly what R9 turns on. Re-run at every
        // compared step, because it has to hold for 2000 steps and not only at
        // setup.
        //-------------------------------------------------------------------//
        auto checkOwnedPartition = [&]( long long step, bool verbose )
        {
            long long owned[3] = { mesh.ownedVertexCount(),
                                   mesh.ownedEdgeCount(),
                                   mesh.ownedFaceCount() };
            long long total[3] = { 0, 0, 0 };
            MPI_Allreduce( owned, total, 3, MPI_LONG_LONG, MPI_SUM,
                           mesh.comm() );
            if ( verbose )
            {
                std::ostringstream os;
                os << "claim A step " << step
                   << " owned partition: sum over ranks V " << total[0] << " E "
                   << total[1] << " F " << total[2] << "; this rank owns V "
                   << owned[0] << " of local V " << mesh.totalVertexCount();
                rec.note( os.str() );
            }
            if ( total[0] != kVertices || total[1] != kEdges ||
                 total[2] != kFaces )
            {
                std::ostringstream os;
                os << "ENTITY COUNTS CHANGED at claim A step " << step
                   << ": summed owned V " << total[0] << " E " << total[1]
                   << " F " << total[2] << ", expected " << kVertices << " / "
                   << kEdges << " / " << kFaces
                   << ". Adaptivity leaked into the frozen-mesh "
                      "configuration, or the owned sets stopped partitioning "
                      "the global ones.";
                rec.fail( os.str() );
                return false;
            }
            return true;
        };
        checkOwnedPartition( 0, true );

        //-------------------------------------------------------------------//
        // The two carried scalars, pinned before the first step against THIS
        // LEVEL's gold values.
        //-------------------------------------------------------------------//
        const double initial_volume =
            static_cast<double>( solver.initialVolume() );
        const double h0 = static_cast<double>( solver.initialMinEdge() );
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "initial_volume " << initial_volume << " vs gold "
               << kInitialVolume << ", initial_min_edge " << h0 << " vs gold "
               << kInitialMinEdge;
            rec.note( os.str() );
        }
        BEATNIK_CHECK_CLOSE( rec, initial_volume, kInitialVolume, kScalarRtol );
        BEATNIK_CHECK_CLOSE( rec, h0, kInitialMinEdge, kScalarRtol );

        //-------------------------------------------------------------------//
        // R9 DISCRIMINATOR 2 -- the polyhedral deficit at step 0, re-derived
        // per level. Partition-independent, so double-counting even a handful
        // of ghost faces moves it in the second or third digit.
        //-------------------------------------------------------------------//
        {
            const double sphere = 4.0 * M_PI * std::pow( kRadius, 3 ) / 3.0;
            const double ratio = initial_volume / sphere;
            std::ostringstream os;
            os.precision( 17 );
            os << "volume / (4*pi*R^3/3) = " << ratio << " (expected "
               << kVolumeOverSphere << " at subdivision " << kSubdivisions
               << "; partition-independent)";
            rec.note( os.str() );
            BEATNIK_CHECK_CLOSE( rec, ratio, kVolumeOverSphere, 1.0e-12 );
        }

        //-------------------------------------------------------------------//
        // The per-step volume drift, against the REFERENCE's own measured
        // series. OWNED faces only, then one MPI_Allreduce -- the same
        // convention `enclosedVolume` documents and the same one
        // `initial_volume` was computed under.
        //-------------------------------------------------------------------//
        auto checkVolumeDrift = [&]( long long step )
        {
            auto pos = mesh.positions();
            auto owned_faces = Kokkos::subview(
                mesh.faceVertices(), std::make_pair( 0, mesh.ownedFaceCount() ),
                Kokkos::ALL() );
            const Real local =
                Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
            Real volume = 0;
            MPI_Allreduce( &local, &volume, 1, MPI_DOUBLE, MPI_SUM,
                           mesh.comm() );
            const double drift =
                static_cast<double>( volume ) / initial_volume - 1.0;
            const double reference = kRefVolumeDrift[step / kCheckpointEvery];
            const double deviation = reference == 0.0
                                         ? std::fabs( drift )
                                         : std::fabs( drift / reference - 1.0 );
            std::ostringstream os;
            os.precision( 17 );
            os << "claim A step " << step << " relative drift " << drift
               << " reference " << reference;
            os.precision( 6 );
            os << " deviation " << deviation << " (rtol " << kVolumeDriftRtol
               << ", abs cap " << kVolumeDriftAbsCap << ")";
            rec.note( os.str() );
            BEATNIK_CHECK_TRUE( rec, deviation <= kVolumeDriftRtol );
            BEATNIK_CHECK_TRUE( rec, std::fabs( drift ) <= kVolumeDriftAbsCap );
        };

        //-------------------------------------------------------------------//
        // One compared step: the checkpoint Beatnik just wrote against that
        // step's gold file. Rank 0 only -- the comparator is serial Python over
        // one file. THIS IS `compare_output.py`, UNCHANGED, and it is correct
        // here for the reason it is wrong for claim B: a direct-driven run
        // satisfies its quantized-lexsort pairing precondition by nine decades.
        //-------------------------------------------------------------------//
        auto compareStep = [&]( long long step )
        {
            if ( rank != 0 )
                return;
            const std::string written = solver.lastCheckpointPath();
            const std::string gold = goldForStep( gold_dir, step );
            BEATNIK_CHECK_TRUE( rec, fileExists( written ) );
            if ( gold.empty() || !fileExists( written ) )
            {
                rec.fail( "claim A step " + std::to_string( step ) +
                          ": missing gold or output file" );
                return;
            }
            const int status = runComparator( python, script, written, gold );
            std::ostringstream os;
            os << "claim A step " << step << " comparator exit " << status
               << " (0 = match, 1 = compared and disagreed, 2 = LOAD ERROR)";
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, status, 0 );
        };

        //-------------------------------------------------------------------//
        // THE CLAIM-A EVALUATION ITSELF.
        //
        // The preconditions are established in the order `ZModelSolver`'s RHS
        // establishes them (`Beatnik_ZModelSolver.hpp` steps 0-2): one
        // whole-tuple halo exchange, geometry at the CURRENT positions, then
        // the sheet vector. Both BR solvers are then handed the SAME mesh,
        // geometry, state and quadrature, which is what makes this a
        // comparison of the two summations and of nothing else.
        //
        // **It cannot perturb the trajectory.** It writes only into its own two
        // output views and into `state`'s sheet vector, which the next RHS
        // recomputes from scratch at its first stage; the adaptive dt on this
        // configuration is a function of edge lengths alone
        // (`max_sheet_dt_product` is 0), and nothing here moves a vertex. The
        // checkpoint for this step was already written inside
        // `advanceOneStep()` before this runs.
        //
        // The two solvers are constructed ONCE, outside the loop, and reused
        // at all 81 states -- which is also what an FMM-driven run does, so the
        // adapter's forward-distributor reuse branch and Canopy's operator
        // cache are exercised the way claim B exercises them.
        //-------------------------------------------------------------------//
        auto quadrature = Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
            Beatnik::SourceQuadrature::Vertex );
        direct_type direct( comm );
        fmm_type fmm( comm, makeFmmParams() );

        // The compiled defaults, asserted rather than assumed. `tau_A` is a
        // claim about THIS parameter set, so a change to any default below
        // silently changes what the bound means.
        BEATNIK_CHECK_EQ( rec, fmm.farField().params().ncrit, kNcrit );
        BEATNIK_CHECK_EQ( rec, fmm.farField().params().order,
                          kProductionOrder );
        BEATNIK_CHECK_EQ( rec,
                          static_cast<int>( fmm.farField().params().basis ),
                          static_cast<int>(
                              Beatnik::FarFieldBasis::CartesianTaylor ) );
        BEATNIK_CHECK_CLOSE( rec, fmm.farField().params().mac_theta, 0.3,
                             1.0e-15 );
        BEATNIK_CHECK_EQ( rec, fmm.farField().params().max_depth, 10 );
        BEATNIK_CHECK_CLOSE(
            rec, fmm.farField().params().near_softening_factor, 0.0, 0.0 );

        const int n_owned0 = mesh.ownedVertexCount();
        vector_view u_direct( "beatnik_t6_u_direct", n_owned0 );
        vector_view u_fmm( "beatnik_t6_u_fmm", n_owned0 );
        vector_view u_pert( "beatnik_t6_u_pert", n_owned0 );

        const Beatnik::ZModelParams zparams =
            makeParams( claim_a_dir, Beatnik::BRApproximation::Direct ).zmodel;

        auto evaluateClaimA = [&]( long long step ) -> ClaimAPoint
        {
            const int n_owned = mesh.ownedVertexCount();
            mesh.haloExchange();
            geometry_type geometry;
            geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                              mesh.faceVertices() );
            state.updateSheetVector( mesh, geometry );

            if ( static_cast<int>( u_direct.extent( 0 ) ) != n_owned )
            {
                Kokkos::realloc( u_direct, n_owned );
                Kokkos::realloc( u_fmm, n_owned );
                Kokkos::realloc( u_pert, n_owned );
            }
            direct.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                             zparams, u_direct );
            fmm.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                          zparams, u_fmm );

            const auto& diag = fmm.farField().diagnostics();
            const FieldDifference d =
                fieldDifference<ExecSpace>( comm, u_fmm, u_direct, n_owned );

            ClaimAPoint p;
            p.step = step;
            p.scale = fieldScale<ExecSpace>( comm, u_direct, n_owned );
            p.max_abs = d.max_abs;
            p.nonfinite = d.nonfinite;
            p.p2p_fraction = diag.p2p_pair_fraction;
            p.m2l_pairs = diag.global_m2l_pair_count;
            p.m2l_fallback = diag.global_m2l_fallback_pair_count;
            p.particles = diag.global_particle_count;
            // THE STEP-0 STATE HAS NO FIELD. `--initial-potential-strength 0`
            // makes phi and therefore S identically zero, so both velocities
            // are zero and a relative error is 0/0. Compared absolutely
            // instead, and flagged, rather than dropped: 81 states is the
            // claim.
            p.relative = p.scale > kFieldScaleFloor;
            p.rel = p.relative ? p.max_abs / p.scale : p.max_abs;
            return p;
        };

        /// Every assertion claim A makes at one state, in the order R1 needs
        /// them: the round trip, then the far field, then -- only then -- the
        /// accuracy bound. Nothing below the liveness check is evidence about
        /// the expansion unless the liveness check passed.
        auto assertClaimA = [&]( const ClaimAPoint& p )
        {
            // The round trip (R3). A tag mismatch does not crash; it drops or
            // duplicates a source, which is a wrong velocity and not a slow
            // one. An integer reduction, so exact at every rank count.
            BEATNIK_CHECK_EQ( rec, p.particles, kVertices );
            BEATNIK_CHECK_EQ( rec, p.nonfinite, 0 );
            // The far field exists, and -- at level 4 -- dominates. A non-zero
            // fallback count would make the error a mixture of two code paths
            // (R6), which is not the number this claim measures.
            BEATNIK_CHECK_TRUE( rec, p.m2l_pairs > 0 );
            BEATNIK_CHECK_TRUE( rec, p.p2p_fraction < kP2PFractionBound );
            BEATNIK_CHECK_EQ( rec, p.m2l_fallback, 0 );
            // THE BOUND. Both forms, so a failure report names whichever the
            // reader is thinking in.
            if ( p.relative )
            {
                BEATNIK_CHECK_TRUE( rec, p.max_abs <= kTauA * p.scale );
                BEATNIK_CHECK_TRUE( rec, p.rel <= kTauA );
            }
            else
            {
                // A zero-field state: the only honest bound is absolute.
                BEATNIK_CHECK_TRUE( rec, p.max_abs <= kFieldScaleFloor );
            }
        };

        //-------------------------------------------------------------------//
        // STEP 0 IS A COMPARED STEP. `setup()` wrote it unconditionally and the
        // gold depth is steps 0 through 2000 -- 81 files, not 80. It is also
        // the generator gate: a disagreement here is the two icosphere
        // generators differing at this subdivision level, not a divergence
        // measurement, and it must not be read as one.
        //-------------------------------------------------------------------//
        compareStep( 0 );
        {
            const ClaimAPoint p = evaluateClaimA( 0 );
            series.push_back( p );
            assertClaimA( p );
        }

        //-------------------------------------------------------------------//
        // THE DIRECT TRAJECTORY. Driven one step at a time through
        // `advanceOneStep` rather than through `solve()`, so the entity-count
        // check happens at EVERY step and the comparison and the claim-A
        // evaluation at every checkpointed one.
        //
        // `advanceOneStep` is collective and every rank calls it the same
        // number of times -- the BR ring deadlocks otherwise (T2c).
        //-------------------------------------------------------------------//
        const double t_start = MPI_Wtime();
        for ( int step = 1; step <= kSteps; ++step )
        {
            if ( !solver.advanceOneStep() )
            {
                // A STOP IS A REPORTED STOP STEP, NEVER A SHORTER PASS.
                std::ostringstream os;
                os << "claim A run STOPPED EARLY at step " << step << " of "
                   << kSteps << " (non-finite state); solver step "
                   << solver.step() << ", time " << solver.time()
                   << ". This is a reported stop step, not a shorter pass.";
                rec.fail( os.str() );
                claim_a_stopped = true;
                break;
            }
            claim_a_completed = solver.step();
            BEATNIK_CHECK_EQ( rec, solver.step(),
                              static_cast<long long>( step ) );

            // Cheap, integer, and reduced inside Tessera: safe every step.
            if ( mesh.globalVertexCount() != kVertices ||
                 mesh.globalFaceCount() != kFaces )
            {
                std::ostringstream os;
                os << "ENTITY COUNTS CHANGED at claim A step " << step
                   << ": vertices " << mesh.globalVertexCount() << " (expected "
                   << kVertices << "), faces " << mesh.globalFaceCount()
                   << " (expected " << kFaces
                   << "). Adaptivity leaked into the frozen-mesh "
                      "configuration.";
                rec.fail( os.str() );
                break;
            }

            if ( step % kCheckpointEvery != 0 )
                continue;

            if ( !checkOwnedPartition( step, false ) )
                break;

            const double t = static_cast<double>( solver.time() );
            if ( step == kSteps )
            {
                std::ostringstream os;
                os.precision( 17 );
                os << "claim A step " << step << " time " << t << " vs gold "
                   << kFinalTime;
                rec.note( os.str() );
                BEATNIK_CHECK_CLOSE( rec, t, kFinalTime, kTimeRtol );
            }

            checkVolumeDrift( step );
            compareStep( step );

            const ClaimAPoint p = evaluateClaimA( step );
            series.push_back( p );
            assertClaimA( p );
        }
        Kokkos::fence();
        t_claim_a = MPI_Wtime() - t_start;
        claim_a_final_time = static_cast<double>( solver.time() );
        claim_a_last_checkpoint = solver.lastCheckpointPath();

        //-------------------------------------------------------------------//
        // NEGATIVE CASE 2 -- A STATE PERTURBED BY MORE THAN `tau_A` MUST FAIL
        // CLAIM A's COMPARISON.
        //
        // Built on the LAST state, and on the same two views and the same
        // reduction the 81 assertions above ran on, so what it proves live is
        // the comparison those assertions used rather than a copy of it. One
        // component of one owned row on rank 0 is displaced by
        // `kPerturbationFactor * kTauA * scale` -- a factor of two past the
        // bound, so the case cannot be satisfied by round-off and cannot be
        // missed by a bound that is slightly loose.
        //-------------------------------------------------------------------//
        if ( !claim_a_stopped && !series.empty() )
        {
            const ClaimAPoint& last = series.back();
            const int n_owned = mesh.ownedVertexCount();
            Kokkos::deep_copy( u_pert, u_fmm );
            const double delta = kPerturbationFactor * kTauA * last.scale;
            if ( rank == 0 && n_owned > 0 )
            {
                auto view = u_pert;
                const Real d = static_cast<Real>( delta );
                Kokkos::parallel_for(
                    "beatnik_t6_perturb", Kokkos::RangePolicy<ExecSpace>( 0, 1 ),
                    KOKKOS_LAMBDA( const int ) { view( 0, 0 ) += d; } );
                Kokkos::fence();
            }
            const FieldDifference pd =
                fieldDifference<ExecSpace>( comm, u_pert, u_direct, n_owned );
            const double pert_rel =
                last.scale > kFieldScaleFloor ? pd.max_abs / last.scale : 0.0;
            std::ostringstream os;
            os.precision( 17 );
            os << "NEGATIVE CASE 2, claim A on a state perturbed past tau_A: "
                  "unperturbed relative error "
               << last.rel << ", perturbation " << delta
               << " absolute applied to one component of one owned row, "
                  "perturbed relative error "
               << pert_rel << "; tau_A = " << kTauA
               << ". The perturbed value MUST exceed tau_A -- if it does not, "
                  "claim A's bound is not being evaluated.";
            rec.note( os.str() );
            BEATNIK_CHECK_TRUE( rec, pert_rel > kTauA );
        }
        else
        {
            rec.fail( "NEGATIVE CASE 2 not run: claim A's trajectory did not "
                      "complete, so there is no final state to perturb. This "
                      "is a reported gap, not a pass." );
        }

        solver.finalize();

        //-------------------------------------------------------------------//
        // The step budget must have been reached: 2000 is what both gold sets
        // ran, and anything less is a ceiling the next session has to know
        // about.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, claim_a_completed,
                          static_cast<long long>( kSteps ) );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
    }

    //-----------------------------------------------------------------------//
    // THE 81-ENTRY CLAIM-A SERIES, AT 17 DIGITS, IN ONE BLOCK. The exit
    // criterion asks for it in the log so that the next session has the whole
    // curve without re-running an hours-long member -- and so that the one
    // thing T5 could not measure, how the bound behaves out at a deformed
    // sheet, is on the record either way.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( series.size() ),
                      static_cast<long long>( kComparedSteps ) );
    if ( rank == 0 )
    {
        std::printf(
            "[t6] CLAIM A SERIES level=%d space=%s np=%d basis=cartesian-taylor "
            "order=%d ncrit=%d mac_theta=0.3 max_depth=10 softening=0.025 "
            "near_softening_factor=0 tau_A=%.17g\n",
            kSubdivisions, ExecSpace::name(), comm_size, kProductionOrder,
            kNcrit, kTauA );
        std::printf( "[t6] %6s %26s %26s %26s %12s %12s %10s\n", "step",
                     "max|u_fmm - u_direct|", "max|u_direct|",
                     "relative error", "p2p_frac", "m2l_pairs", "fallback" );
        for ( const ClaimAPoint& p : series )
        {
            std::printf( "[t6] %6lld %26.17g %26.17g %26.17g %12.6f %12lld "
                         "%10lld%s\n",
                         p.step, p.max_abs, p.scale, p.rel, p.p2p_fraction,
                         p.m2l_pairs, p.m2l_fallback,
                         p.relative ? "" : "  (ABSOLUTE: zero-field state)" );
        }
        std::fflush( stdout );
    }
    {
        double worst = 0.0;
        long long worst_step = -1;
        double p2p_lo = 1.0;
        double p2p_hi = 0.0;
        for ( const ClaimAPoint& p : series )
        {
            if ( p.relative && p.rel > worst )
            {
                worst = p.rel;
                worst_step = p.step;
            }
            if ( p.p2p_fraction < p2p_lo )
                p2p_lo = p.p2p_fraction;
            if ( p.p2p_fraction > p2p_hi )
                p2p_hi = p.p2p_fraction;
        }
        std::ostringstream os;
        os.precision( 17 );
        os << "CLAIM A: worst relative velocity error " << worst << " at step "
           << worst_step << " over " << series.size() << " states, against "
           << "tau_A = " << kTauA;
        os << " at final time " << claim_a_final_time;
        os.precision( 6 );
        os << "; realized P2P pair fraction " << p2p_lo << " .. " << p2p_hi
           << " (T5 measured " << kP2PFractionReference
           << " on a 5-step state; bound " << kP2PFractionBound << ", far "
           << ( kFarFieldIsLive ? "field LIVE -- this is a far-field bound"
                                : "field a MINORITY at 14.5% of pairs -- this "
                                  "claim is mostly a P2P comparison and the "
                                  "far-field accuracy claim rests on level 4" )
           << ")";
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // NEGATIVE CASE 1 -- claim A's final state against the STEP-0 gold. Same
    // schema, same mesh, same carried scalars, a different time and different
    // positions. It must exit exactly 1 ("compared and disagreed") and NOT 2
    // ("could not load"), because accepting 2 is how a negative case passes
    // vacuously. It also proves 2000 steps MOVED the surface.
    //-----------------------------------------------------------------------//
    if ( rank == 0 && !claim_a_stopped )
    {
        const std::string step0 = goldForStep( gold_dir, 0 );
        if ( !step0.empty() && fileExists( claim_a_last_checkpoint ) )
        {
            const int status = runComparator( python, script,
                                              claim_a_last_checkpoint, step0 );
            std::ostringstream os;
            os << "NEGATIVE CASE 1, claim A's final state vs the step-0 gold: "
                  "exit "
               << status
               << " (1 = detected a mismatch, 2 = LOAD ERROR and therefore a "
                  "vacuous pass)";
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, status, 1 );
        }
        else
        {
            rec.fail( "negative case 1: step-0 gold or claim A output file is "
                      "missing" );
        }
    }

    //=======================================================================//
    // CLAIM B -- 2000 steps FMM-DRIVEN. Stability, then the divergence
    // horizon.
    //
    // Claim A's solver has been destroyed by here, deliberately: two 2000-step
    // solvers alive at once doubles the resident footprint for no reason, and
    // everything claim A's negative case 1 needed was captured as a path.
    //=======================================================================//
    double t_claim_b = 0.0;
    bool claim_b_stopped = false;
    long long claim_b_completed = 0;
    double claim_b_final_time = 0.0;
    double claim_b_final_drift = 0.0;
    double claim_b_worst_deviation = 0.0;
    long long claim_b_worst_step = -1;
    double claim_b_p2p_fraction = 0.0;
    long long claim_b_m2l_pairs = 0;

    {
        solver_type solver( comm, makeParams( claim_b_dir,
                                              Beatnik::BRApproximation::Fmm ) );
        solver.setup();

        auto& mesh = solver.mesh();
        const auto& state = solver.state();

        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );

        const double initial_volume =
            static_cast<double>( solver.initialVolume() );
        BEATNIK_CHECK_CLOSE( rec, initial_volume, kInitialVolume, kScalarRtol );

        // The same two paths to the entity counts the frozen member uses, and
        // for the same reason: Tessera's global counts every step (cheap,
        // integer) and an MPI_Allreduce over OWNED counts at every compared
        // step. Owned-versus-local is what R9 turns on, and under an FMM these
        // are the assertions that fire on develop-canopy's failure mode.
        auto checkOwnedPartition = [&]( long long step )
        {
            long long owned[3] = { mesh.ownedVertexCount(),
                                   mesh.ownedEdgeCount(),
                                   mesh.ownedFaceCount() };
            long long total[3] = { 0, 0, 0 };
            MPI_Allreduce( owned, total, 3, MPI_LONG_LONG, MPI_SUM,
                           mesh.comm() );
            if ( total[0] != kVertices || total[1] != kEdges ||
                 total[2] != kFaces )
            {
                std::ostringstream os;
                os << "ENTITY COUNTS CHANGED at claim B step " << step
                   << ": summed owned V " << total[0] << " E " << total[1]
                   << " F " << total[2] << ", expected " << kVertices << " / "
                   << kEdges << " / " << kFaces
                   << ". The FMM-driven trajectory changed the mesh, which is "
                      "one of the four properties claim B exists to exclude.";
                rec.fail( os.str() );
                return false;
            }
            return true;
        };
        checkOwnedPartition( 0 );

        // The volume drift, against the reference's own series and at the
        // FMM's own rtol. A conserved integral survives decorrelation of the
        // pointwise field, which is what makes it the useful check here -- and
        // it is the one claim-B assertion immune to the pairing problem, since
        // it needs no reference mesh and therefore no vertex correspondence.
        auto checkVolumeDrift = [&]( long long step )
        {
            auto pos = mesh.positions();
            auto owned_faces = Kokkos::subview(
                mesh.faceVertices(), std::make_pair( 0, mesh.ownedFaceCount() ),
                Kokkos::ALL() );
            const Real local =
                Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
            Real volume = 0;
            MPI_Allreduce( &local, &volume, 1, MPI_DOUBLE, MPI_SUM,
                           mesh.comm() );
            const double drift =
                static_cast<double>( volume ) / initial_volume - 1.0;
            const double reference = kRefVolumeDrift[step / kCheckpointEvery];
            const double deviation = reference == 0.0
                                         ? std::fabs( drift )
                                         : std::fabs( drift / reference - 1.0 );
            if ( deviation > claim_b_worst_deviation )
            {
                claim_b_worst_deviation = deviation;
                claim_b_worst_step = step;
            }
            claim_b_final_drift = drift;
            std::ostringstream os;
            os.precision( 17 );
            os << "claim B step " << step << " relative drift " << drift
               << " reference " << reference;
            os.precision( 6 );
            os << " deviation " << deviation << " (rtol "
               << kFmmVolumeDriftRtol << ", abs cap " << kVolumeDriftAbsCap
               << ")";
            rec.note( os.str() );
            BEATNIK_CHECK_TRUE( rec, deviation <= kFmmVolumeDriftRtol );
            BEATNIK_CHECK_TRUE( rec, std::fabs( drift ) <= kVolumeDriftAbsCap );
        };

        checkVolumeDrift( 0 );

        //-------------------------------------------------------------------//
        // THE FMM-DRIVEN TRAJECTORY. `advanceOneStep` returns false on a
        // non-finite state, and that return is claim B's finiteness assertion:
        // every velocity of every stage passed through `SurfaceState::allFinite`
        // to get here, and the abort is a GLOBAL decision so no rank can leave
        // the loop while its peers block in the next collective.
        //-------------------------------------------------------------------//
        const double t_start = MPI_Wtime();
        for ( int step = 1; step <= kSteps; ++step )
        {
            if ( !solver.advanceOneStep() )
            {
                // THIS IS THE ASSERTION R7 IS ABOUT. develop-canopy's FMM
                // tracked the exact solver acceptably for 1362 steps and then
                // a single corrupted node seeded a runaway to whole-field NaN,
                // while the direct solver completed the identical deck. A stop
                // is a REPORTED STOP STEP, never a shorter pass.
                std::ostringstream os;
                os << "claim B FMM-DRIVEN run STOPPED EARLY at step " << step
                   << " of " << kSteps << " (non-finite state); solver step "
                   << solver.step() << ", time " << solver.time()
                   << ". This is a reported stop step, not a shorter pass, and "
                      "it is the failure mode claim B exists to catch.";
                rec.fail( os.str() );
                claim_b_stopped = true;
                break;
            }
            claim_b_completed = solver.step();

            if ( mesh.globalVertexCount() != kVertices ||
                 mesh.globalFaceCount() != kFaces )
            {
                std::ostringstream os;
                os << "ENTITY COUNTS CHANGED at claim B step " << step
                   << ": vertices " << mesh.globalVertexCount() << " (expected "
                   << kVertices << "), faces " << mesh.globalFaceCount()
                   << " (expected " << kFaces << ").";
                rec.fail( os.str() );
                break;
            }

            if ( step % kCheckpointEvery != 0 )
                continue;
            if ( !checkOwnedPartition( step ) )
                break;
            checkVolumeDrift( step );
        }
        Kokkos::fence();
        t_claim_b = MPI_Wtime() - t_start;
        claim_b_final_time = static_cast<double>( solver.time() );

        //-------------------------------------------------------------------//
        // THE REALIZED P2P PAIR FRACTION OF CLAIM B's OWN TRAJECTORY, measured
        // on its FINAL state.
        //
        // `Solver` does not expose its BR solver -- `BRSolverBase` is
        // Canopy-agnostic and the conventions table confines Canopy types to
        // the adapter -- so the fraction the driven run realized is not
        // readable from here. One standalone evaluation at the identical
        // `FmmParams`, on the state that run ended at, is: it is the most
        // deformed state in this member and therefore the one where the tree
        // is least like the 5-step state T5 measured. Reported and not
        // asserted equal to anything; the bound below is the same one claim A
        // asserts at every state.
        //-------------------------------------------------------------------//
        if ( !claim_b_stopped )
        {
            const int n_owned = mesh.ownedVertexCount();
            mesh.haloExchange();
            geometry_type geometry;
            geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                              mesh.faceVertices() );
            state.updateSheetVector( mesh, geometry );
            auto quadrature =
                Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
                    Beatnik::SourceQuadrature::Vertex );
            fmm_type probe( comm, makeFmmParams() );
            vector_view u_probe( "beatnik_t6_u_probe", n_owned );
            const Beatnik::ZModelParams zparams =
                makeParams( claim_b_dir, Beatnik::BRApproximation::Fmm ).zmodel;
            probe.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                            zparams, u_probe );
            const auto& diag = probe.farField().diagnostics();
            claim_b_p2p_fraction = diag.p2p_pair_fraction;
            claim_b_m2l_pairs = diag.global_m2l_pair_count;
            std::ostringstream os;
            os.precision( 6 );
            os << "claim B realized far field on its OWN final state (step "
               << claim_b_completed << "): p2p fraction "
               << claim_b_p2p_fraction << " (bound " << kP2PFractionBound
               << ", T5's 5-step value " << kP2PFractionReference
               << "), m2l pairs " << claim_b_m2l_pairs << ", m2l fallback "
               << diag.global_m2l_fallback_pair_count << ", particles "
               << diag.global_particle_count << ", softening "
               << static_cast<double>( diag.softening );
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, diag.global_particle_count, kVertices );
            BEATNIK_CHECK_TRUE( rec, claim_b_m2l_pairs > 0 );
            BEATNIK_CHECK_TRUE( rec,
                                claim_b_p2p_fraction < kP2PFractionBound );
            BEATNIK_CHECK_EQ( rec, diag.global_m2l_fallback_pair_count, 0LL );
        }

        solver.finalize();

        BEATNIK_CHECK_EQ( rec, claim_b_completed,
                          static_cast<long long>( kSteps ) );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );

        {
            std::ostringstream os;
            os.precision( 17 );
            os << "CLAIM B: reached step " << claim_b_completed << " at time "
               << claim_b_final_time
               << " (NOT compared against the direct trajectory's "
               << kFinalTime
               << ": under --adaptive-dt the timestep is a function of the "
                  "state, so an FMM-driven run is at a different physical time "
                  "at the same step, and asserting it here would report the dt "
                  "rather than the trajectory); final volume drift "
               << claim_b_final_drift << ", worst deviation from the reference "
                  "series ";
            os.precision( 6 );
            os << claim_b_worst_deviation << " at step " << claim_b_worst_step
               << " (rtol " << kFmmVolumeDriftRtol << ")";
            rec.note( os.str() );
        }
    }

    //-----------------------------------------------------------------------//
    // CLAIM B's DIVERGENCE HORIZON.
    //
    // Rank 0 only, and the assertion is broadcast so a single verdict reaches
    // every rank's tally -- the Python runs once, but the failure must not be
    // rank 0's alone.
    //-----------------------------------------------------------------------//
    int horizon_rc = 0;
    if ( rank == 0 && !claim_b_stopped )
    {
        std::vector<LadderRung> rungs;
        std::vector<long long> unpairable;
        std::ostringstream label;
        label << "claimB sub" << kSubdivisions << " " << ExecSpace::name()
              << " np" << comm_size << " fmm vs gold";

        if ( !runLadder( rec, python, ladder_script, claim_b_dir, gold_dir,
                         ladder_json, label.str(), rungs, unpairable ) )
        {
            horizon_rc = 1;
        }
        else
        {
            //---------------------------------------------------------------//
            // The per-rung table the exit criterion asks for, naming every
            // step the pairing refused. A refused step is neither a pass nor a
            // failure: at level 4, steps 1350-1900 are unpairable by any
            // position-based scheme, identically across four independent runs,
            // and the horizon is decided a thousand steps earlier.
            //---------------------------------------------------------------//
            std::ostringstream us;
            us << "claim B UNPAIRABLE steps (" << unpairable.size()
               << ", neither a pass nor a failure; the tool refuses them "
                  "rather than mis-pairing):";
            if ( unpairable.empty() )
                us << " none";
            for ( long long s : unpairable )
                us << " " << s;
            rec.note( us.str() );

            std::printf( "[t6] CLAIM B HORIZON level=%d space=%s np=%d\n",
                         kSubdivisions, ExecSpace::name(), comm_size );
            std::printf( "[t6] %10s %10s %20s %20s %10s\n", "rtol", "atol",
                         "first failing step", "envelope (T5)", "verdict" );

            for ( int i = 0; i < kRungCount; ++i )
            {
                const LadderRung& r = rungs[i];
                // The rung the JSON carries must be the rung this file's
                // envelope was measured at. A rung reordered or retuned on the
                // Python side would otherwise shift every column silently.
                BEATNIK_CHECK_CLOSE( rec, r.rtol, kRungRtol[i], 1.0e-12 );
                BEATNIK_CHECK_CLOSE( rec, r.atol, kRungAtol[i], 1.0e-12 );

                const bool ok = horizonNoEarlierThan(
                    r.null_step, r.first_failing_step, kHorizonEnvelope[i] );
                std::printf( "[t6] %10.0e %10.0e %20s %20lld %10s%s\n", r.rtol,
                             r.atol,
                             r.null_step
                                 ? "None"
                                 : std::to_string( r.first_failing_step )
                                       .c_str(),
                             kHorizonEnvelope[i], ok ? "ok" : "EARLY",
                             i == kLoadBearingRung ? "  <- load-bearing" : "" );
                if ( !ok )
                {
                    std::ostringstream os;
                    os << "claim B HORIZON EARLIER THAN THE ENVELOPE at the "
                       << r.rtol << "/" << r.atol << " rung: first failing "
                       << "checkpointed step " << r.first_failing_step
                       << ", envelope " << kHorizonEnvelope[i]
                       << ". The FMM-driven trajectory decorrelates sooner "
                          "than T5 measured, which is a regression in the far "
                          "field and not a tolerance to widen.";
                    rec.fail( os.str() );
                    horizon_rc = 1;
                }
            }
            std::fflush( stdout );

            // The four tight rungs cannot fail except at step 0, because their
            // envelope IS the first checkpoint. Saying so in the log is what
            // stops a later reader from counting five live assertions where
            // there is one.
            {
                std::ostringstream os;
                os << "claim B: of " << kRungCount << " rungs, "
                   << kRungCount - 1
                   << " have an envelope at the first checkpoint (step "
                   << kCheckpointEvery
                   << ") and therefore cannot fail except at step 0. THE "
                      "LOAD-BEARING RUNG IS "
                   << kRungRtol[kLoadBearingRung] << "/"
                   << kRungAtol[kLoadBearingRung] << ", envelope "
                   << kHorizonEnvelope[kLoadBearingRung]
                   << ". No rung is asserted to PASS at step 2000: at tau_A "
                      "none will, and one that did would be loose enough to "
                      "be meaningless.";
                rec.note( os.str() );
            }

            //---------------------------------------------------------------//
            // NEGATIVE CASE 3 -- A FABRICATED HORIZON ONE CHECKPOINT EARLIER
            // THAN THE ENVELOPE MUST FAIL.
            //
            // Through `horizonNoEarlierThan`, the same predicate the assertion
            // above runs, so what this proves live is that assertion and not a
            // copy of it. Built on the `1e-4` rung: at the four tighter rungs
            // the envelope is the first checkpoint and nothing but step 0 is
            // earlier, so a negative case there would prove only that the
            // comparison runs.
            //---------------------------------------------------------------//
            {
                const long long envelope = kHorizonEnvelope[kLoadBearingRung];
                const long long fabricated = envelope - kCheckpointEvery;
                const bool accepted =
                    horizonNoEarlierThan( false, fabricated, envelope );
                std::ostringstream os;
                os << "NEGATIVE CASE 3, a fabricated horizon on the "
                   << kRungRtol[kLoadBearingRung] << "/"
                   << kRungAtol[kLoadBearingRung] << " rung: step "
                   << fabricated << " against envelope " << envelope
                   << " was " << ( accepted ? "ACCEPTED" : "rejected" )
                   << ". It MUST be rejected -- an accepted one would mean the "
                      "horizon assertion is trivially satisfied.";
                rec.note( os.str() );
                BEATNIK_CHECK_TRUE( rec, !accepted );
                // And the real measurement must not be the fabricated one,
                // which is what stops this case from passing on a rung whose
                // envelope happens to be zero.
                BEATNIK_CHECK_TRUE( rec, fabricated < envelope );
            }
        }
    }
    else if ( rank == 0 )
    {
        rec.fail( "claim B's horizon was NOT measured: the FMM-driven "
                  "trajectory stopped early, so the checkpoint series is "
                  "truncated and a ladder over it would report a horizon for "
                  "a run that did not happen." );
        horizon_rc = 1;
    }
    // ONE VERDICT ACROSS THE RANKS for a rank-0-only measurement. Without this
    // the tier's per-rank tallies disagree and a horizon failure is invisible
    // in every log line but one.
    {
        int global = horizon_rc;
        MPI_Allreduce( &horizon_rc, &global, 1, MPI_INT, MPI_MAX, comm );
        if ( global != 0 && rank != 0 )
            rec.fail( "claim B's horizon assertion failed on rank 0; see that "
                      "rank's detail lines" );
    }

    //-----------------------------------------------------------------------//
    // COST. What tells the next session whether this tier still fits its
    // walltime -- and R9 is precisely the failure mode where a truncated run
    // reads as a shorter pass, so the split between the two claims is what a
    // later `-t` is set from. GPU-side memory is OUT OF SCOPE.
    //-----------------------------------------------------------------------//
    {
        struct rusage ru;
        long peak_kb = 0;
        if ( ::getrusage( RUSAGE_SELF, &ru ) == 0 )
            peak_kb = ru.ru_maxrss; // kB on Linux
        long peak_max = peak_kb;
        MPI_Allreduce( &peak_kb, &peak_max, 1, MPI_LONG, MPI_MAX, comm );
        double comparator_max = g_comparator_seconds;
        MPI_Allreduce( &g_comparator_seconds, &comparator_max, 1, MPI_DOUBLE,
                       MPI_MAX, comm );
        double ladder_max = g_ladder_seconds;
        MPI_Allreduce( &g_ladder_seconds, &ladder_max, 1, MPI_DOUBLE, MPI_MAX,
                       comm );

        std::ostringstream os;
        os.precision( 6 );
        os << "COST: claim A (direct trajectory + 81 FMM evaluations + "
           << g_comparator_calls << " comparator invocations) " << t_claim_a
           << " s";
        if ( claim_a_completed > 0 )
            os << " (" << ( t_claim_a / double( claim_a_completed ) )
               << " s/step)";
        os << "; claim B (FMM-driven trajectory) " << t_claim_b << " s";
        if ( claim_b_completed > 0 )
            os << " (" << ( t_claim_b / double( claim_b_completed ) )
               << " s/step)";
        os << "; of claim A's time, comparator " << comparator_max
           << " s; ladder " << ladder_max << " s; peak RSS this rank "
           << peak_kb << " kB, worst rank " << peak_max << " kB";
        rec.note( os.str() );

        if ( rank == 0 )
        {
            // One machine-greppable line per launch, so a tier log reduces to
            // a table without parsing the prose above.
            std::printf( "[t6] COST level=%d space=%s np=%d steps=%lld "
                         "claimA=%.6f claimB=%.6f comparator=%.6f "
                         "ladder=%.6f peak_rss_kb=%ld\n",
                         kSubdivisions, ExecSpace::name(), comm_size,
                         claim_b_completed, t_claim_a, t_claim_b,
                         comparator_max, ladder_max, peak_max );
            std::fflush( stdout );
        }
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_Milestone0Fmm" );
        try
        {
            // BEATNIK_TEST_EXEC_SPACE is defined by the per-backend shim
            // tests/CMakeLists.txt generates, so the target name's `_SERIAL` /
            // `_HIP` suffix means what the runner's filter assumes it means.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec, argc,
                                                                    argv );
        }
        catch ( const std::exception& e )
        {
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();

    // ONE VERDICT ACROSS THE RANKS. Every rank printed its own tally above, so
    // the log names which rank failed; MPI_MAX then makes any rank's failure
    // the job's failure.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
