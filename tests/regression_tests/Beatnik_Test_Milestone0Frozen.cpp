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
 * @file Beatnik_Test_Milestone0Frozen.cpp
 * @brief **THE MILESTONE-0 TEST** — 2000 TVD-RK3 timesteps of the reference's
 *        default physics on a frozen mesh, compared against the M0-G1 / M0-G2
 *        Python gold set at every one of the 81 checkpointed steps.
 *
 * THIS IS THE `milestone` TIER'S FIRST MEMBER, AND IT IS **NOT** THE SHIP GATE.
 * ---------------------------------------------------------------------------
 * The gate stays at five `regression` members and 60 launches (CLAUDE.md
 * "Minimum test set"); this tier is ranks **1 and 4** on SERIAL and HIP, run on
 * demand through `scripts/<system>/run_milestone.<scheduler>`. A 2000-step run
 * at 642 or 2562 vertices in front of every change is a stall, not a gate.
 *
 * TWO MEMBERS, ONE BODY. This file is the whole test at
 * `--icosphere-subdivisions BEATNIK_M0_LEVEL`, which defaults to **3** — M0-A1's
 * primary member. The second member is
 * `Beatnik_Test_Milestone0FrozenL4.cpp`: three lines that `#define
 * BEATNIK_M0_LEVEL 4` and `#include` this file. Two source stems rather than one
 * parameterized target, because the milestone tier keys its argument lists by
 * source stem (`tests/CMakeLists.txt`) and each member needs its OWN gold
 * directory — one stem with two argument lists would have to teach that loop to
 * carry more than one, and an argument list that names the wrong level's gold
 * set is exactly the mistake the `FATAL_ERROR` guard exists to make impossible.
 * Every per-level literal below is therefore selected by `BEATNIK_M0_LEVEL` and
 * **re-derived for that level**, never transferred from another: the entity
 * counts, the two carried scalars, the polyhedral deficit and the reference
 * volume-drift series all differ between the levels, and T2d's are level-2
 * numbers.
 *
 * WHAT IT COMPARES, AND WHY THAT IS THE WHOLE GOLD SET
 * ---------------------------------------------------
 * All **81** checkpointed steps — step 0, 25, 50, …, 2000 — at
 * `--rtol 1e-10 --atol 1e-12`, **flat**: the same tolerance at every step and at
 * both levels. That is M0-A1's decision and it is a measurement, not a choice.
 * M0-D1 built the tolerance ladder from the comparator's own printed
 * `max|e|` / `max|e|/|g|` bounds over twelve comparisons — two levels x two
 * backends x two rank counts against Python, plus the four rank-1-vs-rank-4
 * pairs — and at this rung **no checkpointed step through 2000 even permits a
 * failure**, so the bound is proved rather than sampled. The rung below,
 * `1e-12/1e-14`, is where the ladder has content: it stops at step 1325 (level
 * 3) or 775 (level 4), which is why it is not what is asserted. See the headroom
 * numbers beside `kRtol`.
 *
 * Because the depth is the ENTIRE gold set, there is no beyond-depth regime:
 * every checkpointed step is compared field-by-field, so nothing is left over to
 * compare structurally or statistically (M0-A1 decision 3).
 *
 * THE GOLD `.npz` CARRIES NO `sheet_vector`. Nine keys, and that is not one of
 * them, so the compared field set is exactly `vertices`, `potential`,
 * `remesh_material_position`, `faces`, `time`, `initial_volume`,
 * `initial_min_edge`. M0-D1 found this the hard way — a Beatnik-vs-Beatnik
 * comparison is over a strictly LARGER field set than a Beatnik-vs-Python one,
 * and reading the two horizons as comparable is what almost made M0-T2 look
 * mandatory. A test here cannot assert on `sheet_vector` however much it would
 * like to.
 *
 * WHY A FROZEN MESH MAKES THIS COMPARISON WELL-BEHAVED
 * ---------------------------------------------------
 * `--no-dynamic-remesh --refine-every 0` removes three of milestone 1's four
 * divergence mechanisms outright (no edit set, no diagonal choice, no greedy
 * cap), leaving cross-rank summation order as the only one. And
 * `compare_output.py` is structural before it is numeric: it fails outright on a
 * differing vertex or face count and requires the canonicalized face lists to be
 * equal. With connectivity frozen the counts are the generator's for the whole
 * run and the face list never changes, so the structural comparison cannot fail
 * at any step and a disagreement is always a smooth, graded field error.
 *
 * That premise is a claim until something checks it, which is what the per-step
 * entity-count assertion below is for — and it is checked TWICE, on purpose: as
 * Tessera's own global counts every step (cheap, integer), and as an
 * `MPI_Allreduce` over OWNED counts at every compared step, which is a second
 * independent path to the same number and is what R9 turns on. A build with
 * remeshing forced on fails it; see `BEATNIK_M0_FORCE_DYNAMIC_REMESH`.
 *
 * VOLUME DRIFT — RE-DERIVED FOR THIS CONFIGURATION, NOT INHERITED FROM T2d
 * -----------------------------------------------------------------------
 * `removeVolumeFlux` makes the *rate* of volume change zero in the discrete
 * sense; the accumulated RK3 truncation still grows, and no `projectToVolume`
 * runs on this path (every call site in the reference sits inside a refine or
 * remesh branch this configuration switches off). So the drift is real, it is
 * the reference's own, and the criterion is **agreement with the reference's
 * measured series** — a strictly stronger statement than smallness, because it
 * fails if Beatnik conserves volume better than the Python as well as worse.
 * T2d's `kGoldVolumeDrift` is a level-2 ten-step series and is not imported; its
 * `kVolumeDriftAbsCap = 1e-9` is not imported either, because the reference's own
 * drift reaches `3.35e-09` at level 3 and `4.74e-09` at level 4 and would fail
 * that cap for the right reason at the wrong scale (risk M0-R3). The relative
 * tolerance `kVolumeDriftRtol = 1e-3` IS reused, and it survived re-derivation
 * with a 36x margin — see its comment.
 *
 * THE R9 DISCRIMINATORS, RE-DERIVED PER LEVEL
 * -------------------------------------------
 * The three structural discriminators T1c established and T2d mechanized are
 * mechanized here too, because they are statements about the PARTITION rather
 * than about the trajectory and stay decisive after 2000 steps:
 *
 *   1. the owned sets partition the global sets, summed with a plain
 *      `MPI_Allreduce` over `ownedXCount()` rather than read from Tessera;
 *   2. `volume / (4 pi R^3 / 3)` at step 0 — the polyhedral deficit of THIS
 *      triangulation. T2d's `0.96616074859858714` is the subdivision-2 value and
 *      is wrong at every other level; the level-3 and level-4 values below were
 *      re-derived from each gold set's own step-0 file.
 *   3. the entity counts never change.
 *
 * WALL TIME AND PEAK MEMORY ARE REPORTED, because they are what tells the next
 * session whether a deeper depth or a finer level is affordable. Wall time comes
 * from `MPI_Wtime` around the step loop (the comparator's own cost is clocked and
 * reported separately, since 81 Python invocations are a real share of a level-3
 * launch); peak resident memory from `getrusage(RUSAGE_SELF).ru_maxrss`, reduced
 * with `MPI_MAX` so the reported number is the worst rank's. **GPU-side memory
 * is out of scope** — there is no mechanism for it here, and saying so is what
 * stops a later session reading its absence as an oversight.
 *
 * FAILURE BEHAVIOR IS LOUD (milestone0.md Conventions). A gold file missing for a
 * compared step is a named failure, not a skipped step. A run that stops early is
 * a reported stop step, not a shorter pass. A comparator exit of 2 (could not
 * load) is never conflated with 1 (compared and disagreed).
 *
 * ARGUMENTS. Both paths; see tests/CMakeLists.txt for the two call sites, which
 * pass them absolute (ctest) and manifest-relative (the installed runner).
 *
 *   argv[1]  the gold DIRECTORY for THIS LEVEL (81 .npz, steps 0-2000 by 25)
 *              regression_tests/milestone0-sub3-2000-steps/gold   (level 3)
 *              regression_tests/milestone0-sub4-2000-steps/gold   (level 4)
 *   argv[2]  the comparator
 *              regression_tests/compare_output.py
 *
 * `BEATNIK_PYTHON` overrides the interpreter (default `python3`). There is no
 * option surface here and none may be added (milestone0.md Conventions, "CLI
 * surface: unchanged").
 */

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

//---------------------------------------------------------------------------//
// The subdivision level, and the ONLY thing that differs between this file and
// Beatnik_Test_Milestone0FrozenL4.cpp. Defaulting it here keeps the file
// compilable on its own and makes the primary member the one you get by
// accident, not the one you have to remember to ask for.
//---------------------------------------------------------------------------//
#ifndef BEATNIK_M0_LEVEL
#define BEATNIK_M0_LEVEL 3
#endif

#if BEATNIK_M0_LEVEL != 3 && BEATNIK_M0_LEVEL != 4
#error "BEATNIK_M0_LEVEL must be 3 or 4: no other gold set exists (M0-G1, M0-G2)"
#endif

/// **THE FAILURE-DIRECTION BUILD, off by default.** Defining this to 1 forces
/// `--dynamic-remesh` on with a sizing field tight enough to mark splits, at
/// `--remesh-every 4`, so the entity counts change at step 4 and the frozen-mesh
/// assertion below fails there — the demonstration M0-T3's exit criterion
/// requires, that this test detects a different mesh rather than passing against
/// one. The knobs are T4b's accepted split-only set (`collapse-factor 0`,
/// `remesh-smooth-iters 0`, `remesh-flip-min-gain 1e12`,
/// `--no-isotropic-cleanup`), because a configuration
/// `requireSupportedConfiguration` rejects would throw at setup and demonstrate
/// nothing about the count assertion. It is a BUILD-TIME define and not an
/// option: milestone0.md's conventions close the CLI surface.
#ifndef BEATNIK_M0_FORCE_DYNAMIC_REMESH
#define BEATNIK_M0_FORCE_DYNAMIC_REMESH 0
#endif

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The M0-G1 / M0-G2 configuration, verbatim from each gold/README.md:
//
//   python examples/run_adaptive_mesh_bubble.py \
//     --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
//     --viscosity-mode laplace-beltrami --br-approximation direct \
//     --adaptive-dt --no-dynamic-remesh --refine-every 0 \
//     --source-quadrature vertex \
//     --icosphere-subdivisions <L> --steps 2000 \
//     --checkpoint-every-steps 25 --no-video --checkpoint-dir results<L>
//
// `makeParams()` below is `Beatnik_Test_Milestone0Run.cpp`'s field for field —
// M0-D1's driver, which has already been exercised for 2000 steps at both levels
// and at both rank counts on both backends — with the level fixed at compile
// time instead of read from argv. It is NOT re-derived from the Python command
// line here; re-deriving a params set that has already been measured against is
// how a test stops comparing what was measured.
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = BEATNIK_M0_LEVEL;
constexpr int kSteps = 2000;
constexpr int kCheckpointEvery = 25;
/// Steps 0, 25, …, 2000 — the entire gold set, and M0-A1's decided depth.
constexpr int kComparedSteps = kSteps / kCheckpointEvery + 1;

constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

//---------------------------------------------------------------------------//
// Tolerances. ONE PAIR OF LITERALS, NOT A TABLE: M0-A1's ladder is flat.
//---------------------------------------------------------------------------//

/// M0-A1's decided rung, and the comparator's own default. **Do not loosen**
/// without a new M0-A entry and the measurement behind it (M0-R1, M0-R8).
///
/// The headroom is about two decades and it is measured, not assumed. Peak
/// `vertices` `max|e|` over the 81 steps, from M0-D1's full-precision growth
/// series (`milestone0_ladder.py growth`, sweep job `f3TT4psJ8it7`):
///
///   level 3 SERIAL   8.53317416726895317e-13   at step 2000
///   level 4 SERIAL   3.17634807345257286e-13   at step 1400 — NOT at 2000,
///                    where it has fallen back to 1.31783473023006081e-13
///
/// The growth is power-law-like rather than exponential (one ulp at step 0 to
/// `8.5e-13` at step 2000) and at level 4 it is not even monotone, which is why
/// the peak and not the final value is the number to compare a tolerance
/// against.
constexpr const char* kRtol = "1e-10";
constexpr const char* kAtol = "1e-12";

/// The same relative tolerance, for this test's own `time` check.
constexpr double kTimeRtol = 1.0e-10;

/// For the two carried scalars, at the tolerance regression test 1 pins them at.
/// M0-D1 step 1 measured step-0 agreement at `1e-12` at levels 2, 3 AND 4, with
/// the same single ulp of absolute error at every level — so M0-R5 did not fire
/// and no per-level step-0 tolerance is needed.
constexpr double kScalarRtol = 1.0e-12;

/// How closely Beatnik's per-step volume drift must track the reference's.
///
/// **Reused from T2d unchanged, and re-derived rather than assumed:** M0-D1 step
/// 5 measured Beatnik's drift against the reference's at every one of the 81
/// steps of all eight 2000-step runs, worst case `2.758331e-05` relative
/// (`sub3_Serial_np4` at step 1975; the level-4 worst is `1.831088e-05`). That is
/// a **36x** margin under this literal. Do not loosen it without a new
/// measurement in `tasks/milestone0-progress-log.md`.
constexpr double kVolumeDriftRtol = 1.0e-3;

/// The blow-up detector, kept absolute so a drift that tracks the reference
/// *proportionally* while both explode still fails.
///
/// **RE-DERIVED FOR 2000 STEPS — T2d's `1e-9` must not be reused** (risk M0-R3).
/// No `projectToVolume` runs on this path, so the drift accumulates, and the
/// reference's own reaches `3.35289418451623078e-09` at level 3 and
/// `4.74141392814431128e-09` at level 4 — both ABOVE T2d's cap, which would
/// therefore fail for the right reason at the wrong scale. This sits a little
/// over twice the larger of the two, which is the same "a decade of room over the
/// measured end point" shape T2d's cap had at its own depth.
constexpr double kVolumeDriftAbsCap = 1.0e-8;

//---------------------------------------------------------------------------//
// PER-LEVEL REFERENCE NUMBERS. Every one is re-derived for its own level; none
// is transferred from another level or from T2d.
//
// Provenance, identical for both blocks: computed from the committed gold `.npz`
// of that level by `tests/regression_tests/milestone0_ladder.py series`, whose
// `enclosed_volume` is T2d's and Beatnik's convention
// (`V = (1/6) sum_f a.(b x c)` over `faces`, drift relative to the file's own
// `initial_volume`). M0-D1 validated that tool against M0-G1's independently
// measured table, and these 17-digit literals reproduce the 7-digit
// `V/V0 - 1` column of each `gold/README.md` at every one of the 81 steps.
//
// `kRefVolumeDrift[i]` is step `25*i`; the row comments carry the first step of
// each row so an index can be checked by eye.
//---------------------------------------------------------------------------//
#if BEATNIK_M0_LEVEL == 3

/// `V = 10*4^3 + 2`, `E = 30*4^3`, `F = 20*4^3`. Constant for the whole run.
constexpr long long kVertices = 642;
constexpr long long kEdges = 1920;
constexpr long long kFaces = 1280;

/// The level-3 gold set's own two carried scalars (its `_step0000000.npz`).
/// `initial_min_edge` is what every adaptive dt of the run is relative to, and
/// is why the two levels do NOT reach the same physical time after 2000 steps.
constexpr double kInitialVolume = 6.48865752670790275e-02;
constexpr double kInitialMinEdge = 3.45707933867918649e-02;

/// R9 discriminator 2 — the polyhedral deficit of the subdivision-3 icosphere,
/// `initial_volume / (4 pi R^3 / 3)` at `R = 0.25`. **T2d's
/// `0.96616074859858714` is the subdivision-2 value and does not apply here**;
/// the deficit shrinks as the triangulation refines, which is the whole reason
/// this literal is per level.
constexpr double kVolumeOverSphere = 9.91393842629754940e-01;

/// `time` at step 2000, from the gold set's own `/time` scalar. The comparator
/// checks `time` at every compared step; this literal is here so the END of the
/// run names itself in the log at 17 digits.
constexpr double kFinalTime = 1.99828394714319368e+00;

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

#else // BEATNIK_M0_LEVEL == 4

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

/// `time` at step 2000. **Not level 3's** — the adaptive dt is relative to each
/// run's own `initial_min_edge`, so the two levels reach different physical times
/// after the same number of steps, and every cross-level statement about this run
/// is by STEP and never by time.
constexpr double kFinalTime = 1.96430414465685987e+00;

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

#endif // BEATNIK_M0_LEVEL

//---------------------------------------------------------------------------//
bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// The gold file for `step`, found by its `_step%07d.npz` suffix rather than by
/// rebuilding the name from a time — the time is exactly what is under test, and
/// a name built from Beatnik's own `time` would compare each step against
/// whichever gold file Beatnik's dt happened to point at. Empty if the directory
/// holds no such file, which the caller reports as a named failure.
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

/// Wall seconds spent inside `runComparator`, accumulated on rank 0. Reported at
/// the end beside the solve time: 81 Python invocations are a real share of a
/// level-3 launch and a later session sizing the tier's walltime needs the split.
double g_comparator_seconds = 0.0;
long long g_comparator_calls = 0;

/// Run `python <script> <a> <b> --rtol .. --atol ..` and return its exit status,
/// or -1 if it could not be run at all. The three outcomes are never conflated:
/// 0 match, 1 compared and disagreed, 2 could not load, -1 plumbing.
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
/// The milestone-0 command line, as a `SolverParams`. Identical field for field
/// to `Beatnik_Test_Milestone0Run.cpp::makeParams( BEATNIK_M0_LEVEL, 2000, 25,
/// dir )`, which M0-D1 measured with.
Beatnik::SolverParams makeParams( const std::string& checkpoint_dir )
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
    // --br-approximation direct. Both gold sets are direct runs; `fmm` would
    // add an approximation error this comparison cannot separate from the
    // round-off divergence it measures.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

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

    // --no-dynamic-remesh --refine-every 0. THE WHOLE POINT of milestone 0:
    // connectivity is frozen for the entire run.
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    // Neither of the other two post-step passes is configured either.
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;

    // --isotropic-cleanup is on by default and is moot with remeshing off: its
    // rejection fires only under `refining || remeshing`, and both are false.
    p.cleanup.enabled = true;

    // --checkpoint-every-steps 25. `setup()` writes step 0 unconditionally, so
    // 2000 steps every 25 gives the gold sets' own 81 files.
    p.checkpoint.every_steps = kCheckpointEvery;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

#if BEATNIK_M0_FORCE_DYNAMIC_REMESH
    // THE FAILURE-DIRECTION BUILD. Everything from here down exists to make the
    // mesh change, and nothing here is reachable in a default build. The sizing
    // field is deliberately far tighter than T4b's (`h_max 0.06`, which at 642 or
    // 2562 vertices marks nothing) so splits are certain at both levels, and the
    // three unimplemented remesh thirds are configured off through the
    // REFERENCE's own knobs, which is what makes the configuration one
    // `requireSupportedConfiguration` accepts rather than one it throws on.
    p.dynamic_remesh = true;
    p.remesh_every = 4;
    p.remesh_tight_after = -1.0;
    p.remesh.sagitta_tolerance = 2.0e-3;
    p.remesh.h_max = 1.0e-3;
    p.remesh.h_min = 1.0e-4;
    p.remesh.split_factor = 1.35;
    p.remesh.max_splits_per_pass = 300;
    p.remesh.passes = 1;
    p.remesh.target_gradation_factor = 1.35;
    p.remesh.target_gradation_iterations = 8;
    p.remesh.min_quality = 0.18;
    p.remesh.collapse_factor = 0.0;
    p.remesh.max_collapses_per_pass = 0;
    p.remesh.smoothing_iterations = 0;
    p.remesh.smoothing_relaxation = 0.04;
    p.remesh.flip_min_gain = Beatnik::kFlipsDisabledMinGain;
    p.remesh.use_proximity = false;
    p.remesh.surgical_proximity = false;
    p.cleanup.enabled = false;
#endif

    return p;
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;

    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    {
        std::ostringstream os;
        os << "execution space " << ExecSpace::name() << ", ranks " << comm_size
           << ", icosphere subdivisions " << kSubdivisions << ", steps "
           << kSteps << ", comparing " << kComparedSteps
           << " checkpointed steps at rtol " << kRtol << " atol " << kAtol;
#if BEATNIK_M0_FORCE_DYNAMIC_REMESH
        os << "  *** BEATNIK_M0_FORCE_DYNAMIC_REMESH BUILD: this build is "
              "EXPECTED TO FAIL at step 4 on the entity-count assertion ***";
#endif
        rec.note( os.str() );
    }

    if ( argc < 3 )
    {
        rec.fail( "usage: <gold-dir> <compare_output.py>; see the ARGUMENTS "
                  "block in this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const std::string gold_dir = argv[1];
    const std::string script = argv[2];
    const char* python_env = std::getenv( "BEATNIK_PYTHON" );
    const std::string python = python_env ? python_env : "python3";

    // Every input path is checked BEFORE it is used, so a mis-plumbed path is
    // reported as itself rather than as a comparison failure (the T1b trap) --
    // and all 81 gold files are checked up front rather than at the step that
    // needs them, because discovering a missing step-1975 file after 20 minutes
    // of solving wastes the run.
    if ( rank == 0 )
    {
        BEATNIK_CHECK_TRUE( rec, fileExists( gold_dir ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( script ) );
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
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/beatnik_milestone0_sub"
        << kSubdivisions << "/" << ExecSpace::name() << "_np" << comm_size;
    rec.note( "checkpoint directory " + dir.str() );

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD,
                                                 makeParams( dir.str() ) );
    solver.setup();

    auto& mesh = solver.mesh();

    //-----------------------------------------------------------------------//
    // Structure, before anything evolves. Reduced as integers, so exact at every
    // rank count.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    // Risk R8: the two-ring RHS needs halo depth 2, set once at construction.
    BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), ( mesh_type::halo_depth ) );

    //-----------------------------------------------------------------------//
    // R9 DISCRIMINATOR 1 -- do the owned sets PARTITION the global sets?
    //
    // Summed with a plain MPI_Allreduce over `ownedXCount()` rather than read
    // from Tessera's `globalOwnedX`, deliberately: two independent paths to the
    // same number, and owned-versus-local is exactly what R9 turns on. This is
    // the precondition every owned-range reduction in the RHS needs. Factored
    // into a lambda because the same check runs again at every compared step --
    // that is M0-T3's "assert the counts with an MPI_Allreduce over owned counts
    // rather than a number read from Tessera", and it has to hold for 2000 steps
    // and not only at setup.
    //-----------------------------------------------------------------------//
    auto checkOwnedPartition = [&]( long long step, bool verbose )
    {
        long long owned[3] = { mesh.ownedVertexCount(), mesh.ownedEdgeCount(),
                               mesh.ownedFaceCount() };
        long long total[3] = { 0, 0, 0 };
        MPI_Allreduce( owned, total, 3, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        if ( verbose )
        {
            std::ostringstream os;
            os << "step " << step << " owned partition: sum over ranks V "
               << total[0] << " E " << total[1] << " F " << total[2]
               << "; this rank owns V " << owned[0] << " of local V "
               << mesh.totalVertexCount();
            rec.note( os.str() );
        }
        if ( total[0] != kVertices || total[1] != kEdges ||
             total[2] != kFaces )
        {
            std::ostringstream os;
            os << "ENTITY COUNTS CHANGED at step " << step
               << ": summed owned V " << total[0] << " E " << total[1] << " F "
               << total[2] << ", expected " << kVertices << " / " << kEdges
               << " / " << kFaces
               << ". Adaptivity leaked into the frozen-mesh configuration, or "
                  "the owned sets stopped partitioning the global ones.";
            rec.fail( os.str() );
            return false;
        }
        return true;
    };
    checkOwnedPartition( 0, true );

    //-----------------------------------------------------------------------//
    // The two carried scalars. Every adaptive dt of the run scales off
    // `initial_min_edge` and the volume drift below is measured against
    // `initial_volume`, so both are pinned before the first step -- against THIS
    // LEVEL's gold values.
    //-----------------------------------------------------------------------//
    const double initial_volume = static_cast<double>( solver.initialVolume() );
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

    //-----------------------------------------------------------------------//
    // R9 DISCRIMINATOR 2 -- the closed form, at step 0, RE-DERIVED FOR THIS
    // LEVEL. `volume / (4 pi R^3 / 3)` is the polyhedral deficit of this
    // triangulation and is independent of the partition, so double-counting even
    // a handful of ghost faces moves it in the second or third digit while a
    // summation-order difference does not move it at all in the digits printed.
    //-----------------------------------------------------------------------//
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

    //-----------------------------------------------------------------------//
    // The per-step volume drift, against the REFERENCE's own measured series.
    //
    // OWNED faces only, then one MPI_Allreduce -- the same convention
    // `enclosedVolume` documents and the same one `initial_volume` was computed
    // under, so the two are comparable. Returns the drift so the caller can log
    // it; asserts both the relative agreement and the absolute blow-up cap.
    //-----------------------------------------------------------------------//
    auto checkVolumeDrift = [&]( long long step )
    {
        auto pos = mesh.positions();
        auto owned_faces = Kokkos::subview(
            mesh.faceVertices(), std::make_pair( 0, mesh.ownedFaceCount() ),
            Kokkos::ALL() );
        const Real local =
            Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
        Real volume = 0;
        MPI_Allreduce( &local, &volume, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
        const double drift =
            static_cast<double>( volume ) / initial_volume - 1.0;
        const double reference = kRefVolumeDrift[step / kCheckpointEvery];
        // Relative to the reference drift where there is one; step 0 is exactly
        // zero on both sides, so compare it absolutely.
        const double deviation = reference == 0.0
                                     ? std::fabs( drift )
                                     : std::fabs( drift / reference - 1.0 );
        std::ostringstream os;
        os.precision( 17 );
        os << "step " << step << " relative drift " << drift << " reference "
           << reference;
        os.precision( 6 );
        os << " deviation " << deviation << " (rtol " << kVolumeDriftRtol
           << ", abs cap " << kVolumeDriftAbsCap << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, deviation <= kVolumeDriftRtol );
        BEATNIK_CHECK_TRUE( rec, std::fabs( drift ) <= kVolumeDriftAbsCap );
    };

    //-----------------------------------------------------------------------//
    // One compared step: the checkpoint Beatnik just wrote against that step's
    // gold file. Rank 0 only -- the comparator is serial Python over one file, so
    // running it everywhere would be N identical runs racing on stdout.
    //-----------------------------------------------------------------------//
    auto compareStep = [&]( long long step )
    {
        if ( rank != 0 )
            return;
        const std::string written = solver.lastCheckpointPath();
        const std::string gold = goldForStep( gold_dir, step );
        BEATNIK_CHECK_TRUE( rec, fileExists( written ) );
        if ( gold.empty() || !fileExists( written ) )
        {
            rec.fail( "step " + std::to_string( step ) +
                      ": missing gold or output file" );
            return;
        }
        const int status = runComparator( python, script, written, gold );
        std::ostringstream os;
        os << "step " << step << " comparator exit " << status
           << " (0 = match, 1 = compared and disagreed, 2 = LOAD ERROR)";
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, status, 0 );
    };

    //-----------------------------------------------------------------------//
    // STEP 0 IS A COMPARED STEP. `setup()` wrote it unconditionally, and M0-A1's
    // depth is steps 0 through 2000 -- 81 files, not 80. It is also the generator
    // gate M0-D1 step 1 made a precondition: a disagreement here is the two
    // icosphere generators differing at this subdivision level (M0-R5), not a
    // divergence measurement, and it must not be read as one.
    //-----------------------------------------------------------------------//
    compareStep( 0 );

    //-----------------------------------------------------------------------//
    // THE RUN. Driven one step at a time through `advanceOneStep` rather than
    // through `solve()`, so the entity-count check happens at EVERY step and the
    // comparison at every checkpointed one. A trajectory that diverges slowly
    // passes an end-state-only comparison at a loose tolerance and fails it at a
    // tight one with no indication of when it went wrong; here the first failing
    // step is the answer.
    //
    // `advanceOneStep` is collective and every rank calls it the same number of
    // times -- the BR ring deadlocks otherwise (T2c), including for a rank that
    // owns zero sources.
    //-----------------------------------------------------------------------//
    const double t_start = MPI_Wtime();
    long long completed = 0;
    bool stopped_early = false;
    for ( int step = 1; step <= kSteps; ++step )
    {
        if ( !solver.advanceOneStep() )
        {
            // M0-R2 / M0-R6. A stop is a REPORTED stop step, never a shorter
            // pass. Neither gold set stopped early and neither did any of
            // M0-D1's eight runs, so this firing is new information.
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << kSteps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time()
               << ". This is a reported stop step, not a shorter pass.";
            rec.fail( os.str() );
            stopped_early = true;
            break;
        }
        completed = solver.step();
        BEATNIK_CHECK_EQ( rec, solver.step(), static_cast<long long>( step ) );

        // Cheap, integer, and reduced inside Tessera: safe every step, and the
        // first thing that catches adaptivity leaking in.
        if ( mesh.globalVertexCount() != kVertices ||
             mesh.globalFaceCount() != kFaces )
        {
            std::ostringstream os;
            os << "ENTITY COUNTS CHANGED at step " << step << ": vertices "
               << mesh.globalVertexCount() << " (expected " << kVertices
               << "), faces " << mesh.globalFaceCount() << " (expected "
               << kFaces
               << "). Adaptivity leaked into the frozen-mesh configuration, "
                  "which is what --no-dynamic-remesh --refine-every 0 exists "
                  "to exclude.";
            rec.fail( os.str() );
            break;
        }

        if ( step % kCheckpointEvery != 0 )
            continue;

        //-------------------------------------------------------------------//
        // A COMPARED STEP: the owned-count partition, the adaptive dt, the
        // volume drift, then the field comparison. In that order because that is
        // increasing cost and decreasing locality -- a partition failure or a dt
        // failure names itself, where a field table does not.
        //-------------------------------------------------------------------//
        if ( !checkOwnedPartition( step, false ) )
            break;

        const double t = static_cast<double>( solver.time() );
        if ( step == kSteps )
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "step " << step << " time " << t << " vs gold " << kFinalTime;
            rec.note( os.str() );
            BEATNIK_CHECK_CLOSE( rec, t, kFinalTime, kTimeRtol );
        }

        checkVolumeDrift( step );
        compareStep( step );
    }
    Kokkos::fence();
    const double t_solve = MPI_Wtime() - t_start;

    solver.finalize();

    //-----------------------------------------------------------------------//
    // The step budget must have been reached: 2000 is what both gold sets ran,
    // and anything less is a ceiling the next session has to know about.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( kSteps ) );
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );

    //-----------------------------------------------------------------------//
    // A NEGATIVE CASE, and it is a real one. T1b's lesson: a check that has only
    // ever seen agreeing data has not been tested. The cheapest genuine negative
    // is comparing the final state against the STEP 0 gold file -- same schema,
    // same mesh, same carried scalars, a different time and different positions.
    // It must exit exactly 1 ("compared and disagreed") and NOT 2 ("could not
    // load"), because accepting 2 is how a negative case passes vacuously.
    //
    // It also proves something specific about this test: that 2000 steps actually
    // MOVED the surface. Skipped only if the run stopped early, where the final
    // state is not the thing the criterion is about.
    //-----------------------------------------------------------------------//
    if ( rank == 0 && !stopped_early )
    {
        const std::string written = solver.lastCheckpointPath();
        const std::string step0 = goldForStep( gold_dir, 0 );
        if ( !step0.empty() && fileExists( written ) )
        {
            const int status = runComparator( python, script, written, step0 );
            std::ostringstream os;
            os << "NEGATIVE case, final state vs the step-0 gold: exit "
               << status
               << " (1 = detected a mismatch, 2 = LOAD ERROR and therefore a "
                  "vacuous pass)";
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, status, 1 );
        }
        else
        {
            rec.fail( "negative case: step-0 gold or output file is missing" );
        }
    }

    //-----------------------------------------------------------------------//
    // COST. What tells the next session whether a deeper depth, a finer level or
    // a tighter checkpoint cadence is affordable. GPU-side memory is OUT OF
    // SCOPE -- there is no mechanism for it here.
    //-----------------------------------------------------------------------//
    {
        struct rusage ru;
        long peak_kb = 0;
        if ( ::getrusage( RUSAGE_SELF, &ru ) == 0 )
            peak_kb = ru.ru_maxrss; // kB on Linux
        long peak_max = peak_kb;
        MPI_Allreduce( &peak_kb, &peak_max, 1, MPI_LONG, MPI_MAX,
                       mesh.comm() );
        double comparator_max = g_comparator_seconds;
        MPI_Allreduce( &g_comparator_seconds, &comparator_max, 1, MPI_DOUBLE,
                       MPI_MAX, mesh.comm() );

        std::ostringstream os;
        os.precision( 17 );
        os << "FINAL step " << solver.step() << " time " << solver.time();
        os.precision( 6 );
        os << "  solve+compare wall " << t_solve << " s";
        if ( completed > 0 )
            os << " (" << ( t_solve / double( completed ) ) << " s/step)";
        os << "  of which comparator " << comparator_max << " s in "
           << g_comparator_calls << " invocation(s)"
           << "  peak RSS this rank " << peak_kb << " kB, worst rank "
           << peak_max << " kB";
        rec.note( os.str() );

        if ( rank == 0 )
        {
            // One machine-greppable line per launch, so a tier log reduces to a
            // table without parsing the prose above.
            std::printf( "[m0t3] COST level=%d space=%s np=%d steps=%lld "
                         "wall=%.6f comparator=%.6f peak_rss_kb=%ld\n",
                         kSubdivisions, ExecSpace::name(), comm_size, completed,
                         t_solve, comparator_max, peak_max );
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
        Beatnik::Test::Recorder rec( "Beatnik_Test_Milestone0Frozen" );
        try
        {
            // BEATNIK_TEST_EXEC_SPACE is defined by the per-backend shim
            // tests/CMakeLists.txt generates, so the target name's `_SERIAL` /
            // `_HIP` suffix means what the runner's filter assumes it means.
            // Defaulting to the default space keeps the file compilable alone.
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
    // the log names which rank failed; MPI_MAX then makes any rank's failure the
    // job's failure. The checks above are deliberately not all rank-0's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
