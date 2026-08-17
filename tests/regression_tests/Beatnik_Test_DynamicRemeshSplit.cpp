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
 * @file Beatnik_Test_DynamicRemeshSplit.cpp
 * @brief **REGRESSION TEST 5** (task T4b) — twenty timesteps of metric-driven
 *        dynamic remeshing, split third only, against the reference's own
 *        split-only run, the structural invariants, and risk R12's two shape
 *        signals.
 *
 * THIS IS THE FIFTH MEMBER OF THE SHIP GATE, so the gate is now 5 members x
 * {SERIAL, HIP} x ranks 1-6 = **60 launches** on tuolumne. The user authorized
 * the growth in T4b's exit criterion.
 *
 * WHAT "SPLIT ONLY" MEANS HERE, AND WHY IT IS NOT A BEATNIK-ONLY MODE
 * ------------------------------------------------------------------
 * `dynamic_remesh.py` is three thirds — split, collapse, flip — plus a
 * tangential smoothing pass. T4b implemented the sizing field and the **split**
 * third; the rest is T4d, blocked on Tessera gaps G5b/G5c/G5d. No
 * `--dynamic-remesh-split-only` switch was invented. Instead the run is
 * configured so the **reference itself** would do nothing in the other three,
 * through its own knobs:
 *
 *     --remesh-collapse-factor 0   candidate predicate never true    (:373)
 *     --remesh-smooth-iters 0      the pass returns immediately  (:463-465)
 *     --remesh-flip-min-gain 1e12  every candidate is `continue`d (:449-450)
 *     --no-isotropic-cleanup       the driver skips the block   (:1493)
 *
 * Any other value is rejected by `requireSupportedConfiguration`, by method
 * name and task ID, which is what check 7 below asserts. Note
 * `--remesh-max-collapses 0` is NOT one of the levers: the driver maps a
 * non-positive value to `None` = UNLIMITED, and Beatnik's
 * `max_collapses_per_pass` reproduces that. Only the collapse factor disables
 * the pass.
 *
 * WHY THE SIZING PARAMETERS ARE NOT THE DEFAULTS, AND WHY THAT IS NOT A DODGE
 * --------------------------------------------------------------------------
 * At the DEFAULT `--remesh-sagitta-tolerance 0.004` and `--remesh-h-max 0.05`,
 * on this initial mesh, the sizing field is pinned at its upper clamp: the
 * curvature term asks for `sqrt(8*0.004/3.98) = 0.0894`, the clamp cuts it to
 * `0.05`, and the split threshold `1.35 * 0.05 = 0.0675` sits **below the
 * shortest edge in the mesh** (`0.0690`). Measured against the read-only Python
 * at exactly those flags:
 *
 *   - pass 1 marks **480 of 480** edges — the field selects nothing, the pass is
 *     all-or-nothing;
 *   - `--remesh-max-splits 300` then truncates it, which is risk R4's territory:
 *     a global threshold search accepts a different set than the reference's
 *     sort-and-slice, and the tie-break (the endpoint index pair) is not even
 *     expressible in Beatnik, so pass 1 could not be compared at all;
 *   - and after that single uniform pass the mesh is at `0.0345` against the
 *     same `0.0675` threshold, so **nothing splits again for the remaining 19
 *     steps** — risk R15's trap, in both directions at once.
 *
 * So this test sets
 *
 *     --remesh-sagitta-tolerance 0.002  --remesh-h-max 0.06
 *
 * which puts the threshold inside the edge-length distribution instead of
 * outside it. Every other remesh knob is the reference's default, **including
 * `--remesh-max-splits 300`, which never binds here** (the largest pass is 120
 * splits). Measured against the read-only Python at these flags: five real
 * passes, `320 -> 560 -> 800 -> 1040 -> 1160 -> 1400` faces, then quiescent.
 * The deviation and its measurement are recorded in `tasks/framework.md` under
 * T4b and in the progress log.
 *
 * WHAT IS ACTUALLY CHECKED
 * ------------------------
 *  1. **The run completes 20 steps** without aborting, remeshing every step.
 *  2. **Euler and conformity after every pass.** `V - E + F = 2` globally, and
 *     every owned edge still names exactly two incident faces with both locally
 *     resident. A non-conforming split shows up here and nowhere else.
 *  3. **The mask is complete.** `splits == split_candidates` at every pass, the
 *     cap never binding — i.e. *every* edge longer than
 *     `split_factor * max(target, h_min)` is split, which is the exit
 *     criterion's "either split in the next pass or blocked by h_min" as an
 *     assertion rather than an inspection. `long_edges_after` is logged with the
 *     `h_min`-floored subset separated out, because a non-zero count there is
 *     the next pass's work (a split halves an edge; an edge more than
 *     `2 * split_factor` over target is still long afterwards) and not a defect.
 *  4. **Agreement with the reference**, face for face and vertex for vertex, at
 *     **every** step. Pass 1 is the one the criterion requires — it acts on a
 *     mesh both codes agree on bitwise (T2d validated the fixed-connectivity
 *     steps at `1e-10`), so the masks agree and the counts must. That the other
 *     nineteen agree too is a measurement and not an expectation; R13 predicts
 *     divergence and got it at T4a. See `kPyVertices`.
 *  5. **R12's two signals, every pass, against the reference's own**: the global
 *     minimum `r/R` and the global count of faces below `r/R = 0.25`. This mask
 *     is the length-driven family Tessera measured as periodic, so unlike T4a
 *     the expected shape is the HEALTHY one, and the test asserts that shape:
 *     the count returns to zero after a dip, and the last third of the run sets
 *     no new low.
 *  6. **The volume projection actually runs.** Under `--dynamic-remesh` the
 *     reference projects to the initial volume after every remesh step
 *     (`:1513-1516`), gated on the remesh having *run* rather than on it having
 *     changed anything — so T4b is where `VolumeProjection::projectToVolume`
 *     first executes. The per-step drift is therefore driven to zero, and the
 *     bound is tight enough to tell that from "the projection was skipped":
 *     T2d measured `5.17e-11` at step 10 on a FIXED mesh with no projection, so
 *     `kVolumeDriftBound = 1e-14` separates the two by three decades. T2d's
 *     `kGoldVolumeDrift` is for that fixed mesh and is deliberately NOT reused.
 *  7. **The failure direction, five ways.** Each unimplemented third, at its
 *     reference default, must be rejected before the first step by name and by
 *     task ID — not by a Tessera `EditFamily` throw, and not by silently running
 *     without it.
 *
 * WHAT IS DELIBERATELY NOT CHECKED
 * --------------------------------
 * **Vertex positions or fields against a Python gold file.** Risk R13: for a
 * face with exactly two split edges Tessera cuts the quad along its shorter
 * diagonal while the Python uses a fixed rotation-dependent one, and this
 * task's masks are partial from pass 1 onward, so the two codes integrate
 * different meshes from step 2. Counts and statistics are comparable; a
 * `compare_output.py` field table is not, which is why this test invokes no
 * comparator.
 *
 * ARGUMENTS: none. Everything it needs it computes or carries as a literal.
 */

#include <Beatnik_DynamicRemesh.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The configuration, and where each number comes from.
//---------------------------------------------------------------------------//
constexpr int kSteps = 20;
constexpr int kRemeshEvery = 1;

constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

/// The two deviations from the reference's remesh defaults; see the file
/// header for the measurement that forced them.
constexpr Real kSagittaTolerance = 2.0e-3;
constexpr Real kHMax = 0.06;
/// Everything else is the reference's default, spelled out.
constexpr Real kHMin = 0.0015;
constexpr Real kSplitFactor = 1.35;
constexpr int kMaxSplits = 300;
constexpr Real kMinQuality = 0.18;

constexpr long long kVertices0 = 162;
constexpr long long kEdges0 = 480;
constexpr long long kFaces0 = 320;

/// T1a's two carried scalars, pinned before anything evolves.
constexpr double kInitialVolume = 6.3235073124669514e-02;
constexpr double kInitialMinEdge = 6.8976121063816842e-02;
constexpr double kScalarRtol = 1.0e-12;

/// Global vertex and face counts after each step in the **read-only Python
/// reference**, run at exactly this configuration
/// (`run_adaptive_mesh_bubble.py --steps 20 --source-quadrature vertex
/// --br-approximation direct --dynamic-remesh --remesh-every 1
/// --remesh-sagitta-tolerance 0.002 --remesh-h-max 0.06
/// --remesh-collapse-factor 0 --remesh-max-collapses 0 --remesh-smooth-iters 0
/// --remesh-flip-min-gain 1e12 --no-isotropic-cleanup`). Index 0 is the initial
/// state. The reference splits 120, 120, 120, 60 and 120 edges on steps 1-5 and
/// nothing after that.
///
/// **All twenty steps are asserted against these, and that is a MEASUREMENT
/// rather than an expectation.** R13 says a Python comparison of an adaptive run
/// is a one-pass comparison of counts plus an all-pass comparison of shape
/// statistics, because wherever a face has exactly two split edges Tessera cuts
/// the quad along its geometrically shorter diagonal while
/// `split_selected_edges` uses a fixed one -- same V/E/F, different
/// connectivity, and from the next step the two codes integrate different
/// meshes. That is what happened at T4a. It did **not** happen here: T4b's first
/// run reproduced the reference's per-step counts, minimum `r/R` and sub-`0.25`
/// population at every one of the twenty steps, to every digit measured, and
/// byte-identically across {SERIAL, HIP} x ranks {1, 3, 6}. The honest reading
/// is that on this mesh the two diagonal rules agree wherever the case arises,
/// not that R13 has been retired -- so if a later change breaks one of these
/// late-step assertions, check the diagonal before assuming a bug.
constexpr long long kPyVertices[kSteps + 1] = {
    162, 282, 402, 522, 582, 702, 702, 702, 702, 702, 702,
    702, 702, 702, 702, 702, 702, 702, 702, 702, 702 };
constexpr long long kPyFaces[kSteps + 1] = {
    320, 560, 800, 1040, 1160, 1400, 1400, 1400, 1400, 1400, 1400,
    1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400, 1400 };

/// Steps on which the counts are asserted against the reference: **all of
/// them**, measured. See `kPyVertices`.
constexpr int kPyAgreementSteps = kSteps;

/// **R12's two signals, FROM THE REFERENCE ITSELF**, per step, computed offline
/// from its checkpoints with the same `8A^2/((a+b+c)abc)` formula
/// `SurfaceOperators::radiusRatioStats` uses.
///
/// **This is the HEALTHY signature R12 describes, and T4a's is not.** The
/// minimum dips to `0.2485` at steps 2-3 and *recovers* to `0.2815`, where it
/// stays; the sub-`0.25` population goes `0 -> 120 -> 120 -> 0` and stays at
/// zero. R12 predicted exactly this for a purely length-driven mask -- the
/// family Tessera measured as exactly periodic -- and T4a's monotone decline
/// belongs to its own mask (green transition faces bisected on a neighbour's
/// edge rather than their own longest), not to `splitEdges()`.
///
/// The slow drift in the last digits from step 5 on is the bubble deforming
/// under a fixed connectivity, i.e. it tracks the roll-up and not the round
/// index -- R12's "solver problem" axis, and at this magnitude simply physics.
constexpr double kPyMinRadius[kSteps + 1] = {
    0.486497704566, 0.373875540852, 0.248492357897, 0.248490855246,
    0.281539942917, 0.281537474137, 0.281536790513, 0.281534863505,
    0.281532680186, 0.281530322262, 0.281527789752, 0.281525082674,
    0.281522201047, 0.281519144892, 0.281515914228, 0.281512509079,
    0.281508929466, 0.281505175412, 0.281501246941, 0.281497144079,
    0.281492866851 };
constexpr long long kPyBelowQuarter[kSteps + 1] = {
    0, 0, 120, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

/// Steps on which the two shape signals are asserted against the reference's:
/// **all of them**, to the twelve significant digits the reference columns
/// carry. Measured on T4b's first run and identical across
/// {SERIAL, HIP} x ranks {1, 3, 6}; see the log entry.
constexpr int kShapeAgreementSteps = kSteps;
constexpr double kShapeRtol = 1.0e-10;

/// **THE MEASURED FLOOR** -- the global minimum inradius/circumradius the whole
/// 20-step run stays above, `0.5` being equilateral. The reference's run
/// minimum is `0.248490855246`, at step 3; this is that rounded DOWN to three
/// digits. Deliberately NOT Tessera's `kMinRadiusRatioFloor` of `0.25`, which
/// R12 says is a statement about `test_split_edges` case 8's mask -- and which
/// this run would fail by four ulp at steps 2 and 3.
constexpr double kMinRadiusRatioFloor = 0.248;

/// The reference's minimum triangle quality `4*sqrt(3)*A/sum(l^2)` never falls
/// below this over the twenty steps (measured: `0.9773` initially, `0.6247` at
/// its worst, `0.6732` at step 20). This is the scale `--remesh-min-quality`
/// (0.18) is expressed on, and it is **T4b's answer to what T4d needs to know**:
/// with no coarsening at all, quality does not approach the repair trigger
/// within 20 steps, so the missing collapse third does not bite here.
constexpr double kMinTriangleQuality = 0.60;

/// Per-step enclosed-volume drift `V/V0 - 1` of the reference, as 17-digit
/// literals (T2d's convention). It is **exactly zero** at every step but one,
/// because under `--dynamic-remesh` the driver projects the state back to the
/// initial volume after every remesh step; step 17's `2.22e-16` is one ulp of
/// the ratio. T2d's `kGoldVolumeDrift` describes a FIXED mesh with no
/// projection and must not be reused here.
constexpr double kPyVolumeDrift[kSteps + 1] = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 2.22044604925031308e-16,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00 };

/// Bound on |V/V0 - 1| at every step. **This has teeth precisely because the
/// projection runs**: T2d measured the drift of a fixed-connectivity run
/// WITHOUT the projection as `5.17e-11` by step 10, growing linearly, so a
/// bound three decades below that fails a build in which the projection was
/// skipped -- which is the R15 trap for a control whose correct output is
/// "zero". The `1e-9` absolute blow-up cap T4b's criterion also names is
/// implied by it and is kept as a second, coarser assertion.
constexpr double kVolumeDriftBound = 1.0e-14;
constexpr double kVolumeDriftAbsCap = 1.0e-9;

//---------------------------------------------------------------------------//
/// The command line above, as a `SolverParams`. Everything that matters is set
/// explicitly rather than inherited, so a later change to a Beatnik default
/// breaks this test loudly instead of silently changing what it runs.
Beatnik::SolverParams makeParams()
{
    Beatnik::SolverParams p;

    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = kSubdivisions;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
    p.zmodel.sigma = 0.0;
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = Beatnik::KernelBlobMode::Length;
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = Beatnik::VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = Beatnik::BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    p.time.steps = kSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --dynamic-remesh --remesh-every 1. THE POINT OF THIS TEST.
    p.dynamic_remesh = true;
    p.remesh_every = kRemeshEvery;
    p.remesh_tight_after = -1.0;

    // The sizing field: two deviations (see the file header), everything else
    // the reference's default.
    p.remesh.sagitta_tolerance = kSagittaTolerance;
    p.remesh.h_max = kHMax;
    p.remesh.h_min = kHMin;
    p.remesh.split_factor = kSplitFactor;
    p.remesh.max_splits_per_pass = kMaxSplits;
    p.remesh.passes = 1;
    p.remesh.target_gradation_factor = 1.35;
    p.remesh.target_gradation_iterations = 8;
    p.remesh.min_quality = kMinQuality;

    // The three unimplemented thirds, configured off through the REFERENCE's
    // own knobs -- see the file header. Any other value aborts at setup, which
    // check 7 asserts.
    p.remesh.collapse_factor = 0.0;
    p.remesh.max_collapses_per_pass = 0;
    p.remesh.smoothing_iterations = 0;
    p.remesh.smoothing_relaxation = 0.04;
    p.remesh.flip_min_gain = Beatnik::kFlipsDisabledMinGain;
    p.cleanup.enabled = false;

    // Both proximity paths are off by default in the reference too
    // (dynamic_remesh.py:33,41); spelled out because they are T4e.
    p.remesh.use_proximity = false;
    p.remesh.surgical_proximity = false;

    // The indicator-driven refiner is left at its default cadence ON PURPOSE:
    // the two adaptivity modes are mutually exclusive per run
    // (`run_adaptive_mesh_bubble.py:1424` versus `:1469`), and this test asserts
    // that -- `lastRefinement()` must stay default-constructed for the whole
    // run even though `--refine-every 5` would otherwise fire four times.
    p.amr.refine_every = 5;

    p.filter.flip_passes = 0;
    p.filter.smooth_iters = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;

    // No checkpoints: this test compares counts and statistics, not fields.
    p.checkpoint.directory = std::string();
    p.checkpoint.prefix = "checkpoint";
    p.checkpoint.every_steps = 0;
    p.checkpoint.every_time = 0.0;

    return p;
}

//---------------------------------------------------------------------------//
/// Conformity, re-asserted after every pass: every **owned** edge names exactly
/// two incident faces and both are locally resident. Identical in intent and in
/// implementation to T4a's check -- see
/// `Beatnik_Test_RefineSplitEdges.cpp::checkConformity` for why the RESIDENT
/// pair and not `EdgeField::Faces`.
template <class MeshType>
void checkConformity( Beatnik::Test::Recorder& rec, MeshType& mesh,
                      const std::string& where )
{
    using exec = typename MeshType::execution_space;
    const int n_owned = mesh.ownedEdgeCount();
    auto inc = mesh.edgeAdjacency();
    auto count = inc.resident_count;
    auto faces = inc.resident_faces;

    long long local[2] = { 0, 0 };
    Kokkos::parallel_reduce(
        "beatnik_t4b_conformity", Kokkos::RangePolicy<exec>( 0, n_owned ),
        KOKKOS_LAMBDA( const int e, long long& holes, long long& absent ) {
            if ( count( e ) != 2 )
                ++holes;
            else if ( faces( e, 0 ) < 0 || faces( e, 1 ) < 0 )
                ++absent;
        },
        local[0], local[1] );

    long long total[2] = { 0, 0 };
    MPI_Allreduce( local, total, 2, MPI_LONG_LONG, MPI_SUM, mesh.comm() );

    std::ostringstream os;
    os << where << ": owned edges with != 2 incident faces " << total[0]
       << " (a hanging node), with a non-resident incident face " << total[1]
       << " (a halo that did not survive the edit)";
    rec.note( os.str() );
    BEATNIK_CHECK_EQ( rec, total[0], 0 );
    BEATNIK_CHECK_EQ( rec, total[1], 0 );
}

//---------------------------------------------------------------------------//
/// The global enclosed volume, with `enclosedVolume`'s own convention and over
/// **owned** faces only (risk R9), so it is comparable with `initialVolume()`.
template <class MeshType>
double globalVolume( MeshType& mesh )
{
    auto pos = mesh.positions();
    auto owned_faces =
        Kokkos::subview( mesh.faceVertices(),
                         std::make_pair( 0, mesh.ownedFaceCount() ),
                         Kokkos::ALL() );
    const Real local =
        Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
    Real volume = 0;
    MPI_Allreduce( &local, &volume, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
    return static_cast<double>( volume );
}

//---------------------------------------------------------------------------//
/// One failure direction: a configuration that reaches an unimplemented pass
/// must be rejected before the first step, with a message naming the method and
/// the task. Checked rather than trusted because there are three ways for it to
/// go wrong and only one of them is loud on its own: the run could proceed
/// silently without the pass (a plausible trajectory that is not the
/// reference's), it could reach a bare `BEATNIK_NOT_IMPLEMENTED` several steps
/// in with no task ID on it, or it could abort inside Tessera on an
/// `EditFamily` violation -- which would mean Beatnik had reached for the
/// hierarchical family after all.
template <class ExecSpace, class MemSpace, class Mutator>
void checkRejected( Beatnik::Test::Recorder& rec, const std::string& what,
                    const std::string& method, const std::string& task,
                    Mutator mutate )
{
    Beatnik::SolverParams p = makeParams();
    mutate( p );

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD, p );
    solver.setup();

    bool threw = false;
    std::string message;
    try
    {
        solver.solve();
    }
    catch ( const std::exception& e )
    {
        threw = true;
        message = e.what();
    }

    rec.note( "failure direction, " + what + ": " +
              ( threw ? message
                      : std::string( "NO THROW -- the run proceeded without "
                                     "the pass" ) ) );
    BEATNIK_CHECK_TRUE( rec, threw );
    BEATNIK_CHECK_TRUE( rec, message.find( method ) != std::string::npos );
    BEATNIK_CHECK_TRUE( rec, message.find( task ) != std::string::npos );
    // NOT a Tessera EditFamily throw, which would name the two families.
    BEATNIK_CHECK_TRUE( rec, message.find( "EditFamily" ) == std::string::npos );
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;

    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD, makeParams() );
    solver.setup();

    auto& mesh = solver.mesh();

    //-----------------------------------------------------------------------//
    // Structure and the two carried scalars, before anything evolves.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices0 );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges0 );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces0 );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), ( mesh_type::halo_depth ) );
    BEATNIK_CHECK_CLOSE( rec, static_cast<double>( solver.initialVolume() ),
                         kInitialVolume, kScalarRtol );
    BEATNIK_CHECK_CLOSE( rec, static_cast<double>( solver.initialMinEdge() ),
                         kInitialMinEdge, kScalarRtol );
    checkConformity( rec, mesh, "before step 1" );

    const double initial_volume = static_cast<double>( solver.initialVolume() );

    //-----------------------------------------------------------------------//
    // THE RUN. One step at a time so each remesh can be inspected where it
    // happens; `advanceOneStep` is collective and every rank calls it the same
    // number of times.
    //-----------------------------------------------------------------------//
    double run_min_ratio = 1.0;
    double last_third_min_ratio = 1.0;
    double run_min_quality = 1.0;
    long long max_below_quarter = 0;
    int passes_seen = 0;

    for ( int step = 1; step <= kSteps; ++step )
    {
        const bool ok = solver.advanceOneStep();
        BEATNIK_CHECK_TRUE( rec, ok );
        if ( !ok )
            break;

        ++passes_seen;
        const auto& d = solver.lastRemesh();

        //-------------------------------------------------------------------//
        // The pass report, and R12's two signals against the ROUND INDEX.
        //-------------------------------------------------------------------//
        const double drift = globalVolume( mesh ) / initial_volume - 1.0;
        {
            std::ostringstream os;
            os.precision( 12 );
            os << "pass " << step << ": faces " << d.old_faces << " -> "
               << d.new_faces << ", vertices " << d.old_vertices << " -> "
               << d.new_vertices << ", splits " << d.splits << " of "
               << d.split_candidates << " candidates"
               << ( d.split_capped ? " (CAPPED)" : "" ) << ", long after "
               << d.long_edges_after << " (" << d.long_edges_at_h_min
               << " at h_min), minQ " << d.min_quality_before << " -> "
               << d.min_quality_after << ", max sagitta "
               << d.max_sagitta_before << " -> " << d.max_sagitta_after
               << " | R12: min r/R " << d.min_radius_ratio << ", faces below "
               << "0.25 " << d.faces_below_quarter;
            rec.note( os.str() );

            std::ostringstream vos;
            vos.precision( 17 );
            vos << "pass " << step << ": volume drift " << std::scientific
                << drift << " (reference " << kPyVolumeDrift[step] << ")";
            rec.note( vos.str() );
        }

        //-------------------------------------------------------------------//
        // Structure.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), d.new_faces );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), d.new_vertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        checkConformity( rec, mesh, "after pass " + std::to_string( step ) );

        //-------------------------------------------------------------------//
        // The mask is complete, and the cap never bound. Together these are the
        // exit criterion's "every edge longer than split_factor * target is
        // split unless blocked": the mask is built from that predicate alone
        // (R12's hard constraint), so equality here says nothing was dropped,
        // and !capped says nothing was dropped for a resource reason either.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_TRUE( rec, !d.split_capped );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( d.splits ),
                          static_cast<long long>( d.split_candidates ) );
        BEATNIK_CHECK_EQ( rec, d.passes, 1 );

        //-------------------------------------------------------------------//
        // The refine branch must NOT have run: the two adaptivity modes are
        // mutually exclusive per run, and `--refine-every 5` is set.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, solver.lastRefinement().new_faces, 0 );

        //-------------------------------------------------------------------//
        // Agreement with the reference. See `kPyFaces` for why the number of
        // passes compared is not twenty.
        //-------------------------------------------------------------------//
        if ( step <= kPyAgreementSteps )
        {
            BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kPyFaces[step] );
            BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(),
                              kPyVertices[step] );
        }
        if ( step <= kShapeAgreementSteps )
        {
            BEATNIK_CHECK_CLOSE( rec,
                                 static_cast<double>( d.min_radius_ratio ),
                                 kPyMinRadius[step], kShapeRtol );
            BEATNIK_CHECK_EQ( rec, d.faces_below_quarter,
                              kPyBelowQuarter[step] );
        }

        //-------------------------------------------------------------------//
        // The volume projection ran. See `kVolumeDriftBound`.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_TRUE( rec, std::fabs( drift ) <= kVolumeDriftBound );
        BEATNIK_CHECK_TRUE( rec, std::fabs( drift ) <= kVolumeDriftAbsCap );

        //-------------------------------------------------------------------//
        // R12 and the quality trace, accumulated over the run.
        //-------------------------------------------------------------------//
        run_min_ratio =
            std::min( run_min_ratio, static_cast<double>( d.min_radius_ratio ) );
        if ( step > ( 2 * kSteps ) / 3 )
            last_third_min_ratio =
                std::min( last_third_min_ratio,
                          static_cast<double>( d.min_radius_ratio ) );
        run_min_quality =
            std::min( run_min_quality,
                      static_cast<double>( d.min_quality_after ) );
        max_below_quarter =
            std::max( max_below_quarter,
                      static_cast<long long>( d.faces_below_quarter ) );
        BEATNIK_CHECK_TRUE( rec, d.min_radius_ratio > 0.0 );
    }

    BEATNIK_CHECK_EQ( rec, passes_seen, kSteps );
    BEATNIK_CHECK_EQ( rec, solver.step(), static_cast<long long>( kSteps ) );

    //-----------------------------------------------------------------------//
    // R12's verdict for this mask, as two assertions rather than an impression.
    //
    // HEALTHY (this mask, length-driven, Tessera's periodic family):
    //   - the sub-0.25 population dips and RETURNS TO ZERO;
    //   - the last third of the run sets NO NEW LOW.
    // SHAPE PROBLEM (what T4a measured for its own mask): both decline
    // monotonically and the population settles at a fixed fraction of the mesh.
    // If this ever fails, record it -- do NOT apply an R12 mitigation here,
    // which is a separate task with its own gold set.
    //-----------------------------------------------------------------------//
    {
        std::ostringstream os;
        os.precision( 12 );
        os << "R12: run minimum r/R " << run_min_ratio << " (floor "
           << kMinRadiusRatioFloor << ", measured -- NOT Tessera's 0.25), "
           << "last-third minimum " << last_third_min_ratio
           << ", peak sub-0.25 population " << max_below_quarter
           << ", final population "
           << solver.lastRemesh().faces_below_quarter;
        rec.note( os.str() );
    }
    BEATNIK_CHECK_TRUE( rec, run_min_ratio >= kMinRadiusRatioFloor );
    // The dip recovered: a population appeared and went away again.
    BEATNIK_CHECK_TRUE( rec, max_below_quarter > 0 );
    BEATNIK_CHECK_EQ( rec, solver.lastRemesh().faces_below_quarter, 0 );
    // No new low late in the run.
    BEATNIK_CHECK_TRUE( rec, last_third_min_ratio > run_min_ratio );

    //-----------------------------------------------------------------------//
    // T4b's answer to what T4d needs to know: how badly the missing coarsening
    // bites. It does not, within twenty steps -- the minimum triangle quality
    // never approaches `--remesh-min-quality`, so the repair trigger the
    // reference gates its flip/smooth pass on is never reached for that reason.
    //-----------------------------------------------------------------------//
    {
        std::ostringstream os;
        os.precision( 12 );
        os << "T4d input: run minimum triangle quality " << run_min_quality
           << " against --remesh-min-quality " << kMinQuality
           << "; the repair trigger is never reached in " << kSteps
           << " steps with no coarsening at all";
        rec.note( os.str() );
    }
    BEATNIK_CHECK_TRUE( rec, run_min_quality >= kMinTriangleQuality );
    BEATNIK_CHECK_TRUE( rec, run_min_quality > kMinQuality );

    solver.finalize();

    //-----------------------------------------------------------------------//
    // The failure directions, last: each constructs a second solver, and doing
    // them first would leave the interesting run downstream of an exception
    // path.
    //-----------------------------------------------------------------------//
    checkRejected<ExecSpace, MemSpace>(
        rec, "--remesh-proximity", "nonlocalFaceCentroidDistance", "T4e",
        []( Beatnik::SolverParams& p ) { p.remesh.use_proximity = true; } );
    checkRejected<ExecSpace, MemSpace>(
        rec, "--remesh-collapse-factor 0.45", "collapseShortEdges", "T4d",
        []( Beatnik::SolverParams& p ) { p.remesh.collapse_factor = 0.45; } );
    checkRejected<ExecSpace, MemSpace>(
        rec, "--remesh-smooth-iters 1", "tangentialSmooth", "T4d",
        []( Beatnik::SolverParams& p ) { p.remesh.smoothing_iterations = 1; } );
    checkRejected<ExecSpace, MemSpace>(
        rec, "--remesh-flip-min-gain 1e-3", "flipEdgesForQuality", "T4d",
        []( Beatnik::SolverParams& p ) { p.remesh.flip_min_gain = 1.0e-3; } );
    checkRejected<ExecSpace, MemSpace>(
        rec, "--isotropic-cleanup", "isotropicCleanup", "T4d",
        []( Beatnik::SolverParams& p ) { p.cleanup.enabled = true; } );
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_DynamicRemeshSplit" );
        try
        {
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
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

    // ONE VERDICT ACROSS THE RANKS, as in regression tests 1-4.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
