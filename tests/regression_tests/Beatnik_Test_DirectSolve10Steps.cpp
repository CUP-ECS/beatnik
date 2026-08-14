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
 * @file Beatnik_Test_DirectSolve10Steps.cpp
 * @brief **REGRESSION TEST 2** — ten TVD-RK3 timesteps with the direct
 *        Birkhoff-Rott solver, compared against the T2a Python gold set at
 *        every step.
 *
 * THIS IS THE THIRD MEMBER OF THE SHIP GATE.
 * `tasks/framework.md` states T2d's exit criterion as: *regression test 2
 * (direct-solve-10-steps) passes at all 10 timesteps, `--rtol 1e-10`, at ranks
 * 1, 2, 3, 4 and 5; volume drift stays below `1e-12` relative.* The tier — not
 * this file — supplies the rank sweep, per the convention T1c set and T2c
 * followed, so the criterion's ranks 1-5 are a verified subset of the gate's
 * 1-6 on SERIAL and HIP. The test reads its own comm size and adapts.
 *
 * WHAT IT ACTUALLY EXERCISES, WHICH IS EVERYTHING T2d BUILT
 * --------------------------------------------------------
 * One `Solver::advanceOneStep` per gold file, so every step drives
 * `TimeIntegrator::chooseStepSize` -> three
 * `ZModelSolver::computeRightHandSidePotential` evaluations (each one halo
 * exchange, one geometry, one `SurfaceState::updateSheetVector`, one collective
 * BR ring, one `VolumeProjection::removeVolumeFlux`, one area-weighted
 * re-centring) -> three stage combinations with a re-centred potential ->
 * `SurfaceState::allFinite` -> `Solver::checkpointDue` -> `CheckpointIO::write`
 * -> `DiagnosticsCalculator::compute`. Ten steps is 30 BR evaluations and 30
 * volume projections.
 *
 * THE ADAPTIVE dt IS THE FIRST THING THIS TEST CATCHES, AND DELIBERATELY SO
 * ------------------------------------------------------------------------
 * `time` is a compared scalar (`compare_output.py`'s `FLOAT_SCALARS`), and the
 * gold set's `time` is **not** a uniform multiple of `--dt 0.003`: it is
 * `0.003` exactly at step 1 and then drifts, reaching `0.029996631612342662`
 * at step 10, because the Python re-chose dt from the state every step
 * (`choose_step_dt`). A `chooseStepSize` stubbed to a constant therefore fails
 * this test at **step 2, on `time`**, before any field disagreement — which is
 * the useful ordering, since it separates "the step size is wrong" from "the
 * right-hand side is wrong". T2a's progress-log entry records this as the trap.
 *
 * The per-step `time` is additionally asserted here against a hard-coded literal
 * read from the gold set, so the failure names the step and prints 17 digits
 * rather than arriving as a field-table row.
 *
 * VOLUME DRIFT IS THE SECOND HALF OF THE CRITERION
 * ------------------------------------------------
 * With `preserve_volume` on, `removeVolumeFlux` makes the *rate* of volume
 * change exactly zero in the discrete sense, so what is left after ten steps is
 * RK3 truncation and round-off only — and the *rate* being zero says nothing
 * about the accumulated truncation, which grows linearly in the step count.
 * The reference carries exactly the same truncation: measured offline from the
 * eleven gold `.npz` files, the Python's own relative drift is `5.19e-12` at
 * step 1 and `5.17e-11` at step 10, so the `1e-12` bound this test first
 * asserted was written a priori and sits an order of magnitude below the
 * discretization's floor. What is checked instead is **agreement with the
 * reference's drift** — `kGoldVolumeDrift`, at `kVolumeDriftRtol` relative —
 * which is a strictly stronger statement than any bound: it fails if Beatnik
 * conserves volume better than the Python as well as worse. `kVolumeDriftAbsCap`
 * stays as the blow-up detector. Both are checked every step, against the
 * `initial_volume` the whole run keys off. This test does NOT run
 * `projectToVolume` — no configuration reachable today does (every call site in
 * the reference is inside a refine or remesh branch), so a deviation here is
 * the rate projection failing and nothing else. T2d's log entry records both
 * drift series at 17 digits and the reasoning that separated RK3 truncation
 * from a projection bug.
 *
 * RISK R8 — THE SEAM, WHICH IS WHY THIS TEST IS IN THE MULTI-RANK GATE
 * -------------------------------------------------------------------
 * The RHS is a two-ring stencil on the potential (one surface gradient builds
 * the sheet vector, a second differentiates the Bernoulli potential in the
 * sheet-vector model), and the easy bug is a single exchange of a halo that
 * needed depth 2. That is wrong **only near partition boundaries and only by a
 * small amount**, so it produces a plausible trajectory with a seam that moves
 * when the rank count changes — invisible at one rank, and invisible to any
 * check that does not compare against an external reference. The mitigation is
 * structural (`halo_depth = 2`, set once at construction and asserted below)
 * and the detector is this test running at ranks 1-6 against the same gold set.
 *
 * Distinguishing R8 from R2 if it ever does disagree: a seam localized on
 * partition boundaries is R8; uniformly distributed noise at the `1e-15` level
 * is summation order (R2). The three structural R9 discriminators T1c and T2c
 * established are mechanized below and stay decisive here, because they are
 * statements about the partition rather than about the trajectory:
 *
 *   1. the owned sets partition the global sets (162 / 480 / 320), summed with
 *      a plain `MPI_Allreduce` rather than read from Tessera;
 *   2. `volume / (4 pi R^3 / 3) = 0.96616074859858714` at step 0 — the
 *      polyhedral deficit, a property of the triangulation and not of the
 *      partition;
 *   3. the entity counts never change, which is what `--no-dynamic-remesh
 *      --refine-every 0` is for.
 *
 * R9's third T1c ground — "the deviation does not trend with rank count" — is
 * deliberately NOT relied on here: once positions evolve, a real seam bug also
 * moves with the partition, so that argument no longer separates the two.
 *
 * HOW THE GOLD FILE FOR A STEP IS FOUND
 * -------------------------------------
 * By scanning the gold directory for the file whose name ends `_step%07d.npz`,
 * NOT by reconstructing the name from the time — the time is exactly what is
 * under test, and a name built from Beatnik's own `time` would compare each
 * step against whichever gold file Beatnik's dt happened to point at. A missing
 * step is a named failure.
 *
 * ARGUMENTS. All paths; see tests/CMakeLists.txt for both call sites, which
 * pass them absolute (ctest) and manifest-relative (the installed gate).
 *
 *   argv[1]  the T2a gold DIRECTORY (eleven .npz files, steps 0-10)
 *              regression_tests/direct-solve-10-steps/gold
 *   argv[2]  the comparator
 *              regression_tests/compare_output.py
 *
 * `BEATNIK_PYTHON` overrides the interpreter (default `python3`).
 */

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The T2a configuration.
//
//   python examples/run_adaptive_mesh_bubble.py --steps 10
//       --source-quadrature vertex --br-approximation direct
//       --no-dynamic-remesh --refine-every 0 --checkpoint-every-steps 1
//       --no-video --checkpoint-dir results
//
// Everything else is a Python default, and the ones that matter are set
// explicitly in `makeParams()` below rather than inherited, so a later change to
// a Beatnik default breaks this test loudly instead of silently changing what it
// compares. (T2a's log entry confirms this command and T1a's describe the same
// physics: the extra flags T1a lists are all `parse_args` defaults, and step 0
// of this gold set is bitwise identical to T1a's gold file.)
//---------------------------------------------------------------------------//
constexpr int kSteps = 10;
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

constexpr long long kVertices = 162;
constexpr long long kEdges = 480;
constexpr long long kFaces = 320;

/// T1a's two carried scalars, which this run must reproduce before anything
/// evolves. Both are the same literals regression test 1 asserts.
constexpr double kInitialVolume = 6.3235073124669514e-02;
constexpr double kInitialMinEdge = 6.8976121063816842e-02;

/// The polyhedral deficit of the subdivision-2 icosphere — a property of the
/// triangulation, not of the partition. R9 discriminator 2 (T1c).
constexpr double kVolumeOverSphere = 0.96616074859858714;

/// `time` at each of the ten steps, read from the gold set's `/time` scalar.
/// **NOT round multiples of `--dt 0.003`** — see the file header. These are the
/// numbers a constant `chooseStepSize` fails.
constexpr double kGoldTime[kSteps + 1] = {
    0.0,
    0.003,
    0.005999988175164871,
    0.008999940879087079,
    0.011999834470727973,
    0.01499964531509802,
    0.017999349783936778,
    0.020998924256392593,
    0.023998345119701996,
    0.026997588769868765,
    0.029996631612342662,
};

//---------------------------------------------------------------------------//
// Tolerances.
//---------------------------------------------------------------------------//

/// The criterion's number, and the comparator's own default. **Do not loosen**
/// without the measurement and the justification `tasks/framework.md` R2
/// demands.
constexpr const char* kRtol = "1e-10";
constexpr const char* kAtol = "1e-12";

/// The same relative tolerance, for this test's own per-step `time` check.
constexpr double kTimeRtol = 1.0e-10;

/// The **reference's own** per-step relative volume drift, measured offline
/// from the eleven gold `.npz` files with the same convention `enclosedVolume`
/// uses (`V = (1/6) sum_f a.(b x c)` over `faces`, drift relative to step 0).
/// See the file header: the criterion is agreement with these, not smallness.
/// Signed, and every entry is positive — the reference gains volume.
constexpr double kGoldVolumeDrift[kSteps + 1] = {
    0.0,
    5.1898485509127568e-12,
    1.0375700298936863e-11,
    1.5557333199467394e-11,
    2.0734747252504349e-11,
    2.5907276324232953e-11,
    3.1075142459258132e-11,
    3.6238345657579885e-11,
    4.1396441829988362e-11,
    4.6549430976483563e-11,
    5.1697091052460564e-11,
};

/// How closely Beatnik's per-step drift must track `kGoldVolumeDrift`.
///
/// The drift is `V/V0 - 1` with `V ~ V0 ~ 6.3e-2` and a drift of `5e-12`, so
/// **one ulp of the ratio is already `2.2e-16 / 5.19e-12 = 4.3e-5` of the step-1
/// drift** — a hard round-off floor that no correct implementation can beat, and
/// it shrinks by a decade by step 10 as the drift grows. Across the whole
/// 36-launch gate (SERIAL and HIP, ranks 1-6) the step-1 drift takes exactly
/// **three** distinct values, one ulp apart, so the largest deviation is two
/// ulps — `8.5568818722459028e-05` — and `1e-3` sits a little over a decade
/// above it. **Do not loosen** without a new measurement recorded in
/// `tasks/framework-progress-log.md`.
constexpr double kVolumeDriftRtol = 1.0e-3;

/// The blow-up detector, kept as an absolute cap so a drift that tracks the
/// reference *proportionally* while both explode still fails. Two decades
/// above the reference's step-10 drift.
constexpr double kVolumeDriftAbsCap = 1.0e-9;

/// For the two carried scalars, which regression test 1 pins at the same value.
constexpr double kScalarRtol = 1.0e-12;

//---------------------------------------------------------------------------//
bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// The gold file for `step`, found by its `_step%07d.npz` suffix rather than by
/// rebuilding the name from a time. Empty if the directory holds no such file.
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

/// Run `python <script> <a> <b> --rtol .. --atol ..` and return its exit
/// status, or -1 if it could not be run at all. The two are distinguished on
/// purpose; see regression test 1's header for why.
int runComparator( const std::string& python, const std::string& script,
                   const std::string& lhs, const std::string& rhs )
{
    std::ostringstream cmd;
    cmd << "'" << python << "' '" << script << "' '" << lhs << "' '" << rhs
        << "' --rtol " << kRtol << " --atol " << kAtol << " --quiet";
    std::printf( "[cmd] %s\n", cmd.str().c_str() );
    std::fflush( stdout );

    const int raw = std::system( cmd.str().c_str() );
    if ( raw == -1 || !WIFEXITED( raw ) )
        return -1;
    return WEXITSTATUS( raw );
}

//---------------------------------------------------------------------------//
/// The T2a command line, as a `SolverParams`.
Beatnik::SolverParams makeParams( const std::string& checkpoint_dir )
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere,
    // --icosphere-subdivisions 2, --radius 0.25, --center-z 0.25.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = kSubdivisions;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0:
    // the fast path, which is all InitialCondition implements (T5a).
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
    // --br-approximation direct. Not `fmm`: this must not depend on Canopy
    // (T3a), and the gold set is a direct run.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex. See R11 -- a `face` gold file would not be
    // comparable, and the port implements `vertex` only.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // --steps 10, and the dt controls the gold set was generated under. Every
    // one is a Python default and every one changes the trajectory.
    p.time.steps = kSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --no-dynamic-remesh --refine-every 0. Adaptivity is off DELIBERATELY:
    // test 2 isolates the evolution, and refinement brings its own ordering and
    // tie-breaking differences (risks R4 and R7).
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    // Neither of the other two post-step passes is configured either.
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;

    // --isotropic-cleanup is on by default and is moot with remeshing off.
    p.cleanup.enabled = true;

    // --checkpoint-every-steps 1, so there is one file per gold file.
    p.checkpoint.every_steps = 1;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

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

    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

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
    // reported as itself rather than as a comparison failure (the T1b trap).
    if ( rank == 0 )
    {
        BEATNIK_CHECK_TRUE( rec, fileExists( gold_dir ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( script ) );
        for ( long long s = 0; s <= kSteps; ++s )
        {
            const std::string g = goldForStep( gold_dir, s );
            if ( g.empty() )
                rec.fail( "no gold file for step " + std::to_string( s ) +
                          " in " + gold_dir );
        }
    }

    // Resolution order, and why there are three levels: the installed gate path
    // runs from the manifest's directory, which is inside a spack install prefix
    // and is READ-ONLY. `BEATNIK_TEST_SCRATCH` is what the gate wrapper sets
    // (absolute); TMPDIR covers a hand-run from an install prefix; "." covers
    // ctest, which runs in the build tree.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/beatnik_regression_t2d/"
        << ExecSpace::name() << "_np" << comm_size;

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
    // This is the STRUCTURAL half of the R8 mitigation; the rank sweep against
    // the gold set is the empirical half.
    BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), ( mesh_type::halo_depth ) );

    //-----------------------------------------------------------------------//
    // R9 DISCRIMINATOR 1 — do the owned sets PARTITION the global sets?
    //
    // Summed with a plain MPI_Allreduce over `ownedXCount()` rather than read
    // from Tessera's `globalOwnedX`, deliberately: two independent paths to the
    // same number, and owned-versus-local is exactly what R9 turns on. This is
    // the precondition every owned-range reduction in the RHS needs -- the two
    // volume-projection inner products, the area-weighted mean of phi_dot, and
    // the adaptive dt's minimum edge.
    //-----------------------------------------------------------------------//
    {
        long long owned[3] = { mesh.ownedVertexCount(), mesh.ownedEdgeCount(),
                               mesh.ownedFaceCount() };
        long long total[3] = { 0, 0, 0 };
        MPI_Allreduce( owned, total, 3, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        std::ostringstream os;
        os << "owned partition: sum over ranks V " << total[0] << " E "
           << total[1] << " F " << total[2] << "; this rank owns V " << owned[0]
           << " of local V " << mesh.totalVertexCount() << " (ghost fraction "
           << ( mesh.totalVertexCount() > 0
                    ? double( mesh.totalVertexCount() - owned[0] ) /
                          double( mesh.totalVertexCount() )
                    : 0.0 )
           << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, total[0], kVertices );
        BEATNIK_CHECK_EQ( rec, total[1], kEdges );
        BEATNIK_CHECK_EQ( rec, total[2], kFaces );
    }

    //-----------------------------------------------------------------------//
    // The two carried scalars. Every adaptive dt of the run scales off
    // `initial_min_edge` and the volume drift below is measured against
    // `initial_volume`, so both are pinned before the first step.
    //-----------------------------------------------------------------------//
    const double initial_volume =
        static_cast<double>( solver.initialVolume() );
    const double h0 = static_cast<double>( solver.initialMinEdge() );
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "initial_volume " << initial_volume << " vs T1a "
           << kInitialVolume << ", initial_min_edge " << h0 << " vs T1a "
           << kInitialMinEdge;
        rec.note( os.str() );
    }
    BEATNIK_CHECK_CLOSE( rec, initial_volume, kInitialVolume, kScalarRtol );
    BEATNIK_CHECK_CLOSE( rec, h0, kInitialMinEdge, kScalarRtol );

    //-----------------------------------------------------------------------//
    // R9 DISCRIMINATOR 2 — the closed form, at step 0.
    //
    // `volume / (4 pi R^3 / 3)` is the polyhedral deficit of this triangulation
    // and is independent of the partition, so double-counting even a handful of
    // ghost faces moves it in the second or third digit while a summation-order
    // difference does not move it at all in the digits printed. Asserted here
    // (T1c only reported it) because this test's volume drift check would
    // otherwise be measured against a number that could itself be wrong.
    //-----------------------------------------------------------------------//
    {
        const double sphere = 4.0 * M_PI * std::pow( kRadius, 3 ) / 3.0;
        const double ratio = initial_volume / sphere;
        std::ostringstream os;
        os.precision( 17 );
        os << "volume / (4*pi*R^3/3) = " << ratio << " (expected "
           << kVolumeOverSphere << "; partition-independent)";
        rec.note( os.str() );
        BEATNIK_CHECK_CLOSE( rec, ratio, kVolumeOverSphere, 1.0e-12 );
    }

    //-----------------------------------------------------------------------//
    // THE EXIT CRITERION — ten steps, each compared against its gold file.
    //
    // Driven one step at a time through `advanceOneStep` rather than through
    // `solve()`, so the comparison happens at every step rather than only at the
    // end. That is not merely more thorough: a trajectory that diverges slowly
    // passes an end-state-only comparison at a loose tolerance and fails it at a
    // tight one, with no indication of WHEN it went wrong. Here the first
    // failing step is the answer.
    //
    // `advanceOneStep` is collective and every rank calls it the same number of
    // times -- the BR ring deadlocks otherwise (T2c), including for a rank that
    // owns zero sources.
    //-----------------------------------------------------------------------//
    for ( int step = 1; step <= kSteps; ++step )
    {
        const bool ok = solver.advanceOneStep();
        BEATNIK_CHECK_TRUE( rec, ok );
        if ( !ok )
            break;

        BEATNIK_CHECK_EQ( rec, solver.step(), static_cast<long long>( step ) );

        //-------------------------------------------------------------------//
        // The adaptive dt, first, because it fails first and localizes best.
        //-------------------------------------------------------------------//
        const double t = static_cast<double>( solver.time() );
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "step " << step << " time " << t << " vs gold "
               << kGoldTime[step];
            os.precision( 3 );
            os << "  rel "
               << std::fabs( t - kGoldTime[step] ) /
                      std::fabs( kGoldTime[step] );
            rec.note( os.str() );
        }
        BEATNIK_CHECK_CLOSE( rec, t, kGoldTime[step], kTimeRtol );

        //-------------------------------------------------------------------//
        // The mesh must not have changed. `--no-dynamic-remesh --refine-every 0`
        // is what test 2 exists to hold fixed, and a growth here would mean
        // adaptivity leaked into the test that excludes it.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );

        //-------------------------------------------------------------------//
        // The second half of the criterion: the per-step volume drift must
        // match the REFERENCE's own drift, not merely be small. See the file
        // header -- `removeVolumeFlux` zeroes the rate, not the accumulated
        // truncation, and the Python carries the same truncation.
        //
        // OWNED faces only, then one MPI_Allreduce -- the same convention
        // `enclosedVolume` documents and the same one `initial_volume` was
        // computed under, so the two are comparable.
        //-------------------------------------------------------------------//
        {
            auto pos = mesh.positions();
            auto owned_faces = Kokkos::subview(
                mesh.faceVertices(),
                std::make_pair( 0, mesh.ownedFaceCount() ), Kokkos::ALL() );
            const Real local =
                Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
            Real volume = 0;
            MPI_Allreduce( &local, &volume, 1, MPI_DOUBLE, MPI_SUM,
                           mesh.comm() );
            const double drift =
                static_cast<double>( volume ) / initial_volume - 1.0;
            const double gold_drift = kGoldVolumeDrift[step];
            // Relative to the reference drift where there is one; step 0 is
            // exactly zero on both sides, so compare it absolutely.
            const double deviation =
                gold_drift == 0.0
                    ? std::fabs( drift )
                    : std::fabs( drift / gold_drift - 1.0 );
            std::ostringstream os;
            os.precision( 17 );
            os << "step " << step << " volume " << volume << " relative drift "
               << drift << " reference " << gold_drift << " deviation "
               << deviation << " (rtol " << kVolumeDriftRtol << ", abs cap "
               << kVolumeDriftAbsCap << ")";
            rec.note( os.str() );
            BEATNIK_CHECK_TRUE( rec, deviation <= kVolumeDriftRtol );
            BEATNIK_CHECK_TRUE( rec,
                                std::fabs( drift ) <= kVolumeDriftAbsCap );
        }

        //-------------------------------------------------------------------//
        // The comparison itself. Rank 0 only: the comparator is serial Python
        // over one file, so running it everywhere would be N identical runs
        // racing on stdout.
        //-------------------------------------------------------------------//
        if ( rank == 0 )
        {
            const std::string written = solver.lastCheckpointPath();
            const std::string gold = goldForStep( gold_dir, step );
            BEATNIK_CHECK_TRUE( rec, fileExists( written ) );
            if ( gold.empty() || !fileExists( written ) )
            {
                rec.fail( "step " + std::to_string( step ) +
                          ": missing gold or output file" );
                continue;
            }

            const int status = runComparator( python, script, written, gold );
            std::ostringstream os;
            os << "step " << step << " comparator exit " << status
               << " (0 = match, 1 = compared and disagreed, 2 = LOAD ERROR)";
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, status, 0 );
        }
    }

    //-----------------------------------------------------------------------//
    // `finalize()` writes the LAST FINITE state, which T2d made distinct from
    // "current" for the first time. This run never went non-finite, so the two
    // coincide -- and that is the check: the final file must still match step
    // 10's gold, which it cannot if the record/restore round trip corrupts the
    // fields or the (time, step) pair.
    //-----------------------------------------------------------------------//
    solver.finalize();
    BEATNIK_CHECK_EQ( rec, solver.step(), static_cast<long long>( kSteps ) );
    BEATNIK_CHECK_CLOSE( rec, static_cast<double>( solver.time() ),
                         kGoldTime[kSteps], kTimeRtol );

    if ( rank == 0 )
    {
        const std::string written = solver.lastCheckpointPath();
        const std::string gold = goldForStep( gold_dir, kSteps );
        rec.note( "final (last-finite) checkpoint " + written );
        if ( !gold.empty() && fileExists( written ) )
        {
            const int status = runComparator( python, script, written, gold );
            std::ostringstream os;
            os << "last-finite checkpoint vs step " << kSteps
               << " gold: exit " << status;
            rec.note( os.str() );
            BEATNIK_CHECK_EQ( rec, status, 0 );
        }
        else
        {
            rec.fail( "final checkpoint or its gold file is missing" );
        }
    }

    //-----------------------------------------------------------------------//
    // A NEGATIVE CASE, and it is a real one.
    //
    // T1b's lesson: a check that has only ever seen agreeing data has not been
    // tested. Here the cheapest genuine negative is comparing the final state
    // against the STEP 0 gold file -- same schema, same mesh, same carried
    // scalars, and a different time and different vertex positions. It must exit
    // exactly 1 ("compared and disagreed") and NOT 2 ("could not load"), because
    // accepting 2 is how a negative case passes vacuously.
    //
    // It also proves something specific about this test: that ten steps actually
    // MOVED the surface. A solver whose RHS returned zero would pass every
    // positive comparison above only if the gold set were also static -- it is
    // not -- but this makes the claim directly.
    //-----------------------------------------------------------------------//
    if ( rank == 0 )
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
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_DirectSolve10Steps" );
        try
        {
            // BEATNIK_TEST_EXEC_SPACE is defined by the per-backend shim
            // tests/CMakeLists.txt generates, so the target name's `_SERIAL` /
            // `_HIP` suffix means what the gate's `-R <backend>` filter assumes
            // it means. Defaulting to the default space keeps the file
            // compilable on its own.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec, argc,
                                                                    argv );
        }
        catch ( const std::exception& e )
        {
            // Most likely a BEATNIK_NOT_IMPLEMENTED from a stub on a path this
            // test did not expect to touch. Reported as a named failure rather
            // than allowed to abort, so the tally line still appears in the log.
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
    // job's failure. Without this a launcher that reports only rank 0's status
    // would report success for a run that failed elsewhere -- and the checks
    // above are deliberately not all rank-0's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
