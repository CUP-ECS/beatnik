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
 * @file Beatnik_Test_InitialConditions.cpp
 * @brief **REGRESSION TEST 1** — the whole driver path at 0 timesteps, compared
 *        against the T1a Python gold checkpoint.
 *
 * THIS IS THE FIRST MEMBER OF THE SHIP GATE.
 * `tasks/framework.md` states T1c's exit criterion as: *regression test 1
 * passes at 0 timesteps; `compare_output.py beatnik.h5 gold.npz --rtol 1e-12
 * --atol 1e-14` exits 0, at ranks 1, 2 and 4.* Before this file the
 * `regression` tier was empty and a green gate proved nothing (89ec015 removed
 * the pre-redesign solver and its only end-to-end test). It is `regression`
 * and not `unit` because it composes the whole pipeline that exists today —
 * `InitialCondition::build` -> `RestartReader::coldStart` -> `Solver::setup` ->
 * `solve()` -> `finalize()` -> `CheckpointIO::write` -> the comparator — rather
 * than exercising one component.
 *
 * WHY THE TEST SHELLS OUT TO PYTHON
 * --------------------------------
 * The criterion names `compare_output.py`, and that is not incidental: the
 * comparator does the one thing a C++ check cannot do cheaply, which is
 * **recover the vertex correspondence** between two independently ordered
 * meshes (quantize, then lexsort — see that script's module docstring).
 * Reimplementing it here would be a second implementation of the hard part, and
 * the two would drift. So this binary produces the `.h5` and then judges itself
 * by the comparator's exit status.
 *
 * WHY THE VERDICT IS THE BINARY'S OWN
 * -----------------------------------
 * In spack mode there is no build tree and therefore no ctest, so a directly
 * launched binary is judged by its exit code and nothing else (CLAUDE.md,
 * "Running tests in `spack` mode"). Exit 0 iff **every** check passed on
 * **every** rank *and at least one check ran* — the same contract T1b
 * established, with the cross-rank reduction added; see
 * `Beatnik_TestAssert.hpp`.
 *
 * THE NEGATIVE CASE, AND THE T1b TRAP IT AVOIDS
 * ---------------------------------------------
 * A test that has only ever seen matching data has not been tested. T1b found
 * the sharp edge here the hard way: a `WILL_FAIL`-style case can pass
 * **vacuously**, because a missing file also exits non-zero, and the failure is
 * invisible in a green log. So this test runs the comparator a *second* time,
 * against the deliberately mismatched `synthetic_gold.npz` fixture, and
 * requires
 *
 *     exit status == 1, exactly — NOT merely non-zero.
 *
 * `compare_output.py` returns **1** for a comparison failure and **2** for a
 * `LoadError` (missing file, unreadable, `FIELD_MAP` drift). Demanding exactly
 * 1 is what makes the negative case prove the comparator *compared and
 * disagreed* rather than never having opened a file. Every input path is
 * additionally checked for existence before use, so a mis-plumbed path is
 * reported as itself.
 *
 * ARGUMENTS. All paths; see tests/CMakeLists.txt for both call sites, which
 * pass them absolute (ctest) and manifest-relative (the installed gate).
 *
 *   argv[1]  the T1a gold checkpoint
 *              regression_tests/initial_conditions/gold.npz
 *   argv[2]  the comparator
 *              regression_tests/compare_output.py
 *   argv[3]  the deliberately mismatched fixture
 *              regression_tests/fixtures/synthetic_gold.npz
 *
 * `BEATNIK_PYTHON` overrides the interpreter (default `python3`).
 */

#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <sys/stat.h>
#include <sys/wait.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// T1a reference values and configuration.
//
// Produced by the Python driver, `tasks/framework.md` task T1a (met
// 2026-08-07):
//
//   python examples/run_adaptive_mesh_bubble.py
//       --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
//       --viscosity-mode laplace-beltrami --br-approximation direct
//       --isotropic-cleanup --checkpoint-every-steps 1 --no-video
//       --steps 0 --source-quadrature vertex
//
// The parameters set in `makeParams()` below reproduce that command line. Only
// the options that differ from the Beatnik defaults are set, and each is
// commented with the flag it stands for, so a reader can check the
// correspondence without reading the Python.
//
// `--source-quadrature vertex` is not optional even at 0 timesteps, where the
// quadrature is never evaluated: risk R11 in `tasks/framework.md` records that
// every gold file is generated that way so the trap cannot bite at T2a.
//---------------------------------------------------------------------------//
constexpr double kInitialVolume = 6.3235073124669514e-02;
constexpr double kInitialMinEdge = 6.8976121063816842e-02;

/// The comparator's tolerance, and therefore this test's. **Do not loosen**
/// without the measurement and the justification `tasks/framework.md` R2
/// demands; the two carried scalars are the only quantities for which a
/// cross-rank relaxation is even arguable, and the vertex and face fields are
/// never candidates.
constexpr const char* kRtol = "1e-12";
constexpr const char* kAtol = "1e-14";

/// The same numbers, for this test's own checks on the two scalars. Applied on
/// top of the comparator's identical check, because a failure here names the
/// scalar and prints 17 digits at every rank count, which localizes an R2/R9
/// question far faster than the comparator's field table.
constexpr double kScalarRtol = 1.0e-12;

// The Python's defaults, from `run_adaptive_mesh_bubble.py::parse_args`:
// --icosphere-subdivisions 2, --radius 0.25, --center-z 0.25.
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

// Closed-form counts for a subdivision-2 icosphere: V = 10*4^s + 2,
// F = 20*4^s, E = 3F/2. Euler: V - E + F = 2.
constexpr long long kVertices = 162;
constexpr long long kFaces = 320;
constexpr long long kEdges = 480;

//---------------------------------------------------------------------------//
bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// Run `python <script> <a> <b> --rtol .. --atol ..` and return its exit
/// status, or -1 if it could not be run at all.
///
/// The two are distinguished on purpose: "the comparator ran and disagreed" and
/// "the comparator never ran" are different findings, and conflating them is
/// how a negative case passes vacuously (see the file header).
int runComparator( const std::string& python, const std::string& script,
                   const std::string& lhs, const std::string& rhs )
{
    std::ostringstream cmd;
    cmd << "'" << python << "' '" << script << "' '" << lhs << "' '" << rhs
        << "' --rtol " << kRtol << " --atol " << kAtol;
    std::printf( "[cmd] %s\n", cmd.str().c_str() );
    std::fflush( stdout );

    const int raw = std::system( cmd.str().c_str() );
    if ( raw == -1 || !WIFEXITED( raw ) )
        return -1;
    return WEXITSTATUS( raw );
}

//---------------------------------------------------------------------------//
/// The T1a command line, as a `SolverParams`.
Beatnik::SolverParams makeParams( const std::string& checkpoint_dir )
{
    Beatnik::SolverParams p;

    // --state-model potential (the default), --mesh-kind icosphere (default),
    // --icosphere-subdivisions 2, --radius 0.25, --center-z 0.25.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = kSubdivisions;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0:
    // the fast path, which is all T1c implements. Set explicitly rather than
    // left to the defaults, because a later change to a default must break this
    // test loudly rather than silently change what it compares.
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
    // --viscosity-mode laplace-beltrami
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    // --br-approximation direct. Not `fmm`: this must not depend on Canopy
    // (T3a), and at 0 timesteps the BR evaluator is constructed but never
    // called.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex. See R11.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // --steps 0. This is what makes the run compare initial conditions only.
    p.time.steps = 0;

    // --isotropic-cleanup
    p.cleanup.enabled = true;

    // --checkpoint-every-steps 1, and the directory the run writes into.
    p.checkpoint.every_steps = 1;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

    return p;
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    if ( argc < 4 )
    {
        rec.fail( "usage: <gold.npz> <compare_output.py> <synthetic_gold.npz>; "
                  "see the ARGUMENTS block in this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const std::string gold = argv[1];
    const std::string script = argv[2];
    const std::string mismatched = argv[3];
    const char* python_env = std::getenv( "BEATNIK_PYTHON" );
    const std::string python = python_env ? python_env : "python3";

    // Every input path is checked BEFORE it is used. A missing file makes the
    // comparator exit 2, which the negative case below would otherwise accept
    // as "non-zero, therefore failed" -- the T1b trap.
    if ( rank == 0 )
    {
        BEATNIK_CHECK_TRUE( rec, fileExists( gold ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( script ) );
        BEATNIK_CHECK_TRUE( rec, fileExists( mismatched ) );
    }

    //-----------------------------------------------------------------------//
    // Run the solver.
    //
    // The output directory is unique per (rank count, execution space) so the
    // gate's rank sweep -- and a parallel `ctest` -- cannot have two cases
    // writing the same file.
    //-----------------------------------------------------------------------//
    // Resolution order, and why there are three levels: the installed gate path
    // runs from the manifest's directory, which is inside a spack install
    // prefix and is READ-ONLY -- so a relative default would fail there and
    // nowhere else. `BEATNIK_TEST_SCRATCH` is what the gate wrapper sets
    // (absolute); TMPDIR covers a hand-run from an install prefix; "." covers
    // ctest, which runs in the build tree.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/beatnik_regression_t1c/"
        << ExecSpace::name() << "_np" << comm_size;

    std::string checkpoint_path;
    {
        Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD,
                                                     makeParams( dir.str() ) );

        solver.setup();

        // `--steps 0`, so this returns immediately and reports a completed
        // budget. A false here would mean the guard changed meaning.
        BEATNIK_CHECK_TRUE( rec, solver.solve() );

        solver.finalize();

        checkpoint_path = solver.lastCheckpointPath();

        const auto& mesh = solver.mesh();

        //-------------------------------------------------------------------//
        // Structure. Reduced as integers, so exact at every rank count.
        //-------------------------------------------------------------------//
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        // Risk R8: the two-ring RHS needs depth 2, set once at construction.
        BEATNIK_CHECK_EQ(
            rec, mesh.haloDepth(),
            ( Beatnik::SurfaceMesh<ExecSpace, MemSpace>::halo_depth ) );

        //-------------------------------------------------------------------//
        // R9 DISCRIMINATOR 1 — do the owned sets PARTITION the global sets?
        //
        // Summed here with a plain MPI_Allreduce over `ownedXCount()` rather
        // than read from Tessera's `globalOwnedX`, deliberately: the two are
        // independent paths to the same number, and it is exactly the
        // owned-vs-local distinction that R9's double-counting hinges on. If
        // these agree with 162/480/320 then the owned ranges the volume and
        // edge-length reductions were handed cover the global sets exactly
        // once, which is the precondition those reductions need and the
        // assumption R9 says to check rather than make.
        //-------------------------------------------------------------------//
        long long owned[3] = { mesh.ownedVertexCount(), mesh.ownedEdgeCount(),
                               mesh.ownedFaceCount() };
        long long owned_total[3] = { 0, 0, 0 };
        MPI_Allreduce( owned, owned_total, 3, MPI_LONG_LONG, MPI_SUM,
                       mesh.comm() );
        {
            std::ostringstream os;
            os << "owned partition: sum over ranks V " << owned_total[0]
               << " E " << owned_total[1] << " F " << owned_total[2]
               << "; this rank owns V " << owned[0] << " E " << owned[1]
               << " F " << owned[2] << " of local V " << mesh.totalVertexCount()
               << " E " << mesh.totalEdgeCount() << " F "
               << mesh.totalFaceCount();
            rec.note( os.str() );
        }
        BEATNIK_CHECK_EQ( rec, owned_total[0], kVertices );
        BEATNIK_CHECK_EQ( rec, owned_total[1], kEdges );
        BEATNIK_CHECK_EQ( rec, owned_total[2], kFaces );

        //-------------------------------------------------------------------//
        // The two carried scalars, against T1a.
        //
        // Both are what the whole run keys off (`Beatnik_Restart.hpp`), and
        // both are the quantities risks R2 and R9 disagree about. Reported to
        // 17 digits with the relative error, at every rank count, WHETHER OR
        // NOT they pass -- that measurement is what tells a later reader
        // whether a cross-rank difference is summation order (last few ulp, no
        // trend with rank count) or ghost inclusion (orders larger, scaling
        // with the ghost fraction).
        //-------------------------------------------------------------------//
        const double volume = static_cast<double>( solver.initialVolume() );
        const double h_min = static_cast<double>( solver.initialMinEdge() );
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "initial_volume   " << volume << " vs T1a " << kInitialVolume;
            os.precision( 3 );
            os << "  rel "
               << std::fabs( volume - kInitialVolume ) / kInitialVolume;
            rec.note( os.str() );
        }
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "initial_min_edge " << h_min << " vs T1a " << kInitialMinEdge;
            os.precision( 3 );
            os << "  rel "
               << std::fabs( h_min - kInitialMinEdge ) / kInitialMinEdge;
            rec.note( os.str() );
        }
        BEATNIK_CHECK_CLOSE( rec, volume, kInitialVolume, kScalarRtol );
        BEATNIK_CHECK_CLOSE( rec, h_min, kInitialMinEdge, kScalarRtol );

        //-------------------------------------------------------------------//
        // R9 DISCRIMINATOR 2 — the closed form.
        //
        // The enclosed volume of a sphere of this radius is 4*pi*R^3/3; the
        // inscribed subdivision-2 icosphere is smaller by a fixed polyhedral
        // deficit, so the RATIO is a constant of the mesh and not of the
        // partition. Reported, not asserted against a literal: its value is a
        // property of the triangulation, and the assertion that matters is the
        // one against T1a above. What it is for is scale -- a ghost-inclusion
        // bug inflates the sum by the ghost fraction, which at these rank
        // counts is tens of percent and unmistakable here, while a
        // summation-order difference does not move this ratio at all in the
        // digits printed.
        //-------------------------------------------------------------------//
        {
            const double sphere = 4.0 * M_PI * std::pow( kRadius, 3 ) / 3.0;
            std::ostringstream os;
            os.precision( 17 );
            os << "volume / (4*pi*R^3/3) = " << volume / sphere
               << " (polyhedral deficit; partition-independent)";
            rec.note( os.str() );
        }
    }

    //-----------------------------------------------------------------------//
    // THE EXIT CRITERION. Rank 0 only: the comparator is serial Python over one
    // file, so running it on every rank would be N identical runs racing on
    // stdout.
    //-----------------------------------------------------------------------//
    if ( rank == 0 )
    {
        rec.note( "checkpoint written to " + checkpoint_path );
        BEATNIK_CHECK_TRUE( rec, fileExists( checkpoint_path ) );

        // Positive case: this IS T1c's exit criterion.
        const int positive =
            runComparator( python, script, checkpoint_path, gold );
        std::ostringstream os;
        os << "comparator vs T1a gold: exit " << positive << " (0 = match)";
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, positive, 0 );

        // Negative case: exactly 1, not merely non-zero. See the file header --
        // 1 is "compared and disagreed", 2 is "could not load", and accepting 2
        // is how this case would pass without testing anything.
        const int negative =
            runComparator( python, script, checkpoint_path, mismatched );
        std::ostringstream os2;
        os2 << "comparator vs deliberately mismatched fixture: exit "
            << negative
            << " (1 = detected a mismatch, 2 = LOAD ERROR and therefore a "
               "vacuous pass)";
        rec.note( os2.str() );
        BEATNIK_CHECK_EQ( rec, negative, 1 );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_InitialConditions" );
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
            // than allowed to abort, so the tally line still appears in the
            // log.
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
    // the job's failure. Without this a launcher that reports only rank 0's
    // status would report success for a run that failed elsewhere -- and the
    // checks above are deliberately not all rank-0's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
