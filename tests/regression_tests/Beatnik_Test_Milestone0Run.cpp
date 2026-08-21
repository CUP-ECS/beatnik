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
 * @file Beatnik_Test_Milestone0Run.cpp
 * @brief **M0-D1's MEASUREMENT DRIVER** — runs the milestone-0 frozen-mesh
 *        configuration and writes checkpoints. It compares nothing.
 *
 * THIS IS NOT A TEST AND IS IN NO TIER.
 * ------------------------------------
 * It carries no `LABELS`, no ctest case and no manifest line, so neither
 * `run_regression_minset.flux` (the ship gate) nor `run_milestone.flux` can
 * pick it up; see the registration loop in `tests/CMakeLists.txt`, which is the
 * milestone tier's loop stopped short of the point where it applies a label.
 * The reason it lives here anyway is mechanical and is M0-D1's: per-backend
 * binaries exist only for **test sources**, through the generated translation
 * unit that pins `BEATNIK_TEST_EXEC_SPACE`, and
 * `examples/02_adaptive_mesh_bubble` cannot answer for SERIAL at all —
 * `adaptive_mesh_bubble.cpp:209` fixes the space to
 * `Kokkos::DefaultExecutionSpace` and the installed binary is `+rocm`, so that
 * example is HIP-only. M0-D1's matrix is (level 3, level 4) x (SERIAL, HIP) x
 * (ranks 1, 4), and half of it has no other driver.
 *
 * WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
 * ----------------------------------------------
 * It builds `SolverParams` for the M0-G1/M0-G2 command line, runs the step loop
 * one `advanceOneStep` at a time, and lets `Solver` write its own checkpoints on
 * the `--checkpoint-every-steps` cadence. **Every comparison M0-D1 makes is
 * offline in Python**, in `milestone0_ladder.py`, against these checkpoints —
 * so nothing here loads a gold file, spawns `compare_output.py`, or knows a
 * tolerance. That split is the point: the tolerance ladder is built by
 * re-reading one run's output at five tolerances, and a driver that compared
 * as it ran would have to re-run the trajectory to answer each of them.
 *
 * It does assert three things, because they are cheap here and unrecoverable
 * afterwards:
 *
 *   1. The global vertex and face counts, **every step**. M0-D1's exit criterion
 *      fails outright if either changes: `--no-dynamic-remesh --refine-every 0`
 *      is the whole configuration, and a change would mean adaptivity leaked
 *      into the run that exists to exclude it. `compare_output.py` would also
 *      catch it structurally, but only for the steps that have a gold file and
 *      only after the run finished.
 *   2. The two carried scalars, at 17 digits, so the level's
 *      `initial_min_edge` — which every adaptive dt of the run scales off — is
 *      in the log beside the numbers it produced.
 *   3. That the run reached its step budget. A run that goes non-finite stops
 *      early (M0-R2 / M0-R6); `Solver::advanceOneStep` returns false and the
 *      stop step is REPORTED as a failure here rather than left to look like a
 *      shorter successful sweep.
 *
 * Wall time is printed per progress line and in total, from `MPI_Wtime` on rank
 * 0. That is the driver's own measure and it excludes launch overhead; the
 * batch script clocks the whole launch separately, and peak resident memory per
 * rank comes from `/usr/bin/time -v` wrapped around this binary inside
 * `flux run`. **GPU-side memory is out of scope for M0-D1** — there is no
 * mechanism for it here.
 *
 * ARGUMENTS. Three positionals, all required and all integers; there is no
 * option surface here and none may be added (milestone0.md Conventions, "CLI
 * surface: unchanged").
 *
 *   argv[1]  --icosphere-subdivisions   the level under test (2, 3 or 4)
 *   argv[2]  --steps                    2000 for the sweep, 0 for the step-0
 *                                       generator gate of M0-D1 step 1
 *   argv[3]  --checkpoint-every-steps   25, matching both gold sets
 *
 * Output goes to `${BEATNIK_TEST_SCRATCH}/sub<L>_<space>_np<N>` — a subdirectory
 * even though the batch script already hands each run its own scratch, so that
 * two runs sharing one scratch by mistake cannot overwrite each other's
 * checkpoints and be read back as one series. `BEATNIK_TEST_SCRATCH` **must** be
 * on a parallel filesystem: the checkpoints go through MPI-IO (CLAUDE.md).
 */

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The M0-G1 / M0-G2 configuration.
//
//   python examples/run_adaptive_mesh_bubble.py \
//     --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
//     --viscosity-mode laplace-beltrami --br-approximation direct \
//     --adaptive-dt --no-dynamic-remesh --refine-every 0 \
//     --source-quadrature vertex \
//     --icosphere-subdivisions <L> --steps 2000 \
//     --checkpoint-every-steps 25 --no-video --checkpoint-dir results<L>
//
// This is T2d's `makeParams()` field for field, with exactly the three
// differences M0-D1's task entry names: the subdivision level, the step count
// and the checkpoint cadence, all three of which arrive as arguments. Every
// other value is set EXPLICITLY rather than inherited from a Beatnik default,
// for the reason T2d gives: a later change to a default must break this loudly
// instead of silently changing what was measured.
//---------------------------------------------------------------------------//
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

/// Entity counts of the subdivision-L icosphere: `V = 10*4^L + 2`,
/// `F = 20*4^L`, `E = 30*4^L`. Computed rather than tabulated, so the level
/// stays an argument.
long long verticesForLevel( int level )
{
    long long f = 1;
    for ( int i = 0; i < level; ++i )
        f *= 4;
    return 10 * f + 2;
}
long long facesForLevel( int level )
{
    long long f = 1;
    for ( int i = 0; i < level; ++i )
        f *= 4;
    return 20 * f;
}

//---------------------------------------------------------------------------//
/// The milestone-0 command line, as a `SolverParams`.
Beatnik::SolverParams makeParams( int subdivisions, int steps,
                                  int checkpoint_every,
                                  const std::string& checkpoint_dir )
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere, --radius 0.25,
    // --center-z 0.25, --icosphere-subdivisions <L>.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = subdivisions;
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
    // introduce an approximation error the comparison cannot separate from the
    // divergence this task measures.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // --steps <N>, --adaptive-dt, and the dt controls the gold sets were
    // generated under. Every one is a Python default and every one changes the
    // trajectory.
    p.time.steps = steps;
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

    // --isotropic-cleanup is on by default and is moot with remeshing off.
    p.cleanup.enabled = true;

    // --checkpoint-every-steps <N>. `setup()` writes step 0 unconditionally, so
    // 2000 steps every 25 gives the gold sets' own 81 files.
    p.checkpoint.every_steps = checkpoint_every;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

    return p;
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runDriver( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    if ( argc < 4 )
    {
        rec.fail( "usage: <icosphere-subdivisions> <steps> "
                  "<checkpoint-every-steps>; see the ARGUMENTS block in this "
                  "file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const int level = std::atoi( argv[1] );
    const int steps = std::atoi( argv[2] );
    const int every = std::atoi( argv[3] );
    if ( level < 0 || steps < 0 || every < 0 )
    {
        rec.fail( "all three arguments must be non-negative integers" );
        return;
    }

    // Same resolution order, and the same three levels, as the regression
    // tests': the installed path runs from a read-only spack prefix, so "." is
    // not writable there and BEATNIK_TEST_SCRATCH is what the batch script sets.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/sub" << level << "_"
        << ExecSpace::name() << "_np" << comm_size;

    {
        std::ostringstream os;
        os << "milestone0 run: subdivisions " << level << ", steps " << steps
           << ", checkpoint every " << every << " step(s), directory "
           << dir.str();
        rec.note( os.str() );
    }

    const long long want_vertices = verticesForLevel( level );
    const long long want_faces = facesForLevel( level );

    Beatnik::Solver<ExecSpace, MemSpace> solver(
        MPI_COMM_WORLD, makeParams( level, steps, every, dir.str() ) );
    solver.setup();

    auto& mesh = solver.mesh();

    //-----------------------------------------------------------------------//
    // Structure, before anything evolves. The generator's counts are the run's
    // counts for its whole length; anything else is adaptivity leaking in.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), want_vertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), want_faces );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

    //-----------------------------------------------------------------------//
    // The two carried scalars, at 17 digits. `initial_min_edge` is what the
    // adaptive dt is relative to, and it is why the two levels do NOT reach the
    // same physical time after 2000 steps -- so every cross-level comparison
    // M0-D1 makes is by STEP, never by time (M0-G1/M0-G2's log entry).
    //-----------------------------------------------------------------------//
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "initial_volume " << static_cast<double>( solver.initialVolume() )
           << ", initial_min_edge "
           << static_cast<double>( solver.initialMinEdge() );
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // The run. Driven one step at a time rather than through `solve()`, for one
    // reason only: the per-step count check above has to happen per step.
    // `advanceOneStep` is collective and every rank calls it the same number of
    // times -- the BR ring deadlocks otherwise (T2c).
    //-----------------------------------------------------------------------//
    const double t_start = MPI_Wtime();
    long long completed = 0;
    for ( int step = 1; step <= steps; ++step )
    {
        if ( !solver.advanceOneStep() )
        {
            // M0-R2 / M0-R6. A stop is a REPORTED stop step, never a shorter
            // pass: it is the compare-depth ceiling on Beatnik's side and M0-A1
            // needs the number.
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << steps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time();
            rec.fail( os.str() );
            break;
        }
        completed = solver.step();

        // Cheap, integer, and reduced inside Tessera: safe to do every step.
        if ( mesh.globalVertexCount() != want_vertices ||
             mesh.globalFaceCount() != want_faces )
        {
            std::ostringstream os;
            os << "ENTITY COUNTS CHANGED at step " << step << ": vertices "
               << mesh.globalVertexCount() << " (expected " << want_vertices
               << "), faces " << mesh.globalFaceCount() << " (expected "
               << want_faces << "). Adaptivity leaked into the frozen-mesh "
               << "configuration.";
            rec.fail( os.str() );
            break;
        }

        if ( step % 100 == 0 || step == steps )
        {
            Kokkos::fence();
            const double elapsed = MPI_Wtime() - t_start;
            std::ostringstream os;
            os.precision( 17 );
            os << "step " << step << " time " << solver.time();
            os.precision( 6 );
            os << "  elapsed " << elapsed << " s  (" << ( elapsed / step )
               << " s/step)";
            rec.note( os.str() );
        }
    }
    Kokkos::fence();
    const double t_total = MPI_Wtime() - t_start;

    solver.finalize();

    //-----------------------------------------------------------------------//
    // The budget must have been reached. `steps` is what the gold sets ran, so
    // anything less is a ceiling M0-A1 has to know about.
    //-----------------------------------------------------------------------//
    BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( steps ) );
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), want_vertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), want_faces );

    if ( rank == 0 )
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "FINAL step " << solver.step() << " time " << solver.time();
        os.precision( 6 );
        os << "  solve wall " << t_total << " s";
        if ( steps > 0 )
            os << "  (" << ( t_total / steps ) << " s/step)";
        os << "  last checkpoint " << solver.lastCheckpointPath();
        rec.note( os.str() );

        // One machine-greppable line per run, so the batch log can be reduced
        // to a table without parsing the prose above.
        std::printf( "[m0d1] TIMING level=%d space=%s np=%d steps=%d "
                     "wall=%.6f s_per_step=%.6f\n",
                     level, ExecSpace::name(), comm_size, steps, t_total,
                     steps > 0 ? t_total / steps : 0.0 );
        std::fflush( stdout );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_Milestone0Run" );
        try
        {
            // Defined by the per-backend shim tests/CMakeLists.txt generates, so
            // the binary's `_SERIAL` / `_HIP` suffix means what the batch script
            // assumes it means. Defaulting keeps the file compilable alone.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runDriver<ExecSpace, typename ExecSpace::memory_space>( rec, argc,
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

    // ONE VERDICT ACROSS THE RANKS, as every standalone test here does: each
    // rank printed its own tally, MPI_MAX makes any rank's failure the job's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
