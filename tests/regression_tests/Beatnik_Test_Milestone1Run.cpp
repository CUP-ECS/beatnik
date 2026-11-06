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
 * @file Beatnik_Test_Milestone1Run.cpp
 * @brief **THE MILESTONE-1 MEASUREMENT DRIVER** — runs the reference's default
 *        rising-bubble configuration (M1-G2's command line) and prints one
 *        machine-readable line per step. It compares nothing.
 *
 * THIS IS NOT A TEST AND IS IN NO TIER.
 * ------------------------------------
 * It carries no `LABELS`, no ctest case and no manifest line, so neither the
 * ship gate nor the `milestone` runner can pick it up; it is in
 * `BEATNIK_DRIVER_SOURCES` in `tests/CMakeLists.txt`, beside
 * `Beatnik_Test_Milestone0Run`, for the same mechanical reason: per-backend
 * binaries exist only for test sources, and
 * `examples/02_adaptive_mesh_bubble` is HIP-only
 * (`adaptive_mesh_bubble.cpp:209` fixes the space to
 * `Kokkos::DefaultExecutionSpace`). `Beatnik_Test_Milestone0Run` cannot serve
 * either: it asserts a constant mesh every step and admits no new option.
 *
 * WHAT IT DOES
 * ------------
 * It builds `SolverParams` from `Beatnik::Test::Milestone1::makeParams()`
 * (`Beatnik_Milestone1Params.hpp`, shared with the milestone comparison test),
 * overrides the step count, the checkpoint cadence and the gid-renumbering
 * cadence from its three positionals, and runs the step loop one
 * `advanceOneStep` at a time. Every judgment is made afterwards from its
 * output (task T4d6 in `tasks/milestone1.md`); the only thing it asserts is
 * that the run reached its step budget, reporting an early non-finite stop as
 * a failure rather than as a shorter run.
 *
 * THE PER-STEP LINE
 * -----------------
 * **Every rank** prints, after every step, one line of `key=value` fields:
 *
 *     [m1run] step=<n> time=<t> rank=<r> V=<global vertices> F=<global faces>
 *       gid_space=<high-water> gid_space_after=<after any renumbering>
 *       renumbered=<0|1> gid_ratio=<gid_space / V>
 *       volume_drift=<V/V0 - 1> min_quality=<global min 4√3A/Σℓ²>
 *       splits=<cum> collapses=<cum> flips=<cum> smooth_steps=<cum>
 *       rss_kb=<this rank's VmRSS> rss_peak_kb=<this rank's VmHWM>
 *       elapsed=<s since the loop began, on this rank's clock>
 *
 * (one line in the output; wrapped here). Every field except `rank`, `rss_kb`,
 * `rss_peak_kb` and `elapsed` is global and identical on every rank.
 *
 *   - `gid_space` is `RemeshDiagnostics::gid_space_high_water` of the step's
 *     remesh: the sum over vertices, edges and faces of (global max live gid
 *     + 1), measured after the passes and before any renumbering, so it is
 *     the largest gid space the step reached. `gid_ratio` divides it by `V`.
 *   - `min_quality` is measured after the whole step — remesh, cleanup and
 *     volume projection — on owned faces, with the reference's
 *     `_finite_min(triangle_quality(...))` convention.
 *   - The four counters are the cumulative sums of the remesh pass's
 *     `RemeshDiagnostics::{splits, collapses, flips, smooth_steps}`; the
 *     isotropic cleanup's valence flips are not in `flips`.
 *   - `rss_kb` and `rss_peak_kb` are read from `/proc/self/status`. Peak RSS
 *     per rank is also reported by `/usr/bin/time -v` around this binary
 *     inside `flux run`, which the batch script does. GPU memory is out of
 *     scope.
 *
 * ARGUMENTS. Three required positionals; there is no option surface.
 *
 *   argv[1]  --steps                   600 for the cadence run, 50 for the
 *                                      T4d umbrella
 *   argv[2]  --checkpoint-every-steps  `0` = no checkpoints at all (not even
 *                                      step 0); 10 matches the gold set
 *   argv[3]  gid-renumbering cadence   `RemeshParams::renumber_gids_every`;
 *                                      `0` = never. Pass the
 *                                      `RemeshParams` default (50) for a
 *                                      milestone run.
 *
 * Checkpoints go to
 * `${BEATNIK_TEST_SCRATCH}/m1run_<space>_np<N>_renum<C>`, which **must** be on
 * a parallel filesystem: they go through MPI-IO (CLAUDE.md).
 */

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_Milestone1Params.hpp"
#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
/// A `kB` field of `/proc/self/status` (`VmRSS`, `VmHWM`), or -1 if absent.
long long procStatusKb( const char* key )
{
    std::FILE* f = std::fopen( "/proc/self/status", "r" );
    if ( !f )
        return -1;
    char line[256];
    long long value = -1;
    const std::size_t n = std::strlen( key );
    while ( std::fgets( line, sizeof( line ), f ) )
    {
        if ( std::strncmp( line, key, n ) == 0 && line[n] == ':' )
        {
            value = std::atoll( line + n + 1 );
            break;
        }
    }
    std::fclose( f );
    return value;
}

//---------------------------------------------------------------------------//
/// The global enclosed volume over **owned** faces (risk R9), comparable with
/// `Solver::initialVolume()`.
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
/// Global minimum triangle quality \f$4\sqrt3 A/\sum\ell^2\f$ over owned
/// faces, non-finite entries skipped — the reference's
/// `_finite_min(triangle_quality(...))`, the scale `--remesh-min-quality` is
/// expressed on.
template <class ExecSpace, class MemSpace, class MeshType>
double globalMinQuality( MeshType& mesh )
{
    using view_type =
        Kokkos::View<Real*, Kokkos::Device<ExecSpace, MemSpace>>;
    const int n_owned = mesh.ownedFaceCount();
    view_type quality( "beatnik_m1run_quality", mesh.totalFaceCount() );
    Beatnik::SurfaceOperators::triangleQuality( mesh.positions(),
                                                mesh.faceVertices(), quality );
    auto q = quality;
    Real local = Real( 1.0e300 );
    Kokkos::parallel_reduce(
        "beatnik_m1run_min_quality",
        Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int f, Real& mn ) {
            if ( Kokkos::isfinite( q( f ) ) && q( f ) < mn )
                mn = q( f );
        },
        Kokkos::Min<Real>( local ) );
    Real total = Real( 0 );
    MPI_Allreduce( &local, &total, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
    return static_cast<double>( total );
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
        rec.fail( "usage: <steps> <checkpoint-every-steps> "
                  "<gid-renumber-every>; see the ARGUMENTS block in this "
                  "file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const int steps = std::atoi( argv[1] );
    const int every = std::atoi( argv[2] );
    const int renumber_every = std::atoi( argv[3] );
    if ( steps < 0 || every < 0 || renumber_every < 0 )
    {
        rec.fail( "the three arguments must be non-negative integers" );
        return;
    }

    // Same resolution order as the other drivers: the installed path runs
    // from a read-only prefix, so BEATNIK_TEST_SCRATCH is what the batch
    // script sets. No directory at all when checkpoints are off, so `setup()`
    // writes no step-0 file either.
    std::string dir;
    if ( every > 0 )
    {
        const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
        if ( !scratch_env )
            scratch_env = std::getenv( "TMPDIR" );
        std::ostringstream os;
        os << ( scratch_env ? scratch_env : "." ) << "/m1run_"
           << ExecSpace::name() << "_np" << comm_size << "_renum"
           << renumber_every;
        dir = os.str();
    }

    Beatnik::SolverParams params = Beatnik::Test::Milestone1::makeParams( dir );
    params.time.steps = steps;
    params.checkpoint.every_steps = every;
    params.remesh.renumber_gids_every = renumber_every;

    {
        std::ostringstream os;
        os << "milestone1 run: steps " << steps << ", checkpoint every "
           << every << " step(s), gid renumbering every " << renumber_every
           << " remesh call(s) (RemeshParams default "
           << Beatnik::RemeshParams{}.renumber_gids_every << "), directory '"
           << dir << "'";
        rec.note( os.str() );
    }

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD, params );
    solver.setup();

    auto& mesh = solver.mesh();
    const double initial_volume = static_cast<double>( solver.initialVolume() );
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "initial V " << mesh.globalVertexCount() << ", F "
           << mesh.globalFaceCount() << ", gid space " << mesh.gidSpace()
           << ", initial_volume " << initial_volume << ", initial_min_edge "
           << static_cast<double>( solver.initialMinEdge() );
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // The run. `advanceOneStep` is collective and every rank calls it the
    // same number of times; so is every measurement below.
    //-----------------------------------------------------------------------//
    long long cum_splits = 0;
    long long cum_collapses = 0;
    long long cum_flips = 0;
    long long cum_smooth = 0;
    const int remesh_every = params.remesh_every;

    const double t_start = MPI_Wtime();
    long long completed = 0;
    for ( int step = 1; step <= steps; ++step )
    {
        if ( !solver.advanceOneStep() )
        {
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << steps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time();
            rec.fail( os.str() );
            break;
        }
        completed = solver.step();

        const bool remeshed = remesh_every > 0 && step % remesh_every == 0;
        const Beatnik::RemeshDiagnostics& d = solver.lastRemesh();
        if ( remeshed )
        {
            cum_splits += d.splits;
            cum_collapses += d.collapses;
            cum_flips += d.flips;
            cum_smooth += d.smooth_steps;
        }

        const long long nv = mesh.globalVertexCount();
        const long long nf = mesh.globalFaceCount();
        const double drift = globalVolume( mesh ) / initial_volume - 1.0;
        const double min_q =
            globalMinQuality<ExecSpace, MemSpace>( mesh );
        Kokkos::fence();
        const double elapsed = MPI_Wtime() - t_start;
        const long long hw = static_cast<long long>( d.gid_space_high_water );

        std::printf(
            "[m1run] step=%d time=%.17g rank=%d V=%lld F=%lld gid_space=%lld "
            "gid_space_after=%lld renumbered=%d gid_ratio=%.6f "
            "volume_drift=%.6e min_quality=%.10f splits=%lld collapses=%lld "
            "flips=%lld smooth_steps=%lld rss_kb=%lld rss_peak_kb=%lld "
            "elapsed=%.3f\n",
            step, static_cast<double>( solver.time() ), rank, nv, nf, hw,
            static_cast<long long>( d.gid_space_after ),
            ( remeshed && d.gids_renumbered ) ? 1 : 0,
            nv > 0 ? static_cast<double>( hw ) / static_cast<double>( nv )
                   : 0.0,
            drift, min_q, cum_splits, cum_collapses, cum_flips, cum_smooth,
            procStatusKb( "VmRSS" ), procStatusKb( "VmHWM" ), elapsed );
        std::fflush( stdout );
    }
    Kokkos::fence();
    const double t_total = MPI_Wtime() - t_start;

    solver.finalize();

    // The budget must have been reached; anything less is a reported stop.
    BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( steps ) );

    std::printf( "[m1run] FINAL rank=%d space=%s np=%d steps=%d "
                 "renumber_every=%d completed=%lld wall=%.6f s_per_step=%.6f "
                 "splits=%lld collapses=%lld flips=%lld smooth_steps=%lld "
                 "rss_peak_kb=%lld\n",
                 rank, ExecSpace::name(), comm_size, steps, renumber_every,
                 completed, t_total, steps > 0 ? t_total / steps : 0.0,
                 cum_splits, cum_collapses, cum_flips, cum_smooth,
                 procStatusKb( "VmHWM" ) );
    std::fflush( stdout );
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_Milestone1Run" );
        try
        {
            // Defined by the per-backend shim tests/CMakeLists.txt generates.
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

    // ONE VERDICT ACROSS THE RANKS: MPI_MAX makes any rank's failure the job's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
