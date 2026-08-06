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
 * @file adaptive_mesh_bubble.cpp
 * @brief Driver for the adaptive-triangle 3D z-model rising bubble.
 *
 * Port of examples/run_adaptive_mesh_bubble.py::main (lines 1195-1652), with
 * the matplotlib figure, the mp4 writer and the plane-section diagnostic
 * removed
 *
 * The option names and defaults match the Python script exactly, so **the same
 * command line drives the Python gold-file run and this one** — which is what
 * makes the regression harness in `tests/regression_tests/` possible. Video and
 * plotting options are accepted and ignored with a warning rather than
 * rejected, for the same reason.
 *
 * The default configuration is README configuration (a), the viscous
 * Rayleigh-Taylor rising bubble:
 *
 *     --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --viscosity-mode laplace-beltrami
 *     --br-approximation treecode --adaptive-dt --dynamic-remesh
 *     --isotropic-cleanup
 *
 * (`--br-approximation treecode` maps to Beatnik's `fmm`; see
 * `Beatnik_BRSolverFMM.hpp`.) The three switches are on by default and are
 * cleared individually with `--no-adaptive-dt`, `--no-dynamic-remesh` and
 * `--no-isotropic-cleanup`. README configuration (b), the pure self-induction
 * roll-up, is `--A 0.0 --g 0.0 --eps 0.05`.
 *
 * CURRENT BEHAVIOR
 * ----------------
 * Every solver body is a stub. A real invocation parses its arguments, prints
 * the resolved configuration, and then dies with a `std::logic_error` naming
 * the first unimplemented routine — which is the intended state of this
 * framework and the thing to check when picking up the next task.
 */

#include <Beatnik_Config.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "InputFile.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace
{

/// Set by the SIGINT handler; polled by the solver loop.
///
/// Port of run_adaptive_mesh_bubble.py::main (lines 1637-1639) — the
/// `KeyboardInterrupt` handler. The Python catches the exception, stops the
/// loop, and still writes the final last-finite checkpoint. A C++ program has
/// no exception to catch, so the equivalent is a flag the loop checks and a
/// `finalize()` that runs regardless.
volatile std::sig_atomic_t g_interrupted = 0;

extern "C" void handleInterrupt( int )
{
    g_interrupted = 1;
}

/// Echo the resolved configuration. Rank 0 only.
///
/// Not a Python port — the Python prints nothing equivalent. It exists because
/// a run with ~140 options and several derived values is otherwise impossible
/// to reconstruct from a log, and because the derived proximity radii are the
/// values a reader most often wants and cannot see on the command line.
void printConfiguration( std::ostream& os,
                         const Beatnik::Example::ClArgs& cl, int comm_size )
{
    const auto& p = cl.solver;
    os << "adaptive_mesh_bubble: Beatnik " << Beatnik_VERSION_STRING << " ("
       << Beatnik_GIT_COMMIT_HASH << ")\n"
       << "  ranks              " << comm_size << "\n"
       << "  execution space    "
       << Kokkos::DefaultExecutionSpace::name() << "\n"
       << "  state model        " << Beatnik::toString( p.state_model ) << "\n"
       << "  mesh               " << Beatnik::toString( p.initial.mesh_kind )
       << ", subdivisions " << p.initial.icosphere_subdivisions
       << ", radius " << p.initial.radius << ", center_z "
       << p.initial.center_z << "\n"
       << "  initial shape      "
       << Beatnik::toString( p.initial.shape ) << "\n"
       << "  physics            A " << p.zmodel.A << ", g " << p.zmodel.g
       << ", eps " << p.zmodel.eps << ", mu " << p.zmodel.mu << ", sigma "
       << p.zmodel.sigma << "\n"
       << "  kernel blob        " << Beatnik::toString( p.zmodel.blob_mode )
       << " (b = " << p.zmodel.blob() << ")\n"
       << "  viscosity          "
       << Beatnik::toString( p.zmodel.viscosity_mode ) << "\n"
       << "  BR                 "
       << Beatnik::toString( p.zmodel.br_approximation ) << ", quadrature "
       << Beatnik::toString( p.zmodel.source_quadrature ) << ", velocity "
       << Beatnik::toString( p.zmodel.velocity_mode ) << "\n"
       << "  Bernoulli scalar   "
       << Beatnik::toString( p.zmodel.bernoulli_scalar_mode ) << "\n"
       << "  volume projection  "
       << ( p.zmodel.preserve_volume ? "on" : "off" ) << "\n"
       << "  time               steps " << p.time.steps << ", dt "
       << p.time.dt << ", adaptive " << ( p.time.adaptive_dt ? "on" : "off" )
       << ", min_dt " << p.time.min_dt;
    if ( p.time.have_t_end ) os << ", t_end " << p.time.t_end;
    os << "\n"
       << "  adaptivity         "
       << ( p.dynamic_remesh ? "dynamic remesh" : "indicator AMR" );
    if ( p.dynamic_remesh )
        os << ", every " << p.remesh_every << ", passes " << p.remesh.passes
           << ", h in [" << p.remesh.h_min << ", " << p.remesh.h_max << "]";
    else
        os << ", every " << p.amr.refine_every << ", max_faces "
           << p.amr.max_faces;
    os << "\n"
       << "  isotropic cleanup  " << ( p.cleanup.enabled ? "on" : "off" )
       << "\n";
    if ( p.remesh_tight_after >= 0.0 )
        os << "  tight remesh after " << p.remesh_tight_after << ", every "
           << p.remesh_tight_every << "\n";
    os << "  proximity          "
       << ( p.remesh.use_proximity ? "on" : "off" )
       << ", activation " << cl.proximity_activation_distance << " or "
       << cl.proximity_activation_factor << " x h0_min"
       << ", material exclusion " << cl.proximity_material_exclusion_radius
       << " or " << cl.proximity_material_exclusion_factor << " x h0_min\n";
    if ( p.checkpoint.writing() )
        os << "  checkpoints        " << p.checkpoint.directory << "/"
           << p.checkpoint.prefix << "_*, every " << p.checkpoint.every_steps
           << " steps / " << p.checkpoint.every_time << " time\n";
    else
        os << "  checkpoints        disabled (no --checkpoint-dir)\n";
    if ( p.checkpoint.restarting() )
        os << "  restart from       " << p.checkpoint.restart_from << "\n";
    os << std::flush;
}

} // namespace

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // MPI first: Kokkos may consult the rank when selecting a GPU.
    MPI_Init( &argc, &argv );

    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    int status = EXIT_SUCCESS;

    // Kokkos is initialized and finalized around an explicit scope so every
    // Kokkos::View created below is destroyed BEFORE Kokkos::finalize(). A
    // View outliving finalize deallocates into a torn-down memory space, which
    // aborts (or worse, corrupts) at exit rather than where the mistake is.
    Kokkos::initialize( argc, argv );
    {
        try
        {
            Beatnik::Example::ClArgs cl;

            // RemeshParams carries the BASELINE defaults, so the tight set
            // must be initialized to the tight defaults before parsing or a
            // run that never passes a --remesh-tight-* option would silently
            // use baseline values in the tight phase.
            Beatnik::Example::applyTightRemeshDefaults(
                cl.solver.remesh_tight );

            Beatnik::Example::parseCommandLine( argc, argv, cl );

            if ( cl.help )
            {
                if ( rank == 0 ) Beatnik::Example::printSchema( std::cout );
                Kokkos::finalize();
                MPI_Finalize();
                return EXIT_SUCCESS;
            }

            // Fill in the values the Python resolves between parse_args() and
            // building its parameter objects. Anything depending on
            // initial_min_edge is deliberately NOT resolved here — the mesh
            // does not exist yet — and is finished by Solver::setup().
            Beatnik::Example::reconcileDerivedParams( cl );

            if ( rank == 0 )
            {
                Beatnik::Example::warnIgnored( cl, std::cerr );
                printConfiguration( std::cout, cl, comm_size );
            }

            std::signal( SIGINT, handleInterrupt );

            using ExecutionSpace = Kokkos::DefaultExecutionSpace;
            using MemorySpace = ExecutionSpace::memory_space;

            Beatnik::Solver<ExecutionSpace, MemorySpace> solver(
                MPI_COMM_WORLD, cl.solver );

            solver.setup();
            const bool completed = solver.solve();

            // Runs unconditionally, including after a non-finite abort or an
            // interrupt, and writes the recorded LAST FINITE state — see
            // Beatnik_Solver.hpp.
            solver.finalize();

            if ( !completed )
            {
                if ( rank == 0 )
                    std::cerr << "run stopped early (non-finite state or "
                                 "interrupt); the final checkpoint holds the "
                                 "last finite state\n";
                status = EXIT_FAILURE;
            }
        }
        catch ( const std::exception& e )
        {
            // Every unimplemented routine throws std::logic_error("<Class>::"
            // "<method> not implemented"), so at this stage of the port that
            // message names the next thing to write.
            if ( rank == 0 )
                std::cerr << "adaptive_mesh_bubble: error: " << e.what()
                          << std::endl;
            status = EXIT_FAILURE;
        }
    }
    Kokkos::finalize();

    // Agree on the exit status so a failure on one rank is a failure of the
    // job. Without this, a launcher that reports only rank 0's status can
    // report success for a run that died elsewhere.
    int global_status = status;
    MPI_Allreduce( &status, &global_status, 1, MPI_INT, MPI_MAX,
                   MPI_COMM_WORLD );

    MPI_Finalize();
    return global_status;
}
