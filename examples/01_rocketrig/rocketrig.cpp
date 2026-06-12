/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/*
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jacob McCullough <jmccullough12@unm.edu>
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * General rocket rig fluid interface example using the Beatnik z-model
 * fluid interface solver.
 */


#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef MEASURETIME
#define MEASURETIME 0
#endif


// Include Statements
#include <Beatnik_Config.hpp>
#include <BoundaryCondition.hpp>
#include <Solver.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

#if DEBUG
#include <iostream>
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stdlib.h>

using namespace Beatnik;

enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};
enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};

/**
 * @struct ClArgs
 * @brief Holds the resolved run-time parameters for one rocketrig run.
 *
 * Populated from the input file (see InputFile.hpp). Kept as a flat
 * aggregate so the post-parse derivations (bounding-box array,
 * gravity Gs->m/s^2, weak-scale, auto delta_t / t_final) and the
 * rank-0 banner can both walk it without indirection.
 */
struct ClArgs
{
    /* Problem physical setup */
    enum InitialConditionModel initial_condition; /**< Model used to set initial conditions */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of scale of initial interface */
    double variation; /**< Variation in scale of initial interface */
    enum Beatnik::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */
    double bounding_box; /**< Size of global bounding box. From (-B, -B, -B) to (B, B, B) */

    /* Problem simulation parameters */
    std::array<int, 2> num_nodes;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    int weak_scale;     /**< Amount to scale up resulting problem */

    /* I/O parameters */
    int write_freq;     /**< Write frequency */

    /* Solution method constants */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */

    /* Solver-order / BR-solver type, FMM tunables, etc. */
    Params params;
};

#include "InputFile.hpp"

/* Print the input-file schema (rank 0 only). */
void help( const int rank )
{
    if ( rank == 0 )
        Beatnik::Example::printSchema( std::cout );
}

/* Populate `cl` with built-in defaults, then overlay the contents of
 * `path` on top, then derive secondary quantities (bounding-box
 * array, gravity unit conversion, weak-scale multipliers, and the
 * auto values for delta_t / t_final).
 *
 * Returns 0 on success. On error, prints a `rocketrig: <message>`
 * diagnostic on rank 0 and returns a non-zero code so main() can
 * finalize cleanly. */
int parseInput( const int rank, const std::string& path, ClArgs& cl )
{
    /* --- defaults ---------------------------------------------------
     * Match the historical getopt defaults so a near-empty input file
     * reproduces the previous "no flags" run. */
    cl.weak_scale = 1;
    cl.write_freq = 10;

    cl.params.cutoff_distance      = 0.5;
    cl.params.heffte_configuration = 6;
    cl.params.br_solver            = BR_EXACT;
    cl.params.solver_order         = SolverOrder::ORDER_LOW;
    cl.params.period               = 1.0;

    cl.num_nodes        = { 128, 128 };
    cl.bounding_box     = 1.0;
    cl.initial_condition = IC_COS;
    cl.boundary         = Beatnik::BoundaryType::PERIODIC;
    cl.tilt             = 0.0;
    cl.magnitude        = 0.05;
    cl.variation        = 0.0;
    cl.gravity          = 25.0;
    cl.atwood           = 0.5;

    /* Z-Model defaults, scaled by the solver to sqrt(dx*dy). */
    cl.mu  = 1.0;
    cl.eps = 0.25;

    /* Sentinels: <= 0 means "auto" — derived after the file is read. */
    cl.delta_t = -1.0;
    cl.t_final = -1.0;

    try
    {
        Beatnik::Example::parseInputFile( path, cl );
    }
    catch ( const std::exception& e )
    {
        if ( rank == 0 )
            std::cerr << "rocketrig: " << e.what() << "\n";
        return -1;
    }

    /* --- post-parse derivations ------------------------------------- */
    cl.params.global_bounding_box = { -cl.bounding_box, -cl.bounding_box,
                                      -cl.bounding_box,
                                       cl.bounding_box,  cl.bounding_box,
                                       cl.bounding_box };
    cl.gravity *= 9.81;

    /* Scale up global bounding box and node count by the weak-scale
     * factor (linear in sqrt(weak_scale) so total work scales linearly). */
    const double s = std::sqrt( static_cast<double>( cl.weak_scale ) );
    for ( int i = 0; i < 6; ++i )
        cl.params.global_bounding_box[i] *= s;
    for ( int i = 0; i < 2; ++i )
        cl.num_nodes[i] = static_cast<int>( cl.num_nodes[i] * s );

    /* Characteristic period of the interface. */
    const double tau = 1.0 / std::sqrt( cl.atwood * cl.gravity );

    if ( cl.delta_t <= 0.0 )
    {
        cl.delta_t = ( cl.params.solver_order == SolverOrder::ORDER_HIGH )
                     ? tau / 50.0
                     : tau / 25.0;
    }

    if ( cl.t_final <= 0.0 )
    {
        /* 2 characteristic periods — what the low-order model can
         * faithfully resolve. */
        cl.t_final = 2.0 * tau;
    }
    else
    {
        /* Historical semantics: the `timesteps` key holds a count of
         * steps; convert to physical end-time. */
        cl.t_final = cl.t_final * cl.delta_t;
    }

    return 0;
}

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    // Initialize Variables

    MeshInitFunc( std::array<double, 6> box, enum InitialConditionModel i,
                  double t, double m, double v, double p,
                  const std::array<int, 2> nodes, enum Beatnik::BoundaryType boundary )
        : _i(i)
        , _t( t )
        , _m( m )
        , _v( v)
        , _p( p )
        , _b( boundary )
    {
	    _ncells[0] = nodes[0] - 1;
        _ncells[1] = nodes[1] - 1;

        _dx = (box[3] - box[0]) / _ncells[0];
        _dy = (box[4] - box[1]) / _ncells[1];


    };

    template <class RandNumGenType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Position,
                     RandNumGenType random_pool,
                     [[maybe_unused]] const int index[2],
                     const double coord[2],
                     double &z1, double &z2, double &z3) const
    {
        double lcoord[2];
        /* Compute the physical position of the interface from its global
         * coordinate in mesh space */
        for (int i = 0; i < 2; i++) {
            lcoord[i] = coord[i];
            if (_b == BoundaryType::FREE && (_ncells[i] % 2 == 1) ) {
                lcoord[i] += 0.5;
            }
        }
        z1 = _dx * lcoord[0];
        z2 = _dy * lcoord[1];

        // We don't currently support tilting the initial interface

        /* Need to initialize these values here to avoid "jump to case label "case IC_FILE:"
         * crosses initialization of ‘double gaussian’, etc." errors */
        auto generator = random_pool.get_state();
        double rand_num = generator.drand(-1.0, 1.0);
        double mean = 0.0;
        double std_dev = 1.0;
        double gaussian = (1 / (std_dev * Kokkos::sqrt(2 * Kokkos::numbers::pi_v<double>))) *
            Kokkos::exp(-0.5 * Kokkos::pow(((rand_num - mean) / std_dev), 2));
        switch (_i) {
        case IC_COS:
            z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
            break;
        case IC_SECH2:
            z3 = _m * pow(1.0 / cosh(_p * (z1 * z1 + z2 * z2)), 2);
            break;
        case IC_RANDOM:
            z3 = _m * (2*rand_num - 1.0);
            break;
        case IC_GAUSSIAN:
            /* The built-in C++ std::normal_distribution<double> doesn't
             * work here, so coding the gaussian distribution itself.
             */
            z3 = _m * gaussian;
            break;
        case IC_FILE:
            break;
        }

        random_pool.free_state(generator);

        return true;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Vorticity,
                     [[maybe_unused]] const int index[2],
                     [[maybe_unused]] const double coord[2],
                     double& w1, double &w2 ) const
    {
        // Initial vorticity along the interface is 0.
        w1 = 0; w2 = 0;
        return true;
    };
    enum InitialConditionModel _i;
    double _t, _m, _v, _p;
    Kokkos::Array<int, 3> _ncells;
    double _dx, _dy;
    enum Beatnik::BoundaryType _b;
};

// Create Solver and Run
void rocketrig( ClArgs& cl )
{
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    Cabana::Grid::DimBlockPartitioner<2> partitioner; // Create Cabana::Grid Partitioner
    Beatnik::BoundaryCondition bc;
    for (int i = 0; i < 6; i++)
    {
        bc.bounding_box[i] = cl.params.global_bounding_box[i];

    }
    bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

    MeshInitFunc initializer( cl.params.global_bounding_box, cl.initial_condition,
                              cl.tilt, cl.magnitude, cl.variation, cl.params.period,
                              cl.num_nodes, cl.boundary );

    std::shared_ptr<Beatnik::SolverBase> solver;
    if (cl.params.solver_order == SolverOrder::ORDER_LOW) {
        solver = Beatnik::createSolver(
            MPI_COMM_WORLD, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Low(), cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else if (cl.params.solver_order == SolverOrder::ORDER_MEDIUM) {
        solver = Beatnik::createSolver(
            MPI_COMM_WORLD, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Medium(), cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
        solver = Beatnik::createSolver(
            MPI_COMM_WORLD, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else {
        std::cerr << "Invalid Model Order parameter!\n";
        Kokkos::finalize();
        MPI_Finalize();
        exit( -1 );

    }

    // Solve
    solver->solve( cl.t_final, cl.write_freq );
}

int main( int argc, char* argv[] )
{

    #if MEASURETIME
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    #endif

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    /* After Kokkos::initialize strips its own flags, exactly one
     * non-program argument must remain: the input file path (or
     * -h/--help). Any other count is a usage error. */
    std::string input_path;
    if ( argc == 2 )
    {
        const std::string a( argv[1] );
        if ( a == "-h" || a == "--help" )
        {
            help( rank );
            Kokkos::finalize();
            MPI_Finalize();
            return 0;
        }
        input_path = a;
    }
    else
    {
        if ( rank == 0 )
        {
            std::cerr << "rocketrig: expected exactly one argument (the "
                         "input file path).\n"
                      << "Run `rocketrig --help` for the input-file schema.\n";
        }
        Kokkos::finalize();
        MPI_Finalize();
        return -1;
    }

    // Parse Input
    ClArgs cl;
    if ( parseInput( rank, input_path, cl ) != 0 )
    {
        Kokkos::finalize();
        MPI_Finalize();
        return -1;
    }

    // Only Rank 0 Prints Command Line Options
    if ( rank == 0 )
    {
        // Print Command Line Options
        std::cout << "RocketRig\n";
        std::cout << "============Command line arguments============\n";
        std::cout << std::left << std::setw( 30 ) << "Input file"
                  << ": " << input_path << "\n";
        std::cout << std::left << std::setw( 30 ) << "Execution Space"
                  << ": " << std::setw( 8 ) << Kokkos::DefaultExecutionSpace::name()
                  << "\n";
        std::cout << std::left << std::setw( 30 ) << "Mesh Dimension"
                  << ": " << cl.num_nodes[0] << ", "
                  << cl.num_nodes[1] << "\n"; // Number of Cells
        std::cout <<  std::left << std::setw( 30 ) << "Solver Order"
                  << ": " << std::setw( 8 ) << cl.params.solver_order << "\n";

        // Solver-order specific arguments
        if (cl.params.solver_order == SolverOrder::ORDER_LOW)
        {
            std::cout << std::left << std::setw( 30 ) << "HeFFTe configuration"
                  << ": " << std::setw( 8 ) << cl.params.heffte_configuration  << "\n";
        }
        else
        {
            // High or medium-order solver
            if (cl.params.br_solver == BRSolverType::BR_EXACT)
            {
                std::cout <<  std::left << std::setw( 30 ) << "BR Solver type"
                    << ": " << std::setw( 8 ) << "exact" << "\n";
            }
            else if (cl.params.br_solver == BRSolverType::BR_CUTOFF)
            {
                std::cout <<  std::left << std::setw( 30 ) << "BR Solver type"
                    << ": " << std::setw( 8 ) << "cutoff" << "\n";
                std::cout << std::left << std::setw( 30 ) << "Cutoff distance"
                    << ": " << std::setw( 8 ) << cl.params.cutoff_distance  << "\n";
            }
            else if (cl.params.br_solver == BRSolverType::BR_FMM)
            {
                std::cout <<  std::left << std::setw( 30 ) << "BR Solver type"
                    << ": " << std::setw( 8 ) << "fmm" << "\n";
            }
        }
        std::cout << std::left << std::setw( 30 ) << "Total Simulation Time"
                  << ": " << std::setw( 8 ) << cl.t_final << "\n";
        std::cout << std::left << std::setw( 30 ) << "Timestep Size"
                  << ": " << std::setw( 8 ) << cl.delta_t << "\n";
        std::cout << std::left << std::setw( 30 ) << "Write Frequency"
                  << ": " << std::setw( 8 ) << cl.write_freq
                  << "\n"; // Steps between write
        std::cout << std::left << std::setw( 30 ) << "Atwood Constant"
                  << ": " << std::setw( 8 ) << cl.atwood << "\n";
        std::cout << std::left << std::setw( 30 ) << "Gravity"
                  << ": " << std::setw( 8 ) << (cl.gravity/9.81) << "\n";
        std::cout << std::left << std::setw( 30 ) << "Artificial Viscosity"
                  << ": " << std::setw( 8 ) << cl.mu << "\n";
        std::cout << std::left << std::setw( 30 ) << "Desingularization"
                  << ": " << std::setw( 8 ) << cl.eps  << "\n";
        std::cout << std::left << std::setw( 30 ) << "Weak-scaling factor"
                  << ": " << std::setw( 8 ) << cl.weak_scale << "\n";
        std::cout << std::left << std::setw( 30 ) << "Bounding Box Low/High"
                  << ": " << cl.params.global_bounding_box[0]
                  << ", " << cl.params.global_bounding_box[3] << "\n";
        std::cout << "==============================================\n";
    }

    // Call advection solver
    rocketrig( cl );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    #if MEASURETIME
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "measured_time: " << elapsed_seconds.count() << std::endl;
    #endif

    return 0;
};
