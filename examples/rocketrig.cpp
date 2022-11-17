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
 *
 * @section DESCRIPTION
 * General rocket rig fluid interface example using the Beatnik z-model
 * fluid interface solver.
 */


#ifndef DEBUG
#define DEBUG 0
#endif


// Include Statements
#include <BoundaryCondition.hpp>
#include <Solver.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#if DEBUG
#include <iostream>
#endif

// Include Statements
#include <iomanip>
#include <iostream>

#include <getopt.h>
#include <stdlib.h>

using namespace Beatnik;

static char* shortargs = (char*)"n:t:d:x:F:o:I:b:g:a:T:m:v:p:i:w:O:M:e:h";

static option longargs[] = {
    // Basic simulation parameters
    { "nodes", required_argument, NULL, 'n' },
    { "timesteps", required_argument, NULL, 't' },
    { "delta_t", required_argument, NULL, 'd' },
    { "driver", required_argument, NULL, 'x' },
    { "write_frequency", required_argument, NULL, 'F' },
    { "outdir", required_argument, NULL, 'o' },

    // Z-model simulation parameters
    { "initial_condition", required_argument, NULL, 'I' },
    { "boundary", required_argument, NULL, 'b' },
    { "gravity", required_argument, NULL, 'g' },
    { "atwood", required_argument, NULL, 'a' },
    { "tilt", required_argument, NULL, 'T' },
    { "magnitude", required_argument, NULL, 'm' },
    { "variation", required_argument, NULL, 'v' },
    { "period", required_argument, NULL, 'p' },
    { "indir", required_argument, NULL, 'i' },
    { "weak-scale", required_argument, NULL, 'w'},

    // Z-model simulation parameters
    { "order", required_argument, NULL, 'O' },
    { "mu", required_argument, NULL, 'M' },
    { "epsilon", required_argument, NULL, 'e' },

    // Miscellaneous other arguments
    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};
enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};
/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain */
    enum InitialConditionModel initial_condition; /**< Model used to set initial conditions */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of scale of initial interface */
    double variation; /**< Variation in scale of initial interface */
    double period;   /**< Period of initial variation in interface */
    enum Beatnik::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */
    int model;      /**< Model used to set initial conditions */

    /* Problem simulation parameters */
    std::array<int, 2> num_nodes;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */
    int weak_scale;     /**< Amount to scale up resulting problem */

    /* I/O parameters */
    char *indir;        /**< Where to read initial conditions from */
    char *outdir;       /**< Directory to write output to */
    int write_freq;     /**< Write frequency */

    /* Solution method constants */
    enum SolverOrder order;  /**< Order of z-model solver to use */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */
};

/**
 * Outputs help message explaining command line options.
 * @param rank The rank calling the function
 * @param progname The name of the program
 */
void help( const int rank, char* progname )
{
    if ( rank == 0 )
    {
        /* XXX Needs to be fixed for change in arguments */
        std::cout << "Usage: " << progname << "\n";
        std::cout << std::left << std::setw( 10 ) << "-x" << std::setw( 40 )
                  << "On-node Parallelism Model (default serial)" << std::left
                  << "\n";
        std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 )
                  << "Number of points in each dimension (default 128)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 )
                  << "Weak Scaling Factor (default 1)" << std::left << "\n";
     //   std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
     //             << "Amount of time to simulate" << std::left
     //             << "\n";
     //   std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 )
     //             << "Timestep increment" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-F" << std::setw( 40 )
                  << "Write Frequency (default 10)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-O" << std::setw( 40 )
                  << "Solver Order (default \"low\")" << std::left << "\n";

        std::cout << std::left << std::setw( 10 ) << "-I" << std::setw( 40 )
                  << "Initial Condition Model (default \"cos\")" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-m" << std::setw( 40 )
                  << "Initial Condition Magnitude (default 0.05)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-v" << std::setw( 40 )
                  << "Initial Condition Variation (default 0.0)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-p" << std::setw( 40 )
                  << "Initial Condition Period (default 1.0)" << std::left << "\n";

        std::cout << std::left << std::setw( 10 ) << "-g" << std::setw( 40 )
                  << "Gravity in Gs (default 25.0)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-a" << std::setw( 40 )
                  << "Atwood number (default 0.5)" << std::left << "\n";

        std::cout << std::left << std::setw( 10 ) << "-M" << std::setw( 40 )
                  << "Artificial Viscosity Constant (default 1.0)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-e" << std::setw( 40 )
		<< "Desingularization Constant (defailt 0.25)" << std::left << "\n";

        std::cout << std::left << std::setw( 10 ) << "-h" << std::setw( 40 )
                  << "Print Help Message" << std::left << "\n";
    }
}

/**
 * Parses command line input and updates the command line variables
 * accordingly.
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
int parseInput( const int rank, const int argc, char** argv, ClArgs& cl )
{
    char ch;

    /// Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.order = SolverOrder::ORDER_LOW;;
    cl.weak_scale = 1;
    cl.write_freq = 10;

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 128, 128 };
    cl.initial_condition = IC_COS;
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.variation = 0.00;
    cl.period = 1.0;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the soler  to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = -1.0;

    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) !=
            -1 )
    {
        switch ( ch )
        {
        case 'n':
            cl.num_nodes[0] = atoi( optarg );

            if ( cl.num_nodes[0] < 1 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid number of nodes argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            cl.num_nodes[1] = cl.num_nodes[0];

            break;
        case 'x':
            cl.driver = strdup( optarg );
            if ( ( cl.driver.compare( "serial" ) != 0 ) &&
                 ( cl.driver.compare( "openmp" ) != 0 ) &&
                 ( cl.driver.compare( "threads" ) != 0 ) &&
                 ( cl.driver.compare( "cuda" ) != 0 ) &&
                 ( cl.driver.compare( "hip" ) != 0 ) )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid  parallel driver argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'F':
            cl.write_freq = atoi( optarg) ;
            if ( cl.write_freq < 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid write frequency argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'O':
        {
            std::string order(optarg);
            if (order.compare("low") == 0 ) {
                cl.order = SolverOrder::ORDER_LOW;
            } else if (order.compare("medium") == 0 ) {
                cl.order =  SolverOrder::ORDER_MEDIUM;
            } else if (order.compare("high") == 0 ) {
                cl.order = SolverOrder::ORDER_HIGH;
            } else {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid model order argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        }
        case 'b':
        {
            std::string order(optarg);
            if (order.compare("periodic") == 0) {
                cl.boundary = Beatnik::BoundaryType::PERIODIC;
            } else if (order.compare("free") == 0) {
                cl.boundary = Beatnik::BoundaryType::FREE;
            } else {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid boundary condition type.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        }
        case 'w':
            cl.weak_scale = atoi(optarg);
            if (cl.weak_scale < 1) {
                if ( rank == 0 ) {
                    std::cerr << "Invalid weak scaling factor order argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'I':
        {
            std::string model(optarg);
            if (model.compare("cos") == 0 ) {
                cl.initial_condition = InitialConditionModel::IC_COS;
            } else if (model.compare("sech2") == 0 ) {
                cl.initial_condition =  InitialConditionModel::IC_SECH2;
            } else {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid initial condition model argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        }
        case 'm':
            cl.magnitude = atof( optarg );
            if ( cl.magnitude <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid initial condition magnitude.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'v':
            cl.variation = atof( optarg );
            if ( cl.variation < 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid initial condition variation.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'p':
            cl.period = atof( optarg );
            if ( cl.period <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid initial condition period.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'a':
            cl.atwood = atof( optarg );
            if ( cl.atwood <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid atwood number.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'g':
            cl.gravity = atof( optarg );
            if ( cl.gravity <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid gravity.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'M':
            cl.mu = atof( optarg );
            if ( cl.mu <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid artificial viscosity.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'e':
            cl.eps = atof( optarg );
            if ( cl.eps <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid desingularization constant.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'h':
            help( rank, argv[0] );
            exit( 0 );
            break;
        case 't':
          cl.t_final = atof( optarg );
          if ( cl.t_final <= 0.0 )
          {
              if ( rank == 0 )
              {
                  std::cerr << "Invalid number of timesteps.\n";
                  help( rank, argv[0] );
              }
              exit( -1 );
          }
          break;
        default:
            if ( rank == 0 )
            {
                std::cerr << "Invalid argument.\n";
                help( rank, argv[0] );
            }
            exit( -1 );
            break;
        }
    }

    /* Physical setup of problem */
    cl.global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes[i] *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.order == SolverOrder::ORDER_HIGH) {
            cl.delta_t = tau/50.0;  // Should this depend on dx and dy? XXX
        } else {
            cl.delta_t = tau/25.0;
        }
    }

    if (cl.t_final <= 0.0) {
        cl.t_final = tau * 2.0; // Simulate for 2 characterisic periods, which is all
                                // the low-order model can really handle
    }
    else {
        cl.t_final = cl.t_final * cl.delta_t;
    }

    // Return Successfully
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

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Node, Beatnik::Field::Position,
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
        switch (_i) {
        case IC_COS:
            z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
            break;
        case IC_SECH2:
            z3 = _m * pow(1.0 / cosh(_p * (z1 * z1 + z2 * z2)), 2);
            break;
        case IC_RANDOM:
            /* XXX Use p to seed the random number generator XXX */
            /* Also need to use the Kokkos random number generator, not
             * drand48 */
            // z3 = _m * (2*drand48() - 1.0);
            break;
        case IC_GAUSSIAN:
        case IC_FILE:
            break;
        }
        return true;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Node, Beatnik::Field::Vorticity,
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

    Cajita::DimBlockPartitioner<2> partitioner; // Create Cajita Partitioner
    Beatnik::BoundaryCondition bc;
    for (int i = 0; i < 6; i++)
        bc.bounding_box[i] = cl.global_bounding_box[i];
    bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

    MeshInitFunc initializer( cl.global_bounding_box, cl.initial_condition,
                              cl.tilt, cl.magnitude, cl.variation, cl.period,
                              cl.num_nodes, cl.boundary );

    std::shared_ptr<Beatnik::SolverBase> solver;
    if (cl.order == SolverOrder::ORDER_LOW) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Low(), cl.mu, cl.eps, cl.delta_t );
    } else if (cl.order == SolverOrder::ORDER_MEDIUM) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Medium(), cl.mu, cl.eps, cl.delta_t );
    } else if (cl.order == SolverOrder::ORDER_HIGH) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t );
    } else {
        std::cerr << "Invalid Model Order parameter!\n";
        exit(-1);
    }

    // Solve
    solver->solve( cl.t_final, cl.write_freq );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    // Parse Input
    ClArgs cl;
    if ( parseInput( rank, argc, argv, cl ) != 0 )
        return -1;

    // Only Rank 0 Prints Command Line Options
    if ( rank == 0 )
    {
        // Print Command Line Options
        std::cout << "RocketRig\n";
        std::cout << "============Command line arguments============\n";
        std::cout << std::left << std::setw( 30 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.driver
                  << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 30 ) << "Mesh Dimension"
                  << ": " << std::setw( 8 ) << cl.num_nodes[0]
                  << std::setw( 8 ) << cl.num_nodes[1] 
                  << "\n"; // Number of Cells
        std::cout <<  std::left << std::setw( 30 ) << "Solver Order"
                  << ": " << std::setw( 8 ) << cl.order << "\n";
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
        std::cout << "==============================================\n";
    }

    // Call advection solver
    rocketrig( cl );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
