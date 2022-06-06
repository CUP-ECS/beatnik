/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 *
 * @section DESCRIPTION
 * Rocket rig example, including support for tilt, using z-model fluid interface 
 * solver on a unit 1x1 square
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

// Short Args: n - Cell Count,  t - Time Steps, w - Write Frequency, d - delta_t
// w - Write Frequency, x - On-node Parallelism ( Serial/OpenMP/CUDA/etc ),
// g - Gravity, a - atwood number, T - tilt of rocket rig,
// v - magnitude of variation in interface, p - interface periods per unit space 
// b - boundary condition (periodic, free, freeslip), 
// o - model order (low, medium, high), m - mu, the articificial viscosity constant
// e - epsilon, the desingularization constant
static char* shortargs = (char*)"n:t:d:w:x:g:a:T:v:p:b:o:m:e:h";

static option longargs[] = {
    // Basic simulation parameters
    { "cells", required_argument, NULL, 'n' },
    { "timesteps", required_argument, NULL, 't' },
    { "deltat", required_argument, NULL, 'd' },
    { "write-freq", required_argument, NULL, 'w' },
    { "driver", required_argument, NULL, 'x' },

    // Z-model simulation parameters
    { "gravity", required_argument, NULL, 'g' },
    { "atwood", required_argument, NULL, 'a' },
    { "tilt", required_argument, NULL, 'T' },
    { "variation", required_argument, NULL, 'v' },
    { "period", required_argument, NULL, 'p' },
    { "boundary", required_argument, NULL, 'b' },

    { "order", required_argument, NULL, 'o' },
    { "mu", required_argument, NULL, 'm' },
    { "epsilon", required_argument, NULL, 'e' },

    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of initial variation in interface */
    double period;   /**< Period of initial variation in interface */
    enum Beatnik::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */

    /* Problem simulation parameters */
    std::array<int, 2> global_num_cells;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    int write_freq;     /**< Write frequency */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */

    /* Solution method constants */
    int order;      /**< Order of z-model solver to use */
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
        std::cout << "Usage: " << progname << "\n";
        std::cout << std::left << std::setw( 10 ) << "-x" << std::setw( 40 )
                  << "On-node Parallelism Model (default serial)" << std::left
                  << "\n";
        std::cout << std::left << std::setw( 10 ) << "-n" << std::setw( 40 )
                  << "Number of Cells (default 200)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
                  << "Amount of time to simulate (default 4.0)" << std::left
                  << "\n";
        std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
                  << "Timestep increment (default 0.005)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 )
                  << "Write Frequency (default 20)" << std::left << "\n";

        std::cout << std::left << std::setw( 10 ) << "-g" << std::setw( 40 )
                  << "Gravity in Gs (default 25.0)" << std::left << "\n";

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
    cl.global_num_cells = { 128, 128 };
    cl.order = 0;

    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) !=
            -1 )
    {
        switch ( ch )
        {
        case 'n':
            cl.global_num_cells[0] = atoi( optarg );
            if ( cl.global_num_cells[0] < 1 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid cell number argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            cl.global_num_cells[1] = cl.global_num_cells[0];
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
        case 'o':
        { 
            std::string order(optarg);
            if (order.compare("low") == 0 ) {
                cl.order = 0;
            } else if (order.compare("medium") == 0 ) {
                cl.order = 1;
            } else if (order.compare("high") == 0 ) {
                cl.order = 2;
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
        case 'h':
            help( rank, argv[0] );
            exit( 0 );
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
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.period = 1.0;
    cl.gravity = 25.0 * 9.8;
    cl.atwood = 0.5;
    cl.boundary =  Beatnik::BoundaryType::PERIODIC;

    /* Simulation Parameters */

    /* Figure out parameters we need for the timestep and such. Simulate long 
     * enough for the interface to evolve significantly */ 
    double tau = 1/sqrt(cl.atwood * cl.gravity);
    cl.delta_t = tau/25.0;  // This should depend on dx, dy, and num_cells?
    cl.t_final = tau * 2.0; // Simulate for 2 characterisic periods, which is all
                            // the low-order model can really handle
    cl.write_freq = 1;

    /* Z-Model Solver Parameters */
    double dx = (cl.global_bounding_box[4] - cl.global_bounding_box[0])
                    / (cl.global_num_cells[0]);
    double dy = (cl.global_bounding_box[5] - cl.global_bounding_box[1])
                    / (cl.global_num_cells[1]);

    cl.mu = 1.0*sqrt(dx * dy);
    cl.eps = 0.25*sqrt(dx * dy);

    // Return Successfully
    return 0;
}

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    // Initialize Variables

    MeshInitFunc( std::array<double, 6> box, double t, double m, double p, 
                  const std::array<int, 2> cells )
        : _t( t )
        , _m( m )
        , _p( p )
    {
        double xcells = (cells[0] % 2) ? cells[0] + 1 : cells[0];
        double ycells = (cells[1] % 2) ? cells[1] + 1 : cells[1];

        x[0] = box[0];
        x[1] = box[1];
        x[2] = (box[2] + box[5]) / 2;
        _dx = (box[3] - box[0]) / xcells;
        _dy = (box[4] - box[1]) / ycells;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Node, Beatnik::Field::Position,
                     [[maybe_unused]] const int index[2], 
                     const double coord[2], 
                     double &z1, double &z2, double &z3) const
    {
        /* Compute the physical position of the interface from its global
         * coordinate in mesh space */
        z1 = _dx * coord[0];
        z2 = _dy * coord[1];
        // We don't currently tilt the interface
        z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
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
    double _t, _m, _p;
    Kokkos::Array<double, 3> x;
    double _dx, _dy;
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

    MeshInitFunc initializer( cl.global_bounding_box, cl.tilt, cl.magnitude, 
                              cl.period, cl.global_num_cells);

    std::shared_ptr<Beatnik::SolverBase> solver;
    if (cl.order == 0) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD, 
            cl.global_bounding_box, cl.global_num_cells,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Low(), cl.mu, cl.eps, cl.delta_t );
    } else if (cl.order == 1) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD,
            cl.global_bounding_box, cl.global_num_cells,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Medium(), cl.mu, cl.eps, cl.delta_t );
    } else {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD,
            cl.global_bounding_box, cl.global_num_cells,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t );
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
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.driver
                  << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Nodes"
                  << ": " << std::setw( 8 ) << cl.global_num_cells[0]
                  << std::setw( 8 ) << cl.global_num_cells[1]
                  << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Total Simulation Time"
                  << ": " << std::setw( 8 ) << cl.t_final << "\n";
        std::cout << std::left << std::setw( 20 ) << "Timestep Size"
                  << ": " << std::setw( 8 ) << cl.delta_t << "\n";
        std::cout << std::left << std::setw( 20 ) << "Write Frequency"
                  << ": " << std::setw( 8 ) << cl.write_freq
                  << "\n"; // Steps between write
        std::cout << "====================================\n";
    }

    // Call advection solver
    rocketrig( cl );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
