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
//#include "ArtificialViscosity.hpp"
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
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <stdlib.h>

using namespace Beatnik;

// Short Args: n - Cell Count, p - On-node Parallelism ( Serial/OpenMP/CUDA/etc ),
// t - Time Steps, w - Write Frequency, i - delta_t
// g - Gravity, a - atwood number, T - tilt of rocket rig,
// v - magnitude of variation in interface
static char* shortargs = (char*)"n:t:d:w:x:o:g:a:T:v:p:m:h";

static option longargs[] = {
    // Basic simulation parameters
    { "cells", required_argument, NULL, 'n' },
    { "timesteps", required_argument, NULL, 't' },
    { "deltat", required_argument, NULL, 'd' },
    { "write-freq", required_argument, NULL, 'w' },
    { "driver", required_argument, NULL, 'x' },

    // Z-model simulation parameters
    { "order", required_argument, NULL, 'o' },
    { "gravity", required_argument, NULL, 'g' },
    { "atwood", required_argument, NULL, 'a' },
    { "tilt", required_argument, NULL, 'T' },
    { "variation", required_argument, NULL, 'v' },
    { "periodic", required_argument, NULL, 'p' },
    { "mu", required_argument, NULL, 'm' },


    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem basic parameters */`
    std::array<int, 2> global_num_cells;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    int write_freq;     /**< Write frequency */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */

    /* Problem physical setup */
    double tilt;    /**< Initial tilt of interface */
    double variation; /**< Magnitude of initial variation in interface */
    bool periodic; /**< Periodic or non-periodic boundary conditions */
    double gravity; /**< Gravitational accelaration in Gs in -Z direction */
    double atwood;  /**< Atwood pressure differential number */

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

    /// Set default values

    cl.device = "serial"; // Default Thread Setting
    cl.global_num_cells = { 128, 128 };


    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) !=
            -1 )
    {
        switch ( ch )
        {
        case 'n':
            cl.global_num_cells[0] = atoi( optarg );
            if ( cl.global_num_cells[0] <= 0 )
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
            cl.device = strdup( optarg );
            if ( ( cl.device.compare( "serial" ) != 0 ) &&
                 ( cl.device.compare( "cuda" ) != 0 ) &&
                 ( cl.device.compare( "openmp" ) != 0 ) &&
                 ( cl.device.compare( "pthreads" ) != 0 ) )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid  parallel device argument.\n";
                    help( rank, argv[0] );
                }
                exit( -1 );
            }
            break;
        case 'g':
            cl.gravity = atof( optarg );
            break;
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

    cl.t_final = 1.0;
    cl.write_freq = 1;
    cl.gravity = 25.0 * 9.8;
    cl.atwood = 0.5;
    double tau = 1/fsqrt(cl.atwood * cl.gravity);
    cl.delta_t = tau/25.0;

    double dx = 1/cl.global_num_cells[0];
    double dy = 1/cl.global_num_cells[1];

    cl.mu = fsqrt(dx * dy);
    cl.eps = 0.25*fsqrt(dx * dy);
    cl.model = 0;

    // Return Successfully
    return 0;
}

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    // Initialize Variables
    double _m, _xmax, _ymax, _dx, _dy;

    MeshInitFunc( double q, std::array<double, Dim> u )
        : _q( q )
    {
        _u[0] = u[0];
        _u[1] = u[1];
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Node, Beatnik::Field::Position,
                     const int index[2], double&  ) const
    {
        quantity = _q;

        return true;
    };
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cajita::Node, Beatnik::Field::Vorticity,
                     [[maybe_unused]] const int index[Dim],
                     double& xvelocity ) const
    {
        xvelocity = _u[0];
        return true;
    };
};

// Create Solver and Run 
void advect( ClArgs& cl )
{
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    Cajita::DimBlockPartitioner<2> partitioner; // Create Cajita Partitioner
    CajitaFluids::BoundaryCondition<2> bc;
    bc.boundary_type = {
        Beatnik::BoundaryType::PERIODIC, Beatnik::BoundaryType::PERIODIC,
        Beatnik::BoundaryType::PERIODIC, Beatnik::BoundaryType::PERIODIC };

    MeshInitFunc<2> initializer( 0.0, { 0.0, 0.0 } );
    auto solver = Beatnik::createSolver(
        cl.device, MPI_COMM_WORLD, cl.global_num_cells,
        partitioner, cl.density, initializer, bc, av, cl.delta_t );
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
        std::cout << "CajitaFluids\n";
        std::cout << "=======Command line arguments=======\n";
        std::cout << std::left << std::setw( 20 ) << "Thread Setting"
                  << ": " << std::setw( 8 ) << cl.device
                  << "\n"; // Threading Setting
        std::cout << std::left << std::setw( 20 ) << "Cells"
                  << ": " << std::setw( 8 ) << cl.global_num_cells[0]
                  << std::setw( 8 ) << cl.global_num_cells[1]
                  << "\n"; // Number of Cells
        std::cout << std::left << std::setw( 20 ) << "Domain"
                  << ": " << std::setw( 8 ) << cl.global_bounding_box[2]
                  << std::setw( 8 ) << cl.global_bounding_box[3]
                  << "\n"; // Span of Domain
        std::cout << std::left << std::setw( 20 ) << "Input Flow"
                  << ": " << std::setw( 8 ) << cl.inQuantity << " at "
                  << "Location (" << std::setw( 8 ) << cl.inLocation[0]
                  << std::setw( 8 ) << cl.inLocation[1] << " ) "
                  << "Size (" << std::setw( 8 ) << cl.inSize[0]
                  << std::setw( 8 ) << cl.inSize[1] << " ) "
                  << "Velocity (" << std::setw( 8 ) << cl.inVelocity[0]
                  << std::setw( 8 ) << cl.inVelocity[1] << " )\n";
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
    advect( cl );

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI

    return 0;
};
