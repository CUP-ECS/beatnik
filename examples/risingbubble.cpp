/****************************************************************************
 * Copyright (c) 2025 by the Beatnik authors                                *
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
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * General rising bubble example using the Beatnik z-model
 * fluid interface solver.
 */

#include <Solver.hpp>
 
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
 
#include <mpi.h>
#include <iomanip>
#include <iostream>

#include <getopt.h>
#include <stdlib.h>

using namespace Beatnik;

static char* shortargs = (char*)"n:t:d:x:F:o:g:a:i:w:M:e:h:";

static option longargs[] = {
    // Basic simulation parameters
    { "nodes", required_argument, NULL, 'n' },
    { "timesteps", required_argument, NULL, 't' },
    { "delta_t", required_argument, NULL, 'd' },
    { "driver", required_argument, NULL, 'x' },
    { "write_frequency", required_argument, NULL, 'F' },
    { "outdir", required_argument, NULL, 'o' },

    // Z-model simulation parameters
    { "gravity", required_argument, NULL, 'g' },
    { "atwood", required_argument, NULL, 'a' },
    { "indir", required_argument, NULL, 'i' },
    { "weak-scale", required_argument, NULL, 'w'},

    // Z-model simulation parameters
    { "mu", required_argument, NULL, 'M' },
    { "epsilon", required_argument, NULL, 'e' },

    // Miscellaneous other arguments
    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};
enum MeshType {MESH_STRUCTURED = 0, MESH_UNSTRUCTURED};

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    enum MeshType mesh_type;
    // std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain: MOVED TO PARAMS */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */

    /* Problem simulation parameters */
    int num_nodes;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */
    int weak_scale;     /**< Amount to scale up resulting problem */

    /* I/O parameters */
    char *indir;        /**< Where to read initial conditions from */
    char *outdir;       /**< Directory to write output to */
    int write_freq;     /**< Write frequency */

    /* Solution method constants */
    enum BRSolverType br_solver; /**< BRSolver to use */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */

    /* Parameters specific to solver order and BR solver type:
     *  - Period for particle initialization
     *  - Global bounding box
     *  - Periodicity
     *  - Heffte configuration (For low-order solver)
     *  - solver order (Order of z-model solver to use)
     *  - BR solver type
     *  - Cutoff distance (If using cutoff solver)
     */
    Params params;
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
                  << "Number of points" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-w" << std::setw( 40 )
                  << "Weak Scaling Factor (default 1)" << std::left << "\n";
       std::cout << std::left << std::setw( 10 ) << "-t" << std::setw( 40 )
                 << "Amount of timesteps to simulate" << std::left << "\n";
     //   std::cout << std::left << std::setw( 10 ) << "-d" << std::setw( 40 )
     //             << "Timestep increment" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-F" << std::setw( 40 )
                  << "Write Frequency (default 10)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-g" << std::setw( 40 )
                  << "Gravity in Gs (default 25.0)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-a" << std::setw( 40 )
                  << "Atwood number (default 0.5)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-M" << std::setw( 40 )
                  << "Artificial Viscosity Constant (default 1.0)" << std::left << "\n";
        std::cout << std::left << std::setw( 10 ) << "-e" << std::setw( 40 )
		          << "Desingularization Constant (default 0.25)" << std::left << "\n";
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
    signed char ch;

    /// Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.weak_scale = 1;
    cl.write_freq = 10;

    // Set default extra parameters
    cl.params.cutoff_distance = -1; // Not currently used
    cl.params.heffte_configuration = -1; // Not currently used
    cl.params.br_solver = BR_EXACT;
    cl.params.solver_order = SolverOrder::ORDER_HIGH;
    // cl.params.period below

    /* Default problem is the cosine rocket rig */
    cl.mesh_type = MeshType::MESH_UNSTRUCTURED;
    cl.num_nodes = 100;
    cl.params.period = -1;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the solver to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = -1.0;

    // Now parse any arguments
    while ( ( ch = getopt_long( argc, argv, shortargs, longargs, NULL ) ) != -1 )
    {
        switch ( ch )
        {
        case 'n':
        {
            cl.num_nodes = atoi( optarg );

            if ( cl.num_nodes < 1 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid number of nodes argument.\n";
                    help( rank, argv[0] );
                }
                Kokkos::finalize(); 
                MPI_Finalize(); 
                exit( -1 );  
            }

            break;
        }
        case 'x':
        {
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
                Kokkos::finalize(); 
                MPI_Finalize(); 
                exit( -1 );  
            }
            break;
        }
        case 'F':
            cl.write_freq = atoi( optarg ) ;
            if ( cl.write_freq < 0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid write frequency argument.\n";
                    help( rank, argv[0] );
                }
                Kokkos::finalize(); 
                MPI_Finalize(); 
                exit( -1 );  
            }
            break;
        case 'w':
            cl.weak_scale = atoi(optarg);
            if (cl.weak_scale < 1) {
                if ( rank == 0 ) {
                    std::cerr << "Invalid weak scaling factor order argument.\n";
                    help( rank, argv[0] );
                }
                Kokkos::finalize(); 
                MPI_Finalize(); 
                exit( -1 );  
            }
            break;
        case 'a':
        {
            cl.atwood = atof( optarg );
            if ( cl.atwood <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid atwood number.\n";
                    help( rank, argv[0] );
                }
                Kokkos::finalize(); 
                MPI_Finalize(); 
                exit( -1 );  
            }
            break;
        }
        case 'g':
            cl.gravity = atof( optarg );
            if ( cl.gravity <= 0.0 )
            {
                if ( rank == 0 )
                {
                    std::cerr << "Invalid gravity.\n";
                    help( rank, argv[0] );
                }
                Kokkos::finalize(); 
                MPI_Finalize(); 
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
                Kokkos::finalize(); 
                MPI_Finalize(); 
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
                Kokkos::finalize(); 
                MPI_Finalize(); 
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
              Kokkos::finalize(); 
              MPI_Finalize(); 
              exit( -1 );  
          }
          break;
        default:
            if ( rank == 0 )
            {
                std::cerr << "Invalid argument.\n";
                help( rank, argv[0] );
            }
            Kokkos::finalize(); 
            MPI_Finalize(); 
            exit( -1 );  
            break;
        }
    }

    /* Physical setup of problem */
    cl.params.global_bounding_box = {1 * -1.0,
                                     1 * -1.0, 
                                     1 * -1.0,
                                     1,
                                     1,
                                     1};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.params.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
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


// Create Solver and Run
int risingbubble( ClArgs& cl )
{

    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Construct filename based on rank
    std::string filename = "mesh_" + std::to_string(rank) + ".vtu";
    // std::cout << "Rank " << rank << " reading " << filename << std::endl;

    // Read the VTU file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkSmartPointer<vtkUnstructuredGrid> grid = reader->GetOutput();
    vtkSmartPointer<vtkPoints> points = grid->GetPoints();
    
    if (!points) {
        std::cerr << "Rank " << rank << " failed to read points from " << filename << std::endl;
        return 1;
    }

    int num_points = points->GetNumberOfPoints();
    int num_cells = grid->GetNumberOfCells();

    // Read ghost point flags
    vtkSmartPointer<vtkDataArray> ghost_flags_array = grid->GetPointData()->GetArray("ghost_points");
    if (!ghost_flags_array) {
        std::cerr << "Rank " << rank << " failed to read ghost point flags from " << filename << std::endl;
        return 1;
    }

    vtkSmartPointer<vtkDataArray> vertex_owners_array = grid->GetPointData()->GetArray("vertex_owner");
    if (!vertex_owners_array) {
        std::cerr << "Rank " << rank << " failed to read vertex owner data from " << filename << std::endl;
        return 1;
    }

    vtkSmartPointer<vtkDataArray> vertex_gids_array = grid->GetPointData()->GetArray("vertex_gids");
    if (!vertex_gids_array) {
        std::cerr << "Rank " << rank << " failed to read vertex gid data from " << filename << std::endl;
        return 1;
    }

    // Host-side AoSoAs for storing VTU data
    using vertices_d = Cabana::MemberTypes<int,      // Vertex global ID                                 
                                           int       // Owning rank
                                           >;
    using face_d = Cabana::MemberTypes<int[3],       // Vertex LIDs forming the triangle                                
                                       bool          // Flag indicating if the cell contains a ghost point
                                       >;
    using edge_d = Cabana::MemberTypes<int[2]>;       // Vertex LIDs forming the edge

    using triple_d = Cabana::MemberTypes<double[3]>; // Vertex positions

    using vert_aosoa = Cabana::AoSoA<vertices_d, Kokkos::HostSpace, 4>;
    using face_aosoa = Cabana::AoSoA<face_d, Kokkos::HostSpace, 4>;
    using triple_aosoa = Cabana::AoSoA<triple_d, Kokkos::HostSpace, 4>;
    vert_aosoa vertices("vertices", num_points);
    face_aosoa faces("faces", num_cells);
    triple_aosoa positions("positions", num_cells);
    auto v_gid = Cabana::slice<0>(vertices);
    auto v_owner = Cabana::slice<1>(vertices);
    auto f_vids = Cabana::slice<0>(faces);
    auto f_isGhost = Cabana::slice<1>(faces);
    auto p_xyz = Cabana::slice<0>(positions);

    std::unordered_map<int, bool> is_ghost_point;
    for (int i = 0; i < num_points; ++i) {
        int ghost_flag = static_cast<int>(ghost_flags_array->GetComponent(i, 0));
        is_ghost_point[i] = (ghost_flag == 1);  // Mark as ghost point if flag is 1
    }

    // Mapping from global VTK point index to local vertex ID
    std::unordered_map<int, int> global_to_local;
    // std::vector<Vertex> vertices;
        
    int owned_verts = 0;
    for (int i = 0; i < num_points; ++i) {
        global_to_local[i] = i;  // Assign local ID
        int vertex_owner = static_cast<int>(vertex_owners_array->GetComponent(i, 0));
        int vertex_gid = static_cast<int>(vertex_gids_array->GetComponent(i, 0));
        v_gid(i) = vertex_gid;
        v_owner(i) = vertex_owner;
        if (vertex_owner == rank) owned_verts++;

        // Populate coordinates
        double coords[3];
        points->GetPoint(i, coords);
        for (int j = 0; j < 3; j++) p_xyz(i, j) = coords[j];
        // vertices.push_back({i, vertex_gid, vertex_owner, coords[0], coords[1], coords[2]});
    }
    // Resize positions to only include owned points
    positions.resize(owned_verts);

    // Read cells (triangles) and assign local cell IDs
    for (int i = 0; i < num_cells; ++i) {
        vtkCell* cell = grid->GetCell(i);
        if (cell->GetNumberOfPoints() != 3) continue;  // Skip non-triangle cells

        int v0 = cell->GetPointId(0);
        int v1 = cell->GetPointId(1);
        int v2 = cell->GetPointId(2);

        // Check if the cell contains a ghost point
        bool contains_ghost = is_ghost_point[v0] || is_ghost_point[v1] || is_ghost_point[v2];

        f_vids(i, 0) = v0; f_vids(i, 1) = v1; f_vids(i, 2) = v2;
        f_isGhost(i) = contains_ghost;
    }

    Beatnik::BoundaryCondition bc; // Unused
    std::shared_ptr<Beatnik::SolverBase> solver;
    if (cl.params.solver_order == SolverOrder::ORDER_HIGH && (cl.mesh_type == MeshType::MESH_UNSTRUCTURED)) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD, positions, vertices, faces, bc, cl.atwood, cl.gravity,
            Beatnik::Order::High(), Beatnik::Mesh::Unstructured(),cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else {
        std::cerr << "Invalid Model Order parameter!\n";
        Kokkos::finalize(); 
        MPI_Finalize(); 
        exit( -1 );  

    }

    // Solve
    solver->solve( cl.t_final, cl.write_freq );

    return 0;
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
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

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
        std::cout << std::left << std::setw( 30 ) << "Mesh type"
                  << ": " << std::setw( 8 ) << cl.mesh_type
                  << "\n"; // Mesh type
        std::cout << std::left << std::setw( 30 ) << "Number of points:"
                  << ": " << cl.num_nodes << "\n";
        std::cout <<  std::left << std::setw( 30 ) << "Solver Order"
                  << ": " << std::setw( 8 ) << cl.params.solver_order << "\n";

        // Solver-order specific arguments
        if (cl.params.solver_order == SolverOrder::ORDER_LOW)
        {
            // std::cout << std::left << std::setw( 30 ) << "HeFFTe configuration"
            //       << ": " << std::setw( 8 ) << cl.params.heffte_configuration  << "\n";
        }
        else
        {
            // High or medium-order solver
            if (cl.params.br_solver == BRSolverType::BR_EXACT)
            {
                std::cout <<  std::left << std::setw( 30 ) << "BR Solver type"
                    << ": " << std::setw( 8 ) << "exact" << "\n";
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
        std::cout << "==============================================\n";
    }

    // Call advection solver
    risingbubble( cl );

    Kokkos::finalize();
    MPI_Finalize();

    #if MEASURETIME
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "measured_time: " << elapsed_seconds.count() << std::endl;
    #endif

    return 0;
};
