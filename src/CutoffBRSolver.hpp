/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier,BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file CutoffBRSolver.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Class that uses a brute force approach to calculating the Birkhoff-Rott 
 * velocity intergral by using a all-pairs approach. Communication
 * uses a standard ring-pass communication algorithm. Does not attempt to 
 * reduce amount of computation per ring pass by using symetry of forces
 * as this complicates the GPU kernel.
 * 
 * Unlike the ExactBRSolver, calculations are limited by a cutoff distance.
 */

#ifndef BEATNIK_CUTOFFBRSOLVER_HPP
#define BEATNIK_CUTOFFBRSOLVER_HPP

#ifndef DEVELOP
#define DEVELOP 1
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <BRSolverBase.hpp>
#include <HaloComm.hpp>
#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>

namespace Beatnik
{

/**
 * The CutoffBRSolver Class
 * @class CutoffBRSolver
 * @brief Directly solves the Birkhoff-Rott integral using brute-force 
 * all-pairs calculation, limited by a cutoff distance
 * XXX - Make all functions but computeInterfaceVelocity private?
 **/
template <class ExecutionSpace, class MemorySpace, class Params>
class CutoffBRSolver : public BRSolverBase<ExecutionSpace, MemorySpace, Params>
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using spatial_mesh_type = SpatialMesh<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    
    using Node = Cabana::Grid::Node;
    using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    using node_array = typename pm_type::node_array;
    //using node_view = typename pm_type::node_view;
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    using particle_node = Cabana::MemberTypes<double[3], // xyz position in space                           0
                                              double[3], // Own omega for BR                                1
                                              double[3], // zdot                                            2
                                              double,    // Simpson weight                                  3
                                              int[2],    // Index in PositionView z and VorticityView w     4
                                              int,       // Point ID in 2D                                  5
                                              int,       // Owning rank in 2D space                         6
                                              int,       // Owning rank in 3D space                         7
                                              int        // Point ID in 3D                                  8
                                              >;
    // XXX Change the final parameter of particle_array_type, vector type, to
    // be aligned with the machine we are using
    using particle_array_type = Cabana::AoSoA<particle_node, device_type, 4>;

    // XXX - Get these from SpatialMesh class?
    using spatial_mesh_layout = Cabana::Grid::UniformMesh<double, 3>;
    using local_grid_layout = Cabana::Grid::LocalGrid<spatial_mesh_layout>;


    CutoffBRSolver( const pm_type &pm, const BoundaryCondition &bc,
                    const double epsilon, const double dx, const double dy,
                    const Params params)
        : _pm( pm )
        , _bc( bc )
        , _epsilon( epsilon )
        , _dx( dx )
        , _dy( dy )
        , _params( params )
        , _local_L2G( *_pm.mesh().localGrid() )
    {
	    _comm = _pm.mesh().localGrid()->globalGrid().comm();

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        // Create the spatial mesh
        _spatial_mesh = std::make_shared<SpatialMesh<ExecutionSpace, MemorySpace>>(
            params.global_bounding_box, params.periodic,
	        params.cutoff_distance, _comm );
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    static KOKKOS_INLINE_FUNCTION int isOnBoundary(const int local_location[3],
                                                   const int max_location[3])
    {
        for (int i = 0; i < 2; i++)
        {
            if (local_location[i] == 0 || local_location[i] == max_location[i]-1)
            {
                return 1;
            }
        }
        return 0;
    }

    void getPeriodicNeighbors(int is_neighbor[26]) const
    {
        for (int i = 0; i < 26; i++)
        {
            is_neighbor[i] = 0;
        }

        const auto local_grid = _spatial_mesh->localGrid();
        auto topology = _spatial_mesh->getBoundaryInfo();
        //Kokkos::Array<Cabana::Grid::IndexSpace<4>, topology_size> index_spaces;

        // Store all neighboring shared index space mesh bounds so we only have
        // to launch one kernel during the actual ghost search.
        int n = 0;
        for ( int k = -1; k < 2; ++k )
        {
            for ( int j = -1; j < 2; ++j )
            {
                for ( int i = -1; i < 2; ++i, ++n )
                {
                    if ( i != 0 || j != 0 || k != 0 )
                    {
                        int neighbor_rank = local_grid->neighborRank( i, j, k );
                        if (neighbor_rank != -1)
                        {
                            for (int w = 1; w < 3; w++)
                            {
                                if (abs(topology(_rank, w) - topology(neighbor_rank, w)) > 1)
                                {
                                    is_neighbor[neighbor_rank] = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Creates a populates particle array
     **/
    void initializeParticles(particle_array_type &particle_array, node_view z, node_view w, node_view o) const
    {
        Kokkos::Profiling::pushRegion("initializeParticles");

        auto local_grid = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        int istart = local_space.min(0), jstart = local_space.min(1);
        int iend = local_space.max(0), jend = local_space.max(1);

        // Create the AoSoA
        int array_size = (iend - istart) * (jend - jstart);

        int rank = _rank;

        // Resize the particle array to hold all the mesh points we're migrating
	    particle_array.resize(array_size);

        // Get slices of each piece of the particle array
        auto z_part = Cabana::slice<0>(particle_array);
        auto o_part = Cabana::slice<1>(particle_array);
        auto zdot_part = Cabana::slice<2>(particle_array);
        auto weight_part = Cabana::slice<3>(particle_array);
        auto idx_part = Cabana::slice<4>(particle_array);
        auto id_part = Cabana::slice<5>(particle_array);
        auto rank2d_part = Cabana::slice<6>(particle_array);
        auto rank3d_part = Cabana::slice<7>(particle_array);

        int mesh_dimension = _pm.mesh().get_surface_mesh_size();
        l2g_type local_L2G = _local_L2G;

	    // We should convert this to a Cabana::simd_parallel_for at some point to get better write behavior
        Kokkos::parallel_for("populate_particles", Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
        KOKKOS_LAMBDA(int i, int j) {

            int particle_id = (i - istart) * (jend - jstart) + (j - jstart);
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            
            // XYZ position, BR omega, zdot
            for (int dim = 0; dim < 3; dim++) {
                z_part(particle_id, dim) = z(i, j, dim);
                o_part(particle_id, dim) = o(i, j, dim);
                zdot_part(particle_id, dim) = 0.0;
            }
            weight_part(particle_id) = simpsonWeight(local_gi[0], mesh_dimension) * simpsonWeight(local_gi[1], mesh_dimension);

            // Local index
            idx_part(particle_id, 0) = i;
            idx_part(particle_id, 1) = j;
            
            // Particle ID and rank
            id_part(particle_id) = particle_id;
            rank2d_part(particle_id) = rank;
            rank3d_part(particle_id) = -1;

        });

        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    /** 
     * Move particles to their 3D rank of ownership. 
     * @return Updated particle AoSoA
     **/
    void migrateParticlesTo3D(particle_array_type &particle_array) const
    {
        Kokkos::Profiling::pushRegion("migrateParticlesTo3D");

        Kokkos::View<int*, memory_space> destination_ranks("destination_ranks", particle_array.size());
        auto positions = Cabana::slice<0>(particle_array, "positions");
        auto particle_comm = Cabana::Grid::createGlobalParticleComm<memory_space>(*_spatial_mesh->localGrid());
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>(*_spatial_mesh->localGrid());
        particle_comm->storeRanks(local_mesh);
        particle_comm->build(positions);
        particle_comm->migrate(_comm, particle_array);

        // Populate 3D rank of origin and ID
        int rank = _rank;
        auto rank3d_part = Cabana::slice<7>(particle_array);
        auto id3d_part = Cabana::slice<8>(particle_array);
        Kokkos::parallel_for("3D origin rank", Kokkos::RangePolicy<exec_space>(0, particle_array.size()), KOKKOS_LAMBDA(int i) {
            rank3d_part(i) = rank;
            id3d_part(i) = i;
        });

        Kokkos::Profiling::popRegion();
    }

    void performHaloExchange3D(particle_array_type &particle_array) const
    {
        Kokkos::Profiling::pushRegion("performHaloExchange3D");

        // Halo exchange done in Comm constructor
        Comm<memory_space, particle_array_type, local_grid_layout>(particle_array, *_spatial_mesh->localGrid(), 40);

        Kokkos::Profiling::popRegion();
    }

    /** 
     * Correct the x/y position of particles that are ghosted across x/y boundaries.
     * 
     * // XXX - Does not support periodic BCs with 1 process
     * 
     * To perform in 3D:
     * 1. Determine if the ghosted particles came from a rank on an x/y boundary
     * 2a. If so, depending on which x/y boundary, adjust the position as necessary
     * 2b. If we only have 1 rank, this doesn't work because this rank sits on 
     *      all boundaries so 1) is ambiguous.  
     * 
     * @return Updated particle AoSoA
     **/
    void correctPeriodicBoundaries(particle_array_type &particle_array, int local_count) const
    {
        Kokkos::Profiling::pushRegion("correctPeriodicBoundaries");

        if (!_params.periodic[0])
        {
            return;
        }
        if (_comm_size == 1)
        {
            std::cerr << "Error: Communicator size is " << _comm_size
            << " < 4 to support periodic boundary conditions using the cutoff solver\n";
            exit(-1);
        }

        auto position_part = Cabana::slice<0>(particle_array);
        auto rank3d_part = Cabana::slice<7>(particle_array);

        int total_size = particle_array.size();
        auto boundary_topology = _spatial_mesh->getBoundaryInfo();
        int local_location[3] = {boundary_topology(_rank, 1), boundary_topology(_rank, 2), boundary_topology(_rank, 3)};
        int max_location[3] = {boundary_topology(_comm_size, 1), boundary_topology(_comm_size, 2), boundary_topology(_comm_size, 3)};
        int rank = _rank;

        // Copy Boundary_topology to execution space
        Kokkos::View<int*[4], device_type> boundary_topology_device("boundary_topology_device", _comm_size+1);
        Kokkos::deep_copy(boundary_topology_device, boundary_topology);

        if (isOnBoundary(local_location, max_location))
        {
            std::array<double, 6> global_bounding_box = _params.global_bounding_box;
            int is_neighbor[26];
            getPeriodicNeighbors(is_neighbor);


            Kokkos::parallel_for("fix_haloed_particle_positions", Kokkos::RangePolicy<exec_space>(local_count, total_size), 
                             KOKKOS_LAMBDA(int index) {

                /* If local process is not on a boundary, exit. No particles
                * accross the boundary would have been recieved.
                * We only consider the x and y postions here because the
                * z-direction will never be periodic.
                */
                int remote_rank = rank3d_part(index);
                if (is_neighbor[remote_rank] == 1)
                {
                    // Get the dimenions to adjust
                    // Dimensions across a boundary will be more than one distance away in x/y/z space
                    int traveled[3];
                    for (int dim = 1; dim < 4; dim++)
                    {
                        if (boundary_topology_device(remote_rank, dim) - boundary_topology_device(rank, dim) > 1)
                        {
                            traveled[dim-1] = -1;
                        }
                        else if (boundary_topology_device(remote_rank, dim) - boundary_topology_device(rank, dim) < -1)
                        {
                            traveled[dim-1] = 1;
                        }
                        else
                        {
                            traveled[dim-1] = 0;
                        }
                    }

                    if (rank == 12)
                        {
                            printf("R%d: from R%d (index %d): traveled: %d, %d, %d, ", rank, remote_rank, index, traveled[0], traveled[1], traveled[2]);
                            printf("old pos: %0.5lf, %0.5lf, %0.5lf, ", position_part(index, 0), position_part(index, 1), position_part(index, 2));
                            //printf("Adjusting pos dim %d: diff: %0.5lf, old: %0.5lf new: %0.5lf\n", dim, diff, new_pos);
                        }
                    for (int dim = 0; dim < 3; dim++)
                    {
                        if (traveled[dim] != 0)
                        {
                            // -1, -1, -1, 1, 1, 1
                            double diff = global_bounding_box[dim+3] - global_bounding_box[dim];
                            // Adjust position
                            double new_pos = position_part(index, dim) + diff * traveled[dim];
                            position_part(index, dim) = new_pos;
                        }
                    }
                    if (rank == 12)
                        {
                            //printf("R%d: from R%d (index %d): traveled: %d, %d, %d\n", rank, remote_rank, index, traveled[0], traveled[1], traveled[2]);
                            printf("new pos: %0.5lf, %0.5lf, %0.5lf\n", 
                            position_part(index, 0), position_part(index, 1), position_part(index, 2));
                            //printf("Adjusting pos dim %d: diff: %0.5lf, old: %0.5lf new: %0.5lf\n", dim, diff, new_pos);
                        }
                }
            });
        }
        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    void migrateParticlesTo2D(particle_array_type &particle_array, int owned_3D_count) const
    {
        Kokkos::Profiling::pushRegion("migrateParticlesTo2D");

        // We only want to send back the non-ghosted particles to 2D
        // Resize to original 3D size to remove ghosted particles
        particle_array.resize(owned_3D_count);
        auto destinations = Cabana::slice<6>(particle_array, "destinations");
        Cabana::Distributor<memory_space> distributor(_comm, destinations);
        Cabana::migrate(distributor, particle_array);

        Kokkos::Profiling::popRegion();
    }

    void computeInterfaceVelocityNeighbors(particle_array_type &particle_array,
            int owned_3D_count, double dy, double dx, double epsilon) const
    {
        Kokkos::Profiling::pushRegion("computeInterfaceVelocityNeighbors");

        /* Project the Birkhoff-Rott calculation between all pairs of points on the 
        * interface, including accounting for any periodic boundary conditions.
        * Right now we brute force all of the points with no tiling to improve
        * memory access or optimizations to remove duplicate calculations. */
        /* Figure out which directions we need to project the k/l point to
        * for any periodic boundary conditions */

        // int kstart, lstart, kend, lend;
        // if (_bc.isPeriodicBoundary({0, 1})) {
        //     kstart = -1; kend = 1;
        // } else {
        //     kstart = kend = 0;
        // }
        // if (_bc.isPeriodicBoundary({1, 1})) {
        //     lstart = -1; lend = 1;
        // } else {
        //     lstart = lend = 0;
        // }

        /* Figure out how wide the bounding box is in each direction */
        // auto low = _pm.mesh().boundingBoxMin();
        // auto high = _pm.mesh().boundingBoxMax();;
        // double width[3];
        // for (int d = 0; d < 3; d++) {
        //     width[d] = high[d] - low[d];
        // }

        //double dx = _dx, dy = _dy, epsilon = _epsilon;

        // Find neighbors using ArborX
        //auto ids = Cabana::slice<3>(_particle_array);
        auto positions = Cabana::slice<0>(particle_array);

        std::size_t num_particles = positions.size();

        auto neighbor_list = Cabana::Experimental::makeNeighborList(
        Cabana::FullNeighborTag{}, positions, 0, num_particles,
            _params.cutoff_distance);

        using list_type = decltype(neighbor_list);

        auto position_part = Cabana::slice<0>(particle_array);
        auto omega_part = Cabana::slice<1>(particle_array);
        auto zdot_part = Cabana::slice<2>(particle_array);
        auto weight_part = Cabana::slice<3>(particle_array);
        Kokkos::parallel_for("compute_BR_with_neighbors", Kokkos::RangePolicy<exec_space>(0, owned_3D_count), 
                             KOKKOS_LAMBDA(int my_id) {
            int num_neighbors = Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, my_id);
            double brsum[3] = {0.0, 0.0, 0.0};

            // printf("Neighbors: R%d: particle %d/%lu, num neighbors %d\n", rank, my_id, num_particles, num_neighbors);

            for (int j = 0; j < num_neighbors; j++) {
                int neighbor_id = Cabana::NeighborList<list_type>::getNeighbor(neighbor_list, my_id, j);

                // XXX Offset initializtion not correct for periodic boundaries
                double offset[3] = {0.0, 0.0, 0.0}, br[3];
                
                /* Do the Birkhoff-Rott evaluation for this point */
                Operators::BR_with_slice(br, my_id, neighbor_id, position_part, omega_part, weight_part, 
                                         epsilon, dx, dy, offset);
                for (int d = 0; d < 3; d++) {
                    brsum[d] += br[d];
                }
            }
            
            // Update the AoSoA
            for (int n = 0; n < 3; n++) {
                zdot_part(my_id, n) = brsum[n];
            }
        });

        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    template <class PositionView>
    void populate_zdot(particle_array_type &particle_array, PositionView zdot) const
    {
        Kokkos::Profiling::pushRegion("populate_zdot");

        int rank = _rank;

        auto zdot_part = Cabana::slice<2>(particle_array);
        auto idx_part = Cabana::slice<4>(particle_array);
        Kokkos::parallel_for("update_zdot", Kokkos::RangePolicy<exec_space>(0, particle_array.size()), 
            KOKKOS_LAMBDA(int i) {
            int i_index = idx_part(i, 0);
            int j_index = idx_part(i, 1);

            for (int n = 0; n < 3; n++) {
                zdot(i_index, j_index, n) = zdot_part(i, n);
            }
        });

        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }
    

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. 
     * This function is called three times per time step to compute the initial, forward, and half-step
     * derivatives for velocity and vorticity.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view w, node_view o) const override
    {
        particle_array_type particle_array;
        initializeParticles(particle_array, z, w, o);
        migrateParticlesTo3D(particle_array);
        int owned_3D_count = particle_array.size();
        performHaloExchange3D(particle_array);
        correctPeriodicBoundaries(particle_array, owned_3D_count);
        computeInterfaceVelocityNeighbors(particle_array, owned_3D_count, _dy, _dx, _epsilon);
        migrateParticlesTo2D(particle_array, owned_3D_count);
        populate_zdot(particle_array, zdot);
    }

    std::shared_ptr<spatial_mesh_type> get_spatial_mesh()
    {
        return _spatial_mesh;
    }
    
    template <class l2g_type, class View>
    void printView(l2g_type local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
	Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            
            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            //printf("global: %d %d\n", local_gi[0], local_gi[1]);
            if (option == 1){
                if (dims == 3) {
                    printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }   
                    else if (dims == 2) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                    }
                }
            }
        });
    }

  private:
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    const Params _params;
    int _rank, _comm_size;
    std::shared_ptr<spatial_mesh_type> _spatial_mesh;
    double _epsilon, _dx, _dy;
    MPI_Comm _comm;
    l2g_type _local_L2G;
    // XXX Communication views and extents to avoid allocations during each ring pass
};

}; // namespace Beatnik

#endif // BEATNIK_CUTOFFBRSOLVER_HPP
