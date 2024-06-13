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

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <BRSolverBase.hpp>
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


    CutoffBRSolver( const pm_type &pm, const BoundaryCondition &bc,
                    const double epsilon, const double dx, const double dy,
                    Params params)
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

        // Create the spatial mesh
        _spatial_mesh = std::make_unique<SpatialMesh<ExecutionSpace, MemorySpace>>(
            _params.global_bounding_box, _params.periodic,
	        _params.cutoff_distance, _comm );
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    void initializeParticles(node_view z, node_view w, node_view o)
    {
        Kokkos::Profiling::pushRegion("initializeParticles");

        auto local_grid = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        int istart = local_space.min(0), jstart = local_space.min(1);
        int iend = local_space.max(0), jend = local_space.max(1);

        // Create the AoSoA
        int array_size = (iend - istart) * (jend - jstart);
        // printf("IN INIT: Before making particle_array\n");

        int rank = _rank;

        // Resize the particle array to hold all the mesh points we're migrating
	    _particle_array.resize(array_size);

        // Get slices of each piece of the particle array
        auto z_part = Cabana::slice<0>(_particle_array);
        auto o_part = Cabana::slice<1>(_particle_array);
        auto zdot_part = Cabana::slice<2>(_particle_array);
        auto weight_part = Cabana::slice<3>(_particle_array);
        auto idx_part = Cabana::slice<4>(_particle_array);
        auto id_part = Cabana::slice<5>(_particle_array);
        auto rank2d_part = Cabana::slice<6>(_particle_array);
        auto rank3d_part = Cabana::slice<7>(_particle_array);

        int mesh_dimension = _pm.mesh().get_surface_mesh_size();
        l2g_type local_L2G = _local_L2G;
        // printf("IN INIT: Before parallel for\n");

	    // We should convert this to a Cabana::simd_parallel_for at some point to get better write behavior
        Kokkos::parallel_for("populate_particles", Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
        KOKKOS_LAMBDA(int i, int j) {

            int particle_id = (i - istart) * (jend - jstart) + (j - jstart);

            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            
            //auto particle = particle_array.getTuple(particle_id);
            //printf("id: %d, get #1\n", particle_id);
            // XYZ position, BR omega, zdot
            for (int dim = 0; dim < 3; dim++) {
                z_part(particle_id, dim) = z(i, j, dim);
                o_part(particle_id, dim) = o(i, j, dim);
                zdot_part(particle_id, dim) = 0.0;
            }
            weight_part(particle_id) = simpsonWeight(local_gi[0], mesh_dimension) * simpsonWeight(local_gi[1], mesh_dimension);
            //printf("R%d: w(%d, %d), simp: %0.6lf\n", rank, local_gi[0], local_gi[1], Cabana::get<3>(particle));

            // Local index
            //printf("id: %d, get #3\n", particle_id);
            idx_part(particle_id, 0) = i;
            idx_part(particle_id, 1) = j;
            
            // Particle ID and rank
            //printf("id: %d, get #4\n", particle_id);
            id_part(particle_id) = particle_id;
            rank2d_part(particle_id) = rank;
            rank3d_part(particle_id) = -1;

            //printf("R%d: (%d, %d), simpson: %0.6lf\n", rank, Cabana::get<4>(particle, 0), Cabana::get<4>(particle, 1), Cabana::get<3>(particle));
        });

        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    void migrateParticlesTo3D()
    {
        Kokkos::Profiling::pushRegion("migrateParticlesTo3D");

        Kokkos::View<int*, memory_space> destination_ranks("destination_ranks", _particle_array.size());
        auto positions = Cabana::slice<0>(_particle_array, "positions");
        auto particle_comm = Cabana::Grid::createGlobalParticleComm<memory_space>(*_spm.localGrid());
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>(*_spm.localGrid());
        particle_comm->storeRanks(local_mesh);
        particle_comm->build(positions);
        particle_comm->migrate(_comm, _particle_array);

        // Populate 3D rank of origin and ID
        int rank = _rank;
        auto rank3d_part = Cabana::slice<7>(_particle_array);
        auto id3d_part = Cabana::slice<8>(_particle_array);
        Kokkos::parallel_for("3D origin rank", Kokkos::RangePolicy<exec_space>(0, _particle_array.size()), KOKKOS_LAMBDA(int i) {
            rank3d_part(i) = rank;
            id3d_part(i) = i;
        });

        // Updated owned 3D count for migration back to 2D
        _owned_3D_count = _particle_array.size();

        //printf("To 3D: R%d: owns %lu, _%lu particles\n", _rank, particle_array.size(), _particle_array.size());
        // for (int i = 0; i < _particle_array.size(); i++)
        // {
        //     auto particle = _particle_array.getTuple(i);
        //     printf("To 3D: R%d particle id: %d, 2D: %d, 3D: %d\n", _rank, Cabana::get<5>(particle), Cabana::get<6>(particle),  Cabana::get<7>(particle));
        // }

        Kokkos::Profiling::popRegion();
    }

    void performHaloExchange3D()
    {
        Kokkos::Profiling::pushRegion("performHaloExchange3D");

        // Halo exchange done in Comm constructor
        Comm<memory_space, particle_array_type, local_grid_type2>(_particle_array, *_spm.localGrid(), 40);

        Kokkos::Profiling::popRegion();
    }

    void migrateParticlesTo2D()
    {
        Kokkos::Profiling::pushRegion("migrateParticlesTo2D");

        // We only want to send back the non-ghosted particles to 2D
        // XXX Assume all ghosted particles are at the end of the array
        int rank = _rank;
        
        _particle_array.resize(_owned_3D_count);
        auto destinations = Cabana::slice<6>(_particle_array, "destinations");
        Cabana::Distributor<memory_space> distributor(_comm, destinations);
        Cabana::migrate(distributor, _particle_array);
        
        //printf("To 2D: R%d: owns %lu, _%lu particles\n", _rank, particle_array.size(), _particle_array.size());
        // for (int i = 0; i < _particle_array.size(); i++)
        // {
        //     auto particle = _particle_array.getTuple(i);
        //     printf("To 2D: R%d particle id: %d, 2D: %d, 3D: %d\n", _rank, Cabana::get<5>(particle), Cabana::get<6>(particle),  Cabana::get<7>(particle));
        // }

        Kokkos::Profiling::popRegion();
    }

    void computeInterfaceVelocityNeighbors(double dy, double dx, double epsilon)
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
        auto positions = Cabana::slice<0>(_particle_array);
        // for (int i = 0; i < positions.size(); i++) {
        //     auto tp = _particle_array.getTuple( i );
        //     printf("R%d: ID %d: %0.5lf %0.5lf %0.5lf\n", rank, Cabana::get<4>(tp),
        //         Cabana::get<0>(tp, 0), Cabana::get<0>(tp, 1), Cabana::get<0>(tp, 2));
        // }

        std::size_t num_particles = positions.size();

        auto neighbor_list = Cabana::Experimental::makeNeighborList(
        Cabana::FullNeighborTag{}, positions, 0, num_particles,
            _cutoff_distance);

        using list_type = decltype(neighbor_list);
        int rank = _rank;

        auto position_part = Cabana::slice<0>(_particle_array);
        auto omega_part = Cabana::slice<1>(_particle_array);
        auto zdot_part = Cabana::slice<2>(_particle_array);
        auto weight_part = Cabana::slice<3>(_particle_array);
        Kokkos::parallel_for("compute_BR_with_neighbors", Kokkos::RangePolicy<exec_space>(0, _owned_3D_count), 
                             KOKKOS_LAMBDA(int my_id) {
            int num_neighbors = Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, my_id);
            double brsum[3] = {0.0, 0.0, 0.0};

            //printf("R%d: particle %d/%lu, num neighbors %d\n", rank, i, num_particles, num_neighbors);

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
                // if (i == 20) {
                //     printf("R%d: neighbor %d, weight: %0.13lf\n", rank, neighbor_id, weight);
                // }
            }
            
            // Update the AoSoA
            for (int n = 0; n < 3; n++) {
                zdot_part(my_id, n) = brsum[n];
            }
            // if (i == 20) {
            //     printf("R%d: Particle %d/%lu (%d, %d),br_sum: %0.13lf %0.13lf %0.13lf\n", rank, i, num_particles, i_index, j_index, brsum[0], brsum[1], brsum[2]);
            // }
        });

        Kokkos::fence();

        Kokkos::Profiling::popRegion();
    }

    template <class PositionView>
    void populate_zdot(PositionView zdot)
    {
        Kokkos::Profiling::pushRegion("populate_zdot");

        int rank = _rank;

        auto zdot_part = Cabana::slice<2>(_particle_array);
        auto idx_part = Cabana::slice<4>(_particle_array);
        Kokkos::parallel_for("update_zdot", Kokkos::RangePolicy<exec_space>(0, _particle_array.size()), 
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
        initializeParticles(z, w, o);
        migrateParticlesTo3D();
        performHaloExchange3D();
        computeInterfaceVelocityNeighbors(_dy, _dx, _epsilon);
        migrateParticlesTo2D();
        populate_zdot(zdot);
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
    std::unique_ptr<spatial_mesh_type> _spatial_mesh;
    const Params _params;
    double _epsilon, _dx, _dy;
    const double _cutoff_distance;
    MPI_Comm _comm;
    int _rank;
    particle_array_type _particle_array;
    l2g_type _local_L2G;
    // XXX Communication views and extents to avoid allocations during each ring pass
};

}; // namespace Beatnik

#endif // BEATNIK_CUTOFFBRSOLVER_HPP
