/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik author                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_MIGRATOR
#define BEATNIK_MIGRATOR

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class Migrator
  \brief Migrator between surface and spatial meshes
*/
template <class ExecutionSpace, class MemorySpace>
class Migrator
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

    using particle_node = Cabana::MemberTypes<double[3], // xyz position in space
                                              double[3], // Own omega for BR
                                              double[3], // zdot
                                              double,    // Simpson weight
                                              int[2],    // Index in PositionView z and VorticityView w
                                              int,       // Point ID
                                              int        // Rank of origin in 2D space
                                              >;
    using particle_array_type = Cabana::AoSoA<particle_node, device_type, 4>;

    // Construct a mesh.
    Migrator(const pm_type &pm, const spatial_mesh_type &spm)
        : _pm( pm )
        , _spm( spm )
        , _local_L2G( *_pm.mesh().localGrid() )
    {
        _comm = _spm.localGrid()->globalGrid().comm();
        MPI_Comm_size(_comm, &_comm_size);
        MPI_Comm_rank(_comm, &_rank);
        auto local_grid = _spm.localGrid();
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>(*local_grid);
        
        // printf("R%d: (%0.2lf, %0.2lf, %0.2lf), (%0.2lf, %0.2lf, %0.2lf)\n", _rank, local_mesh.lowCorner(Cabana::Grid::Own(), 0), local_mesh.lowCorner(Cabana::Grid::Own(), 1), local_mesh.lowCorner(Cabana::Grid::Own(), 2),
        //     local_mesh.highCorner(Cabana::Grid::Own(), 0), local_mesh.highCorner(Cabana::Grid::Own(), 1), local_mesh.highCorner(Cabana::Grid::Own(), 2));


        double own_space[6];
        for (int i = 0; i < 3; i++)
        {
            own_space[i] = local_mesh.lowCorner(Cabana::Grid::Own(), i);
            own_space[i+3] = local_mesh.highCorner(Cabana::Grid::Own(), i);
        }

        // Gather all ranks' spaces
        _grid_space = Kokkos::View<double*, memory_space>("grid_space", _comm_size * 6);
        MPI_Allgather(own_space, 6, MPI_DOUBLE, _grid_space.data(), 6, MPI_DOUBLE, _comm);
        if (_rank == 0)
        {
            printf("Comm size: %d\n", _comm_size);
            for (int i = 0; i < _comm_size * 6; i+=6)
            {
                printf("R%d: (%0.3lf %0.3lf %0.3lf), (%0.3lf %0.3lf %0.3lf)\n",
                    i/6, _grid_space(i), _grid_space(i+1), _grid_space(i+2),
                    _grid_space(i+3), _grid_space(i+4), _grid_space(i+5));
            }
        }

    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    void initializeParticles(node_view z, node_view w, node_view o)
    {
        auto local_grid = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        int istart = local_space.min(0), jstart = local_space.min(1);
        int iend = local_space.max(0), jend = local_space.max(1);

        // Create the AoSoA
        _array_size = (iend - istart) * (jend - jstart);
        _particle_array = particle_array_type("particle_array", _array_size);

        int rank = _rank;
        particle_array_type particle_array = _particle_array;
        int mesh_dimension = _pm.mesh().get_surface_mesh_size();
        l2g_type local_L2G = _local_L2G;
        Kokkos::parallel_for("populate_particles", Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
        KOKKOS_LAMBDA(int i, int j) {

            int particle_id = (i - istart) * (jend - jstart) + (j - jstart);

            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            
            Cabana::Tuple<particle_node> particle;
            //auto particle = particle_array.getTuple(particle_id);
            //printf("id: %d, get #1\n", particle_id);
            // XYZ position, BR omega, zdot
            for (int dim = 0; dim < 3; dim++) {
                Cabana::get<0>(particle, dim) = z(i, j, dim);
                Cabana::get<1>(particle, dim) = o(i, j, dim);
                Cabana::get<2>(particle, dim) = 0.0;
            }
            //printf("id: %d, get #1\n", particle_id);
            // Simpson weight
            // Cabana::get<3>(particle) = simpsonWeight(gi[0], mesh_size) * simpsonWeight(gi[1], mesh_size)
            //Cabana::get<3>(particle) = simpsonWeight(i - offset[0], mesh_dimension) * simpsonWeight(j - offset[1], mesh_dimension);
            Cabana::get<3>(particle) = simpsonWeight(local_gi[0], mesh_dimension) * simpsonWeight(local_gi[1], mesh_dimension);

            // Local index
            //printf("id: %d, get #3\n", particle_id);
            Cabana::get<4>(particle, 0) = i;
            Cabana::get<4>(particle, 1) = j;
            
            // Particle ID
            //printf("id: %d, get #4\n", particle_id);
            Cabana::get<5>(particle) = particle_id;

            // Rank of origin
            //printf("id: %d, get #5\n", particle_id);
            Cabana::get<6>(particle) = rank;

            //printf("id: %d, set tuple\n", particle_id);
            particle_array.setTuple(particle_id, particle);

            //printf("R%d: (%d, %d), simpson: %0.6lf\n", rank, Cabana::get<4>(particle, 0), Cabana::get<4>(particle, 1), Cabana::get<3>(particle));
        });

        // Wait for all parallel tasks to complete
        Kokkos::fence();
    }

    void migrateParticles() const
    {
        Kokkos::View<int*, memory_space> destination_ranks("destination_ranks", _array_size);
        auto positions = Cabana::slice<0>(_particle_array, "positions");
        //Cabana::Grid::particleGridMigrate(_spm.localGrid(), positions, _particle_array, 0, true);
        auto particle_comm = Cabana::Grid::createGlobalParticleComm<memory_space>(*_spm.localGrid());
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>(*_spm.localGrid());
        particle_comm->storeRanks(local_mesh);
        particle_comm->build(positions);
        

        
        // const std::size_t num_space_dim = 3;
        // int myrank = _rank;
        // auto rank = particle_comm->_rank;
        // auto print_corners = KOKKOS_LAMBDA( const std::size_t d )
        // {
        //     printf("R%d: %d: %d: (%0.3lf, %0.3lf)\n", myrank, rank[d], d, particle_comm->_local_corners( rank[d], d, 0 ),
        //         particle_comm->_local_corners( rank[d], d, 1 ));
        // };
        // using exec_space = typename memory_space::execution_space;
        // Kokkos::RangePolicy<exec_space> policy( 0, num_space_dim );
        // Kokkos::parallel_for( "print_corners",
        //                       policy, print_corners );
        // Kokkos::fence();




        particle_array_type particle_array = _particle_array;
        particle_comm->migrate(_comm, particle_array);
        if (_rank == 0)
        {
            for (int i = 0; i < _array_size; i++)
            {
                auto particle = particle_array.getTuple(i);
                if (Cabana::get<6>(particle) != _rank)
                {
                    printf("R%d: Got particle: (%0.3lf, %0.3lf, %0.3lf) from R%d\n", _rank,
                        Cabana::get<0>(particle, 0), Cabana::get<0>(particle, 1), Cabana::get<0>(particle, 2),
                        Cabana::get<6>(particle));
                }
            }
        }
    }

    void computeInterfaceVelocityNeighbors(int neighbor_radius)
    {
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

        double dx = _dx, dy = _dy, epsilon = _epsilon;

        // Find neighbors using ArborX
        //auto ids = Cabana::slice<3>(_particle_array);
        auto positions = Cabana::slice<0>(_particle_array);
        // for (int i = 0; i < positions.size(); i++) {
        //     auto tp = _particle_array.getTuple( i );
        //     printf("R%d: ID %d: %0.5lf %0.5lf %0.5lf\n", rank, Cabana::get<4>(tp),
        //         Cabana::get<0>(tp, 0), Cabana::get<0>(tp, 1), Cabana::get<0>(tp, 2));
        // }

        std::size_t num_particles = positions.size();

        auto neighbor_list = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, positions, 0, num_particles,
            neighbor_radius);

        using list_type = decltype(neighbor_list);
        particle_array_type particle_array = _particle_array;
        int rank = _rank;

        Kokkos::parallel_for("compute_BR_with_neighbors", num_particles, KOKKOS_LAMBDA(int i) {

            int num_neighbors = Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, i);
            double brsum[3] = {0.0, 0.0, 0.0};
            auto particle = particle_array.getTuple(i);

            if (Cabana::get<6>(particle) != rank) return;

            // Compute BR forces from neighbors
            int i_index = Cabana::get<4>(particle, 0);
            int j_index = Cabana::get<4>(particle, 1);
            //printf("Particle %d/%lu, num neighbors %d\n", i, num_particles, num_neighbors);

            for (int j = 0; j < num_neighbors; j++) {
                int neighbor_id = Cabana::NeighborList<list_type>::getNeighbor(neighbor_list, i, j);
                auto neighbor_particle =  particle_array.getTuple(neighbor_id);
                double weight = Cabana::get<3>(neighbor_particle);

                // XXX Offset initializtion not correct for periodic boundaries
                double offset[3] = {0.0, 0.0, 0.0}, br[3];
                
                /* Do the Birkhoff-Rott evaluation for this point */
                Operators::BR_with_aosoa(br, particle, neighbor_particle, epsilon, dx, dy, weight, offset);
                for (int d = 0; d < 3; d++) {
                    brsum[d] += br[d];
                }
                // if (i == 20) {
                //     printf("R%d: neighbor %d, weight: %0.13lf\n", rank, neighbor_id, weight);
                // }
            }
            
            // Update the AoSoA
            for (int n = 0; n < 3; n++) {
                Cabana::get<2>(particle, n) = brsum[n];
            }
            particle_array.setTuple(i, particle);
            // if (i == 20) {
            //     printf("R%d: Particle %d/%lu (%d, %d),br_sum: %0.13lf %0.13lf %0.13lf\n", rank, i, num_particles, i_index, j_index, brsum[0], brsum[1], brsum[2]);
            // }
        });

        // Wait for all parallel tasks to complete
        Kokkos::fence();

        // for ( std::size_t i = 0; i < positions.size(); ++i )
        // {
        //     int num_neighbors = Cabana::NeighborList<list_type>::numNeighbor(neighbor_list, i);

        //     //std::cout << "R" << rank << ": Particle " << i << " # neighbor = " << num_neighbors << std::endl;
        //     // int num_n = Cabana::NeighborList<list_type>::numNeighbor( neighbor_list, i );
        //     // for ( int j = 0; j < num_n; ++j )
        //     //     std::cout << "    neighbor " << j << " = "
        //     //             << Cabana::NeighborList<list_type>::getNeighbor(
        //     //                     neighbor_list, i, j )
        //     //             << std::endl;

        //     // Compute BR force from neighbors
            
        //     double brsum[3] = {0.0, 0.0, 0.0};
        //     auto particle = _particle_array.getTuple(i);

        //     // Only calculate for particles owned by this rank, not ghosts
        //     if (Cabana::get<6>(particle) != _rank) continue;

        //     int i_index = Cabana::get<4>(particle, 0);
        //     int j_index = Cabana::get<4>(particle, 1);
        //     for (int j = 0; j < num_neighbors; j++) {
        //         int neighbor_id = Cabana::NeighborList<list_type>::getNeighbor(neighbor_list, i, j);
        //         auto neighbor_particle = _particle_array.getTuple(neighbor_id);

        //         double weight = Cabana::get<3>(neighbor_particle);

        //         // XXX Offset initializtion not correct for periodic boundaries
        //         double offset[3] = {0.0, 0.0, 0.0}, br[3];
                
        //         /* Do the Birkhoff-Rott evaluation for this point */
        //         Operators::BR_with_aosoa(br, particle, neighbor_particle, epsilon, dx, dy, weight, offset);
        //         for (int d = 0; d < 3; d++) {
        //             brsum[d] += br[d];
        //         }
        //     }
        //     /* Add it its contribution to the integral */
        //     for (int n = 0; n < 3; n++) {
        //         Cabana::get<2>(particle, n) = brsum[n];
        //     }

        //     // Update the AoSoA
        //     _particle_array.setTuple(i, particle);
            

        //     // if (i_index == 4 && j_index == 9) {
        //     //     printf("xyz: %0.5lf %0.5lf %0.5lf\nomega: %0.5lf %0.5lf %0.5lf\nzdot: %0.13lf %0.13lf %0.13lf\nsimpson: %0.5lf, (%d, %d), id: %d\n", 
        //     //         Cabana::get<0>(particle, 0), Cabana::get<0>(particle, 1), Cabana::get<0>(particle, 2),
        //     //         Cabana::get<1>(particle, 0), Cabana::get<1>(particle, 1), Cabana::get<1>(particle, 2),
        //     //         Cabana::get<2>(particle, 0), Cabana::get<2>(particle, 1), Cabana::get<2>(particle, 2),
        //     //         Cabana::get<3>(particle), Cabana::get<4>(particle, 0), Cabana::get<4>(particle, 1), Cabana::get<5>(particle));
        //     // }
        // }
    }

    template <class PositionView>
    void populate_zdot(PositionView zdot)
    {
        particle_array_type particle_array = _particle_array;
        int rank = _rank;

        // Parallel for loop using Kokkos
        Kokkos::parallel_for("update_zdot", _array_size, KOKKOS_LAMBDA(int i) {
            auto particle = particle_array.getTuple(i);
            int particle_rank = Cabana::get<6>(particle);

            if (particle_rank == rank) {
                int i_index = Cabana::get<4>(particle, 0);
                int j_index = Cabana::get<4>(particle, 1);

                for (int n = 0; n < 3; n++) {
                    Kokkos::atomic_add(&zdot(i_index, j_index, n), Cabana::get<2>(particle, n));
                }
            }
        });

        // Wait for all parallel tasks to complete
        Kokkos::fence();


        //if (_rank == 0) {
            //printf("R%d: ****START**** size = %d, array_size = %d\n", _rank, _particle_array.size(), _array_size);
        //}
        // for (int i = 0; i < _array_size; i++)
        // {
        //     if (_rank == 0) {
        //         //printf("R%d: i = %d\n", _rank, i);
        //     }
        //     auto particle = _particle_array.getTuple(i);

        //     // Check that the particle is owned by our rank, and thus in zdot
        //     int particle_rank = Cabana::get<6>(particle);
        //     if (_rank == 0) {
        //         //printf("R%d: i%d: p-rank: %d\n", _rank, i, particle_rank);
        //     }
        //     if (particle_rank == _rank)
        //     {
        //         int i_index = Cabana::get<4>(particle, 0);
        //         int j_index = Cabana::get<4>(particle, 1);
        //         //if (_rank == 0) {
        //            // printf("R%d: i%d: p-rank: %d, i-index: %d, j-index: %d\n", _rank, i, particle_rank, i_index, j_index);
        //         //}
        //         for (int n = 0; n < 3; n++) 
        //         {
        //             zdot(i_index, j_index, n) = Cabana::get<2>(particle, n);
        //         }
        //     }
        // }
    }

  //private:
    particle_array_type _particle_array;
    const spatial_mesh_type &_spm;
    const pm_type &_pm;
    const l2g_type _local_L2G;
    MPI_Comm _comm;
    int _rank, _comm_size, _array_size;

    Kokkos::View<double*, memory_space> _grid_space;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MIGRATOR
