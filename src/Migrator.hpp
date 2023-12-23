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
        particle_comm->build(positions);
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
