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

#ifndef BEATNIK_SPATIAL_MESH
#define BEATNIK_SPATIAL_MESH

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class SpatialMesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace>
class SpatialMesh
{
  public:
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 3>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using global_particle_comm_type = Cabana::Grid::GlobalParticleComm<memory_space, local_grid_type>;

    SpatialMesh( const std::array<double, 6>& global_bounding_box,
	      const std::array<bool, 2>& periodic,
          // const Cabana::Grid::BlockPartitioner<3>& partitioner,
          const double cutoff_distance, MPI_Comm comm )
    {
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_comm_size );
        // Declare the partioner here for now
        // Put particle type here
        // Make a cuttoff BRSolver and declare the spatial mesh inside the cuttoff BRSolver class
        // Make a migration.hpp class, declare spatialmesh in solver.hpp, put all 
        // the spatial mesh stuff in migration
        //Cabana::Grid::DimBlockPartitioner<3> partitioner;

        // Partition in x and y only
        // Try to partition evenly, otherwise set the x-dim to have sqrt(_comm_size)
        // ranks and the y-dim to have the remaining ranks.
        int ranks_in_xy = (int) floor(sqrt((float) _comm_size));
        if (_comm_size % ranks_in_xy && _rank == 0) 
        {
            printf("ERROR: The square root of the number of ranks must be an integer. There are %d ranks.\n", _comm_size);
            exit(1);
        }
        std::array<int, 3> input_ranks_per_dim = { ranks_in_xy, ranks_in_xy, 1 };

        // Create the manual partitioner in 2D.
        Cabana::Grid::ManualBlockPartitioner<3> partitioner(
            input_ranks_per_dim );

        std::array<int, 3> ranks_per_dim_manual =
            partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

        // Print the created decomposition.
        if ( _rank == 0 )
        {
            std::cout << "Ranks per dimension (manual): ";
            for ( int d = 0; d < 3; ++d )
                std::cout << ranks_per_dim_manual[d] << " ";
            std::cout << std::endl;
        }
        
        

        for (int i = 0; i < 3; i++) {
            _low_point[i] = global_bounding_box[i];
            _high_point[i] = global_bounding_box[i+3];
        }
        _low_point[2] = -20;
        _high_point[2] = 20;

        std::array<bool, 3> is_dim_periodic = { periodic[0], periodic[1], false };

        // Finally, create the global mesh, global grid, and local grid.
        _cell_size = 0.05;
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            _low_point, _high_point, _cell_size );

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     is_dim_periodic, partitioner );
        // Build the local grid.
        //_halo_width = fmax(100000, min_halo_width);
        _halo_width = (int) (cutoff_distance / _cell_size);

        // Halo width must be at least one
        if (_halo_width < 1)
        {
            std::cerr << "Halo width is " << _halo_width << ", which must be at least 1. \n";
            exit(-1);
        }

        _local_grid = Cabana::Grid::createLocalGrid( global_grid, _halo_width );
    }

    // Get the local grid.
    const std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> localGrid() const
    {
        return _local_grid;
    }

    const std::array<double, 3> & boundingBoxMin() const
    {
        return _low_point;
    }
    const std::array<double, 3> & boundingBoxMax() const
    {
        return _high_point;
    }

    int halo_width() const
    {
        return _halo_width;
    }

    double cell_size() const
    {
        return _cell_size;
    }


    // Get the boundary indexes on the periodic boundary. local_grid.boundaryIndexSpace()
    // doesn't work on periodic boundaries.
    // XXX Needs more error checking to make sure the boundary is in fact periodic
    template <class DecompositionType, class EntityType>
    Cabana::Grid::IndexSpace<2>
    periodicIndexSpace(DecompositionType dt, EntityType et, std::array<int, 2> dir) const
    {
        auto & global_grid = _local_grid->globalGrid();
        for ( int d = 0; d < 2; d++ ) {
            if ((dir[d] == -1 && global_grid.onLowBoundary(d))
                || (dir[d] == 1 && global_grid.onHighBoundary(d))) {
                return _local_grid->sharedIndexSpace(dt, et, dir);
            }
        }

        std::array<long, 2> zero_size;
        for ( std::size_t d = 0; d < 2; ++d )
            zero_size[d] = 0;
        return Cabana::Grid::IndexSpace<2>( zero_size, zero_size );
    }

    int rank() const { return _rank; }

  private:
    std::array<double, 3> _low_point, _high_point;
    std::shared_ptr<local_grid_type> _local_grid;
    std::shared_ptr<global_particle_comm_type> _global_particle_comm;
    int _rank, _comm_size, _halo_width;
    double _cell_size;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_SPATIAL_MESH
