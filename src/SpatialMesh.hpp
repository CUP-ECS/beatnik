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

    // Construct a mesh.
    SpatialMesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	      const std::array<bool, 2>& periodic,
          // const Cabana::Grid::BlockPartitioner<3>& partitioner,
          const int min_halo_width, MPI_Comm comm )
		  : _num_nodes( num_nodes )
    {
        // Declare the partioner here for now
        // Put particle type here
        // Make a cuttoff BRSolver and declare the spatial mesh inside the cuttoff BRSolver class
        // Make a migration.hpp class, declare spatialmesh in solver.hpp, put all 
        // the spatil mesh stuff in migration
        Cabana::Grid::DimBlockPartitioner<3> partitioner;

        MPI_Comm_rank( comm, &_rank );

        for (int i = 0; i < 3; i++) {
            _low_point[i] = global_bounding_box[i];
            _high_point[i] = global_bounding_box[i+3];
        } 

        std::array<bool, 3> is_dim_periodic = { true, true, false };

        // Finally, create the global mesh, global grid, and local grid.
        double cell_size = 0.05;
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            _low_point, _high_point, cell_size );

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     is_dim_periodic, partitioner );
        // Build the local grid.
        int halo_width = fmax(2, min_halo_width);
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        _global_particle_comm = Cabana::Grid::createGlobalParticleComm(_local_grid);

        auto _local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *_local_grid );


        // Get the current rank for printing output.
        int comm_rank = global_grid->blockId();
        if ( comm_rank == 0 )
        {
            std::cout << "Cabana::Grid Global Grid Example" << std::endl;
            std::cout << "    (intended to be run with MPI)\n" << std::endl;
            printf("Low: %0.2lf %0.2lf %0.2lf, high: %0.2lf %0.2lf %0.2lf\n", _low_point[0], _low_point[1], _low_point[2],
                _high_point[0], _high_point[1], _high_point[2]);
        }
        if ( comm_rank == 0 )
        {
            std::cout << "Global global grid information:" << std::endl;
            bool periodic_x = global_grid->isPeriodic( Cabana::Grid::Dim::I );
            std::cout << "Periodicity in X: " << periodic_x << std::endl;

            int num_blocks_y = global_grid->dimNumBlock( Cabana::Grid::Dim::J );
            std::cout << "Number of blocks in Y: " << num_blocks_y << std::endl;

            int num_blocks = global_grid->totalNumBlock();
            std::cout << "Number of blocks total: " << num_blocks << std::endl;

            int num_cells_x = global_grid->globalNumEntity( Cabana::Grid::Cell(),
                                                            Cabana::Grid::Dim::I );
            std::cout << "Number of cells in X: " << num_cells_x << std::endl;

            int num_faces_y = global_grid->globalNumEntity(
                Cabana::Grid::Face<Cabana::Grid::Dim::I>(), Cabana::Grid::Dim::J );
            std::cout << "Number of X Faces in Y: " << num_faces_y << std::endl;

            std::cout << "\nPer rank global grid information:" << std::endl;
        }

        /*
        The global grid also stores information to describe each separate block
        (MPI rank): whether it sits on a global system boundary, it's position
        ("ID") within the MPI block decomposition, and the number of mesh cells
        "owned" (uniquely managed by this rank).
        */
        bool on_lo_x = global_grid->onLowBoundary( Cabana::Grid::Dim::I );
        std::cout << "Rank-" << comm_rank << " on low X boundary: " << on_lo_x
                << std::endl;

        bool on_hi_y = global_grid->onHighBoundary( Cabana::Grid::Dim::J );
        std::cout << "Rank-" << comm_rank << " on high Y boundary: " << on_hi_y
                << std::endl;

        bool block_id = global_grid->blockId();
        std::cout << "Rank-" << comm_rank << " block ID: " << block_id << std::endl;

        bool block_id_x = global_grid->dimBlockId( Cabana::Grid::Dim::I );
        std::cout << "Rank-" << comm_rank << " block ID in X: " << block_id_x
                << std::endl;

        int num_cells_y = global_grid->ownedNumCell( Cabana::Grid::Dim::J );
        std::cout << "Rank-" << comm_rank
                << " owned mesh cells in Y: " << num_cells_y << std::endl;

        // Barrier for cleaner printing.
        MPI_Barrier( MPI_COMM_WORLD );

        /*
        Other information can be extracted which is somewhat lower level. First,
        the MPI rank of a given block ID can be obtained; this returns -1 if it
        an invalid ID for the current decomposition.
        */
        if ( comm_rank == 0 )
        {
            std::cout << std::endl;
            // In this case, if the block ID is passed as an array with length equal
            // to the spatial dimension.
            bool block_rank = global_grid->blockRank( { 0, 0 } );
            std::cout << "MPI rank of the first block: " << block_rank << std::endl;
        }

        /*
        Second, the offset (the index from the bottom left corner in a given
        dimension) of one block can be extracted.
        */
        int offset_x = global_grid->globalOffset( Cabana::Grid::Dim::I );
        std::cout << "Rank-" << comm_rank << " offset in X: " << offset_x
                << std::endl;
        
        std::cout << "Low corner local: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.lowCorner( Cabana::Grid::Own(), d ) << " ";
        std::cout << "\nHigh corner local: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.highCorner( Cabana::Grid::Own(), d ) << " ";
        std::cout << "\nExtent local: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.extent( Cabana::Grid::Own(), d ) << " ";
        std::cout << "\nLow corner ghost: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.lowCorner( Cabana::Grid::Ghost(), d ) << " ";
        std::cout << "\nHigh corner ghost: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.highCorner( Cabana::Grid::Ghost(), d ) << " ";
        std::cout << "\nExtent ghost: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << _local_mesh.extent( Cabana::Grid::Ghost(), d ) << " ";

        /*
        Note that this information is taken directly from the global grid and mesh
        information.
        */
        std::cout << "\n\nLow corner from global offset ";
        for ( int d = 0; d < 3; ++d )
            std::cout << cell_size * global_grid->globalOffset( d ) << " ";

        /*
        The local mesh is most often useful to get information about individual
        cells within a kernel. Note this is done on the host here, but would most
        often be in a Kokkos parallel kernel. The local mesh is designed such that
        it can be captured by value in Kokkos parallel kernels.

        This information includes the coordinates and the measure (size). The
        measure depends on the entity type used: Nodes return zero, Edges return
        length, Faces return area, and Cells return volume.
        */
        double loc[3];
        int idx[3] = { 0, 0, 0 };
        _local_mesh.coordinates( Cabana::Grid::Cell(), idx, loc );
        std::cout << "\nRandom cell coordinates: ";
        for ( int d = 0; d < 3; ++d )
            std::cout << loc[d] << " ";
        std::cout << "\nRandom cell measure: "
                << _local_mesh.measure( Cabana::Grid::Cell(), idx ) << std::endl;
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
	
    // Get the mesh size
    int get_surface_mesh_size() const
    {
        return _num_nodes[0];
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
    int _rank;
	std::array<int, 2> _num_nodes;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_SPATIAL_MESH
