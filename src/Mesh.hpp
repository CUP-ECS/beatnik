/****************************************************************************
 * Copyright (c) 2021 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_MESH_HPP
#define BEATNIK_MESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class Mesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace>
class Mesh
{
  public:
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;

    // Construct a mesh.
    Mesh( const std::array<int, 2>& global_num_cells,
	  const std::array<bool, 2>& periodic,
          const Cajita::BlockPartitioner<2>& partitioner,
          const int min_halo_width, MPI_Comm comm )
    {
        MPI_Comm_rank( comm, &_rank );

        // Make a copy of the global number of cells so we can modify it.
        std::array<int, 2> num_cells = global_num_cells;

        /* Create global mesh bounds. There are a few caveats here that are important to 
         * understand:
         * 1. Each mesh point has multiple locations - it's i/j location [0...n),
         *    [0...m), it's location in node coordinate space [n/2, n/2), 
         *    [m/2, n/2), its initial spatial location in x/y space, and the 
         *    x/y/z location of its points at any given time.
         * 2. Of these, the first and last are used often in calculations, no 
         *    matter the the order of the model, and the second is used for every
         *    derivative calculation in models that use the Reisz transform. As a
         *    result, we don't store the initial x/y spatial location.
         * 3. Low and medium order models need to have the same number of mesh 
         *    point the same number of nodes above and below 0, but how the mesh 
         *    sets this up depends on the boundary conditions. We basically 
         *    always want an even number of cells. When the mesh isn't periodic, 
         *    this results in the same number of nodes above and below zero. 
         *    When the mesh *is* periodic, the last mesh node above 0 is implicit
         * from the wrap-around of the mesh. 
         */
        for (int i = 0; i < 2; i++) {
            if (num_cells[i] % 2 == 0) {
                num_cells[i]++;
            }
        }

        /* Then split those cells above and below 0 appropriately */
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        { 
            /* periodic -> ncells = 3 -> global low == -1, global high = 2 -> nodes = 3.
             * non-periodic -> ncells = 4 -> global low = -2, global high = 2 -> nodes = 4*/
            global_low_corner[d] = -1 * num_cells[d]/2;
            global_high_corner[d] = num_cells[d] / 2;
        }

        // Finally, create the global mesh, global grid, and local grid.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );


        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );
        // Build the local grid.
        int halo_width = fmax(2, min_halo_width);
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    }

    // Get the local grid.
    const std::shared_ptr<Cajita::LocalGrid<mesh_type>> localGrid() const
    {
        return _local_grid;
    }

    // Get the boundary indexes on the periodic boundary. local_grid.boundaryIndexSpace()
    // doesn't work on periodic boundaries.
    // XXX Needs more error checking to make sure the boundary is in fact periodic
    template <class DecompositionType, class EntityType>
    Cajita::IndexSpace<2>
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
        return Cajita::IndexSpace<2>( zero_size, zero_size );
    }

    int rank() const { return _rank; }

  public:
    std::shared_ptr<Cajita::LocalGrid<mesh_type>> _local_grid;
    int _rank;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MESH_HPP
