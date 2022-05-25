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
          const int halo_cell_width, MPI_Comm comm )
    {
        MPI_Comm_rank( comm, &_rank );

        // Make a copy of the global number of cells so we can modify it.
        std::array<int, 2> num_cells = global_num_cells;

        /* Create global mesh bounds. This is tricky because we want to make sure we have
         * the same number of nodes above and below 0, but how the mesh sets this up depends 
         * on the boundary conditions.
         * 1) If we're periodic in a dimension, we want an odd number of cells so we have 
         *    an odd number of nodes. That means we want one more node above 0 than below 0;
         * 2) If we're not periodic in a dimension, we have an even number of cells so we
         *    have an odd number of nodes. That means we want the same number of nodes above
         *    and below 0.
         * Start by adjusting the number of cells we request appropriately */
        for (int i = 0; i < 2; i++) {
            if (((num_cells[i] % 2 == 0) && periodic[i])
                || ((num_cells[i] % 2 == 1) && !periodic[i])) {
                if (_rank == 0) {
                    std::cout << "Increasing number of cells in direction " << i
                        << " by one to match boundary condition and fourier transform"
                        << " requirements.\n";
                }
                num_cells[i]++;
            }
        }

        /* Then split those cells above and below 0 appropriately */
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        { 
            /* periodic -> ncells = 3 -> global low == -1, global high = 2 -> nodes = 3.
             * non-periodic -> ncells = 4 -> global low = -2, global high = 2 -> nodes = 4*/
            global_low_corner[d] = -1 * (num_cells[d]/2);
            global_high_corner[d] = num_cells[d] / 2 + num_cells[d] % 2; 
        }

        // Finally, create the global mesh, global grid, and local grid.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );


        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );
        for ( int d = 0; d < 2; ++d )
        {
            _min_domain_global_node_index[d] = 0;
            _max_domain_global_node_index[d] = global_grid->globalNumEntity(Cajita::Node(), d) - 1;
        }

        // Build the local grid.
        int halo_width = halo_cell_width;
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    }

    // Get the local grid.
    const std::shared_ptr<Cajita::LocalGrid<mesh_type>> localGrid() const
    {
        return _local_grid;
    }

    // Get the cell size.
    double cellSize() const
    {
        return 1;
    }

    // Get the minimum node index in the domain.
    Kokkos::Array<int, 2> minDomainGlobalNodeIndex() const
    {
        return _min_domain_global_node_index;
    }

    // Get the maximum node index in the domain.
    Kokkos::Array<int, 2> maxDomainGlobalNodeIndex() const
    {
        return _max_domain_global_node_index;
    }

    int rank() const { return _rank; }

  public:
    std::shared_ptr<Cajita::LocalGrid<mesh_type>> _local_grid;

    Kokkos::Array<int, 2> _min_domain_global_node_index;
    Kokkos::Array<int, 2> _max_domain_global_node_index;
    int _rank;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MESH_HPP
