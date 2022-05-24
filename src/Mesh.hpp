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
        // Make a copy of the global number of cells so we can modify it.
        std::array<int, 2> num_cells = global_num_cells;

        // Create global mesh bounds.
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        {
            global_low_corner[d] = 0;
            global_high_corner[d] = global_num_cells[d];
        }

        for ( int d = 0; d < 2; ++d )
        {
            _min_domain_global_node_index[d] = 0;
            _max_domain_global_node_index[d] = num_cells[d] - 1;
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );

        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );

        // Build the local grid.
        int halo_width = halo_cell_width;
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );

        MPI_Comm_rank( comm, &_rank );
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
