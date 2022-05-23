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
    Mesh( const Kokkos::Array<double, 4>& global_bounding_box,
          const std::array<int, 2>& global_num_cell,
          const Cajita::BlockPartitioner<2>& partitioner,
          const int halo_cell_width, MPI_Comm comm )
    {
        // Make a copy of the global number of cells so we can modify it.
        std::array<int, 2> num_cell = global_num_cell;

        // Compute the cell size.
        double cell_size =
            ( global_bounding_box[2] - global_bounding_box[0] ) / num_cell[0];

        // Because the mesh is uniform width in all directions, check that the
        // domain is evenly divisible by the cell size in each dimension
        // within round-off error.
        for ( int d = 0; d < 2; ++d )
        {
            double extent = num_cell[d] * cell_size;
            if ( std::abs( extent - ( global_bounding_box[d + 2] -
                                      global_bounding_box[d] ) ) >
                 double( 10.0 ) * std::numeric_limits<double>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        {
            global_low_corner[d] = global_bounding_box[d];
            global_high_corner[d] = global_bounding_box[d + 2];
        }

        for ( int d = 0; d < 2; ++d )
        {
            _min_domain_global_cell_index[d] = 0;
            _max_domain_global_cell_index[d] = num_cell[d] - 1;
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, num_cell );

        // Build the global grid.
        std::array<bool, 2> periodic;
        for ( int i = 0; i < 2; i++ )
            periodic[i] = false;

        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );

        // Build the local grid.
        int halo_width = halo_cell_width;
        _local_grid = Cajita::createLocalGrid( global_grid, halo_width );

        MPI_Comm_rank( comm, &_rank );
    }

    // Get the local grid.
    const std::shared_ptr<Cajita::LocalGrid<mesh_type>>& localGrid() const
    {
        return _local_grid;
    }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().cellSize( 0 );
    }

    // Get the minimum node index in the domain.
    Kokkos::Array<int, 2> minDomainGlobalCellIndex() const
    {
        return _min_domain_global_cell_index;
    }

    // Get the maximum node index in the domain.
    Kokkos::Array<int, 2> maxDomainGlobalCellIndex() const
    {
        return _max_domain_global_cell_index;
    }

    int rank() const { return _rank; }

  public:
    std::shared_ptr<Cajita::LocalGrid<mesh_type>> _local_grid;

    Kokkos::Array<int, 2> _min_domain_global_cell_index;
    Kokkos::Array<int, 2> _max_domain_global_cell_index;
    int _rank;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MESH_HPP
