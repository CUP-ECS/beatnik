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
    Mesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	  const std::array<bool, 2>& periodic,
          const Cajita::BlockPartitioner<2>& partitioner,
          const int min_halo_width, MPI_Comm comm )
		  : _num_nodes( num_nodes )
    {
        MPI_Comm_rank( comm, &_rank );

        for (int i = 0; i < 3; i++) {
            _low_point[i] = global_bounding_box[i];
            _high_point[i] = global_bounding_box[i+3];
        } 

        /* Create global mesh bounds. There are a few caveats here that are
         * important to understand:
         * 1. Each mesh point has multiple locations:
         *    1.1 Its i/j location [0..n), [0...m), 
         *    1.2 Its location in node coordinate space [-n/2, n/2) based on
                  its initial spatial location in x/y space, and
         *    1.3 the x/y/z location of its points at any given time.
         * 2. Of these, the first and last are used often in calculations, no 
         *    matter the the order of the model, and the second is used to 
         *    calculate Reisz weights in every derivative calculation in 
         *    the low and medium order model. 
         * 3. In periodic meshes, the last point is implicit in the Cabana
         *    representation because it actually mirrors the first point.
         * 4. For a non-periodic model, the number of cells is one less than the 
         *    the number of nodes. For a periodic model, the number of cells is 
         *    the same as the number of nodes, with the last node being
         *    implicitly the same as the first.
         */
       
        /* Split those cells above and below 0 appropriately into coordinates that
         * are used to construct reisz weights. This mainly matters for low and medium
         * order calculations and so mainly with peiodic boundary conditions */
        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        {
            /* Even number of nodes
             * periodic -> nnodes = 4, 3 cabana nodes - 3 cells
	     *             global low == -1, global high = 2 
             *                   -> nodes = (-1,-0,1).
             * non-periodic -> nnodes = 4, 4 cabana nodes - 3 cells
             *              -> global low == -2, global high = 1 
             *                             -> nodes = (-2,-1,0,1).
             * Odd number of nodes
             * periodic -> nnodes = 5, 4 cabana nodes - 4 cells
	     *             global low == -2, global high = 2 
             *                   -> nodes = (-2,-1,0,1).
             * non-periodic -> nnodes = 5, 5 cabana nodes - 4 cells
             *              -> global low == -2, global high = 2 
             *                             -> nodes = (-2,-1,0,1,2).
	     * So we always have (nnodes - 1 cells) */

	    int cabana_nodes = num_nodes[d] - (periodic[d] ? 1 : 0);

            global_low_corner[d] = -cabana_nodes/2;
            global_high_corner[d] = global_low_corner[d] + num_nodes[d] - 1;
#if 0
            std::cout << "Dim " << d << ": " 
                      << num_nodes[d] << " nodes, "
                      << cabana_nodes << " cabana nodes, "
                      << " [ " << global_low_corner[d]
                      << ", " << global_high_corner[d] << " ]"
                      << "\n";
#endif
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

    const std::array<double, 3> & boundingBoxMin() const
    {
        return _low_point;
    }
    const std::array<double, 3> & boundingBoxMax() const
    {
        return _high_point;
    }
	
    // Get the mesh size
    int get_mesh_size() const
    {
        return _num_nodes[0];
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

  private:
    std::array<double, 3> _low_point, _high_point;
    std::shared_ptr<Cajita::LocalGrid<mesh_type>> _local_grid;
    int _rank;
	std::array<int, 2> _num_nodes;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MESH_HPP
