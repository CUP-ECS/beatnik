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

#ifndef BEATNIK_UNSTRUCTURED_MESH_HPP
#define BEATNIK_UNSTRUCTURED_MESH_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class SurfaceMesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace>
class SurfaceMesh
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using Node = Cabana::Grid::Node;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using container_layout_type = ArrayUtils::ArrayLayout<local_grid_type, Node>;
    using node_array = ArrayUtils::Array<container_layout_type, double, memory_space>;

    // Construct a mesh.
    SurfaceMesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	      const std::array<bool, 2>& periodic,
          const Cabana::Grid::BlockPartitioner<2>& partitioner,
          const int min_halo_width, MPI_Comm comm )
		        : _num_nodes( num_nodes )
                , _periodic( periodic )
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
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );
        // Build the local grid.
        _surface_halo_width = fmax(2, min_halo_width);
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, _surface_halo_width );
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

    // Get whether the mesh is periodic
    // XXX - Assumes if the x-boundary is periodic, the mesh
    // is also periodic along the y-boundary
    int is_periodic() const
    {
        return _periodic[0];
    }

    int get_halo_width() const
    {
        return _surface_halo_width;
    }

    /**
     * Compute fourth-order central difference calculation for derivatives along the 
     * interface surface
     */
    std::shared_ptr<node_array> Dx(const node_array& in, const double dx, Cabana::Grid::Node) const
    {
        using Node = Cabana::Grid::Node;
        auto out = ArrayUtils::ArrayOp::clone(in);
        auto out_view = out->array()->view();
        auto in_view = in.array()->view();
        auto layout = in.clayout()->layout();
        auto index_space = layout->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            out_view(i, j, k) = Operators::Dx(in_view, i, j, k, dx);
        });
        return out;
    }
    std::shared_ptr<node_array> Dy(const node_array& in, const double dy, Cabana::Grid::Node) const
    {
        using Node = Cabana::Grid::Node;
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        auto out_view = out->array()->view();
        auto in_view = in.array()->view();
        auto layout = in.clayout()->layout();
        auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
            for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::Dy(in_view, i, j, k, dy);
        });
        return out;
    }

    /* 9-point laplace stencil operator for computing artificial viscosity */
    // std::shared_ptr<node_array> laplace(const node_array& in, const double dx, const double dy, Cabana::Grid::Node) const
    // {
    //     using Node = Cabana::Grid::Node;
    //     auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
    //     auto out_view = out->array()->view();
    //     auto in_view = in.array()->view();
    //     auto layout = in.clayout()->layout();
    //     auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
    //     int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
    //     auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
    //     Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
    //         // double laplace(ViewType f, int i, int j, int d, double dx, double dy) 
    //         for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::laplace(in_view, i, j, k, dx, dy);
    //     });
    //     return out;
    // }

    // XXX - Assert that the mesh and node_arrays are the right type 
    // at the beginning of these functions
    // std::shared_ptr<node_array> omega(const node_array& w, const node_array& z_dx, const node_array& z_dy, Cabana::Grid::Node) const
    // {
    //     using Node = Cabana::Grid::Node;
    //     auto zdx_view = z_dx.array()->view();
    //     auto zdy_view = z_dy.array()->view();
    //     auto w_view = w.array()->view();
    //     auto layout = z_dx.clayout()->layout();
    //     auto node_triple_layout = ArrayUtils::createArrayLayout( layout->localGrid(), 3, Node() );
    //     std::shared_ptr<node_array> out = ArrayUtils::createArray<double, memory_space>("omega", 
    //                                                    node_triple_layout);
    //     auto out_view = out->array()->view();
    //     auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Node(), Cabana::Grid::Local());
    //     int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
    //     auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
    //     Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
    //         for (int k = 0; k < dim2; k++)
    //             out_view(i, j, k) = w_view(i, j, 1) * zdx_view(i, j, k) - w_view(i, j, 0) * zdy_view(i, j, k);
    //     });
    //     return out;
    // }

    int rank() const { return _rank; }

  private:
    std::array<double, 3> _low_point, _high_point;
    std::array<int, 2> _num_nodes;
    const std::array<bool, 2> _periodic;
    std::shared_ptr<local_grid_type> _local_grid;
    int _rank, _surface_halo_width;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // BEATNIK_UNSTRUCTURED_MESH_HPP