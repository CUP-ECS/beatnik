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

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <MeshBase.hpp>
#include <Operators.hpp>
#include <Beatnik_ArrayUtils.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class StructuredMesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag>
class StructuredMesh : public MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using mesh_type_tag = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type_tag;
    using entity_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::entity_type;
    using cabana_local_grid_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type;
    using mesh_array_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_array_type;
    using value_type = typename mesh_array_type::value_type;

    StructuredMesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	      const std::array<bool, 2>& periodic,
          const Cabana::Grid::BlockPartitioner<2>& partitioner,
          const int min_halo_width, MPI_Comm comm )
		        : _num_nodes( num_nodes )
                , _periodic( periodic )
    {
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_comm_size );

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

    // Get the object used to create array layouts, which in
    // the structured case is also the local grid.
    std::shared_ptr<cabana_local_grid_type> layoutObj() const override
    {
        return _local_grid;
    }
    std::shared_ptr<cabana_local_grid_type> localGrid() const override
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
    int mesh_size() const override
    {
        return _num_nodes[0];
    }

    // Get whether the mesh is periodic
    // XXX - Assumes if the x-boundary is periodic, the mesh
    // is also periodic along the y-boundary
    int is_periodic() const override
    {
        return _periodic[0];
    }

    int halo_width() const override
    {
        return _surface_halo_width;
    }

    /**
     * Compute fourth-order central difference calculation for derivatives along the 
     * interface surface
     */
    std::shared_ptr<mesh_array_type> Dx(const mesh_array_type& in, const double dx) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        auto out_view = out->array()->view();
        auto in_view = in.array()->view();
        auto layout = in.clayout()->layout();
        auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
            for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::Dx(in_view, i, j, k, dx);
        });
        return out;
    }
    std::shared_ptr<mesh_array_type> Dy(const mesh_array_type& in, const double dy) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        auto out_view = out->array()->view();
        auto in_view = in.array()->view();
        auto layout = in.clayout()->layout();
        auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate Dy", policy, KOKKOS_LAMBDA(const int i, const int j) {
            for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::Dy(in_view, i, j, k, dy);
        });
        return out;
    }

    /* 9-point laplace stencil operator for computing artificial viscosity */
    std::shared_ptr<mesh_array_type> laplace(const mesh_array_type& in, const double dx, const double dy) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        auto out_view = out->array()->view();
        auto in_view = in.array()->view();
        auto layout = in.clayout()->layout();
        auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate laplace", policy, KOKKOS_LAMBDA(const int i, const int j) {
            // double laplace(ViewType f, int i, int j, int d, double dx, double dy) 
            for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::laplace(in_view, i, j, k, dx, dy);
        });
        // out views not the same at 0, 7.
        // printView(local_L2G, out->array()->view(), 2, 0, 7);
        return out;
    }

    // XXX - Assert that the mesh and mesh_array_types are the right type 
    // at the beginning of these functions
    std::shared_ptr<mesh_array_type> omega(const mesh_array_type& w, const mesh_array_type& z_dx, const mesh_array_type& z_dy) const override
    {
        using Node = Cabana::Grid::Node;
        auto zdx_view = z_dx.array()->view();
        auto zdy_view = z_dy.array()->view();
        auto w_view = w.array()->view();
        auto layout = z_dx.clayout()->layout();
        auto node_triple_layout = ArrayUtils::createArrayLayout<value_type>( layout->localGrid(), 3, Node() );
        std::shared_ptr<mesh_array_type> out = ArrayUtils::createArray<memory_space>("omega", 
                                                       node_triple_layout);
        auto out_view = out->array()->view();
        auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Node(), Cabana::Grid::Local());
        int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        Kokkos::parallel_for("Calculate Omega", policy, KOKKOS_LAMBDA(const int i, const int j) {
            for (int k = 0; k < dim2; k++)
                out_view(i, j, k) = w_view(i, j, 1) * zdx_view(i, j, k) - w_view(i, j, 0) * zdy_view(i, j, k);
        });
        return out;
    }

    // Get the boundary indexes on the periodic boundary. local_grid.boundaryIndexSpace()
    // doesn't work on periodic boundaries.
    // XXX Needs more error checking to make sure the boundary is in fact periodic
    Cabana::Grid::IndexSpace<2>
    periodicIndexSpace(Cabana::Grid::Ghost, Cabana::Grid::Node, std::array<int, 2> dir) const override
    {
        auto & global_grid = _local_grid->globalGrid();
        for ( int d = 0; d < 2; d++ ) {
            if ((dir[d] == -1 && global_grid.onLowBoundary(d))
                || (dir[d] == 1 && global_grid.onHighBoundary(d))) {
                return _local_grid->sharedIndexSpace(Cabana::Grid::Ghost(), Cabana::Grid::Node(), dir);
            }
        }

        std::array<long, 2> zero_size;
        for ( std::size_t d = 0; d < 2; ++d )
            zero_size[d] = 0;
        return Cabana::Grid::IndexSpace<2>( zero_size, zero_size );
    }

    int rank() const override { return _rank; }
    int comm_size() const override { return _comm_size; }

    template <class l2g_type, class View>
    void printView(l2g_type local_L2G, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
	    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, execution_space()),
            KOKKOS_LAMBDA(int i, int j) {
            
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // global i, j
            local_L2G(local_li, local_gi);
            if (option == 1){
                if (dims == 3) {
                    printf("%d %d %.12lf %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("%d %d %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                }
                else if (dims == 1) {
                    printf("%d %d %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                    printf("%d %d %.12lf %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }
                    else if (dims == 2) {
                        printf("%d %d %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                    }
                    else if (dims == 1) {
                    printf("%d %d %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0));
                }
                }
            }
        });
    }

  private:
    std::array<double, 3> _low_point, _high_point;
    std::array<int, 2> _num_nodes;
    const std::array<bool, 2> _periodic;
    std::shared_ptr<cabana_local_grid_type> _local_grid;
    int _rank, _comm_size, _surface_halo_width;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MESH_HPP
