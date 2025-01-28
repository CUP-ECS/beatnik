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

#include <MeshBase.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class StructuredMesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag>
class UnstructuredMesh : public MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using mesh_type_tag = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type_tag;
    using entity_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::entity_type;
    using mesh_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type;
    using mesh_array_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_array_type;
    using cabana_local_grid_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::cabana_local_grid_type;

    UnstructuredMesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	      const std::array<bool, 2>& periodic,
          const Cabana::Grid::BlockPartitioner<2>& partitioner,
          const int min_halo_width, MPI_Comm comm )
		        : _comm( comm )
                , _num_nodes( num_nodes )
                , _periodic( periodic )
    {
        MPI_Comm_rank( _comm, &_rank );

        // Copy the same code used to create the local grid in the constructor
        // of StructuredMesh
        // XXX - Don't copy-paste code
        for (int i = 0; i < 3; i++) {
            _low_point[i] = global_bounding_box[i];
            _high_point[i] = global_bounding_box[i+3];
        } 

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
        }

        // Finally, create the global mesh, global grid, and local grid.
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );
        // Build the local grid.
        int surface_halo_width = 1; // Halo width doesn't matter here
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, surface_halo_width );


        _mesh = NuMesh::createEmptyMesh<execution_space, memory_space>(_comm);

        // Initialize the mesh with garbage position and vorticity values
        // auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
        // auto array = Cabana::Grid::createArray<double, memory_space>("for_initialization", layout);
        // _mesh->initializeFromArray(*array);
    }

    std::shared_ptr<mesh_type> layoutObj() const override
    {
        return _mesh;
    }

    std::shared_ptr<cabana_local_grid_type> localGrid() const
    {
        return _local_grid;
    }

    // Get whether the mesh is periodic
    // XXX - Assumes if the x-boundary is periodic, the mesh
    // is also periodic along the y-boundary
    int is_periodic() const override
    {
        return _periodic[0];
    }

    /**
     * Compute fourth-order central difference calculation for derivatives along the 
     * interface surface
     */
    std::shared_ptr<mesh_array_type> Dx(const mesh_array_type& in, const double dx) const override
    {
        auto out = ArrayUtils::ArrayOp::clone(in);
        // auto out_view = out->array()->view();
        // auto in_view = in.array()->view();
        // auto layout = in.clayout()->layout();
        // auto index_space = layout->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
        //     out_view(i, j, k) = Operators::Dx(in_view, i, j, k, dx);
        // });
        return out;
    }
    std::shared_ptr<mesh_array_type> Dy(const mesh_array_type& in, const double dy) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        // auto out_view = out->array()->view();
        // auto in_view = in.array()->view();
        // auto layout = in.clayout()->layout();
        // auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
        //     for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::Dy(in_view, i, j, k, dy);
        // });
        return out;
    }

    /* 9-point laplace stencil operator for computing artificial viscosity */
    std::shared_ptr<mesh_array_type> laplace(const mesh_array_type& in, const double dx, const double dy) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        // auto out_view = out->array()->view();
        // auto in_view = in.array()->view();
        // auto layout = in.clayout()->layout();
        // auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
        //     // double laplace(ViewType f, int i, int j, int d, double dx, double dy) 
        //     for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::laplace(in_view, i, j, k, dx, dy);
        // });
        return out;
    }

    // XXX - Assert that the mesh and mesh_array_types are the right type 
    // at the beginning of these functions
    std::shared_ptr<mesh_array_type> omega(const mesh_array_type& w, const mesh_array_type& z_dx, const mesh_array_type& z_dy) const override
    {
        // using Node = Cabana::Grid::Node;
        // auto zdx_view = z_dx.array()->view();
        // auto zdy_view = z_dy.array()->view();
        // auto w_view = w.array()->view();
        // auto layout = z_dx.clayout()->layout();
        // auto node_triple_layout = ArrayUtils::createArrayLayout( layout->localGrid(), 3, Node() );
        // std::shared_ptr<mesh_array_type> out = ArrayUtils::createArray<double, memory_space>("omega", 
        //                                                node_triple_layout);
        // auto out_view = out->array()->view();
        // auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Node(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
        //     for (int k = 0; k < dim2; k++)
        //         out_view(i, j, k) = w_view(i, j, 1) * zdx_view(i, j, k) - w_view(i, j, 0) * zdy_view(i, j, k);
        // });
        return NULL;
    }

    /**
     * These functions are only needed for structured meshes, but it must be overridden here
     * to avoid errors.
     */
    Cabana::Grid::IndexSpace<2>
    periodicIndexSpace(Cabana::Grid::Ghost, Cabana::Grid::Node, std::array<int, 2> dir) const override
    { 
        std::array<long, 2> zero_size;
        for ( std::size_t d = 0; d < 2; ++d )
            zero_size[d] = 0;
        return Cabana::Grid::IndexSpace<2>( zero_size, zero_size );
    }
    const std::array<double, 3> & boundingBoxMin() const override {return _low_point;}
    const std::array<double, 3> & boundingBoxMax() const override {return _high_point;}
    int mesh_size() const override { return -1; }
    int halo_width() const override { return -1; }

    int rank() const override { return _rank; }

  private:
    MPI_Comm _comm;
    int _rank;

    std::array<int, 2> _num_nodes;
    const std::array<bool, 2> _periodic;
    
    std::array<double, 3> _low_point, _high_point;
    
    // The local grid the unstructured mesh is built from
    std::shared_ptr<cabana_local_grid_type> _local_grid;

    // Unstructured mesh object
    std::shared_ptr<mesh_type> _mesh;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // BEATNIK_UNSTRUCTURED_MESH_HPP