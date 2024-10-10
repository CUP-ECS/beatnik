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

    UnstructuredMesh( MPI_Comm comm, const std::array<bool, 2>& periodic )
		: _comm( comm )
        , _periodic( periodic )
    {
        MPI_Comm_rank( comm, &_rank );

        _mesh = NuMesh::createMesh<execution_space, memory_space>(_comm);
    }

    std::shared_ptr<mesh_type> layoutObj() const override
    {
        return _mesh;
    }

    // Get whether the mesh is periodic
    // XXX - Assumes if the x-boundary is periodic, the mesh
    // is also periodic along the y-boundary
    int is_periodic() const
    {
        return _periodic[0];
    }

    /**
     * Compute fourth-order central difference calculation for derivatives along the 
     * interface surface
     */
    std::shared_ptr<mesh_array_type> Dx(const mesh_array_type& in, const double dx, Cabana::Grid::Node) const override
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
    std::shared_ptr<mesh_array_type> Dy(const mesh_array_type& in, const double dy, Cabana::Grid::Node) const override
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
    std::shared_ptr<mesh_array_type> laplace(const mesh_array_type& in, const double dx, const double dy, Cabana::Grid::Node) const override
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
    std::shared_ptr<mesh_array_type> omega(const mesh_array_type& w, const mesh_array_type& z_dx, const mesh_array_type& z_dy, Cabana::Grid::Node) const override
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

    int rank() const { return _rank; }

  private:
    MPI_Comm _comm;
    const std::array<bool, 2> _periodic;
    int _rank;
    // std::array<double, 3> _low_point, _high_point;
    // std::array<int, 2> _num_nodes;
    // const std::array<bool, 2> _periodic;
    std::shared_ptr<mesh_type> _mesh;
    
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // BEATNIK_UNSTRUCTURED_MESH_HPP