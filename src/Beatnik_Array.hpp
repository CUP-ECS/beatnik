/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Array and ArrayLayouts that use Cabana::Grid::Arrays or NuMesh::Arrays depending on
 * the mesh variant
 */

#ifndef BEATNIK_ARRAY_HPP
#define BEATNIK_ARRAY_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include <variant> 

namespace Beatnik
{
namespace Array
{

template <class ExecutionSpace, class MemorySpace>
class ArrayLayout
{
public:
    // Define types for Cabana and NuMesh
    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;

    // The variant type that holds either Cabana or NuMesh
    using background_variant_t = std::variant<cabana_t, numesh_t>;

    // Constructor that takes either a Cabana or NuMesh object
    template <typename MeshType>
    ArrayLayout(const std::shared_ptr<MeshType>& mesh)
    {
        initialize(mesh);
    }

private:
    // Method to initialize the background variant based on the mesh type
    template <typename MeshType>
    void initialize(const std::shared_ptr<MeshType>& mesh)
    {
        if constexpr (std::is_same_v<MeshType, cabana_t>)
        {
            //_background_variant = std::make_shared<background_variant_t>(cabana_t(*mesh));
        }
        else if constexpr (std::is_same_v<MeshType, numesh_t>)
        {
            //_background_variant = std::make_shared<background_variant_t>(numesh_t(*mesh));
        }
        else
        {
            //static_assert(false, "Unsupported mesh type provided to ArrayLayout");
        }
    }

    // The shared pointer to the variant
    std::shared_ptr<background_variant_t> _background_variant;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
// Define the Cabana local grid type.
// using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
// using cabana_grid_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;

// // Define the NuMesh type.
// template <class ExecutionSpace, class MemorySpace>
// using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
// /*!
//   \brief Cabana version: Create an array layout over the entities of a local grid.
//   \param local_grid The local grid over which to create the layout.
//   \param dofs_per_entity The number of degrees-of-freedom per grid entity.
//   \return Shared pointer to an ArrayLayout.
//   \note EntityType The entity: Cell, Node, Face, or Edge
// */
// template <class ExecutionSpace, class MemorySpace>
// std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace>>
// createArrayLayout(const std::shared_ptr<cabana_grid_t>& cabana_grid)
// {
//     return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace>>(cabana_grid);
// }

// /*!
//   \brief NuMesh version: Create an array layout over the entities of a mesh.
//   \param numesh The NuMesh mesh over which to create the layout.
//   \return Shared pointer to an ArrayLayout.
// */
// template <class ExecutionSpace, class MemorySpace>
// std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace>>
// createArrayLayout(const std::shared_ptr<numesh_t<ExecutionSpace, MemorySpace>>& numesh)
// {
//     return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace>>(numesh);
// }


} // end namespace Array

} // end namespace Beatnik

#endif // BEATNIK_ARRAY_HPP