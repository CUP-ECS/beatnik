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
 * the mesh variant.
 * NOTE: Only Cabana::Grid::Node layout types are compatiable with this class.
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

template <class ExecutionSpace, class MemorySpace, class EntityType>
class ArrayLayout
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using entity_type = EntityType;
    // Define types for Cabana and NuMesh
    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;

    // The variant type that holds either Cabana or NuMesh
    using cabana_array_layout_t = Cabana::Grid::ArrayLayout<EntityType, cabana_mesh_t>;
    using numesh_array_layout_t = NuMesh::Array::ArrayLayout<EntityType, numesh_t>;
    using background_variant_t = std::variant<cabana_array_layout_t, numesh_array_layout_t>;

    // Constructor that takes either a Cabana or NuMesh object
    template <typename MeshType>
    ArrayLayout(const std::shared_ptr<MeshType>& mesh, const int dofs_per_entity, EntityType tag)
    {
        if constexpr (std::is_same_v<MeshType, cabana_t>)
        {
            printf("cabana type\n");
            auto layout = Cabana::Grid::createArrayLayout(mesh, dofs_per_entity, tag);
            _layout_background_variant = std::make_shared<background_variant_t>(cabana_array_layout_t(*layout)); 
        }
        else if constexpr (std::is_same_v<MeshType, numesh_t>)
        {
            printf("numesh type\n");
            auto layout = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag);
            _layout_background_variant = std::make_shared<background_variant_t>(numesh_array_layout_t(*layout)); 
        }
        else
        {
            printf("no type\n");
            //static_assert(false, "Unsupported mesh type provided to ArrayLayout");
        }
    }

private:
    // The shared pointer to the variant
    std::shared_ptr<background_variant_t> _layout_background_variant;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
// Define the Cabana local grid type.
using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
using cabana_grid_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;

// // Define the NuMesh type.
template <class ExecutionSpace, class MemorySpace>
using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
// /*!
//   \brief Cabana version: Create an array layout over the entities of a local grid.
//   \param local_grid The local grid over which to create the layout.
//   \param dofs_per_entity The number of degrees-of-freedom per grid entity.
//   \return Shared pointer to an ArrayLayout.
//   \note EntityType The entity: Cell, Node, Face, or Edge
// */
template <class ExecutionSpace, class MemorySpace, class EntityType>
std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>
createArrayLayout(const std::shared_ptr<Cabana::Grid::LocalGrid<cabana_mesh_t>>& cabana_grid, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>(cabana_grid, dofs_per_entity, tag);
}

template <class ExecutionSpace, class MemorySpace, class EntityType>
std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>
createArrayLayout(const std::shared_ptr<numesh_t<ExecutionSpace, MemorySpace>>& mesh, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>(mesh, dofs_per_entity, tag);
}

//---------------------------------------------------------------------------//
// Array class.
//---------------------------------------------------------------------------//
template <class ExecutionSpace, class MemorySpace, class Scalar, class EntityType, class MeshType, class... Params>
class Array
{
  public:
    // The variant type that holds either Cabana or NuMesh
    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;

    using cabana_array_layout_t = Cabana::Grid::ArrayLayout<EntityType, cabana_mesh_t>;
    using numesh_array_layout_t = NuMesh::Array::ArrayLayout<EntityType, numesh_t>;

    using cabana_array_t = Cabana::Grid::Array<Scalar, EntityType, cabana_mesh_t, Params...>;
    using numesh_array_t = NuMesh::Array::Array<Scalar, EntityType, numesh_t, Params...>;
    using background_variant_t = std::variant<cabana_array_t, numesh_array_t>;

    // Constructor that takes either a Cabana or NuMesh object
    template <typename LayoutType>
    Array(const std::string& label, const std::shared_ptr<LayoutType>& layout)
    {
        if constexpr (std::is_same_v<LayoutType, cabana_array_layout_t>)
        {
            printf("cabana type\n");
            auto array = Cabana::Grid::createArray(label, layout);
            _background_variant = std::make_shared<background_variant_t>(cabana_array_t(*array)); 
        }
        else if constexpr (std::is_same_v<LayoutType, numesh_array_layout_t>)
        {
            printf("numesh type\n");
            auto array = NuMesh::Array::createArray(label, layout);
            _background_variant = std::make_shared<background_variant_t>(numesh_array_t(*array)); 
        }
        else
        {
            printf("no type\n");
            //static_assert(false, "Unsupported mesh type provided to ArrayLayout");
        }
    }

  private:
    // The shared pointer to the variant
    std::shared_ptr<background_variant_t> _background_variant;
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
  \return Shared pointer to an Array.
*/
template <class LayoutType, class Scalar, class... Params>
std::shared_ptr<Array<LayoutType, Scalar, Params...>>
createArray(const std::string& label,
            const std::shared_ptr<LayoutType>& layout)
{
    using ExecutionSpace = typename LayoutType::execution_space;
    using MemorySpace = typename LayoutType::memory_space;
    using EntityType = typename LayoutType::entity_type;

    return std::make_shared<Array<ExecutionSpace, MemorySpace, Scalar, EntityType, cabana_mesh_t Params...>>(
        label, layout);
}


} // end namespace Array

} // end namespace Beatnik

#endif // BEATNIK_ARRAY_HPP