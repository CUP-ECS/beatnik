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

template <class EntityType, class BackgroundType>
class ArrayLayout
{
  public:
    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using background_type = BackgroundType;

    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::ArrayLayout<entity_type, background_type>;

    using execution_space = typename background_type::execution_space;
    using memory_space = typename background_type::memory_space;
    using numesh_t = NuMesh::Mesh<execution_space, memory_space>;

    // using 

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = background_type::num_space_dim;

    /*! 
      \brief Constructor that handles both Cabana::Grid and NuMesh layouts. 
      \param local_grid The local grid or mesh over which the layout will be constructed.
      \param dofs_per_entity The number of degrees-of-freedom per entity.
    */
    ArrayLayout( const std::shared_ptr<background_type>& local_grid,
                 const int dofs_per_entity, EntityType )
    {
        if constexpr (std::is_same_v<background_type, Cabana::Grid::LocalGrid<cabana_mesh_t>>) 
        {
            _background_variant = cabana_t(Cabana::Grid::createArrayLayout(local_grid, dofs_per_entity, EntityType()));
        }
        else if constexpr (std::is_same_v<background_type, numesh_t>) 
        {
            _background_variant = numesh_t(NuMesh::Array::createArrayLayout(local_grid, dofs_per_entity, EntityType()));
        }
        else 
        {
            static_assert(false, "Unsupported background type");
        }
    }

    //! Get the local grid over which this layout is defined.
    // const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    // {
    //     return _local_grid;
    // }

    //! Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const { return _dofs_per_entity; }

    //! Get the index space of the array elements in the given
    //! decomposition.
    // template <class DecompositionTag, class IndexType>
    // IndexSpace<num_space_dim + 1>
    // indexSpace( DecompositionTag decomposition_tag, IndexType index_type ) const
    // {
    //     return appendDimension( _local_grid->indexSpace( decomposition_tag,
    //                                                      EntityType(),
    //                                                      index_type ),
    //                             _dofs_per_entity );
    // }

    // /*!
    //   Get the local index space of the array elements we shared with the
    //   given neighbor in the given decomposition.

    //   \param decomposition_tag Decomposition type: Own or Ghost
    //   \param off_ijk %Array of neighbor offset indices.
    //   \param halo_width Optional depth of shared indices within the halo. Must
    //   be less than or equal to the halo width of the local grid. Default is to
    //   use the halo width of the local grid.
    // */
    // template <class DecompositionTag>
    // IndexSpace<num_space_dim + 1>
    // sharedIndexSpace( DecompositionTag decomposition_tag,
    //                   const std::array<int, num_space_dim>& off_ijk,
    //                   const int halo_width = -1 ) const
    // {
    //     return appendDimension(
    //         _local_grid->sharedIndexSpace( decomposition_tag, EntityType(),
    //                                        off_ijk, halo_width ),
    //         _dofs_per_entity );
    // }

    // /*!
    //   Get the local index space of the array elements we shared with the
    //   given neighbor in the given decomposition.

    //   \param decomposition_tag Decomposition type: Own or Ghost
    //   \param off_i, off_j, off_k Neighbor offset index in a given dimension.
    //   \param halo_width Optional depth of shared indices within the halo. Must
    //   be less than or equal to the halo width of the local grid. Default is to
    //   use the halo width of the local grid.
    // */
    // template <class DecompositionTag, std::size_t NSD = num_space_dim>
    // std::enable_if_t<3 == NSD, IndexSpace<4>>
    // sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
    //                   const int off_j, const int off_k,
    //                   const int halo_width = -1 ) const
    // {
    //     std::array<int, 3> off_ijk = { off_i, off_j, off_k };
    //     return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    // }

    // /*!
    //   Get the local index space of the array elements we shared with the
    //   given neighbor in the given decomposition.

    //   \param decomposition_tag Decomposition type: Own or Ghost
    //   \param off_i, off_j Neighbor offset index in a given dimension.
    //   \param halo_width Optional depth of shared indices within the halo. Must
    //   be less than or equal to the halo width of the local grid. Default is to
    //   use the halo width of the local grid.
    // */
    // template <class DecompositionTag, std::size_t NSD = num_space_dim>
    // std::enable_if_t<2 == NSD, IndexSpace<3>>
    // sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
    //                   const int off_j, const int halo_width = -1 ) const
    // {
    //     std::array<int, 2> off_ijk = { off_i, off_j };
    //     return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    // }

  private:
    // Pointer to local grid (Cabana) or mesh (NuMesh)
    std::shared_ptr<std::variant<cabana_t, numesh_t>> _background_variant;
    //std::shared_ptr<LocalGrid<MeshType>> _local_grid;
    // int _dofs_per_entity;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Cabana version: Create an array layout over the entities of a local grid.
  \param local_grid The local grid over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
  \return Shared pointer to an ArrayLayout.
  \note EntityType The entity: Cell, Node, Face, or Edge
*/
template <class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType>>
createArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                   const int dofs_per_entity, EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType>>(
        local_grid, dofs_per_entity );
}
/*!
  \brief NuMesh version
*/
template <class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType>>
createArrayLayout( const std::shared_ptr<MeshType>& mesh,
                   const int dofs_per_entity, EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType>>(
        mesh, dofs_per_entity );
}


} // end namespace Array

} // end namespace Beatnik

#endif // BEATNIK_ARRAY_HPP