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
 * Array and ArrayLayouts that use Cabana::Grid::Arrays or an unstructured version
 * based on Cabana::AoSoAs.
 */

#ifndef BEATNIK_ARRAY_HPP
#define BEATNIK_ARRAY_HPP

namespace Beatnik
{

template <class EntityType, class MeshType>
class ArrayLayout
{
  public:
    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using mesh_type = MeshType;

    using LocalGrid_t = LocalGrid<MeshType>;
    // using 

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
      \brief Constructor.
      \param local_grid The local grid over which the layout will be
      constructed.
      \param dofs_per_entity The number of degrees-of-freedom per grid entity.
    */
    ArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                 const int dofs_per_entity )
        : _local_grid( local_grid )
        , _dofs_per_entity( dofs_per_entity )
    {
    }

    //! Get the local grid over which this layout is defined.
    const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    {
        return _local_grid;
    }

    //! Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const { return _dofs_per_entity; }

    //! Get the index space of the array elements in the given
    //! decomposition.
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim + 1>
    indexSpace( DecompositionTag decomposition_tag, IndexType index_type ) const
    {
        return appendDimension( _local_grid->indexSpace( decomposition_tag,
                                                         EntityType(),
                                                         index_type ),
                                _dofs_per_entity );
    }

    /*!
      Get the local index space of the array elements we shared with the
      given neighbor in the given decomposition.

      \param decomposition_tag Decomposition type: Own or Ghost
      \param off_ijk %Array of neighbor offset indices.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag>
    IndexSpace<num_space_dim + 1>
    sharedIndexSpace( DecompositionTag decomposition_tag,
                      const std::array<int, num_space_dim>& off_ijk,
                      const int halo_width = -1 ) const
    {
        return appendDimension(
            _local_grid->sharedIndexSpace( decomposition_tag, EntityType(),
                                           off_ijk, halo_width ),
            _dofs_per_entity );
    }

    /*!
      Get the local index space of the array elements we shared with the
      given neighbor in the given decomposition.

      \param decomposition_tag Decomposition type: Own or Ghost
      \param off_i, off_j, off_k Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<4>>
    sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
                      const int off_j, const int off_k,
                      const int halo_width = -1 ) const
    {
        std::array<int, 3> off_ijk = { off_i, off_j, off_k };
        return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    }

    /*!
      Get the local index space of the array elements we shared with the
      given neighbor in the given decomposition.

      \param decomposition_tag Decomposition type: Own or Ghost
      \param off_i, off_j Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag, std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, IndexSpace<3>>
    sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
                      const int off_j, const int halo_width = -1 ) const
    {
        std::array<int, 2> off_ijk = { off_i, off_j };
        return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    }

  private:
    std::shared_ptr<LocalGrid<MeshType>> _local_grid;
    int _dofs_per_entity;
};

} // end namespace Beatnik

#endif // BEATNIK_ARRAY_HPP