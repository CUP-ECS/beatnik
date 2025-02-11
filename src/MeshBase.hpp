/****************************************************************************
 * Copyright (c) 2020-2022 by the Beatnik authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is                     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_MESHBASE_HPP
#define BEATNIK_MESHBASE_HPP

#include <Beatnik_Config.hpp>

#include <Kokkos_Core.hpp>
#include <Cabana_Grid.hpp>
#include <NuMesh_Core.hpp>

#include <Beatnik_Types.hpp>
#include <Beatnik_ArrayUtils.hpp>


#include <memory>
#include <string>
#include <type_traits>

#include <mpi.h>

namespace Beatnik
{

/* Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag>
class MeshBase
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    /* Simple identifier for structured or unstructured mesh */
    using mesh_type_tag = MeshTypeTag;

    using base_type = double;

    /* Both structured and unstructured meshes hold doubles,
     * but for unstructured meshes this needs to be a 
     * Cabana::MemberTypes<...> tuple for an AoSoA.
     */
    using base_triple_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        base_type,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            Cabana::MemberTypes<base_type[3]>,
            void
        >
    >;

    using base_pair_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        base_type,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            Cabana::MemberTypes<base_type[2]>,
            void
        >
    >;

    using base_scalar_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        base_type,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            Cabana::MemberTypes<base_type>,
            void
        >
    >;

    // Set the Cabana Local Grid type as the value type or the base type of the tuple
    using cabana_local_grid_type = Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<base_type, 2>>;

    /*
     * Structured mesh is calculations are done from Cabana::Grid::Node entities
     * Unstructured mesh calcuations are done from NuMesh::Face entities
     */ 
    using entity_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        Cabana::Grid::Node,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            NuMesh::Vertex,
            void
        >
    >;

    using mesh_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<base_type, 2>>,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            NuMesh::Mesh<execution_space, memory_space>,
            void
        >
    >;

    using triple_layout_type = ArrayUtils::ArrayLayout<mesh_type, entity_type, base_triple_type>;
    using pair_layout_type = ArrayUtils::ArrayLayout<mesh_type, entity_type, base_pair_type>;
    using scalar_layout_type = ArrayUtils::ArrayLayout<mesh_type, entity_type, base_scalar_type>;
    using triple_array_type = ArrayUtils::Array<triple_layout_type, memory_space>;
    using pair_array_type = ArrayUtils::Array<pair_layout_type, memory_space>;
    using scalar_array_type = ArrayUtils::Array<scalar_layout_type, memory_space>;
    
    virtual ~MeshBase() = default;

    /*
     * Return the object needed to create an array layout
     */
    virtual std::shared_ptr<mesh_type> layoutObj(void) const = 0;
    virtual std::shared_ptr<cabana_local_grid_type> localGrid(void) const = 0;
    virtual std::shared_ptr<triple_array_type> Dx(const triple_array_type& in, const double dx) const = 0;
    virtual std::shared_ptr<triple_array_type> Dy(const triple_array_type& in, const double dy) const = 0;
    virtual std::shared_ptr<triple_array_type> omega(const pair_array_type& w, const triple_array_type& z_dx, const triple_array_type& z_dy) const = 0;
    virtual std::shared_ptr<triple_array_type> laplace(const triple_array_type& in, const double dx, const double dy) const = 0;
    virtual int is_periodic(void) const = 0;
    virtual MPI_Comm comm(void) const = 0;
    virtual int rank(void) const = 0;
    virtual int comm_size(void) const = 0;

    /**
     * Temporary function to get the periodic index space of a structured mesh,
     * until this functionality gets added into Cabana.
     * Unstructured mesh does not implement this function.
     */
    virtual Cabana::Grid::IndexSpace<2> periodicIndexSpace(Cabana::Grid::Ghost, Cabana::Grid::Node, std::array<int, 2> dir) const = 0;

    /**
     * More functions only needed for structured meshes
     * XXX - find a way to remove these?
     */
    virtual const std::array<double, 3> & boundingBoxMin() const = 0;
    virtual const std::array<double, 3> & boundingBoxMax() const = 0;
    virtual int mesh_size() const = 0;
    virtual int halo_width() const = 0;
};

} // end namespace Beantik

#endif // end BEATNIK_MESHBASE_HPP
