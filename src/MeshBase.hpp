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

#ifndef BEATNIK_BRSOLVERBASE_HPP
#define BEATNIK_BRSOLVERBASE_HPP

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
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag, class EntityType, class Scalar>
class MeshBase
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using entity_type = EntityType;
    using value_type = Scalar;

    using mesh_type = std::conditional_t<
        std::is_same_v<MeshTypeTag, Mesh::Structured>,
        Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<value_type, 2>>,
        std::conditional_t<
            std::is_same_v<MeshTypeTag, Mesh::Unstructured>,
            NuMesh::Mesh<execution_space, memory_space>,
            void
        >
    >;

    using container_layout_type = ArrayUtils::ArrayLayout<mesh_type, entity_type>;
    using mesh_array_type = ArrayUtils::Array<container_layout_type, value_type, memory_space>;
    
    virtual ~MeshBase() = default;
    virtual mesh_array_type Dx(const mesh_array_type& in, const double dx) const = 0;
    virtual mesh_array_type Dy(const mesh_array_type& in, const double dy) const = 0;
    virtual mesh_array_type omega(const mesh_array_type& w, const mesh_array_type& z_dx, const mesh_array_type& z_dy) const = 0;
    virtual mesh_array_type laplace(const mesh_array_type& in, const double dx, const double dy) const = 0;
};

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
