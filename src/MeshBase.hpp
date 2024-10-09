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

#include <Cabana_Grid.hpp>

#include <Beatnik_Types.hpp>
#include <BoundaryCondition.hpp>
#include <SurfaceMesh.hpp>
#include <SpatialMesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>

#include <ZModel.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace Beatnik
{

/* Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
template <class ExecutionSpace, class MemorySpace, class MeshType, class Scalar, class EntityType, class Params>
class MeshBase
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

    using entity_type = typename ContainerLayoutType::entity_type;
    using mesh_type   = typename ContainerLayoutType::mesh_type;
    using layout_type = typename ContainerLayoutType::array_layout_type;
    using value_type = Scalar;

    using array_layout_type = std::conditional_t<
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::ArrayLayout<entity_type, cabana_mesh_type<mesh_type>>, // Case A: Cabana UniformMesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<MeshType>::value,
            NuMesh::Array::ArrayLayout<entity_type, mesh_type>, // Case B: NuMesh Mesh
            void // Fallback type or an error type if neither condition is satisfied
        >
    >;

    // Determine array_type using std::conditional_t
    using array_type = std::conditional_t<
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::Array<value_type, entity_type, cabana_mesh_type<mesh_type>, memory_space>, // Case A: Cabana Mesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<mesh_type>::value,
            NuMesh::Array::Array<value_type, entity_type, mesh_type, memory_space>, // Case B: NuMesh Mesh
            void // Fallback or error type if neither condition is satisfied
        >
    >;
    virtual ~MeshBase() = default;
    virtual void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const = 0;
};

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
