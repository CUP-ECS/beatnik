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
#include <StructuredMesh.hpp>
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
template <class ProblemManagerType, class Params>
class BRSolverBase
{
  public:
    using memory_space = typename ProblemManagerType::memory_space;
    using value_type = typename ProblemManagerType::beatnik_mesh_type::value_type;
    using view_t = Kokkos::View<value_type***, memory_space>;
    virtual ~BRSolverBase() = default;
    virtual void computeInterfaceVelocity(view_t zdot, view_t z, view_t o) const = 0;
};

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
