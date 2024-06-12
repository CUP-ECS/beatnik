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

#include <BoundaryCondition.hpp>
#include <SurfaceMesh.hpp>
#include <SpatialMesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>
#include <Migrator.hpp>

#include <ZModel.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace Beatnik
{

/*
 * Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
template <class ExecutionSpace, class MemorySpace, class Params>
class BRSolverBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using node_view = Kokkos::View<double***, device_type>;
    virtual ~BRSolverBase() = default;
    virtual void computeInterfaceVelocity(node_view zdot, node_view z, node_view w, node_view o) const = 0;
};

//---------------------------------------------------------------------------//
// // Creation method.
// template <class pm_type, class ExecutionSpace, class MemorySpace, class Params>
// std::shared_ptr<BRSolverBase<ExecutionSpace, MemorySpace, Params>>
// createBRSolver( const pm_type &pm, const BoundaryCondition &bc,
//                 const double epsilon, const double dx, const double dy,
//                 const Params params )
// {
//     if ( params.br_solver == BR_EXACT )
//     {
//         using br_type = Beatnik::ExactBRSolver<ExecutionSpace, MemorySpace, Params>;
//         // *_pm, _bc, *_spatial_mesh, *_migrator, _eps, dx, dy, _params.cutoff_distance)
//         // ExactBRSolver( const pm_type &pm, const BoundaryCondition &bc,
//         //            const double epsilon, const double dx, const double dy )

//         return std::make_shared<br_type>(
//             pm, bc, epsilon, dx, dy, params);
//     }
//     if ( params.br_solver = BR_CUTOFF )
//     {
//         using br_type = Beatnik::CutoffBRSolver<ExecutionSpace, MemorySpace, Params>
//     }
// }

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
