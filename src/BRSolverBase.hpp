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
#include <ExactBRSolver.hpp>
#include <Migrator.hpp>

#include <ZModel.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace Beatnik
{

// Type tags designating different BRSolvers
namespace BRSolver
{
    struct Exact {};

    struct Cutoff {};
} // namespace BRSolver

/*
 * Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
template <class ExecutionSpace, class MemorySpace>
class BRSolverBase
{
  public:
    
    virtual ~BRSolverBase() = default;

    template <class node_view>
    virtual void computeInterfaceVelocity(node_view zdot, node_view z, node_view w, node_view o) const = 0;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class pm_type, class spatial_mesh_type, class BRSolverType>
std::shared_ptr<BRSolverBase<class ExecutionSpace, class MemorySpace>>
createBRSolver( const BRSolverType,
                const pm_type &pm, const BoundaryCondition &bc, const spatial_mesh_type &spm,
                const double epsilon, const double dx, const double dy,
                const double cutoff_distance )
{
    if ( BRSolverType() == BRSolver::Exact )
    {
        return std::make_shared<
            Beatnik::BRSolver<ExecutionSpace, MemorySpace>>(
            comm, global_bounding_box, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, cutoff_distance);
    }
}

}

#endif // end BEATNIK_BRSOLVERBASE_HPP
