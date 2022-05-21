/****************************************************************************
 * Copyright (c) 2022 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is                     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_TIMEINTEGRATOR_HPP
#define BEATNIK_TIMEINTEGRATOR_HPP

#include <BoundaryCondition.hpp>
#include <ProblemManager.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

// The time integrator requires temporary state for runge kutta interpolation
// which are stored as part of this object 
template <class ProblemManagerType>
class TimeIntegrator
{
  using Node = Cajita::Cell;
  using exec_space = typename ProblemManagerType::execution_space;
  using mem_space = typename ProblemManagerType::memory_space;
  using mesh_type = typename ProblemManagerType::mesh_type;
  using halo_type = typename ProblemManagerType::halo_type;
  using node_array = typename ProblemManagerType::array_type;

  public:
    TimeIntegrator( const std::shared_ptr<mesh_type> & mesh, ZModelType & zm, ProblemManagerType & pm, BoundaryCondition &bc )
    : _mesh(mesh)
    , _pm(pm)
    , _bc(bc)
    {
        // Create a layout of the temporary arrays we'll need for velocity
        // intermediate positions, and change in vorticity
        auto node_triple_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 3, Cajita::Node() );
        auto node_double_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Node() );

        _zdot = Cajita::createArray<double, mem_space>("velocity", 
                                                       node_triple_layout);
        _wdot = Cajita::createArray<double, mem_space>("vorticity derivative",
                                                       node_triple_layout);
        _ztmp = Cajita::createArray<double, mem_space>("position temporary", 
                                                       node_triple_layout);
        _wtmp = Cajita::createArray<double, mem_space>("vorticity temporary", 
                                                       node_triple_layout);
    }

    void step( const double delta_t ) 
    { 
        // Compute the derivatives of position and vorticityat our current point
        auto z_orig = pm.get(Cajita::Node(), Field::Position());
        auto w_orig = pm.get(Cajita::Node(), Field::Vorticity());

        // TVD RK3 Step One - derivative at forward euler point
        //zm.computeDerivatives(z_orig, w_orig, _zdot, _wdot);
        //Compute derivative at forward euler point
        //parallel_for();
        //zm.computeDerivatives(_ztmp, _wtmp, _zdot, _wdot, _bc);
 
        // TVD RK3 Step Two - derivative at half-step position
        // derivatives
        //parallel_for();
        //zm.computeDerivatives(_ztmp, _wtmp, _zdot, _wdot, _bc);
        
        // TVD RK3 Step Three - Combine start, forward euler, and half step
        // derivatives to take the final full step.
        // unew = 1/3 uold + 2/3 utmp + 2/3 du_dt_tmp * deltat (classic)
        //parallel_for();
    }

  private:
    node_array _vtmp, _wtmp, _ztmp;
};

//---------------------------------------------------------------------------//
// Advect a field in a divergence-free velocity field
template <std::size_t NumSpaceDims, class ProblemManagerType,
          class ExecutionSpace, class Entity_t, class Field_t>
void advect( ExecutionSpace& exec_space, ProblemManagerType& pm, double delta_t,
             [[maybe_unused]] const BoundaryCondition<NumSpaceDims>& bc,
             Entity_t entity, Field_t field )
{
    auto field_current = pm.get( entity, field, Version::Current() );
    auto field_next = pm.get( entity, field, Version::Next() );

    auto u = pm.get( FaceI(), Field::Velocity(), Version::Current() );
    auto v = pm.get( FaceJ(), Field::Velocity(), Version::Current() );
    //    auto w = pm.get(FaceK(), Field::Velocity(), Version::Current());

    auto local_grid = pm.mesh()->localGrid();
    auto local_mesh = *( pm.mesh()->localMesh() );

    auto owned_items =
        local_grid->indexSpace( Cajita::Own(), entity, Cajita::Local() );
    parallel_for(
        "advection loop", createExecutionPolicy( owned_items, exec_space ),
        KOKKOS_LAMBDA( int i, int j ) {
            int idx[2] = { i, j };
            double start[NumSpaceDims], trace[NumSpaceDims];
            // 1. Get the location of the entity in question
            local_mesh.coordinates( entity, idx, start );

            // 2. Trace the location back through the velocity field
            rk3<NumSpaceDims>( start, local_mesh, u, v, delta_t, trace );

            // 3. Interpolate the value of the advected quantity at that
            // location
            field_next( i, j, 0 ) =
                Interpolation::interpolateField<NumSpaceDims, 3, Entity_t>(
                    trace, local_mesh, field_current );
        } );
}

//---------------------------------------------------------------------------//
// Take a time step.
template <std::size_t NumSpaceDims, class ProblemManagerType,
          class ExecutionSpace>
void step( const ExecutionSpace& exec_space, ProblemManagerType& pm,
           const double delta_t, const BoundaryCondition<NumSpaceDims>& bc )
{
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step" );

    // Get up-to-date copies of the fields being advected and the velocity field
    // into the ghost cells so we can interpolate velocity correctly and
    // retrieve the value being advected into owned cells
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Gather" );
    pm.gather( Version::Current() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Advect" );
    // Advect the fields we care about into the next versions of the fields
    Kokkos::Profiling::pushRegion( "TimeIntegrator::Step::Advect::Quantity" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, Cell(),
                          Field::Quantity() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(
        "TimeIntegrator::Step::Advect::Velocity::FaceI" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, FaceI(),
                          Field::Velocity() );
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(
        "TimeIntegrator::Step::Advect::Velocity::FaceJ" );
    advect<NumSpaceDims>( exec_space, pm, delta_t, bc, FaceJ(),
                          Field::Velocity() );
    Kokkos::Profiling::popRegion();

    if constexpr ( NumSpaceDims == 3 )
    {
        Kokkos::Profiling::pushRegion(
            "TimeIntegrator::Step::Advect::Velocity::FaceK" );
        advect<NumSpaceDims>( exec_space, pm, delta_t, FaceK(),
                              Field::Velocity() );
        Kokkos::Profiling::popRegion();
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion( "Beatnik::TimeIntegrator::Advance" );
    // Once all calculations with the current versions of the fields (including
    // Velocity!) are done, swap the old values with the new one to finish the
    // time step.
    pm.advance( Cell(), Field::Quantity() );
    pm.advance( FaceI(), Field::Velocity() );
    pm.advance( FaceJ(), Field::Velocity() );
    if constexpr ( NumSpaceDims == 3 )
    {
        pm.advance( FaceK(), Field::Velocity() );
        Kokkos::Profiling::popRegion();
    }

    Kokkos::Profiling::popRegion();
}

//---------------------------------------------------------------------------//

} // end namespace TimeIntegrator
} // end namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
