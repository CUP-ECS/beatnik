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
#include <ZModel.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

// The time integrator requires temporary state for runge kutta interpolation
// which are stored as part of this object 
template <class ExecutionSpace, class MemorySpace, class ModelOrder>
class TimeIntegrator
{
    using Node = Cajita::Cell;
    using exec_space = ExecutionSpace;
    using mem_space = MemorySpace;
    using device_type = Kokkos::Device<exec_space, mem_space>;
    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      device_type>;

    using halo_type = Cajita::Halo<MemorySpace>;

  public:
    TimeIntegrator( const ProblemManager<exec_space, mem_space> & pm,
                    const BoundaryCondition & bc,
                    const ZModel<exec_space, mem_space, ModelOrder> & zm )
    : _pm(pm)
    , _bc(bc)
    , _zm(zm)
    {
       
        // Create a layout of the temporary arrays we'll need for velocity
        // intermediate positions, and change in vorticity
        auto node_triple_layout =
            Cajita::createArrayLayout( pm.mesh().localGrid(), 3, Cajita::Node() );
        auto node_double_layout =
            Cajita::createArrayLayout( pm.mesh().localGrid(), 2, Cajita::Node() );

        _zdot = Cajita::createArray<double, device_type>("velocity", 
                                                       node_triple_layout);
        _wdot = Cajita::createArray<double, device_type>("vorticity derivative",
                                                       node_triple_layout);
        _ztmp = Cajita::createArray<double, device_type>("position temporary", 
                                                       node_triple_layout);
        _wtmp = Cajita::createArray<double, device_type>("vorticity temporary", 
                                                       node_triple_layout);
    }

    void step( const double delta_t ) 
    { 
        // Compute the derivatives of position and vorticityat our current point
        auto z_orig = _pm.get( Cajita::Node(), Field::Position() );
        auto w_orig = _pm.get( Cajita::Node(), Field::Vorticity() );
        auto z_tmp = _ztmp->view();
        auto w_tmp = _wtmp->view();
        auto & halo = _pm.halo(); 

        auto local_grid = _pm.mesh().localGrid();

        // TVD RK3 Step One - derivative at forward euler point
        auto z_dot = _zdot->view();
        auto w_dot = _wdot->view();

        // Find foward euler point using initial derivative. This requires up-to-date
        // Halos in the current position and vorticity.
        _pm.gather();
        _zm.computeDerivatives(z_orig, w_orig, z_dot, w_dot);

        auto own_node_space = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        Kokkos::parallel_for("RK3 Euler Step",
            Cajita::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
	        z_tmp(i, j, d) = z_orig(i, j, d) + delta_t * z_dot(i, j, d);
            }
            for (int d = 0; d < 2; d++) {
	        w_tmp(i, j, d) = w_orig(i, j, d) + delta_t * w_dot(i, j, d);
            }
        });

        // Compute derivative at forward euler point, which requires having its halos
        // (and periodic boundary ghosts) up-to-date
        halo.gather( ExecutionSpace(), *_ztmp, *_wtmp );
        _bc.apply(_pm.mesh(), *_ztmp, *_wtmp);
        _zm.computeDerivatives(z_tmp, w_tmp, z_dot, w_dot);
 
        // TVD RK3 Step Two - derivative at half-step position
        // derivatives
        
        // Take the half-step
        Kokkos::parallel_for("RK3 Half Step",
            Cajita::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
	        z_tmp(i, j, d) = 0.75*z_orig(i, j, d) 
                    + 0.25 * z_tmp(i, j, d) 
                    + 0.25 * delta_t * z_dot(i, j, d);
            }
            for (int d = 0; d < 2; d++) {
	        w_tmp(i, j, d) = 0.75*w_orig(i, j, d) 
                    + 0.25 * w_tmp(i, j, d) 
                    + 0.25 * delta_t * w_dot(i, j, d);
            }
        });
        // Get the derivatives there
        halo.gather( ExecutionSpace(), *_ztmp, *_wtmp );
        _bc.apply(_pm.mesh(), *_ztmp, *_wtmp);
        _zm.computeDerivatives(z_tmp, w_tmp, z_dot, w_dot);
        
        // TVD RK3 Step Three - Combine start, forward euler, and half step
        // derivatives to take the final full step.
        // unew = 1/3 uold + 2/3 utmp + 2/3 du_dt_tmp * deltat
        Kokkos::parallel_for("RK3 Full Step",
            Cajita::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
	        z_orig(i, j, d) = ( 1.0 / 3.0 ) * z_orig(i, j, d) 
                    + ( 2.0 / 3.0 ) * z_tmp(i, j, d) 
                    + ( 2.0 / 3.0 ) * delta_t * z_dot(i, j, d);
            }
            for (int d = 0; d < 2; d++) {
	        w_orig(i, j, d) = ( 1.0 / 3.0 ) * w_orig(i, j, d) 
                    + ( 2.0 / 3.0 ) * w_tmp(i, j, d) 
                    + ( 2.0 / 3.0 ) * delta_t * w_dot(i, j, d);
            }
        });
    }

  private:
    const ProblemManager<ExecutionSpace, MemorySpace> & _pm;
    const BoundaryCondition &_bc;
    const ZModel<ExecutionSpace, MemorySpace, ModelOrder> & _zm;
    std::shared_ptr<node_array> _zdot, _wdot, _wtmp, _ztmp;
};

} // end namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
