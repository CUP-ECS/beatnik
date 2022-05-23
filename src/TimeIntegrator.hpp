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
class TimeIntegrator;

template <class ExecutionSpace, class MemorySpace, class ModelOrder>
class TimeIntegrator
<ExecutionSpace, MemorySpace, ZModel<ExecutionSpace, MemorySpace, ModelOrder>>
{
    using Node = Cajita::Cell;
    using exec_space = ExecutionSpace;
    using mem_space = MemorySpace;
    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      MemorySpace>;

    using halo_type = Cajita::Halo<MemorySpace>;

  public:
    TimeIntegrator( const std::shared_ptr<ProblemManager<exec_space, mem_space>> & pm,
                    ZModel<exec_space, mem_space, ModelOrder> & zm )
    : _pm(pm)
    , _zm(zm)
    {
       
        // Create a layout of the temporary arrays we'll need for velocity
        // intermediate positions, and change in vorticity
        auto node_triple_layout =
            Cajita::createArrayLayout( pm->mesh()->localGrid(), 3, Cajita::Node() );
        auto node_double_layout =
            Cajita::createArrayLayout( pm->mesh()->localGrid(), 2, Cajita::Node() );

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
    BoundaryCondition &_bc;
    ZModel<ExecutionSpace, MemorySpace, ModelOrder> & _zm;
    std::shared_ptr<ProblemManager<ExecutionSpace, MemorySpace>> & _pm;
    node_array _zdot, _vtmp, _wtmp, _ztmp;
};

} // end namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
