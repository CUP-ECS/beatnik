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

namespace Beatnik
{

/* Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
template <class ExecutionSpace, class MemorySpace, class Params>
class BRSolverBase
{
  public:
    using node_view = Kokkos::View<double***, MemorySpace>;
    virtual ~BRSolverBase() = default;
    virtual void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const = 0;

    /* Profiling hooks. Solver calls beginBeatnikStep() before each
     * outer step, then flushProfile() after printing its own
     * [Beatnik profile] line so any per-substep records can be
     * emitted indented underneath. The base records the current step
     * so any caller (e.g. ZModel's NaN/Inf guard) can report which
     * timestep a blowup occurred on, for any BR backend. Overrides
     * should set _beatnik_step (or call this base method). */
    virtual void beginBeatnikStep( int step ) const { _beatnik_step = step; }
    virtual void flushProfile() const {}

    /* Current outer (Beatnik) timestep, as last set by beginBeatnikStep().
     * Only meaningful when the Solver drives beginBeatnikStep(), which it
     * does under profiling. */
    int beatnikStep() const { return _beatnik_step; }

  protected:
    mutable int _beatnik_step{ 0 };
};

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
