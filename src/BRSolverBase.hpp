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
};

} // end namespace Beantik

#endif // end BEATNIK_BRSOLVERBASE_HPP
