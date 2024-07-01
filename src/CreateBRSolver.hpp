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

#ifndef BEATNIK_CREATEBRSOLVER_HPP
#define BEATNIK_CREATEBRSOLVER_HPP

#include <BRSolverBase.hpp>
#include <ExactBRSolver.hpp>
#include <CutoffBRSolver.hpp>


namespace Beatnik
{

enum BRSolverType {BR_EXACT = 0, BR_CUTOFF};


/* createBRSolver is pulled into a seperate file to avoid circular 
 * dependencies between BRSolverBase and the BR solver options.
 */
template <class pm_type, class ExecutionSpace, class MemorySpace, class Params>
std::unique_ptr<BRSolverBase<ExecutionSpace, MemorySpace, Params>>
createBRSolver( const pm_type &pm, const BoundaryCondition &bc,
                const double epsilon, const double dx, const double dy,
                const Params params )
{
    if ( params.br_solver == BR_EXACT )
    {
        using br_type = Beatnik::ExactBRSolver<ExecutionSpace, MemorySpace, Params>;
        return std::make_unique<br_type>(pm, bc, epsilon, dx, dy, params);
    }
    if ( params.br_solver == BR_CUTOFF )
    {
        using br_type = Beatnik::CutoffBRSolver<ExecutionSpace, MemorySpace, Params>;
        return std::make_unique<br_type>(pm, bc, epsilon, dx, dy, params);
    }
    std::cerr << "Invalid BR solver type.\n";
    exit(-1);
}

} // end namespace Beatnik

#endif /* BEATNIK_CREATEBRSOLVER_HPP */
