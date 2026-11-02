/****************************************************************************
 * Copyright (c) 2025 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file Beatnik_CreateBRSolver.hpp
 * @brief Factory for the Birkhoff-Rott evaluator.
 *
 * The only place in the library that knows both concrete BR solvers exist. The
 * solver holds a `BRSolverBase` pointer and never names either subclass, so
 * adding a third approximation touches this file and nothing else.
 */

#ifndef BEATNIK_CREATEBRSOLVER_HPP
#define BEATNIK_CREATEBRSOLVER_HPP

#include <Beatnik_BRSolverBase.hpp>
#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_BRSolverFMM.hpp>
#include <Beatnik_Config.hpp>

#include <mpi.h>

#include <memory>
#include <stdexcept>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Construct the Birkhoff-Rott evaluator named by the parameters.
 *
 * @param comm       Communicator the surface is decomposed over.
 * @param params     Supplies `br_approximation`.
 * @param fmm_params Tunables, used only by the FMM path.
 * @return Owning pointer to the evaluator. Never null.
 *
 * @throws std::runtime_error if `fmm` is requested but Beatnik was built
 *         without Canopy. This is a configuration error, not a
 *         not-implemented stub, so it is a `runtime_error` — the distinction
 *         matters when reading a failure: a `logic_error` means "write this
 *         code", a `runtime_error` here means "rebuild with `+canopy`".
 */
template <class ExecutionSpace, class MemorySpace>
std::unique_ptr<BRSolverBase<ExecutionSpace, MemorySpace>>
createBRSolver( MPI_Comm comm, const ZModelParams& params,
                const FmmParams& fmm_params )
{
    (void)fmm_params; // unused on the Direct path and in a ~canopy build

    switch ( params.br_approximation )
    {
    case BRApproximation::Direct:
        return std::unique_ptr<BRSolverBase<ExecutionSpace, MemorySpace>>(
            new BRSolverDirect<ExecutionSpace, MemorySpace>( comm ) );

    case BRApproximation::Fmm:
#ifndef BEATNIK_ENABLE_CANOPY
        throw std::runtime_error(
            "--br-approximation fmm requires Beatnik to be built with Canopy "
            "support (Beatnik_ENABLE_CANOPY=ON, spack '+canopy'). Use "
            "--br-approximation direct instead." );
#else
        return std::unique_ptr<BRSolverBase<ExecutionSpace, MemorySpace>>(
            new BRSolverFMM<ExecutionSpace, MemorySpace>( comm, fmm_params ) );
#endif
    }

    throw std::runtime_error( "unrecognized br_approximation" );
}

} // namespace Beatnik

#endif // BEATNIK_CREATEBRSOLVER_HPP
