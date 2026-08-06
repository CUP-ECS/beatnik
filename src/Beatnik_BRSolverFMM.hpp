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
 * @file Beatnik_BRSolverFMM.hpp
 * @brief Fast-multipole Birkhoff-Rott evaluation, delegating to Canopy through
 *        `Beatnik_FarFieldInterface.hpp`.
 *
 * This class contains **no Canopy types**. Its entire job is to turn a
 * `(mesh, geometry, state, quadrature)` tuple into the point/strength arrays
 * `FarFieldSolver` wants, call it, and hand back the result. That split is what
 * lets the FMM task (`tasks/framework.md`, task F1) open `../canopy` and
 * rewrite only the adapter.
 *
 * RELATION TO THE PYTHON
 * ----------------------
 * The Python's fast path is a Barnes-Hut **treecode**
 * (`zmodel3d/treecode.py`), a different algorithm with a different error
 * structure: monopole-through-quadrupole expansions accepted by an opening
 * angle, with no local expansion and no downward pass. Beatnik does **not**
 * port it. Consequences worth stating plainly:
 *
 *   - There is **no line-for-line Python counterpart** for this file. The trace
 *     comments below point at the treecode only to identify the *role* being
 *     replaced, not an implementation to reproduce.
 *   - A Beatnik `--br-approximation fmm` run is therefore **not** expected to
 *     match a Python `--br-approximation treecode` run to tight tolerance. The
 *     testing ladder handles this by comparing regression test 3 (Beatnik FMM)
 *     against the Python **direct** gold file, not against the Python treecode
 *     — see `tasks/framework.md`, testing task T3.
 *   - `--br-treecode-theta/-order/-ncrit` are still accepted and mapped onto
 *     `FmmParams`, so a Python command line runs. The mapping is nominal; the
 *     numbers do not mean the same thing to the two algorithms.
 *
 * See `Beatnik_BRSolverBase.hpp` for the equation and the conventions.
 */

#ifndef BEATNIK_BRSOLVERFMM_HPP
#define BEATNIK_BRSOLVERFMM_HPP

#include <Beatnik_BRSolverBase.hpp>
#include <Beatnik_FarFieldInterface.hpp>

#include <mpi.h>

#include <memory>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Birkhoff-Rott evaluation by fast multipole.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class BRSolverFMM : public BRSolverBase<ExecutionSpace, MemorySpace>
{
  public:
    using base_type = BRSolverBase<ExecutionSpace, MemorySpace>;
    using scalar_view = typename base_type::scalar_view;
    using vector_view = typename base_type::vector_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;
    using quadrature_type = typename base_type::quadrature_type;

    using far_field_type = FarFieldSolver<ExecutionSpace, MemorySpace>;

    /**
     * @param comm       Communicator the surface is decomposed over.
     * @param fmm_params Tunables handed through to the far-field solver.
     */
    BRSolverFMM( MPI_Comm comm, const FmmParams& fmm_params )
        : _comm( comm )
        , _far_field( new far_field_type( comm, fmm_params ) )
    {
    }

    BRApproximation kind() const override { return BRApproximation::Fmm; }

    /**
     * @brief Induced velocity by fast multipole.
     *
     * Replaces the role of treecode.py::treecode_velocity_unsigned
     * (lines 96-...) — see the file header on why this is a replacement rather
     * than a port.
     *
     * Sequence: generate sources from the quadrature, hand them to
     * `FarFieldSolver::setSources`, then `evaluateCurl` at the owned vertices,
     * then apply `br_sign`. The \f$1/4\pi\f$ is applied inside `evaluateCurl`.
     *
     * @note The FMM's accuracy is controlled by the acceptance criterion, and
     *       the kernel it expands is the **softened** \f$1/r^2\f$ field, not
     *       the bare one. Near self-contact the sheet separation approaches
     *       \f$\sqrt{b}\f$, where the softening dominates the geometry; an
     *       acceptance criterion tuned on the bare kernel is optimistic there.
     *       Recorded as risk R6 in `tasks/framework.md`.
     *
     * @note MPI. Collective, inside Canopy. The tree rebuild also carries a
     *       bounding-box `MPI_Allreduce` and a source redistribution — see
     *       `FarFieldSolver::setSources`.
     */
    void computeInterfaceVelocity( const mesh_type& mesh,
                                   const geometry_type& geometry,
                                   const state_type& state,
                                   const quadrature_type& quadrature,
                                   const ZModelParams& params,
                                   vector_view& velocity ) override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)quadrature;
        (void)params;
        (void)velocity;
        BEATNIK_NOT_IMPLEMENTED( "BRSolverFMM", "computeInterfaceVelocity" );
    }

    /**
     * @brief Surface Riesz scalar by fast multipole.
     *
     * Same source generation, but through
     * `SourceQuadratureBase::generateGradient` and `FarFieldSolver::evaluateDot`
     * with the \f$-1/4\pi^2\f$ normalization.
     *
     * The Python explicitly refuses this combination
     * (`mesh_solver.py:605` raises for `treecode`), so there is no gold file
     * for it — see `BRSolverBase::computeSurfaceRieszScalar`.
     */
    void computeSurfaceRieszScalar( const mesh_type& mesh,
                                    const geometry_type& geometry,
                                    const state_type& state,
                                    const quadrature_type& quadrature,
                                    const ZModelParams& params,
                                    scalar_view& scalar ) override
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)quadrature;
        (void)params;
        (void)scalar;
        BEATNIK_NOT_IMPLEMENTED( "BRSolverFMM", "computeSurfaceRieszScalar" );
    }

  private:
    MPI_Comm _comm;
    std::unique_ptr<far_field_type> _far_field;
};

} // namespace Beatnik

#endif // BEATNIK_BRSOLVERFMM_HPP
