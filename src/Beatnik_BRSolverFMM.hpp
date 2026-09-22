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
    using point_view = typename quadrature_type::point_view;
    using strength_view = typename quadrature_type::strength_view;

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
     * Sequence: generate sources from the quadrature, then
     * `FarFieldSolver::evaluateVelocity` with them. The \f$1/4\pi\f$ and
     * `br_sign` are both applied inside that call, and the tree maintenance
     * the evaluation needs is decided inside it too, so nothing here knows
     * about Canopy's lifecycle.
     *
     * @note **Which kernel the far field expands depends on the basis**, so
     *       the claim cannot be made without naming one. Under
     *       `FarFieldBasis::CartesianTaylor` the expanded kernel is the
     *       **softened** \f$1/r^2\f$ field: the blob enters as
     *       \f$w_b = b + r^2\f$ inside the expansion at every order, so the
     *       far field and the near field run the same regularization. Under
     *       `FarFieldBasis::SolidHarmonic` — which is what Canopy's basis
     *       template parameter defaults to, and therefore what an
     *       instantiation that omits it silently gets — the expanded kernel is
     *       the **bare** one, and the softening survives only in the near
     *       field. `FmmParams::basis` is what decides which of the two is in
     *       force.
     *
     *       Because the softened basis carries the blob at every order, the
     *       old reading of this note — that the acceptance criterion is tuned
     *       on the bare kernel and so is optimistic where the sheet separation
     *       approaches \f$\sqrt{b}\f$ — does not describe the error under it:
     *       self-contact is not a special case for a kernel that is
     *       regularized inside the expansion. What binds instead is **Taylor
     *       truncation at the accepted separation ratio**. Canopy accepts a
     *       pair only beyond \f$R/w > 2\sqrt3/\theta\f$, with \f$R\f$ the
     *       separation and \f$w\f$ the source cell's half-width, and an
     *       order-\f$p\f$ truncation leaves a relative error
     *       \f$\sim(cw/R)^{p}\f$ on the **gradient** — one order worse than on
     *       the potential, which is why `FmmParams::order` defaults to 3 and
     *       not 2. `FmmParams::order` is the knob, not `mac_theta`. See
     *       `tasks/canopy/add-canopy.md` ("The order that reaches the target").
     *       No accuracy figure is claimed for this path until T5 measures one.
     *
     * @note MPI. Collective, inside Canopy, and every rank must call it the
     *       same number of times per step. Beyond Canopy's own collectives the
     *       adapter runs the round trip that maps Beatnik's decomposition onto
     *       Canopy's — see `FarFieldSolver::evaluateVelocity`.
     */
    void computeInterfaceVelocity( mesh_type& mesh,
                                   const geometry_type& geometry,
                                   const state_type& state,
                                   const quadrature_type& quadrature,
                                   const ZModelParams& params,
                                   vector_view& velocity ) override
    {
        // Owned rows only, in owned order: `generate` emits over
        // [0, ownedVertexCount()) and reallocates both views to that count, so
        // default-constructed views are what it wants (risk R9 on
        // `Beatnik_SourceQuadrature.hpp` -- a ghost emitted here is an owned
        // source on another rank and would be double-counted in the global
        // sum). Under `SourceQuadrature::Vertex` the targets ARE the sources,
        // which is why no target array is built: one index serves source,
        // target and output row alike.
        point_view points;
        strength_view strengths;
        quadrature.generate( mesh, geometry, state, points, strengths );

        // Everything else belongs to the adapter, deliberately and exactly
        // once each: it sizes `velocity` to the source count and zeroes it,
        // applies `br_sign/4pi`, resolves the softening from `params.blob()`,
        // drives Canopy's tree maintenance, and rejects a non-`Vertex`
        // quadrature. Duplicating any of those here is a silent double
        // application, not a redundant safety net -- the output is
        // OVERWRITTEN, not accumulated, per `BRSolverBase.hpp:137-139`.
        // `BRSolverDirect`'s realloc/zero/coefficient lines
        // (`Beatnik_BRSolverDirect.hpp:111-115`, `:125-127`) are what
        // `FarFieldSolver::evaluateVelocity` mirrors, not what this
        // reimplements.
        _far_field->evaluateVelocity( points, strengths, params, velocity );
    }

    /**
     * @brief Surface Riesz scalar by fast multipole.
     *
     * Same source generation, but through
     * `SourceQuadratureBase::generateGradient` and
     * `FarFieldSolver::evaluateRieszScalar` with the \f$-1/4\pi^2\f$
     * normalization, which that call applies and `br_sign`, which it does not.
     *
     * The Python explicitly refuses this combination
     * (`mesh_solver.py:605` raises for `treecode`), so there is no gold file
     * for it — see `BRSolverBase::computeSurfaceRieszScalar`.
     */
    void computeSurfaceRieszScalar( mesh_type& mesh,
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

    /**
     * @brief The far-field adapter, for its diagnostics.
     *
     * `_far_field` is private and `BRSolverBase`'s two virtuals return `void`,
     * so without this a test holding a `BRSolverFMM` could not read what
     * Canopy actually did — the maintenance action, the global particle count,
     * the P2P pair fraction, the M2L operator counts, or the basis, order and
     * softening in force — without naming a Canopy type itself, which the
     * adapter contract forbids. `FarFieldDiagnostics` names none, so this
     * hands back a Beatnik POD.
     *
     * Valid to call before any evaluation: the diagnostics are
     * default-constructed until one has run, and in a `~canopy` build they
     * stay that way because both evaluations throw.
     */
    const far_field_type& farField() const { return *_far_field; }

  private:
    MPI_Comm _comm;
    std::unique_ptr<far_field_type> _far_field;
};

} // namespace Beatnik

#endif // BEATNIK_BRSOLVERFMM_HPP
