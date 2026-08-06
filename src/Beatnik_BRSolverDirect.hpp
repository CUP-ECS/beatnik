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
 * @file Beatnik_BRSolverDirect.hpp
 * @brief The O(N^2) reference Birkhoff-Rott sum. Slow, and definitive.
 *
 * This is not an approximation of the Birkhoff-Rott integral — after the
 * quadrature has been chosen, it *is* the discretization. Everything faster is
 * validated against it. The testing ladder therefore runs regression tests 1
 * and 2 with this solver and only introduces the FMM at test 3, so that a
 * failure at test 2 is unambiguously a bug in the surrounding mathematics
 * rather than in the far field.
 *
 * See `Beatnik_BRSolverBase.hpp` for the equation, the sign conventions and
 * the normalization; none of them are restated here, so there is one place to
 * change them.
 */

#ifndef BEATNIK_BRSOLVERDIRECT_HPP
#define BEATNIK_BRSOLVERDIRECT_HPP

#include <Beatnik_BRSolverBase.hpp>

#include <mpi.h>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Direct pairwise Birkhoff-Rott evaluation.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class BRSolverDirect : public BRSolverBase<ExecutionSpace, MemorySpace>
{
  public:
    using base_type = BRSolverBase<ExecutionSpace, MemorySpace>;
    using scalar_view = typename base_type::scalar_view;
    using vector_view = typename base_type::vector_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;
    using quadrature_type = typename base_type::quadrature_type;

    /// @param comm Communicator the surface is decomposed over.
    explicit BRSolverDirect( MPI_Comm comm )
        : _comm( comm )
    {
    }

    BRApproximation kind() const override { return BRApproximation::Direct; }

    /**
     * @brief Direct \f$O(N_t N_s)\f$ evaluation of the induced velocity.
     *
     * Port of mesh_solver.py::_source_velocity_direct_unsigned
     * (lines 437-454), with the `br_sign` multiplication from
     * ::_mesh_birkhoff_rott_velocity_from_sources (line 397)
     *
     * \f[
     *   u_t = \frac{\sigma_{BR}}{4\pi}\sum_{s}
     *     \frac{\delta_{ts}\times S_s}{(b+|\delta_{ts}|^2)^{3/2}},
     *   \qquad \delta_{ts} = x_t - y_s .
     * \f]
     *
     * The Python chunks the target loop to bound the temporary
     * \f$N_\text{chunk}\times N_s\times 3\f$ array at roughly 2M elements
     * (line 445). That is a NumPy memory concern with no analogue here: the
     * Kokkos form is a `TeamPolicy` over targets with a nested reduction over
     * sources and allocates no temporary at all. The chunking therefore is
     * **not** ported, and its absence changes nothing but the summation order —
     * which is already not reproducible across devices.
     *
     * Cost: \f$O(N_t N_s)\f$ per call, three calls per RK3 step, so nine
     * kernel evaluations of the full pairwise sum per accepted timestep. At the
     * default 162-vertex icosphere that is trivial; by the time adaptivity has
     * driven the surface to \f$10^5\f$ vertices it is the entire runtime, which
     * is the whole reason `BRSolverFMM` exists.
     *
     * @note MPI. Every target needs every source. The reference parallel
     *       implementation circulates the source block around a ring of ranks
     *       (P steps of `MPI_Sendrecv`, accumulating into the target's velocity
     *       at each), which needs \f$O(N_s/P)\f$ storage rather than the
     *       \f$O(N_s)\f$ of an `MPI_Allgatherv`. Either is correct; the ring is
     *       what scales.
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
        BEATNIK_NOT_IMPLEMENTED( "BRSolverDirect", "computeInterfaceVelocity" );
    }

    /**
     * @brief Direct evaluation of the surface Riesz scalar.
     *
     * Port of mesh_solver.py::_source_riesz_scalar_direct (lines 457-489)
     *
     * \f[
     *   \Psi_t = -\frac{1}{4\pi^2}\sum_s
     *     \frac{\delta_{ts}\cdot G_s}{(b+|\delta_{ts}|^2)^{3/2}} .
     * \f]
     *
     * Same kernel and same ring-exchange structure as the velocity; only the
     * contraction (dot instead of cross) and the prefactor differ. Note this
     * one is **not** multiplied by `br_sign` in the reference
     * (`mesh_solver.py:581-587` returns it unsigned) — an asymmetry with the
     * velocity path that is faithfully reproduced here.
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
        BEATNIK_NOT_IMPLEMENTED( "BRSolverDirect",
                                 "computeSurfaceRieszScalar" );
    }

  private:
    MPI_Comm _comm;
};

} // namespace Beatnik

#endif // BEATNIK_BRSOLVERDIRECT_HPP
