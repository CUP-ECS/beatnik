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

#include <cmath>
#include <type_traits>
#include <utility>

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
    using device_type = typename base_type::device_type;
    using scalar_view = typename base_type::scalar_view;
    using vector_view = typename base_type::vector_view;
    using mesh_type = typename base_type::mesh_type;
    using geometry_type = typename base_type::geometry_type;
    using state_type = typename base_type::state_type;
    using quadrature_type = typename base_type::quadrature_type;
    using point_view = typename quadrature_type::point_view;
    using strength_view = typename quadrature_type::strength_view;

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
    void computeInterfaceVelocity( mesh_type& mesh,
                                   const geometry_type& geometry,
                                   const state_type& state,
                                   const quadrature_type& quadrature,
                                   const ZModelParams& params,
                                   vector_view& velocity ) override
    {
        const int nt = mesh.ownedVertexCount();
        if ( static_cast<int>( velocity.extent( 0 ) ) != nt )
            Kokkos::realloc( velocity, nt );
        Kokkos::deep_copy( velocity, Real( 0 ) );

        point_view points;
        strength_view strengths;
        quadrature.generate( mesh, geometry, state, points, strengths );

        auto pos = mesh.positions();
        auto u = velocity;
        const Real blob = params.blob();
        // 1/4pi ONCE, here, and `br_sign` on the velocity only. See
        // Beatnik_BRSolverBase.hpp.
        const Real coefficient =
            params.br_sign / ( Real( 4 ) * static_cast<Real>( M_PI ) );

        ringAccumulate(
            points, strengths, static_cast<int>( points.extent( 0 ) ),
            [&]( const point_view& sp, const strength_view& ss, const int nb )
            {
                Kokkos::parallel_for(
                    "beatnik_br_direct_velocity",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, nt ),
                    KOKKOS_LAMBDA( const int t ) {
                        const Real x0 = pos( t, 0 );
                        const Real x1 = pos( t, 1 );
                        const Real x2 = pos( t, 2 );
                        Real acc[3] = { Real( 0 ), Real( 0 ), Real( 0 ) };
                        for ( int s = 0; s < nb; ++s )
                        {
                            // delta = x_t - y_s. The cross product order is
                            // delta x S; reversing it negates the whole field.
                            const Real d0 = x0 - sp( s, 0 );
                            const Real d1 = x1 - sp( s, 1 );
                            const Real d2 = x2 - sp( s, 2 );
                            const Real r2 = d0 * d0 + d1 * d1 + d2 * d2;
                            // b is added to r^2, not to r, and the power is
                            // 3/2 on the sum. `pow(.., 1.5)` rather than
                            // x*sqrt(x) so this is the reference expression.
                            // The self term has delta = 0 and contributes
                            // exactly zero -- no exclusion list.
                            const Real k =
                                Real( 1 ) /
                                Kokkos::pow( blob + r2, Real( 1.5 ) );
                            acc[0] += ( d1 * ss( s, 2 ) - d2 * ss( s, 1 ) ) * k;
                            acc[1] += ( d2 * ss( s, 0 ) - d0 * ss( s, 2 ) ) * k;
                            acc[2] += ( d0 * ss( s, 1 ) - d1 * ss( s, 0 ) ) * k;
                        }
                        for ( int d = 0; d < 3; ++d )
                            u( t, d ) += coefficient * acc[d];
                    } );
                Kokkos::fence();
            } );
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
    void computeSurfaceRieszScalar( mesh_type& mesh,
                                    const geometry_type& geometry,
                                    const state_type& state,
                                    const quadrature_type& quadrature,
                                    const ZModelParams& params,
                                    scalar_view& scalar ) override
    {
        const int nt = mesh.ownedVertexCount();
        if ( static_cast<int>( scalar.extent( 0 ) ) != nt )
            Kokkos::realloc( scalar, nt );
        Kokkos::deep_copy( scalar, Real( 0 ) );

        point_view points;
        strength_view gradients;
        quadrature.generateGradient( mesh, geometry, state, points, gradients );

        auto pos = mesh.positions();
        auto psi = scalar;
        const Real blob = params.blob();
        // -1/(4 pi^2). NOT multiplied by `br_sign` -- the reference returns
        // this one unsigned (`mesh_solver.py:581-587`), an asymmetry with the
        // velocity path that is reproduced deliberately.
        const Real coefficient =
            Real( -1 ) / ( Real( 4 ) * static_cast<Real>( M_PI ) *
                           static_cast<Real>( M_PI ) );

        ringAccumulate(
            points, gradients, static_cast<int>( points.extent( 0 ) ),
            [&]( const point_view& sp, const strength_view& ss, const int nb )
            {
                Kokkos::parallel_for(
                    "beatnik_br_direct_riesz",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, nt ),
                    KOKKOS_LAMBDA( const int t ) {
                        const Real x0 = pos( t, 0 );
                        const Real x1 = pos( t, 1 );
                        const Real x2 = pos( t, 2 );
                        Real acc = Real( 0 );
                        for ( int s = 0; s < nb; ++s )
                        {
                            const Real d0 = x0 - sp( s, 0 );
                            const Real d1 = x1 - sp( s, 1 );
                            const Real d2 = x2 - sp( s, 2 );
                            const Real r2 = d0 * d0 + d1 * d1 + d2 * d2;
                            const Real k =
                                Real( 1 ) /
                                Kokkos::pow( blob + r2, Real( 1.5 ) );
                            // Same kernel as the velocity; only the
                            // contraction is a dot instead of a cross.
                            acc += ( d0 * ss( s, 0 ) + d1 * ss( s, 1 ) +
                                     d2 * ss( s, 2 ) ) *
                                   k;
                        }
                        psi( t ) += coefficient * acc;
                    } );
                Kokkos::fence();
            } );
    }

  private:
    /**
     * @brief Circulate the local source block once around the rank ring,
     *        invoking `kernel` on every block as it passes.
     *
     * The structure the class doc names: `P` steps of `MPI_Sendrecv`, each rank
     * accumulating into its **own** targets as each block arrives, so the peak
     * storage is \f$O(N_s/P)\f$ rather than the \f$O(N_s)\f$ an
     * `MPI_Allgatherv` would need. `kernel( points, strengths, count )` is
     * called exactly once per rank's block, starting with this rank's own —
     * which is what includes the self-interaction, whose \f$\delta = 0\f$ term
     * contributes exactly zero.
     *
     * Factored rather than written twice because the velocity and the Riesz
     * scalar differ only in the contraction and the prefactor; two copies of a
     * collective loop are two places for a deadlock to be introduced
     * independently.
     *
     * @note MPI. Collective on `_comm`, and every rank executes exactly `P`
     *       kernel invocations and `P-1` `Sendrecv` pairs regardless of how
     *       many sources it owns — including zero. Block sizes differ across
     *       ranks, so the buffers are sized at the global maximum and the live
     *       count travels with the data.
     *
     * @note Buffers are device views handed straight to MPI. Cray MPICH is
     *       GPU-aware here (`MPICH_GPU_SUPPORT_ENABLED=1`, set once in
     *       `scripts/tuolumne/runtime_env.sh`); staging through the host would
     *       add a round trip per ring step for nothing.
     */
    template <class Kernel>
    void ringAccumulate( const point_view& points,
                         const strength_view& strengths, const int local_count,
                         const Kernel& kernel )
    {
        static_assert( std::is_same<Real, double>::value,
                       "the ring exchange below hardcodes MPI_DOUBLE" );

        int rank = 0;
        int size = 1;
        MPI_Comm_rank( _comm, &rank );
        MPI_Comm_size( _comm, &size );

        // This rank's own block first, so the self term is included.
        kernel( points, strengths, local_count );
        if ( size == 1 )
            return;

        int max_count = 0;
        MPI_Allreduce( &local_count, &max_count, 1, MPI_INT, MPI_MAX, _comm );

        point_view send_points( "beatnik_br_ring_send_points", max_count );
        strength_view send_strengths( "beatnik_br_ring_send_strengths",
                                      max_count );
        point_view recv_points( "beatnik_br_ring_recv_points", max_count );
        strength_view recv_strengths( "beatnik_br_ring_recv_strengths",
                                      max_count );
        if ( local_count > 0 )
        {
            const auto rows = std::make_pair( 0, local_count );
            Kokkos::deep_copy(
                Kokkos::subview( send_points, rows, Kokkos::ALL ),
                Kokkos::subview( points, rows, Kokkos::ALL ) );
            Kokkos::deep_copy(
                Kokkos::subview( send_strengths, rows, Kokkos::ALL ),
                Kokkos::subview( strengths, rows, Kokkos::ALL ) );
        }

        const int dst = ( rank + 1 ) % size;
        const int src = ( rank + size - 1 ) % size;
        const int words = 3 * max_count;
        int count = local_count;

        for ( int step = 1; step < size; ++step )
        {
            int received = 0;
            MPI_Sendrecv( &count, 1, MPI_INT, dst, 1400, &received, 1, MPI_INT,
                          src, 1400, _comm, MPI_STATUS_IGNORE );
            MPI_Sendrecv( send_points.data(), words, MPI_DOUBLE, dst, 1401,
                          recv_points.data(), words, MPI_DOUBLE, src, 1401,
                          _comm, MPI_STATUS_IGNORE );
            MPI_Sendrecv( send_strengths.data(), words, MPI_DOUBLE, dst, 1402,
                          recv_strengths.data(), words, MPI_DOUBLE, src, 1402,
                          _comm, MPI_STATUS_IGNORE );

            std::swap( send_points, recv_points );
            std::swap( send_strengths, recv_strengths );
            count = received;

            kernel( send_points, send_strengths, count );
        }
    }

    MPI_Comm _comm;
};

} // namespace Beatnik

#endif // BEATNIK_BRSOLVERDIRECT_HPP
