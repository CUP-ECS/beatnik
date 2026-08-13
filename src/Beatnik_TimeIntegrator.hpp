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
 * @file Beatnik_TimeIntegrator.hpp
 * @brief TVD-RK3 time integration and the adaptive timestep control.
 */

#ifndef BEATNIK_TIMEINTEGRATOR_HPP
#define BEATNIK_TIMEINTEGRATOR_HPP

#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>
#include <Beatnik_ZModelSolver.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Three-stage TVD (SSP) Runge-Kutta integrator for the z-model.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class TimeIntegrator
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;
    using zmodel_type = ZModelSolver<ExecutionSpace, MemorySpace>;

    /// @param zmodel Right-hand side evaluator. Borrowed, not owned.
    explicit TimeIntegrator( zmodel_type& zmodel )
        : _zmodel( &zmodel )
    {
    }

    /**
     * @brief Advance the mesh and state one TVD-RK3 step.
     *
     * Port of mesh_solver.py::potential_mesh_rk3_step (lines 1291-1311) and
     * ::mesh_rk3_step (lines 1272-1288); structured-grid ancestor
     * solver.py::rk3_step (lines 220-238)
     *
     * The Shu-Osher SSP(3,3) form. Writing \f$q\f$ for the pair
     * \f$(x, \phi)\f$ (or \f$(x, S)\f$) and \f$L(q)\f$ for the right-hand side:
     * \f[
     *   \begin{aligned}
     *     q^{(1)} &= q^n + \Delta t\, L(q^n) \\
     *     q^{(2)} &= \tfrac34 q^n + \tfrac14 q^{(1)}
     *                + \tfrac14 \Delta t\, L(q^{(1)}) \\
     *     q^{n+1} &= \tfrac13 q^n + \tfrac23 q^{(2)}
     *                + \tfrac23 \Delta t\, L(q^{(2)})
     *   \end{aligned}
     * \f]
     * Third-order accurate, strong-stability-preserving with SSP coefficient 1,
     * so the stable step is the same as forward Euler's.
     *
     * Both components of \f$q\f$ take the **same** convex combination — the
     * position and the potential are not stepped with different weights. Note
     * the second stage combines \f$q^n\f$ with \f$q^{(1)}\f$, not with
     * \f$q^{(2)}\f$'s own predictor.
     *
     * **Geometry is re-evaluated at each stage.** \f$L\f$ depends on the
     * surface through areas, normals and the BR kernel, so a stage evaluated
     * at stale geometry silently degrades to first order. This is the most
     * likely way to get a "works but converges wrong" bug here.
     *
     * **What RK3 does not preserve.** The stage combinations are taken in the
     * ambient \f$\mathbb{R}^3\f$, so the intermediate states are not
     * constrained to satisfy the tangency of \f$S\f$ (re-projected after each
     * construction) or the zero-mean gauge of \f$\phi\f$ (re-centred after each
     * construction). Both restorations happen inside the state construction,
     * not here.
     *
     * @param mesh   Surface, advanced in place.
     * @param state  Solution, advanced in place.
     * @param dt     Step size, already chosen by `chooseStepSize` and clamped.
     *
     * @note MPI. Three full right-hand-side evaluations, hence three BR
     *       collectives and three sets of reductions. Also three ghost
     *       exchanges of the vertex positions, one after each stage update —
     *       `Comm::haloExchangeVertices`.
     *
     * **T2d — WHAT THE STAGE CONSTRUCTION HAS TO DO BESIDES THE ARITHMETIC.**
     * The reference builds each stage through `state.with_arrays(...)`, which
     * runs `MeshPotentialZModelState.__post_init__` and therefore
     * **re-centres the potential at the NEW vertices** (`mesh_solver.py:155-159`)
     * — every stage, not only at the end of the step. So `finishStage` below is
     * `haloExchange` -> geometry at the new positions -> `centerPotential`, and
     * the next stage's combination reads the *centred* value, exactly as the
     * Python's `stage1.potential` is the centred array. Dropping the
     * per-stage re-centring changes the answer, because the mean is subtracted
     * from a field the next stage then differentiates.
     *
     * The stage geometry is computed twice per stage — once by `finishStage` for
     * the centring and once inside the RHS — which at 162 vertices is free and
     * keeps the RHS's documented signature (it takes no geometry, so that it
     * cannot be handed a stale one).
     */
    void step( mesh_type& mesh, state_type& state, Real dt )
    {
        const int n = mesh.ownedVertexCount();
        resizeVector( _x0, "beatnik_rk3_x0", n );
        resizeScalar( _p0, "beatnik_rk3_p0", n );

        {
            auto pos = mesh.positions();
            auto phi = mesh.potential();
            auto x0 = _x0;
            auto p0 = _p0;
            Kokkos::parallel_for(
                "beatnik_rk3_save",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                        x0( i, d ) = pos( i, d );
                    p0( i ) = phi( i );
                } );
            Kokkos::fence();
        }

        // Stage 1: q1 = q0 + dt L(q0). Written as the general combination with
        // (a0, a1) = (0, 1) so the three stages are one kernel and cannot drift.
        _zmodel->computeRightHandSidePotential( mesh, state, _vertex_dot,
                                                _potential_dot );
        combine( mesh, n, Real( 0 ), Real( 1 ), dt );
        finishStage( mesh, state );

        // Stage 2: q2 = 3/4 q0 + 1/4 q1 + 1/4 dt L(q1). Note it combines with
        // q0, not with q2's own predictor.
        _zmodel->computeRightHandSidePotential( mesh, state, _vertex_dot,
                                                _potential_dot );
        combine( mesh, n, Real( 0.75 ), Real( 0.25 ), Real( 0.25 ) * dt );
        finishStage( mesh, state );

        // Stage 3: q^{n+1} = 1/3 q0 + 2/3 q2 + 2/3 dt L(q2).
        _zmodel->computeRightHandSidePotential( mesh, state, _vertex_dot,
                                                _potential_dot );
        const Real third = Real( 1 ) / Real( 3 );
        const Real two_thirds = Real( 2 ) / Real( 3 );
        combine( mesh, n, third, two_thirds, two_thirds * dt );
        finishStage( mesh, state );
    }

    /**
     * @brief Choose the step size for the next step.
     *
     * Port of run_adaptive_mesh_bubble.py::choose_step_dt (lines 889-901),
     * plus the two clamps applied by the caller at lines 1407-1410
     *
     * With `--adaptive-dt` (the default):
     * \f[
     *   \rho = \min\!\Big(1, \frac{h_{\min}}{h_{\min}^0}\Big), \qquad
     *   \Delta t = \max\!\big(\Delta t_{\min},\; \Delta t_0\, \rho^{\,p}\big)
     * \f]
     * with \f$p\f$ = `--dt-edge-power` (default 1) and \f$h_{\min}^0\f$ the
     * *initial* minimum edge length, which is why that quantity is carried in
     * the checkpoint. The ratio is capped at 1, so a *coarsening* mesh never
     * increases dt above \f$\Delta t_0\f$. Without `--adaptive-dt`,
     * \f$\Delta t = \Delta t_0\f$ flat.
     *
     * Then, if `--max-sheet-dt-product` \f$= c > 0\f$ and
     * \f$\|S\|_\infty\f$ is finite and positive,
     * \f$\Delta t \leftarrow \min\big(\Delta t,\ \max(\Delta t_{\min},\,
     * c/\|S\|_\infty)\big)\f$ — a CFL-like condition on the circulation rather
     * than on the geometry. Note the floor is applied *inside* the min, so this
     * clamp cannot push dt below `--min-dt` either.
     *
     * Two further clamps are the caller's, not this function's:
     *   - past `--dt-switch-time`, \f$\Delta t \leftarrow \min(\Delta t,
     *     \text{`--dt-after-switch'})\f$;
     *   - with `--t-end` set, \f$\Delta t \leftarrow \min(\Delta t,
     *     t_{\text{end}} - t)\f$ so the run lands exactly on the end time.
     *
     * @param mesh             Current surface.
     * @param state            Current solution, for \f$\|S\|_\infty\f$.
     * @param time_params      dt controls.
     * @param initial_min_edge \f$h_{\min}^0\f$, from setup or the checkpoint.
     * @return The step size, in time units.
     *
     * @note MPI. \f$h_{\min}\f$ is an `MPI_Allreduce`/`MPI_MIN` and
     *       \f$\|S\|_\infty\f$ an `MPI_Allreduce`/`MPI_MAX`. Every rank **must**
     *       obtain the same dt — see `Comm::allReduceMin`. `MPI_MIN` on a fixed
     *       set of values is order-independent, so \f$h_{\min}\f$ is
     *       reproducible across rank counts *by construction* and not by luck
     *       (measured at T1c: spread exactly zero). That is what keeps the
     *       adaptive dt from being the leading term in R2's trajectory
     *       divergence.
     *
     * **T2d — `mesh` is no longer `const`** (`positions()` / `edgeVertices()`),
     * and **this must not be stubbed to a constant.** The T2a gold set was
     * generated with `--adaptive-dt` live: `time` is `0.003` exactly at step 1
     * and then drifts (`0.0059999881751648708` at step 2), so a fixed-dt run
     * fails on `time` at step 2 and on the fields shortly after, for a reason
     * that has nothing to do with the RHS.
     */
    Real chooseStepSize( mesh_type& mesh, const state_type& state,
                         const TimeParams& time_params,
                         Real initial_min_edge ) const
    {
        Real dt = time_params.dt;

        if ( time_params.adaptive_dt )
        {
            // OWNED edges (risk R9). Owned edges form a global partition, so the
            // reduced minimum is the global minimum exactly once -- and this is
            // the same quantity, computed the same way, as the `initial_min_edge`
            // it is divided by.
            const int n_owned_edges = mesh.ownedEdgeCount();
            auto owned_edges =
                Kokkos::subview( mesh.edgeVertices(),
                                 std::make_pair( 0, n_owned_edges ),
                                 Kokkos::ALL() );
            scalar_view lengths(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_dt_edge_lengths" ),
                n_owned_edges );
            SurfaceOperators::edgeLengths( mesh.positions(), owned_edges,
                                           lengths );

            Real local_min = std::numeric_limits<Real>::max();
            Kokkos::parallel_reduce(
                "beatnik_dt_min_edge",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_edges ),
                KOKKOS_LAMBDA( const int e, Real& m ) {
                    if ( lengths( e ) < m )
                        m = lengths( e );
                },
                Kokkos::Min<Real>( local_min ) );

            const Real h_min = Comm::allReduceMin( mesh.comm(), local_min );
            const Real reference = ( initial_min_edge > Real( 1.0e-300 ) )
                                       ? initial_min_edge
                                       : Real( 1.0e-300 );
            // Capped at 1, so a COARSENING mesh never raises dt above dt0.
            Real ratio = h_min / reference;
            if ( ratio > Real( 1 ) )
                ratio = Real( 1 );
            const Real power = ( time_params.dt_edge_power > Real( 0 ) )
                                   ? time_params.dt_edge_power
                                   : Real( 0 );
            const Real scale = std::pow( ratio, power );
            dt = std::max( time_params.min_dt, time_params.dt * scale );
        }

        if ( time_params.max_sheet_dt_product > Real( 0 ) )
        {
            // The reference reads `state.sheet_vector`, which under the potential
            // model is a PROPERTY that recomputes the surface gradient -- so the
            // sheet vector is refreshed here rather than reused from the previous
            // step's last RK3 stage, where it belongs to the pre-final positions.
            // Costs one extra exchange and one geometry, and only on this branch.
            mesh.haloExchange();
            MeshGeometry<ExecutionSpace, MemorySpace> geometry;
            geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                              mesh.faceVertices() );
            state.updateSheetVector( mesh, geometry );

            const Real max_sheet = state.maxSheetStrength( mesh, geometry );
            if ( std::isfinite( max_sheet ) && max_sheet > Real( 1.0e-300 ) )
            {
                // The floor is INSIDE the min, so this clamp cannot push dt
                // below --min-dt either.
                const Real clamp =
                    std::max( time_params.min_dt,
                              time_params.max_sheet_dt_product / max_sheet );
                dt = std::min( dt, clamp );
            }
        }

        return dt;
    }

  private:
    /// One convex-combination kernel for all three stages:
    /// \f$q \leftarrow a_0 q^0 + a_1 q + c\,\dot q\f$, over OWNED rows.
    /// Both components take the SAME weights — the position and the potential
    /// are not stepped differently.
    void combine( mesh_type& mesh, int n, Real a0, Real a1, Real c )
    {
        auto pos = mesh.positions();
        auto phi = mesh.potential();
        auto x0 = _x0;
        auto p0 = _p0;
        auto xdot = _vertex_dot;
        auto pdot = _potential_dot;
        Kokkos::parallel_for(
            "beatnik_rk3_combine",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                    pos( i, d ) = a0 * x0( i, d ) + a1 * pos( i, d ) +
                                  c * xdot( i, d );
                phi( i ) = a0 * p0( i ) + a1 * phi( i ) + c * pdot( i );
            } );
        Kokkos::fence();
    }

    /// What `state.with_arrays(...)` does beyond storing the arrays: refresh the
    /// ghosts of the fields just written on owned rows, rebuild the geometry at
    /// the new positions, and re-centre the potential against it. See the
    /// T2d note on `step`.
    void finishStage( mesh_type& mesh, state_type& state )
    {
        mesh.haloExchange();
        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );
        state.centerPotential( mesh, geometry );
    }

    /// The label is a `std::string`, not a `const char*` — see the same helper
    /// in `Beatnik_ZModelSolver.hpp` for why a decayed pointer is read by
    /// `Kokkos::view_alloc` as pointer-to-memory rather than as a label.
    static void resizeScalar( scalar_view& v, const std::string& label, int n )
    {
        if ( static_cast<int>( v.extent( 0 ) ) != n )
            v = scalar_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing, label ), n );
    }

    static void resizeVector( vector_view& v, const std::string& label, int n )
    {
        if ( static_cast<int>( v.extent( 0 ) ) != n )
            v = vector_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing, label ), n );
    }

    zmodel_type* _zmodel = nullptr;

    /// The step's initial state and the current stage's rates. Allocated on
    /// first use and reused; the RHS reallocates its two out-parameters only on
    /// an extent mismatch (T2c), so one pair serves all three stages.
    vector_view _x0;
    scalar_view _p0;
    vector_view _vertex_dot;
    scalar_view _potential_dot;
};

} // namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
