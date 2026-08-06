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

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>
#include <Beatnik_ZModelSolver.hpp>

#include <Kokkos_Core.hpp>

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
     */
    void step( mesh_type& mesh, state_type& state, Real dt )
    {
        (void)mesh;
        (void)state;
        (void)dt;
        BEATNIK_NOT_IMPLEMENTED( "TimeIntegrator", "step" );
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
     *       obtain the same dt — see `Comm::allReduceMin`.
     */
    Real chooseStepSize( const mesh_type& mesh, const state_type& state,
                         const TimeParams& time_params,
                         Real initial_min_edge ) const
    {
        (void)mesh;
        (void)state;
        (void)time_params;
        (void)initial_min_edge;
        BEATNIK_NOT_IMPLEMENTED( "TimeIntegrator", "chooseStepSize" );
    }

  private:
    zmodel_type* _zmodel = nullptr;
};

} // namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
