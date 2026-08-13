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
 * @file Beatnik_ZModelSolver.hpp
 * @brief The z-model right-hand side: interface velocity and circulation
 *        forcing.
 *
 * THE EVOLUTION EQUATIONS
 * -----------------------
 * The higher-order 3D z-model evolves an interface \f$\Sigma(t)\f$ carrying a
 * vortex sheet. Two coupled equations, in the potential formulation:
 *
 * **Kinematic (the interface moves with the induced velocity):**
 * \f[
 *   \dot x \;=\; \Pi_{\text{vol}}\Big[\, \mathcal{V}\big[u_{BR}\big]
 *                \;+\; \sigma\,\Delta_{LB}x \,\Big]
 * \f]
 * where \f$u_{BR}\f$ is the Birkhoff-Rott velocity, \f$\mathcal{V}\f$ selects
 * either the full vector or its normal part (`--velocity-mode`), the
 * \f$\sigma\f$ term is optional surface tension, and \f$\Pi_{\text{vol}}\f$ is
 * the discrete mean-normal-flux projection that removes the net rate of volume
 * change (`--no-preserve-volume` disables it).
 *
 * **Dynamic (Bernoulli forcing of the circulation):**
 * \f[
 *   \dot\phi \;=\; \sigma_f A \,\underbrace{\Big[(u\!\cdot\!\hat n)^2
 *     \;-\; \tfrac14 |S|^2 \;-\; 2 g z_3\Big]}_{\textstyle V}
 *     \;+\; \mu\,\Delta_s\phi \;-\; \overline{(\cdot)}
 * \f]
 * with \f$A\f$ the Atwood number, \f$\sigma_f\f$ = `--forcing-sign`,
 * \f$g\f$ gravity, \f$z_3\f$ the vertical coordinate, \f$S\f$ the sheet vector,
 * \f$\mu\f$ the artificial viscosity, and \f$\overline{(\cdot)}\f$ the
 * area-weighted mean subtracted to pin the gauge.
 *
 * The Bernoulli potential \f$V = (u\cdot\hat n)^2 - \tfrac14|S|^2 - 2gz_3\f$ is
 * the validated baseline, identical to the structured-grid form at
 * `solver.py:199`. Every coefficient in it is load-bearing:
 *
 * - **\f$(u\cdot\hat n)^2\f$**, not \f$|u|^2\f$ and not \f$\tfrac12(u\cdot
 *   \hat n)^2\f$. `--bernoulli-scalar-mode normal-proxy` substitutes
 *   \f$\tfrac12 u\!\cdot\!\hat n\f$ *before* squaring, i.e. it changes the
 *   scalar being squared, not the exponent.
 * - **\f$-\tfrac14\f$** on \f$|S|^2\f$. This is the tangential kinetic-energy
 *   term; the factor of a quarter comes from the sheet carrying half the
 *   velocity jump on each side.
 * - **\f$-2g z_3\f$**, factor of two, sign negative with \f$z_3\f$ measured
 *   upward. Gravity acts in \f$-\hat z\f$.
 *
 * In the **sheet-vector** formulation the same \f$V\f$ appears but the
 * dynamic equation is the surface curl of it rather than \f$V\f$ itself:
 * \f[
 *   \dot S \;=\; -\sigma_f A\,\big(\hat n \times \nabla_s V\big)
 *              \;+\; \mu\,\mathcal{L}_{\text{graph}} S,
 * \f]
 * re-projected onto the tangent plane afterwards. Note the extra minus sign and
 * the different viscous operator — both are in the reference
 * (`mesh_solver.py:1229-1232`) and neither is a slip.
 */

#ifndef BEATNIK_ZMODELSOLVER_HPP
#define BEATNIK_ZMODELSOLVER_HPP

#include <Beatnik_BRSolverBase.hpp>
#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>
#include <Beatnik_VolumeProjection.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <string>
#include <utility>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Right-hand side of the z-model evolution equations.
 *
 * Holds no state of its own beyond scratch space and the parameters; one
 * instance serves all three RK3 stages.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class ZModelSolver
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
    using geometry_type = MeshGeometry<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;
    using quadrature_type = SourceQuadratureBase<ExecutionSpace, MemorySpace>;
    using br_solver_type = BRSolverBase<ExecutionSpace, MemorySpace>;

    /**
     * @param params     Physics and BR parameters.
     * @param br_solver  Birkhoff-Rott evaluator. Borrowed, not owned; the
     *                   caller keeps it alive for the run.
     * @param quadrature Source quadrature rule. Borrowed.
     */
    ZModelSolver( const ZModelParams& params, br_solver_type& br_solver,
                  const quadrature_type& quadrature )
        : _params( params )
        , _br_solver( &br_solver )
        , _quadrature( &quadrature )
    {
    }

    /// Physics parameters in force.
    const ZModelParams& params() const { return _params; }

    /**
     * @brief Evaluate \f$(\dot x, \dot\phi)\f$ for the potential formulation.
     *
     * Port of mesh_solver.py::potential_mesh_rhs (lines 1236-1269)
     *
     * Order of operations, which matters:
     *  1. Refresh geometry (areas, normals) at the current positions.
     *  2. Rebuild the sheet vector \f$S = -\hat n\times\nabla_s\phi\f$.
     *  3. Birkhoff-Rott velocity \f$u\f$ at the vertices.
     *  4. \f$u\!\cdot\!\hat n\f$, then the marker velocity per
     *     `--velocity-mode`.
     *  5. Add surface tension \f$\sigma\Delta_{LB}x\f$ if \f$\sigma\ne0\f$ —
     *     **before** the volume projection, so the projection compensates for
     *     the volume the pinching neck displaces.
     *  6. Volume projection, if `preserve_volume`.
     *  7. Bernoulli scalar per `--bernoulli-scalar-mode`, then
     *     \f$V = (\text{scalar})^2 - \tfrac14|S|^2 - 2gz_3\f$.
     *  8. \f$\dot\phi = \sigma_f A V + \mu\,\Delta_s\phi\f$.
     *  9. Subtract the area-weighted mean of \f$\dot\phi\f$.
     *
     * Step 5 before step 6 is deliberate and commented as such in the
     * reference (`mesh_solver.py:1134-1138`): the projection restores the
     * displaced volume as a uniform, shape-preserving inflation of the whole
     * body rather than as a local cancellation of the pinch, which is what
     * projecting first would produce.
     *
     * Step 9 is what keeps \f$\phi\f$ from drifting; see
     * `SurfaceOperators::areaWeightedMean`.
     *
     * @param mesh     Surface at the current stage geometry. **T2d — no longer
     *                 `const`**: every accessor this needs (`positions()`,
     *                 `potential()`, `sheetVector()`, `faceVertices()`) is
     *                 non-const, and the BR solver's own parameter was widened
     *                 at T2c for the same reason.
     * @param state    Current \f$\phi\f$ (and the derived sheet vector).
     * @param[out] vertex_dot `(N_owned,3)` \f$\dot x\f$, units velocity.
     *                 Reallocated on an extent mismatch, so one view serves all
     *                 three RK3 stages.
     * @param[out] potential_dot `(N_owned,)` \f$\dot\phi\f$, units velocity^2.
     *
     * **T2d — the output range is the OWNED vertices, not the local range.**
     * `BRSolverDirect::computeInterfaceVelocity` writes `(N_owned, 3)` (T2c),
     * every reduction below is over owned rows (risk R9), and the integrator
     * updates owned rows and then exchanges. Intermediate *assemblies* are still
     * allocated over the whole local range — the two opposite conventions stated
     * under DISTRIBUTED ASSEMBLY in `Beatnik_MeshGeometry.hpp`.
     *
     * @note MPI. Collective at four points: the BR evaluation, the two
     *       reductions of the volume projection, and the area-weighted mean of
     *       \f$\dot\phi\f$. It also opens with **one whole-tuple
     *       `mesh.haloExchange()`**, which is what makes the ghost potential
     *       current for step 2's one-ring gradient and the ghost positions
     *       current for step 1's geometry. One exchange, not two: the depth-2
     *       halo built once in `SurfaceMesh` is what covers the two-ring
     *       stencil, not a second exchange (risk R8). The ordering
     *       `haloExchange()` -> `updateSheetVector` -> BR evaluation is the one
     *       T2c's regression test performs and checks; getting it wrong is wrong
     *       only near partition boundaries.
     */
    void computeRightHandSidePotential( mesh_type& mesh,
                                        const state_type& state,
                                        vector_view& vertex_dot,
                                        scalar_view& potential_dot )
    {
        const int n_owned = mesh.ownedVertexCount();
        const int n_local = mesh.totalVertexCount();

        // 0. Ghosts. See the @note above for why this is one exchange and why
        //    it is here rather than in the integrator: the RHS is what has the
        //    precondition, so it is the RHS that establishes it.
        mesh.haloExchange();

        // 1. Geometry at the CURRENT stage positions. Recomputed every stage --
        //    stale geometry silently degrades RK3 to first order.
        geometry_type geometry;
        geometry.compute( mesh.positions(), n_local, mesh.faceVertices() );

        // 2. The sheet vector, S = -n x grad_s(phi). Owned rows are complete;
        //    ghost rows hold partial sums, which is why the quadrature reads
        //    owned rows only.
        state.updateSheetVector( mesh, geometry );

        // 3. The Birkhoff-Rott velocity. Overwritten, with 1/4pi and br_sign
        //    ALREADY APPLIED (T2c) -- do not re-apply either here.
        if ( static_cast<int>( vertex_dot.extent( 0 ) ) != n_owned )
            Kokkos::realloc( vertex_dot, n_owned );
        _br_solver->computeInterfaceVelocity( mesh, geometry, state,
                                              *_quadrature, _params,
                                              vertex_dot );

        // 4. u . n from the RAW BR velocity, BEFORE the velocity mode is
        //    applied (mesh_solver.py:1243-1244 takes it in that order, and
        //    under `normal` the two are not the same number).
        resizeScalar( _normal_speed, "beatnik_rhs_normal_speed", n_owned );
        {
            auto u = vertex_dot;
            auto vn = geometry.vertex_normal;
            auto ns = _normal_speed;
            Kokkos::parallel_for(
                "beatnik_rhs_normal_speed",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) {
                    ns( i ) = u( i, 0 ) * vn( i, 0 ) + u( i, 1 ) * vn( i, 1 ) +
                              u( i, 2 ) * vn( i, 2 );
                } );
            Kokkos::fence();
        }

        // The Bernoulli scalar's INPUT. Under `surface-riesz` it is a second BR
        // evaluation, which is collective -- so it is driven by a parameter that
        // is the same on every rank, and every rank reaches it the same number
        // of times per step. Resolved before the velocity mode for the same
        // reason step 4 is.
        const bool riesz =
            ( _params.bernoulli_scalar_mode == BernoulliScalarMode::SurfaceRiesz );
        if ( riesz )
        {
            resizeScalar( _bernoulli_input, "beatnik_rhs_riesz", n_owned );
            _br_solver->computeSurfaceRieszScalar( mesh, geometry, state,
                                                   *_quadrature, _params,
                                                   _bernoulli_input );
        }

        applyVelocityMode( geometry, vertex_dot );

        // 5. Surface tension, BEFORE the volume projection (see the declaration
        //    of computeSurfaceTension for why the order is load-bearing).
        if ( _params.sigma != Real( 0 ) )
            computeSurfaceTension( mesh, geometry, vertex_dot );

        // 6. The volume projection.
        if ( _params.preserve_volume )
            VolumeProjection<ExecutionSpace, MemorySpace>::removeVolumeFlux(
                mesh, vertex_dot );

        // 7. The Bernoulli potential.
        resizeScalar( potential_dot, "beatnik_rhs_potential_dot", n_owned );
        computeBernoulliPotential( mesh, state,
                                   riesz ? _bernoulli_input : _normal_speed,
                                   potential_dot );

        // 8. phi_dot = forcing_sign * A * V + mu * Laplacian(phi).
        {
            const Real gain = _params.forcing_sign * _params.A;
            auto pd = potential_dot;
            Kokkos::parallel_for(
                "beatnik_rhs_forcing",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) { pd( i ) *= gain; } );
            Kokkos::fence();
        }
        if ( _params.mu != Real( 0 ) )
        {
            resizeScalar( _viscosity, "beatnik_rhs_viscosity", n_owned );
            computeScalarViscosity( mesh, geometry, state, _viscosity );
            auto pd = potential_dot;
            auto vis = _viscosity;
            Kokkos::parallel_for(
                "beatnik_rhs_add_viscosity",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) { pd( i ) += vis( i ); } );
            Kokkos::fence();
        }

        // 9. Re-centre phi_dot, AFTER the viscous term (mesh_solver.py:1264-1268
        //    subtracts the mean of the sum, not of the forcing alone).
        subtractAreaWeightedMean( mesh, geometry, potential_dot );
    }

    /**
     * @brief Evaluate \f$(\dot x, \dot S)\f$ for the sheet-vector formulation.
     *
     * Port of mesh_solver.py::mesh_rhs (lines 1207-1233)
     *
     * Same steps 1-7 as the potential form, then:
     *  8. \f$\dot S = -\sigma_f A\,(\hat n\times\nabla_s V)\f$.
     *  9. Add \f$\mu\,\mathcal{L}_{\text{graph}}S\f$ if \f$\mu \ne 0\f$ — note
     *     the **graph** Laplacian here, not the cotangent one, regardless of
     *     `--viscosity-mode` (which only affects the potential form). This
     *     asymmetry is in the reference.
     * 10. Re-project \f$\dot S\f$ onto the tangent plane.
     *
     * There is no gauge to fix, so no mean subtraction.
     *
     * @param[out] vertex_dot `(Nv,3)` \f$\dot x\f$, units velocity.
     * @param[out] sheet_dot  `(Nv,3)` \f$\dot S\f$, units velocity/time.
     *
     * **Still T5c's, and still throwing.** Its `mesh` parameter was widened to
     * `mesh_type&` at T2d with the rest of them, so T5c implements a body only.
     */
    void computeRightHandSideSheet( mesh_type& mesh,
                                    const state_type& state,
                                    vector_view& vertex_dot,
                                    vector_view& sheet_dot )
    {
        (void)mesh;
        (void)state;
        (void)vertex_dot;
        (void)sheet_dot;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver", "computeRightHandSideSheet" );
    }

    /**
     * @brief The Bernoulli potential \f$V\f$.
     *
     * Port of mesh_solver.py::potential_mesh_rhs (lines 1255-1260) and
     * ::mesh_rhs (lines 1222-1227); structured-grid ancestor
     * solver.py::rhs (line 199)
     *
     * \f[
     *   V \;=\; \Psi^2 \;-\; \tfrac14 |S|^2 \;-\; 2 g z_3
     * \f]
     * where \f$\Psi\f$ is the scalar chosen by `--bernoulli-scalar-mode`:
     * \f$u\!\cdot\!\hat n\f$ (`normal-speed`), \f$\tfrac12 u\!\cdot\!\hat n\f$
     * (`normal-proxy`), or the surface Riesz scalar (`surface-riesz`).
     *
     * Units: velocity^2. Note the three terms are all velocity-squared:
     * \f$\Psi^2\f$ and \f$|S|^2\f$ obviously, and \f$g z_3\f$ because
     * \f$[g]=\f$ length/time^2.
     *
     * @param normal_speed `(N_owned,)` \f$u\!\cdot\!\hat n\f$, or the Riesz
     *        scalar under `surface-riesz` — the caller resolves which, because
     *        the Riesz path is a second collective BR evaluation and does not
     *        belong inside a per-vertex kernel.
     * @param[out] bernoulli `(N_owned,)` \f$V\f$.
     *
     * The `normal-proxy` factor of \f$\tfrac12\f$ is applied **here**, on the
     * scalar, before squaring (`mesh_solver.py:922-923`) — it changes the
     * quantity being squared, not the exponent.
     */
    void computeBernoulliPotential( mesh_type& mesh, const state_type& state,
                                    const scalar_view& normal_speed,
                                    scalar_view& bernoulli )
    {
        (void)state;
        const int n = static_cast<int>( bernoulli.extent( 0 ) );
        auto pos = mesh.positions();
        auto sheet = mesh.sheetVector();
        auto psi = normal_speed;
        auto out = bernoulli;
        const Real g = _params.g;
        const Real scale =
            ( _params.bernoulli_scalar_mode == BernoulliScalarMode::NormalProxy )
                ? Real( 0.5 )
                : Real( 1 );
        Kokkos::parallel_for(
            "beatnik_bernoulli_potential",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                const Real s = scale * psi( i );
                const Real s2 = sheet( i, 0 ) * sheet( i, 0 ) +
                                sheet( i, 1 ) * sheet( i, 1 ) +
                                sheet( i, 2 ) * sheet( i, 2 );
                // V = Psi^2 - 1/4 |S|^2 - 2 g z. Every coefficient is
                // load-bearing; see the file header.
                out( i ) = s * s - Real( 0.25 ) * s2 -
                           Real( 2 ) * g * pos( i, 2 );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Select the marker velocity from the Birkhoff-Rott velocity.
     *
     * Port of mesh_solver.py::_interface_velocity (lines 259-269)
     *
     * `full` returns \f$u\f$ unchanged; `normal` returns
     * \f$(u\cdot\hat n)\hat n\f$.
     *
     * For a **closed** surface the two produce the same *shape* evolution —
     * tangential marker motion is a reparameterization, not a physical
     * motion — but they produce very different *meshes*: `full` lets markers
     * slide along the surface and pile up in the roll-up, `normal` keeps them
     * where they are. The default is `full`, with the dynamic remesher cleaning
     * up the resulting distribution.
     */
    void applyVelocityMode( const geometry_type& geometry,
                            vector_view& velocity )
    {
        if ( _params.velocity_mode == VelocityMode::Full )
            return; // `velocity` unchanged -- the reference returns it as is.

        const int n = static_cast<int>( velocity.extent( 0 ) );
        auto u = velocity;
        auto vn = geometry.vertex_normal;
        Kokkos::parallel_for(
            "beatnik_velocity_mode_normal",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                const Real s = u( i, 0 ) * vn( i, 0 ) + u( i, 1 ) * vn( i, 1 ) +
                               u( i, 2 ) * vn( i, 2 );
                for ( int d = 0; d < 3; ++d )
                    u( i, d ) = s * vn( i, d );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Localized surface-tension velocity \f$\sigma\,\Delta_{LB}x\f$.
     *
     * Port of mesh_solver.py::_surface_tension_velocity (lines 1113-1139)
     *
     * Zero (and skipped entirely) when \f$\sigma=0\f$. Otherwise
     * \f$\sigma\,\Delta_{LB}x\f$, which by the sign convention of
     * `SurfaceOperators::meanCurvatureNormal` is the **area-decreasing**
     * direction — added with a plus sign, not subtracted.
     *
     * When `--sigma-radius` \f$R > 0\f$ the flow is tapered by a smoothstep in
     * the distance \f$d\f$ from `--sigma-center`:
     * \f[
     *   t = \mathrm{clamp}\!\Big(\frac{R-d}{0.4R},\,0,\,1\Big),\qquad
     *   w = t^2(3-2t),
     * \f]
     * so \f$w=1\f$ inside \f$0.6R\f$ and \f$w=0\f$ beyond \f$R\f$, with a
     * \f$C^1\f$ transition between. This confines the smoothing to the
     * pinch-off region rather than reshaping distant parts of the body.
     *
     * The subsequent volume projection is deliberately **not** localized to the
     * same ball. Confining it would cancel the pinch itself, because the neck
     * carries the largest localization weight — the reference is explicit about
     * this (`mesh_solver.py:1134-1138`).
     *
     * **T2d — this ADDS into `velocity` rather than returning a field.** The
     * reference returns `None` when \f$\sigma=0\f$ and the caller adds it
     * (`mesh_solver.py:1245-1247`); accumulating in place is the same
     * arithmetic without an `(Nv,3)` temporary, and the \f$\sigma=0\f$ branch
     * is the caller's guard. **No sign flip**: `meanCurvatureNormal` returns
     * \f$-2H\hat n_{\text{out}}\f$, verified inward at every vertex at T2b, and
     * that is already the area-decreasing direction.
     */
    void computeSurfaceTension( mesh_type& mesh,
                                const geometry_type& geometry,
                                vector_view& velocity )
    {
        const int n_owned = static_cast<int>( velocity.extent( 0 ) );
        const int n_local = mesh.totalVertexCount();

        // Local-sized: meanCurvatureNormal is a face-loop scatter and takes Nv
        // from its OUTPUT view, so an owned-sized result would index out of
        // bounds on a ghost corner. Only owned rows are read back.
        resizeVector( _curvature_normal, "beatnik_surface_tension_curvature",
                      n_local );
        auto pos = mesh.positions();
        SurfaceOperators::meanCurvatureNormal( pos, mesh.faceVertices(),
                                               geometry.vertex_area,
                                               _curvature_normal );

        auto u = velocity;
        auto h = _curvature_normal;
        const Real sigma = _params.sigma;
        const Real radius = _params.sigma_radius;
        const Real cx = _params.sigma_center[0];
        const Real cy = _params.sigma_center[1];
        const Real cz = _params.sigma_center[2];
        Kokkos::parallel_for(
            "beatnik_surface_tension",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) {
                Real weight = Real( 1 );
                if ( radius > Real( 0 ) )
                {
                    const Real dx = pos( i, 0 ) - cx;
                    const Real dy = pos( i, 1 ) - cy;
                    const Real dz = pos( i, 2 ) - cz;
                    const Real dist =
                        Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
                    Real t = ( radius - dist ) / ( Real( 0.4 ) * radius );
                    t = ( t < Real( 0 ) ) ? Real( 0 )
                                          : ( t > Real( 1 ) ? Real( 1 ) : t );
                    // smoothstep: 1 inside 0.6 R, 0 at R, C^1 between.
                    weight = t * t * ( Real( 3 ) - Real( 2 ) * t );
                }
                for ( int d = 0; d < 3; ++d )
                    u( i, d ) += sigma * weight * h( i, d );
            } );
        Kokkos::fence();
    }

    /**
     * @brief The viscous term on the potential.
     *
     * Port of mesh_solver.py::_scalar_viscosity (lines 1062-1065)
     *
     * \f$\mu\,\Delta_s\phi\f$ with the cotangent Laplace-Beltrami under
     * `--viscosity-mode laplace-beltrami` (the default) or the graph Laplacian
     * under `graph`.
     *
     * The two are **not** interchangeable at the same \f$\mu\f$: the graph
     * stencil is \f$O(h^2)\Delta\f$, so it weakens exactly where the mesh is
     * refined and the sheet spike lives, while the cotangent form is a true
     * \f$\Delta\f$ and dissipates hardest there. Switching modes without
     * rescaling \f$\mu\f$ changes the effective viscosity by orders of
     * magnitude on a graded mesh.
     *
     * **T2d — `result` is `mu * Laplacian(phi)`, not the bare Laplacian**, and
     * it is written over the **owned** range (its own extent). The `mu` factor
     * lives here rather than at the call site because the two modes are not
     * interchangeable at the same `mu` and keeping the coefficient adjacent to
     * the operator choice is what makes that visible.
     *
     * @pre \f$\phi\f$'s ghost values are current — the caller's
     *      `mesh.haloExchange()`.
     */
    void computeScalarViscosity( mesh_type& mesh,
                                 const geometry_type& geometry,
                                 const state_type& state, scalar_view& result )
    {
        (void)state;
        const int n_owned = static_cast<int>( result.extent( 0 ) );
        const Real mu = _params.mu;

        if ( _params.viscosity_mode == ViscosityMode::Graph )
        {
            // A GATHER over the one-ring, so it writes only the rows it is
            // given and an owned-sized output is correct. `mesh.potential()` is
            // read at local indices, which the exchange above made current.
            SurfaceOperators::graphLaplacianScalar(
                mesh.vertexOneRing(), mesh.potential(), result );
        }
        else
        {
            // A face-loop SCATTER, so its output must span the whole local
            // range (T2b: Nv comes from the output view). This is exactly the
            // call the `OutScalarView` template split at T2b exists for --
            // `values` is `mesh.potential()`, a Cabana slice, and the result is
            // a Beatnik-owned view.
            const int n_local = mesh.totalVertexCount();
            resizeScalar( _laplacian, "beatnik_viscosity_laplacian", n_local );
            SurfaceOperators::cotangentLaplacianScalar(
                mesh.positions(), mesh.faceVertices(), mesh.potential(),
                geometry.vertex_area, _laplacian );
            auto lap = _laplacian;
            auto out = result;
            Kokkos::parallel_for(
                "beatnik_viscosity_gather",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) { out( i ) = lap( i ); } );
            Kokkos::fence();
        }

        auto out = result;
        Kokkos::parallel_for(
            "beatnik_viscosity_scale",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) { out( i ) *= mu; } );
        Kokkos::fence();
    }

  private:
    /// Reallocate only on a size change, so the three RK3 stages reuse one
    /// allocation. `WithoutInitializing` because every consumer below either
    /// overwrites the whole range or is documented to zero it first.
    ///
    /// The label is a `std::string` and NOT a `const char*`: `Kokkos::view_alloc`
    /// treats a *decayed* `const char*` as a **pointer to memory**, not as a
    /// label, and the resulting `static_assert` ("Cannot give pointer-to-memory
    /// for view allocation") is several screens from the call site. A string
    /// literal passed directly works because it is still an array type there.
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

    /// Subtract the global area-weighted mean of an owned-range per-vertex
    /// scalar from itself.
    ///
    /// Port of mesh_solver.py::_area_weighted_scalar_mean (lines 239-244), as
    /// applied at ::potential_mesh_rhs (lines 1264-1268)
    ///
    /// REDUCE BOTH SUMS, THEN DIVIDE -- the same contract, and the same reason,
    /// as `SurfaceState::centerPotential`: an `allReduceSum` of per-rank means
    /// is not the global mean, and subtracting a per-rank mean would give
    /// `phi_dot` a piecewise-constant jump across every partition boundary.
    /// Invisible at one rank.
    static void subtractAreaWeightedMean( mesh_type& mesh,
                                          const geometry_type& geometry,
                                          scalar_view& values )
    {
        const int n_owned = static_cast<int>( values.extent( 0 ) );
        auto area_owned = Kokkos::subview( geometry.vertex_area,
                                           std::make_pair( 0, n_owned ) );

        Real weighted = 0, area = 0;
        SurfaceOperators::areaWeightedMeanPartials( values, area_owned,
                                                    weighted, area );

        Real pair[2] = { weighted, area };
        Real reduced[2] = { 0, 0 };
        MPI_Allreduce( pair, reduced, 2, MPI_DOUBLE, MPI_SUM, mesh.comm() );

        Real mean = 0;
        if ( reduced[1] > Real( 0 ) )
        {
            mean = reduced[0] / reduced[1];
        }
        else
        {
            // The Python's fallback is the UNWEIGHTED mean (lines 242-243).
            // Reduced too, so ranks cannot disagree about the shift.
            auto v = values;
            Real local_sum = 0;
            Kokkos::parallel_reduce(
                "beatnik_rhs_unweighted_mean",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i, Real& acc ) { acc += v( i ); },
                local_sum );
            const Real total = Comm::allReduceSum( mesh.comm(), local_sum );
            const long long count = mesh.globalVertexCount();
            mean = ( count > 0 ) ? total / static_cast<Real>( count ) : Real( 0 );
        }

        auto v = values;
        Kokkos::parallel_for(
            "beatnik_rhs_subtract_mean",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) { v( i ) -= mean; } );
        Kokkos::fence();
    }

    ZModelParams _params;
    br_solver_type* _br_solver = nullptr;
    const quadrature_type* _quadrature = nullptr;

    /// Scratch, allocated on first use and reused across stages and steps.
    scalar_view _normal_speed;
    scalar_view _bernoulli_input;
    scalar_view _viscosity;
    scalar_view _laplacian;
    vector_view _curvature_normal;
};

} // namespace Beatnik

#endif // BEATNIK_ZMODELSOLVER_HPP
