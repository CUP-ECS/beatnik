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
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

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
     * @param mesh     Surface at the current stage geometry.
     * @param state    Current \f$\phi\f$ (and the derived sheet vector).
     * @param[out] vertex_dot `(Nv,3)` \f$\dot x\f$, units velocity.
     * @param[out] potential_dot `(Nv,)` \f$\dot\phi\f$, units velocity^2.
     *
     * @note MPI. Collective at four points: the BR evaluation, the two
     *       reductions of the volume projection, and the area-weighted mean of
     *       \f$\dot\phi\f$. Also requires a ghost exchange of \f$\phi\f$ before
     *       step 2 and of \f$V\f$ before its gradient in step 8 — see
     *       `Comm::haloExchangeField`.
     */
    void computeRightHandSidePotential( const mesh_type& mesh,
                                        const state_type& state,
                                        vector_view& vertex_dot,
                                        scalar_view& potential_dot )
    {
        (void)mesh;
        (void)state;
        (void)vertex_dot;
        (void)potential_dot;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver",
                                 "computeRightHandSidePotential" );
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
     */
    void computeRightHandSideSheet( const mesh_type& mesh,
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
     * @param normal_speed `(Nv,)` \f$u\!\cdot\!\hat n\f$, or the Riesz scalar.
     * @param[out] bernoulli `(Nv,)` \f$V\f$.
     */
    void computeBernoulliPotential( const mesh_type& mesh,
                                    const state_type& state,
                                    const scalar_view& normal_speed,
                                    scalar_view& bernoulli )
    {
        (void)mesh;
        (void)state;
        (void)normal_speed;
        (void)bernoulli;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver", "computeBernoulliPotential" );
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
        (void)geometry;
        (void)velocity;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver", "applyVelocityMode" );
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
     */
    void computeSurfaceTension( const mesh_type& mesh,
                                const geometry_type& geometry,
                                vector_view& velocity )
    {
        (void)mesh;
        (void)geometry;
        (void)velocity;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver", "computeSurfaceTension" );
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
     */
    void computeScalarViscosity( const mesh_type& mesh,
                                 const geometry_type& geometry,
                                 const state_type& state, scalar_view& result )
    {
        (void)mesh;
        (void)geometry;
        (void)state;
        (void)result;
        BEATNIK_NOT_IMPLEMENTED( "ZModelSolver", "computeScalarViscosity" );
    }

  private:
    ZModelParams _params;
    br_solver_type* _br_solver = nullptr;
    const quadrature_type* _quadrature = nullptr;
};

} // namespace Beatnik

#endif // BEATNIK_ZMODELSOLVER_HPP
