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
 * @file Beatnik_InitialCondition.hpp
 * @brief Construction of the initial surface: base sphere, shape deformation,
 *        and initial vorticity seeding.
 *
 * Port of run_adaptive_mesh_bubble.py::apply_initial_geometry (lines 713-886)
 * and ::_apply_polar_mode (lines 698-710)
 *
 * All coordinates below are **relative to the bubble centre**
 * \f$c = (0,0,\,\text{`--center-z'})\f$ unless stated otherwise, and
 * \f$\hat z_u = z/R\f$ is the vertical coordinate normalized by the undeformed
 * radius \f$R\f$ = `--radius`.
 *
 * THE FAST PATH
 * -------------
 * With `--initial-shape sphere`, `--initial-potential-strength 0` and
 * `--polar-amp 0` — i.e. **the defaults** — this whole file is a no-op and the
 * generated sphere is used as is (`run_adaptive_mesh_bubble.py:714-717`). That
 * is the configuration regression test 1 compares, so getting the deformations
 * exactly right is not on the critical path for the first testing rung.
 */

#ifndef BEATNIK_INITIALCONDITION_HPP
#define BEATNIK_INITIALCONDITION_HPP

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Builds the initial surface and its initial fields.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class InitialCondition
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    /// @param params Geometry and vorticity-seeding parameters.
    explicit InitialCondition( const InitialConditionParams& params )
        : _params( params )
    {
    }

    /**
     * @brief Generate the base sphere, deform it, and seed the fields.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1215-1240) and
     * ::apply_initial_geometry (lines 714-717)
     *
     * Sequence: generate (`icosphere` or `latlon`), apply the geometry
     * deformation, zero the potential (or sheet vector), seed the vorticity,
     * then seed the material position from the resulting positions.
     *
     * **T1c implements THE FAST PATH ONLY** — `--initial-shape sphere`,
     * `--initial-potential-strength 0`, `--polar-amp 0`, i.e. the defaults and
     * the configuration regression test 1 compares. Anything else reaches
     * `applyShapeDeformation` / `applyPolarMode` / `seedInitialVorticity`,
     * which are T5a and throw. The dispatch below mirrors the Python's, which
     * is a three-way branch and not two independent conditions:
     *
     *   sphere && strength == 0 && amp == 0  ->  nothing (the fast path)
     *   sphere && strength == 0 && amp != 0  ->  applyPolarMode ONLY
     *   otherwise                            ->  deformation + vorticity
     *
     * The middle case is *instead of*, not in addition to, the deformation
     * (`apply_initial_geometry` returns early at line 716), which is why this
     * is written as one branch rather than three `if`s.
     *
     * ORDER, and why it is this order. The Python's dataclass constructor
     * re-centres the potential inside `__post_init__`, and
     * `apply_initial_geometry` returns a *new* state — so the centring happens
     * after the geometry is final. Transcribed here as an explicit step, after
     * the deformation and before the material seed. The centring is a no-op on
     * the fast path (the potential is identically zero, so its area-weighted
     * mean is zero), and it is called anyway rather than skipped, because it is
     * the *only* collective in the cold-start field path and a run that skipped
     * it here would first exercise it at T2d with the RHS in flight.
     *
     * @param[out] mesh  Surface to build.
     * @param[out] state Fields to initialize.
     *
     * @note MPI. Collective throughout: `generateIcosphere` distributes and
     *       exchanges, and `centerPotential` reduces. Every rank must call it.
     */
    void build( mesh_type& mesh, state_type& state ) const
    {
        const Real center[3] = { 0.0, 0.0, _params.center_z };

        switch ( _params.mesh_kind )
        {
        case MeshKind::Icosphere:
            mesh.generateIcosphere( _params.icosphere_subdivisions,
                                    _params.radius, center );
            break;
        case MeshKind::LatLon:
            // Tessera has the generator (M1 gap G7 closed); the adapter body is
            // T5a-era, so this throws rather than silently building an
            // icosphere. `--mesh-kind latlon` is on no regression path.
            mesh.generateLatLonSphere( _params.n_theta, _params.n_phi,
                                       _params.radius, center );
            break;
        }

        // The fields must be defined before anything reads them: Tessera's
        // vertex user pack is uninitialized storage on a freshly built mesh.
        state.initializeFields( mesh );

        const bool fast_path =
            ( _params.shape == InitialShape::Sphere ) &&
            ( _params.initial_potential_strength == Real( 0 ) );
        if ( !fast_path )
        {
            applyShapeDeformation( mesh );
            seedInitialVorticity( mesh, state );
        }
        else if ( _params.polar_amp != Real( 0 ) )
        {
            applyPolarMode( mesh );
        }

        // Both deformation paths move vertices, so the geometry the centring
        // weights by must be computed after them, not before.
        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );
        state.centerPotential( mesh, geometry );

        // Last, so the material coordinate records the FINAL initial geometry.
        state.seedMaterialPosition( mesh );
    }

    /**
     * @brief Deform the sphere into an oblate / mushroom / skirt shape.
     *
     * Port of run_adaptive_mesh_bubble.py::apply_initial_geometry
     * (lines 719-786)
     *
     * With \f$(x,y,z)\f$ relative to the centre, \f$\rho=\sqrt{x^2+y^2}\f$,
     * \f$\varphi = \operatorname{atan2}(y,x)\f$, \f$r = \|(x,y,z)\|\f$,
     * \f$\hat z_u = \mathrm{clamp}(z/R,-1,1)\f$ and
     * \f$s^2 = \mathrm{clamp}((\rho/r)^2, 0, 1) = \sin^2\theta\f$:
     *
     * **All non-sphere shapes** apply the anisotropic scaling
     * \f$(x,y,z) \to (\alpha x,\ \alpha y,\ \beta z)\f$ with \f$\alpha\f$ =
     * `--horizontal-scale` (1.28) and \f$\beta\f$ = `--vertical-scale` (0.68).
     * `oblate` stops here.
     *
     * **`mushroom-seed`** modulates \f$\alpha\f$ by a Gaussian rim bulge plus an
     * azimuthal ripple:
     * \f[
     *   \text{rim} = a_r \exp\!\Big[-\tfrac12\big(\tfrac{\hat z_u - c_r}{w_r}
     *     \big)^2\Big], \qquad
     *   \text{ripple} = a_\varphi\, s^2 \cos(m\varphi),
     * \f]
     * \f$\alpha \to \alpha(1 + \text{rim} + \text{ripple})\f$. The \f$s^2\f$
     * factor kills the ripple at the poles, where \f$\varphi\f$ is undefined.
     *
     * **`skirt-seed`** replaces the single rim by a bulge minus a neck,
     * \f$\text{rim} = \text{skirt} - \text{neck}\f$ with both Gaussians in
     * \f$\hat z_u\f$, and additionally drops the lip vertically:
     * \f[
     *   z \to z - d\,R\,\exp\!\Big[-\tfrac12\big(\tfrac{\hat z_u-c_s}{w_s}
     *     \big)^2\Big]\, s^2 ,
     * \f]
     * with \f$d\f$ = `--skirt-drop`. Note the drop is applied to \f$z\f$
     * **before** the \f$\beta\f$ scaling, so the effective displacement is
     * \f$\beta d R\f$.
     *
     * All Gaussian widths are floored at 1e-12 to keep a zero width finite.
     */
    void applyShapeDeformation( mesh_type& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "InitialCondition", "applyShapeDeformation" );
    }

    /**
     * @brief Axisymmetric Legendre radial perturbation (the RT seed).
     *
     * Port of run_adaptive_mesh_bubble.py::_apply_polar_mode (lines 698-710)
     *
     * \f[
     *   r \;\to\; r\,\big(1 + a\,P_\ell(\cos\theta)\big),
     *   \qquad \cos\theta = \mathrm{clamp}(z/r, -1, 1),
     * \f]
     * with \f$\ell\f$ = `--polar-mode`, \f$a\f$ = `--polar-amp`, and
     * \f$\varphi\f$ unchanged. Applied **instead of** the shape deformation
     * when the shape is `sphere` and no vorticity is seeded
     * (`run_adaptive_mesh_bubble.py:714-717`), which is the single-mode
     * Rayleigh-Taylor initial condition.
     *
     * The Python calls `scipy.special.eval_legendre`. The C++ port needs its
     * own \f$P_\ell\f$; use the standard three-term recurrence
     * \f$(\ell+1)P_{\ell+1} = (2\ell+1)xP_\ell - \ell P_{\ell-1}\f$, which is
     * stable for \f$|x|\le 1\f$ and the low \f$\ell\f$ this is used at. Do not
     * use the explicit polynomial form — it loses precision by \f$\ell\approx
     * 10\f$.
     *
     * A no-op when \f$a = 0\f$, which is the default.
     */
    void applyPolarMode( mesh_type& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "InitialCondition", "applyPolarMode" );
    }

    /**
     * @brief Seed the initial circulation.
     *
     * Port of run_adaptive_mesh_bubble.py::apply_initial_geometry
     * (lines 788-886)
     *
     * A no-op when `--initial-potential-strength` is 0 (the default).
     * Otherwise \f$\phi \mathrel{+}= \Lambda\,p(x)\f$ (potential model) or
     * \f$S \mathrel{+}= \Lambda\,p(x)\,\hat e_\varphi\f$ (sheet-vector model),
     * with \f$\Lambda\f$ the strength and \f$p\f$ a profile.
     *
     * With \f$q = (\hat z_u - c_v)/w_v\f$ (centre and width from
     * `--initial-vorticity-center/-width`, and \f$\hat z_u\f$ normalized by
     * \f$R\beta\f$ for a deformed shape) and the radial weight
     * \f$\omega = (\rho/\rho_{\max})^{\,p_r}\f$ from
     * `--initial-vorticity-radial-power`:
     *
     * | mode | potential model | sheet-vector model |
     * |------|-----------------|--------------------|
     * | `vertical`  | \f$z - c_z\f$ (absolute z) | \f$1\f$ |
     * | `rim-shear` | \f$\tanh q\f$              | \f$e^{-q^2/2}\f$ |
     * | `rim-bump`  | \f$e^{-q^2/2}\f$           | \f$q\,e^{-q^2/2}\f$ |
     * | `lip-shear` | \f$\omega e^{-q^2/2}\f$    | \f$\omega e^{-q^2/2}\f$ |
     * | `lip-bump`  | \f$\omega q e^{-q^2/2}\f$  | \f$\omega q e^{-q^2/2}\f$ |
     *
     * **The two columns are not the same function, and that is intentional.**
     * The sheet vector is (up to the normal rotation) the *derivative* of the
     * potential, so the sheet-vector profiles are the derivatives of the
     * potential ones — \f$\tanh \to\f$ bump, bump \f$\to\f$ its derivative. A
     * port that shares one table between the models produces a different
     * initial vorticity in one of them.
     *
     * \f$\hat e_\varphi = (-y,\,x,\,0)/\rho\f$ is the azimuthal unit vector,
     * set to zero where \f$\rho \le 10^{-14}\f$ (on the polar axis).
     *
     * @note MPI. \f$\rho_{\max}\f$ is a global maximum
     *       (`Comm::allReduceMax`); a per-rank maximum makes the radial weight
     *       partition-dependent.
     */
    void seedInitialVorticity( const mesh_type& mesh, state_type& state ) const
    {
        (void)mesh;
        (void)state;
        BEATNIK_NOT_IMPLEMENTED( "InitialCondition", "seedInitialVorticity" );
    }

  private:
    InitialConditionParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_INITIALCONDITION_HPP
