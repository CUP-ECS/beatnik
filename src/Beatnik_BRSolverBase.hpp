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
 * @file Beatnik_BRSolverBase.hpp
 * @brief Interface for the Birkhoff-Rott evaluation: the induced interface
 *        velocity, and the optional surface Riesz scalar.
 *
 * THE EQUATION
 * ------------
 * Port of mesh_solver.py::_mesh_birkhoff_rott_velocity_from_sources
 * (lines 388-434)
 *
 * The regularized Birkhoff-Rott integral for the velocity a vortex sheet
 * induces on itself:
 * \f[
 *   u(x) \;=\; \frac{\sigma_{BR}}{4\pi}
 *     \int_\Sigma \frac{(x-y)\times S(y)}{\big(b + |x-y|^2\big)^{3/2}}\; dS_y ,
 * \f]
 * discretized by the source quadrature into
 * \f[
 *   u(x_t) \;=\; \frac{\sigma_{BR}}{4\pi}\sum_s
 *     \frac{\delta_{ts}\times S_s\,\omega_s}{(b + |\delta_{ts}|^2)^{3/2}},
 *   \qquad \delta_{ts} = x_t - y_s .
 * \f]
 *
 * Conventions, all of which are easy to get wrong and none of which are free
 * parameters:
 *
 * - **\f$1/4\pi\f$ is applied exactly once**, inside the kernel sum. Not in the
 *   quadrature weights, not by the caller.
 * - **The cross product order is \f$\delta \times S\f$**, not
 *   \f$S \times \delta\f$. Reversing it negates the whole velocity, i.e. it is
 *   equivalent to `--br-sign -1`.
 * - **\f$b\f$ is added to \f$r^2\f$, not to \f$r\f$**, and the power is
 *   \f$3/2\f$ on the sum. `ZModelParams::blob()` supplies \f$b\f$: it is
 *   \f$\epsilon^2\f$ under `--kernel-blob-mode length` and \f$\epsilon\f$ under
 *   `matlab`, so \f$b\f$ always has units of length squared.
 * - **Self-interaction is included.** Under the `Vertex` quadrature a target
 *   *is* a source; its own term has \f$\delta = 0\f$ and therefore contributes
 *   exactly zero. There is no exclusion list, and adding one changes the
 *   answer (it would also drop the near-neighbour terms that dominate).
 * - **\f$\sigma_{BR}\f$ is `--br-sign`**, \f$\pm 1\f$, applied to the final
 *   sum. It exists to flip a global orientation convention, not to fix a local
 *   sign error.
 *
 * THE SURFACE RIESZ SCALAR
 * ------------------------
 * Port of mesh_solver.py::_source_riesz_scalar_direct (lines 457-489)
 *
 * An alternative scalar for the Bernoulli forcing, selected by
 * `--bernoulli-scalar-mode surface-riesz`:
 * \f[
 *   \Psi(x_t) \;=\; -\frac{1}{4\pi^2}\sum_s
 *     \frac{\delta_{ts}\cdot G_s\,\omega_s}{(b+|\delta_{ts}|^2)^{3/2}},
 * \f]
 * with \f$G_s = \nabla_s\phi\f$ at the source. Note the **different**
 * normalization \f$-1/(4\pi^2)\f$ — negative, and \f$\pi^2\f$ not \f$\pi\f$.
 * On a flat periodic patch this reproduces the MATLAB spectral operator
 * \f$\mathcal{F}^{-1}(i k_j \hat w_j/|k|)/(2\pi)\f$, which is where the
 * normalization comes from; it is not a free choice.
 *
 * ONE INTERFACE, TWO IMPLEMENTATIONS
 * ----------------------------------
 * `BRSolverDirect` — the \f$O(N^2)\f$ reference sum. Slow, but it is the
 * definition, so it is what regression tests 1 and 2 use and what test 3
 * compares the FMM against.
 *
 * `BRSolverFMM` — Canopy fast multipole via `Beatnik_FarFieldInterface.hpp`.
 *
 * The Python's `local` and `clustered` intermediate approximations are **not**
 * ported. They exist there as stepping stones between the direct sum and the
 * treecode; Beatnik has a real FMM, so they would be dead weight. The CLI
 * accepts their names and maps them to `fmm` with a warning.
 */

#ifndef BEATNIK_BRSOLVERBASE_HPP
#define BEATNIK_BRSOLVERBASE_HPP

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
 * @brief Interface to a Birkhoff-Rott evaluator.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class BRSolverBase
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

    virtual ~BRSolverBase() = default;

    /// Which approximation this is.
    virtual BRApproximation kind() const = 0;

    /**
     * @brief Evaluate the induced velocity at every owned vertex.
     *
     * @param mesh       Surface providing the target positions.
     * @param geometry   Areas and normals, for the quadrature weights.
     * @param state      Solution state providing the sheet strength.
     * @param quadrature Rule producing the source points and strengths.
     * @param params     Supplies `blob()`, `br_sign`, `source_quadrature`.
     * @param[out] velocity `(Nv,3)` induced velocity at the owned vertices,
     *             **overwritten**. Units: velocity. The \f$1/4\pi\f$ and
     *             `br_sign` are both already applied.
     *
     * @note MPI. Collective, and the most expensive collective in the run:
     *       every target must see every source on every rank. The direct
     *       implementation does this by circulating source blocks around a ring
     *       of ranks; the FMM does it inside Canopy. Either way, all ranks must
     *       call it the same number of times per step — which is why the
     *       finiteness abort is a reduction rather than a local branch.
     */
    virtual void computeInterfaceVelocity( mesh_type& mesh,
                                           const geometry_type& geometry,
                                           const state_type& state,
                                           const quadrature_type& quadrature,
                                           const ZModelParams& params,
                                           vector_view& velocity ) = 0;

    /**
     * @brief Evaluate the surface Riesz scalar at every owned vertex.
     *
     * Only called under `--bernoulli-scalar-mode surface-riesz`.
     *
     * @param[out] scalar `(Nv,)` result, **overwritten**. The
     *             \f$-1/4\pi^2\f$ is already applied. Units: velocity.
     *
     * @note The Python raises for this combination under `treecode`
     *       (`mesh_solver.py:605`), so no Python run exercises
     *       `surface-riesz` + a fast far field. Beatnik supports it, which
     *       means there is no gold file to validate that combination against —
     *       recorded as risk R5 in `tasks/framework.md`.
     */
    virtual void computeSurfaceRieszScalar( mesh_type& mesh,
                                            const geometry_type& geometry,
                                            const state_type& state,
                                            const quadrature_type& quadrature,
                                            const ZModelParams& params,
                                            scalar_view& scalar ) = 0;
};

} // namespace Beatnik

#endif // BEATNIK_BRSOLVERBASE_HPP
