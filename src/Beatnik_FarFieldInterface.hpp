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
 * @file Beatnik_FarFieldInterface.hpp
 * @brief ADAPTER (3 of 3). Far-field summation of the regularized
 *        Birkhoff-Rott kernel, backed by **Canopy**'s fast multipole method.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Canopy type. `Beatnik_BRSolverFMM.hpp`
 * calls only through `FarFieldSolver` below. Canopy has **not been read** while
 * writing this header (see `tasks/framework.md`, task F1); the interface is
 * shaped by the kernel the physics needs.
 *
 * THE KERNEL
 * ----------
 * Both the velocity and the Riesz-scalar evaluations are sums over the same
 * regularized \f$1/r^2\f$ field. Writing \f$\delta = x_t - y_s\f$ and
 * \f$r^2 = |\delta|^2\f$, with blob parameter \f$b\f$ (see
 * `ZModelParams::blob()`), the shared kernel is
 *
 * \f[
 *   K(x_t, y_s) \;=\; \frac{\delta}{(b + r^2)^{3/2}} .
 * \f]
 *
 * Note the kernel is **not** \f$\delta/|\delta|^3\f$: the denominator carries
 * the additive blob, which both removes the singularity at \f$r=0\f$ (a target
 * *is* a source in this problem — every vertex appears on both sides, including
 * its own self-interaction, which contributes exactly zero to the velocity
 * because \f$\delta = 0\f$ there) and sets the sheet thickness. An FMM
 * expansion of this kernel is therefore an expansion of a *softened* Coulomb
 * field, not of the bare one; that difference matters for the error estimate at
 * separations comparable to \f$\sqrt{b}\f$.
 *
 * The two evaluations Beatnik needs from it:
 *
 * **Velocity** (`evaluateCurl`), the Birkhoff-Rott integral —
 * Port of mesh_solver.py::_source_velocity_direct_unsigned (lines 437-454):
 * \f[
 *   u(x_t) \;=\; \frac{1}{4\pi} \sum_s K(x_t, y_s) \times S_s ,
 * \f]
 * where \f$S_s\f$ is the area-weighted sheet strength at source \f$s\f$. The
 * \f$1/4\pi\f$ is applied **once, here**; no caller re-applies it.
 *
 * **Riesz scalar** (`evaluateDot`) —
 * Port of mesh_solver.py::_source_riesz_scalar_direct (lines 457-489):
 * \f[
 *   \Psi(x_t) \;=\; -\frac{1}{4\pi^2} \sum_s \frac{\delta \cdot G_s}{(b+r^2)^{3/2}} ,
 * \f]
 * where \f$G_s\f$ is the area-weighted surface gradient at source \f$s\f$. Note
 * the different normalization: \f$-1/(4\pi^2)\f$, **not** \f$1/(4\pi)\f$, and
 * negative. On a flat periodic patch this discretizes
 * \f$\mathcal{F}^{-1}(i k_j \hat w_j / |k|)/(2\pi)\f$.
 *
 * WHY THIS IS AN ADAPTER AND NOT JUST "CALL CANOPY"
 * -------------------------------------------------
 * Both evaluations are sums of the same scalar kernel against a *vector*
 * source, differing only in how the result is contracted (cross product vs dot
 * product). An FMM implementation may expose one, the other, both, or a
 * generic three-component field evaluation. Confining the choice here means the
 * BR solver above does not care.
 */

#ifndef BEATNIK_FARFIELDINTERFACE_HPP
#define BEATNIK_FARFIELDINTERFACE_HPP

#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include <mpi.h>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Far-field kernel summation. Canopy lives behind this.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space holding sources and targets.
 */
template <class ExecutionSpace, class MemorySpace>
class FarFieldSolver
{
  public:
    /**
     * @param comm   Communicator over which sources and targets are
     *               distributed. The sum is global: every target sees every
     *               source on every rank.
     * @param params FMM tunables.
     */
    FarFieldSolver( MPI_Comm comm, const FmmParams& params )
        : _comm( comm )
        , _params( params )
    {
    }

    /**
     * @brief Build or update the spatial decomposition for a source set.
     *
     * Separated from evaluation because the sources move every RK stage but
     * the *tree structure* can often be reused across stages within a step. A
     * tree rebuilt every stage is correct but wasteful; a tree never rebuilt is
     * wrong once the surface has deformed. The FMM task decides the policy —
     * this call is where it lands.
     *
     * @param source_points `(Ns, 3)` source positions.
     *
     * @tparam PointView
     *         // TODO(types): templated pending Tessera/Canopy interface;
     *         // collapse to a concrete type once known.
     *
     * @note MPI. Collective. Building a global tree requires a bounding box
     *       reduction (`MPI_Allreduce` with `MPI_MIN`/`MPI_MAX` over the
     *       coordinates) and then a redistribution of sources to tree cells.
     */
    template <class PointView>
    void setSources( const PointView& source_points )
    {
        (void)source_points;
        BEATNIK_NOT_IMPLEMENTED( "FarFieldSolver", "setSources" );
    }

    /**
     * @brief Evaluate the cross-product (velocity) sum.
     *
     * \f$ u(x_t) = \frac{1}{4\pi}\sum_s
     *     \frac{(x_t - y_s) \times S_s}{(b + |x_t-y_s|^2)^{3/2}} \f$
     *
     * @param targets   `(Nt, 3)` evaluation points.
     * @param strengths `(Ns, 3)` area-weighted sheet strengths \f$S_s\f$, in
     *                  units of (velocity x length), i.e. circulation.
     * @param blob      \f$b\f$, the squared-length kernel offset from
     *                  `ZModelParams::blob()`.
     * @param[out] velocity `(Nt, 3)` result, **overwritten** (not accumulated),
     *                  in velocity units. The \f$1/4\pi\f$ is already applied.
     *
     * @note MPI. Collective. The sum is over sources on all ranks.
     */
    template <class PointView, class StrengthView, class VelocityView>
    void evaluateCurl( const PointView& targets, const StrengthView& strengths,
                       Real blob, VelocityView& velocity )
    {
        (void)targets;
        (void)strengths;
        (void)blob;
        (void)velocity;
        BEATNIK_NOT_IMPLEMENTED( "FarFieldSolver", "evaluateCurl" );
    }

    /**
     * @brief Evaluate the dot-product (Riesz scalar) sum.
     *
     * \f$ \Psi(x_t) = -\frac{1}{4\pi^2}\sum_s
     *     \frac{(x_t - y_s) \cdot G_s}{(b + |x_t-y_s|^2)^{3/2}} \f$
     *
     * Used only when `--bernoulli-scalar-mode surface-riesz` is selected. The
     * Python supports this mode for `direct`, `local` and `clustered` but
     * **not** for `treecode` (`mesh_solver.py:605` raises), so a Python run
     * cannot combine `surface-riesz` with the default far-field path. Beatnik
     * removes that restriction by routing it through the same FMM.
     *
     * @param targets   `(Nt, 3)` evaluation points.
     * @param gradients `(Ns, 3)` area-weighted surface gradients \f$G_s\f$.
     * @param blob      \f$b\f$ from `ZModelParams::blob()`.
     * @param[out] scalar `(Nt,)` result, **overwritten**. The
     *                  \f$-1/4\pi^2\f$ is already applied.
     *
     * @note MPI. Collective, as `evaluateCurl`.
     */
    template <class PointView, class GradientView, class ScalarView>
    void evaluateDot( const PointView& targets, const GradientView& gradients,
                      Real blob, ScalarView& scalar )
    {
        (void)targets;
        (void)gradients;
        (void)blob;
        (void)scalar;
        BEATNIK_NOT_IMPLEMENTED( "FarFieldSolver", "evaluateDot" );
    }

    /// FMM tunables in force.
    const FmmParams& params() const { return _params; }

  private:
    MPI_Comm _comm;
    FmmParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_FARFIELDINTERFACE_HPP
