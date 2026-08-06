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
 * @file Beatnik_VolumeProjection.hpp
 * @brief Enforcement of enclosed-volume conservation, in its two forms.
 *
 * WHY VOLUME IS NOT AUTOMATICALLY CONSERVED
 * -----------------------------------------
 * The continuous Birkhoff-Rott velocity of a closed vortex sheet is divergence
 * free, so the exact evolution conserves the enclosed volume. The *discrete*
 * one does not: the quadrature, the regularization blob, the finite-difference
 * surface gradients and the RK3 truncation each break it at their own order.
 * Left alone, a rising bubble loses (or gains) a few percent of its volume over
 * a few hundred steps, which changes its buoyancy and therefore its whole
 * trajectory.
 *
 * Rather than improve the discretization, the reference **projects** — twice,
 * in two different places, for two different reasons:
 *
 * **1. Rate projection**, on the velocity, inside the RHS
 * (`--no-preserve-volume` disables it). Removes the instantaneous net volume
 * rate so the *exact* evolution of the discrete system is volume-preserving.
 *
 * **2. Position projection**, on the vertex positions, after any geometric
 * edit. Corrects the volume that has already drifted, back to the target.
 *
 * Both are rank-one corrections along the volume gradient — the minimum-norm
 * correction, which is what makes them shape-preserving. A different choice
 * (e.g. moving only the vertices near the discrepancy) would conserve volume
 * just as well and deform the bubble while doing it.
 */

#ifndef BEATNIK_VOLUMEPROJECTION_HPP
#define BEATNIK_VOLUMEPROJECTION_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Enclosed-volume conservation.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class VolumeProjection
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;

    /**
     * @brief Remove the net volume rate from a velocity field.
     *
     * Port of mesh_solver.py::_remove_discrete_volume_flux (lines 285-295)
     *
     * With \f$G = \partial V/\partial x\f$ the discrete volume gradient
     * (`SurfaceOperators::volumeGradient`), the rate of volume change under a
     * velocity \f$u\f$ is \f$\dot V = \sum_v G_v\cdot u_v\f$. Subtracting its
     * projection onto \f$G\f$,
     * \f[
     *   u \;\leftarrow\; u \;-\; \frac{\langle G, u\rangle}{\langle G, G\rangle}\,G ,
     * \f]
     * gives a field with \f$\dot V = 0\f$ **exactly**, in the discrete sense —
     * to round-off, not to truncation order. This is an orthogonal projection
     * in the Euclidean inner product on \f$\mathbb{R}^{3N_v}\f$, so it is the
     * smallest change to \f$u\f$ that achieves it.
     *
     * A no-op when \f$\langle G,G\rangle \le 0\f$, which happens only on a
     * fully degenerate mesh.
     *
     * **Called after surface tension is added, not before** — see
     * `ZModelSolver::computeSurfaceTension` for why the compensation is
     * deliberately global rather than confined to the localized region.
     *
     * @param mesh Surface providing the geometry for \f$G\f$.
     * @param[in,out] velocity `(Nv,3)` field, projected in place.
     *
     * @note MPI. Both inner products are `MPI_Allreduce`/`MPI_SUM` over owned
     *       vertices, and **both must complete before the division** — see
     *       `Comm::allReduceSum`. Reducing them separately and dividing locally
     *       gives a different scalar on every rank and a velocity field that is
     *       discontinuous across partitions.
     */
    static void removeVolumeFlux( const mesh_type& mesh,
                                  vector_view& velocity )
    {
        (void)mesh;
        (void)velocity;
        BEATNIK_NOT_IMPLEMENTED( "VolumeProjection", "removeVolumeFlux" );
    }

    /**
     * @brief Move vertices so the enclosed volume matches a target.
     *
     * Port of run_adaptive_mesh_bubble.py::project_state_to_volume
     * (lines 1054-1077)
     *
     * A Newton iteration on the scalar constraint \f$V(x) = V^*\f$, with the
     * step taken along the gradient:
     * \f[
     *   x \;\leftarrow\; x \;-\;
     *     \frac{V(x) - V^*}{\langle G, G\rangle}\;G .
     * \f]
     * Two iterations by default, which is enough because \f$V\f$ is nearly
     * linear in \f$x\f$ over the correction distances involved (the relative
     * volume error per step is \f$O(10^{-6})\f$). Exits early if
     * \f$\langle G,G\rangle \le 0\f$.
     *
     * The target is `initial_volume` — the enclosed volume of the **initial**
     * surface — for the whole run, which is why it is carried in every
     * checkpoint. Re-targeting to the current volume at restart would let the
     * bubble ratchet.
     *
     * **Only positions change.** The potential (or sheet vector) is carried
     * through untouched, so the correction is purely geometric.
     *
     * Called after: an indicator-driven refine + repair, a dynamic remesh, and
     * the periodic redistribution — i.e. after anything that moves vertices
     * outside the time integration
     * (`run_adaptive_mesh_bubble.py:1465-1468, 1514-1516, 1564-1565`).
     *
     * @param[in,out] mesh   Surface, vertices moved in place.
     * @param target_volume  \f$V^*\f$, units length^3.
     * @param iterations     Newton steps; the reference uses 2.
     *
     * @note MPI. Each iteration is one `MPI_Allreduce` for \f$V\f$ and two more
     *       for the inner products, plus a ghost exchange of the moved
     *       positions before the next iteration's gradient.
     */
    static void projectToVolume( mesh_type& mesh, Real target_volume,
                                 int iterations = 2 )
    {
        (void)mesh;
        (void)target_volume;
        (void)iterations;
        BEATNIK_NOT_IMPLEMENTED( "VolumeProjection", "projectToVolume" );
    }
};

} // namespace Beatnik

#endif // BEATNIK_VOLUMEPROJECTION_HPP
