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

#include <mpi.h>

#include <utility>

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
     * @param mesh Surface providing the geometry for \f$G\f$. **T2d — no longer
     *        `const`**: `positions()` and `faceVertices()` are non-const.
     * @param[in,out] velocity `(N_owned,3)` field, projected in place. The
     *        **owned** range, matching what the BR solver produces (T2c) and
     *        what the integrator updates.
     *
     * **T2d — the two conventions, which are opposite and both load-bearing.**
     * \f$G\f$ is *assembled* from the **whole local face set**, so every owned
     * vertex's row is complete and ghost rows hold partial sums; the two inner
     * products then *reduce* over **owned rows only**, which is what makes them
     * a partition and not a double count (risk R9). Adding a `haloScatterAdd`
     * after the assembly would double-count — see DISTRIBUTED ASSEMBLY in
     * `Beatnik_MeshGeometry.hpp`.
     *
     * @note MPI. Both inner products are `MPI_Allreduce`/`MPI_SUM` over owned
     *       vertices, and **both must complete before the division** — see
     *       `Comm::allReduceSum`. Reducing them separately and dividing locally
     *       gives a different scalar on every rank and a velocity field that is
     *       discontinuous across partitions. They are batched into ONE
     *       collective here.
     */
    static void removeVolumeFlux( mesh_type& mesh, vector_view& velocity )
    {
        const int n_owned = static_cast<int>( velocity.extent( 0 ) );
        const int n_local = mesh.totalVertexCount();

        vector_view gradient(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_volume_flux_gradient" ),
            n_local );
        SurfaceOperators::volumeGradient( mesh.positions(),
                                          mesh.faceVertices(), gradient );

        auto g = gradient;
        auto u = velocity;
        Real gg = 0, gu = 0;
        Kokkos::parallel_reduce(
            "beatnik_volume_flux_products",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& acc_gg, Real& acc_gu ) {
                for ( int d = 0; d < 3; ++d )
                {
                    acc_gg += g( i, d ) * g( i, d );
                    acc_gu += g( i, d ) * u( i, d );
                }
            },
            gg, gu );

        Real local[2] = { gg, gu };
        Real reduced[2] = { 0, 0 };
        MPI_Allreduce( local, reduced, 2, MPI_DOUBLE, MPI_SUM, mesh.comm() );

        // A no-op on a fully degenerate mesh, exactly as the reference returns
        // the velocity unchanged (mesh_solver.py:292-293).
        if ( !( reduced[0] > Real( 0 ) ) )
            return;

        const Real factor = reduced[1] / reduced[0];
        Kokkos::parallel_for(
            "beatnik_volume_flux_project",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                    u( i, d ) -= factor * g( i, d );
            } );
        Kokkos::fence();
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
     * @note MPI. Each iteration is one `MPI_Allreduce` carrying \f$V\f$ and the
     *       gradient's self inner product together, plus a ghost exchange of the
     *       moved positions before the next iteration's gradient. **The two
     *       reduction ranges are opposite**: \f$V\f$ sums over **owned faces**
     *       (a global sum, so a ghost face would be counted twice — R9) while
     *       \f$G\f$ is assembled from **all local faces** (a per-vertex
     *       assembly) and its inner product then reduces over **owned
     *       vertices**.
     *
     * **T2d — implemented but not exercised by regression test 2.** The gold set
     * was generated with `--no-dynamic-remesh --refine-every 0`, and every call
     * site in `run_adaptive_mesh_bubble.py::main` is inside a refine or remesh
     * branch, so no T2d-era configuration reaches it. It is implemented anyway
     * because the alternative is a stub on the volume-conservation path whose
     * first exercise would be at T4a, where a failure would look like a
     * refinement bug.
     */
    static void projectToVolume( mesh_type& mesh, Real target_volume,
                                 int iterations = 2 )
    {
        const int n_owned = mesh.ownedVertexCount();
        const int n_local = mesh.totalVertexCount();

        vector_view gradient(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_volume_project_gradient" ),
            n_local );

        for ( int it = 0; it < iterations; ++it )
        {
            auto pos = mesh.positions();
            auto faces = mesh.faceVertices();
            SurfaceOperators::volumeGradient( pos, faces, gradient );

            auto g = gradient;
            Real gg = 0;
            Kokkos::parallel_reduce(
                "beatnik_volume_project_denominator",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i, Real& acc ) {
                    for ( int d = 0; d < 3; ++d )
                        acc += g( i, d ) * g( i, d );
                },
                gg );

            // OWNED faces for the volume; ALL local faces for the gradient
            // above. Both in one collective.
            auto owned_faces = Kokkos::subview(
                faces, std::make_pair( 0, mesh.ownedFaceCount() ),
                Kokkos::ALL() );
            const Real local_volume =
                SurfaceOperators::enclosedVolume( pos, owned_faces );

            Real local[2] = { gg, local_volume };
            Real reduced[2] = { 0, 0 };
            MPI_Allreduce( local, reduced, 2, MPI_DOUBLE, MPI_SUM,
                           mesh.comm() );

            if ( !( reduced[0] > Real( 0 ) ) )
                break; // Degenerate mesh; the reference breaks out too.

            const Real step = ( reduced[1] - target_volume ) / reduced[0];
            Kokkos::parallel_for(
                "beatnik_volume_project_move",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                        pos( i, d ) -= step * g( i, d );
                } );
            Kokkos::fence();

            // The next iteration's gradient and volume both read ghost
            // positions, and only owned rows were moved.
            mesh.haloExchange();
        }
    }
};

} // namespace Beatnik

#endif // BEATNIK_VOLUMEPROJECTION_HPP
