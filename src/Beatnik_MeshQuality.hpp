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
 * @file Beatnik_MeshQuality.hpp
 * @brief Valence-equalizing cleanup: the regularization the quality-driven
 *        remesher does not provide.
 *
 * THE PROBLEM THIS SOLVES
 * -----------------------
 * Port of mesh_quality.py::isotropic_cleanup (lines 146-167)
 *
 * `DynamicRemesh::flipEdgesForQuality` flips for **triangle quality** and never
 * for **valence**. During a tightening roll-up the curvature sizing field
 * refines hard, and quality-only flips leave behind a high-valence,
 * small-area tangle — the "curvature sliver". Empirically that tangle kills the
 * run, or drives dt to its floor, **long before** the sheets actually
 * self-contact. So the simulation dies of a meshing artifact rather than
 * reaching the physics it was set up to study.
 *
 * The fix, run after every remesh (`--isotropic-cleanup`, on by default), is
 * two passes that regularize connectivity and element size **without changing
 * the surface shape**:
 *   1. valence-equalizing flips, driving vertex valences toward 6;
 *   2. tangential relaxation, equalizing triangle areas.
 *
 * Both preserve the vertex **count**, and the flips change only connectivity,
 * so per-vertex fields stay valid — the caller rebuilds the state on the
 * returned `(vertices, faces)` with the same potential
 * (`run_adaptive_mesh_bubble.py:1491-1504`). No field remap is needed, which is
 * why this is a separate, cheaper operation than a remesh.
 */

#ifndef BEATNIK_MESHQUALITY_HPP
#define BEATNIK_MESHQUALITY_HPP

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Connectivity and parameterization regularization.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class MeshQuality
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;

    /// @param params Cleanup pass counts and relaxation weight.
    explicit MeshQuality( const CleanupParams& params )
        : _params( params )
    {
    }

    /**
     * @brief Flip edges to drive vertex valences toward 6 (Botsch-Kobbelt).
     *
     * Port of mesh_quality.py::_valence_equalizing_flips (lines 44-87)
     *
     * For the shared edge \f$(a,b)\f$ of two triangles with opposite vertices
     * \f$c\f$ and \f$d\f$, flipping to \f$(c,d)\f$ moves one unit of valence
     * from \f$a,b\f$ to \f$c,d\f$. Accept the flip iff it strictly reduces the
     * total deviation from valence 6:
     * \f[
     *   \sum |{\rm val}-6| \;\to\;
     *     |v_a-7|+|v_b-7|+|v_c-5|+|v_d-5| \;<\;
     *     |v_a-6|+|v_b-6|+|v_c-6|+|v_d-6| .
     * \f]
     * (The pre-flip valences \f$v\f$ are used on both sides; the post-flip
     * valences are \f$v_a-1, v_b-1, v_c+1, v_d+1\f$, which is where the 7s and
     * 5s come from.)
     *
     * Rejected if the flip would:
     *   - invert a face normal (\f$\hat n_{\text{new}}\cdot\hat n_{\text{old}} <
     *     0.2\f$ — a generous margin, not merely a sign test, because a
     *     near-inversion is nearly as bad);
     *   - create an already-existing edge, i.e. make the mesh non-manifold;
     *   - produce a triangle with quality below 0.05.
     *
     * A `touched` set keeps the flips within one pass independent, so the
     * valences the pass is deciding against stay consistent. That also makes
     * the pass order-dependent — a parallel implementation will not reproduce
     * the serial flip set, though it should reach a comparable valence
     * histogram. Recorded as risk R7 in `tasks/framework.md`.
     *
     * @param mesh   Surface, connectivity edited in place.
     * @param passes Sweeps; exits early when a sweep flips nothing.
     * @return Total flips applied.
     */
    int valenceEqualizingFlips( mesh_type& mesh, int passes ) const
    {
        (void)mesh;
        (void)passes;
        BEATNIK_NOT_IMPLEMENTED( "MeshQuality", "valenceEqualizingFlips" );
    }

    /**
     * @brief Move vertices toward their neighbor centroid, tangentially.
     *
     * Port of mesh_quality.py::_tangential_relaxation (lines 90-116)
     *
     * \f[
     *   x_v \leftarrow x_v + w\,\Big[\,\bar x_v - x_v
     *     - \big((\bar x_v - x_v)\cdot\hat n_v\big)\hat n_v \Big],
     *   \qquad \bar x_v = \frac{1}{|N(v)|}\sum_{j\in N(v)} x_j .
     * \f]
     *
     * The normal projection is the whole point: it equalizes triangle areas
     * while leaving the resolved **shape** untouched. Without it this is
     * Laplacian smoothing of the interface, which shrinks the bubble and reads
     * as excessive numerical dissipation rather than as the geometry bug it is.
     *
     * The vertex normals are recomputed **each pass**, since the surface
     * parameterization has moved.
     *
     * @note MPI. Ghost positions must be refreshed between passes
     *       (`Comm::haloExchangeVertices`) or boundary vertices relax against
     *       stale neighbors and a visible seam develops.
     */
    int tangentialRelaxation( mesh_type& mesh, int passes, Real weight ) const
    {
        (void)mesh;
        (void)passes;
        (void)weight;
        BEATNIK_NOT_IMPLEMENTED( "MeshQuality", "tangentialRelaxation" );
    }

    /**
     * @brief The sliver-clearing post-pass: valence flips then relaxation.
     *
     * Port of mesh_quality.py::isotropic_cleanup (lines 146-167)
     *
     * `--isotropic-cleanup-flips` (3) flip passes, then
     * `--isotropic-cleanup-relax` (2) relaxation passes at
     * `--isotropic-cleanup-weight` (0.4).
     *
     * Vertex count unchanged and shape preserved, so the caller keeps its
     * per-vertex fields as they are. It **does** change face areas and
     * curvature, so the AMR reference state must be re-based afterwards — see
     * `AdaptiveMesh::resetReferenceState`.
     *
     * The driver follows it with the volume projection whenever anything
     * changed (`run_adaptive_mesh_bubble.py:1465-1468, 1514-1516`), because the
     * tangential relaxation is only volume-preserving to first order.
     */
    void isotropicCleanup( mesh_type& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "MeshQuality", "isotropicCleanup" );
    }

    /**
     * @brief Quality-driven edge flips for the indicator-AMR path.
     *
     * Port of mesh_solver.py::improve_mesh_connectivity_by_edge_flips
     * (lines 1704-1772)
     *
     * Distinct from `valenceEqualizingFlips`: this one accepts a flip iff
     * \f$\min(q_{\text{new}}) > \min(q_{\text{old}})(1+g)\f$, the same
     * criterion `DynamicRemesh::flipEdgesForQuality` uses. Called only from the
     * `--no-dynamic-remesh` refinement path, after a refine pass
     * (`--flip-passes`, default 0, i.e. off).
     *
     * The `reset_reference` flag in the Python is threaded oddly — the caller
     * passes `reset_reference=(smooth_iters == 0)` so the reference is re-based
     * by whichever of the flip and smooth passes runs last
     * (`run_adaptive_mesh_bubble.py:1440-1451`). Here the caller re-bases
     * explicitly after both, which is equivalent and easier to reason about.
     */
    int improveConnectivityByFlips( mesh_type& mesh, int max_passes,
                                    Real min_gain ) const
    {
        (void)mesh;
        (void)max_passes;
        (void)min_gain;
        BEATNIK_NOT_IMPLEMENTED( "MeshQuality", "improveConnectivityByFlips" );
    }

    /**
     * @brief Mild tangential relaxation used by the indicator-AMR path.
     *
     * Port of mesh_solver.py::improve_mesh_quality_tangential
     * (lines 1775-1831)
     *
     * The same operator as `tangentialRelaxation`, reached from a different
     * place: after an indicator-driven refine (`--smooth-iters`,
     * `--smooth-relaxation`) and from the periodic `--redistribute-every`
     * sweep.
     *
     * The reference notes it "targets element quality while avoiding deliberate
     * normal smoothing of the interface geometry" — the same tangential
     * constraint, stated for the same reason.
     */
    void improveQualityTangential( mesh_type& mesh, int iterations,
                                   Real relaxation ) const
    {
        (void)mesh;
        (void)iterations;
        (void)relaxation;
        BEATNIK_NOT_IMPLEMENTED( "MeshQuality", "improveQualityTangential" );
    }

  private:
    CleanupParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_MESHQUALITY_HPP
