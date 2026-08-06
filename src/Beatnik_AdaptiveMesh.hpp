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
 * @file Beatnik_AdaptiveMesh.hpp
 * @brief Indicator-driven red-green AMR: the refine/coarsen indicators, the
 *        mark-set closure, and the reference-state bookkeeping they depend on.
 *
 * WHEN THIS PATH RUNS
 * -------------------
 * Only under `--no-dynamic-remesh`. The default adaptivity is the metric-based
 * remesher in `Beatnik_DynamicRemesh.hpp` (`--dynamic-remesh`, on by default),
 * and the driver runs one or the other, never both
 * (`run_adaptive_mesh_bubble.py:1424`). This path is retained because it is
 * simpler, deterministic, and the right thing to validate first.
 *
 * THE REFERENCE STATE — AND WHY A RESTART CHANGES BEHAVIOR
 * --------------------------------------------------------
 * Two of the three indicators are **change** indicators: they compare a face's
 * current area and curvature against the values that face had when its
 * reference was last set. A `TriangleSurfaceState` stores
 * `reference_face_area` and `reference_face_curvature`
 * (`mesh.py:38-49`), initialized to the current values whenever they are not
 * supplied.
 *
 * The reference is reset (i.e. re-based to "now") at these points:
 *   - initial mesh construction;
 *   - after refinement, for the newly created faces;
 *   - after `improve_mesh_connectivity_by_edge_flips` with
 *     `reset_reference=True` (`mesh_solver.py:1704-1772`);
 *   - after `improve_mesh_quality_tangential` with `reset_reference=True`
 *     (lines 1775-1831);
 *   - after **every** dynamic remesh, which passes `reference_face_area=None`
 *     (`run_adaptive_mesh_bubble.py:1099-1100, 1108`);
 *   - **on every restart**, because the checkpoint does not carry them
 *     (`run_adaptive_mesh_bubble.py:993-1033`).
 *
 * That last one is the consequential one. A run restarted at step N does not
 * continue the trajectory of an uninterrupted run: its area- and
 * curvature-change indicators are measured against the step-N geometry rather
 * than against whatever the reference was before, so the *next* refinement
 * decision differs. The divergence is real, not numerical. This is isolated in
 * `Beatnik_Restart.hpp` and recorded as risk R3 in `tasks/framework.md`.
 */

#ifndef BEATNIK_ADAPTIVEMESH_HPP
#define BEATNIK_ADAPTIVEMESH_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What one refinement pass did.
 *
 * Port of mesh.py::RefinementDiagnostics (lines 69-78)
 */
struct RefinementDiagnostics
{
    int old_vertices = 0;
    int new_vertices = 0;
    int old_faces = 0;
    int new_faces = 0;
    /// Faces marked red after closure. Zero means the pass was a no-op, which
    /// the driver uses to skip the follow-on quality repair entirely.
    int marked_faces = 0;
    int split_edges = 0;
    Real max_area_change = 0.0;
    Real max_curvature = 0.0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Indicator-driven red-green adaptive refinement.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class AdaptiveMesh
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using mark_view = Kokkos::View<char*, device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    /// @param params AMR thresholds and caps.
    explicit AdaptiveMesh( const AmrParams& params )
        : _params( params )
    {
    }

    //-----------------------------------------------------------------------//
    // Indicators
    //-----------------------------------------------------------------------//

    /**
     * @brief Relative area change of each face since its reference.
     *
     * Port of mesh.py::area_change_indicator (lines 215-218)
     *
     * \f[
     *   \eta^{A}_f = \left| \frac{A_f}{A_f^{\text{ref}}} - 1 \right|
     * \f]
     * with \f$A_f^{\text{ref}}\f$ floored at 1e-300. Dimensionless, so
     * `--area-threshold` (default 0.16) means "refine a face that has grown or
     * shrunk by 16% since its reference was set". Note it is **symmetric**: a
     * face that has *shrunk* by 16% is marked too, which is deliberate — a
     * shrinking face signals a compressing region.
     */
    void areaChangeIndicator( const mesh_type& mesh,
                              const scalar_view& reference_area,
                              scalar_view& indicator ) const
    {
        (void)mesh;
        (void)reference_area;
        (void)indicator;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "areaChangeIndicator" );
    }

    /**
     * @brief Relative curvature change of each face since its reference.
     *
     * Port of mesh.py::curvature_change_indicator (lines 221-224), on top of
     * ::face_curvature_indicator (lines 172-174) and
     * ::cotangent_vertex_curvature (lines 150-169)
     *
     * The per-vertex curvature magnitude is the *cheap, robust* estimator — not
     * a true mean curvature:
     * \f[
     *   \kappa_v = \frac{\big\|\,\mathrm{mean}_{j\in N(v)}(x_j - x_v)\,\big\|}
     *                   {\mathrm{mean}_{j\in N(v)}\|x_j-x_v\|^2}
     * \f]
     * i.e. the umbrella displacement over the mean squared edge length, units
     * 1/length. The per-face value is the max over its three vertices, and the
     * indicator is
     * \f[
     *   \eta^{\kappa}_f = \left|\frac{\kappa_f}{\kappa_f^{\text{ref}}}-1\right| .
     * \f]
     *
     * The reference explicitly notes this "is intentionally robust and cheap
     * rather than a high-order mean-curvature estimator" (`mesh.py:154-155`).
     * Substituting the cotangent mean curvature would change every refinement
     * decision, so do not "improve" it — `mean_curvature_normal` exists
     * separately for the places where the true quantity is needed.
     */
    void curvatureChangeIndicator( const mesh_type& mesh,
                                   const scalar_view& reference_curvature,
                                   scalar_view& indicator ) const
    {
        (void)mesh;
        (void)reference_curvature;
        (void)indicator;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "curvatureChangeIndicator" );
    }

    /**
     * @brief Flat-triangle chord ("sagitta") error of each face.
     *
     * Port of mesh.py::curvature_resolution_indicator (lines 203-212) and
     * ::normal_variation_curvature_indicator (lines 177-200)
     *
     * Unlike the two above this is an **absolute** resolution measure, not a
     * change measure, so it does not depend on any reference state and is
     * unaffected by a restart.
     *
     * Curvature is estimated from the dihedral angle across each edge:
     * \f[
     *   \kappa_e = \frac{\arccos(\hat n_{f_0}\cdot\hat n_{f_1})}{\ell_e},
     *   \qquad \kappa_f = \max_{e\in f}\kappa_e ,
     * \f]
     * and the sagitta of a chord of length \f$h\f$ on a circle of curvature
     * \f$\kappa\f$ is \f$\approx \kappa h^2/8\f$:
     * \f[
     *   \eta^{\text{sag}}_f = \tfrac18\,\kappa_f\,h_{\max,f}^2 .
     * \f]
     * Units of length, so `--curvature-resolution-threshold` is a physical
     * distance — how far the flat triangle is allowed to depart from the
     * surface it represents. Disabled by default (threshold 0).
     *
     * @note MPI. The dihedral angle needs both faces of every edge, so this
     *       reads ghost faces. A boundary edge with only one incident face is
     *       skipped, which on a distributed closed surface would silently
     *       under-refine along partition boundaries if the ghost layer were
     *       missing.
     */
    void curvatureResolutionIndicator( const mesh_type& mesh,
                                       scalar_view& indicator ) const
    {
        (void)mesh;
        (void)indicator;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh",
                                 "curvatureResolutionIndicator" );
    }

    //-----------------------------------------------------------------------//
    // Mark selection and closure
    //-----------------------------------------------------------------------//

    /**
     * @brief Seed marks from the three indicators.
     *
     * Port of mesh_solver.py::refine_potential_mesh_state (lines 1388-1408)
     *
     * A face is seeded if **any** indicator exceeds its threshold:
     * \f[
     *   m_f = (\eta^A_f > \tau_A) \;\lor\; (\eta^\kappa_f > \tau_\kappa)
     *         \;\lor\; (\eta^{\text{sag}}_f > \tau_{\text{sag}}) ,
     * \f]
     * with the third clause present only when \f$\tau_{\text{sag}} > 0\f$.
     *
     * Faces whose shortest edge is already below `--min-refine-edge` are then
     * dropped (`mesh_solver.py::_drop_faces_below_min_edge`, lines 1626-1635) —
     * a hard floor that stops refinement chasing a feature it has already
     * resolved to the intended scale.
     *
     * A per-face **score** accompanies the marks, used to rank them when a cap
     * binds:
     * \f[
     *   s_f = \max\!\Big(\frac{\eta^A_f}{\tau_A},\;
     *                    \frac{\eta^\kappa_f}{\tau_\kappa},\;
     *                    \frac{\eta^{\text{sag}}_f}{\tau_{\text{sag}}}\Big),
     * \f]
     * i.e. each indicator normalized by its own threshold so they are
     * comparable. The third term is zero when the sagitta criterion is off.
     */
    void markFaces( const mesh_type& mesh, const scalar_view& reference_area,
                    const scalar_view& reference_curvature, mark_view& marked,
                    scalar_view& score ) const
    {
        (void)mesh;
        (void)reference_area;
        (void)reference_curvature;
        (void)marked;
        (void)score;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "markFaces" );
    }

    /**
     * @brief Cap the seed marks at `--max-refine-fraction` of all faces.
     *
     * Port of mesh_solver.py::_limit_marked_fraction (lines 1434-1451)
     *
     * Budget \f$= \max(1, \lceil f\,N_f\rceil)\f$; if more faces are marked
     * than that, keep the `budget` highest-scoring ones and drop the rest.
     * Applied to the **seeds**, before closure — so the closure can and does
     * exceed the fraction.
     *
     * @note MPI. The fraction is of the *global* face count and the ranking is
     *       global, so a naive per-rank cap gives a different (and rank-count
     *       dependent) mark set. A global ranking needs either a distributed
     *       sort or a parallel threshold search on the score.
     */
    void limitMarkedFraction( mark_view& marked, const scalar_view& score,
                              Real max_fraction ) const
    {
        (void)marked;
        (void)score;
        (void)max_fraction;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "limitMarkedFraction" );
    }

    /**
     * @brief Grow the marked set by whole face-neighbor rings.
     *
     * Port of mesh_solver.py::_expand_marked_face_rings (lines 1515-1530)
     *
     * `--refine-neighbor-rings` (default 1) breadth-first expansions across
     * edge-adjacency. Buffering the refined region keeps the resolution jump
     * away from the feature that triggered it, at the cost of roughly
     * \f$3\times\f$ the marked faces per ring.
     *
     * @note MPI. The expansion crosses rank boundaries and must propagate —
     *       see `Comm::reconcileRefinementMarks`.
     */
    void expandMarkedRings( const mesh_type& mesh, mark_view& marked,
                            int rings ) const
    {
        (void)mesh;
        (void)marked;
        (void)rings;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "expandMarkedRings" );
    }

    /**
     * @brief Promote poor green transitions to full red refinement.
     *
     * Port of mesh_solver.py::_balance_red_green_refinement
     * (lines 1543-1580), with the predicted-quality helper
     * ::_single_green_split_quality (lines 1606-1623)
     *
     * A red face bisects all three of its edges. An unmarked neighbor sharing
     * split edges must be green-split to stay conforming, and green splits
     * produce poorer triangles than red ones. This pass, run to a fixed point:
     *
     *   - **Always** promotes a face sharing \f$\ge 2\f$ split edges. A
     *     two-edge green split is a genuinely bad element and there is no
     *     threshold worth tuning.
     *   - Promotes a **one-edge** green split if the worse of its two children
     *     would have quality below
     *     \f$\max(\tau_{\text{floor}},\ f_q\, q_{\text{parent}})\f$, with
     *     \f$\tau_{\text{floor}}\f$ = `--transition-quality-floor` (0.18) and
     *     \f$f_q\f$ = `--transition-quality-fraction` (0.45). The absolute
     *     floor catches faces that are already poor; the relative fraction
     *     catches good faces about to be badly split.
     *
     * Promotion splits more edges, which can force further promotions — hence
     * the `while changed` loop. It terminates because promotion is monotone
     * and bounded by the face count.
     *
     * @note MPI. This is the fixed-point iteration `Comm::reconcileRefinementMarks`
     *       exists for. A cascade can traverse several ranks, so a single
     *       exchange per sweep is required and the loop's termination test must
     *       be a global `MPI_Allreduce(MPI_LOR)`.
     */
    void balanceRedGreen( const mesh_type& mesh, mark_view& marked ) const
    {
        (void)mesh;
        (void)marked;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "balanceRedGreen" );
    }

    /**
     * @brief Projected face count after conforming red-green refinement.
     *
     * Port of mesh.py::projected_red_green_face_count (lines 251-271)
     *
     * A marked face contributes 4 children; an unmarked face with \f$k\f$ split
     * edges contributes \f$\max(k+1, 1)\f$ — so 1, 2, 3 or 4 for
     * \f$k = 0,1,2,3\f$. Evaluated *before* refining, so the `--max-faces` cap
     * can be enforced without doing the work.
     */
    GlobalIndex projectedFaceCount( const mesh_type& mesh,
                                    const mark_view& marked ) const
    {
        (void)mesh;
        (void)marked;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "projectedFaceCount" );
    }

    /**
     * @brief Full mark selection: seed, cap, expand, balance, obey `max_faces`.
     *
     * Port of mesh_solver.py::_quality_preserving_refinement_marks
     * (lines 1454-1512)
     *
     * If the closed mark set already fits under `--max-faces`, it is used.
     * Otherwise the reference falls back to a **greedy** accept loop: walk the
     * seeds in descending score, tentatively add each, re-close, and keep it
     * only if the projected count still fits. That is \f$O(N_{\text{seeds}})\f$
     * closures, each itself a fixed-point iteration — quadratic in the marked
     * set and by far the most expensive thing in the AMR path.
     *
     * **This greedy loop does not parallelize as written**: each trial depends
     * on the accepted set from the previous trial. A distributed implementation
     * needs a different algorithm with the same *intent* (respect the cap,
     * prefer high scores, keep the closure valid) and will not reproduce the
     * serial mark set exactly. That is acceptable — the cap is a resource
     * limit, not physics — but it means a capped run is not expected to match
     * the Python face-for-face. Recorded as risk R4 in `tasks/framework.md`.
     */
    void selectMarks( const mesh_type& mesh, const scalar_view& reference_area,
                      const scalar_view& reference_curvature,
                      mark_view& marked ) const
    {
        (void)mesh;
        (void)reference_area;
        (void)reference_curvature;
        (void)marked;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "selectMarks" );
    }

    //-----------------------------------------------------------------------//
    // Driver entry point
    //-----------------------------------------------------------------------//

    /**
     * @brief Select marks, refine, and transfer the solution fields.
     *
     * Port of mesh_solver.py::refine_potential_mesh_state (lines 1374-1431)
     * and ::refine_mesh_state (lines 1314-1371)
     *
     * The two Python functions are identical apart from which field rides
     * through; here the state carries its own model tag, so there is one
     * function.
     *
     * The refined state is constructed with `reference_face_area=None` and
     * `reference_face_curvature=None`, so **the reference is re-based to the
     * post-refinement geometry** — see the file header.
     *
     * @param[in,out] mesh  Surface, refined in place.
     * @param[in,out] state Solution, remapped through the refinement.
     * @return What the pass did. `marked_faces == 0` means nothing happened
     *         and the caller should skip the follow-on repair.
     */
    RefinementDiagnostics refine( mesh_type& mesh, state_type& state )
    {
        (void)mesh;
        (void)state;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "refine" );
    }

    /**
     * @brief Re-base the reference area and curvature to the current geometry.
     *
     * Port of mesh_solver.py::_state_with_faces (lines 1676-1701) — the
     * `reset_reference=True` branch
     *
     * Called after any operation that legitimately changes face areas or
     * curvature without that change signalling a need to refine: refinement
     * itself, quality flips, tangential relaxation, dynamic remeshing, and
     * restart. Failing to re-base leaves the change indicators reading a
     * one-off geometric edit as physics and refining in response.
     */
    void resetReferenceState( const mesh_type& mesh, scalar_view& reference_area,
                              scalar_view& reference_curvature ) const
    {
        (void)mesh;
        (void)reference_area;
        (void)reference_curvature;
        BEATNIK_NOT_IMPLEMENTED( "AdaptiveMesh", "resetReferenceState" );
    }

  private:
    AmrParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_ADAPTIVEMESH_HPP
