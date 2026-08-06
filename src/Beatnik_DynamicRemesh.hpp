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
 * @file Beatnik_DynamicRemesh.hpp
 * @brief Metric-based dynamic remeshing: the sizing field, the
 *        split/collapse/flip/smooth cycle, and the nonlocal-proximity logic.
 *
 * This is the **default** adaptivity path (`--dynamic-remesh`, on by default).
 *
 * THE SIZING FIELD
 * ----------------
 * Port of dynamic_remesh.py::vertex_target_edge_length (lines 196-231)
 *
 * Everything here is driven by a per-vertex target edge length \f$h_v\f$, built
 * in four stages:
 *
 * **1. Curvature term.** Requiring the flat-triangle sagitta
 * \f$\kappa h^2/8\f$ to equal the tolerance \f$\tau\f$ gives
 * \f[
 *   h_f = \sqrt{\frac{8\tau}{\max(\kappa_f,\ \kappa_{\text{floor}})}},
 *   \qquad
 *   \kappa_{\text{floor}} = \max\!\Big(\frac{8\tau}{h_{\max}^2},\ 10^{-12}\Big).
 * \f]
 * The floor is exactly the curvature at which the formula would return
 * \f$h_{\max}\f$, so a flat region asks for \f$h_{\max}\f$ rather than
 * infinity. Then clamped to \f$[h_{\min}, h_{\max}]\f$.
 *
 * **2. Proximity term** (optional, `--remesh-proximity`). Where two
 * *nonlocal* pieces of sheet approach within the activation distance, the
 * target drops to a fraction of the gap:
 * \f$h_f \leftarrow \min(h_f,\ f_p\, d_f)\f$, re-clamped. See
 * `nonlocalFaceCentroidDistance` for what "nonlocal" means and why it is hard.
 *
 * **3. Face-to-vertex.** \f$h_v = \min_{f\ni v} h_f\f$, initialized to
 * \f$h_{\max}\f$ — a **minimum**, so the finest requirement at a vertex wins.
 *
 * **4. Gradation.** \f$h_v\f$ is smoothed so adjacent targets differ by at most
 * `--remesh-target-gradation-factor` (1.35), iterated up to
 * `--remesh-target-gradation-iters` (8) times. Without it a locally tiny
 * target produces a refinement cascade with an abrupt size jump around it.
 *
 * THE REMESH CYCLE
 * ----------------
 * Port of dynamic_remesh.py::dynamic_remesh_arrays (lines 118-193)
 *
 * Per pass (`--remesh-passes`, default 1):
 *   1. surgical proximity splits (optional)
 *   2. recompute sizing; **split** edges longer than
 *      `split_factor * target`
 *   3. recompute sizing; **collapse** edges shorter than
 *      `collapse_factor * target`
 *   4. if anything changed, or the worst quality is below `min_quality`:
 *      **flip** for quality, then **tangentially smooth**
 *
 * The sizing field is recomputed between split and collapse because splitting
 * changes the curvature estimate. Skipping that recomputation lets the collapse
 * pass undo the split pass.
 *
 * MATERIAL EXCLUSION — THE SUBTLE PART
 * ------------------------------------
 * Proximity refinement asks "are two pieces of sheet about to touch?". On a
 * tightly rolled spiral, *every* face is geometrically close to some other
 * face — its own immediate neighborhood, one turn of the spiral away, and
 * genuinely approaching sheets all look the same to a distance query. Two
 * exclusions separate them:
 *
 * - **Topological** (`--remesh-proximity-exclusion-rings`, default 3): ignore
 *   faces within N vertex-neighbor rings on the surface. Handles "its own
 *   neighborhood" but nothing else — a spiral turn is topologically far away.
 * - **Material** (`--remesh-proximity-material-exclusion-*`): ignore faces
 *   whose *carried Lagrangian coordinates* are within a radius. Two faces close
 *   in material coordinate came from the same patch of the initial sheet, so
 *   their present proximity is a fold at the mesh scale, not an approach.
 *   This is what the `remesh_material_position` field in the checkpoint exists
 *   for — see `SurfaceState::materialPosition`.
 *
 * Both radii default to multiples of the **initial** minimum edge length
 * (activation \f$6 h^0_{\min}\f$, material exclusion \f$4 h^0_{\min}\f$), which
 * is why `initial_min_edge` is carried in every checkpoint.
 */

#ifndef BEATNIK_DYNAMICREMESH_HPP
#define BEATNIK_DYNAMICREMESH_HPP

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What one dynamic-remesh call did.
 *
 * Port of dynamic_remesh.py::DynamicRemeshDiagnostics (lines 49-62)
 */
struct RemeshDiagnostics
{
    int old_vertices = 0;
    int old_faces = 0;
    int new_vertices = 0;
    int new_faces = 0;
    int splits = 0;
    int collapses = 0;
    int flips = 0;
    int smooth_steps = 0;
    Real min_quality_before = 0.0;
    Real min_quality_after = 0.0;
    Real max_sagitta_before = 0.0;
    Real max_sagitta_after = 0.0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Metric-based dynamic surface remeshing.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class DynamicRemesh
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using edge_view = Kokkos::View<LocalIndex* [2], device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    /// @param params Remeshing controls.
    explicit DynamicRemesh( const RemeshParams& params )
        : _params( params )
    {
    }

    /// Controls in force. The driver swaps in the "tight" set past
    /// `--remesh-tight-after`.
    const RemeshParams& params() const { return _params; }
    void setParams( const RemeshParams& params ) { _params = params; }

    //-----------------------------------------------------------------------//
    // Sizing field
    //-----------------------------------------------------------------------//

    /**
     * @brief Per-face curvature used for sizing.
     *
     * Port of dynamic_remesh.py::face_curvature_for_sizing (lines 661-673),
     * ::normal_variation_curvature (lines 639-655) and
     * ::cotangent_face_curvature (lines 674-678)
     *
     * The larger of the dihedral-angle estimate and the cotangent estimate, so
     * a feature that either one resolves triggers refinement. Units 1/length.
     */
    void faceCurvatureForSizing( const mesh_type& mesh,
                                 scalar_view& curvature ) const
    {
        (void)mesh;
        (void)curvature;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "faceCurvatureForSizing" );
    }

    /**
     * @brief The per-vertex target edge length.
     *
     * Port of dynamic_remesh.py::vertex_target_edge_length (lines 196-231)
     *
     * Stages 1-4 of the file header. Units of length.
     *
     * @param mesh   Current surface.
     * @param state  Supplies the material position for the proximity exclusion.
     * @param[out] target `(Nv,)` target edge length, clamped to
     *             \f$[h_{\min}, h_{\max}]\f$.
     *
     * @note MPI. The gradation sweep (stage 4) is a Jacobi-like iteration over
     *       edges and needs a ghost exchange of `target` between sweeps, or the
     *       gradation constraint is not enforced across rank boundaries and a
     *       size jump appears exactly there.
     */
    void vertexTargetEdgeLength( const mesh_type& mesh, const state_type& state,
                                 scalar_view& target ) const
    {
        (void)mesh;
        (void)state;
        (void)target;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "vertexTargetEdgeLength" );
    }

    /**
     * @brief Cap adjacent target-size ratios.
     *
     * Port of dynamic_remesh.py::graded_vertex_target_edge_length
     * (lines 736-767)
     *
     * Repeatedly enforce \f$h_i \le \gamma h_j\f$ across every edge, with
     * \f$\gamma\f$ = `--remesh-target-gradation-factor`. Monotone decreasing
     * and bounded below by \f$h_{\min}\f$, so it terminates; the iteration cap
     * is a safety net, not the usual exit.
     *
     * A no-op when \f$\gamma \le 1\f$ or the iteration count is zero.
     */
    void gradeTargetEdgeLength( const mesh_type& mesh,
                                scalar_view& target ) const
    {
        (void)mesh;
        (void)target;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "gradeTargetEdgeLength" );
    }

    //-----------------------------------------------------------------------//
    // Nonlocal proximity
    //-----------------------------------------------------------------------//

    /**
     * @brief Distance from each face centroid to the nearest *nonlocal* face.
     *
     * Port of dynamic_remesh.py::nonlocal_face_centroid_distance
     * (lines 977-1039), with the exclusion helpers ::face_vertex_exclusion_rings
     * (lines 1054-1095) and ::_material_faces_are_local (lines 1042-1051)
     *
     * A centroid-to-centroid nearest-neighbor query, skipping any face that is
     * excluded either topologically (within `proximity_exclusion_rings`
     * vertex-neighbor rings) or materially (carried material centroids within
     * `proximity_material_exclusion_radius`). The result is
     * \f$+\infty\f$ for a face with no admissible partner.
     *
     * The reference grows `k` in the k-nearest query — 16, 32, 64, ... — until
     * every face has found an admissible neighbor, because the excluded set can
     * be large enough to swallow the whole first query. That growth loop is
     * part of the algorithm, not an optimization: a fixed `k` silently reports
     * \f$+\infty\f$ for faces deep in a refined patch, which reads as "nothing
     * nearby" and disables the very refinement the query exists to drive.
     *
     * Cheap but approximate — it compares *centroids*, so it under-reports the
     * gap between two large triangles by up to their circumradii.
     * `nonlocalFaceProximityPairs` is the exact version.
     *
     * @note MPI. **This is the one genuinely global spatial query in the
     *       solver.** Two approaching sheets can be on any two ranks; there is
     *       no ghost depth that makes it local. A distributed spatial
     *       structure (an ArborX distributed tree, or Canopy's tree reused) is
     *       required. Restricting the search to on-rank faces gives an answer
     *       that changes with the partition — the failure mode is that a
     *       self-contact is detected on 1 rank and missed on 8.
     */
    void nonlocalFaceCentroidDistance( const mesh_type& mesh,
                                       const state_type& state,
                                       scalar_view& distance ) const
    {
        (void)mesh;
        (void)state;
        (void)distance;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh",
                                 "nonlocalFaceCentroidDistance" );
    }

    /**
     * @brief Exact close nonlocal face pairs, by triangle-triangle distance.
     *
     * Port of dynamic_remesh.py::nonlocal_face_proximity_pairs
     * (lines 783-870), on top of ::triangle_triangle_distance (lines 871-889),
     * ::point_triangle_distance (lines 890-939) and
     * ::segment_segment_distance (lines 940-976)
     *
     * Candidates are found by a centroid query with a radius padded by both
     * faces' circumradii, giving a **lower bound** on the true gap; the exact
     * triangle-triangle distance is then evaluated only for the best
     * `--remesh-surgical-proximity-max-pairs` candidates. The exact distance is
     * the minimum over the nine edge-edge segment distances and the six
     * vertex-face point distances, which is what makes it expensive and why it
     * is not the default.
     *
     * Used by the surgical proximity splits and, with `--exact-gap-diagnostics`,
     * by the progress diagnostics.
     */
    void nonlocalFaceProximityPairs( const mesh_type& mesh,
                                     const state_type& state,
                                     Real activation_distance,
                                     edge_view& pairs,
                                     scalar_view& gaps ) const
    {
        (void)mesh;
        (void)state;
        (void)activation_distance;
        (void)pairs;
        (void)gaps;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "nonlocalFaceProximityPairs" );
    }

    //-----------------------------------------------------------------------//
    // The cycle
    //-----------------------------------------------------------------------//

    /**
     * @brief Split every edge longer than `split_factor * target`.
     *
     * Port of dynamic_remesh.py::split_long_edges (lines 234-260)
     *
     * The per-edge target is the **minimum** of its two endpoints' targets, so
     * an edge is split if either end wants finer resolution. Capped at
     * `max_splits_per_pass`, longest-first.
     */
    int splitLongEdges( mesh_type& mesh, state_type& state,
                        const scalar_view& target ) const
    {
        (void)mesh;
        (void)state;
        (void)target;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "splitLongEdges" );
    }

    /**
     * @brief Collapse every edge shorter than `collapse_factor * target`.
     *
     * Port of dynamic_remesh.py::collapse_short_edges (lines 361-407)
     *
     * Subject to the link condition and the geometric safety test — see
     * `SurfaceMesh::collapseEdges`. Capped at `max_collapses_per_pass`,
     * shortest-first. Note `--remesh-tight-collapse-factor` defaults to 0,
     * i.e. the tight parameter set **disables collapse entirely**: once the
     * roll-up is tight, coarsening is more likely to destroy a resolved feature
     * than to save work.
     */
    int collapseShortEdges( mesh_type& mesh, state_type& state,
                            const scalar_view& target ) const
    {
        (void)mesh;
        (void)state;
        (void)target;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "collapseShortEdges" );
    }

    /**
     * @brief Flip interior edges that improve the worst incident quality.
     *
     * Port of dynamic_remesh.py::flip_edges_for_quality (lines 408-458)
     *
     * A flip is accepted when
     * \f$\min(q_{\text{new}}) > \min(q_{\text{old}})\,(1+g)\f$ with \f$g\f$ =
     * `--remesh-flip-min-gain`. The strict margin prevents a two-flip cycle
     * that oscillates forever at equal quality.
     *
     * This flips for **quality**, never for **valence** — which is exactly the
     * gap `Beatnik_MeshQuality.hpp::isotropicCleanup` exists to fill.
     */
    int flipEdgesForQuality( mesh_type& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "flipEdgesForQuality" );
    }

    /**
     * @brief Tangential vertex relaxation.
     *
     * Port of dynamic_remesh.py::tangential_smooth_vertices (lines 459-491)
     *
     * \f$x_v \leftarrow x_v + \lambda\,\Pi_{\tan}\big(\mathrm{centroid}(N(v))
     * - x_v\big)\f$, with the normal component projected out so the **shape is
     * preserved** and only the parameterization moves. Removing the projection
     * turns this into Laplacian smoothing of the interface itself, which
     * shrinks the bubble — a subtle and destructive bug, because it looks like
     * excessive numerical dissipation rather than a geometry error.
     */
    int tangentialSmooth( mesh_type& mesh ) const
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "tangentialSmooth" );
    }

    /**
     * @brief Directly split faces in exact nonlocal close-pair regions.
     *
     * Port of dynamic_remesh.py::split_surgical_proximity_edges
     * (lines 299-360)
     *
     * Runs **before** the ordinary sizing-driven cycle. The centroid-based
     * proximity term in the sizing field is too coarse to resolve an imminent
     * contact between two large triangles; this pass finds the exact closest
     * pairs and splits those faces directly, to a target of
     * `--remesh-surgical-proximity-fraction` of the true gap, floored at
     * `--remesh-surgical-proximity-h-min` (or `h_min` when that is <= 0).
     *
     * Off by default (`--remesh-surgical-proximity`).
     */
    int splitSurgicalProximityEdges( mesh_type& mesh, state_type& state ) const
    {
        (void)mesh;
        (void)state;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh",
                                 "splitSurgicalProximityEdges" );
    }

    /**
     * @brief Run the full remesh cycle and transfer the solution fields.
     *
     * Port of dynamic_remesh.py::dynamic_remesh_arrays (lines 118-193) and
     * run_adaptive_mesh_bubble.py::dynamic_remesh_state_with_material
     * (lines 1080-1110)
     *
     * The fields carried through are the potential (or sheet vector) **and**
     * the material position. The reference passes them as a `fields` dict
     * precisely so nothing can be forgotten; here they live on the state and
     * `SurfaceState::remap` moves them together.
     *
     * The remeshed state is built with `reference_face_area=None` and
     * `reference_face_curvature=None`, re-basing the AMR change indicators —
     * see `Beatnik_AdaptiveMesh.hpp`.
     *
     * @note MPI. Split, collapse and flip are all collective, and a rank
     *       boundary must make the same decision on both sides. The
     *       conservative pattern — which is what to write first — is to defer
     *       every edit touching a boundary edge to a second phase in which the
     *       owner decides and broadcasts. Optimizing that comes later; getting
     *       it wrong tears the surface.
     */
    RemeshDiagnostics remesh( mesh_type& mesh, state_type& state )
    {
        (void)mesh;
        (void)state;
        BEATNIK_NOT_IMPLEMENTED( "DynamicRemesh", "remesh" );
    }

  private:
    RemeshParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_DYNAMICREMESH_HPP
