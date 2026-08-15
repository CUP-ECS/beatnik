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
 * @brief Indicator-driven refinement: the refine/coarsen indicators, the
 *        mark-set closure, and the reference-state bookkeeping they depend on.
 *
 * WHEN THIS PATH RUNS
 * -------------------
 * Only under `--no-dynamic-remesh`. The default adaptivity is the metric-based
 * remesher in `Beatnik_DynamicRemesh.hpp` (`--dynamic-remesh`, on by default),
 * and the driver runs one or the other, never both
 * (`run_adaptive_mesh_bubble.py:1424` versus `:1469-1471`). This path is
 * retained because it is simpler, deterministic, and the right thing to
 * validate first.
 *
 * THE EDIT IS `splitEdges()`, AND THE MASK IS AN EDGE MASK (T4a)
 * --------------------------------------------------------------
 * Beatnik never calls `Tessera::refine()`; `SurfaceMesh::refine` does not
 * exist. Every topological edit here goes through the **Remesh** family, and
 * the reason is fidelity rather than convenience: for the mask "every edge of
 * every marked face", `Tessera::splitEdges()` **is**
 * `mesh.py::refine_marked_faces` — the same midpoints, the same
 * retriangulation by \f$|S_f|\f$, the same absence of a cascade. Tessera's
 * `refine()` is a different algorithm whose closure layer is *transient*, so it
 * would diverge from the Python in face count from round 2 onward and would
 * churn the per-face reference state through every un-close. The whole argument
 * is in `tasks/framework.md`, Phase 4, *The editing-family question —
 * RESOLVED*.
 *
 * So **the mark representation is an edge mask, everywhere**: a host
 * `std::vector<char>` sized `ownedEdgeCount()`, `1` = bisect, which is
 * `Tessera::splitEdges`'s own convention rather than a second one invented on
 * top. Every step of the pipeline below is a rule for constructing or growing
 * that mask, and the face-level notions the Python works in — "this face is
 * red", "this face has two split edges" — appear only as the *derivation*
 * between the two.
 *
 * The one thing carried per face rather than per edge is the mark itself, in
 * the `FaceFieldId::RefineMark` face user field, and it is there because it has
 * to cross ranks: see MARK TRANSLATION below.
 *
 * SHAPE AT DEPTH IS THE MASK'S PROPERTY, NOT `splitEdges()`'s — RISK R12
 * ---------------------------------------------------------------------
 * `splitEdges()` offers **no** triangle-shape guarantee and does not claim one:
 * a \f$|S|=1\f$ median-cut child is an ordinary face on the next call and can
 * be cut again, so the reachable similarity classes are unbounded in the round
 * count. Tessera measured four mask families to depth
 * (`../tessera/tests/test_split_edges_depth.cpp`): a **length-driven** mask
 * ("split the edges longer than the mean") is exactly periodic in the global
 * minimum \f$r/R\f$ with period 3, while a shorter-than-mean or a length-blind
 * mask degrades geometrically to a zero minimum angle with no floor at all.
 *
 * What supplies the bound is that the rule attacks the long edge of a stretched
 * triangle — a coarse relative of Rivara longest-edge bisection, which is
 * self-correcting. **This file's hard constraint follows: an edge enters the
 * mask only because it is an edge of a face that is red.** No other term is
 * ever unioned in. Every rule below — the seed, the ring growth, the two-edge
 * promotion, the one-edge quality promotion — reds a *face* and therefore marks
 * all three of its edges together, which is the "always the red split" family
 * R12 names as bounded. A curvature, vorticity or region-tag term added
 * directly to the edge mask would inherit none of that, and R12 states the mask
 * transform (Rivara promotion to fixpoint) that would restore it.
 *
 * `refine()` therefore reports both of R12's monitoring signals per pass, in
 * `RefinementDiagnostics`: the global minimum \f$r/R\f$ and the global count of
 * faces below \f$r/R = 0.25\f$. The second threshold is deliberately Tessera's
 * own `kTail` value, so the two diagnostics are directly comparable. **Healthy**
 * is a minimum that cycles and a sub-`0.25` count that returns to zero between
 * dips; a monotone decline at a constant factor per round is the shape problem.
 *
 * MARK TRANSLATION ACROSS RANKS — ROUTE (a)
 * ------------------------------------------
 * An owned edge may be incident on a face owned by another rank, and
 * `splitEdges()` takes its verdict from the **edge owner**. So a face-level
 * decision has to reach that rank. Of the two routes `tasks/framework.md`
 * offered, this file takes **(a)**: the per-face verdict is computed on
 * **owned** faces, written into the `RefineMark` face user field,
 * `haloExchange()`d, and every rank then evaluates its own owned edges from
 * locally-resident faces. One exchange, no duplicated arithmetic, and no
 * dependence on the one-ring of a ghost face's corner being complete — which
 * route (b), evaluating the indicators on ghost faces, would have needed.
 *
 * That is also what makes the balance fixpoint cheap: each round re-exchanges
 * the mark and terminates on one `MPI_Allreduce(MPI_LOR)`, with a hard round
 * cap that **throws** rather than proceeding with a partial mark set. Tessera
 * needs no reconciled mask (the edge coordinator routes the owner's verdict
 * itself), so what is agreed here is Beatnik's own closure and nothing else —
 * which is why `Comm::reconcileRefinementMarks` was deleted at T4a.
 *
 * The precondition route (a) rests on is that **both incident faces of every
 * owned edge, and all three edges of every owned face, are locally resident at
 * halo depth 2**. That is checked, not assumed: `refine()` throws naming the
 * offending count if it does not hold.
 *
 * THE REFERENCE STATE — AND WHY A RESTART CHANGES BEHAVIOR
 * --------------------------------------------------------
 * Two of the three indicators are **change** indicators: they compare a face's
 * current area and curvature against the values that face had when its
 * reference was last set. A `TriangleSurfaceState` stores
 * `reference_face_area` and `reference_face_curvature`
 * (`mesh.py:38-49`), initialized to the current values whenever they are not
 * supplied. Under the M1 storage model both live **in the mesh**, as
 * `FaceFieldId::{ReferenceArea, ReferenceCurvature}`, for the same reason the
 * evolved vertex fields do: a `Kokkos::View` outside the mesh is silently
 * dropped by `splitEdges()` and silently stale after `migrate()`.
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

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What one refinement pass did.
 *
 * Port of mesh.py::RefinementDiagnostics (lines 69-78), plus the fields T4a's
 * exit criterion measures. Every count is **global** — a per-rank number here
 * would be a partition-dependent diagnostic, and the whole point of these is to
 * be comparable across rank counts and against the Python.
 */
struct RefinementDiagnostics
{
    GlobalIndex old_vertices = 0;
    GlobalIndex new_vertices = 0;
    GlobalIndex old_faces = 0;
    GlobalIndex new_faces = 0;
    /// Faces red after closure. Zero means the pass was a no-op, which the
    /// driver uses to skip the follow-on quality repair entirely.
    GlobalIndex marked_faces = 0;
    /// Edges in the mask handed to `splitEdges()`, i.e. edges bisected.
    GlobalIndex split_edges = 0;
    Real max_area_change = 0.0;
    Real max_curvature = 0.0;

    //-- T4a additions ------------------------------------------------------//

    /// \f$\sum_f (|S_f| + 1)\f$ over owned faces, reduced. Computed **before**
    /// the edit; `new_faces` must equal it exactly afterwards.
    GlobalIndex projected_faces = 0;
    /// Score threshold the seed selection settled on. `1` when neither
    /// `--max-refine-fraction` nor `--max-faces` bound.
    Real score_threshold = 1.0;
    /// Rounds the red-green balance fixpoint took to converge.
    int balance_rounds = 0;
    /// **R12 signal 1** — global minimum inradius/circumradius after the pass.
    /// `0.5` is equilateral, `0` degenerate.
    Real min_radius_ratio = 0.0;
    /// **R12 signal 2** — global count of faces below \f$r/R = 0.25\f$ after
    /// the pass. Tessera's own `kTail` threshold, kept so the two diagnostics
    /// compare directly.
    GlobalIndex faces_below_quarter = 0;
    /// Faces present after the pass whose gid is NOT in the pre-pass owned-gid
    /// snapshot, i.e. children of subdivided parents, reduced globally. Reported
    /// rather than inferred: "the snapshot difference was empty" and "nothing
    /// refined" are different failures with the same symptom.
    GlobalIndex new_faces_created = 0;
    /// True when `--max-faces` forced the threshold above the seed threshold,
    /// i.e. the run is in risk R4's territory and will not match the Python
    /// face for face.
    bool max_faces_bound = false;
};

//---------------------------------------------------------------------------//
/**
 * @brief Indicator-driven adaptive refinement through `splitEdges()`.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class AdaptiveMesh
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using scalar_view = Kokkos::View<Real*, device_type>;
    /// Per-entity 0/1 marks. Sized over the **local** range (owned + ghost) for
    /// edges, because a face's three edges may include a ghost edge and the
    /// \f$|S_f|\f$ count has to see it.
    using mark_view = Kokkos::View<char*, device_type>;

    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    /// Hard cap on the red-green balance fixpoint, and on the ring growth's own
    /// exchange loop. Hit means the closure did not converge; `refine()`
    /// **throws** rather than proceeding with a partial mark set, because a
    /// partial one produces a mesh whose face count does not match the
    /// projection and whose two sides of a rank boundary may disagree.
    ///
    /// 64 is far above anything reachable here: each round promotes at least
    /// one face and the cascade length is bounded by the diameter of the marked
    /// region in faces, which is a few for the seeds `--max-refine-fraction`
    /// permits. It is a runaway detector, not a tuning knob.
    static constexpr int max_closure_rounds = 64;

    /// R12's second signal is the population below this radius ratio. Tessera's
    /// `kTail` value (`../tessera/tests/test_split_edges_depth.cpp:98-102`),
    /// borrowed rather than invented so the two diagnostics are comparable.
    /// **This is not the pass/fail floor** — that one is measured, per R12, and
    /// lives in the test.
    static constexpr Real quality_tail_threshold = 0.25;

    /**
     * @brief What each `FaceFieldId` slot means, in slot order. T4a.
     *
     * The declaration `CheckpointIO::write` emits as
     * `/beatnik/face_field_names`, and the face-side answer to risk R14's
     * second consequence: `/faces/u<N>` is positional exactly as
     * `/vertices/u<N>` is, so reordering `FaceFieldId` would silently relabel
     * every checkpoint on disk. R14 asks T4a to **extend M2's mechanism rather
     * than invent a second one**, and this is that extension — same shape, same
     * `static_assert` at the writer, same cross-check in `compare_output.py`.
     *
     * It lives here rather than in the IO adapter or beside the enum for the
     * reason M2 gave for putting `vertex_field_names` on `SurfaceState`: the
     * cross-check is only worth something if the writer and the comparator name
     * the slots **independently**, and `AdaptiveMesh` is what actually decides
     * what the three face slots mean.
     *
     * Order must match `Beatnik::FaceFieldId`; `FaceFieldId::Count` is asserted
     * against it at the use site.
     */
    static constexpr const char* face_field_names[3] = {
        "reference_face_area", "reference_face_curvature", "refine_mark" };

    /// @param params AMR thresholds and caps.
    explicit AdaptiveMesh( const AmrParams& params )
        : _params( params )
    {
    }

    //-----------------------------------------------------------------------//
    // Indicators
    //
    // All three are evaluated over the WHOLE LOCAL face range but are only
    // MEANINGFUL on owned faces: the curvature ones read the one-rings of a
    // face's corners, which are complete for an owned face and may be short for
    // a ghost face at the outermost ring. Route (a) is what makes that
    // sufficient -- only owned rows are ever thresholded, and the verdict is
    // then exchanged. Sizing them over the local range rather than the owned one
    // is the same convention `Beatnik_MeshGeometry.hpp` states for every
    // assembled quantity: what is ASSEMBLED spans the local range, what is
    // CONSUMED spans the owned range.
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
     *
     * **T4a CHANGE — the reference is read from the mesh**, not passed in. It
     * is `FaceFieldId::ReferenceArea`, because a per-face array outside the
     * mesh does not survive `splitEdges()`; see the file header.
     *
     * @param face_area  `(Nf,)` current areas, from `MeshGeometry::face_area`.
     * @param[out] indicator `(Nf,)` result, allocated by the caller.
     */
    void areaChangeIndicator( mesh_type& mesh, const scalar_view& face_area,
                              scalar_view& indicator ) const
    {
        const int nf = static_cast<int>( indicator.extent( 0 ) );
        auto ref = mesh.referenceFaceArea();
        auto area = face_area;
        auto out = indicator;
        Kokkos::parallel_for(
            "beatnik_amr_area_change",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                // np.maximum(reference, 1e-300): floored, not branched, so a
                // zero reference gives a huge finite value rather than a NaN.
                const Real r = ( ref( f ) > Real( 1.0e-300 ) ) ? ref( f )
                                                              : Real( 1.0e-300 );
                out( f ) = Kokkos::fabs( area( f ) / r - Real( 1 ) );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Per-vertex curvature magnitude, the cheap robust estimator.
     *
     * Port of mesh.py::cotangent_vertex_curvature (lines 150-169)
     *
     * \f[
     *   \kappa_v = \frac{\big\|\,\mathrm{mean}_{j\in N(v)}(x_j - x_v)\,\big\|}
     *                   {\mathrm{mean}_{j\in N(v)}\|x_j-x_v\|^2}
     * \f]
     * i.e. the umbrella displacement over the mean squared edge length, units
     * 1/length, with the denominator floored at 1e-300.
     *
     * The reference explicitly notes this "is intentionally robust and cheap
     * rather than a high-order mean-curvature estimator" (`mesh.py:154-155`).
     * Substituting the cotangent mean curvature would change every refinement
     * decision, so do not "improve" it — `meanCurvatureNormal` exists
     * separately for the places where the true quantity is needed.
     *
     * @param[out] curvature `(Nv,)` over the whole local vertex range.
     */
    void vertexCurvature( mesh_type& mesh, scalar_view& curvature ) const
    {
        const int nv = static_cast<int>( curvature.extent( 0 ) );
        auto pos = mesh.positions();
        auto ring = mesh.vertexOneRing();
        auto offsets = ring.offsets;
        auto neighbors = ring.neighbors;
        auto out = curvature;
        Kokkos::parallel_for(
            "beatnik_amr_vertex_curvature",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                const int b = offsets( i );
                const int e = offsets( i + 1 );
                const int n = e - b;
                if ( n <= 0 )
                {
                    out( i ) = Real( 0 );
                    return;
                }
                Real disp[3] = { Real( 0 ), Real( 0 ), Real( 0 ) };
                Real h2 = Real( 0 );
                for ( int k = b; k < e; ++k )
                {
                    const int j = neighbors( k );
                    Real s = 0;
                    for ( int d = 0; d < 3; ++d )
                    {
                        const Real dx = pos( j, d ) - pos( i, d );
                        disp[d] += dx;
                        s += dx * dx;
                    }
                    h2 += s;
                }
                const Real inv = Real( 1 ) / static_cast<Real>( n );
                h2 *= inv;
                if ( h2 < Real( 1.0e-300 ) )
                    h2 = Real( 1.0e-300 );
                Real norm = 0;
                for ( int d = 0; d < 3; ++d )
                {
                    disp[d] *= inv;
                    norm += disp[d] * disp[d];
                }
                out( i ) = Kokkos::sqrt( norm ) / h2;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Per-face curvature indicator: the max over the face's corners.
     *
     * Port of mesh.py::face_curvature_indicator (lines 172-174)
     */
    void faceCurvature( mesh_type& mesh, scalar_view& curvature ) const
    {
        const int nv = mesh.totalVertexCount();
        const int nf = static_cast<int>( curvature.extent( 0 ) );
        scalar_view vertex_curvature(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_amr_vertex_curvature" ),
            nv );
        vertexCurvature( mesh, vertex_curvature );

        auto fv = mesh.faceVertices();
        auto vk = vertex_curvature;
        auto out = curvature;
        Kokkos::parallel_for(
            "beatnik_amr_face_curvature",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                Real m = Real( 0 );
                bool any = false;
                for ( int k = 0; k < 3; ++k )
                {
                    const int i = fv( f, k );
                    if ( i < 0 )
                        continue;
                    if ( !any || vk( i ) > m )
                        m = vk( i );
                    any = true;
                }
                out( f ) = any ? m : Real( 0 );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Relative curvature change of each face since its reference.
     *
     * Port of mesh.py::curvature_change_indicator (lines 221-224)
     *
     * \f[
     *   \eta^{\kappa}_f = \left|\frac{\kappa_f}{\kappa_f^{\text{ref}}}-1\right| .
     * \f]
     *
     * **T4a CHANGE — the reference is read from the mesh**, as for the area.
     */
    void curvatureChangeIndicator( mesh_type& mesh,
                                   const scalar_view& face_curvature,
                                   scalar_view& indicator ) const
    {
        const int nf = static_cast<int>( indicator.extent( 0 ) );
        auto ref = mesh.referenceFaceCurvature();
        auto cur = face_curvature;
        auto out = indicator;
        Kokkos::parallel_for(
            "beatnik_amr_curvature_change",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const Real r = ( ref( f ) > Real( 1.0e-300 ) ) ? ref( f )
                                                              : Real( 1.0e-300 );
                out( f ) = Kokkos::fabs( cur( f ) / r - Real( 1 ) );
            } );
        Kokkos::fence();
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
     *       reads ghost faces. An edge with fewer than two **resident**
     *       incident faces is skipped, which on a distributed closed surface
     *       would silently under-refine along partition boundaries if the ghost
     *       layer were missing — hence `refine()`'s residency check, which makes
     *       that condition a throw rather than a quiet bias.
     *
     * Scattered per edge into both incident faces with `atomic_max` rather than
     * gathered per face, because the quantity is naturally edge-centred and the
     * per-face max is an associative reduction over it. That makes it
     * order-independent and therefore bitwise reproducible, unlike a summed
     * scatter (risk R2).
     */
    void curvatureResolutionIndicator( mesh_type& mesh,
                                       const typename MeshGeometry<
                                           ExecutionSpace,
                                           MemorySpace>::vector_view&
                                           face_normal,
                                       scalar_view& indicator ) const
    {
        const int nf = static_cast<int>( indicator.extent( 0 ) );
        const int ne = mesh.totalEdgeCount();

        Kokkos::deep_copy( indicator, Real( 0 ) );

        auto pos = mesh.positions();
        auto ev = mesh.edgeVertices();
        auto inc = mesh.edgeAdjacency();
        // The RESIDENT pair, not the gid-recorded one: a dihedral angle needs
        // both faces' normals in this rank's own arrays, and after an edit
        // `EdgeField::Faces` records only this side of a partition boundary.
        // See `SurfaceMesh::EdgeFaceIncidence`.
        auto count = inc.resident_count;
        auto faces = inc.resident_faces;
        auto fn = face_normal;
        auto out = indicator;
        Kokkos::parallel_for(
            "beatnik_amr_edge_dihedral",
            Kokkos::RangePolicy<ExecutionSpace>( 0, ne ),
            KOKKOS_LAMBDA( const int e ) {
                if ( count( e ) != 2 )
                    return;
                const int f0 = faces( e, 0 );
                const int f1 = faces( e, 1 );
                if ( f0 < 0 || f1 < 0 )
                    return;
                const int i0 = ev( e, 0 );
                const int i1 = ev( e, 1 );
                if ( i0 < 0 || i1 < 0 )
                    return;
                Real s = 0;
                for ( int d = 0; d < 3; ++d )
                {
                    const Real dx = pos( i1, d ) - pos( i0, d );
                    s += dx * dx;
                }
                const Real len = Kokkos::sqrt( s );
                if ( len <= Real( 1.0e-300 ) )
                    return;
                Real dot = 0;
                for ( int d = 0; d < 3; ++d )
                    dot += fn( f0, d ) * fn( f1, d );
                // np.clip(dot, -1, 1) before arccos: a dot product of a pair of
                // unit vectors can leave [-1,1] in the last bit and arccos
                // returns NaN there.
                if ( dot > Real( 1 ) )
                    dot = Real( 1 );
                if ( dot < Real( -1 ) )
                    dot = Real( -1 );
                const Real value = Kokkos::acos( dot ) / len;
                Kokkos::atomic_max( &out( f0 ), value );
                Kokkos::atomic_max( &out( f1 ), value );
            } );
        Kokkos::fence();

        // ... then 0.125 * kappa * h_max^2, in place.
        scalar_view h_min(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_amr_h_min" ),
            nf );
        scalar_view h_max(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_amr_h_max" ),
            nf );
        auto fv = mesh.faceVertices();
        SurfaceOperators::faceEdgeExtents( pos, fv, h_min, h_max );

        auto hi = h_max;
        Kokkos::parallel_for(
            "beatnik_amr_sagitta", Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                out( f ) = Real( 0.125 ) * out( f ) * hi( f ) * hi( f );
            } );
        Kokkos::fence();
    }

    //-----------------------------------------------------------------------//
    // Mark selection and closure
    //-----------------------------------------------------------------------//

    /**
     * @brief Seed score for every face, from the three indicators.
     *
     * Port of mesh_solver.py::refine_potential_mesh_state (lines 1388-1408)
     * and ::_drop_faces_below_min_edge (lines 1626-1635)
     *
     * A face is seeded if **any** indicator exceeds its threshold:
     * \f[
     *   m_f = (\eta^A_f > \tau_A) \;\lor\; (\eta^\kappa_f > \tau_\kappa)
     *         \;\lor\; (\eta^{\text{sag}}_f > \tau_{\text{sag}}) ,
     * \f]
     * with the third clause present only when \f$\tau_{\text{sag}} > 0\f$, and
     * the per-face **score** is each indicator normalized by its own threshold:
     * \f[
     *   s_f = \max\!\Big(\frac{\eta^A_f}{\tau_A},\;
     *                    \frac{\eta^\kappa_f}{\tau_\kappa},\;
     *                    \frac{\eta^{\text{sag}}_f}{\tau_{\text{sag}}}\Big).
     * \f]
     *
     * **The two are the same statement**, and T4a leans on that: under this
     * normalization the seed condition is exactly \f$s_f > 1\f$, so "which
     * faces are seeded" and "how highly are they ranked" are one number. That
     * is what lets `--max-refine-fraction` and `--max-faces` both be enforced by
     * *raising the threshold* rather than by a distributed sort or a greedy
     * accept loop.
     *
     * Faces whose shortest edge is already below `--min-refine-edge` score
     * \f$0\f$ — the hard floor that stops refinement chasing a feature it has
     * already resolved to the intended scale. Scoring them zero rather than
     * masking them separately keeps the threshold search's monotonicity, which
     * a separate mask would break.
     *
     * @param[out] score `(Nf,)` over the whole local range; only owned rows are
     *             ever read, because only owned faces are thresholded.
     */
    void markFaces( mesh_type& mesh, const scalar_view& area_indicator,
                    const scalar_view& curvature_indicator,
                    const scalar_view& resolution_indicator,
                    scalar_view& score ) const
    {
        const int nf = static_cast<int>( score.extent( 0 ) );

        scalar_view h_min(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_amr_seed_h_min" ),
            nf );
        scalar_view h_max(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_amr_seed_h_max" ),
            nf );
        SurfaceOperators::faceEdgeExtents( mesh.positions(),
                                           mesh.faceVertices(), h_min, h_max );

        const Real tau_a =
            std::max( _params.area_change_threshold, Real( 1.0e-300 ) );
        const Real tau_k =
            std::max( _params.curvature_change_threshold, Real( 1.0e-300 ) );
        const Real tau_s = _params.curvature_resolution_threshold;
        const bool use_sagitta = tau_s > Real( 0 );
        const Real min_edge = _params.min_refine_edge;

        auto ia = area_indicator;
        auto ik = curvature_indicator;
        auto is = resolution_indicator;
        auto lo = h_min;
        auto out = score;
        Kokkos::parallel_for(
            "beatnik_amr_score", Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                if ( min_edge > Real( 0 ) && !( lo( f ) > min_edge ) )
                {
                    out( f ) = Real( 0 );
                    return;
                }
                Real s = ia( f ) / tau_a;
                const Real k = ik( f ) / tau_k;
                if ( k > s )
                    s = k;
                if ( use_sagitta )
                {
                    const Real g = is( f ) / tau_s;
                    if ( g > s )
                        s = g;
                }
                out( f ) = s;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Global count of owned faces scoring strictly above a threshold.
     *
     * The probe both threshold searches share. One `MPI_Allreduce` of one
     * `long long` per probe — exact because integer, unlike a floating sum
     * (risk R2), and therefore identical on every rank without further
     * agreement.
     */
    GlobalIndex countAbove( mesh_type& mesh, const scalar_view& score,
                            Real threshold ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto s = score;
        const Real t = threshold;
        long long local = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_count_above",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, long long& acc ) {
                if ( s( f ) > t )
                    ++acc;
            },
            local );
        long long total = 0;
        MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        return static_cast<GlobalIndex>( total );
    }

    /**
     * @brief Cap the seed marks at `--max-refine-fraction` of all faces.
     *
     * Port of mesh_solver.py::_limit_marked_fraction (lines 1434-1451)
     *
     * Budget \f$= \max(1, \lceil f\,N_f\rceil)\f$ over the **global** face
     * count; if more faces score above the seed threshold than that, the
     * threshold is raised until they fit. Applied to the **seeds**, before
     * closure — so the closure can and does exceed the fraction.
     *
     * **The ranking is global, and this is a threshold search rather than a
     * distributed sort.** The reference sorts the marked faces by score and
     * keeps the top `budget`; a distributed sort of a quantity that only exists
     * on owned faces is a genuinely expensive collective, while bisecting the
     * threshold costs one `MPI_Allreduce` of one `long long` per probe. The two
     * differ where scores tie across the cut: the sort keeps exactly `budget`
     * faces, the search keeps every face above a threshold and so may keep
     * fewer. That is the same class of deviation as risk R4 and is recorded
     * with it — the cap is a resource limit, not physics.
     *
     * @return The threshold to seed at. `1` (the untouched seed condition
     *         \f$s_f > 1\f$) when the fraction does not bind.
     */
    Real limitMarkedFraction( mesh_type& mesh, const scalar_view& score,
                              Real max_fraction ) const
    {
        if ( !( max_fraction > Real( 0 ) ) )
            return Real( 1 );

        const GlobalIndex n_faces = mesh.globalFaceCount();
        const GlobalIndex seeded = countAbove( mesh, score, Real( 1 ) );
        if ( seeded == 0 )
            return Real( 1 );

        const GlobalIndex budget = std::max<GlobalIndex>(
            1, static_cast<GlobalIndex>( std::ceil(
                   static_cast<double>( max_fraction ) *
                   static_cast<double>( n_faces ) ) ) );
        if ( seeded <= budget )
            return Real( 1 );

        return searchThreshold( mesh, score, Real( 1 ),
                                [&]( Real t ) {
                                    return countAbove( mesh, score, t ) <=
                                           budget;
                                } );
    }

    /**
     * @brief Set the face mark field from a score threshold, and exchange it.
     *
     * The seeding half of route (a): owned faces are thresholded here, the
     * verdict is halo-exchanged, and every rank can then read the mark of every
     * locally-resident face — including faces it does not own.
     *
     * @note MPI. Collective (`haloExchange`). Every rank must reach it, and
     *       does: the threshold is a globally identical number by construction.
     */
    void seedMarks( mesh_type& mesh, const scalar_view& score,
                    Real threshold ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto mark = mesh.refineMark();
        auto s = score;
        const Real t = threshold;
        Kokkos::parallel_for(
            "beatnik_amr_seed_marks",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f ) {
                mark( f ) = ( s( f ) > t ) ? Real( 1 ) : Real( 0 );
            } );
        Kokkos::fence();
        mesh.haloExchange();
    }

    /**
     * @brief Derive the edge mask from the resident face marks.
     *
     * **This is the one place a face-level notion becomes the edge mask**, and
     * it is R12's constraint in code: an edge is marked *if and only if* it is
     * an edge of a marked face. Nothing else can add to the mask.
     *
     * Evaluated over the whole **local** edge range, not the owned one, because
     * \f$|S_f|\f$ has to be countable for an owned face whose third edge is
     * owned elsewhere. The owned prefix is what is eventually handed to
     * `splitEdges()`; the ghost tail exists only so the closure can count.
     */
    void edgeMaskFromMarks( mesh_type& mesh, mark_view& edge_mark ) const
    {
        const int ne = mesh.totalEdgeCount();
        auto inc = mesh.edgeAdjacency();
        // RESIDENT again, and here it is load-bearing: an owned edge whose only
        // marked incident face is owned elsewhere must still be marked, and the
        // gid-recorded pair does not name that face after an edit. Reading the
        // wrong one under-marks along partition boundaries — silently, and by an
        // amount that moves with the rank count.
        auto count = inc.resident_count;
        auto faces = inc.resident_faces;
        auto mark = mesh.refineMark();
        auto out = edge_mark;
        Kokkos::parallel_for(
            "beatnik_amr_edge_mask",
            Kokkos::RangePolicy<ExecutionSpace>( 0, ne ),
            KOKKOS_LAMBDA( const int e ) {
                char m = 0;
                const int n = ( count( e ) < 2 ) ? count( e ) : 2;
                for ( int s = 0; s < n; ++s )
                {
                    const int f = faces( e, s );
                    if ( f >= 0 && mark( f ) > Real( 0.5 ) )
                        m = 1;
                }
                out( e ) = m;
            } );
        Kokkos::fence();
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
     * **On an edge mask one ring step is one line**: red every face that has at
     * least one marked edge. Two faces are neighbours exactly when they share
     * an edge, so "every face adjacent to a red face" and "every face with a
     * marked edge" are the same set — the edge mask already *is* the frontier,
     * and no `_face_neighbors` map is built. Each step re-exchanges the mark, so
     * a ring crosses rank boundaries by the same route (a) machinery as
     * everything else.
     *
     * @note MPI. Collective, `rings` times. `rings` is a globally identical
     *       parameter, so every rank performs the same number of exchanges.
     */
    void expandMarkedRings( mesh_type& mesh, mark_view& edge_mark,
                            int rings ) const
    {
        for ( int r = 0; r < std::max( rings, 0 ); ++r )
        {
            edgeMaskFromMarks( mesh, edge_mark );
            promoteFacesWithMarkedEdges( mesh, edge_mark, /*min_edges=*/1,
                                         /*quality_test=*/false );
            mesh.haloExchange();
        }
        edgeMaskFromMarks( mesh, edge_mark );
    }

    /**
     * @brief Promote poor green transitions to full red refinement.
     *
     * Port of mesh_solver.py::_balance_red_green_refinement
     * (lines 1543-1580), with the predicted-quality helper
     * ::_single_green_split_quality (lines 1606-1623)
     *
     * A red face bisects all three of its edges. An unmarked neighbour sharing
     * split edges is *green*-split to stay conforming, and green splits produce
     * poorer triangles than red ones. This pass, run to a fixed point:
     *
     *   - **Always** promotes a face with \f$\ge 2\f$ marked edges. A two-edge
     *     green split is a genuinely bad element and there is no threshold
     *     worth tuning.
     *   - Promotes a **one-edge** green split if the worse of its two children
     *     would have quality below
     *     \f$\max(\tau_{\text{floor}},\ f_q\, q_{\text{parent}})\f$, with
     *     \f$\tau_{\text{floor}}\f$ = `--transition-quality-floor` (0.18) and
     *     \f$f_q\f$ = `--transition-quality-fraction` (0.45). The absolute
     *     floor catches faces that are already poor; the relative fraction
     *     catches good faces about to be badly split.
     *
     * Promotion marks more edges, which can force further promotions — hence
     * the loop. It terminates because promotion is monotone and bounded by the
     * face count; `max_closure_rounds` is a runaway detector and hitting it
     * **throws**.
     *
     * @note MPI. One `haloExchange` and one `MPI_Allreduce(MPI_LOR)` per round.
     *       The termination test is global and never rank-local: a cascade can
     *       traverse several ranks, so a rank that saw no promotion this round
     *       may still receive one next round.
     *
     * @return Rounds executed.
     */
    int balanceRedGreen( mesh_type& mesh, mark_view& edge_mark ) const
    {
        if ( !_params.balance_refinement )
        {
            edgeMaskFromMarks( mesh, edge_mark );
            return 0;
        }

        for ( int round = 0; round < max_closure_rounds; ++round )
        {
            edgeMaskFromMarks( mesh, edge_mark );
            // min_edges = 1, not 2: a face with a SINGLE marked edge must
            // still be considered, because the one-edge quality promotion is
            // half of this pass. `quality_test` is what makes the two-edge case
            // unconditional and the one-edge case predicated.
            const long long promoted = promoteFacesWithMarkedEdges(
                mesh, edge_mark, /*min_edges=*/1, /*quality_test=*/true );

            int local_changed = ( promoted > 0 ) ? 1 : 0;
            int any_changed = 0;
            MPI_Allreduce( &local_changed, &any_changed, 1, MPI_INT, MPI_LOR,
                           mesh.comm() );
            if ( !any_changed )
                return round;

            mesh.haloExchange();
        }

        throw std::runtime_error(
            "Beatnik::AdaptiveMesh::balanceRedGreen: the red-green closure did "
            "not reach a fixed point in " +
            std::to_string( max_closure_rounds ) +
            " rounds. Proceeding with a partial mark set is not an option: the "
            "post-split face count would not match projectedFaceCount's "
            "prediction, and two ranks could disagree about the same face. "
            "This cap is a runaway detector, not a tuning knob -- a cascade "
            "here is bounded by the diameter of the marked region in faces." );
    }

    /**
     * @brief Projected face count after the split, \f$\sum_f (|S_f|+1)\f$.
     *
     * Port of mesh.py::projected_red_green_face_count (lines 251-271)
     *
     * A face with \f$|S_f|\f$ marked edges becomes \f$|S_f|+1\f$ children —
     * 1, 2, 3 or 4 — which is `splitEdges()`'s contract exactly and is why this
     * is a **closed form** rather than the reference's simulated closure. Summed
     * over **owned** faces and reduced once (risk R9: a ghost face is an owned
     * face somewhere else and would be counted twice).
     *
     * Evaluated *before* the edit, so `--max-faces` is enforced without doing
     * the work — and, because it is this cheap, by a global threshold search
     * rather than the reference's \f$O(N_{\text{seeds}})\f$ greedy accept loop
     * (risk R4).
     *
     * The test asserts the post-split global face count equals this **exactly**.
     * That is the check that catches a mask reconciled differently than it was
     * projected, and the one that fails loudly if the balance fixpoint did not
     * converge.
     */
    GlobalIndex projectedFaceCount( mesh_type& mesh,
                                    const mark_view& edge_mark ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto fe = mesh.faceEdges();
        auto em = edge_mark;
        long long local = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_projected_faces",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, long long& acc ) {
                int s = 0;
                for ( int k = 0; k < 3; ++k )
                {
                    const int e = fe( f, k );
                    if ( e >= 0 && em( e ) )
                        ++s;
                }
                acc += s + 1;
            },
            local );
        long long total = 0;
        MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        return static_cast<GlobalIndex>( total );
    }

    /**
     * @brief Full mark selection: seed, cap, expand, balance, obey `max_faces`.
     *
     * Port of mesh_solver.py::_quality_preserving_refinement_marks
     * (lines 1454-1512)
     *
     * Seed, cap, expand, balance — in that order, which is the reference's. The
     * order matters: capping the *seeds* and then closing is what lets the
     * closure legitimately exceed `--max-refine-fraction`, and closing before
     * capping would make the cap mean something else.
     *
     * **`--max-faces` is a threshold search, not a greedy accept loop (R4).**
     * The reference walks the seeds in descending score, tentatively adds each,
     * re-closes, and keeps it only if the projection still fits — each trial
     * depending on the previous acceptance, which does not parallelize. Here the
     * projection is a closed form (`projectedFaceCount`), so the same intent —
     * respect the cap, prefer high scores, keep the closure valid — is met by
     * bisecting the seed threshold until the *closed* mark set's projection
     * fits. \f$O(\log)\f$ closures instead of \f$O(N_{\text{seeds}})\f$,
     * parallel and deterministic. A threshold search accepts a different mark
     * set than a greedy walk, so **a capped run still will not match the Python
     * face for face**; that half of R4 stands.
     *
     * @param[out] edge_mark `(Ne,)` over the local range, the closed mask.
     * @return The threshold used, and whether `--max-faces` forced it up.
     */
    std::pair<Real, bool> selectMarks( mesh_type& mesh,
                                       const scalar_view& score,
                                       mark_view& edge_mark,
                                       int& balance_rounds ) const
    {
        const Real seed_threshold =
            limitMarkedFraction( mesh, score, _params.max_refine_fraction );

        auto close = [&]( Real t ) {
            seedMarks( mesh, score, t );
            expandMarkedRings( mesh, edge_mark, _params.refine_neighbor_rings );
            balance_rounds = balanceRedGreen( mesh, edge_mark );
        };

        close( seed_threshold );
        if ( _params.max_faces <= 0 )
            return { seed_threshold, false };

        const GlobalIndex cap = static_cast<GlobalIndex>( _params.max_faces );
        if ( projectedFaceCount( mesh, edge_mark ) <= cap )
            return { seed_threshold, false };

        // The cap binds. Raise the threshold until the CLOSED set's projection
        // fits -- the predicate is monotone in the threshold because a higher
        // threshold seeds a subset, every closure rule is monotone in the seed
        // set, and |S_f| is monotone in the mask.
        const Real capped =
            searchThreshold( mesh, score, seed_threshold, [&]( Real t ) {
                close( t );
                return projectedFaceCount( mesh, edge_mark ) <= cap;
            } );
        close( capped );
        return { capped, true };
    }

    //-----------------------------------------------------------------------//
    // Driver entry point
    //-----------------------------------------------------------------------//

    /**
     * @brief Select marks, split the edges, and fix up the reference state.
     *
     * Port of mesh_solver.py::refine_potential_mesh_state (lines 1374-1431)
     * and mesh.py::refine_marked_faces (lines 570-730)
     *
     * The Python's two refiners (`refine_potential_mesh_state` and
     * `refine_mesh_state`) are identical apart from which field rides through;
     * here the state carries its own model tag and Tessera transfers the whole
     * vertex pack through the `RefinePolicy`, so there is one function and it
     * touches no solution field at all.
     *
     * THE REFERENCE-AREA TRANSFER IS NOT INHERITANCE, AND THAT IS THE SUBTLE
     * PART. `mesh.py::refine_marked_faces` gives each child of a subdivided face
     * \f$A^{\text{ref}}_p \cdot A_{\text{child}} / A_p\f$, keeps an unsplit
     * face's reference unchanged, and **resets** reference *curvature* to the
     * child's current value for subdivided faces only. Tessera inherits face
     * user fields **verbatim** and `RefinePolicy` covers vertex fields only (its
     * two hooks are `interpolatePosition` and `interpolateVertexField`), so
     * neither rule is expressible through the policy. Both are done here with
     * two local passes around the call:
     *
     *   - **before** `splitEdges()`, replace the stored reference area by the
     *     ratio \f$\sigma_f = A^{\text{ref}}_f / A_f\f$;
     *   - **after** it, restore \f$A^{\text{ref}}_f = \sigma_f A_f\f$.
     *
     * That reproduces the Python **exactly**, for both cases at once, with no
     * parent map: children of one parent inherit the same \f$\sigma\f$, so
     * \f$A^{\text{ref}}_{\text{child}} = (A^{\text{ref}}_p/A_p)A_{\text{child}}\f$;
     * and an unsplit face's corners are untouched by a split, so its area — and
     * therefore its reference — is unchanged. The one departure is that the
     * Python normalizes by `sum(child_areas)` rather than by the parent's own
     * area; the two differ only in floating-point association.
     *
     * Reference *curvature* needs a "was I subdivided?" discriminator, and the
     * gids supply it: a face with \f$|S| = 0\f$ **keeps its gid** while a
     * subdivided parent's is retired and its children get fresh ones from the
     * child-gid exscan. So the owned gids are snapshotted before the call and
     * the reference curvature is reset for exactly those faces whose gid is not
     * in the snapshot.
     *
     * @param[in,out] mesh  Surface, refined in place.
     * @param[in]     state Solution. Untouched — Tessera carries the vertex
     *                pack through the split itself. Taken by reference so the
     *                signature does not have to change when a future rule
     *                needs it.
     * @return What the pass did. `marked_faces == 0` means nothing happened and
     *         the caller should skip the follow-on repair.
     *
     * @note MPI. **Collective throughout, and uniformly so.** Every branch is
     *       driven by a globally identical quantity — a reduced count, a
     *       threshold, or a parameter — so every rank performs the same number
     *       of exchanges and reductions. A rank-local branch here would
     *       deadlock rather than misbehave.
     */
    RefinementDiagnostics refine( mesh_type& mesh, state_type& state )
    {
        (void)state;
        RefinementDiagnostics diag;

        requireLocalIncidence( mesh );

        diag.old_vertices = mesh.globalVertexCount();
        diag.old_faces = mesh.globalFaceCount();

        const int nf_local = mesh.totalFaceCount();
        const int ne_local = mesh.totalEdgeCount();

        //-------------------------------------------------------------------//
        // Indicators and the seed score, on the pre-split geometry.
        //-------------------------------------------------------------------//
        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );

        scalar_view area_indicator( "beatnik_amr_area_indicator", nf_local );
        scalar_view curvature( "beatnik_amr_curvature", nf_local );
        scalar_view curvature_indicator( "beatnik_amr_curvature_indicator",
                                         nf_local );
        scalar_view resolution( "beatnik_amr_resolution", nf_local );
        scalar_view score( "beatnik_amr_score_view", nf_local );

        areaChangeIndicator( mesh, geometry.face_area, area_indicator );
        faceCurvature( mesh, curvature );
        curvatureChangeIndicator( mesh, curvature, curvature_indicator );
        curvatureResolutionIndicator( mesh, geometry.face_normal, resolution );
        markFaces( mesh, area_indicator, curvature_indicator, resolution,
                   score );

        // The two headline indicator maxima the reference reports, over OWNED
        // faces then reduced -- `Diagnostics::compute` prints the same numbers.
        diag.max_area_change = ownedMax( mesh, area_indicator );
        diag.max_curvature = ownedMax( mesh, curvature );

        //-------------------------------------------------------------------//
        // Seed, cap, expand, balance.
        //-------------------------------------------------------------------//
        mark_view edge_mark( "beatnik_amr_edge_mark", ne_local );
        const auto selected =
            selectMarks( mesh, score, edge_mark, diag.balance_rounds );
        diag.score_threshold = selected.first;
        diag.max_faces_bound = selected.second;

        diag.marked_faces = countMarkedFaces( mesh );
        diag.split_edges = countMarkedOwnedEdges( mesh, edge_mark );
        diag.projected_faces = projectedFaceCount( mesh, edge_mark );

        if ( diag.split_edges == 0 )
        {
            // Nothing to do, and `splitEdges()` would take its documented
            // empty-mask fast path anyway. Returning early keeps the sigma
            // round trip -- which is exact but not the identity in floating
            // point -- out of a pass that changed nothing.
            clearMarks( mesh );
            diag.new_vertices = diag.old_vertices;
            diag.new_faces = diag.old_faces;
            measureShape( mesh, diag );
            return diag;
        }

        //-------------------------------------------------------------------//
        // sigma = A_ref / A, so that inheriting sigma verbatim and multiplying
        // by the child's own area reproduces the Python's per-child scaling.
        //-------------------------------------------------------------------//
        {
            const int n_owned = mesh.ownedFaceCount();
            auto ref = mesh.referenceFaceArea();
            auto area = geometry.face_area;
            Kokkos::parallel_for(
                "beatnik_amr_to_sigma",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int f ) {
                    const Real a = ( area( f ) > Real( 1.0e-300 ) )
                                       ? area( f )
                                       : Real( 1.0e-300 );
                    ref( f ) = ref( f ) / a;
                } );
            Kokkos::fence();
        }

        // The subdivided-face discriminator, taken while the old gids still
        // exist. Owned only: an unsplit face keeps its gid on its own rank, and
        // a face is only ever subdivided by the rank that owns it.
        const std::vector<std::uint64_t> gids_before = mesh.ownedFaceGids();
        std::unordered_set<std::uint64_t> before( gids_before.begin(),
                                                  gids_before.end() );

        //-------------------------------------------------------------------//
        // THE EDIT.
        //-------------------------------------------------------------------//
        std::vector<char> mask = ownedEdgeMaskToHost( mesh, edge_mark );
        mesh.splitEdges( mask );
        mesh.haloExchange();

        diag.new_vertices = mesh.globalVertexCount();
        diag.new_faces = mesh.globalFaceCount();

        //-------------------------------------------------------------------//
        // A_ref = sigma * A on the new geometry, then reset the reference
        // curvature of exactly the faces whose gid is new.
        //-------------------------------------------------------------------//
        MeshGeometry<ExecutionSpace, MemorySpace> refined_geometry;
        refined_geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                                  mesh.faceVertices() );
        {
            const int n_owned = mesh.ownedFaceCount();
            auto ref = mesh.referenceFaceArea();
            auto area = refined_geometry.face_area;
            Kokkos::parallel_for(
                "beatnik_amr_from_sigma",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int f ) {
                    ref( f ) = ref( f ) * area( f );
                } );
            Kokkos::fence();
        }

        resetNewFaceCurvature( mesh, before, diag );

        clearMarks( mesh );
        mesh.haloExchange();

        measureShape( mesh, diag );
        return diag;
    }

    /**
     * @brief Re-base the reference area and curvature to the current geometry.
     *
     * Port of mesh_solver.py::_state_with_faces (lines 1676-1701) — the
     * `reset_reference=True` branch
     *
     * Called after any operation that legitimately changes face areas or
     * curvature without that change signalling a need to refine: initial mesh
     * construction, quality flips, tangential relaxation, dynamic remeshing, and
     * restart. Failing to re-base leaves the change indicators reading a one-off
     * geometric edit as physics and refining in response.
     *
     * **It is also what initializes the two fields.** Tessera's face AoSoA is
     * allocated uninitialized, so a run that never called this would threshold
     * against whatever was in memory. `Solver::setup` therefore calls it
     * unconditionally, not only when `--refine-every > 0`.
     *
     * @note MPI. Collective — it ends with a `haloExchange()`, so a ghost face's
     *       reference is defined too.
     */
    void resetReferenceState( mesh_type& mesh ) const
    {
        const int nf_local = mesh.totalFaceCount();
        const int n_owned = mesh.ownedFaceCount();

        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );

        scalar_view curvature( "beatnik_amr_reset_curvature", nf_local );
        faceCurvature( mesh, curvature );

        auto ref_area = mesh.referenceFaceArea();
        auto ref_curv = mesh.referenceFaceCurvature();
        auto mark = mesh.refineMark();
        auto area = geometry.face_area;
        auto curv = curvature;
        Kokkos::parallel_for(
            "beatnik_amr_reset_reference",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f ) {
                ref_area( f ) = area( f );
                ref_curv( f ) = curv( f );
                mark( f ) = Real( 0 );
            } );
        Kokkos::fence();
        mesh.haloExchange();
    }

  private:
    //-----------------------------------------------------------------------//
    // Closure helpers
    //-----------------------------------------------------------------------//

    /**
     * @brief Red every owned face with at least `min_edges` marked edges,
     *        optionally applying the one-edge quality test.
     *
     * The single kernel both the ring growth and the balance fixpoint are
     * written in terms of, because they differ only in `min_edges` and in
     * whether the one-edge case is tested rather than accepted outright.
     *
     * @return Number of faces newly promoted **on this rank**. The caller
     *         reduces it; a rank-local zero is not a fixed point.
     */
    long long promoteFacesWithMarkedEdges( mesh_type& mesh,
                                           const mark_view& edge_mark,
                                           int min_edges,
                                           bool quality_test ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto fv = mesh.faceVertices();
        auto fe = mesh.faceEdges();
        auto ev = mesh.edgeVertices();
        auto pos = mesh.positions();
        auto mark = mesh.refineMark();
        auto em = edge_mark;
        const int need = min_edges;
        const bool test_one = quality_test;
        const Real floor = std::max( _params.transition_quality_floor,
                                     Real( 0 ) );
        const Real fraction = std::max( _params.transition_quality_fraction,
                                        Real( 0 ) );

        long long promoted = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_promote",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, long long& acc ) {
                if ( mark( f ) > Real( 0.5 ) )
                    return;

                int split = 0;
                int one = -1;
                for ( int k = 0; k < 3; ++k )
                {
                    const int e = fe( f, k );
                    if ( e >= 0 && em( e ) )
                    {
                        ++split;
                        one = e;
                    }
                }
                if ( split < need )
                    return;

                bool promote = split >= 2;
                if ( split == 1 )
                {
                    if ( need <= 1 && !test_one )
                    {
                        // Ring growth: a single marked edge is enough.
                        promote = true;
                    }
                    else if ( test_one &&
                              ( floor > Real( 0 ) || fraction > Real( 0 ) ) )
                    {
                        const Real q_child =
                            singleGreenSplitQuality( pos, fv, ev, f, one );
                        const Real q_parent = triangleQualityOf( pos, fv, f );
                        Real target = fraction * q_parent;
                        if ( floor > target )
                            target = floor;
                        promote = q_child < target;
                    }
                    else
                    {
                        promote = false;
                    }
                }
                if ( promote )
                {
                    mark( f ) = Real( 1 );
                    ++acc;
                }
            },
            promoted );
        Kokkos::fence();
        return promoted;
    }

    /// `4 sqrt(3) A / (a^2+b^2+c^2)`, the reference's `triangle_quality`, for
    /// one face. The same expression `SurfaceOperators::triangleQuality`
    /// evaluates in bulk; inlined here because the promotion kernel needs it
    /// for a face and for two hypothetical children that do not exist yet.
    KOKKOS_INLINE_FUNCTION static Real
    triangleQualityOfPoints( const Real a[3], const Real b[3],
                             const Real c[3] )
    {
        Real ab[3], bc[3], ca[3], ac[3];
        for ( int d = 0; d < 3; ++d )
        {
            ab[d] = b[d] - a[d];
            bc[d] = c[d] - b[d];
            ca[d] = a[d] - c[d];
            ac[d] = c[d] - a[d];
        }
        const Real nx = ab[1] * ac[2] - ab[2] * ac[1];
        const Real ny = ab[2] * ac[0] - ab[0] * ac[2];
        const Real nz = ab[0] * ac[1] - ab[1] * ac[0];
        const Real area =
            Real( 0.5 ) * Kokkos::sqrt( nx * nx + ny * ny + nz * nz );
        Real l2 = 0;
        for ( int d = 0; d < 3; ++d )
            l2 += ab[d] * ab[d] + bc[d] * bc[d] + ca[d] * ca[d];
        return ( l2 > Real( 0 ) )
                   ? Real( 4.0 ) * Kokkos::sqrt( Real( 3.0 ) ) * area / l2
                   : Real( 0 );
    }

    template <class PosView, class FaceView>
    KOKKOS_INLINE_FUNCTION static Real
    triangleQualityOf( const PosView& pos, const FaceView& fv, int f )
    {
        Real p[3][3];
        for ( int k = 0; k < 3; ++k )
        {
            const int i = fv( f, k );
            if ( i < 0 )
                return Real( 0 );
            for ( int d = 0; d < 3; ++d )
                p[k][d] = pos( i, d );
        }
        return triangleQualityOfPoints( p[0], p[1], p[2] );
    }

    /**
     * @brief Worse of the two children a one-edge green split would produce.
     *
     * Port of mesh_solver.py::_single_green_split_quality (lines 1606-1623)
     *
     * The median from the marked edge's midpoint to the opposite corner. Note
     * this is the *predicted* quality of the Python's split; Tessera's
     * \f$|S|=1\f$ case is the same two children (the median from the midpoint),
     * so unlike the two-edge diagonal (risk R13) there is no divergence to
     * account for here.
     */
    template <class PosView, class FaceView, class EdgeView>
    KOKKOS_INLINE_FUNCTION static Real
    singleGreenSplitQuality( const PosView& pos, const FaceView& fv,
                             const EdgeView& ev, int f, int e )
    {
        const int ea = ev( e, 0 );
        const int eb = ev( e, 1 );
        if ( ea < 0 || eb < 0 )
            return Real( 0 );

        int opposite = -1;
        for ( int k = 0; k < 3; ++k )
        {
            const int i = fv( f, k );
            if ( i < 0 )
                return Real( 0 );
            if ( i != ea && i != eb )
                opposite = i;
        }
        if ( opposite < 0 )
            return Real( 0 );

        Real a[3], b[3], o[3], m[3];
        for ( int d = 0; d < 3; ++d )
        {
            a[d] = pos( ea, d );
            b[d] = pos( eb, d );
            o[d] = pos( opposite, d );
            m[d] = Real( 0.5 ) * ( a[d] + b[d] );
        }
        const Real q0 = triangleQualityOfPoints( a, m, o );
        const Real q1 = triangleQualityOfPoints( m, b, o );
        if ( !( q0 == q0 ) || !( q1 == q1 ) )
            return Real( 0 ); // np.all(np.isfinite(...)) else 0.0
        return ( q0 < q1 ) ? q0 : q1;
    }

    /**
     * @brief Smallest threshold at or above `lo` for which `fits` holds.
     *
     * A fixed-iteration bisection on the score, not a sort. `fits` must be
     * monotone (true stays true as the threshold rises), which every caller's
     * predicate is: a higher threshold seeds a subset, and every closure rule
     * and the \f$|S_f|+1\f$ projection are monotone in the seed set.
     *
     * **Fixed iteration count, deliberately.** Every probe is a collective, so
     * the loop bound has to be a globally identical constant — a convergence
     * test on a floating threshold could in principle terminate at different
     * iterations on different ranks and deadlock. 60 halvings of the initial
     * bracket takes any double-precision score interval below its own ulp, so
     * this is exact rather than approximate.
     *
     * The upper bracket comes from the global maximum score, which is above
     * every face's score and therefore always fits (it seeds nothing).
     */
    template <class Predicate>
    Real searchThreshold( mesh_type& mesh, const scalar_view& score, Real lo,
                          Predicate fits ) const
    {
        Real hi = ownedMax( mesh, score );
        if ( !( hi > lo ) )
            return lo;
        // Strictly above every score, so `s > hi` is empty and the predicate
        // holds there by construction.
        hi = hi * ( Real( 1 ) + Real( 1.0e-12 ) ) + Real( 1.0e-300 );

        for ( int i = 0; i < 60; ++i )
        {
            const Real mid = Real( 0.5 ) * ( lo + hi );
            if ( fits( mid ) )
                hi = mid;
            else
                lo = mid;
        }
        return hi;
    }

    //-----------------------------------------------------------------------//
    // Small collective helpers
    //-----------------------------------------------------------------------//

    /// Global maximum of a per-face quantity over **owned** faces.
    /// `MPI_MAX` is reproducible across rank counts, unlike a sum (risk R2).
    Real ownedMax( mesh_type& mesh, const scalar_view& values ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto v = values;
        Real local = Real( 0 );
        Kokkos::parallel_reduce(
            "beatnik_amr_owned_max",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, Real& acc ) {
                if ( v( f ) > acc )
                    acc = v( f );
            },
            Kokkos::Max<Real>( local ) );
        Real total = Real( 0 );
        MPI_Allreduce( &local, &total, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        return total;
    }

    /// Global count of red faces, over **owned** faces.
    GlobalIndex countMarkedFaces( mesh_type& mesh ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto mark = mesh.refineMark();
        long long local = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_count_marked_faces",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, long long& acc ) {
                if ( mark( f ) > Real( 0.5 ) )
                    ++acc;
            },
            local );
        long long total = 0;
        MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        return static_cast<GlobalIndex>( total );
    }

    /// Global count of marked **owned** edges — the size of the edit, and the
    /// number `SplitResult::requested` reports from Tessera's own side.
    GlobalIndex countMarkedOwnedEdges( mesh_type& mesh,
                                       const mark_view& edge_mark ) const
    {
        const int n_owned = mesh.ownedEdgeCount();
        auto em = edge_mark;
        long long local = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_count_marked_edges",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int e, long long& acc ) {
                if ( em( e ) )
                    ++acc;
            },
            local );
        long long total = 0;
        MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        return static_cast<GlobalIndex>( total );
    }

    /// Zero the mark field over the whole local range, so a later pass starts
    /// from a defined state and a checkpoint written in between carries zeros
    /// rather than the last pass's marks.
    void clearMarks( mesh_type& mesh ) const
    {
        const int nf = mesh.totalFaceCount();
        auto mark = mesh.refineMark();
        Kokkos::parallel_for(
            "beatnik_amr_clear_marks",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) { mark( f ) = Real( 0 ); } );
        Kokkos::fence();
    }

    /// The owned prefix of the edge mask, on the host, in the exact shape
    /// `Tessera::splitEdges` requires.
    std::vector<char> ownedEdgeMaskToHost( mesh_type& mesh,
                                           const mark_view& edge_mark ) const
    {
        const int n_owned = mesh.ownedEdgeCount();
        auto sub = Kokkos::subview( edge_mark, std::make_pair( 0, n_owned ) );
        auto host = Kokkos::create_mirror_view( sub );
        Kokkos::deep_copy( host, sub );
        std::vector<char> mask( static_cast<std::size_t>( n_owned ) );
        for ( int e = 0; e < n_owned; ++e )
            mask[e] = host( e );
        return mask;
    }

    /**
     * @brief Reset the reference curvature of faces whose gid is new.
     *
     * The post-split half of the curvature rule. A face absent from the
     * pre-split owned-gid snapshot is a child of a subdivided parent; the
     * Python resets exactly those (`refine_marked_faces`'s `reset_mask`).
     *
     * Done on the host over the owned range, because the discriminator is a hash
     * lookup against the snapshot. Face counts here are in the thousands and the
     * pass runs once per refinement, so the transfer is not worth avoiding —
     * and doing it on device would mean shipping a sorted gid array and a binary
     * search for no measurable gain.
     */
    void resetNewFaceCurvature( mesh_type& mesh,
                                const std::unordered_set<std::uint64_t>& before,
                                RefinementDiagnostics& diag ) const
    {
        const int nf_local = mesh.totalFaceCount();
        const int n_owned = mesh.ownedFaceCount();

        scalar_view curvature( "beatnik_amr_new_curvature", nf_local );
        faceCurvature( mesh, curvature );

        const std::vector<std::uint64_t> gids_after = mesh.ownedFaceGids();

        Kokkos::View<char*, device_type> is_new( "beatnik_amr_is_new_face",
                                                 n_owned );
        auto h_is_new = Kokkos::create_mirror_view( is_new );
        long long local_new = 0;
        for ( int f = 0; f < n_owned; ++f )
        {
            const bool fresh = before.find( gids_after[f] ) == before.end();
            h_is_new( f ) = fresh ? 1 : 0;
            if ( fresh )
                ++local_new;
        }
        Kokkos::deep_copy( is_new, h_is_new );

        long long total_new = 0;
        MPI_Allreduce( &local_new, &total_new, 1, MPI_LONG_LONG, MPI_SUM,
                       mesh.comm() );
        // A silently empty snapshot difference and "nothing refined" are
        // different failures with the same symptom, so the count is reported
        // rather than inferred.
        diag.new_faces_created = static_cast<GlobalIndex>( total_new );

        auto ref_curv = mesh.referenceFaceCurvature();
        auto curv = curvature;
        auto fresh = is_new;
        Kokkos::parallel_for(
            "beatnik_amr_reset_new_curvature",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f ) {
                if ( fresh( f ) )
                    ref_curv( f ) = curv( f );
            } );
        Kokkos::fence();
    }

    /**
     * @brief R12's two signals, measured over owned faces after the pass.
     *
     * \f[
     *   \frac{r}{R} = \frac{8A^2}{(a+b+c)\,abc}
     * \f]
     * — `0.5` for an equilateral triangle, `0` for a degenerate one, and the
     * reciprocal of twice Tessera's published \f$Q = R/2r\f$, so the two
     * measurements compare directly without a conversion at the call site.
     */
    void measureShape( mesh_type& mesh, RefinementDiagnostics& diag ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        auto pos = mesh.positions();
        auto fv = mesh.faceVertices();
        const Real tail = quality_tail_threshold;

        Real local_min = Real( 1.0e300 );
        long long local_tail = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_shape",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, Real& mn, long long& cnt ) {
                Real p[3][3];
                for ( int k = 0; k < 3; ++k )
                {
                    const int i = fv( f, k );
                    if ( i < 0 )
                        return;
                    for ( int d = 0; d < 3; ++d )
                        p[k][d] = pos( i, d );
                }
                Real e0[3], e1[3], e2[3];
                for ( int d = 0; d < 3; ++d )
                {
                    e0[d] = p[1][d] - p[0][d];
                    e1[d] = p[2][d] - p[1][d];
                    e2[d] = p[0][d] - p[2][d];
                }
                const Real a = Kokkos::sqrt( e0[0] * e0[0] + e0[1] * e0[1] +
                                             e0[2] * e0[2] );
                const Real b = Kokkos::sqrt( e1[0] * e1[0] + e1[1] * e1[1] +
                                             e1[2] * e1[2] );
                const Real c = Kokkos::sqrt( e2[0] * e2[0] + e2[1] * e2[1] +
                                             e2[2] * e2[2] );
                Real u[3];
                u[0] = e0[1] * ( -e2[2] ) - e0[2] * ( -e2[1] );
                u[1] = e0[2] * ( -e2[0] ) - e0[0] * ( -e2[2] );
                u[2] = e0[0] * ( -e2[1] ) - e0[1] * ( -e2[0] );
                const Real area =
                    Real( 0.5 ) *
                    Kokkos::sqrt( u[0] * u[0] + u[1] * u[1] + u[2] * u[2] );
                const Real den = ( a + b + c ) * a * b * c;
                const Real ratio =
                    ( den > Real( 0 ) )
                        ? Real( 8 ) * area * area / den
                        : Real( 0 );
                if ( ratio < mn )
                    mn = ratio;
                if ( ratio < tail )
                    ++cnt;
            },
            Kokkos::Min<Real>( local_min ), local_tail );

        Real global_min = Real( 0 );
        MPI_Allreduce( &local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                       mesh.comm() );
        long long global_tail = 0;
        MPI_Allreduce( &local_tail, &global_tail, 1, MPI_LONG_LONG, MPI_SUM,
                       mesh.comm() );

        diag.min_radius_ratio = global_min;
        diag.faces_below_quarter = static_cast<GlobalIndex>( global_tail );
    }

    /**
     * @brief Check route (a)'s precondition rather than assume it.
     *
     * Route (a) is only correct if every rank can see, for each of its **owned
     * edges**, both incident faces, and for each of its **owned faces**, all
     * three of its edges. Both hold at `halo_depth = 2` — an edge of an owned
     * face joins two vertices the rank holds, so the face on its other side is
     * incident on a held vertex and is therefore in Tessera's local face set —
     * but "hold" is exactly the sort of claim that is cheap to check and
     * expensive to be wrong about: a missing incidence does not crash, it
     * silently under-marks along partition boundaries, which reads as a
     * physics difference that moves with the rank count.
     *
     * T1b's test already asserts every edge has exactly two incident faces at
     * ranks 1-6, so the `count == 2` half is a re-assertion; the residency half
     * is new.
     */
    void requireLocalIncidence( mesh_type& mesh ) const
    {
        const int n_owned_e = mesh.ownedEdgeCount();
        const int n_owned_f = mesh.ownedFaceCount();

        auto inc = mesh.edgeAdjacency();
        auto count = inc.resident_count;
        auto faces = inc.resident_faces;
        long long bad_edges = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_check_edges",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_e ),
            KOKKOS_LAMBDA( const int e, long long& acc ) {
                if ( count( e ) != 2 || faces( e, 0 ) < 0 || faces( e, 1 ) < 0 )
                    ++acc;
            },
            bad_edges );

        auto fe = mesh.faceEdges();
        long long bad_faces = 0;
        Kokkos::parallel_reduce(
            "beatnik_amr_check_faces",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_f ),
            KOKKOS_LAMBDA( const int f, long long& acc ) {
                for ( int k = 0; k < 3; ++k )
                    if ( fe( f, k ) < 0 )
                    {
                        ++acc;
                        return;
                    }
            },
            bad_faces );

        long long local[2] = { bad_edges, bad_faces };
        long long total[2] = { 0, 0 };
        MPI_Allreduce( local, total, 2, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        if ( total[0] == 0 && total[1] == 0 )
            return;

        throw std::runtime_error(
            "Beatnik::AdaptiveMesh::refine: the halo does not carry the "
            "incidence route (a) needs -- " +
            std::to_string( total[0] ) +
            " owned edge(s) without two locally resident incident faces and " +
            std::to_string( total[1] ) +
            " owned face(s) with a non-resident edge, summed globally. The "
            "mark translation would under-mark along partition boundaries "
            "instead of failing, which reads as a physics difference that "
            "moves with the rank count. Check that the mesh was distributed at "
            "SurfaceMesh::halo_depth." );
    }

    AmrParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_ADAPTIVEMESH_HPP
