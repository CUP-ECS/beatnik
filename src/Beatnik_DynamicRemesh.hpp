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
 *
 * WHAT T4b IMPLEMENTED, AND WHAT STILL THROWS
 * -------------------------------------------
 * The sizing field, the gradation and the **split** third of the cycle are
 * implemented. The collapse, flip and tangential-smoothing thirds are **T4d**
 * and still throw; the two proximity paths are **T4e** and still throw.
 *
 * A `--dynamic-remesh` run is therefore accepted only when the three
 * unimplemented thirds are configured off **through the reference's own knobs**,
 * so that what runs here is what the reference would run, not a Beatnik-only
 * subset of it:
 *
 * | third | configured off by | reference becomes |
 * | --- | --- | --- |
 * | collapse | `--remesh-collapse-factor 0` | `dynamic_remesh.py:373`'s candidate predicate `length < 0 * target` is never true, so the pass returns before any mutation |
 * | smoothing | `--remesh-smooth-iters 0` | `:463-465` returns immediately |
 * | flips | `--remesh-flip-min-gain >= 1e12` | `:449-450` `continue`s every candidate; see `kFlipsDisabledMinGain` |
 * | cleanup | `--no-isotropic-cleanup` | `run_adaptive_mesh_bubble.py:1493` skips the whole block |
 *
 * `Solver::requireSupportedConfiguration` rejects anything else by name and by
 * task ID before the first step. **`--remesh-max-collapses 0` is NOT one of the
 * levers**, and reading it as one is the trap: the driver maps a non-positive
 * value to `None` = *unlimited* (`run_adaptive_mesh_bubble.py:1350-1352`), which
 * `RemeshParams::max_collapses_per_pass` reproduces ("<= 0 means unlimited").
 * Only the collapse factor disables the pass.
 */

#ifndef BEATNIK_DYNAMICREMESH_HPP
#define BEATNIK_DYNAMICREMESH_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

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
    // The four counts are **global** and `GlobalIndex`, not the Python's `int`,
    // for the reason `RefinementDiagnostics` states: a per-rank entity count is
    // a statement about the partition, and these exist to be compared across
    // rank counts and against the reference.
    GlobalIndex old_vertices = 0;
    GlobalIndex old_faces = 0;
    GlobalIndex new_vertices = 0;
    GlobalIndex new_faces = 0;
    int splits = 0;
    int collapses = 0;
    int flips = 0;
    int smooth_steps = 0;
    Real min_quality_before = 0.0;
    Real min_quality_after = 0.0;
    Real max_sagitta_before = 0.0;
    Real max_sagitta_after = 0.0;

    //-- T4b additions ------------------------------------------------------//
    //
    // Every count here is **global**, for the reason `RefinementDiagnostics`
    // gives: a per-rank number is a statement about the partition and these
    // exist to be compared across rank counts and against the Python.

    /// Passes actually executed, i.e. `max(passes, 0)`.
    int passes = 0;
    /// Edges satisfying \f$\ell > f_s\max(h_{\text{target}}, h_{\min})\f$ over
    /// the whole pass sequence — the split pass's **candidate** set before the
    /// per-pass cap. `splits < split_candidates` exactly when the cap bound,
    /// which is the distinction T4b's exit criterion asserts on.
    GlobalIndex split_candidates = 0;
    /// True when `max_splits_per_pass` truncated the candidate list in any
    /// pass. Risk R4's territory: a truncated pass will not match the Python,
    /// because a global threshold search on the length ratio accepts a
    /// different set than the reference's sort-and-slice.
    bool split_capped = false;
    /// Ratio \f$\ell/\max(h_{\text{target}},10^{-300})\f$ the cap settled on,
    /// `0` when it did not bind.
    Real split_ratio_threshold = 0.0;
    /// Candidate edges remaining **after** the last pass, with the sizing field
    /// recomputed on the edited mesh. Not a defect: a split halves an edge, so
    /// an edge more than \f$2f_s\f$ over target is still long afterwards and is
    /// the next pass's work. Reported so "the split pass keeps up" is a number
    /// rather than an impression.
    GlobalIndex long_edges_after = 0;
    /// Of those, the ones whose per-edge target sits **at** \f$h_{\min}\f$ —
    /// the population the sizing floor, not the algorithm, is holding back.
    GlobalIndex long_edges_at_h_min = 0;
    /// **R12 signal 1** — global minimum inradius/circumradius after the call.
    Real min_radius_ratio = 0.0;
    /// **R12 signal 2** — global count of faces below \f$r/R = 0.25\f$ after
    /// the call. Tessera's own `kTail` threshold.
    GlobalIndex faces_below_quarter = 0;
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

    /// R12's second signal is the population below this radius ratio —
    /// Tessera's `kTail` value
    /// (`../tessera/tests/test_split_edges_depth.cpp:98-102`), borrowed rather
    /// than invented so the two diagnostics are comparable, and the same number
    /// `AdaptiveMesh::quality_tail_threshold` carries for the same reason.
    /// **This is not a pass/fail floor**; that one is measured and lives in the
    /// test.
    static constexpr Real quality_tail_threshold = 0.25;

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
     *
     * **T4b — the mesh argument is non-const**, as it is on every other
     * geometric consumer in the tree: `positions()`, `faceVertices()` and
     * `faceEdges()` are all non-const (Cabana slices behind a generation guard,
     * and CSRs cached against `generation()`). Same widening T2c applied to
     * twelve signatures.
     *
     * Both estimates are assembled over the **whole local face set** and are
     * complete on every entity this pass consumes, by the rule in
     * `Beatnik_MeshGeometry.hpp`'s DISTRIBUTED ASSEMBLY section: at
     * `halo_depth = 2` every corner of a locally held face has its entire
     * incident-face set local, so the cotangent sum at that corner is whole, and
     * every edge of an owned face names two locally resident faces (which
     * `AdaptiveMesh::requireLocalIncidence` checks rather than assumes).
     *
     * Two deliberate departures from `SurfaceOperators::faceCotangents`, both
     * because this is a port of a *different* Python function: the cotangent is
     * **clipped to \f$\pm 50\f$** (`dynamic_remesh.py::_safe_cotangent`,
     * "only to prevent a single nearly degenerate triangle from dominating the
     * sizing field"), and a zero cross product yields **0** rather than a huge
     * finite weight. `faceCotangents` does neither, correctly, because the
     * Laplace-Beltrami operator it feeds must not be clipped.
     *
     * @param[out] curvature `(Nf,)` over the local face range.
     */
    void faceCurvatureForSizing( mesh_type& mesh, scalar_view& curvature ) const
    {
        const int nf = mesh.totalFaceCount();
        const int nv = mesh.totalVertexCount();

        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), nv, mesh.faceVertices() );

        // --- 1. cotangent_vertex_curvature (dynamic_remesh.py:674-712) ------
        vector_view laplace( "beatnik_remesh_cot_laplace", nv );
        {
            auto pos = mesh.positions();
            auto fv = mesh.faceVertices();
            auto area = geometry.face_area;
            auto acc = laplace;
            Kokkos::parallel_for(
                "beatnik_remesh_cot_assemble",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
                KOKKOS_LAMBDA( const int f ) {
                    const int idx[3] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
                    if ( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 )
                        return;
                    // `for face, area in zip(faces, areas): if area <= 1e-300:
                    // continue` -- a collapsed triangle contributes nothing at
                    // all, rather than contributing a clipped weight.
                    if ( area( f ) <= Real( 1.0e-300 ) )
                        return;
                    for ( int k = 0; k < 3; ++k )
                    {
                        // cot at corner k weights the OPPOSITE edge, which the
                        // Python spells out as
                        // _add_cotangent_edge_laplace(L, j, k, cot_i, V).
                        const int p = idx[k];
                        const int q = idx[( k + 1 ) % 3];
                        const int r = idx[( k + 2 ) % 3];
                        Real u[3], w[3];
                        for ( int d = 0; d < 3; ++d )
                        {
                            u[d] = pos( q, d ) - pos( p, d );
                            w[d] = pos( r, d ) - pos( p, d );
                        }
                        const Real cr[3] = { u[1] * w[2] - u[2] * w[1],
                                             u[2] * w[0] - u[0] * w[2],
                                             u[0] * w[1] - u[1] * w[0] };
                        const Real cross_norm = Kokkos::sqrt(
                            cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] );
                        if ( cross_norm <= Real( 1.0e-300 ) )
                            continue;
                        Real dot = 0;
                        for ( int d = 0; d < 3; ++d )
                            dot += u[d] * w[d];
                        Real cot = dot / cross_norm;
                        // np.clip(..., -50.0, 50.0)
                        cot = ( cot < Real( -50 ) )  ? Real( -50 )
                              : ( cot > Real( 50 ) ) ? Real( 50 )
                                                     : cot;
                        if ( cot == Real( 0 ) )
                            continue;
                        for ( int d = 0; d < 3; ++d )
                        {
                            const Real delta =
                                cot * ( pos( r, d ) - pos( q, d ) );
                            Kokkos::atomic_add( &acc( q, d ), delta );
                            Kokkos::atomic_add( &acc( r, d ), -delta );
                        }
                    }
                } );
            Kokkos::fence();
        }

        scalar_view vertex_curvature( "beatnik_remesh_vertex_curvature", nv );
        {
            auto acc = laplace;
            auto va = geometry.vertex_area;
            auto out = vertex_curvature;
            Kokkos::parallel_for(
                "beatnik_remesh_cot_normalize",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nv ),
                KOKKOS_LAMBDA( const int i ) {
                    // np.divide(..., where=vertex_area > 1e-300) leaves 0.
                    if ( !( va( i ) > Real( 1.0e-300 ) ) )
                    {
                        out( i ) = Real( 0 );
                        return;
                    }
                    Real s = 0;
                    for ( int d = 0; d < 3; ++d )
                    {
                        const Real l =
                            acc( i, d ) / ( Real( 2 ) * va( i ) );
                        s += l * l;
                    }
                    out( i ) = Real( 0.5 ) * Kokkos::sqrt( s );
                } );
            Kokkos::fence();
        }

        // --- 2. normal_variation_curvature (dynamic_remesh.py:639-655), and
        //        the max of the two (::face_curvature_for_sizing) ------------
        //
        // The Python loops EDGES and scatters `max` onto both incident faces;
        // this gathers, per face, over its own three edges. The two are the same
        // set of (edge, face) pairs, so the result is identical -- and the
        // gather needs no atomic max, which Kokkos would have to emulate with a
        // CAS loop on a Real.
        {
            const int nf_local = nf;
            auto pos = mesh.positions();
            auto ev = mesh.edgeVertices();
            auto fe = mesh.faceEdges();
            auto inc = mesh.edgeAdjacency();
            // RESIDENT, never the gid-recorded pair: `EdgeField::Faces` is
            // partial by construction after any topological edit, and this runs
            // between splits. T4a paid a whole gate run for that lesson.
            auto count = inc.resident_count;
            auto faces = inc.resident_faces;
            auto fn = geometry.face_normal;
            auto fv = mesh.faceVertices();
            auto vk = vertex_curvature;
            auto out = curvature;
            Kokkos::parallel_for(
                "beatnik_remesh_face_curvature",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nf_local ),
                KOKKOS_LAMBDA( const int f ) {
                    Real cot_face = Real( 0 );
                    bool any = false;
                    for ( int k = 0; k < 3; ++k )
                    {
                        const int i = fv( f, k );
                        if ( i < 0 )
                            continue;
                        if ( !any || vk( i ) > cot_face )
                            cot_face = vk( i );
                        any = true;
                    }
                    if ( !any )
                        cot_face = Real( 0 );

                    Real jump = Real( 0 );
                    for ( int k = 0; k < 3; ++k )
                    {
                        const int e = fe( f, k );
                        if ( e < 0 || count( e ) != 2 )
                            continue;
                        const int f0 = faces( e, 0 );
                        const int f1 = faces( e, 1 );
                        if ( f0 < 0 || f1 < 0 )
                            continue;
                        const int a = ev( e, 0 );
                        const int b = ev( e, 1 );
                        if ( a < 0 || b < 0 )
                            continue;
                        Real s = 0;
                        for ( int d = 0; d < 3; ++d )
                        {
                            const Real dx = pos( b, d ) - pos( a, d );
                            s += dx * dx;
                        }
                        const Real length = Kokkos::sqrt( s );
                        if ( length <= Real( 1.0e-300 ) )
                            continue;
                        Real dot = 0;
                        for ( int d = 0; d < 3; ++d )
                            dot += fn( f0, d ) * fn( f1, d );
                        dot = ( dot < Real( -1 ) )  ? Real( -1 )
                              : ( dot > Real( 1 ) ) ? Real( 1 )
                                                    : dot;
                        const Real value = Kokkos::acos( dot ) / length;
                        if ( value > jump )
                            jump = value;
                    }

                    out( f ) = ( cot_face > jump ) ? cot_face : jump;
                } );
            Kokkos::fence();
        }
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
     *       size jump appears exactly there. **T4b — that exchange is
     *       `SurfaceMesh::haloExchangeVertexView`**, added for this and
     *       explained there; the same call also publishes stage 3's result, so
     *       an outermost-ring ghost vertex (whose incident-face set is short)
     *       carries its owner's value rather than a partial minimum.
     *
     * @param state Unused until T4e: its material position is the *proximity*
     *        exclusion's input, and both proximity switches are rejected at
     *        setup. Kept in the signature so landing T4e changes no caller.
     */
    void vertexTargetEdgeLength( mesh_type& mesh, const state_type& state,
                                 scalar_view& target ) const
    {
        (void)state;
        const int nf = mesh.totalFaceCount();
        const int nv = mesh.totalVertexCount();

        // --- 1. curvature term ---------------------------------------------
        scalar_view curvature( "beatnik_remesh_sizing_curvature", nf );
        faceCurvatureForSizing( mesh, curvature );

        const Real tol = ( _params.sagitta_tolerance > Real( 1.0e-300 ) )
                             ? _params.sagitta_tolerance
                             : Real( 1.0e-300 );
        const Real h_max2 = ( _params.h_max * _params.h_max > Real( 1.0e-300 ) )
                                ? _params.h_max * _params.h_max
                                : Real( 1.0e-300 );
        // Exactly the curvature at which the formula would return h_max, so a
        // flat region asks for h_max rather than infinity.
        const Real floor_kappa =
            std::max( Real( 8 ) * _params.sagitta_tolerance / h_max2,
                      Real( 1.0e-12 ) );
        const Real h_min = _params.h_min;
        const Real h_max = _params.h_max;

        scalar_view face_target( "beatnik_remesh_face_target", nf );
        {
            auto kappa = curvature;
            auto out = face_target;
            Kokkos::parallel_for(
                "beatnik_remesh_face_target",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
                KOKKOS_LAMBDA( const int f ) {
                    const Real k =
                        ( kappa( f ) > floor_kappa ) ? kappa( f ) : floor_kappa;
                    Real h = Kokkos::sqrt( Real( 8 ) * tol / k );
                    h = ( h < h_min ) ? h_min : ( ( h > h_max ) ? h_max : h );
                    out( f ) = h;
                } );
            Kokkos::fence();
        }

        // --- 2. proximity term: T4e. `--remesh-proximity` is rejected in
        //        `Solver::requireSupportedConfiguration`, so there is nothing to
        //        skip here and nothing that could be silently skipped.

        // --- 3. face -> vertex, as a MINIMUM ---------------------------------
        {
            auto fv = mesh.faceVertices();
            auto ft = face_target;
            auto out = target;
            Kokkos::parallel_for(
                "beatnik_remesh_target_init",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nv ),
                KOKKOS_LAMBDA( const int i ) { out( i ) = h_max; } );
            Kokkos::fence();
            Kokkos::parallel_for(
                "beatnik_remesh_target_scatter",
                Kokkos::RangePolicy<ExecutionSpace>( 0, nf ),
                KOKKOS_LAMBDA( const int f ) {
                    for ( int k = 0; k < 3; ++k )
                    {
                        const int i = fv( f, k );
                        if ( i >= 0 )
                            Kokkos::atomic_min( &out( i ), ft( f ) );
                    }
                } );
            Kokkos::fence();
            clampTarget( target );
        }
        mesh.haloExchangeVertexView( target );

        // --- 4. gradation ---------------------------------------------------
        gradeTargetEdgeLength( mesh, target );
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
     *
     * **The Python's sweep is Jacobi, not Gauss-Seidel, and reading it the other
     * way makes this order-dependent.** `graded_vertex_target_edge_length`
     * (`:753-763`) writes into `target` while reading the *neighbour* value from
     * `old`, the copy taken at the top of the sweep — so one sweep is exactly
     * \f$h_i \leftarrow \min\big(h_i^{\text{old}},\ \gamma\min_{j\in N(i)}
     * h_j^{\text{old}}\big)\f$, independent of the edge order it iterates. That
     * is what makes a distributed one-ring form of it the same computation
     * rather than a similar one.
     *
     * After `k` sweeps a vertex's target sees \f$\gamma^d h^0_j\f$ for every
     * vertex `j` at graph distance \f$d \le k\f$, which at the default 8 sweeps
     * is **four times the halo depth**. Hence the exchange per sweep: without
     * it the constraint is enforced over a 2-ring on a boundary vertex and an
     * 8-ring everywhere else, the sizing field bends at every partition seam,
     * and the split set moves with the rank count.
     *
     * @note MPI. One `haloExchangeVertexView` and one `MPI_Allreduce(MPI_LOR)`
     *       per sweep. The termination test is global and never rank-local: a
     *       rank whose own targets stopped moving may still receive a smaller
     *       neighbour next sweep.
     */
    void gradeTargetEdgeLength( mesh_type& mesh, scalar_view& target ) const
    {
        const Real factor = _params.target_gradation_factor;
        const int iterations = std::max( _params.target_gradation_iterations, 0 );
        if ( !( factor > Real( 1 ) ) || iterations == 0 )
        {
            clampTarget( target );
            return;
        }

        const int n_owned = mesh.ownedVertexCount();
        const int nv = mesh.totalVertexCount();
        scalar_view previous( "beatnik_remesh_target_prev", nv );

        for ( int it = 0; it < iterations; ++it )
        {
            Kokkos::deep_copy( previous, target );

            auto ring = mesh.vertexOneRing();
            auto offsets = ring.offsets;
            auto neighbors = ring.neighbors;
            auto old = previous;
            auto out = target;
            const Real gamma = factor;

            long long changed_local = 0;
            Kokkos::parallel_reduce(
                "beatnik_remesh_gradation",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i, long long& acc ) {
                    Real m = old( i );
                    const int b = offsets( i );
                    const int e = offsets( i + 1 );
                    for ( int k = b; k < e; ++k )
                    {
                        const int j = neighbors( k );
                        if ( j < 0 )
                            continue;
                        const Real limited = gamma * old( j );
                        if ( limited < m )
                            m = limited;
                    }
                    if ( m < old( i ) )
                    {
                        out( i ) = m;
                        ++acc;
                    }
                },
                changed_local );

            int local_changed = ( changed_local > 0 ) ? 1 : 0;
            int any_changed = 0;
            MPI_Allreduce( &local_changed, &any_changed, 1, MPI_INT, MPI_LOR,
                           mesh.comm() );
            // The exchange happens even on the last sweep and even when nothing
            // changed on this rank: another rank's owned value may have moved,
            // and this rank's ghost copy of it is what the split predicate
            // reads.
            mesh.haloExchangeVertexView( target );
            if ( !any_changed )
                break;
        }

        clampTarget( target );
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
     *
     * **THIS IS THE WHOLE OF RISK R12's CONSTRAINT, IN ONE PREDICATE.** An edge
     * enters the mask if and only if
     * \f$\ell_e > f_s \max(\min(h_i,h_j),\ h_{\min})\f$. No curvature term, no
     * vorticity term, no region tag, and no union with anything: this is the one
     * mask family Tessera measured as shape-bounded, and it is bounded *because*
     * it is a coarse Rivara longest-edge rule and therefore self-correcting.
     * Adding a non-length term makes those edges inherit none of the bound; R12
     * names the mask transform that would have to come first.
     *
     * **The cap is a global threshold search on the length ratio, not a sort.**
     * The reference sorts the candidates by
     * \f$\ell/\max(h_{\text{target}},10^{-300})\f$ descending, tie-broken on the
     * endpoint index pair, and slices. A distributed sort would be a different
     * algorithm and the tie-break is not even expressible — Beatnik's vertex
     * indices are not the Python's. So the same intent (prefer the most
     * over-long edges, respect the cap) is met by bisecting a ratio threshold
     * until the surviving count fits, as `AdaptiveMesh::selectMarks` does for
     * `--max-faces` and for the same reason (risk R4). **A capped pass therefore
     * will not match the Python edge for edge**, which is why T4b's gate
     * configuration keeps the candidate count under the cap and why
     * `split_capped` is reported rather than inferred.
     *
     * @param[out] diag `splits`, `split_candidates`, `split_capped` and
     *             `split_ratio_threshold` are accumulated here. **Signature
     *             change from the stub** (which returned only the count): the
     *             candidate count is knowable only inside this function and is
     *             half of what the exit criterion asserts on.
     * @return Edges bisected, globally.
     */
    int splitLongEdges( mesh_type& mesh, state_type& state,
                        const scalar_view& target,
                        RemeshDiagnostics& diag ) const
    {
        (void)state;
        const int n_owned_e = mesh.ownedEdgeCount();

        // Per owned edge: is it long, and by what ratio. Both are computed from
        // the same two endpoint targets, but with DIFFERENT floors, which is not
        // a slip in the reference: the predicate floors the target at h_min
        // (`:249`) so the sizing floor cannot demand an unsplittable edge, while
        // the ranking key floors at 1e-300 (`:250`) so the ordering is by how
        // over-long the edge is against what was actually asked for.
        Kokkos::View<char*, device_type> mask( "beatnik_remesh_split_mask",
                                               n_owned_e );
        scalar_view ratio( "beatnik_remesh_split_ratio", n_owned_e );
        {
            auto pos = mesh.positions();
            auto ev = mesh.edgeVertices();
            auto t = target;
            auto m = mask;
            auto r = ratio;
            const Real fs = _params.split_factor;
            const Real hmin = _params.h_min;
            Kokkos::parallel_for(
                "beatnik_remesh_split_candidates",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_e ),
                KOKKOS_LAMBDA( const int e ) {
                    m( e ) = 0;
                    r( e ) = Real( 0 );
                    const int i = ev( e, 0 );
                    const int j = ev( e, 1 );
                    if ( i < 0 || j < 0 )
                        return;
                    Real s = 0;
                    for ( int d = 0; d < 3; ++d )
                    {
                        const Real dx = pos( j, d ) - pos( i, d );
                        s += dx * dx;
                    }
                    const Real length = Kokkos::sqrt( s );
                    const Real local =
                        ( t( i ) < t( j ) ) ? t( i ) : t( j );
                    const Real floored = ( local > hmin ) ? local : hmin;
                    if ( length > fs * floored )
                    {
                        m( e ) = 1;
                        const Real denom =
                            ( local > Real( 1.0e-300 ) ) ? local
                                                         : Real( 1.0e-300 );
                        r( e ) = length / denom;
                    }
                } );
            Kokkos::fence();
        }

        const GlobalIndex candidates = countMask( mesh, mask );
        diag.split_candidates += candidates;
        if ( candidates == 0 )
            return 0;

        // The cap. `max_splits_per_pass <= 0` is unlimited, which is how the
        // driver spells `--remesh-max-splits 0` too
        // (`run_adaptive_mesh_bubble.py:1348`: a non-positive value becomes
        // `None`).
        GlobalIndex splits = candidates;
        if ( _params.max_splits_per_pass > 0 &&
             candidates > static_cast<GlobalIndex>( _params.max_splits_per_pass ) )
        {
            const GlobalIndex cap =
                static_cast<GlobalIndex>( _params.max_splits_per_pass );
            const Real threshold = searchRatioThreshold( mesh, mask, ratio, cap );
            auto m = mask;
            auto r = ratio;
            Kokkos::parallel_for(
                "beatnik_remesh_apply_cap",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_e ),
                KOKKOS_LAMBDA( const int e ) {
                    if ( m( e ) && !( r( e ) > threshold ) )
                        m( e ) = 0;
                } );
            Kokkos::fence();
            splits = countMask( mesh, mask );
            diag.split_capped = true;
            diag.split_ratio_threshold = threshold;
        }

        if ( splits == 0 )
            return 0;

        mesh.splitEdges( ownedEdgeMaskToHost( mesh, mask ) );
        mesh.haloExchange();
        return static_cast<int>( splits );
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
     *
     * **T4d, not T4c**, and T4b assigned it: it is a port of
     * `dynamic_remesh.py::tangential_smooth_vertices` and runs inside the same
     * `if changed or needs_quality_repair` block as the flips, whereas T4c ports
     * `mesh_solver.py::improve_mesh_quality_tangential` — a different function
     * with a different trigger. `--remesh-smooth-iters > 0` is rejected in
     * `Solver::requireSupportedConfiguration` naming this method and T4d.
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
     * precisely so nothing can be forgotten; here they are Tessera vertex user
     * fields and `splitEdges()` interpolates the whole pack at every midpoint
     * through `RefinePolicy`, so this function touches no solution field at all.
     * (`SurfaceState::remap`, which an earlier revision of this comment named,
     * was **deleted** at the M1 rework for exactly that reason.)
     *
     * The remeshed state is built with `reference_face_area=None` and
     * `reference_face_curvature=None`, re-basing the AMR change indicators.
     * **That re-basing is `AdaptiveMesh::resetReferenceState` and the SOLVER
     * calls it**, immediately after this returns — not this function, which
     * would otherwise have to depend on the AMR header to satisfy a
     * configuration in which the AMR indicators are never read. See
     * `Solver::advanceOneStep`'s remesh branch.
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
        RemeshDiagnostics diag;
        diag.old_vertices = static_cast<int>( mesh.globalVertexCount() );
        diag.old_faces = static_cast<int>( mesh.globalFaceCount() );
        diag.min_quality_before = globalMinQuality( mesh );
        diag.max_sagitta_before = globalMaxSagitta( mesh );

        const int passes = std::max( _params.passes, 0 );
        diag.passes = passes;

        for ( int pass = 0; pass < passes; ++pass )
        {
            // 1. Surgical proximity splits. T4e, and `--remesh-surgical-
            //    proximity` is rejected at setup, so this is not a skip.

            // 2. Recompute the sizing field, then split.
            scalar_view target( "beatnik_remesh_target",
                                mesh.totalVertexCount() );
            vertexTargetEdgeLength( mesh, state, target );
            const int splits = splitLongEdges( mesh, state, target, diag );
            diag.splits += splits;

            // 3. Recompute the sizing field, then collapse. T4d (Tessera gap
            //    G5b). The only configuration accepted here has
            //    `collapse_factor <= 0`, which makes the reference's own
            //    `collapse_short_edges` return before touching anything
            //    (`dynamic_remesh.py:373` -- the candidate predicate
            //    `length < 0 * target` is false for every edge). The sizing
            //    recompute at `:151` exists ONLY to feed that call, so with the
            //    call gone it is dead arithmetic and is not performed. It is
            //    listed here rather than silently absent because the recompute
            //    is load-bearing the moment T4d lands: without it the collapse
            //    pass undoes the split pass.
            const int collapses = 0;

            // 4. Repair, if anything changed or the worst element is poor.
            //    Both repairs are T4d and both are configured off, so the gate
            //    is transcribed in full rather than folded away -- landing T4d
            //    then turns it on by deleting a rejection rather than by
            //    remembering to add a call. Same convention T4a used for the
            //    volume-projection gate.
            const bool changed = ( splits > 0 ) || ( collapses > 0 );
            const bool needs_quality_repair =
                globalMinQuality( mesh ) < _params.min_quality;
            if ( changed || needs_quality_repair )
            {
                //   flipEdgesForQuality   T4d (blocked, Tessera G5c)
                //   tangentialSmooth      T4d
                diag.flips += 0;
                diag.smooth_steps += 0;
            }
        }

        diag.new_vertices = static_cast<int>( mesh.globalVertexCount() );
        diag.new_faces = static_cast<int>( mesh.globalFaceCount() );
        diag.min_quality_after = globalMinQuality( mesh );
        diag.max_sagitta_after = globalMaxSagitta( mesh );
        measureShape( mesh, diag );
        auditLongEdges( mesh, state, diag );
        return diag;
    }

  private:
    //-----------------------------------------------------------------------//
    // Helpers
    //-----------------------------------------------------------------------//

    /// `np.clip(target, h_min, h_max)`, over the whole local vertex range.
    void clampTarget( scalar_view& target ) const
    {
        const int nv = static_cast<int>( target.extent( 0 ) );
        auto out = target;
        const Real lo = _params.h_min;
        const Real hi = _params.h_max;
        Kokkos::parallel_for(
            "beatnik_remesh_clamp_target",
            Kokkos::RangePolicy<ExecutionSpace>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                out( i ) = ( out( i ) < lo ) ? lo
                                             : ( ( out( i ) > hi ) ? hi
                                                                   : out( i ) );
            } );
        Kokkos::fence();
    }

    /// Global count of set entries in an owned-edge mask.
    GlobalIndex countMask( mesh_type& mesh,
                           const Kokkos::View<char*, device_type>& mask ) const
    {
        const int n = static_cast<int>( mask.extent( 0 ) );
        auto m = mask;
        long long local = 0;
        Kokkos::parallel_reduce(
            "beatnik_remesh_count_mask",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int e, long long& acc ) {
                if ( m( e ) )
                    ++acc;
            },
            local );
        long long total = 0;
        MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        return static_cast<GlobalIndex>( total );
    }

    /**
     * @brief Smallest ratio threshold whose surviving candidate count fits the
     *        per-pass split cap.
     *
     * A fixed 60-iteration bisection, exactly as `AdaptiveMesh::searchThreshold`
     * and for the same reason: every probe is a collective, so the loop bound
     * must be a globally identical constant. A convergence test on a floating
     * threshold could terminate at different iterations on different ranks and
     * deadlock. 60 halvings takes any double-precision interval below its own
     * ulp, so this is exact rather than approximate.
     *
     * The predicate is monotone in the threshold (a higher threshold keeps a
     * subset), and the upper bracket is above every candidate's ratio and
     * therefore keeps nothing.
     */
    Real searchRatioThreshold( mesh_type& mesh,
                               const Kokkos::View<char*, device_type>& mask,
                               const scalar_view& ratio, GlobalIndex cap ) const
    {
        const int n = static_cast<int>( mask.extent( 0 ) );
        auto m = mask;
        auto r = ratio;

        Real local_max = Real( 0 );
        Kokkos::parallel_reduce(
            "beatnik_remesh_ratio_max",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int e, Real& acc ) {
                if ( m( e ) && r( e ) > acc )
                    acc = r( e );
            },
            Kokkos::Max<Real>( local_max ) );
        Real hi = Real( 0 );
        MPI_Allreduce( &local_max, &hi, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        hi = hi * ( Real( 1 ) + Real( 1.0e-12 ) ) + Real( 1.0e-300 );
        Real lo = Real( 0 );

        auto fits = [&]( Real t ) {
            long long local = 0;
            Kokkos::parallel_reduce(
                "beatnik_remesh_ratio_count",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                KOKKOS_LAMBDA( const int e, long long& acc ) {
                    if ( m( e ) && r( e ) > t )
                        ++acc;
                },
                local );
            long long total = 0;
            MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM,
                           mesh.comm() );
            return static_cast<GlobalIndex>( total ) <= cap;
        };

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

    /// The owned prefix of an edge mask, on the host, in the exact shape
    /// `Tessera::splitEdges` requires. The same ten lines as
    /// `AdaptiveMesh::ownedEdgeMaskToHost`; the two are kept separate because
    /// the mask lives in the pass that built it and neither class is the other's
    /// dependency.
    std::vector<char>
    ownedEdgeMaskToHost( mesh_type& mesh,
                         const Kokkos::View<char*, device_type>& mask ) const
    {
        const int n_owned = mesh.ownedEdgeCount();
        auto host = Kokkos::create_mirror_view( mask );
        Kokkos::deep_copy( host, mask );
        std::vector<char> out( static_cast<std::size_t>( n_owned ) );
        for ( int e = 0; e < n_owned; ++e )
            out[e] = host( e );
        return out;
    }

    /// Global minimum triangle quality \f$4\sqrt3 A/\sum\ell^2\f$ over owned
    /// faces — the reference's `_finite_min(triangle_quality(...))`, which is
    /// the scale `--remesh-min-quality` (0.18) is expressed on. **Not** the
    /// \f$r/R\f$ of R12's signals; the two differ by more than a constant.
    Real globalMinQuality( mesh_type& mesh ) const
    {
        const int n_owned = mesh.ownedFaceCount();
        scalar_view quality( "beatnik_remesh_quality", mesh.totalFaceCount() );
        SurfaceOperators::triangleQuality( mesh.positions(),
                                           mesh.faceVertices(), quality );
        auto q = quality;
        Real local = Real( 1.0e300 );
        Kokkos::parallel_reduce(
            "beatnik_remesh_min_quality",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, Real& mn ) {
                // _finite_min: a non-finite entry is skipped, not propagated.
                if ( Kokkos::isfinite( q( f ) ) && q( f ) < mn )
                    mn = q( f );
            },
            Kokkos::Min<Real>( local ) );
        Real total = Real( 0 );
        MPI_Allreduce( &local, &total, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
        return total;
    }

    /// Global maximum flat-triangle sagitta
    /// \f$\tfrac18 \kappa_f h_f^2\f$ over owned faces, with \f$h_f\f$ the face's
    /// longest edge.
    ///
    /// Port of dynamic_remesh.py::curvature_sagitta_indicator (lines 657-659)
    ///
    /// This is the quantity `--remesh-sagitta-tolerance` is a target for, so it
    /// is the honest report of whether the sizing field got what it asked for.
    Real globalMaxSagitta( mesh_type& mesh ) const
    {
        const int nf = mesh.totalFaceCount();
        const int n_owned = mesh.ownedFaceCount();
        scalar_view curvature( "beatnik_remesh_sagitta_curvature", nf );
        faceCurvatureForSizing( mesh, curvature );
        scalar_view lo( "beatnik_remesh_face_min_edge", nf );
        scalar_view hi( "beatnik_remesh_face_max_edge", nf );
        SurfaceOperators::faceEdgeExtents( mesh.positions(),
                                           mesh.faceVertices(), lo, hi );
        auto k = curvature;
        auto h = hi;
        Real local = Real( 0 );
        Kokkos::parallel_reduce(
            "beatnik_remesh_max_sagitta",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int f, Real& mx ) {
                const Real s = Real( 0.125 ) * k( f ) * h( f ) * h( f );
                if ( Kokkos::isfinite( s ) && s > mx )
                    mx = s;
            },
            Kokkos::Max<Real>( local ) );
        Real total = Real( 0 );
        MPI_Allreduce( &local, &total, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
        return total;
    }

    /// R12's two signals, measured over owned faces after the call. The kernel
    /// is `SurfaceOperators::radiusRatioStats`, shared verbatim with
    /// `AdaptiveMesh::measureShape` so the two tasks' numbers are the same
    /// number.
    void measureShape( mesh_type& mesh, RemeshDiagnostics& diag ) const
    {
        Real local_min = Real( 1.0e300 );
        long long local_tail = 0;
        SurfaceOperators::radiusRatioStats<ExecutionSpace>(
            mesh.positions(), mesh.faceVertices(), mesh.ownedFaceCount(),
            quality_tail_threshold, local_min, local_tail );

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
     * @brief Re-derive the sizing field on the edited mesh and count what is
     *        still long — the exit criterion's "either split in the next pass or
     *        blocked by \f$h_{\min}\f$", as a number.
     *
     * A non-zero count is **not** a defect and the criterion does not ask for
     * zero: a split halves an edge, so an edge that was more than \f$2f_s\f$
     * over its target is still over it afterwards, and the recomputed curvature
     * can legitimately lower a target under an edge that was fine before. What
     * would be a defect is a long edge that the *next* pass then declines to
     * mark, and that cannot happen, because the next pass rebuilds the mask from
     * this same predicate: `split_candidates == splits` whenever the cap did not
     * bind is the assertion that closes it.
     *
     * `long_edges_at_h_min` separates the population the sizing floor holds
     * back, which no number of passes will clear, from the population that is
     * simply the next pass's work.
     */
    void auditLongEdges( mesh_type& mesh, state_type& state,
                         RemeshDiagnostics& diag ) const
    {
        const int n_owned_e = mesh.ownedEdgeCount();
        scalar_view target( "beatnik_remesh_audit_target",
                            mesh.totalVertexCount() );
        vertexTargetEdgeLength( mesh, state, target );

        auto pos = mesh.positions();
        auto ev = mesh.edgeVertices();
        auto t = target;
        const Real fs = _params.split_factor;
        const Real hmin = _params.h_min;

        long long local[2] = { 0, 0 };
        Kokkos::parallel_reduce(
            "beatnik_remesh_audit",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned_e ),
            KOKKOS_LAMBDA( const int e, long long& n_long, long long& n_floor ) {
                const int i = ev( e, 0 );
                const int j = ev( e, 1 );
                if ( i < 0 || j < 0 )
                    return;
                Real s = 0;
                for ( int d = 0; d < 3; ++d )
                {
                    const Real dx = pos( j, d ) - pos( i, d );
                    s += dx * dx;
                }
                const Real length = Kokkos::sqrt( s );
                const Real local_target = ( t( i ) < t( j ) ) ? t( i ) : t( j );
                const Real floored =
                    ( local_target > hmin ) ? local_target : hmin;
                if ( length > fs * floored )
                {
                    ++n_long;
                    if ( !( local_target > hmin ) )
                        ++n_floor;
                }
            },
            local[0], local[1] );

        long long total[2] = { 0, 0 };
        MPI_Allreduce( local, total, 2, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
        diag.long_edges_after = static_cast<GlobalIndex>( total[0] );
        diag.long_edges_at_h_min = static_cast<GlobalIndex>( total[1] );
    }

    RemeshParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_DYNAMICREMESH_HPP
