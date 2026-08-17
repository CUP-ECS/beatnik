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
 * @file Beatnik_MeshGeometry.hpp
 * @brief Discrete differential geometry on a triangulated surface: areas,
 *        normals, gradients, Laplacians, curvature and enclosed volume.
 *
 * Everything here is a pure function of `(vertices, faces)` — no solution
 * state, no parameters. These are the primitives every other mathematical
 * header is written in terms of, so their sign and normalization conventions
 * are load-bearing and are stated explicitly on each routine.
 *
 * GLOBAL CONVENTIONS
 * ------------------
 * - A face `(i, j, k)` is **counter-clockwise seen from outside**, so
 *   \f$(v_j - v_i)\times(v_k - v_i)\f$ points **outward** and the enclosed
 *   volume computed from it is **positive**. Every mesh generator and every
 *   topological edit must preserve this; the AMR and remeshing code explicitly
 *   re-orients children against the parent normal for exactly this reason.
 * - Degenerate denominators are floored at `1e-300` rather than guarded by a
 *   branch, matching the Python (`np.maximum(..., 1.0e-300)`). This keeps the
 *   kernels branch-free for the GPU and reproduces the reference behavior on a
 *   collapsed triangle: a huge but finite value rather than a NaN.
 * - "Vertex area" is the **barycentric lumped** area \f$A_v=\frac13\sum_{f\ni
 *   v} A_f\f$, not the Voronoi or mixed area. The Python uses barycentric
 *   throughout (`mesh_solver.py:223`), so the port must too or the cotangent
 *   Laplacian is normalized differently and the viscous term changes magnitude.
 *
 * DISTRIBUTED ASSEMBLY — PASS ALL LOCAL FACES, AND DO NOT SCATTER-ADD
 * ------------------------------------------------------------------
 * Every routine here that turns a per-face quantity into a per-vertex one does
 * it by looping faces and accumulating onto their three corners. On a
 * distributed surface that looks like it needs a ghost-to-owner scatter-add,
 * and the pre-T1b version of this header said so on four routines. **It does
 * not**, and the reason is a property of Tessera's local set:
 *
 *   Tessera's local face set is *the owned faces plus every face incident on an
 *   owned vertex* (more at halo depth 2, which is what Beatnik builds).
 *
 * So a loop over **all locally held faces** — `[0, totalFaceCount())`, not
 * `[0, ownedFaceCount())` — gives every *owned* vertex contributions from its
 * complete incident-face set, with no communication and no double-counting of
 * an owned vertex. Ghost vertices are left holding partial sums, which is
 * exactly what a scatter-add would leave behind anyway and is harmless as long
 * as consumers read owned rows.
 *
 * Two rules follow, and both are load-bearing:
 *
 *   1. **Pass the whole local face set** to `MeshGeometry::compute` and
 *      `volumeGradient`. Passing only the owned faces makes every
 *      partition-boundary vertex's area and normal too small — a seam of
 *      spurious velocity that *moves* when the rank count changes, which is the
 *      signature to look for.
 *   2. **Do not follow them with `Comm::haloScatterAdd`.** That would
 *      double-count. The scatter-add is for the *other* pattern (owned-face
 *      loop into a mesh-resident field) and is documented as such in
 *      `Beatnik_Communication.hpp`.
 *
 * The routines that reduce to a **global scalar** are the opposite case:
 * `enclosedVolume` and `edgeLengths` must see **owned entities only**, because
 * there each local entity contributes once to a global sum or minimum and a
 * ghost is an owned entity somewhere else (risk R9). They take the range the
 * caller hands them, so the caller passes an owned subrange — see each.
 *
 * DETERMINISM
 * -----------
 * The per-vertex accumulations use `Kokkos::atomic_add`, whose summation order
 * is not reproducible run to run on a GPU. Vertex areas and normals are
 * therefore not bitwise reproducible, at any rank count including one. That is
 * the same class of non-determinism as risk R2 and is why no test compares an
 * assembled per-vertex field bitwise.
 */

#ifndef BEATNIK_MESHGEOMETRY_HPP
#define BEATNIK_MESHGEOMETRY_HPP

#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Discrete geometric quantities of a triangle surface.
 *
 * Bundled because they are almost always wanted together and because computing
 * them in one face loop is one scatter-add halo exchange instead of four.
 *
 * @tparam MemorySpace Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class MeshGeometry
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;

    /// `(Nf,)` triangle areas, \f$A_f = \tfrac12\|(b-a)\times(c-a)\|\f$.
    scalar_view face_area;

    /// `(Nf,3)` outward unit face normals.
    vector_view face_normal;

    /// `(Nv,)` barycentric lumped vertex areas, floored at 1e-300.
    scalar_view vertex_area;

    /// `(Nv,3)` area-weighted unit vertex normals.
    vector_view vertex_normal;

    /**
     * @brief Compute all four arrays in one pass.
     *
     * Port of mesh_solver.py::_mesh_geometry_arrays (lines 216-236) and
     * ::mesh_geometry (lines 203-213)
     *
     * Face quantities:
     * \f[
     *   n_f^{\text{raw}} = (b-a)\times(c-a), \quad
     *   A_f = \tfrac12\|n_f^{\text{raw}}\|, \quad
     *   \hat n_f = n_f^{\text{raw}}/\|n_f^{\text{raw}}\| .
     * \f]
     * A zero-length raw normal yields \f$\hat n_f = 0\f$ (not NaN), matching
     * `np.divide(..., where=lengths > 0)`.
     *
     * Vertex quantities, scattered from the face loop:
     * \f[
     *   A_v = \tfrac13\sum_{f \ni v} A_f, \qquad
     *   \hat n_v = \frac{\sum_{f\ni v} A_f\,\hat n_f}{\|\sum_{f\ni v} A_f\,\hat n_f\|}.
     * \f]
     * Note the vertex normal is weighted by face **area**, not by angle or
     * uniformly. On a strongly graded adaptive mesh the three weightings differ
     * materially, and the area weighting is what the reference uses.
     *
     * The normalization is applied in a **second** pass, after the whole face
     * loop has finished, because the accumulator and the result share storage:
     * `vertex_normal` holds \f$\sum A_f \hat n_f\f$ during pass 1. Normalizing
     * inside the face loop would normalize a partial sum.
     *
     * @param vertices     `(Nv,3)` positions, ghosts included. Indexed
     *                     `(i, d)`; may be a Kokkos view or a
     *                     generation-guarded Cabana slice, which is why the
     * count is separate.
     * @param vertex_count `Nv` = `mesh.totalVertexCount()`. **Explicit because
     *                     `SurfaceMesh::position_slice` deliberately exposes no
     *                     extent** — it is a Cabana slice behind a generation
     *                     guard that forwards `operator()` only. (Signature
     *                     change from the pre-T1b header, forced by the M1
     *                     storage model.)
     * @param faces        `(Nf,3)` **local** vertex indices. Pass the WHOLE
     *                     local set, `mesh.totalFaceCount()` rows — see
     *                     DISTRIBUTED ASSEMBLY in the file header. A corner
     *                     index of `-1` (a ghost face reaching off-rank) is
     *                     skipped, and the whole face with it, so a partial
     *                     triangle cannot contribute a wrong area.
     *
     * @note MPI. **None.** No exchange and no scatter-add: the local-face loop
     *       already gives every owned vertex its complete one-ring, and a
     *       scatter-add afterwards would double-count. The pre-T1b note here
     *       required one; that was written before Tessera's local-set rule was
     *       read, and it is wrong. Ghost rows hold partial sums; a consumer
     *       that needs them consistent should recompute rather than
     *       communicate, since these are derived scratch and not mesh fields.
     */
    template <class VertexView, class FaceView>
    void compute( const VertexView& vertices, int vertex_count,
                  const FaceView& faces )
    {
        const int nv = vertex_count;
        const int nf = static_cast<int>( faces.extent( 0 ) );

        face_area = scalar_view(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_area" ),
            nf );
        face_normal = vector_view(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_normal" ),
            nf );
        // Zero-initialized: both are accumulators in pass 1.
        vertex_area = scalar_view( "vertex_area", nv );
        vertex_normal = vector_view( "vertex_normal", nv );

        auto pos = vertices;
        auto fv = faces;
        auto fa = face_area;
        auto fn = face_normal;
        auto va = vertex_area;
        auto vn = vertex_normal;

        Kokkos::parallel_for(
            "beatnik_mesh_geometry_faces",
            Kokkos::RangePolicy<execution_space>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const int ia = fv( f, 0 );
                const int ib = fv( f, 1 );
                const int ic = fv( f, 2 );
                if ( ia < 0 || ib < 0 || ic < 0 )
                {
                    fa( f ) = Real( 0 );
                    for ( int d = 0; d < 3; ++d )
                        fn( f, d ) = Real( 0 );
                    return;
                }
                Real e1[3], e2[3];
                for ( int d = 0; d < 3; ++d )
                {
                    e1[d] = pos( ib, d ) - pos( ia, d );
                    e2[d] = pos( ic, d ) - pos( ia, d );
                }
                Real raw[3];
                raw[0] = e1[1] * e2[2] - e1[2] * e2[1];
                raw[1] = e1[2] * e2[0] - e1[0] * e2[2];
                raw[2] = e1[0] * e2[1] - e1[1] * e2[0];
                const Real len = Kokkos::sqrt(
                    raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] );
                const Real area = Real( 0.5 ) * len;
                fa( f ) = area;
                // np.divide(..., where=lengths > 0) -- a zero-length raw normal
                // yields the zero vector, not a NaN.
                for ( int d = 0; d < 3; ++d )
                    fn( f, d ) = ( len > Real( 0 ) ) ? raw[d] / len : Real( 0 );

                const Real third = area / Real( 3 );
                const int corner[3] = { ia, ib, ic };
                for ( int k = 0; k < 3; ++k )
                {
                    Kokkos::atomic_add( &va( corner[k] ), third );
                    for ( int d = 0; d < 3; ++d )
                        Kokkos::atomic_add( &vn( corner[k], d ),
                                            area * fn( f, d ) );
                }
            } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "beatnik_mesh_geometry_vertices",
            Kokkos::RangePolicy<execution_space>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                // np.maximum(vertex_area, 1e-300): floored, not branched, so a
                // collapsed one-ring gives a huge finite value not a NaN.
                if ( va( i ) < Real( 1.0e-300 ) )
                    va( i ) = Real( 1.0e-300 );
                const Real len = Kokkos::sqrt( vn( i, 0 ) * vn( i, 0 ) +
                                               vn( i, 1 ) * vn( i, 1 ) +
                                               vn( i, 2 ) * vn( i, 2 ) );
                if ( len > Real( 0 ) )
                    for ( int d = 0; d < 3; ++d )
                        vn( i, d ) /= len;
            } );
        Kokkos::fence();
    }
};

//---------------------------------------------------------------------------//
/**
 * @brief Stateless geometric kernels on `(vertices, faces)`.
 *
 * Free-standing because they are used by the AMR indicators, the remesher and
 * the diagnostics as well as by the solver, and none of them needs the bundled
 * `MeshGeometry`.
 */
class SurfaceOperators
{
  public:
    /**
     * @brief Scale-free triangle quality in [0, 1].
     *
     * Port of mesh.py::triangle_quality (lines 101-124)
     *
     * \f[
     *   q_f = \frac{4\sqrt{3}\,A_f}{\ell_{ab}^2+\ell_{bc}^2+\ell_{ca}^2}
     * \f]
     * Exactly 1 for an equilateral triangle (the \f$4\sqrt3\f$ is what
     * normalizes it there) and \f$\to 0\f$ for any degenerate triangle,
     * whether a needle or a cap. Dimensionless — the ratio of an area to a sum
     * of squared lengths — which is what makes it usable as an absolute
     * threshold across a mesh whose element sizes span two decades.
     *
     * @param vertices `(Nv,3)` positions.
     * @param faces    `(Nf,3)` local vertex indices. A face carrying a `-1`
     *                 corner scores 0, the same as a collapsed triangle: it is
     *                 not a valid triangle here, and 0 is the value a
     *                 quality-based consumer already knows how to handle.
     * @param[out] quality `(Nf,)` result. Zero where the squared-length sum is
     *             zero (a fully collapsed triangle).
     *
     * @note The execution space is deduced from `quality` rather than being a
     *       class template parameter: `SurfaceOperators` is stateless and its
     *       callers are, so there is no natural place to carry one. Same for
     *       every routine below.
     */
    template <class VertexView, class FaceView, class ScalarView>
    static void triangleQuality( const VertexView& vertices,
                                 const FaceView& faces, ScalarView& quality )
    {
        using exec = typename ScalarView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto q = quality;
        Kokkos::parallel_for(
            "beatnik_triangle_quality", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const int ia = fv( f, 0 );
                const int ib = fv( f, 1 );
                const int ic = fv( f, 2 );
                if ( ia < 0 || ib < 0 || ic < 0 )
                {
                    q( f ) = Real( 0 );
                    return;
                }
                Real ab[3], bc[3], ca[3], ac[3];
                for ( int d = 0; d < 3; ++d )
                {
                    ab[d] = pos( ib, d ) - pos( ia, d );
                    bc[d] = pos( ic, d ) - pos( ib, d );
                    ca[d] = pos( ia, d ) - pos( ic, d );
                    ac[d] = pos( ic, d ) - pos( ia, d );
                }
                const Real nx = ab[1] * ac[2] - ab[2] * ac[1];
                const Real ny = ab[2] * ac[0] - ab[0] * ac[2];
                const Real nz = ab[0] * ac[1] - ab[1] * ac[0];
                const Real area =
                    Real( 0.5 ) * Kokkos::sqrt( nx * nx + ny * ny + nz * nz );
                Real l2 = 0;
                for ( int d = 0; d < 3; ++d )
                    l2 += ab[d] * ab[d] + bc[d] * bc[d] + ca[d] * ca[d];
                // 4*sqrt(3) normalizes an equilateral triangle to exactly 1.
                q( f ) =
                    ( l2 > Real( 0 ) )
                        ? Real( 4.0 ) * Kokkos::sqrt( Real( 3.0 ) ) * area / l2
                        : Real( 0 );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Risk R12's two shape signals, reduced over a face range.
     *
     * \f[
     *   \frac{r}{R} = \frac{8A^2}{(a+b+c)\,abc}
     * \f]
     * — `0.5` for an equilateral triangle, `0` for a degenerate one, and the
     * reciprocal of twice Tessera's published \f$Q = R/2r\f$, so Beatnik's
     * diagnostic and Tessera's depth study compare without a conversion at the
     * call site.
     *
     * **Single-sourced at T4b, and the reason is that it is now measured by two
     * unrelated passes.** T4a computed it inline inside
     * `AdaptiveMesh::measureShape`; T4b's remesher needs exactly the same
     * number, per pass, compared against the same twelve-significant-digit
     * Python literals. Two copies of a formula that a gold set is asserted
     * against is one copy too many, so the kernel moved here verbatim —
     * identical arithmetic in identical order, which is why T4a's compiled-in
     * literals did not move.
     *
     * A face with a `-1` corner is **skipped entirely** (it contributes to
     * neither the minimum nor the tail count), which is what a partial ghost
     * triangle deserves and what T4a's kernel already did.
     *
     * @param face_count Rows to reduce. Pass the **OWNED** count: each local
     *        face is owned exactly once globally, so a ghost would be counted
     *        twice in the tail population (risk R9).
     * @param tail Ratio below which a face is counted, `0.25` everywhere in
     *        Beatnik (`AdaptiveMesh::quality_tail_threshold`).
     * @param[out] local_min Minimum over this rank's range, `1e300` if empty.
     * @param[out] local_below Tail population on this rank.
     *
     * @note MPI. **None** — the caller reduces, with `MPI_MIN` and `MPI_SUM`
     *       respectively, because the two consumers report into different
     *       diagnostics structs.
     */
    template <class ExecSpace, class VertexView, class FaceView>
    static void radiusRatioStats( const VertexView& vertices,
                                  const FaceView& faces, int face_count,
                                  Real tail, Real& local_min,
                                  long long& local_below )
    {
        auto pos = vertices;
        auto fv = faces;
        const Real cut = tail;
        Real mn_out = Real( 1.0e300 );
        long long below_out = 0;
        Kokkos::parallel_reduce(
            "beatnik_radius_ratio_stats",
            Kokkos::RangePolicy<ExecSpace>( 0, face_count ),
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
                const Real ratio = ( den > Real( 0 ) )
                                       ? Real( 8 ) * area * area / den
                                       : Real( 0 );
                if ( ratio < mn )
                    mn = ratio;
                if ( ratio < cut )
                    ++cnt;
            },
            Kokkos::Min<Real>( mn_out ), below_out );
        local_min = mn_out;
        local_below = below_out;
    }

    /**
     * @brief Length of every edge in the supplied edge list.
     *
     * Port of run_adaptive_mesh_bubble.py::mesh_edge_lengths (lines 545-555)
     *
     * Feeds the global minimum edge length, which is the adaptive-dt throttle
     * and the unit in which the proximity activation and material-exclusion
     * radii are expressed.
     *
     * **M1-REWORK — this takes EDGES, not faces.** The Python builds a
     * `set` of sorted endpoint pairs out of the face array, because a NumPy
     * triangle soup has no edge list; deriving the unique edge set is the whole
     * body of that function. Tessera maintains the unique edge set as a
     * first-class entity kind, continuously, through every topology op, so
     * rederiving it here would be reimplementing storage the mesh already has —
     * and doing it on device, where a hash set is exactly what one does not
     * want. `SurfaceMesh::edgeVertices()` is the list.
     *
     * @param vertices `(Nv,3)` positions.
     * @param edges    `(Ne,2)` local vertex indices per edge — the unique-edge
     *                 list. Pass the **OWNED** range,
     *                 `subview(edgeVertices(), (0, ownedEdgeCount()), ALL)`:
     *                 owned edges form a global partition, so this is what
     *                 makes the subsequent reduction a partition and not a
     *                 double count (risk R9). Passing the whole local range
     *                 does not change a *minimum*, but it does change a sum or
     *                 a histogram, so the owned range is the contract.
     * @param[out] lengths `(Ne,)` Euclidean length of each edge. An edge with a
     *                 `-1` endpoint yields `+inf`, so it cannot win a minimum;
     *                 that is unreachable for an owned edge and is a guard
     *                 against indexing at -1, not a modelled case.
     *
     * @note MPI. The subsequent minimum is `Comm::allReduceMin`, since every
     *       rank must throttle dt identically. `MPI_MIN` is reproducible across
     *       rank counts, unlike the volume sum.
     */
    template <class VertexView, class EdgeView, class ScalarView>
    static void edgeLengths( const VertexView& vertices, const EdgeView& edges,
                             ScalarView& lengths )
    {
        using exec = typename ScalarView::execution_space;
        const int ne = static_cast<int>( edges.extent( 0 ) );
        auto pos = vertices;
        auto ev = edges;
        auto out = lengths;
        Kokkos::parallel_for(
            "beatnik_edge_lengths", Kokkos::RangePolicy<exec>( 0, ne ),
            KOKKOS_LAMBDA( const int e ) {
                const int i0 = ev( e, 0 );
                const int i1 = ev( e, 1 );
                if ( i0 < 0 || i1 < 0 )
                {
                    out( e ) = Real( 1.0e300 );
                    return;
                }
                Real s = 0;
                for ( int d = 0; d < 3; ++d )
                {
                    const Real dx = pos( i1, d ) - pos( i0, d );
                    s += dx * dx;
                }
                out( e ) = Kokkos::sqrt( s );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Shortest and longest edge of each face.
     *
     * Port of mesh.py::face_max_edge_lengths (lines 88-98) and
     * mesh_solver.py::_face_min_edge_lengths (lines 1638-1648)
     *
     * `h_max` sets the sagitta estimate; `h_min` gates the `--min-refine-edge`
     * floor that stops refinement from chasing an already-resolved feature.
     *
     * These are the *face's own* three edges, so unlike `edgeLengths` there is
     * no owned/ghost question and no double counting: the answer is per face.
     *
     * @param[out] h_min `(Nf,)` shortest edge of each face; `0` for a face with
     *             a `-1` corner.
     * @param[out] h_max `(Nf,)` longest edge of each face; `0` likewise.
     */
    template <class VertexView, class FaceView, class ScalarView>
    static void faceEdgeExtents( const VertexView& vertices,
                                 const FaceView& faces, ScalarView& h_min,
                                 ScalarView& h_max )
    {
        using exec = typename ScalarView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto lo = h_min;
        auto hi = h_max;
        Kokkos::parallel_for(
            "beatnik_face_edge_extents", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const int c[3] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
                if ( c[0] < 0 || c[1] < 0 || c[2] < 0 )
                {
                    lo( f ) = Real( 0 );
                    hi( f ) = Real( 0 );
                    return;
                }
                Real mn = Real( 0 ), mx = Real( 0 );
                for ( int k = 0; k < 3; ++k )
                {
                    const int i0 = c[k];
                    const int i1 = c[( k + 1 ) % 3];
                    Real s = 0;
                    for ( int d = 0; d < 3; ++d )
                    {
                        const Real dx = pos( i1, d ) - pos( i0, d );
                        s += dx * dx;
                    }
                    const Real l = Kokkos::sqrt( s );
                    if ( k == 0 || l < mn )
                        mn = l;
                    if ( k == 0 || l > mx )
                        mx = l;
                }
                lo( f ) = mn;
                hi( f ) = mx;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Signed volume enclosed by the closed surface.
     *
     * Port of run_adaptive_mesh_bubble.py::mesh_enclosed_volume
     * (lines 1036-1040)
     *
     * The divergence-theorem form, summed over triangles with the origin as
     * apex:
     * \f[
     *   V = \frac{1}{6}\sum_f \; a_f \cdot ( b_f \times c_f )
     * \f]
     * where \f$(a_f,b_f,c_f)\f$ are the face's three vertex *positions* (not
     * edge vectors). Positive for an outward-oriented closed surface; a
     * negative result means the connectivity is inward-wound, which is a bug,
     * not a sign convention to absorb downstream. Units of length^3. Note the
     * apex is the coordinate origin, not the bubble centre — the result is
     * independent of that choice **only** if the surface is genuinely closed,
     * which makes this a useful closure check in its own right.
     *
     * @param vertices `(Nv,3)` positions.
     * @param faces    `(Nf,3)` local vertex indices. Pass the **OWNED** range —
     *                 `subview(faceVertices(), (0, ownedFaceCount()), ALL)` —
     *                 because a ghost face is an owned face on another rank and
     *                 would be counted twice (risk R9). The routine sums
     *                 whatever range it is handed; it cannot tell.
     * @return This rank's **partial** sum, already divided by 6. The caller
     *         reduces: `Comm::allReduceSum( comm, enclosedVolume( ... ) )`.
     *         Returning the partial rather than reducing internally is what
     *         lets a caller batch several reductions into one collective, which
     *         `Beatnik_VolumeProjection.hpp` needs for its Rayleigh quotient.
     *
     * @note MPI. Owned faces only, then `Comm::allReduceSum`. That sum is
     *       floating point, so it is **not** bitwise reproducible across rank
     *       counts — risk R2, and this is one of the two quantities the whole
     *       run keys off.
     */
    template <class VertexView, class FaceView>
    static Real enclosedVolume( const VertexView& vertices,
                                const FaceView& faces )
    {
        using exec = typename FaceView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        Real total = 0;
        Kokkos::parallel_reduce(
            "beatnik_enclosed_volume", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f, Real& acc ) {
                const int ia = fv( f, 0 );
                const int ib = fv( f, 1 );
                const int ic = fv( f, 2 );
                if ( ia < 0 || ib < 0 || ic < 0 )
                    return;
                const Real ax = pos( ia, 0 );
                const Real ay = pos( ia, 1 );
                const Real az = pos( ia, 2 );
                const Real bx = pos( ib, 0 );
                const Real by = pos( ib, 1 );
                const Real bz = pos( ib, 2 );
                const Real cx = pos( ic, 0 );
                const Real cy = pos( ic, 1 );
                const Real cz = pos( ic, 2 );
                acc += ax * ( by * cz - bz * cy ) + ay * ( bz * cx - bx * cz ) +
                       az * ( bx * cy - by * cx );
            },
            total );
        return total / Real( 6.0 );
    }

    /**
     * @brief Gradient of the enclosed volume with respect to vertex positions.
     *
     * Port of run_adaptive_mesh_bubble.py::mesh_volume_gradient
     * (lines 1043-1051) and mesh_solver.py::_mesh_volume_gradient
     * (lines 272-282)
     *
     * Differentiating the volume sum above,
     * \f[
     *   \frac{\partial V}{\partial a_f} = \frac{b_f\times c_f}{6},\quad
     *   \frac{\partial V}{\partial b_f} = \frac{c_f\times a_f}{6},\quad
     *   \frac{\partial V}{\partial c_f} = \frac{a_f\times b_f}{6},
     * \f]
     * accumulated onto the three vertices of every face. The result is
     * \f$\tfrac13 A_v \hat n_v\f$ to leading order — it is, up to that factor,
     * the outward area vector at each vertex. Units of length^2.
     *
     * This vector is the whole basis of volume conservation: both the
     * instantaneous flux removal (`Beatnik_VolumeProjection.hpp`) and the
     * iterative position projection are rank-one corrections along it.
     *
     * @param faces `(Nf,3)` local vertex indices. Pass the **WHOLE LOCAL** set
     *        here, not the owned range — this is a per-vertex assembly, not a
     *        global sum, and the opposite convention to `enclosedVolume` for
     *        the reason set out under DISTRIBUTED ASSEMBLY in the file header.
     *        The two are easy to transpose and the symptom of transposing them
     *        is a rank-count-dependent seam, so they are stated on both.
     * @param[in,out] gradient `(Nv,3)`. **Zeroed by this routine**, then
     *        accumulated into, so a caller cannot accidentally add two frames
     *        together.
     *
     * @note MPI. **None** — no scatter-add. The local-face loop already gives
     *       every owned vertex its complete incident-face set; a scatter-add
     *       would double-count. Ghost rows hold partial sums.
     */
    template <class VertexView, class FaceView, class VectorView>
    static void volumeGradient( const VertexView& vertices,
                                const FaceView& faces, VectorView& gradient )
    {
        using exec = typename VectorView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto g = gradient;
        Kokkos::deep_copy( gradient, Real( 0 ) );
        Kokkos::parallel_for(
            "beatnik_volume_gradient", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const int idx[3] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
                if ( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 )
                    return;
                Real p[3][3];
                for ( int k = 0; k < 3; ++k )
                    for ( int d = 0; d < 3; ++d )
                        p[k][d] = pos( idx[k], d );
                // dV/dp_k = (p_{k+1} x p_{k+2}) / 6, cyclically.
                for ( int k = 0; k < 3; ++k )
                {
                    const Real* u = p[( k + 1 ) % 3];
                    const Real* v = p[( k + 2 ) % 3];
                    const Real cr[3] = { u[1] * v[2] - u[2] * v[1],
                                         u[2] * v[0] - u[0] * v[2],
                                         u[0] * v[1] - u[1] * v[0] };
                    for ( int d = 0; d < 3; ++d )
                        Kokkos::atomic_add( &g( idx[k], d ),
                                            cr[d] / Real( 6.0 ) );
                }
            } );
        Kokkos::fence();
    }

    /**
     * @brief The in-plane gradient on one face, on device.
     *
     * Port of mesh_solver.py::_face_scalar_gradient (lines 938-961), the body
     * of the loop.
     *
     * Factored out because `faceScalarGradient` and `surfaceGradient` are the
     * *same* discretization and the second must not drift from the first:
     * `surfaceGradient` fuses this into its own face loop rather than
     * materializing an `(Nf,3)` temporary it would immediately reduce away, so
     * without a shared kernel the 2x2 Gram solve would exist twice. The
     * equations and the degeneracy rule are documented on
     * `faceScalarGradient` below; this is that body and nothing more.
     *
     * @param f Local face index. A `-1` corner yields a zero gradient.
     * @param[out] g The three components.
     */
    template <class VertexView, class FaceView, class ScalarView>
    KOKKOS_INLINE_FUNCTION static void
    faceGradient( const VertexView& vertices, const FaceView& faces,
                  const ScalarView& scalar, int f, Real g[3] )
    {
        const int ia = faces( f, 0 );
        const int ib = faces( f, 1 );
        const int ic = faces( f, 2 );
        if ( ia < 0 || ib < 0 || ic < 0 )
        {
            for ( int d = 0; d < 3; ++d )
                g[d] = Real( 0 );
            return;
        }
        Real e1[3], e2[3];
        for ( int d = 0; d < 3; ++d )
        {
            e1[d] = vertices( ib, d ) - vertices( ia, d );
            e2[d] = vertices( ic, d ) - vertices( ia, d );
        }
        const Real d1 = scalar( ib ) - scalar( ia );
        const Real d2 = scalar( ic ) - scalar( ia );
        Real a = 0, b = 0, c = 0;
        for ( int d = 0; d < 3; ++d )
        {
            a += e1[d] * e1[d];
            b += e1[d] * e2[d];
            c += e2[d] * e2[d];
        }
        const Real det = a * c - b * b;
        Real c1 = 0, c2 = 0;
        // np.divide(..., where=np.abs(det) > 1.0e-300): a degenerate face gives
        // the zero gradient, not a NaN and not a huge finite value. Note this
        // is the one place the file header's "floored, not branched" rule does
        // NOT apply -- the reference branches here, and flooring instead would
        // turn a collapsed triangle into a 1e300 gradient that poisons the
        // area-weighted average of every vertex on it.
        if ( ( det > Real( 1.0e-300 ) ) || ( det < Real( -1.0e-300 ) ) )
        {
            c1 = ( d1 * c - d2 * b ) / det;
            c2 = ( a * d2 - b * d1 ) / det;
        }
        for ( int d = 0; d < 3; ++d )
            g[d] = c1 * e1[d] + c2 * e2[d];
    }

    /**
     * @brief Per-face gradient of a per-vertex scalar, in the face plane.
     *
     * Port of mesh_solver.py::_face_scalar_gradient (lines 938-961)
     *
     * On the triangle with origin \f$p_0\f$ and edge vectors
     * \f$e_1 = p_1-p_0\f$, \f$e_2 = p_2-p_0\f$, the in-plane gradient is the
     * unique \f$g = c_1 e_1 + c_2 e_2\f$ satisfying \f$g\cdot e_1 = \Delta_1\f$
     * and \f$g\cdot e_2 = \Delta_2\f$, with
     * \f$\Delta_m = \phi(p_m) - \phi(p_0)\f$. Solving the 2x2 Gram system with
     * \f$a=e_1\!\cdot\!e_1\f$, \f$b=e_1\!\cdot\!e_2\f$, \f$c=e_2\!\cdot\!e_2\f$,
     * \f$\det = ac-b^2\f$:
     * \f[
     *   c_1 = \frac{\Delta_1 c - \Delta_2 b}{\det}, \qquad
     *   c_2 = \frac{a \Delta_2 - b \Delta_1}{\det}.
     * \f]
     * A degenerate face (\f$|\det| \le 10^{-300}\f$) yields a zero gradient
     * rather than a NaN. The result is exactly tangent to the face by
     * construction, so no projection is needed at this level.
     *
     * Units: [scalar]/length.
     */
    template <class VertexView, class FaceView, class ScalarView,
              class VectorView>
    static void faceScalarGradient( const VertexView& vertices,
                                    const FaceView& faces,
                                    const ScalarView& scalar,
                                    VectorView& gradient )
    {
        using exec = typename VectorView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto s = scalar;
        auto g = gradient;
        Kokkos::parallel_for(
            "beatnik_face_scalar_gradient", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                Real gf[3];
                faceGradient( pos, fv, s, f, gf );
                for ( int d = 0; d < 3; ++d )
                    g( f, d ) = gf[d];
            } );
        Kokkos::fence();
    }

    /**
     * @brief Per-vertex surface gradient of a per-vertex scalar.
     *
     * Port of mesh_solver.py::surface_gradient (lines 964-986)
     *
     * Area-weighted average of the incident face gradients, then projected onto
     * the vertex tangent plane:
     * \f[
     *   g_v = \frac{\sum_{f\ni v} A_f\, g_f}{\sum_{f\ni v} A_f}, \qquad
     *   \nabla_s\phi\,(v) = g_v - (g_v\cdot \hat n_v)\,\hat n_v .
     * \f]
     * The projection matters: the area-weighted average of tangent vectors from
     * differently-oriented faces is not itself tangent, and the leftover normal
     * component would appear in the sheet vector as a spurious normal
     * circulation.
     *
     * This routine is called **twice per RHS evaluation** — once to build the
     * sheet vector from the potential, and once on the Bernoulli potential —
     * so it is a two-ring stencil overall. That is served by building the mesh
     * at `SurfaceMesh::halo_depth = 2` once at setup, **not** by exchanging
     * twice — see the halo section of `Beatnik_MeshInterface.hpp` and risk R8.
     *
     * @note MPI (CORRECTED at the M1 rework; T2b implements this). Two
     *       face-loop scatters (weighted gradient and weight), assembled from
     *       the **whole local face set** and therefore already complete on
     *       every owned vertex — see DISTRIBUTED ASSEMBLY in the file header.
     *       **No scatter-add**, which would double-count. Only a `haloExchange`
     *       of the *input* scalar is needed, so ghost values are current.
     *
     * @param faces    `(Nf,3)` local vertex indices — the **WHOLE LOCAL** set.
     * @param scalar   Per-vertex input. May be a Cabana slice with no extent
     *                 (`mesh.potential()` is exactly that), which is why `Nv`
     *                 is taken from `gradient` and not from here.
     * @param[out] gradient `(Nv,3)`. Its extent defines `Nv`. **Zeroed by this
     *                 routine** before assembly, so a caller cannot
     *                 accidentally accumulate two frames.
     *
     * @note The face area used as the weight is recomputed here rather than
     *       taken from `MeshGeometry::face_area`. It is the same expression
     *       (`0.5*|e1 x e2|`, `mesh.py::face_areas` 81-85) evaluated on the
     *       same positions, so it is bit-identical to that array; passing the
     *       array instead would add a required argument, and a stale one is a
     *       worse failure than a recomputation.
     */
    template <class VertexView, class FaceView, class ScalarView,
              class VectorView, class NormalView>
    static void surfaceGradient( const VertexView& vertices,
                                 const FaceView& faces,
                                 const ScalarView& scalar,
                                 const NormalView& vertex_normal,
                                 VectorView& gradient )
    {
        using exec = typename VectorView::execution_space;
        using mem = typename VectorView::memory_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        const int nv = static_cast<int>( gradient.extent( 0 ) );

        auto pos = vertices;
        auto fv = faces;
        auto s = scalar;
        auto vn = vertex_normal;
        auto g = gradient;
        // The denominator of the area-weighted average, assembled alongside the
        // numerator so the two see exactly the same face set.
        Kokkos::View<Real*, mem> weight( "beatnik_surface_gradient_weight", nv );
        Kokkos::deep_copy( gradient, Real( 0 ) );

        Kokkos::parallel_for(
            "beatnik_surface_gradient_faces", Kokkos::RangePolicy<exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                const int ia = fv( f, 0 );
                const int ib = fv( f, 1 );
                const int ic = fv( f, 2 );
                if ( ia < 0 || ib < 0 || ic < 0 )
                    return;
                Real gf[3];
                faceGradient( pos, fv, s, f, gf );

                Real e1[3], e2[3];
                for ( int d = 0; d < 3; ++d )
                {
                    e1[d] = pos( ib, d ) - pos( ia, d );
                    e2[d] = pos( ic, d ) - pos( ia, d );
                }
                const Real cr[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                                     e1[2] * e2[0] - e1[0] * e2[2],
                                     e1[0] * e2[1] - e1[1] * e2[0] };
                const Real area =
                    Real( 0.5 ) * Kokkos::sqrt( cr[0] * cr[0] + cr[1] * cr[1] +
                                                cr[2] * cr[2] );

                const int corner[3] = { ia, ib, ic };
                for ( int k = 0; k < 3; ++k )
                {
                    Kokkos::atomic_add( &weight( corner[k] ), area );
                    for ( int d = 0; d < 3; ++d )
                        Kokkos::atomic_add( &g( corner[k], d ), area * gf[d] );
                }
            } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "beatnik_surface_gradient_vertices",
            Kokkos::RangePolicy<exec>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
                // np.divide(..., where=weights > 0): a vertex with no incident
                // face gets the zero gradient rather than a NaN.
                if ( weight( i ) > Real( 0 ) )
                    for ( int d = 0; d < 3; ++d )
                        g( i, d ) /= weight( i );
                else
                    for ( int d = 0; d < 3; ++d )
                        g( i, d ) = Real( 0 );

                // The projection. The area-weighted average of tangent vectors
                // from differently-tilted faces is not itself tangent, and the
                // leftover normal component would become a spurious normal
                // circulation in the sheet vector.
                Real dot = 0;
                for ( int d = 0; d < 3; ++d )
                    dot += g( i, d ) * vn( i, d );
                for ( int d = 0; d < 3; ++d )
                    g( i, d ) -= dot * vn( i, d );
            } );
        Kokkos::fence();
    }

    /**
     * @brief The cotangent of each of a face's three corner angles, on device.
     *
     * Port of mesh_solver.py::cotangent_laplacian_scalars's `_cot`
     * (lines 1046-1050)
     *
     * \f$\cot\theta_p = (u\cdot w)/\|u\times w\|\f$ with \f$u = q-p\f$,
     * \f$w = r-p\f$ — numerically better than \f$\cos/\sin\f$, and the
     * denominator is **floored** at 1e-300 rather than branched, so a collapsed
     * triangle gives a huge finite weight and not a NaN.
     *
     * Factored out for the same reason as `faceGradient`:
     * `cotangentLaplacianScalar` and `meanCurvatureNormal` are the same
     * operator applied to a scalar and to the positions, and the corner-to-edge
     * pairing is the sign-critical part of both. One kernel, one place to be
     * wrong.
     *
     * @param[out] cot `cot[k]` is the cotangent at corner `k`, which weights
     *             the **opposite** edge. Zero for a face with a `-1` corner.
     */
    template <class VertexView, class FaceView>
    KOKKOS_INLINE_FUNCTION static void
    faceCotangents( const VertexView& vertices, const FaceView& faces, int f,
                    Real cot[3] )
    {
        const int idx[3] = { faces( f, 0 ), faces( f, 1 ), faces( f, 2 ) };
        if ( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 )
        {
            for ( int k = 0; k < 3; ++k )
                cot[k] = Real( 0 );
            return;
        }
        for ( int k = 0; k < 3; ++k )
        {
            const int p = idx[k];
            const int q = idx[( k + 1 ) % 3];
            const int r = idx[( k + 2 ) % 3];
            Real u[3], w[3];
            for ( int d = 0; d < 3; ++d )
            {
                u[d] = vertices( q, d ) - vertices( p, d );
                w[d] = vertices( r, d ) - vertices( p, d );
            }
            const Real cr[3] = { u[1] * w[2] - u[2] * w[1],
                                 u[2] * w[0] - u[0] * w[2],
                                 u[0] * w[1] - u[1] * w[0] };
            Real denom = Kokkos::sqrt( cr[0] * cr[0] + cr[1] * cr[1] +
                                       cr[2] * cr[2] );
            if ( denom < Real( 1.0e-300 ) )
                denom = Real( 1.0e-300 );
            Real dot = 0;
            for ( int d = 0; d < 3; ++d )
                dot += u[d] * w[d];
            cot[k] = dot / denom;
        }
    }

    /**
     * @brief Area-normalized cotangent Laplace-Beltrami of a per-vertex scalar.
     *
     * Port of mesh_solver.py::cotangent_laplacian_scalars (lines 1020-1059)
     *
     * \f[
     *   (\Delta_s \phi)_i = \frac{1}{A_i}\sum_{j\in N(i)} w_{ij}(\phi_j-\phi_i),
     *   \qquad w_{ij} = \tfrac12(\cot\alpha_{ij} + \cot\beta_{ij}),
     * \f]
     * where \f$\alpha,\beta\f$ are the two angles opposite edge \f$(i,j)\f$.
     * Assembled per face: the angle at vertex 0 contributes \f$\tfrac12\cot\f$
     * to the edge (1,2), and cyclically. The cotangent at a vertex \f$p\f$ with
     * neighbors \f$q,r\f$ is computed as
     * \f$(u\cdot w)/\|u\times w\|\f$ with \f$u=q-p, w=r-p\f$ — numerically
     * better than \f$\cos/\sin\f$ and floored at 1e-300 in the denominator.
     *
     * **Sign convention:** with this sign, \f$\mu\Delta_s\phi\f$ with
     * \f$\mu>0\f$ is *dissipative* — it drives \f$\phi\f$ toward its local
     * average. The opposite sign is anti-diffusive and blows up immediately,
     * which is the fast way to detect getting it wrong.
     *
     * **Why not the graph Laplacian:** the umbrella stencil scales like
     * \f$h^2\Delta\f$, so its diffusive effect *weakens* exactly where the mesh
     * is refined — i.e. where the sheet spike lives and the diffusion is
     * needed. The cotangent form is a true \f$\Delta\f$ and does not have that
     * defect. `--viscosity-mode graph` selects the old behavior.
     *
     * Units: [scalar]/length^2.
     *
     * @note MPI (CORRECTED at the M1 rework; T2b implements this). Per-face
     *       scatter with `+=` onto both endpoints of each edge, over the
     *       **whole local face set**, and therefore already complete on every
     *       owned vertex before the division by the vertex area — see
     *       DISTRIBUTED ASSEMBLY in the file header. **No scatter-add.** Only
     *       the input `values` need a preceding `haloExchange`.
     *
     * @param values Per-vertex input. **T2b — its type is now independent of
     *        `result`'s.** The pre-T2b declaration used one `ScalarView` for
     *        both, which cannot express the call the viscous term actually
     *        makes: `values` is `mesh.potential()`, a Cabana slice, and
     *        `result` is a Beatnik-owned `Kokkos::View`. A widening, so every
     *        conceivable pre-T2b call still compiles.
     * @param[out] result `(Nv,)`. Its extent defines `Nv`; **zeroed here**.
     */
    template <class VertexView, class FaceView, class ScalarView,
              class AreaView, class OutScalarView>
    static void cotangentLaplacianScalar( const VertexView& vertices,
                                          const FaceView& faces,
                                          const ScalarView& values,
                                          const AreaView& vertex_area,
                                          OutScalarView& result )
    {
        using exec = typename OutScalarView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        const int nv = static_cast<int>( result.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto val = values;
        auto va = vertex_area;
        auto out = result;
        Kokkos::deep_copy( result, Real( 0 ) );

        Kokkos::parallel_for(
            "beatnik_cotangent_laplacian_faces",
            Kokkos::RangePolicy<exec>( 0, nf ), KOKKOS_LAMBDA( const int f ) {
                const int idx[3] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
                if ( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 )
                    return;
                Real cot[3];
                faceCotangents( pos, fv, f, cot );
                // The angle at corner k weights the OPPOSITE edge, with the
                // half from this face's side of it: cot0 -> (1,2), cot1 ->
                // (0,2), cot2 -> (0,1). Getting this pairing wrong still gives
                // a symmetric operator, so it does not blow up -- it just is
                // not the Laplacian.
                for ( int k = 0; k < 3; ++k )
                {
                    const int p = idx[( k + 1 ) % 3];
                    const int q = idx[( k + 2 ) % 3];
                    const Real w = Real( 0.5 ) * cot[k];
                    Kokkos::atomic_add( &out( p ), w * ( val( q ) - val( p ) ) );
                    Kokkos::atomic_add( &out( q ), w * ( val( p ) - val( q ) ) );
                }
            } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "beatnik_cotangent_laplacian_normalize",
            Kokkos::RangePolicy<exec>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
                // np.maximum(vertex_area, 1.0e-300): floored, not branched.
                const Real a = ( va( i ) > Real( 1.0e-300 ) ) ? va( i )
                                                              : Real( 1.0e-300 );
                out( i ) /= a;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Uniform graph ("umbrella") Laplacian of a per-vertex scalar.
     *
     * Port of mesh_solver.py::graph_laplacian_scalars (lines 1004-1017)
     *
     * \f$(\mathcal{L}\phi)_i = \frac{1}{|N(i)|}\sum_{j\in N(i)}
     * (\phi_j - \phi_i)\f$ — the mean neighbor difference, *not* divided by any
     * length. It is therefore \f$O(h^2)\Delta\phi\f$ and dimensionless in
     * [scalar], which is why the `mu` coefficient means something different
     * under `--viscosity-mode graph` than under `laplace-beltrami`. Kept for
     * bit-comparability against Python runs that used it.
     *
     * **T2b SIGNATURE CHANGE — this takes the vertex one-ring CSR, not
     * `faces`.** The pre-T2b declaration was `(faces, values, vertex_count,
     * result)`, i.e. a per-face scatter. That cannot reproduce the reference:
     * the Python builds a `set` of neighbours per vertex and averages over the
     * **unique** neighbour set, while a per-face scatter visits every interior
     * neighbour **twice** (once from each of the two faces sharing that edge).
     * On a closed manifold the double count cancels between numerator and
     * denominator, so the two agree algebraically — but not bitwise, and the
     * cancellation argument holds only where every edge has exactly two
     * incident faces, which the reference never asserts and which a partially
     * held ghost row need not satisfy. `SurfaceMesh::vertexOneRing()` *is* the
     * unique set (`Tessera::buildVertexStencil( mesh, 1 )`, ascending unique
     * local indices), so the argument does not have to be made at all.
     *
     * @param one_ring `SurfaceMesh::vertexOneRing()` — CSR `offsets` /
     *        `neighbors`, self excluded. **Complete for every owned vertex; a
     *        ghost row may stop at the edge of the local set**, so only owned
     *        rows of `result` are meaningful. A vertex with an empty row gets
     *        zero, as the Python's `if not nbrs: continue` leaves it.
     * @param values Per-vertex input; may be a Cabana slice with no extent.
     * @param[out] result `(Nv,)`. Its extent defines the range written.
     */
    template <class CsrType, class ScalarView, class OutScalarView>
    static void graphLaplacianScalar( const CsrType& one_ring,
                                      const ScalarView& values,
                                      OutScalarView& result )
    {
        using exec = typename OutScalarView::execution_space;
        const int nv = static_cast<int>( result.extent( 0 ) );
        auto offsets = one_ring.offsets;
        auto neighbors = one_ring.neighbors;
        auto val = values;
        auto out = result;
        Kokkos::parallel_for(
            "beatnik_graph_laplacian_scalar",
            Kokkos::RangePolicy<exec>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
                const int begin = offsets( i );
                const int end = offsets( i + 1 );
                const int n = end - begin;
                if ( n <= 0 )
                {
                    out( i ) = Real( 0 );
                    return;
                }
                Real acc = 0;
                for ( int p = begin; p < end; ++p )
                    acc += val( neighbors( p ) ) - val( i );
                out( i ) = acc / static_cast<Real>( n );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Uniform graph Laplacian of a per-vertex vector.
     *
     * Port of mesh_solver.py::graph_laplacian_vectors (lines 989-1001)
     *
     * Componentwise `graphLaplacianScalar`. This is the viscous operator used
     * on the **sheet-vector** state (`mesh_rhs`, line 1231) — note the
     * asymmetry with the potential state, which uses the cotangent Laplacian by
     * default. That asymmetry is in the reference, not a porting slip.
     *
     * **T2b SIGNATURE CHANGE — takes the vertex one-ring CSR**, for exactly the
     * reason written out on `graphLaplacianScalar` above.
     */
    template <class CsrType, class VectorView, class OutVectorView>
    static void graphLaplacianVector( const CsrType& one_ring,
                                      const VectorView& values,
                                      OutVectorView& result )
    {
        using exec = typename OutVectorView::execution_space;
        const int nv = static_cast<int>( result.extent( 0 ) );
        auto offsets = one_ring.offsets;
        auto neighbors = one_ring.neighbors;
        auto val = values;
        auto out = result;
        Kokkos::parallel_for(
            "beatnik_graph_laplacian_vector",
            Kokkos::RangePolicy<exec>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
                const int begin = offsets( i );
                const int end = offsets( i + 1 );
                const int n = end - begin;
                if ( n <= 0 )
                {
                    for ( int d = 0; d < 3; ++d )
                        out( i, d ) = Real( 0 );
                    return;
                }
                for ( int d = 0; d < 3; ++d )
                {
                    Real acc = 0;
                    for ( int p = begin; p < end; ++p )
                        acc += val( neighbors( p ), d ) - val( i, d );
                    out( i, d ) = acc / static_cast<Real>( n );
                }
            } );
        Kokkos::fence();
    }

    /**
     * @brief Discrete mean-curvature normal \f$\Delta_{LB}x\f$ per vertex.
     *
     * Port of mesh_solver.py::mean_curvature_normal (lines 1068-1110)
     *
     * The cotangent Laplacian applied to the vertex *positions*. By the
     * Meyer-Desbrun-Schroeder-Barr identity,
     * \f[
     *   (\Delta_{LB} x)_i = -2 H_i\,\hat n_i,
     * \f]
     * with \f$H_i\f$ the mean curvature and \f$\hat n_i\f$ the **outward** unit
     * normal. So for a convex bump the vector points **inward**, toward the
     * centre of curvature, with magnitude \f$2H\f$. Sanity check: on a unit
     * sphere it returns \f$\approx -2\hat n_{\text{out}}\f$.
     *
     * Consequently the *area-decreasing* surface-tension flow is
     * \f$\dot x \mathrel{+}= \sigma\,\Delta_{LB}x\f$ with \f$\sigma>0\f$ — i.e.
     * adding this vector directly, with no sign flip. Writing
     * \f$-\sigma\Delta_{LB}x\f$ inflates the bubble instead of smoothing it.
     *
     * Units: 1/length.
     *
     * @note MPI. Same scatter pattern as `cotangentLaplacianScalar`.
     *
     * @param[out] result `(Nv,3)`. Its extent defines `Nv`; **zeroed here**.
     */
    template <class VertexView, class FaceView, class AreaView,
              class VectorView>
    static void meanCurvatureNormal( const VertexView& vertices,
                                     const FaceView& faces,
                                     const AreaView& vertex_area,
                                     VectorView& result )
    {
        using exec = typename VectorView::execution_space;
        const int nf = static_cast<int>( faces.extent( 0 ) );
        const int nv = static_cast<int>( result.extent( 0 ) );
        auto pos = vertices;
        auto fv = faces;
        auto va = vertex_area;
        auto out = result;
        Kokkos::deep_copy( result, Real( 0 ) );

        Kokkos::parallel_for(
            "beatnik_mean_curvature_normal_faces",
            Kokkos::RangePolicy<exec>( 0, nf ), KOKKOS_LAMBDA( const int f ) {
                const int idx[3] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
                if ( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 )
                    return;
                Real cot[3];
                faceCotangents( pos, fv, f, cot );
                // Identical assembly to cotangentLaplacianScalar, with the
                // vertex POSITIONS as the field: contribution w*(x_q - x_p)
                // onto p and its negation onto q. That antisymmetry is what
                // makes the result -2H n_out rather than +2H n_out, so it is
                // the sign the whole surface-tension term rests on.
                for ( int k = 0; k < 3; ++k )
                {
                    const int p = idx[( k + 1 ) % 3];
                    const int q = idx[( k + 2 ) % 3];
                    const Real w = Real( 0.5 ) * cot[k];
                    for ( int d = 0; d < 3; ++d )
                    {
                        const Real contrib = w * ( pos( q, d ) - pos( p, d ) );
                        Kokkos::atomic_add( &out( p, d ), contrib );
                        Kokkos::atomic_add( &out( q, d ), -contrib );
                    }
                }
            } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "beatnik_mean_curvature_normal_normalize",
            Kokkos::RangePolicy<exec>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
                const Real a = ( va( i ) > Real( 1.0e-300 ) ) ? va( i )
                                                              : Real( 1.0e-300 );
                for ( int d = 0; d < 3; ++d )
                    out( i, d ) /= a;
            } );
        Kokkos::fence();
    }

    /**
     * @brief Remove the normal component of a per-vertex vector field.
     *
     * Port of mesh_solver.py::_project_tangent (lines 247-256)
     *
     * \f$ v \leftarrow v - (v\cdot\hat n_v)\hat n_v \f$.
     *
     * The sheet vector is a tangential density by definition, but nothing in
     * the RK3 update preserves that: the stage combination of tangential
     * vectors at *different* geometries is not tangential at the new geometry.
     * The reference therefore re-projects on every state construction
     * (`MeshZModelState.__post_init__`, line 93) and again on the computed
     * `sheet_dot` (line 1232). Both projections are load-bearing; dropping
     * either lets a normal component accumulate that has no physical meaning
     * and feeds straight back into the Bernoulli term through
     * \f$|S|^2\f$.
     *
     * @param[in,out] values `(Nv,3)` field, modified in place. May be a Cabana
     *        slice with no extent (`mesh.sheetVector()` is one), which is why
     *        `Nv` and the execution space both come from `vertex_normal`.
     * @param vertex_normal `(Nv,3)` unit vertex normals. Rows where the normal
     *        is zero — a vertex with no incident face — are left untouched,
     *        which is what subtracting a zero projection does anyway.
     */
    template <class VectorView, class NormalView>
    static void projectTangent( VectorView& values,
                                const NormalView& vertex_normal )
    {
        using exec = typename NormalView::execution_space;
        const int nv = static_cast<int>( vertex_normal.extent( 0 ) );
        auto v = values;
        auto vn = vertex_normal;
        Kokkos::parallel_for(
            "beatnik_project_tangent", Kokkos::RangePolicy<exec>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                Real dot = 0;
                for ( int d = 0; d < 3; ++d )
                    dot += v( i, d ) * vn( i, d );
                for ( int d = 0; d < 3; ++d )
                    v( i, d ) -= dot * vn( i, d );
            } );
        Kokkos::fence();
    }

    /**
     * @brief Area-weighted mean of a per-vertex scalar.
     *
     * Port of mesh_solver.py::_area_weighted_scalar_mean (lines 239-244)
     *
     * \f$\bar\phi = \sum_v A_v\phi_v \big/ \sum_v A_v\f$, falling back to the
     * unweighted mean if the total area is non-positive.
     *
     * Used to pin the arbitrary additive constant of the velocity potential:
     * the potential is re-centred on construction (line 155-159) and its time
     * derivative is re-centred too (line 1264-1268). Without the second, the
     * mean of \f$\phi\f$ random-walks and the *magnitude* of \f$\phi\f$ grows
     * without bound even though the physically meaningful gradient does not —
     * eventually losing precision in the differences that actually matter.
     *
     * @param scalar      `(N,)` per-vertex values.
     * @param vertex_area `(N,)` per-vertex areas. Pass the **OWNED** range of
     *        both: this reduces to a global scalar, so a ghost vertex would be
     *        counted twice (risk R9).
     *
     * @note MPI. This overload divides **locally** and is correct only when the
     *       two ranges it is handed cover the whole surface — i.e. at one rank,
     *       or on a diagnostic that genuinely wants a per-rank mean. A
     *       distributed caller must use `areaWeightedMeanPartials` and reduce
     *       both sums before dividing; see below and `Comm::allReduceSum`.
     */
    template <class ScalarView, class AreaView>
    static Real areaWeightedMean( const ScalarView& scalar,
                                  const AreaView& vertex_area )
    {
        Real weighted = 0, area = 0;
        areaWeightedMeanPartials( scalar, vertex_area, weighted, area );
        if ( !( area > Real( 0 ) ) )
        {
            // The Python's fallback: the unweighted mean.
            const int n = static_cast<int>( scalar.extent( 0 ) );
            if ( n == 0 )
                return Real( 0 );
            using exec = typename AreaView::execution_space;
            auto s = scalar;
            Real sum = 0;
            Kokkos::parallel_reduce(
                "beatnik_unweighted_mean", Kokkos::RangePolicy<exec>( 0, n ),
                KOKKOS_LAMBDA( const int i, Real& acc ) { acc += s( i ); },
                sum );
            return sum / static_cast<Real>( n );
        }
        return weighted / area;
    }

    /**
     * @brief The two partial sums of the area-weighted mean, unreduced.
     *
     * Port of mesh_solver.py::_area_weighted_scalar_mean (lines 239-244)
     *
     * \f$\sum_v A_v\phi_v\f$ and \f$\sum_v A_v\f$ over whatever range is
     * handed in, with **no division**. This exists because a mean is not a
     * reducible quantity: `allReduceSum` of per-rank means is not the global
     * mean, and the only correct order is *reduce both sums, then divide*.
     * Returning a `Real` from `areaWeightedMean` cannot express that, which is
     * why the distributed path calls this instead.
     *
     * If each rank instead subtracted its *local* mean from the potential, the
     * potential would acquire a piecewise-constant jump across every partition
     * boundary and its surface gradient — the sheet vector — would pick up a
     * delta function there. That is the failure this signature exists to
     * prevent, and it is invisible at one rank.
     *
     * @param scalar      `(N,)` per-vertex values, **owned** range.
     * @param vertex_area `(N,)` per-vertex areas, **owned** range.
     * @param[out] weighted_sum \f$\sum_v A_v\phi_v\f$ on this rank.
     * @param[out] area_sum     \f$\sum_v A_v\f$ on this rank.
     *
     * @note MPI. Two `Comm::allReduceSum` calls, numerator and denominator,
     *       **both** before the division. They may be batched into one
     *       collective. See that function's note.
     */
    template <class ScalarView, class AreaView>
    static void areaWeightedMeanPartials( const ScalarView& scalar,
                                          const AreaView& vertex_area,
                                          Real& weighted_sum, Real& area_sum )
    {
        using exec = typename AreaView::execution_space;
        const int n = static_cast<int>( vertex_area.extent( 0 ) );
        auto s = scalar;
        auto a = vertex_area;
        Real w = 0, t = 0;
        Kokkos::parallel_reduce(
            "beatnik_area_weighted_mean", Kokkos::RangePolicy<exec>( 0, n ),
            KOKKOS_LAMBDA( const int i, Real& acc_w, Real& acc_t ) {
                acc_w += a( i ) * s( i );
                acc_t += a( i );
            },
            w, t );
        weighted_sum = w;
        area_sum = t;
    }
};

} // namespace Beatnik

#endif // BEATNIK_MESHGEOMETRY_HPP
