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
     * @param vertices `(Nv,3)` positions, ghosts included.
     * @param faces    `(Nf,3)` connectivity, ghosts included.
     *
     * @note MPI. The vertex scatter deposits onto ghost vertices, so this must
     *       be followed by `Comm::haloScatterAdd` on `vertex_area` and the
     *       *unnormalized* weighted normal, and the normalization applied only
     *       after that. Normalizing before the scatter-add gives boundary
     *       vertices a normal built from a partial one-ring.
     */
    template <class VertexView, class FaceView>
    void compute( const VertexView& vertices, const FaceView& faces )
    {
        (void)vertices;
        (void)faces;
        BEATNIK_NOT_IMPLEMENTED( "MeshGeometry", "compute" );
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
     * @param faces    `(Nf,3)` connectivity.
     * @param[out] quality `(Nf,)` result. Zero where the squared-length sum is
     *             zero (a fully collapsed triangle).
     */
    template <class VertexView, class FaceView, class ScalarView>
    static void triangleQuality( const VertexView& vertices,
                                 const FaceView& faces, ScalarView& quality )
    {
        (void)vertices;
        (void)faces;
        (void)quality;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "triangleQuality" );
    }

    /**
     * @brief Length of every unique edge.
     *
     * Port of run_adaptive_mesh_bubble.py::mesh_edge_lengths (lines 545-555)
     *
     * Edges are keyed by their sorted endpoint pair so each interior edge is
     * measured once, not twice. Feeds the global minimum edge length, which is
     * the adaptive-dt throttle and the unit in which the proximity activation
     * and material-exclusion radii are expressed.
     *
     * @note MPI. Only edges whose *lower-global-id* endpoint is owned may be
     *       counted, or boundary edges appear on both ranks. The subsequent
     *       minimum is an `MPI_Allreduce` with `MPI_MIN`
     *       (`Comm::allReduceMin`), since every rank must throttle dt
     *       identically.
     */
    template <class VertexView, class FaceView, class ScalarView>
    static void edgeLengths( const VertexView& vertices, const FaceView& faces,
                             ScalarView& lengths )
    {
        (void)vertices;
        (void)faces;
        (void)lengths;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "edgeLengths" );
    }

    /**
     * @brief Shortest and longest edge of each face.
     *
     * Port of mesh.py::face_max_edge_lengths (lines 88-98) and
     * mesh_solver.py::_face_min_edge_lengths (lines 1638-1648)
     *
     * `h_max` sets the sagitta estimate; `h_min` gates the `--min-refine-edge`
     * floor that stops refinement from chasing an already-resolved feature.
     */
    template <class VertexView, class FaceView, class ScalarView>
    static void faceEdgeExtents( const VertexView& vertices,
                                 const FaceView& faces, ScalarView& h_min,
                                 ScalarView& h_max )
    {
        (void)vertices;
        (void)faces;
        (void)h_min;
        (void)h_max;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "faceEdgeExtents" );
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
     * @note MPI. Owned faces only, then `Comm::allReduceSum`.
     */
    template <class VertexView, class FaceView>
    static Real enclosedVolume( const VertexView& vertices,
                                const FaceView& faces )
    {
        (void)vertices;
        (void)faces;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "enclosedVolume" );
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
     * @note MPI. Face-loop scatter onto vertices, so
     *       `Comm::haloScatterAdd` afterwards.
     */
    template <class VertexView, class FaceView, class VectorView>
    static void volumeGradient( const VertexView& vertices,
                                const FaceView& faces, VectorView& gradient )
    {
        (void)vertices;
        (void)faces;
        (void)gradient;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "volumeGradient" );
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
        (void)vertices;
        (void)faces;
        (void)scalar;
        (void)gradient;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "faceScalarGradient" );
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
     * so it is a two-ring stencil overall. See
     * `Comm::haloExchangeField` for what that costs in ghost depth.
     *
     * @note MPI. Two face-loop scatters (weighted gradient and weight), so
     *       `Comm::haloScatterAdd` on both before dividing.
     */
    template <class VertexView, class FaceView, class ScalarView,
              class VectorView, class NormalView>
    static void surfaceGradient( const VertexView& vertices,
                                 const FaceView& faces,
                                 const ScalarView& scalar,
                                 const NormalView& vertex_normal,
                                 VectorView& gradient )
    {
        (void)vertices;
        (void)faces;
        (void)scalar;
        (void)vertex_normal;
        (void)gradient;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "surfaceGradient" );
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
     * @note MPI. Per-face scatter with `+=` onto both endpoints of each edge,
     *       so `Comm::haloScatterAdd` before dividing by the vertex area.
     */
    template <class VertexView, class FaceView, class ScalarView,
              class AreaView>
    static void cotangentLaplacianScalar( const VertexView& vertices,
                                          const FaceView& faces,
                                          const ScalarView& values,
                                          const AreaView& vertex_area,
                                          ScalarView& result )
    {
        (void)vertices;
        (void)faces;
        (void)values;
        (void)vertex_area;
        (void)result;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators",
                                 "cotangentLaplacianScalar" );
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
     */
    template <class FaceView, class ScalarView>
    static void graphLaplacianScalar( const FaceView& faces,
                                      const ScalarView& values, int vertex_count,
                                      ScalarView& result )
    {
        (void)faces;
        (void)values;
        (void)vertex_count;
        (void)result;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "graphLaplacianScalar" );
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
     */
    template <class FaceView, class VectorView>
    static void graphLaplacianVector( const FaceView& faces,
                                      const VectorView& values,
                                      int vertex_count, VectorView& result )
    {
        (void)faces;
        (void)values;
        (void)vertex_count;
        (void)result;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "graphLaplacianVector" );
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
     */
    template <class VertexView, class FaceView, class AreaView,
              class VectorView>
    static void meanCurvatureNormal( const VertexView& vertices,
                                     const FaceView& faces,
                                     const AreaView& vertex_area,
                                     VectorView& result )
    {
        (void)vertices;
        (void)faces;
        (void)vertex_area;
        (void)result;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "meanCurvatureNormal" );
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
     */
    template <class VectorView, class NormalView>
    static void projectTangent( VectorView& values,
                                const NormalView& vertex_normal )
    {
        (void)values;
        (void)vertex_normal;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "projectTangent" );
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
     * @note MPI. Two `Comm::allReduceSum` calls, numerator and denominator,
     *       **both** before the division. See that function's note.
     */
    template <class ScalarView, class AreaView>
    static Real areaWeightedMean( const ScalarView& scalar,
                                  const AreaView& vertex_area )
    {
        (void)scalar;
        (void)vertex_area;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceOperators", "areaWeightedMean" );
    }
};

} // namespace Beatnik

#endif // BEATNIK_MESHGEOMETRY_HPP
