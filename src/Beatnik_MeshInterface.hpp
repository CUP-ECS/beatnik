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
 * @file Beatnik_MeshInterface.hpp
 * @brief ADAPTER (1 of 3). The distributed unstructured triangle surface,
 *        backed by **Tessera**.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Tessera type. Everything the rest of the
 * code needs from Tessera — storage of vertices and connectivity, the
 * owned/ghost partition, adjacency, and the topological edit operations that
 * refinement and remeshing perform — passes through `SurfaceMesh` below.
 * Swapping in the real library must touch this file and nothing else.
 *
 * Tessera has deliberately **not been read** while writing this header (see
 * `tasks/framework.md`, task M1). The interface is therefore shaped by what
 * the *Python* algorithms need, not by what Tessera happens to offer. The
 * first task that opens `../tessera` is expected to reshape it; that is the
 * point of confining it to one file.
 *
 * STORAGE MODEL
 * -------------
 * Two arrays, matching the `.npz` gold-file schema exactly (see
 * `Beatnik_IOInterface.hpp`):
 *
 *   - `vertices`  `(Nv, 3)` Real — node positions, owned followed by ghost.
 *   - `faces`     `(Nf, 3)` index — triangle connectivity, **outward oriented**
 *                 (the enclosed volume computed from it is positive).
 *
 * Per-vertex solution fields are *not* stored here; they live in
 * `Beatnik_SurfaceState.hpp` and are indexed by the same local vertex ordering.
 * Any operation that changes the vertex ordering (refinement, collapse,
 * redistribution) must therefore report a permutation or be applied through
 * `SurfaceState` so the fields follow. That is enforced by the
 * `MeshEditResult` return type below.
 *
 * MPI DECOMPOSITION
 * -----------------
 * `SurfaceMesh` owns a contiguous block of vertices and the faces incident on
 * them, plus a ghost layer deep enough for the widest stencil in the solver.
 * The widest stencil is the **cotangent Laplacian / mean-curvature normal**,
 * which reads the one-ring of each owned vertex, so one ghost layer of faces
 * suffices for the differential operators. The *proximity* search
 * (`Beatnik_DynamicRemesh.hpp`) is genuinely nonlocal and does not use ghosts
 * at all — it goes through a global spatial query instead.
 */

#ifndef BEATNIK_MESHINTERFACE_HPP
#define BEATNIK_MESHINTERFACE_HPP

#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Result of a topological mesh edit.
 *
 * Refinement, collapse and redistribution all renumber vertices. Rather than
 * let callers guess, every edit returns the map from the *new* local vertex
 * index to the *old* one so per-vertex fields can be gathered through it.
 *
 * @tparam VertexMapView `(Nv_new,)` view of old-index-or-`-1`.
 *         // TODO(types): templated pending Tessera/Canopy interface; collapse
 *         // to a concrete type once known.
 * @tparam WeightView `(Nv_new, 2)` interpolation weights for vertices created
 *         by an edge split; unused rows are (1, 0).
 *         // TODO(types): templated pending Tessera/Canopy interface; collapse
 *         // to a concrete type once known.
 */
template <class VertexMapView, class WeightView>
struct MeshEditResult
{
    /// For each new vertex, the old vertex it copies, or the first endpoint of
    /// the split edge for a newly created midpoint vertex.
    VertexMapView parent_a;

    /// For a newly created midpoint vertex, the second endpoint of the split
    /// edge; equal to `parent_a` for a carried-over vertex.
    VertexMapView parent_b;

    /// Interpolation weights `(w_a, w_b)`; `(1, 0)` for a carried-over vertex,
    /// `(0.5, 0.5)` for an edge midpoint. A per-vertex field `f` transfers as
    /// `f_new[i] = w_a[i]*f_old[parent_a[i]] + w_b[i]*f_old[parent_b[i]]`.
    WeightView weights;

    /// Number of edges split by this edit.
    int splits = 0;
    /// Number of edges collapsed by this edit.
    int collapses = 0;
    /// Number of edges flipped by this edit.
    int flips = 0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Distributed unstructured triangle surface. Tessera lives behind this.
 *
 * @tparam ExecutionSpace Kokkos execution space the mesh kernels run in.
 * @tparam MemorySpace    Kokkos memory space the mesh arrays live in.
 */
template <class ExecutionSpace, class MemorySpace>
class SurfaceMesh
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /// `(Nv, 3)` vertex positions.
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vertex_view = Kokkos::View<Real* [3], device_type>;

    /// `(Nf, 3)` triangle connectivity, local vertex indices.
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using face_view = Kokkos::View<LocalIndex* [3], device_type>;

    /// `(N,)` global identifiers, used only for I/O gather ordering.
    using gid_view = Kokkos::View<GlobalIndex*, device_type>;

    /**
     * @brief Construct an empty mesh bound to a communicator.
     * @param comm Communicator across which the surface is decomposed. Not
     *             duplicated here; the caller owns its lifetime.
     */
    explicit SurfaceMesh( MPI_Comm comm )
        : _comm( comm )
    {
    }

    /// The communicator the surface is decomposed over.
    MPI_Comm comm() const { return _comm; }

    //-----------------------------------------------------------------------//
    // Construction
    //-----------------------------------------------------------------------//

    /**
     * @brief Build a quasi-uniform closed sphere by recursive icosahedron
     *        subdivision, then distribute it.
     *
     * Port of mesh.py::icosphere_mesh (lines 362-461)
     *
     * Starts from the 12-vertex / 20-face regular icosahedron with golden-ratio
     * coordinates \f$(\pm 1, \pm\varphi, 0)\f$ and cyclic permutations,
     * normalized to the unit sphere. Each subdivision replaces every triangle
     * by four, with new vertices at **normalized** edge midpoints
     * \f$\hat m = (v_a+v_b)/\|v_a+v_b\|\f$ — note this is the *spherical*
     * midpoint, not the linear one, so the result stays exactly on the sphere.
     * Vertices are then scaled by `radius` and translated by `center`, and
     * every face is oriented so its normal points away from `center`.
     *
     * Resulting sizes: \f$N_v = 10\cdot 4^{s} + 2\f$,
     * \f$N_f = 20\cdot 4^{s}\f$.
     *
     * @param subdivisions Subdivision level \f$s \ge 0\f$.
     * @param radius       Sphere radius, in problem length units.
     * @param center       Sphere centre \f$(c_x, c_y, c_z)\f$.
     *
     * @note REPRODUCIBILITY. The vertex *ordering* produced here must not be
     *       assumed to match the Python. The regression comparator sorts on
     *       quantized coordinates precisely so it does not have to. What *must*
     *       match is the vertex set and the positions, to comparison tolerance.
     *       See `tasks/framework.md` "Known risks".
     */
    void generateIcosphere( int subdivisions, Real radius,
                            const Real center[3] )
    {
        (void)subdivisions;
        (void)radius;
        (void)center;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "generateIcosphere" );
    }

    /**
     * @brief Build a structured latitude/longitude closed sphere with true
     *        pole vertices and triangular caps, then distribute it.
     *
     * Port of mesh.py::structured_sphere_mesh (lines 300-359)
     *
     * Latitude rings at \f$\theta_i = i\pi/n_\theta\f$ for
     * \f$i = 1 \ldots n_\theta-1\f$ (poles excluded and added as single
     * vertices), longitudes at \f$\phi_j = 2\pi j/n_\phi\f$. The north cap is
     * `n_phi` triangles fanning from the north pole vertex, each interior band
     * is two triangles per quad, and the south cap fans from the south pole
     * with reversed winding so all normals point outward.
     *
     * Resulting sizes: \f$N_v = n_\phi(n_\theta-1) + 2\f$,
     * \f$N_f = 2 n_\phi (n_\theta - 1)\f$.
     *
     * @param n_theta Latitude bands, \f$\ge 2\f$.
     * @param n_phi   Longitude divisions, \f$\ge 4\f$.
     * @param radius  Sphere radius.
     * @param center  Sphere centre.
     */
    void generateLatLonSphere( int n_theta, int n_phi, Real radius,
                               const Real center[3] )
    {
        (void)n_theta;
        (void)n_phi;
        (void)radius;
        (void)center;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "generateLatLonSphere" );
    }

    /**
     * @brief Adopt an externally supplied vertex/face pair and distribute it.
     *
     * This is the path used by `--restart-from` and by the "read the initial
     * mesh from the gold file" mitigation for regression test 1
     * (`tasks/framework.md`, risk R1). The arrays arrive **replicated on rank
     * 0** and are partitioned here.
     *
     * @param vertices `(Nv, 3)` positions, meaningful on rank 0 only.
     * @param faces    `(Nf, 3)` connectivity, meaningful on rank 0 only.
     *
     * @tparam HostVertexView, HostFaceView Host-accessible views.
     *         // TODO(types): templated pending Tessera/Canopy interface;
     *         // collapse to a concrete type once known.
     */
    template <class HostVertexView, class HostFaceView>
    void adopt( const HostVertexView& vertices, const HostFaceView& faces )
    {
        (void)vertices;
        (void)faces;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "adopt" );
    }

    //-----------------------------------------------------------------------//
    // Accessors
    //-----------------------------------------------------------------------//

    /// Owned + ghost vertex positions. Rows `[0, ownedVertexCount())` are
    /// owned; the remainder are ghosts.
    vertex_view vertices() const { return _vertices; }

    /// Owned + ghost faces, indexing into `vertices()`.
    face_view faces() const { return _faces; }

    /// Number of vertices this rank owns.
    int ownedVertexCount() const { return _owned_vertices; }

    /// Number of vertices this rank stores, owned plus ghost.
    int totalVertexCount() const
    {
        return static_cast<int>( _vertices.extent( 0 ) );
    }

    /// Number of faces this rank owns.
    int ownedFaceCount() const { return _owned_faces; }

    /// Number of faces this rank stores, owned plus ghost.
    int totalFaceCount() const
    {
        return static_cast<int>( _faces.extent( 0 ) );
    }

    /**
     * @brief Global vertex count, summed across the communicator.
     *
     * Involves an `MPI_Allreduce` — see
     * `Beatnik_Communication.hpp::allReduceSum`. Cached between mesh edits.
     */
    GlobalIndex globalVertexCount() const
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "globalVertexCount" );
    }

    /// Global face count. Same reduction caveat as `globalVertexCount`.
    GlobalIndex globalFaceCount() const
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "globalFaceCount" );
    }

    //-----------------------------------------------------------------------//
    // Adjacency
    //-----------------------------------------------------------------------//

    /**
     * @brief Build (or return the cached) unique-edge list and its
     *        edge-to-face adjacency.
     *
     * Port of mesh.py::edges_from_faces (lines 227-237)
     *
     * An edge is keyed by its sorted endpoint pair, so each interior edge maps
     * to exactly two faces and each boundary edge to one. On a closed surface
     * every edge must map to two; a count of one signals either a genuine hole
     * (a bug in refinement) or a missing ghost face.
     *
     * @return Opaque adjacency handle; the concrete type is Tessera's.
     */
    void buildEdgeAdjacency()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "buildEdgeAdjacency" );
    }

    /**
     * @brief Build (or return the cached) vertex one-ring adjacency.
     *
     * Port of mesh.py::vertex_adjacency (lines 141-147)
     *
     * Needed by every umbrella-stencil operator: the graph Laplacian, the
     * tangential relaxation, and the valence computation used by the
     * isotropic cleanup.
     */
    void buildVertexAdjacency()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "buildVertexAdjacency" );
    }

    /**
     * @brief Face-to-face adjacency through shared edges.
     *
     * Port of mesh_solver.py::_face_neighbors (lines 1533-1540)
     *
     * Used to grow AMR marks by neighbor rings and to build the
     * proximity-exclusion rings.
     */
    void buildFaceAdjacency()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "buildFaceAdjacency" );
    }

    //-----------------------------------------------------------------------//
    // Topological edits
    //
    // These are the operations Tessera must support for AMR and remeshing to
    // work in parallel. Each is collective: an edge on a rank boundary is split
    // or collapsed identically on both sides, or the surface tears.
    //-----------------------------------------------------------------------//

    /**
     * @brief Conforming red-green refinement of a marked face set.
     *
     * Port of mesh.py::refine_marked_faces (lines 570-730)
     *
     * A *red* (marked) face is split into four by bisecting all three edges. An
     * unmarked face sharing 1, 2 or 3 split edges is *green*-split into 2, 3 or
     * 4 children respectively, so the result is conforming (no hanging nodes).
     * New vertices sit at the linear edge midpoint — **not** projected back to
     * any surface, since the interface is the surface.
     *
     * @param marked `(Nf,)` boolean marks, already closed and rank-consistent.
     * @return Vertex parent/weight map, so per-vertex fields transfer as the
     *         midpoint average of the split edge's endpoints.
     *
     * @note MPI. Marks must be reconciled across rank boundaries *before*
     *       calling: a face marked on one side of a boundary implies its edges
     *       are split on the other. See
     *       `Beatnik_Communication.hpp::reconcileRefinementMarks`.
     */
    template <class MarkView>
    auto refine( const MarkView& marked )
    {
        (void)marked;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "refine" );
    }

    /**
     * @brief Split a specified set of edges at their midpoints.
     *
     * Port of dynamic_remesh.py::split_selected_edges (lines 261-298)
     *
     * The primitive under both `split_long_edges` and the surgical proximity
     * splits. Unlike `refine`, the caller chooses edges rather than faces; the
     * incident faces are subdivided to stay conforming.
     */
    template <class EdgeListView>
    auto splitEdges( const EdgeListView& edges )
    {
        (void)edges;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "splitEdges" );
    }

    /**
     * @brief Collapse a set of short edges onto their midpoints.
     *
     * Port of dynamic_remesh.py::collapse_short_edges (lines 361-407)
     *
     * A collapse is rejected unless it is both **topologically safe** (the link
     * condition: the one-rings of the two endpoints intersect in exactly the
     * two vertices opposite the edge, else the collapse creates a non-manifold
     * edge — `dynamic_remesh.py:509-518`) and **geometrically safe** (no
     * incident face inverts its normal or becomes a sliver —
     * `dynamic_remesh.py:519-551`).
     */
    template <class EdgeListView>
    auto collapseEdges( const EdgeListView& edges )
    {
        (void)edges;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "collapseEdges" );
    }

    /**
     * @brief Flip a set of interior edges to the opposite diagonal.
     *
     * Port of dynamic_remesh.py::flip_edges_for_quality (lines 408-458)
     *
     * A flip changes connectivity only, so per-vertex fields are untouched and
     * no `MeshEditResult` map is needed. A flip is rejected if the opposite
     * diagonal already exists (non-manifold) or if either child face would
     * invert relative to the pre-flip normal.
     */
    template <class EdgeListView>
    int flipEdges( const EdgeListView& edges )
    {
        (void)edges;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "flipEdges" );
    }

    /**
     * @brief Drop unreferenced vertices and renumber contiguously.
     *
     * Port of dynamic_remesh.py::compact_mesh (lines 492-508)
     *
     * Run after a collapse sweep, which leaves orphaned vertices behind.
     */
    auto compact()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "compact" );
    }

    /**
     * @brief Overwrite vertex positions in place, connectivity unchanged.
     *
     * Used by the time integrator and by every purely geometric pass
     * (tangential relaxation, volume projection, implicit fairing).
     */
    template <class NewVertexView>
    void setVertices( const NewVertexView& vertices )
    {
        (void)vertices;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "setVertices" );
    }

  private:
    MPI_Comm _comm;
    vertex_view _vertices;
    face_view _faces;
    gid_view _vertex_gids;
    gid_view _face_gids;
    int _owned_vertices = 0;
    int _owned_faces = 0;
};

} // namespace Beatnik

#endif // BEATNIK_MESHINTERFACE_HPP
