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
 * RECONCILED AGAINST TESSERA 2026-08-07 (task M1). The pre-M1 version of this
 * header was written without reading `../tessera` and was shaped by what the
 * Python algorithms need. This version is shaped by what Tessera actually
 * provides. Every semantic that changed is flagged inline with **M1 CHANGE**,
 * and every capability Beatnik needs that Tessera does not have is flagged
 * **M1 GAP** here and recorded in `tasks/framework.md` (task M1). *Gaps are not
 * worked around in Beatnik* — in particular Beatnik implements no haloing and
 * no partitioning of its own.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Tessera type. Everything the rest of the
 * code needs from Tessera — storage of vertices and connectivity, the
 * owned/ghost partition, adjacency, the topological edits, the halo, and
 * redistribution — passes through `SurfaceMesh` below. Where a caller needs to
 * spell a type that is Tessera's underneath, it spells the Beatnik alias
 * (`typename mesh_type::position_slice`, `::face_vertex_view`, ...), never the
 * Tessera name.
 *
 * WHAT TESSERA IS
 * ---------------
 * A distributed unstructured triangle-mesh library over Cabana + Kokkos. It
 * owns: entity storage (vertices / edges / faces, each a Cabana AoSoA with a
 * compile-time user field pack), global ids, the owned/ghost partition, a
 * **1-deep** MPI halo, split-based conforming adaptive refinement, Zoltan2
 * load balancing and migration, and HDF5/XDMF I/O. It owns no physics and no
 * discretization convention: normals are unoriented, areas undefined, stencil
 * weights are the caller's.
 *
 * STORAGE MODEL (M1 CHANGE — this replaces the two-array model)
 * ------------------------------------------------------------
 * Tessera stores **three** entity kinds, not two, each owned-first:
 *
 *   - vertices : `Gid`, `Owner`, `Flags`, `Position[3]`, then the user pack.
 *   - edges    : `Gid`, `Owner`, `Level`, endpoint vertex **gids** `v[2]`,
 *                incident face **gids** `f[2]`, then the user pack.
 *   - faces    : `Gid`, `Owner`, `Level`, corner vertex **gids** `v[3]`,
 *                edge **gids** `e[3]`, then the user pack.
 *
 * Two consequences drive most of the rest of this file:
 *
 * 1. **Connectivity is stored as global ids, not local indices.** A `Mesh` is
 *    therefore not capturable into a device kernel, and `faces()` cannot hand
 *    back a `View<LocalIndex*[3]>` taken straight out of storage. Tessera's
 *    answer is `buildMeshGeometry(mesh)`, which derives per-face and per-edge
 *    *local* vertex indices once (host-side, through a `gid -> local` map) and
 *    returns a small device-capturable accessor. `faceVertices()` below is that
 *    accessor's `faceVerts` view, and it is what every ported Python kernel
 *    should index with. It covers **all locally held entities (owned + ghost)**,
 *    so an operator over owned vertices can read its full 1-ring.
 *
 * 2. **Per-vertex solution fields must live inside the mesh** — see the field
 *    pack section below. This is the single largest M1 change.
 *
 * THE VERTEX USER FIELD PACK (M1 CHANGE — replaces `MeshEditResult`)
 * -----------------------------------------------------------------
 * The pre-M1 design had every topological edit return a `parent_a`/`parent_b`/
 * `weights` map, and `SurfaceState` gather its own `Kokkos::View`s through it.
 * **Tessera does not report such a map and does not need one**, because it
 * transfers fields itself:
 *
 *   - `refine()` inserts a midpoint vertex on each bisected edge and fills its
 *     position *and every vertex user field* from the two endpoints through a
 *     pluggable `RefinePolicy` (default: the linear average, i.e. exactly the
 *     `(0.5, 0.5)` weights the old `MeshEditResult` encoded). Existing vertex
 *     and face user fields are preserved.
 *   - `migrate()` / `loadBalance()` move **whole Cabana tuples**, so every user
 *     field follows its entity across ranks with no per-field plumbing.
 *   - `haloExchange()` likewise syncs the whole tuple of every ghost.
 *
 * So a per-vertex field held in a `Kokkos::View` *outside* the Tessera mesh is
 * silently dropped by refinement and silently stale after migration. Beatnik's
 * evolved state therefore lives **in the mesh**, as the vertex user pack:
 *
 * | Beatnik field | Type | Meaning |
 * | --- | --- | --- |
 * | `VertexField::Potential`        | `Real`    | velocity potential jump phi |
 * | `VertexField::SheetVector`      | `Real[3]` | tangential sheet vector S |
 * | `VertexField::MaterialPosition` | `Real[3]` | carried Lagrangian coordinate |
 *
 * The linear average is the correct transfer rule for all three (phi and the
 * material coordinate are interpolated at a split; the sheet vector under the
 * `Potential` model is a cache that `updateSheetVector` overwrites anyway), so
 * Tessera's `DefaultRefinePolicy` is used unchanged and Beatnik supplies no
 * policy of its own. If a conservative (rather than interpolatory) rule is ever
 * wanted for the sheet strength, it is a `RefinePolicy` subclass here, not a
 * change anywhere else.
 *
 * **Follow-up, not M1:** `Beatnik_SurfaceState.hpp` still declares its own
 * `Kokkos::View`s for these three fields. T1b/T1c must re-point it at the
 * accessors below; until then those views are unbacked. Recorded in
 * `tasks/framework.md`.
 *
 * MPI DECOMPOSITION AND THE HALO
 * ------------------------------
 * Ownership is Tessera's **lowest-rank rule**: a face is owned by its assigned
 * partition rank; a vertex or edge by the lowest-ranked owner of an incident
 * face. The local set is the 1-deep closure of the owned vertices' one-rings,
 * so **every owned vertex sees its complete one-ring**, ghosts included. Local
 * indices `[0, ownedXCount())` are owned; `[ownedXCount(), totalXCount())` are
 * ghosts.
 *
 * `haloExchange()` is **collective** on the mesh communicator and syncs the
 * *whole field pack of all three entity kinds* — it is not per-field and not
 * per-kind. Beatnik cannot ask for "just the potential"; the cost of refreshing
 * one field is the cost of refreshing everything.
 *
 * **M1 GAP — depth.** The halo is **1-deep and not configurable** (risk R8).
 * That covers a one-ring stencil exactly. The Beatnik RHS is a **two-ring**
 * stencil, and Tessera documents that `buildVertexStencil(mesh, 2)` is
 * *silently incomplete* within one hop of a partition boundary: missing outer
 * -ring neighbours are absent from the CSR row rather than reported. Two
 * successive `haloExchange()` calls do **not** substitute — the second exchange
 * refreshes the same 1-deep ghost set, it does not widen it. See
 * `tasks/framework.md` M1 gap G1.
 *
 * **M1 GAP — direction.** `haloExchange()` is a pure **gather** (owner -> ghost).
 * There is no scatter-add (ghost -> owner accumulate). See gap G2.
 *
 * GENERATION GUARD — THE LIFETIME CONTRACT
 * ----------------------------------------
 * Every handle Tessera hands out (position slice, CSR, key view, `MeshGeometry`,
 * `VertexStencil`) is stamped with the mesh's `generation()` counter at the
 * moment it was taken. Any op that changes the local entity count or the ghost
 * set — `distribute`, `migrate`, `loadBalance`, `refine`, any `resize` — bumps
 * the counter, and **copying a stale handle aborts with a diagnostic** rather
 * than reading freed storage. `haloExchange()` is topology-preserving and does
 * *not* bump it.
 *
 * Beatnik must therefore re-take every accessor below after any topological
 * edit or redistribution. `generation()` is exposed so a caller can assert it,
 * and the adapter rebuilds its own cached `MeshGeometry`/stencil internally.
 * **Do not cache the return of `positions()`, `faceVertices()` or
 * `vertexOneRing()` across a mesh edit.**
 *
 * THE MANDATORY POST-REFINE SEQUENCE
 * ----------------------------------
 * Tessera's distributed `refine()` leaves each rank holding only its refined
 * *owned* entities and **clears the halo**. A `haloExchange()` on a
 * freshly-refined mesh is a no-op on an empty plan, not a synced ghost layer,
 * and a second `refine()` without an intervening re-halo *throws*. The
 * sanctioned idiom — which `refine()` and `redistribute()` below encapsulate so
 * no caller has to remember it — is:
 *
 *     refine( mask );                  // Tessera::refine, clears the halo
 *     migrate( dest );                 // identity dest is legal; rebuilds halo
 *     haloExchange();                  // now meaningful
 *
 * REFINEMENT MODE
 * ---------------
 * The mesh is instantiated `RefinementMode::Conforming` (Tessera's default): a
 * marked face is red-split 1->4 and the kept neighbours are retriangulated by a
 * transient closure layer, so no hanging node survives. This is what the Python
 * red-green scheme produces and what a surface operator needs to be consistent.
 * Two consequences Beatnik inherits:
 *
 *   - `Level` is the **red** level, so it no longer maps 1:1 to triangle size.
 *   - The *visible* triangulation is rank-count dependent up to which diagonal
 *     some closure quads are split along (Tessera measured ranks 1-4 identical,
 *     rank 5 differing on 4 of 20 blue parents). Mesh size, refinement
 *     decisions and conformity **are** rank-count invariant. Do not compare a
 *     refined visible face set bitwise across rank counts; compare by position
 *     — which is exactly what `compare_output.py`'s quantized matching does.
 */

#ifndef BEATNIK_MESHINTERFACE_HPP
#define BEATNIK_MESHINTERFACE_HPP

#include <Beatnik_Types.hpp>

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief Beatnik's vertex user fields, as offsets into Tessera's vertex user
 *        pack.
 *
 * These are *Beatnik* constants, not Tessera ones, so a caller outside this
 * header can name a field without naming a Tessera type. The adapter maps them
 * onto `Tessera::userVertexField<N>()` internally.
 *
 * Order is part of the checkpoint contract only indirectly — the HDF5 schema is
 * fixed by `Beatnik_IOInterface.hpp` and keyed by name, not by this index — but
 * reordering still invalidates any `RefinePolicy` written against it.
 */
namespace VertexFieldId
{
enum : int
{
    Potential = 0,        ///< `Real`,    velocity potential jump phi.
    SheetVector = 1,      ///< `Real[3]`, tangential sheet vector S.
    MaterialPosition = 2, ///< `Real[3]`, carried Lagrangian coordinate.
    Count = 3
};
}

//---------------------------------------------------------------------------//
/**
 * @brief What a topological edit did.
 *
 * M1 CHANGE. This replaces the pre-M1 `MeshEditResult`, whose `parent_a` /
 * `parent_b` / `weights` map does not exist in Tessera and is not needed:
 * Tessera transfers vertex user fields itself through the refinement policy
 * (see the field-pack section in the file header). What is left is what
 * `Tessera::RefineResult` actually reports, plus the entity counts a caller
 * wants for a progress line.
 *
 * `MeshEditResult` is deleted rather than kept as a shim, because a shim would
 * have to fabricate a parent map that no Beatnik code could correctly consume.
 */
struct MeshEditReport
{
    /// 2:1 mark-propagation rounds Tessera's cross-rank fixpoint executed.
    /// Tessera's own diagnostic; a value that climbs run over run means the
    /// indicator is marking in a pattern that needs a lot of closure.
    int balance_rounds = 0;

    /// Number of edges bisected, counted over the edges **this rank touches**
    /// (`Tessera::RefineResult::midpoints.size()`). Not a global count and not
    /// a partition — an edge on a rank boundary is counted on both sides.
    long long split_edges_local = 0;

    /// Owned vertex / face counts after the edit, for diagnostics.
    long long owned_vertices_after = 0;
    long long owned_faces_after = 0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Distributed unstructured triangle surface. Tessera lives behind this.
 *
 * The class template parameters are unchanged from the pre-M1 header, so every
 * other Beatnik header's `using mesh_type = SurfaceMesh<ExecutionSpace,
 * MemorySpace>;` still compiles. Tessera's `Mesh` takes `<Scalar, Dim,
 * VertexUserFields, EdgeUserFields, FaceUserFields, MemorySpace,
 * ExecutionSpace, RefinementMode>`; the mapping is fixed below.
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

  private:
    /// Beatnik's vertex user pack, in `VertexFieldId` order.
    using vertex_fields =
        Tessera::VertexFields<Real, Real[3], Real[3]>;

    /// No per-edge or per-face user state. Note that Tessera **resets** edge
    /// user fields on every `refine()` (edges are re-derived from the new face
    /// connectivity), so an edge field would be unsafe to carry anyway.
    using edge_fields = Tessera::EdgeFields<>;
    using face_fields = Tessera::FaceFields<>;

  public:
    /// The Tessera mesh Beatnik is a facade over. Named here and nowhere else.
    using tessera_mesh_type =
        Tessera::Mesh<Real, 3, vertex_fields, edge_fields, face_fields,
                      MemorySpace, ExecutionSpace,
                      Tessera::RefinementMode::Conforming>;

    /// The three halo exchange plans (vertices, edges, faces).
    using tessera_halo_type = Tessera::MeshHalo<MemorySpace>;

    /**
     * @brief `(Nv, 3)` vertex positions, as a generation-stamped Cabana slice.
     *
     * M1 CHANGE — this was `Kokkos::View<Real*[3], device_type>`. Positions are
     * a core member of Tessera's vertex AoSoA, so what comes back is a slice of
     * that AoSoA wrapped in the generation guard, not a standalone view.
     * Indexing is unchanged (`pos(i, d)`), and it is capturable by value into a
     * `KOKKOS_LAMBDA`; capturing it after a topology edit aborts by design.
     *
     * Callers outside this header spell it `typename mesh_type::position_slice`.
     */
    using position_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template vertexSlice<Tessera::VertexField::Position>() );

    /// `(Nv,)` scalar vertex user field slice (e.g. the potential).
    using scalar_field_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template vertexSlice<Tessera::userVertexField<
                          VertexFieldId::Potential>()>() );

    /// `(Nv, 3)` vector vertex user field slice (sheet vector, material pos).
    using vector_field_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template vertexSlice<Tessera::userVertexField<
                          VertexFieldId::SheetVector>()>() );

    /**
     * @brief `(Nf, 3)` **local** vertex indices per face.
     *
     * M1 CHANGE — this replaces `face_view = Kokkos::View<LocalIndex*[3]>`
     * taken from storage. Tessera stores corner vertex *gids*; this view is the
     * `faceVerts` member of `Tessera::MeshGeometry`, derived host-side once per
     * topology generation. A corner whose gid is not held locally is
     * `Tessera::invalid_local` (-1) — which the 1-ring closure invariant makes
     * unreachable for a locally-held face, but which a kernel must still skip
     * rather than index with. Plain Kokkos, so it names no Tessera type.
     */
    using face_vertex_view = Kokkos::View<int* [3], memory_space>;

    /// `(Ne, 2)` local vertex indices per edge. Same derivation and caveat.
    using edge_vertex_view = Kokkos::View<int* [2], memory_space>;

    /**
     * @brief Vertex one-ring adjacency in CSR form.
     *
     * `offsets(i) .. offsets(i+1)` index `neighbors`, giving the **local**
     * indices of the vertices adjacent to local vertex `i`, ascending.
     */
    struct AdjacencyCsr
    {
        Kokkos::View<int*, memory_space> offsets;
        Kokkos::View<int*, memory_space> neighbors;
    };

    /**
     * @brief Construct an empty mesh bound to a communicator.
     * @param comm Communicator across which the surface is decomposed. Not
     *             duplicated here; the caller owns its lifetime. Tessera's
     *             `Mesh` constructor takes it the same way and caches rank and
     *             size.
     */
    explicit SurfaceMesh( MPI_Comm comm )
        : _mesh( comm )
    {
    }

    /// The communicator the surface is decomposed over.
    MPI_Comm comm() const { return _mesh.comm(); }

    /// This rank's index in `comm()`.
    int rank() const { return _mesh.rank(); }

    /// Number of ranks in `comm()`.
    int commSize() const { return _mesh.commSize(); }

    /**
     * @brief Tessera's topology generation counter.
     *
     * Bumped by every op that changes the local entity count or the ghost set.
     * Any accessor taken at generation `g` aborts if copied once the mesh has
     * moved past `g`. Exposed so a caller can assert its cached handles are
     * still current rather than discovering it inside a kernel launch.
     */
    std::size_t generation() const { return _mesh.generation(); }

    //-----------------------------------------------------------------------//
    // Construction
    //
    // All three entry points below produce a *replicated* mesh — identical on
    // every rank, every entity owned — and then call `distribute()` to cut it
    // to owned + 1-deep ghost and build the halo plans. That is Tessera's
    // only distribution path from a serial generator: `buildFromTriangleSoup`
    // is host-side and single-rank by construction.
    //
    // M1 GAP (G6, scalability, not correctness): the coarse mesh is built and
    // held in full on every rank before it is cut. At the default subdivision 2
    // that is 162 vertices, so it is irrelevant to Beatnik's regression tests;
    // it becomes a memory ceiling only if an initial mesh is ever generated at
    // a resolution comparable to the refined running mesh.
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
     * \f$\hat m = (v_a+v_b)/\|v_a+v_b\|\f$ — the *spherical* midpoint, not the
     * linear one, so the result stays exactly on the sphere.
     *
     * Resulting sizes: \f$N_v = 10\cdot 4^{s} + 2\f$,
     * \f$N_f = 20\cdot 4^{s}\f$.
     *
     * TESSERA MAPPING
     * ---------------
     *   1. `Tessera::buildIcosphere( mesh, subdivisions )` — generates the
     *      triangle soup and derives full connectivity (edges, both CSR
     *      one-rings, the canonical key side tables). Tessera's generator uses
     *      the same golden-ratio base table and the same normalized-midpoint
     *      rule, and computes the midpoint as `normalize(0.5*(a+b))`, which is
     *      identical in IEEE double to the Python's
     *      `(a+b)/||a+b||` up to the shared scaling of numerator and
     *      denominator. Its **face winding and vertex ordering differ** from the
     *      Python's; see the reproducibility note below.
     *   2. Scale by `radius` and translate by `center`, in place on the
     *      position slice. **M1 CHANGE — Tessera's generator is unit-sphere
     *      only and takes neither argument**, so this is Beatnik's step, not a
     *      parameter forwarded to Tessera.
     *   3. `Tessera::facePartitionByAxis( mesh, axis = 2 )` then
     *      `Tessera::distribute( mesh, halo, faceOwner )` then
     *      `Tessera::haloExchange( mesh, halo )`.
     *
     * Orientation: Tessera's base table is documented CCW seen from outside and
     * subdivision preserves winding, so faces come out outward-oriented as the
     * Python requires. Beatnik must still *verify* it rather than assume — the
     * enclosed volume of the generated sphere must be positive — because
     * Tessera imposes no orientation convention of its own (`faceNormalRaw` is
     * explicitly unoriented).
     *
     * @param subdivisions Subdivision level \f$s \ge 0\f$.
     * @param radius       Sphere radius, in problem length units.
     * @param center       Sphere centre \f$(c_x, c_y, c_z)\f$.
     *
     * @note REPRODUCIBILITY. The vertex *ordering* produced here must not be
     *       assumed to match the Python — it does not, since Tessera's base
     *       icosahedron table and subdivision loop are its own. The regression
     *       comparator sorts on quantized coordinates precisely so it does not
     *       have to. What *must* match is the vertex set and the positions, to
     *       comparison tolerance. See `tasks/framework.md` risk R1.
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
     * M1 GAP (G7, minor). **Tessera has no lat/lon generator** — `Icosphere` is
     * its only one. Beatnik builds the `positions` / `triangles` soup itself
     * (host, serial, replicated: the arithmetic above, transcribed from the
     * Python) and hands it to `Tessera::buildFromTriangleSoup`, which derives
     * all connectivity. This is *generation*, not haloing or partitioning, so
     * writing it in Beatnik is not a workaround for a missing Tessera
     * capability — but a `buildLatLonSphere` alongside `buildIcosphere` would
     * be the natural Tessera-side home for it, and is offered as such in
     * `tasks/framework.md`.
     *
     * Distribution is then identical to `generateIcosphere` steps 2-3.
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
     * (`tasks/framework.md`, risk R1).
     *
     * TESSERA MAPPING. `Tessera::buildFromTriangleSoup( mesh, soup )` where
     * `soup.positions` is `3*Nv` scalars and `soup.triangles` is `3*Nf` vertex
     * indices, then partition + distribute + halo exchange as above.
     *
     * **M1 CHANGE — the arrays must be replicated on EVERY rank, not rank 0.**
     * The pre-M1 contract said "meaningful on rank 0 only", with Beatnik
     * partitioning from there. `buildFromTriangleSoup` is a single-rank host
     * routine with no communication: it assumes identical input everywhere and
     * `distribute()` then relies on that (its ownership rule is computed
     * locally *because* the mesh is replicated). So the caller must broadcast
     * first — `Beatnik_Communication.hpp::broadcastFromRoot`, which for this
     * reason is a prerequisite of `adopt`, not an alternative to it.
     *
     * @param vertices `(Nv, 3)` positions, identical on every rank.
     * @param faces    `(Nf, 3)` connectivity, identical on every rank.
     *
     * @tparam HostVertexView, HostFaceView Host-accessible views. These stay
     *         templated: Tessera's soup is `std::vector`-backed and Beatnik's
     *         readers produce Kokkos host views, so `adopt` transcribes rather
     *         than forwards.
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
    //
    // Every one of these returns a generation-stamped handle. Re-take them
    // after any edit; see the lifetime contract in the file header.
    //-----------------------------------------------------------------------//

    /**
     * @brief Owned + ghost vertex positions, `pos(i, d)`.
     *
     * Rows `[0, ownedVertexCount())` are owned; the remainder are ghosts, whose
     * values are current only after a `haloExchange()` on an un-cleared halo.
     *
     * TESSERA MAPPING: `mesh.vertexSlice<Tessera::VertexField::Position>()`.
     */
    position_slice positions()
    {
        return _mesh.template vertexSlice<Tessera::VertexField::Position>();
    }

    /**
     * @brief `(Nv,)` velocity potential jump phi, the evolved unknown under
     *        `StateModel::Potential`.
     *
     * Units: velocity x length. Carried through refinement (interpolated at
     * split midpoints) and migration automatically — see the field-pack section
     * of the file header.
     */
    scalar_field_slice potential()
    {
        return _mesh.template vertexSlice<
            Tessera::userVertexField<VertexFieldId::Potential>()>();
    }

    /**
     * @brief `(Nv, 3)` tangential sheet vector S. Units: velocity.
     *
     * The evolved unknown under `StateModel::SheetVector`; under `Potential` a
     * cache refreshed from \f$S = -\hat n \times \nabla_s \phi\f$.
     */
    vector_field_slice sheetVector()
    {
        return _mesh.template vertexSlice<
            Tessera::userVertexField<VertexFieldId::SheetVector>()>();
    }

    /**
     * @brief `(Nv, 3)` carried Lagrangian ("material") coordinate.
     *
     * Initialized to the vertex positions at t=0 and advected as an ordinary
     * per-vertex field through every remesh, but **never** integrated in time.
     * Its only consumer is the nonlocal-proximity material exclusion.
     */
    vector_field_slice materialPosition()
    {
        return _mesh.template vertexSlice<
            Tessera::userVertexField<VertexFieldId::MaterialPosition>()>();
    }

    /**
     * @brief `(Nf, 3)` local vertex indices per face; the connectivity every
     *        ported kernel indexes with.
     *
     * Covers owned **and** ghost faces. Derived from the gid-based storage by
     * `Tessera::buildMeshGeometry`, which the adapter caches and rebuilds on a
     * generation change — so this is O(1) on the common path and a host-side
     * `gid -> local` pass on the first call after an edit.
     */
    face_vertex_view faceVertices()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "faceVertices" );
    }

    /// `(Ne, 2)` local vertex indices per edge. Same derivation as
    /// `faceVertices`; this is the unique-edge list, so it is also the answer
    /// to "enumerate the edges" for the edge-length reductions.
    edge_vertex_view edgeVertices()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "edgeVertices" );
    }

    /// Number of vertices this rank owns.
    int ownedVertexCount() const
    {
        return static_cast<int>( _mesh.numOwnedVertices() );
    }

    /// Number of vertices this rank stores, owned plus ghost.
    int totalVertexCount() const
    {
        return static_cast<int>( _mesh.numVertices() );
    }

    /// Number of edges this rank owns. Owned edges form a global partition, so
    /// this is the count to reduce over for a global edge statistic (risk R9).
    int ownedEdgeCount() const
    {
        return static_cast<int>( _mesh.numOwnedEdges() );
    }

    /// Number of edges this rank stores, owned plus ghost.
    int totalEdgeCount() const { return static_cast<int>( _mesh.numEdges() ); }

    /// Number of faces this rank owns.
    int ownedFaceCount() const
    {
        return static_cast<int>( _mesh.numOwnedFaces() );
    }

    /// Number of faces this rank stores, owned plus ghost.
    int totalFaceCount() const { return static_cast<int>( _mesh.numFaces() ); }

    /**
     * @brief Global vertex count, summed across the communicator.
     *
     * M1 GAP (G3). Tessera exposes exactly one global reduction, `globalMin`;
     * there is no `globalSum`. Its own invariant checks hand-roll
     * `MPI_Allreduce(MPI_SUM)` over `numOwnedVertices()`, and so must Beatnik
     * — through `Beatnik_Communication.hpp::allReduceSum`, so the collective
     * appears in one place. Summing the **owned** count is what makes this a
     * partition and not a double count.
     *
     * Involves a collective. Cached between mesh edits (keyed on
     * `generation()`).
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
    //
    // M1 CHANGE — the three pre-M1 "build*Adjacency" calls do not survive as
    // three builders. Tessera builds two of the three relations itself as part
    // of every topology op (`buildFromTriangleSoup`, `distribute`, `migrate`),
    // so there is nothing for Beatnik to trigger; and the third does not exist
    // at all. What is left is: one accessor, one build, one gap.
    //-----------------------------------------------------------------------//

    /**
     * @brief Unique-edge list and edge-to-face adjacency.
     *
     * Port of mesh.py::edges_from_faces (lines 227-237)
     *
     * M1 CHANGE — **there is nothing to build.** Tessera derives the unique
     * edge set on construction and maintains it through every topology op, as a
     * first-class entity kind: `EdgeField::Verts` holds the two endpoint vertex
     * gids (sorted, so an edge is keyed identically on both sides of a rank
     * boundary — `Tessera::makeEdgeKey`), and `EdgeField::Faces` holds the two
     * incident face gids, `Tessera::invalid_gid` where absent. Beatnik's job is
     * to *read* it: `edgeVertices()` above for the endpoints in local indices.
     *
     * The Python's closed-surface check carries over unchanged and is worth
     * keeping: on a closed surface every edge must name two faces. A count of
     * one signals a genuine hole (a bug in refinement) — but note that after
     * `refine()` and before the mandatory re-halo, an edge's second incident
     * face may simply have moved off-rank, so the check is only meaningful on a
     * halo-consistent mesh.
     *
     * @note `EdgeField::Faces` stores **gids**, and Tessera's own migration
     *       comment records that it carries them verbatim — an edge's incident
     *       -face gid may name a face now owned elsewhere and not held locally.
     *       It is metadata, not a usable local index. Do not build a face
     *       neighbourhood from it; see `faceAdjacency` below.
     */
    void edgeAdjacency() const
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "edgeAdjacency" );
    }

    /**
     * @brief Vertex one-ring adjacency, in local indices.
     *
     * Port of mesh.py::vertex_adjacency (lines 141-147)
     *
     * Needed by every umbrella-stencil operator: the graph Laplacian, the
     * tangential relaxation, and the valence computation used by the isotropic
     * cleanup.
     *
     * TESSERA MAPPING: `Tessera::buildVertexStencil( mesh, k = 1 )`, whose
     * CSR is exactly this relation (self excluded, neighbours ascending by
     * local index for determinism). Complete for every owned vertex, because
     * the 1-deep halo is by construction the one-ring closure.
     *
     * The adapter caches it and rebuilds on a generation change. Tessera also
     * maintains `vertexEdges()` and `vertexFaces()` CSRs directly on the mesh;
     * those are the right handle when the *incident edges* or *incident faces*
     * are wanted rather than the adjacent vertices.
     *
     * @warning **k = 2 is not available** — see gap G1 in the file header. A
     *          two-ring CSR can be *asked for*, and Tessera will return one,
     *          but it is silently truncated within one hop of a partition
     *          boundary. Do not build the RHS's two-ring stencil this way.
     */
    AdjacencyCsr vertexOneRing()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "vertexOneRing" );
    }

    /**
     * @brief Face-to-face adjacency through shared edges.
     *
     * Port of mesh_solver.py::_face_neighbors (lines 1533-1540)
     *
     * Used to grow AMR marks by neighbor rings (T4a) and to build the
     * proximity-exclusion rings (T4b).
     *
     * **M1 GAP (G4) — Tessera does not provide this and it cannot be derived
     * locally.** The two candidate local derivations both fail:
     *
     *   - From `EdgeField::Faces`: those are gids of faces that may not be held
     *     locally at all, and Tessera carries them verbatim through migration
     *     without repair.
     *   - From the vertex one-ring: two faces sharing an *edge* share two
     *     vertices, so a vertex-incidence walk finds them — but only when both
     *     are locally held. The halo is the 1-deep closure of owned *vertices*,
     *     which does **not** guarantee an owned face's edge-neighbour across a
     *     partition boundary is present. Tessera's own `refine()` says this in
     *     as many words, which is why its 2:1 balance routes every cross-rank
     *     decision through an edge coordinator rather than through the halo.
     *
     * So a correct distributed face-neighbour relation needs communication, and
     * building it in Beatnik would mean writing exactly the halo/topology code
     * this adapter exists to avoid. Left throwing. The Tessera-side addition —
     * a `buildFaceAdjacency(mesh)` CSR over the same edge-coordinator machinery
     * `refine()` already runs — is specified in `tasks/framework.md`.
     */
    AdjacencyCsr faceAdjacency()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "faceAdjacency" );
    }

    //-----------------------------------------------------------------------//
    // Communication
    //
    // Thin, deliberately: the adapter's job here is to name Tessera's one
    // exchange and to record what it does *not* do. Beatnik implements no
    // haloing of its own.
    //-----------------------------------------------------------------------//

    /**
     * @brief Refresh every ghost entity from its owner.
     *
     * TESSERA MAPPING: `Tessera::haloExchange( mesh, halo )`.
     *
     * **Collective** on `comm()`; every rank must call it. Syncs the *whole
     * field pack of all three entity kinds* in one shot — positions, gids,
     * ownership, connectivity, and all three Beatnik vertex fields. There is no
     * way to exchange one field, so the "exchange the potential twice per RHS
     * evaluation" of risk R8 is two whole-mesh exchanges.
     *
     * Topology-preserving: does **not** bump `generation()`, so handles taken
     * before it stay valid.
     *
     * @pre The halo plans must be live. They are cleared by `refine()` and
     *      rebuilt by `redistribute()`; calling this in between is a silent
     *      no-op on an empty plan, **not** an error. `refine()` below therefore
     *      never returns with a cleared halo.
     */
    void haloExchange()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "haloExchange" );
    }

    //-----------------------------------------------------------------------//
    // Topological edits
    //-----------------------------------------------------------------------//

    /**
     * @brief Conforming red-green refinement of a marked face set.
     *
     * Port of mesh.py::refine_marked_faces (lines 570-730)
     *
     * A *red* (marked) face is split into four by bisecting all three edges. An
     * unmarked face sharing split edges is retriangulated so the result is
     * conforming (no hanging nodes). New vertices sit at the linear edge
     * midpoint — **not** projected back to any surface, since the interface is
     * the surface. Tessera's `DefaultRefinePolicy` implements exactly that, and
     * documents that AMR does not project onto the sphere.
     *
     * **Conforming red-green refinement is NATIVE** (T4a asked this directly).
     * Beatnik does not drive it edge by edge and does not implement the closure.
     * Tessera's `RefinementMode::Conforming` — the mode this mesh is
     * instantiated with — does the 1->4 red split, the cross-rank 2:1 level
     * balance, and the transient red/green/blue closure of the kept neighbours.
     *
     * TESSERA MAPPING: `Tessera::refine( mesh, halo, mask )`, then the
     * mandatory re-halo (see the file header); this method performs both, so it
     * returns with a live halo.
     *
     * @param marked Per-**owned-face** marks, `1` = refine. **M1 CHANGE — this
     *        is a host `std::vector<char>` sized `ownedFaceCount()`, not a
     *        device `MarkView` sized `Nf`.** Tessera's mask is host-side and
     *        owned-only. A device-computed indicator (T4a) must therefore be
     *        copied to the host and truncated to the owned range before it can
     *        be used, which is a real cost on the AMR path and is recorded as
     *        such in `tasks/framework.md`.
     * @return What the edit did; see `MeshEditReport`. **No parent/weight map
     *         — Tessera transfers the fields itself.**
     *
     * @note MPI. **M1 CHANGE — marks do NOT need reconciling first.** The
     *       pre-M1 header required the caller to close the mark set across rank
     *       boundaries via `reconcileRefinementMarks`. Tessera runs that itself:
     *       its Phase-1 coordinator drives a 2:1 mark-propagation fixpoint to
     *       convergence, guarded by an `MPI_Allreduce` and a hard iteration cap,
     *       and reports the round count in `MeshEditReport::balance_rounds`. So
     *       an arbitrary, unreconciled, rank-local mask is a legal input and
     *       `Beatnik_Communication.hpp::reconcileRefinementMarks` has no work
     *       left to do.
     *
     * @note The refined set is a **superset** of `marked` — the 2:1 fixpoint
     *       pulls in whatever it must, and the closure adds children on top. A
     *       `--max-faces` cap (risk R4) therefore cannot be enforced by
     *       trimming the mask exactly; it can only be approached from below.
     */
    MeshEditReport refine( const std::vector<char>& marked )
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
     * splits: unlike `refine`, the caller chooses *edges* rather than faces.
     *
     * **M1 GAP (G5a) — Tessera has no caller-driven edge split.** Its only
     * topological edit is the face-mask `refine()` above; the split-edge map it
     * carries internally (`RefineResult::midpoints`) is an output, not an
     * input. The closest expressible thing is to mark every face incident on a
     * wanted edge, which splits all three of that face's edges and pulls in the
     * 2:1 closure — a strictly larger, differently-shaped edit than the Python
     * performs. Left throwing rather than approximated, because silently
     * substituting a coarser edit would make T4b's mesh diverge from the
     * reference for a reason no test would attribute correctly.
     */
    template <class EdgeListView>
    MeshEditReport splitEdges( const EdgeListView& edges )
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
     *
     * **M1 GAP (G5b) — Tessera implements split-based refinement only.** Edge
     * collapse does not exist, at any level: no public entry point, no
     * primitive, and the data model has no coarsening path (`Level` only ever
     * rises). The README records this as a milestone-1 design limitation.
     * Neither the link condition nor the cross-rank owner-decides protocol the
     * pre-M1 header anticipated has anywhere to attach.
     */
    template <class EdgeListView>
    MeshEditReport collapseEdges( const EdgeListView& edges )
    {
        (void)edges;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "collapseEdges" );
    }

    /**
     * @brief Flip a set of interior edges to the opposite diagonal.
     *
     * Port of dynamic_remesh.py::flip_edges_for_quality (lines 408-458)
     *
     * A flip changes connectivity only, so per-vertex fields are untouched. A
     * flip is rejected if the opposite diagonal already exists (non-manifold)
     * or if either child face would invert relative to the pre-flip normal.
     *
     * **M1 GAP (G5c) — Tessera implements no edge flip**, same limitation as
     * `collapseEdges`. Note this is not merely a missing convenience: a flip
     * across a rank boundary needs the two incident faces, which the 1-deep
     * *vertex* halo does not guarantee are co-resident (the same reason
     * `faceAdjacency` is a gap), so it needs Tessera's edge-coordinator
     * machinery to be correct.
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
     * **M1 GAP (G5d), consequential only.** Compaction exists in the Python to
     * clean up after a collapse sweep. With no collapse (G5b) nothing orphans a
     * vertex, so there is nothing to compact and Tessera provides no such call.
     * If collapse is ever added Tessera-side, compaction must come with it.
     */
    MeshEditReport compact()
    {
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "compact" );
    }

    /**
     * @brief Overwrite vertex positions in place, connectivity unchanged.
     *
     * Used by the time integrator and by every purely geometric pass
     * (tangential relaxation, volume projection, implicit fairing).
     *
     * Writes the position slice directly. Purely geometric, so it does **not**
     * bump `generation()` and every outstanding handle stays valid — but it
     * *does* invalidate the cached `MeshGeometry`'s numerical content (the
     * accessor holds the live slice, so positions are seen; it is derived
     * quantities the caller has cached that go stale). Ghost positions are not
     * updated: write owned rows and `haloExchange()`.
     */
    template <class NewVertexView>
    void setVertices( const NewVertexView& vertices )
    {
        (void)vertices;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "setVertices" );
    }

    //-----------------------------------------------------------------------//
    // Redistribution (load balancing)
    //-----------------------------------------------------------------------//

    /**
     * @brief Recompute the partition and move entities to it.
     *
     * This is what `Beatnik_Communication.hpp::redistribute` (T5d) calls, and
     * it is also the mandatory post-`refine()` halo rebuild — `refine()` above
     * invokes it with an identity assignment for exactly that reason.
     *
     * TESSERA MAPPING, two forms, both of which rebuild the 1-deep ghost layer
     * and all three halo plans as a side effect:
     *
     *   - `rebalance == true`: `Tessera::loadBalance( mesh, halo )`. Computes a
     *     Zoltan2 geometric **MultiJagged** partition of the owned-face
     *     centroids (never RCB — Zoltan2's RCB is documented broken on
     *     Tuolumne) and migrates to it. Default imbalance tolerance 0.05.
     *   - `rebalance == false`: `Tessera::migrate( mesh, halo, dest )` with
     *     `dest[f] == rank()`, a legal no-op move whose side effect is the
     *     ownership recompute and ghost rebuild.
     *
     * An externally computed partition (e.g. one aligned with Canopy's FMM
     * tree, so the BR evaluation and the mesh agree on locality) is the same
     * `migrate()` call with a different `dest`, built from
     * `Tessera::ownedFaceCentroids( mesh )`. That is the documented Canopy
     * contract and is available to Beatnik unchanged — worth knowing at T3a.
     *
     * **Field data follows automatically.** Migration ships whole Cabana
     * tuples, so all three Beatnik vertex fields move with their vertices with
     * no per-field plumbing and no parent map. This is the same mechanism that
     * makes `MeshEditReport` carry no weights.
     *
     * COST, for T5d's benefit:
     *   - Host-orchestrated, not GPU-resident: entity payloads travel as tuple
     *     byte images over `allToAllV`, in four rounds (gather-missing,
     *     move, ownership/ghost discovery via gid coordinators, ghost fetch),
     *     then the local AoSoAs, both CSRs, the key tables and the three halo
     *     plans are rebuilt.
     *   - **M1 GAP (G8, scalability).** `Tessera::computeLoadBalance` gathers
     *     *every* rank's owned-face centroids and weights to **rank 0**, solves
     *     there over a `Teuchos::SerialComm`, and scatters the assignment back.
     *     That is deliberate — MultiJagged is not guaranteed deterministic
     *     across ranks — but it makes the partitioner's memory and time
     *     rank-0-bound in the global face count. Fine at T5d's 16-rank exit
     *     criterion; a ceiling at production scale. A distributed solve, or a
     *     sampled one, is the Tessera-side fix.
     *   - Bumps `generation()`: re-take every accessor afterwards.
     *
     * @param rebalance `true` to recompute the partition, `false` for the
     *        ownership/ghost rebuild alone.
     * @return Owned face count after the move, for the imbalance diagnostic
     *         T5d's exit criterion measures.
     */
    long long redistribute( bool rebalance )
    {
        (void)rebalance;
        BEATNIK_NOT_IMPLEMENTED( "SurfaceMesh", "redistribute" );
    }

  private:
    /// The Tessera mesh. All entity storage, connectivity, ownership and the
    /// user field pack live here.
    tessera_mesh_type _mesh;

    /// The three halo exchange plans. Cleared by `Tessera::refine`, rebuilt by
    /// `Tessera::migrate` / `Tessera::loadBalance`.
    tessera_halo_type _halo;

    /// Cached device-capturable geometry accessor (`faceVerts` / `edgeVerts` /
    /// the position slice), and the `generation()` it was built at. Rebuilt
    /// lazily when the mesh has moved past it.
    Tessera::MeshGeometry<tessera_mesh_type> _geometry;
    std::size_t _geometry_generation = static_cast<std::size_t>( -1 );

    /// Cached one-ring stencil (`buildVertexStencil(mesh, 1)`), same lazy
    /// rebuild rule. Generation-guarded by Tessera in its own right.
    Tessera::VertexStencil<MemorySpace> _one_ring;
    std::size_t _one_ring_generation = static_cast<std::size_t>( -1 );
};

} // namespace Beatnik

#endif // BEATNIK_MESHINTERFACE_HPP
