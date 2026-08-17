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
 * RECONCILED AGAINST TESSERA 2026-08-07 (task M1), REWORKED 2026-08-11 (the
 * M1 adapter rework, ahead of T1b). The pre-M1 version of this header was
 * written without reading `../tessera`; the M1 version was shaped by Tessera as
 * it stood on 2026-08-07, when eleven capabilities Beatnik needs were missing.
 * **Eight of those eleven have since landed upstream** (branch
 * `conforming-refinement`), so most of the "M1 GAP" text in the previous
 * revision described a Tessera that no longer exists and has been replaced by
 * the real call. Semantics that changed at M1 are still flagged **M1 CHANGE**;
 * text that changed in the rework is flagged **M1-REWORK**. The three gaps that
 * remain open — edge collapse, edge flip, compaction — are flagged **M1 GAP**
 * and recorded in `tasks/framework.md` (task M1, gaps G5b/G5c/G5d). *Gaps are
 * not worked around in Beatnik* — in particular Beatnik implements no haloing
 * and no partitioning of its own.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Tessera type. Everything the rest of the
 * code needs from Tessera — storage of vertices and connectivity, the
 * owned/ghost partition, adjacency, the topological edits, the halo, the global
 * reductions, and redistribution — passes through `SurfaceMesh` below. Where a
 * caller needs to spell a type that is Tessera's underneath, it spells the
 * Beatnik alias (`typename mesh_type::position_slice`, `::face_vertex_view`,
 * ...), never the Tessera name.
 *
 * WHAT TESSERA IS
 * ---------------
 * A distributed unstructured triangle-mesh library over Cabana + Kokkos. It
 * owns: entity storage (vertices / edges / faces, each a Cabana AoSoA with a
 * compile-time user field pack), global ids, the owned/ghost partition, an MPI
 * halo of **configurable depth**, both families of topological edit, the global
 * scalar reductions, Zoltan2 load balancing and migration, and HDF5/XDMF I/O.
 * It owns no physics and no discretization convention: normals are unoriented,
 * areas undefined, stencil weights are the caller's.
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
 *    should index with. It covers **all locally held entities (owned +
 *    ghost)**, so an operator over owned vertices can read its full ring.
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
 *   - `splitEdges()` inserts a midpoint vertex on each bisected edge and fills
 *     its position *and every vertex user field* from the two endpoints through
 *     a pluggable `RefinePolicy` (default: the linear average, i.e. exactly the
 *     `(0.5, 0.5)` weights the old `MeshEditResult` encoded). Existing vertex
 *     user fields are preserved, and **face** user fields are inherited
 *     verbatim by every child of a subdivided face — which is not the rule
 *     Beatnik's reference area wants, and is why `AdaptiveMesh::refine` wraps
 *     the call in the two local passes described there. Edge user fields are
 *     **reset**, which is why Beatnik declares none.
 *   - `migrate()` / `loadBalance()` move **whole Cabana tuples**, so every user
 *     field follows its entity across ranks with no per-field plumbing.
 *   - `haloExchange()` likewise syncs the whole tuple of every ghost.
 *
 * So a per-vertex field held in a `Kokkos::View` *outside* the Tessera mesh is
 * silently dropped by refinement and silently stale after migration. Beatnik's
 * evolved state therefore lives **in the mesh**, as the vertex user pack:
 *
 * | `VertexFieldId::` slot | Type | Meaning |
 * | --- | --- | --- |
 * | `Potential`        | `Real`    | velocity potential jump phi |
 * | `SheetVector`      | `Real[3]` | tangential sheet vector S |
 * | `MaterialPosition` | `Real[3]` | carried Lagrangian coordinate |
 *
 * The linear average is the correct transfer rule for all three (phi and the
 * material coordinate are interpolated at a split; the sheet vector under the
 * `Potential` model is a cache that `updateSheetVector` overwrites anyway), so
 * Tessera's `DefaultRefinePolicy` is used unchanged and Beatnik supplies no
 * policy of its own. If a conservative (rather than interpolatory) rule is ever
 * wanted for the sheet strength, it is a `RefinePolicy` subclass here, not a
 * change anywhere else.
 *
 * **Scratch fields are a different matter, and deliberately stay outside.** The
 * *derived* per-vertex quantities — vertex area, vertex normal, the volume
 * gradient, an assembled Laplacian — are recomputed from scratch after every
 * edit, so nothing is lost by holding them in plain `Kokkos::View`s outside the
 * mesh, and putting them in the pack would change the checkpoint schema (see
 * `VertexFieldId` below). They live in `Beatnik_MeshGeometry.hpp`, and the
 * assembly rule that makes that correct without a scatter-add is stated under
 * DISTRIBUTED ASSEMBLY below.
 *
 * **DONE at T1c.** `Beatnik_SurfaceState.hpp` used to declare its own
 * `Kokkos::View`s for the three *evolved* fields, and they were never
 * allocated. It now holds no storage at all and every method takes the mesh,
 * reading the `potential()` / `sheetVector()` / `materialPosition()` accessors
 * below. See that header's "THE FIELDS LIVE IN THE MESH" section for the table
 * of what changed.
 *
 * MPI DECOMPOSITION AND THE HALO
 * ------------------------------
 * Ownership is Tessera's **lowest-rank rule**: a face is owned by its assigned
 * partition rank; a vertex or edge by the lowest-ranked owner of an incident
 * face. Local indices `[0, ownedXCount())` are owned; `[ownedXCount(),
 * totalXCount())` are ghosts.
 *
 * **M1-REWORK — the halo takes a depth, and Beatnik sets it to 2 once.** At M1
 * the ghost layer was 1-deep and not configurable, which was risk R8's blocker:
 * the Beatnik RHS is a **two-ring** stencil (one surface gradient builds the
 * sheet vector from the potential, a second is taken of the Bernoulli
 * potential), and `buildVertexStencil(mesh, 2)` was *silently* short within one
 * hop of a partition boundary. Tessera now takes a `depth` on `distribute()`
 * and `rebuildHalo()`, reports it as `mesh.haloDepth()`, and **preserves it
 * across `splitEdges()` and `migrate()`**. So:
 *
 *   - `SurfaceMesh::halo_depth` is **2**, passed to `distribute()` exactly
 * once,
 *     inside the construction entry points below. Nothing downstream re-states
 *     it, and no code path may narrow it.
 *   - A stencil request with `k > haloDepth()` now **throws
 *     `std::invalid_argument`** naming both numbers, rather than returning
 *     short rows. `vertexOneRing()` (k = 1) and a future two-ring stencil are
 *     therefore either correct or loud, never quietly incomplete.
 *   - Two successive `haloExchange()` calls still do **not** substitute for
 *     depth — the second refreshes the same ghost set rather than widening it.
 *     Depth is a property of the local entity closure, not of how many times
 *     the plan is replayed. That was the trap R8 named; what remains of R8 is
 *     only forgetting to set the depth, which is now loud.
 *
 * `haloExchange()` is **collective** on the mesh communicator and is a pure
 * **gather** (owner -> ghost, overwrite) that syncs the *whole field pack of
 * all three entity kinds* in one shot — it is not per-field and not per-kind.
 * Beatnik cannot ask for "just the potential"; the cost of refreshing one field
 * is the cost of refreshing everything. That is why
 * `Beatnik_Communication.hpp` has one exchange entry point and not a per-field
 * family.
 *
 * DISTRIBUTED ASSEMBLY — WHY BEATNIK NEEDS NO SCATTER-ADD FOR ITS GEOMETRY
 * -----------------------------------------------------------------------
 * **M1-REWORK.** Tessera now provides the reverse halo,
 * `haloScatterAddVertices<FieldIndex>( mesh, halo )` — ghost -> owner, `+=`,
 * one
 * *named mesh field* per call. It closes M1 gap G2 and
 * `haloScatterAddVertexField()` below forwards to it.
 *
 * It is, however, **not** what the per-vertex geometry assembly needs, and the
 * reason is a property of Tessera's local set worth stating once here rather
 * than rediscovering per kernel. Tessera's local face set is *the owned faces
 * plus every face incident on an owned vertex* (and, at depth 2, more). So a
 * kernel that loops **all locally held faces** and accumulates into their
 * **local** corner slots gives every *owned* vertex contributions from its
 * complete incident-face set, with no communication at all and no
 * double-counting of an owned vertex. Ghost vertices end up holding partial
 * sums, which is harmless as long as consumers read owned rows — and is exactly
 * what the scatter-add contract would leave behind anyway.
 *
 * Two consequences:
 *   - `MeshGeometry::compute`, `SurfaceOperators::volumeGradient` and the
 *     assembled Laplacians are correct over owned vertices from a *local-face*
 *     loop, and must **not** be followed by a scatter-add: doing so would
 *     double-count. Their `@note MPI` blocks say so.
 *   - `haloScatterAddVertexField()` exists for the case the rule does not
 * cover:
 *     accumulating into a field that lives *in the mesh* from a loop over
 *     **owned** faces only. Nothing in T1b does that; it is kept because it is
 *     the only correct primitive for it and because Tessera's contract (ghosts
 *     untouched, not idempotent, peer-ordered summation) must be recorded
 *     somewhere in Beatnik.
 *
 * GLOBAL REDUCTIONS
 * -----------------
 * **M1-REWORK.** At M1 Tessera exposed only `globalMin`, and Beatnik was to
 * hand-roll the rest. It now provides `globalSum`, `globalMax`,
 * `globalAllFinite` and the exact integer counts
 * `globalOwnedVertices/Edges/Faces/Euler`. Beatnik's four `Comm::allReduce*`
 * wrappers and the three `global*Count()` accessors below forward to them; no
 * Beatnik code calls `MPI_Allreduce` directly. `globalSum` on `double` is
 * **not** bitwise reproducible across rank counts — that is risk R2, and
 * Tessera now states it too.
 *
 * GENERATION GUARD — THE LIFETIME CONTRACT
 * ----------------------------------------
 * Every handle Tessera hands out (position slice, CSR, key view,
 * `MeshGeometry`, `VertexStencil`, `FaceAdjacency`) is stamped with the mesh's
 * `generation()` counter at the moment it was taken. Any op that changes the
 * local entity count or the ghost set — `distribute`, `migrate`, `loadBalance`,
 * `refine`, `splitEdges`, any `resize` — bumps the counter, and **copying a
 * stale handle aborts with a diagnostic** rather than reading freed storage.
 * `haloExchange()` is topology-preserving and does *not* bump it.
 *
 * Beatnik must therefore re-take every accessor below after any topological
 * edit or redistribution. `generation()` is exposed so a caller can assert it,
 * and the adapter rebuilds its own cached `MeshGeometry`/stencil internally.
 * **Do not cache the return of `positions()`, `faceVertices()` or
 * `vertexOneRing()` across a mesh edit.**
 *
 * THE TWO EDITING FAMILIES ARE DISJOINT — AND BEATNIK USES EXACTLY ONE
 * --------------------------------------------------------------------
 * **M1-REWORK — a constraint that did not exist at M1; SETTLED at T4a.**
 * Tessera's topological edits fall into two families and **a mesh belongs to
 * exactly one**:
 *
 * | Family | Operations | Invariant kept | `Level` is |
 * | --- | --- | --- | --- |
 * | Hierarchical | `refine`, `refineLocal` | 2:1 balance + closure | binding |
 * | Remesh | `splitEdges`, later collapse/flip | conformity only | advisory |
 *
 * A mesh carries an `EditFamily` tag, `None` until its first topological edit
 * and fixed thereafter, and **each entry point throws `std::runtime_error`** if
 * the other family is then used on it. This is not a Beatnik-side check and
 * cannot be one.
 *
 * **Beatnik never uses the Hierarchical family.** Every topological edit, in
 * every configuration — the indicator-driven refiner (T4a) as much as the
 * metric remesher (T4b) — goes through the **Remesh** family, so Tessera's
 * guard is a backstop that should never fire. `SurfaceMesh::refine` was
 * **deleted** at T4a rather than left unused; see `tasks/framework.md`,
 * Phase 4, *The editing-family question — RESOLVED*. Two findings drove it:
 * the two adaptivity modes are mutually exclusive per run
 * (`run_adaptive_mesh_bubble.py:1424` versus `:1469-1471`), so no mesh ever
 * needed both; and `Tessera::splitEdges()` **is** `mesh.py::refine_marked_faces`
 * — for the mask "every edge of every marked face" the two are the same
 * algorithm — whereas `Tessera::refine()`'s transient closure is not, and would
 * diverge from the Python in face count from round 2 onward.
 *
 * REFINEMENT MODE
 * ---------------
 * The mesh is still instantiated `RefinementMode::Conforming`, which is
 * Tessera's default and what fixes the face tuple layout; nothing Beatnik calls
 * exercises the closure, because `splitEdges()` is conforming on exit with no
 * closure layer and no 2:1 pass. `Level` is therefore advisory throughout, and
 * every local face is part of the current triangulation.
 */

#ifndef BEATNIK_MESHINTERFACE_HPP
#define BEATNIK_MESHINTERFACE_HPP

#include <Beatnik_Types.hpp>

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
 * **M2 CHANGE — this order IS the checkpoint contract.** The pre-M2 note here
 * said the HDF5 schema was "keyed by name, not by this index". It is not:
 * `Tessera::writeMesh` names user fields *positionally*, so slot `N` is written
 * as `/vertices/u<N>` and nothing in the file records what it means. Reordering
 * this enum silently relabels every checkpoint already on disk, and
 * additionally invalidates any `RefinePolicy` written against it. Changing it
 * means changing the schema table in `Beatnik_IOInterface.hpp`, and `FIELD_MAP`
 * and `H5_PATH` under `tests/regression_tests/`, in one commit. The writer
 * emits `/beatnik/vertex_field_names` so the comparator can catch a mismatch.
 *
 * The same reasoning is why the *derived* geometry arrays are **not** in this
 * pack: adding one would silently append a `/vertices/u3` to every checkpoint.
 * See DISTRIBUTED ASSEMBLY in the file header for why they do not need to be.
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
 * @brief Beatnik's **face** user fields, as offsets into Tessera's face user
 *        pack. Added at T4a.
 *
 * The same reasoning as `VertexFieldId`, applied to per-face state: a
 * `Kokkos::View` of per-face values held *outside* the mesh is silently dropped
 * by `splitEdges()` (children get no value) and silently stale after
 * `migrate()`. Tessera inherits a parent's face user fields **verbatim** by its
 * children and ships them whole on migration, so anything per-face that must
 * survive an edit belongs here.
 *
 * Three slots, and each is here because it must cross one of those two events:
 *
 * | `FaceFieldId::` slot | Type | Meaning |
 * | --- | --- | --- |
 * | `ReferenceArea`      | `Real` | \f$A^{\text{ref}}_f\f$, the area the area-change indicator measures against. |
 * | `ReferenceCurvature` | `Real` | \f$\kappa^{\text{ref}}_f\f$, likewise for the curvature-change indicator. |
 * | `RefineMark`         | `Real` | `1` if the face is red this pass, else `0`. **Scratch between passes, but it must be halo-exchanged during one**, which is the whole of route (a) below. |
 *
 * **Why `RefineMark` is in the pack even though it is scratch.** T4a's mark
 * translation is route (a) of the two `tasks/framework.md` offered: a face-level
 * verdict is computed on **owned** faces, `haloExchange()`d, and each rank then
 * evaluates its **owned edges** from locally-resident faces. `haloExchange()` is
 * whole-tuple and addresses fields by their compile-time Cabana member index, so
 * there is no way to exchange a Beatnik-side view — the mark has to be a mesh
 * field or it cannot cross a rank boundary at all. The red-green balance
 * fixpoint re-exchanges it once per round, which is what makes its termination
 * test a single `MPI_Allreduce(MPI_LOR)` rather than a mark-propagation
 * protocol of Beatnik's own. The per-face **score** deliberately stays outside
 * the mesh: only owned faces are ever thresholded, so it never needs to cross.
 *
 * **T4a CHANGE — this widens every checkpoint**, exactly as risk R14 predicted:
 * `Tessera::writeMesh` writes the face user pack unconditionally, so a file now
 * carries `/faces/u0`, `/faces/u1` and `/faces/u2`. Two consequences:
 *
 *   1. A checkpoint written by a pre-T4a binary is **not readable** by a
 *      post-T4a one — `Tessera::readMesh` treats a field-pack mismatch as an
 *      `MPI_Abort` inside Tessera, not a catchable exception (M2's trap (b)).
 *      Nothing depends on it yet (`CheckpointIO::read` still throws, T5b), but
 *      T5b must not assume the two packs are compatible.
 *   2. `/faces/u<N>` is **positional**, so reordering this enum silently
 *      relabels every file on disk. The same mitigation the vertex pack got at
 *      M2 applies: the writer emits `/beatnik/face_field_names` alongside
 *      `/beatnik/vertex_field_names`, and `compare_output.py` cross-checks it.
 */
namespace FaceFieldId
{
enum : int
{
    ReferenceArea = 0,      ///< `Real`, reference face area.
    ReferenceCurvature = 1, ///< `Real`, reference face curvature indicator.
    RefineMark = 2,         ///< `Real`, 1 = red this pass. Scratch, exchanged.
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
 * `Tessera::RefineResult` / `Tessera::SplitResult` actually report, plus the
 * entity counts a caller wants for a progress line.
 *
 * `MeshEditResult` is deleted rather than kept as a shim, because a shim would
 * have to fabricate a parent map that no Beatnik code could correctly consume.
 */
struct MeshEditReport
{
    /// 2:1 mark-propagation rounds Tessera's cross-rank fixpoint executed.
    /// Tessera's own diagnostic; a value that climbs run over run means the
    /// indicator is marking in a pattern that needs a lot of closure. Always 0
    /// for a remesh-family edit, which runs no balance pass.
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

    /**
     * @brief Ghost rings the local closure carries. **Two, set once.**
     *
     * The Beatnik RHS is a two-ring stencil on the potential (risk R8), so
     * every construction entry point passes this to `Tessera::distribute` and
     * nothing afterwards re-states it: Tessera's `splitEdges()` and
     * `migrate()` preserve the recorded depth. Raising it costs ghost memory
     * and halo traffic; **lowering it silently breaks the RHS at partition
     * boundaries**, which is why it is a compile-time constant here rather than
     * a per-call argument.
     */
    static constexpr int halo_depth = 2;

  private:
    /// Beatnik's vertex user pack, in `VertexFieldId` order.
    using vertex_fields = Tessera::VertexFields<Real, Real[3], Real[3]>;

    /// No per-edge user state. Tessera **resets** edge user fields on every
    /// `splitEdges()` (edges are re-derived from the new face connectivity), so
    /// an edge field would be unsafe to carry anyway. That is also why the
    /// refinement mask is passed as a plain host vector rather than parked in
    /// an edge field: an edge field could not survive the edit that consumes
    /// it.
    using edge_fields = Tessera::EdgeFields<>;

    /// Beatnik's face user pack, in `FaceFieldId` order. **T4a CHANGE — this
    /// was `FaceFields<>`.** See `FaceFieldId` for what each slot is for and
    /// for the checkpoint consequences (risk R14).
    using face_fields = Tessera::FaceFields<Real, Real, Real>;

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
     * It deliberately exposes **no extent**: the guard forwards `operator()`
     * only. A routine that needs `Nv` takes it as an argument from
     * `totalVertexCount()` / `ownedVertexCount()`, which is why
     * `MeshGeometry::compute` has an explicit `vertex_count` parameter.
     *
     * Callers outside this header spell it `typename
     * mesh_type::position_slice`.
     */
    using position_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template vertexSlice<Tessera::VertexField::Position>() );

    /// `(Nv,)` scalar vertex user field slice (e.g. the potential).
    using scalar_field_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template vertexSlice<Tessera::userVertexField<
                          VertexFieldId::Potential>()>() );

    /**
     * @brief `(Nf,)` scalar **face** user field slice. T4a.
     *
     * The face analogue of `scalar_field_slice`, and it carries the same
     * generation guard and the same "exposes no extent" property — a routine
     * that needs `Nf` takes it from `ownedFaceCount()` / `totalFaceCount()`.
     */
    using face_scalar_slice =
        decltype( std::declval<tessera_mesh_type&>()
                      .template faceSlice<Tessera::userFaceField<
                          FaceFieldId::ReferenceArea>()>() );

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
     * topology generation. A corner whose gid is not held locally is `-1`
     * (`Tessera::invalid_local`), which a kernel must **skip** rather than
     * index with. That is unreachable for an *owned* face, whose corners are
     * owned or ghosted by construction, but it is reachable for a ghost face
     * acquired at the outermost ring — so every kernel below guards its corner
     * indices. Plain Kokkos, so it names no Tessera type.
     */
    using face_vertex_view = Kokkos::View<int* [3], memory_space>;

    /// `(Ne, 2)` local vertex indices per edge. Same derivation and caveat.
    /// This is the **unique**-edge list Tessera maintains as a first-class
    /// entity kind, so it is also the answer to "enumerate the edges" for the
    /// edge-length reduction.
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
     * @brief Edge-to-face incidence, two ways, because the two answer different
     *        questions and **only one of them survives a topological edit**.
     *
     * M1-REWORK, corrected at T4a.
     *
     * `count` / `faces` are a *read* of Tessera's `EdgeField::Faces`, which
     * holds an edge's incident face **gids**. Immediately after `distribute()`
     * that field is complete on every rank that holds the edge, because
     * distribution cuts a replicated mesh in which every rank knew both
     * incidences — which is what made `count == 2` a global closed-surface
     * assertion at T1b.
     *
     * **It does not stay complete.** Tessera fills `EdgeField::Faces` from
     * *each rank's own incidences only* — its own words, at
     * `../tessera/src/Tessera_DistributedBuilder.hpp:698`: "the same
     * partial-by-construction contract `migrate()` leaves, and the reason
     * `buildFaceAdjacency()` exists rather than reading this field". So after
     * `splitEdges()` rebuilds the edge set from each rank's local faces, an edge
     * on a partition boundary records only the face on this side and `count`
     * drops to 1 for a perfectly conforming mesh. **Measured at T4a:** 0 such
     * edges at 1 rank, 24 at 2, 45 at 3, 104 at 6 — a partition-boundary
     * population, not a hole.
     *
     * `resident_count` / `resident_faces` are therefore derived the other way,
     * from `FaceField::Edges` over **all locally held faces** — the face's own
     * record of which edges it uses, which is complete for every resident face
     * at every generation. That makes `resident_count(e) == 2` for an owned edge
     * mean exactly "this rank can see both faces of this edge", which is
     * simultaneously the conformity statement and the residency precondition
     * `Beatnik_AdaptiveMesh.hpp`'s route (a) rests on. **Post-edit code wants
     * this pair; `count` is only trustworthy before the first edit.**
     */
    struct EdgeFaceIncidence
    {
        /// `(Ne,)` incident faces **recorded by gid** for each local edge.
        /// Complete only until the first topological edit; see above.
        Kokkos::View<int*, memory_space> count;

        /// `(Ne, 2)` local index of each gid-recorded incident face, or `-1`
        /// when that face is recorded but not held here. **A `-1` is not a
        /// hole**; check `count`.
        Kokkos::View<int* [2], memory_space> faces;

        /// `(Ne,)` **locally resident** faces incident on each local edge,
        /// counted from the faces themselves. T4a. `2` for every owned edge of
        /// a closed conforming surface at `halo_depth = 2`, at every generation;
        /// a ghost edge at the outermost ring may legitimately read `1`. A value
        /// above 2 is a non-manifold edge, and only the first two are stored.
        Kokkos::View<int*, memory_space> resident_count;

        /// `(Ne, 2)` local indices of those faces, ascending, `-1` in unused
        /// slots. Never `-1` below `resident_count`.
        Kokkos::View<int* [2], memory_space> resident_faces;
    };

    /**
     * @brief Face-to-face adjacency through shared edges, both halves.
     *
     * M1-REWORK — this is `Tessera::FaceAdjacency`, adapted. It has two halves
     * and **which one a consumer may use is a precondition, not a preference**:
     *
     *   - `neighbor_gid` / `neighbor_owner` are **always valid**, resident or
     *     not. A *topological* consumer — AMR mark growth by neighbour rings
     *     (T4a), remesh conflict resolution (T4b) — uses these: it sends to the
     *     owner and names the face by gid, and never needs it locally.
     *   - `neighbors` (local indices) is usable only where the neighbour is
     *     resident, and `-1` elsewhere. A *geometric* consumer, one that reads
     *     the neighbour's vertex positions, must **check
     *     `non_resident == 0`** rather than assume it.
     *
     * Rows are sorted ascending by neighbour **gid**, not by local index, so
     * row order is rank-count invariant and two runs at different rank counts
     * compare as exact lists rather than as sets.
     */
    struct FaceAdjacencyCsr
    {
        /// Row per **local** face; `offsets(f) .. offsets(f+1)` index the three
        /// parallel arrays below. Rows for owned faces are complete; rows for
        /// **ghost** faces are best-effort and may be short — do not iterate
        /// them.
        Kokkos::View<int*, memory_space> offsets;
        /// Local face index of each neighbour, or `-1` when not held here.
        Kokkos::View<int*, memory_space> neighbors;
        /// Global id of each neighbour. Always valid.
        Kokkos::View<std::uint64_t*, memory_space> neighbor_gid;
        /// Owning rank of each neighbour. Always valid.
        Kokkos::View<std::int32_t*, memory_space> neighbor_owner;
        /// Owned-row entries that are `-1`, summed over this rank. Zero means a
        /// geometric consumer may use `neighbors` directly.
        long long non_resident = 0;
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
     * @brief The underlying Tessera mesh. **For the sibling adapters only.**
     *
     * **T1c CHANGE, and the narrowest one that made `CheckpointIO::write`
     * possible.** `Tessera::writeMesh` / `readMesh` take the Tessera `Mesh`
     * itself — they are whole-mesh operations over storage, connectivity,
     * ownership and the user pack at once, so there is no subset of this facade
     * that could stand in for it, and re-exposing each piece would be a worse
     * leak than one accessor.
     *
     * The adapter contract is unbroken because it is scoped to *"no **other**
     * Beatnik header may name a Tessera type"* and
     * `Beatnik_IOInterface.hpp` is **adapter 2 of 3**. Nothing outside the
     * three adapter headers may call this. It is a deliberate hole with a
     * named caller, not a general escape hatch: if a fourth caller ever wants
     * it, the right move is to add the operation here rather than to widen the
     * hole.
     */
    tessera_mesh_type& tesseraMesh() { return _mesh; }

    /// Const overload of `tesseraMesh()`; `Tessera::writeMesh` takes its mesh
    /// by const reference, so the write path needs only this one.
    const tessera_mesh_type& tesseraMesh() const { return _mesh; }

    /// The three halo plans, for the same two callers and the same reason:
    /// `Tessera::readMesh` and `rebuildHalo` need the halo, not the mesh alone.
    tessera_halo_type& tesseraHalo() { return _halo; }

    /**
     * @brief Tessera's topology generation counter.
     *
     * Bumped by every op that changes the local entity count or the ghost set.
     * Any accessor taken at generation `g` aborts if copied once the mesh has
     * moved past `g`. Exposed so a caller can assert its cached handles are
     * still current rather than discovering it inside a kernel launch.
     */
    std::size_t generation() const { return _mesh.generation(); }

    /**
     * @brief Ghost rings actually in force, as Tessera reports them.
     *
     * Must equal `halo_depth` after any construction entry point. Exposed so a
     * test can assert the depth was set rather than inferring it from a stencil
     * that happens not to be short — see risk R8.
     */
    int haloDepth() const { return _mesh.haloDepth(); }

    //-----------------------------------------------------------------------//
    // Construction
    //
    // All three entry points below produce a *replicated* mesh — identical on
    // every rank, every entity owned — and then call `Tessera::distribute()` to
    // cut it to owned + a `halo_depth`-deep ghost layer and build the three
    // halo plans.
    //
    // M1-REWORK (G6 closed). Tessera now also offers
    // `buildIcosphereDistributed()` / `buildFromTriangleSoupDistributed()`,
    // which never materialize the global mesh on any rank and produce a
    // **bitwise identical** vertex-position multiset to the replicated path.
    // Beatnik keeps the replicated path here because Tessera documents it as
    // the right choice for a small initial mesh, and the default subdivision 2
    // is 162 vertices. Switch when an initial mesh is generated at a resolution
    // comparable to the refined running mesh; it is a two-line change and needs
    // no canonical keys for the icosphere.
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
     * \f$N_f = 20\cdot 4^{s}\f$ — 162 / 320 at the default \f$s = 2\f$.
     *
     * TESSERA MAPPING
     * ---------------
     *   1. `Tessera::buildIcosphere( mesh, subdivisions )` — generates the
     *      triangle soup and derives full connectivity (edges, both CSR
     *      one-rings, the canonical key side tables). Its base table and its
     *      face list are **the same literals** as the Python's, and its
     *      midpoint rule is `normalize3( 0.5*(a+b) )` against the Python's
     *      `(a+b)/‖a+b‖` — identical up to the shared scaling of numerator and
     *      denominator, and to the reciprocal-multiply-versus-divide difference
     *      noted below.
     *   2. Scale by `radius` and translate by `center`, in place on the
     * position
     *      slice, as `c + r\,\hat v` — the Python's association
     *      (`center[None,:] + radius * vertices`), which matters in the last
     *      bit. **M1 CHANGE — Tessera's generator is unit-sphere only and takes
     *      neither argument**, so this is Beatnik's step, not a parameter
     *      forwarded to Tessera.
     *   3. `Tessera::facePartitionByAxis( mesh, axis = 2 )` then
     *      `Tessera::distribute( mesh, halo, faceOwner, halo_depth )` then
     *      `Tessera::haloExchange( mesh, halo )`.
     *   4. Verify the winding, see below.
     *
     * ORIENTATION IS VERIFIED, NOT ASSUMED. Tessera's base table is documented
     * CCW seen from outside and subdivision preserves winding, so faces come
     * out outward-oriented as the Python requires — but Tessera imposes no
     * orientation convention of its own (`faceNormalRaw` is explicitly
     * unoriented), so an inward mesh would flip the sign of every normal, every
     * curvature and the enclosed volume downstream and nothing else would
     * notice. This routine therefore computes the enclosed volume
     * \f$V = \tfrac16\sum_f a_f\cdot(b_f\times c_f)\f$ over **owned** faces,
     * reduces it, and throws `std::runtime_error` unless \f$V > 0\f$. The
     * Python instead *repairs* the winding face by face
     * (`icosphere_mesh` lines 452-461); Beatnik rejects rather than repairs,
     * because a generator that needs repairing is a Tessera bug to report, not
     * a condition to absorb silently.
     *
     * @param subdivisions Subdivision level \f$s \ge 0\f$.
     * @param radius       Sphere radius, in problem length units.
     * @param center       Sphere centre \f$(c_x, c_y, c_z)\f$.
     *
     * @note REPRODUCIBILITY. The vertex *ordering* produced here must not be
     *       assumed to match the Python — it does not, since Tessera's
     *       subdivision loop numbers midpoints in its own order. The regression
     *       comparator sorts on quantized coordinates precisely so it does not
     *       have to. What *must* match is the vertex set and the positions, to
     *       comparison tolerance; the positions differ from the Python's in the
     *       last bits only, because `Tessera::detail::normalize3` multiplies by
     *       a reciprocal where NumPy divides. See `tasks/framework.md` risk R1.
     *
     * @note MPI. `Tessera::distribute` and `haloExchange` are collective, and
     *       so is the orientation reduction. Every rank must call this.
     */
    void generateIcosphere( int subdivisions, Real radius,
                            const Real center[3] )
    {
        Tessera::buildIcosphere( _mesh, subdivisions );
        scaleAndTranslate( radius, center );
        distributeReplicated();
        requireOutwardWinding( "generateIcosphere" );
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
     * **M1-REWORK (G7 closed) — Tessera now has this generator.**
     * `Tessera::buildLatLonSphere( mesh, nLat, nLon )` produces exactly this
     * surface, with the four classic UV-sphere traps pinned by its own tests:
     * exact literal poles (a computed south pole is off the unit sphere in its
     * last bits *and different for different phi*), no duplicated seam
     * meridian, a fixed quad diagonal, and CCW-outward winding matching the
     * icosphere. So Beatnik no longer builds the soup itself.
     *
     * Two parameter differences to reconcile when this is implemented (T5a-era;
     * `--mesh-kind latlon` is on no regression path today, which is why the
     * body still throws):
     *   - Tessera's `nLat` counts latitude rings **including** both poles, so
     *     `nLat = n_theta + 1` against the Python's convention.
     *   - Tessera's generator is unit-sphere only, like the icosphere's, so
     *     `radius` / `center` are applied here exactly as in
     *     `generateIcosphere` step 2.
     *
     * @note Tessera documents that these positions come from `sin`/`cos` at
     *       computed angles and are **not** bit-reproducible across libm
     *       implementations, unlike the icosphere's rational base table plus
     *       `sqrt`. That is exactly the `latlon` half of risk R1: a gold-file
     *       comparison for this mesh kind needs a tolerance, not equality.
     *
     * @param n_theta Latitude bands, \f$\ge 2\f$ (Python convention).
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
     * This is the path used by the "read the initial mesh from the gold file"
     * mitigation for regression test 1 (`tasks/framework.md`, risk R1).
     * **M2 CHANGE — it is *not* the `--restart-from` path**, which goes through
     * `Tessera::readMesh` and never materializes the global mesh anywhere.
     *
     * TESSERA MAPPING. `Tessera::buildFromTriangleSoup( mesh, soup )` where
     * `soup.positions` is `3*Nv` scalars and `soup.triangles` is `3*Nf` vertex
     * indices, then partition + distribute + halo exchange as in
     * `generateIcosphere` step 3.
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
     * The winding is **not** verified here, unlike `generateIcosphere`: an
     * adopted surface's orientation is a property of the file it came from, and
     * a caller reading a Python gold file wants the mesh it was given rather
     * than an exception. Verify at the call site if the source is untrusted.
     *
     * @param vertices `(Nv, 3)` positions, identical on every rank.
     * @param faces    `(Nf, 3)` connectivity, identical on every rank.
     *
     * @tparam HostVertexView, HostFaceView Host-accessible, `(i, d)`-indexable
     *         with an `extent(0)`. These stay templated: Tessera's soup is
     *         `std::vector`-backed and Beatnik's readers produce Kokkos host
     *         views, so `adopt` transcribes rather than forwards.
     */
    template <class HostVertexView, class HostFaceView>
    void adopt( const HostVertexView& vertices, const HostFaceView& faces )
    {
        const std::size_t nv = static_cast<std::size_t>( vertices.extent( 0 ) );
        const std::size_t nf = static_cast<std::size_t>( faces.extent( 0 ) );

        Tessera::TriangleSoup<Real> soup;
        soup.positions.resize( 3 * nv );
        soup.triangles.resize( 3 * nf );
        for ( std::size_t i = 0; i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                soup.positions[3 * i + d] =
                    static_cast<Real>( vertices( i, d ) );
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
                soup.triangles[3 * f + k] = static_cast<int>( faces( f, k ) );

        Tessera::buildFromTriangleSoup( _mesh, soup );
        distributeReplicated();
    }

    //-----------------------------------------------------------------------//
    // Accessors
    //
    // Every one of these returns a generation-stamped handle, or a view derived
    // from one and cached against the same counter. Re-take them after any
    // edit; see the lifetime contract in the file header.
    //-----------------------------------------------------------------------//

    /**
     * @brief Owned + ghost vertex positions, `pos(i, d)`.
     *
     * Rows `[0, ownedVertexCount())` are owned; the remainder are ghosts, whose
     * values are current only after a `haloExchange()`.
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
     * Covers owned **and** ghost faces, which is what makes the local-face
     * assembly rule in the file header correct. Derived from the gid-based
     * storage by `Tessera::buildMeshGeometry`, which the adapter caches and
     * rebuilds on a generation change — so this is O(1) on the common path and
     * a host-side `gid -> local` pass on the first call after an edit.
     *
     * A corner index may be `-1`; see `face_vertex_view`. Guard it.
     */
    face_vertex_view faceVertices()
    {
        ensureGeometry();
        return _geometry.faceVerts;
    }

    /**
     * @brief `(Nf,)` reference face area \f$A^{\text{ref}}_f\f$. T4a.
     *
     * What `AdaptiveMesh::areaChangeIndicator` measures the current area
     * against. Set to the current areas at construction and re-based after any
     * edit that legitimately changes areas
     * (`AdaptiveMesh::resetReferenceState`). Inherited **verbatim** by a
     * `splitEdges()` child, which is *not* the rule the Python uses — see
     * `AdaptiveMesh::refine`'s \f$\sigma = A^{\text{ref}}/A\f$ round trip, which
     * is where the reference's per-child area scaling actually happens.
     *
     * Covers owned **and** ghost faces; ghost rows are current only after a
     * `haloExchange()`.
     */
    face_scalar_slice referenceFaceArea()
    {
        return _mesh.template faceSlice<
            Tessera::userFaceField<FaceFieldId::ReferenceArea>()>();
    }

    /// `(Nf,)` reference face curvature indicator. T4a. Same storage contract
    /// as `referenceFaceArea()`; **reset**, not scaled, for a subdivided face.
    face_scalar_slice referenceFaceCurvature()
    {
        return _mesh.template faceSlice<
            Tessera::userFaceField<FaceFieldId::ReferenceCurvature>()>();
    }

    /// `(Nf,)` red-refinement mark, `1` = this face bisects all three of its
    /// edges. T4a. Scratch between refinement passes, but a **mesh field**
    /// because the balance fixpoint has to halo-exchange it — see
    /// `FaceFieldId`.
    face_scalar_slice refineMark()
    {
        return _mesh.template faceSlice<
            Tessera::userFaceField<FaceFieldId::RefineMark>()>();
    }

    /// `(Ne, 2)` local vertex indices per edge, from the same accessor. This is
    /// the unique-edge list, so it is also the answer to "enumerate the edges"
    /// for the edge-length reduction — and the reason
    /// `SurfaceOperators::edgeLengths` takes edges rather than deriving them
    /// from faces the way the Python must.
    edge_vertex_view edgeVertices()
    {
        ensureGeometry();
        return _geometry.edgeVerts;
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
     * **M1-REWORK (G3 closed).** At M1 Tessera exposed only `globalMin` and
     * this was to be a hand-rolled `MPI_Allreduce(MPI_SUM)` routed through
     * `Comm::allReduceSum`. `Tessera::globalOwnedVertices( mesh )` now does it,
     * as a `long long` — **exact, because integer**, unlike the floating-point
     * sums of risk R2. Summing the **owned** count is what makes this a
     * partition and not a double count.
     *
     * Involves a collective; every rank must call it. Not cached: it is one
     * `MPI_Allreduce` of one `long long`, and a cache keyed on `generation()`
     * would have to be invalidated by `haloExchange()` too (which does not bump
     * the counter) if the ghost set ever entered the answer. It does not, but
     * the cache would be a standing invitation to that bug for no measurable
     * gain.
     */
    GlobalIndex globalVertexCount() const
    {
        return static_cast<GlobalIndex>(
            Tessera::globalOwnedVertices( _mesh ) );
    }

    /// Global unique-edge count. Same reduction and exactness as
    /// `globalVertexCount`. Needed by the Euler check \f$V - E + F = 2\f$,
    /// which is the cheapest structural test of a closed surface there is.
    GlobalIndex globalEdgeCount() const
    {
        return static_cast<GlobalIndex>( Tessera::globalOwnedEdges( _mesh ) );
    }

    /// Global face count. Same reduction and exactness as `globalVertexCount`.
    GlobalIndex globalFaceCount() const
    {
        return static_cast<GlobalIndex>( Tessera::globalOwnedFaces( _mesh ) );
    }

    /**
     * @brief Euler characteristic \f$V - E + F\f$ over owned entities.
     *
     * **2** for a closed conforming surface, at every rank count. Tessera
     * reduces the three owned counts in one collective
     * (`Tessera::globalOwnedEuler`), which is why this is not
     * `globalVertexCount() - globalEdgeCount() + globalFaceCount()`.
     */
    GlobalIndex globalEulerCharacteristic() const
    {
        return static_cast<GlobalIndex>( Tessera::globalOwnedEuler( _mesh ) );
    }

    //-----------------------------------------------------------------------//
    // Adjacency
    //
    // M1 CHANGE — the three pre-M1 "build*Adjacency" calls do not survive as
    // three symmetric builders, because Tessera does not treat the three
    // relations symmetrically. Edge-to-face is maintained continuously and is
    // only *read* here; the vertex one-ring is a cheap local BFS Tessera
    // exposes as a stencil; face-to-face genuinely needs communication and is a
    // collective builder. All three are cached against `generation()` where the
    // build is not free.
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
     * incident face gids, `Tessera::invalid_gid` where absent. So this reads
     * that relation and resolves it to local indices; `edgeVertices()` above
     * gives the endpoints.
     *
     * The Python's closed-surface check carries over unchanged and is worth
     * keeping: on a closed surface every edge must name two faces, i.e.
     * `count(e) == 2` for every local edge. A count of 1 signals a genuine
     * hole. **M1-REWORK — the pre-rework caveat that this is only meaningful on
     * a halo-consistent mesh no longer applies**: `splitEdges()` rebuilds the
     * halo itself before returning, so there is no window in which the mesh is
     * left with a cleared halo.
     *
     * @note `EdgeField::Faces` stores **gids**, and Tessera carries them
     *       verbatim through migration — an edge's incident-face gid may name a
     *       face now owned elsewhere and not held locally, which is exactly why
     *       `faces` can hold `-1` while `count` is 2. It is metadata, not a
     *       usable local index. Do not build a face neighbourhood from it; see
     *       `faceAdjacency()` below, which is a collective for this reason.
     *
     * Cached against `generation()`: the host pass is `O(Ne + Nf)`.
     */
    EdgeFaceIncidence edgeAdjacency()
    {
        ensureEdgeFaceIncidence();
        return _edge_faces;
    }

    /**
     * @brief `(Nf, 3)` **local** edge indices per face. T4a.
     *
     * The inverse of `edgeAdjacency()`, and the one new adapter accessor T4a
     * needed. Tessera publishes no face->edge CSR — it has `vertexEdges()`, and
     * the face AoSoA carries `FaceField::Edges` as **gids** — so this is the
     * same host `gid -> local` resolution `edgeAdjacency()` does, run the other
     * way, and cached against `generation()` for the same reason.
     *
     * It exists because the whole of T4a is a translation between a face-level
     * verdict and an edge-level mask, in both directions: "mark all three edges
     * of this face" needs face->edge, and "how many of this face's edges are
     * marked" (\f$|S_f|\f$, which decides the child count and therefore
     * `AdaptiveMesh::projectedFaceCount`) needs it too. Doing it inline in the
     * AMR code would put a host-side gid map inside a per-pass loop.
     *
     * Edge slot `k` is the edge Tessera records in `FaceField::Edges[k]`; it is
     * **not** promised to be the edge between corners `k` and `k+1` of
     * `faceVertices()`, so a kernel that needs the endpoints of a face's edge
     * must read them from `edgeVertices()` rather than assume the pairing.
     *
     * An entry is `-1` when the named edge is not held on this rank. That is
     * unreachable for an **owned** face at `halo_depth = 2` — every edge of an
     * owned face is an edge of a locally held face — and `AdaptiveMesh` asserts
     * it rather than assuming it. Guard it anyway, exactly as
     * `face_vertex_view`'s `-1` corners are guarded.
     */
    face_vertex_view faceEdges()
    {
        ensureFaceEdges();
        return _face_edges;
    }

    /**
     * @brief Global ids of this rank's **owned** faces, on the host, in local
     *        index order. T4a.
     *
     * The discriminator for "was this face subdivided?", which nothing else in
     * the adapter can answer: a face with \f$|S| = 0\f$ **keeps its gid**
     * through `splitEdges()` while a subdivided parent's gid is retired and its
     * children get fresh ones from the child-gid exscan
     * (`../tessera/src/Tessera_EdgeSplit.hpp`, the |S| table). So a snapshot
     * taken before the call, differenced against the gids after it, is exactly
     * the set of new faces — which is what the Python's "reset reference
     * curvature for subdivided faces only" rule needs and what `RefinePolicy`
     * cannot supply (its two hooks are `interpolatePosition` and
     * `interpolateVertexField`; face fields are inherited verbatim and vertex
     * fields are all it can touch).
     *
     * A host `std::vector` and not a device view on purpose: its consumer is a
     * set membership test against the post-edit gids, which is a host hash
     * lookup, and it is `O(ownedFaceCount())` once per refinement pass.
     */
    std::vector<std::uint64_t> ownedFaceGids()
    {
        const std::size_t nf = static_cast<std::size_t>( ownedFaceCount() );
        Cabana::AoSoA<typename tessera_mesh_type::face_member_types,
                      Kokkos::HostSpace>
            hf( "beatnik_owned_face_gids", _mesh.numFaces() );
        Cabana::deep_copy( hf, _mesh.faces() );
        auto f_gid = Cabana::slice<Tessera::FaceField::Gid>( hf );

        std::vector<std::uint64_t> out( nf );
        for ( std::size_t f = 0; f < nf; ++f )
            out[f] = static_cast<std::uint64_t>( f_gid( f ) );
        return out;
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
     * local index for determinism). Complete for every owned vertex.
     *
     * The adapter caches it and rebuilds on a generation change. Tessera also
     * maintains `vertexEdges()` and `vertexFaces()` CSRs directly on the mesh;
     * those are the right handle when the *incident edges* or *incident faces*
     * are wanted rather than the adjacent vertices.
     *
     * **M1-REWORK — k = 2 is now available and checked.** The pre-rework
     * warning here said a two-ring CSR would be silently truncated near a
     * partition boundary. It no longer can be: the mesh is built at
     * `halo_depth = 2`, and `buildVertexStencil` **throws
     * `std::invalid_argument`** naming `k` and the depth if `k` exceeds it. The
     * RHS's two-ring stencil is therefore expressible; it is not built here
     * only because no T1b caller needs it yet.
     */
    AdjacencyCsr vertexOneRing()
    {
        ensureOneRing();
        const auto& csr = _one_ring.csr.get();
        AdjacencyCsr out;
        out.offsets = csr.offsets;
        out.neighbors = csr.neighbors;
        return out;
    }

    /**
     * @brief Face-to-face adjacency through shared edges.
     *
     * Port of mesh_solver.py::_face_neighbors (lines 1533-1540)
     *
     * Used to grow AMR marks by neighbour rings (T4a) and to build the
     * proximity-exclusion rings (T4b).
     *
     * **M1-REWORK (G4 closed) — `Tessera::buildFaceAdjacency( mesh )` provides
     * it, collectively.** The pre-rework header recorded this as a gap, and the
     * reasoning it gave for why it *cannot* be derived locally is still exactly
     * right and is why the builder is a collective rather than a walk:
     *
     *   - From `EdgeField::Faces`: those are gids of faces that may not be held
     *     locally at all, carried verbatim through migration without repair.
     *   - From the vertex one-ring: two faces sharing an *edge* share two
     *     vertices, so a vertex-incidence walk finds them — but only when both
     *     are locally held, and the vertex-ring closure does not guarantee an
     *     owned face's edge-neighbour is present.
     *
     * Tessera reuses the **edge coordinator** rather than the halo,
     * adding no new communication mechanism, and returns both halves — see
     * `FaceAdjacencyCsr`, whose precondition split (`non_resident == 0` for a
     * geometric consumer) is the part to read before using it.
     *
     * On this `Conforming` mesh every face is a visible face, so every row has
     * degree exactly 3 on a closed surface and there is nothing to skip.
     *
     * @note MPI. **Collective** and generation-guarded. Cached here against
     *       `generation()`, which also means every rank must reach the first
     *       call after an edit — do not call it inside a rank-local branch.
     */
    FaceAdjacencyCsr faceAdjacency()
    {
        ensureFaceAdjacency();
        const auto& csr = _face_adj.csr.get();
        FaceAdjacencyCsr out;
        out.offsets = csr.offsets;
        out.neighbors = csr.neighbors;
        out.neighbor_gid = _face_adj.nbrGid;
        out.neighbor_owner = _face_adj.nbrOwner;
        out.non_resident = _face_adj.numNonResident;
        return out;
    }

    //-----------------------------------------------------------------------//
    // Communication
    //
    // Thin, deliberately: the adapter's job here is to name Tessera's two
    // halo directions and to record their contracts. Beatnik implements no
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
     * way to exchange one field, so `Beatnik_Communication.hpp` offers one
     * exchange and not a per-field family.
     *
     * Topology-preserving: does **not** bump `generation()`, so handles taken
     * before it stay valid.
     *
     * @pre The halo plans must be live. **M1-REWORK — they always are.** At M1
     *      Tessera's edits returned with the halo *cleared*, so an exchange in
     *      between was a silent no-op on an empty plan and the adapter had to
     *      insert an identity `migrate()`. `splitEdges()` now calls
     *      `rebuildHalo()` itself, at the recorded depth, before returning.
     *      There is no longer a window in which this is a no-op, and the
     *      workaround is gone from `splitEdges()` below.
     */
    void haloExchange() { Tessera::haloExchange( _mesh, _halo ); }

    /**
     * @brief Refresh the ghost rows of a **Beatnik-owned** per-vertex scalar
     *        view from their owners. T4b.
     *
     * TESSERA MAPPING: `Tessera::haloExchange( comm, aosoa, halo.vplan )` — the
     * *same* plan and the same pack/unpack machinery `haloExchange()` uses,
     * applied to a one-member scratch AoSoA laid out over the same local vertex
     * range. Beatnik posts no message of its own; the staging AoSoA exists only
     * because Tessera's exchange is expressed over an AoSoA.
     *
     * **Why this exists, since the file header says a field outside the mesh
     * cannot cross a rank boundary.** That statement is about the *reverse*
     * halo: `haloScatterAddVertices` accumulates into a field addressed by its
     * compile-time Cabana member index, so an external view genuinely has no
     * way in. A forward exchange has no such obstacle — it is a gather from
     * owned rows and a scatter into ghost rows, and the plan's index lists are
     * ordinary integers.
     *
     * **Why not a fourth vertex user field instead.** The one consumer is
     * `DynamicRemesh`'s sizing field, whose gradation sweep (T4b) propagates a
     * minimum `--remesh-target-gradation-iters` (8) rings — past
     * `halo_depth = 2`, so it must exchange between sweeps or the sizing field
     * differs at every partition boundary and the split set moves with the rank
     * count. Adding a slot to `vertex_fields` would put a *scratch* quantity in
     * `/vertices/u3` of every checkpoint and make every existing file
     * unreadable (risk R14) for a field no restart needs. T4a paid that price
     * for `RefineMark` because the reverse direction left it no choice; here
     * there is one.
     *
     * @param values `(Nv,)` over the **local** vertex range,
     *        `totalVertexCount()` rows. Owned rows are read, ghost rows are
     *        overwritten.
     *
     * @note MPI. Collective on `comm()`, and a no-op at one rank (the plan is
     *       empty, exactly as `haloExchange()` is).
     */
    template <class ScalarView>
    void haloExchangeVertexView( ScalarView& values )
    {
        using aosoa_type =
            Cabana::AoSoA<Cabana::MemberTypes<Real>, memory_space>;
        const int nv = totalVertexCount();
        aosoa_type buffer( "beatnik_vertex_view_halo", nv );
        auto slice = Cabana::slice<0>( buffer );
        auto in = values;
        Kokkos::parallel_for(
            "beatnik_vertex_view_halo_load",
            Kokkos::RangePolicy<execution_space>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) { slice( i ) = in( i ); } );
        Kokkos::fence();

        Tessera::haloExchange( comm(), buffer, _halo.vplan );

        auto out = values;
        Kokkos::parallel_for(
            "beatnik_vertex_view_halo_store",
            Kokkos::RangePolicy<execution_space>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) { out( i ) = slice( i ); } );
        Kokkos::fence();
    }

    /**
     * @brief Accumulate one **mesh-resident** vertex field from ghost slots
     *        into their owners.
     *
     * TESSERA MAPPING: `Tessera::haloScatterAddVertices<
     * Tessera::userVertexField<FieldId>()>( mesh, halo )` — the reverse halo
     * (ghost -> owner, `+=`), which closes M1 gap G2.
     *
     * **Read the assembly rule in the file header before reaching for this.**
     * Beatnik's *derived geometry* — vertex areas, vertex normals, the volume
     * gradient, the assembled Laplacians — is assembled from a loop over **all
     * locally held faces** and is therefore already complete on every owned
     * vertex with no communication; calling this after such a loop
     * **double-counts**. This entry point is for the other pattern: a loop over
     * **owned faces only**, accumulating into a field that lives in the mesh.
     *
     * Tessera's three contract properties, which a caller will get wrong if
     * they are not written down:
     *
     *   1. **Ghost slots are left untouched.** Afterwards an owned entry holds
     *      the complete global sum while every ghost copy still holds that
     *      rank's local partial, so the mesh is **not halo-consistent** for the
     *      field — follow with `haloExchange()` if kernels read ghosts.
     *   2. **Calling it twice double-counts.** It is not idempotent, precisely
     *      because of (1).
     *   3. **The summation order is fixed by peer order, not by rank count**,
     * so
     *      the result is bitwise reproducible within a run and **not** across
     *      rank counts. Same caveat as `globalSum`; risk R2.
     *
     * @tparam FieldId A `Beatnik::VertexFieldId` value. It is a **template**
     *         parameter and not a runtime argument because Tessera addresses
     *         the field by its compile-time Cabana member index — which is also
     *         why there is no way to scatter-add a Beatnik-owned `Kokkos::View`
     *         that lives outside the mesh, and why the assembly rule above
     * matters.
     *
     * @note MPI. Collective on `comm()`.
     */
    template <int FieldId>
    void haloScatterAddVertexField()
    {
        Tessera::haloScatterAddVertices<Tessera::userVertexField<FieldId>()>(
            _mesh, _halo );
    }

    //-----------------------------------------------------------------------//
    // Topological edits
    //
    // EDITING FAMILIES — READ THE FILE HEADER. Every edit named below is in the
    // REMESH family, which is the only family Beatnik uses:
    // `splitEdges()` today, `collapseEdges()` / `flipEdges()` / `compact()`
    // when Tessera lands them. `SurfaceMesh::refine` — the HIERARCHICAL
    // family's entry point — was **deleted** at T4a rather than left unused, so
    // Tessera's `EditFamily` guard is a backstop that should never fire. The
    // decision and its two findings are in `tasks/framework.md`, Phase 4,
    // *The editing-family question — RESOLVED*.
    //-----------------------------------------------------------------------//

    /**
     * @brief Split a specified set of edges at their midpoints. **The only
     *        topological edit Beatnik performs.**
     *
     * Port of dynamic_remesh.py::split_selected_edges (lines 261-298) **and**
     * of mesh.py::refine_marked_faces (lines 570-730)
     *
     * The second of those is the T4a finding and is not a coincidence. For the
     * mask "every edge of every marked face", `splitEdges()` and
     * `refine_marked_faces` are the *same algorithm*: the Python mints
     * midpoints on marked faces' edges only and retriangulates every face on
     * the bit pattern of its own split edges — `|S|` of 1, 2, 3 giving 2, 3, 4
     * children — with no cascade, because its `existing_midpoint` only ever
     * finds midpoints a marked face minted. That is `splitEdges()`'s contract
     * verbatim. So the indicator-driven refiner (T4a) and the metric remesher
     * (T4b) are two mask-construction rules over one primitive, rather than two
     * editing families.
     *
     * **M1-REWORK (G5a closed).** `Tessera::splitEdges( mesh, halo, edgeMask )`
     * bisects exactly the marked edges, the edge **owner** decides, every
     * incident face becomes 2, 3 or 4 children, and the result is **conforming
     * on exit with no closure layer and no 2:1 pass** — an edge-addressed split
     * needs neither, because both faces incident on a bisected edge subdivide
     * that edge the same way and no hanging node survives. `rebuildHalo()` is
     * called on the way out, so the halo is live at `halo_depth` on return, a
     * `haloExchange()` immediately afterwards is meaningful, and a second
     * `splitEdges()` may follow with nothing in between.
     *
     * **T4a CHANGE — the parameter is a concrete `const std::vector<char>&`,
     * not a template.** It was left templated at M1 because T4b had not settled
     * how the remesher would produce it; T4a settles it for both, and the
     * answer is Tessera's own convention rather than a second one invented on
     * top: a **host** mask sized `ownedEdgeCount()`, indexed by owned edge local
     * index, `1` = bisect. A device-computed indicator is copied to the host and
     * truncated to the owned range, which is a real cost on the AMR path and is
     * where it belongs — `Tessera::splitEdges` takes a host vector, so a
     * templated parameter could only have hidden the copy, not avoided it.
     *
     * @param edgeMask Per-**owned-edge** marks, `1` = bisect. Must be exactly
     *        `ownedEdgeCount()` long; Tessera checks this and an empty mask is
     *        a documented no-op fast path (V/E/F unchanged, no communication
     *        beyond the two global counts).
     * @return What the edit did; see `MeshEditReport`. **No parent/weight map
     *         — Tessera transfers the vertex fields itself** through the
     *         `RefinePolicy`, and inherits face user fields verbatim.
     *
     * @note MPI. **Collective**, and the mask needs no reconciling first: the
     *       edge owner's verdict is routed to every rank holding an incident
     *       face by Tessera's edge coordinator, so an arbitrary, unreconciled,
     *       rank-local mask is a legal input. This is what left
     *       `Comm::reconcileRefinementMarks` with nothing to do, and T4a
     *       deleted it. What Beatnik must still agree across ranks is its own
     *       *mark closure*, which happens before this call and terminates on
     *       one `MPI_Allreduce(MPI_LOR)`.
     *
     * @note INVALIDATION. Reallocates the AoSoAs, the key views and the CSRs
     *       and replaces the halo plans, so every slice, CSR and cached
     *       accessor taken before this call is dangling. `generation()` is
     *       bumped, the adapter's four caches invalidate themselves, and every
     *       accessor must be re-taken.
     *
     * **EDITING FAMILY — Remesh**, and this is the call that tags the mesh.
     * Nothing can subsequently violate it, because `refine()` no longer exists
     * in this facade.
     */
    MeshEditReport splitEdges( const std::vector<char>& edgeMask )
    {
        const Tessera::SplitResult result =
            Tessera::splitEdges( _mesh, _halo, edgeMask );

        MeshEditReport report;
        // A remesh-family edit runs no balance pass, so this is always 0. Kept
        // in the report because the field is part of the shared shape.
        report.balance_rounds = 0;
        report.split_edges_local =
            static_cast<long long>( result.midpoints.size() );
        report.owned_vertices_after = ownedVertexCount();
        report.owned_faces_after = ownedFaceCount();
        return report;
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
     * **M1 GAP (G5b) — STILL OPEN.** Tessera implements split-based edits only.
     * Edge collapse does not exist at any level: no public entry point, no
     * primitive, and the data model has no coarsening path (`Level` only ever
     * rises). Neither the link condition nor the cross-rank owner-decides
     * protocol has anywhere to attach. Tessera task
     * `../tessera/tasks/edge-collapse.md` (NOT STARTED, the largest of the
     * eleven; its hard dependency on halo depth >= 2 has landed). This is what
     * still blocks T4b.
     *
     * **EDITING FAMILY — Remesh**, when it lands: it will tag the mesh
     * `EditFamily::Remesh`, the same family `splitEdges()` already tags this
     * mesh with — so it composes with it freely, and Tessera's guard stays a
     * backstop that never fires.
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
     * **M1 GAP (G5c) — STILL OPEN.** Tessera implements no edge flip.
     * `../tessera/tasks/edge-flip.md` (NOT STARTED). Its one prerequisite is
     * met: a correct flip needs both incident faces, which the vertex halo does
     * not guarantee are co-resident, so it needs the edge-coordinator machinery
     * — and `buildFaceAdjacency` (G4) now exposes exactly that. This blocks
     * T4d's flips (T4c is the tangential pass and needs no flip).
     *
     * **EDITING FAMILY — Remesh**, when it lands: the same family
     * `splitEdges()` already tags this mesh with, so it composes with it.
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
     * **M1 GAP (G5d) — STILL OPEN**, and in Tessera's own ordering a
     * *prerequisite* of collapse rather than a consequence of it.
     * `../tessera/tasks/mesh-compaction.md` (NOT STARTED). With no collapse
     * (G5b) nothing orphans a vertex, so there is nothing to compact today.
     *
     * **EDITING FAMILY — Remesh**, when it lands.
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
     * Writes the position slice directly, over the **owned** range only. Purely
     * geometric, so it does **not** bump `generation()` and every outstanding
     * handle stays valid — but it *does* invalidate any derived quantity a
     * caller has cached (the accessor holds the live slice, so positions
     * themselves are seen). Ghost positions are **not** updated: follow with
     * `haloExchange()`.
     *
     * @param vertices `(>= ownedVertexCount(), 3)` device-accessible positions.
     */
    template <class NewVertexView>
    void setVertices( const NewVertexView& vertices )
    {
        auto pos = positions();
        const int n = ownedVertexCount();
        auto src = vertices;
        Kokkos::parallel_for(
            "beatnik_set_vertices",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                    pos( i, d ) = src( i, d );
            } );
        Kokkos::fence();
    }

    //-----------------------------------------------------------------------//
    // Redistribution (load balancing)
    //-----------------------------------------------------------------------//

    /**
     * @brief Recompute the partition and move entities to it.
     *
     * This is what `Beatnik_Communication.hpp::redistribute` (T5d) calls.
     *
     * **M1-REWORK — it is no longer also the post-edit halo rebuild.**
     * `splitEdges()` rebuilds its own halo now, so the identity-`migrate()` call
     * that used to be forced through here is gone and `rebalance == false` is a
     * genuine no-op path a caller has no reason to ask for.
     *
     * TESSERA MAPPING, two forms, both of which rebuild the ghost layer at the
     * recorded `halo_depth` and all three halo plans as a side effect:
     *
     *   - `rebalance == true`: `Tessera::loadBalance( mesh, halo )`. A Zoltan2
     *     geometric **MultiJagged** partition of the owned-face centroids
     *     (never RCB — Zoltan2's RCB is documented broken on Tuolumne), then
     *     migrate to it. Default imbalance tolerance 0.05.
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
     *     byte images over `allToAllV`, in four rounds, then the local AoSoAs,
     *     both CSRs, the key tables and the three halo plans are rebuilt.
     *   - **M1-REWORK (G8 closed).** The partition solve is no longer
     *     rank-0-bound. `Tessera::LoadBalanceMode` offers `GatherRoot` (the old
     *     path, kept as the reference), `Distributed` (rank 0 receives **zero**
     *     faces) and `Sampled`, and **`Sampled` is the default** because it is
     *     the only mode Tessera measured run-to-run reproducible — its cuts
     *     come from a gid-order sample solved once and broadcast, and each rank
     *     then classifies its own centroids in exact local arithmetic. T5d
     *     should take the default and report
     *     `LoadBalanceStats::rootSolveFaces`, which is `O(comm size)` in
     *     `Sampled`. Note `Distributed` is *not* reproducible run to run — a
     *     measured property of Zoltan2, in Tessera's Known Issues, not a
     * Tessera defect.
     *   - Bumps `generation()`: re-take every accessor afterwards. The
     * adapter's
     *     own caches invalidate themselves.
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
    //-----------------------------------------------------------------------//
    // Construction helpers
    //-----------------------------------------------------------------------//

    /// Map the unit sphere Tessera generated onto `radius` / `center`, in
    /// place. Written as `c + r*p`, associated exactly as the Python's
    /// `center[None,:] + radius * vertices`, because the association is visible
    /// in the last bit and the gold-file comparison is at 1e-14.
    void scaleAndTranslate( Real radius, const Real center[3] )
    {
        auto pos = _mesh.template vertexSlice<Tessera::VertexField::Position>();
        const int n = totalVertexCount();
        const Real r = radius;
        const Real cx = center[0];
        const Real cy = center[1];
        const Real cz = center[2];
        Kokkos::parallel_for(
            "beatnik_scale_translate_sphere",
            Kokkos::RangePolicy<execution_space>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                pos( i, 0 ) = cx + r * pos( i, 0 );
                pos( i, 1 ) = cy + r * pos( i, 1 );
                pos( i, 2 ) = cz + r * pos( i, 2 );
            } );
        Kokkos::fence();
    }

    /// Partition a replicated mesh by the z axis, cut it to owned plus a
    /// `halo_depth`-deep ghost layer, and fill the ghosts. The one place
    /// `halo_depth` is passed to Tessera.
    void distributeReplicated()
    {
        auto face_owner = Tessera::facePartitionByAxis( _mesh, /*axis=*/2 );
        Tessera::distribute( _mesh, _halo, face_owner, halo_depth );
        Tessera::haloExchange( _mesh, _halo );
    }

    /**
     * @brief Throw unless the generated surface is outward-wound.
     *
     * \f$V = \tfrac16\sum_f a_f\cdot(b_f\times c_f)\f$ over **owned** faces
     * (risk R9 — a ghost face is an owned face somewhere else and would be
     * counted twice), reduced with `Tessera::globalSum`.
     *
     * The formula is `SurfaceOperators::enclosedVolume`'s, deliberately
     * duplicated in these six lines rather than reached for through
     * `Beatnik_MeshGeometry.hpp`: this adapter's only Beatnik include is
     * `Beatnik_Types.hpp`, and an orientation *precondition* should not make
     * the mesh layer depend on the geometry layer.
     *
     * @note MPI. Collective. Called from every construction entry point that
     *       generates its own geometry, so every rank reaches it.
     */
    void requireOutwardWinding( const char* where )
    {
        auto pos = positions();
        auto fv = faceVertices();
        const int nf = ownedFaceCount();
        Real local = 0.0;
        Kokkos::parallel_reduce(
            "beatnik_winding_check",
            Kokkos::RangePolicy<execution_space>( 0, nf ),
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
            local );
        const Real volume = Tessera::globalSum( _mesh, local ) / Real( 6.0 );
        if ( !( volume > Real( 0.0 ) ) )
            throw std::runtime_error(
                std::string( "Beatnik::SurfaceMesh::" ) + where +
                ": the generated surface is not outward-wound (enclosed "
                "volume " +
                std::to_string( volume ) +
                " is not positive). Every normal, curvature and volume "
                "downstream would carry the wrong sign. Tessera's generators "
                "document CCW-outward winding, so this is a Tessera defect to "
                "report rather than a condition to absorb here." );
    }

    //-----------------------------------------------------------------------//
    // Lazy derived-structure caches, all keyed on `generation()`
    //-----------------------------------------------------------------------//

    void ensureGeometry()
    {
        if ( _geometry_generation != _mesh.generation() )
        {
            _geometry = Tessera::buildMeshGeometry( _mesh );
            _geometry_generation = _mesh.generation();
        }
    }

    void ensureOneRing()
    {
        if ( _one_ring_generation != _mesh.generation() )
        {
            _one_ring = Tessera::buildVertexStencil( _mesh, /*k=*/1 );
            _one_ring_generation = _mesh.generation();
        }
    }

    void ensureFaceAdjacency()
    {
        if ( _face_adj_generation != _mesh.generation() )
        {
            _face_adj = Tessera::buildFaceAdjacency( _mesh );
            _face_adj_generation = _mesh.generation();
        }
    }

    /// Resolve `FaceField::Edges` (gids) to local edge indices, on the host,
    /// once per topology generation — the inverse of
    /// `ensureEdgeFaceIncidence()` below and built the same way, for the same
    /// reason. T4a.
    void ensureFaceEdges()
    {
        if ( _face_edges_generation == _mesh.generation() )
            return;

        const std::size_t ne = _mesh.numEdges();
        const std::size_t nf = _mesh.numFaces();

        Cabana::AoSoA<typename tessera_mesh_type::edge_member_types,
                      Kokkos::HostSpace>
            he( "beatnik_face_edges_he", ne );
        Cabana::AoSoA<typename tessera_mesh_type::face_member_types,
                      Kokkos::HostSpace>
            hf( "beatnik_face_edges_hf", nf );
        Cabana::deep_copy( he, _mesh.edges() );
        Cabana::deep_copy( hf, _mesh.faces() );
        auto e_gid = Cabana::slice<Tessera::EdgeField::Gid>( he );
        auto f_e = Cabana::slice<Tessera::FaceField::Edges>( hf );

        std::unordered_map<Tessera::GlobalId, int> gid2local;
        gid2local.reserve( ne * 2 );
        for ( std::size_t e = 0; e < ne; ++e )
            gid2local[e_gid( e )] = static_cast<int>( e );

        face_vertex_view edges(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_face_edges" ),
            nf );
        auto h_edges = Kokkos::create_mirror_view( edges );
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const Tessera::GlobalId g = f_e( f, k );
                if ( g == Tessera::invalid_gid )
                {
                    h_edges( f, k ) = -1;
                    continue;
                }
                auto it = gid2local.find( g );
                h_edges( f, k ) = ( it == gid2local.end() ) ? -1 : it->second;
            }
        Kokkos::deep_copy( edges, h_edges );

        _face_edges = edges;
        _face_edges_generation = _mesh.generation();
    }

    /// Resolve `EdgeField::Faces` (gids) to local face indices, on the host,
    /// once per topology generation. Same locus and idiom as Tessera's own
    /// `buildMeshGeometry` / `buildVertexStencil`.
    void ensureEdgeFaceIncidence()
    {
        if ( _edge_faces_generation == _mesh.generation() )
            return;

        const std::size_t ne = _mesh.numEdges();
        const std::size_t nf = _mesh.numFaces();

        Cabana::AoSoA<typename tessera_mesh_type::edge_member_types,
                      Kokkos::HostSpace>
            he( "beatnik_edge_faces_he", ne );
        Cabana::AoSoA<typename tessera_mesh_type::face_member_types,
                      Kokkos::HostSpace>
            hf( "beatnik_edge_faces_hf", nf );
        Cabana::deep_copy( he, _mesh.edges() );
        Cabana::deep_copy( hf, _mesh.faces() );
        auto e_f = Cabana::slice<Tessera::EdgeField::Faces>( he );
        auto f_gid = Cabana::slice<Tessera::FaceField::Gid>( hf );

        std::unordered_map<Tessera::GlobalId, int> gid2local;
        gid2local.reserve( nf * 2 );
        for ( std::size_t f = 0; f < nf; ++f )
            gid2local[f_gid( f )] = static_cast<int>( f );

        Kokkos::View<int*, memory_space> count(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_edge_face_count" ),
            ne );
        Kokkos::View<int* [2], memory_space> faces(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_edge_faces" ),
            ne );
        auto h_count = Kokkos::create_mirror_view( count );
        auto h_faces = Kokkos::create_mirror_view( faces );
        for ( std::size_t e = 0; e < ne; ++e )
        {
            int n = 0;
            for ( int s = 0; s < 2; ++s )
            {
                const Tessera::GlobalId g = e_f( e, s );
                if ( g == Tessera::invalid_gid )
                {
                    h_faces( e, s ) = -1;
                    continue;
                }
                ++n;
                auto it = gid2local.find( g );
                h_faces( e, s ) = ( it == gid2local.end() ) ? -1 : it->second;
            }
            h_count( e ) = n;
        }
        Kokkos::deep_copy( count, h_count );
        Kokkos::deep_copy( faces, h_faces );

        // The second, edit-durable view: scatter each LOCALLY HELD face into
        // its three edges. `FaceField::Edges` is the face's own record and is
        // complete for every resident face at every generation, which
        // `EdgeField::Faces` above is not once an edit has rebuilt the edge set
        // from each rank's local faces. See `EdgeFaceIncidence`.
        ensureFaceEdges();
        auto h_face_edges = Kokkos::create_mirror_view( _face_edges );
        Kokkos::deep_copy( h_face_edges, _face_edges );

        Kokkos::View<int*, memory_space> resident_count(
            "beatnik_edge_resident_count", ne );
        Kokkos::View<int* [2], memory_space> resident_faces(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "beatnik_edge_resident_faces" ),
            ne );
        auto h_resident_count = Kokkos::create_mirror_view( resident_count );
        auto h_resident_faces = Kokkos::create_mirror_view( resident_faces );
        for ( std::size_t e = 0; e < ne; ++e )
        {
            h_resident_count( e ) = 0;
            h_resident_faces( e, 0 ) = -1;
            h_resident_faces( e, 1 ) = -1;
        }
        // Ascending local face index, because the loop runs that way — so the
        // slot order is a property of the local numbering and not of insertion
        // race, which matters for nothing today (every consumer is symmetric in
        // the two) and would matter the moment one is not.
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const int e = h_face_edges( f, k );
                if ( e < 0 )
                    continue;
                const int n = h_resident_count( e );
                if ( n < 2 )
                    h_resident_faces( e, n ) = static_cast<int>( f );
                h_resident_count( e ) = n + 1;
            }
        Kokkos::deep_copy( resident_count, h_resident_count );
        Kokkos::deep_copy( resident_faces, h_resident_faces );

        _edge_faces.count = count;
        _edge_faces.faces = faces;
        _edge_faces.resident_count = resident_count;
        _edge_faces.resident_faces = resident_faces;
        _edge_faces_generation = _mesh.generation();
    }

    /// The Tessera mesh. All entity storage, connectivity, ownership and the
    /// user field pack live here.
    tessera_mesh_type _mesh;

    /// The three halo exchange plans, carrying `halo_depth`. Rebuilt in place
    /// by `Tessera::splitEdges` / `migrate` / `loadBalance`, each of
    /// which preserves the recorded depth.
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

    /// Cached face-to-face adjacency. The build is **collective**, so the
    /// first call after an edit must be reached on every rank.
    Tessera::FaceAdjacency<MemorySpace> _face_adj;
    std::size_t _face_adj_generation = static_cast<std::size_t>( -1 );

    /// Cached edge-to-face incidence, resolved from gids on the host.
    EdgeFaceIncidence _edge_faces;
    std::size_t _edge_faces_generation = static_cast<std::size_t>( -1 );

    /// Cached face-to-edge indices, resolved from gids on the host. T4a.
    face_vertex_view _face_edges;
    std::size_t _face_edges_generation = static_cast<std::size_t>( -1 );
};

} // namespace Beatnik

#endif // BEATNIK_MESHINTERFACE_HPP
