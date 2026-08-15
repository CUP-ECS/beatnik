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
 * @file Beatnik_Communication.hpp
 * @brief Every point in the control flow where ranks must talk, as a named,
 *        documented entry point.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The Python reference is serial, so none of these calls have a Python
 * counterpart to port — they are *new* requirements that arise only because the
 * surface is distributed. Collecting them here, named and documented up front,
 * is what keeps them from being discovered one deadlock at a time later.
 *
 * Each entry point records three things: **what** moves, **which** MPI
 * operation is expected, and **why it must happen at that point** — that is,
 * the invariant that breaks without it. The last is the part that cannot be
 * reconstructed from the code afterwards.
 *
 * WHO OWNS THE COLLECTIVE (M1-REWORK)
 * -----------------------------------
 * **Nothing here calls `MPI_Allreduce` or posts a message.** Every function
 * below forwards either to Tessera — through `SurfaceMesh`, so no Tessera type
 * is named outside the adapter — or, for the reductions, to Tessera's
 * single-sourced collective wrappers. At M1 that was not possible: Tessera
 * exposed only `globalMin` and no reverse halo, so this file was specified as
 * the one place Beatnik would hand-roll the rest. Both gaps (M1 G2 and G3) are
 * closed, so the hand-rolling is gone and the value of this file is now
 * entirely in **where** these calls must sit in the control flow, which follows
 * from the mathematics and is unchanged.
 *
 * The reduction wrappers take an `MPI_Comm` rather than a mesh because their
 * callers (the time integrator, the diagnostics, `SurfaceState`) hold a
 * communicator and not always a mesh. Tessera's own wrappers take a mesh, so
 * these route through a minimal `MPI_Allreduce` on the given communicator with
 * the identical datatype/op — see the note on each.
 *
 * A NOTE ON REDUCTION DETERMINISM
 * -------------------------------
 * `MPI_Allreduce` with `MPI_SUM` on floating point is **not** reproducible
 * across rank counts, and on GPU the local partial sums are not reproducible
 * across runs either (atomics, reduction tree order). Two of the reductions
 * below feed quantities the whole run keys off — `initial_volume` and
 * `initial_min_edge` — so a 4-rank run and a 6-rank run will not produce
 * bit-identical trajectories. This is expected and is why
 * `compare_output.py` compares with `rtol`/`atol` rather than exactly. It is
 * recorded as risk R2 in `tasks/framework.md`, and Tessera now documents the
 * same caveat on its own `globalSum`. **Integer sums are exact**, which is why
 * the entity counts on `SurfaceMesh` are reduced as `long long`.
 */

#ifndef BEATNIK_COMMUNICATION_HPP
#define BEATNIK_COMMUNICATION_HPP

#include <Beatnik_Types.hpp>

#include <mpi.h>

namespace Beatnik
{
namespace Comm
{

//---------------------------------------------------------------------------//
// 1. Halo / ghost exchange
//---------------------------------------------------------------------------//

/**
 * @brief Refresh every ghost entity from its owner.
 *
 * **What moves:** the whole Cabana tuple of every ghost vertex, edge and face —
 * position, gid, ownership, connectivity, and all three Beatnik vertex fields —
 * from the rank that owns it to every rank that ghosts it.
 *
 * **MPI operation:** neighbour point-to-point over the persistent halo plans
 * (`MPI_Isend`/`MPI_Irecv`, one message per peer, peers in ascending rank).
 * Collective on the mesh communicator in the sense that every rank must call
 * it, but not a collective over `MPI_COMM_WORLD`.
 *
 * **M1-REWORK — this is the ONLY gather, and it is whole-tuple.** The
 * pre-rework header had a second entry point, `haloExchangeField( mesh, field
 * )`, for refreshing one named per-vertex field. Tessera has no such operation
 * and cannot have one cheaply: `haloExchange` ships the tuple as one opaque
 * `MPI_Type_contiguous` of `sizeof(tuple)` bytes, which is correct for an
 * overwrite of every field at once. So `haloExchangeField` is **deleted**
 * rather than kept as a shim that would have implied a cost model Beatnik does
 * not have — the M2 precedent set by `gatherForCheckpoint`. The consequence for
 * a caller is only that refreshing one field costs the same as refreshing
 * everything.
 *
 * **Why here:** every differential operator in the solver — face normals and
 * areas, the cotangent Laplacian, the surface gradient, the mean-curvature
 * normal — reads the full ring of an owned vertex, and part of that ring is
 * ghost. If positions are stale the operators are evaluated on a torn surface:
 * face normals on boundary-straddling triangles point the wrong way, the
 * cotangent weights go negative, and the resulting velocity field has a
 * discontinuity along every rank boundary. This must run **after every write to
 * vertex positions or to an evolved vertex field** — that is, after each RK
 * stage update, after the volume projection, after tangential relaxation, and
 * after any remeshing pass that moves vertices.
 *
 * **Why it is not needed for ring width.** Exchanging twice does **not** widen
 * the ghost set; it refreshes the same set. The two-ring RHS is served by
 * building the mesh at `SurfaceMesh::halo_depth = 2`, once, at setup. That is
 * what retired most of risk R8; see the halo section of
 * `Beatnik_MeshInterface.hpp`.
 *
 * @pre The halo plans must be live. **They always are**: Tessera's
 *      `splitEdges()` rebuilds the halo at the recorded depth before
 *      returning, so unlike at M1 there is no window in which this is a silent
 *      no-op on an empty plan.
 */
template <class MeshType>
void haloExchangeVertices( MeshType& mesh )
{
    mesh.haloExchange();
}

/**
 * @brief Sum ghost contributions of one **mesh-resident** vertex field back
 *        onto their owners.
 *
 * **What moves:** the named field's value on every ghost vertex, sent back to
 * the owning rank and added there. One field per call, scalar or `Real[3]`
 * (accumulated componentwise).
 *
 * **MPI operation:** the exact reverse of the halo exchange — pack from the
 * ghost index list, send along the receive peers, accumulate into the owned
 * index list with `+=`. Tessera serializes the unpack per peer (one kernel per
 * peer) rather than paying for atomics, which is what fixes the summation
 * order.
 *
 * **Why here — and why almost nothing in Beatnik needs it.** The natural
 * parallelization of a per-vertex quantity built from a face loop is "scatter
 * from faces to their three corners", and a face straddling a rank boundary is
 * owned by one rank but deposits onto vertices owned by another. The textbook
 * fix is this call. **Beatnik does not need it for its geometry**, because
 * Tessera's local face set is *the owned faces plus every face incident on an
 * owned vertex*: a loop over **all locally held faces** therefore gives every
 * owned vertex its complete incident-face set with no communication at all.
 * Vertex areas, vertex normals, the volume gradient and the assembled
 * Laplacians are all assembled that way, and calling this after such a loop
 * would **double-count**. The rule is stated once, with its consequences, under
 * DISTRIBUTED ASSEMBLY in `Beatnik_MeshInterface.hpp`.
 *
 * What it is for is the other pattern: accumulating into a field that lives *in
 * the mesh* from a loop over **owned faces only**. Nothing in T1b does that;
 * this exists because it is the only correct primitive for it, and because a
 * later task reaching for a scatter-add must find Tessera's contract written
 * down rather than assume the usual one.
 *
 * **The three contract properties**, all of which a caller will get wrong if
 * they are not stated:
 *   1. **Ghost slots are left untouched** — afterwards owners hold the global
 *      sum and every ghost still holds that rank's partial, so the field is
 *      *not* halo-consistent. Follow with `haloExchangeVertices` if kernels
 *      read ghosts.
 *   2. **Calling it twice double-counts.** Not idempotent, because of (1).
 *   3. **Summation order is fixed by peer order, not by rank count** — bitwise
 *      reproducible within a run, *not* across rank counts. Risk R2 again.
 *
 * **M1-REWORK — the signature changed, and it had to.** The pre-rework
 * signature was `haloScatterAdd( mesh, field )` with `field` an arbitrary
 * Beatnik-owned `Kokkos::View`. That **cannot be implemented**: Tessera
 * accumulates a field *inside* the mesh AoSoA, addressed by its compile-time
 * Cabana member index, and an external view has no such index. The field is
 * therefore a template parameter naming a slot in Beatnik's vertex pack, and a
 * scatter-add of an external view is not available from anyone — which is
 * exactly why the local-face assembly rule above is load-bearing rather than
 * merely convenient.
 *
 * @tparam FieldId A `Beatnik::VertexFieldId` value.
 */
template <int FieldId, class MeshType>
void haloScatterAdd( MeshType& mesh )
{
    mesh.template haloScatterAddVertexField<FieldId>();
}

//---------------------------------------------------------------------------//
// 2. Global reductions
//
// M1-REWORK. Tessera now single-sources these as `globalSum` / `globalMin` /
// `globalMax` / `globalAllFinite` over `mesh.comm()`, closing M1 gap G3. The
// wrappers here keep an `MPI_Comm` signature because their callers hold a
// communicator rather than a mesh, and they perform the identical single
// `MPI_Allreduce` with the identical datatype and op. Where a caller does hold
// a mesh, either spelling is correct and gives the same answer.
//---------------------------------------------------------------------------//

/**
 * @brief Global sum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_SUM, comm)`.
 * Equivalent to `Tessera::globalSum( mesh, local )` on `mesh.comm()`.
 *
 * **Why here:** four quantities in the solver are global sums over the surface
 * and every rank needs the answer, not just rank 0:
 *
 *   1. **Enclosed volume**
 *      \f$V = \frac{1}{6}\sum_f a_f\cdot(b_f\times c_f)\f$
 *      (`run_adaptive_mesh_bubble.py::mesh_enclosed_volume`, lines 1036-1040).
 *      Only owned faces may contribute or the shared ones are counted twice
 *      (risk R9).
 *   2. **The two inner products of the volume projection**,
 *      \f$\sum_v |\nabla_v V|^2\f$ and \f$\sum_v \nabla_v V \cdot u_v\f$
 *      (`mesh_solver.py::_remove_discrete_volume_flux`, lines 285-295). These
 *      form a *single* global Rayleigh quotient, so both must be reduced before
 *      either is used; reducing them separately and dividing locally gives a
 *      different (wrong) answer on every rank.
 *   3. **The area-weighted mean of the potential**,
 *      \f$\bar\phi = \sum_v A_v\phi_v / \sum_v A_v\f$
 *      (`mesh_solver.py::_area_weighted_scalar_mean`, lines 239-244),
 *      subtracted to fix the arbitrary additive constant of the potential. If
 *      each rank subtracts its *local* mean the potential acquires a
 *      piecewise-constant jump across every partition boundary, and its surface
 *      gradient — the sheet vector — picks up a delta function there. This is
 *      why `SurfaceOperators::areaWeightedMeanPartials` returns the numerator
 *      and denominator separately instead of a mean.
 *   4. **Total surface area**, the denominator of (3).
 */
inline Real allReduceSum( MPI_Comm comm, Real local )
{
    Real global = local;
    MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, comm );
    return global;
}

/**
 * @brief Global minimum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_MIN, comm)`.
 * Equivalent to `Tessera::globalMin( mesh, local )` on `mesh.comm()`.
 *
 * **Why here:** the adaptive timestep is
 * \f$\Delta t = \max(\Delta t_{\min}, \Delta t \cdot
 * (h_{\min}/h_{\min}^0)^p)\f$ (`run_adaptive_mesh_bubble.py::choose_step_dt`,
 * lines 889-901), where \f$h_{\min}\f$ is the shortest edge **anywhere on the
 * surface**. Every rank must step with the *same* dt or the surface is
 * integrated inconsistently and the RK3 stages no longer correspond to a single
 * time level — the result is not merely inaccurate, it is not a discretization
 * of anything. The same applies to the minimum triangle quality that triggers
 * the remesh repair pass.
 *
 * Unlike the sum, `MPI_MIN` **is** reproducible across rank counts: it selects
 * an element rather than combining them, so it does not depend on the partition
 * into partial results.
 */
inline Real allReduceMin( MPI_Comm comm, Real local )
{
    Real global = local;
    MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_MIN, comm );
    return global;
}

/**
 * @brief Global maximum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_MAX, comm)`.
 * Equivalent to `Tessera::globalMax( mesh, local )` on `mesh.comm()`.
 *
 * **Why here:** `max|sheet_vector|` gates two control decisions — the
 * `--max-sheet-dt-product` dt clamp (line 897-900) and the
 * `--field-filter-threshold` filter trigger (lines 1536-1540). Both must be
 * taken identically on every rank, for the same reason as the dt minimum: a
 * rank that filters while its neighbors do not produces a discontinuous field.
 * Also used for the maxima reported by the diagnostics.
 */
inline Real allReduceMax( MPI_Comm comm, Real local )
{
    Real global = local;
    MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_MAX, comm );
    return global;
}

/**
 * @brief Global "are all values finite" test.
 *
 * **What moves:** one `int` (0/1).
 * **MPI operation:** `MPI_Allreduce(..., MPI_INT, MPI_LAND, comm)` — on an
 * `int` rather than a `bool` because `MPI_CXX_BOOL` is not universally
 * available. Equivalent to `Tessera::globalAllFinite( mesh, local )`.
 *
 * **It takes a verdict, not data.** The local "everything I hold is finite"
 * sweep is the caller's Kokkos reduction over whichever fields it cares about
 * (`SurfaceState::allFinite`), which is knowledge neither this function nor
 * Tessera has. There is deliberately no device-side all-finite helper here.
 *
 * **Why here:** the driver aborts the run the moment any vertex position or
 * sheet value goes non-finite, and writes a final "last finite state"
 * checkpoint (`run_adaptive_mesh_bubble.py:1413-1423, 1517-1527, 1546-1556`).
 * That decision must be unanimous: if one rank sees a NaN and breaks out of the
 * loop while the others continue to the next RHS evaluation, the survivors
 * block forever in the next collective. This is the single most likely
 * hang in the whole solver, and it is why the finiteness check is a
 * *reduction* and not a local `if`.
 */
inline bool allReduceAllFinite( MPI_Comm comm, bool local_finite )
{
    int local = local_finite ? 1 : 0;
    int global = local;
    MPI_Allreduce( &local, &global, 1, MPI_INT, MPI_LAND, comm );
    return global != 0;
}

//---------------------------------------------------------------------------//
// 3. Refinement and load balancing
//---------------------------------------------------------------------------//

// NOTE (T4a): `reconcileRefinementMarks` was DELETED here, not left as a stub
// and not made a no-op.
//
// Red-green closure is a fixed-point computation
// (`mesh_solver.py::_balance_red_green_refinement`, lines 1543-1580, loops
// `while changed`): marking a face implies splitting its three edges, which can
// force a neighbour from green to red, which splits *its* edges, and so on.
// When that cascade crosses a rank boundary it must propagate, or the two sides
// make different decisions about the same edge and the surface tears.
//
// **Tessera closes the half of that which concerns the EDIT itself.**
// `splitEdges()` performs no closure and no 2:1 pass: the edge **owner**
// decides, and its verdict is routed to every rank holding an incident face by
// the edge coordinator, so an arbitrary, unreconciled, rank-local owned-edge
// mask is a legal input to `SurfaceMesh::splitEdges`.
//
// What is left is the half that concerns Beatnik's own MARK CLOSURE, and it is
// not a communication primitive: `AdaptiveMesh::refine` carries the promotion
// mark in the `FaceFieldId::RefineMark` face user field, re-`haloExchange()`es
// it once per round, and terminates on a single `MPI_Allreduce(MPI_LOR)` with a
// hard round cap. There is nothing for a separate entry point here to do, and a
// no-op one would let a caller keep a reconciliation step in its control flow
// and believe it was doing something.

/**
 * @brief Redistribute the surface after adaptation and migrate its fields.
 *
 * **What moves:** vertices, edges, faces, and every per-vertex field
 * (potential, sheet vector, material position) for the entities changing owner.
 *
 * **MPI operation:** an all-to-all migration (`allToAllV` over tuple byte
 * images, in four rounds), followed by a rebuild of the ghost layer at the
 * recorded halo depth and of all three halo plans.
 *
 * **Why here:** refinement and collapse are strongly localized in space — the
 * whole point of the sizing field is to concentrate resolution in the
 * roll-up — so a partition that was balanced at t=0 is badly imbalanced after a
 * few hundred steps, with one rank holding most of the refined spiral. Since
 * the Birkhoff-Rott evaluation costs \f$O(N_\text{local})\f$ *targets* against
 * a global source set, imbalance in the vertex count translates directly into
 * imbalance in the dominant cost, and every rank waits on the slowest at the
 * next collective. Correctness does not require this call; throughput does.
 * The natural point is immediately after the remesh/refine pass and before the
 * next RHS evaluation.
 *
 * **Every field migrates with the mesh, automatically.** Tessera moves whole
 * Cabana tuples, so there is no per-field plumbing to forget — which is worth
 * knowing precisely because the failure it removes (migrating vertices without
 * the potential, silently reinitializing the solution) looks like a physics bug
 * rather than a communication bug.
 *
 * **M1-REWORK — this is no longer also the post-`refine()` rebuild.** At M1
 * `Tessera::refine()` returned with the halo cleared and the adapter had to
 * follow it with an identity `migrate()`. It rebuilds its own halo now, so the
 * only reason to call this is a genuine rebalance. See
 * `SurfaceMesh::redistribute` for the modes and for G8's closure.
 *
 * T5d's territory; the `state` parameter survives from the pre-M1 design in
 * which fields lived outside the mesh and will be dropped when T5d implements
 * this.
 */
template <class MeshType, class StateType>
void redistribute( MeshType& mesh, StateType& state )
{
    (void)mesh;
    (void)state;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "redistribute" );
}

//---------------------------------------------------------------------------//
// 4. Whole-surface transfer for I/O
//
// **M2 CHANGE — `gatherForCheckpoint` is deleted.** It collected the owned
// vertices, faces and fields of every rank onto rank 0, renumbering face
// connectivity from local to global indices on the way, so a serial writer
// could emit one whole-surface array per field.
//
// `Tessera::writeMesh` does all of that and does it collectively: it writes
// each rank's owned entities directly, at `MPI_Exscan` offsets, with
// connectivity already translated to dense global indices. So the gather is not
// just redundant, it is strictly worse — O(global) memory on rank 0, a
// serialized write, and a hand-rolled reimplementation of the one genuinely
// error-prone step. See `Beatnik_IOInterface.hpp` for the division of labour.
//
// The reverse direction survives, but for a different caller: `readMesh`
// reconstructs a Beatnik checkpoint without any rank holding the global mesh,
// so `broadcastFromRoot` is no longer part of the restart path either.
//---------------------------------------------------------------------------//

/**
 * @brief Broadcast a rank-0-read surface to all ranks for partitioning.
 *
 * **What moves:** the whole vertex and face arrays plus the fields, from rank 0
 * outward.
 *
 * **MPI operation:** `MPI_Bcast` of the array extents, then of the arrays.
 *
 * **Why here:** `SurfaceMesh::adopt` wraps `Tessera::buildFromTriangleSoup`,
 * which has no communication and therefore requires its input replicated on
 * **every** rank (M1). This is how it gets there. **M2 CHANGE — this is no
 * longer on the `--restart-from` path**, which goes through `Tessera::readMesh`
 * and never materializes the global mesh anywhere; the one remaining caller is
 * the "read the initial mesh from the gold file" mitigation for regression
 * test 1 (risk R1), whose input is a Python `.npz`.
 *
 * Ranks other than 0 have nothing until this completes, so the extents must be
 * broadcast first — allocating from a stale or zero extent is the usual bug.
 */
template <class HostArrays>
void broadcastFromRoot( MPI_Comm comm, HostArrays& arrays )
{
    (void)comm;
    (void)arrays;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "broadcastFromRoot" );
}

} // namespace Comm
} // namespace Beatnik

#endif // BEATNIK_COMMUNICATION_HPP
