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
 *        documented stub.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The Python reference is serial, so none of these calls have a Python
 * counterpart to port — they are *new* requirements that arise only because the
 * surface is distributed. Collecting them here, named and documented up front,
 * is what keeps them from being discovered one deadlock at a time later.
 *
 * Each stub records three things: **what** moves, **which** MPI operation is
 * expected, and **why it must happen at that point** — that is, the invariant
 * that breaks without it. The last is the part that cannot be reconstructed
 * from the code afterwards.
 *
 * The final partitioning is Tessera's decision, so the signatures here take a
 * mesh and views rather than naming a partitioner. What is *not* negotiable is
 * the placement of these calls in the control flow; that follows from the
 * mathematics and is fixed.
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
 * recorded as risk R2 in `tasks/framework.md`.
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
 * @brief Refresh ghost vertex positions from their owners.
 *
 * **What moves:** the `(3,)` position of every ghost vertex, from the rank that
 * owns it to every rank that ghosts it.
 *
 * **MPI operation:** neighbor point-to-point exchange
 * (`MPI_Isend`/`MPI_Irecv` over the persistent neighbor list, or
 * `MPI_Neighbor_alltoallv` on a distributed graph communicator). Not a
 * collective over `MPI_COMM_WORLD`.
 *
 * **Why here:** every differential operator in the solver — face normals and
 * areas, the cotangent Laplacian, the surface gradient, the mean-curvature
 * normal — reads the full one-ring of an owned vertex, and part of that ring is
 * ghost. If positions are stale the operators are evaluated on a torn surface:
 * face normals on boundary-straddling triangles point the wrong way, the
 * cotangent weights go negative, and the resulting velocity field has a
 * discontinuity along every rank boundary. This must run **after every write to
 * vertex positions** — that is, after each RK stage update, after the volume
 * projection, after tangential relaxation, and after any remeshing pass that
 * moves vertices.
 */
template <class MeshType>
void haloExchangeVertices( MeshType& mesh )
{
    (void)mesh;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "haloExchangeVertices" );
}

/**
 * @brief Refresh an arbitrary per-vertex field on the ghost layer.
 *
 * **What moves:** `ncomp` components per ghost vertex — the potential (1), the
 * sheet vector (3), the material position (3), or a per-vertex target edge
 * length (1).
 *
 * **MPI operation:** the same neighbor exchange as `haloExchangeVertices`,
 * reusing the same neighbor lists and offsets.
 *
 * **Why here:** the sheet vector is reconstructed from the *surface gradient*
 * of the potential, which is a one-ring operator; the RHS then takes a further
 * surface gradient of the Bernoulli potential. So the RHS is a two-ring
 * stencil on the potential. With only one ghost layer, the potential must be
 * exchanged **twice** per RHS evaluation — once before the gradient that builds
 * the sheet vector and once before the gradient of the Bernoulli potential — or
 * the ghost layer must be two faces deep. The width/exchange-count tradeoff is
 * a decision for the meshing task; what is fixed is that a single exchange of a
 * single-deep halo is *insufficient*, which is the easy bug to write here.
 */
template <class MeshType, class FieldView>
void haloExchangeField( MeshType& mesh, FieldView& field )
{
    (void)mesh;
    (void)field;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "haloExchangeField" );
}

/**
 * @brief Sum ghost contributions back onto their owners.
 *
 * **What moves:** partial accumulations deposited on ghost vertices by
 * face-loop kernels, sent back to the owning rank and summed there.
 *
 * **MPI operation:** the reverse of the halo exchange (neighbor
 * point-to-point), with an `+=` on arrival rather than an overwrite.
 *
 * **Why here:** the natural parallelization of every vertex quantity built from
 * a face loop — vertex area, the area-weighted vertex normal, the cotangent
 * Laplacian, the volume gradient, the area-weighted surface gradient — is
 * "scatter from faces to their three vertices". A face straddling a rank
 * boundary is owned by one rank but deposits onto vertices owned by another. If
 * the ghost deposits are dropped, boundary vertices see only part of their
 * one-ring's area: vertex areas are too small there, normals are mis-weighted,
 * and the Laplacian is wrong by exactly the missing sector. The symptom is a
 * seam of spurious velocity along partition boundaries that *moves* when the
 * rank count changes, which is the signature to look for.
 */
template <class MeshType, class FieldView>
void haloScatterAdd( MeshType& mesh, FieldView& field )
{
    (void)mesh;
    (void)field;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "haloScatterAdd" );
}

//---------------------------------------------------------------------------//
// 2. Global reductions
//---------------------------------------------------------------------------//

/**
 * @brief Global sum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_SUM, comm)`.
 *
 * **Why here:** four quantities in the solver are global sums over the surface
 * and every rank needs the answer, not just rank 0:
 *
 *   1. **Enclosed volume**
 *      \f$V = \frac{1}{6}\sum_f a_f\cdot(b_f\times c_f)\f$
 *      (`run_adaptive_mesh_bubble.py::mesh_enclosed_volume`, lines 1036-1040).
 *      Only owned faces may contribute or the shared ones are counted twice.
 *   2. **The two inner products of the volume projection**,
 *      \f$\sum_v |\nabla_v V|^2\f$ and \f$\sum_v \nabla_v V \cdot u_v\f$
 *      (`mesh_solver.py::_remove_discrete_volume_flux`, lines 285-295). These
 *      form a *single* global Rayleigh quotient, so both must be reduced before
 *      either is used; reducing them separately and dividing locally gives a
 *      different (wrong) answer on every rank.
 *   3. **The area-weighted mean of the potential**,
 *      \f$\bar\phi = \sum_v A_v\phi_v / \sum_v A_v\f$
 *      (`mesh_solver.py::_area_weighted_scalar_mean`, lines 239-244), subtracted
 *      to fix the arbitrary additive constant of the potential. If each rank
 *      subtracts its *local* mean the potential acquires a piecewise-constant
 *      jump across every partition boundary, and its surface gradient — the
 *      sheet vector — picks up a delta function there.
 *   4. **Total surface area**, the denominator of (3).
 */
inline Real allReduceSum( MPI_Comm comm, Real local )
{
    (void)comm;
    (void)local;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "allReduceSum" );
}

/**
 * @brief Global minimum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_MIN, comm)`.
 *
 * **Why here:** the adaptive timestep is
 * \f$\Delta t = \max(\Delta t_{\min}, \Delta t \cdot (h_{\min}/h_{\min}^0)^p)\f$
 * (`run_adaptive_mesh_bubble.py::choose_step_dt`, lines 889-901), where
 * \f$h_{\min}\f$ is the shortest edge **anywhere on the surface**. Every rank
 * must step with the *same* dt or the surface is integrated inconsistently and
 * the RK3 stages no longer correspond to a single time level — the result is
 * not merely inaccurate, it is not a discretization of anything. The same
 * applies to the minimum triangle quality that triggers the remesh repair pass.
 */
inline Real allReduceMin( MPI_Comm comm, Real local )
{
    (void)comm;
    (void)local;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "allReduceMin" );
}

/**
 * @brief Global maximum of a scalar.
 *
 * **What moves:** one `Real`.
 * **MPI operation:** `MPI_Allreduce(..., MPI_DOUBLE, MPI_MAX, comm)`.
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
    (void)comm;
    (void)local;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "allReduceMax" );
}

/**
 * @brief Global "are all values finite" test.
 *
 * **What moves:** one `int` (0/1).
 * **MPI operation:** `MPI_Allreduce(..., MPI_INT, MPI_LAND, comm)`.
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
    (void)comm;
    (void)local_finite;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "allReduceAllFinite" );
}

//---------------------------------------------------------------------------//
// 3. Refinement and load balancing
//---------------------------------------------------------------------------//

/**
 * @brief Make refinement marks consistent across rank boundaries.
 *
 * **What moves:** the boolean mark of every ghost face, plus the split-status
 * of every edge on a rank boundary; iterated to a fixed point.
 *
 * **MPI operation:** neighbor exchange of the ghost-face mark array inside a
 * loop, terminated by an `MPI_Allreduce(MPI_LOR)` on "did anything change this
 * sweep".
 *
 * **Why here:** red-green closure is a *fixed-point* computation
 * (`mesh_solver.py::_balance_red_green_refinement`, lines 1543-1580, loops
 * `while changed`). Marking a face implies splitting its three edges, which can
 * force a neighbor to be promoted from green to red, which splits *its* edges,
 * and so on. When that cascade crosses a rank boundary it must propagate, or
 * the two sides of the boundary make different decisions about the same edge
 * and the surface tears: one side has a midpoint vertex the other does not.
 * The iteration must be global — a single exchange is not enough, because a
 * cascade can traverse several ranks.
 */
template <class MeshType, class MarkView>
void reconcileRefinementMarks( MeshType& mesh, MarkView& marked )
{
    (void)mesh;
    (void)marked;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "reconcileRefinementMarks" );
}

/**
 * @brief Redistribute the surface after adaptation and migrate its fields.
 *
 * **What moves:** vertices, faces, and every per-vertex field (potential or
 * sheet vector, plus the material position) for the elements changing owner.
 *
 * **MPI operation:** an all-to-all migration
 * (`MPI_Alltoallv`, or Cabana's `Distributor` over the same communicator),
 * followed by a rebuild of the ghost layer.
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
 * **Every field must migrate with the mesh.** Migrating vertices without the
 * potential silently reinitializes the solution — a failure that looks like a
 * physics bug rather than a communication bug.
 */
template <class MeshType, class StateType>
void redistribute( MeshType& mesh, StateType& state )
{
    (void)mesh;
    (void)state;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "redistribute" );
}

//---------------------------------------------------------------------------//
// 4. Checkpoint gather
//---------------------------------------------------------------------------//

/**
 * @brief Collect the distributed surface into the rank-0 arrays the checkpoint
 *        writer expects.
 *
 * **What moves:** all owned vertices, faces and per-vertex fields, to rank 0.
 *
 * **MPI operation:** `MPI_Gatherv` per array, ordered by global vertex/face id
 * so the file layout does not depend on the rank count. (An HDF5 collective
 * write with per-rank hyperslabs is the scalable alternative and is what
 * Tessera is expected to provide; the gather is the reference path.)
 *
 * **Why here:** the gold files are single, whole-surface `.npz` arrays, and
 * `compare_output.py` compares one file against one file. More importantly,
 * **face connectivity must be renumbered from local to global vertex indices
 * before the gather** — local indices are meaningless once the arrays from
 * different ranks are concatenated. Getting this wrong produces a file that
 * loads cleanly and describes a completely scrambled surface, which the
 * comparator will report as a vertex mismatch rather than as the connectivity
 * bug it is.
 *
 * Ghost entities must be excluded, or vertices appear multiple times and the
 * vertex count exceeds the true global count — the comparator's structural
 * check catches that one immediately.
 */
template <class MeshType, class StateType, class HostArrays>
void gatherForCheckpoint( const MeshType& mesh, const StateType& state,
                          HostArrays& out )
{
    (void)mesh;
    (void)state;
    (void)out;
    BEATNIK_NOT_IMPLEMENTED( "Comm", "gatherForCheckpoint" );
}

/**
 * @brief Broadcast a rank-0-read surface to all ranks for partitioning.
 *
 * **What moves:** the whole vertex and face arrays plus the fields, from rank 0
 * outward.
 *
 * **MPI operation:** `MPI_Bcast` of the array extents, then of the arrays.
 *
 * **Why here:** the inverse of `gatherForCheckpoint`, used by `--restart-from`
 * and by the "read the initial mesh from the gold file" mitigation for
 * regression test 1. Ranks other than 0 have nothing until this completes, so
 * the extents must be broadcast first — allocating from a stale or zero extent
 * is the usual bug.
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
