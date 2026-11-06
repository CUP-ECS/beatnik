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
 * @file Beatnik_MeshQuality.hpp
 * @brief Valence-equalizing cleanup: the regularization the quality-driven
 *        remesher does not provide.
 *
 * THE PROBLEM THIS SOLVES
 * -----------------------
 * Port of mesh_quality.py::isotropic_cleanup (lines 146-167)
 *
 * `DynamicRemesh::flipEdgesForQuality` flips for **triangle quality** and never
 * for **valence**. During a tightening roll-up the curvature sizing field
 * refines hard, and quality-only flips leave behind a high-valence,
 * small-area tangle — the "curvature sliver". Empirically that tangle kills the
 * run, or drives dt to its floor, **long before** the sheets actually
 * self-contact. So the simulation dies of a meshing artifact rather than
 * reaching the physics it was set up to study.
 *
 * The fix, run after every remesh (`--isotropic-cleanup`, on by default), is
 * two passes that regularize connectivity and element size **without changing
 * the surface shape**:
 *   1. valence-equalizing flips, driving vertex valences toward 6;
 *   2. tangential relaxation, equalizing triangle areas.
 *
 * Both preserve the vertex **count**, and the flips change only connectivity,
 * so per-vertex fields stay valid — the caller rebuilds the state on the
 * returned `(vertices, faces)` with the same potential
 * (`run_adaptive_mesh_bubble.py:1491-1504`). No field remap is needed, which is
 * why this is a separate, cheaper operation than a remesh.
 */

#ifndef BEATNIK_MESHQUALITY_HPP
#define BEATNIK_MESHQUALITY_HPP

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What one quality-flip sweep did: Tessera's verdict counters summed
 *        over the sweep's rounds, and the number of rounds.
 *
 * Every count is **global** (Tessera's own, never re-reduced), so it is the
 * same on every rank and comparable across rank counts. `requested ==
 * accepted + Σ rejected_*` over the sweep, as for one `flipEdges()` call.
 * Risk M1-R7 is the reason this exists: a milestone run whose flip counts are
 * all zero is a silent no-op, not agreement.
 */
struct FlipSweepStats
{
    /// `SurfaceMesh::flipEdges` calls. A round whose mask is empty on every
    /// rank ends the sweep **without** a call and is not counted.
    int rounds = 0;
    long long requested = 0;
    long long accepted = 0;
    long long rejected_boundary = 0;
    long long rejected_duplicate_edge = 0;
    long long rejected_geometric = 0;
    long long rejected_conflict = 0;
};

/**
 * @brief What one valence-equalizing pass did: Tessera's counters summed over
 *        the pass's rounds, and every round's accepted flips.
 *
 * `rounds[r]` is round `r`'s accepted flips, gathered from every rank and
 * deduplicated by old edge, so it is the same on every rank and its size is
 * that round's global `accepted`. It is what lets a caller check the pass's
 * touched-set rule from outside: no accepted flip has an endpoint among the
 * corners of a flip accepted in an earlier round of the same pass.
 */
struct ValenceFlipPass
{
    FlipSweepStats stats;
    std::vector<std::vector<FlippedEdge>> rounds;
};

/**
 * @brief The reference's quality-flip acceptance test, with the quality floor
 *        as a parameter: one predicate for both entry points.
 *
 * Port of dynamic_remesh.py::flip_edges_for_quality (lines 430-451) and
 * mesh_solver.py::improve_mesh_connectivity_by_edge_flips (lines 1744-1758)
 *
 * The two differ only in the floor: the remesh pass rejects a flip whose worse
 * new face is below `--remesh-min-quality` (`dynamic_remesh.py:446-447`), and
 * the refine-path pass has no floor (`mesh_solver.py:1755-1758`).
 */
struct QualityFlipCriterion
{
    /// Whether `floor` applies.
    bool has_floor = false;
    /// Reject when \f$\min(q_{\text{new}}) < \f$ `floor`.
    double floor = 0.0;
    /// Accept only when \f$\min(q_{\text{new}}) > \min(q_{\text{old}})(1+g)\f$.
    double min_gain = 0.0;
};

namespace QualityFlips
{

/**
 * @brief The reference triangle quality \f$4\sqrt3 A/\sum\ell^2\f$ on host
 *        3-vectors.
 *
 * Port of dynamic_remesh.py::triangle_quality (lines 626-636)
 *
 * The formula of `SurfaceOperators::triangleQuality`, evaluated on the host
 * because mask construction is a host pass. A zero edge-length sum gives 0,
 * as the reference's `np.divide( …, where=l2_sum > 0 )` does; any other
 * non-finite input propagates, so a caller can apply the reference's
 * `np.isfinite` test itself.
 */
inline double triangleQuality( const double* p0, const double* p1,
                               const double* p2 )
{
    double u[3], v[3], w[3];
    for ( int d = 0; d < 3; ++d )
    {
        u[d] = p1[d] - p0[d];
        v[d] = p2[d] - p1[d];
        w[d] = p0[d] - p2[d];
    }
    const double cx = u[1] * w[2] - u[2] * w[1];
    const double cy = u[2] * w[0] - u[0] * w[2];
    const double cz = u[0] * w[1] - u[1] * w[0];
    const double area = 0.5 * std::sqrt( cx * cx + cy * cy + cz * cz );
    const double l2 = ( u[0] * u[0] + u[1] * u[1] + u[2] * u[2] ) +
                      ( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] ) +
                      ( w[0] * w[0] + w[1] * w[1] + w[2] * w[2] );
    if ( !( l2 > 0.0 ) )
        return 0.0;
    return 4.0 * std::sqrt( 3.0 ) * area / l2;
}

/// Unit normal \f$(p_1-p_0)\times(p_2-p_0)\f$ normalized, or zero for a
/// degenerate triangle. Port of dynamic_remesh.py::face_normals (lines
/// 605-611).
inline void unitNormal( const double* p0, const double* p1, const double* p2,
                        double n[3] )
{
    const double u[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    const double v[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
    const double len = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
    for ( int d = 0; d < 3; ++d )
        n[d] = ( len > 0.0 ) ? n[d] / len : 0.0;
}

/**
 * @brief Does flipping edge \f$(a,b)\f$ to \f$(c,d)\f$ pass the reference's
 *        test?
 *
 * Port of dynamic_remesh.py::flip_edges_for_quality (lines 430-451) and
 * mesh_solver.py::improve_mesh_connectivity_by_edge_flips (lines 1744-1758)
 *
 * `f0` and `f1` are the two old faces' corners in their stored (outward)
 * order; `c` is `f0`'s corner off the edge and `d` is `f1`'s. In order:
 *
 *   1. the summed unit normal of the old faces must not vanish — the
 *      reference's orientation reference (`:434-436`, `:1745-1748`). The
 *      remesh pass tests \f$\|n\| \le 10^{-300}\f$ and the refine-path pass
 *      \f$\|n\| = 0\f$ or non-finite; this tests the former, which also
 *      rejects a non-finite norm. A sum of two unit vectors is either zero or
 *      far above \f$10^{-300}\f$, so the two agree on every finite input;
 *   2. both new faces \f$(c,d,a)\f$ and \f$(d,c,b)\f$ have a finite quality;
 *   3. where a floor is given, the worse new face is not below it;
 *   4. \f$\min(q_{\text{new}}) > \min(q_{\text{old}})(1+g)\f$ — strictly, so
 *      a flip and its inverse cannot both pass.
 *
 * The reference re-orients each new face against the summed normal
 * (`:437-443`); quality does not depend on orientation, and Tessera's rewrite
 * preserves it, so that step has nothing to decide here. The reference also
 * skips a flip whose diagonal already exists; the mask leaves that to Tessera,
 * which rejects it exactly into `rejected_duplicate_edge`
 * (`tasks/milestone1.md`, "Read this first" #6).
 */
inline bool accepts( const double* const f0[3], const double* const f1[3],
                     const double* a, const double* b, const double* c,
                     const double* d, const QualityFlipCriterion& k )
{
    double n0[3], n1[3];
    unitNormal( f0[0], f0[1], f0[2], n0 );
    unitNormal( f1[0], f1[1], f1[2], n1 );
    const double s[3] = { n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2] };
    const double norm = std::sqrt( s[0] * s[0] + s[1] * s[1] + s[2] * s[2] );
    if ( !( norm > 1.0e-300 ) )
        return false;

    const double q_old = std::min( triangleQuality( f0[0], f0[1], f0[2] ),
                                   triangleQuality( f1[0], f1[1], f1[2] ) );
    const double qa = triangleQuality( c, d, a );
    const double qb = triangleQuality( d, c, b );
    if ( !std::isfinite( qa ) || !std::isfinite( qb ) )
        return false;
    const double q_new = std::min( qa, qb );
    if ( k.has_floor && q_new < k.floor )
        return false;
    if ( q_new <= q_old * ( 1.0 + k.min_gain ) )
        return false;
    return true;
}

/// An edge named by its endpoints' vertex gids, ascending.
using GidPair = std::pair<std::uint64_t, std::uint64_t>;

inline GidPair gidPair( std::uint64_t u, std::uint64_t v )
{
    return ( u < v ) ? GidPair( u, v ) : GidPair( v, u );
}

/**
 * @brief Gather every rank's `MeshEditReport::flipped` and deduplicate it by
 *        the old edge: the whole round's accepted flips, on every rank.
 *
 * `flipped` lists only the flips the calling rank touches, so one flip appears
 * on every rank that owns its edge or either face and nowhere else. After the
 * gather every rank holds the same set, in the same order, which is the whole
 * round.
 *
 * @throws std::runtime_error when the deduplicated count differs from the
 *         global `accepted`. Both are the same on every rank, so every rank
 *         throws or none does.
 *
 * @note MPI. Collective: one `MPI_Allgather` and one `MPI_Allgatherv`.
 */
inline std::vector<FlippedEdge> gatherFlips( MPI_Comm comm,
                                             const MeshEditReport& report )
{
    std::vector<std::uint64_t> mine;
    mine.reserve( 4 * report.flipped.size() );
    for ( const FlippedEdge& f : report.flipped )
    {
        mine.push_back( f.old_v[0] );
        mine.push_back( f.old_v[1] );
        mine.push_back( f.new_v[0] );
        mine.push_back( f.new_v[1] );
    }
    int size = 1;
    MPI_Comm_size( comm, &size );
    int n = static_cast<int>( mine.size() );
    std::vector<int> counts( size ), displs( size, 0 );
    MPI_Allgather( &n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );
    for ( int r = 1; r < size; ++r )
        displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<std::uint64_t> all( static_cast<std::size_t>(
        displs[size - 1] + counts[size - 1] ) );
    MPI_Allgatherv( mine.data(), n, MPI_UINT64_T, all.data(), counts.data(),
                    displs.data(), MPI_UINT64_T, comm );

    std::set<GidPair> old_seen;
    std::vector<FlippedEdge> flips;
    for ( std::size_t k = 0; k + 3 < all.size(); k += 4 )
    {
        if ( !old_seen.insert( gidPair( all[k], all[k + 1] ) ).second )
            continue;
        FlippedEdge f;
        f.old_v[0] = all[k];
        f.old_v[1] = all[k + 1];
        f.new_v[0] = all[k + 2];
        f.new_v[1] = all[k + 3];
        flips.push_back( f );
    }
    if ( static_cast<long long>( flips.size() ) != report.accepted )
        throw std::runtime_error(
            "Beatnik flips: the gathered flip map names " +
            std::to_string( flips.size() ) +
            " distinct flips but flipEdges accepted " +
            std::to_string( report.accepted ) +
            "; every accepted flip is touched by at least its edge's owner "
            "(Tessera_EdgeFlip.hpp, FlipResult::flipped)." );
    return flips;
}

/**
 * @brief The new diagonals of a round's accepted flips, gathered from every
 *        rank — `gatherFlips` reduced to what the quality sweep's touched set
 *        holds.
 *
 * @note MPI. Collective, as `gatherFlips`.
 */
inline std::vector<GidPair> gatherNewDiagonals( MPI_Comm comm,
                                                const MeshEditReport& report )
{
    std::vector<GidPair> diagonals;
    for ( const FlippedEdge& f : gatherFlips( comm, report ) )
        diagonals.push_back( gidPair( f.new_v[0], f.new_v[1] ) );
    return diagonals;
}

/**
 * @brief Every local vertex position, owned then ghost, in a host view whose
 *        rows are contiguous 3-vectors.
 *
 * LayoutRight, so `&pos( i, 0 )` is a `const double*` to a 3-vector whatever
 * the device's default layout — the form `triangleQuality` and `unitNormal`
 * take.
 */
template <class Mesh>
Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace>
hostPositions( Mesh& mesh )
{
    using exec_space = typename Mesh::execution_space;
    using mem_space = typename Mesh::memory_space;
    const int nv = mesh.totalVertexCount();
    Kokkos::View<double* [3], Kokkos::LayoutRight, mem_space> pos_d(
        "beatnik_flip_positions", nv );
    auto pos = mesh.positions();
    Kokkos::parallel_for(
        "beatnik_flip_copy_positions",
        Kokkos::RangePolicy<exec_space>( 0, nv ), KOKKOS_LAMBDA( const int i ) {
            for ( int d = 0; d < 3; ++d )
                pos_d( i, d ) = pos( i, d );
        } );
    Kokkos::fence();
    return Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pos_d );
}

/// The corner of face `f` that is neither `a` nor `b`, or -1. Port of
/// dynamic_remesh.py::opposite_vertex (lines 1138-1144).
template <class FaceVertices>
int oppositeCorner( const FaceVertices& fv, const int f, const int a,
                    const int b )
{
    for ( int k = 0; k < 3; ++k )
        if ( fv( f, k ) != a && fv( f, k ) != b )
            return fv( f, k );
    return -1;
}

} // namespace QualityFlips

/**
 * @brief One reference quality-flip **sweep**, as several
 *        `SurfaceMesh::flipEdges` rounds.
 *
 * Port of the sweep in dynamic_remesh.py::flip_edges_for_quality
 * (lines 408-457), which is also one pass of
 * mesh_solver.py::improve_mesh_connectivity_by_edge_flips (lines 1720-1768)
 *
 * The reference tests every edge of one edge-map snapshot, in order, and
 * skips an edge if either incident face was already rewritten in the sweep
 * (`touched`, `:416-421`). One Tessera call accepts one independent set that
 * is not maximal, so one sweep here is several rounds sharing a **sweep-wide
 * touched set**:
 *
 *   1. mark every owned edge that passes `QualityFlips::accepts` and has no
 *      touched incident face;
 *   2. if no rank marked anything, end the sweep **without** calling
 *      `flipEdges` — a call on an empty mask would still rebuild the halo
 *      and bump `generation()`, and the disabled configurations must mutate
 *      nothing;
 *   3. otherwise `flipEdges` with the permissive policy
 *      `{maxNormalDeviation = π, minQuality = 0}` and the default
 *      `FlipConflict::SharedFace` relation — two flips may not share a face,
 *      which is the reference's rule here (`:416-421`);
 *   4. gather the accepted flips and add their new diagonals to the touched
 *      set; end the sweep if the round accepted nothing.
 *
 * **The touched set holds new diagonals, by vertex gid.** A flip preserves
 * every gid and the two faces it rewrites are exactly the two faces of its new
 * diagonal, so a face is touched iff one of its edges is a diagonal created in
 * this sweep. No touched face is rewritten again within the sweep, so that
 * stays true for the whole sweep, and no edge created in it is tested in it.
 * Re-running rounds with a fresh mask instead would re-test rewritten faces
 * and flip more than the reference.
 *
 * Each round's verdicts are Tessera's: the accepted set is not the
 * reference's serial set (framework risk R7), so compare statistics, never
 * flip sets.
 *
 * @pre Vertex positions are halo-consistent: the mask reads ghost corners and
 *      `flipEdges` does not exchange. A flip moves no vertex, so this holds
 *      for every later round.
 *
 * @throws std::runtime_error if an owned edge's quad is not fully resident —
 *         `faceAdjacency().non_resident` nonzero on any rank, or an owned edge
 *         with other than two resident faces or a face corner not held here.
 *         The counts are reduced before the test, so every rank throws or
 *         none does.
 *
 * @note MPI. Collective. Every branch is on a global count, so every rank
 *       reaches the same `faceAdjacency()`, reductions and `flipEdges` calls.
 */
template <class ExecutionSpace, class MemorySpace>
FlipSweepStats qualityFlipSweep( SurfaceMesh<ExecutionSpace, MemorySpace>& mesh,
                                 const QualityFlipCriterion& criterion )
{
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using QualityFlips::GidPair;

    const MPI_Comm comm = mesh.comm();

    typename mesh_type::flip_policy_type policy;
    policy.maxNormalDeviation = std::acos( -1.0 );
    policy.minQuality = 0.0;
    policy.conflict = mesh_type::flip_conflict_type::SharedFace;

    FlipSweepStats stats;
    std::set<GidPair> diagonals;

    while ( true )
    {
        // The quad is read from the neighbour face's positions, which makes
        // this a geometric consumer of the face adjacency: it must be fully
        // resident. Collective, so called unconditionally on every rank.
        const long long non_resident = mesh.faceAdjacency().non_resident;

        const int n_owned_e = mesh.ownedEdgeCount();
        const auto incidence = mesh.edgeAdjacency();
        auto rc = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), incidence.resident_count );
        auto rf = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), incidence.resident_faces );
        auto ev = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       mesh.edgeVertices() );
        auto fv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       mesh.faceVertices() );
        const auto pos = QualityFlips::hostPositions( mesh );
        const std::vector<std::uint64_t> gid = mesh.vertexGids();

        auto touched = [&]( const int f )
        {
            if ( diagonals.empty() )
                return false;
            for ( int k = 0; k < 3; ++k )
            {
                const int u = fv( f, k );
                const int v = fv( f, ( k + 1 ) % 3 );
                if ( diagonals.count( QualityFlips::gidPair( gid[u], gid[v] ) ) )
                    return true;
            }
            return false;
        };

        std::vector<char> mask( static_cast<std::size_t>( n_owned_e ), 0 );
        long long counts[3] = { non_resident, 0, 0 }; // non-resident,
                                                      // unresolved, marked
        for ( int e = 0; e < n_owned_e; ++e )
        {
            const int a = ev( e, 0 );
            const int b = ev( e, 1 );
            const int f0 = rf( e, 0 );
            const int f1 = rf( e, 1 );
            if ( rc( e ) != 2 || a < 0 || b < 0 || f0 < 0 || f1 < 0 ||
                 fv( f0, 0 ) < 0 || fv( f0, 1 ) < 0 || fv( f0, 2 ) < 0 ||
                 fv( f1, 0 ) < 0 || fv( f1, 1 ) < 0 || fv( f1, 2 ) < 0 )
            {
                ++counts[1];
                continue;
            }
            if ( touched( f0 ) || touched( f1 ) )
                continue;
            const int c = QualityFlips::oppositeCorner( fv, f0, a, b );
            const int d = QualityFlips::oppositeCorner( fv, f1, a, b );
            if ( c < 0 || d < 0 || c == d )
                continue;

            const double* p0[3] = { &pos( fv( f0, 0 ), 0 ),
                                    &pos( fv( f0, 1 ), 0 ),
                                    &pos( fv( f0, 2 ), 0 ) };
            const double* p1[3] = { &pos( fv( f1, 0 ), 0 ),
                                    &pos( fv( f1, 1 ), 0 ),
                                    &pos( fv( f1, 2 ), 0 ) };
            if ( QualityFlips::accepts( p0, p1, &pos( a, 0 ), &pos( b, 0 ),
                                        &pos( c, 0 ), &pos( d, 0 ),
                                        criterion ) )
            {
                mask[static_cast<std::size_t>( e )] = 1;
                ++counts[2];
            }
        }

        long long global[3] = { 0, 0, 0 };
        MPI_Allreduce( counts, global, 3, MPI_LONG_LONG, MPI_SUM, comm );
        if ( global[0] != 0 || global[1] != 0 )
            throw std::runtime_error(
                "Beatnik quality flips: " + std::to_string( global[0] ) +
                " owned face-adjacency entries are not resident and " +
                std::to_string( global[1] ) +
                " owned edges have no fully resident quad; the flip test "
                "reads both incident faces' corners and cannot be evaluated "
                "(SurfaceMesh::FaceAdjacencyCsr::non_resident)." );
        if ( global[2] == 0 )
            break;

        const MeshEditReport report = mesh.flipEdges( mask, policy );
        ++stats.rounds;
        stats.requested += report.requested;
        stats.accepted += report.accepted;
        stats.rejected_boundary += report.rejected_boundary;
        stats.rejected_duplicate_edge += report.rejected_duplicate_edge;
        stats.rejected_geometric += report.rejected_geometric;
        stats.rejected_conflict += report.rejected_conflict;

        for ( const GidPair& g : QualityFlips::gatherNewDiagonals( comm, report ) )
            diagonals.insert( g );
        if ( report.accepted == 0 )
            break;
    }
    return stats;
}

/// The valence flip's normal test: each new face's unit normal against the
/// old face it replaces (`mesh_quality.py:78`).
constexpr double kMinValenceFlipNormalDot = 0.2;
/// The valence flip's quality floor on the worse new face
/// (`mesh_quality.py:80`).
constexpr double kMinValenceFlipQuality = 0.05;

//---------------------------------------------------------------------------//
/**
 * @brief Connectivity and parameterization regularization.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class MeshQuality
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;

    /// @param params Cleanup pass counts and relaxation weight.
    explicit MeshQuality( const CleanupParams& params )
        : _params( params )
    {
    }

    /**
     * @brief Flip edges to drive vertex valences toward 6 (Botsch-Kobbelt).
     *
     * Port of mesh_quality.py::_valence_equalizing_flips (lines 44-87)
     *
     * For the shared edge \f$(a,b)\f$ of two triangles \f$f_0\f$ (opposite
     * vertex \f$c\f$) and \f$f_1\f$ (opposite \f$d\f$), flipping to
     * \f$(c,d)\f$ moves one unit of valence from \f$a,b\f$ to \f$c,d\f$. The
     * edge is marked iff the flip strictly reduces the total deviation from
     * valence 6:
     * \f[
     *   |v_a-7|+|v_b-7|+|v_c-5|+|v_d-5| \;<\;
     *   |v_a-6|+|v_b-6|+|v_c-6|+|v_d-6| ,
     * \f]
     * with \f$v\f$ the valences at the start of the pass; and iff each new
     * face's unit normal has a dot product of at least 0.2 with the
     * pass-start unit normal of the old face it replaces (a generous margin,
     * not merely a sign test, because a near-inversion is nearly as bad); and
     * iff the worse new face has reference quality at least 0.05
     * (`QualityFlips::triangleQuality`). An edge whose two opposite corners
     * coincide is skipped, as in the reference.
     *
     * **The new faces are wound as Tessera writes them**, consistently with
     * the mesh: the child holding \f$a\f$ replaces \f$f_0\f$ and the child
     * holding \f$b\f$ replaces \f$f_1\f$. The reference writes them as
     * \f$(c,d,a)\f$ and \f$(d,c,b)\f$ whatever \f$f_0\f$'s winding, so on
     * every edge whose first-indexed face runs \f$a\to b\f$ its normal test
     * compares an inverted child and rejects; on the default icosphere that
     * is 206 of 480 edges. The rank-local face order here has no such index,
     * so the test is the one the reference intends, which accepts flips the
     * reference does not (`tasks/milestone1.md`, Deliberate deviations).
     *
     * Valence is the face-incidence count, which on a closed manifold is the
     * one-ring size `vertexOneRing()` holds for every owned vertex. Whether the
     * new diagonal already exists is **not** tested: Tessera rejects that
     * flip into `rejected_duplicate_edge` (`tasks/milestone1.md`, Read this
     * first #6).
     *
     * **One reference pass is several `flipEdges` rounds** under
     * `FlipConflict::EndpointInQuad`, sharing a pass-wide set of touched
     * vertex gids:
     *
     *   1. mark every owned edge that passes the tests above and has neither
     *      endpoint touched;
     *   2. if no rank marked anything, end the pass **without** calling
     *      `flipEdges`;
     *   3. otherwise `flipEdges` with the permissive policy
     *      `{maxNormalDeviation = π, minQuality = 0,
     *      conflict = EndpointInQuad}` — no accepted flip has an endpoint
     *      among another's four corners, the reference's `touched` rule;
     *   4. gather the round's flips and add all four corners of each to the
     *      touched set; end the pass if the round accepted nothing.
     *
     * Re-running rounds with a fresh mask instead would re-test edges at
     * touched vertices and flip more than the reference.
     *
     * **The pass-start valences** are reconstructed each round, exactly: the
     * current valence (owned rows from the one-ring, ghost rows by
     * `haloExchangeVertexView`) plus, per gid, the number of this pass's
     * flips that removed an edge at the vertex, minus the number that added
     * one, counted from the gathered flips, which are the same on every
     * rank. A vertex that enters a rank's halo mid-pass is covered by the same
     * rule. **The pass-start face normals need no storing**: a face rewritten
     * in the pass has all three corners touched, so no later candidate edge
     * lies on it, and every face the mask reads still has the positions and
     * connectivity it had at the start of the pass.
     *
     * The accepted set is Tessera's and is not the reference's serial set
     * (framework risk R7): compare valence statistics, never flip sets.
     *
     * Volume is not controlled: a flip on a curved surface changes the
     * enclosed volume by its quad's tetrahedron, as in the reference, whose
     * call sites restore it with `project_state_to_volume`.
     *
     * @param mesh   Surface, connectivity edited in place. Halo-exchanged on
     *               entry, because the mask reads ghost corners.
     * @param passes Sweeps; a non-positive count returns 0 before anything
     *               collective. Exits early after a pass that flips nothing,
     *               as `mesh_quality.py:85-86` does.
     * @param[out] record When non-null, one `ValenceFlipPass` is appended per
     *             pass run, with its counters and each round's gathered flips.
     * @return Total flips accepted, globally.
     *
     * @throws std::runtime_error if an owned edge's quad is not fully
     *         resident, as `qualityFlipSweep` does; every rank throws or none.
     *
     * @note MPI. Collective whenever `passes > 0`. Every branch is on a global
     *       count, a parameter or the gathered flips, so every rank reaches the
     *       same `faceAdjacency()`, halo exchanges, reductions and `flipEdges`
     *       calls.
     */
    int valenceEqualizingFlips(
        mesh_type& mesh, int passes,
        std::vector<ValenceFlipPass>* record = nullptr ) const
    {
        const int n_passes = ( passes > 0 ) ? passes : 0;
        if ( n_passes == 0 )
            return 0;

        const MPI_Comm comm = mesh.comm();
        mesh.haloExchange();

        typename mesh_type::flip_policy_type policy;
        policy.maxNormalDeviation = std::acos( -1.0 );
        policy.minQuality = 0.0;
        policy.conflict = mesh_type::flip_conflict_type::EndpointInQuad;

        long long total = 0;
        for ( int pass = 0; pass < n_passes; ++pass )
        {
            ValenceFlipPass this_pass;
            std::set<std::uint64_t> touched;
            // Pass-start valence minus current valence, by vertex gid.
            std::map<std::uint64_t, int> shift;

            while ( true )
            {
                // Collective, so called unconditionally on every rank; see
                // qualityFlipSweep.
                const long long non_resident =
                    mesh.faceAdjacency().non_resident;

                const int n_owned_e = mesh.ownedEdgeCount();
                const int n_owned_v = mesh.ownedVertexCount();
                const int nv = mesh.totalVertexCount();
                const auto incidence = mesh.edgeAdjacency();
                auto rc = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), incidence.resident_count );
                auto rf = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), incidence.resident_faces );
                auto ev = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), mesh.edgeVertices() );
                auto fv = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), mesh.faceVertices() );
                const auto pos = QualityFlips::hostPositions( mesh );
                const std::vector<std::uint64_t> gid = mesh.vertexGids();

                // Current valence: owned rows from the one-ring, ghost rows
                // from their owners.
                Kokkos::View<Real*, MemorySpace> val_d( "beatnik_valence",
                                                        nv );
                {
                    auto off = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), mesh.vertexOneRing().offsets );
                    auto val_h = Kokkos::create_mirror_view( val_d );
                    for ( int v = 0; v < nv; ++v )
                        val_h( v ) = ( v < n_owned_v )
                                         ? Real( off( v + 1 ) - off( v ) )
                                         : Real( 0 );
                    Kokkos::deep_copy( val_d, val_h );
                }
                mesh.haloExchangeVertexView( val_d );
                auto val_h = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), val_d );
                auto valence = [&]( const int v )
                {
                    const auto it = shift.find( gid[v] );
                    return static_cast<int>( val_h( v ) ) +
                           ( ( it == shift.end() ) ? 0 : it->second );
                };

                std::vector<char> mask( static_cast<std::size_t>( n_owned_e ),
                                        0 );
                long long counts[3] = { non_resident, 0, 0 }; // non-resident,
                                                              // unresolved,
                                                              // marked
                for ( int e = 0; e < n_owned_e; ++e )
                {
                    const int a = ev( e, 0 );
                    const int b = ev( e, 1 );
                    const int f0 = rf( e, 0 );
                    const int f1 = rf( e, 1 );
                    if ( rc( e ) != 2 || a < 0 || b < 0 || f0 < 0 || f1 < 0 ||
                         fv( f0, 0 ) < 0 || fv( f0, 1 ) < 0 ||
                         fv( f0, 2 ) < 0 || fv( f1, 0 ) < 0 ||
                         fv( f1, 1 ) < 0 || fv( f1, 2 ) < 0 )
                    {
                        ++counts[1];
                        continue;
                    }
                    if ( touched.count( gid[a] ) || touched.count( gid[b] ) )
                        continue;
                    const int c = QualityFlips::oppositeCorner( fv, f0, a, b );
                    const int d = QualityFlips::oppositeCorner( fv, f1, a, b );
                    if ( c < 0 || d < 0 || c == d )
                        continue;

                    const int va = valence( a );
                    const int vb = valence( b );
                    const int vc = valence( c );
                    const int vd = valence( d );
                    const int dev0 = std::abs( va - 6 ) + std::abs( vb - 6 ) +
                                     std::abs( vc - 6 ) + std::abs( vd - 6 );
                    const int dev1 = std::abs( va - 7 ) + std::abs( vb - 7 ) +
                                     std::abs( vc - 5 ) + std::abs( vd - 5 );
                    if ( dev1 >= dev0 )
                        continue;

                    // The children as Tessera winds them: if f0 runs a -> b
                    // the quad's boundary is a, d, b, c and the children are
                    // (c, a, d) and (d, b, c); otherwise (c, d, a) and
                    // (d, c, b), the reference's literal pair.
                    int k = 0;
                    while ( fv( f0, k ) != a )
                        ++k;
                    const bool forward = ( fv( f0, ( k + 1 ) % 3 ) == b );
                    const int t0[3] = { c, forward ? a : d, forward ? d : a };
                    const int t1[3] = { d, forward ? b : c, forward ? c : b };

                    double n_old0[3], n_old1[3], n_new0[3], n_new1[3];
                    QualityFlips::unitNormal( &pos( fv( f0, 0 ), 0 ),
                                              &pos( fv( f0, 1 ), 0 ),
                                              &pos( fv( f0, 2 ), 0 ), n_old0 );
                    QualityFlips::unitNormal( &pos( fv( f1, 0 ), 0 ),
                                              &pos( fv( f1, 1 ), 0 ),
                                              &pos( fv( f1, 2 ), 0 ), n_old1 );
                    QualityFlips::unitNormal( &pos( t0[0], 0 ),
                                              &pos( t0[1], 0 ),
                                              &pos( t0[2], 0 ), n_new0 );
                    QualityFlips::unitNormal( &pos( t1[0], 0 ),
                                              &pos( t1[1], 0 ),
                                              &pos( t1[2], 0 ), n_new1 );
                    const double dot0 = n_new0[0] * n_old0[0] +
                                        n_new0[1] * n_old0[1] +
                                        n_new0[2] * n_old0[2];
                    const double dot1 = n_new1[0] * n_old1[0] +
                                        n_new1[1] * n_old1[1] +
                                        n_new1[2] * n_old1[2];
                    if ( dot0 < kMinValenceFlipNormalDot ||
                         dot1 < kMinValenceFlipNormalDot )
                        continue;

                    const double q0 = QualityFlips::triangleQuality(
                        &pos( t0[0], 0 ), &pos( t0[1], 0 ), &pos( t0[2], 0 ) );
                    const double q1 = QualityFlips::triangleQuality(
                        &pos( t1[0], 0 ), &pos( t1[1], 0 ), &pos( t1[2], 0 ) );
                    if ( q0 < kMinValenceFlipQuality ||
                         q1 < kMinValenceFlipQuality )
                        continue;

                    mask[static_cast<std::size_t>( e )] = 1;
                    ++counts[2];
                }

                long long global[3] = { 0, 0, 0 };
                MPI_Allreduce( counts, global, 3, MPI_LONG_LONG, MPI_SUM,
                               comm );
                if ( global[0] != 0 || global[1] != 0 )
                    throw std::runtime_error(
                        "Beatnik valence flips: " +
                        std::to_string( global[0] ) +
                        " owned face-adjacency entries are not resident and " +
                        std::to_string( global[1] ) +
                        " owned edges have no fully resident quad; the flip "
                        "test reads both incident faces' corners and cannot "
                        "be evaluated (SurfaceMesh::FaceAdjacencyCsr::"
                        "non_resident)." );
                if ( global[2] == 0 )
                    break;

                const MeshEditReport report = mesh.flipEdges( mask, policy );
                FlipSweepStats& st = this_pass.stats;
                ++st.rounds;
                st.requested += report.requested;
                st.accepted += report.accepted;
                st.rejected_boundary += report.rejected_boundary;
                st.rejected_duplicate_edge += report.rejected_duplicate_edge;
                st.rejected_geometric += report.rejected_geometric;
                st.rejected_conflict += report.rejected_conflict;

                std::vector<FlippedEdge> flips =
                    QualityFlips::gatherFlips( comm, report );
                for ( const FlippedEdge& f : flips )
                {
                    for ( int j = 0; j < 2; ++j )
                    {
                        touched.insert( f.old_v[j] );
                        touched.insert( f.new_v[j] );
                        ++shift[f.old_v[j]];
                        --shift[f.new_v[j]];
                    }
                }
                this_pass.rounds.push_back( std::move( flips ) );
                if ( report.accepted == 0 )
                    break;
            }

            total += this_pass.stats.accepted;
            const bool flipped = ( this_pass.stats.accepted > 0 );
            if ( record != nullptr )
                record->push_back( std::move( this_pass ) );
            if ( !flipped )
                break;
        }
        return static_cast<int>( total );
    }

    /**
     * @brief Move vertices toward their neighbor centroid, tangentially.
     *
     * Port of mesh_quality.py::_tangential_relaxation (lines 90-116)
     *
     * \f[
     *   x_v \leftarrow x_v + w\,\Big[\,\bar x_v - x_v
     *     - \big((\bar x_v - x_v)\cdot\hat n_v\big)\hat n_v \Big],
     *   \qquad \bar x_v = \frac{1}{|N(v)|}\sum_{j\in N(v)} x_j .
     * \f]
     *
     * The normal projection is the whole point: it equalizes triangle areas
     * while leaving the resolved **shape** untouched. Without it this is
     * Laplacian smoothing of the interface, which shrinks the bubble and reads
     * as excessive numerical dissipation rather than as the geometry bug it is.
     *
     * The vertex normals are recomputed **each pass**, since the surface
     * parameterization has moved.
     *
     * Implemented, and it shares `improveQualityTangential`'s kernel: both
     * entry points delegate to `relaxTangential` below, because the
     * reference's two functions (`mesh_quality.py::_tangential_relaxation` and
     * `mesh_solver.py::improve_mesh_quality_tangential`) are the same operator
     * written twice, one over raw arrays and one over a state. Its consumer is
     * `isotropicCleanup`, the second half of which it is.
     *
     * @param mesh   Surface; **owned** vertex positions moved in place.
     * @param passes Sweeps; a non-positive count is a no-op, as is `weight` 0.
     * @param weight Relaxation factor \f$w\f$.
     * @return Sweeps actually applied.
     *
     * @note MPI. Ghost positions are refreshed between passes by
     *       `mesh.haloExchange()` — see `relaxTangential`, which is where the
     *       requirement this note records is discharged.
     */
    int tangentialRelaxation( mesh_type& mesh, int passes, Real weight ) const
    {
        return relaxTangential( mesh, passes, weight );
    }

    /**
     * @brief The sliver-clearing post-pass: valence flips then relaxation.
     *
     * Port of mesh_quality.py::isotropic_cleanup (lines 146-167)
     *
     * `--isotropic-cleanup-flips` (3) passes of `valenceEqualizingFlips`, then
     * `--isotropic-cleanup-relax` (2) passes of `tangentialRelaxation` at
     * `--isotropic-cleanup-weight` (0.4).
     *
     * **`--no-isotropic-cleanup` is checked here**, before anything collective:
     * with `CleanupParams::enabled` false the call returns at once and the mesh
     * is untouched. The reference has the switch at its two call sites
     * (`run_adaptive_mesh_bubble.py:1452`, `:1491`); `Solver::advanceOneStep`
     * gates both of its calls on the same flag, which is the struct this class
     * holds, so the check here only makes the switch testable on its own.
     *
     * Vertex count unchanged and shape preserved, so the caller keeps its
     * per-vertex fields as they are. It **does** change face areas and
     * curvature, and **it does not re-base the AMR reference state**: the
     * reference's callers build a fresh `MeshPotentialZModelState` on the
     * returned arrays, which re-seeds the reference area and curvature and
     * re-centres the potential (`mesh_solver.py:138-163`). That rebuild is a
     * property of the call site, so `Solver::advanceOneStep` does it after
     * this call (`AdaptiveMesh::resetReferenceState`, then
     * `SurfaceState::centerPotential`); this class does not depend on the AMR
     * header.
     *
     * The driver follows it with the volume projection
     * (`run_adaptive_mesh_bubble.py:1465-1468, 1513-1516`): the flips change
     * the volume by their quads' tetrahedra and the tangential relaxation
     * preserves it only to first order.
     *
     * @pre None beyond a closed surface: `valenceEqualizingFlips` exchanges
     *      the halo on entry, and the relaxation exchanges between passes and
     *      once at the end, so the mesh is halo-consistent on return.
     *
     * @note MPI. Collective when enabled; the pass counts and the early stop
     *       are parameters or global counts.
     */
    void isotropicCleanup( mesh_type& mesh ) const
    {
        if ( !_params.enabled )
            return;
        valenceEqualizingFlips( mesh, _params.flip_passes );
        relaxTangential( mesh, _params.relax_passes, _params.relax_weight );
    }

    /**
     * @brief Quality-driven edge flips for the indicator-AMR path.
     *
     * Port of mesh_solver.py::improve_mesh_connectivity_by_edge_flips
     * (lines 1704-1772)
     *
     * Distinct from `valenceEqualizingFlips`: this one accepts a flip iff
     * \f$\min(q_{\text{new}}) > \min(q_{\text{old}})(1+g)\f$, the criterion
     * `DynamicRemesh::flipEdgesForQuality` uses, but with **no** quality floor
     * (`:1755-1758`). Called from the `--no-dynamic-remesh` refinement path,
     * after a refine pass that marked a face (`--flip-passes`, default 0).
     *
     * Up to `max_passes` sweeps of `qualityFlipSweep`, each with a fresh
     * touched set, stopping after a sweep that flips nothing (`:1720-1768`).
     * `max_passes <= 0` returns 0 **before anything collective**
     * (`:1713-1715`), so `--flip-passes 0` mutates nothing.
     *
     * **It does not re-base the AMR reference state.** The reference's call
     * site passes `reset_reference=(smooth_iters == 0)`
     * (`run_adaptive_mesh_bubble.py:1440-1445`), and `_state_with_faces`
     * re-seeds the reference area and curvature only when that is set and a
     * flip happened (`:1770-1772`). The re-base is a property of the call site,
     * so `Solver::advanceOneStep` calls `AdaptiveMesh::resetReferenceState`
     * itself under exactly that condition; this class does not depend on the
     * AMR header. The tangential pass that follows is handed
     * `reset_reference=False` at both of the reference's call sites (`:1449`,
     * `:1562`), so `improveQualityTangential` never re-bases.
     *
     * @pre Vertex positions are halo-consistent, as `qualityFlipSweep`
     *      requires. `AdaptiveMesh::refine` ends with a `haloExchange()`.
     *
     * @param mesh       Surface, connectivity edited in place.
     * @param max_passes `--flip-passes`; a non-positive count is a no-op.
     * @param min_gain   The gain \f$g\f$.
     * @param[out] sweeps When non-null, one `FlipSweepStats` is appended per
     *             sweep run, with its rounds and Tessera's verdict counts.
     * @return Total flips accepted, globally.
     *
     * @note MPI. Collective whenever `max_passes > 0`; the early stop is on a
     *       global count.
     */
    int improveConnectivityByFlips(
        mesh_type& mesh, int max_passes, Real min_gain,
        std::vector<FlipSweepStats>* sweeps = nullptr ) const
    {
        const int passes = ( max_passes > 0 ) ? max_passes : 0;
        if ( passes == 0 )
            return 0;

        QualityFlipCriterion criterion;
        criterion.has_floor = false;
        criterion.min_gain = min_gain;

        long long total = 0;
        for ( int pass = 0; pass < passes; ++pass )
        {
            const FlipSweepStats s = qualityFlipSweep( mesh, criterion );
            if ( sweeps != nullptr )
                sweeps->push_back( s );
            total += s.accepted;
            if ( s.accepted == 0 )
                break;
        }
        return static_cast<int>( total );
    }

    /**
     * @brief Mild tangential relaxation used by the indicator-AMR path.
     *
     * Port of mesh_solver.py::improve_mesh_quality_tangential
     * (lines 1775-1831)
     *
     * The same operator as `tangentialRelaxation`, reached from a different
     * place: after an indicator-driven refine (`--smooth-iters`,
     * `--smooth-relaxation`) and from the periodic `--redistribute-every`
     * sweep.
     *
     * The reference notes it "targets element quality while avoiding deliberate
     * normal smoothing of the interface geometry" — the same tangential
     * constraint, stated for the same reason.
     *
     * One kernel, two entry points: this is `relaxTangential` below,
     * as is `tangentialRelaxation`. Called from `Solver::advanceOneStep` at both
     * of the reference's call sites — after an indicator-driven refine
     * (`--smooth-iters`, `--smooth-relaxation`) and from the periodic
     * `--redistribute-every` sweep. **It does not re-base the AMR reference
     * state**: both call sites pass `reset_reference=False`, so re-basing here
     * would change every subsequent refinement decision.
     *
     * @param mesh       Surface; **owned** vertex positions moved in place.
     * @param iterations `--smooth-iters`; a non-positive count is a no-op, as is
     *                   a zero `relaxation` (`mesh_solver.py:1793-1794`).
     * @param relaxation `--smooth-relaxation`.
     */
    void improveQualityTangential( mesh_type& mesh, int iterations,
                                   Real relaxation ) const
    {
        relaxTangential( mesh, iterations, relaxation );
    }

  private:
    /**
     * @brief The one tangential-relaxation sweep loop.
     *
     * Port of mesh_solver.py::improve_mesh_quality_tangential (lines 1792-1812)
     * and mesh_quality.py::_tangential_relaxation (lines 108-115)
     *
     * \f[
     *   x_v \mathrel{+}= w\,P_v\big(\bar x_v - x_v\big), \qquad
     *   P_v u = u - (u\cdot\hat n_v)\hat n_v,
     * \f]
     * with \f$\bar x_v\f$ the mean over the **unique** neighbour set. That
     * offset is exactly `SurfaceOperators::graphLaplacianVector` over
     * `mesh.vertexOneRing()` and the projection is exactly
     * `SurfaceOperators::projectTangent` — both validated (task T2b) against this
     * same reference on this same mesh (`max` of the raw offset
     * `1.2663750374617e-2`, agreeing to `4e-17`), so no new stencil is written
     * here and a failure localizes to the projection rather than to the
     * neighbourhood.
     *
     * **THE SWEEP IS JACOBI, NOT GAUSS-SEIDEL.** The reference builds the whole
     * `displacement` array from `vertices` and only then adds it
     * (`mesh_solver.py:1804-1812`), so one sweep is independent of vertex order
     * and therefore of the partition. Reading it as Gauss-Seidel — updating a
     * position and letting the next vertex see it — is a faithful-*looking*
     * transcription that is partition dependent; the same trap T4b records for
     * the gradation sweep.
     *
     * **Two things the distributed form needs that the serial reference does
     * not.** The vertex normals are recomputed **every sweep** (the surface
     * parameterization has moved, and the reference recomputes them too), and
     * the positions are halo-exchanged **between** sweeps — `mesh.haloExchange()`,
     * since positions live in the mesh; `haloExchangeVertexView` is for
     * views held outside it and is not needed here. Without the exchange a
     * boundary vertex relaxes against stale neighbours and the seam moves with
     * the rank count. The exchange at the top of the loop also covers whatever
     * edited the mesh just before the call, and the one after it leaves the mesh
     * halo-consistent for the volume projection that follows.
     *
     * Only **owned** rows of `positions()` are written (risk R9): a ghost is an
     * owned vertex somewhere else and its own rank moves it. `displacement` is
     * allocated over the **whole local range** because `projectTangent` takes
     * its extent from `vertex_normal`, which is local-ranged — T2d's decision 4,
     * both halves of it.
     *
     * The `reference_area` / `reference_curvature` re-base at
     * `mesh_solver.py:1814-1818` is **deliberately not done**: it is gated on
     * `reset_reference`, and both of the reference's call sites pass `False`.
     *
     * @return Sweeps applied — 0 when the pass is configured off, which is the
     *         reference's own early return.
     *
     * @note MPI. Collective: `haloExchange()` and `vertexOneRing()` both are,
     *       and the sweep count is a parameter identical on every rank, so every
     *       rank reaches each of them the same number of times.
     */
    int relaxTangential( mesh_type& mesh, int iterations,
                         Real relaxation ) const
    {
        // `iterations = max(int(iterations), 0)`, then the two-clause early
        // return (mesh_solver.py:1792-1794). `relaxation == 0` is compared
        // exactly, as the Python does.
        const int sweeps = ( iterations > 0 ) ? iterations : 0;
        if ( sweeps == 0 || relaxation == Real( 0 ) )
            return 0;

        using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
        // TODO(types): templated pending Tessera/Canopy interface; collapse to a
        // concrete type once known.
        using vector_view = Kokkos::View<Real* [3], device_type>;

        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        const Real w = relaxation;

        for ( int sweep = 0; sweep < sweeps; ++sweep )
        {
            mesh.haloExchange();

            const int nv = mesh.totalVertexCount();
            const int n_owned = mesh.ownedVertexCount();
            auto pos = mesh.positions();

            geometry.compute( pos, nv, mesh.faceVertices() );

            vector_view displacement( "beatnik_tangential_displacement", nv );
            SurfaceOperators::graphLaplacianVector( mesh.vertexOneRing(), pos,
                                                   displacement );
            SurfaceOperators::projectTangent( displacement,
                                              geometry.vertex_normal );

            Kokkos::parallel_for(
                "beatnik_tangential_relaxation_apply",
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_owned ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                        pos( i, d ) += w * displacement( i, d );
                } );
            Kokkos::fence();
        }

        // Leave the ghosts consistent with the moved owners, so the caller's
        // next geometry computation (the volume projection, at both call sites)
        // does not read stale positions.
        mesh.haloExchange();
        return sweeps;
    }

    CleanupParams _params;
};

} // namespace Beatnik

#endif // BEATNIK_MESHQUALITY_HPP
