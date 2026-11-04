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
 * @file Beatnik_Test_MeshEdits.cpp
 * @brief `unit`-tier test of the three `SurfaceMesh` edit adapters over
 *        Tessera: `collapseEdges`, `flipEdges` and `compact`.
 *
 * THIS IS T4d1's EXIT CRITERION, MECHANIZED (`tasks/milestone1.md`). Its rank
 * sweep is `scripts/tuolumne/t4d1_mesh_edits.flux`, ranks 1-6 in one job; the
 * tier registers it at one rank like every other member.
 *
 * WHAT IT ASSERTS
 * ---------------
 * Case 1, on the default (subdivision-2) icosphere, V 162 / E 480 / F 320:
 *
 *   1. **The policy is forwarded.** Six named, well-separated edges are
 *      collapsed under a policy whose `minQuality` exceeds every attainable
 *      quality. All six are counted in `rejected_quality` and the mesh is
 *      unchanged. The adapter has no other way to produce that verdict.
 *   2. **(a) The V/E/F identities.** The same six edges under a permissive
 *      policy are all accepted; the report's removed counts are `accepted`,
 *      `3 * accepted` and `2 * accepted`, and so are the differences of the
 *      global counts taken before and after. **(b)** `V - E + F == 2`.
 *   3. **Flips.** Eight named, well-separated edges are flipped under the
 *      permissive flip policy: all accepted, V/E/F unchanged, every owned face
 *      gid preserved.
 *   4. **(c) Gid preservation across `compact()`.** Nothing is tombstoned
 *      (collapse compacts itself), so nothing is removed, the gid space is
 *      unchanged, and each rank's owned face gid set is identical before and
 *      after. The gid space is first asserted to exceed the live entity count
 *      — the collapse left holes in it — so a `compact()` that renumbered
 *      would change both and fail.
 *
 * Case 2, **(d) the failure direction**, on a fresh icosphere. Two spoke flips
 * bring a valence-5 vertex `v` to valence 3, with ring `{x, y, z}`. The ring
 * edge `(x, y)` then violates the link condition: its endpoints share the
 * neighbours `v` and `z` and the far opposite corner `w`, three where a
 * collapsible edge has two. The test asserts that premise from the gathered
 * one-rings before relying on it. Marking `(x, y)` alone is rejected into
 * `rejected_link_condition` with `accepted == 0` and the mesh bit-unchanged
 * in its counts and face gids. Marking it together with a valid antipodal
 * edge accepts exactly the valid one, and the identities hold.
 *
 * NAMING EDGES WITHOUT GIDS
 * -------------------------
 * Gids are not on the adapter's public surface, so an edge is named
 * geometrically: "the owned edge whose midpoint direction has the largest dot
 * product with a fixed, deliberately generic unit vector". Owned edges
 * partition the mesh and positions are bit-identical at every rank count, so
 * one global `MPI_MAX` picks the same edge everywhere. Each selection asserts
 * that exactly one edge was marked globally.
 *
 * Every check is a global quantity, so one set of literals holds at every rank
 * count. `main` reduces the per-rank exit codes with `MPI_Allreduce(MPI_MAX)`
 * (`Beatnik_TestAssert.hpp`). The test writes no files.
 */

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace
{

using Beatnik::Real;
using Point = std::array<double, 3>;

// The subdivision-2 icosphere: `Params::icosphere_subdivisions`' default.
constexpr int kSubdivisions = 2;
constexpr long long kV0 = 162;
constexpr long long kE0 = 480;
constexpr long long kF0 = 320;

// Six collapse targets near +-x, +-y, +-z (pairwise >= ~85 degrees apart, far
// beyond the two-ring a collapse candidate names) and eight flip targets near
// the cube diagonals (>= ~50 degrees from every collapse target and ~70 from
// each other). The perturbations make every argmax tie-free.
const double kCollapseDirs[6][3] = {
    { 1.0, 0.11, 0.07 },  { -1.0, 0.05, -0.13 }, { 0.09, 1.0, 0.03 },
    { -0.04, -1.0, 0.12 }, { 0.06, -0.08, 1.0 },  { 0.13, 0.02, -1.0 } };
const double kFlipPerturb[3] = { 0.031, -0.017, 0.023 };

// Case 2's choices: the valence-5 vertex, the spokes to flip, the ring edge.
const double kVertexDir[3] = { 0.21, 0.37, 0.9 };
const double kSpokeDir[3] = { 0.53, -0.29, 0.11 };
const double kRingEdgeDir[3] = { -0.17, 0.44, 0.61 };

Point unit( const double* d )
{
    const double n = std::sqrt( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
    return { d[0] / n, d[1] / n, d[2] / n };
}

double dot( const Point& a, const Point& b )
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point midDir( const Point& a, const Point& b )
{
    const double m[3] = { 0.5 * ( a[0] + b[0] ), 0.5 * ( a[1] + b[1] ),
                          0.5 * ( a[2] + b[2] ) };
    return unit( m );
}

//---------------------------------------------------------------------------//
/// A host copy of what the selections need: every local vertex position, the
/// owned edges' local endpoints, and the one-ring CSR. Re-taken after every
/// edit, because every edit invalidates the accessors it is built from.
struct HostMesh
{
    std::vector<Point> pos;
    std::vector<std::array<int, 2>> owned_edges;
    std::vector<int> ring_offsets;
    std::vector<int> ring_neighbors;
    int owned_vertices = 0;
};

template <class ExecSpace, class MeshType>
HostMesh snapshot( MeshType& mesh )
{
    using mem_space = typename MeshType::memory_space;
    HostMesh h;
    const int nv = mesh.totalVertexCount();
    h.owned_vertices = mesh.ownedVertexCount();

    Kokkos::View<double* [3], mem_space> pos_d( "beatnik_t4d1_pos", nv );
    auto pos = mesh.positions();
    Kokkos::parallel_for(
        "beatnik_t4d1_copy_positions", Kokkos::RangePolicy<ExecSpace>( 0, nv ),
        KOKKOS_LAMBDA( const int i ) {
            for ( int d = 0; d < 3; ++d )
                pos_d( i, d ) = pos( i, d );
        } );
    Kokkos::fence();
    auto pos_h =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pos_d );
    h.pos.resize( nv );
    for ( int i = 0; i < nv; ++i )
        h.pos[i] = { pos_h( i, 0 ), pos_h( i, 1 ), pos_h( i, 2 ) };

    auto ev = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   mesh.edgeVertices() );
    const int ne = mesh.ownedEdgeCount();
    h.owned_edges.resize( ne );
    for ( int e = 0; e < ne; ++e )
        h.owned_edges[e] = { ev( e, 0 ), ev( e, 1 ) };

    auto csr = mesh.vertexOneRing();
    auto off = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    csr.offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    csr.neighbors );
    h.ring_offsets.assign( off.data(), off.data() + off.extent( 0 ) );
    h.ring_neighbors.assign( nbr.data(), nbr.data() + nbr.extent( 0 ) );
    return h;
}

long long globalCount( MPI_Comm comm, const std::vector<char>& mask )
{
    long long local = 0;
    for ( char m : mask )
        local += ( m != 0 );
    long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return global;
}

bool globalAll( MPI_Comm comm, bool local )
{
    int in = local ? 1 : 0;
    int out = 0;
    MPI_Allreduce( &in, &out, 1, MPI_INT, MPI_MIN, comm );
    return out == 1;
}

//---------------------------------------------------------------------------//
/// Mark the one owned edge, over all ranks, with the largest `score`. An edge
/// is ineligible when `score` returns -infinity, and so is one whose local
/// endpoints are not held.
std::vector<char>
markArgmax( MPI_Comm comm, const HostMesh& h,
            const std::function<double( const Point&, const Point& )>& score )
{
    const double ninf = -std::numeric_limits<double>::infinity();
    std::vector<double> s( h.owned_edges.size(), ninf );
    double local_best = ninf;
    for ( std::size_t e = 0; e < h.owned_edges.size(); ++e )
    {
        const int a = h.owned_edges[e][0];
        const int b = h.owned_edges[e][1];
        if ( a < 0 || b < 0 )
            continue;
        s[e] = score( h.pos[a], h.pos[b] );
        local_best = std::max( local_best, s[e] );
    }
    double best = ninf;
    MPI_Allreduce( &local_best, &best, 1, MPI_DOUBLE, MPI_MAX, comm );
    std::vector<char> mask( h.owned_edges.size(), 0 );
    if ( best == ninf )
        return mask;
    for ( std::size_t e = 0; e < s.size(); ++e )
        if ( s[e] == best )
            mask[e] = 1;
    return mask;
}

/// The edge whose midpoint direction is closest to `dir`.
std::vector<char> markNearest( MPI_Comm comm, const HostMesh& h,
                               const double* dir )
{
    const Point u = unit( dir );
    return markArgmax( comm, h, [u]( const Point& a, const Point& b )
                       { return dot( midDir( a, b ), u ); } );
}

/// The owned edge joining the two given points exactly.
std::vector<char> markBetween( const HostMesh& h, const Point& p,
                               const Point& q )
{
    std::vector<char> mask( h.owned_edges.size(), 0 );
    for ( std::size_t e = 0; e < h.owned_edges.size(); ++e )
    {
        const int a = h.owned_edges[e][0];
        const int b = h.owned_edges[e][1];
        if ( a < 0 || b < 0 )
            continue;
        if ( ( h.pos[a] == p && h.pos[b] == q ) ||
             ( h.pos[a] == q && h.pos[b] == p ) )
            mask[e] = 1;
    }
    return mask;
}

void orMask( std::vector<char>& into, const std::vector<char>& from )
{
    for ( std::size_t e = 0; e < into.size(); ++e )
        into[e] = static_cast<char>( into[e] || from[e] );
}

/// The one-ring of the vertex at exactly `p`, as positions, gathered from its
/// owner to every rank. Empty if no rank owns a vertex there.
std::vector<Point> gatherRing( MPI_Comm comm, const HostMesh& h,
                               const Point& p )
{
    std::vector<double> mine;
    for ( int i = 0; i < h.owned_vertices; ++i )
    {
        if ( h.pos[i] != p )
            continue;
        for ( int k = h.ring_offsets[i]; k < h.ring_offsets[i + 1]; ++k )
            for ( int d = 0; d < 3; ++d )
                mine.push_back( h.pos[h.ring_neighbors[k]][d] );
    }
    int size = 1;
    MPI_Comm_size( comm, &size );
    int n = static_cast<int>( mine.size() );
    std::vector<int> counts( size ), displs( size, 0 );
    MPI_Allgather( &n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );
    for ( int r = 1; r < size; ++r )
        displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<double> all( displs[size - 1] + counts[size - 1] );
    MPI_Allgatherv( mine.data(), n, MPI_DOUBLE, all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, comm );
    std::vector<Point> ring;
    for ( std::size_t k = 0; k + 2 < all.size(); k += 3 )
        ring.push_back( { all[k], all[k + 1], all[k + 2] } );
    return ring;
}

/// The owned vertex of valence `valence` maximizing `dot(p, dir)`, over all
/// ranks, broadcast to every rank.
Point argmaxVertex( MPI_Comm comm, const HostMesh& h, int valence,
                    const double* dir )
{
    const Point u = unit( dir );
    struct
    {
        double score;
        int rank;
    } local{ -std::numeric_limits<double>::infinity(), 0 }, best{};
    MPI_Comm_rank( comm, &local.rank );
    Point mine{ 0.0, 0.0, 0.0 };
    for ( int i = 0; i < h.owned_vertices; ++i )
    {
        if ( h.ring_offsets[i + 1] - h.ring_offsets[i] != valence )
            continue;
        const double s = dot( h.pos[i], u );
        if ( s > local.score )
        {
            local.score = s;
            mine = h.pos[i];
        }
    }
    MPI_Allreduce( &local, &best, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm );
    MPI_Bcast( mine.data(), 3, MPI_DOUBLE, best.rank, comm );
    return mine;
}

/// Each rank's owned face gids, sorted.
template <class MeshType>
std::vector<std::uint64_t> sortedFaceGids( MeshType& mesh )
{
    auto g = mesh.ownedFaceGids();
    std::sort( g.begin(), g.end() );
    return g;
}

std::string reportLine( const char* what, const Beatnik::MeshEditReport& r )
{
    return std::string( what ) + ": requested=" + std::to_string( r.requested ) +
           " accepted=" + std::to_string( r.accepted ) +
           " rejectedBoundary=" + std::to_string( r.rejected_boundary ) +
           " rejectedLinkCondition=" +
           std::to_string( r.rejected_link_condition ) +
           " rejectedNormalFlip=" + std::to_string( r.rejected_normal_flip ) +
           " rejectedQuality=" + std::to_string( r.rejected_quality ) +
           " rejectedDuplicateEdge=" +
           std::to_string( r.rejected_duplicate_edge ) +
           " rejectedGeometric=" + std::to_string( r.rejected_geometric ) +
           " rejectedConflict=" + std::to_string( r.rejected_conflict ) +
           " removed V/E/F=" + std::to_string( r.vertices_removed ) + "/" +
           std::to_string( r.edges_removed ) + "/" +
           std::to_string( r.faces_removed );
}

template <class MeshType>
std::string vefLine( const char* when, MeshType& mesh )
{
    return std::string( when ) +
           ": V=" + std::to_string( mesh.globalVertexCount() ) +
           " E=" + std::to_string( mesh.globalEdgeCount() ) +
           " F=" + std::to_string( mesh.globalFaceCount() ) +
           " chi=" + std::to_string( mesh.globalEulerCharacteristic() );
}

//---------------------------------------------------------------------------//
// Case 1 -- collapse, flip, compact on the default icosphere.
//---------------------------------------------------------------------------//
template <class ExecSpace, class MeshType>
void caseCollapseFlipCompact( Beatnik::Test::Recorder& rec )
{
    MeshType mesh( MPI_COMM_WORLD );
    const MPI_Comm comm = mesh.comm();
    const Real center[3] = { 0.0, 0.0, 0.0 };
    mesh.generateIcosphere( kSubdivisions, 1.0, center );

    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
    rec.note( vefLine( "case 1 start", mesh ) );

    const double pi = std::acos( -1.0 );
    typename MeshType::collapse_policy_type permissive_collapse;
    permissive_collapse.maxNormalRotation = pi;
    permissive_collapse.minQuality = 0.0;
    typename MeshType::flip_policy_type permissive_flip;
    permissive_flip.maxNormalDeviation = pi;
    permissive_flip.minQuality = 0.0;

    // The six named collapse edges. Premise: six distinct edges marked.
    HostMesh h = snapshot<ExecSpace>( mesh );
    std::vector<char> collapse_mask( h.owned_edges.size(), 0 );
    for ( const auto& d : kCollapseDirs )
    {
        const auto one = markNearest( comm, h, d );
        BEATNIK_CHECK_EQ( rec, globalCount( comm, one ), 1 );
        orMask( collapse_mask, one );
    }
    BEATNIK_CHECK_EQ( rec, globalCount( comm, collapse_mask ), 6 );

    // 1. The policy is forwarded: a minQuality above every attainable quality
    //    rejects all six into rejected_quality, and nothing changes.
    {
        typename MeshType::collapse_policy_type strict = permissive_collapse;
        strict.minQuality = 1.1;
        const auto gids_before = sortedFaceGids( mesh );
        const auto r = mesh.collapseEdges( collapse_mask, strict );
        rec.note( reportLine( "case 1 collapse, strict policy", r ) );
        BEATNIK_CHECK_EQ( rec, r.requested, 6 );
        BEATNIK_CHECK_EQ( rec, r.accepted, 0 );
        BEATNIK_CHECK_EQ( rec, r.rejected_quality, 6 );
        BEATNIK_CHECK_EQ( rec, r.rejected_boundary + r.rejected_link_condition +
                                   r.rejected_normal_flip + r.rejected_conflict,
                          0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
    }

    // 2. (a) and (b): the same six under the permissive policy.
    {
        const long long v0 = mesh.globalVertexCount();
        const long long e0 = mesh.globalEdgeCount();
        const long long f0 = mesh.globalFaceCount();
        const auto r = mesh.collapseEdges( collapse_mask, permissive_collapse );
        rec.note( reportLine( "case 1 collapse, permissive policy", r ) );
        rec.note( vefLine( "case 1 after collapse", mesh ) );
        BEATNIK_CHECK_EQ( rec, r.requested, 6 );
        BEATNIK_CHECK_EQ( rec, r.accepted, 6 );
        BEATNIK_CHECK_EQ( rec,
                          r.rejected_boundary + r.rejected_link_condition +
                              r.rejected_normal_flip + r.rejected_quality +
                              r.rejected_conflict,
                          0 );
        // (a) The three identities, on the report and on the global counts.
        BEATNIK_CHECK_EQ( rec, r.vertices_removed, r.accepted );
        BEATNIK_CHECK_EQ( rec, r.edges_removed, 3 * r.accepted );
        BEATNIK_CHECK_EQ( rec, r.faces_removed, 2 * r.accepted );
        BEATNIK_CHECK_EQ( rec, v0 - mesh.globalVertexCount(), r.accepted );
        BEATNIK_CHECK_EQ( rec, e0 - mesh.globalEdgeCount(), 3 * r.accepted );
        BEATNIK_CHECK_EQ( rec, f0 - mesh.globalFaceCount(), 2 * r.accepted );
        // (b) Still a closed surface of genus 0.
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), MeshType::halo_depth );
    }

    // 3. Eight named flips near the cube diagonals.
    {
        h = snapshot<ExecSpace>( mesh );
        std::vector<char> flip_mask( h.owned_edges.size(), 0 );
        for ( int s = 0; s < 8; ++s )
        {
            const double d[3] = { ( s & 1 ? -1.0 : 1.0 ) + kFlipPerturb[0],
                                  ( s & 2 ? -1.0 : 1.0 ) + kFlipPerturb[1],
                                  ( s & 4 ? -1.0 : 1.0 ) + kFlipPerturb[2] };
            const auto one = markNearest( comm, h, d );
            BEATNIK_CHECK_EQ( rec, globalCount( comm, one ), 1 );
            orMask( flip_mask, one );
        }
        BEATNIK_CHECK_EQ( rec, globalCount( comm, flip_mask ), 8 );

        const long long v0 = mesh.globalVertexCount();
        const long long e0 = mesh.globalEdgeCount();
        const long long f0 = mesh.globalFaceCount();
        const auto gids_before = sortedFaceGids( mesh );
        const auto r = mesh.flipEdges( flip_mask, permissive_flip );
        rec.note( reportLine( "case 1 flip, permissive policy", r ) );
        BEATNIK_CHECK_EQ( rec, r.requested, 8 );
        BEATNIK_CHECK_EQ( rec, r.accepted, 8 );
        BEATNIK_CHECK_EQ( rec, r.rejected_boundary + r.rejected_duplicate_edge +
                                   r.rejected_geometric + r.rejected_conflict,
                          0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), v0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), e0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), f0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
    }

    // 4. (c) compact() preserves gids.
    {
        const long long v0 = mesh.globalVertexCount();
        const long long e0 = mesh.globalEdgeCount();
        const long long f0 = mesh.globalFaceCount();
        const auto gids_before = sortedFaceGids( mesh );
        const auto r = mesh.compact();
        rec.note( "case 1 compact: removed V/E/F=" +
                  std::to_string( r.vertices_removed ) + "/" +
                  std::to_string( r.edges_removed ) + "/" +
                  std::to_string( r.faces_removed ) +
                  " gidSpace before/after=" +
                  std::to_string( r.gid_space_before ) + "/" +
                  std::to_string( r.gid_space_after ) +
                  " live V+E+F=" + std::to_string( v0 + e0 + f0 ) );
        // Premise: the collapse left holes in the gid space, so renumbering
        // would be observable.
        BEATNIK_CHECK_TRUE( rec, r.gid_space_before > v0 + e0 + f0 );
        BEATNIK_CHECK_EQ( rec, r.vertices_removed, 0 );
        BEATNIK_CHECK_EQ( rec, r.edges_removed, 0 );
        BEATNIK_CHECK_EQ( rec, r.faces_removed, 0 );
        BEATNIK_CHECK_EQ( rec, r.gid_space_after, r.gid_space_before );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), v0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), e0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), f0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
        BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), MeshType::halo_depth );
    }
}

//---------------------------------------------------------------------------//
// Case 2 -- (d) a link-condition-violating edge is rejected, not applied.
//---------------------------------------------------------------------------//
template <class ExecSpace, class MeshType>
void caseLinkCondition( Beatnik::Test::Recorder& rec )
{
    MeshType mesh( MPI_COMM_WORLD );
    const MPI_Comm comm = mesh.comm();
    const Real center[3] = { 0.0, 0.0, 0.0 };
    mesh.generateIcosphere( kSubdivisions, 1.0, center );

    const double pi = std::acos( -1.0 );
    typename MeshType::flip_policy_type permissive_flip;
    permissive_flip.maxNormalDeviation = pi;
    permissive_flip.minQuality = 0.0;

    HostMesh h = snapshot<ExecSpace>( mesh );
    const Point v = argmaxVertex( comm, h, 5, kVertexDir );
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( gatherRing( comm, h, v ).size() ),
                      5 );

    // Two spoke flips, one per call, each the spoke whose far endpoint lies
    // furthest along kSpokeDir. Each lowers v's valence by one.
    const Point spoke_dir = unit( kSpokeDir );
    for ( int k = 0; k < 2; ++k )
    {
        h = snapshot<ExecSpace>( mesh );
        const auto mask = markArgmax(
            comm, h,
            [v, spoke_dir]( const Point& a, const Point& b )
            {
                if ( a == v )
                    return dot( b, spoke_dir );
                if ( b == v )
                    return dot( a, spoke_dir );
                return -std::numeric_limits<double>::infinity();
            } );
        BEATNIK_CHECK_EQ( rec, globalCount( comm, mask ), 1 );
        const auto r = mesh.flipEdges( mask, permissive_flip );
        rec.note( reportLine( "case 2 spoke flip", r ) );
        BEATNIK_CHECK_EQ( rec, r.accepted, 1 );
    }
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

    // Premise: v has valence 3, and the chosen ring edge (x, y) has endpoints
    // sharing three neighbours -- v, the third ring vertex z, and the far
    // opposite corner w -- where a collapsible edge's share exactly two.
    h = snapshot<ExecSpace>( mesh );
    const auto ring = gatherRing( comm, h, v );
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( ring.size() ), 3 );
    if ( ring.size() != 3 )
    {
        rec.fail( "case 2 fixture: v did not reach valence 3; the "
                  "link-condition case cannot be built" );
        return;
    }
    const Point ring_dir = unit( kRingEdgeDir );
    int bi = 0, bj = 1;
    double best = -std::numeric_limits<double>::infinity();
    for ( int i = 0; i < 3; ++i )
        for ( int j = i + 1; j < 3; ++j )
        {
            const double s = dot( midDir( ring[i], ring[j] ), ring_dir );
            if ( s > best )
            {
                best = s;
                bi = i;
                bj = j;
            }
        }
    const Point x = ring[bi];
    const Point y = ring[bj];
    const auto ring_x = gatherRing( comm, h, x );
    const auto ring_y = gatherRing( comm, h, y );
    long long common = 0;
    for ( const auto& p : ring_x )
        common += std::count( ring_y.begin(), ring_y.end(), p );
    rec.note( "case 2 fixture: valence(v)=" + std::to_string( ring.size() ) +
              " valence(x)=" + std::to_string( ring_x.size() ) +
              " valence(y)=" + std::to_string( ring_y.size() ) +
              " |link(x) & link(y)|=" + std::to_string( common ) );
    BEATNIK_CHECK_EQ( rec, common, 3 );

    const auto bad = markBetween( h, x, y );
    BEATNIK_CHECK_EQ( rec, globalCount( comm, bad ), 1 );

    // The bad edge alone, at the default policy: rejected on the link
    // condition, and the mesh is untouched.
    {
        const long long v0 = mesh.globalVertexCount();
        const long long e0 = mesh.globalEdgeCount();
        const long long f0 = mesh.globalFaceCount();
        const auto gids_before = sortedFaceGids( mesh );
        const auto r = mesh.collapseEdges( bad );
        rec.note( reportLine( "case 2 collapse, bad edge only", r ) );
        rec.note( vefLine( "case 2 after bad-only collapse", mesh ) );
        BEATNIK_CHECK_EQ( rec, r.requested, 1 );
        BEATNIK_CHECK_EQ( rec, r.accepted, 0 );
        BEATNIK_CHECK_EQ( rec, r.rejected_link_condition, 1 );
        BEATNIK_CHECK_EQ( rec, r.rejected_boundary + r.rejected_normal_flip +
                                   r.rejected_quality + r.rejected_conflict,
                          0 );
        BEATNIK_CHECK_EQ( rec, r.vertices_removed + r.edges_removed +
                                   r.faces_removed,
                          0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), v0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), e0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), f0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
    }

    // The bad edge beside a valid antipodal one: only the valid one is
    // accepted, and the identities hold.
    {
        typename MeshType::collapse_policy_type permissive_collapse;
        permissive_collapse.maxNormalRotation = pi;
        permissive_collapse.minQuality = 0.0;

        h = snapshot<ExecSpace>( mesh );
        const double anti[3] = { -kVertexDir[0], -kVertexDir[1],
                                 -kVertexDir[2] };
        auto mask = markNearest( comm, h, anti );
        BEATNIK_CHECK_EQ( rec, globalCount( comm, mask ), 1 );
        orMask( mask, markBetween( h, x, y ) );
        BEATNIK_CHECK_EQ( rec, globalCount( comm, mask ), 2 );

        const long long v0 = mesh.globalVertexCount();
        const long long e0 = mesh.globalEdgeCount();
        const long long f0 = mesh.globalFaceCount();
        const auto r = mesh.collapseEdges( mask, permissive_collapse );
        rec.note( reportLine( "case 2 collapse, bad + antipodal", r ) );
        rec.note( vefLine( "case 2 after mixed collapse", mesh ) );
        BEATNIK_CHECK_EQ( rec, r.requested, 2 );
        BEATNIK_CHECK_EQ( rec, r.accepted, 1 );
        BEATNIK_CHECK_EQ( rec, r.rejected_link_condition, 1 );
        BEATNIK_CHECK_EQ( rec, r.rejected_boundary + r.rejected_normal_flip +
                                   r.rejected_quality + r.rejected_conflict,
                          0 );
        BEATNIK_CHECK_EQ( rec, v0 - mesh.globalVertexCount(), 1 );
        BEATNIK_CHECK_EQ( rec, e0 - mesh.globalEdgeCount(), 3 );
        BEATNIK_CHECK_EQ( rec, f0 - mesh.globalFaceCount(), 2 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    }
}

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    int comm_size = 1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    caseCollapseFlipCompact<ExecSpace, mesh_type>( rec );
    caseLinkCondition<ExecSpace, mesh_type>( rec );
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_MeshEdits" );
        try
        {
            // One binary on the default execution space, no backend suffix --
            // the `unit` tier's convention; see tests/unit_tests/CMakeLists.txt.
            using ExecSpace = Kokkos::DefaultExecutionSpace;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
        }
        catch ( const std::exception& e )
        {
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();

    // ONE VERDICT ACROSS THE RANKS. Every rank reported, so the log names which
    // rank failed; see Beatnik_TestAssert.hpp for why the reduction is here.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
