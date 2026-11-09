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
 *        Tessera — `collapseEdges`, `flipEdges` and `compact` — and of the
 *        passes built on them: `DynamicRemesh::collapseShortEdges`,
 *        `DynamicRemesh::flipEdgesForQuality`,
 *        `MeshQuality::improveConnectivityByFlips`,
 *        `DynamicRemesh::tangentialSmooth`,
 *        `MeshQuality::valenceEqualizingFlips` and
 *        `MeshQuality::isotropicCleanup`.
 *
 * THIS IS T4d1's TO T4d5's EXIT CRITERIA, MECHANIZED
 * (`tasks/milestone1.md`).
 * Its rank sweep is `scripts/tuolumne/t4d1_mesh_edits.flux`, ranks 1-6 in one
 * job; the tier registers it at one rank like every other member.
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
 * Case 3, **T4d2**, each part on a fresh default icosphere whose vertices are
 * moved radially by a smooth function of position, so that no two collapse
 * ratios tie:
 *
 *   - **3a, the failure direction.** `collapse_factor = 0` returns 0 with no
 *     candidates, and V/E/F, the face gids, every local position and the mesh
 *     generation are unchanged — the path the split-only regression member
 *     takes.
 *   - **3b.** An artificially large `collapse_factor` at
 *     `max_collapses_per_pass = 0`, which is **unlimited**: the candidate and
 *     collapse counts are fixed literals, faces fall by `2 * collapses`, and
 *     `V - E + F == 2`.
 *   - **3c.** The same at a binding cap: exactly the cap's count is handed to
 *     Tessera, and the collapse count is a fixed literal.
 *   - **3d.** `remesh()`, splitting disabled by an unreachable split factor,
 *     collapses exactly 3b's count; at `collapse_factor = 0` it changes
 *     nothing.
 *
 * Case 4, **T4d3**, each part on a fresh default icosphere whose vertices are
 * moved tangentially by a smooth, high-frequency field and put back on the
 * sphere, which leaves faces below `--remesh-min-quality` (0.18) and edges
 * whose flip improves the worse face. Statistics are compared, never flip
 * sets (framework risk R7):
 *
 *   - **4a, the failure direction.** `flip_min_gain = kFlipsDisabledMinGain`
 *     (1e12) gives `flips == 0`, no `flipEdges` call, and V/E/F, the face gids,
 *     every local position and the generation unchanged — the path the T4b
 *     regression member takes.
 *   - **4b.** At the default gain (1e-3) and floor (0.18), one sweep flips a
 *     fixed count over more than one round, the global minimum quality rises
 *     or holds, the count of faces below 0.18 does not increase, V/E/F and
 *     the face gids are unchanged, and Tessera's counters partition
 *     `requested`.
 *   - **4c.** `remesh()` with split and collapse off reaches the flips through
 *     its quality-repair gate and flips 4b's count; at 1e12 it changes
 *     nothing.
 *   - **4d.** `improveConnectivityByFlips`: `max_passes = 0` returns 0 and
 *     changes nothing; three passes flip a fixed count over a fixed number of
 *     sweeps, and the minimum quality rises or holds.
 *
 * Case 5, **T4d4**, the guarded tangential smoothing:
 *
 *   - **5a, off.** `smoothing_iterations = 0` (the T4b regression member's
 *     value), and `relaxation` 0 and negative (the reference's
 *     `relaxation <= 0`), each return 0 with every local position
 *     bit-identical and the generation unchanged.
 *   - **5b, the healthy direction**, on the default icosphere at the
 *     default relaxation 0.04. Three single-iteration calls: each returns 1,
 *     its displacement satisfies `max|dx.n| <= 1e-13 max|dx|` against the
 *     normals it started from, and its minimum quality satisfies the guard's
 *     own bound `min_after + 1e-14 >= max(min_quality, 0.85 min_before)`.
 *     The minimum is printed and compared with the reference, never required
 *     to hold: the reference lowers it. The mean rises. One call at three
 *     iterations returns 3 and lands where the single calls did. Its relative
 *     volume change is at least 1e3 times smaller than that of one
 *     un-projected iteration, built here from the same public operators.
 *   - **5c, the rejecting direction.** At `min_quality = 0` and relaxation
 *     5.0 the trial loses more than 15 % of the minimum (asserted from the
 *     test's own unguarded trial first), so the `0.85` branch rejects it:
 *     `smooth_steps == 0` and every local position is bit-identical.
 *   - **5d, the pass loop.** On case 4's fixture the repair gate fires; at
 *     `smoothing_iterations = 0` with flips off `remesh()` changes nothing,
 *     and with both on it flips 4b's count and then smooths.
 *
 * Case 6, **T4d5**, the valence-equalizing flips and the isotropic cleanup,
 * on case 3's perturbed icosphere after one round of SharedFace flips in a
 * cap, which leaves a patch of vertices up to valence 10. Statistics are
 * compared, never flip sets (framework risk R7):
 *
 *   - **6a, off.** `isotropicCleanup` under `CleanupParams::enabled = false`
 *     (the `Solver`'s `--no-isotropic-cleanup`), and `valenceEqualizingFlips`
 *     at zero passes, leave every local position, the face gids and the
 *     generation bit-identical.
 *   - **6b, the valence passes.** Three passes flip fixed counts over fixed
 *     round counts, one pass taking more than one round. From each pass's
 *     gathered flips: no accepted flip has an endpoint among the corners of a
 *     flip accepted earlier in the pass or of another flip of its round. The
 *     count of vertices with valence != 6 falls to a fixed count; V/E/F and
 *     the Euler characteristic are unchanged. The volume change is printed,
 *     not bounded: a flip on a curved surface changes it.
 *   - **6c, the relaxation passes**, one per call at weight 0.4: each
 *     displacement satisfies `max|dx.n| <= 1e-13 max|dx|` against the normals
 *     it used. The two passes' relative volume change is at least 1e3 times
 *     smaller than that of one un-projected pass on the same flipped mesh.
 *   - **6d, on.** `isotropicCleanup` under `enabled = true` changes the
 *     positions and the generation and lands where 6b and 6c's separate calls
 *     did.
 *   - **6e, the early stop.** On a fixture whose second pass flips nothing,
 *     three requested passes run two.
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

#include <Beatnik_DynamicRemesh.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_MeshQuality.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Case 3's controls. On the unit sphere the curvature target
// sqrt(8 * 0.004 / kappa) exceeds the default h_max, so every vertex target is
// h_max = 0.05 and an edge is a candidate iff it is shorter than
// kCollapseFactor * 0.05 = 0.3, which is inside the subdivision-2 edge-length
// range: some edges qualify and some do not. The radial perturbation
// amplitude breaks the icosphere's length ties. kCap is below the candidate
// count, so the cap binds.
constexpr double kPerturbAmp = 0.02;
constexpr double kCollapseFactor = 6.0;
constexpr int kCap = 25;
// Measured, identical at ranks 1-6; see tasks/milestone1-progress-log.md
// `## T4d2`. Tessera's one independent set per call accepts only the
// candidates that are shortest in both endpoints' one-rings, which on this
// smoothly perturbed mesh is 4 of 240, and the same 4 survive the cap.
constexpr long long kCandidates = 240;
constexpr int kCollapsesUnlimited = 4;
constexpr int kCollapsesCapped = 4;

// Case 4's fixture: every vertex p moves by kTangentialAmp times the tangential
// part of (sin(12y+1), sin(15z+2), sin(18x+3)) and is renormalized onto the
// unit sphere. The field and amplitude were chosen with a Python model of the
// sweep (tasks/milestone1-progress-log.md `## T4d3`): it leaves faces below
// the 0.18 floor and no inverted face.
constexpr double kTangentialAmp = 0.1;
constexpr double kQualityFloor = 0.18;
// Measured, identical at ranks 1-6; see tasks/milestone1-progress-log.md
// `## T4d3`. The fixture's worst face is 0.0775 and three faces are below the
// floor; one sweep flips 44 edges over two rounds and leaves none below it.
// improveConnectivityByFlips' three sweeps flip 44, 10 and 2.
constexpr long long kBelowBefore = 3;
constexpr long long kBelowAfter = 0;
constexpr int kQualityFlips = 44;
constexpr int kQualityRounds = 2;
constexpr int kImprovePasses = 3;
constexpr int kImproveFlips = 56;
constexpr int kImproveSweeps = 3;

// Case 5's controls: the reference's default relaxation (0.04) at three
// iterations, and an overshooting relaxation for the guard's rejecting
// direction. At 5.0 a projected iteration on the default icosphere takes the
// minimum quality from 0.97727 to 0.62203 (ratio 0.636), well past the
// guard's 0.85 (Python model, tasks/milestone1-progress-log.md `## T4d4`);
// the test re-measures that premise itself.
constexpr double kSmoothRelaxation = 0.04;
constexpr int kSmoothIterations = 3;
constexpr double kRejectRelaxation = 5.0;
// Measured, identical at ranks 1-6; see tasks/milestone1-progress-log.md
// `## T4d4`: on case 4's fixture, `remesh()` flips 44 edges and then accepts
// its one smoothing iteration.
constexpr int kRemeshSmoothSteps = 1;
// The reference, `dynamic_remesh.py::tangential_smooth_vertices` with
// `smoothing_relaxation = 0.04`, `min_quality = 0.18`, on
// `mesh.icosphere_mesh( subdivisions=2, radius=1 )`: minimum and mean
// quality after one and after three iterations, and the relative volume
// change after three. Not to be edited to make a check pass.
constexpr double kPyMinQuality1 = 0.97725307555529173;
constexpr double kPyMeanQuality1 = 0.989142962939909;
constexpr double kPyMinQuality3 = 0.97721315828587241;
constexpr double kPyMeanQuality3 = 0.99019655118968153;
constexpr double kPyVolumeRel3 = -1.16636709257544879e-06;
// T4c's tolerances, for T4c's reasons (Beatnik_Test_TangentialRelaxation):
// 1e-12 relative against a reference summary scalar, 1e-8 for a volume
// change formed by cancellation, the a-priori tangency bound, and 1e-13
// absolute between two runs equal in exact arithmetic (the vertex normals
// are assembled with atomics, so not bitwise).
constexpr double kPyTolerance = 1.0e-12;
constexpr double kPyCancellationTolerance = 1.0e-8;
constexpr double kTangencyRatio = 1.0e-13;
constexpr double kIdenticalPositions = 1.0e-13;
// The exit criterion's volume separation: the guarded pass changes the
// volume at least this many times less than one un-projected iteration.
constexpr double kVolumeFactor = 1.0e3;

// Case 6's fixture: case 3's radially perturbed icosphere, then SharedFace
// flips of every owned edge whose midpoint direction lies within a cap about
// kScrambleDir, which pushes valence away from 6 inside the cap. One round in
// a 45-degree cap gives the main fixture; two rounds in a 60-degree cap give
// one whose second valence pass flips nothing. Chosen with a Python model of
// the scramble round, the valence pass and Tessera's EndpointInQuad round
// (tasks/milestone1-progress-log.md `## T4d5`), which predicted the literals
// below; the model's pass 0 takes three rounds.
const double kScrambleDir[3] = { 0.3, 0.5, 0.8 };
constexpr double kScrambleCapDeg = 45.0;
constexpr int kScrambleRounds = 1;
constexpr long long kScrambleFlips = 14;
constexpr long long kIrregularBefore = 35;
constexpr int kMaxValenceBefore = 10;
constexpr int kCleanupFlipPasses = 3;
constexpr int kValencePasses = 3;
const long long kValencePassFlips[kValencePasses] = { 6, 3, 1 };
const int kValencePassRounds[kValencePasses] = { 3, 1, 1 };
constexpr long long kIrregularAfter = 20;
constexpr double kEarlyStopCapDeg = 60.0;
constexpr int kEarlyStopScrambleRounds = 2;
constexpr long long kEarlyStopFlips = 1;
// `--isotropic-cleanup-relax` and `--isotropic-cleanup-weight` defaults.
constexpr int kCleanupRelaxPasses = 2;
constexpr double kCleanupWeight = 0.4;

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

//---------------------------------------------------------------------------//
// Case 3 -- DynamicRemesh::collapseShortEdges (T4d2).
//---------------------------------------------------------------------------//

/// A fresh default icosphere with every owned vertex moved radially by a
/// smooth function of its own position, then halo-exchanged. The icosphere's
/// symmetry makes many edge lengths, and so many collapse ratios, exactly
/// equal; the perturbation breaks those ties so the cap's threshold search
/// has a unique answer. It is a function of position alone, so it is
/// bit-identical at every rank count.
template <class ExecSpace, class MeshType>
void makePerturbedIcosphere( MeshType& mesh )
{
    const Real center[3] = { 0.0, 0.0, 0.0 };
    mesh.generateIcosphere( kSubdivisions, 1.0, center );
    auto pos = mesh.positions();
    const double amp = kPerturbAmp;
    Kokkos::parallel_for(
        "beatnik_t4d2_perturb", Kokkos::RangePolicy<ExecSpace>(
                                    0, mesh.ownedVertexCount() ),
        KOKKOS_LAMBDA( const int i ) {
            const double s =
                1.0 + amp * Kokkos::sin( 3.0 * pos( i, 0 ) +
                                                 5.0 * pos( i, 1 ) +
                                                 7.0 * pos( i, 2 ) );
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) = s * pos( i, d );
        } );
    Kokkos::fence();
    mesh.haloExchange();
}

template <class ExecSpace, class MeshType>
void caseCollapseShortEdges( Beatnik::Test::Recorder& rec )
{
    using remesh_type =
        Beatnik::DynamicRemesh<ExecSpace, typename MeshType::memory_space>;
    using state_type =
        Beatnik::SurfaceState<ExecSpace, typename MeshType::memory_space>;

    state_type state( Beatnik::StateModel::Potential );

    auto runPass = [&]( MeshType& mesh, const Beatnik::RemeshParams& p,
                        Beatnik::RemeshDiagnostics& diag )
    {
        remesh_type dr( p );
        typename remesh_type::scalar_view target( "beatnik_t4d2_target",
                                                  mesh.totalVertexCount() );
        dr.vertexTargetEdgeLength( mesh, state, target );
        return dr.collapseShortEdges( mesh, state, target, diag );
    };

    Beatnik::RemeshParams base;
    base.collapse_factor = kCollapseFactor;
    base.max_collapses_per_pass = 0;

    // 3a. The failure direction: collapse_factor = 0 returns 0 before any
    //     mutation -- counts, face gids, every local position and the
    //     generation are unchanged.
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makePerturbedIcosphere<ExecSpace>( mesh );
        const HostMesh before = snapshot<ExecSpace>( mesh );
        const auto gids_before = sortedFaceGids( mesh );
        const std::size_t gen_before = mesh.generation();

        Beatnik::RemeshParams p = base;
        p.collapse_factor = 0.0;
        Beatnik::RemeshDiagnostics diag;
        const int collapses = runPass( mesh, p, diag );
        rec.note( "case 3a collapse_factor=0: collapses=" +
                  std::to_string( collapses ) + " candidates=" +
                  std::to_string( diag.collapse_candidates ) );
        BEATNIK_CHECK_EQ( rec, collapses, 0 );
        BEATNIK_CHECK_EQ( rec, diag.collapse_candidates, 0 );
        BEATNIK_CHECK_EQ( rec, diag.collapse_requested, 0 );
        BEATNIK_CHECK_TRUE( rec, !diag.collapse_capped );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, mesh.generation() == gen_before ) );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
        const HostMesh after = snapshot<ExecSpace>( mesh );
        BEATNIK_CHECK_TRUE( rec, globalAll( comm, after.pos == before.pos ) );
    }

    // 3b. An artificially large factor, at max_collapses_per_pass = 0, which
    //     is UNLIMITED and does not disable the pass. Faces fall by
    //     2 * collapses, by the same literal at every rank count.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makePerturbedIcosphere<ExecSpace>( mesh );
        Beatnik::RemeshDiagnostics diag;
        const int collapses = runPass( mesh, base, diag );
        rec.note( "case 3b collapse_factor=" + std::to_string( kCollapseFactor ) +
                  " max_collapses=0: candidates=" +
                  std::to_string( diag.collapse_candidates ) + " requested=" +
                  std::to_string( diag.collapse_requested ) + " accepted=" +
                  std::to_string( collapses ) + " rejectedConflict=" +
                  std::to_string( diag.collapse_rejected_conflict ) +
                  " capped=" + std::to_string( diag.collapse_capped ) );
        rec.note( vefLine( "case 3b after", mesh ) );
        BEATNIK_CHECK_EQ( rec, diag.collapse_candidates, kCandidates );
        BEATNIK_CHECK_EQ( rec, diag.collapse_requested, kCandidates );
        BEATNIK_CHECK_TRUE( rec, !diag.collapse_capped );
        BEATNIK_CHECK_EQ( rec, collapses, kCollapsesUnlimited );
        BEATNIK_CHECK_TRUE( rec, collapses > 0 );
        BEATNIK_CHECK_EQ( rec, kF0 - mesh.globalFaceCount(),
                          2LL * collapses );
        BEATNIK_CHECK_EQ( rec, kV0 - mesh.globalVertexCount(),
                          static_cast<long long>( collapses ) );
        BEATNIK_CHECK_EQ( rec, kE0 - mesh.globalEdgeCount(),
                          3LL * collapses );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    }

    // 3c. The cap binds: the smallest-ratio kCap candidates are handed to
    //     Tessera, and the threshold is reported.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makePerturbedIcosphere<ExecSpace>( mesh );
        Beatnik::RemeshParams p = base;
        p.max_collapses_per_pass = kCap;
        Beatnik::RemeshDiagnostics diag;
        const int collapses = runPass( mesh, p, diag );
        rec.note( "case 3c max_collapses=" + std::to_string( kCap ) +
                  ": candidates=" + std::to_string( diag.collapse_candidates ) +
                  " requested=" + std::to_string( diag.collapse_requested ) +
                  " accepted=" + std::to_string( collapses ) +
                  " rejectedConflict=" +
                  std::to_string( diag.collapse_rejected_conflict ) +
                  " capped=" + std::to_string( diag.collapse_capped ) );
        rec.note( vefLine( "case 3c after", mesh ) );
        BEATNIK_CHECK_EQ( rec, diag.collapse_candidates, kCandidates );
        BEATNIK_CHECK_TRUE( rec, diag.collapse_capped );
        BEATNIK_CHECK_TRUE( rec, diag.collapse_ratio_threshold > 0.0 );
        BEATNIK_CHECK_EQ( rec, diag.collapse_requested,
                          static_cast<Beatnik::GlobalIndex>( kCap ) );
        BEATNIK_CHECK_EQ( rec, collapses, kCollapsesCapped );
        BEATNIK_CHECK_TRUE( rec, collapses > 0 );
        BEATNIK_CHECK_EQ( rec, kF0 - mesh.globalFaceCount(),
                          2LL * collapses );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    }

    // 3d. The pass loop: `remesh()` with splitting switched off by an
    //     unreachable split factor recomputes the sizing field and collapses
    //     exactly what 3b collapsed; at collapse_factor = 0 it changes
    //     nothing.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makePerturbedIcosphere<ExecSpace>( mesh );
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        rec.note( "case 3d remesh(): splits=" + std::to_string( d.splits ) +
                  " collapses=" + std::to_string( d.collapses ) +
                  " faces " + std::to_string( d.old_faces ) + " -> " +
                  std::to_string( d.new_faces ) );
        BEATNIK_CHECK_EQ( rec, d.splits, 0 );
        BEATNIK_CHECK_EQ( rec, d.collapses, kCollapsesUnlimited );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( d.new_faces ),
                          kF0 - 2LL * kCollapsesUnlimited );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    }
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makePerturbedIcosphere<ExecSpace>( mesh );
        const auto gids_before = sortedFaceGids( mesh );
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        p.collapse_factor = 0.0;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        rec.note( "case 3d remesh(), collapse_factor=0: collapses=" +
                  std::to_string( d.collapses ) + " faces " +
                  std::to_string( d.old_faces ) + " -> " +
                  std::to_string( d.new_faces ) );
        BEATNIK_CHECK_EQ( rec, d.splits, 0 );
        BEATNIK_CHECK_EQ( rec, d.collapses, 0 );
        BEATNIK_CHECK_EQ( rec, d.collapse_candidates, 0 );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( d.new_faces ), kF0 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
    }
}

//---------------------------------------------------------------------------//
// Case 4 -- the quality flips (T4d3).
//---------------------------------------------------------------------------//

/// A fresh default icosphere with every owned vertex moved tangentially by a
/// smooth, high-frequency field and put back on the unit sphere, then
/// halo-exchanged. A function of position alone, so bit-identical at every
/// rank count.
template <class ExecSpace, class MeshType>
void makeTangentialIcosphere( MeshType& mesh )
{
    const Real center[3] = { 0.0, 0.0, 0.0 };
    mesh.generateIcosphere( kSubdivisions, 1.0, center );
    auto pos = mesh.positions();
    const double amp = kTangentialAmp;
    Kokkos::parallel_for(
        "beatnik_t4d3_tangential", Kokkos::RangePolicy<ExecSpace>(
                                       0, mesh.ownedVertexCount() ),
        KOKKOS_LAMBDA( const int i ) {
            const double p[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) };
            double t[3] = { Kokkos::sin( 12.0 * p[1] + 1.0 ),
                            Kokkos::sin( 15.0 * p[2] + 2.0 ),
                            Kokkos::sin( 18.0 * p[0] + 3.0 ) };
            const double tn = t[0] * p[0] + t[1] * p[1] + t[2] * p[2];
            double w[3];
            for ( int d = 0; d < 3; ++d )
            {
                t[d] -= tn * p[d];
                w[d] = p[d] + amp * t[d];
            }
            const double n =
                Kokkos::sqrt( w[0] * w[0] + w[1] * w[1] + w[2] * w[2] );
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) = w[d] / n;
        } );
    Kokkos::fence();
    mesh.haloExchange();
}

/// Global minimum reference quality over owned faces, and the global count of
/// owned faces below `threshold`. Owned faces partition the mesh, so both are
/// rank-count invariant.
struct QualityStats
{
    double min_q = 0.0;
    long long below = 0;
};

template <class ExecSpace, class MeshType>
QualityStats qualityStats( MPI_Comm comm, MeshType& mesh,
                           double threshold )
{
    const HostMesh h = snapshot<ExecSpace>( mesh );
    auto fv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   mesh.faceVertices() );
    double local_min = std::numeric_limits<double>::infinity();
    long long local_below = 0;
    for ( int f = 0; f < mesh.ownedFaceCount(); ++f )
    {
        const double q = Beatnik::QualityFlips::triangleQuality(
            h.pos[fv( f, 0 )].data(), h.pos[fv( f, 1 )].data(),
            h.pos[fv( f, 2 )].data() );
        local_min = std::min( local_min, q );
        local_below += ( q < threshold );
    }
    QualityStats s;
    MPI_Allreduce( &local_min, &s.min_q, 1, MPI_DOUBLE, MPI_MIN, comm );
    MPI_Allreduce( &local_below, &s.below, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return s;
}

std::string statsLine( const char* when, const QualityStats& s )
{
    char buf[64];
    std::snprintf( buf, sizeof( buf ), "%.17g", s.min_q );
    return std::string( when ) + ": min quality=" + buf +
           " faces below 0.18=" + std::to_string( s.below );
}

std::string sweepLine( const char* what, int rounds, long long requested,
                       long long accepted, long long boundary,
                       long long duplicate, long long geometric,
                       long long conflict )
{
    return std::string( what ) + ": rounds=" + std::to_string( rounds ) +
           " requested=" + std::to_string( requested ) +
           " accepted=" + std::to_string( accepted ) +
           " rejectedBoundary=" + std::to_string( boundary ) +
           " rejectedDuplicateEdge=" + std::to_string( duplicate ) +
           " rejectedGeometric=" + std::to_string( geometric ) +
           " rejectedConflict=" + std::to_string( conflict );
}

template <class ExecSpace, class MeshType>
void caseQualityFlips( Beatnik::Test::Recorder& rec )
{
    using mem_space = typename MeshType::memory_space;
    using remesh_type = Beatnik::DynamicRemesh<ExecSpace, mem_space>;
    using quality_type = Beatnik::MeshQuality<ExecSpace, mem_space>;
    using state_type = Beatnik::SurfaceState<ExecSpace, mem_space>;

    state_type state( Beatnik::StateModel::Potential );

    Beatnik::RemeshParams base;
    base.min_quality = kQualityFloor;
    base.flip_min_gain = 1.0e-3;

    // A no-mutation check: counts, face gids, every local position and the
    // generation are what they were.
    struct Frozen
    {
        HostMesh h;
        std::vector<std::uint64_t> gids;
        std::size_t gen;
    };
    auto freeze = [&]( MeshType& mesh )
    {
        return Frozen{ snapshot<ExecSpace>( mesh ), sortedFaceGids( mesh ),
                       mesh.generation() };
    };
    auto unchanged = [&]( MeshType& mesh, const Frozen& b, const char* what )
    {
        const MPI_Comm comm = mesh.comm();
        const bool gen = globalAll( comm, mesh.generation() == b.gen );
        const bool gids = globalAll( comm, sortedFaceGids( mesh ) == b.gids );
        const bool pos =
            globalAll( comm, snapshot<ExecSpace>( mesh ).pos == b.h.pos );
        rec.note( std::string( what ) + ": generation " +
                  ( gen ? "unchanged" : "CHANGED" ) + ", face gids " +
                  ( gids ? "unchanged" : "CHANGED" ) + ", positions " +
                  ( pos ? "unchanged" : "CHANGED" ) );
        BEATNIK_CHECK_TRUE( rec, gen );
        BEATNIK_CHECK_TRUE( rec, gids );
        BEATNIK_CHECK_TRUE( rec, pos );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
    };

    // 4a. The failure direction: gain 1e12 flips nothing and calls nothing
    //     that mutates. The fixture's premise -- no zero-quality face, which
    //     is what makes 1e12 unsatisfiable -- is asserted first.
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makeTangentialIcosphere<ExecSpace>( mesh );
        const QualityStats s0 = qualityStats<ExecSpace>( comm, mesh,
                                                         kQualityFloor );
        rec.note( statsLine( "case 4a fixture", s0 ) );
        BEATNIK_CHECK_TRUE( rec, s0.min_q > 0.0 );
        BEATNIK_CHECK_EQ( rec, s0.below, kBelowBefore );

        const Frozen before = freeze( mesh );
        Beatnik::RemeshParams p = base;
        p.flip_min_gain = Beatnik::kFlipsDisabledMinGain;
        remesh_type dr( p );
        Beatnik::RemeshDiagnostics diag;
        const int flips = dr.flipEdgesForQuality( mesh, diag );
        rec.note( sweepLine( "case 4a flip_min_gain=1e12", diag.flip_rounds,
                             diag.flip_requested, flips,
                             diag.flip_rejected_boundary,
                             diag.flip_rejected_duplicate_edge,
                             diag.flip_rejected_geometric,
                             diag.flip_rejected_conflict ) );
        BEATNIK_CHECK_EQ( rec, flips, 0 );
        BEATNIK_CHECK_EQ( rec, diag.flip_sweeps, 1 );
        BEATNIK_CHECK_EQ( rec, diag.flip_rounds, 0 );
        BEATNIK_CHECK_EQ( rec, diag.flip_requested, 0 );
        unchanged( mesh, before, "case 4a after" );
    }

    // 4b. The passing direction: one sweep at the default gain and floor.
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makeTangentialIcosphere<ExecSpace>( mesh );
        const QualityStats s0 = qualityStats<ExecSpace>( comm, mesh,
                                                         kQualityFloor );
        const auto gids_before = sortedFaceGids( mesh );

        remesh_type dr( base );
        Beatnik::RemeshDiagnostics diag;
        const int flips = dr.flipEdgesForQuality( mesh, diag );
        const QualityStats s1 = qualityStats<ExecSpace>( comm, mesh,
                                                         kQualityFloor );
        rec.note( sweepLine( "case 4b flip_min_gain=1e-3 min_quality=0.18",
                             diag.flip_rounds, diag.flip_requested, flips,
                             diag.flip_rejected_boundary,
                             diag.flip_rejected_duplicate_edge,
                             diag.flip_rejected_geometric,
                             diag.flip_rejected_conflict ) );
        rec.note( statsLine( "case 4b before", s0 ) );
        rec.note( statsLine( "case 4b after", s1 ) );
        rec.note( vefLine( "case 4b after", mesh ) );

        BEATNIK_CHECK_EQ( rec, flips, kQualityFlips );
        BEATNIK_CHECK_TRUE( rec, flips > 0 );
        BEATNIK_CHECK_EQ( rec, diag.flip_rounds, kQualityRounds );
        BEATNIK_CHECK_TRUE( rec, diag.flip_rounds > 1 );
        BEATNIK_CHECK_EQ( rec, diag.flip_requested,
                          static_cast<Beatnik::GlobalIndex>( flips ) +
                              diag.flip_rejected_boundary +
                              diag.flip_rejected_duplicate_edge +
                              diag.flip_rejected_geometric +
                              diag.flip_rejected_conflict );
        BEATNIK_CHECK_TRUE( rec, s1.min_q >= s0.min_q );
        BEATNIK_CHECK_TRUE( rec, s1.below <= s0.below );
        BEATNIK_CHECK_EQ( rec, s1.below, kBelowAfter );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
        BEATNIK_CHECK_TRUE(
            rec, globalAll( comm, sortedFaceGids( mesh ) == gids_before ) );
    }

    // 4c. The pass loop: `remesh()` with split and collapse off. The
    //     fixture's worst face is below 0.18, so the repair gate fires and the
    //     sweep flips exactly 4b's count; at 1e12 nothing changes, which is the
    //     T4b regression member's path.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeTangentialIcosphere<ExecSpace>( mesh );
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        p.collapse_factor = 0.0;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        rec.note( "case 4c remesh(): splits=" + std::to_string( d.splits ) +
                  " collapses=" + std::to_string( d.collapses ) +
                  " flips=" + std::to_string( d.flips ) +
                  " flip_sweeps=" + std::to_string( d.flip_sweeps ) +
                  " flip_rounds=" + std::to_string( d.flip_rounds ) );
        BEATNIK_CHECK_EQ( rec, d.splits, 0 );
        BEATNIK_CHECK_EQ( rec, d.collapses, 0 );
        BEATNIK_CHECK_EQ( rec, d.flip_sweeps, 1 );
        BEATNIK_CHECK_EQ( rec, d.flips, kQualityFlips );
        BEATNIK_CHECK_EQ( rec, d.flip_rounds, kQualityRounds );
        BEATNIK_CHECK_TRUE( rec, d.min_quality_after >= d.min_quality_before );
    }
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeTangentialIcosphere<ExecSpace>( mesh );
        const Frozen before = freeze( mesh );
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        p.collapse_factor = 0.0;
        p.flip_min_gain = Beatnik::kFlipsDisabledMinGain;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        rec.note( "case 4c remesh(), flip_min_gain=1e12: flips=" +
                  std::to_string( d.flips ) + " flip_sweeps=" +
                  std::to_string( d.flip_sweeps ) + " flip_rounds=" +
                  std::to_string( d.flip_rounds ) );
        BEATNIK_CHECK_EQ( rec, d.flips, 0 );
        BEATNIK_CHECK_EQ( rec, d.flip_sweeps, 1 );
        BEATNIK_CHECK_EQ( rec, d.flip_rounds, 0 );
        unchanged( mesh, before, "case 4c after, flip_min_gain=1e12" );
    }

    // 4d. improveConnectivityByFlips: max_passes = 0 is a no-op before
    //     anything collective; three passes flip a fixed count.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeTangentialIcosphere<ExecSpace>( mesh );
        const Frozen before = freeze( mesh );
        quality_type mq( Beatnik::CleanupParams{} );
        std::vector<Beatnik::FlipSweepStats> sweeps;
        const int flips =
            mq.improveConnectivityByFlips( mesh, 0, 1.0e-3, &sweeps );
        rec.note( "case 4d max_passes=0: flips=" + std::to_string( flips ) +
                  " sweeps=" + std::to_string( sweeps.size() ) );
        BEATNIK_CHECK_EQ( rec, flips, 0 );
        BEATNIK_CHECK_TRUE( rec, sweeps.empty() );
        unchanged( mesh, before, "case 4d after, max_passes=0" );
    }
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makeTangentialIcosphere<ExecSpace>( mesh );
        const QualityStats s0 = qualityStats<ExecSpace>( comm, mesh,
                                                         kQualityFloor );
        quality_type mq( Beatnik::CleanupParams{} );
        std::vector<Beatnik::FlipSweepStats> sweeps;
        const int flips = mq.improveConnectivityByFlips(
            mesh, kImprovePasses, 1.0e-3, &sweeps );
        const QualityStats s1 = qualityStats<ExecSpace>( comm, mesh,
                                                         kQualityFloor );
        long long summed = 0;
        for ( std::size_t k = 0; k < sweeps.size(); ++k )
        {
            const auto& s = sweeps[k];
            const std::string what =
                "case 4d max_passes=3, sweep " + std::to_string( k );
            rec.note( sweepLine( what.c_str(), s.rounds, s.requested,
                                 s.accepted, s.rejected_boundary,
                                 s.rejected_duplicate_edge,
                                 s.rejected_geometric, s.rejected_conflict ) );
            summed += s.accepted;
        }
        rec.note( "case 4d max_passes=3: flips=" + std::to_string( flips ) +
                  " sweeps=" + std::to_string( sweeps.size() ) );
        rec.note( statsLine( "case 4d before", s0 ) );
        rec.note( statsLine( "case 4d after", s1 ) );
        BEATNIK_CHECK_EQ( rec, flips, kImproveFlips );
        BEATNIK_CHECK_TRUE( rec, flips > 0 );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( flips ), summed );
        BEATNIK_CHECK_EQ( rec, static_cast<int>( sweeps.size() ),
                          kImproveSweeps );
        BEATNIK_CHECK_TRUE( rec, s1.min_q >= s0.min_q );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    }
}

//---------------------------------------------------------------------------//
// Case 5 -- guarded tangential smoothing (T4d4).
//---------------------------------------------------------------------------//

/// Mean and minimum reference quality over owned faces, reduced globally.
struct QualityMeanMin
{
    double mean = 0.0;
    double min_q = 0.0;
};

template <class ExecSpace, class MeshType>
QualityMeanMin qualityMeanMin( MeshType& mesh )
{
    const MPI_Comm comm = mesh.comm();
    const HostMesh h = snapshot<ExecSpace>( mesh );
    auto fv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   mesh.faceVertices() );
    double sum = 0.0;
    double mn = std::numeric_limits<double>::infinity();
    const int nf = mesh.ownedFaceCount();
    for ( int f = 0; f < nf; ++f )
    {
        const double q = Beatnik::QualityFlips::triangleQuality(
            h.pos[fv( f, 0 )].data(), h.pos[fv( f, 1 )].data(),
            h.pos[fv( f, 2 )].data() );
        sum += q;
        mn = std::min( mn, q );
    }
    long long n = nf;
    long long n_total = 0;
    double sum_total = 0.0;
    QualityMeanMin out;
    MPI_Allreduce( &n, &n_total, 1, MPI_LONG_LONG, MPI_SUM, comm );
    MPI_Allreduce( &sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &mn, &out.min_q, 1, MPI_DOUBLE, MPI_MIN, comm );
    out.mean = sum_total / static_cast<double>( n_total );
    return out;
}

/// The global enclosed volume over owned faces.
template <class MeshType>
double globalVolume( MeshType& mesh )
{
    auto owned_faces =
        Kokkos::subview( mesh.faceVertices(),
                         std::make_pair( 0, mesh.ownedFaceCount() ),
                         Kokkos::ALL() );
    auto pos = mesh.positions();
    double local = static_cast<double>(
        Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces ) );
    double total = 0.0;
    MPI_Allreduce( &local, &total, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
    return total;
}

/// Host copy of the vertex normals `MeshGeometry::compute` gives at the
/// current positions -- the normals a smoothing iteration starting here uses.
template <class ExecSpace, class MeshType>
std::vector<Point> hostNormals( MeshType& mesh )
{
    using mem_space = typename MeshType::memory_space;
    Beatnik::MeshGeometry<ExecSpace, mem_space> geometry;
    geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                      mesh.faceVertices() );
    auto n = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                  geometry.vertex_normal );
    std::vector<Point> out( n.extent( 0 ) );
    for ( std::size_t i = 0; i < out.size(); ++i )
        out[i] = { n( i, 0 ), n( i, 1 ), n( i, 2 ) };
    return out;
}

/// `max|dx|` and `max|dx . n|` over owned vertices, reduced globally, and
/// the largest owned-row difference between two snapshots.
struct StepStats
{
    double max_step = 0.0;
    double max_normal = 0.0;
};

StepStats stepStats( MPI_Comm comm, const HostMesh& before,
                     const HostMesh& after, const std::vector<Point>& normals )
{
    double step = 0.0;
    double normal = 0.0;
    for ( int i = 0; i < before.owned_vertices; ++i )
    {
        Point dx;
        for ( int d = 0; d < 3; ++d )
            dx[d] = after.pos[i][d] - before.pos[i][d];
        step = std::max( step, std::sqrt( dot( dx, dx ) ) );
        normal = std::max( normal, std::fabs( dot( dx, normals[i] ) ) );
    }
    StepStats s;
    MPI_Allreduce( &step, &s.max_step, 1, MPI_DOUBLE, MPI_MAX, comm );
    MPI_Allreduce( &normal, &s.max_normal, 1, MPI_DOUBLE, MPI_MAX, comm );
    return s;
}

/// One reference smoothing iteration with no guard, built in the test from
/// the same public operators `tangentialSmooth` uses: `MeshGeometry`
/// normals, `graphLaplacianVector` over the one-ring, and -- when `project`
/// is set -- `projectTangent`. Owned rows with fewer than three neighbours
/// stay put, as in the reference. With `project` false this is the
/// un-projected failure direction, as in `Beatnik_Test_TangentialRelaxation`
/// group 5; with it true, it is the trial the guard judges.
template <class ExecSpace, class MeshType>
void unguardedIteration( MeshType& mesh, double relaxation, bool project )
{
    using mem_space = typename MeshType::memory_space;
    using vector_view = Kokkos::View<Real* [3], mem_space>;
    mesh.haloExchange();
    const int nv = mesh.totalVertexCount();
    const int n_owned = mesh.ownedVertexCount();
    auto pos = mesh.positions();
    auto ring = mesh.vertexOneRing();
    Beatnik::MeshGeometry<ExecSpace, mem_space> geometry;
    geometry.compute( pos, nv, mesh.faceVertices() );
    vector_view disp( "beatnik_t4d4_displacement", nv );
    Beatnik::SurfaceOperators::graphLaplacianVector( ring, pos, disp );
    if ( project )
        Beatnik::SurfaceOperators::projectTangent( disp,
                                                   geometry.vertex_normal );
    auto offsets = ring.offsets;
    const Real w = relaxation;
    Kokkos::parallel_for(
        "beatnik_t4d4_unguarded", Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int i ) {
            if ( offsets( i + 1 ) - offsets( i ) < 3 )
                return;
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) += w * disp( i, d );
        } );
    Kokkos::fence();
    mesh.haloExchange();
}

std::string g17( double x )
{
    char buf[40];
    std::snprintf( buf, sizeof( buf ), "%.17g", x );
    return buf;
}

template <class ExecSpace, class MeshType>
void caseTangentialSmooth( Beatnik::Test::Recorder& rec )
{
    using mem_space = typename MeshType::memory_space;
    using remesh_type = Beatnik::DynamicRemesh<ExecSpace, mem_space>;
    using state_type = Beatnik::SurfaceState<ExecSpace, mem_space>;

    state_type state( Beatnik::StateModel::Potential );
    const Real center[3] = { 0.0, 0.0, 0.0 };

    Beatnik::RemeshParams base;
    base.min_quality = kQualityFloor;
    base.smoothing_relaxation = kSmoothRelaxation;

    // A rejected or skipped pass leaves every local position (owned and
    // ghost) bit-identical and the generation unchanged.
    auto expectFrozen = [&]( MeshType& mesh, const HostMesh& before,
                             std::size_t gen, const std::string& what )
    {
        const MPI_Comm comm = mesh.comm();
        const bool pos =
            globalAll( comm, snapshot<ExecSpace>( mesh ).pos == before.pos );
        const bool g = globalAll( comm, mesh.generation() == gen );
        rec.note( what + ": positions " +
                  ( pos ? "bit-identical" : "CHANGED" ) + ", generation " +
                  ( g ? "unchanged" : "CHANGED" ) );
        BEATNIK_CHECK_TRUE( rec, pos );
        BEATNIK_CHECK_TRUE( rec, g );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
    };

    // 5a. Off: iterations 0 (the T4b regression member's value), and the
    //     reference's `relaxation <= 0` at zero and below. Each returns 0
    //     and changes nothing.
    {
        const int iters[3] = { 0, kSmoothIterations, kSmoothIterations };
        const double relax[3] = { kSmoothRelaxation, 0.0, -kSmoothRelaxation };
        for ( int k = 0; k < 3; ++k )
        {
            MeshType mesh( MPI_COMM_WORLD );
            mesh.generateIcosphere( kSubdivisions, 1.0, center );
            const HostMesh before = snapshot<ExecSpace>( mesh );
            const std::size_t gen = mesh.generation();
            Beatnik::RemeshParams p = base;
            p.smoothing_iterations = iters[k];
            p.smoothing_relaxation = relax[k];
            remesh_type dr( p );
            const int steps = dr.tangentialSmooth( mesh );
            const std::string what =
                "case 5a smoothing_iterations=" + std::to_string( iters[k] ) +
                " relaxation=" + g17( relax[k] ) +
                ": smooth_steps=" + std::to_string( steps );
            BEATNIK_CHECK_EQ( rec, steps, 0 );
            expectFrozen( mesh, before, gen, what );
        }
    }

    // 5b. The healthy direction, one iteration per call so each accepted
    //     iteration's displacement can be judged against the normals it
    //     used and its guard bound checked; then the same three iterations
    //     in one call.
    QualityMeanMin q0;
    double volume0 = 0.0;
    HostMesh stepped;
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        mesh.generateIcosphere( kSubdivisions, 1.0, center );
        q0 = qualityMeanMin<ExecSpace>( mesh );
        volume0 = globalVolume( mesh );
        rec.note( "case 5b icosphere: min quality=" + g17( q0.min_q ) +
                  " mean quality=" + g17( q0.mean ) );

        Beatnik::RemeshParams p = base;
        p.smoothing_iterations = 1;
        remesh_type dr( p );
        QualityMeanMin q = q0;
        for ( int k = 0; k < kSmoothIterations; ++k )
        {
            const HostMesh before = snapshot<ExecSpace>( mesh );
            const auto normals = hostNormals<ExecSpace>( mesh );
            const int steps = dr.tangentialSmooth( mesh );
            const HostMesh after = snapshot<ExecSpace>( mesh );
            const QualityMeanMin q1 = qualityMeanMin<ExecSpace>( mesh );
            const StepStats s = stepStats( comm, before, after, normals );
            const double bound =
                std::max( kQualityFloor, 0.85 * q.min_q );
            rec.note( "case 5b iteration " + std::to_string( k + 1 ) +
                      ": smooth_steps=" + std::to_string( steps ) +
                      " max|dx|=" + g17( s.max_step ) + " max|dx.n|=" +
                      g17( s.max_normal ) + " ratio=" +
                      g17( s.max_normal / s.max_step ) + " min quality " +
                      g17( q.min_q ) + " -> " + g17( q1.min_q ) +
                      " (guard bound " + g17( bound ) + ") mean quality " +
                      g17( q.mean ) + " -> " + g17( q1.mean ) );
            BEATNIK_CHECK_EQ( rec, steps, 1 );
            BEATNIK_CHECK_TRUE( rec, s.max_step > 0.0 );
            BEATNIK_CHECK_TRUE( rec,
                                s.max_normal <= kTangencyRatio * s.max_step );
            BEATNIK_CHECK_TRUE( rec, q1.min_q + 1.0e-14 >= bound );
            if ( k == 0 )
            {
                BEATNIK_CHECK_CLOSE( rec, q1.min_q, kPyMinQuality1,
                                   kPyTolerance );
                BEATNIK_CHECK_CLOSE( rec, q1.mean, kPyMeanQuality1,
                                   kPyTolerance );
            }
            q = q1;
        }
        const double dv = globalVolume( mesh ) / volume0 - 1.0;
        rec.note( "case 5b after " + std::to_string( kSmoothIterations ) +
                  " single calls: relative volume change=" + g17( dv ) );
        BEATNIK_CHECK_TRUE( rec, q.mean > q0.mean );
        BEATNIK_CHECK_CLOSE( rec, q.min_q, kPyMinQuality3, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, q.mean, kPyMeanQuality3, kPyTolerance );
        stepped = snapshot<ExecSpace>( mesh );
    }
    double dv_pass = 0.0;
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        mesh.generateIcosphere( kSubdivisions, 1.0, center );
        Beatnik::RemeshParams p = base;
        p.smoothing_iterations = kSmoothIterations;
        remesh_type dr( p );
        const int steps = dr.tangentialSmooth( mesh );
        const QualityMeanMin q = qualityMeanMin<ExecSpace>( mesh );
        dv_pass = globalVolume( mesh ) / volume0 - 1.0;
        const HostMesh after = snapshot<ExecSpace>( mesh );
        double diff = 0.0;
        for ( int i = 0; i < after.owned_vertices; ++i )
            for ( int d = 0; d < 3; ++d )
                diff = std::max( diff, std::fabs( after.pos[i][d] -
                                                  stepped.pos[i][d] ) );
        double diff_g = 0.0;
        MPI_Allreduce( &diff, &diff_g, 1, MPI_DOUBLE, MPI_MAX, comm );
        rec.note( "case 5b one call, smoothing_iterations=" +
                  std::to_string( kSmoothIterations ) +
                  ": smooth_steps=" + std::to_string( steps ) +
                  " min quality=" + g17( q.min_q ) + " mean quality=" +
                  g17( q.mean ) + " relative volume change=" + g17( dv_pass ) +
                  " max|x - x_single_calls|=" + g17( diff_g ) );
        BEATNIK_CHECK_EQ( rec, steps, kSmoothIterations );
        BEATNIK_CHECK_TRUE( rec, diff_g <= kIdenticalPositions );
        BEATNIK_CHECK_TRUE( rec, q.mean > q0.mean );
        BEATNIK_CHECK_CLOSE( rec, dv_pass, kPyVolumeRel3,
                           kPyCancellationTolerance );
    }
    {
        // The un-projected failure direction at the same relaxation.
        MeshType mesh( MPI_COMM_WORLD );
        mesh.generateIcosphere( kSubdivisions, 1.0, center );
        unguardedIteration<ExecSpace>( mesh, kSmoothRelaxation, false );
        const double dv_raw = globalVolume( mesh ) / volume0 - 1.0;
        rec.note( "case 5b un-projected iteration: relative volume change=" +
                  g17( dv_raw ) + " factor over the pass=" +
                  g17( std::fabs( dv_raw / dv_pass ) ) );
        BEATNIK_CHECK_TRUE( rec, dv_raw != 0.0 );
        BEATNIK_CHECK_TRUE( rec, kVolumeFactor * std::fabs( dv_pass ) <=
                                     std::fabs( dv_raw ) );
    }

    // 5c. The rejecting direction. min_quality = 0, so the guard's bound is
    //     0.85 * the old minimum; at kRejectRelaxation the trial overshoots
    //     the neighbour centroids. The premise -- the trial loses more than
    //     15 % of the minimum, so the 0.85 branch binds -- is measured on a
    //     separate mesh from the test's own unguarded iteration first.
    {
        MeshType mesh( MPI_COMM_WORLD );
        mesh.generateIcosphere( kSubdivisions, 1.0, center );
        const QualityMeanMin before = qualityMeanMin<ExecSpace>( mesh );
        unguardedIteration<ExecSpace>( mesh, kRejectRelaxation, true );
        const QualityMeanMin trial = qualityMeanMin<ExecSpace>( mesh );
        rec.note( "case 5c premise, relaxation=" + g17( kRejectRelaxation ) +
                  ": trial min quality " + g17( before.min_q ) + " -> " +
                  g17( trial.min_q ) + " (ratio " +
                  g17( trial.min_q / before.min_q ) + ")" );
        BEATNIK_CHECK_TRUE( rec, trial.min_q + 1.0e-14 <
                                     std::max( 0.0, 0.85 * before.min_q ) );
        BEATNIK_CHECK_TRUE( rec, 0.85 * before.min_q > 0.0 );
    }
    {
        MeshType mesh( MPI_COMM_WORLD );
        mesh.generateIcosphere( kSubdivisions, 1.0, center );
        const HostMesh before = snapshot<ExecSpace>( mesh );
        const std::size_t gen = mesh.generation();
        Beatnik::RemeshParams p = base;
        p.min_quality = 0.0;
        p.smoothing_iterations = kSmoothIterations;
        p.smoothing_relaxation = kRejectRelaxation;
        remesh_type dr( p );
        const int steps = dr.tangentialSmooth( mesh );
        BEATNIK_CHECK_EQ( rec, steps, 0 );
        expectFrozen( mesh, before, gen,
                      "case 5c min_quality=0 relaxation=" +
                          g17( kRejectRelaxation ) +
                          ": smooth_steps=" + std::to_string( steps ) );
    }

    // 5d. The pass loop. On case 4's fixture the repair gate fires (worst
    //     face below 0.18). With flips off and smoothing_iterations = 0 --
    //     the T4b regression member's path -- nothing changes. With both on,
    //     the flips run first and the smoothing after them.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeTangentialIcosphere<ExecSpace>( mesh );
        const HostMesh before = snapshot<ExecSpace>( mesh );
        const std::size_t gen = mesh.generation();
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        p.collapse_factor = 0.0;
        p.flip_min_gain = Beatnik::kFlipsDisabledMinGain;
        p.smoothing_iterations = 0;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        BEATNIK_CHECK_EQ( rec, d.flip_sweeps, 1 );
        BEATNIK_CHECK_EQ( rec, d.flips, 0 );
        BEATNIK_CHECK_EQ( rec, d.smooth_steps, 0 );
        expectFrozen( mesh, before, gen,
                      "case 5d remesh(), flip_min_gain=1e12 "
                      "smoothing_iterations=0: flip_sweeps=" +
                          std::to_string( d.flip_sweeps ) +
                          " smooth_steps=" +
                          std::to_string( d.smooth_steps ) );
    }
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeTangentialIcosphere<ExecSpace>( mesh );
        Beatnik::RemeshParams p = base;
        p.split_factor = 1.0e6;
        p.collapse_factor = 0.0;
        p.flip_min_gain = 1.0e-3;
        p.smoothing_iterations = 1;
        remesh_type dr( p );
        const Beatnik::RemeshDiagnostics d = dr.remesh( mesh, state );
        rec.note( "case 5d remesh(), flips and smoothing_iterations=1: "
                  "flips=" +
                  std::to_string( d.flips ) + " smooth_steps=" +
                  std::to_string( d.smooth_steps ) + " min quality " +
                  g17( d.min_quality_before ) + " -> " +
                  g17( d.min_quality_after ) );
        BEATNIK_CHECK_EQ( rec, d.flips, kQualityFlips );
        BEATNIK_CHECK_EQ( rec, d.smooth_steps, kRemeshSmoothSteps );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
    }
}

//---------------------------------------------------------------------------//
// Case 6 -- valence-equalizing flips and the isotropic cleanup (T4d5).
//---------------------------------------------------------------------------//

/// Case 3's perturbed icosphere, then `rounds` SharedFace flip calls, each
/// marking every owned edge whose midpoint direction is within `cap_deg` of
/// kScrambleDir. Tessera's accepted set is a function of positions and gids
/// alone, so the fixture is the same at every rank count. Returns the total
/// accepted.
template <class ExecSpace, class MeshType>
long long makeScrambledIcosphere( MeshType& mesh, double cap_deg, int rounds )
{
    makePerturbedIcosphere<ExecSpace>( mesh );
    typename MeshType::flip_policy_type permissive;
    permissive.maxNormalDeviation = std::acos( -1.0 );
    permissive.minQuality = 0.0;
    const Point u = unit( kScrambleDir );
    const double cos_cap = std::cos( cap_deg * std::acos( -1.0 ) / 180.0 );
    long long accepted = 0;
    for ( int r = 0; r < rounds; ++r )
    {
        const HostMesh h = snapshot<ExecSpace>( mesh );
        std::vector<char> mask( h.owned_edges.size(), 0 );
        for ( std::size_t e = 0; e < h.owned_edges.size(); ++e )
        {
            const int a = h.owned_edges[e][0];
            const int b = h.owned_edges[e][1];
            if ( a >= 0 && b >= 0 &&
                 dot( midDir( h.pos[a], h.pos[b] ), u ) >= cos_cap )
                mask[e] = 1;
        }
        accepted += mesh.flipEdges( mask, permissive ).accepted;
    }
    mesh.haloExchange();
    return accepted;
}

/// The global count of vertices whose valence is not 6, and the largest
/// valence, from the owned rows of the one-ring (valence on a closed
/// manifold).
struct ValenceStats
{
    long long irregular = 0;
    int max_valence = 0;
};

template <class ExecSpace, class MeshType>
ValenceStats valenceStats( MeshType& mesh )
{
    const HostMesh h = snapshot<ExecSpace>( mesh );
    long long irregular = 0;
    int mx = 0;
    for ( int i = 0; i < h.owned_vertices; ++i )
    {
        const int v = h.ring_offsets[i + 1] - h.ring_offsets[i];
        irregular += ( v != 6 );
        mx = std::max( mx, v );
    }
    ValenceStats s;
    MPI_Allreduce( &irregular, &s.irregular, 1, MPI_LONG_LONG, MPI_SUM,
                   mesh.comm() );
    MPI_Allreduce( &mx, &s.max_valence, 1, MPI_INT, MPI_MAX, mesh.comm() );
    return s;
}

/// The pass's touched-set rule, checked from the gathered flips: no accepted
/// flip has an endpoint among the corners of a flip accepted earlier in its
/// pass, nor among the corners of another flip of its own round (the
/// EndpointInQuad relation). Returns the number of violations.
long long touchedViolations( const Beatnik::ValenceFlipPass& pass )
{
    long long bad = 0;
    std::vector<std::uint64_t> touched;
    auto in = []( const std::vector<std::uint64_t>& s, std::uint64_t g )
    { return std::find( s.begin(), s.end(), g ) != s.end(); };
    for ( const auto& round : pass.rounds )
    {
        for ( std::size_t i = 0; i < round.size(); ++i )
        {
            for ( int j = 0; j < 2; ++j )
            {
                const std::uint64_t g = round[i].old_v[j];
                if ( in( touched, g ) )
                    ++bad;
                for ( std::size_t k = 0; k < round.size(); ++k )
                {
                    if ( k == i )
                        continue;
                    const auto& o = round[k];
                    if ( g == o.old_v[0] || g == o.old_v[1] ||
                         g == o.new_v[0] || g == o.new_v[1] )
                        ++bad;
                }
            }
        }
        for ( const auto& f : round )
            for ( int j = 0; j < 2; ++j )
            {
                touched.push_back( f.old_v[j] );
                touched.push_back( f.new_v[j] );
            }
    }
    return bad;
}

template <class ExecSpace, class MeshType>
void caseIsotropicCleanup( Beatnik::Test::Recorder& rec )
{
    using mem_space = typename MeshType::memory_space;
    using quality_type = Beatnik::MeshQuality<ExecSpace, mem_space>;

    Beatnik::CleanupParams on;
    on.enabled = true;
    on.flip_passes = kCleanupFlipPasses;
    on.relax_passes = kCleanupRelaxPasses;
    on.relax_weight = kCleanupWeight;
    Beatnik::CleanupParams off = on;
    off.enabled = false;

    // 6a. The switch, off: `isotropicCleanup` under `enabled = false` -- the
    //     `Solver`'s `--no-isotropic-cleanup` -- and `valenceEqualizingFlips`
    //     at zero passes change nothing: every local position, the face gids
    //     and the generation are bit-identical.
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makeScrambledIcosphere<ExecSpace>( mesh, kScrambleCapDeg,
                                           kScrambleRounds );
        const HostMesh before = snapshot<ExecSpace>( mesh );
        const auto gids = sortedFaceGids( mesh );
        const std::size_t gen = mesh.generation();
        quality_type( off ).isotropicCleanup( mesh );
        const int zero = quality_type( on ).valenceEqualizingFlips( mesh, 0 );
        const bool pos =
            globalAll( comm, snapshot<ExecSpace>( mesh ).pos == before.pos );
        const bool fg = globalAll( comm, sortedFaceGids( mesh ) == gids );
        const bool g = globalAll( comm, mesh.generation() == gen );
        rec.note( std::string( "case 6a enabled=false, and passes=0: " ) +
                  "positions " + ( pos ? "bit-identical" : "CHANGED" ) +
                  ", face gids " + ( fg ? "unchanged" : "CHANGED" ) +
                  ", generation " + ( g ? "unchanged" : "CHANGED" ) );
        BEATNIK_CHECK_EQ( rec, zero, 0 );
        BEATNIK_CHECK_TRUE( rec, pos );
        BEATNIK_CHECK_TRUE( rec, fg );
        BEATNIK_CHECK_TRUE( rec, g );
    }

    // 6b. The valence passes, one call, recorded per pass and round.
    double volume_flipped = 0.0;
    HostMesh stepped;
    ValenceStats stepped_stats;
    double dv_relax = 0.0;
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        const long long scrambled = makeScrambledIcosphere<ExecSpace>(
            mesh, kScrambleCapDeg, kScrambleRounds );
        const ValenceStats v0 = valenceStats<ExecSpace>( mesh );
        const double volume0 = globalVolume( mesh );
        rec.note( "case 6b fixture: scramble flips=" +
                  std::to_string( scrambled ) + " valence != 6: " +
                  std::to_string( v0.irregular ) + " max valence " +
                  std::to_string( v0.max_valence ) );
        BEATNIK_CHECK_EQ( rec, scrambled, kScrambleFlips );
        BEATNIK_CHECK_EQ( rec, v0.irregular, kIrregularBefore );
        BEATNIK_CHECK_EQ( rec, v0.max_valence, kMaxValenceBefore );

        std::vector<Beatnik::ValenceFlipPass> record;
        const int flips = quality_type( on ).valenceEqualizingFlips(
            mesh, kValencePasses, &record );
        BEATNIK_CHECK_EQ( rec, static_cast<int>( record.size() ),
                          kValencePasses );
        long long sum = 0;
        int max_rounds = 0;
        for ( std::size_t k = 0; k < record.size(); ++k )
        {
            const auto& p = record[k];
            long long gathered = 0;
            for ( const auto& r : p.rounds )
                gathered += static_cast<long long>( r.size() );
            const long long bad = touchedViolations( p );
            const std::string what =
                "case 6b valence pass " + std::to_string( k );
            rec.note( sweepLine( what.c_str(), p.stats.rounds,
                                 p.stats.requested, p.stats.accepted,
                                 p.stats.rejected_boundary,
                                 p.stats.rejected_duplicate_edge,
                                 p.stats.rejected_geometric,
                                 p.stats.rejected_conflict ) +
                      " gathered=" + std::to_string( gathered ) +
                      " touched-rule violations=" + std::to_string( bad ) );
            BEATNIK_CHECK_EQ( rec, bad, 0 );
            BEATNIK_CHECK_EQ( rec, gathered, p.stats.accepted );
            BEATNIK_CHECK_EQ( rec, static_cast<int>( p.rounds.size() ),
                              p.stats.rounds );
            BEATNIK_CHECK_EQ( rec, p.stats.requested,
                              p.stats.accepted + p.stats.rejected_boundary +
                                  p.stats.rejected_duplicate_edge +
                                  p.stats.rejected_geometric +
                                  p.stats.rejected_conflict );
            if ( k < static_cast<std::size_t>( kValencePasses ) )
            {
                BEATNIK_CHECK_EQ( rec, p.stats.accepted,
                                  kValencePassFlips[k] );
                BEATNIK_CHECK_EQ( rec, p.stats.rounds, kValencePassRounds[k] );
            }
            sum += p.stats.accepted;
            max_rounds = std::max( max_rounds, p.stats.rounds );
        }
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( flips ), sum );
        // The premise: some pass needed more than one round.
        BEATNIK_CHECK_TRUE( rec, max_rounds > 1 );

        const ValenceStats v1 = valenceStats<ExecSpace>( mesh );
        volume_flipped = globalVolume( mesh );
        rec.note( "case 6b after the valence passes: flips=" +
                  std::to_string( flips ) + " valence != 6: " +
                  std::to_string( v0.irregular ) + " -> " +
                  std::to_string( v1.irregular ) + " max valence " +
                  std::to_string( v0.max_valence ) + " -> " +
                  std::to_string( v1.max_valence ) +
                  " relative volume change=" +
                  g17( volume_flipped / volume0 - 1.0 ) );
        BEATNIK_CHECK_TRUE( rec, v1.irregular < v0.irregular );
        BEATNIK_CHECK_EQ( rec, v1.irregular, kIrregularAfter );
        BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kV0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kE0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kF0 );
        BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

        // 6c. The relaxation passes, one per call so each can be judged
        //     against the normals it used.
        quality_type q( on );
        for ( int k = 0; k < kCleanupRelaxPasses; ++k )
        {
            const HostMesh before = snapshot<ExecSpace>( mesh );
            const auto normals = hostNormals<ExecSpace>( mesh );
            const int sweeps = q.tangentialRelaxation( mesh, 1, kCleanupWeight );
            const HostMesh after = snapshot<ExecSpace>( mesh );
            const StepStats s = stepStats( comm, before, after, normals );
            rec.note( "case 6c relaxation pass " + std::to_string( k + 1 ) +
                      ": max|dx|=" + g17( s.max_step ) + " max|dx.n|=" +
                      g17( s.max_normal ) + " ratio=" +
                      g17( s.max_normal / s.max_step ) );
            BEATNIK_CHECK_EQ( rec, sweeps, 1 );
            BEATNIK_CHECK_TRUE( rec, s.max_step > 0.0 );
            BEATNIK_CHECK_TRUE( rec,
                                s.max_normal <= kTangencyRatio * s.max_step );
        }
        dv_relax = globalVolume( mesh ) / volume_flipped - 1.0;
        stepped = snapshot<ExecSpace>( mesh );
        stepped_stats = valenceStats<ExecSpace>( mesh );
        rec.note( "case 6c after " + std::to_string( kCleanupRelaxPasses ) +
                  " relaxation passes: relative volume change=" +
                  g17( dv_relax ) );
    }
    {
        // The un-projected failure direction: one pass at the same weight on
        // the same flipped mesh.
        MeshType mesh( MPI_COMM_WORLD );
        makeScrambledIcosphere<ExecSpace>( mesh, kScrambleCapDeg,
                                           kScrambleRounds );
        quality_type( on ).valenceEqualizingFlips( mesh, kValencePasses );
        const double v_flipped = globalVolume( mesh );
        unguardedIteration<ExecSpace>( mesh, kCleanupWeight, false );
        const double dv_raw = globalVolume( mesh ) / v_flipped - 1.0;
        rec.note( "case 6c un-projected pass: relative volume change=" +
                  g17( dv_raw ) + " factor over the relaxation passes=" +
                  g17( std::fabs( dv_raw / dv_relax ) ) );
        BEATNIK_CHECK_TRUE( rec, v_flipped == volume_flipped );
        BEATNIK_CHECK_TRUE( rec, dv_raw != 0.0 );
        BEATNIK_CHECK_TRUE( rec, kVolumeFactor * std::fabs( dv_relax ) <=
                                     std::fabs( dv_raw ) );
    }

    // 6d. The switch, on: `isotropicCleanup` in one call lands where 6b and
    //     6c's separate calls did, and the mesh is not what it was.
    {
        MeshType mesh( MPI_COMM_WORLD );
        const MPI_Comm comm = mesh.comm();
        makeScrambledIcosphere<ExecSpace>( mesh, kScrambleCapDeg,
                                           kScrambleRounds );
        const HostMesh before = snapshot<ExecSpace>( mesh );
        const std::size_t gen = mesh.generation();
        quality_type( on ).isotropicCleanup( mesh );
        const HostMesh after = snapshot<ExecSpace>( mesh );
        const ValenceStats s = valenceStats<ExecSpace>( mesh );
        double diff = 0.0;
        for ( int i = 0; i < after.owned_vertices; ++i )
            for ( int d = 0; d < 3; ++d )
                diff = std::max( diff, std::fabs( after.pos[i][d] -
                                                  stepped.pos[i][d] ) );
        double diff_g = 0.0;
        MPI_Allreduce( &diff, &diff_g, 1, MPI_DOUBLE, MPI_MAX, comm );
        const bool pos_same = globalAll( comm, after.pos == before.pos );
        const bool gen_same = globalAll( comm, mesh.generation() == gen );
        rec.note( "case 6d enabled=true: valence != 6: " +
                  std::to_string( s.irregular ) +
                  " max|x - x_separate_calls|=" + g17( diff_g ) +
                  ", positions " + ( pos_same ? "UNCHANGED" : "changed" ) +
                  ", generation " + ( gen_same ? "UNCHANGED" : "changed" ) );
        BEATNIK_CHECK_EQ( rec, s.irregular, stepped_stats.irregular );
        BEATNIK_CHECK_TRUE( rec, diff_g <= kIdenticalPositions );
        BEATNIK_CHECK_TRUE( rec, !pos_same );
        BEATNIK_CHECK_TRUE( rec, !gen_same );
    }

    // 6e. The early stop: on a fixture whose first pass flips once and
    //     second pass flips nothing, three passes requested run two.
    {
        MeshType mesh( MPI_COMM_WORLD );
        makeScrambledIcosphere<ExecSpace>( mesh, kEarlyStopCapDeg,
                                           kEarlyStopScrambleRounds );
        std::vector<Beatnik::ValenceFlipPass> record;
        const int flips = quality_type( on ).valenceEqualizingFlips(
            mesh, kValencePasses, &record );
        std::string per_pass;
        for ( const auto& p : record )
            per_pass += " " + std::to_string( p.stats.accepted ) + "/" +
                        std::to_string( p.stats.rounds );
        rec.note( "case 6e early stop: passes run=" +
                  std::to_string( record.size() ) +
                  " flips/rounds per pass:" + per_pass );
        BEATNIK_CHECK_EQ( rec, static_cast<int>( record.size() ), 2 );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( flips ),
                          kEarlyStopFlips );
        if ( record.size() == 2 )
            BEATNIK_CHECK_EQ( rec, record[1].stats.accepted, 0LL );
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
    caseCollapseShortEdges<ExecSpace, mesh_type>( rec );
    caseQualityFlips<ExecSpace, mesh_type>( rec );
    caseTangentialSmooth<ExecSpace, mesh_type>( rec );
    caseIsotropicCleanup<ExecSpace, mesh_type>( rec );
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
