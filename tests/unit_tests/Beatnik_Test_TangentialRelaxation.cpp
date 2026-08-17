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
 * @file Beatnik_Test_TangentialRelaxation.cpp
 * @brief `unit`-tier test for T4c: `MeshQuality::improveQualityTangential` —
 *        the neighbour-centroid Laplacian displacement projected onto the local
 *        tangent plane.
 *
 * THIS IS T4c's EXIT CRITERION, MECHANIZED. It is a `unit`-tier member on
 * purpose (the criterion says so): the ship gate stays at five members and 60
 * launches, and this test needs no time integration at all — the operator is a
 * pure function of the mesh.
 *
 * WHAT IT ASSERTS, AND WHY EACH PIECE IS SHAPED THE WAY IT IS
 * ----------------------------------------------------------
 * 1. **Tangency, PER SWEEP.** \f$\max_v|\Delta x_v\cdot\hat n_v| \le
 *    10^{-13}\max_v|\Delta x_v|\f$ against the normals the sweep itself used.
 *    Two things forbid the per-vertex ratio
 *    \f$|\Delta x\cdot\hat n|/|\Delta x|\f$ the criterion originally asked for:
 *
 *      * **42 of the 162 vertices move by exactly zero** — icosahedral symmetry
 *        makes their neighbour-centroid offset exactly radial, so the whole
 *        displacement projects away and the ratio is `0/0` on a quarter of the
 *        mesh. The count itself is only asserted to be non-empty: which
 *        near-zero displacements land on exactly `0.0` is a property of the
 *        arithmetic and not of the operator, and Beatnik and the Python disagree
 *        about it (`31/30/32` against `42/34/31`) while agreeing about `max|dx|`
 *        to `1e-16` relative. See the note on `kPySweepZeroMoves`.
 *      * **The identity is per sweep, not cumulative.** Each sweep re-projects
 *        against the geometry the previous one moved, so at `iterations = 3` the
 *        accumulated \f$\max|\Delta x\cdot\hat n_0|\f$ against the *pre-pass*
 *        normals is `2.05e-6` — eleven decades above the per-sweep residual.
 *        That number is asserted too, as a positive statement, so a future
 *        reader does not "fix" a cumulative tangency check that was never true.
 *
 * 2. **The mean triangle quality rises, and V/E/F are unchanged.** The
 *    operator's whole purpose. The **minimum** quality *decreases* slightly —
 *    a property of the operator, not of this port — so it is reported and NOT
 *    asserted, exactly as the criterion says.
 *
 * 3. **The pass is not a no-op** (risk R15's trap): \f$\max|\Delta x\f$| is
 *    `1.2%` of the shortest edge at one sweep and `3.0%` at three, both
 *    asserted against the reference to `1e-12`.
 *
 * 4. **The failure direction, with the separation MEASURED rather than
 *    asserted.** A deliberately un-projected sweep — `relaxTangential` verbatim
 *    minus the one `projectTangent` call, built here in the test out of the same
 *    public operators rather than through any library switch — changes the
 *    enclosed volume by `1.606e-2` relative against the projected pass's
 *    `3.898e-6`, a factor of ~4120. A tangency check that cannot see that factor
 *    has no teeth, so the factor itself is checked.
 *
 * 5. **Rank-count invariance of every scalar above.** Every quantity is a global
 *    reduction over **owned** entities (risk R9) compared against the same
 *    literal, so running the tier at `BEATNIK_UNIT_RANKS=4` is what checks the
 *    position halo exchange *between* sweeps. At one rank that precondition is
 *    unobservable: getting it wrong moves a seam with the rank count rather than
 *    failing.
 *
 * THE FIRST MULTI-RANK MEMBER OF THIS TIER
 * ---------------------------------------
 * `tests/unit_tests/CMakeLists.txt` registers the tier at one rank and its
 * comment said the multi-rank question belonged to "the task that first needs
 * one". This is that task, and the answer is the one T1c already established for
 * the `regression` tier: every rank calls `report()`, so the log names which rank
 * failed, and `main` reduces the returned exit codes with
 * `MPI_Allreduce(MPI_MAX)`. See `Beatnik_TestAssert.hpp`. The registered ctest
 * entry stays at one rank; the four-rank run is the batch wrapper's
 * `BEATNIK_UNIT_RANKS`.
 *
 * HOW THE REFERENCE NUMBERS WERE OBTAINED
 * ---------------------------------------
 * Every `kPy*` literal was computed by calling the **read-only** reference
 * (`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/mesh_solver.py::
 * improve_mesh_quality_tangential`, lines 1775-1832) on
 * `mesh.icosphere_mesh( subdivisions=2, radius=0.25, center=(0,0,0.25) )` — the
 * Python's own defaults, i.e. the mesh T1a's gold file describes — at the
 * reference's own `--smooth-relaxation 0.12`. All of them are order-invariant
 * summary scalars (a max, a min, a mean, a global volume), so this test does not
 * have to match Beatnik's vertex numbering to the Python's.
 *
 * They are hard-coded 17-digit literals and **must not be adjusted to make a
 * check pass**: a mismatch is a real disagreement with the reference and is the
 * finding. Same rule as every other test in this tree.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp`.
 */

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_MeshQuality.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <exception>
#include <sstream>
#include <string>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The mesh and the knobs. All four are the reference's own defaults:
// `parse_args` subdivision 2, radius 0.25, centre (0, 0, 0.25), and
// `--smooth-relaxation 0.12` (`run_adaptive_mesh_bubble.py:388`).
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;
constexpr Real kRelaxation = 0.12;

constexpr long long kVertices = 162;
constexpr long long kEdges = 480;
constexpr long long kFaces = 320;

//---------------------------------------------------------------------------//
// Reference values from the read-only Python, on the mesh above. See the file
// header for how they were produced and why they must not be edited.
//---------------------------------------------------------------------------//

/// The shortest edge, the scale the displacement is judged against.
constexpr double kPyShortestEdge = 6.89761210638168420e-02;

/// The raw (un-projected) neighbour-centroid offset, i.e.
/// `graphLaplacianVector` of the positions. T2b measured Beatnik's max as
/// `1.26637503746173247e-02` against this, agreeing at `4e-17` absolute — which
/// is what localizes a failure here to the projection rather than to the
/// stencil.
constexpr double kPyRawOffsetMax = 1.26637503746173715e-02;

/// Enclosed volume before the pass. Identical to T1a's `kInitialVolume`.
constexpr double kPyVolume0 = 6.32350731246695136e-02;

/// Mean and minimum \f$4\sqrt3 A/\sum\ell^2\f$ before the pass.
constexpr double kPyQualityMean0 = 9.88528666232462605e-01;
constexpr double kPyQualityMin0 = 9.77274131408830016e-01;

/// Per sweep, indexed 0, 1, 2 for sweeps 1, 2, 3. Each row is measured with the
/// normals THAT sweep used, i.e. from the geometry it started at.
constexpr double kPySweepMaxStep[3] = { 7.97280408632468937e-04,
                                        6.96133377671392896e-04,
                                        6.08979480504877584e-04 };
/// \f$\max_v|\Delta x_v\cdot\hat n_v|\f$ for that sweep. Reported, not
/// asserted against: the ASSERTION is the ratio bound below, because these are
/// values at the rounding floor and their exact size is arithmetic-dependent.
constexpr double kPySweepTangential[3] = { 2.05185261142881714e-17,
                                           2.60886147754324504e-17,
                                           2.51534904016637029e-17 };
/// Vertices whose displacement is EXACTLY zero in that sweep.
///
/// **REPORTED, NOT ASSERTED, and the reason is a finding rather than a
/// concession.** For an icosahedrally symmetric vertex the neighbour-centroid
/// offset is radial *in exact arithmetic*, so the projection removes all of it —
/// but "all of it" lands on exactly `0.0` only if the three subtractions cancel
/// to the last bit, which is a property of the arithmetic and not of the
/// operator. Beatnik measures `31 / 30 / 32` against the Python's
/// `42 / 34 / 31`, with `max|dx|` agreeing to `1e-16` relative in the same
/// sweep — i.e. the two codes disagree only about which near-zero
/// displacements round to zero. Asserting these would be asserting a rounding
/// coincidence. What the count IS good for is the criterion's point: the
/// per-vertex ratio `|dx.n|/|dx|` is `0/0` on a quarter of the mesh, so the
/// tangency statement has to be a max-over-max. That much is asserted, as
/// `zero_moves > 0`.
constexpr long long kPySweepZeroMoves[3] = { 42, 34, 31 };
/// Mean and minimum quality after that sweep.
constexpr double kPySweepQualityMean[3] = { 9.90272901161699748e-01,
                                            9.91531783096093022e-01,
                                            9.92444425261716723e-01 };
constexpr double kPySweepQualityMin[3] = { 9.77210671167454525e-01,
                                           9.77154045647789848e-01,
                                           9.77103592672459276e-01 };
/// Relative change of the enclosed volume after that many sweeps. Negative:
/// the tangential pass shrinks the polyhedron slightly, which is why the
/// driver follows it with `projectToVolume`.
constexpr double kPySweepVolumeRel[3] = { -3.89810833234527365e-06,
                                          -6.61155496195497960e-06,
                                          -8.50273211649987815e-06 };

/// Cumulative displacement after 1 and after 3 sweeps.
constexpr double kPyCumulativeMax1 = 7.97280408632468937e-04;
constexpr double kPyCumulativeMax3 = 2.10239251514152516e-03;
/// Cumulative \f$\max|\Delta x\cdot\hat n_0|\f$ against the PRE-PASS normals
/// after 3 sweeps. Eleven decades above the per-sweep residual, and that is
/// correct: see the file header.
constexpr double kPyCumulativeTangential3 = 2.04599811150858908e-06;

/// THE FAILURE DIRECTION, at one sweep: the same pass with the projection
/// removed.
constexpr double kPyUnprojectedVolumeRel = -1.60587136326278968e-02;
/// The factor between the two, DERIVED from the two measured literals rather
/// than carried as a third one. Both are volumes of the same mesh, so the ratio
/// is a reference number too — and deriving it is what keeps it from drifting
/// away from its own numerator and denominator. `4.11961707153640509e+03`.
constexpr double kPyUnprojectedFactor =
    kPyUnprojectedVolumeRel / kPySweepVolumeRel[0];

//---------------------------------------------------------------------------//
// Tolerances.
//---------------------------------------------------------------------------//

/// Against a Python reference scalar, relative. T2b's number, for T2b's reason:
/// Tessera's icosphere positions and the Python's differ only in their last
/// bits, so a derived summary scalar agrees to ~1e-15 and 1e-12 leaves three
/// decades for the non-reproducible atomic scatter and for the cross-rank
/// reduction order (risk R2). Do NOT loosen this.
constexpr double kPyTolerance = 1.0e-12;

/// For a reference formed by cancellation: `V/V0 - 1` is ~4e-6 built from two
/// volumes of ~6.3e-2, so one ulp of the ratio is `2.2e-16/3.9e-6 = 5.7e-11` of
/// the answer — a hard round-off floor no implementation beats. Two decades
/// above it, chosen from that arithmetic and not from the measurement. The same
/// argument T2d's `kVolumeDriftRtol` records.
constexpr double kPyCancellationTolerance = 1.0e-8;

/// THE TANGENCY BOUND, stated a priori by the exit criterion:
/// `max|dx.n| <= 1e-13 max|dx|` per sweep. The reference measures a ratio of
/// `2.6e-14`, so this is a factor of four of margin over the reference's own
/// rounding floor and it is the criterion's number, not a fitted one.
constexpr double kTangencyRatio = 1.0e-13;

/// Two runs that must agree bitwise in exact arithmetic — `iterations = 3` in
/// one call versus three calls of `iterations = 1`, and the two entry points on
/// the same kernel. Positions are ~0.25, so an ulp is `5.5e-17`; `1e-13`
/// absolute is three decades of slack and still eleven decades below the
/// `1e-4`-scale disagreement a Gauss-Seidel sweep or a missing halo exchange
/// would produce.
constexpr double kIdenticalPositions = 1.0e-13;

//---------------------------------------------------------------------------//
// Global reductions. Every reported scalar goes through one of these, over
// OWNED entities only (risk R9), so that a four-rank run and a one-rank run are
// comparing the same quantity against the same literal.
//---------------------------------------------------------------------------//

double globalMax( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MAX, comm );
    return out;
}

double globalSum( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_SUM, comm );
    return out;
}

double globalMin( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MIN, comm );
    return out;
}

long long globalCount( MPI_Comm comm, long long local )
{
    long long out = local;
    MPI_Allreduce( &local, &out, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

//---------------------------------------------------------------------------//
/// The global enclosed volume over **owned** faces, with `enclosedVolume`'s own
/// convention — the same shape T4b's regression test uses, for the same reason.
template <class MeshType>
double globalVolume( MeshType& mesh )
{
    auto owned_faces =
        Kokkos::subview( mesh.faceVertices(),
                         std::make_pair( 0, mesh.ownedFaceCount() ),
                         Kokkos::ALL() );
    auto pos = mesh.positions();
    const Real local =
        Beatnik::SurfaceOperators::enclosedVolume( pos, owned_faces );
    return globalSum( mesh.comm(), static_cast<double>( local ) );
}

//---------------------------------------------------------------------------//
/// Mean and minimum triangle quality over **owned** faces, reduced globally.
template <class ExecSpace, class MemSpace, class MeshType>
void globalQuality( MeshType& mesh, double& mean, double& minimum )
{
    using scalar_view = Kokkos::View<Real*, MemSpace>;
    const int n_owned = mesh.ownedFaceCount();
    auto owned_faces = Kokkos::subview(
        mesh.faceVertices(), std::make_pair( 0, n_owned ), Kokkos::ALL() );
    scalar_view quality( "beatnik_t4c_quality", n_owned );
    Beatnik::SurfaceOperators::triangleQuality( mesh.positions(), owned_faces,
                                                quality );

    Real sum = 0;
    Real worst = 1.0e300;
    Kokkos::parallel_reduce(
        "beatnik_t4c_quality_reduce", Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int f, Real& s, Real& m ) {
            s += quality( f );
            if ( quality( f ) < m )
                m = quality( f );
        },
        sum, Kokkos::Min<Real>( worst ) );

    MPI_Comm comm = mesh.comm();
    const long long faces = globalCount( comm, n_owned );
    mean = globalSum( comm, static_cast<double>( sum ) ) /
           static_cast<double>( faces );
    minimum = globalMin( comm, static_cast<double>( worst ) );
}

//---------------------------------------------------------------------------//
/// Snapshot the **whole local** position range, so the caller can both difference
/// it against the moved positions on owned rows and recompute the pre-sweep
/// geometry from it.
template <class ExecSpace, class MemSpace, class MeshType>
Kokkos::View<Real* [3], MemSpace> snapshotPositions( MeshType& mesh )
{
    const int nv = mesh.totalVertexCount();
    Kokkos::View<Real* [3], MemSpace> snap( "beatnik_t4c_positions", nv );
    auto pos = mesh.positions();
    Kokkos::parallel_for(
        "beatnik_t4c_snapshot", Kokkos::RangePolicy<ExecSpace>( 0, nv ),
        KOKKOS_LAMBDA( const int i ) {
            for ( int d = 0; d < 3; ++d )
                snap( i, d ) = pos( i, d );
        } );
    Kokkos::fence();
    return snap;
}

//---------------------------------------------------------------------------//
/// The vertex normals of a snapshot, copied out of the geometry object so a
/// later `compute()` cannot overwrite them.
template <class ExecSpace, class MemSpace, class MeshType, class VectorView>
Kokkos::View<Real* [3], MemSpace> normalsOf( MeshType& mesh,
                                             const VectorView& positions )
{
    Beatnik::MeshGeometry<ExecSpace, MemSpace> geometry;
    geometry.compute( positions, mesh.totalVertexCount(), mesh.faceVertices() );
    Kokkos::View<Real* [3], MemSpace> out( "beatnik_t4c_normals",
                                           mesh.totalVertexCount() );
    Kokkos::deep_copy( out, geometry.vertex_normal );
    return out;
}

//---------------------------------------------------------------------------//
/// Displacement statistics on **owned** vertices, against a given normal field.
struct DisplacementStats
{
    double max_step = 0.0;
    double max_tangential = 0.0;
    long long zero_moves = 0;
};

template <class ExecSpace, class MemSpace, class MeshType, class ViewA,
          class ViewB>
DisplacementStats displacementStats( MeshType& mesh, const ViewA& before,
                                     const ViewB& normals )
{
    const int n_owned = mesh.ownedVertexCount();
    auto pos = mesh.positions();
    Real max_step = 0;
    Real max_normal = 0;
    long long zeros = 0;
    Kokkos::parallel_reduce(
        "beatnik_t4c_displacement",
        Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int i, Real& m_step, Real& m_norm,
                       long long& n_zero ) {
            Real dx[3];
            for ( int d = 0; d < 3; ++d )
                dx[d] = pos( i, d ) - before( i, d );
            const Real mag =
                Kokkos::sqrt( dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] );
            if ( mag > m_step )
                m_step = mag;
            if ( mag == Real( 0 ) )
                ++n_zero;
            const Real dn = Kokkos::fabs( dx[0] * normals( i, 0 ) +
                                          dx[1] * normals( i, 1 ) +
                                          dx[2] * normals( i, 2 ) );
            if ( dn > m_norm )
                m_norm = dn;
        },
        Kokkos::Max<Real>( max_step ), Kokkos::Max<Real>( max_normal ), zeros );

    MPI_Comm comm = mesh.comm();
    DisplacementStats out;
    out.max_step = globalMax( comm, static_cast<double>( max_step ) );
    out.max_tangential = globalMax( comm, static_cast<double>( max_normal ) );
    out.zero_moves = globalCount( comm, zeros );
    return out;
}

//---------------------------------------------------------------------------//
/// `MeshQuality::relaxTangential` VERBATIM MINUS THE PROJECTION — the failure
/// direction, built here out of the same two public operators rather than
/// through a library switch. Adding a "skip the projection" knob to the library
/// would have been a CLI/API surface the reference does not have (the
/// `framework.md` **CLI surface** convention), and a test that cannot construct
/// its own negative case is not testing the projection at all.
template <class ExecSpace, class MemSpace, class MeshType>
void unprojectedSweep( MeshType& mesh, Real relaxation )
{
    using vector_view = Kokkos::View<Real* [3], MemSpace>;
    mesh.haloExchange();
    const int nv = mesh.totalVertexCount();
    const int n_owned = mesh.ownedVertexCount();
    auto pos = mesh.positions();
    vector_view displacement( "beatnik_t4c_unprojected", nv );
    Beatnik::SurfaceOperators::graphLaplacianVector( mesh.vertexOneRing(), pos,
                                                     displacement );
    // NO projectTangent here. That is the point.
    const Real w = relaxation;
    Kokkos::parallel_for(
        "beatnik_t4c_unprojected_apply",
        Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int i ) {
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) += w * displacement( i, d );
        } );
    Kokkos::fence();
    mesh.haloExchange();
}

//---------------------------------------------------------------------------//
/// Global entity counts, summed from the per-rank OWNED counts with a plain
/// `MPI_Allreduce` rather than through Tessera's own globals — R9's
/// discriminator 1, and here also the "V/E/F are unchanged" half of criterion 2.
template <class MeshType>
void checkCounts( Beatnik::Test::Recorder& rec, MeshType& mesh,
                  const std::string& where )
{
    MPI_Comm comm = mesh.comm();
    const long long v = globalCount( comm, mesh.ownedVertexCount() );
    const long long e = globalCount( comm, mesh.ownedEdgeCount() );
    const long long f = globalCount( comm, mesh.ownedFaceCount() );
    std::ostringstream os;
    os << where << ": V/E/F " << v << "/" << e << "/" << f;
    rec.note( os.str() );
    BEATNIK_CHECK_EQ( rec, v, kVertices );
    BEATNIK_CHECK_EQ( rec, e, kEdges );
    BEATNIK_CHECK_EQ( rec, f, kFaces );
    // Euler characteristic, which the counts above imply but which is the
    // statement a reader wants to see for a pass that must change no
    // connectivity at all.
    BEATNIK_CHECK_EQ( rec, v - e + f, 2 );
}

//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using quality_type = Beatnik::MeshQuality<ExecSpace, MemSpace>;
    using vector_view = Kokkos::View<Real* [3], MemSpace>;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    MPI_Comm_size( comm, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    const Real center[3] = { 0.0, 0.0, kCenterZ };

    // `CleanupParams` is what the class holds; this test drives the operator
    // through its arguments, which is how both call sites do it.
    Beatnik::CleanupParams cleanup;
    quality_type quality( cleanup );

    //-----------------------------------------------------------------------//
    // Group 0 -- the mesh, the scale, and the raw offset the pass is built on.
    //
    // The raw offset is `graphLaplacianVector` of the positions, checked here
    // against the Python so that a later tangency failure localizes to the
    // PROJECTION rather than to the neighbourhood. T2b validated the same
    // number; it is re-measured because this is the first test that depends on
    // it at four ranks, where the one-ring's completeness on owned rows is a
    // real precondition rather than a tautology.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh( comm );
        mesh.generateIcosphere( kSubdivisions, kRadius, center );
        checkCounts( rec, mesh, "initial" );

        const int nv = mesh.totalVertexCount();
        const int n_owned = mesh.ownedVertexCount();
        vector_view offset( "beatnik_t4c_raw_offset", nv );
        Beatnik::SurfaceOperators::graphLaplacianVector( mesh.vertexOneRing(),
                                                        mesh.positions(),
                                                        offset );
        Real worst = 0;
        Kokkos::parallel_reduce(
            "beatnik_t4c_raw_offset_max",
            Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                const Real mag = Kokkos::sqrt( offset( i, 0 ) * offset( i, 0 ) +
                                               offset( i, 1 ) * offset( i, 1 ) +
                                               offset( i, 2 ) * offset( i, 2 ) );
                if ( mag > m )
                    m = mag;
            },
            Kokkos::Max<Real>( worst ) );
        const double offset_max = globalMax( comm, static_cast<double>( worst ) );

        // The shortest edge, over owned edges.
        auto edge_verts = mesh.edgeVertices();
        auto pos = mesh.positions();
        const int n_owned_edges = mesh.ownedEdgeCount();
        Real shortest = 1.0e300;
        Kokkos::parallel_reduce(
            "beatnik_t4c_shortest_edge",
            Kokkos::RangePolicy<ExecSpace>( 0, n_owned_edges ),
            KOKKOS_LAMBDA( const int e, Real& m ) {
                const int ia = edge_verts( e, 0 );
                const int ib = edge_verts( e, 1 );
                if ( ia < 0 || ib < 0 )
                    return;
                Real d[3];
                for ( int k = 0; k < 3; ++k )
                    d[k] = pos( ib, k ) - pos( ia, k );
                const Real len =
                    Kokkos::sqrt( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
                if ( len < m )
                    m = len;
            },
            Kokkos::Min<Real>( shortest ) );
        const double shortest_edge =
            globalMin( comm, static_cast<double>( shortest ) );

        double mean = 0.0, minimum = 0.0;
        globalQuality<ExecSpace, MemSpace>( mesh, mean, minimum );
        const double volume = globalVolume( mesh );

        std::ostringstream os;
        os.precision( 17 );
        os << "initial: raw offset max " << offset_max << " (python "
           << kPyRawOffsetMax << "), shortest edge " << shortest_edge
           << " (python " << kPyShortestEdge << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "initial: quality mean " << mean << " (python "
            << kPyQualityMean0 << "), min " << minimum << " (python "
            << kPyQualityMin0 << "), volume " << volume << " (python "
            << kPyVolume0 << ")";
        rec.note( os2.str() );

        BEATNIK_CHECK_CLOSE( rec, offset_max, kPyRawOffsetMax, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, shortest_edge, kPyShortestEdge,
                             kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, mean, kPyQualityMean0, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, minimum, kPyQualityMin0, kPyTolerance );
        BEATNIK_CHECK_CLOSE( rec, volume, kPyVolume0, kPyTolerance );
    }

    //-----------------------------------------------------------------------//
    // Group 1 -- the two no-op configurations, which are the reference's own
    // early return (`mesh_solver.py:1792-1794`). Positions must be BITWISE
    // unchanged, not merely close: nothing has been computed, so there is
    // nothing to round.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh( comm );
        mesh.generateIcosphere( kSubdivisions, kRadius, center );
        auto before = snapshotPositions<ExecSpace, MemSpace>( mesh );

        quality.improveQualityTangential( mesh, 0, kRelaxation );
        quality.improveQualityTangential( mesh, -3, kRelaxation );
        quality.improveQualityTangential( mesh, 5, Real( 0 ) );

        const int n_owned = mesh.ownedVertexCount();
        auto pos = mesh.positions();
        long long moved = 0;
        Kokkos::parallel_reduce(
            "beatnik_t4c_noop", Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, long long& n ) {
                for ( int d = 0; d < 3; ++d )
                    if ( pos( i, d ) != before( i, d ) )
                    {
                        ++n;
                        return;
                    }
            },
            moved );
        const long long total_moved = globalCount( comm, moved );
        std::ostringstream os;
        os << "no-op configurations (iters 0, iters -3, relaxation 0): vertices "
              "moved "
           << total_moved;
        rec.note( os.str() );
        BEATNIK_CHECK_EQ( rec, total_moved, 0 );
    }

    //-----------------------------------------------------------------------//
    // Group 2 -- THE CRITERION. Three sweeps, one `iterations = 1` call each,
    // measured against the normals that sweep started from.
    //
    // Driving three sweeps as three calls rather than as `iterations = 3` is
    // what makes the PER-SWEEP tangency observable: each call recomputes the
    // geometry from the positions it is handed, so three calls of one sweep and
    // one call of three sweeps are the same arithmetic -- which group 3 then
    // asserts rather than assumes.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh( comm );
        mesh.generateIcosphere( kSubdivisions, kRadius, center );

        // The PRE-PASS normals, kept for the cumulative statement below.
        auto initial_positions = snapshotPositions<ExecSpace, MemSpace>( mesh );
        auto initial_normals = normalsOf<ExecSpace, MemSpace>(
            mesh, initial_positions );
        const double volume0 = globalVolume( mesh );

        for ( int sweep = 0; sweep < 3; ++sweep )
        {
            auto before = snapshotPositions<ExecSpace, MemSpace>( mesh );
            auto normals = normalsOf<ExecSpace, MemSpace>( mesh, before );

            const int applied =
                quality.tangentialRelaxation( mesh, 1, kRelaxation );
            BEATNIK_CHECK_EQ( rec, applied, 1 );

            const DisplacementStats stats =
                displacementStats<ExecSpace, MemSpace>( mesh, before, normals );
            double mean = 0.0, minimum = 0.0;
            globalQuality<ExecSpace, MemSpace>( mesh, mean, minimum );
            const double volume = globalVolume( mesh );
            const double ratio = ( stats.max_step > 0.0 )
                                     ? stats.max_tangential / stats.max_step
                                     : 0.0;

            std::ostringstream os;
            os.precision( 17 );
            os << "sweep " << ( sweep + 1 ) << ": max|dx| " << stats.max_step
               << " (python " << kPySweepMaxStep[sweep] << "), max|dx.n| "
               << stats.max_tangential << " (python "
               << kPySweepTangential[sweep] << "), ratio " << ratio;
            rec.note( os.str() );
            std::ostringstream os2;
            os2.precision( 17 );
            os2 << "sweep " << ( sweep + 1 ) << ": zero-move vertices "
                << stats.zero_moves << " (python " << kPySweepZeroMoves[sweep]
                << "), quality mean " << mean << " (python "
                << kPySweepQualityMean[sweep] << "), min " << minimum
                << " (python " << kPySweepQualityMin[sweep] << "), volume rel "
                << ( volume / volume0 - 1.0 ) << " (python "
                << kPySweepVolumeRel[sweep] << ")";
            rec.note( os2.str() );

            // CRITERION, part 1: TANGENCY, PER SWEEP.
            BEATNIK_CHECK_TRUE(
                rec, stats.max_tangential <= kTangencyRatio * stats.max_step );
            // The displacement scale, so the tangency bound above is a
            // statement about something that happened (risk R15).
            BEATNIK_CHECK_CLOSE( rec, stats.max_step, kPySweepMaxStep[sweep],
                                 kPyTolerance );
            // The exactly-zero population -- icosahedral symmetry. Only its
            // NON-EMPTINESS is asserted, because that is the part of it that is
            // a property of the operator rather than of the arithmetic; see the
            // note on `kPySweepZeroMoves`. It is what makes the per-vertex ratio
            // `|dx.n|/|dx|` a `0/0` on a quarter of the mesh and the max-over-max
            // above the only satisfiable form of the criterion. The rank-count
            // detector is `max|dx|` and the quality mean, both asserted against
            // the reference and both moved by a stale ghost neighbour.
            BEATNIK_CHECK_TRUE( rec, stats.zero_moves > 0 );
            // CRITERION, part 2: the mean quality against the reference. The
            // MINIMUM is reported above and deliberately not asserted -- it
            // decreases slightly, which is a property of the operator.
            BEATNIK_CHECK_CLOSE( rec, mean, kPySweepQualityMean[sweep],
                                 kPyTolerance );
            BEATNIK_CHECK_CLOSE( rec, volume / volume0 - 1.0,
                                 kPySweepVolumeRel[sweep],
                                 kPyCancellationTolerance );
            // The mean quality rises monotonically over the three sweeps --
            // asserted as a direction, independently of the literals.
            BEATNIK_CHECK_TRUE( rec, mean > ( sweep == 0
                                                  ? kPyQualityMean0
                                                  : kPySweepQualityMean[sweep - 1] ) *
                                             ( 1.0 - kPyTolerance ) );
            // V/E/F: this pass changes NO connectivity, at any sweep.
            checkCounts( rec, mesh, "after sweep " +
                                        std::to_string( sweep + 1 ) );
        }

        // CRITERION, part 1 again, the half that keeps a future reader from
        // "fixing" it: the identity is PER SWEEP. Against the pre-pass normals
        // the accumulated normal component after three sweeps is 2.05e-6 --
        // eleven decades above the per-sweep residual -- because each sweep
        // re-projects against the geometry the previous one moved.
        {
            const DisplacementStats cumulative =
                displacementStats<ExecSpace, MemSpace>( mesh,
                                                        initial_positions,
                                                        initial_normals );
            std::ostringstream os;
            os.precision( 17 );
            os << "cumulative over 3 sweeps: max|dx| " << cumulative.max_step
               << " (python " << kPyCumulativeMax3 << "), max|dx.n_0| "
               << cumulative.max_tangential << " (python "
               << kPyCumulativeTangential3 << ")";
            rec.note( os.str() );
            BEATNIK_CHECK_CLOSE( rec, cumulative.max_step, kPyCumulativeMax3,
                                 kPyTolerance );
            // The cancellation tolerance and not `kPyTolerance`: this is what
            // SURVIVES three sweeps of projection out of a 2.1e-3 displacement,
            // i.e. three decades of cancellation on a quantity itself known to
            // ~1e-16 relative, so ~1e-12 is its floor and 1e-12 is not a
            // tolerance it can meet. Measured 1.5e-12 relative at one rank.
            BEATNIK_CHECK_CLOSE( rec, cumulative.max_tangential,
                                 kPyCumulativeTangential3,
                                 kPyCancellationTolerance );
            // And it is genuinely NOT at the rounding floor, which is the
            // statement that makes the per-sweep form the only satisfiable one.
            BEATNIK_CHECK_TRUE( rec, cumulative.max_tangential >
                                         kTangencyRatio * cumulative.max_step );
        }
    }

    //-----------------------------------------------------------------------//
    // Group 3 -- `iterations = 3` in ONE call is the same three sweeps.
    //
    // The sweep loop is the only place the port could have drifted from the
    // reference's structure (a Gauss-Seidel reading of it, or a geometry
    // computed once outside the loop), and this is what pins it: the two paths
    // must agree to the last bits, and the cumulative displacement must match
    // the Python's.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh_one( comm );
        mesh_one.generateIcosphere( kSubdivisions, kRadius, center );
        mesh_type mesh_three( comm );
        mesh_three.generateIcosphere( kSubdivisions, kRadius, center );

        auto before = snapshotPositions<ExecSpace, MemSpace>( mesh_three );
        auto normals = normalsOf<ExecSpace, MemSpace>( mesh_three, before );

        for ( int i = 0; i < 3; ++i )
            quality.improveQualityTangential( mesh_one, 1, kRelaxation );
        quality.improveQualityTangential( mesh_three, 3, kRelaxation );

        const DisplacementStats stats = displacementStats<ExecSpace, MemSpace>(
            mesh_three, before, normals );
        BEATNIK_CHECK_CLOSE( rec, stats.max_step, kPyCumulativeMax3,
                             kPyTolerance );

        const int n_owned = mesh_one.ownedVertexCount();
        auto pos_one = mesh_one.positions();
        auto pos_three = mesh_three.positions();
        Real worst = 0;
        Kokkos::parallel_reduce(
            "beatnik_t4c_iterations_equivalence",
            Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                for ( int d = 0; d < 3; ++d )
                {
                    const Real e =
                        Kokkos::fabs( pos_one( i, d ) - pos_three( i, d ) );
                    if ( e > m )
                        m = e;
                }
            },
            Kokkos::Max<Real>( worst ) );
        const double difference = globalMax( comm, static_cast<double>( worst ) );
        std::ostringstream os;
        os.precision( 17 );
        os << "3 x iterations=1 versus iterations=3: max|dx| between them "
           << difference << ", cumulative max|dx| " << stats.max_step
           << " (python " << kPyCumulativeMax3 << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, difference <= kIdenticalPositions );
    }

    //-----------------------------------------------------------------------//
    // Group 4 -- ONE KERNEL, TWO ENTRY POINTS.
    //
    // `tangentialRelaxation` (whose only production caller is
    // `isotropicCleanup`, T4d) and `improveQualityTangential` are the same
    // operator, and this is the only thing that exercises the former. If they
    // ever diverge, T4d inherits a second implementation of a pass whose
    // reference numbers were measured against this one.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh_a( comm );
        mesh_a.generateIcosphere( kSubdivisions, kRadius, center );
        mesh_type mesh_b( comm );
        mesh_b.generateIcosphere( kSubdivisions, kRadius, center );

        quality.improveQualityTangential( mesh_a, 2, kRelaxation );
        const int applied = quality.tangentialRelaxation( mesh_b, 2,
                                                         kRelaxation );
        BEATNIK_CHECK_EQ( rec, applied, 2 );

        const int n_owned = mesh_a.ownedVertexCount();
        auto pos_a = mesh_a.positions();
        auto pos_b = mesh_b.positions();
        Real worst = 0;
        Kokkos::parallel_reduce(
            "beatnik_t4c_entry_point_equivalence",
            Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i, Real& m ) {
                for ( int d = 0; d < 3; ++d )
                {
                    const Real e = Kokkos::fabs( pos_a( i, d ) - pos_b( i, d ) );
                    if ( e > m )
                        m = e;
                }
            },
            Kokkos::Max<Real>( worst ) );
        const double difference = globalMax( comm, static_cast<double>( worst ) );
        std::ostringstream os;
        os.precision( 17 );
        os << "improveQualityTangential versus tangentialRelaxation: max "
              "position difference "
           << difference;
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, difference <= kIdenticalPositions );
    }

    //-----------------------------------------------------------------------//
    // Group 5 -- THE FAILURE DIRECTION, MEASURED.
    //
    // The un-projected sweep is Laplacian smoothing of the interface: it shrinks
    // the bubble. The projected pass changes the enclosed volume by 3.9e-6
    // relative and the un-projected one by 1.6e-2 -- a factor of ~4120, which is
    // the separation the tangency check has to be able to see. Both are compared
    // against the Python, and the FACTOR is asserted, because a tangency bound
    // that passes on both would be no check at all.
    //-----------------------------------------------------------------------//
    {
        mesh_type mesh_projected( comm );
        mesh_projected.generateIcosphere( kSubdivisions, kRadius, center );
        mesh_type mesh_raw( comm );
        mesh_raw.generateIcosphere( kSubdivisions, kRadius, center );

        const double volume0 = globalVolume( mesh_projected );

        auto raw_before = snapshotPositions<ExecSpace, MemSpace>( mesh_raw );
        auto raw_normals = normalsOf<ExecSpace, MemSpace>( mesh_raw,
                                                           raw_before );

        quality.improveQualityTangential( mesh_projected, 1, kRelaxation );
        unprojectedSweep<ExecSpace, MemSpace>( mesh_raw, kRelaxation );

        const double projected_rel =
            globalVolume( mesh_projected ) / volume0 - 1.0;
        const double raw_rel = globalVolume( mesh_raw ) / volume0 - 1.0;
        const double factor = ( projected_rel != 0.0 )
                                  ? std::fabs( raw_rel / projected_rel )
                                  : 0.0;

        const DisplacementStats raw_stats =
            displacementStats<ExecSpace, MemSpace>( mesh_raw, raw_before,
                                                    raw_normals );

        std::ostringstream os;
        os.precision( 17 );
        os << "failure direction: projected volume rel " << projected_rel
           << " (python " << kPySweepVolumeRel[0] << "), UN-projected "
           << raw_rel << " (python " << kPyUnprojectedVolumeRel
           << "), factor " << factor << " (python " << kPyUnprojectedFactor
           << ")";
        rec.note( os.str() );
        std::ostringstream os2;
        os2.precision( 17 );
        os2 << "failure direction: the un-projected sweep's max|dx.n| "
            << raw_stats.max_tangential << " against max|dx| "
            << raw_stats.max_step << ", ratio "
            << ( raw_stats.max_tangential / raw_stats.max_step )
            << " -- the tangency bound is " << kTangencyRatio;
        rec.note( os2.str() );

        BEATNIK_CHECK_CLOSE( rec, projected_rel, kPySweepVolumeRel[0],
                             kPyCancellationTolerance );
        BEATNIK_CHECK_CLOSE( rec, raw_rel, kPyUnprojectedVolumeRel,
                             kPyCancellationTolerance );
        BEATNIK_CHECK_CLOSE( rec, factor, kPyUnprojectedFactor,
                             kPyCancellationTolerance );
        // The teeth: the separation is three decades, so a tangency check has
        // something to catch.
        BEATNIK_CHECK_TRUE( rec, factor > 1.0e3 );
        // And the check itself does catch it -- the same bound that passes on
        // every projected sweep above FAILS here by eleven decades.
        BEATNIK_CHECK_TRUE( rec, raw_stats.max_tangential >
                                     kTangencyRatio * raw_stats.max_step );
        // The un-projected pass still changes no connectivity, so the volume
        // difference is the projection's and nothing else's.
        checkCounts( rec, mesh_raw, "after the un-projected sweep" );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_TangentialRelaxation" );
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
    // rank failed; the reduction is here and not inside the recorder because a
    // collective in there would deadlock exactly when one rank took an early
    // exception path. See Beatnik_TestAssert.hpp.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
