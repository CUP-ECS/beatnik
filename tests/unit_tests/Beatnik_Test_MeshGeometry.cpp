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
 * @file Beatnik_Test_MeshGeometry.cpp
 * @brief `unit`-tier test for T1b: icosphere generation and mesh geometry.
 *
 * THIS IS T1b's EXIT CRITERION, MECHANIZED.
 * `tasks/framework.md` states it as: *a 1-rank run reproduces the Python's
 * vertex and face counts (162 / 320 at the default subdivision 2) and its
 * enclosed volume and minimum edge length to `1e-14` relative.*
 *
 * There is no driver path to check that through — `Solver::setup` and
 * `CheckpointIO::write` are T1c, so no `beatnik.h5` exists to hand to
 * `compare_output.py`. This test is what stands in until there is, and it
 * checks strictly more than the exit criterion: the two reference scalars, the
 * three entity counts, and the internal consistency of all three adjacency
 * relations, which is what would catch a mesh that has the right *number* of
 * things connected wrongly.
 *
 * The two reference scalars are **hard-coded literals** taken from the T1a run
 * recorded in `tasks/framework.md`. They are not recomputed here and must not
 * be adjusted to make this pass: a mismatch at `1e-14` is a real disagreement
 * with the Python reference and is the finding.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp` for why the
 * verdict is the binary's own and not ctest's.
 */

#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <exception>
#include <sstream>
#include <string>
#include <utility>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// T1a reference values.
//
// Produced by the Python driver, `tasks/framework.md` task T1a (met
// 2026-08-07):
//
//   python examples/run_adaptive_mesh_bubble.py
//       --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
//       --viscosity-mode laplace-beltrami --br-approximation direct
//       --isotropic-cleanup --checkpoint-every-steps 1 --no-video
//       --steps 0 --source-quadrature vertex
//
// i.e. the default icosphere: subdivision 2, radius 0.25, centre (0, 0, 0.25).
// Carried in the gold checkpoint as `/beatnik/initial_volume` and
// `/beatnik/initial_min_edge`.
//---------------------------------------------------------------------------//
constexpr double kInitialVolume = 6.3235073124669514e-02;
constexpr double kInitialMinEdge = 6.8976121063816842e-02;

/// The exit criterion's tolerance. Do NOT loosen this; see the file header.
constexpr double kRefTolerance = 1.0e-14;

/// Slack for identities that hold exactly in exact arithmetic but are computed
/// through an atomic scatter, so their rounding is not even reproducible run to
/// run (see DETERMINISM in `Beatnik_MeshGeometry.hpp`). These are consistency
/// checks on Beatnik against itself, not comparisons against the Python, so the
/// tolerance is a numerical statement rather than a contract.
constexpr double kIdentityTolerance = 1.0e-12;

// The Python's defaults, from `run_adaptive_mesh_bubble.py::parse_args`:
// --icosphere-subdivisions 2, --radius 0.25, --center-z 0.25.
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;

// Closed-form counts for a subdivision-2 icosphere: V = 10*4^s + 2,
// F = 20*4^s, and E = 3F/2 for a closed triangle mesh. Euler: V - E + F = 2.
constexpr long long kVertices = 162;
constexpr long long kFaces = 320;
constexpr long long kEdges = 480;

//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using Ops = Beatnik::SurfaceOperators;

    int comm_size = 1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    mesh_type mesh( MPI_COMM_WORLD );
    const Real center[3] = { 0.0, 0.0, 0.25 };
    mesh.generateIcosphere( kSubdivisions, kRadius, center );

    //-----------------------------------------------------------------------//
    // Group 1 — entity counts and the halo depth.
    //-----------------------------------------------------------------------//
    // Reduced as integers, so these are exact at every rank count.
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), kVertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), kFaces );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), kEdges );
    // V - E + F == 2 for a closed surface: 162 - 480 + 320.
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );
    // Risk R8: the two-ring RHS needs depth 2, set once at construction. If
    // this is 1, a two-ring stencil throws instead of being silently short --
    // which is the loud failure the depth work bought, but it is still a
    // failure.
    BEATNIK_CHECK_EQ( rec, mesh.haloDepth(), mesh_type::halo_depth );

    const int nv_local = mesh.totalVertexCount();
    const int nv_owned = mesh.ownedVertexCount();
    const int ne_owned = mesh.ownedEdgeCount();
    const int nf_owned = mesh.ownedFaceCount();
    const int nf_local = mesh.totalFaceCount();
    {
        std::ostringstream os;
        os << "local V " << nv_local << " (owned " << nv_owned << "), owned E "
           << ne_owned << ", local F " << nf_local << " (owned " << nf_owned
           << ")";
        rec.note( os.str() );
    }

    auto pos = mesh.positions();
    auto face_verts = mesh.faceVertices();
    auto edge_verts = mesh.edgeVertices();

    // Owned subranges. `enclosedVolume` and `edgeLengths` reduce to a global
    // scalar, so a ghost entity would be counted twice (risk R9).
    auto owned_faces = Kokkos::subview(
        face_verts, std::make_pair( 0, nf_owned ), Kokkos::ALL() );
    auto owned_edges = Kokkos::subview(
        edge_verts, std::make_pair( 0, ne_owned ), Kokkos::ALL() );

    //-----------------------------------------------------------------------//
    // Group 2 — enclosed volume against the T1a reference.
    //-----------------------------------------------------------------------//
    const Real volume = Beatnik::Comm::allReduceSum(
        mesh.comm(), Ops::enclosedVolume( pos, owned_faces ) );
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "enclosed volume " << volume << " vs T1a " << kInitialVolume;
        rec.note( os.str() );
    }
    BEATNIK_CHECK_CLOSE( rec, volume, kInitialVolume, kRefTolerance );
    // Positive, i.e. outward-wound. `generateIcosphere` already throws
    // otherwise, so this is a guard on that guard rather than a fresh check --
    // cheap, and it fails loudly if the orientation check is ever weakened.
    BEATNIK_CHECK_TRUE( rec, volume > 0.0 );

    //-----------------------------------------------------------------------//
    // Group 3 — minimum edge length against the T1a reference.
    //-----------------------------------------------------------------------//
    Kokkos::View<Real*, MemSpace> lengths( "edge_lengths", ne_owned );
    Ops::edgeLengths( pos, owned_edges, lengths );
    Real min_local = 0.0;
    Kokkos::parallel_reduce(
        "test_min_edge", Kokkos::RangePolicy<ExecSpace>( 0, ne_owned ),
        KOKKOS_LAMBDA( const int e, Real& m ) {
            if ( lengths( e ) < m )
                m = lengths( e );
        },
        Kokkos::Min<Real>( min_local ) );
    const Real h_min = Beatnik::Comm::allReduceMin( mesh.comm(), min_local );
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "min edge " << h_min << " vs T1a " << kInitialMinEdge;
        rec.note( os.str() );
    }
    BEATNIK_CHECK_CLOSE( rec, h_min, kInitialMinEdge, kRefTolerance );

    //-----------------------------------------------------------------------//
    // Group 4 — adjacency consistency.
    //
    // The counts above say the mesh has the right NUMBER of things. These say
    // they are connected to each other, which is the part a count cannot see.
    //-----------------------------------------------------------------------//

    // 4a. Every edge of a closed surface has exactly two incident faces.
    //     Checked over the whole local edge set, since `EdgeField::Faces`
    //     records both incidences by gid on every rank that holds the edge.
    {
        auto inc = mesh.edgeAdjacency();
        const int ne_local = mesh.totalEdgeCount();
        auto count = inc.count;
        int bad = 0;
        Kokkos::parallel_reduce(
            "test_edge_incidence",
            Kokkos::RangePolicy<ExecSpace>( 0, ne_local ),
            KOKKOS_LAMBDA( const int e, int& acc ) {
                if ( count( e ) != 2 )
                    ++acc;
            },
            bad );
        BEATNIK_CHECK_EQ( rec, bad, 0 );
    }

    // 4b. Vertex adjacency is symmetric, and its total entry count is 2E.
    //     Symmetry is asserted only for owned-owned pairs: a ghost vertex's row
    //     may legitimately stop at the edge of the local set, so asserting it
    //     there would be asserting something false at >1 rank. At 1 rank every
    //     vertex is owned, so nothing is skipped.
    {
        auto ring = mesh.vertexOneRing();
        auto offsets = ring.offsets;
        auto neighbors = ring.neighbors;
        int asymmetric = 0;
        Kokkos::parallel_reduce(
            "test_ring_symmetry", Kokkos::RangePolicy<ExecSpace>( 0, nv_owned ),
            KOKKOS_LAMBDA( const int i, int& acc ) {
                for ( int p = offsets( i ); p < offsets( i + 1 ); ++p )
                {
                    const int j = neighbors( p );
                    if ( j >= nv_owned )
                        continue;
                    bool found = false;
                    for ( int q = offsets( j ); q < offsets( j + 1 ); ++q )
                        if ( neighbors( q ) == i )
                            found = true;
                    if ( !found )
                        ++acc;
                    // The self-loop the CSR must not contain.
                    if ( j == i )
                        ++acc;
                }
            },
            asymmetric );
        BEATNIK_CHECK_EQ( rec, asymmetric, 0 );

        // Each edge contributes its two endpoints to each other's row, so the
        // total entry count over ALL local vertices is twice the local edge
        // count only when the local set is closed -- true at 1 rank. Assert the
        // exact global figure there and only the lower bound otherwise.
        int entries = 0;
        Kokkos::parallel_reduce(
            "test_ring_entries", Kokkos::RangePolicy<ExecSpace>( 0, nv_local ),
            KOKKOS_LAMBDA( const int i, int& acc ) {
                acc += offsets( i + 1 ) - offsets( i );
            },
            entries );
        if ( comm_size == 1 )
            BEATNIK_CHECK_EQ( rec, entries, 2 * kEdges );
        else
            BEATNIK_CHECK_TRUE( rec, entries >= 2 * ne_owned );
    }

    // 4c. Face adjacency: degree exactly 3 on a closed surface, and reciprocal.
    //     The local-index half of the CSR is only usable where every neighbour
    //     is resident, which Tessera reports as `non_resident == 0` -- so the
    //     reciprocity check is gated on that rather than assuming it.
    {
        auto adj = mesh.faceAdjacency();
        auto offsets = adj.offsets;
        auto neighbors = adj.neighbors;
        {
            std::ostringstream os;
            os << "face adjacency non-resident entries " << adj.non_resident;
            rec.note( os.str() );
        }
        int wrong_degree = 0;
        Kokkos::parallel_reduce(
            "test_face_degree", Kokkos::RangePolicy<ExecSpace>( 0, nf_owned ),
            KOKKOS_LAMBDA( const int f, int& acc ) {
                if ( offsets( f + 1 ) - offsets( f ) != 3 )
                    ++acc;
            },
            wrong_degree );
        BEATNIK_CHECK_EQ( rec, wrong_degree, 0 );

        if ( adj.non_resident == 0 )
        {
            int not_reciprocal = 0;
            Kokkos::parallel_reduce(
                "test_face_reciprocity",
                Kokkos::RangePolicy<ExecSpace>( 0, nf_owned ),
                KOKKOS_LAMBDA( const int f, int& acc ) {
                    for ( int p = offsets( f ); p < offsets( f + 1 ); ++p )
                    {
                        const int g = neighbors( p );
                        if ( g < 0 )
                        {
                            ++acc;
                            continue;
                        }
                        bool found = false;
                        for ( int q = offsets( g ); q < offsets( g + 1 ); ++q )
                            if ( neighbors( q ) == f )
                                found = true;
                        if ( !found )
                            ++acc;
                    }
                },
                not_reciprocal );
            BEATNIK_CHECK_EQ( rec, not_reciprocal, 0 );
        }
        else
        {
            rec.note( "face reciprocity skipped: some neighbours are "
                      "non-resident, so the local-index CSR is incomplete by "
                      "contract" );
            BEATNIK_CHECK_TRUE( rec, comm_size > 1 );
        }
    }

    //-----------------------------------------------------------------------//
    // Group 5 — the derived geometry is self-consistent.
    //
    // `MeshGeometry::compute` and `volumeGradient` are T1b bodies with no
    // reference value of their own at this task, so they are pinned by two
    // identities that hold exactly in exact arithmetic. Both are stated over
    // the WHOLE surface, so they are asserted at 1 rank only -- a per-rank
    // partial sum satisfies neither.
    //-----------------------------------------------------------------------//
    {
        Beatnik::MeshGeometry<ExecSpace, MemSpace> geom;
        // The whole local face set, per DISTRIBUTED ASSEMBLY: every owned
        // vertex then sees its complete incident-face set with no scatter-add.
        geom.compute( pos, nv_local, face_verts );

        Kokkos::View<Real* [3], MemSpace> grad( "volume_gradient", nv_local );
        Ops::volumeGradient( pos, face_verts, grad );

        if ( comm_size == 1 )
        {
            // A_v = (1/3) sum_{f in v} A_f, so sum_v A_v == sum_f A_f exactly.
            auto fa = geom.face_area;
            auto va = geom.vertex_area;
            Real face_total = 0.0, vertex_total = 0.0;
            Kokkos::parallel_reduce(
                "test_face_area_sum",
                Kokkos::RangePolicy<ExecSpace>( 0, nf_local ),
                KOKKOS_LAMBDA( const int f, Real& acc ) { acc += fa( f ); },
                face_total );
            Kokkos::parallel_reduce(
                "test_vertex_area_sum",
                Kokkos::RangePolicy<ExecSpace>( 0, nv_local ),
                KOKKOS_LAMBDA( const int i, Real& acc ) { acc += va( i ); },
                vertex_total );
            {
                std::ostringstream os;
                os.precision( 17 );
                os << "surface area (faces) " << face_total
                   << ", (lumped vertices) " << vertex_total;
                rec.note( os.str() );
            }
            BEATNIK_CHECK_CLOSE( rec, vertex_total, face_total,
                                 kIdentityTolerance );

            // V is homogeneous of degree 3 in the vertex coordinates, so
            // Euler's theorem gives sum_v p_v . dV/dp_v == 3V. This pins the
            // per-corner cross products AND their 1/6 factor at once; getting
            // the cyclic order wrong breaks it.
            Real contracted = 0.0;
            Kokkos::parallel_reduce(
                "test_volume_gradient_euler",
                Kokkos::RangePolicy<ExecSpace>( 0, nv_local ),
                KOKKOS_LAMBDA( const int i, Real& acc ) {
                    for ( int d = 0; d < 3; ++d )
                        acc += pos( i, d ) * grad( i, d );
                },
                contracted );
            {
                std::ostringstream os;
                os.precision( 17 );
                os << "sum_v p.dV/dp " << contracted << " vs 3V "
                   << 3.0 * volume;
                rec.note( os.str() );
            }
            BEATNIK_CHECK_CLOSE( rec, contracted, 3.0 * volume,
                                 kIdentityTolerance );
        }
        else
        {
            rec.note( "geometry identities skipped: both are whole-surface "
                      "statements and this run has more than one rank" );
        }
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_MeshGeometry" );
        try
        {
            // The default execution space is what the build targets (HIP on
            // tuolumne). The `unit` tier registers one binary rather than one
            // per backend, so there is no `_SERIAL`/`_HIP` name suffix here --
            // see tests/unit_tests/CMakeLists.txt.
            using ExecSpace = Kokkos::DefaultExecutionSpace;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec );
        }
        catch ( const std::exception& e )
        {
            // Most likely a BEATNIK_NOT_IMPLEMENTED from a stub on a path this
            // test did not expect to touch. Reported as a named failure rather
            // than allowed to abort, so the tally line still appears in the
            // log.
            rec.fail( std::string( "unexpected exception: " ) + e.what() );
        }
        catch ( ... )
        {
            rec.fail( "unexpected non-std exception" );
        }
        rc = rec.report();
    }

    Kokkos::finalize();
    MPI_Finalize();
    return rc;
}
