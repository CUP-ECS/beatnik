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
 * @file Beatnik_Test_Restart.cpp
 * @brief `unit`-tier test for restart: `CheckpointIO::read`,
 *        `RestartReader::load`, and the checkpoint series across a restart.
 *
 * WHAT IS CHECKED
 * ---------------
 *  1. **Round trip.** A level-2 cold start with a non-trivial potential,
 *     sheet vector and material position is written with `CheckpointIO::write`
 *     and read with `RestartReader::load` into a fresh mesh. Asserted:
 *       - the five header scalars come back exactly;
 *       - global `V`/`E`/`F` are equal;
 *       - every local vertex (owned and ghost) of the read mesh has a gid the
 *         written mesh had, and its position and three fields equal the
 *         written ones exactly. Pairing is by gid: `readMesh` keeps the gids
 *         the file was written with (`/vertices/gid`);
 *       - the owned vertices partition the surface (their global sum is `V`);
 *       - the halo depth is 2;
 *       - `RestartState.from_checkpoint`, `amr_reference_rebased` and
 *         `material_position_loaded` are all true.
 *  2. **The series.** A restart into a directory with no index starts a new
 *     series (one index line after its startup frame). A restart into the
 *     writer's directory from its latest frame continues the series: the
 *     startup frame is not re-appended, the next one is, and master and index
 *     both list three frames. A restart there from an earlier frame throws,
 *     naming the index, and leaves the master byte-unchanged.
 *  3. **Failure directions.** Each prints its own PASS line on rank 0:
 *       - a missing path throws `std::runtime_error` whose message holds the
 *         path;
 *       - a file written by `Tessera::writeMesh` alone (no `/beatnik` group)
 *         throws naming `/beatnik`;
 *       - a state model differing from the run's throws naming both.
 *     None aborts: each is a throw before `readMesh`, on every rank.
 *
 * Rank-count aware: every comparison is per vertex by gid against a record
 * gathered on every rank, or a global reduction, so the same assertions hold
 * at 1 and at `BEATNIK_UNIT_RANKS=4`. Writes to `BEATNIK_TEST_SCRATCH`, which
 * must name a parallel filesystem: the checkpoints go through MPI-IO.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp`.
 */

#include <Beatnik_AdaptiveMesh.hpp>
#include <Beatnik_IOInterface.hpp>
#include <Beatnik_InitialCondition.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Restart.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Tessera.hpp>

#include <mpi.h>

#include <sys/stat.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

using Beatnik::Real;

/// The header the round trip writes. Distinctive values, none of them a
/// cold start's, so a reader that returned defaults would fail.
constexpr double kTime = 0.375;
constexpr long long kStep = 7;
constexpr double kInitialVolume = 0.0654321;
constexpr double kInitialMinEdge = 0.0412345;

/// An earlier frame of the same series, for the restart-from-earlier case.
constexpr double kEarlyTime = 0.125;
constexpr long long kEarlyStep = 3;

/// The frame the continued series appends after its restart point.
constexpr double kNextTime = 0.5;
constexpr long long kNextStep = 9;

/// Per-vertex record: position, potential, sheet vector, material position.
constexpr int kRecordWidth = 10;

std::string readFile( const std::string& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return std::string();
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

int countOccurrences( const std::string& haystack, const std::string& needle )
{
    int n = 0;
    std::size_t at = haystack.find( needle );
    while ( at != std::string::npos )
    {
        ++n;
        at = haystack.find( needle, at + needle.size() );
    }
    return n;
}

bool contains( const std::string& text, const std::string& needle )
{
    return text.find( needle ) != std::string::npos;
}

/// Host copy of every local vertex: gid -> record, plus the owned count.
template <class MeshType>
std::map<std::uint64_t, std::vector<double>> localRecords( MeshType& mesh,
                                                           bool owned_only )
{
    auto& tmesh = mesh.tesseraMesh();
    using tmesh_type = std::remove_reference_t<decltype( tmesh )>;
    Cabana::AoSoA<typename tmesh_type::vertex_member_types, Kokkos::HostSpace>
        hv( "restart_test_vertices", tmesh.numVertices() );
    Cabana::deep_copy( hv, tmesh.vertices() );
    auto gid = Cabana::slice<Tessera::VertexField::Gid>( hv );
    auto pos = Cabana::slice<Tessera::VertexField::Position>( hv );
    auto phi = Cabana::slice<
        Tessera::userVertexField<Beatnik::VertexFieldId::Potential>()>( hv );
    auto sv = Cabana::slice<
        Tessera::userVertexField<Beatnik::VertexFieldId::SheetVector>()>( hv );
    auto mat = Cabana::slice<Tessera::userVertexField<
        Beatnik::VertexFieldId::MaterialPosition>()>( hv );

    const int n =
        owned_only ? mesh.ownedVertexCount() : mesh.totalVertexCount();
    std::map<std::uint64_t, std::vector<double>> out;
    for ( int v = 0; v < n; ++v )
    {
        std::vector<double> r( kRecordWidth );
        for ( int d = 0; d < 3; ++d )
        {
            r[d] = pos( v, d );
            r[4 + d] = sv( v, d );
            r[7 + d] = mat( v, d );
        }
        r[3] = phi( v );
        out[static_cast<std::uint64_t>( gid( v ) )] = r;
    }
    return out;
}

/// Every rank's OWNED records, gathered onto every rank.
std::map<std::uint64_t, std::vector<double>>
gatherRecords( MPI_Comm comm,
               const std::map<std::uint64_t, std::vector<double>>& owned )
{
    int size = 1;
    MPI_Comm_size( comm, &size );
    std::vector<std::uint64_t> gids;
    std::vector<double> values;
    for ( const auto& kv : owned )
    {
        gids.push_back( kv.first );
        values.insert( values.end(), kv.second.begin(), kv.second.end() );
    }
    int n = static_cast<int>( gids.size() );
    std::vector<int> counts( size ), displs( size, 0 );
    MPI_Allgather( &n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );
    for ( int r = 1; r < size; ++r )
        displs[r] = displs[r - 1] + counts[r - 1];
    const int total = displs[size - 1] + counts[size - 1];

    std::vector<std::uint64_t> all_gids( total );
    MPI_Allgatherv( gids.data(), n, MPI_UINT64_T, all_gids.data(),
                    counts.data(), displs.data(), MPI_UINT64_T, comm );

    std::vector<int> vcounts( size ), vdispls( size );
    for ( int r = 0; r < size; ++r )
    {
        vcounts[r] = counts[r] * kRecordWidth;
        vdispls[r] = displs[r] * kRecordWidth;
    }
    std::vector<double> all_values( static_cast<std::size_t>( total ) *
                                    kRecordWidth );
    MPI_Allgatherv( values.data(), n * kRecordWidth, MPI_DOUBLE,
                    all_values.data(), vcounts.data(), vdispls.data(),
                    MPI_DOUBLE, comm );

    std::map<std::uint64_t, std::vector<double>> out;
    for ( int i = 0; i < total; ++i )
        out[all_gids[i]] = std::vector<double>(
            all_values.begin() + static_cast<std::ptrdiff_t>( i ) * kRecordWidth,
            all_values.begin() +
                static_cast<std::ptrdiff_t>( i + 1 ) * kRecordWidth );
    return out;
}

Beatnik::CheckpointHeader makeHeader( double time, long long step )
{
    Beatnik::CheckpointHeader header;
    header.state_model = Beatnik::StateModel::Potential;
    header.time = static_cast<Real>( time );
    header.step = step;
    header.initial_volume = static_cast<Real>( kInitialVolume );
    header.initial_min_edge = static_cast<Real>( kInitialMinEdge );
    header.has_material_position = true;
    return header;
}

//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using state_type = Beatnik::SurfaceState<ExecSpace, MemSpace>;
    using amr_type = Beatnik::AdaptiveMesh<ExecSpace, MemSpace>;
    using io_type = Beatnik::CheckpointIO<ExecSpace, MemSpace>;
    using reader_type = Beatnik::RestartReader<ExecSpace, MemSpace>;

    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    // Output directory, resolved as Beatnik_Test_CheckpointSeries does.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir_stream;
    dir_stream << ( scratch_env ? scratch_env : "." )
               << "/beatnik_unit_restart/" << ExecSpace::name() << "_np"
               << comm_size;
    const std::string root = dir_stream.str();
    const std::string dir = root + "/run";
    const std::string fresh_dir = root + "/fresh";
    const std::string prefix = "checkpoint";
    const std::string master = dir + "/" + prefix + ".xmf";
    const std::string index = dir + "/" + prefix + ".xmfindex";
    rec.note( "output directory " + root );
    if ( rank == 0 )
    {
        const std::string rm = "rm -rf '" + root + "'";
        if ( std::system( rm.c_str() ) != 0 )
            rec.note( "warning: could not clear " + root );
    }
    MPI_Barrier( comm );

    //-----------------------------------------------------------------------//
    // The written state: a level-2 cold start with non-trivial fields.
    //-----------------------------------------------------------------------//
    Beatnik::InitialConditionParams ic;
    ic.icosphere_subdivisions = 2;
    mesh_type mesh( comm );
    state_type state( Beatnik::StateModel::Potential );
    Beatnik::InitialCondition<ExecSpace, MemSpace>( ic ).build( mesh, state );
    {
        auto pos = mesh.positions();
        auto phi = mesh.potential();
        auto sv = mesh.sheetVector();
        auto mat = mesh.materialPosition();
        Kokkos::parallel_for(
            "restart_test_fields",
            Kokkos::RangePolicy<ExecSpace>( 0, mesh.totalVertexCount() ),
            KOKKOS_LAMBDA( const int i ) {
                const Real x = pos( i, 0 );
                const Real y = pos( i, 1 );
                const Real z = pos( i, 2 );
                phi( i ) = 0.3 * x - 1.7 * y * z + 0.01 * z;
                sv( i, 0 ) = y * z + 0.125;
                sv( i, 1 ) = -x * z;
                sv( i, 2 ) = 0.5 * x * y;
                mat( i, 0 ) = x + 0.03 * z;
                mat( i, 1 ) = y - 0.02 * x;
                mat( i, 2 ) = z + 0.01 * y;
            } );
        Kokkos::fence();
    }
    const Beatnik::AmrParams amr_params;
    amr_type amr( amr_params );
    amr.resetReferenceState( mesh );

    const long long nv0 = static_cast<long long>( mesh.globalVertexCount() );
    const long long ne0 = static_cast<long long>( mesh.globalEdgeCount() );
    const long long nf0 = static_cast<long long>( mesh.globalFaceCount() );
    const auto written = gatherRecords( comm, localRecords( mesh, true ) );
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( written.size() ), nv0 );

    // Two frames of one series: an earlier one and the restart point.
    io_type writer( comm, dir, prefix );
    const std::string early_path =
        writer.write( makeHeader( kEarlyTime, kEarlyStep ), mesh );
    const std::string path = writer.write( makeHeader( kTime, kStep ), mesh );

    //-----------------------------------------------------------------------//
    // 1. Round trip, into a directory with no series.
    //-----------------------------------------------------------------------//
    {
        mesh_type restored( comm );
        state_type restored_state( Beatnik::StateModel::Potential );
        io_type io( comm, fresh_dir, prefix );
        const Beatnik::RestartState start =
            reader_type::load( io, path, restored, restored_state );

        BEATNIK_CHECK_TRUE( rec, static_cast<double>( start.time ) == kTime );
        BEATNIK_CHECK_EQ( rec, start.step, kStep );
        BEATNIK_CHECK_TRUE(
            rec, static_cast<double>( start.initial_volume ) == kInitialVolume );
        BEATNIK_CHECK_TRUE( rec, static_cast<double>( start.initial_min_edge ) ==
                                     kInitialMinEdge );
        BEATNIK_CHECK_TRUE( rec, start.from_checkpoint );
        BEATNIK_CHECK_TRUE( rec, start.amr_reference_rebased );
        BEATNIK_CHECK_TRUE( rec, start.material_position_loaded );

        const Beatnik::CheckpointHeader header = io.readHeader( path );
        BEATNIK_CHECK_TRUE( rec, header.state_model ==
                                     Beatnik::StateModel::Potential );
        BEATNIK_CHECK_TRUE( rec, header.has_initial_volume );
        BEATNIK_CHECK_TRUE( rec, header.has_initial_min_edge );
        BEATNIK_CHECK_TRUE( rec, header.has_material_position );

        BEATNIK_CHECK_EQ(
            rec, static_cast<long long>( restored.globalVertexCount() ), nv0 );
        BEATNIK_CHECK_EQ(
            rec, static_cast<long long>( restored.globalEdgeCount() ), ne0 );
        BEATNIK_CHECK_EQ(
            rec, static_cast<long long>( restored.globalFaceCount() ), nf0 );
        BEATNIK_CHECK_EQ( rec, restored.haloDepth(), 2 );

        // Owned vertices partition the surface.
        long long owned = restored.ownedVertexCount();
        long long owned_sum = 0;
        MPI_Allreduce( &owned, &owned_sum, 1, MPI_LONG_LONG, MPI_SUM, comm );
        BEATNIK_CHECK_EQ( rec, owned_sum, nv0 );

        // Every local vertex, ghosts included, paired by gid, exactly.
        const auto local = localRecords( restored, false );
        long long missing = 0;
        long long differing = 0;
        for ( const auto& kv : local )
        {
            const auto it = written.find( kv.first );
            if ( it == written.end() )
            {
                ++missing;
                continue;
            }
            for ( int k = 0; k < kRecordWidth; ++k )
                if ( kv.second[k] != it->second[k] )
                {
                    ++differing;
                    break;
                }
        }
        BEATNIK_CHECK_EQ( rec, missing, 0 );
        BEATNIK_CHECK_EQ( rec, differing, 0 );
        rec.note( "round trip: " + std::to_string( local.size() ) +
                  " local vertices on this rank, " + std::to_string( owned ) +
                  " owned, compared exactly by gid" );

        // The startup checkpoint of a restart into a directory with no index
        // starts a new series there.
        io.write( makeHeader( kTime, kStep ), restored );
        if ( rank == 0 )
        {
            const std::string fresh_index =
                readFile( fresh_dir + "/" + prefix + ".xmfindex" );
            BEATNIK_CHECK_EQ( rec, countOccurrences( fresh_index, "\n" ), 1 );
            BEATNIK_CHECK_EQ(
                rec,
                countOccurrences( readFile( fresh_dir + "/" + prefix + ".xmf" ),
                                  "<Time Value=" ),
                1 );
        }
    }

    //-----------------------------------------------------------------------//
    // 2. The series in the writer's directory.
    //-----------------------------------------------------------------------//
    {
        // From the EARLIER frame: must throw naming the index, before readMesh,
        // and leave the master alone.
        const std::string master_before =
            ( rank == 0 ) ? readFile( master ) : std::string();
        bool threw = false;
        bool names_index = false;
        try
        {
            mesh_type restored( comm );
            state_type restored_state( Beatnik::StateModel::Potential );
            io_type io( comm, dir, prefix );
            reader_type::load( io, early_path, restored, restored_state );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            names_index = contains( e.what(), index ) &&
                          contains( e.what(), "--checkpoint-dir" );
            if ( rank == 0 )
                rec.note( "restart-from-earlier throw said: " +
                          std::string( e.what() ) );
        }
        BEATNIK_CHECK_TRUE( rec, threw );
        BEATNIK_CHECK_TRUE( rec, names_index );
        if ( rank == 0 )
            BEATNIK_CHECK_TRUE( rec, readFile( master ) == master_before );

        // From the LATEST frame: continues the series.
        mesh_type restored( comm );
        state_type restored_state( Beatnik::StateModel::Potential );
        io_type io( comm, dir, prefix );
        reader_type::load( io, path, restored, restored_state );
        // The startup checkpoint repeats the restart frame: not re-appended.
        const std::string again =
            io.write( makeHeader( kTime, kStep ), restored );
        BEATNIK_CHECK_TRUE( rec, again == path );
        io.write( makeHeader( kNextTime, kNextStep ), restored );
        if ( rank == 0 )
        {
            const std::string index_text = readFile( index );
            const std::string master_text = readFile( master );
            BEATNIK_CHECK_EQ( rec, countOccurrences( index_text, "\n" ), 3 );
            BEATNIK_CHECK_EQ( rec, countOccurrences( master_text, "<Time Value=" ),
                              3 );
            rec.note( "continued series index:\n" + index_text );
        }
    }

    //-----------------------------------------------------------------------//
    // 3. Failure directions. Every rank calls each one and must throw.
    //-----------------------------------------------------------------------//
    {
        const std::string missing_path = root + "/does_not_exist.h5";
        bool threw = false;
        bool names_path = false;
        try
        {
            mesh_type restored( comm );
            state_type restored_state( Beatnik::StateModel::Potential );
            io_type io( comm, fresh_dir, prefix );
            reader_type::load( io, missing_path, restored, restored_state );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            names_path = contains( e.what(), missing_path );
            if ( rank == 0 )
                rec.note( "missing-path throw said: " + std::string( e.what() ) );
        }
        BEATNIK_CHECK_TRUE( rec, threw );
        BEATNIK_CHECK_TRUE( rec, names_path );
        if ( rank == 0 && threw && names_path )
            std::printf( "[Beatnik_Test_Restart] PASS failure direction: "
                         "missing path throws std::runtime_error naming it\n" );
    }
    {
        // A Tessera-only file: the mesh and fields, no /beatnik group.
        const std::string stem = root + "/tessera_only";
        Tessera::writeMesh( mesh.tesseraMesh(), stem );
        MPI_Barrier( comm );
        bool threw = false;
        bool names_group = false;
        try
        {
            mesh_type restored( comm );
            state_type restored_state( Beatnik::StateModel::Potential );
            io_type io( comm, fresh_dir, prefix );
            reader_type::load( io, stem + ".h5", restored, restored_state );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            names_group = contains( e.what(), "/beatnik" ) &&
                          contains( e.what(), stem + ".h5" );
            if ( rank == 0 )
                rec.note( "no-/beatnik throw said: " + std::string( e.what() ) );
        }
        BEATNIK_CHECK_TRUE( rec, threw );
        BEATNIK_CHECK_TRUE( rec, names_group );
        if ( rank == 0 && threw && names_group )
            std::printf( "[Beatnik_Test_Restart] PASS failure direction: "
                         "a file with no /beatnik group throws naming "
                         "/beatnik\n" );
    }
    {
        bool threw = false;
        bool names_both = false;
        try
        {
            mesh_type restored( comm );
            state_type restored_state( Beatnik::StateModel::SheetVector );
            io_type io( comm, fresh_dir, prefix );
            reader_type::load( io, path, restored, restored_state );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            names_both = contains( e.what(), "'potential'" ) &&
                         contains( e.what(), "'sheet-vector'" );
            if ( rank == 0 )
                rec.note( "state-model throw said: " + std::string( e.what() ) );
        }
        BEATNIK_CHECK_TRUE( rec, threw );
        BEATNIK_CHECK_TRUE( rec, names_both );
        if ( rank == 0 && threw && names_both )
            std::printf( "[Beatnik_Test_Restart] PASS failure direction: "
                         "a state-model mismatch throws naming both\n" );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_Restart" );
        try
        {
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

    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
