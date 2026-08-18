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
 * @file Beatnik_Test_CheckpointSeries.cpp
 * @brief `unit`-tier test for grouped-io T2: the master XDMF temporal
 *        collection `CheckpointIO::write` maintains.
 *
 * THIS IS T2's EXIT CRITERION, MECHANIZED, and it exists to take Paraview out
 * of the loop. Risk R1 in `tasks/grouped-io.md` is that the master opens and
 * shows one timestep, which looks like a broken file and is almost always the
 * reader — only the *temporal* XDMF3 reader walks a collection. A visual check
 * therefore cannot distinguish "the light data is wrong" from "the reader was
 * wrong", so this test asserts on the emitted **text** instead.
 *
 * WHAT IS AND IS NOT UNDER TEST
 * -----------------------------
 * `CheckpointIO` is driven **directly**, not through `Solver`: the subject is
 * the light data one `write()` call emits and the state it keeps between calls,
 * and the solver loop is not part of that. Nothing is asserted about field
 * *values* — the vertex user pack is never initialized here, so `u0`/`u1`/`u2`
 * hold whatever was in memory. `AdaptiveMesh::resetReferenceState` IS called,
 * because Tessera's face AoSoA is allocated uninitialized and writing it from
 * uninitialized memory is undefined rather than merely arbitrary.
 *
 * THE TWO FAILURE DIRECTIONS, and why both are needed
 * ---------------------------------------------------
 * The rule under test is three-way (see the GROUPED OUTPUT section of
 * `src/Beatnik_IOInterface.hpp`), so a test that only checked the happy path
 * would pass against an implementation that had either of the other two wrong:
 *
 *   (a) A *decreasing* time must throw, and the message must name **both**
 *       stems — "something threw" is satisfied by an unrelated abort.
 *   (b) An *equal* time with an *equal* stem must NOT append a fourth timestep.
 *       That is `Solver::finalize()`'s shape: it re-writes the last finite
 *       state at the same `(time, step)` as the previous checkpoint.
 *
 * Direction (a) is the guard doing its job and (b) is the equal-time rule doing
 * its job. Both additionally assert the master is left **byte-unchanged**, so a
 * rejected or non-appended frame cannot have half-rewritten it. Reverting the
 * equal-time branch in `write()` to an unconditional `_series.write()` is what
 * this test is calibrated against: it then fails with Tessera's
 * `time must be strictly increasing` message rather than passing.
 *
 * Exit code 0 iff every check passes; see `Beatnik_TestAssert.hpp` for why the
 * verdict is the binary's own and not ctest's.
 */

#include <Beatnik_AdaptiveMesh.hpp>
#include <Beatnik_IOInterface.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The mesh. Subdivision 2 is the project default, and nothing here depends on
// the size -- the subject is the light data, which names counts it does not
// interpret.
//---------------------------------------------------------------------------//
constexpr int kSubdivisions = 2;
constexpr Real kRadius = 0.25;

/// The three frames written in increasing time, plus the fourth call that
/// repeats the third exactly -- `Solver::finalize()`'s shape.
constexpr double kFrameTimes[3] = { 0.0, 0.1, 0.2 };
constexpr long long kFrameSteps[3] = { 0, 1, 2 };

/// The `<Time Value=>` text is written with `%g`, so it is not a fixed string
/// to match: parse the number and compare. Loose on purpose -- `%g`'s six
/// significant digits are the only precision loss in play.
constexpr double kTimeTolerance = 1.0e-9;

//---------------------------------------------------------------------------//
// Small file helpers. Rank 0 only, so no MPI-IO and no Tessera involvement.
//---------------------------------------------------------------------------//

bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// Whole-file slurp. Returns "" for a missing file, which every caller then
/// fails on through an emptiness check rather than by aborting here.
std::string readFile( const std::string& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return std::string();
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/// Number of non-overlapping occurrences of `needle` in `haystack`.
int countOccurrences( const std::string& haystack, const std::string& needle )
{
    if ( needle.empty() )
        return 0;
    int n = 0;
    std::size_t at = haystack.find( needle );
    while ( at != std::string::npos )
    {
        ++n;
        at = haystack.find( needle, at + needle.size() );
    }
    return n;
}

/// Every `<Time Value="X"/>` value in the file, in document order.
std::vector<double> collectTimeValues( const std::string& text )
{
    const std::string open = "<Time Value=\"";
    std::vector<double> values;
    std::size_t at = text.find( open );
    while ( at != std::string::npos )
    {
        const std::size_t start = at + open.size();
        const std::size_t end = text.find( '"', start );
        if ( end == std::string::npos )
            break;
        values.push_back( std::atof( text.substr( start, end - start ).c_str() ) );
        at = text.find( open, end );
    }
    return values;
}

/// The document-order block of the `i`-th child `<Grid>` of the collection:
/// from its own `<Time Value=` to the next one, or to end of file for the last.
/// This is how "each child names only its own frame" is checked -- a whole-file
/// substring search cannot tell a correct master from one whose every child
/// names frame 0.
std::string childBlock( const std::string& text, int i )
{
    const std::string open = "<Time Value=\"";
    std::size_t at = text.find( open );
    for ( int k = 0; k < i && at != std::string::npos; ++k )
        at = text.find( open, at + open.size() );
    if ( at == std::string::npos )
        return std::string();
    const std::size_t next = text.find( open, at + open.size() );
    return next == std::string::npos ? text.substr( at )
                                     : text.substr( at, next - at );
}

/// Trailing path component, for the basenames the master references.
std::string baseName( const std::string& path )
{
    const std::size_t slash = path.find_last_of( '/' );
    return slash == std::string::npos ? path : path.substr( slash + 1 );
}

/// Strip the trailing `.h5` from what `write()` returns, giving the STEM.
/// `write()` returns a path and `CheckpointIO` names stems internally, so the
/// two are not interchangeable in a substring search -- which is exactly the way
/// the first version of this test failed.
std::string stemOf( const std::string& h5_path )
{
    const std::string ext = ".h5";
    if ( h5_path.size() >= ext.size() &&
         h5_path.compare( h5_path.size() - ext.size(), ext.size(), ext ) == 0 )
        return h5_path.substr( 0, h5_path.size() - ext.size() );
    return h5_path;
}

//---------------------------------------------------------------------------//

template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec )
{
    using mesh_type = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using amr_type = Beatnik::AdaptiveMesh<ExecSpace, MemSpace>;
    using io_type = Beatnik::CheckpointIO<ExecSpace, MemSpace>;

    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    //-----------------------------------------------------------------------//
    // Output directory.
    //
    // Resolution order, and why there are three levels: the installed path runs
    // from the manifest's directory, which is inside a spack install prefix and
    // is READ-ONLY -- so a relative default would fail there and nowhere else.
    // `BEATNIK_TEST_SCRATCH` is what the batch wrapper's submitting environment
    // sets (absolute, and it must name a PARALLEL filesystem: the checkpoints go
    // through MPI-IO); TMPDIR covers a hand-run from an install prefix; "."
    // covers ctest, which runs in the build tree. Unique per
    // (execution space, rank count) so two cases cannot write the same file.
    //-----------------------------------------------------------------------//
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir_stream;
    dir_stream << ( scratch_env ? scratch_env : "." )
               << "/beatnik_unit_checkpoint_series/" << ExecSpace::name()
               << "_np" << comm_size;
    const std::string dir = dir_stream.str();
    const std::string prefix = "checkpoint";
    const std::string master = dir + "/" + prefix + ".xmf";
    rec.note( "output directory " + dir );

    // Start from an empty directory. Tessera APPENDS to `<master>.xmfindex`, and
    // the master itself is rewritten rather than truncated-and-grown, so a
    // leftover directory from a previous run makes both counts unreadable. Rank
    // 0 only, then a barrier, for the same reason `write`'s own mkdir is.
    if ( rank == 0 )
    {
        const std::string rm = "rm -rf '" + dir + "'";
        if ( std::system( rm.c_str() ) != 0 )
            rec.note( "warning: could not clear " + dir );
    }
    MPI_Barrier( MPI_COMM_WORLD );

    //-----------------------------------------------------------------------//
    // The mesh, built without a solver.
    //-----------------------------------------------------------------------//
    mesh_type mesh( MPI_COMM_WORLD );
    const Real center[3] = { 0.0, 0.0, 0.25 };
    mesh.generateIcosphere( kSubdivisions, kRadius, center );

    // Initializes the two reference face fields and clears the mark. Tessera's
    // face AoSoA is allocated uninitialized and `writeMesh` writes the whole
    // face user pack, so without this the frames would be written from
    // uninitialized memory.
    const Beatnik::AmrParams amr_params;
    amr_type amr( amr_params );
    amr.resetReferenceState( mesh );

    //-----------------------------------------------------------------------//
    // Four writes: three increasing times, then the exact repeat of the third.
    // COLLECTIVE -- every rank calls every one of them, in order.
    //-----------------------------------------------------------------------//
    io_type io( MPI_COMM_WORLD, dir, prefix );

    std::vector<std::string> frame_paths;
    for ( int i = 0; i < 3; ++i )
    {
        Beatnik::CheckpointHeader header;
        header.state_model = Beatnik::StateModel::Potential;
        header.time = static_cast<Real>( kFrameTimes[i] );
        header.step = kFrameSteps[i];
        header.has_material_position = true;
        frame_paths.push_back( io.write( header, mesh ) );
    }

    // The fourth call: `Solver::finalize()`'s re-write of the last finite state,
    // which carries the SAME (time, step) as the previous checkpoint whenever
    // the last accepted step also checkpointed. This must not throw and must not
    // add a timestep.
    std::string equal_time_path;
    {
        Beatnik::CheckpointHeader header;
        header.state_model = Beatnik::StateModel::Potential;
        header.time = static_cast<Real>( kFrameTimes[2] );
        header.step = kFrameSteps[2];
        header.has_material_position = true;
        equal_time_path = io.write( header, mesh );
    }
    // Same `(time, step)` means the same stem, which is exactly what the
    // equal-time branch requires; asserted rather than assumed, because the
    // branch is unreachable if the stem construction ever changes.
    BEATNIK_CHECK_TRUE( rec, equal_time_path == frame_paths[2] );

    //-----------------------------------------------------------------------//
    // Direction (a): a DECREASING time must throw, and the message must name
    // both stems. Recorded against the master's bytes, so a rejected frame is
    // also shown to have left the master alone.
    //
    // Every rank calls it and every rank must throw -- Tessera validates on
    // every rank precisely so the throw is symmetric and cannot deadlock, and
    // Beatnik's own guard runs before any I/O for the same reason. So this is
    // NOT rank-0-only: a rank that did not throw would have gone on to a
    // collective write with no partner.
    //-----------------------------------------------------------------------//
    const std::string master_before = ( rank == 0 ) ? readFile( master )
                                                    : std::string();

    bool threw = false;
    bool message_names_both = false;
    {
        Beatnik::CheckpointHeader header;
        header.state_model = Beatnik::StateModel::Potential;
        header.time = static_cast<Real>( kFrameTimes[0] ); // backwards
        header.step = kFrameSteps[0];
        header.has_material_position = true;
        try
        {
            io.write( header, mesh );
        }
        catch ( const std::runtime_error& e )
        {
            threw = true;
            const std::string what( e.what() );
            // Both stems, not merely "something threw": the guard's whole value
            // is that it says which two frames disagreed. STEMS, not the paths
            // `write()` returns -- the message names what CheckpointIO handed
            // Tessera, which carries no extension.
            message_names_both =
                what.find( baseName( stemOf( frame_paths[0] ) ) ) !=
                    std::string::npos &&
                what.find( baseName( stemOf( frame_paths[2] ) ) ) !=
                    std::string::npos;
            if ( rank == 0 )
                rec.note( "decreasing-time throw said: " + what );
        }
    }
    BEATNIK_CHECK_TRUE( rec, threw );
    BEATNIK_CHECK_TRUE( rec, message_names_both );

    //-----------------------------------------------------------------------//
    // The text assertions. RANK 0 ONLY -- everything above was collective and
    // had to run everywhere; reading a file does not.
    //-----------------------------------------------------------------------//
    if ( rank == 0 )
    {
        BEATNIK_CHECK_TRUE( rec, fileExists( master ) );
        const std::string text = readFile( master );
        BEATNIK_CHECK_TRUE( rec, !text.empty() );

        // The rejected frame left the master byte-unchanged.
        BEATNIK_CHECK_TRUE( rec, text == master_before );

        // Exactly one temporal collection. Two would mean a master that had
        // been appended to rather than rewritten.
        BEATNIK_CHECK_EQ( rec,
                          countOccurrences( text,
                                            "CollectionType=\"Temporal\"" ),
                          1 );

        // THREE timesteps, not four: direction (b). The fourth `write()` above
        // repeated `(0.2, step 2)` exactly, and the master already named that
        // frame at that time.
        const std::vector<double> times = collectTimeValues( text );
        BEATNIK_CHECK_EQ( rec, static_cast<long long>( times.size() ), 3 );
        BEATNIK_CHECK_EQ( rec, countOccurrences( text, "<Time Value=" ), 3 );

        // One `<Topology` and one `<Geometry` per child. Repeating topology per
        // child is deliberate in Tessera -- an adaptively remeshing series
        // genuinely changes Nv/Nf frame to frame -- so three is correct and one
        // would mean a hoisted-into-Domain master that readers break on.
        BEATNIK_CHECK_EQ( rec, countOccurrences( text, "<Topology" ), 3 );
        BEATNIK_CHECK_EQ( rec, countOccurrences( text, "<Geometry" ), 3 );

        if ( times.size() == 3 )
        {
            std::ostringstream os;
            os.precision( 17 );
            os << "master times " << times[0] << ", " << times[1] << ", "
               << times[2];
            rec.note( os.str() );
            for ( int i = 0; i < 3; ++i )
                BEATNIK_CHECK_CLOSE( rec, times[i] - kFrameTimes[i], 0.0,
                                     kTimeTolerance );
        }

        // Each child names ONLY its own frame. A whole-file substring search
        // would pass on a master whose every child named frame 0, so the check
        // is per child block: its own basename present, and neither sibling's.
        for ( int i = 0; i < 3; ++i )
        {
            const std::string block = childBlock( text, i );
            BEATNIK_CHECK_TRUE( rec, !block.empty() );
            for ( int j = 0; j < 3; ++j )
            {
                const std::string h5 = baseName( frame_paths[j] );
                const bool present = block.find( h5 ) != std::string::npos;
                BEATNIK_CHECK_TRUE( rec, present == ( i == j ) );
            }
        }

        // Every frame pair is still on disk, unchanged in layout: the master is
        // an addition, not a replacement.
        for ( int i = 0; i < 3; ++i )
        {
            const std::string stem = stemOf( frame_paths[i] );
            BEATNIK_CHECK_TRUE( rec, fileExists( stem + ".h5" ) );
            BEATNIK_CHECK_TRUE( rec, fileExists( stem + ".xmf" ) );
        }

        // Tessera's restart record, one line per APPENDED frame -- so it is the
        // second, independent witness that the equal-time call did not append.
        const std::string index_text = readFile( dir + "/" + prefix +
                                                ".xmfindex" );
        BEATNIK_CHECK_EQ( rec, countOccurrences( index_text, "\n" ), 3 );

        // The `_latest` alias deliberately still names the newest FRAME, not
        // the master: it is half of the pair a restart consumes.
        BEATNIK_CHECK_TRUE( rec,
                            fileExists( dir + "/" + prefix + "_latest.h5" ) );
        BEATNIK_CHECK_TRUE( rec,
                            fileExists( dir + "/" + prefix + "_latest.xmf" ) );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_CheckpointSeries" );
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
