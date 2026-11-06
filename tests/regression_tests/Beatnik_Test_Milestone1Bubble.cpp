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
 * @file Beatnik_Test_Milestone1Bubble.cpp
 * @brief **MILESTONE 1** — the reference's default rising-bubble configuration
 *        (M1-G2's command line, every remesh pass and the isotropic cleanup
 *        live) run for 600 steps and compared against the M1-G2 Python gold
 *        set at every checkpointed step.
 *
 * A MEMBER OF THE `milestone` TIER, NOT OF THE SHIP GATE. It is registered in
 * `BEATNIK_MILESTONE_TEST_SOURCES` (`tests/CMakeLists.txt`) at ranks 1 and 4
 * on every backend, and the 60-launch `regression` gate does not run it.
 *
 * WHAT IT ASSERTS
 * ---------------
 *   1. **The comparison.** Every checkpointed step, 0 through 600 every 10
 *      (61 files), is compared against the gold file for the same step by
 *      `compare_output.py`, at that step's entry in the compiled-in tolerance
 *      table `kTolerance` (initialized to the comparator's own defaults,
 *      `--rtol 1e-10 --atol 1e-12`, at every step). The run **continues past
 *      a failing step**: every step is compared and reported, and the test
 *      fails at the end if any step failed. The comparator's exit status is
 *      read exactly: 0 is a match, 1 is a detected mismatch, and anything else
 *      is a DEFECT (a load or path error) that is never a pass and never a
 *      mismatch.
 *   2. **Liveness (risk M1-R7).** By the last step the cumulative remesh
 *      counters `splits`, `collapses`, `flips` and `smooth_steps` and
 *      `Solver::cleanupFlips()` are all non-zero, and at every checkpointed
 *      step Beatnik's `time` is within `kTimeLivenessRtol` of the gold's. A
 *      fixed-dt run is off by a factor of about 3 at step 10 and fails that;
 *      the tight per-step `time` agreement is the comparator's, through
 *      `FLOAT_SCALARS`, at the table's tolerance.
 *   3. **The budget.** The run reaches step 600; a non-finite early stop is a
 *      failure, not a shorter run.
 *
 * Edit sets are never compared against the reference (risk R7): Beatnik's
 * accepted flip and collapse sets are independent sets, not the reference's
 * serial sweeps, and its valence flips accept flips the reference rejects
 * (`tasks/milestone1.md`, Deliberate deviations). The cleanup flip total is
 * therefore asserted non-zero and printed, never compared with Python.
 *
 * WHAT IT PRINTS, FOR THE READER OF THE OUTPUT (rank 0 only)
 * ---------------------------------------------------------
 * One line per step, the face-count trajectory and the five counters:
 *
 *     [m1traj] step=<n> time=<t> V=<global V> F=<global F> splits=<cum>
 *       collapses=<cum> flips=<cum> smooth_steps=<cum> cleanup_flips=<cum>
 *       elapsed=<s>
 *
 * One line per checkpointed step, the comparison:
 *
 *     [m1step] step=<n> verdict=<match|mismatch|DEFECT> exit=<status>
 *       rtol=<r> atol=<a> time=<Beatnik t> gold_time=<gold t>
 *       time_rel=<|t/t_gold - 1|> V_cpp=<V> F_cpp=<F> V_gold=<V> F_gold=<F>
 *       ambiguous=<cpp>/<gold> faces_identical=<k>/<F>
 *       max_e[<field>]=<max|e|> ... cmp_s=<seconds in the comparator>
 *
 * (one line each in the output; wrapped here). `V_cpp`/`F_cpp` and
 * `V_gold`/`F_gold` are the two FILES' row counts as the comparator read them
 * (its `structure:` line), `-1` where it printed none. `max_e[<field>]` is
 * repeated once per per-vertex field the comparator reported (`vertices`, the
 * solution field, `remesh_material_position`), plus `max_e[time]`; a field
 * the comparator did not reach (it stops at a count mismatch) is absent. The
 * comparator's full report for the step follows, every line prefixed
 * `[cmp <step>] `.
 *
 * Closing summary:
 *
 *     [m1final] mode=<mode> space=<space> np=<N> steps=<completed>/<budget>
 *       wall=<s> cmp_wall=<s> run_wall=<s> s_per_step=<run_wall/steps>
 *       splits=<n> collapses=<n> flips=<n> smooth_steps=<n> cleanup_flips=<n>
 *       compared=<n> matched=<n> mismatched=<n> defects=<n>
 *       first_failing_step=<n or none> time_liveness_failures=<n>
 *
 * `cmp_wall` is the time rank 0 spent in comparator subprocesses and
 * `run_wall = wall - cmp_wall`.
 *
 * FAILURE DIRECTIONS, selected by the third argument
 * --------------------------------------------------
 *   `fail-step0`       runs to the first checkpoint after step 0 (step 10)
 *                      and compares it against the **step-0** gold. Passes iff
 *                      the comparator exits exactly 1: a detected mismatch,
 *                      not a load error.
 *   `fail-no-cleanup`  runs with `--no-isotropic-cleanup` forced and compares
 *                      each checkpoint until one mismatches. Passes iff a step
 *                      exits exactly 1 before any exits other than 0 or 1,
 *                      and `cleanupFlips()` stays 0: the comparison fails
 *                      rather than passing with a different mesh.
 *
 * Both print the same `[m1traj]`, `[m1step]` and `[m1final]` lines.
 *
 * ARGUMENTS
 *
 *   argv[1]  the M1-G2 gold DIRECTORY (61 numbered .npz files)
 *              regression_tests/milestone1-bubble/gold
 *   argv[2]  the comparator
 *              regression_tests/compare_output.py
 *   argv[3]  optional: `compare` (the default), `fail-step0` or
 *            `fail-no-cleanup`
 *
 * `BEATNIK_PYTHON` overrides the interpreter (default `python3`). Checkpoints
 * go to `${BEATNIK_TEST_SCRATCH}/beatnik_milestone1/<space>_np<N>_<mode>`,
 * which **must** be on a parallel filesystem: they go through MPI-IO.
 */

#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_Milestone1Params.hpp"
#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace M1 = Beatnik::Test::Milestone1;

constexpr int kSteps = M1::kSteps;
constexpr int kEvery = M1::kCheckpointEvery;
constexpr int kCheckpoints = kSteps / kEvery + 1;

//---------------------------------------------------------------------------//
// The gold set, read from the 61 numbered `.npz` files of
// `regression_tests/milestone1-bubble/gold` (M1-G2): each file's `step`,
// `time` (all 17 digits) and the row counts of `vertices` and `faces`. The
// counts match the gold README's table.
//---------------------------------------------------------------------------//
struct GoldRow
{
    int step;
    double time;
    long long vertices;
    long long faces;
};

constexpr GoldRow kGold[kCheckpoints] = {
    { 0, 0.0, 162, 320 },
    { 10, 0.01101694018526759, 547, 1090 },
    { 20, 0.019424763852420666, 544, 1084 },
    { 30, 0.02764722148239303, 542, 1080 },
    { 40, 0.03573164737334241, 542, 1080 },
    { 50, 0.043733409842731615, 542, 1080 },
    { 60, 0.05251582547660846, 539, 1074 },
    { 70, 0.06221093104373272, 539, 1074 },
    { 80, 0.07183002288733918, 539, 1074 },
    { 90, 0.08143892400437756, 539, 1074 },
    { 100, 0.09104341360360306, 539, 1074 },
    { 110, 0.10064342203837187, 539, 1074 },
    { 120, 0.11023834854746112, 539, 1074 },
    { 130, 0.11982763912652993, 539, 1074 },
    { 140, 0.12941080709248845, 539, 1074 },
    { 150, 0.13898740262175938, 539, 1074 },
    { 160, 0.14855699074582313, 539, 1074 },
    { 170, 0.15811913921607063, 539, 1074 },
    { 180, 0.16767341244660672, 539, 1074 },
    { 190, 0.1772193688486418, 539, 1074 },
    { 200, 0.18675656001476051, 539, 1074 },
    { 210, 0.19628453089236317, 539, 1074 },
    { 220, 0.20580282046155707, 539, 1074 },
    { 230, 0.21531096264220165, 539, 1074 },
    { 240, 0.22480848727442135, 539, 1074 },
    { 250, 0.23429687658476397, 538, 1072 },
    { 260, 0.2437986970251245, 538, 1072 },
    { 270, 0.25331585545466995, 538, 1072 },
    { 280, 0.262837186151417, 538, 1072 },
    { 290, 0.27235592809456655, 538, 1072 },
    { 300, 0.2818680330636531, 538, 1072 },
    { 310, 0.291370834445422, 538, 1072 },
    { 320, 0.3008624475279937, 539, 1074 },
    { 330, 0.31034117866526945, 538, 1072 },
    { 340, 0.31979159982870725, 538, 1072 },
    { 350, 0.3292045212469648, 538, 1072 },
    { 360, 0.3385894509439249, 538, 1072 },
    { 370, 0.34795109396476026, 538, 1072 },
    { 380, 0.35729099879835297, 538, 1072 },
    { 390, 0.3666096117617827, 538, 1072 },
    { 400, 0.37590703622444366, 540, 1076 },
    { 410, 0.3851832128835166, 540, 1076 },
    { 420, 0.394438083741857, 540, 1076 },
    { 430, 0.40367158822170385, 540, 1076 },
    { 440, 0.4128833722890705, 540, 1076 },
    { 450, 0.42207287576577107, 540, 1076 },
    { 460, 0.43123952411741784, 540, 1076 },
    { 470, 0.4403828103832484, 540, 1076 },
    { 480, 0.44950230379895273, 540, 1076 },
    { 490, 0.45859763183337793, 540, 1076 },
    { 500, 0.46766845859279943, 540, 1076 },
    { 510, 0.476714467594406, 540, 1076 },
    { 520, 0.48573535026743064, 540, 1076 },
    { 530, 0.49473079924420826, 540, 1076 },
    { 540, 0.5037005050477837, 540, 1076 },
    { 550, 0.5126441549644762, 540, 1076 },
    { 560, 0.5215614332136723, 540, 1076 },
    { 570, 0.5304520218203523, 540, 1076 },
    { 580, 0.5396080560065922, 539, 1074 },
    { 590, 0.5488465473588036, 538, 1072 },
    { 600, 0.5579341242218034, 538, 1072 },
};

//---------------------------------------------------------------------------//
// The per-step tolerance table.
//
// One entry per checkpointed step, passed to the comparator as
// `--rtol` / `--atol`. Every entry is the comparator's default; task M1-A1j
// edits the entries (with the measured `max|e|` beside each as a comment),
// never this structure. **Do not loosen** an entry without that measurement
// (tasks/milestone1.md, risk M1-R1).
//---------------------------------------------------------------------------//
struct Tolerance
{
    double rtol;
    double atol;
};

constexpr Tolerance kDefaultTolerance = { 1.0e-10, 1.0e-12 };

Tolerance toleranceForStep( int /*step*/ ) { return kDefaultTolerance; }

/// Liveness bound on Beatnik's `time` against the gold's at a checkpointed
/// step, as `|t / t_gold - 1|`. Wide on purpose: it separates the adaptive
/// step size from a fixed one, while a mesh that differs from the
/// reference's moves the adaptive dt through its minimum edge, which is a
/// measurement, not a liveness failure. A fixed `--dt 0.003` is off by 1.72 at
/// step 10 (`t = 0.03` against the gold's `0.0110`), 2.43 at step 50 and 2.23
/// at step 600; Beatnik's adaptive run at the milestone defaults was 0.16
/// ahead at step 10 and 0.34 at step 50 (task T4d6's driver output). The
/// tight `time` agreement is the comparator's.
constexpr double kTimeLivenessRtol = 1.0;

//---------------------------------------------------------------------------//
bool fileExists( const std::string& path )
{
    struct stat sb;
    return ::stat( path.c_str(), &sb ) == 0;
}

/// The gold file for `step`, found by its `_step%07d.npz` suffix rather than by
/// rebuilding the name from a time, because the time is under test.
std::string goldForStep( const std::string& directory, long long step )
{
    char suffix[32];
    std::snprintf( suffix, sizeof( suffix ), "_step%07lld.npz", step );
    const std::string want( suffix );

    DIR* dir = ::opendir( directory.c_str() );
    if ( !dir )
        return std::string();

    std::string found;
    while ( struct dirent* entry = ::readdir( dir ) )
    {
        const std::string name( entry->d_name );
        if ( name.size() >= want.size() &&
             name.compare( name.size() - want.size(), want.size(), want ) == 0 )
        {
            found = directory + "/" + name;
            break;
        }
    }
    ::closedir( dir );
    return found;
}

//---------------------------------------------------------------------------//
/// What one comparator run reported.
struct Comparison
{
    /// Exit status; -1 if the subprocess could not be run or did not exit.
    int status = -1;
    long long v_cpp = -1;
    long long f_cpp = -1;
    long long v_gold = -1;
    long long f_gold = -1;
    std::string ambiguous = "n/a";
    std::string faces_identical = "n/a";
    /// `max_e[<field>]=<value>` pairs, in the comparator's order.
    std::vector<std::pair<std::string, std::string>> max_e;
    double seconds = 0.0;
};

/// Parse one line of `compare_output.py`'s full (not `--quiet`) report into
/// `c`. Lines the parser does not recognize are left alone.
void parseComparatorLine( const std::string& line, Comparison& c )
{
    long long a = 0, b = 0, d = 0, e = 0, f = 0, g = 0;
    // structure: vertices cpp=(547, 3) gold=(547, 3); faces cpp=(1090, 3) ...
    if ( std::sscanf( line.c_str(),
                      "structure: vertices cpp=(%lld, %lld) gold=(%lld, %lld); "
                      "faces cpp=(%lld, %lld) gold=(%lld, ",
                      &a, &b, &d, &e, &f, &g, &c.f_gold ) == 7 )
    {
        c.v_cpp = a;
        c.v_gold = d;
        c.f_cpp = f;
        return;
    }
    // matching (eps=1e-09): 538/538 unambiguous, ambiguous cpp=0 gold=0
    const std::size_t amb = line.find( "ambiguous cpp=" );
    if ( line.compare( 0, 9, "matching " ) == 0 && amb != std::string::npos )
    {
        long long ac = 0, ag = 0;
        if ( std::sscanf( line.c_str() + amb, "ambiguous cpp=%lld gold=%lld",
                          &ac, &ag ) == 2 )
            c.ambiguous = std::to_string( ac ) + "/" + std::to_string( ag );
        return;
    }
    // faces: 1070/1072 triangles identical after remap + ...
    if ( line.compare( 0, 7, "faces: " ) == 0 &&
         line.find( "triangles identical" ) != std::string::npos )
    {
        char frac[64] = { 0 };
        if ( std::sscanf( line.c_str(), "faces: %63s", frac ) == 1 )
            c.faces_identical = frac;
        return;
    }
    // "  <field>   n=<N>   max|e|=<value>  ..." -- a per-vertex field.
    // "  time      cpp=<v> gold=<v> |e|=<value>" -- the float scalar.
    char name[128] = { 0 };
    char value[64] = { 0 };
    const std::size_t pos_e = line.find( "max|e|=" );
    if ( line.compare( 0, 2, "  " ) == 0 && pos_e != std::string::npos &&
         line.find( " n=" ) != std::string::npos &&
         std::sscanf( line.c_str(), " %127s", name ) == 1 &&
         std::sscanf( line.c_str() + pos_e, "max|e|=%63s", value ) == 1 )
    {
        c.max_e.emplace_back( name, value );
        return;
    }
    const std::size_t pos_t = line.find( " |e|=" );
    if ( line.compare( 0, 2, "  " ) == 0 && pos_t != std::string::npos &&
         line.find( " cpp=" ) != std::string::npos &&
         std::sscanf( line.c_str(), " %127s", name ) == 1 &&
         std::strcmp( name, "time" ) == 0 &&
         std::sscanf( line.c_str() + pos_t, " |e|=%63s", value ) == 1 )
    {
        c.max_e.emplace_back( name, value );
    }
}

/// Run `python <script> <cpp> <gold> --rtol .. --atol ..` with the full
/// report, echo it prefixed `[cmp <step>] `, and return what it reported. The
/// exit status is returned verbatim: 0 match, 1 mismatch, anything else a
/// defect. Rank 0 only.
Comparison runComparator( const std::string& python, const std::string& script,
                          const std::string& cpp, const std::string& gold,
                          const Tolerance& tol, int step )
{
    Comparison c;
    char tol_str[96];
    std::snprintf( tol_str, sizeof( tol_str ), " --rtol %.17g --atol %.17g",
                   tol.rtol, tol.atol );
    std::ostringstream cmd;
    cmd << "'" << python << "' '" << script << "' '" << cpp << "' '" << gold
        << "'" << tol_str << " 2>&1";
    std::printf( "[cmd] %s\n", cmd.str().c_str() );
    std::fflush( stdout );

    const double t0 = MPI_Wtime();
    std::FILE* pipe = ::popen( cmd.str().c_str(), "r" );
    if ( !pipe )
    {
        c.seconds = MPI_Wtime() - t0;
        return c;
    }
    char buf[4096];
    while ( std::fgets( buf, sizeof( buf ), pipe ) )
    {
        std::string line( buf );
        while ( !line.empty() &&
                ( line.back() == '\n' || line.back() == '\r' ) )
            line.pop_back();
        std::printf( "[cmp %d] %s\n", step, line.c_str() );
        parseComparatorLine( line, c );
    }
    std::fflush( stdout );
    const int raw = ::pclose( pipe );
    c.seconds = MPI_Wtime() - t0;
    if ( raw == -1 || !WIFEXITED( raw ) )
        c.status = -1;
    else
        c.status = WEXITSTATUS( raw );
    return c;
}

//---------------------------------------------------------------------------//
enum class Mode
{
    Compare,
    FailStep0,
    FailNoCleanup
};

const char* modeName( Mode m )
{
    switch ( m )
    {
    case Mode::Compare:
        return "compare";
    case Mode::FailStep0:
        return "fail-step0";
    case Mode::FailNoCleanup:
        return "fail-no-cleanup";
    }
    return "?";
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runChecks( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    rec.note( std::string( "execution space " ) + ExecSpace::name() +
              ", ranks " + std::to_string( comm_size ) );

    if ( argc < 3 )
    {
        rec.fail( "usage: <gold-dir> <compare_output.py> "
                  "[compare|fail-step0|fail-no-cleanup]; see the ARGUMENTS "
                  "block in this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const std::string gold_dir = argv[1];
    const std::string script = argv[2];
    Mode mode = Mode::Compare;
    if ( argc >= 4 )
    {
        const std::string m = argv[3];
        if ( m == "compare" )
            mode = Mode::Compare;
        else if ( m == "fail-step0" )
            mode = Mode::FailStep0;
        else if ( m == "fail-no-cleanup" )
            mode = Mode::FailNoCleanup;
        else
        {
            rec.fail( "unknown mode '" + m +
                      "' (expected compare, fail-step0 or fail-no-cleanup)" );
            return;
        }
    }
    const char* python_env = std::getenv( "BEATNIK_PYTHON" );
    const std::string python = python_env ? python_env : "python3";

    // Every input path is checked BEFORE it is used, so a mis-plumbed path is
    // reported as itself rather than as a comparison failure. The verdict is
    // broadcast so every rank stops together.
    int inputs_ok = 1;
    if ( rank == 0 )
    {
        if ( !fileExists( gold_dir ) )
        {
            rec.fail( "gold directory not found: " + gold_dir );
            inputs_ok = 0;
        }
        if ( !fileExists( script ) )
        {
            rec.fail( "comparator not found: " + script );
            inputs_ok = 0;
        }
        for ( int i = 0; i < kCheckpoints && inputs_ok; ++i )
        {
            if ( goldForStep( gold_dir, kGold[i].step ).empty() )
            {
                rec.fail( "no gold file for step " +
                          std::to_string( kGold[i].step ) + " in " +
                          gold_dir );
                inputs_ok = 0;
            }
        }
    }
    MPI_Bcast( &inputs_ok, 1, MPI_INT, 0, MPI_COMM_WORLD );
    if ( !inputs_ok )
        return;

    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/beatnik_milestone1/"
        << ExecSpace::name() << "_np" << comm_size << "_" << modeName( mode );

    Beatnik::SolverParams params = M1::makeParams( dir.str() );
    if ( mode == Mode::FailNoCleanup )
        params.cleanup.enabled = false;
    // fail-step0 needs only the first checkpoint after step 0.
    const int steps = ( mode == Mode::FailStep0 ) ? kEvery : kSteps;
    params.time.steps = steps;

    {
        std::ostringstream os;
        os << "mode " << modeName( mode ) << ", steps " << steps
           << ", checkpoint every " << params.checkpoint.every_steps
           << ", isotropic cleanup "
           << ( params.cleanup.enabled ? "on" : "OFF" )
           << ", gid renumbering every " << params.remesh.renumber_gids_every
           << ", directory '" << dir.str() << "', python '" << python << "'";
        rec.note( os.str() );
    }

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD, params );
    solver.setup();
    auto& mesh = solver.mesh();

    long long cum_splits = 0;
    long long cum_collapses = 0;
    long long cum_flips = 0;
    long long cum_smooth = 0;
    const int remesh_every = params.remesh_every;

    int compared = 0;
    int matched = 0;
    int mismatched = 0;
    int defects = 0;
    int first_failing = -1;
    int time_liveness_failures = 0;
    double cmp_wall = 0.0;
    // fail-no-cleanup: stop at the first detected mismatch (broadcast).
    int stop_now = 0;

    // Compare the checkpoint `written` for gold row `row` (rank 0) and print
    // its `[m1step]` line. `gold_row` is the row the gold FILE comes from,
    // which differs from `row` only in fail-step0.
    auto compareStep = [&]( int row, int gold_row, const std::string& written,
                            double t ) {
        const int step = kGold[row].step;
        const Tolerance tol = toleranceForStep( step );
        const std::string gold = goldForStep( gold_dir, kGold[gold_row].step );
        Comparison c;
        if ( !written.empty() && fileExists( written ) && !gold.empty() )
            c = runComparator( python, script, written, gold, tol, step );
        else
            rec.fail( "step " + std::to_string( step ) +
                      ": missing output or gold file ('" + written + "', '" +
                      gold + "')" );
        cmp_wall += c.seconds;
        ++compared;

        const char* verdict = "DEFECT";
        if ( c.status == 0 )
        {
            verdict = "match";
            ++matched;
        }
        else if ( c.status == 1 )
        {
            verdict = "mismatch";
            ++mismatched;
        }
        else
        {
            ++defects;
        }
        if ( c.status != 0 && first_failing < 0 )
            first_failing = step;

        const double tg = kGold[gold_row].time;
        const double time_rel =
            tg != 0.0 ? std::fabs( t / tg - 1.0 ) : std::fabs( t );

        std::printf( "[m1step] step=%d verdict=%s exit=%d rtol=%.3g "
                     "atol=%.3g time=%.17g gold_time=%.17g time_rel=%.6e "
                     "V_cpp=%lld F_cpp=%lld V_gold=%lld F_gold=%lld "
                     "ambiguous=%s faces_identical=%s",
                     step, verdict, c.status, tol.rtol, tol.atol, t, tg,
                     time_rel, c.v_cpp, c.f_cpp, c.v_gold, c.f_gold,
                     c.ambiguous.c_str(), c.faces_identical.c_str() );
        for ( const auto& kv : c.max_e )
            std::printf( " max_e[%s]=%s", kv.first.c_str(),
                         kv.second.c_str() );
        std::printf( " cmp_s=%.3f\n", c.seconds );
        std::fflush( stdout );

        // A gold file whose counts are not the compiled-in table's is a
        // mis-resolved gold set, not a measurement.
        if ( c.v_gold >= 0 && ( c.v_gold != kGold[gold_row].vertices ||
                                c.f_gold != kGold[gold_row].faces ) )
            rec.fail( "step " + std::to_string( step ) +
                      ": the gold file's counts " + std::to_string( c.v_gold ) +
                      "/" + std::to_string( c.f_gold ) +
                      " are not the gold table's " +
                      std::to_string( kGold[gold_row].vertices ) + "/" +
                      std::to_string( kGold[gold_row].faces ) );
        return c.status;
    };

    // Rank 0's verdict on one compared step, per mode.
    auto judgeStep = [&]( int row, int status, double t ) {
        const int step = kGold[row].step;
        if ( status != 0 && status != 1 )
        {
            rec.fail( "step " + std::to_string( step ) +
                      ": comparator exit " + std::to_string( status ) +
                      " is a DEFECT (load or path error), neither a match nor "
                      "a mismatch" );
            return;
        }
        if ( mode == Mode::Compare )
        {
            if ( status != 0 )
                rec.fail( "step " + std::to_string( step ) +
                          ": comparison mismatch (exit 1) at rtol " +
                          std::to_string( toleranceForStep( step ).rtol ) );
            else
                BEATNIK_CHECK_EQ( rec, status, 0 );
            // Liveness: the adaptive time step.
            const double tg = kGold[row].time;
            const double rel =
                tg != 0.0 ? std::fabs( t / tg - 1.0 ) : std::fabs( t );
            if ( rel > kTimeLivenessRtol )
            {
                ++time_liveness_failures;
                std::ostringstream os;
                os.precision( 17 );
                os << "step " << step << ": time " << t << " vs gold " << tg
                   << ", relative " << rel << " > liveness bound "
                   << kTimeLivenessRtol << " (a fixed dt?)";
                rec.fail( os.str() );
            }
        }
    };

    //-----------------------------------------------------------------------//
    // Step 0, written by `setup()`.
    //-----------------------------------------------------------------------//
    if ( rank == 0 && mode != Mode::FailStep0 )
    {
        const double t = static_cast<double>( solver.time() );
        const int status = compareStep( 0, 0, solver.lastCheckpointPath(), t );
        judgeStep( 0, status, t );
        if ( mode == Mode::FailNoCleanup && status != 0 )
            stop_now = 1;
    }
    MPI_Bcast( &stop_now, 1, MPI_INT, 0, MPI_COMM_WORLD );

    //-----------------------------------------------------------------------//
    // The run. `advanceOneStep` is collective and every rank calls it the
    // same number of times; rank 0's comparator subprocess runs between steps
    // while the other ranks wait in the next step's first collective.
    //-----------------------------------------------------------------------//
    const double t_start = MPI_Wtime();
    long long completed = 0;
    for ( int step = 1; step <= steps && !stop_now; ++step )
    {
        const std::string before = solver.lastCheckpointPath();
        if ( !solver.advanceOneStep() )
        {
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << steps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time();
            rec.fail( os.str() );
            break;
        }
        completed = solver.step();

        if ( remesh_every > 0 && step % remesh_every == 0 )
        {
            const Beatnik::RemeshDiagnostics& d = solver.lastRemesh();
            cum_splits += d.splits;
            cum_collapses += d.collapses;
            cum_flips += d.flips;
            cum_smooth += d.smooth_steps;
        }

        // Collective counts, on every rank.
        const long long nv = mesh.globalVertexCount();
        const long long nf = mesh.globalFaceCount();
        const double t = static_cast<double>( solver.time() );

        if ( rank == 0 )
        {
            std::printf( "[m1traj] step=%d time=%.17g V=%lld F=%lld "
                         "splits=%lld collapses=%lld flips=%lld "
                         "smooth_steps=%lld cleanup_flips=%lld "
                         "elapsed=%.3f\n",
                         step, t, nv, nf, cum_splits, cum_collapses,
                         cum_flips, cum_smooth, solver.cleanupFlips(),
                         MPI_Wtime() - t_start );
            std::fflush( stdout );
        }

        if ( step % kEvery == 0 && rank == 0 )
        {
            const int row = step / kEvery;
            const std::string written = solver.lastCheckpointPath();
            if ( written == before )
                rec.fail( "step " + std::to_string( step ) +
                          ": no new checkpoint was written" );
            if ( mode == Mode::FailStep0 )
            {
                // Direction (a): this step's file against the STEP-0 gold.
                const int status = compareStep( row, 0, written, t );
                std::printf( "[m1fail] direction=a step=%d vs step-0 gold: "
                             "comparator exit %d (1 = detected mismatch, "
                             "anything else = DEFECT)\n",
                             step, status );
                std::fflush( stdout );
                BEATNIK_CHECK_EQ( rec, status, 1 );
            }
            else
            {
                const int status = compareStep( row, row, written, t );
                judgeStep( row, status, t );
                if ( mode == Mode::FailNoCleanup && status != 0 )
                    stop_now = 1;
            }
        }
        if ( step % kEvery == 0 )
            MPI_Bcast( &stop_now, 1, MPI_INT, 0, MPI_COMM_WORLD );
    }
    Kokkos::fence();
    const double wall = MPI_Wtime() - t_start;

    solver.finalize();

    //-----------------------------------------------------------------------//
    // The verdicts that need the whole run.
    //-----------------------------------------------------------------------//
    if ( mode == Mode::FailNoCleanup )
    {
        // Direction (b): the cleanup is off, so no cleanup flip, and the
        // comparison must have failed -- by a mismatch, not a defect.
        BEATNIK_CHECK_EQ( rec, solver.cleanupFlips(), 0 );
        if ( rank == 0 )
        {
            std::printf( "[m1fail] direction=b --no-isotropic-cleanup: "
                         "first failing step %d, mismatched %d, defects %d, "
                         "compared %d (pass iff a mismatch and no defect)\n",
                         first_failing, mismatched, defects, compared );
            std::fflush( stdout );
            BEATNIK_CHECK_TRUE( rec, mismatched > 0 );
            BEATNIK_CHECK_EQ( rec, defects, 0 );
        }
    }
    else if ( mode == Mode::Compare )
    {
        // The budget, then liveness (M1-R7). Every counter is global.
        BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( kSteps ) );
        BEATNIK_CHECK_TRUE( rec, cum_splits > 0 );
        BEATNIK_CHECK_TRUE( rec, cum_collapses > 0 );
        BEATNIK_CHECK_TRUE( rec, cum_flips > 0 );
        BEATNIK_CHECK_TRUE( rec, cum_smooth > 0 );
        BEATNIK_CHECK_TRUE( rec, solver.cleanupFlips() > 0 );
        if ( rank == 0 )
            BEATNIK_CHECK_EQ( rec, compared, kCheckpoints );
    }
    else
    {
        BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( steps ) );
    }

    if ( rank == 0 )
    {
        const double run_wall = wall - cmp_wall;
        const std::string first =
            first_failing < 0 ? std::string( "none" )
                              : std::to_string( first_failing );
        std::printf(
            "[m1final] mode=%s space=%s np=%d steps=%lld/%d wall=%.3f "
            "cmp_wall=%.3f run_wall=%.3f s_per_step=%.6f splits=%lld "
            "collapses=%lld flips=%lld smooth_steps=%lld cleanup_flips=%lld "
            "compared=%d matched=%d mismatched=%d defects=%d "
            "first_failing_step=%s time_liveness_failures=%d\n",
            modeName( mode ), ExecSpace::name(), comm_size, completed, steps,
            wall, cmp_wall, run_wall,
            completed > 0 ? run_wall / static_cast<double>( completed ) : 0.0,
            cum_splits, cum_collapses, cum_flips, cum_smooth,
            solver.cleanupFlips(), compared, matched, mismatched, defects,
            first.c_str(), time_liveness_failures );
        std::fflush( stdout );
    }
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Test_Milestone1Bubble" );
        try
        {
            // Defined by the per-backend shim tests/CMakeLists.txt generates.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using ExecSpace = BEATNIK_TEST_EXEC_SPACE;
            runChecks<ExecSpace, typename ExecSpace::memory_space>( rec, argc,
                                                                    argv );
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

    // ONE VERDICT ACROSS THE RANKS: MPI_MAX makes any rank's failure the job's.
    // Every comparison is rank 0's, so a non-zero rank's tally can be all
    // passes while the job fails on rank 0's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
