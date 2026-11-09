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
 *        live) run for 1610 steps: compared field-wise against the gold at
 *        step 0, and by time-matched mesh statistics against the steps
 *        0–1610 gold set (M1-G3) at every gold checkpoint after it.
 *
 * A MEMBER OF THE `milestone` TIER, NOT OF THE SHIP GATE. It is registered in
 * `BEATNIK_MILESTONE_TEST_SOURCES` (`tests/CMakeLists.txt`) at ranks 1 and 4
 * on every backend, and the 60-launch `regression` gate does not run it.
 *
 * WHAT IT ASSERTS (`tasks/milestone1.md`, M1-A1j **Decided**)
 * -----------------------------------------------------------
 *   1. **Field-wise comparison at step 0 only.** The step-0 checkpoint is
 *      compared against the M1-G2 step-0 gold file by `compare_output.py` at
 *      the comparator's defaults, `--rtol 1e-10 --atol 1e-12`
 *      (`kStep0Tolerance`). The comparator's exit status is read exactly: 0 is
 *      a match, 1 a detected mismatch, anything else a DEFECT (a load or path
 *      error) that is never a pass and never a mismatch. From step 1 on the
 *      two codes' meshes differ by selection — Tessera's shorter quad diagonal
 *      at step 1, its one-round collapse independent set at step 2 — so the
 *      comparator's vertex pairing and count checks cannot apply
 *      (progress log, `## M1-D2`).
 *   2. **Time-matched statistics, gold steps 10 through 1610.** Beatnik's mesh
 *      statistics are computed after every step and linearly interpolated in
 *      `time` to each gold checkpoint's `time` (the table
 *      `Beatnik_Milestone1GoldStats.hpp`, 162 rows, generated from the M1-G3
 *      `.npz` files). At each of the 161 checkpoints after step 0:
 *        - rise (area-weighted centroid `z`): `|Beatnik - gold| <= 1e-3`
 *          at gold steps 10-1050, `<= 4e-3` at gold steps 1060-1610;
 *        - total area: `|Beatnik / gold - 1| <= 1e-3` at gold steps
 *          10-1050, `<= 1.5e-2` at gold steps 1060-1610;
 *        - Beatnik's `|volume drift| <= 1e-12`;
 *        - face count: `|F_Beatnik / F_gold - 1| <= 0.10`;
 *        - Beatnik's minimum triangle quality `>= 0.18`
 *          (`--remesh-min-quality`).
 *      Valence 5/6/7 counts and the minimum and mean edge length are printed
 *      beside the gold's, never asserted. A gold checkpoint whose `time` is
 *      later than Beatnik's final `time` is a `[FAIL]`, never a skip. The
 *      rise and area segment is chosen by the gold row's step, never by
 *      Beatnik's step index. The late limits cover a mesh-selection
 *      discretisation gap that the reference itself shows when restarted from
 *      Beatnik's step-10 mesh (progress log, `## M1-D3`).
 *   3. **Liveness (risk M1-R7).** By the last step the cumulative remesh
 *      counters `splits`, `collapses`, `flips` and `smooth_steps` and
 *      `Solver::cleanupFlips()` are all non-zero, and at every step that is a
 *      multiple of 10 Beatnik's `time` is at most twice the gold's at the same
 *      step index (`kTimeLivenessRtol`). A fixed-dt run is off by a factor of
 *      about 3 at step 10 and fails that.
 *   4. **The budget.** The run reaches step 1610; a non-finite early stop is a
 *      failure, not a shorter run.
 *
 * Edit sets are never compared against the reference (risk R7): Beatnik's
 * accepted flip and collapse sets are independent sets, not the reference's
 * serial sweeps, and its valence flips accept flips the reference rejects
 * (`tasks/milestone1.md`, Deliberate deviations). The cleanup flip total is
 * therefore asserted non-zero and printed, never compared with Python.
 *
 * WHERE THE STATISTICS COME FROM. Every rank sends its owned vertices (gid and
 * position) and its owned faces (corner gids) to rank 0 in one gather per
 * step, and rank 0 computes the statistics serially on the assembled global
 * mesh with the generator's definitions: area `sum 1/2 |(b-a) x (c-a)|`,
 * centroid `z` area-weighted over face centroids, volume `sum a.(b x c) / 6`
 * against `Solver::initialVolume()`, quality from
 * `SurfaceOperators::triangleQuality` (`4 sqrt(3) A / sum l^2`), and valence
 * and edge lengths over the unique edges of the face list. The counts and the
 * minima are therefore independent of the rank count; the sums depend on it
 * only through the face order, at round-off.
 *
 * WHAT IT PRINTS, FOR THE READER OF THE OUTPUT (rank 0 only)
 * ---------------------------------------------------------
 * One line per step, the face-count trajectory and the five counters:
 *
 *     [m1traj] step=<n> time=<t> V=<global V> F=<global F> splits=<cum>
 *       collapses=<cum> flips=<cum> smooth_steps=<cum> cleanup_flips=<cum>
 *       elapsed=<s>
 *
 * One line per field-wise comparison (step 0; step 10 in `fail-step0`):
 *
 *     [m1step] step=<n> verdict=<match|mismatch|DEFECT> exit=<status>
 *       rtol=<r> atol=<a> time=<Beatnik t> gold_time=<gold t>
 *       time_rel=<|t/t_gold - 1|> V_cpp=<V> F_cpp=<F> V_gold=<V> F_gold=<F>
 *       ambiguous=<cpp>/<gold> faces_identical=<k>/<F>
 *       max_e[<field>]=<max|e|> ... cmp_s=<seconds in the comparator>
 *
 * followed by the comparator's full report, every line prefixed
 * `[cmp <step>] `. `V_cpp`…`F_gold` are the two files' row counts from the
 * comparator's `structure:` line (`-1` where it printed none); a field the
 * comparator did not reach is absent.
 *
 * One line per gold checkpoint, the statistics (step 0 printed with
 * `verdict=unasserted`):
 *
 *     [m1stat] gold_step=<s> gold_time=<t> step_lo=<i-1> step_hi=<i>
 *       w=<interpolation weight> F=<Beatnik> F_gold=<n> F_rel=<r>
 *       zc=<Beatnik> zc_gold=<z> rise_err=<|dz|> area=<Beatnik>
 *       area_gold=<A> area_rel=<r> vol_drift=<Beatnik> vol_drift_gold=<d>
 *       qmin=<Beatnik> qmin_gold=<q> v5=<Beatnik> v5_gold=<n> v6=.. v6_gold=..
 *       v7=.. v7_gold=.. lmin=<Beatnik> lmin_gold=<l> lmean=<Beatnik>
 *       lmean_gold=<l> rise_tol=<limit> area_tol=<limit>
 *       verdict=<pass|FAIL|unasserted> failed=<bands|none>
 *
 * where `step_lo`/`step_hi` are the Beatnik steps whose `time` brackets the
 * gold's and every Beatnik value is the interpolant between them.
 * `rise_tol`/`area_tol` are the rise and area limits of the gold step's
 * segment, printed `%.3g`: `0.001`/`0.001` at gold steps 10-1050 (and at the
 * unasserted step 0), `0.004`/`0.015` at 1060-1610. The worst value of each
 * asserted band over the run follows on one line, rise and area per segment:
 *
 *     [m1statworst] rise_err[10-1050]=<max> at=<gold step>
 *       area_rel[10-1050]=<max> at=<s> rise_err[1060-1610]=<max> at=<s>
 *       area_rel[1060-1610]=<max> at=<s> vol_drift=<max |d|> at=<s>
 *       F_rel=<max> at=<s> qmin=<min> at=<s>
 *
 * A segment with no asserted checkpoint prints `0.000000e+00 at=-1`.
 *
 * Closing summary:
 *
 *     [m1final] mode=<mode> space=<space> np=<N> steps=<completed>/<budget>
 *       wall=<s> cmp_wall=<s> stat_wall=<s> run_wall=<s>
 *       s_per_step=<run_wall/steps> splits=<n> collapses=<n> flips=<n>
 *       smooth_steps=<n> cleanup_flips=<n> compared=<n> matched=<n>
 *       mismatched=<n> defects=<n> first_failing_step=<n or none>
 *       time_liveness_failures=<n> stat_checked=<n> stat_failed=<n>
 *       stat_unreached=<n> first_failing_stat_step=<gold step or none>
 *
 * `wall` runs from just before the step-0 comparison to the end of the last
 * step; `cmp_wall` is the time rank 0 spent in comparator subprocesses,
 * `stat_wall` the time spent gathering and computing the statistics, and
 * `run_wall = wall - cmp_wall - stat_wall`. `first_failing_step` is the
 * field-wise comparison's.
 *
 * FAILURE DIRECTIONS, selected by the third argument
 * --------------------------------------------------
 *   `fail-step0`       runs to the first checkpoint after step 0 (step 10)
 *                      and compares it against the **step-0** gold. Passes iff
 *                      the comparator exits exactly 1: a detected mismatch,
 *                      not a load error.
 *   `fail-no-cleanup`  runs with `--no-isotropic-cleanup` forced and checks
 *                      the statistics until a band fails. Passes iff a
 *                      statistics band fails, the step-0 comparison has no
 *                      DEFECT, and `cleanupFlips()` stays 0: the statistics
 *                      check fails rather than passing with a different mesh.
 *                      The face-count band is the one expected to catch it, at
 *                      gold step 10 (1280 faces against 1090).
 *
 * ARGUMENTS
 *
 *   argv[1]  the M1-G2 gold DIRECTORY; only its step-0 file is read
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

#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_Milestone1GoldStats.hpp"
#include "Beatnik_Milestone1Params.hpp"
#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

namespace M1 = Beatnik::Test::Milestone1;

constexpr int kSteps = M1::kSteps;
constexpr int kEvery = M1::kCheckpointEvery;

static_assert( M1::kGoldStatsRows == kSteps / kEvery + 1,
               "the gold-statistics table must have one row per checkpoint "
               "of the run" );
static_assert( M1::kGoldStats[M1::kGoldStatsRows - 1].step == kSteps,
               "the gold-statistics table must end at the run's last step" );

//---------------------------------------------------------------------------//
// The field-wise tolerance, step 0 only: the comparator's defaults. Every
// launch of the M1-T2j run matched at step 0 with max|e| 5.55e-17 on
// `vertices` and `remesh_material_position` and 0 on `potential` and `time`
// (progress log, `## M1-T2`). **Do not loosen** it without that measurement
// beside it (tasks/milestone1.md, risk M1-R1).
//---------------------------------------------------------------------------//
struct Tolerance
{
    double rtol;
    double atol;
};

constexpr Tolerance kStep0Tolerance = { 1.0e-10, 1.0e-12 };

//---------------------------------------------------------------------------//
// The statistics bands, asserted at every gold checkpoint from step 10 to
// 1610 (tasks/milestone1.md, M1-A1j and M1-A1 **Decided**). Measured through
// step 600 at matched time (progress log, `## M1-D2`): rise within 2.683e-4,
// area within 1.805e-4 relative, |volume drift| at most 4.4e-16 on both sides,
// Beatnik 3-5 % fewer faces, its minimum quality 0.840-0.870.
//
// Rise and area are piecewise in the GOLD row's step, split at
// `kLateFirstGoldStep`. Measured at matched time on all four compare launches
// of job f3j79G2VDX4T (progress log, `## M1-A1`):
//   - gold steps 10-1050: worst area_rel 9.88e-4 (step 1050), worst rise_err
//     2.71e-4 (step 310);
//   - gold steps 1060-1610: worst area_rel 1.063e-2 and worst rise_err
//     2.545e-3, both at step 1610. The reference's own solver restarted from
//     Beatnik's step-10 mesh (`bk10`) misses the gold there by area 1.047e-2
//     and rise 2.808e-3 (progress log, `## M1-D3`): a mesh-selection
//     discretisation gap, not a defect. The late limits hold both with about
//     1.4x headroom. 1060 is the first gold step at which area drifts past
//     the early limit.
//---------------------------------------------------------------------------//
/// The first gold step of the late rise/area segment.
constexpr int kLateFirstGoldStep = 1060;
/// `|zc_Beatnik - zc_gold|`, absolute, units of length, gold steps 10-1050.
constexpr double kRiseAtolEarly = 1.0e-3;
/// `|zc_Beatnik - zc_gold|`, absolute, units of length, gold steps 1060-1610.
constexpr double kRiseAtolLate = 4.0e-3;
/// `|area_Beatnik / area_gold - 1|`, gold steps 10-1050.
constexpr double kAreaRtolEarly = 1.0e-3;
/// `|area_Beatnik / area_gold - 1|`, gold steps 1060-1610.
constexpr double kAreaRtolLate = 1.5e-2;

static_assert( kLateFirstGoldStep % kEvery == 0 &&
                   kEvery < kLateFirstGoldStep && kLateFirstGoldStep <= kSteps,
               "the late segment must start at a gold checkpoint after "
               "step 10 and within the run" );

/// The rise/area segment of a gold checkpoint: 0 early, 1 late.
constexpr int bandSegment( int gold_step )
{
    return gold_step >= kLateFirstGoldStep ? 1 : 0;
}
/// The rise limit applied at a gold checkpoint.
constexpr double riseAtol( int gold_step )
{
    return bandSegment( gold_step ) == 1 ? kRiseAtolLate : kRiseAtolEarly;
}
/// The area limit applied at a gold checkpoint.
constexpr double areaRtol( int gold_step )
{
    return bandSegment( gold_step ) == 1 ? kAreaRtolLate : kAreaRtolEarly;
}

/// Beatnik's `|vol / initial_volume - 1|`.
constexpr double kVolumeDriftMax = 1.0e-12;
/// `|F_Beatnik / F_gold - 1|`.
constexpr double kFaceCountRtol = 0.10;
/// Beatnik's minimum triangle quality, `--remesh-min-quality`.
constexpr double kMinQuality = 0.18;

/// Liveness bound on Beatnik's `time` against the gold's at the same step
/// index, as `|t / t_gold - 1|`: at most twice the gold's. Wide on purpose:
/// it separates the adaptive step size from a fixed one, while a mesh that
/// differs from the reference's moves the adaptive dt through its minimum
/// edge, which is a measurement, not a liveness failure. A fixed `--dt 0.003`
/// is off by 1.72 at step 10 (`t = 0.03` against the gold's `0.0110`), 2.43 at
/// step 50 and 2.23 at step 600; Beatnik's adaptive run leads the gold by
/// `time_rel` 0.116-0.336 through step 600 (progress log, `## M1-T2`).
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
/// Beatnik's mesh statistics at one step, every field a `double` so that the
/// counts interpolate in `time` like the rest.
struct MeshStats
{
    double time = 0.0;
    double vertices = 0.0;
    double faces = 0.0;
    double area = 0.0;
    double centroid_z = 0.0;
    double volume_drift = 0.0;
    double min_quality = 0.0;
    double valence5 = 0.0;
    double valence6 = 0.0;
    double valence7 = 0.0;
    double min_edge = 0.0;
    double mean_edge = 0.0;
};

MeshStats interpolate( const MeshStats& a, const MeshStats& b, double w )
{
    auto lerp = [w]( double x, double y ) { return x + w * ( y - x ); };
    MeshStats r;
    r.time = lerp( a.time, b.time );
    r.vertices = lerp( a.vertices, b.vertices );
    r.faces = lerp( a.faces, b.faces );
    r.area = lerp( a.area, b.area );
    r.centroid_z = lerp( a.centroid_z, b.centroid_z );
    r.volume_drift = lerp( a.volume_drift, b.volume_drift );
    r.min_quality = lerp( a.min_quality, b.min_quality );
    r.valence5 = lerp( a.valence5, b.valence5 );
    r.valence6 = lerp( a.valence6, b.valence6 );
    r.valence7 = lerp( a.valence7, b.valence7 );
    r.min_edge = lerp( a.min_edge, b.min_edge );
    r.mean_edge = lerp( a.mean_edge, b.mean_edge );
    return r;
}

/**
 * @brief Gather the global mesh to rank 0 and compute its statistics there.
 *
 * Collective: every rank calls it on the same steps. Rank 0's return value is
 * the statistics; the other ranks' is default. `ok` is set false on rank 0 if
 * an owned face names a vertex gid no rank owns, which is a broken mesh, not a
 * statistic.
 */
template <class ExecSpace, class MemSpace, class MeshType>
MeshStats gatherStats( MeshType& mesh, double time, double initial_volume,
                       bool& ok )
{
    using Real = Beatnik::Real;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm comm = mesh.comm();
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );
    ok = true;

    const int nv_owned = mesh.ownedVertexCount();
    const int nf_owned = mesh.ownedFaceCount();

    // Owned positions to the host.
    Kokkos::View<Real* [3], MemSpace> owned_pos( "m1_owned_positions",
                                                 nv_owned );
    {
        auto pos = mesh.positions();
        Kokkos::parallel_for(
            "m1_copy_owned_positions",
            Kokkos::RangePolicy<ExecSpace>( 0, nv_owned ),
            KOKKOS_LAMBDA( const int i ) {
                owned_pos( i, 0 ) = pos( i, 0 );
                owned_pos( i, 1 ) = pos( i, 1 );
                owned_pos( i, 2 ) = pos( i, 2 );
            } );
        Kokkos::fence();
    }
    auto h_pos =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), owned_pos );
    auto h_faces = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        Kokkos::subview( mesh.faceVertices(), std::make_pair( 0, nf_owned ),
                         Kokkos::ALL() ) );
    const std::vector<std::uint64_t> gids = mesh.vertexGids();

    std::vector<std::uint64_t> send_vgid( nv_owned );
    std::vector<double> send_pos( 3 * static_cast<std::size_t>( nv_owned ) );
    for ( int i = 0; i < nv_owned; ++i )
    {
        send_vgid[i] = gids[i];
        for ( int d = 0; d < 3; ++d )
            send_pos[3 * i + d] = static_cast<double>( h_pos( i, d ) );
    }
    // A corner of an owned face is owned or ghosted here by construction; a
    // -1 is flagged by the all-ones gid, which no rank owns.
    std::vector<std::uint64_t> send_fgid( 3 *
                                          static_cast<std::size_t>( nf_owned ) );
    for ( int f = 0; f < nf_owned; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const int v = h_faces( f, k );
            send_fgid[3 * f + k] = v >= 0 ? gids[v] : ~std::uint64_t( 0 );
        }

    int counts[2] = { nv_owned, nf_owned };
    std::vector<int> all_counts( 2 * comm_size, 0 );
    MPI_Gather( counts, 2, MPI_INT, all_counts.data(), 2, MPI_INT, 0, comm );

    std::vector<int> nv_r( comm_size ), nv3_r( comm_size ), nf3_r( comm_size );
    std::vector<int> nv_d( comm_size ), nv3_d( comm_size ), nf3_d( comm_size );
    int nv = 0, nf = 0;
    if ( rank == 0 )
    {
        for ( int r = 0; r < comm_size; ++r )
        {
            nv_r[r] = all_counts[2 * r];
            nv3_r[r] = 3 * all_counts[2 * r];
            nf3_r[r] = 3 * all_counts[2 * r + 1];
            nv_d[r] = nv;
            nv3_d[r] = 3 * nv;
            nf3_d[r] = 3 * nf;
            nv += all_counts[2 * r];
            nf += all_counts[2 * r + 1];
        }
    }
    std::vector<std::uint64_t> vgid( rank == 0 ? nv : 0 );
    std::vector<double> vpos( rank == 0 ? 3 * nv : 0 );
    std::vector<std::uint64_t> fgid( rank == 0 ? 3 * nf : 0 );
    MPI_Gatherv( send_vgid.data(), nv_owned, MPI_UINT64_T, vgid.data(),
                 nv_r.data(), nv_d.data(), MPI_UINT64_T, 0, comm );
    MPI_Gatherv( send_pos.data(), 3 * nv_owned, MPI_DOUBLE, vpos.data(),
                 nv3_r.data(), nv3_d.data(), MPI_DOUBLE, 0, comm );
    MPI_Gatherv( send_fgid.data(), 3 * nf_owned, MPI_UINT64_T, fgid.data(),
                 nf3_r.data(), nf3_d.data(), MPI_UINT64_T, 0, comm );

    MeshStats s;
    if ( rank != 0 )
        return s;

    std::unordered_map<std::uint64_t, int> index;
    index.reserve( 2 * static_cast<std::size_t>( nv ) );
    for ( int i = 0; i < nv; ++i )
        index.emplace( vgid[i], i );

    Kokkos::View<Real* [3], Kokkos::HostSpace> P( "m1_global_positions", nv );
    for ( int i = 0; i < nv; ++i )
        for ( int d = 0; d < 3; ++d )
            P( i, d ) = vpos[3 * i + d];
    Kokkos::View<int* [3], Kokkos::HostSpace> F( "m1_global_faces", nf );
    for ( int f = 0; f < nf; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const auto it = index.find( fgid[3 * f + k] );
            if ( it == index.end() )
            {
                ok = false;
                F( f, k ) = 0;
            }
            else
                F( f, k ) = it->second;
        }

    Kokkos::View<Real*, Kokkos::HostSpace> quality( "m1_global_quality", nf );
    Beatnik::SurfaceOperators::triangleQuality( P, F, quality );
    Kokkos::fence();

    double area = 0.0, wz = 0.0, vol6 = 0.0;
    double qmin = std::numeric_limits<double>::infinity();
    std::vector<std::pair<int, int>> edges;
    edges.reserve( 3 * static_cast<std::size_t>( nf ) );
    for ( int f = 0; f < nf; ++f )
    {
        const int ia = F( f, 0 ), ib = F( f, 1 ), ic = F( f, 2 );
        double a[3], b[3], c[3], ab[3], ac[3];
        for ( int d = 0; d < 3; ++d )
        {
            a[d] = P( ia, d );
            b[d] = P( ib, d );
            c[d] = P( ic, d );
            ab[d] = b[d] - a[d];
            ac[d] = c[d] - a[d];
        }
        const double nx = ab[1] * ac[2] - ab[2] * ac[1];
        const double ny = ab[2] * ac[0] - ab[0] * ac[2];
        const double nz = ab[0] * ac[1] - ab[1] * ac[0];
        const double af = 0.5 * std::sqrt( nx * nx + ny * ny + nz * nz );
        area += af;
        wz += af * ( a[2] + b[2] + c[2] ) / 3.0;
        vol6 += a[0] * ( b[1] * c[2] - b[2] * c[1] ) +
                a[1] * ( b[2] * c[0] - b[0] * c[2] ) +
                a[2] * ( b[0] * c[1] - b[1] * c[0] );
        qmin = std::min( qmin, static_cast<double>( quality( f ) ) );
        const int corner[3] = { ia, ib, ic };
        for ( int k = 0; k < 3; ++k )
        {
            const int u = corner[k], v = corner[( k + 1 ) % 3];
            edges.emplace_back( std::min( u, v ), std::max( u, v ) );
        }
    }
    std::sort( edges.begin(), edges.end() );
    edges.erase( std::unique( edges.begin(), edges.end() ), edges.end() );

    std::vector<int> valence( nv, 0 );
    double lmin = std::numeric_limits<double>::infinity(), lsum = 0.0;
    for ( const auto& e : edges )
    {
        ++valence[e.first];
        ++valence[e.second];
        double l2 = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            const double dx = P( e.first, d ) - P( e.second, d );
            l2 += dx * dx;
        }
        const double l = std::sqrt( l2 );
        lmin = std::min( lmin, l );
        lsum += l;
    }
    long long v5 = 0, v6 = 0, v7 = 0;
    for ( int i = 0; i < nv; ++i )
    {
        v5 += valence[i] == 5;
        v6 += valence[i] == 6;
        v7 += valence[i] == 7;
    }

    s.time = time;
    s.vertices = nv;
    s.faces = nf;
    s.area = area;
    s.centroid_z = area > 0.0 ? wz / area : 0.0;
    s.volume_drift = ( vol6 / 6.0 ) / initial_volume - 1.0;
    s.min_quality = qmin;
    s.valence5 = static_cast<double>( v5 );
    s.valence6 = static_cast<double>( v6 );
    s.valence7 = static_cast<double>( v7 );
    s.min_edge = lmin;
    s.mean_edge =
        edges.empty() ? 0.0 : lsum / static_cast<double>( edges.size() );
    return s;
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
        if ( inputs_ok && goldForStep( gold_dir, 0 ).empty() )
        {
            rec.fail( "no step-0 gold file in " + gold_dir );
            inputs_ok = 0;
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
    // The statistics are a parameter-only schedule: every step, on every
    // rank, in every mode but fail-step0.
    const bool track_stats = ( mode != Mode::FailStep0 );

    {
        std::ostringstream os;
        os << "mode " << modeName( mode ) << ", steps " << steps
           << ", checkpoint every " << params.checkpoint.every_steps
           << ", isotropic cleanup "
           << ( params.cleanup.enabled ? "on" : "OFF" )
           << ", gid renumbering every " << params.remesh.renumber_gids_every
           << ", statistics " << ( track_stats ? "every step" : "off" )
           << ", directory '" << dir.str() << "', python '" << python << "'";
        rec.note( os.str() );
    }

    Beatnik::Solver<ExecSpace, MemSpace> solver( MPI_COMM_WORLD, params );
    solver.setup();
    auto& mesh = solver.mesh();
    const double initial_volume =
        static_cast<double>( solver.initialVolume() );

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
    double stat_wall = 0.0;

    int stat_checked = 0;
    int stat_failed = 0;
    int stat_unreached = 0;
    int first_failing_stat = -1;
    // The worst value of each asserted band, and the gold step it occurs at;
    // rise and area per segment (`bandSegment`).
    double worst_rise[2] = { 0.0, 0.0 }, worst_area[2] = { 0.0, 0.0 };
    int worst_rise_at[2] = { -1, -1 }, worst_area_at[2] = { -1, -1 };
    double worst_vol = 0.0, worst_faces = 0.0,
           worst_q = std::numeric_limits<double>::infinity();
    int worst_vol_at = -1, worst_faces_at = -1, worst_q_at = -1;
    // The next gold row to match; row 0 is step 0, printed unasserted.
    int next_gold = 1;
    bool mesh_ok_all = true;
    // Stop at the first statistics failure (fail-no-cleanup; broadcast).
    int stop_now = 0;

    // Compare the checkpoint `written` at Beatnik step `step` against the
    // STEP-0 gold file (rank 0) and print its `[m1step]` line.
    auto compareStep = [&]( int step, const std::string& written, double t ) {
        const Tolerance tol = kStep0Tolerance;
        const std::string gold = goldForStep( gold_dir, 0 );
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

        const double tg = M1::kGoldStats[0].time;
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

        // A gold file whose counts are not the table's step-0 row is a
        // mis-resolved gold set, not a measurement.
        if ( c.v_gold >= 0 && ( c.v_gold != M1::kGoldStats[0].vertices ||
                                c.f_gold != M1::kGoldStats[0].faces ) )
            rec.fail( "step " + std::to_string( step ) +
                      ": the gold file's counts " + std::to_string( c.v_gold ) +
                      "/" + std::to_string( c.f_gold ) +
                      " are not the gold table's step-0 " +
                      std::to_string( M1::kGoldStats[0].vertices ) + "/" +
                      std::to_string( M1::kGoldStats[0].faces ) );
        return c.status;
    };

    // Print one `[m1stat]` line for gold row `row` against Beatnik's
    // interpolant `b`, and judge the five bands unless `asserted` is false.
    // Returns true iff every band holds. Rank 0 only.
    auto checkStats = [&]( int row, const MeshStats& b, int step_lo,
                           int step_hi, double w, bool asserted ) {
        const M1::GoldStats& g = M1::kGoldStats[row];
        const double f_rel = std::fabs( b.faces / g.faces - 1.0 );
        const double rise_err = std::fabs( b.centroid_z - g.centroid_z );
        const double area_rel = std::fabs( b.area / g.area - 1.0 );
        const double vol = std::fabs( b.volume_drift );
        // The segment is the gold row's, never Beatnik's step index.
        const int seg = bandSegment( g.step );
        const double rise_tol = riseAtol( g.step );
        const double area_tol = areaRtol( g.step );

        std::string failed;
        auto band = [&]( bool holds, const char* name ) {
            if ( holds )
                return;
            if ( !failed.empty() )
                failed += ",";
            failed += name;
        };
        // Each band is written `x <= tol`, so a NaN fails it.
        band( rise_err <= rise_tol, "rise" );
        band( area_rel <= area_tol, "area" );
        band( vol <= kVolumeDriftMax, "volume_drift" );
        band( f_rel <= kFaceCountRtol, "face_count" );
        band( b.min_quality >= kMinQuality, "min_quality" );

        const char* verdict =
            !asserted ? "unasserted" : ( failed.empty() ? "pass" : "FAIL" );
        std::printf(
            "[m1stat] gold_step=%d gold_time=%.17g step_lo=%d step_hi=%d "
            "w=%.6f F=%.3f F_gold=%lld F_rel=%.6e zc=%.17g zc_gold=%.17g "
            "rise_err=%.6e area=%.17g area_gold=%.17g area_rel=%.6e "
            "vol_drift=%.6e vol_drift_gold=%.6e qmin=%.6f qmin_gold=%.6f "
            "v5=%.3f v5_gold=%lld v6=%.3f v6_gold=%lld v7=%.3f v7_gold=%lld "
            "lmin=%.6e lmin_gold=%.6e lmean=%.6e lmean_gold=%.6e "
            "rise_tol=%.3g area_tol=%.3g verdict=%s failed=%s\n",
            g.step, g.time, step_lo, step_hi, w, b.faces, g.faces, f_rel,
            b.centroid_z, g.centroid_z, rise_err, b.area, g.area, area_rel,
            b.volume_drift, g.volume_drift, b.min_quality, g.min_quality,
            b.valence5, g.valence5, b.valence6, g.valence6, b.valence7,
            g.valence7, b.min_edge, g.min_edge, b.mean_edge, g.mean_edge,
            rise_tol, area_tol, verdict,
            failed.empty() ? "none" : failed.c_str() );
        std::fflush( stdout );
        if ( !asserted )
            return true;

        ++stat_checked;
        if ( !( rise_err <= worst_rise[seg] ) )
        {
            worst_rise[seg] = rise_err;
            worst_rise_at[seg] = g.step;
        }
        if ( !( area_rel <= worst_area[seg] ) )
        {
            worst_area[seg] = area_rel;
            worst_area_at[seg] = g.step;
        }
        if ( !( vol <= worst_vol ) )
        {
            worst_vol = vol;
            worst_vol_at = g.step;
        }
        if ( !( f_rel <= worst_faces ) )
        {
            worst_faces = f_rel;
            worst_faces_at = g.step;
        }
        if ( !( b.min_quality >= worst_q ) )
        {
            worst_q = b.min_quality;
            worst_q_at = g.step;
        }
        if ( failed.empty() )
            return true;
        ++stat_failed;
        if ( first_failing_stat < 0 )
            first_failing_stat = g.step;
        return false;
    };

    // Judge one band set into the recorder (compare mode only), with the
    // numbers in the failure text.
    auto recordStats = [&]( int row, const MeshStats& b, bool holds ) {
        if ( mode != Mode::Compare )
            return;
        if ( holds )
        {
            BEATNIK_CHECK_TRUE( rec, holds );
            return;
        }
        const M1::GoldStats& g = M1::kGoldStats[row];
        char msg[640];
        std::snprintf(
            msg, sizeof( msg ),
            "gold step %d (time %.17g): statistics outside the decided bands: "
            "rise_err %.6e (<= %.3g), area_rel %.6e (<= %.3g), |vol_drift| "
            "%.6e (<= %.3g), F_rel %.6e (<= %.3g; F %.3f vs %lld), qmin %.6f "
            "(>= %.3g)",
            g.step, g.time, std::fabs( b.centroid_z - g.centroid_z ),
            riseAtol( g.step ), std::fabs( b.area / g.area - 1.0 ),
            areaRtol( g.step ),
            std::fabs( b.volume_drift ), kVolumeDriftMax,
            std::fabs( b.faces / g.faces - 1.0 ), kFaceCountRtol, b.faces,
            g.faces, b.min_quality, kMinQuality );
        rec.fail( msg );
    };

    //-----------------------------------------------------------------------//
    // Step 0, written by `setup()`: the field-wise comparison, and the
    // step-0 statistics (printed, not asserted).
    //-----------------------------------------------------------------------//
    const double t_start = MPI_Wtime();
    if ( rank == 0 && mode != Mode::FailStep0 )
    {
        const double t = static_cast<double>( solver.time() );
        const int status = compareStep( 0, solver.lastCheckpointPath(), t );
        if ( status != 0 && status != 1 )
            rec.fail( "step 0: comparator exit " + std::to_string( status ) +
                      " is a DEFECT (load or path error), neither a match "
                      "nor a mismatch" );
        else if ( mode == Mode::Compare )
        {
            if ( status != 0 )
            {
                char msg[160];
                std::snprintf( msg, sizeof( msg ),
                               "step 0: comparison mismatch (exit 1) at rtol "
                               "%.3g atol %.3g",
                               kStep0Tolerance.rtol, kStep0Tolerance.atol );
                rec.fail( msg );
            }
            else
                BEATNIK_CHECK_EQ( rec, status, 0 );
        }
    }

    MeshStats prev;
    if ( track_stats )
    {
        const double ts0 = MPI_Wtime();
        bool ok = true;
        prev = gatherStats<ExecSpace, MemSpace>(
            mesh, static_cast<double>( solver.time() ), initial_volume, ok );
        if ( rank == 0 )
        {
            mesh_ok_all = mesh_ok_all && ok;
            checkStats( 0, prev, 0, 0, 1.0, false );
        }
        stat_wall += MPI_Wtime() - ts0;
    }

    //-----------------------------------------------------------------------//
    // The run. `advanceOneStep` and the statistics gather are collective and
    // every rank calls them the same number of times; rank 0's comparator
    // subprocess and statistics run between steps while the other ranks wait
    // in the next collective.
    //-----------------------------------------------------------------------//
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

        if ( track_stats )
        {
            const double ts0 = MPI_Wtime();
            bool ok = true;
            const MeshStats cur = gatherStats<ExecSpace, MemSpace>(
                mesh, t, initial_volume, ok );
            if ( rank == 0 )
            {
                mesh_ok_all = mesh_ok_all && ok;
                // Every gold checkpoint whose time this step reached.
                while ( next_gold < M1::kGoldStatsRows &&
                        M1::kGoldStats[next_gold].time <= cur.time &&
                        !stop_now )
                {
                    const double tg = M1::kGoldStats[next_gold].time;
                    const double span = cur.time - prev.time;
                    const double w =
                        span > 0.0 ? ( tg - prev.time ) / span : 1.0;
                    const MeshStats b = interpolate( prev, cur, w );
                    const bool holds =
                        checkStats( next_gold, b, step - 1, step, w, true );
                    recordStats( next_gold, b, holds );
                    if ( mode == Mode::FailNoCleanup && !holds )
                        stop_now = 1;
                    ++next_gold;
                }
            }
            prev = cur;
            stat_wall += MPI_Wtime() - ts0;
        }

        if ( step % kEvery == 0 && rank == 0 )
        {
            const std::string written = solver.lastCheckpointPath();
            if ( written == before )
                rec.fail( "step " + std::to_string( step ) +
                          ": no new checkpoint was written" );
            if ( mode == Mode::FailStep0 )
            {
                // Direction (a): this step's file against the STEP-0 gold.
                const int status = compareStep( step, written, t );
                std::printf( "[m1fail] direction=a step=%d vs step-0 gold: "
                             "comparator exit %d (1 = detected mismatch, "
                             "anything else = DEFECT)\n",
                             step, status );
                std::fflush( stdout );
                BEATNIK_CHECK_EQ( rec, status, 1 );
            }
            else if ( mode == Mode::Compare )
            {
                // Liveness: the adaptive time step, at the same step index.
                const double tg = M1::kGoldStats[step / kEvery].time;
                const double rel = std::fabs( t / tg - 1.0 );
                if ( !( rel <= kTimeLivenessRtol ) )
                {
                    ++time_liveness_failures;
                    std::ostringstream os;
                    os.precision( 17 );
                    os << "step " << step << ": time " << t << " vs gold "
                       << tg << ", relative " << rel << " > liveness bound "
                       << kTimeLivenessRtol << " (a fixed dt?)";
                    rec.fail( os.str() );
                }
            }
        }
        MPI_Bcast( &stop_now, 1, MPI_INT, 0, MPI_COMM_WORLD );
    }
    Kokkos::fence();
    const double wall = MPI_Wtime() - t_start;

    solver.finalize();

    //-----------------------------------------------------------------------//
    // The verdicts that need the whole run.
    //-----------------------------------------------------------------------//
    if ( rank == 0 && track_stats )
    {
        if ( !mesh_ok_all )
            rec.fail( "an owned face named a vertex gid that no rank owns "
                      "in the statistics gather: the mesh is broken" );
        if ( mode == Mode::Compare )
        {
            // A gold checkpoint Beatnik's time never reached is a failure,
            // never a skip.
            for ( ; next_gold < M1::kGoldStatsRows; ++next_gold )
            {
                ++stat_unreached;
                std::ostringstream os;
                os.precision( 17 );
                os << "gold step " << M1::kGoldStats[next_gold].step
                   << " (time " << M1::kGoldStats[next_gold].time
                   << ") is later than Beatnik's final time " << prev.time
                   << " at step " << completed
                   << ": the statistics were NOT checked there";
                rec.fail( os.str() );
            }
        }
        const int early_last = kLateFirstGoldStep - kEvery;
        std::printf( "[m1statworst] rise_err[%d-%d]=%.6e at=%d "
                     "area_rel[%d-%d]=%.6e at=%d rise_err[%d-%d]=%.6e at=%d "
                     "area_rel[%d-%d]=%.6e at=%d vol_drift=%.6e at=%d "
                     "F_rel=%.6e at=%d qmin=%.6f at=%d\n",
                     kEvery, early_last, worst_rise[0], worst_rise_at[0],
                     kEvery, early_last, worst_area[0], worst_area_at[0],
                     kLateFirstGoldStep, kSteps, worst_rise[1],
                     worst_rise_at[1], kLateFirstGoldStep, kSteps,
                     worst_area[1], worst_area_at[1], worst_vol, worst_vol_at,
                     worst_faces, worst_faces_at, worst_q, worst_q_at );
        std::fflush( stdout );
    }

    if ( mode == Mode::FailNoCleanup )
    {
        // Direction (b): the cleanup is off, so no cleanup flip, and a
        // statistics band must have failed, with no comparator defect.
        BEATNIK_CHECK_EQ( rec, solver.cleanupFlips(), 0 );
        if ( rank == 0 )
        {
            std::printf( "[m1fail] direction=b --no-isotropic-cleanup: "
                         "first failing gold step %d, statistics checked %d, "
                         "failed %d; step-0 comparison defects %d (pass iff "
                         "a statistics band fails and no defect)\n",
                         first_failing_stat, stat_checked, stat_failed,
                         defects );
            std::fflush( stdout );
            BEATNIK_CHECK_TRUE( rec, stat_failed > 0 );
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
        {
            BEATNIK_CHECK_EQ( rec, compared, 1 );
            BEATNIK_CHECK_EQ( rec, stat_checked, M1::kGoldStatsRows - 1 );
        }
    }
    else
    {
        BEATNIK_CHECK_EQ( rec, completed, static_cast<long long>( steps ) );
    }

    if ( rank == 0 )
    {
        const double run_wall = wall - cmp_wall - stat_wall;
        const std::string first =
            first_failing < 0 ? std::string( "none" )
                              : std::to_string( first_failing );
        const std::string first_stat =
            first_failing_stat < 0 ? std::string( "none" )
                                   : std::to_string( first_failing_stat );
        std::printf(
            "[m1final] mode=%s space=%s np=%d steps=%lld/%d wall=%.3f "
            "cmp_wall=%.3f stat_wall=%.3f run_wall=%.3f s_per_step=%.6f "
            "splits=%lld collapses=%lld flips=%lld smooth_steps=%lld "
            "cleanup_flips=%lld compared=%d matched=%d mismatched=%d "
            "defects=%d first_failing_step=%s time_liveness_failures=%d "
            "stat_checked=%d stat_failed=%d stat_unreached=%d "
            "first_failing_stat_step=%s\n",
            modeName( mode ), ExecSpace::name(), comm_size, completed, steps,
            wall, cmp_wall, stat_wall, run_wall,
            completed > 0 ? run_wall / static_cast<double>( completed ) : 0.0,
            cum_splits, cum_collapses, cum_flips, cum_smooth,
            solver.cleanupFlips(), compared, matched, mismatched, defects,
            first.c_str(), time_liveness_failures, stat_checked, stat_failed,
            stat_unreached, first_stat.c_str() );
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
