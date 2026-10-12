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
 * @file Beatnik_Probe_FmmCost.cpp
 * @brief **canopy_profiling T2's MEASUREMENT DRIVER**
 *        (`tasks/canopy_profiling.md`) — the per-phase cost of an
 *        **FMM-driven** frozen-mesh trajectory, read from Beatnik's and
 *        Canopy's timer registries around every timestep, with a claim-A
 *        accuracy check and a direct-cost estimate at sampled states.
 *
 * THIS IS NOT A TEST AND IS IN NO TIER.
 * ------------------------------------
 * It carries no `LABELS`, no ctest case and no manifest line — see the
 * "Measurement drivers — IN NO TIER" loop in `tests/CMakeLists.txt`. **It
 * asserts nothing about any measured value.** It exits non-zero only when it
 * cannot run: a bad argument (an unknown `key=` names itself), a non-finite
 * state, a broken frozen mesh, or a driving BR solver that is not
 * `BRSolverFMM`.
 *
 * WHY FMM-DRIVEN. Canopy's maintenance path (`Migrate`, `Rebalance`,
 * `Rebuild`) depends on how far the particles moved since the last
 * evaluation, and production presents it one RK3 stage at a time, three
 * evaluations per step. Only a trajectory the FMM itself drives measures that
 * pattern (`tasks/canopy_profiling.md`, "Why the trajectory is FMM-driven").
 *
 * ARGUMENTS.
 *
 *   argv[1]  level          --icosphere-subdivisions (>= 0)
 *   argv[2]  steps          timesteps to run (>= 1)
 *   argv[3]  sample_every   emit a [cost] and an [acc] row every this many
 *                           steps, and at step 1 and the last step (>= 1)
 *   argv[4..] key=value     optional overrides; an unknown key is an error:
 *     ncrit=<n>             leaf occupancy, default 8 (the milestone members')
 *     order=<n>             CartesianTaylor order, default 3; the far-field
 *                           adapter rejects any order it has no arm for
 *     mac_theta=<t>         default 0.3
 *     m2l_op_count_cap=<n>  default 65536 (the level-4 member's
 *                           `kM2LOpCountCap`)
 *     canopy_report=<0|1>   Canopy's own timing tables and diagnostic lines,
 *                           default 0 (off, through
 *                           `Canopy::Profiling::set_reporting`)
 *     run=<label>           the `run=` value on every row; default
 *                           `L<level>_<space>_np<N>_ncrit<n>_order<p>_rep<r>`
 *
 * The configuration is otherwise the milestone-0 frozen-mesh bubble of
 * `Beatnik_Test_Milestone0Fmm.cpp::makeParams` with `--br-approximation fmm`
 * and **no checkpoints**. That setup is re-derived here rather than shared
 * (`tasks/canopy_profiling.md`, "Deliberate deviations"); the constants that
 * must agree with the member are `kRadius`, `kCenterZ`, the `zmodel` and
 * `time` blocks of `makeParams`, `kTauA`'s metric (`fieldScale`,
 * `fieldDifference`, `kFieldScaleFloor`) and the `FmmParams` defaults above.
 *
 * WHAT A STEP MEASURES. Around each `advanceOneStep()` the driver snapshots
 * Beatnik's `Profiling::timer_totals()`, Canopy's
 * `Canopy::Profiling::timer_totals()` and the driving solver's cumulative
 * `FarFieldDiagnostics` counters (`evaluations`, `maintenance_counts`,
 * `local_m2l_op_keys_built`), fences, steps, fences, and differences. The
 * differences accumulate into a window that closes at each sampled step.
 *
 * OUTPUT. T0's row grammar (`canopy_profiling-progress-log.md` `## T0`,
 * "Decisions"; read by `fmm_profile_reduce.py`): one line per row,
 * `[<tag>] key=value ...`, each key once, `run=` on every row and `step=` on
 * `[cost]` and `[acc]`. A value that is not compiled in prints `-1`; a timer
 * that is compiled in but did not fire in a window prints `0`.
 *
 *   [cost] — one per sampled step. Window `steps` and `evals`; `time`
 *     (simulation time at the step); `step_wall` per step; every `ff_*` phase
 *     and every Canopy key listed in `kCanopyKeys`, each **per evaluation**
 *     (window sum over window evaluations). Rank-local quantities are reduced
 *     over ranks: the bare key is the mean, `<key>.max` the max. Then the
 *     window's maintenance counts (`maint_*`, which sum to `evals`),
 *     `keys_built_delta` against `cache_size`, `unique_ops`,
 *     `occupied_depths`, and the global fields of the window's LAST
 *     evaluation: `fb_range_guard`, `fb_count_cap`, `fb_dropped`,
 *     `m2l_pairs`, `m2l_fallback`, `p2p_frac`, `particles`.
 *   [acc] — at the same state, after the window closed, so it never
 *     pollutes `[cost]`: a standalone `BRSolverDirect` and a standalone
 *     `BRSolverFMM`, both constructed once outside the loop. `acc_rel` is
 *     claim A's metric `max|u_fmm - u_direct| / max|u_direct|`
 *     (`Beatnik_Test_Milestone0Fmm.cpp:1447-1490`); `direct_wall` and
 *     `fmm_wall` are the two evaluation walls (max over ranks);
 *     `direct_step_est` is `3 * direct_wall + (step wall - the step's three
 *     ff_evaluate_total)` for the last step of the window.
 *   [cost-total] — once, at the end: the action histogram, each phase's share
 *     of `ff_evaluate_total`, `ff_evaluate_total`'s share of step wall, the
 *     operator-build share of `ff_solve`, and `canopy_hash` from the
 *     environment variable `BEATNIK_CANOPY_HASH` (else `unknown`).
 */

#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_BRSolverFMM.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>
#include <Profiling.hpp>

#ifdef BEATNIK_ENABLE_CANOPY
#include <Canopy_Profiling.hpp>
#endif

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The configuration shared with `Beatnik_Test_Milestone0Fmm.cpp`.
//---------------------------------------------------------------------------//
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

/// Claim A's floor below which the field scale is treated as zero and the
/// error is reported absolutely (`Beatnik_Test_Milestone0Fmm.cpp:330`).
constexpr double kFieldScaleFloor = 1.0e-12;

/// `FmmParams` defaults this driver applies unless overridden.
constexpr int kDefaultNcrit = 8;
constexpr int kDefaultOrder = 3;
constexpr double kDefaultMacTheta = 0.3;
constexpr int kDefaultOpCountCap = 65536;

//---------------------------------------------------------------------------//
// Timer keys. Beatnik's ten phases tile `ff_evaluate_total`; the Canopy keys
// are the ones T3 splits `ff_solve` and `ff_maintain` into. `detailed` marks a
// key timed only at CANOPY_PROFILING_LEVEL >= 2, which prints -1 below it.
//---------------------------------------------------------------------------//
const std::vector<std::string> kFfPhases = {
    "ff_pack",       "ff_fwd_distribute", "ff_fwd_migrate",
    "ff_maintain",   "ff_solve",          "ff_contract",
    "ff_rev_distribute", "ff_rev_migrate", "ff_scatter",
    "ff_diagnostics" };
const char* const kFfTotal = "ff_evaluate_total";

struct CanopyKey
{
    const char* name;
    bool detailed;
};
const std::vector<CanopyKey> kCanopyKeys = {
    { "upward_total", false },
    { "downward_total", false },
    { "dn_build_ilist", true },
    { "ilist_s1_collect_entries", true },
    { "ilist_s2_sort_by_depth", true },
    { "ilist_s3_classify_pairs", true },
    { "ilist_s4_op_table_build", true },
    { "ilist_s4_op_table_copy", true },
    { "ilist_s5_csr_partition_sort", true },
    { "ilist_s5_csr_build", true },
    { "ilist_s5_fallback_table", true },
    { "ilist_s5_device_upload", true },
    { "m2l_kernel", false },
    { "l2p", false },
    { "p2p_total", false },
    { "migrate_total", false },
    { "rebalance_total", false },
    { "setup_total", false },
    { "partition", false },
    { "repartition", false },
    { "redistribute", false },
    { "builder_build", false },
    { "sort_by_leaf", false },
    { "comm_plan_build", false },
    { "commplan_dtt_traversal", true } };

/// Whether each family of timers is compiled into this binary.
#if defined( BEATNIK_ENABLE_PROFILING ) && BEATNIK_PROFILING_LEVEL >= 2
constexpr bool kHaveFfTimers = true;
#else
constexpr bool kHaveFfTimers = false;
#endif

#if defined( BEATNIK_ENABLE_CANOPY ) && defined( CANOPY_ENABLE_PROFILING )
constexpr bool kHaveCanopyTimers = true;
constexpr int kCanopyLevel = CANOPY_PROFILING_LEVEL;
#else
constexpr bool kHaveCanopyTimers = false;
constexpr int kCanopyLevel = 0;
#endif

bool canopyKeyAvailable( const CanopyKey& k )
{
    return kHaveCanopyTimers && ( !k.detailed || kCanopyLevel >= 2 );
}

using Snapshot = std::unordered_map<std::string, double>;

Snapshot beatnikTotals()
{
#if defined( BEATNIK_ENABLE_PROFILING )
    return Snapshot( Beatnik::Profiling::timer_totals().begin(),
                     Beatnik::Profiling::timer_totals().end() );
#else
    return Snapshot();
#endif
}

Snapshot canopyTotals()
{
#if defined( BEATNIK_ENABLE_CANOPY ) && defined( CANOPY_ENABLE_PROFILING )
    return Snapshot( Canopy::Profiling::timer_totals().begin(),
                     Canopy::Profiling::timer_totals().end() );
#else
    return Snapshot();
#endif
}

/// `after[k] - before[k]`, a missing key reading as 0.
double delta( const Snapshot& after, const Snapshot& before,
              const std::string& k )
{
    auto a = after.find( k );
    auto b = before.find( k );
    return ( a == after.end() ? 0.0 : a->second ) -
           ( b == before.end() ? 0.0 : b->second );
}

//---------------------------------------------------------------------------//
/// Entity counts of the subdivision-L icosphere, as
/// `Beatnik_Probe_FmmKeyDemand.cpp` computes them.
long long powFour( int level )
{
    long long f = 1;
    for ( int i = 0; i < level; ++i )
        f *= 4;
    return f;
}
long long verticesForLevel( int level ) { return 10 * powFour( level ) + 2; }
long long facesForLevel( int level ) { return 20 * powFour( level ); }

//---------------------------------------------------------------------------//
/// The parsed command line.
struct Options
{
    int level = -1;
    int steps = 0;
    int sample_every = 0;
    int ncrit = kDefaultNcrit;
    int order = kDefaultOrder;
    double mac_theta = kDefaultMacTheta;
    int m2l_op_count_cap = kDefaultOpCountCap;
    int canopy_report = 0;
    std::string run;
};

int parseInt( const std::string& key, const std::string& v )
{
    std::size_t used = 0;
    int out = 0;
    try
    {
        out = std::stoi( v, &used );
    }
    catch ( const std::exception& )
    {
        used = 0;
    }
    if ( used == 0 || used != v.size() )
        throw std::invalid_argument( "Beatnik_Probe_FmmCost: '" + key +
                                     "' needs an integer, got '" + v + "'" );
    return out;
}

double parseDouble( const std::string& key, const std::string& v )
{
    std::size_t used = 0;
    double out = 0.0;
    try
    {
        out = std::stod( v, &used );
    }
    catch ( const std::exception& )
    {
        used = 0;
    }
    if ( used == 0 || used != v.size() )
        throw std::invalid_argument( "Beatnik_Probe_FmmCost: '" + key +
                                     "' needs a number, got '" + v + "'" );
    return out;
}

Options parseOptions( int argc, char* argv[] )
{
    if ( argc < 4 )
        throw std::invalid_argument(
            "Beatnik_Probe_FmmCost: usage: <level> <steps> <sample_every> "
            "[key=value ...]; see this file's header. Got " +
            std::to_string( argc - 1 ) + " argument(s)." );

    Options o;
    o.level = parseInt( "level", argv[1] );
    o.steps = parseInt( "steps", argv[2] );
    o.sample_every = parseInt( "sample_every", argv[3] );
    if ( o.level < 0 || o.steps < 1 || o.sample_every < 1 )
        throw std::invalid_argument(
            "Beatnik_Probe_FmmCost: need level >= 0, steps >= 1 and "
            "sample_every >= 1" );

    for ( int i = 4; i < argc; ++i )
    {
        const std::string arg = argv[i];
        const auto eq = arg.find( '=' );
        if ( eq == std::string::npos || eq == 0 )
            throw std::invalid_argument( "Beatnik_Probe_FmmCost: argument '" +
                                         arg + "' is not key=value" );
        const std::string key = arg.substr( 0, eq );
        const std::string val = arg.substr( eq + 1 );
        if ( key == "ncrit" )
            o.ncrit = parseInt( key, val );
        else if ( key == "order" )
            o.order = parseInt( key, val );
        else if ( key == "mac_theta" )
            o.mac_theta = parseDouble( key, val );
        else if ( key == "m2l_op_count_cap" )
            o.m2l_op_count_cap = parseInt( key, val );
        else if ( key == "canopy_report" )
        {
            o.canopy_report = parseInt( key, val );
            if ( o.canopy_report != 0 && o.canopy_report != 1 )
                throw std::invalid_argument(
                    "Beatnik_Probe_FmmCost: canopy_report must be 0 or 1" );
        }
        else if ( key == "run" )
        {
            if ( val.empty() ||
                 val.find_first_of( " \t=" ) != std::string::npos )
                throw std::invalid_argument(
                    "Beatnik_Probe_FmmCost: run= needs a label with no "
                    "whitespace and no '='" );
            o.run = val;
        }
        else
            throw std::invalid_argument(
                "Beatnik_Probe_FmmCost: unknown key '" + key +
                "'. Accepted: ncrit, order, mac_theta, m2l_op_count_cap, "
                "canopy_report, run." );
    }
    if ( o.ncrit < 1 || o.m2l_op_count_cap < 0 || !( o.mac_theta > 0.0 ) )
        throw std::invalid_argument(
            "Beatnik_Probe_FmmCost: need ncrit >= 1, m2l_op_count_cap >= 0 "
            "and mac_theta > 0" );
    return o;
}

//---------------------------------------------------------------------------//
/// The milestone-0 frozen-mesh bubble, FMM-driven, no checkpoints. Field for
/// field `Beatnik_Probe_FmmKeyDemand.cpp::makeParams` except the BR
/// approximation, the far-field knobs and the checkpoint directory.
Beatnik::SolverParams makeParams( const Options& o )
{
    Beatnik::SolverParams p;

    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = o.level;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
    p.zmodel.sigma = 0.0;
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = Beatnik::KernelBlobMode::Length;
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = Beatnik::VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = Beatnik::BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    // --br-approximation fmm: the FMM drives the trajectory.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Fmm;
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    p.fmm.ncrit = o.ncrit;
    p.fmm.order = o.order;
    p.fmm.mac_theta = o.mac_theta;
    p.fmm.m2l_op_count_cap = o.m2l_op_count_cap;

    p.time.steps = o.steps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;
    p.cleanup.enabled = true;

    // No checkpoints: an empty directory makes writeCheckpoint() a no-op.
    p.checkpoint.every_steps = 0;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = "";

    return p;
}

//---------------------------------------------------------------------------//
/// Claim A's two field reductions, as `Beatnik_Test_Milestone0Fmm.cpp`
/// defines them: the max magnitude of `v`, and the max `|a - b|` with the
/// count of non-finite rows in `a`, both over owned rows and reduced globally.
template <class ExecSpace, class ViewType>
double fieldScale( MPI_Comm comm, const ViewType& v, int n )
{
    Real worst = 0;
    Kokkos::parallel_reduce(
        "beatnik_fmmcost_field_scale", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m ) {
            const Real mag = Kokkos::sqrt( v( i, 0 ) * v( i, 0 ) +
                                           v( i, 1 ) * v( i, 1 ) +
                                           v( i, 2 ) * v( i, 2 ) );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ) );
    double local = static_cast<double>( worst );
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MAX, comm );
    return out;
}

struct FieldDifference
{
    double max_abs = 0.0;
    long long nonfinite = 0;
};

template <class ExecSpace, class ViewA, class ViewB>
FieldDifference fieldDifference( MPI_Comm comm, const ViewA& a, const ViewB& b,
                                 int n )
{
    Real worst = 0;
    long long bad = 0;
    Kokkos::parallel_reduce(
        "beatnik_fmmcost_field_difference",
        Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m, long long& nb ) {
            Real d[3];
            bool finite = true;
            for ( int c = 0; c < 3; ++c )
            {
                d[c] = a( i, c ) - b( i, c );
                if ( !Kokkos::isfinite( a( i, c ) ) ||
                     !Kokkos::isfinite( b( i, c ) ) )
                    finite = false;
            }
            if ( !finite )
            {
                ++nb;
                return;
            }
            const Real mag =
                Kokkos::sqrt( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ), bad );

    FieldDifference out;
    double local = static_cast<double>( worst );
    MPI_Allreduce( &local, &out.max_abs, 1, MPI_DOUBLE, MPI_MAX, comm );
    MPI_Allreduce( &bad, &out.nonfinite, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

//---------------------------------------------------------------------------//
/// Mean and max over ranks of each element of `local`, in two reductions.
void reduceMeanMax( MPI_Comm comm, const std::vector<double>& local,
                    std::vector<double>& mean, std::vector<double>& max )
{
    int comm_size = 1;
    MPI_Comm_size( comm, &comm_size );
    const int n = static_cast<int>( local.size() );
    mean.assign( n, 0.0 );
    max.assign( n, 0.0 );
    MPI_Allreduce( local.data(), mean.data(), n, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( local.data(), max.data(), n, MPI_DOUBLE, MPI_MAX, comm );
    for ( double& m : mean )
        m /= comm_size;
}

/// A row under construction: `key=value` pairs in insertion order, issued in
/// one `printf` so concurrent ranks cannot interleave inside it.
class Row
{
  public:
    explicit Row( const char* tag ) { _os << "[" << tag << "]"; }

    Row& add( const std::string& k, const std::string& v )
    {
        _os << " " << k << "=" << v;
        return *this;
    }
    Row& add( const std::string& k, long long v )
    {
        return add( k, std::to_string( v ) );
    }
    Row& add( const std::string& k, int v )
    {
        return add( k, std::to_string( v ) );
    }
    Row& add( const std::string& k, double v )
    {
        char buf[32];
        std::snprintf( buf, sizeof( buf ), "%.9g", v );
        return add( k, std::string( buf ) );
    }
    /// The bare key is the mean over ranks and `<k>.max` the max, or `-1`
    /// for both when the quantity is not compiled in.
    Row& addMeanMax( const std::string& k, bool available, double mean,
                     double max )
    {
        add( k, available ? mean : -1.0 );
        return add( k + ".max", available ? max : -1.0 );
    }

    void print() const
    {
        std::printf( "%s\n", _os.str().c_str() );
        std::fflush( stdout );
    }

  private:
    std::ostringstream _os;
};

//---------------------------------------------------------------------------//
/// Everything one window accumulates on this rank.
struct Window
{
    int steps = 0;
    long long evals = 0;
    double step_wall = 0.0;
    std::map<std::string, double> ff;
    std::map<std::string, double> canopy;
    long long maint[Beatnik::FarFieldDiagnostics::num_maintenance] = {
        0, 0, 0, 0 };
    long long keys_built_delta = 0;

    // The window's last step, for [acc]'s direct estimate.
    double last_step_wall = 0.0;
    double last_step_ff_total = 0.0;

    void clear() { *this = Window(); }
};

const char* const kMaintNames[Beatnik::FarFieldDiagnostics::num_maintenance] =
    { "maint_setup", "maint_migrate", "maint_rebalance", "maint_rebuild" };

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runProbe( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
#ifndef BEATNIK_ENABLE_CANOPY
    (void)argc;
    (void)argv;
    rec.fail( "Beatnik_Probe_FmmCost needs a Canopy build (+canopy): it "
              "measures the FMM path" );
    return;
#else
    using solver_type = Beatnik::Solver<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using fmm_type = Beatnik::BRSolverFMM<ExecSpace, MemSpace>;
    using direct_type = Beatnik::BRSolverDirect<ExecSpace, MemSpace>;
    using vector_view =
        Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;
    constexpr int NM = Beatnik::FarFieldDiagnostics::num_maintenance;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );

    const Options opt = parseOptions( argc, argv );
    std::string run = opt.run;
    if ( run.empty() )
    {
        std::ostringstream os;
        os << "L" << opt.level << "_" << ExecSpace::name() << "_np"
           << comm_size << "_ncrit" << opt.ncrit << "_order" << opt.order
           << "_rep" << opt.canopy_report;
        run = os.str();
    }

    // Identical on every rank: argv is. The tables it gates are collectives.
    Canopy::Profiling::set_reporting( opt.canopy_report != 0 );

    const long long want_vertices = verticesForLevel( opt.level );
    const long long want_faces = facesForLevel( opt.level );

    const char* hash_env = std::getenv( "BEATNIK_CANOPY_HASH" );
    const std::string canopy_hash =
        ( hash_env && *hash_env ) ? hash_env : "unknown";

    if ( rank == 0 )
    {
        std::printf( "[fmmcost] header run=%s level=%d vertices=%lld ranks=%d "
                     "space=%s steps=%d sample_every=%d ncrit=%d order=%d "
                     "mac_theta=%.17g m2l_op_count_cap=%d canopy_report=%d "
                     "beatnik_ff_timers=%d canopy_timers=%d "
                     "canopy_profiling_level=%d canopy_hash=%s\n",
                     run.c_str(), opt.level, want_vertices, comm_size,
                     ExecSpace::name(), opt.steps, opt.sample_every,
                     opt.ncrit, opt.order, opt.mac_theta,
                     opt.m2l_op_count_cap, opt.canopy_report,
                     kHaveFfTimers ? 1 : 0, kHaveCanopyTimers ? 1 : 0,
                     kCanopyLevel, canopy_hash.c_str() );
        std::fflush( stdout );
    }

    const Beatnik::SolverParams params = makeParams( opt );
    solver_type solver( comm, params );
    solver.setup();

    auto& mesh = solver.mesh();
    const auto& state = solver.state();
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), want_vertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), want_faces );

    const auto* driving =
        dynamic_cast<const fmm_type*>( &solver.brSolver() );
    if ( !driving )
        throw std::runtime_error(
            "Beatnik_Probe_FmmCost: the driving BR solver is not BRSolverFMM; "
            "the probe needs --br-approximation fmm" );
    const auto& ddiag = driving->farField().diagnostics();

    // The [acc] comparators: constructed ONCE, so the standalone FMM's
    // operator cache behaves as the add-canopy-t6 probe's did.
    auto quadrature = Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
        Beatnik::SourceQuadrature::Vertex );
    direct_type direct( comm );
    fmm_type fmm( comm, params.fmm );
    const Beatnik::ZModelParams& zparams = solver.params().zmodel;
    vector_view u_direct( "beatnik_fmmcost_u_direct", mesh.ownedVertexCount() );
    vector_view u_fmm( "beatnik_fmmcost_u_fmm", mesh.ownedVertexCount() );

    Window win;
    // Run totals of the per-rank window sums, for [cost-total].
    double run_wall = 0.0;
    long long run_evals = 0;
    std::map<std::string, double> run_ff;
    std::map<std::string, double> run_canopy;
    long long run_maint[NM] = { 0, 0, 0, 0 };

    for ( int step = 1; step <= opt.steps; ++step )
    {
        // ---- snapshot, step, difference ----
        const long long ev0 = ddiag.evaluations;
        long long m0[NM];
        for ( int a = 0; a < NM; ++a )
            m0[a] = ddiag.maintenance_counts[a];
        const long long kb0 = ddiag.local_m2l_op_keys_built;
        const Snapshot b0 = beatnikTotals();
        const Snapshot c0 = canopyTotals();

        Kokkos::fence();
        const double t0 = MPI_Wtime();
        const bool ok = solver.advanceOneStep();
        Kokkos::fence();
        const double t1 = MPI_Wtime();

        if ( !ok )
        {
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << opt.steps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time();
            rec.fail( os.str() );
            break;
        }
        if ( mesh.globalVertexCount() != want_vertices ||
             mesh.globalFaceCount() != want_faces )
        {
            rec.fail( "ENTITY COUNTS CHANGED at step " +
                      std::to_string( step ) +
                      ": adaptivity leaked into the frozen-mesh "
                      "configuration" );
            break;
        }

        const Snapshot b1 = beatnikTotals();
        const Snapshot c1 = canopyTotals();

        ++win.steps;
        win.step_wall += t1 - t0;
        win.evals += ddiag.evaluations - ev0;
        for ( int a = 0; a < NM; ++a )
            win.maint[a] += ddiag.maintenance_counts[a] - m0[a];
        win.keys_built_delta += ddiag.local_m2l_op_keys_built - kb0;
        for ( const auto& k : kFfPhases )
            win.ff[k] += delta( b1, b0, k );
        win.ff[kFfTotal] += delta( b1, b0, kFfTotal );
        for ( const auto& k : kCanopyKeys )
            win.canopy[k.name] += delta( c1, c0, k.name );
        win.last_step_wall = t1 - t0;
        win.last_step_ff_total = delta( b1, b0, kFfTotal );

        const bool sample =
            step == 1 || step == opt.steps || step % opt.sample_every == 0;
        if ( !sample )
            continue;

        // ---- [cost] ----
        run_wall += win.step_wall;
        run_evals += win.evals;
        for ( int a = 0; a < NM; ++a )
            run_maint[a] += win.maint[a];
        for ( const auto& kv : win.ff )
            run_ff[kv.first] += kv.second;
        for ( const auto& kv : win.canopy )
            run_canopy[kv.first] += kv.second;

        const double per_eval =
            win.evals > 0 ? 1.0 / static_cast<double>( win.evals ) : 0.0;
        std::vector<double> local;
        local.push_back( win.step_wall / win.steps );
        for ( const auto& k : kFfPhases )
            local.push_back( win.ff[k] * per_eval );
        local.push_back( win.ff[kFfTotal] * per_eval );
        for ( const auto& k : kCanopyKeys )
            local.push_back( win.canopy[k.name] * per_eval );
        local.push_back( static_cast<double>( win.keys_built_delta ) );
        local.push_back( static_cast<double>( ddiag.local_m2l_op_cache_size ) );
        local.push_back(
            static_cast<double>( ddiag.local_m2l_unique_op_count ) );
        local.push_back(
            static_cast<double>( ddiag.local_m2l_occupied_depths ) );
        local.push_back( win.last_step_wall );
        local.push_back( win.last_step_ff_total );
        std::vector<double> mean, max;
        reduceMeanMax( comm, local, mean, max );

        std::size_t i = 0;
        Row cost( "cost" );
        cost.add( "run", run ).add( "step", step ).add(
            "time", static_cast<double>( solver.time() ) );
        cost.add( "steps", win.steps ).add( "evals", win.evals );
        cost.addMeanMax( "step_wall", true, mean[i], max[i] );
        ++i;
        for ( const auto& k : kFfPhases )
        {
            cost.addMeanMax( k, kHaveFfTimers, mean[i], max[i] );
            ++i;
        }
        cost.addMeanMax( kFfTotal, kHaveFfTimers, mean[i], max[i] );
        ++i;
        for ( const auto& k : kCanopyKeys )
        {
            cost.addMeanMax( k.name, canopyKeyAvailable( k ), mean[i],
                             max[i] );
            ++i;
        }
        for ( int a = 0; a < NM; ++a )
            cost.add( kMaintNames[a], win.maint[a] );
        cost.addMeanMax( "keys_built_delta", true, mean[i], max[i] );
        ++i;
        cost.addMeanMax( "cache_size", true, mean[i], max[i] );
        ++i;
        cost.addMeanMax( "unique_ops", true, mean[i], max[i] );
        ++i;
        cost.addMeanMax( "occupied_depths", true, mean[i], max[i] );
        ++i;
        const double last_wall_max = max[i++];
        const double last_ff_total_max = max[i++];
        cost.add( "fb_range_guard", ddiag.global_m2l_fallback_pairs_range_guard )
            .add( "fb_count_cap", ddiag.global_m2l_fallback_pairs_count_cap )
            .add( "fb_dropped", ddiag.global_m2l_fallback_pairs_depth_dropped )
            .add( "m2l_pairs", ddiag.global_m2l_pair_count )
            .add( "m2l_fallback", ddiag.global_m2l_fallback_pair_count )
            .add( "p2p_frac", ddiag.p2p_pair_fraction )
            .add( "particles", ddiag.global_particle_count );
        if ( rank == 0 )
            cost.print();
        win.clear();

        // ---- [acc], outside the window ----
        const int n_owned = mesh.ownedVertexCount();
        mesh.haloExchange();
        geometry_type geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );
        state.updateSheetVector( mesh, geometry );
        if ( static_cast<int>( u_direct.extent( 0 ) ) != n_owned )
        {
            Kokkos::realloc( u_direct, n_owned );
            Kokkos::realloc( u_fmm, n_owned );
        }

        Kokkos::fence();
        MPI_Barrier( comm );
        double a0 = MPI_Wtime();
        direct.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                         zparams, u_direct );
        Kokkos::fence();
        double direct_wall_local = MPI_Wtime() - a0;
        MPI_Barrier( comm );
        a0 = MPI_Wtime();
        fmm.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                      zparams, u_fmm );
        Kokkos::fence();
        double fmm_wall_local = MPI_Wtime() - a0;

        double walls_local[2] = { direct_wall_local, fmm_wall_local };
        double walls[2] = { 0.0, 0.0 };
        MPI_Allreduce( walls_local, walls, 2, MPI_DOUBLE, MPI_MAX, comm );

        const FieldDifference d =
            fieldDifference<ExecSpace>( comm, u_fmm, u_direct, n_owned );
        const double scale = fieldScale<ExecSpace>( comm, u_direct, n_owned );
        const bool relative = scale > kFieldScaleFloor;
        const double acc_rel = relative ? d.max_abs / scale : d.max_abs;
        if ( d.nonfinite != 0 )
            rec.fail( "non-finite velocity in the [acc] comparison at step " +
                      std::to_string( step ) );

        const double direct_step_est =
            kHaveFfTimers
                ? 3.0 * walls[0] + ( last_wall_max - last_ff_total_max )
                : -1.0;

        Row acc( "acc" );
        acc.add( "run", run )
            .add( "step", step )
            .add( "time", static_cast<double>( solver.time() ) )
            .add( "acc_rel", acc_rel )
            .add( "acc_relative", relative ? 1 : 0 )
            .add( "acc_max_abs", d.max_abs )
            .add( "acc_scale", scale )
            .add( "acc_nonfinite", d.nonfinite )
            .add( "direct_wall", walls[0] )
            .add( "fmm_wall", walls[1] )
            .add( "fmm_step_wall", last_wall_max )
            .add( "direct_step_est", direct_step_est );
        if ( rank == 0 )
            acc.print();
    }

    // ---- [cost-total] ----
    std::vector<double> local;
    local.push_back( run_wall );
    for ( const auto& k : kFfPhases )
        local.push_back( run_ff[k] );
    local.push_back( run_ff[kFfTotal] );
    for ( const auto& k : kCanopyKeys )
        local.push_back( run_canopy[k.name] );
    std::vector<double> mean, max;
    reduceMeanMax( comm, local, mean, max );

    std::size_t i = 0;
    const double wall = mean[i++];
    std::map<std::string, double> ff_sum;
    for ( const auto& k : kFfPhases )
        ff_sum[k] = mean[i++];
    const double ff_total = mean[i++];
    std::map<std::string, double> canopy_sum;
    for ( const auto& k : kCanopyKeys )
        canopy_sum[k.name] = mean[i++];

    auto share = [&]( bool available, double num, double den ) {
        return ( available && den > 0.0 ) ? num / den : -1.0;
    };

    Row total( "cost-total" );
    total.add( "run", run )
        .add( "level", opt.level )
        .add( "ranks", comm_size )
        .add( "space", std::string( ExecSpace::name() ) )
        .add( "steps", solver.step() )
        .add( "evals", run_evals )
        .add( "wall", wall )
        .add( "ff_evaluate_total_sum", kHaveFfTimers ? ff_total : -1.0 );
    for ( int a = 0; a < NM; ++a )
        total.add( std::string( "hist_" ) + ( kMaintNames[a] + 6 ),
                   run_maint[a] );
    for ( const auto& k : kFfPhases )
        total.add( "share." + k, share( kHaveFfTimers, ff_sum[k], ff_total ) );
    total.add( "share.ff_evaluate_total_of_step",
               share( kHaveFfTimers, ff_total, wall ) );
    const CanopyKey op_build{ "ilist_s4_op_table_build", true };
    total.add( "share.op_build_of_ff_solve",
               share( kHaveFfTimers && canopyKeyAvailable( op_build ),
                      canopy_sum[op_build.name], ff_sum["ff_solve"] ) );
    total.add( "canopy_hash", canopy_hash );
    if ( rank == 0 )
        total.print();

    solver.finalize();
#endif
}

} // namespace

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int rc = 1;
    {
        Beatnik::Test::Recorder rec( "Beatnik_Probe_FmmCost" );
        try
        {
            // Defined by the per-backend shim tests/CMakeLists.txt generates.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using exec_space = BEATNIK_TEST_EXEC_SPACE;
            runProbe<exec_space, typename exec_space::memory_space>( rec, argc,
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

    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
