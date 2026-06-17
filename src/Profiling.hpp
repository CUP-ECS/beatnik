/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                   *
 ****************************************************************************/
/**
 * @file Profiling.hpp
 * @brief Beatnik-local profiling helpers, modeled on Canopy's
 *        Canopy_Profiling.hpp. Provides a process-local timer registry, an
 *        RAII ScopedTimer, and a per-rank Min/Max/Mean/Imbalance print table.
 *        Used to attribute the per-call cost of the FMM BR solver's
 *        computeInterfaceVelocity at profiling level >= 2.
 */

#ifndef BEATNIK_PROFILING_HPP
#define BEATNIK_PROFILING_HPP

// ---------------------------------------------------------------------------
// Profiling level hierarchy (mirrors Canopy's semantics)
//
//   0 — off (no instrumentation)
//   1 — basic: per-call total + auto_maintain action histogram (already
//       implemented inline in FmmBRSolver.hpp)
//   2 — detailed: per-sub-phase timing inside
//       FmmBRSolver::computeInterfaceVelocity (pack, distributor builds,
//       migrates, auto_maintain, Canopy solve, cross product, writeZdot)
//   3 — verbose (reserved for future even finer-grained timers)
//
// BEATNIK_ENABLE_PROFILING / BEATNIK_PROFILING_LEVEL are added as compile
// definitions by the top-level CMake when the resolved level is > 0. Set with
// -DBeatnik_PROFILING_LEVEL=2 (or the `profiling_level=2` spack variant).
// ---------------------------------------------------------------------------
#ifndef BEATNIK_PROFILING_LEVEL
#  ifdef BEATNIK_ENABLE_PROFILING
#    define BEATNIK_PROFILING_LEVEL 1
#  else
#    define BEATNIK_PROFILING_LEVEL 0
#  endif
#endif

#ifdef BEATNIK_ENABLE_PROFILING

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace Beatnik
{
namespace Profiling
{

// ---------------------------------------------------------------------------
// Phase-key constants — used as keys in the timer registry and as labels in
// the PhaseEntry descriptors. Defined here so all instrumented headers share
// the same strings without risk of typos.
// ---------------------------------------------------------------------------

// FmmBRSolver::computeInterfaceVelocity sub-phases (level 2).
static constexpr const char* TIMER_CIV_TOTAL     = "civ_total";
static constexpr const char* TIMER_PACK          = "pack_grid_particles";
static constexpr const char* TIMER_SETUP         = "canopy_setup";
static constexpr const char* TIMER_FWD_DIST      = "build_fwd_distributor";
static constexpr const char* TIMER_FWD_MIGRATE   = "fwd_migrate";
static constexpr const char* TIMER_AUTO_MAINTAIN = "auto_maintain";
static constexpr const char* TIMER_SOLVE         = "canopy_solve";
static constexpr const char* TIMER_CROSS         = "cross_product";
static constexpr const char* TIMER_REV_DIST      = "rev_distributor_build";
static constexpr const char* TIMER_REV_MIGRATE   = "rev_migrate";
static constexpr const char* TIMER_WRITE_ZDOT    = "write_zdot";

// ---------------------------------------------------------------------------
// Timer registry — process-local accumulator map, key -> elapsed seconds.
// Using a function-local static so this is safe in a header-only library:
// exactly one instance per process, initialized on first use.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, double>& timer_registry()
{
    static std::unordered_map<std::string, double> s_reg;
    return s_reg;
}

inline void reset_timers()
{
    timer_registry().clear();
}

inline void accumulate( const char* key, double elapsed )
{
    timer_registry()[key] += elapsed;
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII guard. Records wall time at construction via MPI_Wtime
// and accumulates the elapsed time at destruction. Non-copyable.
//
// Unlike Canopy's ScopedTimer, the destructor issues a Kokkos::fence() before
// reading the clock: Beatnik's sub-phase kernels (packGridParticles,
// crossProduct, writeZdot) do not all fence internally, so the fence is what
// makes the captured wall time reflect true device work. Because this struct
// only compiles at profiling level >= 2, the fence costs nothing in
// production builds.
// ---------------------------------------------------------------------------
struct ScopedTimer
{
    const char* key;
    double      t0;

    explicit ScopedTimer( const char* phase_key )
        : key( phase_key )
        , t0( MPI_Wtime() )
    {
        Kokkos::Profiling::pushRegion( key );
    }

    ~ScopedTimer()
    {
        Kokkos::fence();
        Kokkos::Profiling::popRegion();
        accumulate( key, MPI_Wtime() - t0 );
    }

    ScopedTimer( const ScopedTimer& ) = delete;
    ScopedTimer& operator=( const ScopedTimer& ) = delete;
};

// ---------------------------------------------------------------------------
// PhaseEntry — describes one row in the printed timing table.
// ---------------------------------------------------------------------------
struct PhaseEntry
{
    const char* label;
    const char* key;
    int         indent; // 0 = top-level, 1 = sub-phase (2 leading spaces each)
};

// ---------------------------------------------------------------------------
// Ordered phase entry list for the FmmBRSolver breakdown. Sub-phases sum to
// approximately the "computeInterfaceVel total" minus the small zeroZdot
// kernel, which is intentionally left unattributed (as Canopy leaves slack).
// ---------------------------------------------------------------------------
inline std::vector<PhaseEntry> fmm_phase_entries()
{
    return {
        { "computeInterfaceVel total", TIMER_CIV_TOTAL,     0 },
        { "Pack grid particles",       TIMER_PACK,          1 },
        { "Canopy setup (first call)", TIMER_SETUP,         1 },
        { "Build forward distributor", TIMER_FWD_DIST,      1 },
        { "Forward migrate",           TIMER_FWD_MIGRATE,   1 },
        { "auto_maintain",             TIMER_AUTO_MAINTAIN, 1 },
        { "Canopy solve",              TIMER_SOLVE,         1 },
        { "Cross product (gradients)", TIMER_CROSS,         1 },
        { "Build reverse distributor", TIMER_REV_DIST,      1 },
        { "Reverse migrate",           TIMER_REV_MIGRATE,   1 },
        { "Write zdot",                TIMER_WRITE_ZDOT,    1 },
    };
}

// ---------------------------------------------------------------------------
// print_timing_table
//
// Gathers per-rank timing data via three MPI_Reduce calls (MIN, MAX, SUM)
// to rank 0. Only rank 0 prints the formatted table. Other ranks return
// immediately after the reduce, so this is a collective call.
//
// Parameters:
//   comm         - MPI communicator (same one used for the solve)
//   section_name - printed in the header line (e.g.
//                  "FmmBRSolver::computeInterfaceVelocity")
//   phases       - ordered list of PhaseEntry rows to print
// ---------------------------------------------------------------------------
inline void print_timing_table( MPI_Comm comm, const char* section_name,
                                const std::vector<PhaseEntry>& phases )
{
    int rank, nprocs;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &nprocs );

    const auto& reg = timer_registry();
    const int   N   = static_cast<int>( phases.size() );

    std::vector<double> local_vals( N, 0.0 );
    for ( int i = 0; i < N; i++ )
    {
        auto it = reg.find( phases[i].key );
        if ( it != reg.end() )
            local_vals[i] = it->second;
    }

    std::vector<double> min_vals( N ), max_vals( N ), sum_vals( N );
    MPI_Reduce( local_vals.data(), min_vals.data(), N,
                MPI_DOUBLE, MPI_MIN, 0, comm );
    MPI_Reduce( local_vals.data(), max_vals.data(), N,
                MPI_DOUBLE, MPI_MAX, 0, comm );
    MPI_Reduce( local_vals.data(), sum_vals.data(), N,
                MPI_DOUBLE, MPI_SUM, 0, comm );

    if ( rank != 0 )
        return;

    // Column widths
    static constexpr int COL_LABEL = 36;
    static constexpr int COL_NUM   = 9;

    std::printf( "\n[Beatnik Diagnostics] %s timing (%d MPI rank%s)\n",
                 section_name, nprocs, nprocs > 1 ? "s" : "" );
    std::printf( "  %-*s  %*s  %*s  %*s  %s\n",
                 COL_LABEL, "Phase",
                 COL_NUM,   "Min (s)",
                 COL_NUM,   "Max (s)",
                 COL_NUM,   "Mean (s)",
                 "Imbalance" );

    const int sep_len = COL_LABEL + 3 * ( COL_NUM + 2 ) + 12;
    for ( int i = 0; i < sep_len; i++ )
        std::putchar( '-' );
    std::putchar( '\n' );

    const double inv_nprocs = 1.0 / static_cast<double>( nprocs );
    for ( int i = 0; i < N; i++ )
    {
        const double mn   = min_vals[i];
        const double mx   = max_vals[i];
        const double mean = sum_vals[i] * inv_nprocs;
        const double imb  = ( mean > 0.0 )
                            ? ( mx - mean ) / mean * 100.0
                            : 0.0;

        // Build indented label
        char buf[64];
        const int indent_spaces = 2 * phases[i].indent;
        std::snprintf( buf, sizeof( buf ), "%*s%s",
                       indent_spaces, "", phases[i].label );

        std::printf( "  %-*s  %*.3f  %*.3f  %*.3f  %.1f%%\n",
                     COL_LABEL, buf,
                     COL_NUM,   mn,
                     COL_NUM,   mx,
                     COL_NUM,   mean,
                     imb );
    }
    std::putchar( '\n' );
    std::fflush( stdout );
}

} // namespace Profiling
} // namespace Beatnik

#endif // BEATNIK_ENABLE_PROFILING

// ---------------------------------------------------------------------------
// Convenience macros — defined whether or not profiling is enabled so
// instrumentation in other headers compiles in every mode. Detailed (level 2)
// timers compile away to no-ops below level 2 so call sites can be left in
// place unguarded.
// ---------------------------------------------------------------------------
// Token-paste helpers so __LINE__ is expanded (a bare `_t_##__LINE__` would
// paste the literal text "__LINE__"). Unique per line -> no -Wshadow between
// nested timer scopes.
#define BEATNIK_PROF_CONCAT_( a, b ) a##b
#define BEATNIK_PROF_CONCAT( a, b ) BEATNIK_PROF_CONCAT_( a, b )

#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 2
#  define BEATNIK_SCOPED_TIMER_DETAILED( key )                                 \
       ::Beatnik::Profiling::ScopedTimer                                       \
           BEATNIK_PROF_CONCAT( _beatnik_timer_d_, __LINE__ )( (key) )
#  define BEATNIK_PRINT_FMM_TIMERS( comm ) \
       ::Beatnik::Profiling::print_timing_table( \
           (comm), "FmmBRSolver::computeInterfaceVelocity", \
           ::Beatnik::Profiling::fmm_phase_entries() )
#else
#  define BEATNIK_SCOPED_TIMER_DETAILED( key ) do {} while ( 0 )
#  define BEATNIK_PRINT_FMM_TIMERS( comm )     do { (void)( comm ); } while ( 0 )
#endif

#endif // BEATNIK_PROFILING_HPP
</content>
