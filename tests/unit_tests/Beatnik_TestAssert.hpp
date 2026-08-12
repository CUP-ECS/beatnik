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
 * @file Beatnik_TestAssert.hpp
 * @brief Minimal, header-only, self-validating assertion recorder for the
 *        `unit` tier.
 *
 * WHY NOT GTEST, WHICH THE PROJECT ALREADY DEPENDS ON
 * ---------------------------------------------------
 * `cmake/test_harness/test_harness.cmake` drives gtest and is the right tool
 * for a test that wants one binary per Kokkos backend generated from a
 * `tst<Name>.hpp`. It is the wrong tool for the `unit`-tier tests this file
 * serves, for one reason that is about the *build mode* rather than about
 * taste:
 *
 *   **In spack mode there is no build tree, so there is no `ctest` to run.**
 *   The gate wrapper already works around this by walking an installed
 *   manifest and launching binaries directly (see
 *   `scripts/tuolumne/run_regression_minset.flux`), and the unit tier has to do
 *   the same.
 *
 * A binary launched directly is judged by exactly one thing: its **exit code**.
 * So a `unit` test must decide its own verdict and return non-zero on failure.
 * That single property satisfies both modes at once — ctest's default success
 * criterion *is* exit code zero, so `add_test` needs no
 * `PASS_REGULAR_EXPRESSION` and no other special handling, and the bare binary
 * is authoritative when run by hand or from a batch script.
 *
 * WHAT THIS GIVES YOU
 * -------------------
 * Three checks (boolean, exact integer equality, relative-tolerance floating
 * point), each recording the expression, the expected and actual values, the
 * tolerance where relevant, and the file and line. Failures **accumulate**
 * rather than aborting: a run that reports all of its assertion groups is worth
 * far more than one that dies on the first count mismatch, because the pattern
 * of which checks failed is usually what identifies the bug.
 *
 * The output is one greppable tally line plus one detail line per failure:
 *
 *     [FAIL] Beatnik_Test_MeshGeometry (7/9 checks)
 *     [FAIL]   check 4: volume ... expected 6.32350731246695e-02
 *                        actual   6.32350731246712e-02  rel 2.7e-14 > 1.0e-14
 *
 * so whoever reads a batch log does not have to interpret it.
 *
 * No external test framework and no dependency beyond the standard library —
 * deliberately, since a `unit` test must be runnable from an installed prefix
 * where gtest may not be on the link line.
 */

#ifndef BEATNIK_TESTASSERT_HPP
#define BEATNIK_TESTASSERT_HPP

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace Beatnik
{
namespace Test
{

//---------------------------------------------------------------------------//
/**
 * @brief Accumulates check results and prints the verdict.
 *
 * Not thread-safe and **not MPI-aware**, deliberately: it records what happened
 * on *this* rank.
 *
 * **The multi-rank question, settled at T1c** (which needed it first, for the
 * `regression` tier's rank sweep). The reduction stays **outside** this class:
 * a multi-rank test calls `report()` on every rank — so the log names which
 * rank failed, which a rank-0-only tally would hide — and then reduces the
 * returned exit codes with `MPI_Allreduce(..., MPI_MAX, comm)` to reach one
 * verdict. See `tests/regression_tests/Beatnik_Test_InitialConditions.cpp`,
 * which does exactly that in four lines.
 *
 * The reduction is not put in here because it would have to be collective, and
 * a collective inside the reporter would deadlock precisely in the case that
 * matters most: one rank taking an early exception path and never reaching
 * `report()` while its peers block in the reduce. Keeping it at the call site
 * makes that visible.
 */
class Recorder
{
  public:
    explicit Recorder( std::string name )
        : _name( std::move( name ) )
    {
    }

    /// Boolean check.
    void checkTrue( bool value, const char* expr, const char* file, int line )
    {
        ++_total;
        if ( value )
            return;
        std::ostringstream os;
        os << "check " << _total << ": expected true, got false\n"
           << "           expr:   " << expr << "\n"
           << "           at:     " << file << ":" << line;
        _failures.push_back( os.str() );
    }

    /// Exact equality of two integers. Exact is the right comparison for a
    /// count, an index, or a reduced entity total (Tessera reduces those as
    /// `long long`, so they are exact even across rank counts).
    void checkEqual( long long actual, long long expected, const char* expr,
                     const char* file, int line )
    {
        ++_total;
        if ( actual == expected )
            return;
        std::ostringstream os;
        os << "check " << _total << ": expected " << expected << ", got "
           << actual << "\n"
           << "           expr:   " << expr << "\n"
           << "           at:     " << file << ":" << line;
        _failures.push_back( os.str() );
    }

    /**
     * @brief Relative-tolerance floating-point check.
     *
     * Passes when `|actual - expected| <= rtol * |expected|`, or when both are
     * exactly zero. A non-finite `actual` always fails — otherwise a NaN would
     * pass every comparison of the form `!(diff > tol)`, which is the classic
     * way a tolerance check silently stops testing anything.
     */
    void checkClose( double actual, double expected, double rtol,
                     const char* expr, const char* file, int line )
    {
        ++_total;
        if ( std::isfinite( actual ) )
        {
            const double diff = std::fabs( actual - expected );
            const double tol = rtol * std::fabs( expected );
            if ( diff <= tol )
                return;
            std::ostringstream os;
            os.precision( 17 );
            os << "check " << _total << ": expected " << expected << ", got "
               << actual << "\n";
            os.precision( 3 );
            os << "           rel err "
               << ( std::fabs( expected ) > 0.0 ? diff / std::fabs( expected )
                                                : diff )
               << " > rtol " << rtol << "\n"
               << "           expr:   " << expr << "\n"
               << "           at:     " << file << ":" << line;
            _failures.push_back( os.str() );
            return;
        }
        std::ostringstream os;
        os.precision( 17 );
        os << "check " << _total << ": expected " << expected
           << ", got a NON-FINITE value (" << actual << ")\n"
           << "           expr:   " << expr << "\n"
           << "           at:     " << file << ":" << line;
        _failures.push_back( os.str() );
    }

    /**
     * @brief Record a failure that is not a check — an unexpected exception, a
     *        precondition the test could not establish.
     *
     * Counts as one failed check, so an aborted run can never report
     * `[PASS] ... (0/0 checks)`. That matters: the most likely unexpected
     * exception in this codebase is a `BEATNIK_NOT_IMPLEMENTED` from a stub on
     * a path the test author did not realize it was on, and "zero checks, all
     * passed" is the worst possible way to report it.
     */
    void fail( const std::string& why )
    {
        ++_total;
        _failures.push_back( "check " + std::to_string( _total ) + ": " + why );
    }

    /// Note something in the log without asserting on it. Use for the measured
    /// values a later reader will want (rank count, execution space, the actual
    /// scalars), so a failure report carries its own context.
    void note( const std::string& text ) const
    {
        std::printf( "[note] %s: %s\n", _name.c_str(), text.c_str() );
    }

    /**
     * @brief Print the tally and every failure detail; return the process exit
     *        code.
     * @return `0` iff every check passed and at least one check ran.
     */
    int report() const
    {
        const int passed = _total - static_cast<int>( _failures.size() );
        const bool ok = _failures.empty() && _total > 0;
        std::printf( "[%s] %s (%d/%d checks)\n", ok ? "PASS" : "FAIL",
                     _name.c_str(), passed, _total );
        for ( const auto& f : _failures )
            std::printf( "[FAIL]   %s\n", f.c_str() );
        if ( _total == 0 )
            std::printf(
                "[FAIL]   no checks ran at all, which is not a pass\n" );
        std::fflush( stdout );
        return ok ? 0 : 1;
    }

  private:
    std::string _name;
    int _total = 0;
    std::vector<std::string> _failures;
};

} // namespace Test
} // namespace Beatnik

//! Boolean check. `rec` is a `Beatnik::Test::Recorder`.
#define BEATNIK_CHECK_TRUE( rec, expr )                                        \
    ( rec ).checkTrue( ( expr ), #expr, __FILE__, __LINE__ )

//! Exact integer equality.
#define BEATNIK_CHECK_EQ( rec, actual, expected )                              \
    ( rec ).checkEqual( static_cast<long long>( actual ),                      \
                        static_cast<long long>( expected ),                    \
                        #actual " == " #expected, __FILE__, __LINE__ )

//! Relative-tolerance floating-point equality.
#define BEATNIK_CHECK_CLOSE( rec, actual, expected, rtol )                     \
    ( rec ).checkClose( static_cast<double>( actual ),                         \
                        static_cast<double>( expected ),                       \
                        static_cast<double>( rtol ), #actual " ~= " #expected, \
                        __FILE__, __LINE__ )

#endif // BEATNIK_TESTASSERT_HPP
