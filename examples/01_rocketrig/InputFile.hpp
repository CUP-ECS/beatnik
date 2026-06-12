/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file InputFile.hpp
 * @brief key = value input-file parser for the rocketrig example.
 *
 * Scoped to the rocketrig example — not part of the public Beatnik
 * library. The parser reads a flat list of `key = value` lines, with
 * `#` to-EOL comments and blank lines allowed. Unknown keys, malformed
 * lines, and bad enum values all throw with the file path and line
 * number in the message. Missing keys keep the caller-supplied
 * defaults in `cl`, so a near-empty `.in` file is valid.
 */

#ifndef BEATNIK_EXAMPLE_INPUTFILE_HPP
#define BEATNIK_EXAMPLE_INPUTFILE_HPP

#include <Beatnik_Config.hpp>
#include <BoundaryCondition.hpp>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Beatnik
{
namespace Example
{

inline std::string trim( const std::string& s )
{
    const char* ws = " \t\r\n";
    const auto a = s.find_first_not_of( ws );
    if ( a == std::string::npos ) return {};
    const auto b = s.find_last_not_of( ws );
    return s.substr( a, b - a + 1 );
}

/* Parse a value as int with full string consumption; throw on garbage. */
inline int parseIntValue( const std::string& key, const std::string& v )
{
    try {
        size_t pos = 0;
        const long long x = std::stoll( v, &pos );
        if ( pos != v.size() )
            throw std::runtime_error( "trailing characters" );
        return static_cast<int>( x );
    } catch ( const std::exception& e ) {
        throw std::runtime_error( "invalid integer for '" + key +
                                  "': '" + v + "' (" + e.what() + ")" );
    }
}

inline double parseDoubleValue( const std::string& key, const std::string& v )
{
    try {
        size_t pos = 0;
        const double x = std::stod( v, &pos );
        if ( pos != v.size() )
            throw std::runtime_error( "trailing characters" );
        return x;
    } catch ( const std::exception& e ) {
        throw std::runtime_error( "invalid float for '" + key +
                                  "': '" + v + "' (" + e.what() + ")" );
    }
}

/* Apply an enum map; on miss, throw with the offending value and the
 * full list of accepted options. */
template <class Enum>
Enum parseEnumValue( const std::string& key, const std::string& v,
                     const std::vector<std::pair<std::string, Enum>>& options )
{
    for ( const auto& kv : options )
        if ( kv.first == v ) return kv.second;

    std::ostringstream oss;
    oss << "invalid value for '" << key << "': '" << v
        << "' (expected one of:";
    for ( size_t i = 0; i < options.size(); ++i )
        oss << ( i == 0 ? " " : ", " ) << options[i].first;
    oss << ")";
    throw std::runtime_error( oss.str() );
}

/* Read `path` into `cl`. `cl` is expected to be pre-populated with
 * defaults; only keys present in the file are overwritten. Throws
 * std::runtime_error with file:line context on any error. */
template <class ClArgsT>
void parseInputFile( const std::string& path, ClArgsT& cl )
{
    using Beatnik::BoundaryType;

    // Setter map. Each entry parses the value string and assigns the
    // matching cl.* field, throwing with key context on a bad value.
    std::unordered_map<std::string, std::function<void(const std::string&)>> setters;

    // --- mesh / domain ---
    setters["nodes"] = [&]( const std::string& v ) {
        const int n = parseIntValue( "nodes", v );
        if ( n < 1 )
            throw std::runtime_error( "'nodes' must be >= 1, got " + v );
        cl.num_nodes = { n, n };
    };
    setters["bounding_box"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "bounding_box", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'bounding_box' must be > 0, got " + v );
        cl.bounding_box = x;
    };
    setters["weak_scale"] = [&]( const std::string& v ) {
        const int x = parseIntValue( "weak_scale", v );
        if ( x < 1 )
            throw std::runtime_error( "'weak_scale' must be >= 1, got " + v );
        cl.weak_scale = x;
    };

    // --- time integration ---
    setters["timesteps"] = [&]( const std::string& v ) {
        // 0 (or anything <= 0) means "auto: 2*tau" — preserve old semantics.
        cl.t_final = parseDoubleValue( "timesteps", v );
    };
    setters["delta_t"] = [&]( const std::string& v ) {
        cl.delta_t = parseDoubleValue( "delta_t", v );
    };
    setters["write_frequency"] = [&]( const std::string& v ) {
        const int x = parseIntValue( "write_frequency", v );
        if ( x < 0 )
            throw std::runtime_error( "'write_frequency' must be >= 0, got " + v );
        cl.write_freq = x;
    };

    // --- initial condition ---
    setters["initial_condition"] = [&]( const std::string& v ) {
        cl.initial_condition = parseEnumValue<InitialConditionModel>(
            "initial_condition", v,
            { { "cos",      IC_COS },
              { "sech2",    IC_SECH2 },
              { "gaussian", IC_GAUSSIAN },
              { "random",   IC_RANDOM } } );
    };
    setters["magnitude"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "magnitude", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'magnitude' must be > 0, got " + v );
        cl.magnitude = x;
    };
    setters["variation"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "variation", v );
        if ( x < 0.0 )
            throw std::runtime_error( "'variation' must be >= 0, got " + v );
        cl.variation = x;
    };
    setters["period"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "period", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'period' must be > 0, got " + v );
        cl.params.period = x;
    };
    setters["tilt"] = [&]( const std::string& v ) {
        cl.tilt = parseDoubleValue( "tilt", v );
    };

    // --- physics / boundary ---
    setters["boundary"] = [&]( const std::string& v ) {
        cl.boundary = parseEnumValue<BoundaryType>(
            "boundary", v,
            { { "periodic", BoundaryType::PERIODIC },
              { "free",     BoundaryType::FREE     } } );
    };
    setters["gravity"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "gravity", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'gravity' must be > 0, got " + v );
        cl.gravity = x;
    };
    setters["atwood"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "atwood", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'atwood' must be > 0, got " + v );
        cl.atwood = x;
    };

    // --- solver ---
    setters["solver_order"] = [&]( const std::string& v ) {
        cl.params.solver_order = parseEnumValue<int>(
            "solver_order", v,
            { { "low",    SolverOrder::ORDER_LOW    },
              { "medium", SolverOrder::ORDER_MEDIUM },
              { "high",   SolverOrder::ORDER_HIGH   } } );
    };
    setters["br_solver"] = [&]( const std::string& v ) {
        // Validate against the full set first, then check Canopy support
        // separately so the error message about FMM is specific.
        cl.params.br_solver = parseEnumValue<BRSolverType>(
            "br_solver", v,
            { { "exact",  BRSolverType::BR_EXACT  },
              { "cutoff", BRSolverType::BR_CUTOFF },
              { "fmm",    BRSolverType::BR_FMM    } } );
#ifndef BEATNIK_ENABLE_CANOPY
        if ( cl.params.br_solver == BRSolverType::BR_FMM )
            throw std::runtime_error(
                "br_solver = fmm requires Beatnik to be built with Canopy "
                "support (Beatnik_ENABLE_CANOPY=ON)" );
#endif
    };
    setters["cutoff_distance"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "cutoff_distance", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'cutoff_distance' must be > 0, got " + v );
        cl.params.cutoff_distance = x;
    };
    setters["heffte_configuration"] = [&]( const std::string& v ) {
        const int x = parseIntValue( "heffte_configuration", v );
        if ( x < 0 || x > 7 )
            throw std::runtime_error(
                "'heffte_configuration' must be in [0,7], got " + v );
        cl.params.heffte_configuration = x;
    };
    setters["mu"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "mu", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'mu' must be > 0, got " + v );
        cl.mu = x;
    };
    setters["epsilon"] = [&]( const std::string& v ) {
        const double x = parseDoubleValue( "epsilon", v );
        if ( x <= 0.0 )
            throw std::runtime_error( "'epsilon' must be > 0, got " + v );
        cl.eps = x;
    };

    // --- FMM tunables (only consulted when br_solver = fmm) ---
    setters["fmm_ncrit"] = [&]( const std::string& v ) {
        cl.params.fmm_ncrit = parseIntValue( "fmm_ncrit", v );
    };
    setters["fmm_max_depth"] = [&]( const std::string& v ) {
        cl.params.fmm_max_depth = parseIntValue( "fmm_max_depth", v );
    };
    setters["fmm_mac_theta"] = [&]( const std::string& v ) {
        cl.params.fmm_mac_theta = parseDoubleValue( "fmm_mac_theta", v );
    };
    setters["fmm_replication_depth"] = [&]( const std::string& v ) {
        cl.params.fmm_replication_depth =
            parseIntValue( "fmm_replication_depth", v );
    };
    setters["fmm_imbalance_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_imbalance_tol =
            parseDoubleValue( "fmm_imbalance_tol", v );
    };
    setters["fmm_ncrit_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_ncrit_tol = parseDoubleValue( "fmm_ncrit_tol", v );
    };
    setters["fmm_xmin_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_xmin_tol = parseDoubleValue( "fmm_xmin_tol", v );
    };
    setters["fmm_xmax_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_xmax_tol = parseDoubleValue( "fmm_xmax_tol", v );
    };
    setters["fmm_ymin_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_ymin_tol = parseDoubleValue( "fmm_ymin_tol", v );
    };
    setters["fmm_ymax_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_ymax_tol = parseDoubleValue( "fmm_ymax_tol", v );
    };
    setters["fmm_zmin_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_zmin_tol = parseDoubleValue( "fmm_zmin_tol", v );
    };
    setters["fmm_zmax_tol"] = [&]( const std::string& v ) {
        cl.params.fmm_zmax_tol = parseDoubleValue( "fmm_zmax_tol", v );
    };

    // Open + parse.
    std::ifstream in( path );
    if ( !in )
        throw std::runtime_error( "cannot open input file: " + path );

    std::string line;
    int line_no = 0;
    while ( std::getline( in, line ) )
    {
        ++line_no;

        // Strip `#`-to-EOL.
        const auto hash = line.find( '#' );
        if ( hash != std::string::npos ) line.erase( hash );

        line = trim( line );
        if ( line.empty() ) continue;

        const auto eq = line.find( '=' );
        if ( eq == std::string::npos )
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) +
                ": expected 'key = value', got: " + line );

        const std::string key   = trim( line.substr( 0, eq ) );
        const std::string value = trim( line.substr( eq + 1 ) );
        if ( key.empty() )
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) +
                ": empty key before '='" );
        if ( value.empty() )
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) +
                ": empty value for key '" + key + "'" );

        const auto it = setters.find( key );
        if ( it == setters.end() )
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) +
                ": unknown key '" + key + "'" );

        try {
            it->second( value );
        } catch ( const std::exception& e ) {
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) + ": " + e.what() );
        }
    }
}

/* Help / schema dump for `rocketrig --help`. Mirrors the shipped
 * rocketrig.in so the same text is documentation and template. */
inline void printSchema( std::ostream& os )
{
    os <<
"Usage: rocketrig <input_file>\n"
"       rocketrig --help\n"
"\n"
"All run-time parameters come from an input file of `key = value`\n"
"lines. `#` starts a comment to end-of-line. Blank lines are ignored.\n"
"Missing keys keep their built-in defaults; unknown keys are an error\n"
"with the file path and line number.\n"
"\n"
"--- mesh / domain ---\n"
"  nodes                int    >=1     NxN interface mesh (default 128)\n"
"  bounding_box         double >0      half-size; domain is (-B,-B,-B)..(B,B,B) (default 1.0)\n"
"  weak_scale           int    >=1     multiplies bounding_box and nodes by sqrt(w) (default 1)\n"
"\n"
"--- time integration ---\n"
"  timesteps            double 0 -> auto: simulate 2*tau (default auto)\n"
"  delta_t              double 0 -> auto: tau/25 (low/med) or tau/50 (high) (default auto)\n"
"  write_frequency      int    >=0     steps between I/O writes (default 10)\n"
"\n"
"--- initial condition ---\n"
"  initial_condition    enum   cos | sech2 | gaussian | random (default cos)\n"
"  magnitude            double >0      IC amplitude (default 0.05)\n"
"  variation            double >=0     IC variation (default 0.0)\n"
"  period               double >0      IC period (default 1.0)\n"
"  tilt                 double         IC tilt (default 0.0)\n"
"\n"
"--- physics / boundary ---\n"
"  boundary             enum   periodic | free (default periodic)\n"
"  gravity              double >0      gravity in Gs (default 25.0)\n"
"  atwood               double >0      Atwood number (default 0.5)\n"
"\n"
"--- solver ---\n"
"  solver_order         enum   low | medium | high (default low)\n"
"  br_solver            enum   exact | cutoff | fmm (default exact)\n"
"  cutoff_distance      double >0      used by br_solver=cutoff (default 0.5)\n"
"  heffte_configuration int    [0,7]   used by low-order solver (default 6)\n"
"  mu                   double >0      artificial viscosity (default 1.0)\n"
"  epsilon              double >0      desingularization (default 0.25)\n"
"\n"
"--- FMM tunables (used only when br_solver = fmm) ---\n"
"  fmm_ncrit             int    (default 32)\n"
"  fmm_max_depth         int    (default 15)\n"
"  fmm_mac_theta         double (default 0.5)\n"
"  fmm_replication_depth int    (default 3)\n"
"  fmm_imbalance_tol     double (default 0.10)\n"
"  fmm_ncrit_tol         double (default 0.10)\n"
"  fmm_{x,y,z}{min,max}_tol  double  per-face bbox padding (default 0.10 each)\n";
}

} // namespace Example
} // namespace Beatnik

#endif // BEATNIK_EXAMPLE_INPUTFILE_HPP
