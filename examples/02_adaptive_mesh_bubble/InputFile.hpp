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
 * @file InputFile.hpp
 * @brief Command-line and `key = value` deck parsing for the
 *        `adaptive_mesh_bubble` example.
 *
 * Scoped to this example — not part of the public Beatnik library.
 *
 * RELATION TO EXAMPLE 01
 * ----------------------
 * The scalar parse helpers (`trim`, `parseIntValue`, `parseDoubleValue`,
 * `parseEnumValue`) and the setter-table discipline are carried forward
 * verbatim from `examples/01_rising_bubble/InputFile.hpp`: unknown keys are an
 * error with position context, malformed values are an error naming the key,
 * and a missing key keeps the built-in default. What is *extended* is the front
 * end. Example 01 reads only a `key = value` deck; this example must accept
 * an **argparse-style `--flag value` command line**, because
 * `examples/run_adaptive_mesh_bubble.py` is driven that way and the whole point
 * of the regression harness is that **one command line drives both the Python
 * gold run and the Beatnik run**.
 *
 * Both front ends share one setter table, so the deck and the CLI cannot drift:
 * the deck key for `--remesh-h-min` is `remesh_h_min`, mechanically (strip the
 * leading dashes, `-` to `_`).
 *
 * DEFAULTS
 * --------
 * Every default here is transcribed from
 * `run_adaptive_mesh_bubble.py::parse_args` (lines 64-533). Where a default is
 * `None` in Python (`--t-end`, `--remesh-tight-proximity-fraction`) the struct
 * carries an explicit "was it set" flag rather than a sentinel value, because
 * the Python distinguishes "unset" from "zero" for both.
 *
 * ACCEPTED-AND-IGNORED OPTIONS
 * ----------------------------
 * Plotting and video options are **accepted and ignored**, each emitting one
 * line to `stderr`. Rejecting them would break the shared-command-line
 * property; silently accepting them would let a user believe an mp4 was
 * written. The same treatment applies to `--br-cluster-count`,
 * `--br-near-radius` and `--br-near-factor`, which tune the Python's `local`
 * and `clustered` approximations that this port does not have.
 */

#ifndef BEATNIK_EXAMPLE02_INPUTFILE_HPP
#define BEATNIK_EXAMPLE02_INPUTFILE_HPP

#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_Types.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Beatnik
{
namespace Example
{

//---------------------------------------------------------------------------//
// Scalar parse helpers, carried forward from example 01.
//---------------------------------------------------------------------------//

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
        throw std::runtime_error( "invalid integer for '" + key + "': '" + v +
                                  "' (" + e.what() + ")" );
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
        throw std::runtime_error( "invalid float for '" + key + "': '" + v +
                                  "' (" + e.what() + ")" );
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

/* Parse a sign option: the Python restricts these to exactly -1.0 or +1.0
 * via `choices=(-1.0, 1.0)`, so anything else is an error rather than a
 * scale factor. */
inline double parseSignValue( const std::string& key, const std::string& v )
{
    const double x = parseDoubleValue( key, v );
    if ( x != -1.0 && x != 1.0 )
        throw std::runtime_error( "'" + key + "' must be -1 or 1, got " + v );
    return x;
}

//---------------------------------------------------------------------------//
/**
 * @brief Everything the command line can set.
 *
 * `solver` is what the library consumes; the remaining members are either
 * accepted-and-ignored presentation options or driver-level bookkeeping.
 */
struct ClArgs
{
    /// The run configuration handed to `Beatnik::Solver`.
    SolverParams solver;

    // --- resolved-later inputs ---------------------------------------------
    // These are the raw CLI values for quantities the solver resolves against
    // `initial_min_edge`, which is not known until the mesh exists. The solver
    // does the resolution in `setup()`; see
    // run_adaptive_mesh_bubble.py:1272-1286.

    // T1c CHANGE: the four proximity distance/factor fields that used to live
    // here now live in `RemeshParams`, because `Solver::setup` is what resolves
    // them against `initial_min_edge` and it is handed only a `SolverParams`.
    // The CLI option names and defaults are unchanged; see
    // `Beatnik_Params.hpp::RemeshParams::proximity_activation_factor`.

    /// `--remesh-tight-proximity`, default false. ORed with
    /// `--remesh-proximity` to set the tight set's `use_proximity`
    /// (run_adaptive_mesh_bubble.py:1376).
    bool remesh_tight_proximity = false;
    /// `--remesh-tight-proximity-fraction`. Python default is `None`, meaning
    /// "reuse `--remesh-proximity-fraction`" (lines 1377-1379).
    double remesh_tight_proximity_fraction = 0.0;
    bool have_remesh_tight_proximity_fraction = false;

    // --- accepted and ignored ----------------------------------------------

    /// `--output`, default "results/adaptive-mesh-bubble-rhs.mp4". Ignored.
    std::string output = "results/adaptive-mesh-bubble-rhs.mp4";
    /// `--no-video`, default false. Ignored — this port never renders video.
    bool no_video = false;

    /// True once a `--help` was seen; the driver prints and exits 0.
    bool help = false;

    /// Names of accepted-but-ignored options actually supplied, in order.
    /// Reported once each, on stderr, by `warnIgnored`.
    std::vector<std::string> ignored_seen;
};

//---------------------------------------------------------------------------//
/// How an option consumes its argument.
enum class OptionArity
{
    Value,     ///< `--opt V`
    Flag,      ///< `--opt` (store_true)
    Boolean,   ///< `--opt` / `--no-opt` (argparse.BooleanOptionalAction)
    Triple,    ///< `--opt X Y Z`
};

/// One row of the setter table.
struct Option
{
    OptionArity arity;
    /// Applies the (already-split) value strings. For `Flag`/`Boolean` the
    /// vector holds a single "1" or "0".
    std::function<void( const std::vector<std::string>& )> set;
    /// True for options accepted only for command-line compatibility.
    bool ignored = false;
};

using OptionTable = std::map<std::string, Option>;

//---------------------------------------------------------------------------//
/**
 * @brief Build the setter table over a `ClArgs` that already holds the
 *        defaults.
 *
 * Option names are the Python long names **without** the leading `--`. The
 * `key = value` deck front end accepts the same names with `-` replaced by `_`.
 *
 * One table, one source of truth for names, types, ranges and defaults.
 */
inline OptionTable buildOptionTable( ClArgs& cl )
{
    OptionTable t;

    // Small binders to keep the table readable.
    auto D = [&]( const char* name, double& dst ) {
        t[name] = { OptionArity::Value,
                    [&dst, name]( const std::vector<std::string>& v ) {
                        dst = parseDoubleValue( name, v[0] );
                    },
                    false };
    };
    auto I = [&]( const char* name, int& dst ) {
        t[name] = { OptionArity::Value,
                    [&dst, name]( const std::vector<std::string>& v ) {
                        dst = parseIntValue( name, v[0] );
                    },
                    false };
    };
    auto S = [&]( const char* name, std::string& dst ) {
        t[name] = { OptionArity::Value,
                    [&dst]( const std::vector<std::string>& v ) {
                        dst = v[0];
                    },
                    false };
    };
    auto F = [&]( const char* name, bool& dst ) {
        t[name] = { OptionArity::Flag,
                    [&dst]( const std::vector<std::string>& v ) {
                        dst = ( v[0] == "1" );
                    },
                    false };
    };
    auto B = [&]( const char* name, bool& dst ) {
        t[name] = { OptionArity::Boolean,
                    [&dst]( const std::vector<std::string>& v ) {
                        dst = ( v[0] == "1" );
                    },
                    false };
    };
    // Accepted-and-ignored value option.
    auto XV = [&]( const char* name ) {
        t[name] = { OptionArity::Value,
                    []( const std::vector<std::string>& ) {}, true };
    };
    // Accepted-and-ignored store_true flag.
    auto XF = [&]( const char* name ) {
        t[name] = { OptionArity::Flag,
                    []( const std::vector<std::string>& ) {}, true };
    };

    auto& ic = cl.solver.initial;
    auto& zm = cl.solver.zmodel;
    auto& fmm = cl.solver.fmm;
    auto& tp = cl.solver.time;
    auto& amr = cl.solver.amr;
    auto& rm = cl.solver.remesh;
    auto& rt = cl.solver.remesh_tight;
    auto& cu = cl.solver.cleanup;
    auto& fl = cl.solver.filter;
    auto& cp = cl.solver.checkpoint;

    //-----------------------------------------------------------------------//
    // Base mesh and initial geometry — parse_args lines 66-107
    //-----------------------------------------------------------------------//
    I( "n-theta", ic.n_theta );
    I( "n-phi", ic.n_phi );
    t["mesh-kind"] = { OptionArity::Value,
                       [&ic]( const std::vector<std::string>& v ) {
                           ic.mesh_kind = parseEnumValue<MeshKind>(
                               "mesh-kind", v[0],
                               { { "icosphere", MeshKind::Icosphere },
                                 { "latlon", MeshKind::LatLon } } );
                       },
                       false };
    I( "icosphere-subdivisions", ic.icosphere_subdivisions );
    D( "radius", ic.radius );
    D( "center-z", ic.center_z );
    t["initial-shape"] = {
        OptionArity::Value,
        [&ic]( const std::vector<std::string>& v ) {
            ic.shape = parseEnumValue<InitialShape>(
                "initial-shape", v[0],
                { { "sphere", InitialShape::Sphere },
                  { "oblate", InitialShape::Oblate },
                  { "mushroom-seed", InitialShape::MushroomSeed },
                  { "skirt-seed", InitialShape::SkirtSeed } } );
        },
        false };
    D( "horizontal-scale", ic.horizontal_scale );
    D( "vertical-scale", ic.vertical_scale );
    D( "rim-amp", ic.rim_amp );
    D( "rim-center", ic.rim_center );
    D( "rim-width", ic.rim_width );
    D( "skirt-amp", ic.skirt_amp );
    D( "skirt-center", ic.skirt_center );
    D( "skirt-width", ic.skirt_width );
    D( "skirt-neck-amp", ic.skirt_neck_amp );
    D( "skirt-neck-center", ic.skirt_neck_center );
    D( "skirt-neck-width", ic.skirt_neck_width );
    D( "skirt-drop", ic.skirt_drop );
    I( "azimuthal-mode", ic.azimuthal_mode );
    D( "azimuthal-amp", ic.azimuthal_amp );
    I( "polar-mode", ic.polar_mode );
    D( "polar-amp", ic.polar_amp );

    //-----------------------------------------------------------------------//
    // Initial vorticity — parse_args lines 108-132
    //-----------------------------------------------------------------------//
    D( "initial-potential-strength", ic.initial_potential_strength );
    t["initial-vorticity-mode"] = {
        OptionArity::Value,
        [&ic]( const std::vector<std::string>& v ) {
            ic.vorticity_mode = parseEnumValue<InitialVorticityMode>(
                "initial-vorticity-mode", v[0],
                { { "vertical", InitialVorticityMode::Vertical },
                  { "rim-shear", InitialVorticityMode::RimShear },
                  { "rim-bump", InitialVorticityMode::RimBump },
                  { "lip-shear", InitialVorticityMode::LipShear },
                  { "lip-bump", InitialVorticityMode::LipBump } } );
        },
        false };
    D( "initial-vorticity-center", ic.vorticity_center );
    D( "initial-vorticity-width", ic.vorticity_width );
    D( "initial-vorticity-radial-power", ic.vorticity_radial_power );

    //-----------------------------------------------------------------------//
    // Time stepping — parse_args lines 133-157
    //-----------------------------------------------------------------------//
    I( "steps", tp.steps );
    t["t-end"] = { OptionArity::Value,
                   [&tp]( const std::vector<std::string>& v ) {
                       tp.t_end = parseDoubleValue( "t-end", v[0] );
                       tp.have_t_end = true;
                   },
                   false };
    D( "dt", tp.dt );
    D( "dt-switch-time", tp.dt_switch_time );
    D( "dt-after-switch", tp.dt_after_switch );
    B( "adaptive-dt", tp.adaptive_dt );
    D( "min-dt", tp.min_dt );
    D( "dt-edge-power", tp.dt_edge_power );
    D( "max-sheet-dt-product", tp.max_sheet_dt_product );

    //-----------------------------------------------------------------------//
    // Physics — parse_args lines 158-217
    //-----------------------------------------------------------------------//
    D( "A", zm.A );
    D( "g", zm.g );
    D( "eps", zm.eps );
    D( "mu", zm.mu );
    D( "sigma", zm.sigma );
    D( "sigma-radius", zm.sigma_radius );
    t["sigma-center"] = { OptionArity::Triple,
                          [&zm]( const std::vector<std::string>& v ) {
                              zm.sigma_center = {
                                  parseDoubleValue( "sigma-center", v[0] ),
                                  parseDoubleValue( "sigma-center", v[1] ),
                                  parseDoubleValue( "sigma-center", v[2] ) };
                          },
                          false };
    t["viscosity-mode"] = {
        OptionArity::Value,
        [&zm]( const std::vector<std::string>& v ) {
            zm.viscosity_mode = parseEnumValue<ViscosityMode>(
                "viscosity-mode", v[0],
                { { "laplace-beltrami", ViscosityMode::LaplaceBeltrami },
                  { "graph", ViscosityMode::Graph } } );
        },
        false };
    t["kernel-blob-mode"] = {
        OptionArity::Value,
        [&zm]( const std::vector<std::string>& v ) {
            zm.blob_mode = parseEnumValue<KernelBlobMode>(
                "kernel-blob-mode", v[0],
                { { "length", KernelBlobMode::Length },
                  { "matlab", KernelBlobMode::Matlab } } );
        },
        false };
    t["forcing-sign"] = { OptionArity::Value,
                          [&zm]( const std::vector<std::string>& v ) {
                              zm.forcing_sign =
                                  parseSignValue( "forcing-sign", v[0] );
                          },
                          false };
    t["br-sign"] = { OptionArity::Value,
                     [&zm]( const std::vector<std::string>& v ) {
                         zm.br_sign = parseSignValue( "br-sign", v[0] );
                     },
                     false };

    //-----------------------------------------------------------------------//
    // Birkhoff-Rott — parse_args lines 218-257
    //-----------------------------------------------------------------------//
    t["source-quadrature"] = {
        OptionArity::Value,
        [&zm]( const std::vector<std::string>& v ) {
            zm.source_quadrature = parseEnumValue<SourceQuadrature>(
                "source-quadrature", v[0],
                { { "face", SourceQuadrature::Face },
                  { "triangle3", SourceQuadrature::Triangle3 },
                  { "vertex", SourceQuadrature::Vertex } } );
        },
        false };
    t["velocity-mode"] = { OptionArity::Value,
                           [&zm]( const std::vector<std::string>& v ) {
                               zm.velocity_mode = parseEnumValue<VelocityMode>(
                                   "velocity-mode", v[0],
                                   { { "normal", VelocityMode::Normal },
                                     { "full", VelocityMode::Full } } );
                           },
                           false };
    // The Python's four choices map onto Beatnik's two. `local`, `clustered`
    // and `treecode` are intermediate approximations this port replaces with a
    // real FMM, so they map to `fmm` and say so. `fmm` itself is a Beatnik
    // extension the Python does not accept.
    t["br-approximation"] = {
        OptionArity::Value,
        [&zm]( const std::vector<std::string>& v ) {
            const std::string& s = v[0];
            if ( s == "direct" )
            {
                zm.br_approximation = BRApproximation::Direct;
            }
            else if ( s == "fmm" )
            {
                zm.br_approximation = BRApproximation::Fmm;
            }
            else if ( s == "treecode" || s == "local" || s == "clustered" )
            {
                zm.br_approximation = BRApproximation::Fmm;
                std::cerr << "warning: --br-approximation " << s
                          << " has no Beatnik counterpart; using fmm "
                             "(Canopy fast multipole). Results will not match "
                             "a Python "
                          << s << " run to tight tolerance.\n";
            }
            else
            {
                throw std::runtime_error(
                    "invalid value for 'br-approximation': '" + s +
                    "' (expected one of: direct, fmm, local, clustered, "
                    "treecode)" );
            }
        },
        false };
    // Tunables for the Python's local/clustered paths. No Beatnik counterpart.
    XV( "br-cluster-count" );
    XV( "br-near-radius" );
    XV( "br-near-factor" );
    // Treecode tunables, mapped nominally onto the FMM. The numbers do not mean
    // the same thing to the two algorithms — see Beatnik_BRSolverFMM.hpp.
    t["br-treecode-theta"] = { OptionArity::Value,
                               [&fmm]( const std::vector<std::string>& v ) {
                                   fmm.mac_theta = parseDoubleValue(
                                       "br-treecode-theta", v[0] );
                               },
                               false };
    t["br-treecode-order"] = { OptionArity::Value,
                               [&fmm]( const std::vector<std::string>& v ) {
                                   fmm.order = parseIntValue(
                                       "br-treecode-order", v[0] );
                               },
                               false };
    t["br-treecode-ncrit"] = { OptionArity::Value,
                               [&fmm]( const std::vector<std::string>& v ) {
                                   fmm.ncrit = parseIntValue(
                                       "br-treecode-ncrit", v[0] );
                               },
                               false };
    t["bernoulli-scalar-mode"] = {
        OptionArity::Value,
        [&zm]( const std::vector<std::string>& v ) {
            zm.bernoulli_scalar_mode = parseEnumValue<BernoulliScalarMode>(
                "bernoulli-scalar-mode", v[0],
                { { "normal-speed", BernoulliScalarMode::NormalSpeed },
                  { "surface-riesz", BernoulliScalarMode::SurfaceRiesz },
                  { "normal-proxy", BernoulliScalarMode::NormalProxy } } );
        },
        false };
    // Python stores this as `no_preserve_volume`; the struct stores the
    // positive sense, so the flag inverts.
    t["no-preserve-volume"] = { OptionArity::Flag,
                                [&zm]( const std::vector<std::string>& v ) {
                                    zm.preserve_volume = ( v[0] != "1" );
                                },
                                false };

    //-----------------------------------------------------------------------//
    // Indicator-driven AMR — parse_args lines 263-302
    //-----------------------------------------------------------------------//
    D( "area-threshold", amr.area_change_threshold );
    D( "curvature-change-threshold", amr.curvature_change_threshold );
    D( "curvature-resolution-threshold", amr.curvature_resolution_threshold );
    I( "max-faces", amr.max_faces );
    D( "max-refine-fraction", amr.max_refine_fraction );
    I( "refine-neighbor-rings", amr.refine_neighbor_rings );
    t["no-balance-refinement"] = { OptionArity::Flag,
                                   [&amr]( const std::vector<std::string>& v ) {
                                       amr.balance_refinement = ( v[0] != "1" );
                                   },
                                   false };
    D( "transition-quality-floor", amr.transition_quality_floor );
    D( "transition-quality-fraction", amr.transition_quality_fraction );
    D( "min-refine-edge", amr.min_refine_edge );
    I( "refine-every", amr.refine_every );

    //-----------------------------------------------------------------------//
    // Output, checkpointing, diagnostics — parse_args lines 303-339
    //-----------------------------------------------------------------------//
    XV( "output" );   // no video is ever written, so the path is inert
    XF( "no-video" ); // ...which makes suppressing it inert too
    S( "restart-from", cp.restart_from );
    S( "checkpoint-dir", cp.directory );
    S( "checkpoint-prefix", cp.prefix );
    D( "checkpoint-every-time", cp.every_time );
    I( "checkpoint-every-steps", cp.every_steps );
    XV( "fps" );
    XV( "stride" );
    D( "progress-time-interval", cl.solver.progress_time_interval );
    F( "exact-gap-diagnostics", cl.solver.exact_gap_diagnostics );

    //-----------------------------------------------------------------------//
    // Plot-only options — parse_args lines 340-381. Accepted and ignored.
    //-----------------------------------------------------------------------//
    XV( "surface-alpha" );
    XV( "wire-width" );
    XV( "wire-alpha" );
    XV( "plot-half" );
    XV( "plot-half-origin" );
    XV( "view-elev" );
    XV( "view-azim" );
    XV( "section-axis" );
    XV( "section-origin" );
    XF( "section-panel" );

    //-----------------------------------------------------------------------//
    // State model and field filtering — parse_args lines 382-405
    //-----------------------------------------------------------------------//
    t["state-model"] = {
        OptionArity::Value,
        [&cl]( const std::vector<std::string>& v ) {
            cl.solver.state_model = parseEnumValue<StateModel>(
                "state-model", v[0],
                { { "potential", StateModel::Potential },
                  { "sheet-vector", StateModel::SheetVector } } );
        },
        false };
    I( "smooth-iters", fl.smooth_iters );
    D( "smooth-relaxation", fl.smooth_relaxation );
    I( "redistribute-every", fl.redistribute_every );
    D( "field-filter-after", fl.field_filter_after );
    I( "field-filter-every", fl.field_filter_every );
    I( "field-filter-iters", fl.field_filter_iters );
    D( "field-filter-relaxation", fl.field_filter_relaxation );
    D( "field-filter-threshold", fl.field_filter_threshold );
    I( "flip-passes", fl.flip_passes );

    //-----------------------------------------------------------------------//
    // Dynamic remeshing, baseline set — parse_args lines 406-495
    //-----------------------------------------------------------------------//
    B( "dynamic-remesh", cl.solver.dynamic_remesh );
    I( "remesh-every", cl.solver.remesh_every );
    I( "remesh-passes", rm.passes );
    D( "remesh-sagitta-tolerance", rm.sagitta_tolerance );
    D( "remesh-h-min", rm.h_min );
    D( "remesh-h-max", rm.h_max );
    D( "remesh-split-factor", rm.split_factor );
    D( "remesh-collapse-factor", rm.collapse_factor );
    D( "remesh-min-quality", rm.min_quality );
    D( "remesh-flip-min-gain", rm.flip_min_gain );
    I( "remesh-smooth-iters", rm.smoothing_iterations );
    D( "remesh-smooth-relaxation", rm.smoothing_relaxation );
    I( "remesh-max-splits", rm.max_splits_per_pass );
    I( "remesh-max-collapses", rm.max_collapses_per_pass );
    F( "remesh-proximity", rm.use_proximity );
    D( "remesh-proximity-fraction", rm.proximity_fraction );
    D( "remesh-proximity-activation-distance",
       rm.proximity_activation_distance );
    D( "remesh-proximity-activation-factor", rm.proximity_activation_factor );
    I( "remesh-proximity-exclusion-rings", rm.proximity_exclusion_rings );
    D( "remesh-proximity-material-exclusion-radius",
       rm.proximity_material_exclusion_radius );
    D( "remesh-proximity-material-exclusion-factor",
       rm.proximity_material_exclusion_factor );
    t["remesh-proximity-max-faces"] = {
        OptionArity::Value,
        [&rm]( const std::vector<std::string>& v ) {
            rm.proximity_max_faces =
                parseIntValue( "remesh-proximity-max-faces", v[0] );
        },
        false };
    F( "remesh-surgical-proximity", rm.surgical_proximity );
    D( "remesh-surgical-proximity-fraction", rm.surgical_proximity_fraction );
    D( "remesh-surgical-proximity-h-min", rm.surgical_proximity_h_min );
    D( "remesh-surgical-proximity-activation-distance",
       rm.surgical_proximity_activation_distance );
    I( "remesh-surgical-proximity-max-pairs",
       rm.surgical_proximity_max_pairs );
    I( "remesh-surgical-proximity-query-k", rm.surgical_proximity_query_k );
    D( "remesh-target-gradation-factor", rm.target_gradation_factor );
    I( "remesh-target-gradation-iters", rm.target_gradation_iterations );

    //-----------------------------------------------------------------------//
    // Isotropic cleanup — parse_args lines 502-513
    //-----------------------------------------------------------------------//
    B( "isotropic-cleanup", cu.enabled );
    I( "isotropic-cleanup-flips", cu.flip_passes );
    I( "isotropic-cleanup-relax", cu.relax_passes );
    D( "isotropic-cleanup-weight", cu.relax_weight );

    //-----------------------------------------------------------------------//
    // Dynamic remeshing, tight set — parse_args lines 496-532
    //-----------------------------------------------------------------------//
    D( "remesh-tight-after", cl.solver.remesh_tight_after );
    I( "remesh-tight-every", cl.solver.remesh_tight_every );
    I( "remesh-tight-passes", rt.passes );
    D( "remesh-tight-sagitta-tolerance", rt.sagitta_tolerance );
    D( "remesh-tight-h-min", rt.h_min );
    D( "remesh-tight-h-max", rt.h_max );
    D( "remesh-tight-split-factor", rt.split_factor );
    D( "remesh-tight-collapse-factor", rt.collapse_factor );
    D( "remesh-tight-min-quality", rt.min_quality );
    D( "remesh-tight-flip-min-gain", rt.flip_min_gain );
    I( "remesh-tight-smooth-iters", rt.smoothing_iterations );
    D( "remesh-tight-smooth-relaxation", rt.smoothing_relaxation );
    I( "remesh-tight-max-splits", rt.max_splits_per_pass );
    I( "remesh-tight-max-collapses", rt.max_collapses_per_pass );
    F( "remesh-tight-proximity", cl.remesh_tight_proximity );
    t["remesh-tight-proximity-fraction"] = {
        OptionArity::Value,
        [&cl]( const std::vector<std::string>& v ) {
            cl.remesh_tight_proximity_fraction =
                parseDoubleValue( "remesh-tight-proximity-fraction", v[0] );
            cl.have_remesh_tight_proximity_fraction = true;
        },
        false };

    return t;
}

//---------------------------------------------------------------------------//
/**
 * @brief Populate the tight-remesh defaults that the Python takes from the CLI.
 *
 * `RemeshParams` carries the *baseline* defaults, so a freshly constructed
 * `remesh_tight` has the baseline values, not the tight ones. Call this before
 * parsing so `--remesh-tight-*` overrides land on the right starting point.
 *
 * Port of run_adaptive_mesh_bubble.py::parse_args (lines 514-526)
 */
inline void applyTightRemeshDefaults( RemeshParams& rt )
{
    rt.passes = 2;
    rt.sagitta_tolerance = 0.0025;
    rt.h_min = 0.0008;
    rt.h_max = 0.055;
    rt.split_factor = 1.05;
    rt.collapse_factor = 0.0;
    rt.min_quality = 0.14;
    rt.flip_min_gain = 1.0e-3;
    rt.smoothing_iterations = 1;
    rt.smoothing_relaxation = 0.015;
    rt.max_splits_per_pass = 900;
    rt.max_collapses_per_pass = 0;
}

//---------------------------------------------------------------------------//
/**
 * @brief Reconcile the values the Python resolves after `parse_args`.
 *
 * Port of run_adaptive_mesh_bubble.py::main (lines 1272-1286, 1339-1346,
 * 1376-1385)
 *
 * Four things the Python does between parsing and constructing its parameter
 * objects, reproduced here so the solver receives a fully resolved
 * configuration:
 *
 *  1. `proximity_max_faces <= 0` means "no cap"; the Python substitutes
 *     \f$10^{12}\f$.
 *  2. The tight set inherits the baseline's proximity settings except for
 *     `use_proximity`, which is `--remesh-proximity OR
 *     --remesh-tight-proximity`, and `proximity_fraction`, which falls back to
 *     the baseline value when `--remesh-tight-proximity-fraction` is unset.
 *  3. Both sets inherit the baseline's gradation and surgical-proximity
 *     settings verbatim — there are no `--remesh-tight-` variants of those.
 *  4. The activation and material-exclusion radii are *not* resolved here: they
 *     depend on `initial_min_edge`, which is only known once the mesh exists.
 *     `Solver::setup` resolves them from the four raw values carried in
 *     `ClArgs`.
 *
 * Anything not done here is done nowhere, and shows up as a tight-remesh pass
 * that silently uses baseline proximity behavior.
 */
inline void reconcileDerivedParams( ClArgs& cl )
{
    auto& rm = cl.solver.remesh;
    auto& rt = cl.solver.remesh_tight;

    if ( rm.proximity_max_faces <= 0 )
        rm.proximity_max_faces = 1000000000000LL;

    rt.use_proximity = rm.use_proximity || cl.remesh_tight_proximity;
    rt.proximity_fraction = cl.have_remesh_tight_proximity_fraction
                                ? cl.remesh_tight_proximity_fraction
                                : rm.proximity_fraction;
    rt.proximity_exclusion_rings = rm.proximity_exclusion_rings;
    rt.proximity_max_faces = rm.proximity_max_faces;

    // The activation distance and the material-exclusion radius are resolved
    // against initial_min_edge by `Solver::setup`, which resolves BOTH
    // parameter sets from the same CLI values
    // (run_adaptive_mesh_bubble.py:1272-1286 resolves once and hands the
    // result to both). So the raw values and their factors are copied across
    // here and the resolution is not duplicated.
    rt.proximity_activation_distance = rm.proximity_activation_distance;
    rt.proximity_activation_factor = rm.proximity_activation_factor;
    rt.proximity_material_exclusion_radius =
        rm.proximity_material_exclusion_radius;
    rt.proximity_material_exclusion_factor =
        rm.proximity_material_exclusion_factor;

    rt.target_gradation_factor = rm.target_gradation_factor;
    rt.target_gradation_iterations = rm.target_gradation_iterations;

    rt.surgical_proximity = rm.surgical_proximity;
    rt.surgical_proximity_fraction = rm.surgical_proximity_fraction;
    rt.surgical_proximity_h_min = rm.surgical_proximity_h_min;
    rt.surgical_proximity_activation_distance =
        rm.surgical_proximity_activation_distance;
    rt.surgical_proximity_max_pairs = rm.surgical_proximity_max_pairs;
    rt.surgical_proximity_query_k = rm.surgical_proximity_query_k;
}

//---------------------------------------------------------------------------//
/**
 * @brief Parse an argparse-style command line into `cl`.
 *
 * Accepts `--opt value`, `--opt=value`, `--flag`, and `--no-flag` for the
 * boolean-optional options. Unknown options and missing values are errors with
 * the offending token in the message.
 *
 * `cl` must already hold the defaults; only options present on the command
 * line are overwritten. `--help` sets `cl.help` and stops parsing, so
 * `prog --help --nonsense` still prints help rather than erroring.
 *
 * @throws std::runtime_error on an unknown option, a missing value, or a bad
 *         value.
 */
inline void parseCommandLine( int argc, char* argv[], ClArgs& cl )
{
    OptionTable table = buildOptionTable( cl );

    for ( int i = 1; i < argc; ++i )
    {
        std::string tok = argv[i];

        if ( tok == "-h" || tok == "--help" )
        {
            cl.help = true;
            return;
        }

        if ( tok.rfind( "--", 0 ) != 0 )
            throw std::runtime_error( "unexpected positional argument: '" +
                                      tok + "' (all options start with --)" );

        // Split --opt=value.
        std::string inline_value;
        bool has_inline = false;
        const auto eq = tok.find( '=' );
        if ( eq != std::string::npos )
        {
            inline_value = tok.substr( eq + 1 );
            has_inline = true;
            tok = tok.substr( 0, eq );
        }

        std::string name = tok.substr( 2 );

        // A `--no-` prefix on a boolean-optional option clears it. Checked
        // before the plain lookup so an option literally named `no-...` (e.g.
        // `--no-video`, `--no-preserve-volume`) still resolves to itself.
        auto it = table.find( name );
        bool negated = false;
        if ( it == table.end() && name.rfind( "no-", 0 ) == 0 )
        {
            auto positive = table.find( name.substr( 3 ) );
            if ( positive != table.end() &&
                 positive->second.arity == OptionArity::Boolean )
            {
                it = positive;
                negated = true;
            }
        }

        if ( it == table.end() )
            throw std::runtime_error( "unknown option: '" + tok + "'" );

        std::vector<std::string> values;
        switch ( it->second.arity )
        {
        case OptionArity::Flag:
            if ( has_inline )
                throw std::runtime_error( "option '" + tok +
                                          "' takes no value" );
            values.push_back( "1" );
            break;

        case OptionArity::Boolean:
            if ( has_inline )
                throw std::runtime_error( "option '" + tok +
                                          "' takes no value" );
            values.push_back( negated ? "0" : "1" );
            break;

        case OptionArity::Value:
            if ( has_inline )
            {
                values.push_back( inline_value );
            }
            else
            {
                if ( i + 1 >= argc )
                    throw std::runtime_error( "option '" + tok +
                                              "' requires a value" );
                values.push_back( argv[++i] );
            }
            break;

        case OptionArity::Triple:
            if ( has_inline )
                throw std::runtime_error(
                    "option '" + tok +
                    "' takes three values and cannot use --opt=value form" );
            if ( i + 3 >= argc )
                throw std::runtime_error( "option '" + tok +
                                          "' requires three values (X Y Z)" );
            values.push_back( argv[++i] );
            values.push_back( argv[++i] );
            values.push_back( argv[++i] );
            break;
        }

        try
        {
            it->second.set( values );
        }
        catch ( const std::exception& e )
        {
            throw std::runtime_error( std::string( "while parsing '" ) + tok +
                                      "': " + e.what() );
        }

        if ( it->second.ignored )
            cl.ignored_seen.push_back( tok );
    }
}

//---------------------------------------------------------------------------//
/**
 * @brief Read a `key = value` deck into `cl`, using the same setter table.
 *
 * Carried forward from `examples/01_rising_bubble/InputFile.hpp`: `#` starts a
 * comment to end-of-line, blank lines are skipped, unknown keys and malformed
 * lines throw with `path:line` context, and missing keys keep their defaults.
 *
 * Deck keys are the CLI long names with `-` replaced by `_`. A flag key takes
 * `true`/`false` (or `1`/`0`); a triple takes three whitespace-separated
 * numbers.
 *
 * This front end is a convenience for reproducible runs from a committed deck.
 * The **command line** is the interface the regression harness uses, because
 * it is the one the Python shares.
 */
inline void parseInputFile( const std::string& path, ClArgs& cl )
{
    OptionTable table = buildOptionTable( cl );

    std::ifstream in( path );
    if ( !in )
        throw std::runtime_error( "cannot open input file: " + path );

    std::string line;
    int line_no = 0;
    while ( std::getline( in, line ) )
    {
        ++line_no;

        const auto hash = line.find( '#' );
        if ( hash != std::string::npos ) line.erase( hash );

        line = trim( line );
        if ( line.empty() ) continue;

        const auto eq = line.find( '=' );
        if ( eq == std::string::npos )
            throw std::runtime_error( path + ":" + std::to_string( line_no ) +
                                      ": expected 'key = value', got: " +
                                      line );

        std::string key = trim( line.substr( 0, eq ) );
        const std::string value = trim( line.substr( eq + 1 ) );
        if ( key.empty() )
            throw std::runtime_error( path + ":" + std::to_string( line_no ) +
                                      ": empty key before '='" );
        if ( value.empty() )
            throw std::runtime_error( path + ":" + std::to_string( line_no ) +
                                      ": empty value for key '" + key + "'" );

        // Deck keys use underscores; the table is keyed on the CLI spelling.
        std::string name = key;
        for ( auto& c : name )
            if ( c == '_' ) c = '-';

        const auto it = table.find( name );
        if ( it == table.end() )
            throw std::runtime_error( path + ":" + std::to_string( line_no ) +
                                      ": unknown key '" + key + "'" );

        std::vector<std::string> values;
        if ( it->second.arity == OptionArity::Triple )
        {
            std::istringstream iss( value );
            std::string v;
            while ( iss >> v ) values.push_back( v );
            if ( values.size() != 3 )
                throw std::runtime_error(
                    path + ":" + std::to_string( line_no ) + ": key '" + key +
                    "' needs three values (X Y Z)" );
        }
        else if ( it->second.arity == OptionArity::Flag ||
                  it->second.arity == OptionArity::Boolean )
        {
            if ( value == "true" || value == "1" )
                values.push_back( "1" );
            else if ( value == "false" || value == "0" )
                values.push_back( "0" );
            else
                throw std::runtime_error(
                    path + ":" + std::to_string( line_no ) + ": key '" + key +
                    "' expects true/false, got '" + value + "'" );
        }
        else
        {
            values.push_back( value );
        }

        try
        {
            it->second.set( values );
        }
        catch ( const std::exception& e )
        {
            throw std::runtime_error(
                path + ":" + std::to_string( line_no ) + ": " + e.what() );
        }

        if ( it->second.ignored )
            cl.ignored_seen.push_back( "--" + name );
    }
}

//---------------------------------------------------------------------------//
/**
 * @brief Emit one `warning:` line per accepted-and-ignored option supplied.
 *
 * Rank 0 only — the driver guards the call. Duplicates are collapsed so a deck
 * plus a command line that both set `--fps` warns once.
 */
inline void warnIgnored( const ClArgs& cl, std::ostream& os )
{
    std::vector<std::string> seen;
    for ( const auto& name : cl.ignored_seen )
    {
        bool already = false;
        for ( const auto& s : seen )
            if ( s == name ) already = true;
        if ( already ) continue;
        seen.push_back( name );
        os << "warning: " << name
           << " is accepted for CLI compatibility and ignored\n";
    }
}

//---------------------------------------------------------------------------//
/**
 * @brief Print the usage/schema text.
 *
 * Mirrors the option table above and the Python's `parse_args`, so the same
 * text serves as documentation and as a check that nothing was dropped.
 * Accepted arguments are also listed in the top-level `README.md`, which must
 * be updated in the same change as this function (CLAUDE.md general
 * guidelines).
 */
inline void printSchema( std::ostream& os )
{
    os <<
"Usage: adaptive_mesh_bubble [OPTIONS]\n"
"       adaptive_mesh_bubble --help\n"
"\n"
"C++/Kokkos/MPI port of zmodel3d-amr's run_adaptive_mesh_bubble.py. The option\n"
"names and defaults match that script, so ONE command line can drive both the\n"
"Python gold-file run and this one. Options may also be given in a `key = value`\n"
"deck (same names, `-` replaced by `_`).\n"
"\n"
"Video and plotting options are ACCEPTED AND IGNORED, with a warning to stderr.\n"
"\n"
"--- base mesh and initial geometry ---\n"
"  --n-theta INT                        latitude bands, latlon mesh (7)\n"
"  --n-phi INT                          longitude divisions, latlon mesh (14)\n"
"  --mesh-kind {icosphere,latlon}       base sphere generator (icosphere)\n"
"  --icosphere-subdivisions INT         subdivision level (2 -> 162 verts)\n"
"  --radius FLOAT                       sphere radius (0.25)\n"
"  --center-z FLOAT                     sphere centre height (0.25)\n"
"  --initial-shape {sphere,oblate,mushroom-seed,skirt-seed}   (sphere)\n"
"  --horizontal-scale FLOAT             radial stretch, non-sphere (1.28)\n"
"  --vertical-scale FLOAT               vertical stretch, non-sphere (0.68)\n"
"  --rim-amp/-center/-width FLOAT       mushroom rim bulge (0.14/0.05/0.32)\n"
"  --skirt-amp/-center/-width FLOAT     skirt bulge (0.42/-0.42/0.16)\n"
"  --skirt-neck-amp/-center/-width FLOAT  skirt neck (0.16/-0.04/0.24)\n"
"  --skirt-drop FLOAT                   lip drop, fraction of radius (0.11)\n"
"  --azimuthal-mode INT                 ripple mode m (4)\n"
"  --azimuthal-amp FLOAT                ripple amplitude (0.035)\n"
"  --polar-mode INT                     Legendre mode l for the RT seed (0)\n"
"  --polar-amp FLOAT                    RT seed amplitude (0.0)\n"
"\n"
"--- initial vorticity ---\n"
"  --initial-potential-strength FLOAT   seed amplitude; 0 = quiescent (0.0)\n"
"  --initial-vorticity-mode {vertical,rim-shear,rim-bump,lip-shear,lip-bump}\n"
"                                       profile (vertical)\n"
"  --initial-vorticity-center FLOAT     reference z/radius (-0.15)\n"
"  --initial-vorticity-width FLOAT      reference width (0.18)\n"
"  --initial-vorticity-radial-power FLOAT  lip localization power (2.0)\n"
"\n"
"--- time stepping ---\n"
"  --steps INT                          steps this invocation takes (140)\n"
"  --t-end FLOAT                        stop at this time (unset)\n"
"  --dt FLOAT                           nominal step (0.003)\n"
"  --dt-switch-time FLOAT               clamp dt past this time; <0 off (-1)\n"
"  --dt-after-switch FLOAT              the clamp value (0.001)\n"
"  --adaptive-dt / --no-adaptive-dt     throttle dt by the smallest triangle (on)\n"
"  --min-dt FLOAT                       adaptive dt floor (2.5e-4)\n"
"  --dt-edge-power FLOAT                exponent on the edge ratio (1.0)\n"
"  --max-sheet-dt-product FLOAT         also cap dt*max|S|; 0 = off (0.0)\n"
"\n"
"--- physics ---\n"
"  --A FLOAT                            Atwood number (0.3)\n"
"  --g FLOAT                            gravity (1.0)\n"
"  --eps FLOAT                          kernel desingularization (0.025)\n"
"  --mu FLOAT                           artificial viscosity (0.002)\n"
"  --sigma FLOAT                        surface tension (0.0)\n"
"  --sigma-radius FLOAT                 localize surface tension; 0 = global (0)\n"
"  --sigma-center X Y Z                 localization centre (0 0 0)\n"
"  --viscosity-mode {laplace-beltrami,graph}   (laplace-beltrami)\n"
"  --kernel-blob-mode {length,matlab}   eps^2+r^2 or eps+r^2 (length)\n"
"  --forcing-sign {-1,1}                sign on the Bernoulli forcing (1)\n"
"  --br-sign {-1,1}                     sign on the BR velocity (1)\n"
"\n"
"--- Birkhoff-Rott ---\n"
"  --source-quadrature {face,triangle3,vertex}\n"
"                                       (face) NOTE: only `vertex` is\n"
"                                       implemented in this port; see README.\n"
"  --velocity-mode {normal,full}        marker motion rule (full)\n"
"  --br-approximation {direct,fmm,local,clustered,treecode}\n"
"                                       (treecode -> fmm). Beatnik offers\n"
"                                       `direct` and `fmm`; the Python's\n"
"                                       local/clustered/treecode map to `fmm`\n"
"                                       with a warning.\n"
"  --br-treecode-theta FLOAT            -> FMM acceptance criterion (0.3)\n"
"  --br-treecode-order INT              -> FMM expansion order (2)\n"
"  --br-treecode-ncrit INT              -> FMM leaf occupancy (64)\n"
"  --br-cluster-count INT               IGNORED (Python `clustered` only)\n"
"  --br-near-radius FLOAT               IGNORED (Python local/clustered only)\n"
"  --br-near-factor FLOAT               IGNORED (Python local/clustered only)\n"
"  --bernoulli-scalar-mode {normal-speed,surface-riesz,normal-proxy}\n"
"                                       (normal-speed)\n"
"  --no-preserve-volume                 disable the volume projection\n"
"\n"
"--- indicator-driven AMR (only with --no-dynamic-remesh) ---\n"
"  --area-threshold FLOAT               (0.16)\n"
"  --curvature-change-threshold FLOAT   (0.35)\n"
"  --curvature-resolution-threshold FLOAT  sagitta, length units; 0 = off (0)\n"
"  --max-faces INT                      projected face cap (1400)\n"
"  --max-refine-fraction FLOAT          seed-mark cap (0.05)\n"
"  --refine-neighbor-rings INT          mark expansion rings (1)\n"
"  --no-balance-refinement              disable green->red promotion\n"
"  --transition-quality-floor FLOAT     (0.18)\n"
"  --transition-quality-fraction FLOAT  (0.45)\n"
"  --min-refine-edge FLOAT              refinement floor; 0 = none (0.0)\n"
"  --refine-every INT                   steps between refinements (5)\n"
"\n"
"--- dynamic remeshing (the default adaptivity path) ---\n"
"  --dynamic-remesh / --no-dynamic-remesh  (on)\n"
"  --remesh-every INT                   (1)\n"
"  --remesh-passes INT                  (1)\n"
"  --remesh-sagitta-tolerance FLOAT     (0.004)\n"
"  --remesh-h-min / --remesh-h-max FLOAT   (0.0015 / 0.05)\n"
"  --remesh-split-factor FLOAT          (1.35)\n"
"  --remesh-collapse-factor FLOAT       (0.45)\n"
"  --remesh-min-quality FLOAT           (0.18)\n"
"  --remesh-flip-min-gain FLOAT         (1e-3)\n"
"  --remesh-smooth-iters INT            (1)\n"
"  --remesh-smooth-relaxation FLOAT     (0.04)\n"
"  --remesh-max-splits INT              per pass; <=0 unlimited (300)\n"
"  --remesh-max-collapses INT           per pass; <=0 unlimited (120)\n"
"  --remesh-target-gradation-factor FLOAT  (1.35)\n"
"  --remesh-target-gradation-iters INT  (8)\n"
"\n"
"--- nonlocal proximity sizing ---\n"
"  --remesh-proximity                   enable (off)\n"
"  --remesh-proximity-fraction FLOAT    target = this * gap (0.25)\n"
"  --remesh-proximity-activation-distance FLOAT   <=0 uses the factor (0.0)\n"
"  --remesh-proximity-activation-factor FLOAT     x initial min edge (6.0)\n"
"  --remesh-proximity-exclusion-rings INT         same-surface rings (3)\n"
"  --remesh-proximity-material-exclusion-radius FLOAT  <=0 uses factor (0.0)\n"
"  --remesh-proximity-material-exclusion-factor FLOAT  x initial min edge (4.0)\n"
"  --remesh-proximity-max-faces INT     cap; <=0 uncapped (100000)\n"
"  --remesh-surgical-proximity          exact close-pair splits (off)\n"
"  --remesh-surgical-proximity-fraction FLOAT     (0.35)\n"
"  --remesh-surgical-proximity-h-min FLOAT        <=0 uses --remesh-h-min (0)\n"
"  --remesh-surgical-proximity-activation-distance FLOAT  <=0 reuses above (0)\n"
"  --remesh-surgical-proximity-max-pairs INT      (64)\n"
"  --remesh-surgical-proximity-query-k INT        (48)\n"
"\n"
"--- tight remeshing, active past --remesh-tight-after ---\n"
"  --remesh-tight-after FLOAT           <0 disables (-1.0)\n"
"  --remesh-tight-every INT             (1)\n"
"  --remesh-tight-passes INT            (2)\n"
"  --remesh-tight-sagitta-tolerance FLOAT  (0.0025)\n"
"  --remesh-tight-h-min / -h-max FLOAT  (0.0008 / 0.055)\n"
"  --remesh-tight-split-factor FLOAT    (1.05)\n"
"  --remesh-tight-collapse-factor FLOAT (0.0 -- collapse disabled)\n"
"  --remesh-tight-min-quality FLOAT     (0.14)\n"
"  --remesh-tight-flip-min-gain FLOAT   (1e-3)\n"
"  --remesh-tight-smooth-iters INT      (1)\n"
"  --remesh-tight-smooth-relaxation FLOAT  (0.015)\n"
"  --remesh-tight-max-splits INT        (900)\n"
"  --remesh-tight-max-collapses INT     (0)\n"
"  --remesh-tight-proximity             (off)\n"
"  --remesh-tight-proximity-fraction FLOAT  unset -> --remesh-proximity-fraction\n"
"\n"
"--- isotropic cleanup (after each remesh) ---\n"
"  --isotropic-cleanup / --no-isotropic-cleanup  (on)\n"
"  --isotropic-cleanup-flips INT        valence flip passes (3)\n"
"  --isotropic-cleanup-relax INT        relaxation passes (2)\n"
"  --isotropic-cleanup-weight FLOAT     relaxation weight (0.4)\n"
"\n"
"--- state model and field filtering ---\n"
"  --state-model {potential,sheet-vector}   (potential)\n"
"  --smooth-iters INT                   post-refine relaxation sweeps (1)\n"
"  --smooth-relaxation FLOAT            (0.12)\n"
"  --redistribute-every INT             0 = off (0)\n"
"  --field-filter-after FLOAT           <0 = always (-1.0)\n"
"  --field-filter-every INT             0 = off (0)\n"
"  --field-filter-iters INT             (1)\n"
"  --field-filter-relaxation FLOAT      (0.01)\n"
"  --field-filter-threshold FLOAT       on max|S|; 0 = always (0.0)\n"
"  --flip-passes INT                    post-refine quality flips (0)\n"
"\n"
"--- checkpointing and restart ---\n"
"  --checkpoint-dir PATH                empty disables checkpointing (\"\")\n"
"  --checkpoint-prefix STR              (checkpoint)\n"
"  --checkpoint-every-time FLOAT        0 = off (0.0)\n"
"  --checkpoint-every-steps INT         0 = off (0)\n"
"  --restart-from PATH                  restart from a checkpoint (\"\")\n"
"\n"
"--- diagnostics ---\n"
"  --progress-time-interval FLOAT       sim time between progress lines (0.25)\n"
"  --exact-gap-diagnostics              exact triangle-triangle gaps (off)\n"
"\n"
"--- accepted and IGNORED (video / plotting) ---\n"
"  --output PATH, --no-video, --fps INT, --stride INT,\n"
"  --surface-alpha, --wire-width, --wire-alpha,\n"
"  --plot-half, --plot-half-origin, --view-elev, --view-azim,\n"
"  --section-axis, --section-origin, --section-panel\n";
}

} // namespace Example
} // namespace Beatnik

#endif // BEATNIK_EXAMPLE02_INPUTFILE_HPP
