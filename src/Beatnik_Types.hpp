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
 * @file Beatnik_Types.hpp
 * @brief Scalar aliases, the not-implemented stub macro, and the enumerations
 *        that mirror the Python driver's `--choices` options.
 *
 * PORT NOTE (applies to every `Beatnik_*.hpp` in this directory)
 * -------------------------------------------------------------
 * Beatnik is a C++/Kokkos/MPI port of the Python package
 * `~/research-bridges/zmodel-steve/zmodel3d-amr`. The `// Port of <file>::<fn>
 * (lines N-M)` comments name the exact Python origin of each routine.
 *
 * The task brief named `zmodel3d/solver.py` as the port source. That file is
 * the *structured-grid* z-model and is **not** what the adaptive-mesh bubble
 * driver calls. The adaptive-mesh code path lives in:
 *
 *   - `examples/run_adaptive_mesh_bubble.py`  — CLI, control flow, checkpoints
 *   - `zmodel3d/mesh_solver.py`               — states, BR velocity, RHS, RK3,
 *                                               red-green AMR, quality repair
 *   - `zmodel3d/mesh.py`                      — surface primitives, indicators,
 *                                               sphere generators, refinement
 *   - `zmodel3d/dynamic_remesh.py`            — metric-based dynamic remeshing,
 *                                               nonlocal proximity sizing
 *   - `zmodel3d/mesh_quality.py`              — isotropic cleanup
 *
 * Traces therefore name the real origin file rather than `solver.py`. Where a
 * routine also has a recognizable structured-grid ancestor in `solver.py`, the
 * trace names both.
 */

#ifndef BEATNIK_TYPES_HPP
#define BEATNIK_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>

namespace Beatnik
{

//---------------------------------------------------------------------------//
// Scalars
//---------------------------------------------------------------------------//

/// Working precision. The Python reference is float64 throughout and the
/// regression comparison is against float64 gold files, so `double` is not a
/// tunable here: narrowing it would fail the comparison tolerances.
using Real = double;

/// Global (cross-rank) index type for vertices and faces.
using GlobalIndex = long long;

/// Rank-local index type.
using LocalIndex = int;

//---------------------------------------------------------------------------//
// Stub reporting
//---------------------------------------------------------------------------//

/**
 * @brief Throw the canonical "not implemented" error for a stubbed routine.
 *
 * Every unimplemented body in this framework calls this, so a run that reaches
 * unported code dies with a message naming exactly which routine to write next.
 *
 * @param cls    Class (or free-function scope) name, e.g. `"ZModelSolver"`.
 * @param method Method name, e.g. `"computeRightHandSide"`.
 */
[[noreturn]] inline void notImplemented( const char* cls, const char* method )
{
    throw std::logic_error( std::string( cls ) + "::" + method +
                            " not implemented" );
}

/// Convenience wrapper so a stub body is a single line.
#define BEATNIK_NOT_IMPLEMENTED( cls, method )                                 \
    ::Beatnik::notImplemented( cls, method )

//---------------------------------------------------------------------------//
// Enumerations mirroring the Python CLI choices
//
// Each enum reproduces the `choices=(...)` tuple of the matching
// `run_adaptive_mesh_bubble.py` argument, in the same order, so the CLI parser
// can map strings one-to-one and an unrecognized value is an error rather than
// a silent fallback.
//---------------------------------------------------------------------------//

/// `--state-model` (default `potential`). Which unknown is evolved: the
/// per-vertex velocity potential jump, or the per-vertex sheet vector.
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 382-386)
enum class StateModel
{
    Potential,   ///< `potential`   — evolve phi; sheet vector is derived.
    SheetVector, ///< `sheet-vector`— evolve the tangential sheet vector.
};

/// `--mesh-kind` (default `icosphere`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (line 68)
enum class MeshKind
{
    Icosphere, ///< `icosphere` — quasi-uniform, no polar singularity.
    LatLon,    ///< `latlon`    — structured lat/lon sphere with pole caps.
};

/// `--initial-shape` (default `sphere`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 72-76)
enum class InitialShape
{
    Sphere,
    Oblate,
    MushroomSeed,
    SkirtSeed,
};

/// `--initial-vorticity-mode` (default `vertical`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 109-114)
enum class InitialVorticityMode
{
    Vertical,
    RimShear,
    RimBump,
    LipShear,
    LipBump,
};

/// `--source-quadrature` (default `face`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 218-223)
///
/// All three are accepted at the CLI. **Only `Vertex` is required to be
/// implemented in the C++ port** — see README "Source quadrature".
enum class SourceQuadrature
{
    Face,      ///< One quadrature point per face at the centroid.
    Triangle3, ///< Three interior points per face (barycentric 2/3,1/6,1/6).
    Vertex,    ///< One quadrature point per vertex, weighted by vertex area.
};

/// `--velocity-mode` (default `full`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 224-229)
enum class VelocityMode
{
    Full,   ///< Move markers with the full Birkhoff-Rott velocity.
    Normal, ///< Move markers with (u.n) n only.
};

/// Birkhoff-Rott far-field approximation.
///
/// The Python offers `direct | local | clustered | treecode` (default
/// `treecode`, lines 230-237). **Beatnik replaces the approximate family with a
/// single fast-multipole path backed by Canopy**, so the C++ CLI accepts
/// `direct` and `fmm`. `local`, `clustered` and `treecode` are accepted at the
/// CLI and mapped to `Fmm` with a warning, so a Python command line still runs.
enum class BRApproximation
{
    Direct, ///< Reference O(N^2) regularized sum.
    Fmm,    ///< Canopy fast multipole (replaces the Python treecode).
};

/// `--bernoulli-scalar-mode` (default `normal-speed`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 252-257)
enum class BernoulliScalarMode
{
    NormalSpeed,  ///< Use u.n directly.
    SurfaceRiesz, ///< Use the surface Riesz scalar of the sheet.
    NormalProxy,  ///< Use 0.5 * (u.n).
};

/// `--viscosity-mode` (default `laplace-beltrami`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 188-194)
enum class ViscosityMode
{
    LaplaceBeltrami, ///< Area-normalized cotangent Laplacian (mesh-consistent).
    Graph,           ///< Uniform graph (umbrella) Laplacian.
};

/// `--kernel-blob-mode` (default `length`).
/// Port of run_adaptive_mesh_bubble.py::parse_args (lines 195-203)
///
/// Selects the desingularization denominator of the Birkhoff-Rott kernel:
/// `Length` uses \f$(\epsilon^2 + r^2)^{3/2}\f$ (eps is a physical smoothing
/// length); `Matlab` uses \f$(\epsilon + r^2)^{3/2}\f$ (eps is the raw MATLAB
/// blob parameter, i.e. already a squared length).
enum class KernelBlobMode
{
    Length,
    Matlab,
};

//---------------------------------------------------------------------------//
// Enum <-> string
//
// One table per enum, used by the example CLI parser and by any diagnostic
// that echoes the resolved configuration. Kept here rather than in the example
// so the library and the driver cannot disagree about a spelling.
//---------------------------------------------------------------------------//

inline const char* toString( StateModel v )
{
    return v == StateModel::Potential ? "potential" : "sheet-vector";
}

inline const char* toString( MeshKind v )
{
    return v == MeshKind::Icosphere ? "icosphere" : "latlon";
}

inline const char* toString( InitialShape v )
{
    switch ( v )
    {
    case InitialShape::Sphere:
        return "sphere";
    case InitialShape::Oblate:
        return "oblate";
    case InitialShape::MushroomSeed:
        return "mushroom-seed";
    case InitialShape::SkirtSeed:
        return "skirt-seed";
    }
    return "sphere";
}

inline const char* toString( InitialVorticityMode v )
{
    switch ( v )
    {
    case InitialVorticityMode::Vertical:
        return "vertical";
    case InitialVorticityMode::RimShear:
        return "rim-shear";
    case InitialVorticityMode::RimBump:
        return "rim-bump";
    case InitialVorticityMode::LipShear:
        return "lip-shear";
    case InitialVorticityMode::LipBump:
        return "lip-bump";
    }
    return "vertical";
}

inline const char* toString( SourceQuadrature v )
{
    switch ( v )
    {
    case SourceQuadrature::Face:
        return "face";
    case SourceQuadrature::Triangle3:
        return "triangle3";
    case SourceQuadrature::Vertex:
        return "vertex";
    }
    return "face";
}

inline const char* toString( VelocityMode v )
{
    return v == VelocityMode::Full ? "full" : "normal";
}

inline const char* toString( BRApproximation v )
{
    return v == BRApproximation::Direct ? "direct" : "fmm";
}

inline const char* toString( BernoulliScalarMode v )
{
    switch ( v )
    {
    case BernoulliScalarMode::NormalSpeed:
        return "normal-speed";
    case BernoulliScalarMode::SurfaceRiesz:
        return "surface-riesz";
    case BernoulliScalarMode::NormalProxy:
        return "normal-proxy";
    }
    return "normal-speed";
}

inline const char* toString( ViscosityMode v )
{
    return v == ViscosityMode::LaplaceBeltrami ? "laplace-beltrami" : "graph";
}

inline const char* toString( KernelBlobMode v )
{
    return v == KernelBlobMode::Length ? "length" : "matlab";
}

} // namespace Beatnik

#endif // BEATNIK_TYPES_HPP
