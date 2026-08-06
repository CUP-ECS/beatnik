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
 * @file Beatnik_Diagnostics.hpp
 * @brief Per-step scalar diagnostics printed by the driver.
 *
 * Port of run_adaptive_mesh_bubble.py::frame_diagnostics (lines 1113-1192)
 *
 * The Python computes these for three purposes: the progress line, the
 * per-frame history behind the video's diagnostics panel, and the frame list.
 * **Only the progress line is ported** — the plotting is out of scope, and
 * none of these quantities is written to a checkpoint, so none of them is
 * compared by the regression harness. They exist to make a run's behavior
 * legible while it is running.
 *
 * COST WARNING
 * ------------
 * `frame_diagnostics` is not cheap. It recomputes both AMR indicators, the
 * sagitta indicator, the triangle qualities, all edge lengths, and — most
 * expensively — the nonlocal centroid-proximity query, which is a global
 * spatial search (`DynamicRemesh::nonlocalFaceCentroidDistance`). With
 * `--exact-gap-diagnostics` it additionally runs the exact
 * triangle-triangle pair search.
 *
 * The reference calls it on a *throttle*: at step 1, every `steps/10` steps,
 * and whenever `--progress-time-interval` of simulation time has elapsed
 * (`run_adaptive_mesh_bubble.py:1604-1608`). Calling it every step is a real
 * slowdown, not a rounding error, and on a distributed run it adds a global
 * spatial query per step. Keep the throttle.
 */

#ifndef BEATNIK_DIAGNOSTICS_HPP
#define BEATNIK_DIAGNOSTICS_HPP

#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <iosfwd>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief One row of run diagnostics.
 *
 * Port of run_adaptive_mesh_bubble.py::frame_diagnostics return dict
 * (lines 1168-1192)
 *
 * The plot-only history entries the Python accumulates are not carried; each
 * of these is a scalar for one step.
 */
struct Diagnostics
{
    /// Global vertex count.
    GlobalIndex vertices = 0;
    /// Global face count.
    GlobalIndex faces = 0;

    /// Area-weighted mean height,
    /// \f$\sum_v A_v z_v / \sum_v A_v\f$. The bubble's rise is read off this
    /// rather than off the centroid of the vertices, which would be biased by
    /// the adaptive vertex distribution. Units of length.
    Real centroid_z = 0.0;
    /// Highest vertex.
    Real zmax = 0.0;
    /// Lowest vertex.
    Real zmin = 0.0;

    /// Current enclosed volume, units length^3.
    Real volume = 0.0;
    /// Relative volume drift \f$V/V_0 - 1\f$. The single most useful health
    /// indicator: with the volume projection on it should stay at round-off,
    /// and a drift means the projection is not being applied where it should
    /// be.
    Real volume_rel = 0.0;

    /// \f$\|S\|_\infty\f$ over both vertex and face sheet vectors.
    Real max_sheet = 0.0;

    /// Largest area-change indicator.
    Real max_area_change = 0.0;
    /// Largest curvature-change indicator.
    Real max_curvature_change = 0.0;
    /// Largest sagitta / curvature-resolution indicator, units of length.
    Real max_curvature_resolution = 0.0;
    /// Largest raw face curvature, units 1/length.
    Real max_curvature = 0.0;

    /// Worst and mean triangle quality, both in [0,1].
    Real min_quality = 0.0;
    Real mean_quality = 0.0;

    /// Shortest edge anywhere, units of length.
    Real min_edge = 0.0;

    /// Smallest nonlocal centroid-to-centroid gap, or +inf if every face is
    /// excluded. Units of length.
    Real min_nonlocal_distance = 0.0;
    /// Same, by exact triangle-triangle distance. NaN unless
    /// `--exact-gap-diagnostics`.
    Real exact_min_nonlocal_distance = 0.0;

    /// \f$d_{\text{gap}} / h_{\min}\f$. **The self-contact indicator**: the
    /// simulation is resolving an approach as long as this stays comfortably
    /// above 1, and the sheets are touching at the mesh scale when it reaches
    /// it. Watching this rather than the raw gap is what distinguishes "the
    /// sheets are close" from "the mesh can no longer tell them apart".
    Real gap_edge_ratio = 0.0;
    /// Same ratio using the exact gap.
    Real exact_gap_edge_ratio = 0.0;
};

//---------------------------------------------------------------------------//
/**
 * @brief Computes and formats the run diagnostics.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class DiagnosticsCalculator
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;

    /**
     * @brief Compute one diagnostics row.
     *
     * Port of run_adaptive_mesh_bubble.py::frame_diagnostics (lines 1113-1192)
     *
     * @param mesh           Current surface.
     * @param state          Current solution.
     * @param initial_volume \f$V_0\f$, for the relative drift. A non-positive
     *                       value reports zero drift rather than dividing.
     * @param remesh_params  Supplies the proximity exclusion rings and the
     *                       material exclusion radius, so the reported gap
     *                       means the same thing the remesher acts on.
     * @param exact_gap      Also run the exact triangle-triangle search.
     *
     * @note MPI. Collective throughout — every field is a global reduction, and
     *       the two gap fields are global spatial queries. Every rank must call
     *       it, on the same steps, or the throttle itself deadlocks the run.
     */
    static Diagnostics compute( const mesh_type& mesh, const state_type& state,
                                Real initial_volume,
                                const RemeshParams& remesh_params,
                                bool exact_gap )
    {
        (void)mesh;
        (void)state;
        (void)initial_volume;
        (void)remesh_params;
        (void)exact_gap;
        BEATNIK_NOT_IMPLEMENTED( "DiagnosticsCalculator", "compute" );
    }

    /**
     * @brief Write the one-line progress record.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1619-1636)
     *
     * The reference's field order and formatting are reproduced so a Beatnik
     * log can be diffed against a Python log of the same configuration. Rank 0
     * only.
     */
    static void writeProgressLine( std::ostream& os, const Diagnostics& diag,
                                   long long step, Real time, Real dt,
                                   long long refine_events, long long flips,
                                   long long remesh_events )
    {
        (void)os;
        (void)diag;
        (void)step;
        (void)time;
        (void)dt;
        (void)refine_events;
        (void)flips;
        (void)remesh_events;
        BEATNIK_NOT_IMPLEMENTED( "DiagnosticsCalculator", "writeProgressLine" );
    }
};

} // namespace Beatnik

#endif // BEATNIK_DIAGNOSTICS_HPP
