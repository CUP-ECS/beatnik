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

#include <Beatnik_Communication.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <limits>
#include <ostream>
#include <utility>

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
     *
     * **T2d — SIX FIELDS ARE REPORTED AS UNAVAILABLE, AND THAT IS DELIBERATE.**
     * Nothing here is written to a checkpoint and nothing here is compared by
     * the regression harness (see the file header), so an unavailable field is a
     * legibility gap and not a correctness one. What is left unavailable, and
     * why, so a reader does not mistake it for a bug:
     *
     * | Field(s) | Needs | Reported as |
     * | --- | --- | --- |
     * | `min_nonlocal_distance`, `gap_edge_ratio` | the global centroid-proximity query (T3-era / T4b) | `+inf` / `+inf` |
     * | `exact_min_nonlocal_distance`, `exact_gap_edge_ratio` | the exact triangle-pair search (T4b) | NaN |
     * | `max_area_change`, `max_curvature_change`, `max_curvature_resolution`, `max_curvature` | the four AMR indicators (T4a) | NaN |
     *
     * `+inf` and NaN are exactly what the Python prints when *it* has nothing —
     * `min_nonlocal_distance` is `np.inf` when every face is excluded, and
     * `exact_min_nonlocal_distance` is `np.nan` without
     * `--exact-gap-diagnostics` — so a Beatnik progress line stays diffable
     * against a Python one, and no field silently reports a zero that reads as a
     * measurement.
     *
     * **T2d — `mesh` is no longer `const`** (`positions()`, `faceVertices()`,
     * `edgeVertices()`, `sheetVector()`).
     */
    static Diagnostics compute( mesh_type& mesh, const state_type& state,
                                Real initial_volume,
                                const RemeshParams& remesh_params,
                                bool exact_gap )
    {
        (void)remesh_params;
        (void)exact_gap;

        using exec = ExecutionSpace;
        using range = Kokkos::RangePolicy<exec>;
        const Real inf = std::numeric_limits<Real>::infinity();
        const Real nan = std::numeric_limits<Real>::quiet_NaN();

        Diagnostics diag;
        diag.vertices = mesh.globalVertexCount();
        diag.faces = mesh.globalFaceCount();

        const int n_owned = mesh.ownedVertexCount();
        const int n_local = mesh.totalVertexCount();
        const int nf_owned = mesh.ownedFaceCount();

        MeshGeometry<ExecutionSpace, MemorySpace> geometry;
        geometry.compute( mesh.positions(), n_local, mesh.faceVertices() );

        auto pos = mesh.positions();
        auto faces = mesh.faceVertices();

        // The area-weighted mean height, and the two extremes. OWNED vertices
        // (risk R9); the mean is reduce-both-then-divide, as everywhere else.
        Real weighted_z = 0, area_sum = 0, zmax = -inf, zmin = inf;
        {
            auto va = geometry.vertex_area;
            Kokkos::parallel_reduce(
                "beatnik_diag_vertices", range( 0, n_owned ),
                KOKKOS_LAMBDA( const int i, Real& wz, Real& a, Real& hi,
                               Real& lo ) {
                    wz += va( i ) * pos( i, 2 );
                    a += va( i );
                    if ( pos( i, 2 ) > hi )
                        hi = pos( i, 2 );
                    if ( pos( i, 2 ) < lo )
                        lo = pos( i, 2 );
                },
                weighted_z, area_sum, Kokkos::Max<Real>( zmax ),
                Kokkos::Min<Real>( zmin ) );
        }

        // The volume and the triangle-quality statistics, over OWNED faces.
        auto owned_faces = Kokkos::subview(
            faces, std::make_pair( 0, nf_owned ), Kokkos::ALL() );
        const Real local_volume =
            SurfaceOperators::enclosedVolume( pos, owned_faces );

        Real quality_min = inf, quality_sum = 0;
        {
            typename MeshGeometry<ExecutionSpace, MemorySpace>::scalar_view
                quality( Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                             "beatnik_diag_quality" ),
                         nf_owned );
            SurfaceOperators::triangleQuality( pos, owned_faces, quality );
            Kokkos::parallel_reduce(
                "beatnik_diag_quality_reduce", range( 0, nf_owned ),
                KOKKOS_LAMBDA( const int f, Real& lo, Real& s ) {
                    if ( quality( f ) < lo )
                        lo = quality( f );
                    s += quality( f );
                },
                Kokkos::Min<Real>( quality_min ), quality_sum );
        }

        // The shortest edge, over OWNED edges -- the same quantity, computed the
        // same way, as the adaptive dt's h_min.
        Real edge_min = inf;
        {
            const int ne_owned = mesh.ownedEdgeCount();
            auto owned_edges =
                Kokkos::subview( mesh.edgeVertices(),
                                 std::make_pair( 0, ne_owned ), Kokkos::ALL() );
            typename MeshGeometry<ExecutionSpace, MemorySpace>::scalar_view
                lengths( Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                             "beatnik_diag_edges" ),
                         ne_owned );
            SurfaceOperators::edgeLengths( pos, owned_edges, lengths );
            Kokkos::parallel_reduce(
                "beatnik_diag_edge_reduce", range( 0, ne_owned ),
                KOKKOS_LAMBDA( const int e, Real& lo ) {
                    if ( lengths( e ) < lo )
                        lo = lengths( e );
                },
                Kokkos::Min<Real>( edge_min ) );
        }

        // Three collectives, batched by operation rather than by field: the
        // area-weighted mean needs both of its sums before the division, and
        // batching is what guarantees that.
        Real local_sum[4] = { weighted_z, area_sum, quality_sum, local_volume };
        Real sum[4] = { 0, 0, 0, 0 };
        MPI_Allreduce( local_sum, sum, 4, MPI_DOUBLE, MPI_SUM, mesh.comm() );

        const Real global_zmax = Comm::allReduceMax( mesh.comm(), zmax );
        Real local_min[3] = { zmin, quality_min, edge_min };
        Real global_min[3] = { 0, 0, 0 };
        MPI_Allreduce( local_min, global_min, 3, MPI_DOUBLE, MPI_MIN,
                       mesh.comm() );

        diag.centroid_z = ( sum[1] > Real( 0 ) ) ? sum[0] / sum[1] : Real( 0 );
        diag.zmax = global_zmax;
        diag.zmin = global_min[0];
        diag.volume = sum[3];
        diag.volume_rel =
            ( initial_volume > Real( 0 ) || initial_volume < Real( 0 ) )
                ? ( diag.volume / initial_volume - Real( 1 ) )
                : Real( 0 );
        diag.min_quality = global_min[1];
        diag.mean_quality = ( diag.faces > 0 )
                                ? sum[2] / static_cast<Real>( diag.faces )
                                : Real( 0 );
        diag.min_edge = global_min[2];

        // Collective, and it refreshes the sheet vector the way the Python's
        // `state.sheet_vector` property does.
        state.updateSheetVector( mesh, geometry );
        diag.max_sheet = state.maxSheetStrength( mesh, geometry );

        // The six fields nothing here can supply yet. See the table above.
        diag.max_area_change = nan;
        diag.max_curvature_change = nan;
        diag.max_curvature_resolution = nan;
        diag.max_curvature = nan;
        diag.min_nonlocal_distance = inf;
        diag.exact_min_nonlocal_distance = nan;
        diag.gap_edge_ratio = inf;
        diag.exact_gap_edge_ratio = nan;

        return diag;
    }

    /**
     * @brief Write the one-line progress record.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1619-1636)
     *
     * The reference's field order and formatting are reproduced so a Beatnik
     * log can be diffed against a Python log of the same configuration. Rank 0
     * only.
     *
     * **T2d — formatted with `snprintf`, not with iostream manipulators.** The
     * Python's f-string specifiers (`:.4f`, `:.2e`, `:+.5e`, `:.3g`, `:5d`) are
     * `printf` specifiers verbatim, and the point of this function is that the
     * two logs diff — so the format string is the reference's, transcribed, and
     * `os` receives the finished line. Reproducing `:.3g` with
     * `setprecision`/`defaultfloat` would be a re-derivation of the same
     * behaviour with its own rounding edge cases.
     *
     * `min(area)` is the one field the reference reads off `geom` rather than
     * off the diagnostics dict (`run_adaptive_mesh_bubble.py:1624`), and it is
     * not carried in `Diagnostics`, so it is **omitted** rather than
     * approximated. Every other field is present, with the unavailable ones
     * printing `inf` / `nan` — which is what the reference's own formatter emits
     * for those values, so the columns still line up.
     */
    static void writeProgressLine( std::ostream& os, const Diagnostics& diag,
                                   long long step, Real time, Real dt,
                                   long long refine_events, long long flips,
                                   long long remesh_events )
    {
        char line[1024];
        std::snprintf(
            line, sizeof( line ),
            "step=%5lld t=%.4f dt=%.2e V=%lld F=%lld centroid_z=%+.5e "
            "zmax=%+.5e volRel=%+.2e max|S|=%.5e minQ=%.3g minEdge=%.3g "
            "minGap=%.3g gapEdge=%.3g exactGap=%.3g exactGapEdge=%.3g "
            "max_dA=%.3g max_dk=%.3g max_sag=%.3g refine_events=%lld "
            "flips=%lld remesh_events=%lld",
            step, static_cast<double>( time ), static_cast<double>( dt ),
            static_cast<long long>( diag.vertices ),
            static_cast<long long>( diag.faces ),
            static_cast<double>( diag.centroid_z ),
            static_cast<double>( diag.zmax ),
            static_cast<double>( diag.volume_rel ),
            static_cast<double>( diag.max_sheet ),
            static_cast<double>( diag.min_quality ),
            static_cast<double>( diag.min_edge ),
            static_cast<double>( diag.min_nonlocal_distance ),
            static_cast<double>( diag.gap_edge_ratio ),
            static_cast<double>( diag.exact_min_nonlocal_distance ),
            static_cast<double>( diag.exact_gap_edge_ratio ),
            static_cast<double>( diag.max_area_change ),
            static_cast<double>( diag.max_curvature_change ),
            static_cast<double>( diag.max_curvature_resolution ),
            refine_events, flips, remesh_events );
        os << line << std::endl;
    }
};

} // namespace Beatnik

#endif // BEATNIK_DIAGNOSTICS_HPP
