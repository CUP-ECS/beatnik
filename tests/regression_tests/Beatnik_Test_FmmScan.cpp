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
 * @file Beatnik_Test_FmmScan.cpp
 * @brief **T5's MEASUREMENT DRIVER** (`tasks/canopy/add-canopy.md`) — the
 *        far-field fidelity scan over `order`, `ncrit`, `mac_theta`,
 *        `max_depth` and `basis`, on one real milestone-0 state.
 *
 * THIS IS NOT A TEST AND IS IN NO TIER.
 * ------------------------------------
 * It carries no `LABELS`, no ctest case and no manifest line, so neither
 * `run_regression_minset.flux` (the ship gate) nor `run_milestone.flux` can
 * pick it up — see the "Measurement drivers — IN NO TIER" loop in
 * `tests/CMakeLists.txt`, which is the milestone tier's loop stopped short of
 * the point where it applies a label. **No tolerance is compiled into this
 * file and none may be**: T5 produces the numbers that T6 compiles, and a
 * measurement that carried its own pass criterion would be asserting the thing
 * it exists to measure (risk **R1**). The only assertions here are structural
 * — entity counts, finiteness, the global source count, the spin-up reaching
 * its step budget — i.e. the conditions under which the printed numbers mean
 * anything at all.
 *
 * WHY ONE LAUNCH AND ONE STATE
 * ----------------------------
 * The spin-up is downstream of five timesteps and **the timestep is not
 * bitwise reproducible**: T3 measured 1.15e-15 (np1) / 1.88e-15 (np4) of field
 * RMS run-to-run on an identical binary, and T4 measured the consequence
 * downstream — the same arm at np=1 gave 5.00765e-4 on one job and 5.00819e-4
 * on another, a floor of about 1e-4 **of the error itself**. This scan resolves
 * differences finer than that: between adjacent `max_depth` values, and between
 * neighbouring `mac_theta`. So the state is spun up **once** and every arm is
 * evaluated against that one state in one process, which makes arm-to-arm
 * differences exact. Any figure that crosses launches — the rank sweep — carries
 * that floor and the log states it beside the number.
 *
 * THE TRAP: A SCAN THAT MEASURES TWO DIRECT SUMS
 * ----------------------------------------------
 * At \f$\theta\f$ Canopy's near field reaches \f$\sqrt3/\theta\f$ cell widths.
 * Beatnik's sources are a 2-manifold, so the occupied leaves form a
 * two-dimensional grid and that neighbourhood covers
 * \f$\pi(\sqrt3/\theta)^2\approx105\f$ of them at \f$\theta=0.3\f$. A surface
 * of \f$N\f$ vertices at leaf occupancy `ncrit` occupies
 * \f$N/\texttt{ncrit}\f$ leaves, so a far field that carries any of the field
 * needs \f$N \gg \pi(\sqrt3/\theta)^2\,\texttt{ncrit}\f$ — \f$N\gg6720\f$ at
 * the default `ncrit = 64`, which neither milestone-0 level meets. Below the
 * bound the solve is a direct sum with FMM bookkeeping around it: it agrees
 * with `BRSolverDirect` to round-off at **every** order, silently, and reads as
 * success (**R1**'s cheapest misreading). `ncrit` is therefore a scan **axis**
 * and not a fixed background, and **every point prints its realized P2P pair
 * fraction** — a point without one is not a point.
 *
 * THE POTENTIAL COLUMN, AND WHY IT IS HERE
 * ----------------------------------------
 * A Cartesian-Taylor truncation at order \f$p\f$ leaves
 * \f$\sim(cw/R)^{p+1}\f$ on the potential and \f$\sim(cw/R)^{p}\f$ on the
 * gradient, because the local expansion is differentiated to get the gradient.
 * Beatnik reads **only the gradient** (the velocity is a curl of Canopy's
 * gradient tensor), but every external figure this scan has to be read against
 * is on a potential: Canopy's own convergence table, and the reference
 * treecode's velocity, which has no target-side expansion and so carries the
 * potential's truncation order. So both columns are printed at every point,
 * through `FarFieldSolver::evaluatePotential` — a measurement surface T5 added
 * to the adapter for exactly this, applying no prefactor.
 *
 * THE TWO REFERENCES
 * ------------------
 *   * **Gradient (velocity).** `BRSolverDirect` on the same mesh, geometry,
 *     state and quadrature — T4's reference, unchanged.
 *   * **Potential.** An \f$O(N^2)\f$ sum written here, over the globally
 *     gathered source set. It must follow **Canopy's** exclusion rule rather
 *     than the Birkhoff-Rott one: `Canopy_P2P.hpp` skips \f$p_j=p_i\f$ and any
 *     pair with \f$|r|^2<10^{-24}\f$, so a reference that includes the self
 *     term differs from the FMM by \f$S_t/\sqrt b\f$ — a factor of 40 at
 *     Beatnik's softening — at every target, which would swamp the truncation
 *     error entirely. The softening squared is taken as
 *     `diagnostics().softening`\f$^2\f$, the value Canopy itself squared, not
 *     re-derived from `blob()`.
 *
 * ARGUMENTS. Two positionals; the second is optional. There is no option
 * surface here and none may be added — a driver's arguments come from the batch
 * script that measures with it (`tests/CMakeLists.txt`, the driver loop).
 *
 *   argv[1]  --icosphere-subdivisions   the level to scan (3 or 4)
 *   argv[2]  spin-up steps              default 5; `--br-approximation direct`
 *                                       steps before anything is compared,
 *                                       because `--initial-potential-strength
 *                                       0` makes the sheet strength identically
 *                                       zero at step 0 and a step-0 comparison
 *                                       is vacuous at every order
 *
 * It writes no checkpoints and touches no filesystem, so `BEATNIK_TEST_SCRATCH`
 * is deliberately not read — `makeScanParams` leaves the checkpoint directory
 * empty and `CheckpointIO` is a no-op without one.
 *
 * OUTPUT. One `[t5arm]` line per arm, `key=value` throughout, so the batch log
 * reduces to a table without parsing prose:
 *
 *     [t5arm] axis=<name> basis=<b> order=<p> ncrit=<n> theta=<t>
 *             max_depth=<d> softening=<s> near_soft_factor=0
 *             grad_abs=<e> grad_rel=<e> pot_abs=<e> pot_rel=<e>
 *             p2p_frac=<f> p2p_pairs=<n> m2l_pairs=<n> m2l_fallback=<n>
 *             ops=<n> op_cap=<n> bytes_per_key=<n> table_bytes=<n>
 *             keys_built=<n> cache=<n> maint=<a> particles=<n> nonfinite=<n>
 *
 * Exit code 0 iff every structural check passed; the accuracy numbers never
 * affect it.
 */

#include <Beatnik_BRSolverDirect.hpp>
#include <Beatnik_FarFieldInterface.hpp>
#include <Beatnik_MeshGeometry.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Solver.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include "Beatnik_TestAssert.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using Beatnik::Real;

//---------------------------------------------------------------------------//
// The state. Field for field `Beatnik_Test_FmmVsDirect.cpp::makeSpinUpParams`,
// which is itself field for field the milestone-0 command line, with the level
// and the step count arriving as arguments instead of as constants. Every value
// is set EXPLICITLY rather than inherited from a Beatnik default, so a later
// change to a default breaks this loudly instead of silently changing what was
// measured.
//---------------------------------------------------------------------------//
constexpr Real kEps = 0.025;

/// Entity counts of the subdivision-L icosphere: `V = 10*4^L + 2`,
/// `F = 20*4^L`, `E = 30*4^L`. Computed rather than tabulated, so the level
/// stays an argument. Copied in shape from
/// `Beatnik_Test_Milestone0Run.cpp::verticesForLevel`.
long long powFour( int level )
{
    long long f = 1;
    for ( int i = 0; i < level; ++i )
        f *= 4;
    return f;
}
long long verticesForLevel( int level ) { return 10 * powFour( level ) + 2; }
long long edgesForLevel( int level ) { return 30 * powFour( level ); }
long long facesForLevel( int level ) { return 20 * powFour( level ); }

//---------------------------------------------------------------------------//
Beatnik::SolverParams makeScanParams( int subdivisions, int spin_up_steps )
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere, --radius 0.25,
    // --center-z 0.25, --icosphere-subdivisions <L>.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = subdivisions;
    p.initial.radius = 0.25;
    p.initial.center_z = 0.25;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0.
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --sigma 0
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = kEps;
    p.zmodel.sigma = 0.0;
    // --forcing-sign 1 --br-sign 1 --kernel-blob-mode length
    p.zmodel.forcing_sign = 1.0;
    p.zmodel.br_sign = 1.0;
    p.zmodel.blob_mode = Beatnik::KernelBlobMode::Length;
    // --viscosity-mode laplace-beltrami, --velocity-mode full,
    // --bernoulli-scalar-mode normal-speed, preserve_volume on.
    p.zmodel.viscosity_mode = Beatnik::ViscosityMode::LaplaceBeltrami;
    p.zmodel.velocity_mode = Beatnik::VelocityMode::Full;
    p.zmodel.bernoulli_scalar_mode = Beatnik::BernoulliScalarMode::NormalSpeed;
    p.zmodel.preserve_volume = true;
    // --br-approximation direct. THE STATE MUST BE ONE THE FMM DID NOT
    // PRODUCE, or every comparison below is partly circular.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex. MANDATORY: ZModelParams defaults to Face,
    // whose generate() throws, and the far-field adapter rejects anything but
    // Vertex regardless -- source, target and output row are one index there.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // The dt controls of the milestone-0 configuration.
    p.time.steps = spin_up_steps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --no-dynamic-remesh --refine-every 0. Frozen connectivity, so the
    // adapter's forward distributor takes its reuse branch and the source set
    // never changes shape under it.
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;
    p.cleanup.enabled = true;

    // No checkpoints at all.
    p.checkpoint.every_steps = 0;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = "";

    return p;
}

//---------------------------------------------------------------------------//
// THE SCAN.
//
// One row per arm. `(basis, order)` is the ONLY axis limited to what T2 built:
// `CartesianTaylor` at 0, 2, 3, 4, 5 and `SolidHarmonic` at 3 only. A seventh
// arm is one line in the adapter's dispatch plus roughly 8 CPU-minutes of build
// per translation-unit set, and nothing in this scan needs one.
// `mac_theta`, `ncrit` and `max_depth` are runtime `FmmParams` members and
// scan freely.
//
// Laid out as five NAMED AXES through a common background rather than as a full
// Cartesian product. The product would be 5*5*5*5 = 625 arms to answer four
// one-dimensional questions, and a cross term that mattered would still not be
// readable out of it. The background is T4's measured-live configuration:
// `CartesianTaylor`, order 3, `ncrit` 8, theta 0.3, `max_depth` 10.
//---------------------------------------------------------------------------//
struct Arm
{
    const char* axis;
    Beatnik::FarFieldBasis basis;
    int order;
    int ncrit;
    Real mac_theta;
    int max_depth;
};

/// The background every axis varies one member of.
constexpr Beatnik::FarFieldBasis kBgBasis = Beatnik::FarFieldBasis::CartesianTaylor;
constexpr int kBgOrder = 3;
constexpr int kBgNcrit = 8;
constexpr Real kBgTheta = 0.3;
constexpr int kBgMaxDepth = 10;

std::vector<Arm> buildScan()
{
    std::vector<Arm> s;

    // AXIS "order" -- the curve R1 and R11 are both about. Under
    // CartesianTaylor the gradient error should fall with `order` at the rate
    // (theta/2sqrt3)^p predicts and then flatten into Canopy's floating-point
    // floor. p=4 and p=5 are MEASURED AND REPORTED BUT MAY NOT BE ADOPTED as
    // the production order: p=4 reaches |k|=8 and Canopy's derivative-ladder
    // oracle stops at 6, so a silent recurrence error there would present as
    // exactly the plateau R1 describes and be attributed to truncation (R11).
    for ( int p : { 0, 2, 3, 4, 5 } )
        s.push_back( { "order", kBgBasis, p, kBgNcrit, kBgTheta, kBgMaxDepth } );

    // AXIS "basis" -- R2's discriminator, at the production order only, which
    // is the one solid-harmonic arm T2 dispatches. THE DISCRIMINATOR IS THE
    // SHAPE OF THE ORDER CURVE ABOVE AGAINST THIS FIXED POINT, not the size of
    // the gap: the bias is in the kernel rather than in the truncation, so no
    // order rescues this arm and what proves the selector is live is that the
    // CartesianTaylor curve keeps falling with `order` while this point does
    // not move with it. A separation of tens of percent is a self-contact
    // figure and no configuration in this tree reaches self-contact.
    s.push_back( { "basis", Beatnik::FarFieldBasis::SolidHarmonic, 3, kBgNcrit,
                   kBgTheta, kBgMaxDepth } );

    // AXIS "ncrit" -- the liveness axis. 64 is the compiled default and is
    // included precisely to show what it measures at these vertex counts.
    for ( int n : { 64, 32, 16, 8, 4 } )
        s.push_back( { "ncrit", kBgBasis, kBgOrder, n, kBgTheta, kBgMaxDepth } );

    // AXIS "theta" -- the acceptance criterion. Canopy accepts equal-size
    // cells only beyond R/w > 2sqrt3/theta, so a larger theta accepts closer
    // pairs: more far field (cheaper, and live at lower N) and a larger
    // truncation error at fixed order. 0.3 is Beatnik's and the reference's;
    // 0.5 is Canopy's own default and the second row of the model table.
    for ( Real t : { Real( 0.2 ), Real( 0.3 ), Real( 0.4 ), Real( 0.5 ),
                     Real( 0.7 ) } )
        s.push_back( { "theta", kBgBasis, kBgOrder, kBgNcrit, t, kBgMaxDepth } );

    // AXIS "max_depth" -- R6 and R12. T4 measured fallback count 0 at the
    // background, so R6 has no evidence to act on and this axis starts from
    // "not binding here". Per R12 it may be LOWERED only on evidence of
    // realized overflow, never pre-emptively, so the axis is measured in both
    // directions: 12 above the default and 5-8 below it, to see whether the
    // realized key count responds to it at all at these depths.
    for ( int d : { 5, 6, 8, 10, 12 } )
        s.push_back( { "max_depth", kBgBasis, kBgOrder, kBgNcrit, kBgTheta, d } );

    // AXIS "order@0.5" -- the model's SECOND row, so the rate is checked at
    // two theta values rather than fitted at one. The model predicts
    // 2.1e-2 / 3.0e-3 / 4.3e-4 at p = 2 / 3 / 4 here, against
    // 7.5e-3 / 6.5e-4 / 5.6e-5 at theta 0.3.
    for ( int p : { 2, 3, 4 } )
        s.push_back( { "order@0.5", kBgBasis, p, kBgNcrit, Real( 0.5 ),
                       kBgMaxDepth } );

    return s;
}

//---------------------------------------------------------------------------//
double globalMax( MPI_Comm comm, double local )
{
    double out = local;
    MPI_Allreduce( &local, &out, 1, MPI_DOUBLE, MPI_MAX, comm );
    return out;
}

long long globalCount( MPI_Comm comm, long long local )
{
    long long out = local;
    MPI_Allreduce( &local, &out, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

//---------------------------------------------------------------------------//
/// Max magnitude of an `(N,3)` field over owned rows, reduced globally.
template <class ExecSpace, class ViewType>
double fieldScale( MPI_Comm comm, const ViewType& v, int n )
{
    Real worst = 0;
    Kokkos::parallel_reduce(
        "beatnik_t5_field_scale", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m ) {
            const Real mag = Kokkos::sqrt( v( i, 0 ) * v( i, 0 ) +
                                           v( i, 1 ) * v( i, 1 ) +
                                           v( i, 2 ) * v( i, 2 ) );
            if ( mag > m )
                m = mag;
        },
        Kokkos::Max<Real>( worst ) );
    return globalMax( comm, static_cast<double>( worst ) );
}

//---------------------------------------------------------------------------//
struct FieldDifference
{
    double max_abs = 0.0;
    long long nonfinite = 0;
};

/// Max \f$|a_i-b_i|\f$ over owned rows, reduced globally, with the count of
/// non-finite rows in `a`. One pass: the finiteness check and the error are the
/// same walk. Copied in shape from `Beatnik_Test_FmmVsDirect.cpp`.
template <class ExecSpace, class ViewA, class ViewB>
FieldDifference fieldDifference( MPI_Comm comm, const ViewA& a, const ViewB& b,
                                 int n )
{
    Real worst = 0;
    long long bad = 0;
    Kokkos::parallel_reduce(
        "beatnik_t5_field_difference", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, Real& m, long long& nb ) {
            Real d[3];
            bool finite = true;
            for ( int c = 0; c < 3; ++c )
            {
                d[c] = a( i, c ) - b( i, c );
                if ( !Kokkos::isfinite( a( i, c ) ) )
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
    out.max_abs = globalMax( comm, static_cast<double>( worst ) );
    out.nonfinite = globalCount( comm, bad );
    return out;
}

//---------------------------------------------------------------------------//
/**
 * @brief The globally gathered source set, on the device.
 *
 * The potential reference is an \f$O(N^2)\f$ sum over **every** source on
 * **every** rank, so the whole set has to be resident somewhere. At
 * milestone-0's 642 and 2562 vertices that is 15 KB and 61 KB per array, so it
 * is gathered whole with one `MPI_Allgatherv` per field rather than streamed
 * through a ring as `BRSolverDirect` does — the ring exists to bound memory at
 * production vertex counts and buys nothing here.
 */
template <class ExecSpace, class MemSpace>
struct GlobalSources
{
    using view_type = Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;
    view_type positions;
    view_type charges;
    int count = 0;
};

template <class ExecSpace, class MemSpace, class ViewP, class ViewQ>
GlobalSources<ExecSpace, MemSpace>
gatherSources( MPI_Comm comm, const ViewP& points, const ViewQ& charges,
               int n_owned )
{
    int comm_size = 1;
    MPI_Comm_size( comm, &comm_size );

    std::vector<int> counts( static_cast<std::size_t>( comm_size ), 0 );
    MPI_Allgather( &n_owned, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );

    // Element counts and displacements in units of Real, i.e. 3 per row.
    std::vector<int> rcounts( static_cast<std::size_t>( comm_size ), 0 );
    std::vector<int> displs( static_cast<std::size_t>( comm_size ), 0 );
    int total = 0;
    for ( int r = 0; r < comm_size; ++r )
    {
        rcounts[static_cast<std::size_t>( r )] = 3 * counts[static_cast<std::size_t>( r )];
        displs[static_cast<std::size_t>( r )] = 3 * total;
        total += counts[static_cast<std::size_t>( r )];
    }

    auto h_points = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), points );
    auto h_charges = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), charges );

    // PACKED BY EXPLICIT INDEXING, NEVER THROUGH `.data()`. A
    // `Kokkos::View<Real*[3], Device>` is LayoutLeft on HIP and LayoutRight on
    // Serial, so `.data()` is column-major in one build and row-major in the
    // other -- and an MPI_Allgatherv over the raw pointer would silently
    // transpose the source set on the GPU backend. That does not crash: it
    // produces a potential reference built from a scrambled point cloud, whose
    // disagreement with the FMM would read as a truncation error of about the
    // size T5 is trying to measure.
    static_assert( sizeof( Real ) == sizeof( double ),
                   "the potential reference's Allgatherv assumes Real is "
                   "MPI_DOUBLE; add a dispatch if that stops being true" );

    std::vector<Real> send_pos( static_cast<std::size_t>( 3 * n_owned ), Real( 0 ) );
    std::vector<Real> send_chg( static_cast<std::size_t>( 3 * n_owned ), Real( 0 ) );
    for ( int i = 0; i < n_owned; ++i )
        for ( int d = 0; d < 3; ++d )
        {
            send_pos[static_cast<std::size_t>( 3 * i + d )] = h_points( i, d );
            send_chg[static_cast<std::size_t>( 3 * i + d )] = h_charges( i, d );
        }

    std::vector<Real> all_pos( static_cast<std::size_t>( 3 * total ), Real( 0 ) );
    std::vector<Real> all_chg( static_cast<std::size_t>( 3 * total ), Real( 0 ) );

    MPI_Allgatherv( send_pos.data(), 3 * n_owned, MPI_DOUBLE, all_pos.data(),
                    rcounts.data(), displs.data(), MPI_DOUBLE, comm );
    MPI_Allgatherv( send_chg.data(), 3 * n_owned, MPI_DOUBLE, all_chg.data(),
                    rcounts.data(), displs.data(), MPI_DOUBLE, comm );

    GlobalSources<ExecSpace, MemSpace> g;
    g.count = total;
    g.positions = typename GlobalSources<ExecSpace, MemSpace>::view_type(
        "beatnik_t5_all_positions", total );
    g.charges = typename GlobalSources<ExecSpace, MemSpace>::view_type(
        "beatnik_t5_all_charges", total );

    auto h_all_pos = Kokkos::create_mirror_view( g.positions );
    auto h_all_chg = Kokkos::create_mirror_view( g.charges );
    for ( int i = 0; i < total; ++i )
        for ( int d = 0; d < 3; ++d )
        {
            h_all_pos( i, d ) = all_pos[static_cast<std::size_t>( 3 * i + d )];
            h_all_chg( i, d ) = all_chg[static_cast<std::size_t>( 3 * i + d )];
        }
    Kokkos::deep_copy( g.positions, h_all_pos );
    Kokkos::deep_copy( g.charges, h_all_chg );
    return g;
}

//---------------------------------------------------------------------------//
/**
 * @brief The direct softened-monopole potential, \f$O(N^2)\f$ over the global
 *        source set.
 *
 * \f$\phi_c(x_t)=\sum_{s}S_{s,c}\,(|x_t-y_s|^2+\varepsilon^2)^{-1/2}\f$ with
 * **Canopy's** exclusion rule, not Beatnik's: the term is skipped when
 * \f$|r|^2<10^{-24}\f$, which is the test `Canopy_P2P.hpp` applies alongside
 * `pj == pi`. On a mesh the two coincide — distinct vertices are separated by
 * the minimum edge length, four orders above \f$10^{-12}\f$ — so the position
 * test alone reproduces the self-exclusion exactly and the reference needs no
 * global index.
 *
 * @param eps2 The softening **squared**, and specifically
 *        `diagnostics().softening * diagnostics().softening` — the value
 *        Canopy itself squared in `set_softening`. Re-deriving it as `blob()`
 *        would be the same number up to one `sqrt`-then-square round trip,
 *        which is not guaranteed exact and is free to avoid.
 */
template <class ExecSpace, class ViewT, class ViewOut, class GS>
void directPotential( const ViewT& targets, int n_owned, const GS& sources,
                      Real eps2, const ViewOut& out )
{
    const int ns = sources.count;
    auto spos = sources.positions;
    auto schg = sources.charges;
    Kokkos::parallel_for(
        "beatnik_t5_direct_potential",
        Kokkos::RangePolicy<ExecSpace>( 0, n_owned ),
        KOKKOS_LAMBDA( const int i ) {
            const Real xi = targets( i, 0 );
            const Real yi = targets( i, 1 );
            const Real zi = targets( i, 2 );
            Real phi[3] = { Real( 0 ), Real( 0 ), Real( 0 ) };
            for ( int j = 0; j < ns; ++j )
            {
                const Real dx = xi - spos( j, 0 );
                const Real dy = yi - spos( j, 1 );
                const Real dz = zi - spos( j, 2 );
                const Real r2 = dx * dx + dy * dy + dz * dz;
                // Canopy_P2P.hpp's own two exclusions, in its own order.
                if ( r2 < Real( 1.0e-24 ) )
                    continue;
                const Real inv_r = Real( 1 ) / Kokkos::sqrt( r2 + eps2 );
                for ( int c = 0; c < 3; ++c )
                    phi[c] += schg( j, c ) * inv_r;
            }
            for ( int c = 0; c < 3; ++c )
                out( i, c ) = phi[c];
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/// Everything one arm produced. Both columns, and the whole qualification list
/// the conventions table requires beside any number taken from it.
struct ArmResult
{
    double grad_abs = 0.0;
    double grad_rel = 0.0;
    double pot_abs = 0.0;
    double pot_rel = 0.0;
    long long nonfinite = 0;

    double p2p_fraction = 0.0;
    long long p2p_pairs = 0;
    long long particles = 0;
    long long m2l_pairs = 0;
    long long m2l_fallback = 0;
    int ops = 0;
    int op_cap = 0;
    std::size_t bytes_per_key = 0;
    int cache = 0;
    long long keys_built = 0;
    double softening = 0.0;
    Beatnik::FarFieldDiagnostics::Maintenance maintenance =
        Beatnik::FarFieldDiagnostics::Maintenance::Setup;

    bool threw = false;
    std::string why;
};

//---------------------------------------------------------------------------//
void printArm( const Arm& a, const ArmResult& r )
{
    if ( r.threw )
    {
        std::printf( "[t5arm] axis=%s basis=%s order=%d ncrit=%d theta=%.4f "
                     "max_depth=%d THREW why=\"%s\"\n",
                     a.axis, Beatnik::toString( a.basis ), a.order, a.ncrit,
                     static_cast<double>( a.mac_theta ), a.max_depth,
                     r.why.c_str() );
        std::fflush( stdout );
        return;
    }
    std::printf(
        "[t5arm] axis=%s basis=%s order=%d ncrit=%d theta=%.4f max_depth=%d "
        "softening=%.17g near_soft_factor=0 "
        "grad_abs=%.17g grad_rel=%.17g pot_abs=%.17g pot_rel=%.17g "
        "p2p_frac=%.17g p2p_pairs=%lld m2l_pairs=%lld m2l_fallback=%lld "
        "ops=%d op_cap=%d bytes_per_key=%zu table_bytes=%zu keys_built=%lld "
        "cache=%d maint=%s particles=%lld nonfinite=%lld\n",
        a.axis, Beatnik::toString( a.basis ), a.order, a.ncrit,
        static_cast<double>( a.mac_theta ), a.max_depth, r.softening,
        r.grad_abs, r.grad_rel, r.pot_abs, r.pot_rel, r.p2p_fraction,
        r.p2p_pairs, r.m2l_pairs, r.m2l_fallback, r.ops, r.op_cap,
        r.bytes_per_key,
        static_cast<std::size_t>( r.ops ) * r.bytes_per_key, r.keys_built,
        r.cache, Beatnik::toString( r.maintenance ), r.particles,
        r.nonfinite );
    std::fflush( stdout );
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runScan( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    using solver_type = Beatnik::Solver<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using direct_type = Beatnik::BRSolverDirect<ExecSpace, MemSpace>;
    using far_field_type = Beatnik::FarFieldSolver<ExecSpace, MemSpace>;
    using vector_view =
        Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );

    if ( argc < 2 )
    {
        rec.fail( "usage: <icosphere-subdivisions> [spin-up-steps]; see the "
                  "ARGUMENTS block in this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const int level = std::atoi( argv[1] );
    const int spin_up = ( argc > 2 ) ? std::atoi( argv[2] ) : 5;
    if ( level < 0 || spin_up < 0 )
    {
        rec.fail( "both arguments must be non-negative integers" );
        return;
    }

    const long long want_vertices = verticesForLevel( level );
    {
        std::ostringstream os;
        os << "execution space " << ExecSpace::name() << ", ranks " << comm_size
           << ", icosphere subdivisions " << level << " (" << want_vertices
           << " vertices), spin-up steps " << spin_up
           << " with --br-approximation direct";
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // The state. Spun up ONCE; every arm below sees these exact bytes.
    //-----------------------------------------------------------------------//
    Beatnik::SolverParams params = makeScanParams( level, spin_up );
    solver_type solver( comm, params );
    solver.setup();

    auto& mesh = solver.mesh();
    const auto& state = solver.state();

    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), want_vertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), edgesForLevel( level ) );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), facesForLevel( level ) );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

    for ( int s = 0; s < spin_up; ++s )
    {
        if ( !solver.advanceOneStep() )
        {
            rec.fail( "the direct spin-up went non-finite at step " +
                      std::to_string( s + 1 ) +
                      "; nothing downstream of this is meaningful" );
            return;
        }
    }
    BEATNIK_CHECK_EQ( rec, solver.step(), spin_up );

    // The evaluation preconditions, in the order ZModelSolver's RHS
    // establishes them (`Beatnik_ZModelSolver.hpp`, steps 0-2): one whole-tuple
    // halo exchange, geometry at the CURRENT positions, then the sheet vector.
    // Every arm is then handed the SAME mesh, geometry, state and quadrature.
    mesh.haloExchange();
    geometry_type geometry;
    geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                      mesh.faceVertices() );
    state.updateSheetVector( mesh, geometry );

    auto quadrature = Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
        Beatnik::SourceQuadrature::Vertex );

    const int n_owned = mesh.ownedVertexCount();
    BEATNIK_CHECK_EQ( rec, globalCount( comm, n_owned ), want_vertices );

    typename Beatnik::SourceQuadratureBase<ExecSpace, MemSpace>::point_view
        points;
    typename Beatnik::SourceQuadratureBase<ExecSpace, MemSpace>::strength_view
        strengths;
    quadrature->generate( mesh, geometry, state, points, strengths );
    BEATNIK_CHECK_EQ( rec, static_cast<long long>( points.extent( 0 ) ),
                      n_owned );

    // THE STATE IS REAL. A zero sheet strength makes every field zero and every
    // comparison below vacuous, so this is asserted rather than inferred from
    // the step count.
    const double strength_scale =
        fieldScale<ExecSpace>( comm, strengths, n_owned );
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "after " << spin_up << " direct steps: max |area * S| "
           << strength_scale << ", simulation time " << solver.time();
        rec.note( os.str() );
    }
    BEATNIK_CHECK_TRUE( rec, strength_scale > 0.0 );

    //-----------------------------------------------------------------------//
    // Reference 1: the direct velocity, i.e. the GRADIENT column's reference.
    //-----------------------------------------------------------------------//
    direct_type direct( comm );
    vector_view u_direct( "beatnik_t5_u_direct", n_owned );
    direct.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                     params.zmodel, u_direct );
    const double grad_scale = fieldScale<ExecSpace>( comm, u_direct, n_owned );
    BEATNIK_CHECK_TRUE( rec, grad_scale > 1.0e-6 );

    //-----------------------------------------------------------------------//
    // Reference 2: the direct potential. Softening squared comes from the
    // adapter's own reported length below, so one throwaway arm is evaluated
    // first to learn it -- cheaper than duplicating the blob()-to-length
    // derivation here and creating a second source of truth for it.
    //-----------------------------------------------------------------------//
    auto global_sources =
        gatherSources<ExecSpace, MemSpace>( comm, points, strengths, n_owned );
    BEATNIK_CHECK_EQ( rec, global_sources.count, want_vertices );

    Real eps2 = Real( 0 );
    {
        Beatnik::FmmParams probe;
        probe.basis = kBgBasis;
        probe.order = kBgOrder;
        probe.ncrit = kBgNcrit;
        far_field_type ff( comm, probe );
        vector_view scratch( "beatnik_t5_probe", n_owned );
        ff.evaluateVelocity( points, strengths, params.zmodel, scratch );
        const Real s = ff.diagnostics().softening;
        eps2 = s * s;
        std::ostringstream os;
        os.precision( 17 );
        os << "softening length from the adapter " << static_cast<double>( s )
           << ", squared " << static_cast<double>( eps2 )
           << " (ZModelParams::blob() is "
           << static_cast<double>( params.zmodel.blob() ) << ")";
        rec.note( os.str() );
        BEATNIK_CHECK_TRUE( rec, s > Real( 0 ) );
    }

    vector_view phi_direct( "beatnik_t5_phi_direct", n_owned );
    directPotential<ExecSpace>( points, n_owned, global_sources, eps2,
                                phi_direct );
    const double pot_scale =
        fieldScale<ExecSpace>( comm, phi_direct, n_owned );
    BEATNIK_CHECK_TRUE( rec, pot_scale > 1.0e-6 );

    {
        std::ostringstream os;
        os.precision( 17 );
        os << "field scales: max |u_direct| " << grad_scale
           << ", max |phi_direct| " << pot_scale;
        rec.note( os.str() );
    }

    // The header line for the [t5arm] table, so the log is readable without
    // this file open beside it.
    if ( rank == 0 )
    {
        std::printf( "[t5] scan begins: level=%d vertices=%lld ranks=%d "
                     "space=%s spin_up=%d grad_scale=%.17g pot_scale=%.17g "
                     "eps2=%.17g\n",
                     level, want_vertices, comm_size, ExecSpace::name(),
                     spin_up, grad_scale, pot_scale,
                     static_cast<double>( eps2 ) );
        std::fflush( stdout );
    }

    //-----------------------------------------------------------------------//
    // The arms.
    //-----------------------------------------------------------------------//
    const std::vector<Arm> scan = buildScan();
    vector_view u_fmm( "beatnik_t5_u_fmm", n_owned );
    vector_view phi_fmm( "beatnik_t5_phi_fmm", n_owned );

    for ( const Arm& a : scan )
    {
        ArmResult r;
        try
        {
            Beatnik::FmmParams f;
            f.basis = a.basis;
            f.order = a.order;
            f.ncrit = a.ncrit;
            f.mac_theta = a.mac_theta;
            f.max_depth = a.max_depth;

            far_field_type ff( comm, f );

            // The gradient column first, then the potential column on the same
            // solver. The second is a second `solve()` over the same tree --
            // the round trip carries one Output tuple -- so its maintenance
            // action is `Migrate` where the first is `Setup`, and the
            // diagnostics reported are the SECOND evaluation's. Both
            // evaluations see byte-identical positions, so the tree, the P2P
            // lists and the pair counts are the same; the printed p2p_frac is
            // the one the potential column was measured at and the check below
            // is what says the gradient column's agreed.
            ff.evaluateVelocity( points, strengths, params.zmodel, u_fmm );
            const auto grad_diag = ff.diagnostics();
            const FieldDifference dg =
                fieldDifference<ExecSpace>( comm, u_fmm, u_direct, n_owned );

            ff.evaluatePotential( points, strengths, params.zmodel, phi_fmm );
            const auto& diag = ff.diagnostics();
            const FieldDifference dp =
                fieldDifference<ExecSpace>( comm, phi_fmm, phi_direct,
                                            n_owned );

            r.grad_abs = dg.max_abs;
            r.grad_rel = dg.max_abs / grad_scale;
            r.pot_abs = dp.max_abs;
            r.pot_rel = dp.max_abs / pot_scale;
            r.nonfinite = dg.nonfinite + dp.nonfinite;

            r.p2p_fraction = diag.p2p_pair_fraction;
            r.p2p_pairs = diag.global_p2p_pair_count;
            r.particles = diag.global_particle_count;
            r.m2l_pairs = diag.global_m2l_pair_count;
            r.m2l_fallback = diag.global_m2l_fallback_pair_count;
            r.ops = diag.local_m2l_unique_op_count;
            r.op_cap = diag.local_m2l_op_cap;
            r.bytes_per_key = diag.local_m2l_bytes_per_key;
            r.cache = diag.local_m2l_op_cache_size;
            r.keys_built = diag.local_m2l_op_keys_built;
            r.softening = static_cast<double>( diag.softening );
            r.maintenance = diag.maintenance;

            // STRUCTURAL, not accuracy: the round trip neither dropped nor
            // duplicated a source (R3), the field is finite, and the two
            // evaluations of this arm saw the same tree -- which is what lets
            // one p2p fraction qualify both columns.
            BEATNIK_CHECK_EQ( rec, r.particles, want_vertices );
            BEATNIK_CHECK_EQ( rec, r.nonfinite, 0 );
            BEATNIK_CHECK_EQ( rec, grad_diag.global_p2p_pair_count,
                              diag.global_p2p_pair_count );
        }
        catch ( const std::exception& e )
        {
            // An arm may legitimately refuse: `validateTreeParams` rejects an
            // `ncrit` or `max_depth` Canopy will not take, and the dispatch
            // throws on a `(basis, order)` pair T2 did not build. A refusal is
            // a RESULT for the scan, printed with its reason, and it is a
            // recorded failure too -- the scan table names only pairs that
            // should be accepted, so a throw means the table and the dispatch
            // disagree.
            r.threw = true;
            r.why = e.what();
            rec.fail( std::string( "arm axis=" ) + a.axis + " basis=" +
                      Beatnik::toString( a.basis ) + " order=" +
                      std::to_string( a.order ) + " ncrit=" +
                      std::to_string( a.ncrit ) + " threw: " + e.what() );
        }
        if ( rank == 0 )
            printArm( a, r );
    }

    if ( rank == 0 )
    {
        std::printf( "[t5] scan complete: %zu arm(s) at level %d, %d rank(s)\n",
                     scan.size(), level, comm_size );
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
        Beatnik::Test::Recorder rec( "Beatnik_Test_FmmScan" );
        try
        {
            // Defined by the per-backend shim tests/CMakeLists.txt generates,
            // so the binary's `_SERIAL` / `_HIP` suffix means what the batch
            // script assumes it means. Defaulting keeps the file compilable
            // alone.
#ifndef BEATNIK_TEST_EXEC_SPACE
#define BEATNIK_TEST_EXEC_SPACE Kokkos::DefaultExecutionSpace
#endif
            using exec_space = BEATNIK_TEST_EXEC_SPACE;
            runScan<exec_space, typename exec_space::memory_space>( rec, argc,
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

    // ONE VERDICT ACROSS THE RANKS, as every standalone test here does: each
    // rank printed its own tally, MPI_MAX makes any rank's failure the job's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
