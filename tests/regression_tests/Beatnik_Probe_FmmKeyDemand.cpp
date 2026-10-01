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
 * @file Beatnik_Probe_FmmKeyDemand.cpp
 * @brief **T3's MEASUREMENT DRIVER** (`tasks/add-canopy-t6.md`) — the M2L
 *        operator-key DEMAND series along the milestone-0 direct trajectory,
 *        at every checkpointed step, per rank.
 *
 * THIS IS NOT A TEST AND IS IN NO TIER.
 * ------------------------------------
 * It carries no `LABELS`, no ctest case and no manifest line, so neither
 * `run_regression_minset.flux` (the ship gate) nor `run_milestone.flux` can
 * pick it up — see the "Measurement drivers — IN NO TIER" loop in
 * `tests/CMakeLists.txt`, which is the milestone tier's loop stopped short of
 * the point where it applies a label. **No tolerance is compiled into this
 * file and none may be** (risk **R6**): it measures the demand, and T8 decides
 * what to do about it. The only assertions here are structural — the entity
 * round trip, finiteness, and the parameter-set check below — i.e. the
 * conditions under which the printed numbers mean anything at all.
 *
 * THE QUESTION. Canopy's M2L operator table admits at most
 * `m2l_effective_op_cap()` distinct operator keys per rank; every pair whose
 * key is refused takes the per-pair fallback instead. `n_unique_ops` is the
 * count ADMITTED and saturates at the cap, so it cannot say how far past the
 * cap the tree actually reaches. T1 added the counter that can —
 * `m2l_n_demanded_ops()` — and T2 carried it into
 * `FarFieldDiagnostics::local_m2l_demanded_op_count`. This driver is what
 * reads it along a real trajectory.
 *
 * PER RANK, UNREDUCED, AND NEVER AVERAGED. The operator-key cap is per rank,
 * so four ranks carry four times the key budget and overflow **less** than one
 * rank, not four times as much — T0 measured exactly that (13 274 fallback
 * pairs at np1 against 5 300 at np4, same step, same level). A mean over ranks
 * would hide the rank that actually overflows. Every rank therefore prints its
 * own `[t6probe]` row at every state, tagged with its rank, and the trailer is
 * per rank too. Only the four fields Canopy itself reduces
 * (`global_m2l_pair_count`, `global_m2l_fallback_pair_count`,
 * `global_particle_count`, `p2p_pair_fraction`) are global, and they are
 * spelled `global_` in the row for that reason.
 *
 * **DO NOT USE THE `[Canopy] M2L op count exceeded cap` WARNING AS AN OVERFLOW
 * INDICATOR.** It is warn-once-per-build and T0 measured it appearing 2 732 and
 * 2 733 times in the two np1 level-4 launches and **zero** times in either np4
 * launch, despite non-zero fallback at 71 of 81 states there. The overflow
 * signal is the printed demand and fallback columns, never a grep for that
 * line.
 *
 * THE DEMAND PEAK IS NOT WHERE THE ERROR PEAKS (**R3**). At level 4 the cap
 * first trips at step 250, the fallback count peaks at step 1650, and claim A's
 * velocity error peaks at step 1375 — three different steps. The trailer below
 * therefore reports the demand series' own peak and its own step, and locates
 * the first exceedance rather than assuming it.
 *
 * `-1` IS "UNAVAILABLE", `0` IS A MEASUREMENT (**R7**). The demand counter is
 * gated on `CANOPY_ENABLE_PROFILING`, which a `canopy ~profiling` build leaves
 * undefined; the count is then `-1` and the saturation flag `false`, and
 * neither is a number about the tree. A run against such a build is **not a
 * measurement**, and the header below says so in one loud line rather than
 * letting a reader take `-1` for a count. The two depth-derived fields beside
 * it — `local_m2l_occupied_depths` and `local_m2l_cells_at_max_depth` — come
 * from the **ungated** `m2l_cells_at_depth()` and carry real counts in every
 * build; a `-1` in those two would be this driver's own bug, not a missing
 * profiling define.
 *
 * THE PARAMETER SET MUST NOT DRIFT FROM THE MEMBER'S (**R5**). This file
 * re-derives `Beatnik_Test_Milestone0Fmm.cpp`'s claim-A setup rather than
 * sharing it — that member is 2 201 lines, carries no step-count or claim
 * selector on purpose, and factoring its driving loop into a shared header
 * would mix a refactor of a currently failing member into its own diagnosis.
 * The cost of duplicating it is that the two can drift apart silently, so the
 * five level-independent knobs the member pins are **echoed out of
 * `fmm.farField().params()` and checked against compiled-in literals** before
 * any state is measured: `ncrit` 8 (`Beatnik_Test_Milestone0Fmm.cpp:340`),
 * `order` 3 (`:345`), `CartesianTaylor`, `mac_theta` 0.3 and `max_depth` 10
 * (asserted there at `:1371-1387`). A probe measuring a different
 * configuration than the member is worse than no probe.
 *
 * WHAT IT DRIVES. The **direct** 2000-step milestone-0 trajectory at the level
 * given on the command line, checkpointing every 25 steps — the same 81 states
 * claim A evaluates at. At each of them the preconditions are established
 * exactly as `evaluateClaimA` does (`Beatnik_Test_Milestone0Fmm.cpp:1397-1437`,
 * and `Beatnik_ZModelSolver.hpp` steps 0-2): one whole-tuple halo exchange,
 * geometry at the current positions, then the sheet vector. Then
 * `computeInterfaceVelocity` is called **once**, on the FMM solver only. The
 * direct comparator is deliberately absent: this driver measures demand, not
 * error, and the direct sum is the expensive half of claim A.
 *
 * The level is a runtime argument here, unlike in the member, whose
 * `kSubdivisions` is compile-time (`BEATNIK_M0_FMM_LEVEL`). The precedent
 * followed is `Beatnik_Test_Milestone0Run.cpp`: the level is `argv[1]`, and
 * `verticesForLevel` COMPUTES `10*4^L + 2` rather than tabulating it, so the
 * round-trip check needs no per-level literal table. All five checked knobs
 * are level-independent, so neither does the parameter check.
 *
 * ARGUMENTS. One required positional and one optional one. There is no option
 * surface here — a driver's arguments come from the batch script that measures
 * with it (`tests/CMakeLists.txt`, the driver loop), as positionals. **The
 * standing refusal is a step-count override**: a knob that can silently
 * shorten a 2000-step run is how a truncated run reads as a shorter pass, so
 * the step count stays the compiled `kSteps` and no argument will ever move
 * it. `argv[2]` is admitted on exactly that reasoning — a column-count cap
 * shortens nothing, every state is still measured, and the cap is printed in
 * the header beside the effective one, so a run records what it was configured
 * with as well as what was in force.
 *
 *   argv[1]  --icosphere-subdivisions   the level to probe (3 or 4)
 *   argv[2]  m2l_op_count_cap          OPTIONAL. Canopy's M2L operator
 *                                      column-count cap, reaching
 *                                      `FmmParams::m2l_op_count_cap`. Absent
 *                                      means the `FmmParams` default (32768).
 *                                      0 is legal and admits no column; a
 *                                      negative value is rejected here rather
 *                                      than left to Canopy's own throw.
 *
 * Checkpoints go to
 * `${BEATNIK_TEST_SCRATCH}/keydemand_sub<L>_<space>_np<N>`, a subdirectory
 * even though the batch script already hands each run its own scratch, so two
 * runs sharing one scratch by mistake cannot overwrite each other's series.
 * **`BEATNIK_TEST_SCRATCH` must be on a parallel filesystem**: the checkpoints
 * go through MPI-IO and a node-local scratch fails every launch spanning more
 * than one node.
 *
 * OUTPUT. `key=value` throughout, so the batch log reduces to a table without
 * parsing prose. One `[t6probe] header` line, then one `[t6probe] row` per
 * (state, rank), then one `[t6probe] trailer` per rank and one
 * `[t6probe] total` from rank 0:
 *
 *     [t6probe] header level=<L> vertices=<n> ranks=<n> space=<s> steps=<n>
 *               every=<n> ncrit=<n> order=<n> basis=<b> mac_theta=<t>
 *               max_depth=<d> near_soft_factor=<f> bytes_per_key=<n>
 *               byte_budget=<n> op_count_cap=<n> op_cap=<n>
 *               demand_available=<0|1>
 *     [t6probe] row rank=<r>/<n> step=<s> demand=<n> demand_saturated=<0|1>
 *               unique_ops=<n> op_cap=<n> occupied_depths=<n>
 *               cells_at_max_depth=<n> cache=<n> keys_built=<n>
 *               keys_built_delta=<n> global_m2l_pairs=<n>
 *               global_m2l_fallback=<n> global_p2p_frac=<f>
 *               global_particles=<n> eval_wall=<s>
 *     [t6probe] trailer rank=<r>/<n> demand_available=<0|1>
 *               first_exceed_step=<s> peak_demand=<n> peak_demand_step=<s>
 *               peak_keys_built_delta=<n> peak_keys_built_delta_step=<s>
 *               states=<n> eval_wall_total=<s>
 *
 * Exit code 0 unless the run could not proceed: a throw, a non-finite velocity,
 * a particle-count round-trip mismatch, an early stop, or a parameter-set
 * mismatch against the member. **No measured value affects it.**
 */

#include <Beatnik_BRSolverFMM.hpp>
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
// The configuration. Every value is set EXPLICITLY rather than inherited from a
// Beatnik default, so a later change to a default breaks this loudly instead of
// silently changing what was measured.
//---------------------------------------------------------------------------//
constexpr Real kRadius = 0.25;
constexpr Real kCenterZ = 0.25;

/// The milestone-0 step budget and checkpoint interval, which together give the
/// gold sets' own 81 states. There is no argument for either (see the header).
constexpr int kSteps = 2000;
constexpr int kCheckpointEvery = 25;

//---------------------------------------------------------------------------//
// THE MEMBER'S PARAMETER SET, as compiled-in literals (R5). All five are
// level-independent, so this is one table and not one per level. The line
// numbers are `Beatnik_Test_Milestone0Fmm.cpp`'s.
//---------------------------------------------------------------------------//

/// Leaf occupancy. `kNcrit` at `:340` — the member's ONE departure from
/// `FmmParams`' compiled defaults, and itself a demand driver: it deepens the
/// tree to make the far field live at these vertex counts at all.
constexpr int kNcrit = 8;

/// `kProductionOrder` at `:345`.
constexpr int kProductionOrder = 3;

/// The three the member leaves at their defaults and then asserts at
/// `:1371-1387`, carried here for the same reason it asserts them.
constexpr Beatnik::FarFieldBasis kBasis = Beatnik::FarFieldBasis::CartesianTaylor;
constexpr double kMacTheta = 0.3;
constexpr int kMaxDepth = 10;
constexpr double kNearSofteningFactor = 0.0;

//---------------------------------------------------------------------------//
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
/// The milestone-0 command line as a `SolverParams`, field for field
/// `Beatnik_Test_Milestone0Fmm.cpp::makeParams` with the level as a parameter
/// and the BR approximation pinned to `Direct` — the trajectory this driver
/// walks is claim A's, which is a direct one, and the FMM is evaluated beside
/// it rather than driving it.
Beatnik::SolverParams makeParams( int subdivisions,
                                  const std::string& checkpoint_dir )
{
    Beatnik::SolverParams p;

    // --state-model potential, --mesh-kind icosphere, --radius 0.25,
    // --center-z 0.25, --icosphere-subdivisions <L>.
    p.state_model = Beatnik::StateModel::Potential;
    p.initial.mesh_kind = Beatnik::MeshKind::Icosphere;
    p.initial.icosphere_subdivisions = subdivisions;
    p.initial.radius = kRadius;
    p.initial.center_z = kCenterZ;
    // --initial-shape sphere, --initial-potential-strength 0, --polar-amp 0.
    p.initial.shape = Beatnik::InitialShape::Sphere;
    p.initial.initial_potential_strength = 0.0;
    p.initial.polar_amp = 0.0;

    // --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --sigma 0
    p.zmodel.A = 0.3;
    p.zmodel.g = 1.0;
    p.zmodel.mu = 0.002;
    p.zmodel.eps = 0.025;
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
    // --br-approximation direct. THE TRAJECTORY MUST BE THE MEMBER'S CLAIM-A
    // TRAJECTORY, which is a direct one; an FMM-driven walk would measure the
    // demand of a different sequence of states.
    p.zmodel.br_approximation = Beatnik::BRApproximation::Direct;
    // --source-quadrature vertex. MANDATORY, not decorative: ZModelParams
    // defaults to Face, whose generate() throws, and the far-field adapter
    // rejects anything but Vertex regardless.
    p.zmodel.source_quadrature = Beatnik::SourceQuadrature::Vertex;

    // --steps 2000, --adaptive-dt, and the dt controls both gold sets were
    // generated under. Every one is a Python default and every one changes the
    // trajectory.
    p.time.steps = kSteps;
    p.time.dt = 0.003;
    p.time.adaptive_dt = true;
    p.time.min_dt = 2.5e-4;
    p.time.dt_edge_power = 1.0;
    p.time.max_sheet_dt_product = 0.0;
    p.time.dt_switch_time = -1.0;
    p.time.have_t_end = false;

    // --no-dynamic-remesh --refine-every 0. Connectivity is frozen for the
    // whole run, which is what makes the entity-count check a statement about
    // the FMM rather than about the remesher.
    p.dynamic_remesh = false;
    p.amr.refine_every = 0;
    p.filter.field_filter_every = 0;
    p.filter.redistribute_every = 0;
    p.cleanup.enabled = true;

    // --checkpoint-every-steps 25. `setup()` writes step 0 unconditionally, so
    // 2000 steps every 25 gives the gold sets' own 81 files.
    p.checkpoint.every_steps = kCheckpointEvery;
    p.checkpoint.every_time = 0.0;
    p.checkpoint.directory = checkpoint_dir;
    p.checkpoint.prefix = "checkpoint";

    return p;
}

/// `FmmParams` for the probed evaluations — field for field the member's
/// `makeFmmParams` (`Beatnik_Test_Milestone0Fmm.cpp:1027-1032`), which sets
/// only `ncrit` and is level-independent, so it carries over verbatim.
///
/// `op_count_cap` is the one place this probe departs from the member's set,
/// and only when `argv[2]` is given: a negative value means "absent", which
/// leaves `FmmParams::m2l_op_count_cap` at its own default so that the
/// configuration measured is the member's exactly. The caller rejects a
/// negative `argv[2]` before reaching here, so the sentinel cannot be a user's
/// value.
Beatnik::FmmParams makeFmmParams( int op_count_cap = -1 )
{
    Beatnik::FmmParams f;
    f.ncrit = kNcrit;
    if ( op_count_cap >= 0 )
        f.m2l_op_count_cap = op_count_cap;
    return f;
}

//---------------------------------------------------------------------------//
/// Count of rows of an `(N,3)` field carrying a non-finite component, reduced
/// globally. The probe asserts nothing about the velocity's VALUE — it is not
/// comparing against anything — but a non-finite field means the trajectory
/// stopped meaning something, and the demand measured at such a state is not a
/// measurement of anything either.
template <class ExecSpace, class ViewType>
long long countNonFinite( MPI_Comm comm, const ViewType& v, int n )
{
    long long bad = 0;
    Kokkos::parallel_reduce(
        "beatnik_t6probe_nonfinite", Kokkos::RangePolicy<ExecSpace>( 0, n ),
        KOKKOS_LAMBDA( const int i, long long& nb ) {
            for ( int c = 0; c < 3; ++c )
                if ( !Kokkos::isfinite( v( i, c ) ) )
                {
                    ++nb;
                    return;
                }
        },
        bad );
    long long out = 0;
    MPI_Allreduce( &bad, &out, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return out;
}

//---------------------------------------------------------------------------//
/// What one state produced on THIS rank. Rank-local throughout except the four
/// fields Canopy reduces itself, which are spelled `global_`.
struct DemandPoint
{
    long long step = 0;
    int demand = -1;
    bool demand_saturated = false;
    int unique_ops = 0;
    int op_cap = 0;
    int occupied_depths = 0;
    int cells_at_max_depth = 0;
    int cache = 0;
    long long keys_built = 0;
    long long keys_built_delta = 0;
    long long global_m2l_pairs = 0;
    long long global_m2l_fallback = 0;
    // T8b's per-reason breakdown of the line above, in PAIRS and GLOBAL (the
    // adapter reduces all three), with -1 the "no profiling" sentinel and 0 a
    // legal count for each.
    long long fb_range_guard = -1;
    long long fb_count_cap = -1;
    long long fb_dropped = -1;
    double global_p2p_fraction = 0.0;
    long long global_particles = 0;
    double eval_wall = 0.0;
};

//---------------------------------------------------------------------------//
/// One row, built whole and issued in a single `printf` so that concurrent
/// ranks interleave between lines rather than inside one.
void printRow( int rank, int comm_size, const DemandPoint& p )
{
    std::printf(
        "[t6probe] row rank=%d/%d step=%lld demand=%d demand_saturated=%d "
        "unique_ops=%d op_cap=%d occupied_depths=%d cells_at_max_depth=%d "
        "cache=%d keys_built=%lld keys_built_delta=%lld "
        "global_m2l_pairs=%lld global_m2l_fallback=%lld "
        "fb_range_guard=%lld fb_count_cap=%lld fb_dropped=%lld "
        "global_p2p_frac=%.17g "
        "global_particles=%lld eval_wall=%.6f\n",
        rank, comm_size, p.step, p.demand, p.demand_saturated ? 1 : 0,
        p.unique_ops, p.op_cap, p.occupied_depths, p.cells_at_max_depth,
        p.cache, p.keys_built, p.keys_built_delta, p.global_m2l_pairs,
        p.global_m2l_fallback, p.fb_range_guard, p.fb_count_cap, p.fb_dropped,
        p.global_p2p_fraction, p.global_particles,
        p.eval_wall );
    std::fflush( stdout );
}

//---------------------------------------------------------------------------//
template <class ExecSpace, class MemSpace>
void runProbe( Beatnik::Test::Recorder& rec, int argc, char* argv[] )
{
    using solver_type = Beatnik::Solver<ExecSpace, MemSpace>;
    using geometry_type = Beatnik::MeshGeometry<ExecSpace, MemSpace>;
    using fmm_type = Beatnik::BRSolverFMM<ExecSpace, MemSpace>;
    using vector_view =
        Kokkos::View<Real* [3], Kokkos::Device<ExecSpace, MemSpace>>;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size = 1;
    int rank = 0;
    MPI_Comm_size( comm, &comm_size );
    MPI_Comm_rank( comm, &rank );

    if ( argc < 2 )
    {
        rec.fail( "usage: <icosphere-subdivisions> [m2l-op-count-cap]; see the "
                  "ARGUMENTS block in this file's header. Got " +
                  std::to_string( argc - 1 ) + " argument(s)." );
        return;
    }
    const int level = std::atoi( argv[1] );
    if ( level < 0 )
    {
        rec.fail( "the level must be a non-negative integer" );
        return;
    }

    // OPTIONAL `argv[2]`: the M2L operator column-count cap. `-1` is this
    // driver's "absent" sentinel and not a value a caller can supply -- a
    // negative cap is refused here rather than allowed to reach Canopy, whose
    // `DownwardSweep::set_m2l_op_count_cap()` throws from inside the
    // `Canopy::Solver` constructor. 0 IS legal and is passed through: it admits
    // no column and puts every pair on the overflow path, which is a
    // measurable configuration rather than a mistake.
    int op_count_cap = -1;
    if ( argc > 2 )
    {
        op_count_cap = std::atoi( argv[2] );
        if ( op_count_cap < 0 )
        {
            rec.fail( "the m2l op count cap must be a non-negative integer" );
            return;
        }
    }

    const long long want_vertices = verticesForLevel( level );
    const long long want_edges = edgesForLevel( level );
    const long long want_faces = facesForLevel( level );

    // Same resolution order, and the same three levels, as the regression
    // tests': the installed path runs from a read-only spack prefix, so "." is
    // not writable there and BEATNIK_TEST_SCRATCH is what the batch script
    // sets. MUST be a parallel filesystem -- MPI-IO, see the header.
    const char* scratch_env = std::getenv( "BEATNIK_TEST_SCRATCH" );
    if ( !scratch_env )
        scratch_env = std::getenv( "TMPDIR" );
    std::ostringstream dir;
    dir << ( scratch_env ? scratch_env : "." ) << "/keydemand_sub" << level
        << "_" << ExecSpace::name() << "_np" << comm_size;

    {
        std::ostringstream os;
        os << "execution space " << ExecSpace::name() << ", ranks " << comm_size
           << ", icosphere subdivisions " << level << " (" << want_vertices
           << " vertices), " << kSteps
           << " DIRECT steps with one FMM evaluation every "
           << kCheckpointEvery << " step(s), m2l op count cap "
           << ( op_count_cap >= 0 ? std::to_string( op_count_cap )
                                  : std::string( "default" ) )
           << ", checkpoints to " << dir.str();
        rec.note( os.str() );
    }

    //-----------------------------------------------------------------------//
    // The trajectory. Claim A's: direct, frozen connectivity, 2000 steps.
    //-----------------------------------------------------------------------//
    solver_type solver( comm, makeParams( level, dir.str() ) );
    solver.setup();

    auto& mesh = solver.mesh();
    const auto& state = solver.state();

    // The round trip, before anything evolves. A generator that produced the
    // wrong mesh makes every demand figure below a figure about a different
    // problem.
    BEATNIK_CHECK_EQ( rec, mesh.globalVertexCount(), want_vertices );
    BEATNIK_CHECK_EQ( rec, mesh.globalEdgeCount(), want_edges );
    BEATNIK_CHECK_EQ( rec, mesh.globalFaceCount(), want_faces );
    BEATNIK_CHECK_EQ( rec, mesh.globalEulerCharacteristic(), 2 );

    auto quadrature = Beatnik::createSourceQuadrature<ExecSpace, MemSpace>(
        Beatnik::SourceQuadrature::Vertex );

    // Constructed ONCE and reused at all 81 states, exactly as claim A does --
    // which is also what an FMM-driven run does, so the adapter's forward
    // distributor takes its reuse branch and Canopy's operator cache is
    // exercised the way a real run exercises it. A solver rebuilt per state
    // would report a cold cache at every state and `keys_built_delta` would
    // measure nothing.
    fmm_type fmm( comm, makeFmmParams( op_count_cap ) );

    //-----------------------------------------------------------------------//
    // R5 -- THE PARAMETER SET, echoed out of the solver and checked against the
    // member's compiled literals BEFORE any state is measured. A probe
    // measuring a different configuration than the member is worse than no
    // probe, and this is the only thing standing between the two files.
    //-----------------------------------------------------------------------//
    const Beatnik::FmmParams& live = fmm.farField().params();
    {
        std::ostringstream os;
        os.precision( 17 );
        os << "R5 parameter check against Beatnik_Test_Milestone0Fmm.cpp: "
              "ncrit "
           << live.ncrit << " vs " << kNcrit << ", order " << live.order
           << " vs " << kProductionOrder << ", basis "
           << Beatnik::toString( live.basis ) << " vs "
           << Beatnik::toString( kBasis ) << ", mac_theta "
           << static_cast<double>( live.mac_theta ) << " vs " << kMacTheta
           << ", max_depth " << live.max_depth << " vs " << kMaxDepth
           << ", near_softening_factor "
           << static_cast<double>( live.near_softening_factor ) << " vs "
           << kNearSofteningFactor;
        rec.note( os.str() );
    }
    BEATNIK_CHECK_EQ( rec, live.ncrit, kNcrit );
    BEATNIK_CHECK_EQ( rec, live.order, kProductionOrder );
    BEATNIK_CHECK_EQ( rec, static_cast<int>( live.basis ),
                      static_cast<int>( kBasis ) );
    BEATNIK_CHECK_CLOSE( rec, live.mac_theta, kMacTheta, 1.0e-15 );
    BEATNIK_CHECK_EQ( rec, live.max_depth, kMaxDepth );
    BEATNIK_CHECK_CLOSE( rec, live.near_softening_factor, kNearSofteningFactor,
                         0.0 );

    //-----------------------------------------------------------------------//
    // ONE STATE. The preconditions in the order `ZModelSolver`'s RHS
    // establishes them (`Beatnik_ZModelSolver.hpp` steps 0-2): one whole-tuple
    // halo exchange, geometry at the CURRENT positions, then the sheet vector.
    // Then ONE `computeInterfaceVelocity` on the FMM solver -- no direct
    // comparator, because this measures demand and not error.
    //
    // It cannot perturb the trajectory: it writes only into its own output view
    // and into `state`'s sheet vector, which the next RHS recomputes from
    // scratch at its first stage, and the adaptive dt on this configuration is
    // a function of edge lengths alone (`max_sheet_dt_product` is 0).
    //-----------------------------------------------------------------------//
    vector_view u_fmm( "beatnik_t6probe_u_fmm", mesh.ownedVertexCount() );
    long long keys_built_prev = 0;

    // The SAME `ZModelParams` the trajectory itself is running under, taken
    // from the same `makeParams` rather than rebuilt: the softening, the
    // blob mode and `br_sign` all enter the far-field evaluation, and a
    // second construction of them here would be a second source of truth for
    // what was measured.
    const Beatnik::ZModelParams zparams = makeParams( level, dir.str() ).zmodel;

    auto evaluate = [&]( long long step ) -> DemandPoint
    {
        const int n_owned = mesh.ownedVertexCount();
        mesh.haloExchange();
        geometry_type geometry;
        geometry.compute( mesh.positions(), mesh.totalVertexCount(),
                          mesh.faceVertices() );
        state.updateSheetVector( mesh, geometry );

        if ( static_cast<int>( u_fmm.extent( 0 ) ) != n_owned )
            Kokkos::realloc( u_fmm, n_owned );

        Kokkos::fence();
        const double t0 = MPI_Wtime();
        fmm.computeInterfaceVelocity( mesh, geometry, state, *quadrature,
                                      zparams, u_fmm );
        Kokkos::fence();
        const double t1 = MPI_Wtime();

        const auto& diag = fmm.farField().diagnostics();

        DemandPoint p;
        p.step = step;
        p.demand = diag.local_m2l_demanded_op_count;
        p.demand_saturated = diag.local_m2l_demand_saturated;
        p.unique_ops = diag.local_m2l_unique_op_count;
        p.op_cap = diag.local_m2l_op_cap;
        p.occupied_depths = diag.local_m2l_occupied_depths;
        p.cells_at_max_depth = diag.local_m2l_cells_at_max_depth;
        p.cache = diag.local_m2l_op_cache_size;
        p.keys_built = diag.local_m2l_op_keys_built;
        p.keys_built_delta = diag.local_m2l_op_keys_built - keys_built_prev;
        keys_built_prev = diag.local_m2l_op_keys_built;
        p.global_m2l_pairs = diag.global_m2l_pair_count;
        p.global_m2l_fallback = diag.global_m2l_fallback_pair_count;
        p.fb_range_guard = diag.global_m2l_fallback_pairs_range_guard;
        p.fb_count_cap = diag.global_m2l_fallback_pairs_count_cap;
        p.fb_dropped = diag.global_m2l_fallback_pairs_depth_dropped;
        p.global_p2p_fraction = diag.p2p_pair_fraction;
        p.global_particles = diag.global_particle_count;
        p.eval_wall = t1 - t0;

        // STRUCTURAL, not a measurement: the round trip neither dropped nor
        // duplicated a source, and the field is finite. Everything printed
        // beside them is a measurement and is asserted on by nothing.
        BEATNIK_CHECK_EQ( rec, p.global_particles, want_vertices );
        BEATNIK_CHECK_EQ( rec, countNonFinite<ExecSpace>( comm, u_fmm, n_owned ),
                          0 );

        // T8b'S SUM IDENTITY, STRUCTURAL and not a measurement: a breakdown
        // that does not account for every fallback pair can name the wrong
        // reason and still read like a number. Both figures are global --
        // Canopy reduces the fallback total itself and the adapter reduces the
        // breakdown into the same MPI_Allreduce -- so the identity is checked
        // at every rank against the same two sides.
        //
        // SKIPPED, NOT PASSED, UNDER THE SENTINEL (R7). In a `~profiling`
        // Canopy build all three counters are -1 and `-1 + -1 == fallback` is
        // a claim about nothing; asserting it would make a row of sentinels
        // read as a verified identity at the one state where fallback happens
        // to be -2. The skip is reported in the header and the trailer rather
        // than being silent.
        if ( p.fb_range_guard >= 0 && p.fb_count_cap >= 0 &&
             p.fb_dropped >= 0 )
        {
            BEATNIK_CHECK_EQ( rec, p.fb_range_guard + p.fb_count_cap,
                              p.global_m2l_fallback );
            // A refused pair whose target depth is out of range reaches
            // neither an operator column nor the fallback table, so its
            // contribution is never evaluated: a wrong velocity, not a slow
            // one.
            BEATNIK_CHECK_EQ( rec, p.fb_dropped, 0 );
        }
        return p;
    };

    //-----------------------------------------------------------------------//
    // STEP 0 IS A MEASURED STATE. `setup()` wrote its checkpoint
    // unconditionally and claim A evaluates 81 states, not 80. It is measured
    // FIRST so the header below can carry the real `bytes_per_key`, the real
    // effective cap and the availability of the demand counter, none of which
    // exist until an evaluation has run -- the diagnostics are
    // default-constructed before that.
    //-----------------------------------------------------------------------//
    std::vector<DemandPoint> series;
    series.push_back( evaluate( 0 ) );

    //-----------------------------------------------------------------------//
    // THE HEADER, and R7's loud line. `demand_available` is a compile-time
    // property of the Canopy build and so is the same on every rank, but it is
    // reduced both ways rather than read off rank 0: a disagreement would mean
    // the ranks are not running the same binary, which is worth finding out
    // here rather than inferring it from a ragged demand column later.
    //-----------------------------------------------------------------------//
    const int avail_local = ( series.front().demand != -1 ) ? 1 : 0;
    int avail_min = avail_local;
    int avail_max = avail_local;
    MPI_Allreduce( &avail_local, &avail_min, 1, MPI_INT, MPI_MIN, comm );
    MPI_Allreduce( &avail_local, &avail_max, 1, MPI_INT, MPI_MAX, comm );
    BEATNIK_CHECK_EQ( rec, avail_min, avail_max );

    // T8b's breakdown has its OWN availability, read separately from the
    // demand counter's even though both are gated on the same
    // CANOPY_ENABLE_PROFILING. They are separate measurements, and a build in
    // which one is live and the other is not is a build whose instrumentation
    // has drifted -- worth seeing as a disagreement here rather than as a
    // column of -1 beside a column of counts. The adapter reduces the
    // breakdown, so a disagreement between ranks is impossible by
    // construction and the value is read off this rank.
    const int fb_avail =
        ( series.front().fb_range_guard >= 0 &&
          series.front().fb_count_cap >= 0 &&
          series.front().fb_dropped >= 0 )
            ? 1
            : 0;
    BEATNIK_CHECK_EQ( rec, fb_avail, avail_min );

    if ( rank == 0 )
    {
        const DemandPoint& p0 = series.front();
        std::printf(
            "[t6probe] header level=%d vertices=%lld ranks=%d space=%s "
            "steps=%d every=%d ncrit=%d order=%d basis=%s mac_theta=%.17g "
            "max_depth=%d near_soft_factor=%.17g bytes_per_key=%zu "
            "byte_budget=%zu op_count_cap=%d op_cap=%d "
            "demand_available=%d fallback_breakdown_available=%d\n",
            level, want_vertices, comm_size, ExecSpace::name(), kSteps,
            kCheckpointEvery, live.ncrit, live.order,
            Beatnik::toString( live.basis ),
            static_cast<double>( live.mac_theta ), live.max_depth,
            static_cast<double>( live.near_softening_factor ),
            fmm.farField().diagnostics().local_m2l_bytes_per_key,
            live.m2l_op_table_byte_budget, live.m2l_op_count_cap, p0.op_cap,
            avail_min, fb_avail );
        if ( avail_min == 0 )
        {
            // R7. `-1` is "this Canopy build carries no profiling", and `0`
            // would be "the tree wants no keys" -- a legal measurement. The two
            // must never be read for each other, so a run that cannot measure
            // says so here rather than letting a reader take the column for a
            // series.
            std::printf(
                "[t6probe] *** DEMAND UNAVAILABLE: this Canopy build has "
                "CANOPY_ENABLE_PROFILING undefined (a `canopy ~profiling` "
                "spec), so local_m2l_demanded_op_count is the -1 SENTINEL and "
                "local_m2l_demand_saturated is meaningless false. THE demand "
                "COLUMN BELOW IS NOT A MEASUREMENT and -1 is NOT zero demand. "
                "Rebuild against `canopy +profiling` to measure. The "
                "occupied_depths and cells_at_max_depth columns ARE live in "
                "this build -- m2l_cells_at_depth() is ungated -- so a -1 in "
                "THOSE would be this probe's own bug. ***\n" );
        }
        if ( fb_avail == 0 )
        {
            // R7 again, for T8b's three counters. The sum identity is SKIPPED
            // rather than evaluated in this build, so the per-reason columns
            // below carry no information at all and the absence of a failed
            // identity check is not evidence that one holds.
            std::printf(
                "[t8bprobe] *** FALLBACK BREAKDOWN UNAVAILABLE: the "
                "fb_range_guard, fb_count_cap and fb_dropped columns below "
                "are the -1 SENTINEL and NOT zero counts. Zero range-guard "
                "refusals is the NORMAL reading, so a 0 here would be a "
                "measurement this build did not make. THE SUM IDENTITY "
                "fb_range_guard + fb_count_cap == global_m2l_fallback IS "
                "SKIPPED, not passed -- its absence from the check tally is "
                "the point. Rebuild against `canopy +profiling` to measure. "
                "***\n" );
        }
        std::fflush( stdout );
    }
    MPI_Barrier( comm );
    printRow( rank, comm_size, series.front() );

    //-----------------------------------------------------------------------//
    // THE TRAJECTORY. Driven one step at a time rather than through `solve()`,
    // so the evaluation happens at every checkpointed step. `advanceOneStep` is
    // collective and every rank must call it the same number of times.
    //-----------------------------------------------------------------------//
    const double t_start = MPI_Wtime();
    bool stopped = false;
    for ( int step = 1; step <= kSteps; ++step )
    {
        if ( !solver.advanceOneStep() )
        {
            // A STOP IS A REPORTED STOP STEP, NEVER A SHORTER SERIES: a demand
            // series that ended early and did not say so would read as a peak
            // that had been reached.
            std::ostringstream os;
            os << "run STOPPED EARLY at step " << step << " of " << kSteps
               << " (non-finite state); solver step " << solver.step()
               << ", time " << solver.time()
               << ". The demand series below this step does not exist.";
            rec.fail( os.str() );
            stopped = true;
            break;
        }

        // Cheap, integer, and reduced inside Tessera: safe every step.
        if ( mesh.globalVertexCount() != want_vertices ||
             mesh.globalFaceCount() != want_faces )
        {
            std::ostringstream os;
            os << "ENTITY COUNTS CHANGED at step " << step << ": vertices "
               << mesh.globalVertexCount() << " (expected " << want_vertices
               << "), faces " << mesh.globalFaceCount() << " (expected "
               << want_faces
               << "). Adaptivity leaked into the frozen-mesh configuration.";
            rec.fail( os.str() );
            stopped = true;
            break;
        }

        if ( step % kCheckpointEvery != 0 )
            continue;

        const DemandPoint p = evaluate( step );
        series.push_back( p );
        printRow( rank, comm_size, p );
    }
    Kokkos::fence();
    const double t_total = MPI_Wtime() - t_start;

    solver.finalize();

    // The step budget must have been reached. `rec.fail` above already
    // recorded the reason; this is the check that makes "ran fewer states"
    // visible in the tally rather than only in the prose.
    BEATNIK_CHECK_TRUE( rec, !stopped );

    //-----------------------------------------------------------------------//
    // THE TRAILER, per rank. The demand series has its OWN peak at its OWN
    // step, which R3 says is not where the fallback peaks and not where claim
    // A's error peaks -- three different steps at level 4. So it is located
    // here rather than assumed, and the first exceedance of the cap is derived
    // from the demand and cap columns and NEVER from the `[Canopy]` warning,
    // which T0 measured as absent from both np4 launches despite non-zero
    // fallback at 71 states.
    //-----------------------------------------------------------------------//
    long long first_exceed = -1;
    long long peak_demand = -1;
    long long peak_demand_step = -1;
    long long peak_delta = -1;
    long long peak_delta_step = -1;
    double eval_wall_total = 0.0;
    // T8b. The per-reason breakdown summed over the series, plus the state
    // counts, because the question T8b answers is "which reason accounts for
    // the fallback" and the per-state rows above are 81 lines to add up by
    // hand. `fb_states` is the count of states with non-zero fallback --
    // T8 measured 71 of 81 at both rank counts and both caps, and that count
    // is the observable the level-4 member's `p.m2l_fallback == 0` assertion
    // actually fails on.
    long long fb_total = 0;
    long long fb_range_total = 0;
    long long fb_cap_total = 0;
    long long fb_dropped_total = 0;
    long long fb_states = 0;
    long long fb_range_states = 0;
    long long fb_cap_states = 0;
    long long first_fb_step = -1;
    for ( const DemandPoint& p : series )
    {
        eval_wall_total += p.eval_wall;
        fb_total += p.global_m2l_fallback;
        if ( p.global_m2l_fallback > 0 )
        {
            ++fb_states;
            if ( first_fb_step < 0 )
                first_fb_step = p.step;
        }
        // The totals stay at 0 under the sentinel rather than accumulating
        // -1 per state, and `fallback_breakdown_available=0` in the header is
        // what says they are not measurements (R7).
        if ( p.fb_range_guard >= 0 )
        {
            fb_range_total += p.fb_range_guard;
            fb_cap_total += p.fb_count_cap;
            fb_dropped_total += p.fb_dropped;
            if ( p.fb_range_guard > 0 )
                ++fb_range_states;
            if ( p.fb_count_cap > 0 )
                ++fb_cap_states;
        }
        if ( p.keys_built_delta > peak_delta )
        {
            peak_delta = p.keys_built_delta;
            peak_delta_step = p.step;
        }
        // The demand columns are only readable at all when the counter is
        // compiled in; under the -1 sentinel every one of these stays -1
        // rather than reporting "never exceeded", which is a measurement this
        // run did not make (R7).
        if ( p.demand < 0 )
            continue;
        if ( p.demand > peak_demand )
        {
            peak_demand = p.demand;
            peak_demand_step = p.step;
        }
        if ( first_exceed < 0 && p.demand > p.op_cap )
            first_exceed = p.step;
    }

    std::printf( "[t6probe] trailer rank=%d/%d demand_available=%d "
                 "first_exceed_step=%lld peak_demand=%lld "
                 "peak_demand_step=%lld peak_keys_built_delta=%lld "
                 "peak_keys_built_delta_step=%lld states=%zu "
                 "eval_wall_total=%.6f\n",
                 rank, comm_size, avail_min, first_exceed, peak_demand,
                 peak_demand_step, peak_delta, peak_delta_step, series.size(),
                 eval_wall_total );
    std::fflush( stdout );

    // T8b's answer in one line per rank. `global_*` here are Canopy's own
    // reductions and the adapter's, so every rank prints the same figures;
    // they are printed per rank anyway, as the trailer above is, so a ragged
    // column is visible rather than inferred.
    std::printf( "[t8bprobe] trailer rank=%d/%d breakdown_available=%d "
                 "states=%zu fb_states=%lld first_fb_step=%lld "
                 "fb_total=%lld fb_range_guard_total=%lld "
                 "fb_count_cap_total=%lld fb_dropped_total=%lld "
                 "fb_range_guard_states=%lld fb_count_cap_states=%lld\n",
                 rank, comm_size, fb_avail, series.size(), fb_states,
                 first_fb_step, fb_total, fb_range_total, fb_cap_total,
                 fb_dropped_total, fb_range_states, fb_cap_states );
    std::fflush( stdout );

    MPI_Barrier( comm );
    if ( rank == 0 )
    {
        std::printf( "[t6probe] total level=%d ranks=%d space=%s states=%zu "
                     "trajectory_wall=%.6f\n",
                     level, comm_size, ExecSpace::name(), series.size(),
                     t_total );
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
        Beatnik::Test::Recorder rec( "Beatnik_Probe_FmmKeyDemand" );
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

    // ONE VERDICT ACROSS THE RANKS, as every standalone test here does: each
    // rank printed its own tally, MPI_MAX makes any rank's failure the job's.
    int global_rc = rc;
    MPI_Allreduce( &rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return global_rc;
}
