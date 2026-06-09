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

/* tstFmmVsExact.hpp — verify that FmmBRSolver and ExactBRSolver agree
 * on the Birkhoff-Rott interface velocity for small, well-behaved
 * inputs. Compiled-in only when Beatnik_ENABLE_CANOPY=ON.
 *
 * Two tests:
 *   1. BRDirectComparison — drives the BR solvers directly via
 *      computeInterfaceVelocity() with a manually-set sinusoidal z
 *      and a non-zero omega. Smallest unit; tightest signal.
 *   2. OneRK3StepComparison — drives the full Beatnik::Solver
 *      (ZModel + TimeIntegrator + BR) one step, comparing the
 *      post-step position field. Uses a custom MeshInitFunc that
 *      sets non-trivial initial vorticity so the BR contribution
 *      is non-zero on the first call.
 */

#include <Beatnik_Config.hpp>

#ifdef BEATNIK_ENABLE_CANOPY

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <BoundaryCondition.hpp>
#include <CreateBRSolver.hpp>
#include <ExactBRSolver.hpp>
#include <FmmBRSolver.hpp>
#include <ProblemManager.hpp>
#include <Solver.hpp>
#include <SurfaceMesh.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

namespace BeatnikTest
{

namespace
{

constexpr double kRelTolerance     = 1.0e-3;  // Plan target at P=6
/* Floor used in the relative-error metric: max(|a|, |b|, kAbsFloor).
 * For nodes where both values are below this floor, the metric
 * degrades to diff / kAbsFloor, which acts as a combined
 * absolute/relative tolerance: a node passes if either
 *   diff < kRelTolerance * max(|a|, |b|)   [relative]
 *   diff < kRelTolerance * kAbsFloor       [absolute, = 1e-8]
 * The OneRK3StepComparison test has many post-step z nodes where
 * the value is naturally near zero (e.g. the z1/z2 components
 * along the boundary), and the BR contribution to those nodes is
 * FP-rounding-level, not physically meaningful — without this
 * floor, those would dominate the relative metric. */
constexpr double kAbsFloor         = 1.0e-5;
constexpr int    kMeshNodesPerSide = 32;      // 32x32 owned nodes total
constexpr double kBoxHalfSide      = 1.0;     // Domain [-1, 1]^2
constexpr double kAmplitude        = 0.05;    // Sinusoidal z amplitude
constexpr double kOmegaScale       = 0.10;    // Non-zero vorticity scale

/* Build the default Beatnik::Params used by both solvers in the
 * BR-direct test. solver_order isn't read by ExactBRSolver or
 * FmmBRSolver directly so it can be left at its default. */
Beatnik::Params makeParams( Beatnik::BRSolverType br_kind,
                            const std::array<double, 6>& bbox )
{
    Beatnik::Params p{};
    p.period               = 1.234;     // RNG seed; unused at the BR layer
    p.global_bounding_box  = bbox;
    p.periodic             = {false, false};
    p.solver_order         = 1;
    p.br_solver            = br_kind;
    p.cutoff_distance      = 0.1;
    p.heffte_configuration = 6;
    // FMM tunables: Params already carries defaults (see Solver.hpp).
    return p;
}

/* Free-boundary BC: edges treated as a one-sided boundary. */
Beatnik::BoundaryCondition makeFreeBC()
{
    Beatnik::BoundaryCondition bc;
    bc.boundary_type[0] = Beatnik::BoundaryType::FREE;
    bc.boundary_type[1] = Beatnik::BoundaryType::FREE;
    bc.boundary_type[2] = Beatnik::BoundaryType::FREE;
    bc.boundary_type[3] = Beatnik::BoundaryType::FREE;
    return bc;
}

/* A no-op init functor for ProblemManager — leaves z and omega at
 * their zero-initialized values. We overwrite them with a known
 * pattern after the ProblemManager is constructed. */
struct ZeroInitFunc
{
    template <class RandPool>
    KOKKOS_INLINE_FUNCTION bool operator()( Cabana::Grid::Node,
                                            Beatnik::Field::Position,
                                            RandPool,
                                            const int[2],
                                            const double[2],
                                            double& z1, double& z2, double& z3 ) const
    {
        z1 = 0.0; z2 = 0.0; z3 = 0.0;
        return true;
    }

    KOKKOS_INLINE_FUNCTION bool operator()( Cabana::Grid::Node,
                                            Beatnik::Field::Vorticity,
                                            const int[2],
                                            const double[2],
                                            double& w1, double& w2 ) const
    {
        w1 = 0.0; w2 = 0.0;
        return true;
    }
};

/* Init functor for the one-RK3-step test. Uses real physical
 * coordinates from the local mesh to produce a smooth interface and
 * a non-trivial vorticity pattern that's bounded and free of jumps
 * at the boundary. */
struct CurvedNonZeroVorticityInitFunc
{
    template <class RandPool>
    KOKKOS_INLINE_FUNCTION bool operator()( Cabana::Grid::Node,
                                            Beatnik::Field::Position,
                                            RandPool,
                                            const int[2],
                                            const double coord[2],
                                            double& z1, double& z2, double& z3 ) const
    {
        z1 = coord[0];
        z2 = coord[1];
        // Mild curvature so that the BR integral has a real contribution.
        z3 = kAmplitude
             * std::cos( M_PI * coord[0] / kBoxHalfSide )
             * std::cos( M_PI * coord[1] / kBoxHalfSide );
        return true;
    }

    KOKKOS_INLINE_FUNCTION bool operator()( Cabana::Grid::Node,
                                            Beatnik::Field::Vorticity,
                                            const int[2],
                                            const double coord[2],
                                            double& w1, double& w2 ) const
    {
        // Smooth, divergence-free vorticity sheet; zero at the
        // boundaries so Simpson weighting near the edge doesn't
        // amplify endpoint artifacts.
        w1 = kOmegaScale * std::sin( M_PI * coord[0] / kBoxHalfSide );
        w2 = kOmegaScale * std::sin( M_PI * coord[1] / kBoxHalfSide );
        return true;
    }
};

/* Compute max_i,j,d  |a(i,j,d) - b(i,j,d)| / max(|a(i,j,d)|, |b(i,j,d)|, floor)
 * over the owned index space of a Beatnik node_view. Returns the
 * pair (max_rel_diff, max_abs_diff). */
template <class ViewType>
std::array<double, 2>
maxRelDiff( ViewType a, ViewType b,
            const Cabana::Grid::IndexSpace<2>& owned )
{
    using exec_space = typename ViewType::execution_space;
    double max_rel = 0.0;
    double max_abs = 0.0;
    Kokkos::parallel_reduce( "tstFmmVsExact::maxRelDiff",
        Cabana::Grid::createExecutionPolicy( owned, exec_space() ),
        KOKKOS_LAMBDA( const int i, const int j,
                       double& lrel, double& labs )
    {
        for ( int d = 0; d < 3; ++d )
        {
            const double av = a( i, j, d );
            const double bv = b( i, j, d );
            const double diff = Kokkos::fabs( av - bv );
            const double denom = Kokkos::fmax(
                Kokkos::fmax( Kokkos::fabs( av ), Kokkos::fabs( bv ) ),
                kAbsFloor );
            const double rel = diff / denom;
            if ( rel > lrel ) lrel = rel;
            if ( diff > labs ) labs = diff;
        }
    },
    Kokkos::Max<double>( max_rel ),
    Kokkos::Max<double>( max_abs ) );

    // MPI reduce across ranks
    double max_rel_all = 0.0, max_abs_all = 0.0;
    MPI_Allreduce( &max_rel, &max_rel_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &max_abs, &max_abs_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return { max_rel_all, max_abs_all };
}

} // anonymous namespace

/* -----------------------------------------------------------------
 * Test 1: BR solvers driven directly with a known z and non-zero omega.
 * ----------------------------------------------------------------- */
TEST( TEST_CATEGORY, BRDirectComparison )
{
    using ExecSpace = TEST_EXECSPACE;
    using MemSpace  = TEST_MEMSPACE;
    using SM        = Beatnik::SurfaceMesh<ExecSpace, MemSpace>;
    using PM        = Beatnik::ProblemManager<ExecSpace, MemSpace>;
    using Exact     = Beatnik::ExactBRSolver<ExecSpace, MemSpace, Beatnik::Params>;
    using Fmm       = Beatnik::FmmBRSolver<ExecSpace, MemSpace, Beatnik::Params>;
    using node_view = Kokkos::View<double***, MemSpace>;

    const std::array<double, 6> bbox{ -kBoxHalfSide, -kBoxHalfSide, -kBoxHalfSide,
                                       kBoxHalfSide,  kBoxHalfSide,  kBoxHalfSide };
    const std::array<int, 2>  nodes{ kMeshNodesPerSide, kMeshNodesPerSide };
    const std::array<bool, 2> periodic{ false, false };
    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    auto bc     = makeFreeBC();
    auto params = makeParams( Beatnik::BR_EXACT, bbox );  // br_kind only for params

    SM mesh( bbox, nodes, periodic, partitioner, 2, MPI_COMM_WORLD );
    PM pm( mesh, bc, params.period, ZeroInitFunc{} );

    // Write a known z (sinusoidal) and omega (non-zero pattern) into pm.
    const int num_cells_x = nodes[0] - 1;
    const int num_cells_y = nodes[1] - 1;
    const double dx = ( bbox[3] - bbox[0] ) / num_cells_x;
    const double dy = ( bbox[4] - bbox[1] ) / num_cells_y;
    const double box_half = kBoxHalfSide;
    const double amp      = kAmplitude;
    const double w_scale  = kOmegaScale;

    auto local_grid  = mesh.localGrid();
    auto local_mesh  = Cabana::Grid::createLocalMesh<MemSpace>( *local_grid );
    auto own_nodes   = local_grid->indexSpace( Cabana::Grid::Own(),
                                               Cabana::Grid::Node(),
                                               Cabana::Grid::Local() );
    auto z = pm.get( Cabana::Grid::Node(), Beatnik::Field::Position() );
    auto w = pm.get( Cabana::Grid::Node(), Beatnik::Field::Vorticity() );

    Kokkos::parallel_for( "tstFmmVsExact::setZOmega",
        Cabana::Grid::createExecutionPolicy( own_nodes, ExecSpace() ),
        KOKKOS_LAMBDA( const int i, const int j )
    {
        const int idx[2] = { i, j };
        double coord[2];
        local_mesh.coordinates( Cabana::Grid::Node(), idx, coord );
        z( i, j, 0 ) = coord[0];
        z( i, j, 1 ) = coord[1];
        z( i, j, 2 ) = amp
                       * Kokkos::cos( M_PI * coord[0] / box_half )
                       * Kokkos::cos( M_PI * coord[1] / box_half );
        w( i, j, 0 ) = w_scale * Kokkos::sin( M_PI * coord[0] / box_half );
        w( i, j, 1 ) = w_scale * Kokkos::sin( M_PI * coord[1] / box_half );
    });
    pm.gather();

    // 3-component omega view (FMM solver expects this shape). We
    // build it from the 2-component vorticity by zeroing the third.
    node_view omega3( "omega3", w.extent( 0 ), w.extent( 1 ), 3 );
    Kokkos::parallel_for( "tstFmmVsExact::buildOmega3",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            { 0, 0 }, { (int)w.extent( 0 ), (int)w.extent( 1 ) } ),
        KOKKOS_LAMBDA( const int i, const int j )
    {
        omega3( i, j, 0 ) = w( i, j, 0 );
        omega3( i, j, 1 ) = w( i, j, 1 );
        omega3( i, j, 2 ) = 0.0;
    });

    // Two zdot views
    node_view zdot_exact( "zdot_exact", z.extent( 0 ), z.extent( 1 ), 3 );
    node_view zdot_fmm  ( "zdot_fmm",   z.extent( 0 ), z.extent( 1 ), 3 );

    const double epsilon = 0.25 * std::sqrt( dx * dy );  // ZModel default

    // Construct each BR solver. Note: FmmBRSolver MPI_Aborts on
    // periodic BC at construction; bc here is free.
    Exact exact_solver( pm, bc, epsilon, dx, dy, params );
    Fmm   fmm_solver  ( pm, bc, epsilon, dx, dy, params );

    exact_solver.computeInterfaceVelocity( zdot_exact, z, omega3 );
    fmm_solver  .computeInterfaceVelocity( zdot_fmm,   z, omega3 );

    Kokkos::fence();
    auto [max_rel, max_abs] = maxRelDiff( zdot_exact, zdot_fmm, own_nodes );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        std::cout << "[BRDirectComparison] max_rel_diff=" << max_rel
                  << " max_abs_diff=" << max_abs
                  << " (rel tolerance=" << kRelTolerance << ")\n";
    }

    EXPECT_LT( max_rel, kRelTolerance )
        << "FmmBRSolver and ExactBRSolver zdot disagree above tolerance";
}

/* -----------------------------------------------------------------
 * Test 2: full Beatnik::Solver, one step each, compare post-step z.
 *
 * Uses CurvedNonZeroVorticityInitFunc so the first RK3 BR call sees
 * non-trivial omega. Instantiates the templated Beatnik::Solver
 * directly (rather than going through createSolver, which returns a
 * shared_ptr<SolverBase>) so we can call Solver::position() to
 * extract the post-step z view for comparison.
 * ----------------------------------------------------------------- */
TEST( TEST_CATEGORY, OneRK3StepComparison )
{
    using ExecSpace = TEST_EXECSPACE;
    using MemSpace  = TEST_MEMSPACE;
    using ModelOrderTag = Beatnik::Order::High;
    using TypedSolver = Beatnik::Solver<ExecSpace, MemSpace, ModelOrderTag>;

    const std::array<double, 6> bbox{ -kBoxHalfSide, -kBoxHalfSide, -kBoxHalfSide,
                                       kBoxHalfSide,  kBoxHalfSide,  kBoxHalfSide };
    const std::array<int, 2> nodes{ kMeshNodesPerSide, kMeshNodesPerSide };

    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    auto bc = makeFreeBC();
    const double atwood  = 0.5;
    const double gravity = 25.0;
    const double mu      = 1.0;
    const double epsilon = 0.25;
    const double dt      = 0.01;

    auto base_params = makeParams( Beatnik::BR_EXACT, bbox );
    base_params.solver_order = 2;  // high-order path goes through BR

    auto params_exact = base_params; params_exact.br_solver = Beatnik::BR_EXACT;
    auto params_fmm   = base_params; params_fmm.br_solver   = Beatnik::BR_FMM;

    CurvedNonZeroVorticityInitFunc init;

    TypedSolver solver_exact( MPI_COMM_WORLD, nodes, partitioner, atwood, gravity,
                              init, bc, mu, epsilon, dt, params_exact );
    TypedSolver solver_fmm  ( MPI_COMM_WORLD, nodes, partitioner, atwood, gravity,
                              init, bc, mu, epsilon, dt, params_fmm );

    solver_exact.setup();
    solver_fmm  .setup();
    solver_exact.step();
    solver_fmm  .step();

    // Compare post-step position fields over the owned index space.
    auto local_grid = solver_exact.problemManager().mesh().localGrid();
    auto own_nodes  = local_grid->indexSpace( Cabana::Grid::Own(),
                                              Cabana::Grid::Node(),
                                              Cabana::Grid::Local() );
    auto z_exact = solver_exact.position();
    auto z_fmm   = solver_fmm  .position();

    Kokkos::fence();
    auto [max_rel, max_abs] = maxRelDiff( z_exact, z_fmm, own_nodes );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        std::cout << "[OneRK3StepComparison] max_rel_diff=" << max_rel
                  << " max_abs_diff=" << max_abs
                  << " (rel tolerance=" << kRelTolerance << ")\n";
    }

    EXPECT_LT( max_rel, kRelTolerance )
        << "Post-step z disagrees between -S exact and -S fmm above tolerance";
}

} // namespace BeatnikTest

#endif // BEATNIK_ENABLE_CANOPY
