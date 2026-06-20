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
 * @file FmmBRSolver.hpp
 * @author Jason Stewart <jastewart@unm.edu>
 */

#ifndef BEATNIK_FMMBRSOLVER_HPP
#define BEATNIK_FMMBRSOLVER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Canopy_Solver.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>

#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>
#include <BRSolverBase.hpp>
#include <Profiling.hpp>

namespace Beatnik
{

static constexpr int P_ORDER = 10;
static constexpr int N_COMPS = 3;

/* AoSoA field indices used by FmmBRSolver. The tuple layout is
 *   [0] position  double[3]
 *   [1] charge    double[3]   (Simpson-weighted omega)
 *   [2] u_out     double[3]   (cross-product of Canopy gradients)
 *   [3] tag       int[3]      (origin_rank, local_i, local_j)
 */
namespace FmmField
{
    static constexpr int Position = 0;
    static constexpr int Charge   = 1;
    static constexpr int UOut     = 2;
    static constexpr int Tag      = 3;
}

/**
 * The ExactBRSolver Class
 * @class ExactBRSolver
 * @brief Directly solves the Birkhoff-Rott integral using brute-force 
 * all-pairs calculation
 **/
template <class ExecutionSpace, class MemorySpace, class Params>
class FmmBRSolver : public BRSolverBase<ExecutionSpace, MemorySpace, Params>
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;

    using Node = Cabana::Grid::Node;
    using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    using node_array = typename pm_type::node_array;
    using node_view = Kokkos::View<double***, memory_space>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    using canopy_solver_type =
        Canopy::Solver<MemorySpace, ExecutionSpace, double, P_ORDER, N_COMPS>;

    using particle_member_types =
        Cabana::MemberTypes<double[3], double[3], double[3], int[3]>;
    using aosoa_type =
        Cabana::AoSoA<particle_member_types, MemorySpace>;

    FmmBRSolver( const pm_type &pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy,
                   const Params params)
        : _pm( pm )
        , _bc( bc )
        , _epsilon( epsilon )
        , _dx( dx )
        , _dy( dy )
        , _params( params )
        , _local_L2G( *_pm.mesh().localGrid() )
        , _comm( _pm.mesh().localGrid()->globalGrid().comm() )
        , _canopy( _pm.mesh().localGrid()->globalGrid().comm(),
                   makeCanopyConfig( params, epsilon ) )
    {
        MPI_Comm_size(_comm, &_num_procs);
        MPI_Comm_rank(_comm, &_rank);

        if ( _bc.isPeriodicBoundary({0, 1}) || _bc.isPeriodicBoundary({1, 1}) )
        {
            if ( _rank == 0 )
            {
                std::cerr << "FmmBRSolver: periodic boundary conditions are "
                             "not supported in v1. Use -S exact or -S cutoff, "
                             "or run with non-periodic boundaries.\n";
            }
            MPI_Abort( _comm, 1 );
        }
    }

#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
    ~FmmBRSolver()
    {
        if ( _rank == 0 )
        {
            const long total_am = _am_migrate + _am_rebalance + _am_rebuild;
            std::cout << "[FmmBRSolver] action histogram across "
                      << _call_count << " computeInterfaceVelocity calls: "
                      << "setup=1 "
                      << "Migrate=" << _am_migrate << " "
                      << "Rebalance=" << _am_rebalance << " "
                      << "Rebuild=" << _am_rebuild
                      << " (auto_maintain calls=" << total_am << ")\n";
        }
        // Level-2 sub-phase breakdown (collective; no-op below level 2).
        // Accumulated across every computeInterfaceVelocity call of the run.
        BEATNIK_PRINT_FMM_TIMERS( _comm );
    }
#endif

    /* Pack the grid-ordered owned nodes into an AoSoA tuple per node:
     *   position = z(i, j, :)
     *   charge   = simpson(global_i, N) * simpson(global_j, N) * omega(i, j, :)
     *   u_out    = 0 (filled later from Canopy gradients)
     *   tag      = (origin_rank, local_i, local_j) — travels with the
     *              particle through any Canopy migration so we can
     *              route the FMM result back to its grid origin.
     */
    void packGridParticles( aosoa_type& particles, node_view z, node_view o ) const
    {
        auto local_grid  = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );

        const long imin = local_space.min( 0 );
        const long jmin = local_space.min( 1 );
        const long ni   = local_space.max( 0 ) - imin;
        const long nj   = local_space.max( 1 ) - jmin;
        const long num_local = ni * nj;

        particles.resize( static_cast<std::size_t>( num_local ) );

        auto pos    = Cabana::slice<FmmField::Position>( particles );
        auto charge = Cabana::slice<FmmField::Charge>( particles );
        auto u_out  = Cabana::slice<FmmField::UOut>( particles );
        auto tag    = Cabana::slice<FmmField::Tag>( particles );

        const int mesh_size = _pm.mesh().get_surface_mesh_size();
        const int rank      = _rank;
        l2g_type  L2G       = _local_L2G;

        Kokkos::parallel_for( "FmmBRSolver::packGridParticles",
            Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
            KOKKOS_LAMBDA( const long p )
        {
            const int li = static_cast<int>( imin + p / nj );
            const int lj = static_cast<int>( jmin + p % nj );

            int local_idx[2]  = { li, lj };
            int global_idx[2] = { 0, 0 };
            L2G( local_idx, global_idx );

            const double w = Operators::simpsonWeight( global_idx[0], mesh_size )
                           * Operators::simpsonWeight( global_idx[1], mesh_size );

            for ( int d = 0; d < 3; ++d )
            {
                pos( p, d )    = z( li, lj, d );
                charge( p, d ) = w * o( li, lj, d );
                u_out( p, d )  = 0.0;
            }
            tag( p, 0 ) = rank;
            tag( p, 1 ) = li;
            tag( p, 2 ) = lj;
        });
    }

    /* Build a forward Distributor that routes grid-order particles
     * (one per owned mesh node, currently sitting in _grid_particles)
     * to wherever the matching Canopy-order tuple currently lives in
     * _canopy_particles.
     *
     * Algorithm (the "tag-reverse handshake"):
     *   1. On every canopy rank, for each _canopy_particles[p] with
     *      tag (R, li, lj), pack a small "claim" tuple (li, lj, my_rank).
     *   2. Build a Distributor keyed on tag.origin_rank and migrate
     *      the claims back to their origin grid rank.
     *   3. On the receiving grid rank, fill a 2D map
     *        dest_rank_map(li - imin, lj - jmin) = canopy_rank_that_owns
     *      from the received claims.
     *   4. Walk the local owned index space in the same row-major
     *      order that packGridParticles uses, and emit one
     *      forward_dest_ranks entry per grid tuple from the map.
     *   5. Construct the forward Distributor from forward_dest_ranks.
     *
     * Cost: O(num_canopy + num_grid) compute + 1 Cabana::Distributor
     * build + 1 migrate of small (3-int) claims. Rebuilt every call;
     * caching across `Migrate`-action steps is a follow-up.
     */
    Cabana::Distributor<MemorySpace> buildForwardDistributor() const
    {
        using claim_member_types = Cabana::MemberTypes<int[3]>;
        using claim_aosoa = Cabana::AoSoA<claim_member_types, MemorySpace>;

        const int num_canopy = static_cast<int>( _canopy_particles.size() );

        // Pack claims and their destination ranks (= tag.origin_rank).
        claim_aosoa claims_canopy( "FmmBRSolver_claims_canopy", num_canopy );
        Kokkos::View<int*, MemorySpace> claim_dests(
            Kokkos::ViewAllocateWithoutInitializing( "FmmBRSolver_claim_dests" ),
            num_canopy );
        {
            auto tag = Cabana::slice<FmmField::Tag>( _canopy_particles );
            auto claim = Cabana::slice<0>( claims_canopy );
            const int my_rank = _rank;
            Kokkos::parallel_for( "FmmBRSolver::packClaims",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_canopy ),
                KOKKOS_LAMBDA( const int p )
            {
                claim( p, 0 )       = tag( p, 1 );  // li
                claim( p, 1 )       = tag( p, 2 );  // lj
                claim( p, 2 )       = my_rank;
                claim_dests( p )    = tag( p, 0 );  // grid rank that owns (li, lj)
            });
        }
        Kokkos::fence();

        // Send claims back to the grid ranks that own each (li, lj).
        Cabana::Distributor<MemorySpace> claim_dist( _comm, claim_dests );
        claim_aosoa claims_grid( "FmmBRSolver_claims_grid",
                                 claim_dist.totalNumImport() );
        Cabana::migrate( claim_dist, claims_canopy, claims_grid );

        // Fill (li - imin, lj - jmin) -> canopy_rank from received claims.
        auto local_space = _pm.mesh().localGrid()->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
        const long imin = local_space.min( 0 );
        const long jmin = local_space.min( 1 );
        const long ni   = local_space.max( 0 ) - imin;
        const long nj   = local_space.max( 1 ) - jmin;

        Kokkos::View<int**, MemorySpace> dest_rank_map(
            "FmmBRSolver_dest_rank_map", ni, nj );
        Kokkos::deep_copy( dest_rank_map, -1 );
        {
            auto claim_recv = Cabana::slice<0>( claims_grid );
            const int num_claims = static_cast<int>( claims_grid.size() );
            Kokkos::parallel_for( "FmmBRSolver::fillDestMap",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_claims ),
                KOKKOS_LAMBDA( const int c )
            {
                const int li = claim_recv( c, 0 );
                const int lj = claim_recv( c, 1 );
                dest_rank_map( li - imin, lj - jmin ) = claim_recv( c, 2 );
            });
        }
        Kokkos::fence();

        // Walk the grid in the same row-major order packGridParticles uses,
        // emit one forward dest rank per grid tuple.
        const long num_grid = ni * nj;
        Kokkos::View<int*, MemorySpace> forward_dests(
            Kokkos::ViewAllocateWithoutInitializing( "FmmBRSolver_forward_dests" ),
            num_grid );
        Kokkos::parallel_for( "FmmBRSolver::forwardDests",
            Kokkos::RangePolicy<ExecutionSpace>( 0, num_grid ),
            KOKKOS_LAMBDA( const long k )
        {
            const long li_local = k / nj;
            const long lj_local = k % nj;
            forward_dests( k ) = dest_rank_map( li_local, lj_local );
        });
        Kokkos::fence();

        return Cabana::Distributor<MemorySpace>( _comm, forward_dests );
    }

    /* Directly compute the interface velocity by integrating the
     * vorticity across the surface, using Canopy's fast multipole
     * solver in place of the all-pairs Birkhoff-Rott evaluation.
     *
     * Pipeline:
     *   1. Pack grid-ordered owned nodes into _grid_particles
     *      (position from z, charge = simpson * omega, tag = (rank, i, j)).
     *   2. First call:
     *        deep-copy _grid_particles -> _canopy_particles, then run
     *        _canopy.setup() to build the tree, partition, and migrate.
     *      Subsequent calls:
     *        build a forward Distributor from the current
     *        _canopy_particles tags (see buildForwardDistributor),
     *        migrate _grid_particles -> _canopy_particles, then
     *        _canopy.auto_maintain() to handle position-driven
     *        migration. auto_maintain returns Migrate/Rebalance/Rebuild
     *        based on detected drift.
     *   3. _canopy.solve(..., compute_gradient=true) — returns
     *      gradient(p, c, d) = -sum_j q_c^(j) (x_p - x_j)_d / r^3
     *      with Plummer softening (r^2 + softening^2)^(-3/2). Charges
     *      already carry the per-source w_simpson factor.
     *   4. u_cross = omega x grad G via cross-of-gradients.
     *   5. Reverse-distribute (canopy -> grid origin) via a fresh
     *      Cabana::Distributor keyed on tag.origin_rank.
     *   6. Write zdot(i, j, d) = (dx*dy)/(4*pi) * u_cross[d] using
     *      tag.(i, j) on the receiving (grid-owning) rank.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const override
    {
#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
        const double call_t0 = MPI_Wtime();
#endif
        // Level-2 total spanning the whole body. The small zeroZdot kernel
        // below is intentionally left unattributed (counted in total but not
        // in any sub-phase), matching Canopy's "unaccounted slack" convention.
        BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_CIV_TOTAL );

        auto local_node_space = _pm.mesh().localGrid()->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );

        Kokkos::parallel_for( "FmmBRSolver::zeroZdot",
            Cabana::Grid::createExecutionPolicy( local_node_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( int i, int j ) {
                for ( int d = 0; d < 3; ++d ) zdot( i, j, d ) = 0.0;
            });

        // 1. Pack grid particles into _grid_particles (grid-ordered)
        {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_PACK );
            packGridParticles( _grid_particles, z, o );
        }
        const int num_grid = static_cast<int>( _grid_particles.size() );

        // 2. First call: deep-copy + full setup. Subsequent calls:
        //    forward-migrate fresh grid data into _canopy_particles,
        //    then auto_maintain to handle position drift.
#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
        const char* action_name = "Setup";
#endif
        if ( _first_call )
        {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_SETUP );
            _canopy_particles.resize( num_grid );
            Cabana::deep_copy( _canopy_particles, _grid_particles );
            _canopy.template setup<FmmField::Position, FmmField::Charge>(
                _canopy_particles, num_grid );
            _first_call = false;
        }
        else
        {
            auto forward_dist = [&] {
                BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_FWD_DIST );
                return buildForwardDistributor();
            }();
            aosoa_type new_canopy_particles(
                "FmmBRSolver_canopy_particles",
                forward_dist.totalNumImport() );
            {
                BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_FWD_MIGRATE );
                Cabana::migrate( forward_dist, _grid_particles, new_canopy_particles );
            }
            _canopy_particles = new_canopy_particles;
            const auto action = [&] {
                BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_AUTO_MAINTAIN );
                return _canopy.template auto_maintain<FmmField::Position, FmmField::Charge>(
                    _canopy_particles );
            }();
#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
            action_name = recordAutoMaintainAction( action );
#else
            (void)action;
#endif
        }
#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
        ++_call_count;
#endif
        const int num_local = _canopy.num_local_particles();

#if defined( BEATNIK_FMM_SNAPSHOT_DEBUG )
        // TEMPORARY (debug-nan branch): dump the exact (positions, charges)
        // that the upcoming solve() sees, for the steps bracketing the
        // premature full-rollup NaN, so the offline replay harness can
        // reproduce the blow-up without a 75-min queued run. One binary file
        // per rank per (step, substep). Requires profiling (sets _beatnik_step);
        // outside the window it is a no-op. Remove on resolution.
        maybeDumpSnapshot( num_local );
#endif

        // 3. Solve with compute_gradient=true
        const auto gradient = [&] {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_SOLVE );
            _canopy.template solve<FmmField::Position, FmmField::Charge>(
                _canopy_particles, /*compute_gradient=*/true );
            return _canopy.gradient();
        }();

        // 4. Cross-product of component gradients into u_out
        {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_CROSS );
            auto u_out = Cabana::slice<FmmField::UOut>( _canopy_particles );
            Kokkos::parallel_for( "FmmBRSolver::crossProduct",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
                KOKKOS_LAMBDA( const int p )
            {
                u_out( p, 0 ) = gradient( p, 1, 2 ) - gradient( p, 2, 1 );
                u_out( p, 1 ) = gradient( p, 2, 0 ) - gradient( p, 0, 2 );
                u_out( p, 2 ) = gradient( p, 0, 1 ) - gradient( p, 1, 0 );
            });
        }

        // 5. Reverse-distribute back to the origin grid rank
        Cabana::Distributor<MemorySpace> distributor = [&] {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_REV_DIST );
            Kokkos::View<int*, MemorySpace> origin_ranks(
                Kokkos::ViewAllocateWithoutInitializing( "FmmBRSolver_origin_ranks" ),
                num_local );
            {
                auto tag = Cabana::slice<FmmField::Tag>( _canopy_particles );
                Kokkos::parallel_for( "FmmBRSolver::extractOriginRanks",
                    Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
                    KOKKOS_LAMBDA( const int p ) {
                        origin_ranks( p ) = tag( p, 0 );
                    });
            }
            Kokkos::fence();
            return Cabana::Distributor<MemorySpace>( _comm, origin_ranks );
        }();
        aosoa_type out_particles(
            "FmmBRSolver_out_particles",
            distributor.totalNumImport() );
        {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_REV_MIGRATE );
            Cabana::migrate( distributor, _canopy_particles, out_particles );
        }

        // 6. Write zdot from the migrated (grid-rank) tuples
        {
            BEATNIK_SCOPED_TIMER_DETAILED( Beatnik::Profiling::TIMER_WRITE_ZDOT );
            const int num_recv  = static_cast<int>( out_particles.size() );
            const double scale  = ( _dx * _dy ) / ( 4.0 * M_PI );
            auto out_tag = Cabana::slice<FmmField::Tag>( out_particles );
            auto out_u   = Cabana::slice<FmmField::UOut>( out_particles );
            Kokkos::parallel_for( "FmmBRSolver::writeZdot",
                Kokkos::RangePolicy<ExecutionSpace>( 0, num_recv ),
                KOKKOS_LAMBDA( const int p )
            {
                const int li = out_tag( p, 1 );
                const int lj = out_tag( p, 2 );
                for ( int d = 0; d < 3; ++d )
                    zdot( li, lj, d ) = scale * out_u( p, d );
            });
            Kokkos::fence();
        }

#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
        const double call_dt = MPI_Wtime() - call_t0;
        if ( _rank == 0 && _substep_idx < static_cast<int>( _substep_records.size() ) )
        {
            _substep_records[_substep_idx] = { action_name, call_dt };
        }
        ++_substep_idx;
#endif
    }

#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
    void beginBeatnikStep( int step ) const override
    {
        // this-> required: _beatnik_step is a dependent base-class member.
        this->_beatnik_step = step;
        _substep_idx  = 0;
    }

    void flushProfile() const override
    {
        if ( _rank != 0 ) return;
        const int n = std::min( _substep_idx,
                                static_cast<int>( _substep_records.size() ) );
        for ( int s = 0; s < n; ++s )
        {
            printf( "    [FmmBRSolver step %d.%d] action=%s solve_time=%.6f s\n",
                    this->_beatnik_step, s,
                    _substep_records[s].action,
                    _substep_records[s].seconds );
        }
    }
#endif

  private:

#if defined( BEATNIK_FMM_SNAPSHOT_DEBUG )
    /* TEMPORARY (debug-nan branch): step window to dump particle snapshots.
     * The premature NaN aborts at step 1364; dump a margin on either side in
     * case the failing step shifts between runs. Inclusive bounds. */
    static constexpr int SNAP_FIRST_STEP = 1350;
    static constexpr int SNAP_LAST_STEP  = 1370;

    mutable int _snap_step{ -1 };
    mutable int _snap_sub{ 0 };

    /* Dump _canopy_particles (positions + charges) as seen by the upcoming
     * solve() to one binary file per rank per (step, substep), when the
     * current Beatnik step is inside [SNAP_FIRST_STEP, SNAP_LAST_STEP].
     * File layout (little-endian host): int32 num_local, then num_local
     * records of 6 doubles {px,py,pz,qx,qy,qz}. The replay harness globs all
     * rank files for a chosen (step, substep) and concatenates them. */
    void maybeDumpSnapshot( int num_local ) const
    {
        const int step = this->_beatnik_step;
        if ( step < SNAP_FIRST_STEP || step > SNAP_LAST_STEP )
            return;

        // Track substep index per step without depending on the profiling
        // counter (which may be compiled out independently).
        if ( step != _snap_step )
        {
            _snap_step = step;
            _snap_sub  = 0;
        }
        const int sub = _snap_sub++;

        auto pos = Cabana::slice<FmmField::Position>( _canopy_particles );
        auto chg = Cabana::slice<FmmField::Charge>( _canopy_particles );

        Kokkos::View<double* [6], MemorySpace> packed(
            Kokkos::ViewAllocateWithoutInitializing( "FmmBRSolver_snap_packed" ),
            num_local );
        Kokkos::parallel_for( "FmmBRSolver::packSnapshot",
            Kokkos::RangePolicy<ExecutionSpace>( 0, num_local ),
            KOKKOS_LAMBDA( const int p ) {
                packed( p, 0 ) = pos( p, 0 );
                packed( p, 1 ) = pos( p, 1 );
                packed( p, 2 ) = pos( p, 2 );
                packed( p, 3 ) = chg( p, 0 );
                packed( p, 4 ) = chg( p, 1 );
                packed( p, 5 ) = chg( p, 2 );
            } );
        Kokkos::fence();

        auto h_packed =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), packed );

        char fname[256];
        std::snprintf( fname, sizeof( fname ),
                       "fmm_snapshot_step%04d_sub%d_rank%04d.bin", step, sub,
                       _rank );
        std::FILE* f = std::fopen( fname, "wb" );
        if ( !f )
        {
            std::fprintf( stderr,
                          "[FmmBRSolver snapshot] rank %d: failed to open %s\n",
                          _rank, fname );
            return;
        }
        const int n = num_local;
        std::fwrite( &n, sizeof( int ), 1, f );
        std::fwrite( h_packed.data(), sizeof( double ),
                     static_cast<size_t>( num_local ) * 6, f );
        std::fclose( f );
        if ( _rank == 0 )
            std::fprintf( stderr,
                          "[FmmBRSolver snapshot] wrote step %d sub %d "
                          "(rank 0 num_local=%d)\n",
                          step, sub, num_local );
    }
#endif // BEATNIK_FMM_SNAPSHOT_DEBUG

    /* Build a Canopy::FmmConfig from Beatnik's Params + epsilon. The
     * softening is sqrt(epsilon) so Canopy's
     * (r^2 + softening^2)^(-3/2) matches Beatnik's
     * (r^2 + epsilon)^(-3/2) — Operators::BR does not square epsilon. */
    static Canopy::FmmConfig makeCanopyConfig( const Params& params,
                                               double epsilon )
    {
        Canopy::FmmConfig cfg;
        cfg.ncrit               = params.fmm_ncrit;
        cfg.max_depth           = params.fmm_max_depth;
        cfg.xmin_tol            = params.fmm_xmin_tol;
        cfg.xmax_tol            = params.fmm_xmax_tol;
        cfg.ymin_tol            = params.fmm_ymin_tol;
        cfg.ymax_tol            = params.fmm_ymax_tol;
        cfg.zmin_tol            = params.fmm_zmin_tol;
        cfg.zmax_tol            = params.fmm_zmax_tol;
        cfg.ncrit_tol           = params.fmm_ncrit_tol;
        cfg.replication_depth   = params.fmm_replication_depth;
        cfg.imbalance_tolerance = params.fmm_imbalance_tol;
        cfg.mac_theta           = params.fmm_mac_theta;
        cfg.softening           = std::sqrt( epsilon );
        cfg.near_softening_factor = params.fmm_near_softening_factor;
        return cfg;
    }

    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;
    const Params _params;
    
    l2g_type _local_L2G;
    MPI_Comm _comm;

    int _num_procs, _rank;

    /* Persistent Canopy FMM solver instance. Holds the tree, the
     * partitioner, and the comm plan. After the first setup() call,
     * subsequent steps use auto_maintain() instead of setup() so
     * the tree is built incrementally. */
    mutable canopy_solver_type _canopy;

    /* AoSoA holding one tuple per FMM particle in Canopy's current
     * partitioning. The `tag` field preserves each particle's grid
     * origin (rank, i, j) so a forward Distributor (rebuilt every
     * step by buildForwardDistributor) can refresh positions/charges
     * from the grid each call, and the post-solve reverse distribute
     * can route the FMM result back to the grid-owning rank. */
    mutable aosoa_type _canopy_particles{ "FmmBRSolver_canopy_particles", 0 };

    /* AoSoA holding one tuple per owned grid node, in row-major
     * (li, lj) order. Repacked from z, o on every computeInterfaceVelocity
     * call. On the first call, deep-copied into _canopy_particles
     * directly; on subsequent calls, forwarded to _canopy_particles
     * via buildForwardDistributor. */
    mutable aosoa_type _grid_particles{ "FmmBRSolver_grid_particles", 0 };

    /* On the first computeInterfaceVelocity call, _canopy_particles
     * is empty and Canopy's tree hasn't been built yet — take the
     * `setup()` path. Flipped to false at the end of that call so
     * subsequent calls take the `auto_maintain()` path. */
    mutable bool _first_call{ true };

#if defined(BEATNIK_ENABLE_PROFILING) && BEATNIK_PROFILING_LEVEL >= 1
    /* Instrumentation for the auto_maintain switch. Logged per call
     * at rank 0, plus a final histogram in the destructor. Useful
     * when a failure correlates with a Rebalance/Rebuild action
     * (which exercise more of Canopy's internal state than Migrate).
     * Gated behind Beatnik_ENABLE_PROFILING level >= 1; configure with
     * `-DBeatnik_ENABLE_PROFILING=ON` or `-DBeatnik_PROFILING_LEVEL=1`,
     * or via the `+profiling` / `profiling_level=N` spack variants. */
    mutable long _call_count{ 0 };
    mutable long _am_migrate{ 0 };
    mutable long _am_rebalance{ 0 };
    mutable long _am_rebuild{ 0 };

    /* Per-Beatnik-step buffer of substep records. Filled inside
     * computeInterfaceVelocity, drained by flushProfile() once the
     * Solver has printed its [Beatnik profile] header line. Sized
     * for the 3 RK3 substeps; extras are silently dropped. */
    struct SubstepRecord { const char* action; double seconds; };
    mutable std::array<SubstepRecord, 3> _substep_records{};
    /* _beatnik_step lives in BRSolverBase now (shared by all backends so
     * ZModel can report the offending timestep on a NaN/Inf blowup). */
    mutable int _substep_idx{ 0 };

    const char* recordAutoMaintainAction( typename canopy_solver_type::MaintenanceAction action ) const
    {
        using A = typename canopy_solver_type::MaintenanceAction;
        const char* name = "?";
        switch ( action )
        {
            case A::Migrate:   ++_am_migrate;   name = "Migrate";   break;
            case A::Rebalance: ++_am_rebalance; name = "Rebalance"; break;
            case A::Rebuild:   ++_am_rebuild;   name = "Rebuild";   break;
        }
        return name;
    }
#endif
};

}; // namespace Beatnik

#endif // BEATNIK_FMMBRSOLVER_HPP
