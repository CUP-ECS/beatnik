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

#include <cmath>
#include <iostream>
#include <memory>

#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>
#include <BRSolverBase.hpp>

namespace Beatnik
{

static constexpr int P_ORDER = 6;
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
                   params.fmm_ncrit,
                   params.fmm_max_depth,
                   std::array<double, 3>{ params.fmm_bbox_tol,
                                          params.fmm_bbox_tol,
                                          params.fmm_bbox_tol },
                   params.fmm_ncrit_tol,
                   params.fmm_replication_depth,
                   params.fmm_imbalance_tol,
                   params.fmm_mac_theta,
                   /* softening = sqrt(epsilon) so Canopy's
                    * (r^2 + softening^2)^(-3/2) matches Beatnik's
                    * (r^2 + epsilon)^(-3/2) — Operators::BR does not
                    * square epsilon. */
                   std::sqrt( epsilon ) )
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

    /* Directly compute the interface velocity by integrating the
     * vorticity across the surface, using Canopy's fast multipole
     * solver in place of the all-pairs Birkhoff-Rott evaluation.
     *
     * Pipeline (run on every call — auto_maintain is a follow-up):
     *   1. Pack grid-ordered owned nodes into _canopy_particles
     *   2. _canopy.setup() — Canopy builds the tree, partitions, and
     *      migrates particles across ranks; the `tag` field travels
     *      with each particle so the FMM result can be routed back.
     *   3. _canopy.solve(..., compute_gradient=true) — returns
     *      gradient(p, c, d) = -sum_j q_c^(j) (x_p - x_j)_d / r^3
     *      with Plummer softening (r^2 + softening^2)^(-3/2).
     *      We packed charges as w_simpson * omega so the per-source
     *      prefactor is folded in.
     *   4. u_cross = omega x grad G via cross-of-gradients:
     *        u[0] = grad(p,1,2) - grad(p,2,1)
     *        u[1] = grad(p,2,0) - grad(p,0,2)
     *        u[2] = grad(p,0,1) - grad(p,1,0)
     *   5. Reverse-distribute (canopy -> grid origin) via a
     *      Cabana::Distributor keyed on tag.origin_rank.
     *   6. Write zdot(i, j, d) = (dx*dy)/(4*pi) * u_cross[d] using
     *      tag.(i, j) on the receiving (grid-owning) rank.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const override
    {
        auto local_node_space = _pm.mesh().localGrid()->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );

        Kokkos::parallel_for( "FmmBRSolver::zeroZdot",
            Cabana::Grid::createExecutionPolicy( local_node_space, ExecutionSpace() ),
            KOKKOS_LAMBDA( int i, int j ) {
                for ( int d = 0; d < 3; ++d ) zdot( i, j, d ) = 0.0;
            });

        // 1. Pack grid particles into _canopy_particles (grid-ordered)
        packGridParticles( _canopy_particles, z, o );
        const int num_before = static_cast<int>( _canopy_particles.size() );

        // 2. Setup: build tree, partition, migrate. Calling setup() every
        //    step (rather than auto_maintain) is a v1 simplification —
        //    see tasks/integrate_canopy.md for the follow-up note.
        _canopy.template setup<FmmField::Position, FmmField::Charge>(
            _canopy_particles, num_before );
        const int num_local = _canopy.num_local_particles();

        // 3. Solve with compute_gradient=true
        _canopy.template solve<FmmField::Position, FmmField::Charge>(
            _canopy_particles, /*compute_gradient=*/true );
        const auto gradient = _canopy.gradient();

        // 4. Cross-product of component gradients into u_out
        {
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

        Cabana::Distributor<MemorySpace> distributor( _comm, origin_ranks );
        aosoa_type out_particles(
            "FmmBRSolver_out_particles",
            distributor.totalNumImport() );
        Cabana::migrate( distributor, _canopy_particles, out_particles );

        // 6. Write zdot from the migrated (grid-rank) tuples
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

  private:
    
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;
    const Params _params;
    
    l2g_type _local_L2G;
    MPI_Comm _comm;

    int _num_procs, _rank;

    /* Persistent Canopy FMM solver instance. Holds the tree, the
     * partitioner, and the comm plan. Currently rebuilt every step
     * via setup() — switching to auto_maintain is a follow-up. */
    mutable canopy_solver_type _canopy;

    /* AoSoA holding one tuple per FMM particle. After setup() it
     * lives in Canopy order; the `tag` field preserves each
     * particle's grid origin (rank, i, j) so the post-solve reverse
     * distribute can return the FMM result to the grid-owning rank. */
    mutable aosoa_type _canopy_particles{ "FmmBRSolver_canopy_particles", 0 };
};

}; // namespace Beatnik

#endif // BEATNIK_FMMBRSOLVER_HPP
