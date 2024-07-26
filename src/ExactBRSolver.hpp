/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier,BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file ExactBRSolver.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Thomas Hines <thomas-hines-01@utc.edu>
 * @author Jacob McCullough <jmccullough12@unm.edu>
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Class that uses a brute force approach to calculating the Birkhoff-Rott 
 * velocity intergral by using a all-pairs approach. Communication
 * uses a standard ring-pass communication algorithm. Does not attempt to 
 * reduce amount of computation per ring pass by using symetry of forces
 * as this complicates the GPU kernel.
 */

#ifndef BEATNIK_EXACTBRSOLVER_HPP
#define BEATNIK_EXACTBRSOLVER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>
#include <BRSolverBase.hpp>

/* Scaler reductions:
 * https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html
 * The following code was derived from the example at the link above.
 */
namespace Reduction
{  // namespace helps with name resolution in reduction identity 
   template< class ScalarType, int N >
   struct array_type {
     ScalarType the_array[N];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     array_type() { 
       for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     array_type(const array_type &rhs) { 
        for (int i = 0; i < N; i++ ){
           the_array[i] = rhs.the_array[i];
        }
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     array_type& operator += (const array_type &src) {
       for ( int i = 0; i < N; i++ ) {
          the_array[i]+=src.the_array[i];
       }
       return *this;
     }
   };
   typedef array_type<double, 3> ValueType;  // used to simplify code below
} // end namespace Reduction

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity<Reduction::ValueType> {
      KOKKOS_FORCEINLINE_FUNCTION static Reduction::ValueType sum() {
         return Reduction::ValueType();
      }
   };
}

namespace Beatnik
{

/**
 * The ExactBRSolver Class
 * @class ExactBRSolver
 * @brief Directly solves the Birkhoff-Rott integral using brute-force 
 * all-pairs calculation
 * XXX - Make all functions but computeInterfaceVelocity private?
 **/
template <class ExecutionSpace, class MemorySpace, class Params>
class ExactBRSolver : public BRSolverBase<ExecutionSpace, MemorySpace, Params>
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using spatial_mesh_type = SpatialMesh<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    
    using Node = Cabana::Grid::Node;
    using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    using node_array = typename pm_type::node_array;
    //using node_view = typename pm_type::node_view;
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    ExactBRSolver( const pm_type &pm, const BoundaryCondition &bc,
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
    {
        MPI_Comm_size(_comm, &_num_procs);
        MPI_Comm_rank(_comm, &_rank);
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    void computeInterfaceVelocityPiece(node_view zdot, node_view z, 
                                       node_view zremote, 
                                       node_view oremote,
                                       l2g_type remote_L2G) const
    {
        /* Project the Birkhoff-Rott calculation between all pairs of points on the 
         * interface, including accounting for any periodic boundary conditions.
         * Right now we brute force all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. */

        // Get the local index spaces of pieces we're working with. For the local surface piece
        // this is just the nodes we own. For the remote surface piece, we extract it from the
        // L2G converter they sent us.
        auto local_grid = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = remote_L2G.local_own_min[d];
            rmax[d] = remote_L2G.local_own_max[d];
        }
	    // Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        /* Figure out which directions we need to project the k/l point to
         * for any periodic boundary conditions */
        int kstart, lstart, kend, lend;
        if (_bc.isPeriodicBoundary({0, 1})) {
            kstart = -1; kend = 1;
        } else {
            kstart = kend = 0;
        }
        if (_bc.isPeriodicBoundary({1, 1})) {
            lstart = -1; lend = 1;
        } else {
            lstart = lend = 0;
        }

        /* Figure out how wide the bounding box is in each direction */
        auto low = _pm.mesh().boundingBoxMin();
        auto high = _pm.mesh().boundingBoxMax();;
        double width[3];
        for (int d = 0; d < 3; d++) {
            width[d] = high[d] - low[d];
        }

        /* Local temporaries for any instance variables we need so that we
         * don't have to lambda-capture "this" */
        double epsilon = _epsilon;
        double dx = _dx, dy = _dy;

        // Mesh dimensions for Simpson weight calc
        

    
        /* If the mesh is periodic, the index range is from
         * (halo width) to (halo width + mesh size)
         * If the mesh is non-periodic, the index range is from
         * (halo width) to (halo width + mesh size - 1)
         */
        int mesh_size = _pm.mesh().get_surface_mesh_size();
        int halo_width = _pm.mesh().get_halo_width();
        int is_periodic = _pm.mesh().is_periodic();
        std::array<long, 2> lmin;
        std::array<long, 2> lmax;
        for ( int d = 0; d < 2; ++d ) {
            lmin[d] = local_space.min( d );
            lmax[d] = local_space.max( d );
        }

        // int rank = _rank;
        int local_size = (lmax[0] - lmin[0]) * (lmax[1] - lmin[1]);
        int remote_size = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1]);
        int l_num_cols = lmax[1] - lmin[1];
        int r_num_cols = rmax[1] - rmin[1];
        // printf("R%d: li: (%d, %d), lj: (%d, %d) | ri: (%d, %d), rj: (%d, %d)\n", rank,
        //     lmin[0], lmax[0], lmin[1], lmax[1], rmin[0], rmax[0], rmin[1], rmax[1]);
        
        typedef typename Kokkos::TeamPolicy<exec_space>::member_type member_type;
        Kokkos::TeamPolicy<exec_space> mesh_policy(local_size, Kokkos::AUTO);
        Kokkos::parallel_for("Exact BR Force Team Loop", mesh_policy, 
            KOKKOS_LAMBDA(member_type team) 
        {
            //int thread_id = team.league_rank () * team.team_size () + team.team_rank ();
            // Figure out the i/j pieces of the block this team member is responsible for
            int league_rank = team.league_rank();
            //int team_rank = team.team_rank();
            //int team_size = team.team_size();
            int i = (league_rank / l_num_cols) + halo_width;
            int j = (league_rank % l_num_cols) + halo_width;

            // Kokkos::single(Kokkos::PerTeam(team), [=] () {
            //     if (i < lmin[0] || i >= lmax[0] || j < lmin[1] || j >= lmax[1])
            //     {
            //         printf("ERROR: R%d: ij: %d, %d\n", rank, i, j);
            //     }
            // });
        
            auto policy = Kokkos::TeamThreadRange(team, remote_size);
            double brsum[3];
            Kokkos::parallel_reduce(policy, [=] (const int &w, double &lsum0, double &lsum1, double &lsum2) {
                int k = (w / r_num_cols) + halo_width;
                int l = (w % r_num_cols) + halo_width;

                // if (k < rmin[0] || k >= rmax[0] || l < rmin[1] || l >= rmax[1])
                // {
                //     printf("ERROR: R%d: ijlk: %d, %d, %d, %d\n", rank, i, j, l, k);
                // }

                // We need the global indicies of the (k, l) point for Simpson's weight
                int remote_li[2] = {k, l};
                int remote_gi[2] = {0, 0};  // k, l
                remote_L2G(remote_li, remote_gi);

                /* Compute Simpson's 3/8 quadrature weight for this index */
                double weight;
                weight = simpsonWeight(remote_gi[0], mesh_size)
                            * simpsonWeight(remote_gi[1], mesh_size);
                /* We already have N^4 parallelism, so no need to parallelize on 
                    * the BR periodic points. Instead we serialize this in each thread
                    * and reuse the fetch of the i/j and k/l points */
                for (int kdir = kstart; kdir <= kend; kdir++) {
                    for (int ldir = lstart; ldir <= lend; ldir++) {
                        double offset[3] = {0.0, 0.0, 0.0}, br[3];
                        offset[0] = kdir * width[0];
                        offset[1] = ldir * width[1];

                        /* Do the Birkhoff-Rott evaluation for this point */
                        Operators::BR(br, z, zremote, oremote, epsilon, dx, dy, weight,
                                        i, j, k, l, offset);
                        
                        lsum0 += br[0];
                        lsum1 += br[1];
                        lsum2 += br[2];
                    }
                }
            }, brsum[0], brsum[1], brsum[2]);

            // Introduce a team barrier here to synchronize threads
            team.team_barrier();
            // printf("brsum: %0.1lf %0.1lf %0.1lf\n", brsum[0], brsum[1], brsum[2]);

            Kokkos::single(Kokkos::PerTeam(team), [=] () {
                for (int d = 0; d < 3; d++) {
                    zdot(i, j, d) = brsum[d];   
                }
            });
        });

        Kokkos::fence();
    }

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. 
     * This function is called three times per time step to compute the initial, forward, and half-step
     * derivatives for velocity and vorticity.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const override
    {
        auto local_node_space = _pm.mesh().localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        /* Zero out all of the i/j points */
        Kokkos::parallel_for("Exact BR Zero Loop",
            Cabana::Grid::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
               zdot(i, j, n) = 0.0;
        });
    
        // Compute forces for all owned nodes on this process
        computeInterfaceVelocityPiece(zdot, z, z, o, _local_L2G);
        // printView(_local_L2G, _rank, zdot, 1, 5, 5);

        /* Perform a ring pass of data between each process to compute forces of nodes 
         * on other processes on he nodes owned by this process */
        int next_rank = (_rank + 1) % _num_procs;
        int prev_rank = (_rank + _num_procs - 1) % _num_procs;

        // Create views for receiving data. Alternate which views are being sent and received into
        // *remote2 sends first, so it needs to be deep copied. *remote1 can just be allocated
        node_view zremote1(Kokkos::ViewAllocateWithoutInitializing ("zremote1"), z.extent(0), z.extent(1), z.extent(2));
        node_view zremote2(Kokkos::ViewAllocateWithoutInitializing ("zremote2"), z.extent(0), z.extent(1), z.extent(2));
        node_view oremote1(Kokkos::ViewAllocateWithoutInitializing ("oremote1"), o.extent(0), o.extent(1), o.extent(2));
        node_view oremote2(Kokkos::ViewAllocateWithoutInitializing ("oremote2"), o.extent(0), o.extent(1), o.extent(2));
        l2g_type L2G_remote1 = Cabana::Grid::IndexConversion::createL2G(*_pm.mesh().localGrid(), Cabana::Grid::Node());
        l2g_type L2G_remote2 = Cabana::Grid::IndexConversion::createL2G(*_pm.mesh().localGrid(), Cabana::Grid::Node());
        
        int zextents1[3];
        int zextents2[3];
        int oextents1[3];
        int oextents2[3];

        // printf("extents: omega: %d %d %d, z: %d %d %d\n", omega_view.extent(0), omega_view.extent(1), omega_view.extent(2),
        //     z.extent(0), z.extent(1), z.extent(2));
  
        // Now create references to these buffers. We go ahead and assign them here to get 
        // same type declarations. The loop reassigns these references as needed each time
        // around the loop.
        node_view *zsend_view = NULL; 
        node_view *osend_view = NULL;
        int * zsend_extents = NULL;
        int * osend_extents = NULL;
        l2g_type * L2G_send = NULL;

        node_view *zrecv_view = NULL; 
        node_view *orecv_view = NULL;
        int * zrecv_extents = NULL;
        int * orecv_extents = NULL;
        l2g_type * L2G_recv = NULL;

        // Perform the ring pass
        //int DEBUG_RANK = 1;
        for (int i = 0; i < _num_procs - 1; i++) {

            // Alternate between remote1 and remote2 sending and receiving data 
            // to avoid copying data across interations
            if (i % 2) {
                zsend_view = &zremote1;
                osend_view = &oremote1; 
                
                zsend_extents = zextents1;
                osend_extents = oextents1;
                L2G_send = &L2G_remote1;

                zrecv_view = &zremote2;
                orecv_view = &oremote2;
                zrecv_extents = zextents2;
                orecv_extents = oextents2;
                L2G_recv = &L2G_remote2;
            } else {
                if (i == 0) {
                    /* Avoid a deep copy on the first iteration */
                    zsend_view = &z;
                    osend_view = &o;
                } else {
                    zsend_view = &zremote2;
                    osend_view = &oremote2; 
                } 
                
                zsend_extents = zextents2;
                osend_extents = oextents2;
                L2G_send = &L2G_remote2;

                zrecv_view = &zremote1;
                orecv_view = &oremote1;
                zrecv_extents = zextents1;
                orecv_extents = oextents1;
                L2G_recv = &L2G_remote1;
            }

            // Prepare extents to send
            for (int j = 0; j < 3; j++) {
                zsend_extents[j] = zsend_view->extent(j);
                osend_extents[j] = osend_view->extent(j);
            }
                
            // Send o and z view sizes
            MPI_Sendrecv(zsend_extents, 3, MPI_INT, next_rank, 1, 
                        zrecv_extents, 3, MPI_INT, prev_rank, 1, _comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(osend_extents, 3, MPI_INT, next_rank, 6, 
                        orecv_extents, 3, MPI_INT, prev_rank, 6, _comm, MPI_STATUS_IGNORE);

            // Resize *remote2, which is receiving data
            Kokkos::resize(*zrecv_view, zrecv_extents[0], zrecv_extents[1], zrecv_extents[2]);
            Kokkos::resize(*orecv_view, orecv_extents[0], orecv_extents[1], orecv_extents[2]);

            // Send/receive the views
            MPI_Sendrecv(zsend_view->data(), int(zsend_view->size()), MPI_DOUBLE, next_rank, 3, 
                        zrecv_view->data(), int(zrecv_view->size()), MPI_DOUBLE, prev_rank, 3, 
                        _comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(osend_view->data(), int(osend_view->size()), MPI_DOUBLE, next_rank, 4, 
                        orecv_view->data(), int(orecv_view->size()), MPI_DOUBLE, prev_rank, 4, 
                        _comm, MPI_STATUS_IGNORE);

            // Send/receive the L2G structs. They have a constant size of 72 bytes (found using sizeof())
            MPI_Sendrecv(L2G_send, int(sizeof(*L2G_send)), MPI_BYTE, next_rank, 5, 
                         L2G_recv, int(sizeof(*L2G_recv)), MPI_BYTE, prev_rank, 5, 
                         _comm, MPI_STATUS_IGNORE);

            // Do computations
           computeInterfaceVelocityPiece(zdot, z, *zrecv_view, *orecv_view, *L2G_recv);
	    }

        // printf("\n\n*********************\n");
        // printView(_local_L2G, _rank, zdot, 1, 2, 7);
    }
    
    template <class l2g_type, class View>
    void printView(l2g_type local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
	Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            
            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            //printf("global: %d %d\n", local_gi[0], local_gi[1]);
            if (option == 1){
                if (dims == 3) {
                    printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }   
                    else if (dims == 2) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                    }
                }
            }
        });
    }

  private:
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    const Params _params;
    double _epsilon, _dx, _dy;

    MPI_Comm _comm;
    l2g_type _local_L2G;

    int _num_procs, _rank;
    // XXX Communication views and extents to avoid allocations during each ring pass
};

}; // namespace Beatnik

#endif // BEATNIK_EXACTBRSOLVER_HPP
