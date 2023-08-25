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
#include <Cajita.hpp>
#include <Cajita_IndexConversion.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>

namespace Beatnik
{

/**
 * The ExactBRSolver Class
 * @class ExactBRSolver
 * @brief Directly solves the Birkhoff-Rott integral using brute-force 
 * all-pairs calculation
 **/
template <class ExecutionSpace, class MemorySpace>
class ExactBRSolver
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;
    
    using Node = Cajita::Node;
    using l2g_type = Cajita::IndexConversion::L2G<mesh_type, Node>;
    using node_array = typename pm_type::node_array;
    //using node_view = typename pm_type::node_view;
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cajita::Halo<MemorySpace>;

    ExactBRSolver( const pm_type & pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy)
        : _pm( pm )
        , _bc( bc )
        , _epsilon( epsilon )
        , _dx( dx )
        , _dy( dy )
        , _local_L2G( *_pm.mesh().localGrid() )
    {
	_comm = _pm.mesh().localGrid()->globalGrid().comm();
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    template <class AtomicView>
    void computeInterfaceVelocityPiece(AtomicView atomic_zdot, node_view z, 
                                       node_view zremote, node_view wremote, 
                                       l2g_type remote_L2G, int rank) const
    {
        /* Project the Birkhoff-Rott calculation between all pairs of points on the 
         * interface, including accounting for any periodic boundary conditions.
         * Right now we brute force all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. */

        // Get the local index spaces of pieces we're workign with. For the local surface piece
        // this is just hte nodes we own. For the remote surface piece, we extract it from the
        // L2G converter they sent us.
        auto local_grid = _pm.mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = remote_L2G.local_own_min[d];
            rmax[d] = remote_L2G.local_own_max[d];
        }
        Cajita::IndexSpace<2> remote_space(rmin, rmax);

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
        int num_nodes = _pm.mesh().get_mesh_size();

        /* Now loop over the cross product of all the node on the interface */
        auto pair_space = Operators::crossIndexSpace(local_space, remote_space);
        Kokkos::parallel_for("Exact BR Force Loop",
            Cajita::createExecutionPolicy(pair_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j, int k, int l) {

            // We need the global indicies of the (k, l) point for Simpson's weight
            int remote_li[2] = {k, l};
            int remote_gi[2] = {0, 0};  // k, l
            remote_L2G(remote_li, remote_gi);
            
            double brsum[3] = {0.0, 0.0, 0.0};

            /* Compute Simpson's 3/8 quadrature weight for this index */
            double weight;
            weight = simpsonWeight(remote_gi[0], num_nodes)
                        * simpsonWeight(remote_gi[1], num_nodes);
            // if (remote_gi[0] == 2 && remote_gi[1] == 7) {
                // printf("R%d: (%d, %d), simpson weight: %0.5lf\n", rank, remote_gi[0], remote_gi[1], weight);
            // }
            
            /* We already have N^4 parallelism, so no need to parallelize on 
             * the BR periodic points. Instead we serialize this in each thread
             * and reuse the fetch of the i/j and k/l points */
            for (int kdir = kstart; kdir <= kend; kdir++) {
                for (int ldir = lstart; ldir <= lend; ldir++) {
                    double offset[3] = {0.0, 0.0, 0.0}, br[3];
                    offset[0] = kdir * width[0];
                    offset[1] = ldir * width[1];

                    /* Do the Birkhoff-Rott evaluation for this point */
                    Operators::BR(br, z, zremote, wremote, epsilon, dx, dy, weight,
                                  i, j, k, l, offset);
                    for (int d = 0; d < 3; d++) {
                        brsum[d] += br[d];
                    }
                }
            }

            /* Add it its contribution to the integral */
            for (int n = 0; n < 3; n++) {
                atomic_zdot(i, j, n) += brsum[n];
            }
            // if (remote_gi[0] == 2 && remote_gi[1] == 6 && i == 2 && j == 2) {
            //     printf("R%d: (%d, %d), (%d, %d), %0.13lf %0.13lf %0.13lf\n", rank, i, j, remote_gi[0], remote_gi[1], brsum[0], brsum[1], brsum[2]);
            // }
        
        });
    }

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. 
     * This function is called three times per time step to compute the initial, forward, and half-step
     * derivatives for velocity and vorticity.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view w) const
    {
        auto local_node_space = _pm.mesh().localGrid()->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());

        int num_procs = -1;
        int rank = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // int x01 = 1;
        // int y01 = 1;
        // int z01 = 0;
        // if (rank == 0) {
        //     w(x01, y01, z01) = 0.01;
        //     z(x01, y01, z01) = 0.01;
        // }
        // if (rank == 1) {
        //     w(x01, y01, z01) = 1.1;
        //     z(x01, y01, z01) = 1.01;
        // }
        // if (rank == 2) {
        //     w(x01, y01, z01) = 2.2;
        //     z(x01, y01, z01) = 2.02;
        // }
        // if (rank == 3) {
        //     w(x01, y01, z01) = 3.3;
        //     z(x01, y01, z01) = 3.03;
        // }
        // if (rank == 4) {
        //     w(x01, y01, z01) = 4.4;
        //     z(x01, y01, z01) = 4.04;
        // }
        // if (rank == 5) {
        //     w(x01, y01, z01) = 5.5;
        //     z(x01, y01, z01) = 5.05;
        // }
        // if (rank == 0) {
        //     printf("w R%d initial: w(1, 1, 1) = %lf\n", rank, w(1, 1, 1));
        //     printf("z R%d initial: z(1, 1, 1) = %lf\n", rank, z(1, 1, 1));
        // }

        /* Start by zeroing the interface velocity */
        
        /* Get an atomic view of the interface velocity, since each k/l point
         * is going to be updating it in parallel */
        Kokkos::View<double ***,
             typename node_view::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot;
    
        /* Zero out all of the i/j points - XXX Is this needed are is this already zeroed somewhere else? */
        Kokkos::parallel_for("Exact BR Zero Loop",
            Cajita::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                atomic_zdot(i, j, n) = 0.0;
        });
        
        // Compute forces for all owned nodes on this process
        computeInterfaceVelocityPiece(atomic_zdot, z, z, w, _local_L2G);

        /* Perform a ring pass of data between each process to compute forces of nodes 
         * on other processes on he nodes owned by this process */
        int next_rank = (rank + 1) % num_procs;
        int prev_rank = (rank + num_procs - 1) % num_procs;

        // Create views for receiving data. Alternate which views are being sent and received into
        // *remote2 sends first, so it needs to be deep copied. *remote1 can just be allocated
        // Kokkos::Array<Kokkos::View<node_view>, 2> zviews;
        // Kokkos::Array<Kokkos::View<node_view>, 2> wviews;
        node_view wremote1(Kokkos::ViewAllocateWithoutInitializing ("wremote1"), w.extent(0), w.extent(1), w.extent(2));
        node_view wremote2(Kokkos::ViewAllocateWithoutInitializing ("wremote2"), w.extent(0), w.extent(1), w.extent(2));
        node_view zremote1(Kokkos::ViewAllocateWithoutInitializing ("zremote1"), z.extent(0), z.extent(1), z.extent(2));
        node_view zremote2(Kokkos::ViewAllocateWithoutInitializing ("zremote2"), z.extent(0), z.extent(1), z.extent(2));
        l2g_type L2G_remote1 = Cajita::IndexConversion::createL2G(*_pm.mesh().localGrid(), Cajita::Node());
        l2g_type L2G_remote2 = Cajita::IndexConversion::createL2G(*_pm.mesh().localGrid(), Cajita::Node());
        
        // zviews[0] = zremote1;
        // zviews[1] = zremote2;
        // wviews[0] = wremote1;
        // wviews[1] = wremote2;

        int wextents1[3];
        int wextents2[3];
        int zextents1[3];
        int zextents2[3];
  
        // Now create references to these buffers. We go ahead and assign them here to get 
        // same type declarations. The loop reassigns these references as needed each time
        // around the loop.
        node_view *zsend_view = NULL; 
        node_view *wsend_view = NULL; 
        int * zsend_extents = NULL;
        int * wsend_extents = NULL;
        l2g_type * L2G_send = NULL;

        node_view *zrecv_view = NULL; 
        node_view *wrecv_view = NULL; 
        int * zrecv_extents = NULL;
        int * wrecv_extents = NULL;
        l2g_type * L2G_recv = NULL;

        // Perform the ring pass
        //int DEBUG_RANK = 1;
        for (int i = 0; i < num_procs - 1; i++) {

            // Alternate between remote1 and remote2 sending and receiving data 
            // to avoid copying data across interations
            if (i % 2) {
                // if (rank == DEBUG_RANK) {
                //     printf("in 2\n");
                // }
                zsend_view = &zremote1; wsend_view = &wremote1; 
                zsend_extents = zextents1; wsend_extents = wextents1;
                L2G_send = &L2G_remote1;

                zrecv_view = &zremote2; wrecv_view = &wremote2; 
                zrecv_extents = zextents2; wrecv_extents = wextents2;
                L2G_recv = &L2G_remote2;
            } else {
                // if (rank == DEBUG_RANK) {
                //     printf("in 3\n");
                // }
                if (i == 0) {
                    // if (rank == DEBUG_RANK) {
                    //     printf("in 0\n");
                    // }
                    /* Avoid a deep copy on the first iteration */
                    wsend_view = &w; zsend_view = &z;
                } else {
                    // if (rank == DEBUG_RANK) {
                    //     printf("in -1\n");
                    // }
                    wsend_view = &wremote2; zsend_view = &zremote2; 
                } 
                
                zsend_extents = zextents2; wsend_extents = wextents2;
                L2G_send = &L2G_remote2;

                zrecv_view = &zremote1; wrecv_view = &wremote1; 
                zrecv_extents = zextents1; wrecv_extents = wextents1;
                L2G_recv = &L2G_remote1;
            }

            // Prepare extents to send
            for (int j = 0; j < 3; j++) {
                wsend_extents[j] = wsend_view->extent(j);
                zsend_extents[j] = zsend_view->extent(j);
            }
        //    if (rank == DEBUG_RANK) {
        //         printf("sending: %d %d %d\n", wsend_view.extent(0), wsend_view.extent(1), wsend_view.extent(2));
        //     }
                
            // Send w and z view sizes
            MPI_Sendrecv(wsend_extents, 3, MPI_INT, next_rank, 0, 
                        wrecv_extents, 3, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(zsend_extents, 3, MPI_INT, next_rank, 1, 
                        zrecv_extents, 3, MPI_INT, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Resize *remote2, which is receiving data
            Kokkos::resize(*wrecv_view, wrecv_extents[0], wrecv_extents[1], wrecv_extents[2]);
            Kokkos::resize(*zrecv_view, zrecv_extents[0], zrecv_extents[1], zrecv_extents[2]);
            
            // Send/receive the views
            MPI_Sendrecv(wsend_view->data(), int(wsend_view->size()), MPI_DOUBLE, next_rank, 2, 
                        wrecv_view->data(), int(wrecv_view->size()), MPI_DOUBLE, prev_rank, 2, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(zsend_view->data(), int(zsend_view->size()), MPI_DOUBLE, next_rank, 3, 
                        zrecv_view->data(), int(zrecv_view->size()), MPI_DOUBLE, prev_rank, 3, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send/receive the L2G structs. They have a constant size of 72 bytes (found using sizeof())
            MPI_Sendrecv(L2G_send, int(sizeof(*L2G_send)), MPI_BYTE, next_rank, 4, 
                         L2G_recv, int(sizeof(*L2G_recv)), MPI_BYTE, prev_rank, 4, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // if (rank == DEBUG_RANK) {
            //     printf("w %d: R%d received from R%d: w_rec = %0.2lf (w_send = %0.2lf, w = %0.2lf)\n", i, rank, prev_rank, (*wrecv_view)(1, 1, 0), (*wsend_view)(1, 1, 0), w(1, 1, 0));
            //     printf("z %d: R%d received from R%d: z_rec = %0.2lf (z_send = %0.2lf, z = %0.2lf)\n", i, rank, prev_rank, (*zrecv_view)(1, 1, 0), (*zsend_view)(1, 1, 0), z(1, 1, 0));
            // }
            // if (rank == DEBUG_RANK) {
            //     printf("w %d: R%d received from R%d: w_rec = %d (w_send = %d, w = %d)\n", i, rank, prev_rank, wrecv_extents[1], wsend_extents[1], w.extent(1));
            //     //printf("z %d: R%d received from R%d: z_rec = %d (z_send = %d, z = %d)\n", i, rank, prev_rank, zrecv_extents[1], zsend_extents[1], z.extent(1));
            //     printf("R%d: remote1 ex: %d, remote2 ex: %d\n", rank, wextents1[1], wextents2[1]);
            // }

            // Do computations
            computeInterfaceVelocityPiece(atomic_zdot, z, *zrecv_view, *wrecv_view, *L2G_recv);
	    }

        // printView(_local_L2G, rank, z, 2, 2, 7);
        // printView(_local_L2G, rank, w, 2, 2, 7);
        // printView(_local_L2G, rank, zdot, 2, 2, 7);
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
        Cajita::IndexSpace<2> remote_space(rmin, rmax);

        Kokkos::parallel_for("print views",
            Cajita::createExecutionPolicy(remote_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            
            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            //printf("global: %d %d\n", local_gi[0], local_gi[1]);
            if (option == 1){
                if (dims == 3) {
                    printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }   
                    else if (dims == 2) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                    }
                }
            }
        });
    }

  private:
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;
    MPI_Comm _comm;
    l2g_type _local_L2G;
    // XXX Communication views and extents to avoid allocations during each ring pass
};

}; // namespace Beatnik

#endif // BEATNIK_EXACTBRSOLVER_HPP
