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
 *
 * @section DESCRIPTION
 * Class that uses a brute force approach to calculating the Birkhoff-Rott 
 * velocity intergral by using a all-pairs approach. Future communication
 * approach will be a standard ring-pass communication algorithm.
 */

#ifndef BEATNIK_EXACTBRSOLVER_HPP
#define BEATNIK_EXACTBRSOLVER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Cajita_IndexConversion_Edits.hpp>
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

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      device_type>;

    using halo_type = Cajita::Halo<MemorySpace>;

    ExactBRSolver( const pm_type & pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy)
        : _pm( pm )
        , _bc( bc )
        , _epsilon( epsilon )
        , _dx( dx )
        , _dy( dy )
    {
	// auto comm = _pm.mesh().localGrid()->globalGrid().comm();
        /* Create a 1D MPI communicator for the ring-pass on this
         * algorithm */
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    template <class AtomicView, class PositionView, class VorticityView, class L2G>
    void computeInterfaceVelocityPiece(AtomicView atomic_zdot, 
                                       PositionView z, VorticityView w, L2G local_L2G,
                                       PositionView zremote, VorticityView wremote, L2G remote_L2G) const
    {
        /* Project the Birkhoff-Rott calculation between all pairs of points on the 
         * interface, including accounting for any periodic boundary conditions.
         * Right now we brute force all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. */
        /* Figure out which directions we need to project the k/l point to
         * for any periodic boundary conditions */
        auto local_space = local_L2G.local_own_space;
        auto remote_space = remote_L2G.local_own_space;

        int num_procs = 0;
        int rank = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        int istart = local_space.min(0), jstart = local_space.min(1);
        int iend = local_space.max(0), jend = local_space.max(1);
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
            
            /* We already have N^4 parallelism, so no need to parallelize on 
             * the BR periodic points. Instead we serialize this in each thread
             * and reuse the fetch of the i/j and k/l points */
            for (int kdir = kstart; kdir <= kend; kdir++) {
                for (int ldir = lstart; ldir <= lend; ldir++) {
                    double offset[3] = {0.0, 0.0, 0.0}, br[3];
                    offset[0] = kdir * width[0];
                    offset[1] = ldir * width[1];

                    /* Do the Birkhoff-Rott evaluation for this point */
                    Operators::BR_with_remote(br, w, wremote, z, zremote, epsilon, dx, dy, weight,
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
        
        });
    }

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. 
     * This function is called three times per time step to compute the initial, forward, and half-step
     * derivatives for velocity and vorticity.
     */
    template <class PositionView, class VorticityView>
    void computeInterfaceVelocity(PositionView zdot, PositionView z,
                                  VorticityView w) const
    {
        auto local_node_space = _pm.mesh().localGrid()->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());

        int num_procs = -1;
        int rank = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        auto local_L2G = Cajita::IndexConversion_Edits::createL2G(*_pm.mesh().localGrid(), Cajita::Node());

        /* Start by zeroing the interface velocity */
        
        /* Get an atomic view of the interface velocity, since each k/l point
         * is going to be updating it in parallel */
        Kokkos::View<double ***,
             typename PositionView::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot;
    
        /* Zero out all of the i/j points - XXX Is this needed are is this already zeroed somewhere else? */
        Kokkos::parallel_for("Exact BR Zero Loop",
            Cajita::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                atomic_zdot(i, j, n) = 0.0;
        });
        
        computeInterfaceVelocityPiece(atomic_zdot, z, w, local_L2G, z, w, local_L2G);

        /* Perform a ring pass of data between each process */

        int next_rank = (rank + 1) % num_procs;
        int prev_rank = (rank + num_procs - 1) % num_procs;

        // Create subviews for receiving data. Alternate which views are being sent and received into
        // *remote2 sends first, so it needs to be deep copied. *remote1 can just be allocated
        Kokkos::View<double***, typename VorticityView::device_type> wremote1(Kokkos::ViewAllocateWithoutInitializing ("wremote1"), w.extent(0), w.extent(1), w.extent(2));
        Kokkos::View<double***, typename VorticityView::device_type> wremote2(Kokkos::ViewAllocateWithoutInitializing ("wremote2"), w.extent(0), w.extent(1), w.extent(2));
        Kokkos::View<double***, typename PositionView::device_type> zremote1(Kokkos::ViewAllocateWithoutInitializing ("zremote1"), z.extent(0), z.extent(1), z.extent(2));
        Kokkos::View<double***, typename PositionView::device_type> zremote2(Kokkos::ViewAllocateWithoutInitializing ("zremote2"), z.extent(0), z.extent(1), z.extent(2));
        auto L2G_remote1 = Cajita::IndexConversion_Edits::createL2G(*_pm.mesh().localGrid(), Cajita::Node());
        auto L2G_remote2 = Cajita::IndexConversion_Edits::createL2G(*_pm.mesh().localGrid(), Cajita::Node());

        int wextents1[3];
        int wextents2[3];
        int zextents1[3];
        int zextents2[3];
        for (int i = 0; i < num_procs - 1; i++) {

            // Alternate between remote1 and remote2 sending and receiving data to avoid copying data
            if (i % 2) { // remote1 send, remote2 receive

                // Prepare extents to send
                for (int j = 0; j < 3; j++) {
                    wextents1[j] = wremote1.extent(j);
                    zextents1[j] = zremote1.extent(j);
                }
                
                // Send w and z view sizes
                MPI_Sendrecv(&wextents1, 3, MPI_INT, next_rank, 0, &wextents2, 3, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&zextents1, 3, MPI_INT, next_rank, 1, &zextents2, 3, MPI_INT, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Resize *remote2, which is receiving data
                Kokkos::resize(wremote2, wextents2[0], wextents2[1], wextents2[2]);
                Kokkos::resize(zremote2, zextents2[0], zextents2[1], zextents2[2]);
                
                // Send/receive the views
                MPI_Sendrecv(wremote1.data(), int(wremote1.size()), MPI_DOUBLE, next_rank, 2, wremote2.data(), int(wremote2.size()), MPI_DOUBLE, prev_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(zremote1.data(), int(zremote1.size()), MPI_DOUBLE, next_rank, 3, zremote2.data(), int(zremote2.size()), MPI_DOUBLE, prev_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Send/receive the L2G structs. They have a constant size of 72 bytes (found using sizeof())
                MPI_Sendrecv(&L2G_remote1, int(sizeof(L2G_remote1)), MPI_BYTE, next_rank, 4, &L2G_remote2, int(sizeof(L2G_remote2)), MPI_BYTE, prev_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Do computations
                computeInterfaceVelocityPiece(atomic_zdot, z, w, local_L2G, zremote2, wremote2, L2G_remote2);
            }

            else { // remote2 send, remote1 receive
                
                // Prepare extents to send
                for (int j = 0; j < 3; j++) {
                    wextents2[j] = wremote2.extent(j);
                    zextents2[j] = zremote2.extent(j);
                }

                MPI_Sendrecv(&wextents2, 3, MPI_INT, next_rank, 0, &wextents1, 3, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&zextents2, 3, MPI_INT, next_rank, 1, &zextents1, 3, MPI_INT, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Resize *remote1, which is receiving data
                Kokkos::resize(wremote1, wextents1[0], wextents1[1], wextents1[2]);
                Kokkos::resize(zremote1, zextents1[0], zextents1[1], zextents1[2]);

                // Send/receive the views
                // First iteration, send w and z to avoid a deep dopy into wremote2 and zremote2
                if (i == 0) {
                    MPI_Sendrecv(w.data(), int(w.size()), MPI_DOUBLE, next_rank, 2, wremote1.data(), int(wremote1.size()), MPI_DOUBLE, prev_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(z.data(), int(z.size()), MPI_DOUBLE, next_rank, 3, zremote1.data(), int(zremote1.size()), MPI_DOUBLE, prev_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else {
                    MPI_Sendrecv(wremote2.data(), int(wremote2.size()), MPI_DOUBLE, next_rank, 2, wremote1.data(), int(wremote1.size()), MPI_DOUBLE, prev_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(zremote2.data(), int(zremote2.size()), MPI_DOUBLE, next_rank, 3, zremote1.data(), int(zremote1.size()), MPI_DOUBLE, prev_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                // Send/receive the L2G structs. They have a constant size of 72 bytes (found using sizeof())
                MPI_Sendrecv(&L2G_remote2, int(sizeof(L2G_remote2)), MPI_BYTE, next_rank, 4, &L2G_remote1, int(sizeof(L2G_remote1)), MPI_BYTE, prev_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Do computations
                computeInterfaceVelocityPiece(atomic_zdot, z, w, local_L2G, zremote1, wremote1, L2G_remote1);
            }   
        }
    }
    template <class L2G, class View>
    void printView(L2G local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        Kokkos::parallel_for("print views",
            Cajita::createExecutionPolicy(local_L2G.local_own_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {

            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
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
                        printf("R%d: %d: %d: %d: %d: %.5lf: %.5lf: %.5lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }   
                    else if (dims == 2) {
                        printf("R%d: %d: %d: %d: %d: %.5lf: %.5lf\n", rank, i, j, local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                    }
                }
            }
        });
    }

    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;

  private:
    // XXX Save the w and z sizes for each process so we can avoid MPI send/receives in the ring pass?

};

}; // namespace Beatnik

#endif // BEATNIK_EXACTBRSOLVER_HPP
