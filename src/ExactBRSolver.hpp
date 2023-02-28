/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file ExactBRSolver.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Thomas Hines <thomas-hines-01@utc.edu>
 * @author Jacob McCullough <jmccullough12@unm.edu>
 *
 * @section DESCRIPTION
 * Class that uses a brute force approach to calculating the Birchoff-Rott 
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
 * @brief Directly solves the Birchoff-Rott integral using brute-force 
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

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. */
    template <class PositionView, class VorticityView>
    void computeInterfaceVelocity(PositionView zdot, PositionView z,
                                  VorticityView w) const
    {
        auto node_space = _pm.mesh().localGrid()->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        //std::size_t nnodes = node_space.size();

        /* Start by zeroing the interface velocity */
        
        /* Get an atomic view of the interface velocity, since each k/l point
         * is going to be updating it in parallel */
        Kokkos::View<double ***,
             typename PositionView::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot;
    
        /* Zero out all of the i/j points - XXX Is this needed are is this already zeroed somewhere else? */
	Kokkos::parallel_for("Exact BR Zero Loop",
	    Cajita::createExecutionPolicy(node_space, ExecutionSpace()),
	    KOKKOS_LAMBDA(int i, int j) {
	    for (int n = 0; n < 3; n++)
	        atomic_zdot(i, j, n) = 0.0;
	});

        /* Project the birchoff rott calculation between all pairs of points on the 
         * interface, including accounting for any periodic boundary conditions.
         * Right now we brute fore all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. */

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
        int istart = node_space.min(0), jstart = node_space.min(1);
        int iend = node_space.max(0), jend = node_space.max(1);
        double dx = _dx, dy = _dy;

        /* Now loop over the cross product of all the node on the interface */
        auto pair_space = Operators::crossIndexSpace(node_space, node_space);
        Kokkos::parallel_for("Exact BR Force Loop",
            Cajita::createExecutionPolicy(pair_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j, int k, int l) {
                double brsum[3] = {0.0, 0.0, 0.0};;
		/* Compute Simpson's 3/8 quadrature weight for this index */
		double weight = simpsonWeight(k - istart, iend - istart)
			        * simpsonWeight(l - jstart, jend - jstart);

                /* We already have N^4 parallelism, so no need to parallelize on 
                 * the BR periodic points. Instead we serialize this in each thread
                 * and reuse the fetch of the i/j and k/l points */
                for (int kdir = kstart; kdir <= kend; kdir++) {
                    for (int ldir = lstart; ldir <= lend; ldir++) {
                        double offset[3] = {0.0, 0.0, 0.0}, br[3];
                        offset[0] = kdir * width[0];
                        offset[1] = ldir * width[1];
		        /* Do the birchoff rott evaluation for this point */
                        Operators::BR(br, w, z, epsilon, dx, dy, weight, i, j, k, l, offset);
                        for (int d = 0; d < 3; d++) {
                            brsum[d] += br[d];
                        }
                    }
                }

                /* Add it its contribution to the integral */
                for (int n = 0; n < 3; n++)
                    atomic_zdot(i, j, n) += brsum[n];
        }); 
    } 

    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;

};

}; // namespace Beatnik

#endif // BEATNIK_EXACTBRSOLVER_HPP
