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

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. */
    template <class PositionView, class VorticityView>
    void computeInterfaceVelocity(PositionView zdot, PositionView z,
                                  VorticityView w) const
    {
        auto node_space = _pm.mesh().localGrid()->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        //std::size_t nnodes = node_space.size();

        /* Start by zeroing the interface velocity */
        
        /* Get an atomic view of the interface velocity */
        Kokkos::View<double ***,
             typename PositionView::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot;
    
        /* Parallel loop over each point of the interface
         * For each point on the interface
         *   Loop over each of interface higher in layout order than we are
         *   For each pair of this point and target point
         *      1. Compute the BR force between this point and the target point
         *      2. The point with the computed force
         *      3. Use atomics to update the destination point with the inverse target force
         *    points we've been working on */
        double epsilon = _epsilon;
        int istart = node_space.min(0), jstart = node_space.min(1);
        int iend = node_space.max(0), jend = node_space.max(1);
        double dx = _dx, dy = _dy;

        /* XXX Right now we brute fore all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. XXX */
        Kokkos::parallel_for("Exact BR Force Loop",
            Cajita::createExecutionPolicy(node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                atomic_zdot(i, j, n) = 0.0;

            for (int k = istart; k < iend; k++) {
                for (int l = jstart; l < jend; l++) {
                    double br[3];
                    double kweight, lweight;

		    /* Compute Simpson's 3/8 quadrature weight for this index */
                    if ((k == istart) || (k == iend - 1)) kweight = 3.0/8.0;
                    else if (k - (istart % 3) == 0) kweight = 3.0/4.0;
                    //else if (k % 3 == 0) kweight = 3.0/4.0;
                    else kweight = 9.0/8.0;

                    if ((l == jstart) || (l == jend - 1)) lweight = 3.0/8.0;
                    else if (l - (jstart % 3) == 0) lweight = 3.0/4.0;
                    //else if (l % 3 == 0) lweight = 3.0/4.0;
                    else lweight = 9.0/8.0;

		    /* Do the birchoff rott evaluation for this point */
                    Operators::BR(br, w, z, epsilon, dx, dy, kweight * lweight, i, j, k, l);

                    /* Add it its contribution to the integral */
                    for (int n = 0; n < 3; n++)
                        atomic_zdot(i, j, n) += br[n];
                }
            }
        }); 
    } 

    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;

};

}; // namespace Beatnik

#endif // BEATNIK_EXACTBRSOLVER_HPP
