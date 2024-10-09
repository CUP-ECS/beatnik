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
 * @file BoundaryCondition.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Boundary Conditions for Beatnik prototype interface model
 */

#ifndef BEATNIK_BOUNDARYCONDITIONS_HPP
#define BEATNIK_BOUNDARYCONDITIONS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Beatnik_Types.hpp>
#include <SurfaceMesh.hpp>

#include <Kokkos_Core.hpp>
#include "Operators.hpp"

namespace Beatnik
{

/**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with
 * Kokkos inline Functions
 */

struct BoundaryCondition
{
    bool isPeriodicBoundary(std::array<int, 2> dir) const
    {
        if ((dir[0] == -1) && (boundary_type[0] == MeshBoundaryType::PERIODIC))
            return true;
        if ((dir[0] == 1) && (boundary_type[2] == MeshBoundaryType::PERIODIC))
            return true;
        if ((dir[1] == -1) && (boundary_type[1] == MeshBoundaryType::PERIODIC))
            return true;
        if ((dir[1] == 1) && (boundary_type[3] == MeshBoundaryType::PERIODIC))
            return true;
        return false;
    }

    bool isFreeBoundary(std::array<int, 2> dir) const
    {
        if ((dir[0] == -1) && (boundary_type[0] == MeshBoundaryType::FREE))
            return true;
        if ((dir[0] == 1) && (boundary_type[2] == MeshBoundaryType::FREE))
            return true;
        if ((dir[1] == -1) && (boundary_type[1] == MeshBoundaryType::FREE))
            return true;
        if ((dir[1] == 1) && (boundary_type[3] == MeshBoundaryType::FREE))
            return true;
        return false;
    }

    /* Apply boundary conditions to a generic field with DOF elements that 
     * doesn't require special handling */
    template <class MeshType, class ArrayType> 
    void applyField(const MeshType &mesh, ArrayType field, int dof) const
    {
        using exec_space = typename ArrayType::execution_space;
        auto local_grid = *(mesh.localGrid());
        auto f = field.view();
        
        /* Loop through the directions to correct boundary index spaces when
         * needed. */
        for (int i = -1; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                if (i == 0 && j == 0) continue;

                /* In general, halo exchange takes care of periodic boundaries.
                 * For free boundaries, we linearly extrapolate the field into 
                 * the boundary */
                std::array<int, 2> dir = {i, j};
                if (isFreeBoundary(dir))
                {
                    /* For free boundaries, we have to extrapolate from the mesh
                     * into the boundary to support finite differencing and 
                     * laplacian calculations near the boundary.
                     * We want the boundaryIndexSpace of ghosts to loop over. */
                    auto boundary_space 
                        = local_grid.boundaryIndexSpace(Cabana::Grid::Ghost(), 
                              Cabana::Grid::Node(), dir);
                    long min0 = boundary_space.min(0), min1 = boundary_space.min(1);
                    long max0 = boundary_space.max(0), max1 = boundary_space.max(1);
                    Kokkos::parallel_for("Field boundary extrapolation", 
                                         Cabana::Grid::createExecutionPolicy(boundary_space, 
                                         exec_space()),
                                         KOKKOS_LAMBDA(int k, int l) {
                        /* Linear extrapolation from the two points nearest to the boundary. 
                         * XXX - Optimize the following code
                         * XXX - Make this work for any halo distance, not just 2.
                         */
                        int p1[2], p2[2];
                        for (int d = 0; d < dof; d++) {
                            double slope;
                            if (i == -1 && j == 0)
                            {
                                // Top center boundary
                                p1[0] = max0; p1[1] = l;
                                p2[0] = max0+1; p2[1] = l;
                                slope = f(p1[0], p1[1], d) - f(p2[0], p2[1], d);
                                f(k, l, d) = f(p1[0], p1[1], d) + slope * abs(p1[0] - k);
                            }
                            else if (i == 1 && j == 0)
                            {
                                // Bottom center boundary
                                p1[0] = min0-1; p1[1] = l;
                                p2[0] = min0-2; p2[1] = l;
                                slope = f(p1[0], p1[1], d) - f(p2[0], p2[1], d);
                                f(k, l, d) = f(p1[0], p1[1], d) + slope * abs(p1[0] - k);
                            }
                            else if (i == 0 && j == -1)
                            {
                                // Left center boundary
                                p1[0] = k; p1[1] = max1;
                                p2[0] = k; p2[1] = max1+1;
                                slope = f(p1[0], p1[1], d) - f(p2[0], p2[1], d);
                                f(k, l, d) = f(p1[0], p1[1], d) + slope * abs(p1[1] - l);
                            }
                            else if (i == 0 && j == 1)
                            {
                                // Right center boundary
                                p1[0] = k; p1[1] = min1-1;
                                p2[0] = k; p2[1] = min1-2;
                                slope = f(p1[0], p1[1], d) - f(p2[0], p2[1], d);
                                f(k, l, d) = f(p1[0], p1[1], d) + slope * abs(p1[1] - l);
                            }

                            // XXX - the following corner boundary adjustments are hard-coded for a halo width of 2 cells
                            else if (i == -1 && j == -1)
                            {
                                // Top left boundary
                                if (k == l)
                                {
                                    p1[0] = max0; p1[1] = max1;
                                    p2[0] = max0+1; p2[1] = max1+1;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * (max0-k);
                                }
                                else 
                                {
                                    p1[0] = k+2; p1[1] = l+2;
                                    p2[0] = k+3; p2[1] = l+3;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * 2;
                                }
                            }
                            else if (i == -1 && j == 1)
                            {
                                // Top right boundary
                                if ((k == min0 && l == max1-1) ||
                                    (k == min0+1 && l == max1-2))
                                {
                                    p1[0] = max0; p1[1] = min1-1;
                                    p2[0] = max0+1; p2[1] = min1-2;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * (max0-k);
                                }
                                else
                                {
                                    p1[0] = k+2; p1[1] = l-2;
                                    p2[0] = k+3; p2[1] = l-3;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * 2;
                                }
                            }
                            else if (i == 1 && j == -1)
                            {
                                // Bottom left boundary
                                if ((k == min0 && l == max1-1) ||
                                    (k == min0+1 && l == max1-2))
                                {
                                    p1[0] = min0-1; p1[1] = max1;
                                    p2[0] = min0-2; p2[1] = max1+1;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * (k-min0+1);
                                }
                                else
                                {
                                    p1[0] = k-2; p1[1] = l+2;
                                    p2[0] = k-3; p2[1] = l+3;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * 2;
                                }
                            }
                            else if (i == 1 && j == 1)
                            {
                                // Bottom right boundary
                                if (k == l)
                                {
                                    p1[0] = min0-1; p1[1] = min0-1;
                                    p2[0] = min0-2; p2[1] = min0-2;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * (k-min0+1);
                                }
                                else
                                {
                                    p1[0] = k-2; p1[1] = l-2;
                                    p2[0] = k-3; p2[1] = l-3;
                                    slope = (f(p1[0], p1[1], d) - f(p2[0], p2[1], d))/sqrt(2.0);
                                    f(k, l, d) = f(p1[0], p1[1], d) + slope * sqrt(2.0) * 2;
                                }
                            }
                        }
                    });
                }
            }
        }
        Kokkos::fence();
    } 
    
    /* Because we store a position field in the mesh, the position has to
     * be corrected after haloing if it's a periodic boundary */
    template <class MeshType, class ArrayType> 
    void applyPosition(const MeshType &mesh, ArrayType position) const
    {
        using exec_space = typename ArrayType::execution_space;

        auto local_grid = *(mesh.localGrid());

        /* Loop through the directions to correct periodic boundaries */
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) continue;

                std::array<int, 2> dir = {i, j};
                if (isPeriodicBoundary(dir)) {
                    /* For periodic boundaries, the halo exchange takes care of 
                     * most everything *except* the position, which we correct 
                     * here */
                    auto periodic_space = mesh.periodicIndexSpace(Cabana::Grid::Ghost(), 
                        Cabana::Grid::Node(), dir);
                    auto z = position.view();

                    Kokkos::Array<int, 2> kdir = {i, j};
                    Kokkos::Array<double, 2> diff = {(bounding_box[3] - bounding_box[0]),
				                     (bounding_box[4] - bounding_box[1])};
                    Kokkos::parallel_for("Position halo correction", 
                                     Cabana::Grid::createExecutionPolicy(periodic_space, 
                                                                   exec_space()),
                                     KOKKOS_LAMBDA(int k, int l) {
                        /* This subtracts when we're on the low boundary and adds when we're on
                         * the high boundary, which is what we want. */
                        for (int d = 0; d < 2; d++) {
                            z(k, l, d) += kdir[d] * diff[d];
                        }
                    });
                }
            }
        }
        /* Use applyField to correct free boundaries */
        applyField(mesh, position, 3);
    }  


    Kokkos::Array<double, 6> bounding_box;
    Kokkos::Array<int, 4> boundary_type; /* Boundary condition type on all surface edges  */
};

} // namespace Beatnik

#endif // BEATNIK_BOUNDARYCONDITIONS_HPP
