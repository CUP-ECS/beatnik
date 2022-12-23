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
#define DEBUG 1
#endif

// Include Statements

#include <Mesh.hpp>

#include <Kokkos_Core.hpp>
#include "Operators.hpp"

namespace Beatnik
{
/**
 * @struct BoundaryType
 * @brief Struct which contains enums of boundary type options for each boundary
 * These are used for setting ghost cell values prior to calculating surface
 * normals
 */
enum BoundaryType
{
    PERIODIC = 0,
    FREE = 1,
};

/**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with
 * Kokkos inline Functions
 */

struct BoundaryCondition
{
    bool isPeriodicBoundary(std::array<int, 2> dir) const
    {
        if ((dir[0] == -1) && (boundary_type[0] == PERIODIC))
            return true;
        if ((dir[0] == 1) && (boundary_type[2] == PERIODIC))
            return true;
        if ((dir[1] == -1) && (boundary_type[1] == PERIODIC))
            return true;
        if ((dir[1] == 1) && (boundary_type[3] == PERIODIC))
            return true;
        return false;
    }

    bool isFreeBoundary(std::array<int, 2> dir) const
    {
        if ((dir[0] == -1) && (boundary_type[0] == FREE))
            return true;
        if ((dir[0] == 1) && (boundary_type[2] == FREE))
            return true;
        if ((dir[1] == -1) && (boundary_type[1] == FREE))
            return true;
        if ((dir[1] == 1) && (boundary_type[3] == FREE))
            return true;
        return false;
    }

    /* Because we store a position field in the mesh, the position has to
     * be corrected after haloing if it's a periodic boundary */
    template <class MeshType, class ArrayType> 
    void applyPosition(const MeshType &mesh, ArrayType position) const
    {
        using exec_space = typename ArrayType::execution_space;

        auto local_grid = *(mesh.localGrid());

        /* Loop through the directions. If it's periodic, we get the periodic index
         * space. If its not periodic, we get the boundary index space */
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) continue;

                std::array<int, 2> dir = {i, j};
                if (isPeriodicBoundary(dir)) {
                    /* For periodic boundaries, the halo exchange takes care of 
                     * most everything *except* the position, which we correct 
                     * here */
                    auto periodic_space = mesh.periodicIndexSpace(Cajita::Ghost(), 
                        Cajita::Node(), dir);
                    auto z = position.view();

                    Kokkos::Array<int, 2> kdir = {i, j};
                    Kokkos::Array<double, 2> diff = {(bounding_box[3] - bounding_box[0]),
				                     (bounding_box[4] - bounding_box[1])};
                    Kokkos::parallel_for("Position halo correction", 
                                     Cajita::createExecutionPolicy(periodic_space, 
                                                                   exec_space()),
                                     KOKKOS_LAMBDA(int k, int l) {
                        /* This subtracts when we're on the low boundary and adds when we're on
                         * the high boundary, which is what we want. */
                        for (int d = 0; d < 2; d++) {
                            z(k, l, d) += kdir[d] * diff[d];
                        }
                    });
                } else if (isFreeBoundary(dir)) {
                    /* For free boundaries, we have to extrapolate from the mesh
                     * into the boundary to support finite diffrencing and 
                     * laplacian calculations near the boundary. */
		    
                    auto boundary_space = local_grid.boundaryIndexSpace(Cajita::Ghost(), 
                        Cajita::Node(), dir);

                    auto z = position.view();
                    Kokkos::Array<int, 2> kdir = {i, j};
                    Kokkos::parallel_for("Position boundary extrapolation", 
                                         Cajita::createExecutionPolicy(boundary_space, 
                                         exec_space()),
                                         KOKKOS_LAMBDA(int k, int l) {

			/* Find the two points in the interior we want to 
			 * extrapolate from based on the direction and how far 
                         * we are from the interior.  
                         * 
                         * XXX Right now we always go two points aways since
                         * we have a 2-deep halo. This guarantees to get us out
                         * of the boundary, but may take us further into the the
                         * mesh than we want. We should instead Figuring out dist 
                         * to go just to the edge of the boundary and linearly 
                         * extrapolate from that. XXX */
			int p1[2], p2[2];
                        int dist = 2; 
                        p1[0] = k - kdir[0]*(dist); 
                        p1[1] = l - kdir[1]*(dist); 
                        p2[0] = k - kdir[0]*(dist + 1); 
                        p2[1] = l - kdir[1]*(dist + 1); 
                        for (int d = 0; d < 3; d++) {
			    z(k, l, d) = z(p1[0], p1[1], d) 
                                         + dist*(z(p2[0], p2[1], d) 
                                                 - z(p1[0], p1[1], d));
                        }
                    });
                }
            }
        }
    }  

    /* Apply boundary conditions to a generic field with DOF elements that doesn't
     * require special handling */
    template <class MeshType, class ArrayType> 
    void applyField(const MeshType &mesh, ArrayType field, int dof) const
    {
        using exec_space = typename ArrayType::execution_space;

        auto local_grid = *(mesh.localGrid());

        /* Loop through the directions. If it's periodic, we get the periodic index
         * space. If its not periodic, we get the boundary index space */
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) continue;

                std::array<int, 2> dir = {i, j};
		/* For periodic boundaries, nothing needs doing for general fields */

                /* For free boundaries, we linearl extrapolate the field into the 
                 * boundary */
                if (isFreeBoundary(dir)) {
                    /* For free boundaries, we have to extrapolate from the mesh
                     * into the boundary to support finite diffrencing and laplacian
                     * calculations near the boundary. */
		    
                    auto boundary_space = local_grid.boundaryIndexSpace(Cajita::Ghost(), 
                        Cajita::Node(), dir);
                    auto f = field.view();
                    Kokkos::Array<int, 2> kdir = {i, j};
                    Kokkos::parallel_for("Field boundary extrapolation", 
                                         Cajita::createExecutionPolicy(boundary_space, 
                                         exec_space()),
                                         KOKKOS_LAMBDA(int k, int l) {
			/* Find the two points in the interior we want to 
			 * extrapolate from based on the direction and how far 
                         * we are from the interior.  
                         * 
                         * XXX Right now we always go two points aways since
                         * we have a 2-deep halo. This guarantees to get us out
                         * of the boundary, but may take us further into the the
                         * mesh than we want. We should instead Figuring out dist 
                         * to go just to the edge of the boundary and linearly 
                         * extrapolate from that. XXX */
                        int p1[2], p2[2];
                        int dist = 2; 
                        p1[0] = k - kdir[0]*(dist); 
                        p1[1] = l - kdir[1]*(dist); 
                        p2[0] = k - kdir[0]*(dist + 1); 
                        p2[1] = l - kdir[1]*(dist + 1); 
                        for (int d = 0; d < dof; d++) {
			    f(k, l, 0) = f(p1[0], p1[1], d) 
                                         + dist*(f(p2[0], p2[1], d) 
                                                     - f(p1[0], p1[1], d));
                        }
                    });
                }
            }
        }
    } 

    Kokkos::Array<double, 6> bounding_box;
    Kokkos::Array<int, 4> boundary_type; /**< Boundary condition type on all surface edges  */
};

} // namespace Beatnik

#endif // BEATNIK_BOUNDARYCONDITIONS_HPP
