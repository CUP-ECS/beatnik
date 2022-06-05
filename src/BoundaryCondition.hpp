/**
 * @file
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

#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

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
    void correctHalo(const MeshType &mesh, ArrayType position, 
               [[maybe_unused]] ArrayType vorticity) const
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
                    auto periodic_space = mesh.periodicIndexSpace(Cajita::Ghost(), 
                        Cajita::Node(), dir);
                    auto z = position.view();

                    /* The halo takes care of vorticity. We have to correct 
                     * the position */
                    int xoff = dir[0], yoff = dir[1];
                    Kokkos::parallel_for("Position halo correction", 
                                     Cajita::createExecutionPolicy(periodic_space, exec_space()),
                                     KOKKOS_LAMBDA(int i, int j) {
                        /* This subtracts when we're on the low boundary and adds when we're on 
                         * the high boundary, which is what we want. */
                        z(i, j, 0) += xoff * (bounding_box[3] - bounding_box[0]);
                        z(i, j, 1) += yoff * (bounding_box[4] - bounding_box[1]);
                    });
                }
            }
        }
    }  

    /* For non-periodic boundaries, we linearly project vorticity and position into
     * the boundary area. */
    template <class MeshType, class ArrayType> 
    void apply(const MeshType &mesh, ArrayType position, 
               [[maybe_unused]] ArrayType vorticity) const
    {
        using exec_space = typename ArrayType::execution_space;

        auto local_grid = *(mesh.localGrid());

        /* Loop through the directions. If it's periodic, we get the periodic index
         * space. If its not periodic, we get the boundary index space */
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) continue;
                std::array<int, 2> dir = {i, j};
                if (isFreeBoundary(dir)) {
                    auto boundary_space = local_grid.boundaryIndexSpace(Cajita::Ghost(), 
                        Cajita::Node(), dir);
                    Kokkos::parallel_for("Position halo correction", 
                                         Cajita::createExecutionPolicy(boundary_space, exec_space()),
                                         KOKKOS_LAMBDA([[maybe_unused]] int i, [[maybe_unused]] int j) {
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
