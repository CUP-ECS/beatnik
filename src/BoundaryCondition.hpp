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
#include <ProblemManager.hpp>

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
    FREE_SLIP = 1,
    NO_SLIP = 2, 
};

/**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with
 * Kokkos inline Functions
 */

struct BoundaryCondition
{
    using Node = Cajita::Node;
    using Position = Field::Position;

    // The functor operator applies position boundary conditions. Note that the
    // maxes in the bounding box are in terms of global node indexes.
    template <class PType>
    KOKKOS_INLINE_FUNCTION void operator()( Node, Position, PType& p, 
                                            const int gi, const int gj,
                                            const int i, const int j ) const
    {
    }

    Kokkos::Array<int, 4>
        boundary_type; /**< Boundary condition type on all surface edges  */
    Kokkos::Array<int, 2> min;
    Kokkos::Array<int, 2> max;
};

} // namespace Beatnik

#endif // BEATNIK_BOUNDARYCONDITIONS_HPP
