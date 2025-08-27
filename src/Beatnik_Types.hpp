#ifndef BEATNIK_TYPES_HPP
#define BEATNIK_TYPES_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Tessera_Core.hpp>

#include <type_traits>

namespace Beatnik
{

// Design idea for a Types header taken from the Cabana library:
// https://github.com/ECP-copa/Cabana 

//---------------------------------------------------------------------------//
// Enums
//---------------------------------------------------------------------------//

/**
 * @struct BoundaryType
 * @brief Struct which contains enums of boundary type options for each boundary
 * These are used for setting ghost cell values prior to calculating surface
 * normals
 */
enum MeshBoundaryType {PERIODIC = 0, FREE = 1};

enum BRSolverType {BR_EXACT = 0, BR_CUTOFF = 1};

//---------------------------------------------------------------------------//
// Boundary conditions
//---------------------------------------------------------------------------//
// namespace BoundaryCondition
// {
//     struct Free {};
//     struct Periodic {};
// }

//---------------------------------------------------------------------------//
// Mesh tags
//---------------------------------------------------------------------------//
namespace Mesh
{
    struct Structured {};

    struct Unstructured {};
} // end namespace Mesh


//---------------------------------------------------------------------------//
// Tags designating different orders of the zmodel
//---------------------------------------------------------------------------//
namespace Order
{
    struct Low {};

    struct Medium {};

    struct High {};
} // namespace Order


//---------------------------------------------------------------------------//
// Tags designating different fields of state array entities
//---------------------------------------------------------------------------//
/**
 * @namespace Field
 * @brief Field namespace to track state array entities
 **/
namespace Field
{

    /**
     * @struct Position
     * @brief Tag structure for the position of the surface mesh point in 
     * 3-space
     **/
    struct Position {};

    /**
     * @struct Vorticity
     * @brief Tag structure for the magnitude of vorticity at each surface mesh 
     * point 
     **/
    struct Vorticity {};

}; // end namespace Field

//---------------------------------------------------------------------------//
// Static type checkers
//---------------------------------------------------------------------------//

// Cabana helpers
template <typename T>
using cabana_mesh_type = typename T::mesh_type;

template <typename T>
using is_cabana_mesh = Cabana::Grid::isMeshType<cabana_mesh_type<T>>;
// XXX: Make RHS of 40 to not depend on cabana_mesh_type so cabana_mesh_type can be removed.

template<typename T>
struct dependent_false : std::false_type {};

template <typename T>
struct IsValidValueType
{
    static constexpr bool value =
        std::is_fundamental<T>::value || Tessera::IsCabanaMemberTypes<T>::value;
};

} // end namespace Beatnik

#endif // BEATNIK_TYPES_HPP