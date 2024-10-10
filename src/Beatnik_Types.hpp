#ifndef BEATNIK_TYPES_HPP
#define BEATNIK_TYPES_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

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

enum MeshType {MESH_STRUCTURED = 0, MESH_UNSTRUCTURED = 1};


//---------------------------------------------------------------------------//
// Mesh tags
//---------------------------------------------------------------------------//
namespace Mesh
{

struct Structured {};

struct Unstructured {};

} // end namespace Mesh

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

} // end namespace Beatnik

#endif // BEATNIK_TYPES_HPP