#ifndef BEATNIK_TYPES_HPP
#define BEATNIK_TYPES_HPP

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

} // end namespace Beatnik

#endif // BEATNIK_TYPES_HPP