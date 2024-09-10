#ifndef _TESTING_STRUCTS_HPP_
#define _TESTING_STRUCTS_HPP_

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <TestingUtils.hpp>

namespace BeatnikTest
{

namespace Enums
{

} // end namespace Enums

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
    
namespace Structs
{





} // end namespace Structs

} // end namespace BeatnikTest

#endif // _TESTING_STRUCTS_HPP_