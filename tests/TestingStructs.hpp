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
enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};
enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};
enum BRSolverType {BR_EXACT = 0, BR_CUTOFF};
enum BoundaryType {PERIODIC = 0, FREE = 1};
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


/**
 * @struct Params
 * @brief Holds order and solver-specific parameters
 */
struct Params
{
    /* Save the period from command-line args to pass to 
     * ProblemManager to seed the random number generator
     * to initialize position
     */
    double period;

    /* Mesh data, for solvers that create another mesh */
    std::array<double, 6> global_bounding_box;
    std::array<bool, 2> periodic;

    /* Model Order */
    int solver_order;

    /* BR solver type */
    Enums::BRSolverType br_solver;

    /* Cutoff distance for cutoff-based BRSolver */
    double cutoff_distance;

    /* Heffte configuration options for low-order model: 
        Value	All-to-all	Pencils	Reorder
        0	    False	    False	False
        1	    False	    False	True
        2	    False	    True	False
        3	    False	    True	True
        4	    True	    False	False
        5	    True	    False	True
        6	    True	    True	False (Default)
        7	    True	    True	True
    */
    int heffte_configuration;
};


/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    // std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain: MOVED TO PARAMS */
    enum Enums::InitialConditionModel initial_condition; /**< Model used to set initial conditions */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of scale of initial interface */
    double variation; /**< Variation in scale of initial interface */
    double period;   /**< Period of initial variation in interface */
    enum Enums::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */
    int model;      /**< Model used to set initial conditions */
    double bounding_box; /**< Size of global bounding box. From (-B, -B, -B) to (B, B, B) */

    /* Problem simulation parameters */
    std::array<int, 2> num_nodes;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */
    int weak_scale;     /**< Amount to scale up resulting problem */

    /* I/O parameters */
    char *indir;        /**< Where to read initial conditions from */
    char *outdir;       /**< Directory to write output to */
    int write_freq;     /**< Write frequency */

    /* Solution method constants */
    enum Enums::BRSolverType br_solver; /**< BRSolver to use */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */

    /* Parameters specific to solver order and BR solver type:
     *  - Period for particle initialization
     *  - Global bounding box
     *  - Periodicity
     *  - Heffte configuration (For low-order solver)
     *  - solver order (Order of z-model solver to use)
     *  - BR solver type
     *  - Cutoff distance (If using cutoff solver)
     */
    Params params;
};


} // end namespace Structs

} // end namespace BeatnikTest

#endif // _TESTING_STRUCTS_HPP_