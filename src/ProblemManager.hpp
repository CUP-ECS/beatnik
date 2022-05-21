/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Problem manager class that stores mesh state shared between classes and 
 * maintained across multiple processes. Note that specific compitational methods 
 * may also have mesh state, but if it is not shared between classes it resides
 * in that method, not here.
 */

#ifndef BEATNIK_PROBLEMMANAGER_HPP
#define BEATNIK_PROBLEMMANAGER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Mesh.hpp>

namespace Beatnik
{

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
struct Position
{
};

/**
 * @struct Vorticity
 * @brief Tag structure for the magnitude of vorticity at each surface mesh 
 * point 
 **/
struct Vorticity
{
};

}; // end namespace Field

/**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and global state values, and
 * to perform gathers and scatters
 **/
template <class ExecutionSpace, class MemorySpace>
class ProblemManager;

/* The 2D implementation of hte problem manager class */
template <class ExecutionSpace, class MemorySpace>
class ProblemManager<ExecutionSpace, MemorySpace>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using Node = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      MemorySpace>;

    // Meaningless type for now until we have 3D support in.
    using halo_type = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<ExecutionSpace, MemorySpace>;

    template <class InitFunc>
    ProblemManager( const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor )
        : _mesh( mesh )
    // , other initializers
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associuated with them?
        auto node_triple_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 3, Cajita::Node() );
        auto node_pair_layout =
            Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Node() );

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = Cajita::createArray<double, MemorySpace>(
            "position", node_triple_layout );
        Cajita::ArrayOp::assign( *_position, 0.0, Cajita::Ghost() );

        // 2. The magnitude of vorticity at the interface 
        _vorticity = Cajita::createArray<double, MemorySpace>(
            "vorticity", node_pair_layout );
        Cajita::ArrayOp::assign( *_vorticity, 0.0, Cajita::Ghost() );

        /* Halo pattern for the position. The halo is two cells deep so that 
         * to be able to do fourth-order central differencing to compute 
         * surface normals. */
        int halo_depth = _mesh->localGrid()->haloCellWidth();
        _surface_halo =
            Cajita::createHalo( Cajita::NodeHaloPattern<2>(), halo_depth,
                                _position, _vorticity);

        // Initialize State Values ( quantity and velocity )
        initialize( create_functor );
    }

    /**
     * Initializes state values in the cells
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize( const InitFunctor& create_functor )
    {
        // DEBUG: Trace State Initialization
        if ( _mesh->rank() == 0 && DEBUG )
            std::cout << "Initializing Cell Fields\n";

        // Get Local Grid and Local Mesh
        auto local_grid = *( _mesh->localGrid() );
        double cell_size = _mesh->cellSize();

	// Get State Arrays
        auto z = get( Cajita::Node(), Field::Position() );
        auto w = get( Cajita::Node(), Field::Vorticity() );

        // Loop Over All Owned Nodes ( i, j )
        auto own_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Node(),
                                                Cajita::Local() );
        int index[2] = { 0, 0 };
        double loc[2]; // x/y loocation of the cell at 0, 0
        Kokkos::parallel_for(
            "Initialize Cells`",
            Cajita::createExecutionPolicy( own_cells, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                // Initialization Function
                create_functor( Cajita::Node(), Field::Position(), i, j, z );
                create_functor( Cajita::Node(), Field::Vorticity(), i, j, w );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const std::shared_ptr<Mesh<2, ExecutionSpace, MemorySpace>>& mesh() const
    {
        return _mesh;
    };

    /**
     * Return Position Field
     * @param Location::Node
     * @param Field::Position
     * @return Returns view of current position at nodes
     **/
    typename cell_array::view_type get( Cajita::Node, Field::Position ) const
    {
        return _position->view();
    };

    /**
     * Return Vorticity Field
     * @param Location::Node
     * @param Field::Vorticity
     * @return Returns view of current vorticity at nodes
     **/
    typename cell_array::view_type get( Cajita::Node, Field::Vorticity ) const
    {
        return _vorticity->view();
    };

    /**
     * Gather State Data from Neighbors
     * @param Version
     **/
    void gather( ) const
    {
        _surface_halo->gather( ExecutionSpace(), _position, _vorticity );
    };

  private:
    // The mesh on which our data items are stored
    std::shared_ptr<mesh_type> _mesh;

    // Basic long-term quantities stored in the mesh and periodically written
    // to storage (specific computiontional methods may store additional state)
    std::shared_ptr<node_array> _position, _vorticity;

    // Halo communication pattern for problem-manager stored data
    std::shared_ptr<halo_type> _surface_halo;
};

} // namespace Beatnik

#endif // BEATNIK_PROBLEMMANAGER_HPP
