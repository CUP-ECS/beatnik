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
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Problem manager class that stores mesh state shared between classes and 
 * maintained across multiple processes. Specific compitational methods 
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
#include <BoundaryCondition.hpp>

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
class ProblemManager
{
  public:
    using exec_space = ExecutionSpace;
    using mem_space = MemorySpace;
    using device_type = Kokkos::Device<exec_space, mem_space>;

    using Node = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      device_type>;

    using halo_type = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<exec_space, mem_space>;

    template <class InitFunc>
    ProblemManager( const mesh_type & mesh,
                    const BoundaryCondition & bc, 
                    const InitFunc& create_functor )
        : _mesh( mesh )
        , _bc( bc )
    // , other initializers
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associuated with them?
        auto node_triple_layout =
            Cajita::createArrayLayout( _mesh.localGrid(), 3, Cajita::Node() );
        auto node_pair_layout =
            Cajita::createArrayLayout( _mesh.localGrid(), 2, Cajita::Node() );

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = Cajita::createArray<double, device_type>(
            "position", node_triple_layout );
        Cajita::ArrayOp::assign( *_position, 0.0, Cajita::Ghost() );

        // 2. The magnitude of vorticity at the interface 
        _vorticity = Cajita::createArray<double, device_type>(
            "vorticity", node_pair_layout );
        Cajita::ArrayOp::assign( *_vorticity, 0.0, Cajita::Ghost() );

        /* Halo pattern for the position and vorticity. The halo is two cells 
         * deep so that to be able to do fourth-order central differencing to 
         * compute surface normals accurately. It's a Node (8 point) pattern 
         * as opposed to a Face (4 point) pattern so the vorticity laplacian 
         * can use a 9-point stencil. */
        int halo_depth = _mesh.localGrid()->haloCellWidth();
        _surface_halo = Cajita::createHalo( Cajita::NodeHaloPattern<2>(), 
                            halo_depth, *_position, *_vorticity);

        // Initialize State Values ( position and vorticity ) and 
        // then do a halo to make sure the ghosts and boundaries are correct.
        initialize( create_functor );
        gather();
    }

    /**
     * Initializes state values in the cells
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize( const InitFunctor& create_functor )
    {
        // DEBUG: Trace State Initialization
        if ( _mesh.rank() == 0 && DEBUG )
            std::cout << "Initializing Mesh State\n";

        // Get Local Grid and Local Mesh
        auto local_grid = *( _mesh.localGrid() );
        auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

	// Get State Arrays
        auto z = get( Cajita::Node(), Field::Position() );
        auto w = get( Cajita::Node(), Field::Vorticity() );

        // Loop Over All Owned Nodes ( i, j )
        auto own_nodes = local_grid.indexSpace( Cajita::Own(), Cajita::Node(),
                                                Cajita::Local() );
        Kokkos::parallel_for(
            "Initialize Cells`",
            Cajita::createExecutionPolicy( own_nodes, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int index[2] = { i, j };
                double coords[2];
                local_mesh.coordinates( Cajita::Node(), index, coords);

                create_functor( Cajita::Node(), Field::Position(), index, 
                                coords, z(i, j, 0), z(i, j, 1), z(i, j, 2) );
                create_functor( Cajita::Node(), Field::Vorticity(), index, 
                                coords, w(i, j, 0), w(i, j, 1) );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const mesh_type & mesh() const
    {
        return _mesh;
    };

    /**
     * Return Position Field
     * @param Location::Node
     * @param Field::Position
     * @return Returns view of current position at nodes
     **/
    typename node_array::view_type get( Cajita::Node, Field::Position ) const
    {
        return _position->view();
    };

    /**
     * Return Vorticity Field
     * @param Location::Node
     * @param Field::Vorticity
     * @return Returns view of current vorticity at nodes
     **/
    typename node_array::view_type get( Cajita::Node, Field::Vorticity ) const
    {
        return _vorticity->view();
    };

    /**
     * Gather State Data from Neighbors
     **/
    void gather( ) const
    {
        _surface_halo->gather( ExecutionSpace(), *_position, *_vorticity );
        _bc.applyPositionVorticity(_mesh, *_position, *_vorticity);
    };

    /**
     * Gather state data from neighbors for temporary position and vorticity 
     * arrays managed by other modules 
     */
    void gather( node_array &position, node_array &vorticity) const
    {
        _surface_halo->gather( ExecutionSpace(), position, vorticity );
        _bc.applyPositionVorticity(_mesh, position, vorticity);
    }

#if 0
    /**
     * Provide halo pattern used for position and vorticity for classes that
     * needto manage temporary global versions of that state themselves 
     **/
    halo_type & halo( ) const
    {
        return *_surface_halo;
    }
#endif

  private:
    // The mesh on which our data items are stored
    const mesh_type &_mesh;
    const BoundaryCondition &_bc;

    // Basic long-term quantities stored in the mesh and periodically written
    // to storage (specific computiontional methods may store additional state)
    std::shared_ptr<node_array> _position, _vorticity;

    // Halo communication pattern for problem-manager stored data
    std::shared_ptr<halo_type> _surface_halo;
}; // template class ProblemManager

} // namespace Beatnik

#endif // BEATNIK_PROBLEMMANAGER_HPP
