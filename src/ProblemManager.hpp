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
 * maintained across multiple processes. Specific computational methods 
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
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <SurfaceMesh.hpp>
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

    using Node = Cabana::Grid::Node;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      mem_space>;
    using node_view = 
        typename node_array::view_type;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;
    using surface_mesh_type = SurfaceMesh<exec_space, mem_space>;

    template <class InitFunc>
    ProblemManager( const surface_mesh_type & surface_mesh,
                    const BoundaryCondition & bc, 
                    const double period,
                    const InitFunc& create_functor )
        : _surface_mesh( surface_mesh )
        , _bc( bc )
        , _period( period )
    // , other initializers
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data strutures. Do there need to be version with
        // halos associated with them?
        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( _surface_mesh.localGrid(), 3, Cabana::Grid::Node() );
        auto node_pair_layout =
            Cabana::Grid::createArrayLayout( _surface_mesh.localGrid(), 2, Cabana::Grid::Node() );

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = Cabana::Grid::createArray<double, mem_space>(
            "position", node_triple_layout );
	Cabana::Grid::ArrayOp::assign( *_position, 0.0, Cabana::Grid::Ghost() );

        // 2. The magnitude of vorticity at the interface 
        _vorticity = Cabana::Grid::createArray<double, mem_space>(
            "vorticity", node_pair_layout );
	Cabana::Grid::ArrayOp::assign( *_vorticity, 0.0, Cabana::Grid::Ghost() );

        /* Halo pattern for the position and vorticity. The halo is two cells 
         * deep to be able to do fourth-order central differencing to 
         * compute surface normals accurately. It's a Node (8 point) pattern 
         * as opposed to a Face (4 point) pattern so the vorticity laplacian 
         * can use a 9-point stencil. */
        int halo_depth = _surface_mesh.localGrid()->haloCellWidth();
        _surface_halo = Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<2>(), 
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
        if ( _surface_mesh.rank() == 0 && DEBUG )
            std::cout << "Initializing Mesh State\n";

        // Get Local Grid and Local Mesh
        auto local_grid = *( _surface_mesh.localGrid() );
        auto local_mesh = Cabana::Grid::createLocalMesh<device_type>( local_grid );

	    // Get State Arrays
        auto z = get( Cabana::Grid::Node(), Field::Position() );
        auto w = get( Cabana::Grid::Node(), Field::Vorticity() );

        // Loop Over All Owned Nodes ( i, j )
        auto own_nodes = local_grid.indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                Cabana::Grid::Local() );
        
        int seed = (int) (10000000 * _period);
        Kokkos::Random_XorShift64_Pool<mem_space> random_pool(seed);

        Kokkos::parallel_for(
            "Initialize Cells`",
            Cabana::Grid::createExecutionPolicy( own_nodes, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int index[2] = { i, j };
                double coords[2];
                local_mesh.coordinates( Cabana::Grid::Node(), index, coords);

                create_functor( Cabana::Grid::Node(), Field::Position(), random_pool, index, 
                                coords, z(i, j, 0), z(i, j, 1), z(i, j, 2) );
                create_functor( Cabana::Grid::Node(), Field::Vorticity(), index, 
                                coords, w(i, j, 0), w(i, j, 1) );
            } );
    };

    /**
     * Return mesh
     * @return Returns Mesh object
     **/
    const surface_mesh_type & mesh() const
    {
        return _surface_mesh;
    };

    /**
     * Return Position Field
     * @param Location::Node
     * @param Field::Position
     * @return Returns view of current position at nodes
     **/
    typename node_array::view_type get( Cabana::Grid::Node, Field::Position ) const
    {
        return _position->view();
    };

    /**
     * Return Vorticity Field
     * @param Location::Node
     * @param Field::Vorticity
     * @return Returns view of current vorticity at nodes
     **/
    typename node_array::view_type get( Cabana::Grid::Node, Field::Vorticity ) const
    {
        return _vorticity->view();
    };

    /**
     * Gather State Data from Neighbors
     **/
    void gather( ) const
    {
        _surface_halo->gather( ExecutionSpace(), *_position, *_vorticity );
        _bc.applyPosition(_surface_mesh, *_position);
        _bc.applyField(_surface_mesh, *_vorticity, 2);
    };

    /**
     * Gather state data from neighbors for temporary position and vorticity 
     * arrays managed by other modules 
     */
    void gather( node_array &position, node_array &vorticity) const
    {
        _surface_halo->gather( ExecutionSpace(), position, vorticity );
        _bc.applyPosition(_surface_mesh, position);
        _bc.applyField(_surface_mesh, vorticity, 2);
    }

#if 0
    /**
     * Provide halo pattern used for position and vorticity for classes that
     * need to manage temporary global versions of that state themselves 
     **/
    halo_type & halo( ) const
    {
        return *_surface_halo;
    }
#endif

  private:
    // The mesh on which our data items are stored
    const surface_mesh_type &_surface_mesh;
    const BoundaryCondition &_bc;

    // Used to seed the random number generator
    const double _period;

    // Basic long-term quantities stored in the mesh and periodically written
    // to storage (specific computiontional methods may store additional state)
    std::shared_ptr<node_array> _position, _vorticity;

    // Halo communication pattern for problem-manager stored data
    std::shared_ptr<halo_type> _surface_halo;
}; // template class ProblemManager

} // namespace Beatnik

#endif // BEATNIK_PROBLEMMANAGER_HPP
