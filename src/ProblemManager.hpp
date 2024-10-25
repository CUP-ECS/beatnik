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

#include <StructuredMesh.hpp>
#include <BoundaryCondition.hpp>
#include <Beatnik_ArrayUtils.hpp>

namespace Beatnik
{

/**
 * The ProblemManager Class
 * @class ProblemManager
 * @brief ProblemManager class to store the mesh and global state values, and
 * to perform gathers and scatters
 **/
template <class BeatnikMeshType>
class ProblemManager
{
  public:
    using memory_space = typename BeatnikMeshType::memory_space;
    using execution_space = typename BeatnikMeshType::execution_space;
    using beatnik_mesh_type = BeatnikMeshType;
    using entity_type = typename BeatnikMeshType::entity_type; // Cabana::Grid::Node or NuMesh::Face
    using mesh_type = typename BeatnikMeshType::mesh_type;
    using mesh_type_tag = typename BeatnikMeshType::mesh_type_tag;
    using mesh_array_type = typename BeatnikMeshType::mesh_array_type;
    using halo_type = Cabana::Grid::Halo<memory_space>;

    template <class InitFunc>
    ProblemManager( const BeatnikMeshType & mesh,
                    const BoundaryCondition & bc, 
                    const double period,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        , _bc( bc )
        , _period( period )
        // , other initializers
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data structures. Does there need to be version with
        // halos associated with them?
        auto node_triple_layout = ArrayUtils::createArrayLayout(_mesh.layoutObj(), 3, entity_type());
        auto node_pair_layout = ArrayUtils::createArrayLayout(_mesh.layoutObj(), 2, entity_type());

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = ArrayUtils::createArray<double, memory_space>("position", node_triple_layout);
	    // ArrayUtils::ArrayOp::assign( *_position, 0.0 );


        // 2. The magnitude of vorticity at the interface 
        _vorticity = ArrayUtils::createArray<double, memory_space>("vorticity", node_pair_layout);
	    // ArrayUtils::ArrayOp::assign( *_vorticity, 0.0 );

        /* Halo pattern for the position and vorticity. The halo is two cells 
         * deep to be able to do fourth-order central differencing to 
         * compute surface normals accurately. It's a Node (8 point) pattern 
         * as opposed to a Face (4 point) pattern so the vorticity laplacian 
         * can use a 9-point stencil. */
        /* XXX - For now, only apply strucutred halo to the structured arrays */
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            // Initialize State Values ( position and vorticity ) and 
            // then do a halo to make sure the ghosts and boundaries are correct.
            int halo_depth = _mesh.layoutObj()->haloCellWidth();
            _surface_halo = Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<2>(), 
                                halo_depth, *_position->array(), *_vorticity->array());
            initialize( create_functor );
            gather();
        }
        else 
        {
            throw std::invalid_argument("ProblemManager constructor: Unfinished unstructured implementation.");
        }
    }

    /**
     * Initializes state values in the cells
     * Specific for structured grids
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize( const InitFunctor& create_functor )
    {
        // Get Local Grid and Local Mesh
        auto local_grid = *( _mesh.layoutObj() );
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( local_grid );

	    // Get State Arrays
        auto z = get( Field::Position() )->array()->view();
        auto w = get( Field::Vorticity() )->array()->view();

        // Loop Over All Owned Nodes ( i, j )
        auto own_nodes = local_grid.indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                Cabana::Grid::Local() );
        
        int seed = (int) (10000000 * _period);
        Kokkos::Random_XorShift64_Pool<memory_space> random_pool(seed);

        Kokkos::parallel_for(
            "Initialize Cells`",
            Cabana::Grid::createExecutionPolicy( own_nodes, execution_space() ),
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
    const beatnik_mesh_type & mesh() const
    {
        return _mesh;
    };

    /**
     * Return Position Field
     * @param Location::Node
     * @param Field::Position
     * @return Returns Returns Cabana::Array of current position at nodes
     **/
    std::shared_ptr<mesh_array_type> get( Field::Position ) const
    {
        return _position;
    };

    /**
     * Return Vorticity Field
     * @param Location::Node
     * @param Field::Vorticity
     * @return Returns Cabana::Array of current vorticity at nodes
     **/
    std::shared_ptr<mesh_array_type> get( Field::Vorticity ) const
    {
        return _vorticity;
    };

    /**
     * Gather State Data from Neighbors
     * XXX - only apply to Cabana Arrays
     **/
    void gather( ) const
    {
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            _surface_halo->gather( execution_space(), *_position->array(), *_vorticity->array() );
            _bc.applyPosition(_mesh, *_position->array());
            _bc.applyField(_mesh, *_vorticity->array(), 2);
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            // XXX - TODO
            throw std::invalid_argument("ProblemManager::gather: Not yet implemented for unstructured meshes.");
        }
    }

    /**
     * Gather state data from neighbors for temporary position and vorticity 
     * arrays managed by other modules 
     */
    void gather( mesh_array_type &position, mesh_array_type &vorticity) const
    {
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            _surface_halo->gather( execution_space(), *position.array(), *vorticity.array() );
            _bc.applyPosition(_mesh, *position.array());
            _bc.applyField(_mesh, *vorticity.array(), 2);
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            // XXX - TODO
            throw std::invalid_argument("ProblemManager::gather: Not yet implemented for unstructured meshes.");
        }
    }

  private:
    // The mesh on which our data items are stored
    const beatnik_mesh_type &_mesh;
    const BoundaryCondition &_bc;

    // Used to seed the random number generator
    const double _period;

    // Basic long-term quantities stored in the mesh and periodically written
    // to storage (specific computiontional methods may store additional state)
    std::shared_ptr<mesh_array_type> _position, _vorticity;

    // Halo communication pattern for problem-manager stored data
    std::shared_ptr<halo_type> _surface_halo;

};

} // namespace Beatnik

#endif // BEATNIK_PROBLEMMANAGER_HPP
