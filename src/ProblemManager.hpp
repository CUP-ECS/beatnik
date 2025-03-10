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
    using base_triple_type = typename BeatnikMeshType::base_triple_type;
    using base_pair_type = typename BeatnikMeshType::base_pair_type;
    using triple_array_type = typename BeatnikMeshType::triple_array_type;
    using pair_array_type = typename BeatnikMeshType::pair_array_type;
    using halo_type = Cabana::Grid::Halo<memory_space>;

    template <class InitFunc>
    ProblemManager( const BeatnikMeshType& mesh,
                    const BoundaryCondition& bc, 
                    const double period,
                    const InitFunc& create_functor )
        : _mesh( mesh )
        , _bc( bc )
        , _period( period )
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data structures. Does there need to be version with
        // halos associated with them?
        auto node_triple_layout = ArrayUtils::createArrayLayout<base_triple_type>(_mesh.layoutObj(), 3, entity_type());
        auto node_pair_layout = ArrayUtils::createArrayLayout<base_pair_type>(_mesh.layoutObj(), 2, entity_type());

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = ArrayUtils::createArray<memory_space>("position", node_triple_layout);
	    // ArrayUtils::ArrayOp::assign( *_position, 0.0 );


        // 2. The magnitude of vorticity at the interface 
        _vorticity = ArrayUtils::createArray<memory_space>("vorticity", node_pair_layout);
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
            initialize_structured_mesh( create_functor );
            int halo_depth = _mesh.layoutObj()->haloCellWidth();
            _surface_halo = Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<2>(), 
                                halo_depth, *_position->array(), *_vorticity->array());
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            initialize_unstructured_mesh_grid( create_functor );
            // auto zaosoa = get( Field::Position() )->array()->aosoa();
            // auto waosoa = get( Field::Vorticity() )->array()->aosoa();
            // auto zslice = Cabana::slice<0>(zaosoa);
            // auto wslice = Cabana::slice<0>(waosoa);
            // throw std::invalid_argument("ProblemManager constructor: Unfinished unstructured implementation.");
        }

        gather();
    }

    /**
     * Constructor if point coordinates on the mesh are already known and can be passed-in
     */
    template <class PositionsAoSoA>
    ProblemManager( const BeatnikMeshType& mesh, const BoundaryCondition& bc,
                    const PositionsAoSoA& positions_in)
        : _mesh( mesh )
        , _bc( bc )   // Unused
        , _period( -1 ) // Unused
    {
        // The layouts of our various arrays for values on the staggered mesh
        // and other associated data structures. Does there need to be version with
        // halos associated with them?
        auto node_triple_layout = ArrayUtils::createArrayLayout<base_triple_type>(_mesh.layoutObj(), 3, entity_type());
        auto node_pair_layout = ArrayUtils::createArrayLayout<base_pair_type>(_mesh.layoutObj(), 2, entity_type());

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        _position = ArrayUtils::createArray<memory_space>("position", node_triple_layout);
	    // ArrayUtils::ArrayOp::assign( *_position, 0.0 );


        // 2. The magnitude of vorticity at the interface 
        _vorticity = ArrayUtils::createArray<memory_space>("vorticity", node_pair_layout);
	    // ArrayUtils::ArrayOp::assign( *_vorticity, 0.0 );

        /* Halo pattern for the position and vorticity. The halo is two cells 
         * deep to be able to do fourth-order central differencing to 
         * compute surface normals accurately. It's a Node (8 point) pattern 
         * as opposed to a Face (4 point) pattern so the vorticity laplacian 
         * can use a 9-point stencil. */
        /* XXX - For now, only apply strucutred halo to the structured arrays */
       
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            throw std::runtime_error(
                "ProblemManager: This constructor does not yet support structured meshes");
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            initialize_unstructured_mesh_sphere(positions_in);
        }

        gather();
    }

    /**
     * Initializes state values in the cells
     * Specific for structured grids
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize_structured_mesh( const InitFunctor& create_functor )
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
            "Initialize Cells",
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
     * Initializes state values in the cells
     * 
     * @param create_functor Initialization function
     **/
    template <class InitFunctor>
    void initialize_unstructured_mesh_grid( const InitFunctor& create_functor )
    {
        // Get Local Grid and Local Mesh
        auto mesh = _mesh.layoutObj();
        auto local_grid = _mesh.localGrid();
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *local_grid );

        // Get State Arrays. These are AoSoA slices in the unstructured version
        auto zaosoa = get( Field::Position() )->array()->aosoa();
        auto waosoa = get( Field::Vorticity() )->array()->aosoa();
        auto zslice = Cabana::slice<0>(*zaosoa);
        auto wslice = Cabana::slice<0>(*waosoa);

        /**
         * The initialization functions work in a 2D domain. To properly set the values
         * in the unstructured mesh data structures, which are vectors, we must initialize
         * dummy 2D arrays and then copy the values into the vectors, mapping
         * dummy(i, j, x) to mesh(i, x) by copying how the mesh is initialized from a
         * 2D array.
         */
        auto z_layout = Cabana::Grid::createArrayLayout(local_grid, 3, Cabana::Grid::Node());
        auto w_layout = Cabana::Grid::createArrayLayout(local_grid, 2, Cabana::Grid::Node());
        auto z_array = Cabana::Grid::createArray<double, memory_space>("z_for_initialization", z_layout);
        auto w_array = Cabana::Grid::createArray<double, memory_space>("w_for_initialization", w_layout);
        auto z = z_array->view();
        auto w = w_array->view();

        // Loop Over All Owned Nodes ( i, j )
        auto own_nodes = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
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

        // Copy the values in the 2D arrays into the correct indices in the slices
        int istart = own_nodes.min(0), jstart = own_nodes.min(1);
        int iend = own_nodes.max(0), jend = own_nodes.max(1);
        Kokkos::parallel_for("populate_vertex_data", Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
            KOKKOS_LAMBDA(int i, int j) {

            // Same vertex LID calculation as in NuMesh
            int v_lid = (i - istart) * (jend - jstart) + (j - jstart);

            for (int dim = 0; dim < 3; dim++)
                zslice(v_lid, dim) = z(i, j, dim);
            for (int dim = 0; dim < 2; dim++)
                wslice(v_lid, dim) = w(i, j, dim);
        });
        
    };

    /**
     * Initializes state values in the cells, from a given input
     * 
     * @param create_functor Initialization function
     **/
    template <class PositionsAoSoA>
    void initialize_unstructured_mesh_sphere( const PositionsAoSoA& positions_in )
    {
        // Get State Arrays. These are AoSoA slices in the unstructured version
        auto zaosoa = get( Field::Position() )->array()->aosoa();
        auto waosoa = get( Field::Vorticity() )->array()->aosoa();
        auto zslice = Cabana::slice<0>(*zaosoa);
        auto wslice = Cabana::slice<0>(*waosoa);
        auto positions_in_slice = Cabana::slice<0>(positions_in);

        // Fill positions array with positions_in
        assert(zaosoa->size() == positions_in.size());
        assert(waosoa->size() == positions_in.size());
        Kokkos::parallel_for("Initialize Cells", Kokkos::RangePolicy<execution_space>(0, positions_in.size()),
            KOKKOS_LAMBDA( const int i ) {
            for (int j = 0; j < 3; j++) 
                zslice(i, j) = positions_in_slice(i, j);
            for (int j = 0; j < 2; j++) 
                wslice(i, j) = 0.0;
        });
        
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
    std::shared_ptr<triple_array_type> get( Field::Position ) const
    {
        return _position;
    };

    /**
     * Return Vorticity Field
     * @param Location::Node
     * @param Field::Vorticity
     * @return Returns Cabana::Array of current vorticity at nodes
     **/
    std::shared_ptr<pair_array_type> get( Field::Vorticity ) const
    {
        return _vorticity;
    };

    /**
     * Gather State Data from Neighbors
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
            // We must remake the halo each time to ensure it stays up-to-date with the mesh
            auto numesh_halo = NuMesh::createHalo(_mesh.layoutObj(), 0, 1, NuMesh::Vertex());
            _position->array()->update();
            NuMesh::gather(numesh_halo, _position->array());
            _vorticity->array()->update();
            NuMesh::gather(numesh_halo, _vorticity->array());
        }
    }

    /**
     * Gather state data from neighbors for temporary position and vorticity 
     * arrays managed by other modules 
     */
    void gather( triple_array_type &position, pair_array_type &vorticity) const
    {
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            _surface_halo->gather( execution_space(), *position.array(), *vorticity.array() );
            _bc.applyPosition(_mesh, *position.array());
            _bc.applyField(_mesh, *vorticity.array(), 2);
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            // We must remake the halo each time to ensure it stays up-to-date with the mesh
            int halo_level = _mesh.layoutObj()->global_min_max_tree_depth();
            auto numesh_halo = NuMesh::createHalo(_mesh.layoutObj(), halo_level, 1, NuMesh::Vertex());
            _position->array()->update();
            NuMesh::gather(numesh_halo, _position->array());
            _vorticity->array()->update();
            NuMesh::gather(numesh_halo, _vorticity->array());
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
    std::shared_ptr<pair_array_type> _vorticity;
    std::shared_ptr<triple_array_type> _position;

    // Halo communication pattern for problem-manager stored data
    std::shared_ptr<halo_type> _surface_halo;

};

} // namespace Beatnik

#endif // BEATNIK_PROBLEMMANAGER_HPP
