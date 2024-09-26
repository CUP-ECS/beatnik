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
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Array and ArrayLayouts that use Cabana::Grid::Arrays or NuMesh::Arrays depending on
 * the mesh variant.
 * NOTE: Only Cabana::Grid::Node layout types are compatiable with this class.
 */

#ifndef BEATNIK_ARRAY_HPP
#define BEATNIK_ARRAY_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

namespace Beatnik
{
namespace ArrayUtils
{

template <class ExecutionSpace, class MemorySpace, class EntityType>
class ArrayLayout
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using entity_type = EntityType;
    // Define types for Cabana and NuMesh
    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;

    // The variant type that holds either Cabana or NuMesh
    using cabana_array_layout_nt = Cabana::Grid::ArrayLayout<Cabana::Grid::Node, cabana_mesh_t>;
    using numesh_array_layout_vt = NuMesh::Array::ArrayLayout<NuMesh::Vertex, numesh_t>;
    using numesh_array_layout_et = NuMesh::Array::ArrayLayout<NuMesh::Edge, numesh_t>;
    using numesh_array_layout_ft = NuMesh::Array::ArrayLayout<NuMesh::Face, numesh_t>;

    // Constructor that takes either a Cabana or NuMesh object
    template <typename MeshType>
    ArrayLayout(const std::shared_ptr<MeshType>& mesh, const int dofs_per_entity, EntityType tag)
    {
        if constexpr (std::is_same_v<MeshType, cabana_t>)
        {
            _cabana_layout_n = Cabana::Grid::createArrayLayout(mesh, dofs_per_entity, tag);
            _numesh_layout_v = NULL;
            _numesh_layout_e = NULL;
            _numesh_layout_f = NULL;
        }
        else if constexpr (std::is_same_v<MeshType, numesh_t>)
        {
            if constexpr (std::is_same_v<NuMesh::Vertex, entity_type>)
            {
                _cabana_layout_n = NULL;
                _numesh_layout_v = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag); 
                _numesh_layout_e = NULL;
                _numesh_layout_f = NULL;
            }
            else if  constexpr (std::is_same_v<NuMesh::Edge, entity_type>)
            {
                _cabana_layout_n = NULL;
                _numesh_layout_v = NULL;
                _numesh_layout_e = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag);
                _numesh_layout_f = NULL;
            }
            else if  constexpr (std::is_same_v<NuMesh::Face, entity_type>)
            {
                _cabana_layout_n = NULL;
                _numesh_layout_v = NULL;
                _numesh_layout_e = NULL;
                _numesh_layout_f = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag);
            }
        }
        else
        {
            throw std::runtime_error( "Unsupported Beatnik::ArrayUtils::ArrayLayout EntityType!" );
        }
    }

    std::shared_ptr<cabana_array_layout_nt> layout(Cabana::Grid::Node) const
    {
        return _cabana_layout_n;
    }
    std::shared_ptr<numesh_array_layout_vt> layout(NuMesh::Vertex) const
    {
        return _numesh_layout_v;
    }
    std::shared_ptr<numesh_array_layout_et> layout(NuMesh::Edge) const
    {
        return _numesh_layout_e;
    }
    std::shared_ptr<numesh_array_layout_ft> layout(NuMesh::Face) const
    {
        return _numesh_layout_f;
    }

  private:
    std::shared_ptr<cabana_array_layout_nt> _cabana_layout_n;
    std::shared_ptr<numesh_array_layout_vt> _numesh_layout_v;
    std::shared_ptr<numesh_array_layout_et> _numesh_layout_e;
    std::shared_ptr<numesh_array_layout_ft> _numesh_layout_f;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
// Define the Cabana local grid type.
using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
using cabana_grid_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;

// // Define the NuMesh type.
template <class ExecutionSpace, class MemorySpace>
using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
// /*!
//   \brief Cabana version: Create an array layout over the entities of a local grid.
//   \param local_grid The local grid over which to create the layout.
//   \param dofs_per_entity The number of degrees-of-freedom per grid entity.
//   \return Shared pointer to an ArrayLayout.
//   \note EntityType The entity: Cell, Node, Face, or Edge
// */
template <class ExecutionSpace, class MemorySpace, class EntityType>
std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>
createArrayLayout(const std::shared_ptr<Cabana::Grid::LocalGrid<cabana_mesh_t>>& cabana_grid, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>(cabana_grid, dofs_per_entity, tag);
}

template <class ExecutionSpace, class MemorySpace, class EntityType>
std::shared_ptr<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>
createArrayLayout(const std::shared_ptr<numesh_t<ExecutionSpace, MemorySpace>>& mesh, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<ExecutionSpace, MemorySpace, EntityType>>(mesh, dofs_per_entity, tag);
}

//---------------------------------------------------------------------------//
// Array class.
//---------------------------------------------------------------------------//
// template <class ExecutionSpace, class MemorySpace, class Scalar, class EntityType, class MeshType, class... Params>
template <class ExecutionSpace, class MemorySpace, class EntityType>
class Array
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using entity_type = EntityType;
    using value_type = double;
    using cabana_mesh_t = Cabana::Grid::UniformMesh<double, 2>;
    using cabana_t = Cabana::Grid::LocalGrid<cabana_mesh_t>;
    using numesh_t = NuMesh::Mesh<execution_space, memory_space>;
    using layout_t = ArrayLayout<execution_space, memory_space, EntityType>;

    using cabana_array_layout_nt = Cabana::Grid::ArrayLayout<Cabana::Grid::Node, cabana_mesh_t>;
    using cabana_array_nt = Cabana::Grid::Array<double, Cabana::Grid::Node, cabana_mesh_t, memory_space>;

    using numesh_array_layout_vt = NuMesh::Array::ArrayLayout<NuMesh::Vertex, numesh_t>;
    using numesh_array_vt = NuMesh::Array::Array<double, NuMesh::Vertex, numesh_t, memory_space>;
    using numesh_array_layout_et = NuMesh::Array::ArrayLayout<NuMesh::Edge, numesh_t>;
    using numesh_array_et = NuMesh::Array::Array<double, NuMesh::Edge, numesh_t, memory_space>;
    using numesh_array_layout_ft = NuMesh::Array::ArrayLayout<NuMesh::Face, numesh_t>;
    using numesh_array_ft = NuMesh::Array::Array<double, NuMesh::Face, numesh_t, memory_space>;

    // Constructor that takes either a Cabana or NuMesh object
    template <typename LayoutType>
    Array(const std::string& label, const std::shared_ptr<LayoutType>& array_layout, EntityType entity_type)
        : _label( label )
        , _layout( array_layout )
    {
        auto layout = array_layout->layout(entity_type);

        if constexpr (std::is_same_v<EntityType, Cabana::Grid::Node>)
        {
            _cabana_array_n = Cabana::Grid::createArray<double, memory_space>(label, layout);
            _numesh_array_v = NULL;
            _numesh_array_e = NULL;
            _numesh_array_f = NULL;
        }
        else if  constexpr (std::is_same_v<EntityType, NuMesh::Vertex>)
        {
            _cabana_array_n = NULL;
            _numesh_array_v = NuMesh::Array::createArray<double, memory_space>(label, layout);
            _numesh_array_e = NULL;
            _numesh_array_f = NULL;
        }
        else if  constexpr (std::is_same_v<EntityType, NuMesh::Edge>)
        {
            _cabana_array_n = NULL;
            _numesh_array_v = NULL;
            _numesh_array_e = NuMesh::Array::createArray<double, memory_space>(label, layout);
            _numesh_array_f = NULL;
        }
        else if  constexpr (std::is_same_v<EntityType, NuMesh::Face>)
        {
            _cabana_array_n = NULL;
            _numesh_array_v = NULL;
            _numesh_array_e = NULL;
            _numesh_array_f = NuMesh::Array::createArray<double, memory_space>(label, layout);
        }
        else
        {
            throw std::runtime_error( "Unsupported Beatnik::ArrayUtils::Array EntityType!" );
        }
    }

    // Getters
    std::shared_ptr<cabana_array_nt> array(Cabana::Grid::Node) const {return _cabana_array_n;}
    std::shared_ptr<numesh_array_vt> array(NuMesh::Vertex) const {return _numesh_array_v;}
    std::shared_ptr<numesh_array_et> array(NuMesh::Edge) const {return _numesh_array_e;}
    std::shared_ptr<numesh_array_ft> array(NuMesh::Face) const {return _numesh_array_f;}

    std::shared_ptr<layout_t> layout() const {return _layout;}
    std::string label() const {return _label;}

  private:
    // Array pointers
    std::shared_ptr<cabana_array_nt> _cabana_array_n;
    std::shared_ptr<numesh_array_vt> _numesh_array_v;
    std::shared_ptr<numesh_array_et> _numesh_array_e;
    std::shared_ptr<numesh_array_ft> _numesh_array_f;

    // Layout pointers
    std::shared_ptr<layout_t> _layout;
    std::string _label;
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
  \return Shared pointer to an Array.
*/
// template <class LayoutType, class Scalar, class... Params>
template <class ExecutionSpace, class MemorySpace, class LayoutType, class EntityType>
std::shared_ptr<Array<ExecutionSpace, MemorySpace, EntityType>>
createArray(const std::string& label,
            const std::shared_ptr<LayoutType>& layout,
            EntityType entity_type)
{
    return std::make_shared<Array<ExecutionSpace, MemorySpace, EntityType>>(
        label, layout, entity_type);
}

//---------------------------------------------------------------------------//
// Array operations.
//---------------------------------------------------------------------------//
namespace ArrayOp
{

template <class ExecutionSpace, class MemorySpace, class EntityType>
std::shared_ptr<Array<ExecutionSpace, MemorySpace, EntityType>>
clone( const Array<ExecutionSpace, MemorySpace, EntityType>& array )
{
    return createArray<ExecutionSpace, MemorySpace>( array.label(), array.layout(), EntityType() );
}

template <class Array_t, class DecompositionTag>
void copy( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::copy(*a.array(entity_type()), *b.array(entity_type()), tag);
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::copy(*a.array(entity_type()), *b.array(entity_type()), tag);
    }

}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> cloneCopy( const Array_t& array, DecompositionTag tag )
{
    auto cln = clone( array );
    copy( *cln, array, tag );
    return cln;
}

template <class Array_t, class DecompositionTag>
void assign( Array_t& array, const double alpha,
             DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::assign(*array.array(entity_type()), alpha, tag);
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::assign(*array.array(entity_type()), alpha, tag);
    }
}

template <class Array_t, class DecompositionTag>
void update( Array_t& a, const double alpha, const Array_t& b,
        const double beta, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::update(*a.array(entity_type()), alpha, *b.array(entity_type()), beta, tag);
    }
     else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::update(*a.array(entity_type()), alpha, *b.array(entity_type()), beta, tag);
    }
}

template <class Array_t, class DecompositionTag>
void update( Array_t& a, const double alpha, const Array_t& b,
        const double beta, const Array_t& c,
        const double gamma, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::update(*a.array(entity_type()), alpha, *b.array(entity_type()), beta, *c.array(entity_type()), gamma, tag);
    }
     else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::update(*a.array(entity_type()), alpha, *b.array(entity_type()), beta, *c.array(entity_type()), gamma, tag);
    }
}

} // end namespace ArrayOp

} // end namespace Array

} // end namespace Beatnik

#endif // BEATNIK_ARRAY_HPP