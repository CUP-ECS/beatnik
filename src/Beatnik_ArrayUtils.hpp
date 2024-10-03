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

#ifndef BEATNIK_ARRAYUTILS_HPP
#define BEATNIK_ARRAYUTILS_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include <type_traits>

namespace Beatnik
{
namespace ArrayUtils
{

// Cabana helpers
using cabana_mesh_type = Cabana::Grid::UniformMesh<double, 2>;

template <typename T>
using is_cabana_uniform_mesh = std::is_same<T, Cabana::Grid::Node>;
// XXX: Make RHS of 40 to not depend on cabana_mesh_type so cabana_mesh_type can be removed.


template<typename T>
struct dependent_false : std::false_type {};

template <class MeshType, class EntityType>
class ArrayLayout
{
  public:
    /**
     * NuMesh: MeshType = NuMesh::Mesh<ExecutionSpace, MemSpace>
     * Cabana:: MeshType = Cabana::Grid::LocalGrid<cabana_mesh_type>
     */
    using mesh_type = MeshType;
    using entity_type = EntityType;

    // Determine ContainerLayoutType using std::conditional_t
    using array_layout_type = std::conditional_t<
        is_cabana_uniform_mesh<entity_type>::value,
        Cabana::Grid::ArrayLayout<entity_type, cabana_mesh_type>, // Case A: Cabana UniformMesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<MeshType>::value,
            NuMesh::Array::ArrayLayout<entity_type, mesh_type>, // Case B: NuMesh Mesh
            void // Fallback type or an error type if neither condition is satisfied
        >
    >;
  
    ArrayLayout(const std::shared_ptr<mesh_type>& mesh, const int dofs_per_entity, entity_type tag)
    {
        if constexpr (is_cabana_uniform_mesh<entity_type>::value)
        {
            _layout = Cabana::Grid::createArrayLayout(mesh, dofs_per_entity, tag);
        }
        else if constexpr (NuMesh::is_numesh_mesh<MeshType>::value)
        {
            if constexpr (std::is_same_v<NuMesh::Vertex, entity_type>)
            {
                _layout = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag); 
            }
            else if  constexpr (std::is_same_v<NuMesh::Edge, entity_type>)
            {
                _layout = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag);
            }
            else if  constexpr (std::is_same_v<NuMesh::Face, entity_type>)
            {
                _layout = NuMesh::Array::createArrayLayout(mesh, dofs_per_entity, tag);
            }
        }
        else
        { 	// TBH, you might want this to be a compile time error instead
            static_assert(dependent_false<entity_type>::value, "Unsupported Beatnik::ArrayUtils::ArrayLayout EntityType!");
        }
    }

    std::shared_ptr<array_layout_type> layout() const
    {
        return _layout;
    }
	
	static constexpr bool isArrayLayout()
	{
		return true;
	}

  private:
    std::shared_ptr<array_layout_type> _layout;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
template <class MeshType, class EntityType>
std::shared_ptr<ArrayLayout<MeshType, EntityType>>
createArrayLayout(const std::shared_ptr<MeshType>& mesh, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<MeshType, EntityType>>(mesh, dofs_per_entity, tag);
}

//---------------------------------------------------------------------------//
// Array class
//---------------------------------------------------------------------------//
template <class ContainerLayoutType, class Scalar, class MemorySpace>
class Array
{
    static_assert(ContainerLayoutType::isArrayLayout(), "ContainerLayoutType must be a valid array layout.");
  
  public:
    using entity_type = typename ContainerLayoutType::entity_type;
    using mesh_type   = typename ContainerLayoutType::mesh_type;
    using layout_type = typename ContainerLayoutType::array_layout_type;
    using value_type = Scalar;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    using container_layout_type = ContainerLayoutType;

    // Determine array_type using std::conditional_t
    using array_type = std::conditional_t<
        is_cabana_uniform_mesh<entity_type>::value,
        Cabana::Grid::Array<value_type, entity_type, cabana_mesh_type, memory_space>, // Case A: Cabana Mesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<mesh_type>::value,
            NuMesh::Array::Array<value_type, entity_type, mesh_type, memory_space>, // Case B: NuMesh Mesh
            void // Fallback or error type if neither condition is satisfied
        >
    >;
    // Constructor that takes either a Cabana or NuMesh object
    Array(const std::string& label, const std::shared_ptr<container_layout_type>& array_layout)
        : _label( label )
        , _layout( array_layout )
    {
        auto layout = array_layout->layout();

        if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
        {
            _array = Cabana::Grid::createArray<value_type, memory_space>(label, layout);
        }
        else if  constexpr (std::is_same_v<entity_type, NuMesh::Vertex>)
        {
            _array = NuMesh::Array::createArray<value_type, memory_space>(label, layout);
        }
        else if  constexpr (std::is_same_v<entity_type, NuMesh::Edge>)
        {
            _array = NuMesh::Array::createArray<value_type, memory_space>(label, layout);
        }
        else if  constexpr (std::is_same_v<entity_type, NuMesh::Face>)
        {
            _array = NuMesh::Array::createArray<value_type, memory_space>(label, layout);
        }
        else
        {	// TBH, you might want this to be a compile time error instead
            static_assert(dependent_false<entity_type>::value, "Unsupported Beatnik::ArrayUtils::Array EntityType!");
        }
    }

    // Getters
    std::shared_ptr<array_type> array() const {return _array;}
    std::shared_ptr<container_layout_type> clayout() const {return _layout;} // Return the container layout
    std::string label() const {return _label;}

  private:
    // Array pointers
    std::shared_ptr<array_type> _array;

    // Layout pointers
    std::shared_ptr<container_layout_type> _layout;
    std::string _label;
};

template <class Scalar, class MemorySpace, class ContainerLayoutType>
std::shared_ptr<Array<ContainerLayoutType, Scalar, MemorySpace>>
createArray(const std::string& label, const std::shared_ptr<ContainerLayoutType>& layout)
{
    return std::make_shared<Array<ContainerLayoutType, Scalar, MemorySpace>>(label, layout);
}

//---------------------------------------------------------------------------//
// Array operations.
//---------------------------------------------------------------------------//
namespace ArrayOp
{


/**
 * Here, implement Cabana-array-specific ArrayOps not included in Cabana
 * This is mainly because NuMesh arrays are generally in the (i, x) format
 * whereas Cabana arrays are generally in the (i, j, x) format,
 * so they must be treated slightly differently
 */
namespace CabanaOp
{

/**
 * Cabana has a "dot" ArrayOp, but it computes a single dot product for 
 * an entire array rather than on a vector-by-vector basis
 */ 

} // end namespace CabanaOp

template <class ContainerLayoutType, class Scalar, class MemorySpace>
std::shared_ptr<Array<ContainerLayoutType, Scalar, MemorySpace>>
clone( const Array<ContainerLayoutType, Scalar, MemorySpace>& array )
{
    using memory_space = typename Array<ContainerLayoutType, Scalar, MemorySpace>::memory_space;
    using value_type = typename Array<ContainerLayoutType, Scalar, MemorySpace>::value_type;
    return createArray<value_type, memory_space>( array.label(), array.clayout() );
}

template <class Array_t, class DecompositionTag>
void copy( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::copy(*a.array(), *b.array(), tag);
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::copy(*a.array(), *b.array(), tag);
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
        Cabana::Grid::ArrayOp::assign(*array.array(), alpha, tag);
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::assign(*array.array(), alpha, tag);
    }
}

template <class Array_t, class DecompositionTag>
void update( Array_t& a, const double alpha, const Array_t& b,
        const double beta, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::update(*a.array(), alpha, *b.array(), beta, tag);
    }
     else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
              // Can combine this into one custom type trait method 
    {
        NuMesh::Array::ArrayOp::update(*a.array(), alpha, *b.array(), beta, tag);
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
        Cabana::Grid::ArrayOp::update(*a.array(), alpha, *b.array(), beta, *c.array(), gamma, tag);
    }
     else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::update(*a.array(), alpha, *b.array(), beta, *c.array(), gamma, tag);
    }
}

/**
 * Computes the dot product of each (x, y, z) vector stored in 
 * the arrays
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> vector_dot( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    // The resulting 'dot' array has the shape (i, j, 1)
    std::shared_ptr<ArrayLayout<mesh_type, entity_type>> scaler_layout;
    Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2U, Kokkos::Iterate::Default, Kokkos::Iterate::Default>> policy;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        scaler_layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->localGrid(), 1, entity_type());
        policy = Cabana::Grid::createExecutionPolicy(
            scaler_layout->layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
            execution_space() );
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        scaler_layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->mesh(), 1, entity_type());
        policy = Cabana::Grid::createExecutionPolicy(
            scaler_layout->layout()->indexSpace( tag, entity_type(), NuMesh::Local() ),
            execution_space() );
    }
    // auto vertex_triple_layout = Utils::createArrayLayout(nu_mesh, 3, NuMesh::Vertex());
    auto out = ArrayUtils::createArray<value_type, memory_space>("dot", scaler_layout);
    auto out_view = out->array()->view();

    // Check dimensions
    auto a_view = a.array()->view();
    auto b_view = b.array()->view();

    const int n = a_view.extent(0);
    const int m = a_view.extent(1);
    const int w = a_view.extent(2);
    const int out_n = out_view.extent(0);
    const int out_m = out_view.extent(1);

    // Ensure the third dimension is 3 for 3D vectors
    if (w != 3) {
        throw std::invalid_argument("Third dimension must be 3 for 3D vectors.");
    }
    if (out_n != n) {
        throw std::invalid_argument("First dimension of in and out views do not match.");
    }
    if (out_m != m) {
        throw std::invalid_argument("Second dimension of in and out views do not match.");
    }

    // Parallel loop to compute the dot product at each (n, m) location
    Kokkos::parallel_for("compute_dot_product", policy,
        KOKKOS_LAMBDA(const int i, const int j) {
            out_view(i, j, 0) = a_view(i, j, 0) * b_view(i, j, 0)
                              + a_view(i, j, 1) * b_view(i, j, 1)
                              + a_view(i, j, 2) * b_view(i, j, 2);
        });

    return out;
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> cross( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2U, Kokkos::Iterate::Default, Kokkos::Iterate::Default>> policy;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        policy = Cabana::Grid::createExecutionPolicy(
            a.clayout()->layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
            execution_space() );
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        policy = Cabana::Grid::createExecutionPolicy(
            a.clayout()->layout()->indexSpace( tag, entity_type(), NuMesh::Local() ),
            execution_space() );
    }

    auto out = clone(a);
    auto out_view = out->array()->view();

    auto a_view = a.array()->view();
    auto b_view = b.array()->view();
    int n = a_view.extent(0);
    int m = a_view.extent(1);
    int w = a_view.extent(2);

    if (w != 3 || (int)b_view.extent(0) != n || (int)b_view.extent(1) != m || (int)b_view.extent(2) != 3) {
        throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::Cross: Both input views must be of size n x m x 3.");
    }

    // Create output view for cross product results
    Kokkos::parallel_for("CrossProductKernel", policy,
        KOKKOS_LAMBDA(int i, int j) {
        value_type a_x = a_view(i, j, 0);
        value_type a_y = a_view(i, j, 1);
        value_type a_z = a_view(i, j, 2);
        
        value_type b_x = b_view(i, j, 0);
        value_type b_y = b_view(i, j, 1);
        value_type b_z = b_view(i, j, 2);

        // Cross product: a x b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
        out_view(i, j, 0) = a_y * b_z - a_z * b_y;
        out_view(i, j, 1) = a_z * b_x - a_x * b_z;
        out_view(i, j, 2) = a_x * b_y - a_y * b_x;
    });

    return out;
}

} // end namespace ArrayOp

} // end namespace ArrayUtils

} // end namespace Beatnik

#endif // BEATNIK_ARRAYUTILS_HPP