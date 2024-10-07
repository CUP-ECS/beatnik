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
template <typename T>
using cabana_mesh_type = typename T::mesh_type;

template <typename T>
using is_cabana_mesh = Cabana::Grid::isMeshType<cabana_mesh_type<T>>;
// XXX: Make RHS of 40 to not depend on cabana_mesh_type so cabana_mesh_type can be removed.


template<typename T>
struct dependent_false : std::false_type {};

template <class MeshType, class EntityType>
class ArrayLayout
{
  public:
    /**
     * NuMesh: MeshType = NuMesh::Mesh<ExecutionSpace, MemSpace>
     * Cabana:: MeshType = Cabana::Grid::LocalGrid<cabana_mesh_type>, 
     *      where the struct isMeshType() is implemented for cabana_mesh_type.
     */
    using mesh_type = MeshType;
    using entity_type = EntityType;

    // Determine ContainerLayoutType using std::conditional_t
    using array_layout_type = std::conditional_t<
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::ArrayLayout<entity_type, cabana_mesh_type<mesh_type>>, // Case A: Cabana UniformMesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<MeshType>::value,
            NuMesh::Array::ArrayLayout<entity_type, mesh_type>, // Case B: NuMesh Mesh
            void // Fallback type or an error type if neither condition is satisfied
        >
    >;
  
    ArrayLayout(const std::shared_ptr<mesh_type>& mesh, const int dofs_per_entity, entity_type tag)
    {
        if constexpr (is_cabana_mesh<mesh_type>::value)
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
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::Array<value_type, entity_type, cabana_mesh_type<mesh_type>, memory_space>, // Case A: Cabana Mesh
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
 * 
 * All Array_t in this namespace are Cabana::Grid::Array arrays.
 */
namespace CabanaOp
{

/**
 * Cabana has a "dot" ArrayOp, but it computes a single dot product for 
 * an entire array rather than on a vector-by-vector basis
 */ 
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_dot( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    auto layout = Cabana::Grid::createArrayLayout( a.layout()->localGrid(), 1, Cabana::Grid::Node() );
    auto out = Cabana::Grid::createArray<value_type, memory_space>("cabana_dot", layout);
    auto out_view = out->view();

    // Check dimensions
    auto a_view = a.view();
    auto b_view = b.view();

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
        throw std::invalid_argument("First dimension of in and out arrays do not match.");
    }
    if (out_m != m) {
        throw std::invalid_argument("Second dimension of in and out arrays do not match.");
    }

    // Parallel loop to compute the dot product at each (n, m) location
    auto policy = Cabana::Grid::createExecutionPolicy(
            layout->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
            execution_space() );
    Kokkos::parallel_for("compute_dot_product", policy,
        KOKKOS_LAMBDA(const int i, const int j) {
            out_view(i, j, 0) = a_view(i, j, 0) * b_view(i, j, 0)
                              + a_view(i, j, 1) * b_view(i, j, 1)
                              + a_view(i, j, 2) * b_view(i, j, 2);
        });

    return out;
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_cross( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    // auto layout = Cabana::Grid::createArrayLayout( a.layout()->localGrid(), 3, Cabana::Grid::Node() );
    // auto out = Cabana::Grid::createArray<value_type, memory_space>("cabana_dot", layout);
    auto out = Cabana::Grid::ArrayOp::clone(a);
    auto out_view = out->view();

    // Check dimensions
    auto a_view = a.view();
    auto b_view = b.view();

    const int n = a_view.extent(0);
    const int m = a_view.extent(1);
    const int w = a_view.extent(2);
    const int out_n = out_view.extent(0);
    const int out_m = out_view.extent(1);

    if (w != 3 || (int)b_view.extent(0) != n || (int)b_view.extent(1) != m || (int)b_view.extent(2) != 3) {
        throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::CabanaOp::element_cross: Both input views must be of size n x m x 3.");
    }

    // Parallel loop to compute the dot product at each (n, m) location
    auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
            execution_space() );
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

/**
 * Element-wise multiplication, where (1, 3, 6) * (4, 7, 3) = (4, 21, 18)
 * If a and b do not have matching third dimenions, place the view with the 
 * smaller third dimenion first.
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_multiply( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using execution_space = typename Array_t::execution_space;
    auto out = Cabana::Grid::ArrayOp::clone(b);
    auto out_view = out->view();

    // Check dimensions
    auto a_view = a.view();
    auto b_view = b.view();

    const int an = a_view.extent(0);
    const int bn = b_view.extent(0);
    const int am = a_view.extent(1);
    const int bm = b_view.extent(1);
    const int aw = a_view.extent(2);
    const int bw = b_view.extent(2);

    if (an != bn) {
        throw std::invalid_argument("First dimension of a and b arrays do not match.");
    }
    if (am != bm) {
        throw std::invalid_argument("Second dimension of a and b arrays do not match.");
    }

    // Both a and b have matching third dimensions
    // if (aw == bw) {
    //     auto policy = Cabana::Grid::createExecutionPolicy(
    //         a.layout()->indexSpace( tag, Cabana::Grid::Local() ),
    //         execution_space() );
    //     Kokkos::parallel_for(
    //         "ArrayOp::update", policy,
    //         KOKKOS_LAMBDA( const int i, const int j, const int k ) {
    //             out_view( i, j, k ) = a_view( i, j, k ) * b_view( i, j, k );
    //         } );

    //     return out;
    // }

    // // If a has a third dimension of 1
    // if ((aw == 1) && (aw < bw))
    // {
    //     using entity_type = typename Array_t::entity_type;
    //     auto policy = Cabana::Grid::createExecutionPolicy(
    //         a.layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
    //         execution_space() );
    //     Kokkos::parallel_for(
    //         "ArrayOp::update", policy,
    //         KOKKOS_LAMBDA( const int i, const int j) {
    //             for (int)
    //             out_view( i, j, k ) = a_view( i, j, k ) * b_view( i, j, k );
    //         } );

    //     return out;
    // }
    // else
    // {
    //     throw std::invalid_argument("First array argument must have equal or smaller third dimension than second array argument.");
    // }

    
}

/*!
  \brief Apply some function to every element of an array
  \param array The array to operate on.
  \param function A functor that operates on the array elements.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class Function, class DecompositionTag>
std::enable_if_t<2 == Array_t::num_space_dim, void>
apply( Array_t& array, Function& function, DecompositionTag tag )
{
    using entity_t = typename Array_t::entity_type;
    auto view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::apply",
        createExecutionPolicy( array.layout()->indexSpace( tag, Cabana::Grid::Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k) {
            double val = view( i, j, k );
            view( i, j, k ) = function(val);
        } );
}

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
void scale( Array_t& array, const double alpha,
             DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        Cabana::Grid::ArrayOp::scale(*array.array(), alpha, tag);
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::scale(*array.array(), alpha, tag);
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

template <class Array_t, class Function, class DecompositionTag>
void apply( Array_t& a, Function& function, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        CabanaOp::apply(*a.array(), function, tag);
    }
     else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        NuMesh::Array::ArrayOp::apply(*a.array(), function, tag);
    }
}

/**
 * Computes the dot product of each (x, y, z) vector stored in 
 * the arrays.
 * The resulting 'dot' array has the shape (m, n, 1)
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_dot( const Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;

    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        auto cabana_out = CabanaOp::element_dot(*a.array(), *b.array(), tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->localGrid(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("dot", layout);
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_dot(*a.array(), *b.array(), tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->mesh(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("dot", layout);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }
    
    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_dot: Invalid entity_type");
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_cross( const Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;

    auto out = clone(a);
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        auto cabana_out = CabanaOp::element_cross(*a.array(), *b.array(), tag); 
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_cross(*a.array(), *b.array(), tag);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }

    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_cross: Invalid entity_type");
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_multiply( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;

    auto out = clone(a);
    if constexpr (std::is_same_v<entity_type, Cabana::Grid::Node>)
    {
        auto cabana_out = CabanaOp::element_multiply(*a.array(), *b.array(), tag); 
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (std::is_same_v<entity_type, NuMesh::Vertex> ||
              std::is_same_v<entity_type, NuMesh::Edge> ||
              std::is_same_v<entity_type, NuMesh::Face>) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_multiply(*a.array(), *b.array(), tag);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }

    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_multiply: Invalid entity_type");
}

} // end namespace ArrayOp

} // end namespace ArrayUtils

} // end namespace Beatnik

#endif // BEATNIK_ARRAYUTILS_HPP