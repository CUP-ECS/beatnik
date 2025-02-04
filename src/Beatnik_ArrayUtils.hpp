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

#include <Beatnik_Types.hpp>
#include <type_traits>

namespace Beatnik
{
namespace ArrayUtils
{

template <class MeshType, class EntityType, class ValueType>
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
    /**
     * The ValueType is either a primitive type (int, double, float) for Cabana Grid meshes
     * or a Cabana::MemberTypes<...> type for unstructured meshes
     */
    using value_type = ValueType;

    // Determine ContainerLayoutType using std::conditional_t
    using array_layout_type = std::conditional_t<
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::ArrayLayout<entity_type, cabana_mesh_type<mesh_type>>, // Case A: Cabana UniformMesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<MeshType>::value,
            NuMesh::Array::ArrayLayout<entity_type, mesh_type, value_type>, // Case B: NuMesh Mesh
            void // Fallback type or an error type if neither condition is satisfied
        >
    >;
  
    ArrayLayout(const std::shared_ptr<mesh_type>& mesh, const int dofs_per_entity, entity_type tag)
    {
        if constexpr (is_cabana_mesh<mesh_type>::value)
        {
            _layout = Cabana::Grid::createArrayLayout(mesh, dofs_per_entity, tag);
        }
        else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value)
        {
            _layout = NuMesh::Array::createArrayLayout<ValueType>(mesh, dofs_per_entity, tag); 
        }
        else
        {
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
template <class MeshType, class EntityType, class ValueType>
std::shared_ptr<ArrayLayout<MeshType, EntityType, ValueType>>
createArrayLayout(const std::shared_ptr<MeshType>& mesh, const int dofs_per_entity, EntityType tag)
{
    return std::make_shared<ArrayLayout<MeshType, EntityType, ValueType>>(mesh, dofs_per_entity, tag);
}

//---------------------------------------------------------------------------//
// Array class
//---------------------------------------------------------------------------//
template <class ContainerLayoutType, class MemorySpace>
class Array
{
    static_assert(ContainerLayoutType::isArrayLayout(), "ContainerLayoutType must be a valid array layout.");
  
  public:
    using entity_type = typename ContainerLayoutType::entity_type;
    using mesh_type   = typename ContainerLayoutType::mesh_type;
    using layout_type = typename ContainerLayoutType::array_layout_type;
    using value_type = typename ContainerLayoutType::value_type;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    using container_layout_type = ContainerLayoutType;

    // Determine array_type using std::conditional_t
    using array_type = std::conditional_t<
        is_cabana_mesh<mesh_type>::value,
        Cabana::Grid::Array<value_type, entity_type, cabana_mesh_type<mesh_type>, memory_space>, // Case A: Cabana Mesh
        std::conditional_t<
            NuMesh::is_numesh_mesh<mesh_type>::value,
            NuMesh::Array::Array<memory_space, layout_type>, // Case B: NuMesh Mesh
            void // Fallback or error type if neither condition is satisfied
        >
    >;
    // Constructor that takes either a Cabana or NuMesh object
    Array(const std::string& label, const std::shared_ptr<container_layout_type>& array_layout)
        : _label( label )
        , _layout( array_layout )
    {
        auto layout = array_layout->layout();
        if constexpr (is_cabana_mesh<mesh_type>::value)
        {
            _array = Cabana::Grid::createArray<value_type, memory_space>(label, layout);
        }
        else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value)
        {
            _array = NuMesh::Array::createArray<memory_space>(label, layout);
        }
        else
        {	
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

template <class ContainerLayoutType, class MemorySpace>
std::shared_ptr<Array<ContainerLayoutType, MemorySpace>>
createArray(const std::string& label, const std::shared_ptr<ContainerLayoutType>& layout)
{
    return std::make_shared<Array<ContainerLayoutType, MemorySpace>>(label, layout);
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

    auto layout = Cabana::Grid::createArrayLayout( a.layout()->localGrid(), 1, entity_type() );
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
 * If a and b do not have matching third dimensions, place the view with the 
 * smaller third dimension first. Dimensions in b that are not present in a
 * are truncated.
 * 
 * If a has a third dimension of 1, out(x, y, z) = b(x, y, z) * a(x, y, 0) for 0 <= z < b extent
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
    if (aw == bw) {
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, Cabana::Grid::Local() ),
            execution_space() );
        Kokkos::parallel_for(
            "ArrayOp::element_multiply", policy,
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                out_view( i, j, k ) = a_view( i, j, k ) * b_view( i, j, k );
            } );
        return out;
    }

    // If a has a third dimension of 1
    if ((aw == 1) && (aw < bw))
    {
        using entity_type = typename Array_t::entity_type;
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
            execution_space() );
        Kokkos::parallel_for(
            "ArrayOp::element_multiply", policy,
            KOKKOS_LAMBDA( const int i, const int j) {
                for (int k = 0; k < bw; k++)
                {
                    out_view( i, j, k ) = a_view( i, j, 0 ) * b_view( i, j, k );
                }
            } );
        return out;
    }
    else
    {
        throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::CabanaOp:element_multiply: First array argument must have equal or smaller third dimension than second array argument.");
    }   
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> copyDim( Array_t& a, int dimA, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    auto layout = Cabana::Grid::createArrayLayout( a.layout()->localGrid(), 1, entity_type() );
    auto out = Cabana::Grid::createArray<value_type, memory_space>("cabana_dot", layout);
    auto out_view = out->view();

    // Check dimensions
    auto a_view = a.view();

    const int aw = a_view.extent(2);

    if (dimA >= aw) {
        throw std::invalid_argument("ArrayUtils::ArrayOp::CabanaOp::copyDim: Provided dimension is larger than the number of dimensions in the array.");
    }

    auto policy = Cabana::Grid::createExecutionPolicy(
        a.layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
        execution_space() );
    Kokkos::parallel_for(
        "ArrayUtils::ArrayOp::CabanaOp::copyDim", policy,
        KOKKOS_LAMBDA( const int i, const int j) {
            out_view( i, j, 0 ) = a_view( i, j, dimA );
        } );
    return out;
}

template <class Array_t, class DecompositionTag>
void copyDim( Array_t& a, int dimA, Array_t& b, int dimB, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using execution_space = typename Array_t::execution_space;

    auto a_view = a.view();
    auto b_view = b.view();

    const int an = a_view.extent(0);
    const int bn = b_view.extent(0);
    const int am = a_view.extent(1);
    const int bm = b_view.extent(1);
    const int aw = a_view.extent(2);
    const int bw = b_view.extent(2);

    if (an != bn) {
        throw std::invalid_argument("ArrayUtils::ArrayOp::CabanaOp::copyDim: First dimension of a and b arrays do not match.");
    }
    if (am != bm) {
        throw std::invalid_argument("ArrayUtils::ArrayOp::CabanaOp::copyDim: Second dimension of a and b arrays do not match.");
    }
    if (dimA >= aw) {
        throw std::invalid_argument("ArrayUtils::ArrayOp::CabanaOp::copyDim: Provided dimension for 'a' is larger than the number of dimensions in the b array.");
    }
    if (dimB >= bw) {
        throw std::invalid_argument("ArrayUtils::ArrayOp::CabanaOp::copyDim: Provided dimension for 'b' is larger than the number of dimensions in the b array.");
    }

    auto policy = Cabana::Grid::createExecutionPolicy(
        a.layout()->localGrid()->indexSpace( tag, entity_type(), Cabana::Grid::Local() ),
        execution_space() );
    Kokkos::parallel_for(
        "ArrayUtils::ArrayOp::CabanaOp::copyDim", policy,
        KOKKOS_LAMBDA( const int i, const int j) {
            a_view( i, j, dimA ) = b_view( i, j, dimB );
    } );
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

template <class ContainerLayoutType, class MemorySpace>
std::shared_ptr<Array<ContainerLayoutType, MemorySpace>>
clone( const Array<ContainerLayoutType, MemorySpace>& array )
{
    using memory_space = typename Array<ContainerLayoutType, MemorySpace>::memory_space;
    using value_type = typename Array<ContainerLayoutType, MemorySpace>::value_type;
    return createArray<value_type, memory_space>( array.label(), array.clayout() );
}

template <class Array_t, class DecompositionTag>
void copy( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::copy(*a.array(), *b.array(), tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        NuMesh::Array::ArrayOp::copy(*a.array(), *b.array(), tag);
    }

}

/**
 * Create a copy of one dimension of an array
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> copyDim( Array_t& a, int dimA, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using value_type = typename  Array_t::value_type;
    using memory_space = typename Array_t::memory_space;

    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        auto cabana_out = CabanaOp::copyDim(*a.array(), dimA, tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->localGrid(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("copyDim", layout);
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;

    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::copyDim(*a.array(), dimA, tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->mesh(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("copyDim", layout);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }
}

/**
 * Copy dimB from b into dimA from a 
 */
template <class Array_t, class DecompositionTag>
void copyDim( Array_t& a, int dimA, Array_t& b, int dimB, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        CabanaOp::copyDim(*a.array(), dimA, *b.array(), dimB, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        NuMesh::Array::ArrayOp::copyDim(*a.array(), dimA, *b.array(), dimB, tag);
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
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::assign(*array.array(), alpha, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value)
    {
        NuMesh::Array::ArrayOp::assign(*array.array(), alpha, tag);
    }
}

/**
 * Untagged version of assign which defaults to all entities
 */
template <class Array_t>
void assign( Array_t& array, const double alpha )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::assign(*array.array(), alpha, Cabana::Grid::Ghost());
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value)
    {
        NuMesh::Array::ArrayOp::assign(*array.array(), alpha, NuMesh::Ghost());
    }
}

template <class Array_t, class DecompositionTag>
void scale( Array_t& array, const double alpha,
             DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::scale(*array.array(), alpha, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        NuMesh::Array::ArrayOp::scale(*array.array(), alpha, tag);
    }
}

template <class Array_t, class DecompositionTag>
void update( Array_t& a, const double alpha, const Array_t& b,
        const double beta, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::update(*a.array(), alpha, *b.array(), beta, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        NuMesh::Array::ArrayOp::update(*a.array(), alpha, *b.array(), beta, tag);
    }
}

template <class Array_t, class DecompositionTag>
void update( Array_t& a, const double alpha, const Array_t& b,
        const double beta, const Array_t& c,
        const double gamma, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        Cabana::Grid::ArrayOp::update(*a.array(), alpha, *b.array(), beta, *c.array(), gamma, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        NuMesh::Array::ArrayOp::update(*a.array(), alpha, *b.array(), beta, *c.array(), gamma, tag);
    }
}

template <class Array_t, class Function, class DecompositionTag>
void apply( Array_t& a, Function& function, DecompositionTag tag )
{
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        CabanaOp::apply(*a.array(), function, tag);
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
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

    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        auto cabana_out = CabanaOp::element_dot(*a.array(), *b.array(), tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->localGrid(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("dot", layout);
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_dot(*a.array(), *b.array(), tag);
        auto layout = ArrayUtils::createArrayLayout(a.clayout()->layout()->mesh(), 1, entity_type());
        auto out = ArrayUtils::createArray<value_type, memory_space>("dot", layout);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }
    
    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_dot: Invalid mesh_type");
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_cross( const Array_t& a, const Array_t& b, DecompositionTag tag )
{
    auto out = clone(b);
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        auto cabana_out = CabanaOp::element_cross(*a.array(), *b.array(), tag); 
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_cross(*a.array(), *b.array(), tag);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }

    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_cross: Invalid mesh_type");
}

template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_multiply( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    auto out = clone(b);
    using mesh_type = typename Array_t::mesh_type;
    if constexpr (is_cabana_mesh<mesh_type>::value)
    {
        auto cabana_out = CabanaOp::element_multiply(*a.array(), *b.array(), tag); 
        Cabana::Grid::ArrayOp::copy(*out->array(), *cabana_out, tag);
        return out;
    }
    else if constexpr (NuMesh::is_numesh_mesh<mesh_type>::value) 
    {
        auto numesh_out = NuMesh::Array::ArrayOp::element_multiply(*a.array(), *b.array(), tag);
        NuMesh::Array::ArrayOp::copy(*out->array(), *numesh_out, tag);
        return out;
    }

    throw std::invalid_argument("Beatnik::ArrayUtils::ArrayOp::element_multiply: Invalid mesh_type");
}

} // end namespace ArrayOp

} // end namespace ArrayUtils

} // end namespace Beatnik

#endif // BEATNIK_ARRAYUTILS_HPP