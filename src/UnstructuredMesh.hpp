/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik author                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_UNSTRUCTURED_MESH_HPP
#define BEATNIK_UNSTRUCTURED_MESH_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <MeshBase.hpp>

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <NuMesh_Core.hpp>

#include <mpi.h>

namespace Beatnik
{

KOKKOS_INLINE_FUNCTION
double kern( const double d, const double shape_factor, const double hybrid_weight )
{
    return Kokkos::exp( -shape_factor * d * d ) + hybrid_weight * d * d * d;
}

KOKKOS_INLINE_FUNCTION
double kernPrime( const double d, const double shape_factor, const double hybrid_weight )
{
    return -double( 2.0 ) * shape_factor * d * Kokkos::exp( -shape_factor * d * d ) +
        double( 3.0 ) * hybrid_weight * d * d;
}

//---------------------------------------------------------------------------//
/*!
  \class StructuredMesh
  \brief Logically uniform Cartesian mesh.
*/
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag>
class UnstructuredMesh : public MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using mesh_type_tag = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type_tag;
    using entity_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::entity_type;
    using mesh_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::mesh_type;
    using base_triple_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::base_triple_type;
    using triple_array_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::triple_array_type;
    using pair_array_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::pair_array_type;
    using cabana_local_grid_type = typename MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>::cabana_local_grid_type;

    using v2v_type = NuMesh::Maps::V2V<mesh_type>;

    UnstructuredMesh( const std::array<double, 6>& global_bounding_box,
          const std::array<int, 2>& num_nodes,
	      const std::array<bool, 2>& periodic,
          const Cabana::Grid::BlockPartitioner<2>& partitioner,
          MPI_Comm comm )
		        : _comm( comm )
                , _num_nodes( num_nodes )
                , _periodic( periodic )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        // Copy the same code used to create the local grid in the constructor
        // of StructuredMesh
        // XXX - Don't copy-paste code
        for (int i = 0; i < 3; i++) {
            _low_point[i] = global_bounding_box[i];
            _high_point[i] = global_bounding_box[i+3];
        } 

        std::array<double, 2> global_low_corner, global_high_corner;
        for ( int d = 0; d < 2; ++d )
        {
            /* Even number of nodes
             * periodic -> nnodes = 4, 3 cabana nodes - 3 cells
	     *             global low == -1, global high = 2 
             *                   -> nodes = (-1,-0,1).
             * non-periodic -> nnodes = 4, 4 cabana nodes - 3 cells
             *              -> global low == -2, global high = 1 
             *                             -> nodes = (-2,-1,0,1).
             * Odd number of nodes
             * periodic -> nnodes = 5, 4 cabana nodes - 4 cells
	     *             global low == -2, global high = 2 
             *                   -> nodes = (-2,-1,0,1).
             * non-periodic -> nnodes = 5, 5 cabana nodes - 4 cells
             *              -> global low == -2, global high = 2 
             *                             -> nodes = (-2,-1,0,1,2).
	         * So we always have (nnodes - 1 cells) */

	        int cabana_nodes = num_nodes[d] - (periodic[d] ? 1 : 0);

            global_low_corner[d] = -cabana_nodes/2;
            global_high_corner[d] = global_low_corner[d] + num_nodes[d] - 1;
        }

        // Finally, create the global mesh, global grid, and local grid.
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, 1.0 );

        auto global_grid = Cabana::Grid::createGlobalGrid( comm, global_mesh,
                                                     periodic, partitioner );
        // Build the local grid.
        int surface_halo_width = 2; // Halo width doesn't matter here
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, surface_halo_width );


        _mesh = NuMesh::createEmptyMesh<execution_space, memory_space>(_comm);

        // Create dummy array from local grid from which to initialize the mesh
        auto dlayout = Cabana::Grid::createArrayLayout(_local_grid, 3, Cabana::Grid::Node());
        auto darray = Cabana::Grid::createArray<double, memory_space>("dummy_for_init", dlayout);

        // Initialize the unstructured mesh from the connectivity of the local grid.
        _mesh->initializeFromArray(*darray);

        _gradient_version = -1;

        // Initialize vertex connectivity at level 0
        _v2v = std::make_shared<v2v_type>(_mesh, 0);
        
        auto node_triple_layout =
            ArrayUtils::createArrayLayout<base_triple_type>( _mesh, 3, entity_type() );
        auto pos_test = ArrayUtils::createArray<memory_space>("pos_test", node_triple_layout);
        compute_gradient(*pos_test, 0, 1.0, 1.0);
    }
    
    /**
     * Calculates surface gradients at vertices of an unstructured mesh using
     * radial basis functions (RBFs) and a moving least squares (MLS) approach.
     * 
     * Using vertex connectivity at "level" of the mesh
     */
    void compute_gradient(const triple_array_type& positions_array, 
            const int level, const double shape_factor, const double hybrid_weight)
    {
        // Ensure positions array is up-to-date
        assert(positions_array.array()->aosoa().size() == _mesh->vertices().size());

        // Initialize gradients AoSoA, which has "num verts" Cabana::MemberTypes<double[3]> tuples
        auto vertex_triple_layout =
            ArrayUtils::createArrayLayout<base_triple_type>(_mesh, 3, NuMesh::Vertex());
        _gradients = ArrayUtils::createArray<memory_space>("_gradients", vertex_triple_layout);
        auto gradients = Cabana::slice<0>(_gradients->array()->aosoa());

        // Retrieve vertex-to-vertex connectivity, update if needed
        if ((_mesh->version() != _v2v->version()) || (_v2v->level() != level))
        {
            _v2v = std::make_shared<v2v_type>(_mesh, level);
        }

        auto offsets = _v2v->offsets();
        auto indices = _v2v->indices();
        int owned_vertices = _mesh->count(NuMesh::Own(), NuMesh::Vertex());

        auto positions = Cabana::slice<0>(positions_array.array()->aosoa());

        // At worst, each vert is connected 6*3*(max tree level) verts
        int max_stecil_size = 6 * 3 * (_mesh->max_level()+1);

        // Allocate workspace for K and grad_sample
        Kokkos::View<double***, memory_space> K("Kernel Matrix", owned_vertices, max_stecil_size + 1, max_stecil_size + 1);
        Kokkos::View<double***, memory_space> grad_sample("Gradient Samples", owned_vertices, max_stecil_size + 1, 3);

        Kokkos::parallel_for("compute_gradient", Kokkos::RangePolicy<execution_space>(0, owned_vertices),
            KOKKOS_LAMBDA(int vlid) {

            int offset = offsets(vlid);
            int next_offset = (vlid + 1 < (int)offsets.extent(0)) ? offsets(vlid + 1) : (int)indices.extent(0);
            int sten_size = next_offset - offset;

            // Position of current vertex
            double pos_x = positions(vlid, 0);
            double pos_y = positions(vlid, 1);
            double pos_z = positions(vlid, 2);

            // Initialize Kernel matrix and grad_sample
            for (int i = 0; i <= sten_size; ++i) {
                for (int j = 0; j <= sten_size; ++j) {
                    K(vlid, i, j) = 0.0;
                }
                grad_sample(vlid, i, 0) = 0.0;
                grad_sample(vlid, i, 1) = 0.0;
                grad_sample(vlid, i, 2) = 0.0;
            }

            for (int i = 0; i < sten_size; ++i) {
                int vi = indices(offset + i);
                double vix = positions(vi, 0);
                double viy = positions(vi, 1);
                double viz = positions(vi, 2);

                for (int j = 0; j <= i; ++j) {
                    int vj = indices(offset + j);
                    double dx = vix - positions(vj, 0);
                    double dy = viy - positions(vj, 1);
                    double dz = viz - positions(vj, 2);
                    double d = sqrt(dx * dx + dy * dy + dz * dz);

                    K(vlid, i, j) = kern(d, shape_factor, hybrid_weight);
                    K(vlid, j, i) = K(vlid, i, j);
                }
                K(vlid, i, sten_size) = 1.0;
                K(vlid, sten_size, i) = 1.0;

                vix = pos_x - vix;
                viy = pos_y - viy;
                viz = pos_z - viz;
                double ed = sqrt(vix * vix + viy * viy + viz * viz);
                double kp = (i == 0) ? 0.0 : kernPrime(ed, shape_factor, hybrid_weight) / ed;

                grad_sample(vlid, i, 0) = vix * kp;
                grad_sample(vlid, i, 1) = viy * kp;
                grad_sample(vlid, i, 2) = viz * kp;
            }

            K(vlid, sten_size, sten_size) = 0.0;

            // LU factorization
            KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke(Kokkos::subview(K, vlid, Kokkos::ALL, Kokkos::ALL));

            // Solve for gradients
            KokkosBatched::SerialTrsm<KokkosBatched::Side::Left,
                                    KokkosBatched::Uplo::Lower,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Diag::Unit,
                                    KokkosBatched::Algo::Trsm::Unblocked>::invoke(
                1.0, Kokkos::subview(K, vlid, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(grad_sample, vlid, Kokkos::ALL, Kokkos::ALL));

            KokkosBatched::SerialTrsm<KokkosBatched::Side::Left,
                                    KokkosBatched::Uplo::Upper,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Diag::NonUnit,
                                    KokkosBatched::Algo::Trsm::Unblocked>::invoke(
                1.0, Kokkos::subview(K, vlid, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(grad_sample, vlid, Kokkos::ALL, Kokkos::ALL));

            // Project onto surface tangent
            for (int i = 0; i < sten_size; ++i) {
                double xDg = pos_x * grad_sample(vlid, i, 0) + pos_y * grad_sample(vlid, i, 1) + pos_z * grad_sample(vlid, i, 2);
                gradients(vlid, 0) += grad_sample(vlid, i, 0) - xDg * pos_x;
                gradients(vlid, 1) += grad_sample(vlid, i, 1) - xDg * pos_y;
                gradients(vlid, 2) += grad_sample(vlid, i, 2) - xDg * pos_z;
            }
        });
        _gradient_version = _v2v->version();
    }

    std::shared_ptr<mesh_type> layoutObj() const override
    {
        return _mesh;
    }

    std::shared_ptr<cabana_local_grid_type> localGrid() const
    {
        return _local_grid;
    }

    // Get whether the mesh is periodic
    // XXX - Assumes if the x-boundary is periodic, the mesh
    // is also periodic along the y-boundary
    int is_periodic() const override
    {
        return _periodic[0];
    }

    /**
     * Compute fourth-order central difference calculation for derivatives along the 
     * interface surface
     */
    std::shared_ptr<triple_array_type> Dx(const triple_array_type& in, const double dx) const override
    {
        auto out = ArrayUtils::ArrayOp::clone(in);
        // auto out_view = out->array()->view();
        // auto in_view = in.array()->view();
        // auto layout = in.clayout()->layout();
        // auto index_space = layout->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
        //     out_view(i, j, k) = Operators::Dx(in_view, i, j, k, dx);
        // });
        return out;
    }
    std::shared_ptr<triple_array_type> Dy(const triple_array_type& in, const double dy) const override
    {
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        // auto out_view = out->array()->view();
        // auto in_view = in.array()->view();
        // auto layout = in.clayout()->layout();
        // auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
        //     for (int k = 0; k < dim2; k++) out_view(i, j, k) = Operators::Dy(in_view, i, j, k, dy);
        // });
        return out;
    }

    std::shared_ptr<pair_array_type> laplace(const pair_array_type& in, [[maybe_unused]]const double dx, [[maybe_unused]]const double dy) const override
    {
        assert((_v2v->version() == _mesh->version()) && (_v2v->version() == _gradient_version));
    
        auto out = Beatnik::ArrayUtils::ArrayOp::clone(in);
        auto out_aosoa = out->array()->aosoa();
        auto laplace = Cabana::slice<0>(out_aosoa);
        auto in_aosoa = in.array()->aosoa();
        auto vorts = Cabana::slice<0>(in_aosoa);
        auto gradients = Cabana::slice<0>(_gradients->array()->aosoa());
    
        // Retrieve neighbor connectivity
        auto offsets = _v2v->offsets();
        auto indices = _v2v->indices();
    
        int num_vertices = gradients.size();
        auto policy = Kokkos::RangePolicy<ExecutionSpace>(0, num_vertices);
    
        Kokkos::parallel_for("Calculate Laplace", policy, KOKKOS_LAMBDA(const int vertex_id) {
            double laplace_x = 0.0;
            double laplace_y = 0.0;
            int num_neighbors = 0;
    
            int offset = offsets(vertex_id);
            int next_offset = (vertex_id + 1 < (int)offsets.extent(0)) ? offsets(vertex_id + 1) : (int)indices.extent(0);
    
            for (int i = offset; i < next_offset; ++i) {
                int neighbor_id = indices(i);
    
                // Compute Laplacian for both x and y components
                double diff_x = vorts(neighbor_id, 0) - vorts(vertex_id, 0);
                double diff_y = vorts(neighbor_id, 1) - vorts(vertex_id, 1);
    
                laplace_x += diff_x;
                laplace_y += diff_y;
                num_neighbors++;
            }
    
            // Normalize by number of neighbors (if at least one exists)
            if (num_neighbors > 0) {
                laplace_x /= num_neighbors;
                laplace_y /= num_neighbors;
            }
    
            // Store results in Laplace AoSoA slice
            laplace(vertex_id, 0) = laplace_x;
            laplace(vertex_id, 1) = laplace_y;
        });
    
        return out;
    }
    
    // XXX - Assert that the mesh and mesh_array_types are the right type 
    // at the beginning of these functions
        // using Node = Cabana::Grid::Node;
        // auto zdx_view = z_dx.array()->view();
        // auto zdy_view = z_dy.array()->view();
        // auto w_view = w.array()->view();
        // auto layout = z_dx.clayout()->layout();
        // auto node_triple_layout = ArrayUtils::createArrayLayout( layout->localGrid(), 3, Node() );
        // std::shared_ptr<mesh_array_type> out = ArrayUtils::createArray<double, memory_space>("omega", 
        //                                                node_triple_layout);
        // auto out_view = out->array()->view();
        // auto index_space = layout->localGrid()->indexSpace(Cabana::Grid::Own(), Node(), Cabana::Grid::Local());
        // int dim2 = layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() ).extent( 2 );
        // auto policy = Cabana::Grid::createExecutionPolicy(index_space, ExecutionSpace());
        // Kokkos::parallel_for("Calculate Dx", policy, KOKKOS_LAMBDA(const int i, const int j) {
        //     for (int k = 0; k < dim2; k++)
        //         out_view(i, j, k) = w_view(i, j, 1) * zdx_view(i, j, k) - w_view(i, j, 0) * zdy_view(i, j, k);
        // });

    std::shared_ptr<triple_array_type> omega(const pair_array_type& w,
        [[maybe_unused]]const triple_array_type& z_dx, [[maybe_unused]]const triple_array_type& z_dy) const override
    {
        auto w_aosoa = w.array()->aosoa();
        auto vorts = Cabana::slice<0>(w_aosoa);
        auto gradients = Cabana::slice<0>(_gradients->array()->aosoa());

        // Clone the gradients AoSoA structure for the output
        auto omega_array = ArrayUtils::ArrayOp::clone(*_gradients);
        auto omega_aosoa = omega_array->array()->aosoa();
        auto omega = Cabana::slice<0>(omega_aosoa);

        int num_vertices = _gradients->array()->aosoa().size();

        Kokkos::parallel_for("Calculate Omega (Unstructured)", Kokkos::RangePolicy<execution_space>(0, num_vertices),
        KOKKOS_LAMBDA(const int vertex_id) {

            omega(vertex_id, 0) = vorts(vertex_id, 1) * gradients(vertex_id, 2) -
                                   vorts(vertex_id, 2) * gradients(vertex_id, 1);
        
            omega(vertex_id, 1) = vorts(vertex_id, 2) * gradients(vertex_id, 0) -
                                    vorts(vertex_id, 0) * gradients(vertex_id, 2);
            
            omega(vertex_id, 2) = vorts(vertex_id, 0) * gradients(vertex_id, 1) -
                                    vorts(vertex_id, 1) * gradients(vertex_id, 0);
        });

        return omega_array;
    }


    /**
     * These functions are only needed for structured meshes, but it must be overridden here
     * to avoid errors.
     */
    Cabana::Grid::IndexSpace<2>
    periodicIndexSpace(Cabana::Grid::Ghost, Cabana::Grid::Node, std::array<int, 2> dir) const override
    { 
        std::array<long, 2> zero_size;
        for ( std::size_t d = 0; d < 2; ++d )
            zero_size[d] = 0;
        return Cabana::Grid::IndexSpace<2>( zero_size, zero_size );
    }
    const std::array<double, 3> & boundingBoxMin() const override {return _low_point;}
    const std::array<double, 3> & boundingBoxMax() const override {return _high_point;}
    int mesh_size() const override { return -1; }
    int halo_width() const override { return -1; }

    MPI_Comm comm() const override { return _comm; }
    int rank() const override { return _rank; }
    int comm_size() const override { return _comm_size; }

  private:
    MPI_Comm _comm;
    int _rank, _comm_size;

    std::array<int, 2> _num_nodes;
    const std::array<bool, 2> _periodic;
    
    std::array<double, 3> _low_point, _high_point;
    
    // The local grid the unstructured mesh is built from
    std::shared_ptr<cabana_local_grid_type> _local_grid;

    // Unstructured mesh object
    std::shared_ptr<mesh_type> _mesh;

    // Graident array for each owned vertex and the version of the mesh it is from
    std::shared_ptr<triple_array_type> _gradients;
    int _gradient_version;

    // Vertex-to-vertex mapping. Stored so we don't have to re-create it if the mesh does not update
    std::shared_ptr<v2v_type> _v2v;

};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // BEATNIK_UNSTRUCTURED_MESH_HPP