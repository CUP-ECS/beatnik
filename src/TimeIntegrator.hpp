/****************************************************************************
 * Copyright (c) 2021-2022 by the Beatnik authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is                     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_TIMEINTEGRATOR_HPP
#define BEATNIK_TIMEINTEGRATOR_HPP

#include <BoundaryCondition.hpp>
#include <ProblemManager.hpp>
#include <ZModel.hpp>
#include <Beatnik_ArrayUtils.hpp>

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

namespace Beatnik
{

// The time integrator requires temporary state for runge kutta interpolation
// which are stored as part of this object 
template <class ProblemManagerType, class ZModelType>
class TimeIntegrator
{
    using memory_space = typename ProblemManagerType::memory_space;
    using entity_type = typename ProblemManagerType::entity_type;
    using mesh_array_type = typename ProblemManagerType::mesh_array_type;

  public:
    TimeIntegrator( const ProblemManagerType & pm,
                    const BoundaryCondition & bc,
                    const ZModelType & zm )
    : _pm(pm)
    , _bc(bc)
    , _zm(zm)
    {
        // Create a layout of the temporary arrays we'll need for velocity
        // intermediate positions, and change in vorticity
        auto node_triple_layout =
            ArrayUtils::createArrayLayout( pm.mesh().layoutObj(), 3, entity_type() );
        auto node_pair_layout =
            ArrayUtils::createArrayLayout( pm.mesh().layoutObj(), 2, entity_type() );
        //const std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> mesh = pm.mesh().localGrid(); 
        // auto node_pair_layout_2 = Utils::createArrayLayout<ExecutionSpace, MemorySpace>(pm.mesh().localGrid(), 2, Cabana::Grid::Node());
        // auto node_triple_layout_2 = Utils::createArrayLayout<ExecutionSpace, MemorySpace>(pm.mesh().localGrid(), 3, Cabana::Grid::Node());

        // std::array<int, 2> global_num_cell = { 7, 7 };
        // std::array<double, 2> global_low_corner = { -1.0, -1.0 };
        // std::array<double, 2> global_high_corner = { 1.0, 1.0 };
        // std::array<bool, 2> is_dim_periodic = { true, true };
        // Cabana::Grid::DimBlockPartitioner<2> partitioner;


        // auto nu_mesh = NuMesh::createMesh<exec_space, memory_space>(MPI_COMM_WORLD);
        // nu_mesh->initialize_ve(global_low_corner, global_high_corner, global_num_cell,
        //                         is_dim_periodic, partitioner, MPI_COMM_WORLD);
        // nu_mesh->initialize_faces();
        // nu_mesh->initialize_edges();
        // nu_mesh->gather_edges();
        // nu_mesh->assign_ghost_edges_to_faces();
        // auto vertex_triple_layout = Utils::createArrayLayout(nu_mesh, 3, NuMesh::Vertex());
        // auto edge_triple_layout = Utils::createArrayLayout(nu_mesh, 3, NuMesh::Edge());
        // auto face_triple_layout = Utils::createArrayLayout(nu_mesh, 3, NuMesh::Face());

        // auto zdot_array_2 = Utils::createArray<exec_space, memory_space>("cabana-v", node_pair_layout_2, Cabana::Grid::Node());
        // auto zdot_numesh_v = Utils::createArray<exec_space, memory_space>("nu-v", vertex_triple_layout, NuMesh::Vertex());
        // auto zdot_numesh_e = Utils::createArray<exec_space, memory_space>("nu-e", edge_triple_layout, NuMesh::Edge());
        // auto zdot_numesh_f = Utils::createArray<exec_space, memory_space>("nu-f", face_triple_layout, NuMesh::Face());


        _zdot = ArrayUtils::createArray<double, memory_space>("velocity", 
                                                       node_triple_layout);
        _wdot = ArrayUtils::createArray<double, memory_space>("vorticity derivative",
                                                       node_pair_layout);
        _ztmp = ArrayUtils::createArray<double, memory_space>("position temporary", 
                                                       node_triple_layout);
        _wtmp = ArrayUtils::createArray<double, memory_space>("vorticity temporary", 
                                                       node_pair_layout);
        ArrayUtils::ArrayOp::assign(*_zdot, 0.0);
        ArrayUtils::ArrayOp::assign(*_wdot, 0.0);
    }
    
    /**
     * EntityTag = Cabana::Grid::Node or NuMesh variant
     * DecompositionTag = Cabana::Grid::Own or NuMesh variant
     */
    template <class EntityTag, class DecompositionTag>
    void step( const double delta_t, EntityTag etag, DecompositionTag dtag ) 
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        using mesh_type = typename ProblemManagerType::mesh_type::mesh_type; // This is a Cabana::Grid::Mesh type
        using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, entity_type>;
        
        // Compute the derivatives of position and vorticity at our current point
        auto z_orig = _pm.get( Field::Position() );
        auto w_orig = _pm.get( Field::Vorticity() );
        auto z_tmp = ArrayUtils::ArrayOp::cloneCopy(*z_orig, dtag);
        auto w_tmp = ArrayUtils::ArrayOp::cloneCopy(*w_orig, dtag);

        l2g_type local_L2G = Cabana::Grid::IndexConversion::createL2G( *z_orig->clayout()->layout()->localGrid(), entity_type() );

        

        // TVD RK3 Step One - derivative at forward euler point
        auto z_dot = _zdot;
        auto w_dot = _wdot;

        // Find foward euler point using initial derivative. The zmodel solver
	    // uses the problem manager position and derivative by default.
        _zm.computeDerivatives(*z_dot, *w_dot, etag, dtag);
        // w_dot same after the above call

        // X_tmp = X_tmp + X_dot*delta_t
        // update2: Update two vectors such that a = alpha * a + beta * b.
        ArrayUtils::ArrayOp::update(*z_tmp, 1.0, *z_dot, delta_t, dtag);
        ArrayUtils::ArrayOp::update(*w_tmp, 1.0, *w_dot, delta_t, dtag);
        // Compute derivative at forward euler point from the temporaries
        _zm.computeDerivatives( *z_tmp, *w_tmp, *z_dot, *w_dot, etag, dtag);
        // w_dot not the same after the above line, but zdot is the same
        
        
        // TVD RK3 Step Two - derivative at half-step position
        // derivatives
        // X_tmp = X_tmp*0.25 + X_orig*0.75 + X_dot*delta_t*0.25
        // update3: Update three vectors such that a = alpha * a + beta * b + gamma * c.
        ArrayUtils::ArrayOp::update(*z_tmp, 0.25, *z_orig, 0.75, *z_dot, (delta_t*0.25), dtag);
        ArrayUtils::ArrayOp::update(*w_tmp, 0.25, *w_orig, 0.75, *w_dot, (delta_t*0.25), dtag);
        printView(local_L2G, rank, w_tmp->array()->view(), 1, 1, 1);
        printView(local_L2G, rank, z_tmp->array()->view(), 1, 1, 1);
        // Get the derivatives at the half-setp
        // Still same up to here
        _zm.computeDerivatives( *z_tmp, *w_tmp, *z_dot, *w_dot, etag, dtag);
        // zdot different after the line above.
        
        // TVD RK3 Step Three - Combine start, forward euler, and half step
        // derivatives to take the final full step.
        // (unew = 1/3 uold + 2/3 utmp + 2/3 du_dt_tmp * deltat)
        // X_orig = X_orig*(1/3) + X_tmp*(2/3) + X_dot*delta_t*(2/3)
        // update3: Update three vectors such that a = alpha * a + beta * b + gamma * c.
        ArrayUtils::ArrayOp::update(*z_orig, (1.0/3.0), *z_tmp, (2.0/3.0), *z_dot, (delta_t*2.0/3.0), dtag);
        ArrayUtils::ArrayOp::update(*w_orig, (1.0/3.0), *w_tmp, (2.0/3.0), *w_dot, (delta_t*2.0/3.0), dtag);
        //printView(local_L2G, rank, w_orig->array()->view(), 1, 1, 1);
        // printView(local_L2G, rank, z_orig->array()->view(), 1, 1, 1);
        // printf("*******z_DOT******\n");
        // printView(local_l2g, 0, z_dot->view(), 1, 3, 3);
        // printf("*******z_TMP******\n");
        // printView(local_l2g, 0, z_tmp->view(), 1, 3, 3);
        //printf("*******z_ORIG******\n");
    }

    template <class l2g_type, class View>
    void printView(l2g_type local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
	    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        using execution_space = typename ProblemManagerType::execution_space;
        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, execution_space()),
            KOKKOS_LAMBDA(int i, int j) {
            
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // global i, j
            local_L2G(local_li, local_gi);
            if (option == 1){
                if (dims == 3) {
                    printf("%d %d %.12lf %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("%d %d %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }   
                    else if (dims == 2) {
                        printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                    }
                }
            }
        });
    }

  private:
    const ProblemManagerType& _pm;
    const BoundaryCondition&_bc;
    const ZModelType& _zm;
    std::shared_ptr<mesh_array_type> _zdot, _wdot, _wtmp, _ztmp;
};

} // end namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
