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

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

// The time integrator requires temporary state for runge kutta interpolation
// which are stored as part of this object 
template <class ExecutionSpace, class MemorySpace, class ZModelType>
class TimeIntegrator
{
    using exec_space = ExecutionSpace;
    using mem_space = MemorySpace;
    using device_type = Kokkos::Device<exec_space, mem_space>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      mem_space>;

//    using halo_type = Cabana::Grid::Halo<MemorySpace>;

  public:
    TimeIntegrator( const ProblemManager<exec_space, mem_space> & pm,
                    const BoundaryCondition & bc,
                    const ZModelType & zm )
    : _pm(pm)
    , _bc(bc)
    , _zm(zm)
    {
       
        // Create a layout of the temporary arrays we'll need for velocity
        // intermediate positions, and change in vorticity
        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 3, Cabana::Grid::Node() );
        auto node_pair_layout =
            Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 2, Cabana::Grid::Node() );

        _zdot = Cabana::Grid::createArray<double, mem_space>("velocity", 
                                                       node_triple_layout);
        _wdot = Cabana::Grid::createArray<double, mem_space>("vorticity derivative",
                                                       node_pair_layout);
        _ztmp = Cabana::Grid::createArray<double, mem_space>("position temporary", 
                                                       node_triple_layout);
        _wtmp = Cabana::Grid::createArray<double, mem_space>("vorticity temporary", 
                                                       node_pair_layout);

        _counter = 0;
    }

    void step( const double delta_t ) 
    { 
        // Compute the derivatives of position and vorticity at our current point
        auto z_orig = _pm.get( Cabana::Grid::Node(), Field::Position() );
        auto w_orig = _pm.get( Cabana::Grid::Node(), Field::Vorticity() );
        auto z_tmp = _ztmp->view();
        auto w_tmp = _wtmp->view();
        // auto & halo = _pm.halo(); 

        auto local_grid = _pm.mesh().localGrid();

        // TVD RK3 Step One - derivative at forward euler point
        auto z_dot = _zdot->view();
        auto w_dot = _wdot->view();

        // Find foward euler point using initial derivative. The zmodel solver
	    // uses the problem manager position and derivative by default.
        _zm.computeDerivatives(z_dot, w_dot);

        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for("RK3 Euler Step",
            Cabana::Grid::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
	        z_tmp(i, j, d) = z_orig(i, j, d) + delta_t * z_dot(i, j, d);
            }
            for (int d = 0; d < 2; d++) {
	        w_tmp(i, j, d) = w_orig(i, j, d) + delta_t * w_dot(i, j, d);
            }
        });

        // Compute derivative at forward euler point from the temporaries
        _zm.computeDerivatives( *_ztmp, *_wtmp, z_dot, w_dot);
 
        // TVD RK3 Step Two - derivative at half-step position
        // derivatives
        
        // Take the half-step
        Kokkos::parallel_for("RK3 Half Step",
            Cabana::Grid::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
	        z_tmp(i, j, d) = 0.75*z_orig(i, j, d) 
                    + 0.25 * z_tmp(i, j, d) 
                    + 0.25 * delta_t * z_dot(i, j, d);
            }
            for (int d = 0; d < 2; d++) {
	        w_tmp(i, j, d) = 0.75*w_orig(i, j, d) 
                    + 0.25 * w_tmp(i, j, d) 
                    + 0.25 * delta_t * w_dot(i, j, d);
            }
        });
        // Get the derivatives at the half-setp
        _zm.computeDerivatives( *_ztmp, *_wtmp, z_dot, w_dot);
        
        // TVD RK3 Step Three - Combine start, forward euler, and half step
        // derivatives to take the final full step.
        // unew = 1/3 uold + 2/3 utmp + 2/3 du_dt_tmp * deltat
        auto low = _pm.spatial_mesh().boundingBoxMin();
        auto high = _pm.spatial_mesh().boundingBoxMax();
        Kokkos::parallel_for("RK3 Full Step",
            Cabana::Grid::createExecutionPolicy(own_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
                z_orig(i, j, d) = ( 1.0 / 3.0 ) * z_orig(i, j, d) 
                        + ( 2.0 / 3.0 ) * z_tmp(i, j, d) 
                        + ( 2.0 / 3.0 ) * delta_t * z_dot(i, j, d);
                if (z_orig(i, j, d) < low[d] || z_orig(i, j, d) > high[d])
                {
                    printf("Point has moved outside bounding box. Exiting.\n");
                    exit(1);
                }
            }
            for (int d = 0; d < 2; d++) {
	        w_orig(i, j, d) = ( 1.0 / 3.0 ) * w_orig(i, j, d) 
                    + ( 2.0 / 3.0 ) * w_tmp(i, j, d) 
                    + ( 2.0 / 3.0 ) * delta_t * w_dot(i, j, d);
            }
        });

        // _counter++;
        // print_view(50, z_orig);
        // print_view(75, z_orig);
        // print_view(100, z_orig);
    }

    template <class View>
    void print_view(int timestep, View z)
    {
        if (_counter != timestep) return;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("\n\n\n\nR%d: t%d: printing view...\n", rank, _counter);
        // std::string filename = "output_pos_rank" + std::to_string(rank) + ".txt";
        // FILE *fptr;
        // fptr = fopen(filename.c_str(), "w");

        auto local_L2G = Cabana::Grid::IndexConversion::createL2G(*_pm.mesh().localGrid(), Cabana::Grid::Node());
        // auto z = _zdot->view();

        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
        Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

       Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            
            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            //printf("global: %d %d\n", local_gi[0], local_gi[1]);
            if (dims == 3) {
                printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                //fprintf(fptr, "R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
            }
            else if (dims == 2) {
                printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
            }
        });

        //fclose(fptr);

    }

  private:
    const ProblemManager<ExecutionSpace, MemorySpace> & _pm;
    const BoundaryCondition &_bc;
    const ZModelType & _zm;
    std::shared_ptr<node_array> _zdot, _wdot, _wtmp, _ztmp;

    int _counter;
};

} // end namespace Beatnik

#endif // BEATNIK_TIMEINTEGRATOR_HPP
