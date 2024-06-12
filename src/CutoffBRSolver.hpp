/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier,BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file CutoffBRSolver.hpp
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Class that uses a brute force approach to calculating the Birkhoff-Rott 
 * velocity intergral by using a all-pairs approach. Communication
 * uses a standard ring-pass communication algorithm. Does not attempt to 
 * reduce amount of computation per ring pass by using symetry of forces
 * as this complicates the GPU kernel.
 * 
 * Unlike the ExactBRSolver, calculations are limited by a cutoff distance.
 */

#ifndef BEATNIK_CUTOFFBRSOLVER_HPP
#define BEATNIK_CUTOFFBRSOLVER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <BRSolverBase.hpp>
#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>
#include <Migrator.hpp>

namespace Beatnik
{

/**
 * The CutoffBRSolver Class
 * @class CutoffBRSolver
 * @brief Directly solves the Birkhoff-Rott integral using brute-force 
 * all-pairs calculation, limited by a cutoff distance
 **/
template <class ExecutionSpace, class MemorySpace, class Params>
class CutoffBRSolver : public BRSolverBase<ExecutionSpace, MemorySpace, Params>
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using spatial_mesh_type = SpatialMesh<ExecutionSpace, MemorySpace>;
    using migrator_type = Migrator<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    
    using Node = Cabana::Grid::Node;
    using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    using node_array = typename pm_type::node_array;
    //using node_view = typename pm_type::node_view;
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    CutoffBRSolver( const pm_type &pm, const BoundaryCondition &bc, const spatial_mesh_type &spm,
                migrator_type &migrator, const double epsilon, const double dx, const double dy,
                const double cutoff_distance)
        : _pm( pm )
        , _bc( bc )
        , _spm( spm )
        , _migrator( migrator )
        , _epsilon( epsilon )
        , _dx( dx )
        , _dy( dy )
        , _cutoff_distance (cutoff_distance )
        , _local_L2G( *_pm.mesh().localGrid() )
    {
	    _comm = _pm.mesh().localGrid()->globalGrid().comm();
    }

    static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
    {
        if (index == (len - 1) || index == 0) return 3.0/8.0;
        else if (index % 3 == 0) return 3.0/4.0;
        else return 9.0/8.0;
    }

    

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. 
     * This function is called three times per time step to compute the initial, forward, and half-step
     * derivatives for velocity and vorticity.
     */
    void computeInterfaceVelocity(node_view zdot, node_view z, node_view w, node_view o) const override
    {

        // Perform cutoff solve
        _migrator.initializeParticles(z, w, o);
        _migrator.migrateParticlesTo3D();
        _migrator.performHaloExchange3D();
        _migrator.computeInterfaceVelocityNeighbors(_dy, _dx, _epsilon);
        _migrator.migrateParticlesTo2D();
        _migrator.populate_zdot(zdot);
        
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

        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            
            // local_gi = global versions of the local indicies, and convention for remote 
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // i, j
            local_L2G(local_li, local_gi);
            //printf("global: %d %d\n", local_gi[0], local_gi[1]);
            if (option == 1){
                if (dims == 3) {
                    printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
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
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    const spatial_mesh_type &_spm;
    migrator_type &_migrator;
    double _epsilon, _dx, _dy;
    const double _cutoff_distance;
    MPI_Comm _comm;
    l2g_type _local_L2G;
    // XXX Communication views and extents to avoid allocations during each ring pass
};

}; // namespace Beatnik

#endif // BEATNIK_CUTOFFBRSOLVER_HPP
