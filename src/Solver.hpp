/****************************************************************************
 * Copyright (c) 2020-2022 by the Beatnik authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is                     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_SOLVER_HPP
#define BEATNIK_SOLVER_HPP

#include <Beatnik_Config.hpp>

#include <Cabana_Grid.hpp>

#include <BoundaryCondition.hpp>
#include <SurfaceMesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>
#include <CreateBRSolver.hpp>

#include <ZModel.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace Beatnik
{

/**
 * @struct Params
 * @brief Holds order and solver-specific parameters
 */
struct Params
{
    /* Save the period from command-line args to pass to 
     * ProblemManager to seed the random number generator
     * to initialize position
     */
    double period;

    /* Mesh data, for solvers that create another mesh */
    std::array<double, 6> global_bounding_box;
    std::array<bool, 2> periodic;

    /* Model Order */
    int solver_order;

    /* BR solver type */
    BRSolverType br_solver;

    /* Cutoff distance for cutoff-based BRSolver */
    double cutoff_distance;

    /* Heffte configuration options for low-order model: 
        Value	All-to-all	Pencils	Reorder
        0	    False	    False	False
        1	    False	    False	True
        2	    False	    True	False
        3	    False	    True	True
        4	    True	    False	False
        5	    True	    False	True
        6	    True	    True	False (Default)
        7	    True	    True	True
    */
    int heffte_configuration;
};

/*
 * Convenience base class so that examples that use this don't need to know
 * the details of the problem manager/mesh/etc templating.
 */
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void setup( void ) = 0;
    virtual void step( void ) = 0;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

//---------------------------------------------------------------------------//

/* A note on memory management:
 * 1. The BoundaryCondition object is created by the calling application 
 *    and passed in, so we don't control their memory. As a result, 
 *    the Solver object makes a copy of it (it's small) and passes 
 *    references of those to the objects it uses. 
 * 2. The other objects created by the solver (mesh, problem manager, 
 *    time integrator, and zmodel) are owned and managed by the solver, and 
 *    are managed as unique_ptr by the Solver object. They are passed 
 *    by reference to the classes that use them, which store the references
 *    as const references.
 */

template <class ExecutionSpace, class MemorySpace, class ModelOrder>
class Solver : public SolverBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>, MemorySpace>;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using zmodel_type = ZModel<ExecutionSpace, MemorySpace, ModelOrder, Params>;
    using ti_type = TimeIntegrator<ExecutionSpace, MemorySpace, zmodel_type>;
    using Node = Cabana::Grid::Node;

    template <class InitFunc>
    Solver( MPI_Comm comm,
            const std::array<int, 2>& num_nodes,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            const double atwood, const double g, const InitFunc& create_functor,
            const BoundaryCondition& bc, const double mu, 
            const double epsilon, const double delta_t,
            const Params params)
        : _halo_min( 2 )
        , _atwood( atwood )
        , _g( g )
        , _bc( bc )
        , _mu( mu )
        , _eps( epsilon )
        , _dt( delta_t )
        , _time( 0.0 )
        , _params( params )
    {

        _params.periodic[0] = (bc.boundary_type[0] == PERIODIC);
        _params.periodic[1] = (bc.boundary_type[1] == PERIODIC);

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _surface_mesh = std::make_unique<SurfaceMesh<ExecutionSpace, MemorySpace>>(
            _params.global_bounding_box, num_nodes, _params.periodic, partitioner,
	    _halo_min, comm );

        // Check that our timestep is small enough to handle the mesh size,
        // atwood number and acceleration, and solution method. 
	// XXXX

        // Compute dx and dy in the initial problem state XXX What should this
        // be when the mesh doesn't span the bounding box, e.g. rising bubbles?

        // If we're non-periodic, there's one fewer cells than nodes (we don't 
        // have the cell which wraps around
        std::array<int, 2> num_cells = num_nodes;
        for (int i = 0; i < 2; i++)
            if (!_params.periodic[i]) num_cells[i]--;

        double dx = (_params.global_bounding_box[4] - _params.global_bounding_box[0]) 
            / (num_cells[0]);
        double dy = (_params.global_bounding_box[5] - _params.global_bounding_box[1]) 
            / (num_cells[1]);

        // Adjust down mu and epsilon by sqrt(dx * dy)
        _mu = _mu * sqrt(dx * dy);
        _eps = _eps * sqrt(dx * dy);

#if 0
        std::cout << "===== Solver Parameters =====\n"
                  << "dx = " << dx << ", " << "dy = " << dy << "\n"
                  << "dt = " << delta_t << "\n"
                  << "g = " << _g << "\n"
                  << "atwood = " << _atwood << "\n"
                  << "mu = " << _mu << "\n"
                  << "eps = " << _eps << "\n"
                  << "=============================\n";
#endif

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<pm_type>(
            *_surface_mesh, _bc, _params.period, create_functor );

        if (_params.solver_order == 1 || _params.solver_order  == 2)
        {
            _br = Beatnik::createBRSolver<pm_type, ExecutionSpace, MemorySpace, Params>(*_pm, _bc, _eps, dx, dy, _params);
        }
        else
        {
            _br = NULL;
        }

        // Create the ZModel solver
        _zm = std::make_unique<ZModel<ExecutionSpace, MemorySpace, ModelOrder, Params>>(
            *_pm, _bc, _br.get(), dx, dy, _atwood, _g, _mu, _params.heffte_configuration);

        // Make a time integrator to move the zmodel forward
        _ti = std::make_unique<TimeIntegrator<ExecutionSpace, MemorySpace, zmodel_type>>( *_pm, _bc, *_zm );

        // Set up Silo for I/O
        _silo = std::make_unique<SiloWriter<ExecutionSpace, MemorySpace>>( *_pm );
    }

    void setup() override
    {
        // Should assert that _time == 0 here.

	// Apply boundary conditions
    }

    void step() override
    {
        _ti->step(_dt);
        _time += _dt;
    }

    void solve( const double t_final, const int write_freq ) override
    {
        int t = 0;
        int num_step;

        Kokkos::Profiling::pushRegion( "Solve" );

        if (write_freq > 0) {
            _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
        }

        num_step = t_final / _dt;

        // Start advancing time.
        do
        {
            if ( 0 == _surface_mesh->rank() )
                printf( "Step %d / %d at time = %f\n", t, num_step, _time );

            step();
            t++;
            // 4. Output mesh state periodically
            if ( write_freq && (0 == t % write_freq ))
            {
                _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
            }
        } while ( ( _time < t_final ) );
        Kokkos::Profiling::popRegion();
    }

  private:
    /* Solver state variables */
    int _halo_min;
    double _atwood;
    double _g;
    double _mu, _eps;
    double _dt;
    double _time;
    
    BoundaryCondition _bc;
    Params _params;
    
    std::unique_ptr<SurfaceMesh<ExecutionSpace, MemorySpace>> _surface_mesh;
    std::unique_ptr<pm_type> _pm;
    std::unique_ptr<BRSolverBase<ExecutionSpace, MemorySpace, Params>> _br;
    std::unique_ptr<zmodel_type> _zm;
    std::unique_ptr<ti_type> _ti;
    std::unique_ptr<SiloWriter<ExecutionSpace, MemorySpace>> _silo;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc, class ModelOrder, class Params>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const std::array<int, 2>& global_num_cell,
              const Cabana::Grid::BlockPartitioner<2> & partitioner,
              const double atwood, const double g, 
              const InitFunc& create_functor, 
              const BoundaryCondition& bc, 
              const ModelOrder,
              const double mu,
              const double epsilon, 
              const double delta_t,
              const Params params )
{
    if ( 0 == device.compare( "serial" ) )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_shared<
            Beatnik::Solver<Kokkos::Serial, Kokkos::HostSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "threads" ) )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        return std::make_shared<
            Beatnik::Solver<Kokkos::Threads, Kokkos::HostSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "Threads Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_shared<
            Beatnik::Solver<Kokkos::OpenMP, Kokkos::HostSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#if defined(KOKKOS_ENABLE_CUDA)
        return std::make_shared<
            Beatnik::Solver<Kokkos::Cuda, Kokkos::CudaSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<Beatnik::Solver<Kokkos::HIP, 
            Kokkos::Experimental::HIPSpace, ModelOrder>>(
                comm, global_num_cell, partitioner, atwood, g, 
                create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "HIP Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_SOLVER_HPP
