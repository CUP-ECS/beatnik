/****************************************************************************
 * Copyright (c) 2018-2020 by the Beatnik authors                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is           *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_SOLVER_HPP
#define BEATNIK_SOLVER_HPP

#include <Cajita_Partitioner.hpp>
#include <Cajita_Types.hpp>

#include <ArtificialViscosity.hpp>
#include <BoundaryCondition.hpp>
#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>

#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

#include <mpi.h>

namespace Beatnik
{
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
 * 1. The BoundaryCondition and ArtificialViscosity objects are created by
 *    the calling application and passed in, so we don't control their 
 *    memory. As a result, the Solver object makes a copy of them (they're 
 *    small) and passes references of those to the objects it uses. 
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
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>, MemorySpace>;
    using zmodel_type = ZModel<ExecutionSpace, MemorySpace, ModelOrder>;
    using ti_type = TimeIntegrator<ExecutionSpace, MemorySpace, ModelOrder>;
    using Node = Cajita::Node;

    template <class InitFunc>
    Solver( MPI_Comm comm, const std::array<int, 2>& global_num_cell,
            const Cajita::BlockPartitioner<2>& partitioner,
            const double atwood, const double g, const InitFunc& create_functor,
            const BoundaryCondition& bc, const ArtificialViscosity& av,
            const double epsilon, const double delta_t)
        : _halo_min( 2 )
        , _atwood( atwood )
        , _g( g )
        , _bc( bc )
        , _av( av )
        , _dt( delta_t )
        , _eps( epsilon )
        , _time( 0.0 )
    {
	std::array<bool, 2> periodic;
        periodic[0] =  (bc.boundary_type[0] == PERIODIC);
        periodic[1] =  (bc.boundary_type[1] == PERIODIC);

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _mesh = std::make_unique<Mesh<ExecutionSpace, MemorySpace>>(
            global_num_cell, periodic, partitioner, _halo_min, comm );

        // Check that our timestep is small enough to handle the mesh size,
        // atwood number and acceleration, and solution method. 
	// XXXX

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<ProblemManager<ExecutionSpace, MemorySpace>>(
            *_mesh, create_functor );

        // Create the ZModel solver
        _zm = std::make_unique<ZModel<ExecutionSpace, MemorySpace, ModelOrder>>(
            *_pm, _bc, _av, atwood, g);
        // Make a time integrator to move the zmodel forward
        _ti = std::make_unique<TimeIntegrator<ExecutionSpace, MemorySpace, ModelOrder>>( *_pm, *_zm );

        // Set up Silo for I/O
        _silo =
            std::make_unique<SiloWriter<ExecutionSpace, MemorySpace>>( *_pm );
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

        _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
        Kokkos::Profiling::popRegion();

        num_step = t_final / _dt;

        // Start advancing time.
        do
        {
            if ( 0 == _mesh->rank() && 0 == t % write_freq )
                printf( "Step %d / %d at time = %f\n", t, num_step, _time );

            step();

            // 4. Output mesh state periodically
            if ( 0 == t % write_freq )
            {
                //_silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
            }
            t++;
        } while ( ( _time < t_final ) );
    }

  private:
    /* Solver state variables */
    int _halo_min;
    double _atwood;
    double _g;
    BoundaryCondition _bc;
    ArtificialViscosity _av;
    double _dt;
    double _eps;
    double _time;
    std::unique_ptr<zmodel_type> _zm;
    std::unique_ptr<ti_type> _ti;
    std::unique_ptr<Mesh<ExecutionSpace, MemorySpace>> _mesh;
    std::unique_ptr<ProblemManager<ExecutionSpace, MemorySpace>> _pm;
    std::unique_ptr<SiloWriter<ExecutionSpace, MemorySpace>> _silo;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc, class ModelOrder>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const std::array<int, 2>& global_num_cell,
              const Cajita::BlockPartitioner<2> & partitioner,
              const double atwood, const double g, 
              const InitFunc& create_functor, 
              const BoundaryCondition& bc, 
              const ArtificialViscosity& av,
              const ModelOrder,
              const double epsilon, 
              const double delta_t )
{
    if ( 0 == device.compare( "serial" ) )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_shared<
            Beatnik::Solver<Kokkos::Serial, Kokkos::HostSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, av, epsilon, delta_t);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_shared<
            Beatnik::Solver<Kokkos::OpenMP, Kokkos::HostSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, av, epsilon, delta_t);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            Beatnik::Solver<Kokkos::Cuda, Kokkos::CudaSpace, ModelOrder>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, av, epsilon, delta_t);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<Beatnik::Solver<Kokkos::Experimental::HIP, 
            Kokkos::Experimental::HIPSpace, ModelOrder>>(
                comm, global_num_cell, partitioner, atwood, g, 
                create_functor, bc, av, epsilon, delta_t);
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
