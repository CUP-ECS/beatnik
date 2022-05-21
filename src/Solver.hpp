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

#include <Cajita_HypreStructuredSolver.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>
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

namespace BEATNIK
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

template <class ExecutionSpace, class MemorySpace>
class Solver;

//---------------------------------------------------------------------------//
template <class ExecutionSpace, class MemorySpace>
class Solver<ExecutionSpace, MemorySpace> : public SolverBase
{
  public:
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;
    using node_array =
        Cajita::Array<double, Cajita::Node, mesh_type, MemorySpace>;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using bc_type = BoundaryCondition;

    using Node = Cajita::Node;

    template <class InitFunc>
    Solver( MPI_Comm comm, const std::array<int, 2>& global_num_cell,
            const Cajita::BlockPartitioner<2>& partitioner,
            const double atwood, const double g, const InitFunc& create_functor,
            const BoundaryCondition& bc, const ArtificialViscosity& av,
            const double delta_t)
        : _halo_min( 2 )
        , _atwood( atwood )
        , _g( g )
        , _bc( bc )
        , _av( av )
        , _dt( delta_t )
        , _time( 0.0 )
    {

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _mesh = std::make_unique<Mesh<ExecutionSpace, MemorySpace>>(
            global_num_cell, partitioner, _halo_min, comm );

        // Check that our timestep is small enough to handle the mesh size,
        // atwood number and acceleration, and solution method. 
	// XXXX

        // Create a problem manager to manage mesh state
        _pm = std::make_unique<ProblemManager<ExecutionSpace, MemorySpace>>(
            _mesh, create_functor );

        // Set up Silo for I/O
        _silo =
            std::make_unique<SiloWriter<2, ExecutionSpace, MemorySpace>>( _pm );
    }

    void setup() override
    {
        // Should assert that _time == 0 here.

	// Apply boundary conditions
	_bc.apply()
    }

    void step() override
    {
        TimeIntegrator::step( ExecutionSpace(), *_pm, _dt, _bc );
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
                _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
            }
            t++;
        } while ( ( _time < t_final ) );
    }

  private:
    /* Solver state variables */
    int _halo_min;
    double _atwood;
    BoundaryCondition _bc;
    ArtificialViscosity _av;
    ZModel<ZModelOrder> _zm;
    TimeIntegrator<
    double _dt;
    double _time;
    std::unique_ptr<Mesh<ExecutionSpace, MemorySpace>> _mesh;
    std::unique_ptr<ProblemManager<ExecutionSpace, MemorySpace>> _pm;
    std::unique_ptr<SiloWriter<ExecutionSpace, MemorySpace>> _silo;
    int _rank;
};

//---------------------------------------------------------------------------//
// XXX Should this be a shared_ptr or a unique_ptr?
// Creation method.
template <class InitFunc>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const std::array<int, 2>& global_num_cell,
              const Cajita::BlockPartitioner& partitioner,
              const double atwood, const double g, 
              const InitFunc& create_functor, 
              const BoundaryCondition& bc, 
              const ArtificialViscosity& bc, 
              const double delta_t )
{
    if ( 0 == device.compare( "serial" ) )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_shared<
            Beatnik::Solver<Kokkos::Serial, Kokkos::HostSpace>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, delta_t);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_shared<
            Beatnik::Solver<Kokkos::OpenMP, Kokkos::HostSpace>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, delta_t);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            Beatnik::Solver<Kokkos::Cuda, Kokkos::CudaSpace>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, delta_t);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<
            Kokkos::Experimental::HIP, Kokkos::Experimental::HIPSpace>>(
            Beatnik::Solver<Kokkos::Cuda, Kokkos::CudaSpace>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, delta_t);
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
