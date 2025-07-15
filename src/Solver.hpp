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

#include <Beatnik_Types.hpp>
#include <BoundaryCondition.hpp>
#include <CreateMesh.hpp>
#include <ProblemManager.hpp>
#include <SiloWriter.hpp>
#include <TimeIntegrator.hpp>
#include <VTKWriter.hpp>
#include <CreateBRSolver.hpp>

#include <ZModel.hpp>

#include <memory>
#include <string>

#include <mpi.h>

#ifndef WRITE_VIEWS
#define WRITE_VIEWS 1
#endif
#if WRITE_VIEWS
#include "../tests/TestingUtils.hpp"
#endif


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
    /**
     * Mesh type options:
     *  0: Structured, 2D (Regular, rectangular, domain decomposition)
     *  1: Unstructured, 2D (Domain decomposed into triangles with potential for mesh refinement)
     */
    virtual void step( void ) = 0;
    virtual void solve( const double t_final, const int write_freq ) = 0;

    // For testing purposes
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;
    virtual View_t get_positions( Cabana::Grid::Node ) = 0;
    virtual View_t get_vorticities( Cabana::Grid::Node ) = 0;
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
template <class ExecutionSpace, class MemorySpace, class ModelOrderTag, class MeshTypeTag>
class StructuredSolver : public SolverBase
{
  public:
    // using mesh_array_type =
    //     Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>, MemorySpace>;
    using beatnik_mesh_type = MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>;
    using entity_type = typename beatnik_mesh_type::entity_type;
    using pm_type = ProblemManager<beatnik_mesh_type>;
    using br_solver_type = BRSolverBase<pm_type, Params>;
    using zmodel_type = ZModel<pm_type, br_solver_type, ModelOrderTag>;
    using ti_type = TimeIntegrator<pm_type, zmodel_type>;
    using silo_writer_type = SiloWriter<pm_type>;
    
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;


    template <class InitFunc>
    StructuredSolver( MPI_Comm comm,
            const std::array<int, 2>& num_nodes,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            const double atwood, const double g, const InitFunc& create_functor,
            const BoundaryCondition& bc, const double mu, 
            const double epsilon, const double delta_t,
            const Params params)
        : _halo_min( 2 )
        , _atwood( atwood )
        , _g( g )
        , _mu( mu )
        , _eps( epsilon )
        , _dt( delta_t )
        , _time( 0.0 )
        , _bc( bc )
        , _params( params )
    {

        _params.periodic[0] = (bc.boundary_type[0] == MeshBoundaryType::PERIODIC);
        _params.periodic[1] = (bc.boundary_type[1] == MeshBoundaryType::PERIODIC);

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _mesh = createMesh<ExecutionSpace, MemorySpace, MeshTypeTag>(_params.global_bounding_box, num_nodes, _params.periodic,
                            partitioner, _halo_min, comm );

        // XXX - Check that our timestep is small enough to handle the mesh size,
        // atwood number and acceleration, and solution method. 
        // Compute dx and dy in the initial problem state
        // XXX - What should this be when the mesh doesn't span the bounding box, e.g. rising bubbles?

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
            *_mesh, _bc, _params.period, create_functor );

        
        if (_params.solver_order == 1 || _params.solver_order  == 2)
        {
            _br = Beatnik::createBRSolver<pm_type, Params>(*_pm, _bc, _eps, dx, dy, _params);
        }
        else
        {
            _br = NULL;
        }

        // Create the ZModel solver
        _zm = std::make_unique<zmodel_type>(
            *_pm, _bc, _br.get(), dx, dy, _atwood, _g, _mu, _params.heffte_configuration);

        // Make a time integrator to move the zmodel forward
        _ti = std::make_unique<ti_type>( *_pm, _bc, *_zm );

        // Set up Silo for I/O
        _silo = std::make_unique<silo_writer_type>( *_pm );
    }

    void setup() override
    {
        // XXX - Should assert that _time == 0 here.

	    // XXX - Apply boundary conditions
    }

    void step() override
    {
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
        {
            _ti->step(_dt, entity_type(), Cabana::Grid::Own());
        }
        else if constexpr (std::is_same_v<MeshTypeTag, Mesh::Unstructured>)
        {
            printf("WARNING: Solver::step: Unstructured mesh not yet implemented.\n");
            // ti->step(_dt, entity_type(), NuMesh::Own());
            // throw std::invalid_argument("Solver::step: Unstructured mesh not yet implemented.");
        }
        else
        {
            throw std::invalid_argument("Solver::step: Invalid mesh_type argument.");
        }
        _time += _dt;
    }

    void solve( const double t_final, const int write_freq ) override
    {
        using mesh_type_tag = typename pm_type::mesh_type_tag;

        int t = 0;
        int num_step;
        
        Kokkos::Profiling::pushRegion( "Solve" );

        if (write_freq > 0) {
            
            if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
                _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
            else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
            {
                auto vtk_writer = createVTKWriter(*_pm );
                vtk_writer->vtkWrite(t);
            }

        }

        num_step = t_final / _dt;

        // Start advancing time.
        do
        {
            if ( 0 == _mesh->rank() )
                printf( "Step %d / %d at time = %f\n", t, num_step, _time );

            step();
            t++;
            // 4. Output mesh state periodically
            if ( write_freq && (0 == t % write_freq ))
            {
                if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
                {
                    _silo->siloWrite( strdup( "Mesh" ), t, _time, _dt );
                }
                    
                else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
                {
                    auto vtk_writer = createVTKWriter(*_pm );
                    vtk_writer->vtkWrite(t);
                }
            }

            // Write views for future testing, if enabled
            #if WRITE_VIEWS
            if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
            {
                int write_at_time = 180;
                int rank, comm_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

                if (t == write_at_time)
                {
                    auto z = _pm->get(Field::Position())->array()->view();
                    auto w = _pm->get(Field::Vorticity())->array()->view();
                    int mesh_size = _pm->mesh().mesh_size();
                    int periodic = _params.periodic[0];
                    // void writeView(int rank, int comm_size, int mesh_size, const View v)
                    BeatnikTest::Utils::writeView(rank, comm_size, mesh_size, periodic, z);
                    BeatnikTest::Utils::writeView(rank, comm_size, mesh_size, periodic, w);
                }
            }
            #endif

        } while ( ( _time < t_final ) );
        Kokkos::Profiling::popRegion();
    }

    // XXX - Template these functions on EntityType. Currently
    // causes a linking error when trying to do so.
    // For testing purposes
    View_t get_positions(Cabana::Grid::Node) override
    {
        //_pm->gather();
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
        {
            auto view = _pm->get(Field::Position())->array()->view();
            int dim0 = view.extent(0);
            int dim1 = view.extent(1);
            auto temp = Kokkos::create_mirror_view(view);
            View_t ret = View_t("ret_p", dim0, dim1, 3);
            Kokkos::deep_copy(temp, view);
            Kokkos::deep_copy(ret, temp); 
            return ret;
        }
        else { throw std::invalid_argument("Solver::get_positions: Unstructured mesh not yet supported.\n");} 
    }
    View_t get_vorticities(Cabana::Grid::Node) override
    {
        //_pm->gather();
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
        {
            auto view = _pm->get(Field::Vorticity())->array()->view();
            int dim0 = view.extent(0);
            int dim1 = view.extent(1);
            auto temp = Kokkos::create_mirror_view(view);
            View_t ret = View_t("ret_w", dim0, dim1, 2);
            Kokkos::deep_copy(temp, view);
            Kokkos::deep_copy(ret, temp); 
            return ret;
        }
        else { throw std::invalid_argument("Solver::get_positions: Unstructured mesh not yet supported.\n");} 
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
    
    std::unique_ptr<beatnik_mesh_type> _mesh;
    std::unique_ptr<pm_type> _pm;
    std::unique_ptr<br_solver_type> _br;
    std::unique_ptr<zmodel_type> _zm;
    std::unique_ptr<ti_type> _ti;
    std::unique_ptr<silo_writer_type> _silo;
};

//---------------------------------------------------------------------------//
// Creation method which passes structured mesh paramters.
template <class InitFunc, class ModelOrderTag, class MeshTypeTag, class Params>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const std::array<int, 2>& global_num_cell,
              const Cabana::Grid::BlockPartitioner<2> & partitioner,
              const double atwood, const double g, 
              const InitFunc& create_functor, 
              const BoundaryCondition& bc, 
              const ModelOrderTag,
              const MeshTypeTag,
              const double mu,
              const double epsilon, 
              const double delta_t,
              const Params params )
{
    if ( 0 == device.compare( "serial" ) )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_shared<
            Beatnik::StructuredSolver<Kokkos::Serial, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
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
            Beatnik::StructuredSolver<Kokkos::Threads, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
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
            Beatnik::StructuredSolver<Kokkos::OpenMP, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
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
            Beatnik::StructuredSolver<Kokkos::Cuda, Kokkos::CudaSpace, ModelOrderTag, MeshTypeTag>>(
            comm, global_num_cell, partitioner, atwood, g, 
            create_functor, bc, mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<Beatnik::StructuredSolver<Kokkos::HIP, 
            Kokkos::Experimental::HIPSpace, ModelOrderTag, MeshTypeTag>>(
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

template <class ExecutionSpace, class MemorySpace, class ModelOrderTag, class MeshTypeTag>
class UnstructuredSolver : public SolverBase
{
  public:
    // using mesh_array_type =
    //     Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>, MemorySpace>;
    using beatnik_mesh_type = MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>;
    using entity_type = typename beatnik_mesh_type::entity_type;
    using pm_type = ProblemManager<beatnik_mesh_type>;
    using br_solver_type = BRSolverBase<pm_type, Params>;
    using zmodel_type = ZModel<pm_type, br_solver_type, ModelOrderTag>;
    using ti_type = TimeIntegrator<pm_type, zmodel_type>;
    using vtk_writer_type = VTKWriter<pm_type>;
    
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;


    template <class PositionsAoSoA, class VerticesAoSoA, class FacesAoSoA>
    UnstructuredSolver( MPI_Comm comm,
            const PositionsAoSoA& positions,
            const VerticesAoSoA& vertices,
            const FacesAoSoA& faces,
            const BoundaryCondition& bc,
            const double atwood, const double g,
            const double mu, 
            const double epsilon, const double delta_t,
            const Params params)
        : _halo_min( 2 )
        , _bc( bc )
        , _atwood( atwood )
        , _g( g )
        , _mu( mu )
        , _eps( epsilon )
        , _dt( delta_t )
        , _time( 0.0 )
        , _params( params )
    {
        // Copy positions, vertices, and faces to the correct memory space
        // Host-side AoSoAs for storing VTU data
        using vertices_d = Cabana::MemberTypes<int,       // Vertex global ID                                 
                                               int       // Owning rank
                                               >;
        using face_d = Cabana::MemberTypes<int[3],       // Vertex LIDs forming the triangle                                
                                           bool         // Flag indicating if the cell contains a ghost point
                                           >;
        using triple_d = Cabana::MemberTypes<double[3]>; // Vertex positions
        using vert_aosoa = Cabana::AoSoA<vertices_d, MemorySpace, 4>;
        using face_aosoa = Cabana::AoSoA<face_d, MemorySpace, 4>;
        using triple_aosoa = Cabana::AoSoA<triple_d, MemorySpace, 4>;
        vert_aosoa vertices_device("vertices_device", vertices.size());
        face_aosoa faces_device("faces_device", faces.size());
        triple_aosoa positions_device("positions_device", positions.size());
        Cabana::deep_copy(vertices_device, vertices);
        Cabana::deep_copy(faces_device, faces);
        Cabana::deep_copy(positions_device, positions);

        // Create a mesh one which to do the solve and a problem manager to
        // handle state
        _mesh = createMesh<ExecutionSpace, MemorySpace, MeshTypeTag>(vertices, faces, comm );

        // XXX - Check that our timestep is small enough to handle the mesh size,
        // atwood number and acceleration, and solution method. 
        // Compute dx and dy in the initial problem state
        // XXX - What should this be when the mesh doesn't span the bounding box, e.g. rising bubbles?

        // If we're non-periodic, there's one fewer cells than nodes (we don't 
        // have the cell which wraps around
        // 

        // Adjust down mu and epsilon by sqrt(dx * dy)
        // XXX - how does this work for the unstructured case?
        // double dx = (_params.global_bounding_box[4] - _params.global_bounding_box[0]) 
        //     / (num_cells[0]);
        // double dy = (_params.global_bounding_box[5] - _params.global_bounding_box[1]) 
        //     / (num_cells[1]);
        double dx = 1.0;
        double dy = 1.0;
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
            *_mesh, bc, positions );

        _br = NULL;
        // if (_params.solver_order == 1 || _params.solver_order  == 2)
        // {
        //     _br = Beatnik::createBRSolver<pm_type, Params>(*_pm, _bc, _eps, dx, dy, _params);
        // }
        // else
        // {
        //     _br = NULL;
        // }

        // Create the ZModel solver
        // _zm = std::make_unique<zmodel_type>(
        //     *_pm, bc, _br.get(), dx, dy, _atwood, _g, _mu, _params.heffte_configuration);

        // Make a time integrator to move the zmodel forward
        // _ti = std::make_unique<ti_type>( *_pm, bc, *_zm );
    }

    void setup() override
    {
        // XXX - Should assert that _time == 0 here.

	    // XXX - Apply boundary conditions
    }

    void step() override
    {
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Unstructured>)
        {
            // printf("WARNING: Solver::step: Unstructured mesh not yet implemented.\n");
            // ti->step(_dt, entity_type(), NuMesh::Own());
            // throw std::invalid_argument("Solver::step: Unstructured mesh not yet implemented.");
        }
        else
        {
            throw std::invalid_argument("Solver::step: Invalid mesh_type argument.");
        }
        _time += _dt;
    }

    void solve( const double t_final, const int write_freq ) override
    {
        using mesh_type_tag = typename pm_type::mesh_type_tag;

        int t = 0;
        int num_step;
        
        Kokkos::Profiling::pushRegion( "Solve" );

        if (write_freq > 0) {
            
            if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
            {
                auto vtk_writer = createVTKWriter(*_pm );
                vtk_writer->vtkWrite(t);
            }

        }

        num_step = t_final / _dt;

        // Start advancing time.
        do
        {
            if ( 0 == _mesh->rank() )
                printf( "Step %d / %d at time = %f\n", t, num_step, _time );

            // Refine the mesh for testing purposes
            int num_local_faces = _mesh->layoutObj()->count(NuMesh::Own(), NuMesh::Face());
            auto vef_gid_start = _mesh->layoutObj()->vef_gid_start();
            int face_gid_start = vef_gid_start(_mesh->rank(), 2);
            Kokkos::View<int*, MemorySpace> fin("fin", num_local_faces);
            Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<ExecutionSpace>(0, num_local_faces),
                KOKKOS_LAMBDA(int i) {
    
                    fin(i) = face_gid_start + i;
    
                });
            // Kokkos::View<int[1], MemorySpace> fin("fin");
            // Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<ExecutionSpace>(0, fin.extent(0)),
            // KOKKOS_LAMBDA(int i) {

            //     fin(i) = t+15;
            //     //printf("refining face %d\n", t+15);

            // });
            auto positions = _pm->get( Field::Position() );
            // printf("Before refine: R%d: pos: %d, verts: %d\n",
            //     _mesh->rank(), positions->array()->aosoa()->size(),
            //     _mesh->layoutObj()->vertices().size());
            _mesh->refine(fin);
            _pm->gather();
            // printf("After refine and update: R%d: pos: %d, verts: %d\n",
            //     _mesh->rank(), positions->array()->aosoa()->size(),
            //     _mesh->layoutObj()->vertices().size());
            
            _mesh->fill_positions(positions, true, 1);
            // printf("After pm->gather: R%d: pos: %d, verts: %d\n",
            //     _mesh->rank(), positions->array()->aosoa()->size(),
            //     _mesh->layoutObj()->vertices().size());
            _pm->gather();
            step();
            t++;
            // 4. Output mesh state periodically
            if ( write_freq && (0 == t % write_freq ))
            {
                if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
                {
                    auto vtk_writer = createVTKWriter(*_pm );
                    vtk_writer->vtkWrite(t);
                }
            }
            // Look at face 15, 20/21

            // Write views for future testing, if enabled
            #if WRITE_VIEWS
            int write_at_time = 5;
            int rank, comm_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

            if (t == write_at_time)
            {
                auto z = _pm->get(Cabana::Grid::Node(), Field::Position())->view();
                auto w = _pm->get(Cabana::Grid::Node(), Field::Vorticity())->view();
                int mesh_size = _pm->mesh().mesh_size();
                int periodic = _params.periodic[0];
                // void writeView(int rank, int comm_size, int mesh_size, const View v)
                BeatnikTest::Utils::writeView(rank, comm_size, mesh_size, periodic, z);
                BeatnikTest::Utils::writeView(rank, comm_size, mesh_size, periodic, w);
            }
            #endif

        } while ( ( _time < t_final ) );
        Kokkos::Profiling::popRegion();
    }

    // XXX - Template these functions on EntityType. Currently
    // causes a linking error when trying to do so.
    // For testing purposes
    View_t get_positions(Cabana::Grid::Node) override
    {
        //_pm->gather();
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
        {
            auto view = _pm->get(Field::Position())->array()->view();
            int dim0 = view.extent(0);
            int dim1 = view.extent(1);
            auto temp = Kokkos::create_mirror_view(view);
            View_t ret = View_t("ret_p", dim0, dim1, 3);
            Kokkos::deep_copy(temp, view);
            Kokkos::deep_copy(ret, temp); 
            return ret;
        }
        else { throw std::invalid_argument("Solver::get_positions: Unstructured mesh not yet supported.\n");} 
    }
    View_t get_vorticities(Cabana::Grid::Node) override
    {
        //_pm->gather();
        if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
        {
            auto view = _pm->get(Field::Vorticity())->array()->view();
            int dim0 = view.extent(0);
            int dim1 = view.extent(1);
            auto temp = Kokkos::create_mirror_view(view);
            View_t ret = View_t("ret_w", dim0, dim1, 2);
            Kokkos::deep_copy(temp, view);
            Kokkos::deep_copy(ret, temp); 
            return ret;
        }
        else { throw std::invalid_argument("Solver::get_positions: Unstructured mesh not yet supported.\n");} 
    }   

  private:
    /* Solver state variables */
    int _halo_min;
    double _atwood;
    double _g;
    double _mu, _eps;
    double _dt;
    double _time;
    
    Params _params;
    BoundaryCondition _bc;
    
    std::unique_ptr<beatnik_mesh_type> _mesh;
    std::unique_ptr<pm_type> _pm;
    std::unique_ptr<br_solver_type> _br;
    std::unique_ptr<zmodel_type> _zm;
    std::unique_ptr<ti_type> _ti;
};

//---------------------------------------------------------------------------//
// Creation method which passes unstructured mesh paramters.
template <class ModelOrderTag, class MeshTypeTag, class Params,
          class PositionsAoSoA, class VerticesAoSoA, class FacesAoSoA>
std::shared_ptr<SolverBase>
createSolver( const std::string& device, MPI_Comm comm,
              const PositionsAoSoA& positions,
              const VerticesAoSoA& vertices,
              const FacesAoSoA& faces,
              const BoundaryCondition& bc,
              const double atwood, const double g, 
              const ModelOrderTag,
              const MeshTypeTag,
              const double mu,
              const double epsilon, 
              const double delta_t,
              const Params params )
{
    if ( 0 == device.compare( "serial" ) )

    {
#if defined( KOKKOS_ENABLE_SERIAL )
        return std::make_shared<
            Beatnik::UnstructuredSolver<Kokkos::Serial, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
            comm, positions, vertices, faces, bc, atwood, g, 
            mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "threads" ) )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        return std::make_shared<
            Beatnik::UnstructuredSolver<Kokkos::Threads, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
            comm, positions, vertices, faces, bc, atwood, g, 
            mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "Threads Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "openmp" ) )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        return std::make_shared<
            Beatnik::UnstructuredSolver<Kokkos::OpenMP, Kokkos::HostSpace, ModelOrderTag, MeshTypeTag>>(
            comm, positions, vertices, faces, bc, atwood, g, 
            mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "cuda" ) )
    {
#if defined(KOKKOS_ENABLE_CUDA)
        return std::make_shared<
            Beatnik::UnstructuredSolver<Kokkos::Cuda, Kokkos::CudaSpace, ModelOrderTag, MeshTypeTag>>(
            comm, positions, vertices, faces, bc, atwood, g, 
            mu, epsilon, delta_t, params);
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare( "hip" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<Beatnik::UnstructuredSolver<Kokkos::HIP, 
            Kokkos::Experimental::HIPSpace, ModelOrderTag, MeshTypeTag>>(
                comm, positions, vertices, faces, bc, atwood, g, 
                mu, epsilon, delta_t, params);
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
