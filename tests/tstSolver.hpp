#ifndef _TSTSOLVER_HPP_
#define _TSTSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

template <class T>
class SolverTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    using solver_high_type = Beatnik::Solver<ExecutionSpace, MemorySpace, Beatnik::Order::High()>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
  protected:
    MPI_Comm comm_;
    int rank_, comm_size_;
    int mesh_size = this->meshSize_;
    std::shared_ptr<solver_high_type> solver_high;
    std::shared_ptr<node_array> z, w;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);

        // auto node_triple_layout =
        //     Cabana::Grid::createArrayLayout( this->p_pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        // auto node_pair_layout =
        //     Cabana::Grid::createArrayLayout( this->p_pm_->mesh().localGrid(), 2, Cabana::Grid::Node() );

        // z = Cabana::Grid::createArray<double, Kokkos::HostSpace>(
        //     "z_view", node_triple_layout );
        // Cabana::Grid::ArrayOp::assign( *z, 0.0, Cabana::Grid::Ghost() );

        // // 2. The magnitude of vorticity at the interface 
        // w = Cabana::Grid::createArray<double, Kokkos::HostSpace>(
        //     "w_view", node_pair_layout );
        // Cabana::Grid::ArrayOp::assign( *w, 0.0, Cabana::Grid::Ghost() );

    }

    void TearDown() override
    { 
        this->solver_high = NULL;
        TestingBase<T>::TearDown();
    }

  public:
    template <class ModelOrder, class Partitioner, class CreateFunctor, class BoundaryCondition, class Params>
    void init_solver_high(Partitioner partitioner, CreateFunctor create_functor, BoundaryCondition bc, Params params, double delta_t)
    {
        /*
         Solver( MPI_Comm comm,
            const std::array<int, 2>& num_nodes,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            const double atwood, const double g, const InitFunc& create_functor,
            const BoundaryCondition& bc, const double mu, 
            const double epsilon, const double delta_t,
            const Params params)*/
        params.cutoff_distance = 0.25;
        this->solver_high = std::make_shared<Beatnik::Solver<ExecutionSpace, MemorySpace, ModelOrder>>(
            MPI_COMM_WORLD, this->globalNumNodes_, partitioner, this->A_, this->g_, 
            create_functor, bc, this->mu_, this->epsilon_, delta_t, params);

    }

    void init_views()
    {
        auto pm = this->solver_high->get_pm();
        auto local_grid = pm.mesh().localGrid();
    }

    void rocketrig( Utils::ClArgs& cl )
    {
        int comm_size, rank;                         // Initialize Variables
        MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

        Cabana::Grid::DimBlockPartitioner<2> partitioner; // Create Cabana::Grid Partitioner
        Utils::BoundaryCondition bc;
        for (int i = 0; i < 6; i++)
        {
            bc.bounding_box[i] = cl.params.global_bounding_box[i];
            
        }
        bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

        Utils::MeshInitFunc initializer( cl.params.global_bounding_box, cl.initial_condition,
                                cl.tilt, cl.magnitude, cl.variation, cl.params.period,
                                cl.num_nodes, cl.boundary );

        std::shared_ptr<Beatnik::SolverBase> solver;
        if (cl.params.solver_order == Utils::SolverOrder::ORDER_LOW) {
            solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Utils::Order::Low(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else if (cl.params.solver_order == Utils::SolverOrder::ORDER_MEDIUM) {
            solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Utils::Order::Medium(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else if (cl.params.solver_order == Utils::SolverOrder::ORDER_HIGH) {
            solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Utils::Order::High(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else {
            std::cerr << "Invalid Model Order parameter!\n";
            Kokkos::finalize(); 
            MPI_Finalize(); 
            exit( -1 );  

        }

        // Solve
        solver->solve( cl.t_final, cl.write_freq );
    }

    void read_w(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[2]>;  // Use the correct view type here

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 2);

        // Perform deep copy into the destination view
        auto view_d = this->w->view();
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(view_d, temp);
    }

    void read_z(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[3]>;  // Use the correct view type here

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 3);

        // Perform deep copy into the destination view
        auto view_d = this->z->view();
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(view_d, temp);
    }

    template <class View>
    void compare_views(View testView, View correctView)
    {
        for (int d = 0; d < 3; d++)
        {
            if (testView.extent(d) != correctView.extent(d)) 
            {
                printf("View extent(%d) do not match.\n", d);
                return;
            }
        }
        int dim0 = testView.extent(0), dim1 = testView.extent(1), dim2 = testView.extent(2);
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int d = 0; d < 3; d++)
                {
                    printf("(%d, %d, %d): test: %0.6lf, correct: %0.6lf\n",
                        i, j, d, testView(i, j, d), correctView(i, j, d));
                }
            }
        }
    }
};

} // end namespace BeatnikTest

#endif // _TSTSOLVER_HPP_
