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
    using zm_type_h = Beatnik::ZModel<ExecutionSpace, MemorySpace, Beatnik::Order::High, Beatnik::Params>;
    using ti_type = Beatnik::TimeIntegrator<ExecutionSpace, MemorySpace, zm_type_h>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
  protected:
    MPI_Comm comm_;
    int rank_, comm_size_;
    int mesh_size = this->meshSize_;
    std::shared_ptr<ti_type> ti_;
    std::shared_ptr<node_array> z, w;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
        this->comm_ = this->p_pm_->mesh().localGrid()->globalGrid().comm();
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &comm_size_);
        this->ti_ = std::make_shared<ti_type>( *this->p_pm_, this->p_bc_, *this->p_zm_exact_ );

        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( this->p_pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        auto node_pair_layout =
            Cabana::Grid::createArrayLayout( this->p_pm_->mesh().localGrid(), 2, Cabana::Grid::Node() );

        z = Cabana::Grid::createArray<double, Kokkos::HostSpace>(
            "z_view", node_triple_layout );
	    Cabana::Grid::ArrayOp::assign( *z, 0.0, Cabana::Grid::Ghost() );

        // 2. The magnitude of vorticity at the interface 
        w = Cabana::Grid::createArray<double, Kokkos::HostSpace>(
            "w_view", node_pair_layout );
	    Cabana::Grid::ArrayOp::assign( *w, 0.0, Cabana::Grid::Ghost() );
    }

    void TearDown() override
    { 
        this->ti_ = NULL;
        TestingBase<T>::TearDown();
    }

  public:
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
