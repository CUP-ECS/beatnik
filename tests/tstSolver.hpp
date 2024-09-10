#ifndef _TSTSOLVER_HPP_
#define _TSTSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "TestRocketrig.hpp"

#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

template <class T>
class SolverTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using solver_type = std::shared_ptr<Beatnik::SolverBase>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
  protected:
    Rocketrig rg_;
    solver_type solver_;
    std::shared_ptr<node_array> z, w;

    void SetUp() override
    {

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
    
    }

  public:
    

    // void init_views()
    // {
    //     auto pm = this->solver_->get_pm();
    //     auto local_grid = pm.mesh().localGrid();
    // }

    

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
