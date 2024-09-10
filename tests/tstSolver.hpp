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
class SolverTest : public ::testing::Test
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;

    using solver_type = Beatnik::SolverBase;
    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    // using zmodel_type = Beatnik::ZModel<ExecutionSpace, MemorySpace, Beatnik::ModelOrder, Beatnik::Params>;
    // using ti_type = Beatnik::TimeIntegrator<ExecutionSpace, MemorySpace, zmodel_type>;
    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
  protected:
    std::shared_ptr<Rocketrig> rg_;
    View_t z_test;
    View_t z;
    View_t w_test;
    View_t w;

    int rank_, comm_size_;

    void SetUp() override
    {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
    }

    void TearDown() override
    { 
        this->rg_ = NULL;
    }

    void init(ClArgs &cl)
    {
        this->rg_ = std::make_shared<Rocketrig>(cl);
    }
    
    void read_w(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[2]>;  // Use the correct view type here

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 2);
        int dim0 = read_view.extent(0);
        int dim1 = read_view.extent(1);
        this->w = View_t("w", dim0, dim1, 2);

        // Perform deep copy into the destination view
        // auto view_d = this->w->view();
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(w, temp);
    }

    void read_z(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[3]>;  // Use the correct view type here

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 3);
        int dim0 = read_view.extent(0);
        int dim1 = read_view.extent(1);
        this->z = View_t("z", dim0, dim1, 3);
        // Perform deep copy into the destination view
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(z, temp);
    }

    void read_correct_data(std::string filepath)
    {
        auto cl = this->rg_->get_ClArgs();
        int mesh_size = cl.num_nodes[0];
        int periodic = !(cl.boundary);
        //w_64_p_r1.4.view
        std::string z_path = filepath;
        std::string w_path = filepath;
        std::string z_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'z');
        std::string w_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'w');
        z_path += z_name;
        w_path += w_name;
        this->read_z(z_path);
        this->read_w(w_path);
    }

    template <class View>
    void compare_views(View correctView, View testView)
    {
        for (int d = 0; d < 3; d++)
        {
            if (testView.extent(d) != correctView.extent(d)) 
            {
                printf("View extent(%d) do not match.\n", d);
                return;
            }
        }
        int halo_width = 2;
        int dim0 = testView.extent(0), dim1 = testView.extent(1), dim2 = testView.extent(2);
        for (int i = halo_width; i < (dim0-halo_width); i++)
        {
            for (int j = halo_width; j < (dim1-halo_width); j++)
            {
                for (int d = 0; d < 3; d++)
                {
                    double test = testView(i, j, d);
                    double correct = correctView(i, j, d);
                    ASSERT_DOUBLE_EQ(test, correct) << "At (" << i << ", " << j << ", " << d << ")";
                    // printf("(%d, %d, %d): test: %0.6lf, correct: %0.6lf\n",
                    //     i, j, d, testView(i, j, d), correctView(i, j, d));
                }
            }
        }
    }
};

} // end namespace BeatnikTest

#endif // _TSTSOLVER_HPP_
