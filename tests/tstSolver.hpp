#ifndef _TSTSOLVER_HPP_
#define _TSTSOLVER_HPP_

#include <iostream>
#include <filesystem>
#include <regex>

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "TestRocketrig.hpp"
#include "TestingUtils.hpp"

#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

template <class T>
class SolverTest : public ::testing::Test
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;

  protected:
    std::shared_ptr<Rocketrig> rg_;
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
    
    View_t read_w(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[2]>;

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 2);
        int dim0 = read_view.extent(0);
        int dim1 = read_view.extent(1);
        View_t w = View_t("w", dim0, dim1, 2);
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(w, temp);
        return w;
    }

    View_t read_z(const std::string& filename)
    {
        using ViewType = Kokkos::View<double**[3]>;

        // Call the function with the explicit template type
        auto read_view = Utils::readViewFromFile<ViewType>(filename, 3);
        int dim0 = read_view.extent(0);
        int dim1 = read_view.extent(1);
        View_t z = View_t("z", dim0, dim1, 3);
        auto temp = Kokkos::create_mirror_view(read_view);
        Kokkos::deep_copy(temp, read_view);
        Kokkos::deep_copy(z, temp);
        return z;
    }

    template <class View>
    void compare_views(View correctView, View testView)
    {
        for (int d = 0; d < 3; d++)
        {
            if (testView.extent(d) != correctView.extent(d)) 
            {
                printf("View extents in dimension %d do not match. Was the file read correctly? Skipping test.\n", d);
                return;
            }
        }
        int halo_width = 0;
        int dim0 = testView.extent(0), dim1 = testView.extent(1), dim2 = testView.extent(2);
        for (int i = halo_width; i < (dim0-halo_width); i++)
        {
            for (int j = halo_width; j < (dim1-halo_width); j++)
            {
                for (int d = 0; d < dim2; d++)
                {
                    double test = testView(i, j, d);
                    double correct = correctView(i, j, d);
                    ASSERT_NEAR(test, correct, 0.0000000000001) << "Rank " << this->rank_ << ": (" << i << ", " << j << ", " << d << ")";
                }
            }
        }
    }

    void run_test(ClArgs &cl, std::string filepath)
    {
        this->init(cl);
        int mesh_size = cl.num_nodes[0];
        int periodic = !(cl.boundary);
        std::string z_path = filepath;
        std::string w_path = filepath;
        std::string z_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'z');
        std::string w_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'w');
        z_path += z_name;
        w_path += w_name;
        auto z = this->read_z(z_path);
        auto w = this->read_w(w_path);
        this->rg_->rocketrig();
        auto z_test = this->rg_->get_positions<Cabana::Grid::Node>();
        auto w_test = this->rg_->get_vorticities<Cabana::Grid::Node>();
        this->compare_views(z, z_test);
        this->compare_views(w, w_test);
    }

    void remove_view_files()
    {
        std::string path = "./";
        std::stringstream ss;
        
        // Create regex patterns
        std::regex pattern_w("^w_\\d+_[pf]_r\\d+\\.\\d+\\.view$");
        std::regex pattern_z("^z_\\d+_[pf]_r\\d+\\.\\d+\\.view$");

        // Iterate through the directory
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            // Skip directories
            if (std::filesystem::is_directory(entry.status())) {
                continue;
            }

            // Get the file name as a string
            std::string file_name = entry.path().filename().string();
            // Check if the file name matches the pattern
            if (std::regex_match(file_name, pattern_w) || std::regex_match(file_name, pattern_z)) {
                // Delete the file
                try {
                    std::filesystem::remove(entry.path());
                    // std::cout << "Deleted: " << file_name << std::endl;
                } catch (const std::filesystem::filesystem_error& e) {
                    std::cerr << "Error deleting file: " << e.what() << std::endl;
                }
            }
        }
    }
};

} // end namespace BeatnikTest

#endif // _TSTSOLVER_HPP_
