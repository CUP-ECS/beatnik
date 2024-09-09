#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(SolverTest, DeviceTypes);

TYPED_TEST(SolverTest, testStepFuncPeriodic)
{
    int rank = this->rank_;
    int comm_size = this->comm_size_;
    int mesh_size = this->meshSize_;
    std::string filepath = "../tests/views/high_order/cutoff/comm-size-4/";
    //w_64_p_r1.4.view
    // std::string get_filename(int rank, int comm_size, int mesh_size, int periodic, char x)
    std::string w_name = get_filename(rank, comm_size, mesh_size, 1, 'w');
    filepath += w_name;
    auto z_correct = read_w(filepath);
    auto z_test = this->p_pm_->get( Cabana::Grid::Node(), Beatnik::Field::Position());
    this->compare_views(z_test, z_correct);
}


} // end namespace BeatnikTest
