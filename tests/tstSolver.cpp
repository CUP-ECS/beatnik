#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(SolverTest, DeviceTypes);

TYPED_TEST(SolverTest, testPeriodicExactSolver)
{

    Utils::ClArgs clargs;
    Utils::init_cl_args(clargs);
    this->rocketrig(clargs);

    // double delta_t = this->delta_t_high_order;
    // this->init_solver_high<Beatnik::Order::High()>(this->partitioner_, this->p_MeshInitFunc_, this->p_bc_, this->p_params_, delta_t);
    // this->init_views();
    // int rank = this->rank_;
    // int comm_size = this->comm_size_;
    // int mesh_size = this->meshSize_;
    // std::string filepath = "../tests/views/high_order/cutoff/comm-size-4/";
    //w_64_p_r1.4.view
    // std::string get_filename(int rank, int comm_size, int mesh_size, int periodic, char x)
    // std::string z_name = Utils::get_filename(rank, comm_size, mesh_size, 1, 'z');
    // filepath += z_name;
    // this->read_z(filepath);
    // auto z_correct = this->z->view();
    // auto z_test = this->solver->get_pm()->get( Cabana::Grid::Node(), Beatnik::Field::Position());
    // this->compare_views(z_test, z_correct);
}


} // end namespace BeatnikTest
