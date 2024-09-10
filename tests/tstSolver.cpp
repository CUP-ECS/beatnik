#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(SolverTest, DeviceTypes);

TYPED_TEST(SolverTest, testPeriodicLowOrderSolver)
{

    ClArgs cl;

    // Init default values
    init_cl_args(cl);

    // Adjust command-line args for this test
    cl.num_nodes = {64, 64};
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.params.solver_order = SolverOrder::ORDER_LOW;
    this->init(cl);

    std::stringstream ss;
    ss << "../tests/views/low_order/comm-size-" << this->comm_size_ << "/";
    std::string filepath = ss.str();
    this->read_correct_data(filepath);
    this->rg_->rocketrig();
    auto z_test = this->rg_->get_positions();
    auto w_test = this->rg_->get_vorticities();
    this->compare_views(this->z, z_test);
    this->compare_views(this->w, w_test);

    

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
