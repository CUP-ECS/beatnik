#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(SolverTest, DeviceTypes);

// TYPED_TEST(SolverTest, testPeriodicLowOrderSolver)
// {

//     ClArgs cl;

//     // Init default values
//     init_cl_args(cl);

//     // Adjust command-line args for this test
//     cl.num_nodes = {64, 64};
//     cl.boundary = Beatnik::BoundaryType::PERIODIC;
//     cl.params.solver_order = SolverOrder::ORDER_LOW;

//     // Filepath to view directories
//     std::stringstream ss;
//     ss << "../tests/views/low_order/comm-size-" << this->comm_size_ << "/";
//     std::string filepath = ss.str();

//     // Run test
//     this->run_test(cl, filepath);
// }

TYPED_TEST(SolverTest, testPeriodicCutoffSolver)
{
    if (this->comm_size_ == 1)
    {
        printf("Error: Communicator size is 1 < 4 to support periodic boundary conditions using the cutoff solver. Skipping testPeriodicCutoffSolver test.\n");
        return;
    }

    ClArgs cl;

    // Init default values
    init_cl_args(cl);

    // Adjust command-line args for this test
    cl.num_nodes = {64, 64};
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.params.solver_order = SolverOrder::ORDER_HIGH;
    cl.params.br_solver = Beatnik::BRSolverType::BR_CUTOFF;
    cl.params.cutoff_distance = 0.25;

    // Filepath to view directories
    std::stringstream ss;
    ss << "../tests/views/high_order/cutoff/comm-size-" << this->comm_size_ << "/";
    std::string filepath = ss.str();

    // Run test
    this->run_test(cl, filepath);
}

// TYPED_TEST(SolverTest, testFreeCutoffSolver)
// {
//     ClArgs cl;

//     // Init default values
//     init_cl_args(cl);

//     // Adjust command-line args for this test
//     cl.num_nodes = {64, 64};
//     cl.boundary = Beatnik::BoundaryType::FREE;
//     cl.params.solver_order = SolverOrder::ORDER_HIGH;
//     cl.params.br_solver = Beatnik::BRSolverType::BR_CUTOFF;
//     cl.params.cutoff_distance = 0.25;

//     // Filepath to view directories
//     std::stringstream ss;
//     ss << "../tests/views/high_order/cutoff/comm-size-" << this->comm_size_ << "/";
//     std::string filepath = ss.str();

//     // Run test
//     this->run_test(cl, filepath);
// }

// TYPED_TEST(SolverTest, testPeriodicExactSolver)
// {
//     ClArgs cl;

//     // Init default values
//     init_cl_args(cl);

//     // Adjust command-line args for this test
//     cl.num_nodes = {64, 64};
//     cl.boundary = Beatnik::BoundaryType::PERIODIC;
//     cl.params.solver_order = SolverOrder::ORDER_HIGH;
//     cl.params.br_solver = Beatnik::BRSolverType::BR_EXACT;

//     // Filepath to view directories
//     std::stringstream ss;
//     ss << "../tests/views/high_order/exact/comm-size-" << this->comm_size_ << "/";
//     std::string filepath = ss.str();

//     // Run test
//     this->run_test(cl, filepath);
// }

// TYPED_TEST(SolverTest, testFreeExactSolver)
// {
//     ClArgs cl;

//     // Init default values
//     init_cl_args(cl);

//     // Adjust command-line args for this test
//     cl.num_nodes = {64, 64};
//     cl.boundary = Beatnik::BoundaryType::FREE;
//     cl.params.solver_order = SolverOrder::ORDER_HIGH;
//     cl.params.br_solver = Beatnik::BRSolverType::BR_EXACT;

//     // Filepath to view directories
//     std::stringstream ss;
//     ss << "../tests/views/high_order/exact/comm-size-" << this->comm_size_ << "/";
//     std::string filepath = ss.str();

//     // Run test
//     this->run_test(cl, filepath);
// }


} // end namespace BeatnikTest
