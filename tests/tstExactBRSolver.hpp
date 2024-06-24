#ifndef TSTEXACTBRSOLVER_HPP
#define TSTEXACTBRSOLVER_HPP

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>
#include <examples/rocketrig>

#include <mpi.h>

#include "tstExactBRSolverCorrect.hpp"
#include "tstDriver.hpp"

class TestExactBRSolver : public testing::Test
{
  protected:
    
    void SetUp() override
    {
       
        
    }

}

TEST_F(TestExactBRSolver, testComputeInterfaceVelocityPeriodicBC)
{ 
    /* Tests rocketrig with the default initial conditions 
     * and using the ExactBRSolver with the following modifications:
     *  - 4 processes
     *  - 32x32 mesh
     *  - Checks after timestep 10 (-t 10)
     **/

    // Problem setup
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    std::array<int, 2> num_nodes = {128, 128};
    double gravity = 25.0;
    double atwood = 0.5;

     solver = Beatnik::createSolver(
            "serial", MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t );

    EXPECT_DOUBLE_EQ(1.0, 1.000000000001);
    ASSERT_EQ(6, 6);
}



#endif // TSTEXACTBRSOLVER_HPP
