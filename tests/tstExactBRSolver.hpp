#ifndef TSTEXACTBRSOLVER_HPP
#define TSTEXACTBRSOLVER_HPP

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <examples/rocketrig>

#include <mpi.h>

#include "tstExactBRSolverCorrect.hpp"
#include "tstDriver.hpp"

class TestExactBRSolver : public testing::Test
{
  protected:
    
    void SetUp() override
    {
      MPI_Comm_rank( MPI_COMM_WORLD, &_rank );      
      MPI_Comm_size( MPI_COMM_WORLD, &_comm_size );
    }

    int _rank, _comm_size;
}

TEST_F(TestExactBRSolver, testComputeInterfaceVelocityPeriodicBC)
{ 
    ClArgs cl;

    // Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.order = SolverOrder::ORDER_LOW;
    cl.weak_scale = 1;
    cl.write_freq = 10;

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 128, 128 };
    cl.initial_condition = IC_COS;
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.variation = 0.00;
    cl.period = 1.0;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the solver to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = 10.0;

    /* Physical setup of problem */
    cl.global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes[i] *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.order == SolverOrder::ORDER_HIGH) {
            cl.delta_t = tau/50.0;  // Should this depend on dx and dy? XXX
        } else {
            cl.delta_t = tau/25.0;
        }
    }

    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    Beatnik::BoundaryCondition bc;
    for (int i = 0; i < 6; i++)
        bc.bounding_box[i] = cl.global_bounding_box[i];
    bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

    MeshInitFunc initializer( cl.global_bounding_box, cl.initial_condition,
                              cl.tilt, cl.magnitude, cl.variation, cl.period,
                              cl.num_nodes, cl.boundary );

    std::shared_ptr<Beatnik::SolverBase> solver;

    solver = Beatnik::createSolver(
            "serial", MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t );

    solver->solve( cl.t_final, cl.write_freq );
    printf("in test");
    EXPECT_DOUBLE_EQ(1.0, 1.000000000001);
    ASSERT_EQ(6, 7);
}



#endif // TSTEXACTBRSOLVER_HPP
