#ifndef TSTEXACTBRSOLVER_HPP
#define TSTEXACTBRSOLVER_HPP

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "tstExactBRSolverCorrect.hpp"
#include "tstDriver.hpp"


class TestExactBRSolver : public testing::Test
{
  protected:
    

    /* Init the classes required for the BR solver to work */
    void SetUp() override
    {

        // Create the mesh
        Cabana::Grid::DimBlockPartitioner<2> partitioner;


        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 3, Cabana::Grid::Node() );
        auto node_pair_layout =

            Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 2, Cabana::Grid::Node() );
        _zdot = Cabana::Grid::createArray<double, mem_space>("velocity", 
                                                       node_triple_layout);
        _wdot = Cabana::Grid::createArray<double, mem_space>("vorticity derivative",
                                                       node_pair_layout);
        /* ExactBRSolver( const pm_type & pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy)*/
        
    }


    // Variables for needed testing ExactBRSolver
    const 
}

TEST_F(TestExactBRSolver, testComputeInterfaceVelocityPeriodicBC)
{ 

    ASSERT_EQ(6, 6);
}



#endif // TSTEXACTBRSOLVER_HPP
