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
        /* ExactBRSolver( const pm_type & pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy)*/
        
    }


    // Variables for needed testing ExactBRSolver
    const ProblemManager & _pm;
    const BoundaryCondition & _bc;
    double _epsilon, _dx, _dy;
    MPI_Comm _comm;
}

TEST_F(TestExactBRSolver, sampleTest)
{
  ASSERT_EQ(6, 6);
}



#endif // TSTEXACTBRSOLVER_HPP
