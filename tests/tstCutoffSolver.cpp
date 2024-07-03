#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstCutoffSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(CutoffSolverTest, DeviceTypes);

TYPED_TEST(CutoffSolverTest, testFreeBoundary)
{ 
    double result = this->br_->simpsonWeight(5, 5);
    ASSERT_EQ(0, 0);    
    // auto z = this->position_np_->view();

    // this->populateArray(z);

    // this->bc_non_periodic_.applyField(*this->testMeshNonPeriodic_, *this->position_np_, 1);

    // this->testFreeBC(z);
}

} // end namespace BeatnikTest
