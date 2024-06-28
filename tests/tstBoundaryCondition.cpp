#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstBoundaryCondition.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(BoundaryConditionTest, DeviceTypes);

TYPED_TEST(BoundaryConditionTest, testFreeBoundary)
{ 
    auto z = this->position_np_->view();

    this->populateArray(z);

    this->bc_non_periodic_.applyField(*this->testMeshNonPeriodic_, *this->position_np_, 1);

    this->testFreeBC(z);
}

} // end namespace BeatnikTest
