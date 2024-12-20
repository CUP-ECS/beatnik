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
    this->populateArray();

    this->f_bc_.applyField(*this->f_mesh_, *this->f_position_, 1);

    this->testFreeBC();
}

} // end namespace BeatnikTest
