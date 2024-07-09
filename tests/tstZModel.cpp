#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstZModel.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(ZModelTest, DeviceTypes);

TYPED_TEST(ZModelTest, testOmega)
{
    this->testOmegaPeriodic();
}

} // end namespace BeatnikTest
