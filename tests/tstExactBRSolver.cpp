#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstExactBRSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(ExactBRSolverTest, DeviceTypes);

TYPED_TEST(ExactBRSolverTest, testComputeInterfaceVelocityPeriodic)
{
    this->Init(this->p_pm_);
    this->calculateSingleCorrectZdot();
    this->calculateDistributedZdot(this->p_pm_, this->p_zm_exact_);
}

} // end namespace BeatnikTest
