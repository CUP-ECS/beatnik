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
    this->initializeVorticity(this->p_pm_);
    this->calculateSingleCorrectZdot();
    this->calculateDistributedZdot(this->p_pm_, this->p_zm_exact_);
    this->testZdot(this->p_pm_);
}

TYPED_TEST(ExactBRSolverTest, testComputeInterfaceVelocityFree)
{
    this->Init(this->f_pm_);
    this->initializeVorticity(this->f_pm_);
    this->calculateSingleCorrectZdot();
    this->calculateDistributedZdot(this->f_pm_, this->f_zm_exact_);
    this->testZdot(this->f_pm_);
}

} // end namespace BeatnikTest
