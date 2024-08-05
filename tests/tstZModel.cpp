#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstZModel.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(ZModelTest, DeviceTypes);

TYPED_TEST(ZModelTest, testOmegaPeriodic)
{
    this->populateCorrectOmega(this->p_pm_);
    this->testOmega(this->p_pm_, this->p_zm_exact_);
}

TYPED_TEST(ZModelTest, testOmegaFree)
{
    this->populateCorrectOmega(this->f_pm_);
    this->testOmega(this->f_pm_, this->f_zm_exact_);
}


} // end namespace BeatnikTest
