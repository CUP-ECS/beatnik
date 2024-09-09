#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(SolverTest, DeviceTypes);

TYPED_TEST(SolverTest, testStepFuncPeriodic)
{
    this->read_z("../tests/views/z_orig_n32_r0.view");
    auto z_correct = this->z->view();
    this->ti_->step(this->delta_t_high_order);
    auto z_test = this->p_pm_->get( Cabana::Grid::Node(), Beatnik::Field::Position());
    this->compare_views(z_test, z_correct);
}


} // end namespace BeatnikTest
