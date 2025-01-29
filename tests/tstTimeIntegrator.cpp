#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstTimeIntegrator.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(TimeIntegratorTest, DeviceTypes);

TYPED_TEST(TimeIntegratorTest, testStepFuncPeriodic)
{
    // int dim0 = 2, dim1 = 3;
    // Kokkos::View<double**[3], Kokkos::HostSpace> input("input", dim0, dim1);
    // for (int i = 0; i < dim0; i++)
    // {
    //     for (int j = 0; j < dim1; j++)
    //     {
    //         for (int d = 0; d < 3; d++)
    //         {
    //             double val = (double)(i+1)*(j+1)*(d+1);
    //             input(i, j, d) = val;
    //         }
    //     }
    // }
    // using ViewType = Kokkos::View<double**[3]>;
    // Utils::writeViewToFile(input, "input.view");
    // auto read_view = Utils::readViewFromFile<ViewType>("input.view", 3);
    // Kokkos::View<double**[3], Kokkos::HostSpace> output("output", dim0, dim1);
    // auto temp = Kokkos::create_mirror_view(read_view);
    // Kokkos::deep_copy(temp, read_view);
    // Kokkos::deep_copy(output, temp);
    
    // for (int i = 0; i < dim0; i++)
    // {
    //     for (int j = 0; j < dim1; j++)
    //     {
    //         for (int d = 0; d < 3; d++)
    //         {
    //             printf("(%d, %d, %d) in: %0.2lf, out: %0.2lf\n", i, j, d, input(i, j, d), output(i, j, d));
    //         }
    //     }
    // }
    // this->read_z("input.view");



    this->read_z("../tests/views/z_orig_n32_r0.view");
    auto z_correct = this->z->view();
    this->ti_->step(this->delta_t_high_order);
    auto z_test = this->p_pm_->get( Cabana::Grid::Node(), Beatnik::Field::Position());
    this->compare_views(z_test, z_correct);
}


} // end namespace BeatnikTest
