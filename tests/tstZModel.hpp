#ifndef _TSTZMODEL_HPP_
#define _TSTZMODEL_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

template <class T>
class ZModelTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

  protected:
    MPI_Comm comm_;
    int rank_, comm_size_;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
        comm_ = this->p_pm_->mesh().localGrid()->globalGrid().comm();
        MPI_Comm_size(comm_, &comm_size_);
        MPI_Comm_rank(comm_, &rank_);


    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

  public:
    void testOmegaPeriodic()
    {
        auto z = this->p_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = this->p_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        this->p_zm_exact_->prepareOmega(z, w);

        auto omega = this->p_zm_exact_->getOmega();

        // Copy to host memory for serial testing
        int dim0 = z.extent(0);
        int dim1 = z.extent(1);
        int dim2 = z.extent(2);
        Kokkos::View<double***, Kokkos::HostSpace> omega_host("omega_host", dim0, dim1, dim2);
        auto o_host_tmp = Kokkos::create_mirror_view(omega);
        Kokkos::deep_copy(o_host_tmp, omega);
        Kokkos::deep_copy(omega_host, o_host_tmp);

        auto local_grid = this->p_pm_->mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for( "Check Omega",  
            createExecutionPolicy(own_node_space, Kokkos::DefaultHostExecutionSpace()), 
            KOKKOS_LAMBDA(int i, int j) {
            for (int d = 0; d < 3; d++) {
                //omega(i, j, d) = w(i, j, 1) * Operators::Dx(z, i, j, d, dx) - w(i, j, 0) * Operators::Dy(z, i, j, d, dy);
                //printf("omega0: %.15lf\n", w(k, l, 1) * Operators::Dx(z, k, l, d, dx) - w(k, l, 0) * Operators::Dy(z, k, l, d, dy));
                printf("hi\n");
            }
        });

    }

};

} // end namespace BeatnikTest

#endif // _TSTZMODEL_HPP_
