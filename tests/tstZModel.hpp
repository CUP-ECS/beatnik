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

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
    using node_view = typename node_array::view_type;

  protected:
    MPI_Comm comm_;
    int rank_, comm_size_;

    std::shared_ptr<node_array> omega_correct_;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
        comm_ = this->p_pm_->mesh().localGrid()->globalGrid().comm();
        MPI_Comm_size(comm_, &comm_size_);
        MPI_Comm_rank(comm_, &rank_);
        this->populateCorrectOmega();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

  public:
    void populateCorrectOmega()
    {
        auto z = this->p_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = this->p_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());
        double dx = this->dx_;
        double dy = this->dy_;

        auto omega_view_correct = this->omega_correct_->view();

        auto local_grid = this->p_pm_->mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for( "Check Omega",  
            createExecutionPolicy(own_node_space, Kokkos::DefaultHostExecutionSpace()), 
            KOKKOS_LAMBDA(int k, int l) {
                double omega[3];
                for (int d = 0; d < 3; d++) {
                    omega_view_correct(k, l, d) = w(k, l, 1) * Beatnik::Operators::Dx(z, k, l, d, dx) - w(k, l, 0) * Beatnik::Operators::Dy(z, k, l, d, dy);
                }  
        });

    }

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
            KOKKOS_LAMBDA(int k, int l) {
                double omega[3];
                for (int d = 0; d < 3; d++) {
                    omega[d] = w(k, l, 1) * Dx(z, k, l, d, dx) - w(k, l, 0) * Dy(z, k, l, d, dy);
                }  
                double omega_correct = 
                //omega(i, j, d) = w(i, j, 1) * Operators::Dx(z, i, j, d, dx) - w(i, j, 0) * Operators::Dy(z, i, j, d, dy);
                //printf("omega0: %.15lf\n", w(k, l, 1) * Operators::Dx(z, k, l, d, dx) - w(k, l, 0) * Operators::Dy(z, k, l, d, dy));
                printf("hi\n");
        });

    }

};

} // end namespace BeatnikTest

#endif // _TSTZMODEL_HPP_
