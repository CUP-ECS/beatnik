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

    using mesh_array_type =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
    using node_view = typename mesh_array_type::view_type;

  protected:
    std::shared_ptr<mesh_array_type> omega_correct_;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

  public:
    template <class pm_bc_type>
    void populateCorrectOmega(pm_bc_type pm_)
    {
        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        omega_correct_ = Cabana::Grid::createArray<double, MemorySpace>(
            "omega_correct_", node_triple_layout );
        
        auto z = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        double dx = this->dx_;
        double dy = this->dy_;

        auto omega_view_correct = this->omega_correct_->view();

        auto local_grid = pm_->mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for( "Populate Omega",  
            createExecutionPolicy(own_node_space, ExecutionSpace()), 
            KOKKOS_LAMBDA(int k, int l) {
                for (int d = 0; d < 3; d++) {
                    omega_view_correct(k, l, d) = w(k, l, 1) * Beatnik::Operators::Dx(z, k, l, d, dx) - w(k, l, 0) * Beatnik::Operators::Dy(z, k, l, d, dy);
                }  
        });
    }

    template <class pm_bc_type, class zm_type>
    void testOmega(pm_bc_type pm_, zm_type zm_)
    {
        auto z = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        zm_->prepareOmega(z, w);

        auto omega_d_test = zm_->getOmega();
        auto omega_d_correct = this->omega_correct_->view();

        // Copy to host memory for serial testing
        int dim0 = z.extent(0);
        int dim1 = z.extent(1);
        int dim2 = z.extent(2);
        Kokkos::View<double***, Kokkos::HostSpace> omega_h_test("omega_h_test", dim0, dim1, dim2);
        Kokkos::View<double***, Kokkos::HostSpace> omega_h_correct("omega_h_correct", dim0, dim1, dim2);

        auto hvt_tmp = Kokkos::create_mirror_view(omega_d_test);
        auto hvc_tmp = Kokkos::create_mirror_view(omega_d_correct);
        Kokkos::deep_copy(hvt_tmp, omega_d_test);
        Kokkos::deep_copy(hvc_tmp, omega_d_correct);
        Kokkos::deep_copy(omega_h_test, hvt_tmp);
        Kokkos::deep_copy(omega_h_correct, hvc_tmp);

        auto local_grid = pm_->mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for( "Check Omega",  
            createExecutionPolicy(own_node_space, Kokkos::DefaultHostExecutionSpace()), 
            KOKKOS_LAMBDA(int k, int l) {
                for (int dim = 0; dim < dim2; dim++)
                {
                    double omega_test = omega_h_test(k, l, dim);
                    double omega_correct = omega_h_correct(k, l, dim);
                    EXPECT_DOUBLE_EQ(omega_test, omega_correct);
                }
                
        });
    }
};

} // end namespace BeatnikTest

#endif // _TSTZMODEL_HPP_
