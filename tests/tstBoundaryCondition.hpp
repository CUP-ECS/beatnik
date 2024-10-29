#ifndef _TSTBOUNDARYCONDITION_HPP_
#define _TSTBOUNDARYCONDITION_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryCondition.hpp>

#include <mpi.h>

#include "TestingBase.hpp"
#include "tstDriver.hpp"

namespace BeatnikTest
{

// XXX - Fix memory space issues in this test so it works on national labs machines
template <class T>
class BoundaryConditionTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using node_array_layout = Cabana::Grid::ArrayLayout<Cabana::Grid::Node, mesh_type>;

    using node_array = Cabana::Grid::Array<double, Cabana::Grid::Node, mesh_type, MemorySpace>;

  protected:
    std::shared_ptr<node_array_layout> f_node_layout_; 
    std::shared_ptr<node_array> f_position_;

    
    void SetUp() override
    {
        TestingBase<T>::SetUp();
        this->f_node_layout_ = Cabana::Grid::createArrayLayout(this->f_mesh_->localGrid(), 1, Cabana::Grid::Node());
        this->f_position_ = Cabana::Grid::createArray<double, MemorySpace>("position", f_node_layout_);
    }

    void TearDown() override
    { 
        this->f_position_ = NULL;
        this->f_node_layout_ = NULL;
        TestingBase<T>::TearDown();
    }

  public:
    void populateArray()
    {
        auto local_grid = this->f_pm_->mesh().localGrid();
        auto own_nodes = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                 Cabana::Grid::Local());

        auto policy = Cabana::Grid::createExecutionPolicy(own_nodes, ExecutionSpace());
        auto z = this->f_position_->view();
        double dx = 0.3, dy = 0.4;
        Kokkos::parallel_for("Initialize Cells", policy,
            KOKKOS_LAMBDA( const int i, const int j ) {
                z(i, j, 0) = -1+dx*i + -1+dy*j;
            });
    }

    void testFreeBC()
    {
        MPI_Comm comm_ = this->f_pm_->mesh().localGrid()->globalGrid().comm();
        int rank, comm_size;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &comm_size);

        auto z = this->f_position_->view();
        int dim0 = z.extent(0);
        int dim1 = z.extent(1);

        // Copy to host memory for serial testing
        Kokkos::View<double***, Kokkos::HostSpace> z_host("z_host", dim0, dim1, 1);
        auto z_host_tmp = Kokkos::create_mirror_view(z);
        Kokkos::deep_copy(z_host_tmp, z);
        Kokkos::deep_copy(z_host, z_host_tmp);
        
        double dx = 0.3, dy = 0.4;

        ASSERT_EQ(1, comm_size) << "Only testing with one process is supported at this time.\n";

        if (comm_size == 1)
        {
            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    double correct_value = -1+dx*i + -1+dy*j;
                    /* Using EXPECT_NEAR because of floating-point imprecision
                     * between doubles inside and out of a Kokkos view.
                     */
                    EXPECT_NEAR(correct_value, z_host(i, j, 0), 0.000000000001);
                }
            }
        }
    }
};

} // end namespace BeatnikTest

#endif // _TSTBOUNDARYCONDITION_HPP_
