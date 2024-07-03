#ifndef _TSTBOUNDARYCONDITION_HPP_
#define _TSTBOUNDARYCONDITION_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <BoundaryCondition.hpp>

#include <mpi.h>

#include "tstMesh.hpp"
#include "tstDriver.hpp"

namespace BeatnikTest
{

template <class T>
class BoundaryConditionTest : public MeshTest<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using node_array_layout = std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, mesh_type>>;

    using node_array = std::shared_ptr<Cabana::Grid::Array<double, Cabana::Grid::Node, mesh_type, MemorySpace>>;

  protected:
    Beatnik::BoundaryCondition bc_periodic_;
    Beatnik::BoundaryCondition bc_non_periodic_;
    node_array_layout node_layout_np_; 
    node_array position_np_;
    
    void SetUp() override
    {
        MeshTest<T>::SetUp();
        for (int i = 0; i < 6; i++)
        {
            bc_periodic_.bounding_box[i] = MeshTest<T>::globalBoundingBox_[i];
            bc_non_periodic_.bounding_box[i] = MeshTest<T>::globalBoundingBox_[i];
        }
        for (int i = 0; i < 4; i++)
        {
            bc_periodic_.boundary_type[i] = Beatnik::BoundaryType::PERIODIC;
            bc_non_periodic_.boundary_type[i] = Beatnik::BoundaryType::FREE;
        }
        node_layout_np_ = Cabana::Grid::createArrayLayout(this->testMeshNonPeriodic_->localGrid(), 1, Cabana::Grid::Node());
        position_np_ = Cabana::Grid::createArray<double, MemorySpace>("position", node_layout_np_);
    }

    void TearDown() override
    {
        MeshTest<T>::TearDown();
    }

  public:
    template <class View>
    void populateArray(View z)
    {
        /*****************
         *  Getting execution range from MeshTest because the compiler does like the following:
        auto own_nodes = local_grid_->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                 Cabana::Grid::Local());

        auto policy = Cabana::Grid::createExecutionPolicy( own_nodes );
        **************/
        int range = MeshTest<T>::boxNodes_;
        int min = MeshTest<T>::haloWidth_;
        using range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>;
        range_policy policy({min, min}, {min+range, min+range});
        double dx = 0.3, dy = 0.4;
        Kokkos::parallel_for("Initialize Cells", policy,
            KOKKOS_LAMBDA( const int i, const int j ) {
                z(i, j, 0) = -1+dx*i + -1+dy*j;
            });
    }

    template <class View>
    void testFreeBC(View z)
    {
        int range = MeshTest<T>::boxNodes_;
        int min = MeshTest<T>::haloWidth_;
        using range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>;
        range_policy policy({0, 0}, {min+range+min, min+range+min});
        double dx = 0.3, dy = 0.4;
        Kokkos::parallel_for("Initialize Cells", policy,
            KOKKOS_LAMBDA( const int i, const int j ) {
                double correct_value = -1+dx*i + -1+dy*j;
                /* Using EXPECT_NEAR because of floating-point imprecision
                 * between doubles inside and out of a Kokkos view.
                 * XXX - Fix calling the __host__ function from a __host__ __device__ function
                 * warning caused by using EXPECT_NEAR here.
                 */
                EXPECT_NEAR(correct_value, z(i, j, 0), 0.000000000001);
            });
    }
};

} // end namespace BeatnikTest

#endif // _TSTBOUNDARYCONDITION_HPP_
