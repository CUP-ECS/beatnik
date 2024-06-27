#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstBoundaryCondition.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(BoundaryConditionTest, DeviceTypes);

TYPED_TEST(BoundaryConditionTest, testFreeBoundary)
{ 
    //using memory_sapce = typename TestFixture::MemorySpace;
    
    //auto local_grid = MeshTest<TestFixture::T>::testMeshNonperiodic_->localGrid();

    // const std::array<double, 2> global_low_corner = {-1, -1}, global_high_corner = {1, 1};
    // const std::array<int, 2> num_nodes = {6, 6};
    // Cabana::Grid::DimBlockPartitioner<2> partitioner;
    // Beatnik::BoundaryCondition bc;
    // const std::array<bool, 2> periodic = {false, false};

    // auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
    //     global_low_corner, global_high_corner, 1.0 );
    // auto global_grid = Cabana::Grid::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
    //                                                  periodic, partitioner );
        
    // int halo_width = 2;
    // auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
    // auto local_mesh = Cabana::Grid::createLocalMesh<MemorySpace>( local_grid );

    auto node_layout =
            Cabana::Grid::createArrayLayout( this->local_grid_, 1, Cabana::Grid::Node() );


    auto position = Cabana::Grid::createArray<double, Kokkos::HostSpace>(
            "position", node_layout );

	// Cabana::Grid::ArrayOp::assign( *position, 0.0, Cabana::Grid::Ghost() );

    auto z = position->view();

    // auto own_nodes = this->local_grid_->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
    //                                             Cabana::Grid::Local() );
    
    double dx = 0.3, dy = 0.4;
    int min0 = 2, min1 = 8, max0 = 2, max1 = 8;
    for (int i = min0; i < min1; i++)
    {
        for (int j = max0; j < max1; j++)
        {
            z(i, j, 0) = -1+dx*i + -1+dy*j;
        }
    }
    for (int i = min0; i < min1; i++)
    {
        for (int j = max0; j < max1; j++)
        {
            printf("%d, %d, %d -> %0.8lf\n", i, j, 0, z(i, j, 0));
        }
    }

    // this->bc_.applyField(this->testMeshNonPeriodic_, position, 1);


    
        // Kokkos::parallel_for(
        //     "Initialize Cells",
        //     Cabana::Grid::createExecutionPolicy( own_nodes ),
        //     KOKKOS_LAMBDA( const int i, const int j ) {

        //         double dx = 0.3, dy = 0.4;

        //         // z(i, j, 0) = -1+dx*i + -1+dy*j;
        //     });


    
    EXPECT_DOUBLE_EQ(1.0, 1.000000000001);
    ASSERT_EQ(6, 7);
}

} // end namespace BeatnikTest
