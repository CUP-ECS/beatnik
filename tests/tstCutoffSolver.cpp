#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstCutoffSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(CutoffSolverTest, DeviceTypes);

TYPED_TEST(CutoffSolverTest, testNeighboringRanks)
{ 
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
    // int local_location[3] = {boundary_topology(rank, 1), boundary_topology(rank, 2), boundary_topology(rank, 3)};
    // int max_location[3] = {boundary_topology(comm_size, 1), boundary_topology(comm_size, 2), boundary_topology(comm_size, 3)};
    // int remote_location[3] = {-1, -1, -1};
    // int correct = this->areNeighbors(remote_location, local_location, max_location);
    //printf("R%d, %d, %d, %d, is_n: %d\n", rank, local_location[0], local_location[1], local_location[2], correct);

    // if (rank == 0)
    // {
    //     printf("ll: %d, %d, %d\n\n", local_location[0], local_location[1], local_location[2]);
    //     for (int i = 0; i < comm_size; i++)
    //     {
    //         int remote_location[3] = {boundary_topology(i, 1), boundary_topology(i, 2), boundary_topology(i, 3)};
    //         int result = this->br_->isValidRank(remote_location, local_location, max_location);
    //         int correct = this->areNeighbors(remote_location, local_location, max_location);
    //         printf("Neighbor: R%d, %d, %d, %d, is_n: %d\n", i, remote_location[0], remote_location[1], remote_location[2], correct);
    //         //ASSERT_EQ(result, correct); 
    //     }
    // }
    
       
    // auto z = this->position_np_->view();

    // this->populateArray(z);

    // this->bc_non_periodic_.applyField(*this->testMeshNonPeriodic_, *this->position_np_, 1);

    // this->testFreeBC(z);
}

} // end namespace BeatnikTest
