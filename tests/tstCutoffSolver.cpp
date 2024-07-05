#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstCutoffSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(CutoffSolverTest, DeviceTypes);

TYPED_TEST(CutoffSolverTest, testIsOnBoundary)
{
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
    int local_location[3] = {boundary_topology(rank, 1), boundary_topology(rank, 2), boundary_topology(rank, 3)};
    int max_location[3] = {boundary_topology(comm_size, 1), boundary_topology(comm_size, 2), boundary_topology(comm_size, 3)};
    int result = this->p_br_cutoff_->isOnBoundary(local_location, max_location);
    int correct_result = this->isOnBoundaryCorrect(local_location, max_location);
    EXPECT_EQ(result, correct_result) << "Rank " << rank << " incorrect.";
}

TYPED_TEST(CutoffSolverTest, testAreNeighbors)
{ 
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
    int local_location[3] = {boundary_topology(rank, 1), boundary_topology(rank, 2), boundary_topology(rank, 3)};
    int max_location[3] = {boundary_topology(comm_size, 1), boundary_topology(comm_size, 2), boundary_topology(comm_size, 3)};
    //printf("R%d, %d, %d, %d, is_n: %d\n", rank, local_location[0], local_location[1], local_location[2], correct);

    // Each rank will test if all any ranks share a periodic boundary in 3D x/y space
    // Defined as one of its 26 neighboring ranks in 3D space.
    int is_neighbor[26];
    if (rank == 0)
    {
        this->getNeighbors(rank, boundary_topology, is_neighbor);
    // for (int other_rank = 0; other_rank < comm_size; other_rank++)
    // {
    //     int remote_location[3] = {boundary_topology(other_rank, 1), boundary_topology(other_rank, 2), boundary_topology(other_rank, 3)};
    //     int correct_result = this->getNeighbors();
    // }
    }
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
}

} // end namespace BeatnikTest
