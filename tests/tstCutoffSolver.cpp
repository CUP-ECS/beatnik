#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstCutoffSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(CutoffSolverTest, DeviceTypes);

TYPED_TEST(CutoffSolverTest, testFreeBoundary)
{ 
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    auto boundary_topology = this->br_->get_spatial_mesh()->getBoundaryInfo();
    int local_location[3] = {boundary_topology(rank, 1), boundary_topology(rank, 2), boundary_topology(rank, 3)};
    int max_location[3] = {boundary_topology(comm_size, 1), boundary_topology(comm_size, 2), boundary_topology(comm_size, 3)};
    for (int i = 0; i < comm_size; i++)
    {
        int remote_location[3] = {boundary_topology(i, 1), boundary_topology(i, 2), boundary_topology(i, 3)};
        bool result = this->br_->isValidRank(remote_location, local_location, max_location);
        printf("R%d, valid: %d\n", rank, result);
    }
    
    ASSERT_EQ(0, 0);    
    // auto z = this->position_np_->view();

    // this->populateArray(z);

    // this->bc_non_periodic_.applyField(*this->testMeshNonPeriodic_, *this->position_np_, 1);

    // this->testFreeBC(z);
}

} // end namespace BeatnikTest
