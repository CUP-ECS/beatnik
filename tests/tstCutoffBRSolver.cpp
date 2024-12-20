#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstCutoffBRSolver.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace BeatnikTest
{

TYPED_TEST_SUITE(CutoffBRSolverTest, DeviceTypes);

TYPED_TEST(CutoffBRSolverTest, testIsOnBoundary)
{
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (!(comm_size == 1 || comm_size == 16))
    {
        printf("CutoffBRSolver test only supports 1 or 16 ranks. Exiting.\n");
        exit(1);
    } 

    auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
    int local_location[3] = {boundary_topology(rank, 1), boundary_topology(rank, 2), boundary_topology(rank, 3)};
    int max_location[3] = {boundary_topology(comm_size, 1), boundary_topology(comm_size, 2), boundary_topology(comm_size, 3)};
    int result = this->p_br_cutoff_->isOnBoundary(local_location, max_location);
    int correct_result = this->isOnBoundaryCorrect(local_location, max_location);
    EXPECT_EQ(result, correct_result) << "Rank " << rank << " incorrect.";
}

/* Tests whether points are haloed correctly in 3D space
 * across x/y boundaries. Doesn't perform an explicit check, 
 * but rather makes sure that at least one coordinate changed
 * and that the adjusted position is no more than
 * bounding box limit + cutoff distance outside of the mesh.
 */
TYPED_TEST(CutoffBRSolverTest, testPeriodicHalo)
{ 
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (!(comm_size == 1 || comm_size == 16))
    {
        printf("CutoffBRSolver test only supports 1 or 16 ranks. Exiting.\n");
        exit(1);
    } 
    this->tstPeriodicHalo();
}

} // end namespace BeatnikTest
