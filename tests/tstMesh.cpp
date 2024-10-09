#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"

TYPED_TEST_SUITE( MeshTest, MeshDeviceTypes );

TYPED_TEST( MeshTest, BasicParameters )
{
    int r;

    MPI_Comm_rank( MPI_COMM_WORLD, &r );
    EXPECT_EQ( this->testMeshPeriodic_->rank(), r );
    EXPECT_EQ( this->testMeshNonperiodic_->rank(), r );
};

TYPED_TEST( MeshTest, PeriodicGridSetup )
{
    /* Here we check that the local grid is decomposed like
     * we think it should be. That is, the number of ghosts cells
     * is right, the index spaces for owned, ghost, and boundary
     * cells are right, and so on. */
    auto local_grid = this->testMeshPeriodic_->localGrid();
    auto & global_grid = local_grid->globalGrid();
    int cabana_nodes = this->boxNodes_ - 1;

    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( cabana_nodes,
                   global_grid.globalNumEntity( Cabana::Grid::Node(), i ) );
    }
};
TYPED_TEST( MeshTest, NonperiodicGridSetup )
{
    /* Here we check that the local grid is decomposed like
     * we think it should be. That is, the number of ghosts cells
     * is right, the index spaces for owned, ghost, and boundary
     * cells are right, and so on. */
    auto local_grid = this->testMeshNonperiodic_->localGrid();
    auto & global_grid = local_grid->globalGrid();

    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( this->boxNodes_,
                   global_grid.globalNumEntity( Cabana::Grid::Node(), i ) );
    }

    /* Make sure the number of owned nodes is our share of what was requested */
    auto own_local_node_space = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( own_local_node_space.extent( i ),
                   this->boxNodes_/ global_grid.dimNumBlock( i ) );
    }

    /*
     * Next we extract the ghosted nodes, which encompass the owned nodes and
     * the ghosts in each dimension. 
     */
    auto ghost_local_node_space = local_grid->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    for ( int i = 0; i < 2; i++ ) {
        EXPECT_EQ( ghost_local_node_space.extent( i ),
                   this->boxNodes_ / global_grid.dimNumBlock( i ) +
                   2 * this->haloWidth_ );
    }

};
