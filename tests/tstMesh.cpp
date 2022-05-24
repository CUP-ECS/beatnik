#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"

TYPED_TEST_SUITE( MeshTest, MeshDeviceTypes );

#if 0
TYPED_TEST( MeshTest, BasicParameters )
{
    int r;
    EXPECT_EQ( this->testMesh_->cellSize(), 1.0 );

    auto mins = this->testMesh_->minDomainGlobalNodeIndex();
    EXPECT_EQ( mins[0], 0 );
    EXPECT_EQ( mins[1], 0 );
    auto maxs = this->testMesh_->maxDomainGlobalNodeIndex();
    EXPECT_EQ( maxs[0], this->boxCells_ - 1);
    EXPECT_EQ( maxs[1], this->boxCells_ - 1 );

    MPI_Comm_rank( MPI_COMM_WORLD, &r );
    EXPECT_EQ( this->testMesh_->rank(), r );
};
#endif

TYPED_TEST( MeshTest, LocalGridSetup )
{
    /* Here we check that the local grid is decomposed like
     * we think it should be. That is, the number of ghosts cells
     * is right, the index spaces for owned, ghost, and boundary
     * cells are right, and so on. */
    auto local_grid = this->testMesh_->localGrid();
    Cajita::GlobalGrid<Cajita::UniformMesh<double, 2>> & global_grid = local_grid->globalGrid();
    std::cout << "Local Grid reference count: " << local_grid.use_count() << "\n";

    for ( int i = 0; i < 2; i++ )
    {
        // We're periodic, so Cells == Nodes
        EXPECT_EQ( this->boxCells_,
                   global_grid.globalNumEntity( Cajita::Node(), i ) );
    }

    /* Make sure the number of owned nodes is our share of what was requested */
    auto own_local_node_space = local_grid->indexSpace(
        Cajita::Own(), Cajita::Node(), Cajita::Local() );
    for ( int i = 0; i < 2; i++ )
    {
        EXPECT_EQ( own_local_node_space.extent( i ),
                   this->boxCells_ / global_grid.dimNumBlock( i ) );
    }

    /*
     * Next we extract the ghosted nodes, which encompass the owned nodes and
     * the ghosts in each dimension. 
     */
    auto ghost_local_node_space = local_grid->indexSpace(
        Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
    for ( int i = 0; i < 2; i++ ) {
        EXPECT_EQ( ghost_local_node_space.extent( i ),
                   this->boxCells_ / global_grid.dimNumBlock( 0 ) +
                   2 * this->haloWidth_ );
    }

};
