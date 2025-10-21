#ifndef _TSTMESH_HPP_
#define _TSTMESH_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

/*
 * Parameterizing on number of dimensions in here is messy and we
 * don't do it yet. We'll sort that out when we move to 3D as well.
 * These webpage has some ideas on how to I haven't yet deciphered:
 * 1. http://www.ashermancinelli.com/gtest-type-val-param
 * 2.
 * https://stackoverflow.com/questions/8507385/google-test-is-there-a-way-to-combine-a-test-which-is-both-type-parameterized-a
 */

template <class T>
class MeshTest : public TestingBase
{
    // Convenience type declarations
    using Cell = Cabana::Grid::Node;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
        typename TestingBase::MemorySpace>>;
    using mesh_type = Beatnik::SurfaceMesh<typename T::ExecutionSpace, typename T::MemorySpace>;

  protected:
    void SetUp() override
    {
        TestingBase::SetUp();
    }

    void TearDown() override
    { 
        TestingBase::TearDown();
    }
};

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

// template <typename TestCommSpace>
// class MeshTypedTest : public ::testing::Test
// {
// };

// // Add additional backends to test when implemented.
// using CommSpaceTypes = ::testing::Types<Cabana::Mpi>;

// // Need a trailing comma
// // to avoid an error when compiling with clang++
// TYPED_TEST_SUITE( DistributorTypedTest, CommSpaceTypes, );

// TEST( MeshTest, BasicParameters )
// {
//     int r;

//     MPI_Comm_rank( MPI_COMM_WORLD, &r );
//     EXPECT_EQ( this->p_mesh_->rank(), r );
//     EXPECT_EQ( this->f_mesh_->rank(), r );
// };

// TYPED_TEST( MeshTest, PeriodicGridSetup )
// {
//     /* Here we check that the local grid is decomposed like
//      * we think it should be. That is, the number of ghosts cells
//      * is right, the index spaces for owned, ghost, and boundary
//      * cells are right, and so on. */
//     auto local_grid = this->p_mesh_->localGrid();
//     auto & global_grid = local_grid->globalGrid();
//     int cabana_nodes = this->meshSize_ - 1;

//     for ( int i = 0; i < 2; i++ )
//     {
//         EXPECT_EQ( cabana_nodes,
//                    global_grid.globalNumEntity( Cabana::Grid::Node(), i ) );
//     }
// };
// TYPED_TEST( MeshTest, NonperiodicGridSetup )
// {
//     /* Here we check that the local grid is decomposed like
//      * we think it should be. That is, the number of ghosts cells
//      * is right, the index spaces for owned, ghost, and boundary
//      * cells are right, and so on. */
//     auto local_grid = this->f_mesh_->localGrid();
//     auto & global_grid = local_grid->globalGrid();

//     for ( int i = 0; i < 2; i++ )
//     {
//         EXPECT_EQ( this->meshSize_,
//                    global_grid.globalNumEntity( Cabana::Grid::Node(), i ) );
//     }

//     /* Make sure the number of owned nodes is our share of what was requested */
//     auto own_local_node_space = local_grid->indexSpace(
//         Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
//     for ( int i = 0; i < 2; i++ )
//     {
//         EXPECT_EQ( own_local_node_space.extent( i ),
//                    this->meshSize_/ global_grid.dimNumBlock( i ) );
//     }

//     /*
//      * Next we extract the ghosted nodes, which encompass the owned nodes and
//      * the ghosts in each dimension. 
//      */
//     auto ghost_local_node_space = local_grid->indexSpace(
//         Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
//     for ( int i = 0; i < 2; i++ ) {
//         EXPECT_EQ( ghost_local_node_space.extent( i ),
//                    this->meshSize_ / global_grid.dimNumBlock( i ) +
//                    2 * this->haloWidth_ );
//     }

// };

} // end namespace BeatnikTest

#endif // _TSTMESH_HPP_
