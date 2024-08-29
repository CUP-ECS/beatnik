#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstDriver.hpp"
#include "tstMesh.hpp"
#include "tstProblemManager.hpp"

TYPED_TEST_SUITE( ProblemManagerTest, MeshDeviceTypes );

using Node = Cabana::Grid::Cell;
using Position = Beatnik::Field::Position;
using Vorticity = Beatnik::Field::Vorticity;

TYPED_TEST( ProblemManagerTest, StateArrayTest )
{
    using ExecutionSpace = typename TestFixture::ExecutionSpace;

    // Get basic mesh state
    auto pm = this->testPM_;
    auto mesh = pm->mesh();
    auto z = pm->get( Node(), Position() );
    auto w = pm->get( Node(), Vorticity() );
    auto rank = mesh->rank();

    /* Set values in the array based on our rank. Each cell gets a value of
     * rank*1000 + i * 100 + j * 10 + dim
     */
    auto zspace = mesh->localGrid()->indexSpace( Cabana::Grid::Own(), Node(),
                                                 Cabana::Grid::Local() );
    Kokkos::parallel_for(
        "InitializeCellFields",
        createExecutionPolicy( zspace, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
	    for (int d = 0; d < 3; d++)
                z( i, j, d ) = rank * 1000 + i * 100 + j * 10 + d;
	    for (int d = 0; d < 2; d++)
                w( i, j, d ) = rank * 1000 + i * 100 + j * 10 + d;
        } );

    auto zcopy =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), z );
    auto wcopy =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), w );
    for ( int i = zspace.min( 0 ); i < zspace.max( 0 ); i++ )
        for ( int j = zspace.min( 1 ); j < zspace.max( 1 ); j++ )
        {
            for ( int d = 0; d < 3; d++) 
                ASSERT_EQ( zcopy( i, j, d ), rank * 1000 + i * 100 + j * 10 + d );
            for ( int d = 0; d < 2; d++) 
                ASSERT_EQ( wcopy( i, j, d ), rank * 1000 + i * 100 + j * 10 + d );
        }
}

TYPED_TEST( ProblemManagerTest, HaloTest )
{
    using ExecutionSpace = typename TestFixture::ExecutionSpace;

    auto pm = this->testPM_;
    auto mesh = pm->mesh();
    auto rank = mesh->rank();
    auto z = pm->get( Node(), Position() );
    auto zspace = mesh->localGrid()->indexSpace( Cabana::Grid::Own(), Node(),
                                                 Cabana::Grid::Local() );
    Kokkos::parallel_for(
        "InitializePositions",
        createExecutionPolicy( uspace, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            for (int d = 0; d < 3; d++)
                z( i, j, d ) = rank * 1000 + i * 100 + j * 10 + d;
        } );

    // Check that we can halo the views appropriately, using the FaceI direction
    // as the check
    pm->gather( );
    std::array<int, 2> directions[4] = {
        { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
    auto zcopy =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), z );
    for ( int i = 0; i < 4; i++ )
    {
        auto dir = directions[i];
        int neighbor_rank = mesh->localGrid()->neighborRank( dir );
        auto z_shared_space = mesh->localGrid()->sharedIndexSpace(
            Cabana::Grid::Ghost(), Node(), dir );
        for ( int i = z_shared_space.min( 0 ); i < z_shared_space.max( 0 ); i++ )
            for ( int j = z_shared_space.min( 1 ); j < z_shared_space.max( 1 ); j++ )
                for (int d = 0; d < 3; d++)
                    ASSERT_EQ( zcopy( i, j, 0 ),
                               neighbor_rank * 1000 + i * 100 + j * 10 + d );
    }
}
