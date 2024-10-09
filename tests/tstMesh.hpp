#ifndef _TSTMESH_HPP_
#define _TSTMESH_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"

/*
 * Parameterizing on number of dimensions in here is messy and we
 * don't do it yet. We'll sort that out when we move to 3D as well.
 * These webpage has some ideas on how to I haven't yet deciphered:
 * 1. http://www.ashermancinelli.com/gtest-type-val-param
 * 2.
 * https://stackoverflow.com/questions/8507385/google-test-is-there-a-way-to-combine-a-test-which-is-both-type-parameterized-a
 */

template <class T>
class MeshTest : public ::testing::Test
{
    // We need Cabana Arrays
    // Convenience type declarations
    using Cell = Cabana::Grid::Node;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      typename T::MemorySpace>;
    using mesh_type = Beatnik::Mesh<typename T::ExecutionSpace, typename T::MemorySpace>;

  public:
    virtual void SetUp() override
    {
        // Allocate and initialize the Cabana mesh
        globalNumNodes_ = { boxNodes_, boxNodes_ };
        globalBoundingBox_ = {-1, -1, -1, 1, 1, 1};

        std::array<bool, 2> periodic_ = {true, true};
        testMeshPeriodic_ = std::make_unique<mesh_type>( globalBoundingBox_, globalNumNodes_, periodic_, 
                                partitioner_, haloWidth_, MPI_COMM_WORLD );

        periodic_ = {false, false};
        testMeshNonperiodic_ = std::make_unique<mesh_type>( globalBoundingBox_, globalNumNodes_, periodic_, 
                                partitioner_, haloWidth_, MPI_COMM_WORLD );
    }

    virtual void TearDown() override { testMeshPeriodic_ = NULL; testMeshNonperiodic_ = NULL; }

    std::array<double, 6> globalBoundingBox_;
    std::array<int, 2> globalNumNodes_;
    const double boxWidth_ = 1.0;
    const int haloWidth_ = 2;
    const int boxNodes_ = 512;
    Cabana::Grid::DimBlockPartitioner<2> partitioner_;

    std::unique_ptr<mesh_type> testMeshPeriodic_;
    std::unique_ptr<mesh_type> testMeshNonperiodic_;
};

#endif // _TSTMESH_HPP_
