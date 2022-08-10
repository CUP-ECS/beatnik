#ifndef _TSTMESH_HPP_
#define _TSTMESH_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
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
    // We need Cajita Arrays
    // Convenience type declarations
    using Cell = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      typename T::MemorySpace>;
    using mesh_type = Beatnik::Mesh<typename T::ExecutionSpace, typename T::MemorySpace>;

  public:
    virtual void SetUp() override
    {
        // Allocate and initialize the Cajita mesh
        globalNumCells_ = { boxCells_ , boxCells_ };
        globalBoundingBox_ = {-1, -1, -1, 1, 1, 1};
        testMesh_ = std::make_unique<mesh_type>( globalBoundingBox_, globalNumCells_, periodic_, 
                        partitioner_, haloWidth_, MPI_COMM_WORLD );
    }

    virtual void TearDown() override { testMesh_ = NULL; }

    std::array<double, 6> globalBoundingBox_;
    std::array<int, 2> globalNumCells_;
    const std::array<bool, 2> periodic_ = {true, true};
    const double boxWidth_ = 1.0;
    const int haloWidth_ = 2;
    const int boxCells_ = 512;
    Cajita::DimBlockPartitioner<2> partitioner_;

    std::unique_ptr<mesh_type> testMesh_;
};

#endif // _TSTMESH_HPP_
