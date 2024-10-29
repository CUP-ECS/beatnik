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
class MeshTest : public TestingBase<T>
{
    // Convenience type declarations
    using Cell = Cabana::Grid::Node;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      typename T::MemorySpace>;
    using mesh_type = Beatnik::SurfaceMesh<typename T::ExecutionSpace, typename T::MemorySpace>;

  protected:
    void SetUp() override
    {
        TestingBase<T>::SetUp();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }
};

} // end namespace BeatnikTest

#endif // _TSTMESH_HPP_
