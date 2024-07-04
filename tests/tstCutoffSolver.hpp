#ifndef _TSTCUTOFFSOLVER_HPP_
#define _TSTCUTOFFSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "testingBase.hpp"

namespace BeatnikTest
{

template <class T>
class CutoffSolverTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using node_array_layout = std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, mesh_type>>;

    using node_array = std::shared_ptr<Cabana::Grid::Array<double, Cabana::Grid::Node, mesh_type, MemorySpace>>;

    using br_type = Beatnik::CutoffBRSolver<typename T::ExecutionSpace,
                                          typename T::MemorySpace,
                                          Beatnik::Params>;
    using pm_type = Beatnik::ProblemManager<typename T::ExecutionSpace,
                                            typename T::MemorySpace>;

  protected:
    void SetUp() override
    {
        TestingBase<T>::SetUp();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

  public:
    int areNeighbors(const int remote_location[3], const int local_location[3], const int num_procs[3])
    {
        // Handle corners differently
        if ((local_location[0] == 0 && local_location[1] == 0) ||
                    (local_location[0] == num_procs[0]-1 &&
                     local_location[1] == num_procs[1]-1)) // Top left and bottom right corners
        {
            //int n1 = 
        }

    return 0;
    }
};

} // end namespace BeatnikTest

#endif // _TSTCUTOFFSOLVER_HPP_
