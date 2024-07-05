#ifndef _TSTCUTOFFSOLVER_HPP_
#define _TSTCUTOFFSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

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
        printf("***************BEGIN TEARDOWN**********");
        TestingBase<T>::TearDown();
        printf("***************FINISHED TEARDOWN***************");
    }

  public:
    // XXX: Note: Curently identical to the function in the code. Manually checked it works for 16 processes.
    int isOnBoundaryCorrect(const int local_location[3],
                            const int max_location[3])
    {
        for (int i = 0; i < 2; i++)
        {
            if (local_location[i] == 0 || local_location[i] == max_location[i]-1)
            {
                return 1;
            }
        }
        return 0;
    }

    int areNeighbors(const int remote_location[3], const int local_location[3], const int num_procs[3])
    {
        for (int i = -1; i < 2; i++)
        {
            for (int j = -1, j < 2; j++)
            {
                for (int k = -1; k < 2; k++)
                {
                    
                }
            }
        }

        return 0;
    }

    void correctLocPeriodicXY(const int location[3], const int num_procs[3], int new_location[3])
    {
        new_location = {location[0], location[1], location[2]};
        // z-location never corrected because only periodic in X/Y
        for (int i = 0; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                // loc: 0, 0, 0;      loc: 0, 3, 0;     num_procs: 4, 4, 1
                if (location[i] + j >= num_procs[i])
                {
                    new_location[i] = location[i] - (num_procs[i]-1);
                }
            }
        }
    }
};

} // end namespace BeatnikTest

#endif // _TSTCUTOFFSOLVER_HPP_
