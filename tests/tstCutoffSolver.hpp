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

    template <class Topology>
    void getNeighbors(const int my_rank, const Topology topology, int is_neighbor[26])
    {
        for (int i = 0; i < 26; i++)
        {
            is_neighbor[i] = -1;
        }

        const auto local_grid = this->p_br_cutoff_->get_spatial_mesh()->localGrid();
        const auto local_mesh =
            Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( *local_grid );
        


        //Kokkos::Array<Cabana::Grid::IndexSpace<4>, topology_size> index_spaces;

        // Store all neighboring shared index space mesh bounds so we only have
        // to launch one kernel during the actual ghost search.
        int n = 0;
        for ( int k = -1; k < 2; ++k )
        {
            for ( int j = -1; j < 2; ++j )
            {
                for ( int i = -1; i < 2; ++i, ++n )
                {
                    if ( i != 0 || j != 0 || k != 0 )
                    {
                        int neighbor_rank = local_grid->neighborRank( i, j, k );
                        if (neighbor_rank != -1)
                        {
                            for (int w = 1; w < 3; w++)
                            {
                                if (abs(topology(my_rank, w) - topology(neighbor_rank, w)) > 1)
                                {
                                    if (my_rank == 3)
                                    {
                                        printf("R%d: (%d, %d, %d) Neighbor rank: %d\n", my_rank, i, j, k, neighbor_rank);
                                    }
                                }
                            }
                        }
                        // Potentially invalid neighbor ranks (non-periodic
                        // global boundary)
                        
                        // if ( neighbor_rank != -1 )
                        // {
                        //     auto sis = local_grid.sharedIndexSpace(
                        //         Cabana::Grid::Own(), Cabana::Grid::Cell(), i, j, k,
                        //         _min_halo );
                        //     auto min_ind = sis.min();
                        //     auto max_ind = sis.max();
                        //     local_mesh.coordinates( Cabana::Grid::Node(),
                        //                             min_ind.data(),
                        //                             _min_coord[n].data() );
                        //     local_mesh.coordinates( Cabana::Grid::Node(),
                        //                             max_ind.data(),
                        //                             _max_coord[n].data() );
                        // }
                    }
                }
            }
        }
    }

    void correctLocPeriodicXY(const int location[3], const int num_procs[3], int new_location[3])
    {
        // new_location = {location[0], location[1], location[2]};
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
