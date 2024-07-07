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
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using node_array_layout = std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, mesh_type>>;

    using node_array = Cabana::Grid::Array<double, Cabana::Grid::Node, mesh_type, MemorySpace>;

    using br_type = Beatnik::CutoffBRSolver<ExecutionSpace, MemorySpace, Beatnik::Params>;
    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;

    // XXX - Can we get particle_node and particle_array_type from the CutoffBRSolver class
    // instead of re-copying them here?
    using particle_node = Cabana::MemberTypes<double[3], // xyz position in space                           0
                                              double[3], // Own omega for BR                                1
                                              double[3], // zdot                                            2
                                              double,    // Simpson weight                                  3
                                              int[2],    // Index in PositionView z and VorticityView w     4
                                              int,       // Point ID in 2D                                  5
                                              int,       // Owning rank in 2D space                         6
                                              int,       // Owning rank in 3D space                         7
                                              int        // Point ID in 3D                                  8
                                              >;
    using particle_array_type = Cabana::AoSoA<particle_node, device_type, 4>;
   

  protected:
    MPI_Comm comm_;
    int comm_size_, rank_;
    double cutoff_distance_;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
        comm_ = this->p_pm_->mesh().localGrid()->globalGrid().comm();
        MPI_Comm_size(comm_, &comm_size_);
        MPI_Comm_rank(comm_, &rank_);

        cutoff_distance_ = this->p_params_.cutoff_distance;
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
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

    void tstPeriodicHalo()
    {
        /* Start setup */
        this->p_pm_->gather();
        auto z = this->p_pm_->get( Cabana::Grid::Node(), Beatnik::Field::Position() );
        auto w = this->p_pm_->get( Cabana::Grid::Node(), Beatnik::Field::Vorticity() );
        particle_array_type particle_array_;

        int dim = this->comm_size_ + this->haloWidth_*2;
        Kokkos::View<double**[2], device_type> omega_("omega_", dim, dim);
        this->p_br_cutoff_->initializeParticles(particle_array_, z, w, omega_);
        this->p_br_cutoff_->migrateParticlesTo3D(particle_array_);
        int owned_3D_count = particle_array_.size();
        this->p_br_cutoff_->performHaloExchange3D(particle_array_);
        this->p_br_cutoff_->correctPeriodicBoundaries(particle_array_, owned_3D_count);

        auto position_part = Cabana::slice<0>(particle_array_);
        auto rank3d_part = Cabana::slice<7>(particle_array_);

        int total_size = particle_array_.size();
        auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
        int local_location[3] = {boundary_topology(rank_, 1), boundary_topology(rank_, 2), boundary_topology(rank_, 3)};
        int max_location[3] = {boundary_topology(comm_size_, 1), boundary_topology(comm_size_, 2), boundary_topology(comm_size_, 3)};
        int is_neighbor[26];
        this->p_br_cutoff_->getPeriodicNeighbors(is_neighbor);
        int isOnBoundary = this->p_br_cutoff_->isOnBoundary(local_location, max_location);
        /* End setup */

        /* Iterate over each haloed particle recieved and check the following conditions:
         * 
         */
        for (int index = owned_3D_count; index < total_size; index++)
        {
            int remote_rank = rank3d_part(index);
            if (!isOnBoundary)
            {
                // If the rank is not on a boundary, none of the particles it recieves
                // should be from ranks marked as a periodic neighbor
                EXPECT_EQ(is_neighbor[remote_rank], 0) << "Rank " << rank_
                    << " incorrectly marked rank " << remote_rank << " as a periodic neighbor in x/y.";
            }
        }




    //     if (isOnBoundaryCorrect(local_location, max_location))
    //     {
    //         std::array<double, 6> global_bounding_box = this->p_params_.global_bounding_box;
    //         int is_neighbor[26];
    //         this->p_br_cutoff->getPeriodicNeighborsCorrect(is_neighbor);

    //         for (int index = owned_3D_count; index < total_size; index++)
    //         {
    //             /* If local process is not on a boundary, exit. No particles
    //             * accross the boundary would have been recieved.
    //             * We only consider the x and y postions here because the
    //             * z-direction will never be periodic.
    //             */
    //             int remote_rank = rank3d_part(index);
    //             if (is_neighbor[remote_rank] == 1)
    //             {
    //                 // Get the dimenions to adjust
    //                 // Dimensions across a boundary will be more than one distance away in x/y/z space
    //                 int traveled[3];
    //                 for (int dim = 1; dim < 4; dim++)
    //                 {
    //                     if (boundary_topology(remote_rank, dim) - boundary_topology(rank, dim) > 1)
    //                     {
    //                         traveled[dim-1] = -1;
    //                     }
    //                     else if (boundary_topology(remote_rank, dim) - boundary_topology(rank, dim) < -1)
    //                     {
    //                         traveled[dim-1] = 1;
    //                     }
    //                     else
    //                     {
    //                         traveled[dim-1] = 0;
    //                     }
    //                 }

    //                 if (rank == 12)
    //                     {
    //                         printf("R%d: from R%d (index %d): traveled: %d, %d, %d, ", rank, remote_rank, index, traveled[0], traveled[1], traveled[2]);
    //                         printf("old pos: %0.5lf, %0.5lf, %0.5lf, ", position_part(index, 0), position_part(index, 1), position_part(index, 2));
    //                         //printf("Adjusting pos dim %d: diff: %0.5lf, old: %0.5lf new: %0.5lf\n", dim, diff, new_pos);
    //                     }
    //                 for (int dim = 0; dim < 3; dim++)
    //                 {
    //                     if (traveled[dim] != 0)
    //                     {
    //                         // -1, -1, -1, 1, 1, 1
    //                         double diff = global_bounding_box[dim+3] - global_bounding_box[dim];
    //                         // Adjust position
    //                         double new_pos = position_part(index, dim) + diff * traveled[dim];
    //                         position_part(index, dim) = new_pos;
    //                     }
    //                 }
    //                 if (rank == 12)
    //                     {
    //                         //printf("R%d: from R%d (index %d): traveled: %d, %d, %d\n", rank, remote_rank, index, traveled[0], traveled[1], traveled[2]);
    //                         printf("new pos: %0.5lf, %0.5lf, %0.5lf\n", 
    //                         position_part(index, 0), position_part(index, 1), position_part(index, 2));
    //                         //printf("Adjusting pos dim %d: diff: %0.5lf, old: %0.5lf new: %0.5lf\n", dim, diff, new_pos);
    //                     }
    //             }

    //         }

                
    //     }
    }
};

} // end namespace BeatnikTest

#endif // _TSTCUTOFFSOLVER_HPP_
