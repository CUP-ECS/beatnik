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

        // Copy particle_array_ to host memory
        auto host_particle_array = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), particle_array_);

        auto position_part = Cabana::slice<0>(host_particle_array);
        auto rank3d_part = Cabana::slice<7>(host_particle_array);

        int total_size = host_particle_array.size();
        auto boundary_topology = this->p_br_cutoff_->get_spatial_mesh()->getBoundaryInfo();
        int local_location[3] = {boundary_topology(rank_, 1), boundary_topology(rank_, 2), boundary_topology(rank_, 3)};
        int num_procs[3] = {boundary_topology(comm_size_, 1), boundary_topology(comm_size_, 2), boundary_topology(comm_size_, 3)};
        int is_neighbor[26];
        this->p_br_cutoff_->getPeriodicNeighbors(is_neighbor);
        int isOnBoundary = this->p_br_cutoff_->isOnBoundary(local_location, num_procs);
        /* End setup */

        /* Iterate over each haloed particle recieved and check the following conditions:
         * 
         */
        int num_periodic_recieved = 0;
        int procs_recv_in_xy = 0;
        int procs_recv_in_x = 0;
        int procs_recv_in_y = 0;
        for (int index = owned_3D_count; index < total_size; index++)
        {
            int remote_rank = rank3d_part(index);
            if (!isOnBoundary)
            {
                // If the rank is not on a boundary, none of the particles it recieves
                // should be from ranks marked as a periodic neighbor
                EXPECT_EQ(is_neighbor[remote_rank], 0) << "Rank " << rank_
                    << " incorrectly marked rank " << remote_rank << " as a periodic neighbor in x/y.\n";
            }
            else
            {
                // Make sure the process recieved particles across its periodic boundary
                // XXX - Don't use the getPeriodicNeighbors function in CutoffBRSolver to do this?
                if (is_neighbor[remote_rank] == 1)
                {
                    num_periodic_recieved++;

                    /* Check general mechanics of the halo */
                    int traveled[3];
                    for (int dim = 1; dim < 4; dim++)
                    {
                        if (boundary_topology(remote_rank, dim) - boundary_topology(rank_, dim) > 1)
                        {
                            traveled[dim-1] = -1;
                        }
                        else if (boundary_topology(remote_rank, dim) - boundary_topology(rank_, dim) < -1)
                        {
                            traveled[dim-1] = 1;
                        }
                        else
                        {
                            traveled[dim-1] = 0;
                        }
                    }
                    if (traveled[0] != 0 && traveled[1] != 0) procs_recv_in_xy++;
                    else if (traveled[0] != 0 && traveled[1] == 0) procs_recv_in_x++;
                    else if (traveled[0] == 0 && traveled[1] != 0) procs_recv_in_y++;
                    

                    /* Test if the coordinates were adjusted correctly:
                     * The traveled dimension(s) have coordinates outside
                     * of the bounding box but inside the cutoff distance.
                     */
                    for (int dim = 0; dim < 2; dim++)
                    {
                        double abs_pos = abs(position_part(index, dim));
                        double max_dim = this->globalBoundingBox_[dim+3];
                        double max_coord = max_dim + this->cutoff_distance_;
                        if (traveled[dim])
                        {
                            EXPECT_GE(max_dim, abs_pos) << "Rank " << rank_ << ": Absolute value of adjusted coordinate (index " 
                                << index << ") in dimension " << dim << " is not outside of the bounding box.\n";
                            EXPECT_LE(abs_pos, max_coord) << "Rank " << rank_ << ": Absolute value of adjusted coordinate (index " 
                                << index << ") in dimension " << dim << " is outside of the bounding box + cutoff distance.\n";
                        }
                        else 
                        {
                            EXPECT_LE(max_dim, abs_pos) << "Rank " << rank_ << ": Absolute value of non-adjusted coordinate (index " 
                                << index << ") in dimension " << dim << " is not inside of the bounding box.\n";
                        }
                        
                    }
                }
            } 
        }

        if (isOnBoundary)
        {
            EXPECT_GT(num_periodic_recieved, 0) << "Rank " << rank_ 
                << " is on a periodic boundary and did not recieve any particles across the boundary.\n";
        }

        /* Make sure that the number of ranks that recieved particles across
         * both the x and y dimensions equals the number of processes in
         * the z-dimension times four. 
         */ 
        int xy_procs = num_procs[2] * 4;
        int x_procs = num_procs[0] * num_procs[2] * 2 - xy_procs;
        int y_procs = num_procs[1] * num_procs[2] * 2 - xy_procs;
        EXPECT_EQ(procs_recv_in_xy, xy_procs) << "The number of processes recieving points across the x/y dimensions is not as expected.\n";
        EXPECT_EQ(procs_recv_in_x, x_procs) << "The number of processes recieving points across the x dimension is not as expected.\n";
        EXPECT_EQ(procs_recv_in_y, y_procs) << "The number of processes recieving points across the y dimension is not as expected.\n";
    }
};

} // end namespace BeatnikTest

#endif // _TSTCUTOFFSOLVER_HPP_
