#ifndef _TSTCUTOFFSOLVER_HPP_
#define _TSTCUTOFFSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "tstMesh.hpp"
#include "tstDriver.hpp"

namespace BeatnikTest
{

template <std::size_t Dim>
class NullInitFunctor
{
  public:
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Position,
                     [[maybe_unused]] const int index[Dim],
                     [[maybe_unused]] const double x[Dim],
                     [[maybe_unused]] double& z1, 
                     [[maybe_unused]] double& z2, 
                     [[maybe_unused]] double& z3) const
    {
        return true;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Vorticity,
                     [[maybe_unused]] const int index[Dim],
                     [[maybe_unused]] const double x[Dim],
                     [[maybe_unused]] double& w1,
                     [[maybe_unused]] double& w2 ) const
    {
        return true;
    };
};

template <class T>
class CutoffSolverTest : public MeshTest<T>
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
    NullInitFunctor<2> createFunctor_;
    std::shared_ptr<br_type> br_;
    std::shared_ptr<pm_type> pm_;
    Beatnik::BoundaryCondition bc_periodic_;
    Beatnik::BoundaryCondition bc_non_periodic_;
    node_array_layout node_layout_np_; 
    node_array position_np_;
    
    void SetUp() override
    {
        MeshTest<T>::SetUp();
        Beatnik::Params params;
        for (int i = 0; i < 6; i++)
        {
            params.global_bounding_box[i] = MeshTest<T>::globalBoundingBox_[i];
        }
        params.periodic = {true, true};
        params.cutoff_distance = 0.1;

        Beatnik::BoundaryCondition bc;
        // Dummy ProblemManager is not used but needed to create the CutoffBRSolver class
        this->pm_ = std::make_shared<pm_type>( *this->testMeshPeriodic_, bc, createFunctor_ );
        this->br_ = std::make_shared<br_type>(*pm_, bc, 1.0, 1.0, 1.0, params);
    }

    void TearDown() override
    {
        this->br_ = NULL;
        this->pm_ = NULL;
        MeshTest<T>::TearDown();
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
