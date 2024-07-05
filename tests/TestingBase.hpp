#ifndef _TESTINGBASE_HPP_
#define _TESTINGBASE_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

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

/*
 * Parent class to hold all the objects we want to test in one place.
 * Since so many objects depend on one another, it makes sense to initialize
 * one version of them here rather than multiple versions across different tests.
 */

template <class T>
class TestingBase : public ::testing::Test
{
    // Convenience type declarations
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;
    using local_grid_type = Cabana::Grid::LocalGrid<mesh_type>;
    using node_array_layout = std::shared_ptr<Cabana::Grid::ArrayLayout<Cabana::Grid::Node, mesh_type>>;

    using node_array = Cabana::Grid::Array<double, Cabana::Grid::Node, mesh_type, MemorySpace>;

    using surface_mesh_type = Beatnik::SurfaceMesh<ExecutionSpace,MemorySpace>;
    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    using br_type = Beatnik::CutoffBRSolver<ExecutionSpace, MemorySpace, Beatnik::Params>;


  protected:
    // Propterties
    const int meshSize_ = 64;
    const double boxWidth_ = 1.0;
    const int haloWidth_ = 2;
    std::array<double, 6> globalBoundingBox_ = {-1, -1, -1, 1, 1, 1};
    std::array<int, 2> globalNumNodes_ = {meshSize_, meshSize_};
    Cabana::Grid::DimBlockPartitioner<2> partitioner_;
    
    // Objects: p = periodic, f = nonperiodic (free)
    Beatnik::Params p_params_;
    Beatnik::Params f_params_;

    Beatnik::BoundaryCondition p_bc_;
    Beatnik::BoundaryCondition f_bc_;

    std::shared_ptr<surface_mesh_type> p_mesh_;
    std::shared_ptr<surface_mesh_type> f_mesh_;

    NullInitFunctor<2> createFunctor_;
    std::shared_ptr<pm_type> p_pm_;
    std::shared_ptr<pm_type> f_pm_;

    std::shared_ptr<br_type> p_br_cutoff_;
    std::shared_ptr<br_type> f_br_cutoff_;

    void SetUp() override
    {
        // Init boundary conditions
        for (int i = 0; i < 6; i++)
        {
            p_bc_.bounding_box[i] = globalBoundingBox_[i];
            f_bc_.bounding_box[i] = globalBoundingBox_[i];

            p_params_.global_bounding_box[i] = globalBoundingBox_[i];
            f_params_.global_bounding_box[i] = globalBoundingBox_[i];

        }
        for (int i = 0; i < 4; i++)
        {
            p_bc_.boundary_type[i] = Beatnik::BoundaryType::PERIODIC;
            f_bc_.boundary_type[i] = Beatnik::BoundaryType::FREE;
        }
        p_params_.periodic = {true, true};
        p_params_.cutoff_distance = 0.1;
        f_params_.periodic = {false, false};
        f_params_.cutoff_distance = 0.1;

        // Periodic
        this->p_mesh_ = std::make_shared<surface_mesh_type>( globalBoundingBox_, globalNumNodes_, p_params_.periodic, 
                                partitioner_, haloWidth_, MPI_COMM_WORLD );
        this->p_pm_ = std::make_shared<pm_type>( *p_mesh_, p_bc_, createFunctor_ );
        this->p_br_cutoff_ = std::make_shared<br_type>(*p_pm_, p_bc_, 1.0, 1.0, 1.0, p_params_);

        // Free
        this->f_mesh_ = std::make_shared<surface_mesh_type>( globalBoundingBox_, globalNumNodes_, f_params_.periodic, 
                                partitioner_, haloWidth_, MPI_COMM_WORLD );
        this->f_pm_ = std::make_shared<pm_type>( *f_mesh_, f_bc_, createFunctor_ );
        this->f_br_cutoff_ = std::make_shared<br_type>(*f_pm_, f_bc_, 1.0, 1.0, 1.0, f_params_);
    }

    void TearDown() override
    {
        this->p_br_cutoff_ = NULL;
        this->f_br_cutoff_ = NULL;

        this->p_pm_ = NULL;
        this->f_pm_ = NULL;

        this->p_mesh_ = NULL;
        this->f_mesh_ = NULL;
    }

};

} // end namespace BeatnikTest

#endif // _TESTINGBASE_HPP_
