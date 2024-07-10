#ifndef _TSTZMODEL_HPP_
#define _TSTZMODEL_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

template <class T>
class ExactBRSolverTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using surface_mesh_type = Beatnik::SurfaceMesh<ExecutionSpace,MemorySpace>;
    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    using br_exact_type = Beatnik::ExactBRSolver<ExecutionSpace, MemorySpace, Beatnik::Params>;

    using l2g_type = Cabana::Grid::IndexConversion::L2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node, Cabana::Grid::UniformMesh<double, 2>,
                      MemorySpace>;
    using node_view = typename node_array::view_type;

  protected:
    MPI_Comm comm_, comm_single_; 
    int rank_, comm_size_, single_rank_;

    Beatnik::BoundaryCondition single_bc_;
    Beatnik::Params single_params_;
    std::shared_ptr<surface_mesh_type> single_mesh_;
    std::shared_ptr<pm_type> single_pm_;
    std::shared_ptr<node_array> zdot_correct_;

    void SetUp() override
    {
        TestingBase<T>::SetUp();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

    // Get the global mesh comm from ProblemManager
    template <class pm_bc_type>
    void Init(pm_bc_type pm_)
    {
        this->comm_ = pm_->mesh().localGrid()->globalGrid().comm();
        MPI_Comm_size(comm_, &comm_size_);
        MPI_Comm_rank(comm_, &rank_);

        MPI_Comm_split(comm_, rank_, 1, &comm_single_);
        MPI_Comm_rank(comm_single_, &single_rank_);

        // Create new objects with the same parameters
        // as the global one, but only split amoungst one process
        for (int i = 0; i < 6; i++)
        {
            this->single_bc_.bounding_box[i] = this->globalBoundingBox_[i];
            this->single_params_.global_bounding_box[i] = this->globalBoundingBox_[i];
        }
        int is_periodic = single_pm_->mesh().is_periodic(); // 1 if periodic
        for (int i = 0; i < 4; i++)
        {
            // variable = (condition) ? expressionTrue : expressionFalse;
            // PERIODIC = 0, FREE = 1 in BoundaryCondition object
            this->single_bc_.boundary_type[i] = (is_periodic) ? Beatnik::BoundaryType::PERIODIC : Beatnik::BoundaryType::FREE;
        }
        this->single_params_.periodic = {(bool) is_periodic, (bool) is_periodic};

        MeshInitFunc single_MeshInitFunc_(this->globalBoundingBox_, this->tilt_, this->m_, this->v_, this->p_, this->globalNumNodes_, Beatnik::BoundaryType::FREE);
        this->single_mesh_ = std::make_shared<surface_mesh_type>( this->globalBoundingBox_, this->globalNumNodes_, single_params_.periodic, 
                                this->partitioner_, this->haloWidth_, comm_single_ );
        this->single_pm_ = std::make_shared<pm_type>( *single_mesh_, single_bc_, this->p_, single_MeshInitFunc_ );
        this->p_br_exact_ = std::make_shared<br_exact_type>(*single_pm_, single_bc_, this->epsilon_, this->dx_, this->dy_, single_params_);
    }

  public:
    template <class VorticityView, class PositionView>
    KOKKOS_INLINE_FUNCTION
    void BR(double out[3], PositionView z, PositionView z2, VorticityView w2,
            double epsilon, double dx, double dy, double weight, int i, int j, int k, int l,
            double offset[3])
    {
        double omega[3], zdiff[3], zsize;
        zsize = 0.0;
        for (int d = 0; d < 3; d++) {
            omega[d] = w2(k, l, 1) * Beatnik::Operators::Dx(z2, k, l, d, dx) - w2(k, l, 0) * Beatnik::Operators::Dy(z2, k, l, d, dy);
            zdiff[d] = z(i, j, d) - (z2(k, l, d) + offset[d]);
            zsize += zdiff[d] * zdiff[d];
        }  
        zsize = pow(zsize + epsilon, 1.5); // matlab code doesn't square epsilon
        for (int d = 0; d < 3; d++) {
            zdiff[d] /= zsize;
        }
        Beatnik::Operators::cross(out, omega, zdiff);
        for (int d = 0; d < 3; d++) {  
            out[d] *= (dx * dy * weight) / (-4.0 * Kokkos::numbers::pi_v<double>);
        }
    }
    /** Calculate zdot on the original mesh but only with one process
     */
    void populateSingleCorrectZdot()
    {
        // Setup
        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( single_pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        zdot_correct_ = Cabana::Grid::createArray<double, MemorySpace>(
            "zdot_correct_", node_triple_layout );
        
        auto z = single_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = single_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        double dx = this->dx_;
        double dy = this->dy_;

        auto zdot_ = this->zdot_correct_->view();

        auto local_grid = single_pm_->mesh().localGrid();

        Kokkos::View<double ***,
             typename node_view::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot_;
    
        /* Zero out all of the i/j points - XXX Is this needed are is this already zeroed somewhere else? */
        auto local_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for("Exact BR Zero Loop",
            Cabana::Grid::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                atomic_zdot(i, j, n) = 0.0;
        });


        // Calculations
        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = single_l2g_.local_own_min[d];
            rmax[d] = single_l2g_.local_own_max[d];
        }
	    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        /* Figure out which directions we need to project the k/l point to
         * for any periodic boundary conditions */
        int kstart, lstart, kend, lend;
        if (single_bc_.isPeriodicBoundary({0, 1})) {
            kstart = -1; kend = 1;
        } else {
            kstart = kend = 0;
        }
        if (single_bc_.isPeriodicBoundary({1, 1})) {
            lstart = -1; lend = 1;
        } else {
            lstart = lend = 0;
        }

        /* Figure out how wide the bounding box is in each direction */
        auto low = single_pm_->mesh().boundingBoxMin();
        auto high = single_pm_->mesh().boundingBoxMax();;
        double width[3];
        for (int d = 0; d < 3; d++) {
            width[d] = high[d] - low[d];
        }

        /* Local temporaries for any instance variables we need so that we
         * don't have to lambda-capture "this" */
        double epsilon = this->_epsilon;
        double dx = this->_dx, dy = this->_dy;

        // Mesh dimensions for Simpson weight calc
        int num_nodes = single_pm_->mesh().get_mesh_size();

        /* Now loop over the cross product of all the node on the interface */
        auto pair_space = Operators::crossIndexSpace(local_node_space, remote_space);
        Kokkos::parallel_for("Exact BR Force Loop",
            Cabana::Grid::createExecutionPolicy(pair_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j, int k, int l) {

            // We need the global indicies of the (k, l) point for Simpson's weight
            int remote_gi[2] = {k, l};  // k, l
            
            
            double brsum[3] = {0.0, 0.0, 0.0};

            /* Compute Simpson's 3/8 quadrature weight for this index */
            double weight;
            weight = Beatnik::simpsonWeight(remote_gi[0], num_nodes)
                        * Beatnik::simpsonWeight(remote_gi[1], num_nodes);
            
            /* We already have N^4 parallelism, so no need to parallelize on 
             * the BR periodic points. Instead we serialize this in each thread
             * and reuse the fetch of the i/j and k/l points */
            for (int kdir = kstart; kdir <= kend; kdir++) {
                for (int ldir = lstart; ldir <= lend; ldir++) {
                    double offset[3] = {0.0, 0.0, 0.0}, br[3];
                    offset[0] = kdir * width[0];
                    offset[1] = ldir * width[1];

                    /* Do the Birkhoff-Rott evaluation for this point */
                    Beatnik::Operators::BR(br, z, z, w, epsilon, dx, dy, weight,
                                  i, j, k, l, offset);
                    for (int d = 0; d < 3; d++) {
                        brsum[d] += br[d];
                    }
                }
            }

            /* Add it its contribution to the integral */
            for (int n = 0; n < 3; n++) {
                atomic_zdot(i, j, n) += brsum[n];
            }
        });
    }
};

} // end namespace BeatnikTest

#endif // _TSTZMODEL_HPP_
