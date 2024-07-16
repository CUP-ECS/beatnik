#ifndef _TSTEXACT_BRSOLVER_HPP_
#define _TSTEXACT_BRSOLVER_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

/* Helpers */
static KOKKOS_INLINE_FUNCTION double simpsonWeight(int index, int len)
{
    if (index == (len - 1) || index == 0) return 3.0/8.0;
    else if (index % 3 == 0) return 3.0/4.0;
    else return 9.0/8.0;
}

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
        //omega[d] = omega_view(k, l, d);
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

template <class T>
class ExactBRSolverTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using surface_mesh_type = Beatnik::SurfaceMesh<ExecutionSpace,MemorySpace>;
    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    using br_exact_type = Beatnik::ExactBRSolver<ExecutionSpace, MemorySpace, Beatnik::Params>;
    using zm_type_h = Beatnik::ZModel<ExecutionSpace, MemorySpace, Beatnik::Order::High, Beatnik::Params>;

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
    std::shared_ptr<zm_type_h> single_zm_;
    std::shared_ptr<node_array> zdot_correct_;
    std::shared_ptr<node_array> zdot_test_;


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
            single_bc_.bounding_box[i] = this->globalBoundingBox_[i];
            single_params_.global_bounding_box[i] = this->globalBoundingBox_[i];
        }
        int is_periodic = pm_->mesh().is_periodic(); // 1 if periodic
        for (int i = 0; i < 4; i++)
        {
            // variable = (condition) ? expressionTrue : expressionFalse;
            // PERIODIC = 0, FREE = 1 in BoundaryCondition object
            this->single_bc_.boundary_type[i] = (is_periodic) ? Beatnik::BoundaryType::PERIODIC : Beatnik::BoundaryType::FREE;
        }
        this->single_params_.periodic = {(bool) is_periodic, (bool) is_periodic};
        this->single_mesh_ = std::make_shared<surface_mesh_type>( this->globalBoundingBox_, this->globalNumNodes_, single_params_.periodic, 
                                this->partitioner_, this->haloWidth_, comm_single_ );
        if (is_periodic)
        {
            this->single_pm_ = std::make_shared<pm_type>( *single_mesh_, single_bc_, this->p_, this->p_MeshInitFunc_ );
        }
        else 
        {
            this->single_pm_ = std::make_shared<pm_type>( *single_mesh_, single_bc_, this->p_, this->f_MeshInitFunc_ );
        }
    }

  public:
    /* Vorticity is initlized to zero in ProblemManager. Set the single and distributed
     * vorticities to non-zero values based on global index to keep
     * the values the same in the single and distributed versions.
     */
    template <class pm_bc_type>
    void initializeVorticity(pm_bc_type pm_)
    {
        auto w_dist = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());
        auto w_sing = single_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        auto local_grid_single = this->single_pm_->mesh().localGrid();
        auto local_grid = pm_->mesh().localGrid();
        auto local_L2G = Cabana::Grid::IndexConversion::createL2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>(*local_grid, Cabana::Grid::Node());

        // Global index + halo width = local index on single w
        int halo_width = this->haloWidth_;
        auto local_node_space_single = local_grid_single->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for("Single vorticity init",
            Cabana::Grid::createExecutionPolicy(local_node_space_single, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                w_sing(i, j, n) = (double) (i*j) * 0.005;
        });
        auto local_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for("Distributed vorticity init",
            Cabana::Grid::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {

                int li[2] = {i, j};
                int gi[2] = {0, 0}; // Holds global k, l
                local_L2G(li, gi);
                int g0_adj = gi[0] + halo_width;
                int g1_adj = gi[1] + halo_width;
                for (int n = 0; n < 3; n++)
                    w_dist(i, j, n) = (double) (g0_adj*g1_adj) * 0.005;
        });

        // Kokkos::parallel_for("Check vorticity init",
        //     Cabana::Grid::createExecutionPolicy(local_node_space_single, ExecutionSpace()),
        //     KOKKOS_LAMBDA(int k, int l) {
        //         printf("ws: %0.8lf, %0.8lf, %0.8lf, wd:, %0.8lf, %0.8lf, %0.8lf\n", w_sing(k, l, 0), w_sing(k, l, 1), w_sing(k, l, 2),
        //                 w_dist(k, l, 0), w_dist(k, l, 1), w_dist(k, l, 2));
        // });
    }

    template <class pm_bc_type, class BoundaryCondition, class AtomicView>
    void computeInterfaceVelocityPieceCorrect(pm_bc_type pm_,
                                              BoundaryCondition boundary_cond_,
                                              AtomicView atomic_zdot, node_view z, 
                                              node_view zremote, node_view wremote,
                                              l2g_type remote_L2G)
    {
        /* Project the Birkhoff-Rott calculation between all pairs of points on the 
         * interface, including accounting for any periodic boundary conditions.
         * Right now we brute force all of the points with no tiling to improve
         * memory access or optimizations to remove duplicate calculations. */

        // Get the local index spaces of pieces we're working with. For the local surface piece
        // this is just the nodes we own. For the remote surface piece, we extract it from the
        // L2G converter they sent us.
        auto local_grid = pm_->mesh().localGrid();
        auto local_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = remote_L2G.local_own_min[d];
            rmax[d] = remote_L2G.local_own_max[d];
        }
	    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        /* Figure out which directions we need to project the k/l point to
         * for any periodic boundary conditions */
        int kstart, lstart, kend, lend;
        if (boundary_cond_.isPeriodicBoundary({0, 1})) {
            kstart = -1; kend = 1;
        } else {
            kstart = kend = 0;
        }
        if (boundary_cond_.isPeriodicBoundary({1, 1})) {
            lstart = -1; lend = 1;
        } else {
            lstart = lend = 0;
        }

        /* Figure out how wide the bounding box is in each direction */
        auto low = pm_->mesh().boundingBoxMin();
        auto high = pm_->mesh().boundingBoxMax();;
        double width[3];
        for (int d = 0; d < 3; d++) {
            width[d] = high[d] - low[d];
        }

        /* Local temporaries for any instance variables we need so that we
         * don't have to lambda-capture "this" */
        double epsilon = this->epsilon_;
        double dx = this->dx_, dy = this->dy_;

        // Mesh dimensions for Simpson weight calc
        int num_nodes = pm_->mesh().get_surface_mesh_size();

        /* Now loop over the cross product of all the node on the interface */
        auto pair_space = Beatnik::Operators::crossIndexSpace(local_space, remote_space);
        Kokkos::parallel_for("Exact BR Force Loop",
            Cabana::Grid::createExecutionPolicy(pair_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j, int k, int l) {

            // We need the global indicies of the (k, l) point for Simpson's weight
            int remote_li[2] = {k, l};
            int remote_gi[2] = {0, 0};  // k, l
            remote_L2G(remote_li, remote_gi);
            
            double brsum[3] = {0.0, 0.0, 0.0};

            /* Compute Simpson's 3/8 quadrature weight for this index */
            double weight;
            weight = simpsonWeight(remote_gi[0], num_nodes)
                        * simpsonWeight(remote_gi[1], num_nodes);            
            /* We already have N^4 parallelism, so no need to parallelize on 
             * the BR periodic points. Instead we serialize this in each thread
             * and reuse the fetch of the i/j and k/l points */
            for (int kdir = kstart; kdir <= kend; kdir++) {
                for (int ldir = lstart; ldir <= lend; ldir++) {
                    double offset[3] = {0.0, 0.0, 0.0}, br[3];
                    offset[0] = kdir * width[0];
                    offset[1] = ldir * width[1];

                    /* Do the Birkhoff-Rott evaluation for this point */
                    BR(br, z, zremote, wremote, epsilon, dx, dy, weight,
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
    /** Calculate zdot on the original mesh but only with one process
     */
    void calculateSingleCorrectZdot()
    {
        // Setup
        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( single_pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        zdot_correct_ = Cabana::Grid::createArray<double, MemorySpace>(
            "zdot_correct_", node_triple_layout );
        
        auto z = single_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = single_pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());

        auto zdot_ = this->zdot_correct_->view();

        auto local_grid = single_pm_->mesh().localGrid();
        auto local_L2G = Cabana::Grid::IndexConversion::createL2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>(*local_grid, Cabana::Grid::Node());
        auto local_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        
        /* Get an atomic view of the interface velocity, since each k/l point
         * is going to be updating it in parallel */
        Kokkos::View<double ***,
             typename node_view::device_type,
             Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_zdot = zdot_;
    
        /* Zero out all of the i/j points - XXX Is this needed are is this already zeroed somewhere else? */
        Kokkos::parallel_for("Exact BR Zero Loop",
            Cabana::Grid::createExecutionPolicy(local_node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++)
                atomic_zdot(i, j, n) = 0.0;
        });
        
        // Compute forces for all owned nodes on this process
        computeInterfaceVelocityPieceCorrect(single_pm_, single_bc_, atomic_zdot, z, z, w, local_L2G);
    }
    
    template <class pm_bc_type, class zm_type>
    void calculateDistributedZdot(pm_bc_type pm_, zm_type zm_)
    {
        // Get z, w, o views
        auto z = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Position());
        auto w = pm_->get(Cabana::Grid::Node(), Beatnik::Field::Vorticity());
        zm_->prepareOmega(z, w);
        auto o = zm_->getOmega();

        auto node_triple_layout =
            Cabana::Grid::createArrayLayout( pm_->mesh().localGrid(), 3, Cabana::Grid::Node() );
        zdot_test_ = Cabana::Grid::createArray<double, MemorySpace>(
            "zdot_test_", node_triple_layout );
        auto zdot = this->zdot_test_->view();

        int is_periodic = pm_->mesh().is_periodic(); // 1 if periodic
        if (is_periodic)
        {
            this->p_br_exact_->computeInterfaceVelocity(zdot, z, w, o);
        }
        else 
        {
            this->f_br_exact_->computeInterfaceVelocity(zdot, z, w, o);
        }


    }
    template <class pm_bc_type>
    void testZdot(pm_bc_type pm_)
    {
        auto zdot_d_test = this->zdot_test_->view();
        auto zdot_d_correct = this->zdot_correct_->view();

        auto local_grid = pm_->mesh().localGrid();
        auto local_L2G = Cabana::Grid::IndexConversion::createL2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>(*local_grid, Cabana::Grid::Node());

        // Copy views to host memory
        int tdim0 = zdot_d_test.extent(0);
        int tdim1 = zdot_d_test.extent(1);
        int tdim2 = zdot_d_test.extent(2);
        int cdim0 = zdot_d_correct.extent(0);
        int cdim1 = zdot_d_correct.extent(1);
        int cdim2 = zdot_d_correct.extent(2);
        Kokkos::View<double***, Kokkos::HostSpace> zdot_h_test("zdot_h_test", tdim0, tdim1, tdim2);
        Kokkos::View<double***, Kokkos::HostSpace> zdot_h_correct("zdot_h_correct", cdim0, cdim1, cdim2);
        Beatnik::Operators::copy_to_host(zdot_h_test, zdot_d_test);
        Beatnik::Operators::copy_to_host(zdot_h_correct, zdot_d_correct);

        const int halo_width = 2; // XXX - For some reason this->haloWidth_ is not working
        auto own_node_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
        Kokkos::parallel_for( "Check zdot",  
            createExecutionPolicy(own_node_space, Kokkos::DefaultHostExecutionSpace()), 
            KOKKOS_LAMBDA(int k, int l) {

                // We want the global coordinates of the distributed zdot so we
                // can properly index into the single zdot
                int li[2] = {k, l};
                int gi[2] = {0, 0}; // Holds global k, l
                local_L2G(li, gi);
                for (int dim = 0; dim < cdim2; dim++)
                {
                    double zdot_test = zdot_h_test(k, l, dim);
                    double zdot_correct = zdot_h_correct(gi[0]+halo_width, gi[1]+halo_width, dim);
                    EXPECT_DOUBLE_EQ(zdot_test, zdot_correct);
                }
                
        });

    }
};

} // end namespace BeatnikTest

#endif // _TSTEXACT_BRSOLVER_HPP_
