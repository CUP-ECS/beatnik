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
        MeshInitFunc single_MeshInitFunc_(this->globalBoundingBox_, this->tilt_, this->m_, this->v_, this->p_, this->globalNumNodes_, Beatnik::BoundaryType::FREE);
        this->single_mesh_ = std::make_shared<surface_mesh_type>( this->globalBoundingBox_, this->globalNumNodes_, single_params_.periodic, 
                                this->partitioner_, this->haloWidth_, comm_single_ );
        this->single_pm_ = std::make_shared<pm_type>( *single_mesh_, single_bc_, this->p_, single_MeshInitFunc_ );
        this->p_br_exact_ = std::make_shared<br_exact_type>(*single_pm_, single_bc_, this->epsilon_, this->dx_, this->dy_, single_params_);

        // XXX - Make initial vorticity (w) non-zero
    }

  public:
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
        double epsilon = this->epsilon_;
        double dx = this->dx_, dy = this->dy_;

        // Mesh dimensions for Simpson weight calc
        int num_nodes = single_pm_->mesh().get_surface_mesh_size();
        int sr = single_rank_;
        /* Now loop over the cross product of all the node on the interface */
        auto pair_space = Beatnik::Operators::crossIndexSpace(local_node_space, local_node_space);
        Kokkos::parallel_for("Exact BR Force Loop",
            Cabana::Grid::createExecutionPolicy(pair_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j, int k, int l) {

            // printf("R%d: ijlk: %d, %d, %d, %d\n", sr, i, j, k, l);

            // We need the global indicies of the (k, l) point for Simpson's weight
            int remote_gi[2] = {k, l};  // k, l
            
            
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
                    BR(br, z, z, w, epsilon, dx, dy, weight,
                                  i, j, k, l, offset);
                    printf("esp: %0.3lf, dx: %0.3lf, dy: %0.3lf, weight: %0.3lf, offset: %0.5lf, %0.5lf, br: %0.13lf, %0.13lf, %0.13lf\n",
                        epsilon, dx, dy, weight, offset[0], offset[1], br[0], br[1], br[2]);
                    for (int d = 0; d < 3; d++) {
                        brsum[d] += br[d];
                    }
                }
            }

            /* Add it its contribution to the integral */
            for (int n = 0; n < 3; n++) {
                atomic_zdot(i, j, n) += brsum[n];
            }
            //printf("R%d: br: %05.lf, %0.5lf, %05.lf\n", sr, brsum[0], brsum[1], brsum[2]);
        });
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
                printf("R%d: l: %d, %d, g: %d, %d, ztest: %05.lf, %0.5lf, %05.lf, zcor: %05.lf, %0.5lf, %05.lf\n",
                    rank_, k, l, gi[0]+halo_width, gi[1]+halo_width, zdot_h_test(k, l, 0), zdot_h_test(k, l, 1), zdot_h_test(k, l, 2),
                    zdot_h_correct(gi[0]+halo_width, gi[1]+halo_width, 0), zdot_h_correct(gi[0]+halo_width, gi[1]+halo_width, 1), zdot_h_correct(gi[0]+halo_width, gi[1]+halo_width, 2));
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
