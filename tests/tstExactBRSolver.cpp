#ifndef TSTEXACTBRSOLVER_HPP
#define TSTEXACTBRSOLVER_HPP

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};
enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};
enum BoundaryType {PERIODIC = 0, FREE = 1};

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain */
    enum InitialConditionModel initial_condition; /**< Model used to set initial conditions */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of scale of initial interface */
    double variation; /**< Variation in scale of initial interface */
    double period;   /**< Period of initial variation in interface */
    enum Beatnik::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */
    int model;      /**< Model used to set initial conditions */

    /* Problem simulation parameters */
    std::array<int, 2> num_nodes;          /**< Number of cells */
    double t_final;     /**< Ending time */
    double delta_t;     /**< Timestep */
    std::string driver; /**< ( Serial, Threads, OpenMP, CUDA ) */
    int weak_scale;     /**< Amount to scale up resulting problem */

    /* I/O parameters */
    char *indir;        /**< Where to read initial conditions from */
    char *outdir;       /**< Directory to write output to */
    int write_freq;     /**< Write frequency */

    /* Solution method constants */
    enum SolverOrder order;  /**< Order of z-model solver to use */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */
};

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    // Initialize Variables

    MeshInitFunc( std::array<double, 6> box, enum InitialConditionModel i,
                  double t, double m, double v, double p, 
                  const std::array<int, 2> nodes, enum Beatnik::BoundaryType boundary )
        : _i(i)
        , _t( t )
        , _m( m )
        , _v( v)
        , _p( p )
        , _b( boundary )
    {
	_ncells[0] = nodes[0] - 1;
        _ncells[1] = nodes[1] - 1;

        _dx = (box[3] - box[0]) / _ncells[0];
        _dy = (box[4] - box[1]) / _ncells[1];
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Position,
                     [[maybe_unused]] const int index[2],
                     const double coord[2],
                     double &z1, double &z2, double &z3) const
    {
        double lcoord[2];
        /* Compute the physical position of the interface from its global
         * coordinate in mesh space */
        for (int i = 0; i < 2; i++) {
            lcoord[i] = coord[i];
            if (_b == BoundaryType::FREE && (_ncells[i] % 2 == 1) ) {
                lcoord[i] += 0.5;
            }
        }
        z1 = _dx * lcoord[0];
        z2 = _dy * lcoord[1];

        // We don't currently support tilting the initial interface
        switch (_i) {
        case IC_COS:
            z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
            break;
        case IC_SECH2:
            z3 = _m * pow(1.0 / cosh(_p * (z1 * z1 + z2 * z2)), 2);
            break;
        case IC_RANDOM:
            /* XXX Use p to seed the random number generator XXX */
            /* Also need to use the Kokkos random number generator, not
             * drand48 */
            // z3 = _m * (2*drand48() - 1.0);
            break;
        case IC_GAUSSIAN:
        case IC_FILE:
            break;
        }
        return true;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Vorticity,
                     [[maybe_unused]] const int index[2],
                     [[maybe_unused]] const double coord[2],
                     double& w1, double &w2 ) const
    {
        // Initial vorticity along the interface is 0.
        w1 = 0; w2 = 0;
        return true;
    };
    enum InitialConditionModel _i;
    double _t, _m, _v, _p;
    Kokkos::Array<int, 3> _ncells;
    double _dx, _dy;
    enum Beatnik::BoundaryType _b;
};

class TestExactBRSolver : public testing::Test
{
  protected:
    
    void SetUp() override
    {
      MPI_Comm_rank( MPI_COMM_WORLD, &_rank );      
      MPI_Comm_size( MPI_COMM_WORLD, &_comm_size );
    }

    int _rank, _comm_size;
};

TEST_F(TestExactBRSolver, test1)
{ 
    ClArgs cl;

    // Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.order = SolverOrder::ORDER_LOW;
    cl.weak_scale = 1;
    cl.write_freq = 10;

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 128, 128 };
    cl.initial_condition = IC_COS;
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.variation = 0.00;
    cl.period = 1.0;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the solver to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = 10.0;

    /* Physical setup of problem */
    cl.global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes[i] *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.order == SolverOrder::ORDER_HIGH) {
            cl.delta_t = tau/50.0;  // Should this depend on dx and dy? XXX
        } else {
            cl.delta_t = tau/25.0;
        }
    }

    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    Beatnik::BoundaryCondition bc;
    for (int i = 0; i < 6; i++)
        bc.bounding_box[i] = cl.global_bounding_box[i];
    bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

    MeshInitFunc initializer( cl.global_bounding_box, cl.initial_condition,
                              cl.tilt, cl.magnitude, cl.variation, cl.period,
                              cl.num_nodes, cl.boundary );

    std::shared_ptr<Beatnik::SolverBase> solver;

    solver = Beatnik::createSolver(
            "serial", MPI_COMM_WORLD,
            cl.global_bounding_box, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t );

    solver->solve( cl.t_final, cl.write_freq );
    printf("in test");
    EXPECT_DOUBLE_EQ(1.0, 1.000000000001);
    ASSERT_EQ(6, 7);
}

int main( int argc, char* argv[] )
{
    ::testing::InitGoogleTest( &argc, argv );
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    int return_val = RUN_ALL_TESTS();
    Kokkos::finalize();
    MPI_Finalize();
    return return_val;
}


#endif // TSTEXACTBRSOLVER_HPP
