#ifndef _TEST_ROCKETRIG_HPP_
#define _TEST_ROCKETRIG_HPP_

#include <Solver.hpp>


namespace BeatnikTest
{

enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};
enum SolverOrder {ORDER_LOW = 0, ORDER_MEDIUM, ORDER_HIGH};

/**
 * @struct ClArgs
 * @brief Template struct to organize and keep track of parameters controlled by
 * command line arguments
 */
struct ClArgs
{
    /* Problem physical setup */
    // std::array<double, 6> global_bounding_box;    /**< Size of initial spatial domain: MOVED TO PARAMS */
    enum InitialConditionModel initial_condition; /**< Model used to set initial conditions */
    double tilt;    /**< Initial tilt of interface */
    double magnitude;/**< Magnitude of scale of initial interface */
    double variation; /**< Variation in scale of initial interface */
    double period;   /**< Period of initial variation in interface */
    enum Beatnik::BoundaryType boundary;  /**< Type of boundary conditions */
    double gravity; /**< Gravitational accelaration in -Z direction in Gs */
    double atwood;  /**< Atwood pressure differential number */
    int model;      /**< Model used to set initial conditions */
    double bounding_box; /**< Size of global bounding box. From (-B, -B, -B) to (B, B, B) */

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
    enum Beatnik::BRSolverType br_solver; /**< BRSolver to use */
    double mu;      /**< Artificial viscosity constant */
    double eps;     /**< Desingularization constant */

    /* Parameters specific to solver order and BR solver type:
     *  - Period for particle initialization
     *  - Global bounding box
     *  - Periodicity
     *  - Heffte configuration (For low-order solver)
     *  - solver order (Order of z-model solver to use)
     *  - BR solver type
     *  - Cutoff distance (If using cutoff solver)
     */
    Beatnik::Params params;
};

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

    template <class RandNumGenType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Position,
                     RandNumGenType random_pool,
                     [[maybe_unused]] const int index[2],
                     const double coord[2],
                     double &z1, double &z2, double &z3) const
    {
        double lcoord[2];
        /* Compute the physical position of the interface from its global
         * coordinate in mesh space */
        for (int i = 0; i < 2; i++) {
            lcoord[i] = coord[i];
            if (_b == Beatnik::BoundaryType::FREE && (_ncells[i] % 2 == 1) ) {
                lcoord[i] += 0.5;
            }
        }
        z1 = _dx * lcoord[0];
        z2 = _dy * lcoord[1];

        // We don't currently support tilting the initial interface

        /* Need to initialize these values here to avoid "jump to case label "case IC_FILE:"
         * crosses initialization of ‘double gaussian’, etc." errors */
        auto generator = random_pool.get_state();
        double rand_num = generator.drand(-1.0, 1.0);
        double mean = 0.0;
        double std_dev = 1.0;
        double gaussian = (1 / (std_dev * Kokkos::sqrt(2 * Kokkos::numbers::pi_v<double>))) *
            Kokkos::exp(-0.5 * Kokkos::pow(((rand_num - mean) / std_dev), 2));
        switch (_i) {
        case IC_COS:
            z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
            break;
        case IC_SECH2:
            z3 = _m * pow(1.0 / cosh(_p * (z1 * z1 + z2 * z2)), 2);
            break;
        case IC_RANDOM:
            z3 = _m * (2*rand_num - 1.0);
            break;
        case IC_GAUSSIAN:
            /* The built-in C++ std::normal_distribution<double> doesn't
             * work here, so coding the gaussian distribution itself.
             */
            z3 = _m * gaussian;
            break;
        case IC_FILE:
            break;
        }
        
        random_pool.free_state(generator);

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

int init_cl_args( ClArgs& cl )
{
    /// Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.weak_scale = 1;
    cl.write_freq = 0;

    // Set default extra parameters
    cl.params.cutoff_distance = 0.5;
    cl.params.heffte_configuration = 6;
    cl.params.br_solver = Beatnik::BR_EXACT;
    cl.params.solver_order = SolverOrder::ORDER_LOW;
    // cl.params.period below

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 64, 64 };
    cl.bounding_box = 1.0;
    cl.initial_condition = IC_COS;
    cl.boundary = Beatnik::BoundaryType::PERIODIC;
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.variation = 0.00;
    cl.params.period = 1.0;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the solver to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = 5;

    /* Physical setup of problem */
    cl.params.global_bounding_box = {cl.bounding_box * -1.0,
                                     cl.bounding_box * -1.0, 
                                     cl.bounding_box * -1.0,
                                     cl.bounding_box,
                                     cl.bounding_box,
                                     cl.bounding_box};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.params.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes[i] *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
            cl.delta_t = tau/50.0;  // Should this depend on dx and dy? XXX
        } else {
            cl.delta_t = tau/25.0;
        }
    }

    if (cl.t_final <= 0.0) {
        cl.t_final = tau * 2.0; // Simulate for 2 characterisic periods, which is all
                                // the low-order model can really handle
    }
    else {
        cl.t_final = cl.t_final * cl.delta_t;
    }

    // Return Successfully
    return 0;
}

class Rocketrig
{
    using solver_type = std::shared_ptr<Beatnik::SolverBase>;
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;

  public:
    Rocketrig( ClArgs& cl ) : _cl( cl ) {};

    void rocketrig()
    {
        int comm_size, rank;                         // Initialize Variables
        MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank
        ClArgs cl = _cl;

        Cabana::Grid::DimBlockPartitioner<2> partitioner; // Create Cabana::Grid Partitioner
        Beatnik::BoundaryCondition bc;
        for (int i = 0; i < 6; i++)
        {
            bc.bounding_box[i] = cl.params.global_bounding_box[i];
            
        }
        bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};
        /* MeshInitFunc( std::array<double, 6> box, enum InitialConditionModel i,
                  double t, double m, double v, double p, 
                  const std::array<int, 2> nodes, enum Beatnik::BoundaryType boundary ) 
        */
        MeshInitFunc initializer( cl.params.global_bounding_box, cl.initial_condition,
                                cl.tilt, cl.magnitude, cl.variation, cl.params.period,
                                cl.num_nodes, cl.boundary );

        if (cl.params.solver_order == SolverOrder::ORDER_LOW) {
            _solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Beatnik::Order::Low(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else if (cl.params.solver_order == SolverOrder::ORDER_MEDIUM) {
            _solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Beatnik::Order::Medium(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
            _solver = Beatnik::createSolver(
                cl.driver, MPI_COMM_WORLD, cl.num_nodes,
                partitioner, cl.atwood, cl.gravity, initializer,
                bc, Beatnik::Order::High(), cl.mu, cl.eps, cl.delta_t,
                cl.params );
        } else {
            std::cerr << "Invalid Model Order parameter!\n";
            Kokkos::finalize(); 
            MPI_Finalize(); 
            exit( -1 );  

        }

        // Solve
        _solver->solve( cl.t_final, cl.write_freq );
    }

    View_t get_positions()
    {
        return _solver->get_positions();
    }

    View_t get_vorticities()
    {
        return _solver->get_vorticities();
    }

    ClArgs get_ClArgs()
    {
        return _cl;
    }

  private:
    solver_type _solver;
    ClArgs _cl;
}; // class rocketrig


} // end namespace BeantikTest

#endif // _TEST_ROCKETRIG_HPP_
