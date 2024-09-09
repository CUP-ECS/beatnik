

static char* shortargs = (char*)"n:B:t:d:x:F:o:c:H:I:b:g:a:T:m:v:p:i:w:O:S:M:e:h:";

static option longargs[] = {
    // Basic simulation parameters
    { "nodes", required_argument, NULL, 'n' },
    { "bounding_box", required_argument, NULL, 'B'},
    { "timesteps", required_argument, NULL, 't' },
    { "delta_t", required_argument, NULL, 'd' },
    { "driver", required_argument, NULL, 'x' },
    { "write_frequency", required_argument, NULL, 'F' },
    { "outdir", required_argument, NULL, 'o' },
    { "cutoff_distance", required_argument, NULL, 'c' },
    { "heffte_configuration", required_argument, NULL, 'H'},

    // Z-model simulation parameters
    { "initial_condition", required_argument, NULL, 'I' },
    { "boundary", required_argument, NULL, 'b' },
    { "gravity", required_argument, NULL, 'g' },
    { "atwood", required_argument, NULL, 'a' },
    { "tilt", required_argument, NULL, 'T' },
    { "magnitude", required_argument, NULL, 'm' },
    { "variation", required_argument, NULL, 'v' },
    { "period", required_argument, NULL, 'p' },
    { "indir", required_argument, NULL, 'i' },
    { "weak-scale", required_argument, NULL, 'w'},

    // Z-model simulation parameters
    { "order", required_argument, NULL, 'O' },
    { "solver", required_argument, NULL, 'S' },
    { "mu", required_argument, NULL, 'M' },
    { "epsilon", required_argument, NULL, 'e' },

    // Miscellaneous other arguments
    { "help", no_argument, NULL, 'h' },
    { 0, 0, 0, 0 } };

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
    enum BRSolverType br_solver; /**< BRSolver to use */
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

/**
 * Parses command line input and updates the command line variables
 * accordingly.
 * @param rank The rank calling the function
 * @param argc Number of command line options passed to program
 * @param argv List of command line options passed to program
 * @param cl Command line arguments structure to store options
 * @return Error status
 */
int parseInput( const int rank, const int argc, char** argv, ClArgs& cl )
{
    signed char ch;

    /// Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.weak_scale = 1;
    cl.write_freq = 10;

    // Set default extra parameters
    cl.params.cutoff_distance = 0.5;
    cl.params.heffte_configuration = 6;
    cl.params.br_solver = BR_EXACT;
    cl.params.solver_order = SolverOrder::ORDER_LOW;
    // cl.params.period below

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 128, 128 };
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
    cl.t_final = -1.0;

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

// Create Solver and Run
void rocketrig( ClArgs& cl )
{
    int comm_size, rank;                         // Initialize Variables
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // Get My Rank

    Cabana::Grid::DimBlockPartitioner<2> partitioner; // Create Cabana::Grid Partitioner
    Beatnik::BoundaryCondition bc;
    for (int i = 0; i < 6; i++)
    {
        bc.bounding_box[i] = cl.params.global_bounding_box[i];
        
    }
    bc.boundary_type = {cl.boundary, cl.boundary, cl.boundary, cl.boundary};

    MeshInitFunc initializer( cl.params.global_bounding_box, cl.initial_condition,
                              cl.tilt, cl.magnitude, cl.variation, cl.params.period,
                              cl.num_nodes, cl.boundary );

    std::shared_ptr<Beatnik::SolverBase> solver;
    if (cl.params.solver_order == SolverOrder::ORDER_LOW) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Low(), cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else if (cl.params.solver_order == SolverOrder::ORDER_MEDIUM) {
        solver = Beatnik::createSolver(
            cl.driver, MPI_COMM_WORLD, cl.num_nodes,
            partitioner, cl.atwood, cl.gravity, initializer,
            bc, Beatnik::Order::Medium(), cl.mu, cl.eps, cl.delta_t,
            cl.params );
    } else if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
        solver = Beatnik::createSolver(
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
    solver->solve( cl.t_final, cl.write_freq );
}
