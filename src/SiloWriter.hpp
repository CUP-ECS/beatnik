/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 *
 * @section DESCRIPTION
 * Silo Writer class to write results to a silo file using PMPIO
 */

#ifndef BEATNIK_SILOWRITER_HPP
#define BEATNIK_SILOWRITER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cajita.hpp>

#include <pmpio.h>
#include <silo.h>

namespace Beatnik
{

/**
 * The SiloWriter Class
 * @class SiloWriter
 * @brief SiloWriter class to write results to Silo file using PMPIO
 **/
template <class ExecutionSpace, class MemorySpace>
class SiloWriter
{
  public:
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    /**
     * Constructor
     * Create new SiloWriter
     *
     * @param pm Problem manager object
     */
    template <class ProblemManagerType>
    SiloWriter( ProblemManagerType& pm )
        : _pm( pm )
    {
        if ( DEBUG && _pm->mesh()->rank() == 0 )
            std::cerr << "Created Beatnik SiloWriter\n";
    };

    /**
     * Write File
     * @param dbile File handler to dbfile
     * @param name File name
     * @param time_step Current time step
     * @param time Current tim
     * @param dt Time Step (dt)
     * @brief Writes the locally-owned portion of the mesh/variables to a file
     **/
    void writeFile( DBfile* dbfile, char* meshname, int time_step, double time,
                    double dt )
    {
        // Initialize Variables
        int dims[3];
        double *coords[3], *vars[3];
        const char* coordnames[3] = { "X", "Y", "Z" };
        DBoptlist* optlist;

        // Rertrieve the Local Grid and Local Mesh
        auto local_grid = _pm->mesh()->localGrid();


        // Set DB Options: Time Step, Time Stamp and Delta Time
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );
        int dbcoord = DB_CARTESIAN;
        DBAddOption( optlist, DBOPT_COORDSYS, &dbcoord );
        int dborder = DB_ROWMAJOR;
        DBAddOption( optlist, DBOPT_MAJORORDER, &dborder );

        // Get the size of the local cell space and declare the
        // coordinates of the portion of the mesh we're writing
        auto node_domain = local_grid->indexSpace(
            Cajita::Own(), Cajita::Node(), Cajita::Local() );

        for ( unsigned int i = 0; i < 2; i++ )
        {
            dims[i] = node_domain.extent( i ); // zones (cells) in a dimension
            // spacing[i] = _pm->mesh()->cellSize(); // uniform mesh
        }
        dims[2] = 1;

        // Allocate coordinate arrays in each dimension
        for ( unsigned int i = 0; i < 3; i++ )
        {
            coords[i] = (double*)malloc( sizeof( double ) * dims[0] * dims[1] );
        }

        // Fill out coords[] arrays with coordinate values in each dimension
        for ( unsigned int d = 0; d < 3; d++ )
        {
            for ( int i = node_domain.min( 0 ); i < node_domain.max( 0 ); i++ )
            {
                for ( int j = node_domain.min( 1 ); j < node_domain.max( 1 ); j++ )
                {
		    int iown = i - cell_domain.min( 0 );
                    int jown = j - cell_domain.min( 1 );
                    for ( unsigned int j = 0; j < 3; j++ )
                        index[j] = 0;
                    // Get the location data onto the host and use it to set
                    // mesh coordinates for the curvilinear mesh
                    auto z = _pm->get( Cajita::Node(), Field::Position() );
                    auto zHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), z );
	
                    coords[d][iown * node_domain.extent(0) + jown ] = zHost(i, j, d);
		}
            }
        }

        DBPutQuadmesh( dbfile, meshname, (DBCAS_t)coordnames, coords, dims,
                       2, DB_DOUBLE, DB_NONCOLLINEAR, optlist );

        // Now we write the individual variables associated with this
        // portion of the mesh, potentially copying them out of device space
        // and making sure not to write ghost values.

        // Advected quantity first - copy owned portion from the primary
        // execution space to the host execution space
        auto w =
            _pm->get( Cajita::Node(), Field::Vorticity() );
        auto xmin = node_domain.min( 0 );
        auto ymin = node_domain.min( 1 );

        // Silo is expecting row-major data so we make this a LayoutRight
        // array that we copy data into and then get a mirror view of.
        // XXX WHY DOES THIS ONLY WORK LAYOUTLEFT?
        Kokkos::View<typename pm_type::node_array::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::node_array::device_type>
            w1Owned( "w1owned", node_domain.extent( 0 ), node_domain.extent( 1 ),
                    1 );
        Kokkos::View<typename pm_type::node_array::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::node_array::device_type>
            w2Owned( "w2owned", node_domain.extent( 0 ), node_domain.extent( 1 ),
                    1 );
        Kokkos::parallel_for(
            "SiloWriter::wowned copy",
            createExecutionPolicy( node_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                w1Owned( i - xmin, j - ymin, 0 ) = w( i, j, 0 );
                w2Owned( i - xmin, j - ymin, 1 ) = w( i, j, 1 );
            } );
        auto w1Host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), w1Owned );
        auto w2Host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), w2Owned );

        vars[0] = uHost.data();
        vars[1] = vHost.data();

        char *varnames[2] = {"w1", "w2"};
        DBPutQuadvar( dbfile, "vorticity", meshname, 2, (DBCAS_t)varnames,
                      vars, dims, 2, NULL, 0, DB_DOUBLE, DB_NODECENT,
                      optlist );

        for ( unsigned int i = 0; i < 2; i++ )
        {
            free( coords[i] );
        }

        // Free Option List
        DBFreeOptlist( optlist ); 
    };

    /**
     * Create New Silo File for Current Time Step and Owning Group
     * @param filename Name of file
     * @param nsname Name of directory inside of the file
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void* createSiloFile( const char* filename, const char* nsname,
                                 void* user_data )
    {

        int driver = *( (int*)user_data );

        DBfile* silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL,
                                      "BeatnikRaw", driver );

        if ( silo_file )
        {
            DBMkDir( silo_file, nsname );
            DBSetDir( silo_file, nsname );
        }


        return (void*)silo_file;
    };

    /**
     * Open Silo File
     * @param filename Name of file
     * @param nsname Name of directory inside of file
     * @param ioMode Read/Write/Append Mode
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void* openSiloFile( const char* filename, const char* nsname,
                               PMPIO_iomode_t ioMode,
                               [[maybe_unused]] void* user_data )
    {
        DBfile* silo_file = DBOpen(
            filename, DB_UNKNOWN, ioMode == PMPIO_WRITE ? DB_APPEND : DB_READ );

        if ( silo_file )
        {
            if ( ioMode == PMPIO_WRITE )
            {
                DBMkDir( silo_file, nsname );
            }
            DBSetDir( silo_file, nsname );
        }
        return (void*)silo_file;
    };

    /**
     * Close Silo File
     * @param file File pointer
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void closeSiloFile( void* file, [[maybe_unused]] void* user_data )
    {
        DBfile* silo_file = (DBfile*)file;
        if ( silo_file )
            DBClose( silo_file );
    };

    /**
     * Write Multi Object Silo File the References Child Files in order to
     * have entire set of data for the time step writen by each rank in
     * a single logical file
     *
     * @param silo_file Pointer to the Silo File
     * @param baton Baton object from PMPIO
     * @param size Number of Ranks
     * @param time_step Current time step
     * @param file_ext File extension (PDB, HDF5)
     **/
    void writeMultiObjects( DBfile* silo_file, PMPIO_baton_t* baton, int size,
                            int time_step, const char* file_ext )
    {
        char** mesh_block_names = (char**)malloc( size * sizeof( char* ) );
        char** q_block_names = (char**)malloc( size * sizeof( char* ) );
        char** v_block_names = (char**)malloc( size * sizeof( char* ) );

        int* block_types = (int*)malloc( size * sizeof( int ) );
        int* var_types = (int*)malloc( size * sizeof( int ) );

        DBSetDir( silo_file, "/" );

        for ( int i = 0; i < size; i++ )
        {
            int group_rank = PMPIO_GroupRank( baton, i );
            mesh_block_names[i] = (char*)malloc( 1024 );
            q_block_names[i] = (char*)malloc( 1024 );
            v_block_names[i] = (char*)malloc( 1024 );

            sprintf( mesh_block_names[i],
                     "raw/BeatnikOutput%05d%05d.%s:/domain_%05d/Mesh",
                     group_rank, time_step, file_ext, i );
            sprintf( q_block_names[i],
                     "raw/BeatnikOutput%05d%05d.%s:/domain_%05d/quantity",
                     group_rank, time_step, file_ext, i );
            sprintf( v_block_names[i],
                     "raw/BeatnikOutput%05d%05d.%s:/domain_%05d/velocity",
                     group_rank, time_step, file_ext, i );
            block_types[i] = DB_QUADMESH;
            var_types[i] = DB_QUADVAR;
        }

        DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names,
                        block_types, 0 );
        DBPutMultivar( silo_file, "multi_quantity", size, q_block_names,
                       var_types, 0 );
        DBPutMultivar( silo_file, "multi_velocity", size, v_block_names,
                       var_types, 0 );
        for ( int i = 0; i < size; i++ )
        {
            free( mesh_block_names[i] );
            free( q_block_names[i] );
            free( v_block_names[i] );
        }

        free( mesh_block_names );
        free( q_block_names );
        free( v_block_names );
        free( block_types );
        free( var_types );
    }

    // Function to Create New DB File for Current Time Step
    /**
     * Createe New DB File for Current Time Step
     * @param name Name of directory in silo file
     * @param time_step Current time step
     * @param time Current time
     * @param dt Time step (dt)
     **/
    void siloWrite( char* name, int time_step, double time, double dt )
    {
        // Initalize Variables
        DBfile* silo_file;
        DBfile* master_file;
        int size;
        int driver = DB_PDB;
        const char* file_ext = "pdb";
        // TODO: Make the Number of Groups a Constant or a Runtime Parameter (
        // Between 8 and 64 )
        int numGroups = 2;
        char masterfilename[256], filename[256], nsname[256];
        PMPIO_baton_t* baton;


        MPI_Comm_size( MPI_COMM_WORLD, &size );
        MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

        baton =
            PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1,
                        createSiloFile, openSiloFile, closeSiloFile, &driver );

        // Set Filename to Reflect TimeStep
        sprintf( masterfilename, "data/Beatnik%05d.%s", time_step,
                 file_ext );
        sprintf( filename, "data/raw/BeatnikOutput%05d%05d.%s",
                 PMPIO_GroupRank( baton, _pm->mesh()->rank() ), time_step,
                 file_ext );
        sprintf( nsname, "domain_%05d", _pm->mesh()->rank() );

        // Show Errors and Force FLoating Point
        DBShowErrors( DB_ALL, NULL );

        silo_file = (DBfile*)PMPIO_WaitForBaton( baton, filename, nsname );

        writeFile( silo_file, name, time_step, time, dt );
        if ( _pm->mesh()->rank() == 0 )
        {
            master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL,
                                    "Beatnik", driver );
            writeMultiObjects( master_file, baton, size, time_step, "pdb" );
            DBClose( master_file );
        }
    
        PMPIO_HandOffBaton( baton, silo_file );

        PMPIO_Finish( baton );

    }
        
  private:
    std::shared_ptr<pm_type> _pm;
}; 

}; // namespace Beatnik
#endif
