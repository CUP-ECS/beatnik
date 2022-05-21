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
template <std::size_t Dims, class ExecutionSpace, class MemorySpace>
class SiloWriter
{
  public:
    using pm_type = ProblemManager<Dims, ExecutionSpace, MemorySpace>;
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
            std::cerr << "Created CajitaFluids SiloWriter\n";
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
        int dims[Dims], zdims[Dims];
        double *coords[Dims], *vars[Dims];
        // double *spacing[Dims];
        const char* coordnames[3] = { "X", "Y", "Z" };
        DBoptlist* optlist;

        // Rertrieve the Local Grid and Local Mesh
        auto local_grid = _pm->mesh()->localGrid();
        auto local_mesh = *( _pm->mesh()->localMesh() );

        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile" );

        // Set DB Options: Time Step, Time Stamp and Delta Time
        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::SetupOptions" );
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );
        int dbcoord = DB_CARTESIAN;
        DBAddOption( optlist, DBOPT_COORDSYS, &dbcoord );
        int dborder = DB_ROWMAJOR;
        DBAddOption( optlist, DBOPT_MAJORORDER, &dborder );
        Kokkos::Profiling::popRegion();

        // Get the size of the local cell space and declare the
        // coordinates of the portion of the mesh we're writing
        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::WriteMesh" );
        auto cell_domain = local_grid->indexSpace(
            Cajita::Own(), Cajita::Cell(), Cajita::Local() );

        for ( unsigned int i = 0; i < Dims; i++ )
        {
            zdims[i] = cell_domain.extent( i ); // zones (cells) in a dimension
            dims[i] = zdims[i] + 1;             // nodes in a dimension
            // spacing[i] = _pm->mesh()->cellSize(); // uniform mesh
        }

        // Allocate coordinate arrays in each dimension
        for ( unsigned int i = 0; i < Dims; i++ )
        {
            coords[i] = (double*)malloc( sizeof( double ) * dims[i] );
        }

        // Fill out coords[] arrays with coordinate values in each dimension
        for ( unsigned int d = 0; d < Dims; d++ )
        {
            for ( int i = cell_domain.min( d ); i < cell_domain.max( d ) + 1;
                  i++ )
            {
                int iown = i - cell_domain.min( d );
                int index[Dims];
                double location[Dims];
                for ( unsigned int j = 0; j < Dims; j++ )
                    index[j] = 0;
                index[d] = i;
                local_mesh.coordinates( Cajita::Node(), index, location );
                coords[d][iown] = location[d];
            }
        }

        DBPutQuadmesh( dbfile, meshname, (DBCAS_t)coordnames, coords, dims,
                       Dims, DB_DOUBLE, DB_COLLINEAR, optlist );
        Kokkos::Profiling::popRegion();

        // Now we write the individual variables associated with this
        // portion of the mesh, potentially copying them out of device space
        // and making sure not to write ghost values.

        // Advected quantity first - copy owned portion from the primary
        // execution space to the host execution space
        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::WriteQuantity" );
        auto q =
            _pm->get( Cajita::Cell(), Field::Quantity(), Version::Current() );
        auto xmin = cell_domain.min( 0 );
        auto ymin = cell_domain.min( 1 );

        // Silo is expecting row-major data so we make this a LayoutRight
        // array that we copy data into and then get a mirror view of.
        // XXX WHY DOES THIS ONLY WORK LAYOUTLEFT?
        Kokkos::View<typename pm_type::cell_array::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::cell_array::device_type>
            qOwned( "qowned", cell_domain.extent( 0 ), cell_domain.extent( 1 ),
                    1 );
        Kokkos::parallel_for(
            "SiloWriter::qowned copy",
            createExecutionPolicy( cell_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                qOwned( i - xmin, j - ymin, 0 ) = q( i, j, 0 );
            } );
        auto qHost =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qOwned );

        DBPutQuadvar1( dbfile, "quantity", meshname, qHost.data(), zdims, Dims,
                       NULL, 0, DB_DOUBLE, DB_ZONECENT, optlist );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::WriteFile::WriteVelocity" );
        auto u = _pm->get( Cajita::Face<Cajita::Dim::I>(), Field::Velocity(),
                           Version::Current() );
        auto v = _pm->get( Cajita::Face<Cajita::Dim::J>(), Field::Velocity(),
                           Version::Current() );

        /* Because VisIt and all the other things that read Silo files
         * can't currently show edge velocity magnitudes, we
         * instead interpolate to cell-centered velcity vectors for I/O */
        // XXX Why does this only work LayoutLeft???
        Kokkos::View<typename pm_type::cell_array::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::cell_array::device_type>
            uOwned( "uOwned", cell_domain.extent( 0 ), cell_domain.extent( 1 ),
                    1 );
        Kokkos::View<typename pm_type::cell_array::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::cell_array::device_type>
            vOwned( "vOwned", cell_domain.extent( 0 ), cell_domain.extent( 1 ),
                    1 );
        Kokkos::parallel_for(
            "SiloWriter::velocity interpolate and copy",
            createExecutionPolicy( cell_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                int idx[2] = { i, j };
                double loc[2], velocity[2];
                local_mesh.coordinates( Cajita::Cell(), idx, loc );
                Interpolation::interpolateVelocity<2, 1>( loc, local_mesh, u, v,
                                                          velocity );
                uOwned( i - xmin, j - ymin, 0 ) = velocity[0];
                vOwned( i - xmin, j - ymin, 0 ) = velocity[1];
            } );
        auto uHost =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), uOwned );
        auto vHost =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), vOwned );
        const char* varnames[3] = { "u", "v", "w" };
        vars[0] = uHost.data();
        vars[1] = vHost.data();
        DBPutQuadvar( dbfile, "velocity", meshname, Dims, (DBCAS_t)varnames,
                      vars, zdims, Dims, NULL, 0, DB_DOUBLE, DB_ZONECENT,
                      optlist );
        Kokkos::Profiling::popRegion();

        for ( unsigned int i = 0; i < Dims; i++ )
        {
            free( coords[i] );
        }

        // Free Option List
        DBFreeOptlist( optlist );
        Kokkos::Profiling::popRegion(); // writeFile region
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
        Kokkos::Profiling::pushRegion( "SiloWriter::CreateSiloFile" );

        DBfile* silo_file = DBCreate( filename, DB_CLOBBER, DB_LOCAL,
                                      "CajitaFluidsRaw", driver );

        if ( silo_file )
        {
            DBMkDir( silo_file, nsname );
            DBSetDir( silo_file, nsname );
        }

        Kokkos::Profiling::popRegion();

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
        Kokkos::Profiling::pushRegion( "SiloWriter::openSiloFile" );
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
        Kokkos::Profiling::popRegion();
        return (void*)silo_file;
    };

    /**
     * Close Silo File
     * @param file File pointer
     * @param user_data File Driver/Type (PDB, HDF5)
     **/
    static void closeSiloFile( void* file, [[maybe_unused]] void* user_data )
    {
        Kokkos::Profiling::pushRegion( "SiloWriter::closeSiloFile" );
        DBfile* silo_file = (DBfile*)file;
        if ( silo_file )
            DBClose( silo_file );
        Kokkos::Profiling::popRegion();
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
        Kokkos::Profiling::pushRegion( "SiloWriter::writeMultiObjects" );
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
                     "raw/CajitaFluidsOutput%05d%05d.%s:/domain_%05d/Mesh",
                     group_rank, time_step, file_ext, i );
            sprintf( q_block_names[i],
                     "raw/CajitaFluidsOutput%05d%05d.%s:/domain_%05d/quantity",
                     group_rank, time_step, file_ext, i );
            sprintf( v_block_names[i],
                     "raw/CajitaFluidsOutput%05d%05d.%s:/domain_%05d/velocity",
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
        Kokkos::Profiling::popRegion();
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

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite" );

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::Setup" );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

        baton =
            PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1,
                        createSiloFile, openSiloFile, closeSiloFile, &driver );

        // Set Filename to Reflect TimeStep
        sprintf( masterfilename, "data/CajitaFluids%05d.%s", time_step,
                 file_ext );
        sprintf( filename, "data/raw/CajitaFluidsOutput%05d%05d.%s",
                 PMPIO_GroupRank( baton, _pm->mesh()->rank() ), time_step,
                 file_ext );
        sprintf( nsname, "domain_%05d", _pm->mesh()->rank() );

        // Show Errors and Force FLoating Point
        DBShowErrors( DB_ALL, NULL );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::batonWait" );
        silo_file = (DBfile*)PMPIO_WaitForBaton( baton, filename, nsname );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::writeState" );
        writeFile( silo_file, name, time_step, time, dt );
        if ( _pm->mesh()->rank() == 0 )
        {
            master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL,
                                    "CajitaFluids", driver );
            writeMultiObjects( master_file, baton, size, time_step, "pdb" );
            DBClose( master_file );
        }
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::batonHandoff" );
        PMPIO_HandOffBaton( baton, silo_file );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion( "SiloWriter::siloWrite::finish" );
        PMPIO_Finish( baton );
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::popRegion(); // siloWrite
    }

  private:
    std::shared_ptr<pm_type> _pm;
};

}; // namespace CajitaFluids
#endif
