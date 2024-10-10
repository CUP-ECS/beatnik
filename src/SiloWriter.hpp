/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik authors                          *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
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
#include <Cabana_Grid.hpp>

#include <pmpio.h>
#include <silo.h>
#include <sys/stat.h>

namespace Beatnik
{

/**
 * The SiloWriter Class
 * XXX - currently specific to structured grids
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
        if ( DEBUG && _pm.mesh().rank() == 0 )
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
        double *coords[3], *vars[2];
        const char* coordnames[3] = { "X", "Y", "Z" };
        DBoptlist* optlist;
        int rank = _pm.mesh().rank();

        // Retrieve the Local Grid and Local Mesh
        const auto & local_grid = _pm.mesh().localGrid();

        // Set DB Options: Time Step, Time Stamp and Delta Time
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );
        int dbcoord = DB_CARTESIAN;
        DBAddOption( optlist, DBOPT_COORDSYS, &dbcoord );
        int dborder = DB_ROWMAJOR;
        DBAddOption( optlist, DBOPT_MAJORORDER, &dborder );
        int dbtopo = 2;
        DBAddOption( optlist, DBOPT_TOPO_DIM, &dbtopo );

        // Declare the coordinates of the portion of the mesh we're writing
        auto node_domain = local_grid->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );

        for ( unsigned int i = 0; i < 2; i++ )
        {
            dims[i] = node_domain.extent( i ); 
        }
        dims[2] = 1;

	if (DEBUG) {
            std::cout << "Rank " << rank << ": "
                      << "Writing " << dims[0] << " by " << dims[1] 
                      << " mesh element.\n";
        }

        // Allocate coordinate arrays in each dimension
        for ( unsigned int i = 0; i < 3; i++ )
        {
            coords[i] = (double*)malloc( sizeof( double ) * dims[0] * dims[1]);
        }

        // Fill out coords[] arrays with coordinate values in each dimension
        auto z = _pm.get( Field::Position() )->array()->view();
        auto zHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), z );
        auto xmin = node_domain.min( 0 );
        auto ymin = node_domain.min( 1 );
        for ( int i = node_domain.min( 0 ); i < node_domain.max( 0 ); i++ )
        {
            for ( int j = node_domain.min( 1 ); j < node_domain.max( 1 ); j++ )
            {
                int iown = i - xmin;
                int jown = j - ymin;
                /* XXX Figure out why we have to write this column major when the optlist
                 * explicitly says we'll be passing them row major XXX. */
                for ( unsigned int d = 0; d < 3; d++ )
                {
                    coords[d][jown * node_domain.extent(0) + iown ] = zHost(i, j, d);
                }
             }
        }

        DBPutQuadmesh( dbfile, meshname, (DBCAS_t)coordnames, coords, dims,
                       3, DB_DOUBLE, DB_NONCOLLINEAR, optlist );

        // Now we write the individual variables associated with this
        // portion of the mesh, potentially copying them out of device space
        // and making sure not to write ghost values.

        // Mesh vorticity values - copy owned portion from the primary
        // execution space to the host execution space
        auto w = _pm.get( Field::Vorticity() )->array()->view();

        // array that we copy data into and then get a mirror view of.
        Kokkos::View<typename pm_type::mesh_array_type::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::mesh_array_type::memory_space>
            w1Owned( "w1o", node_domain.extent( 0 ), node_domain.extent( 1 ),
                    1 );
        Kokkos::View<typename pm_type::mesh_array_type::value_type***,
                     Kokkos::LayoutLeft,
                     typename pm_type::mesh_array_type::memory_space>
            w2Owned( "w2o", node_domain.extent( 0 ), node_domain.extent( 1 ),
                    1 );

        Kokkos::parallel_for(
            "SiloWriter::wowned copy",
            createExecutionPolicy( node_domain, ExecutionSpace() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                w1Owned( i - xmin, j - ymin, 0 ) = w( i, j, 0 );
                w2Owned( i - xmin, j - ymin, 0 ) = w( i, j, 1 );
            } );
        auto w1Host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), w1Owned );
        auto w2Host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), w2Owned );
        vars[0] = w1Host.data();
        vars[1] = w2Host.data();

        const char *varnames[2] = {"w1", "w2"};
        DBPutQuadvar( dbfile, "vorticity", meshname, 2, (DBCAS_t)varnames,
                      vars, dims, 3, NULL, 0, DB_DOUBLE, DB_NODECENT,
                      optlist );

        for ( unsigned int i = 0; i < 3; i++ )
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
     * Write Multi-Object Silo File the References Child Files in order to
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
                            int time_step, double time, double dt, 
                            const char* file_ext )
    {
        char** mesh_block_names = (char**)malloc( size * sizeof( char* ) );
        char** w_block_names = (char**)malloc( size * sizeof( char* ) );

        int* block_types = (int*)malloc( size * sizeof( int ) );
        int* var_types = (int*)malloc( size * sizeof( int ) );

        DBSetDir( silo_file, "/" );

        for ( int i = 0; i < size; i++ )
        {
            int group_rank = PMPIO_GroupRank( baton, i );
            mesh_block_names[i] = (char*)malloc( 1024 );
            w_block_names[i] = (char*)malloc( 1024 );

            snprintf( mesh_block_names[i], 1024, 
                      "raw/BeatnikOutput-%05d-%05d.%s:/domain_%05d/Mesh",
                      group_rank, time_step, file_ext, i );
            snprintf( w_block_names[i], 1024,
                      "raw/BeatnikOutput-%05d-%05d.%s:/domain_%05d/vorticity",
                      group_rank, time_step, file_ext, i );
            block_types[i] = DB_QUADMESH;
            var_types[i] = DB_QUADVAR;
        }

        // Set DB Options: Time Step, Time Stamp and Delta Time
        DBoptlist* optlist;
        optlist = DBMakeOptlist( 10 );
        DBAddOption( optlist, DBOPT_CYCLE, &time_step );
        DBAddOption( optlist, DBOPT_TIME, &time );
        DBAddOption( optlist, DBOPT_DTIME, &dt );
        int dbcoord = DB_CARTESIAN;
        DBAddOption( optlist, DBOPT_COORDSYS, &dbcoord );
        int dborder = DB_ROWMAJOR;
        DBAddOption( optlist, DBOPT_MAJORORDER, &dborder );
        int dborigin = 0;
        DBAddOption( optlist, DBOPT_BLOCKORIGIN, &dborigin );
        int dbtopo = 2;
        DBAddOption( optlist, DBOPT_TOPO_DIM, &dbtopo );

        DBPutMultimesh( silo_file, "multi_mesh", size, mesh_block_names,
                        block_types, optlist );
        DBPutMultivar( silo_file, "multi_vorticity", size, w_block_names,
                       var_types, optlist );
        for ( int i = 0; i < size; i++ )
        {
            free( mesh_block_names[i] );
            free( w_block_names[i] );
        }

        free( mesh_block_names );
        free( w_block_names );
        free( block_types );
        free( var_types );

        // Free Option List
        DBFreeOptlist( optlist ); 
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
        char masterfilename[256], filename[256], nsname[256];
  
        int rank = _pm.mesh().rank();
        /* Make sure the output directory exists */
        if (rank == 0) {
            // Make sure directories for output exist
            if (mkdir("data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
                if( errno != EEXIST ) {
                    // something else
                    std::cerr << "cannot create data directory. error:" 
                              << strerror(errno) << std::endl;
                    throw std::runtime_error( strerror(errno) );
                }
            }
            if (mkdir("data/raw", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
                if( errno != EEXIST ) {
                    // something else
                    std::cerr << "cannot create raw data directory. Error:" 
                              << strerror(errno) << std::endl;
                    throw std::runtime_error( strerror(errno) );
                }
            }
        }

        int size;
        // XXX Figure out how to set the number of groups intelligently
        int numGroups = 1;
        int driver = DB_PDB;
        const char* file_ext = "silo";
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        MPI_Bcast( &numGroups, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &driver, 1, MPI_INT, 0, MPI_COMM_WORLD );

        PMPIO_baton_t * baton =
            PMPIO_Init( numGroups, PMPIO_WRITE, MPI_COMM_WORLD, 1,
                        createSiloFile, openSiloFile, closeSiloFile, &driver );


        // Set Filename to Reflect TimeStep
        snprintf( masterfilename, 256, "data/Beatnik-%05d.%s", time_step,
                  file_ext );
        snprintf( filename, 256, "data/raw/BeatnikOutput-%05d-%05d.%s",
                  PMPIO_GroupRank( baton, rank ), time_step,
                  file_ext );
        snprintf( nsname, 256, "domain_%05d", rank );

        // Show Errors and Force Floating Point
        DBShowErrors( DB_ALL, NULL );

        DBfile * silo_file = (DBfile*)PMPIO_WaitForBaton( baton, filename, nsname );

        writeFile( silo_file, name, time_step, time, dt );
        if ( _pm.mesh().rank() == 0 )
        {
            DBfile * master_file = DBCreate( masterfilename, DB_CLOBBER, DB_LOCAL,
                                    "Beatnik", driver );
            writeMultiObjects( master_file, baton, size, time_step, 
                               time, dt, file_ext );
            DBClose( master_file );
        }
    
        PMPIO_HandOffBaton( baton, silo_file );

        PMPIO_Finish( baton );

    }
        
  private:
    const pm_type &_pm;
}; 

}; // namespace Beatnik
#endif
