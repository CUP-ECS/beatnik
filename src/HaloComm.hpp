/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/* PARTS OF THIS CODE IS COPIED FROM CABANAPD: https://github.com/ORNL/CabanaPD/blob/main/src/CabanaPD_Comm.hpp */

#ifndef HALO_COMM_H
#define HALO_COMM_H

#include <string>

#include "mpi.h"

#include <Cabana_Grid.hpp>


namespace Beatnik
{
template <std::size_t Size, class Scalar>
auto vectorToArray( std::vector<Scalar> vector )
{
    Kokkos::Array<Scalar, Size> array;
    for ( std::size_t i = 0; i < Size; ++i )
        array[i] = vector[i];
    return array;
}

// Functor to determine which particles should be ghosted with Cabana grid.
template <class MemorySpace, class LocalGridType>
struct HaloIds
{
    static constexpr std::size_t num_space_dim = LocalGridType::num_space_dim;
    static constexpr int topology_size = 26;

    using memory_space = MemorySpace;

    int _min_halo;

    using coord_type = Kokkos::Array<double, num_space_dim>;
    Kokkos::Array<coord_type, topology_size> _min_coord;
    Kokkos::Array<coord_type, topology_size> _max_coord;

    Kokkos::Array<int, topology_size> _device_topology;

    using DestinationRankView = typename Kokkos::View<int*, memory_space>;
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, memory_space,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    CountView _send_count;
    DestinationRankView _destinations;
    DestinationRankView _ids;

    template <class PositionSliceType>
    HaloIds( const LocalGridType& local_grid,
             const PositionSliceType& positions, const int minimum_halo_width,
             const int max_export_guess )
    {
        _destinations = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
            max_export_guess );
        _ids = DestinationRankView(
            Kokkos::ViewAllocateWithoutInitializing( "ids" ),
            max_export_guess );
        _send_count = CountView( "halo_send_count" );

        // Check within the halo width, within the local domain.
        _min_halo = minimum_halo_width;

        auto topology = Cabana::Grid::getTopology( local_grid );
        _device_topology = vectorToArray<topology_size>( topology );

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Get the neighboring mesh bounds (only needed once unless load
        // balancing).
        neighborBounds( local_grid );

        build( positions );
    }

    // Find the bounds of each neighbor rank and store for determining which
    // ghost particles to communicate.
    void neighborBounds( const LocalGridType& local_grid )
    {
        const auto& local_mesh =
            Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( local_grid );

        Kokkos::Array<Cabana::Grid::IndexSpace<4>, topology_size> index_spaces;

        // Store all neighboring shared index space mesh bounds so we only have
        // to launch one kernel during the actual ghost search.
        int n = 0;
        for ( int k = -1; k < 2; ++k )
        {
            for ( int j = -1; j < 2; ++j )
            {
                for ( int i = -1; i < 2; ++i, ++n )
                {
                    if ( i != 0 || j != 0 || k != 0 )
                    {
                        int neighbor_rank = local_grid.neighborRank( i, j, k );
                        // Potentially invalid neighbor ranks (non-periodic
                        // global boundary)
                        if ( neighbor_rank != -1 )
                        {
                            auto sis = local_grid.sharedIndexSpace(
                                Cabana::Grid::Own(), Cabana::Grid::Cell(), i, j, k,
                                _min_halo );
                            auto min_ind = sis.min();
                            auto max_ind = sis.max();
                            local_mesh.coordinates( Cabana::Grid::Node(),
                                                    min_ind.data(),
                                                    _min_coord[n].data() );
                            local_mesh.coordinates( Cabana::Grid::Node(),
                                                    max_ind.data(),
                                                    _max_coord[n].data() );
                        }
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------//
    // Locate particles within the local grid and determine if any from this
    // rank need to be ghosted to one (or more) of the 26 neighbor ranks,
    // keeping track of destination rank and index.
    template <class PositionSliceType, class UserFunctor>
    void build( const PositionSliceType& positions, UserFunctor user_functor )
    {
        using execution_space = typename PositionSliceType::execution_space;

        // Local copies of member variables for lambda capture.
        auto send_count = _send_count;
        auto destinations = _destinations;
        auto ids = _ids;
        auto device_topology = _device_topology;
        auto min_coord = _min_coord;
        auto max_coord = _max_coord;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Look for ghosts within the halo width of the local mesh boundary,
        // potentially for each of the 26 neighbors cells.
        // Do this one neighbor rank at a time so that sends are contiguous.
        auto ghost_search = KOKKOS_LAMBDA( const int p )
        {
            for ( std::size_t n = 0; n < topology_size; n++ )
            {
                // Potentially invalid neighbor ranks (non-periodic global
                // boundary)
                if ( device_topology[n] != -1 )
                {
                    // Check the if particle is both in the owned
                    // space and the ghosted space of this neighbor
                    // (ignore the current cell).
                    bool within_halo = false;
                    if ( positions( p, 0 ) > min_coord[n][0] &&
                         positions( p, 0 ) < max_coord[n][0] &&
                         positions( p, 1 ) > min_coord[n][1] &&
                         positions( p, 1 ) < max_coord[n][1] &&
                         positions( p, 2 ) > min_coord[n][2] &&
                         positions( p, 2 ) < max_coord[n][2] )
                        within_halo = true;
                    if ( within_halo )
                    {
                        double px[3] = { positions( p, 0 ), positions( p, 1 ),
                                         positions( p, 2 ) };
                        
                        // Let the user restrict to a subset of the boundary.
                        bool create_ghost = user_functor( p, px );
                        if ( create_ghost )
                        {
                            const std::size_t sc = send_count()++;
                            // If the size of the arrays is exceeded,
                            // keep counting to resize and fill next.
                            if ( sc < destinations.extent( 0 ) )
                            {
                                // Keep the destination MPI rank.
                                destinations( sc ) = device_topology[n];
                                // Keep the particle ID.
                                ids( sc ) = p;
                            }
                        }
                    }
                }
            }
        };

        auto policy =
            Kokkos::RangePolicy<execution_space>( 0, positions.size() );
        Kokkos::parallel_for( "CabanaPD::Comm::GhostSearch", policy,
                              ghost_search );
        Kokkos::fence();
    }

    template <class PositionSliceType>
    void build( const PositionSliceType& positions )
    {
        auto empty_functor = KOKKOS_LAMBDA( const int, const double[3] )
        {
            return true;
        };
        build( positions, empty_functor );
    }

    template <class PositionSliceType>
    void rebuild( const PositionSliceType& positions )
    {
        // Resize views to actual send sizes.
        int dest_size = _destinations.extent( 0 );
        int dest_count = 0;
        Kokkos::deep_copy( dest_count, _send_count );
        if ( dest_count != dest_size )
        {
            Kokkos::resize( _destinations, dest_count );
            Kokkos::resize( _ids, dest_count );
        }

        // If original view sizes were exceeded, only counting was done so
        // we need to rerun.
        if ( dest_count > dest_size )
        {
            Kokkos::deep_copy( _send_count, 0 );
            build( positions );
        }
    }
};

template <class MemorySpace, class ParticleType, class LocalGridType>
class Comm
{
  public:
    int mpi_size = -1;
    int mpi_rank = -1;
    int max_export;

    using memory_space = MemorySpace;
    using halo_type = Cabana::Halo<memory_space>;
    using gather_particles_type =
        Cabana::Gather<halo_type, ParticleType>;
    std::shared_ptr<gather_particles_type> gather_particles;
    std::shared_ptr<halo_type> halo;

    Comm( ParticleType& particles,
          const LocalGridType& local_grid, int max_export_guess = 100 )
        : max_export( max_export_guess )
    {
        MPI_Comm_size( local_grid.globalGrid().comm(), &mpi_size );
        MPI_Comm_rank( local_grid.globalGrid().comm(), &mpi_rank );

        auto positions = Cabana::slice<0>(particles, "positions");

        // Get all 26 neighbor ranks.
        auto halo_width = local_grid.haloCellWidth();
        auto topology = Cabana::Grid::getTopology( local_grid );

        // Determine which particles need to be ghosted to neighbors.
        auto halo_ids =
            createHaloIds( local_grid, positions, halo_width, max_export );
        // Rebuild if needed.
        halo_ids.rebuild( positions );

        // Number of local particles
        int num_local = particles.size();

        // Create the Cabana Halo.
        halo = std::make_shared<halo_type>( local_grid.globalGrid().comm(),
                                            num_local, halo_ids._ids,
                                            halo_ids._destinations, topology );

        int total_particles = halo->numLocal() + halo->numGhost();
        particles.resize( total_particles );

        Cabana::gather( *halo, particles );

        gather_particles = std::make_shared<gather_particles_type>( *halo, particles );
        gather_particles->apply();
    }
    ~Comm() {}

    // Determine which particles should be ghosted, reallocating and recounting
    // if needed.
    template <class PositionSliceType>
    auto createHaloIds( const LocalGridType& local_grid,
                        const PositionSliceType& positions,
                        const int min_halo_width, const int max_export )
    {
        return HaloIds<typename PositionSliceType::memory_space, LocalGridType>(
            local_grid, positions, min_halo_width, max_export );
    }
};

} // namespace Beatnik

#endif /* HALO_COMM_H */
