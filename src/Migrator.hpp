/****************************************************************************
 * Copyright (c) 2021, 2022 by the Beatnik author                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik benchmark. Beatnik is                   *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_MIGRATOR
#define BEATNIK_MIGRATOR

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Beatnik
{
//---------------------------------------------------------------------------//
/*!
  \class Migrator
  \brief Migrator between surface and spatial meshes
*/
template <class ExecutionSpace, class MemorySpace>
class Migrator
{
  public:
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;

    // Construct a mesh.
    Migrator( const std::array<double, 6>& global_bounding_box,
        const std::array<int, 2>& num_nodes,
        const std::array<bool, 2>& periodic,
        const Cabana::Grid::BlockPartitioner<2>& partitioner,
        const int min_halo_width, MPI_Comm comm )
        : _num_nodes( num_nodes )
    {
        MPI_Comm_rank( comm, &_rank );

    }

    

  private:
    std::array<double, 3> _low_point, _high_point;
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
    int _rank;
	std::array<int, 2> _num_nodes;
};

//---------------------------------------------------------------------------//

} // end namespace Beatnik

#endif // end BEATNIK_MIGRATOR
