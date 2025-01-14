/****************************************************************************
 * Copyright (c) 2020-2022 by the Beatnik authors                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is                     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef BEATNIK_CREATEMESH_HPP
#define BEATNIK_CREATEMESH_HPP

#include <MeshBase.hpp>
#include <StructuredMesh.hpp>
#include <UnstructuredMesh.hpp>


namespace Beatnik
{


/* Separate header for createMesh to avoid circular 
 * dependencies between MeshBase and the mesh classes.
 */
template <class ExecutionSpace, class MemorySpace, class MeshTypeTag>
std::unique_ptr<MeshBase<ExecutionSpace, MemorySpace, MeshTypeTag>>
createMesh( const std::array<double, 6>& global_bounding_box,
            const std::array<int, 2>& num_nodes,
	        const std::array<bool, 2>& periodic,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            const int min_halo_width, MPI_Comm comm )
{
    if constexpr (std::is_same_v<MeshTypeTag, Mesh::Structured>)
    {
        using beatnik_mesh_type = StructuredMesh<ExecutionSpace, MemorySpace, MeshTypeTag>;
        return std::make_unique<beatnik_mesh_type>(global_bounding_box, num_nodes, periodic,
            partitioner, min_halo_width, comm);
    }
    else if constexpr (std::is_same_v<MeshTypeTag, Mesh::Unstructured>)
    {
        // Create the local grid that will build this mesh
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        auto global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, periodic, partitioner );
        int halo_width = 1; // Halo width doesn't matter
        auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        using beatnik_mesh_type = UnstructuredMesh<ExecutionSpace, MemorySpace, MeshTypeTag>;
        auto mesh = std::make_unique<beatnik_mesh_type>(comm, periodic);
        
        return mesh;
    }
    std::cerr << "createMesh:: Invalid mesh type.\n";
    Kokkos::finalize();
    MPI_Finalize();
    exit(-1);
}

} // end namespace Beatnik

#endif /* BEATNIK_CREATEMESH_HPP */
