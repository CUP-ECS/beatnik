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
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * VTKWriter class for unstructured mesh I/O
 */

#ifndef BEATNIK_VTKWRITER_HPP
#define BEATNIK_VTKWRITER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Grid.hpp>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkTriangle.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>

namespace Beatnik
{

/**
 * The VTKWriter Class
 * @class VTKWriter
 * @brief VTKWriter class for unstructured mesh I/O
 **/
template <class ProblemManagerType>
class VTKWriter
{
  public:
    using memory_space = typename ProblemManagerType::memory_space;
    using execution_space = typename ProblemManagerType::execution_space;
    using base_triple_type = typename ProblemManagerType::beatnik_mesh_type::base_triple_type;
    using base_pair_type = typename ProblemManagerType::beatnik_mesh_type::base_pair_type;
    using mesh_type_tag = typename ProblemManagerType::mesh_type_tag;

    using face_aosoa_type = Cabana::AoSoA<typename ProblemManagerType::
                                        beatnik_mesh_type::
                                            mesh_type::face_data, Kokkos::HostSpace, 4>;
    using vertex_aosoa_type = Cabana::AoSoA<typename ProblemManagerType::
                                        beatnik_mesh_type::
                                            mesh_type::vertex_data, Kokkos::HostSpace, 4>;

    /**
     * Constructor
     * Create new VTKWriter
     *
     * @param pm Problem manager object
     */
    VTKWriter( ProblemManagerType& pm )
        : _pm( pm )
    {
        // Get face and vertex data from mesh and copy to host memory
        const auto & mesh = _pm.mesh().layoutObj();
        auto vertices = mesh->vertices();
        auto faces = mesh->faces();

        _faces_h = face_aosoa_type("faces_h", faces.size());
        Cabana::deep_copy(_faces_h, faces);
        
        _vertices_h = vertex_aosoa_type("vertices_h", vertices.size());
        Cabana::deep_copy(_vertices_h, vertices);
    };

    // void setFilename( const idx_t step )
    // {
    //     std::ostringstream ss;
    //     ss << d_basename << "_" << step << ".vtu";
    //     d_filename = ss.str();
    // }

    void createDataFile()
    {
        // Retrieve the Local Grid and Local Mesh
        const auto & mesh = _pm.mesh().layoutObj();

        vtkNew<vtkPoints> points;
        vtkNew<vtkCellArray> cells;
        vtkNew<vtkIntArray> ownerRanks;  // Array to store rank of each cell
        ownerRanks->SetName("OwnerRank");
        ownerRanks->SetNumberOfComponents(1);

        auto vertices = mesh->vertices();
        auto faces = mesh->faces();
        int num_faces = mesh->count(NuMesh::Own(), NuMesh::Face());

        // Get positions AoSoA and copy to host memory
        using z_aosoa_type = Cabana::AoSoA<base_triple_type, Kokkos::HostSpace, 4>;
        auto zaosoa = _pm.get(Field::Position())->array()->aosoa();
        z_aosoa_type zhaosoa("positions_host", zaosoa->size());
        Cabana::deep_copy(zhaosoa, *zaosoa);
        auto z_slice = Cabana::slice<0>(zhaosoa);

        // Mesh slices
        auto v_gids = Cabana::slice<V_GID>(_vertices_h);
        auto f_vids = Cabana::slice<F_VIDS>(_faces_h);
        auto f_gid = Cabana::slice<F_GID>(_faces_h);
        auto f_cids = Cabana::slice<F_CID>(_faces_h);
        auto f_owner = Cabana::slice<F_OWNER>(_faces_h);

        // Create points and cells
        for (size_t n = 0; n < vertices.size(); ++n) {
            points->InsertNextPoint(z_slice(n, 0), z_slice(n, 1), z_slice(n, 2));
        }

        for (int n = 0; n < num_faces; ++n) {
            if (f_cids(n, 0) != -1) {
                continue;  // Skip child faces
            }

            int fvid0 = f_vids(n, 0);
            int vlid0 = NuMesh::Utils::get_lid(v_gids, fvid0, 0, _vertices_h.size());
            assert(vlid0 != -1);
            int fvid1 = f_vids(n, 1);
            int vlid1 = NuMesh::Utils::get_lid(v_gids, fvid1, 0, _vertices_h.size());
            assert(vlid1 != -1);
            int fvid2 = f_vids(n, 2);
            int vlid2 = NuMesh::Utils::get_lid(v_gids, fvid2, 0, _vertices_h.size());
            assert(vlid2 != -1);

            vtkIdType tri_ids[3] = {vlid0, vlid1, vlid2};
            cells->InsertNextCell(3, tri_ids);

            // Add the owner rank to the cell data
            ownerRanks->InsertNextValue(f_owner(n)); 
        }

        // Create unstructured mesh
        _d_vtk_mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
        _d_vtk_mesh->SetPoints(points);
        _d_vtk_mesh->SetCells(VTK_TRIANGLE, cells);

        // Add cell data
        _d_vtk_mesh->GetCellData()->AddArray(ownerRanks);
    }


    // void VTKWriter<Policy>::writeData( const std::string name, Kokkos::View<scalar_t *> field )
    // {
    //     using idx_t     = typename Policy::idx_t;
    //     using FaceEntry = typename FaceArray<idx_t>::Entry;

    //     const auto num_verts = field.extent( 0 );

    //     auto h_field = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), field );

    //     vtkNew<vtkDoubleArray> vtk_data;
    //     vtk_data->SetName( name.c_str() );
    //     vtk_data->SetNumberOfComponents( 1 );
    //     vtk_data->SetNumberOfTuples( num_verts );
    //     for ( idx_t n = 0; n < num_verts; ++n ) {
    //         vtk_data->SetTuple1( n, h_field( n ) );
    //     }
    //     _d_vtk_mesh->GetPointData()->AddArray( vtk_data );
    // }

    void writeVorticity()
    {
        const auto& mesh = _pm.mesh().layoutObj();
        // int num_verts = mesh->count(NuMesh::Own(), NuMesh::Vertex());
        int num_faces = mesh->count(NuMesh::Own(), NuMesh::Face());
        
        // Copy vorticity to host memory
        using w_aosoa_type = Cabana::AoSoA<base_pair_type, Kokkos::HostSpace, 4>;
        auto waosoa = _pm.get( Field::Vorticity() )->array()->aosoa();
        assert(waosoa->size() == mesh->vertices().size()); // Ensure aosoa is up-to-date
        w_aosoa_type whaosoa("vorticity_host", waosoa->size());
        Cabana::deep_copy(whaosoa, *waosoa);
        auto w_slice = Cabana::slice<0>(whaosoa);

        // Mesh slices
        auto v_gids = Cabana::slice<V_GID>(_vertices_h);
        auto f_vids = Cabana::slice<F_VIDS>(_faces_h);
        auto f_gid = Cabana::slice<F_GID>(_faces_h);
        auto f_cids = Cabana::slice<F_CID>(_faces_h);

        std::string name = "vorticity";

        vtkNew<vtkDoubleArray> vtk_data;
        const int num_vars = 2;
        vtk_data->SetName( name.c_str() );
        vtk_data->SetNumberOfComponents( num_vars );
        vtk_data->SetNumberOfTuples( mesh->vertices().size() );

        // Write vorticity data for all points on all owned faces
        for ( int n = 0; n < num_faces; ++n ) {
            if ( f_cids( n, 0 ) != -1 ) {
                // Only consider child faces
                continue;
            }
            int fvid0 = f_vids(n, 0);
            int vlid0 = NuMesh::Utils::get_lid(v_gids, fvid0, 0, _vertices_h.size());
            assert(vlid0 != -1);
            int fvid1 = f_vids(n, 1);
            int vlid1 = NuMesh::Utils::get_lid(v_gids, fvid1, 0, _vertices_h.size());
            assert(vlid1 != -1);
            int fvid2 = f_vids(n, 2);
            int vlid2 = NuMesh::Utils::get_lid(v_gids, fvid2, 0, _vertices_h.size());
            assert(vlid2 != -1);
            vtk_data->SetTuple2( vlid0, w_slice( vlid0, 0 ), w_slice( vlid0, 1 ) );
            vtk_data->SetTuple2( vlid1, w_slice( vlid1, 0 ), w_slice( vlid1, 1 ) );
            vtk_data->SetTuple2( vlid2, w_slice( vlid2, 0 ), w_slice( vlid2, 1 ) );
        }
        _d_vtk_mesh->GetPointData()->AddArray( vtk_data );
    }

    // finalizeDataFile()
    // {
    //     vtkNew<vtkXMLUnstructuredGridWriter> writer;
    //     writer->SetFileName( d_filename.c_str() );
    //     writer->SetInputData( d_vtk_mesh );
    //     writer->Write();
    // }

    void vtkWrite(int time_step)
    {
        const int rank = _pm.mesh().rank();
        const int comm_size = _pm.mesh().comm_size();

        /* Make sure the output directory exists */
        if (rank == 0) {
            if (mkdir("data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST) {
                std::cerr << "Cannot create data directory. Error: " << strerror(errno) << std::endl;
                throw std::runtime_error(strerror(errno));
            }
            if (mkdir("data/raw", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST) {
                std::cerr << "Cannot create raw data directory. Error: " << strerror(errno) << std::endl;
                throw std::runtime_error(strerror(errno));
            }
        }

        // Ensure all processes sync before writing
        MPI_Barrier(MPI_COMM_WORLD);

        // Set filename for each rank
        std::string filename = "data/raw/BeatnikOutput-" + std::to_string(rank) + "-" +
                            std::to_string(time_step) + ".vtu";

        createDataFile();
        writeVorticity();

        // Each rank writes its own VTU file
        auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(_d_vtk_mesh);
        writer->Write();

        // Synchronize again to ensure all ranks have written their files
        MPI_Barrier(MPI_COMM_WORLD);

        // Rank 0 generates the PVTU file
        if (rank == 0)
        {
            std::string masterfilename = "data/Beatnik-" + std::to_string(time_step) + ".pvtu";

            std::ofstream pvtuFile(masterfilename);
            if (!pvtuFile.is_open()) {
                std::cerr << "Error opening PVTU file for writing: " << masterfilename << std::endl;
                throw std::runtime_error("Could not open PVTU file.");
            }

            // Write the PVTU XML header
            // This is hard-coded until we can figure out how to correctly write the pvtu file
            pvtuFile << "<?xml version=\"1.0\"?>\n";
            pvtuFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            pvtuFile << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
            pvtuFile << "    <PPoints>\n";
            pvtuFile << "      <PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\"/>\n";
            pvtuFile << "    </PPoints>\n";
            pvtuFile << "    <PCells>\n";
            pvtuFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"/>\n";
            pvtuFile << "      <PDataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"/>\n";
            pvtuFile << "      <PDataArray type=\"UInt8\" Name=\"types\" format=\"ascii\"/>\n";
            pvtuFile << "    </PCells>\n";

            // Define point data (for vorticity)
            pvtuFile << "    <PPointData Scalars=\"vorticity\">\n";
            pvtuFile << "      <PDataArray type=\"Float64\" Name=\"vorticity\" NumberOfComponents=\"2\" format=\"ascii\"/>\n";
            pvtuFile << "    </PPointData>\n";

            // Add the cell data section to describe `OwnerRank`
            pvtuFile << "    <PCellData Scalars=\"OwnerRank\">\n";
            pvtuFile << "      <PDataArray type=\"Int32\" Name=\"OwnerRank\" format=\"ascii\"/>\n";
            pvtuFile << "    </PCellData>\n";

            // Add piece entries for each rank
            for (int i = 0; i < comm_size; i++)
            {
                pvtuFile << "    <Piece Source=\"raw/BeatnikOutput-" << i << "-" << time_step << ".vtu\"/>\n";
            }

            pvtuFile << "  </PUnstructuredGrid>\n";
            pvtuFile << "</VTKFile>\n";

            pvtuFile.close();
        }
    }

        
  private:
    const ProblemManagerType &_pm;
    vtkSmartPointer<vtkUnstructuredGrid> _d_vtk_mesh;

    face_aosoa_type _faces_h;
    vertex_aosoa_type _vertices_h;

    /**
     * The number of vertices that reside on all faces we own.
     * Some of these vertices are ghosted.
     * Populated during the call to createDataFile
     * and used to set the number of tuples for vorticity data in writeVorticity.
     */
    // int _num_vertices;
};

template <class ProblemManagerType>
std::shared_ptr<VTKWriter<ProblemManagerType>> createVTKWriter( ProblemManagerType& pm )
{
    return std::make_shared<VTKWriter<ProblemManagerType>>( pm );
}

}; // namespace Beatnik

#endif // BEATNIK_VTKWRITER_HPP
