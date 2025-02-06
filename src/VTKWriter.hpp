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
    using value_type = typename ProblemManagerType::beatnik_mesh_type::value_type;
    using mesh_type_tag = typename ProblemManagerType::mesh_type_tag;
    /**
     * Constructor
     * Create new VTKWriter
     *
     * @param pm Problem manager object
     */
    VTKWriter( ProblemManagerType& pm, const std::string base )
        : _pm( pm )
        , _base_filename( base )
    {};

    // void setFilename( const idx_t step )
    // {
    //     std::ostringstream ss;
    //     ss << d_basename << "_" << step << ".vtu";
    //     d_filename = ss.str();
    // }

    void createDataFile( const int step )
    {
        // setFilename( step );

        // Retrieve the Local Grid and Local Mesh
        const auto & mesh = _pm.mesh().layoutObj();

        // for simplicity create new points/triangles every time
        // assuming that they change frequently enough
        vtkNew<vtkPoints> points;
        vtkNew<vtkCellArray> cells;

        // Declare the coordinates of the portion of the mesh we're writing
        auto vertices = mesh->vertices();
        auto faces = mesh->faces();
        int num_verts = mesh->count(NuMesh::Own(), NuMesh::Vertex());
        int num_faces = mesh->count(NuMesh::Own(), NuMesh::Face());

        // Get positions AoSoA and copy to host memory
        using z_aosoa_type = Cabana::AoSoA<value_type, Kokkos::HostSpace, 4>;
        auto zaosoa = _pm.get( Field::Position() )->array()->aosoa();
        assert(zaosoa.size() == vertices.size()); // Ensure aosoa is up-to-date
        z_aosoa_type zhaosoa("positions_host", zaosoa.size());
        Cabana::deep_copy(zhaosoa, zaosoa);
        auto z_slice = Cabana::slice<0>(zhaosoa);

        // Get face and vertex data and copy to host memory
        using face_aosoa_type = Cabana::AoSoA<typename ProblemManagerType::
                                            beatnik_mesh_type::
                                                mesh_type::face_data, Kokkos::HostSpace, 4>;
        face_aosoa_type faces_h("faces_h", faces.size());
        Cabana::deep_copy(faces_h, faces);
        using vertex_aosoa_type = Cabana::AoSoA<typename ProblemManagerType::
                                            beatnik_mesh_type::
                                                mesh_type::vertex_data, Kokkos::HostSpace, 4>;
        vertex_aosoa_type vertices_h("vertices_h", vertices.size());
        Cabana::deep_copy(vertices_h, vertices);

        auto v_gids = Cabana::slice<V_GID>(vertices_h);
        auto f_vids = Cabana::slice<F_VIDS>(faces_h);
        auto f_gid = Cabana::slice<F_GID>(faces_h);
        auto f_cids = Cabana::slice<F_CID>(faces_h);

        // create all points and triangles
        for ( size_t n = 0; n < vertices.size(); ++n ) {
            points->InsertNextPoint( z_slice( n, 0 ), z_slice( n, 1 ), z_slice( n, 2 ) );
        }
        for ( int n = 0; n < num_faces; ++n ) {
            if ( f_cids( n, 0 ) != -1 ) {
                // Only consider child faces
                continue;
            }
            int fvid0 = f_vids(n, 0);
            int vlid0 = NuMesh::Utils::get_lid(v_gids, fvid0, 0, vertices_h.size());
            assert(vlid0 != -1);
            int fvid1 = f_vids(n, 0);
            int vlid1 = NuMesh::Utils::get_lid(v_gids, fvid1, 0, vertices_h.size());
            assert(vlid1 != -1);
            int fvid2 = f_vids(n, 0);
            int vlid2 = NuMesh::Utils::get_lid(v_gids, fvid2, 0, vertices_h.size());
            assert(vlid2 != -1);
            vtkNew<vtkTriangle> tri;
            tri->GetPointIds()->SetId( 0, vlid0 );
            tri->GetPointIds()->SetId( 1, vlid1 );
            tri->GetPointIds()->SetId( 2, vlid2 );
            cells->InsertNextCell( tri );
        }

        // create unstructured mesh
        _d_vtk_mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
        _d_vtk_mesh->SetPoints( points );
        _d_vtk_mesh->SetCells( VTK_TRIANGLE, cells );
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

    // writeData( const std::string name, Kokkos::View<scalar_t **> field )
    // {
    //     const auto num_verts = field.extent( 0 );
    //     const auto num_vars  = field.extent( 1 );

    //     if ( num_vars > 3 ) {
    //         std::cout << "VTKWriter<Policy>::writeData Too many components in passed field. Ignoring."
    //                 << std::endl;
    //         return;
    //     }

    //     auto h_field = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), field );

    //     vtkNew<vtkDoubleArray> vtk_data;
    //     vtk_data->SetName( name.c_str() );
    //     vtk_data->SetNumberOfComponents( num_vars );
    //     vtk_data->SetNumberOfTuples( num_verts );
    //     for ( idx_t n = 0; n < num_verts; ++n ) {
    //         if ( num_vars == 1 ) {
    //             vtk_data->SetTuple1( n, h_field( n, 0 ) );
    //         } else if ( num_vars == 2 ) {
    //             vtk_data->SetTuple2( n, h_field( n, 0 ), h_field( n, 1 ) );
    //         } else {
    //             vtk_data->SetTuple3( n, h_field( n, 0 ), h_field( n, 1 ), h_field( n, 2 ) );
    //         }
    //     }
    //     _d_vtk_mesh->GetPointData()->AddArray( vtk_data );
    // }

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

        createDataFile(time_step);

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
    const std::string _base_filename;

    vtkSmartPointer<vtkUnstructuredGrid> _d_vtk_mesh;
};

template <class ProblemManagerType>
std::shared_ptr<VTKWriter<ProblemManagerType>> createVTKWriter( ProblemManagerType& pm, const std::string base)
{
    return std::make_shared<VTKWriter<ProblemManagerType>>( pm, base );
}

}; // namespace Beatnik

#endif // BEATNIK_VTKWRITER_HPP
