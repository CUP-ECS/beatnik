#ifndef _TESTING_UTILS_HPP_
#define _TESTING_UTILS_HPP_

#include <Kokkos_Core.hpp>
#include <fstream>
#include <type_traits>

namespace BeatnikTest
{
    
namespace Utils
{

std::string get_filename(int rank, int comm_size, int mesh_size, int periodic, char x)
{
    std::string filename = "z_";
    if (x == 'w') filename = "w_";
    filename += std::to_string(mesh_size);
    if (periodic == 1) filename += "_p_";
    else filename += "_f_";
    filename += "r";
    filename += std::to_string(rank);
    filename += ".";
    filename += std::to_string(comm_size);
    filename += ".view";
    return filename;
}


// Generalized function to write the contents of a Kokkos view to a file
template <class View>
void writeViewToFile(const View& view, const std::string& filename) {
    static_assert(View::rank == 3, "View must have a rank of 3");

    // Create a host mirror to access the data
    auto hostView = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hostView, view);

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Get the dimensions of the view
    size_t dim0 = view.extent(0);
    size_t dim1 = view.extent(1);
    size_t dim2 = view.extent(2);

    // Write dimensions to file
    file.write(reinterpret_cast<const char*>(&dim0), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&dim1), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&dim2), sizeof(size_t));

    // Write the data in linear order
    for (int i = 0; i < (int)dim0; ++i) {
        for (int j = 0; j < (int)dim1; ++j) {
            for (int k = 0; k < (int)dim2; ++k) {
                file.write(reinterpret_cast<const char*>(&hostView(i, j, k)), sizeof(double));
            }
        }
    }

    file.close();
}

template <class View>
View readViewFromFile(const std::string& filename, const int dim2) {
    static_assert(View::rank == 3, "View must have a rank of 3");

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return View();
    }

    // Read the dimensions from the file
    size_t dim0, dim1, fileDim2;
    file.read(reinterpret_cast<char*>(&dim0), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&dim1), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&fileDim2), sizeof(size_t));

    if ((int)fileDim2 != dim2) {
        std::cerr << "Dimension mismatch! Expected dim2 = " << dim2 << ", but file contains dim2 = " << fileDim2 << std::endl;
        return View();
    }

    // Allocate a Kokkos view with the read dimensions
    View view("from_file_view", dim0, dim1, dim2);

    // Create a host mirror to store the data temporarily
    auto hostView = Kokkos::create_mirror_view(view);

    // Read the data in linear order
    for (int i = 0; i < (int)dim0; ++i) {
        for (int j = 0; j < (int)dim1; ++j) {
            for (int k = 0; k < (int)dim2; ++k) {
                file.read(reinterpret_cast<char*>(&hostView(i, j, k)), sizeof(double));
            }
        }
    }

    // Copy the host view back to the device
    Kokkos::deep_copy(view, hostView);

    file.close();
    return view;
}

template <class View>
void writeView(int rank, int comm_size, int mesh_size, int periodic, const View v)
{
    char x = 'z';
    if (v.extent(2) == 2) x = 'w';
    std::string filename = get_filename(rank, comm_size, mesh_size, periodic, x);
    writeViewToFile(v, filename);
}

} // end namespace Utils

} // end namespace BeatnikTest

#endif // _TESTING_UTILS_HPP_