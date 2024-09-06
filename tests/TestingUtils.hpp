#ifndef _TESTING_UTILS_HPP
#define _TESTING_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <fstream>
#include <type_traits>

namespace BeatnikTest
{
    
namespace Utils
{

// Generalized function to write the contents of a Kokkos view to a file
template <class View>
void writeViewToFile(const View& view, const std::string& filename) {
    // Ensure the view is of the correct type (with **[x])
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

    // Write the data
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            file.write(reinterpret_cast<const char*>(&hostView(i, j, 0)), sizeof(double) * dim2);
        }
    }

    file.close();
}

// Generalized function to read the contents of a file into a Kokkos view
template <class View>
View readViewFromFile(const std::string& filename, const int dim2) {
    // Ensure the view is of the correct type (with **[x])
    static_assert(View::rank == 3, "View must have a rank of 3");

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return View();
    }

    // Read the dimensions from the file
    size_t dim0, dim1;
    file.read(reinterpret_cast<char*>(&dim0), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&dim1), sizeof(size_t));

    // Allocate a Kokkos view with the read dimensions
    View view("from_file_view", dim0, dim1, dim2);

    // Create a host mirror to store the data temporarily
    auto hostView = Kokkos::create_mirror_view(view);

    // Read the data and populate the host view
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            file.read(reinterpret_cast<char*>(&hostView(i, j, 0)), sizeof(double) * dim2);
        }
    }

    // Copy the host view back to the device
    Kokkos::deep_copy(view, hostView);

    file.close();
    return view;
}

} // end namespace Utils

} // end namespace BeatnikTest

#endif // _TESTING_UTILS_HPP