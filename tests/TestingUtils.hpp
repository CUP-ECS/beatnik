#ifndef _TESTING_UTILS_HPP
#define _TESTING_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <fstream>

namespace BeatnikTest
{
    
namespace Utils
{

// Function to write the contents of a Kokkos view to a file
void writeViewToFile(const Kokkos::View<double**[3]>& view, const std::string& filename) {
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

    // Write dimensions to file
    file.write(reinterpret_cast<const char*>(&dim0), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&dim1), sizeof(size_t));

    // Write the data (each subarray of size [3] for every element of (dim0 x dim1))
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            file.write(reinterpret_cast<const char*>(&hostView(i, j, 0)), sizeof(double) * 3);
        }
    }

    file.close();
}

// Function to read the contents of a file into a Kokkos view
Kokkos::View<double**[3]> readViewFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return Kokkos::View<double**[3]>();
    }

    // Read the dimensions from the file
    size_t dim0, dim1;
    file.read(reinterpret_cast<char*>(&dim0), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&dim1), sizeof(size_t));

    // Allocate a Kokkos view with the read dimensions
    Kokkos::View<double**[3]> view("view", dim0, dim1);

    // Create a host mirror to store the data temporarily
    auto hostView = Kokkos::create_mirror_view(view);

    // Read the data and populate the host view
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            file.read(reinterpret_cast<char*>(&hostView(i, j, 0)), sizeof(double) * 3);
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