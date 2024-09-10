#ifndef _TESTING_UTILS_HPP_
#define _TESTING_UTILS_HPP_

#include <Kokkos_Core.hpp>
#include <fstream>
#include <type_traits>

namespace BeatnikTest
{
    
namespace Utils
{

int init_cl_args( ClArgs& cl )
{
    /// Set default values
    cl.driver = "serial"; // Default Thread Setting
    cl.weak_scale = 1;
    cl.write_freq = 0;

    // Set default extra parameters
    cl.params.cutoff_distance = 0.5;
    cl.params.heffte_configuration = 6;
    cl.params.br_solver = BR_EXACT;
    cl.params.solver_order = SolverOrder::ORDER_LOW;
    // cl.params.period below

    /* Default problem is the cosine rocket rig */
    cl.num_nodes = { 64, 64 };
    cl.bounding_box = 1.0;
    cl.initial_condition = IC_COS;
    cl.boundary = BoundaryType::PERIODIC;
    cl.tilt = 0.0;
    cl.magnitude = 0.05;
    cl.variation = 0.00;
    cl.params.period = 1.0;
    cl.gravity = 25.0;
    cl.atwood = 0.5;

    /* Defaults for Z-Model method, translated by the solver to be relative
     * to sqrt(dx*dy) */
    cl.mu = 1.0;
    cl.eps = 0.25;

    /* Defaults computed once other arguments known */
    cl.delta_t = -1.0;
    cl.t_final = 5;

    /* Physical setup of problem */
    cl.params.global_bounding_box = {cl.bounding_box * -1.0,
                                     cl.bounding_box * -1.0, 
                                     cl.bounding_box * -1.0,
                                     cl.bounding_box,
                                     cl.bounding_box,
                                     cl.bounding_box};
    cl.gravity = cl.gravity * 9.81;

    /* Scale up global bounding box and number of cells by weak scaling factor */
    for (int i = 0; i < 6; i++) {
        cl.params.global_bounding_box[i] *= sqrt(cl.weak_scale);
    }
    for (int i = 0; i < 2; i++) {
        cl.num_nodes[i] *= sqrt(cl.weak_scale);
    }

    /* Figure out parameters we need for the timestep and such. Simulate long
     * enough for the interface to evolve significantly */
    double tau = 1/sqrt(cl.atwood * cl.gravity);

    if (cl.delta_t <= 0.0) {
        if (cl.params.solver_order == SolverOrder::ORDER_HIGH) {
            cl.delta_t = tau/50.0;  // Should this depend on dx and dy? XXX
        } else {
            cl.delta_t = tau/25.0;
        }
    }

    if (cl.t_final <= 0.0) {
        cl.t_final = tau * 2.0; // Simulate for 2 characterisic periods, which is all
                                // the low-order model can really handle
    }
    else {
        cl.t_final = cl.t_final * cl.delta_t;
    }

    // Return Successfully
    return 0;
}

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