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
 * @author Thomas Hines <thomas-hines-01@utc.edu>
 *
 * @section DESCRIPTION
 * ZModel class that handles computing derivatives of ionterface position and
 * velocity, using external classses for different velocity calculation
 * strategies
 */

#ifndef BEATNIK_PVFMMBRSOLVER_HPP
#define BEATNIK_PVFMMBRSOLVER_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Mesh.hpp>
#include <ProblemManager.hpp>
#include <Operators.hpp>

#include <pvfmm.hpp>

namespace Beatnik
{

#if 0
// XXX Attempt at a custom desingularized kernel for PvFMM. Resulted in long
// setup times and large setup files, no easy way to define epsilon at runtime, 
// and still didn't fix correctness problems.

struct birchoff_rott_ {
  static const int FLOPS = 24;
  static constexpr double _eps = 2.0/128.0;
  template <class Real> static Real ScaleFactor() { return 1 / (4 * sctl::const_pi<Real>()); }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
      VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2] + _eps*_eps;
      VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
      VecType rinv3 = rinv*rinv*rinv;

      u[0] = FMA(rinv3, f[1]*r[2] - f[2]*r[1], u[0]);
      u[1] = FMA(rinv3, f[2]*r[0] - f[0]*r[2], u[1]);
      u[2] = FMA(rinv3, f[0]*r[1] - f[1]*r[0], u[2]);
  }
};

struct birchoff_rott : public pvfmm::GenericKernel<birchoff_rott_> {};

template<class T>
struct BirchoffRottKernel{
    inline static const pvfmm::Kernel<T>& potential() 
    {
        static pvfmm::Kernel<T> ker = pvfmm::BuildKernel<T, birchoff_rott::Eval<T>>("birchoff_rott", 3, std::pair<int, int>(3, 3));
        return ker;
    }
};
#endif

/**
 * The PvfmmBRSolver Class
 * @class PvfmmBRSolver
 * @brief Directly solves the Birchoff-Rott integral using the PVFMM
 * fast multipole method.
 **/
template <class ExecutionSpace, class MemorySpace>
class PvfmmBRSolver
{
  public:
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>;

    using Node = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      device_type>;

    using halo_type = Cajita::Halo<MemorySpace>;

    PvfmmBRSolver( const pm_type & pm, const BoundaryCondition &bc,
                   const double epsilon, const double dx, const double dy)
        : _pm( pm )
        , _bc( bc )
        , _dx( dx )
        , _dy( dy )
        , _pvfmm_kernel_fn( pvfmm::BiotSavartKernel<double>::potential() )
    {
        _pvfmm_mem_mgr = std::make_unique<pvfmm::mem::MemoryManager>(10000000);
        _pvfmm_matricies = std::make_unique<pvfmm::PtFMM<double>>(_pvfmm_mem_mgr.get());

        // FMM Setup
	auto comm = _pm.mesh().localGrid()->globalGrid().comm();
        std::cout << "Initializing PVFMM Matricies.\n";
        _pvfmm_matricies->Initialize(10, comm, &_pvfmm_kernel_fn);
        std::cout << "Initialized PVFMM Matricies.\n";
    }

    /* Directly compute the interface velocity by integrating the vorticity 
     * across the surface. Uses a fast multipole method for computing the 
     * these integrals */
    template <class PositionView, class VorticityView>
    void computeInterfaceVelocity(PositionView zdot, PositionView z,
                                  VorticityView w) const
    {
        auto node_space = _pm.mesh().localGrid()->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        std::size_t nnodes = node_space.size();

        /* Create C-Order (LayoutRight) views on the device for the coordinates 
         * and densities */
        Kokkos::View<typename node_array::value_type***,
                     Kokkos::LayoutRight,
                     typename node_array::device_type>
            coord( "node coords", node_space.extent( 0 ), node_space.extent( 1 ),
                    3 );
        Kokkos::View<typename node_array::value_type***,
                     Kokkos::LayoutRight,
                     typename node_array::device_type>
            vortex( "node vortex", node_space.extent( 0 ), node_space.extent( 1 ),
                    3 );

        /* Figure out how we need to translate and scale down coordinates 
         * and scale down the vorticities to map the problem tothe unit cube */
        auto low_point = _pm.mesh().boundingBoxMin();
        auto high_point = _pm.mesh().boundingBoxMax();
        Kokkos::Array<double, 3> translation;
        double coord_scale = 1;
        for (int i = 0; i < 3; i++) {
            translation[i] = -low_point[i];
            low_point[i] += translation[i];
            high_point[i] += translation[i];
            if (high_point[i] > coord_scale) coord_scale = high_point[i];
        }
        /* Biot-Savart is a 1/s^2 kernel. The denominator goes down by s^2, so the 
         * numerator needs to as well. */
        double value_scale = (coord_scale * coord_scale);
        std::cout << "Scaling down coordinates by factor of " << coord_scale << "\n";
        std::cout << "Scaling down omega by factor of " << value_scale << "\n";
        /* Iterate through the mesh filling out the coordinate and weighted
         * vorticity magnitude device arrays */
        int xmin = node_space.min(0), ymin = node_space.min(1);
        int xmax = node_space.max(0), ymax = node_space.max(1);
	double dx = _dx, dy = _dy;

        Kokkos::parallel_for("FMM Coordinate/Point Calculation",
            Cajita::createExecutionPolicy(node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++) {
                double dx_z = Operators::Dx<4>(z, i, j, n, dx);
                double dy_z = Operators::Dy<4>(z, i, j, n, dy);
                double kweight, lweight;

                /* Compute simpson's 3/8 quadrature weight for this index */
                if ((i == xmin) || (i == xmax - 1)) kweight = 3.0/8.0;
                else if ((i - xmin) % 3 == 0) kweight = 3.0/4.0;
                else kweight = 9.0/8.0;

                if ((j == ymin) || (j == ymax - 1)) lweight = 3.0/8.0;
                else if (j - ymin % 3 == 0) lweight = 3.0/4.0;
                else lweight = 9.0/8.0;

                coord(i - xmin, j - ymin, n) = (z(i, j, n) + translation[n]) / coord_scale;
                vortex(i - xmin, j - ymin, n) = kweight * lweight * (w(i, j, 1) * dx_z - w(i, j, 0) * dy_z) / value_scale;
	    } 
        });

        auto coordHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coord );
        auto vortexHost = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), vortex );

        /* Create vectors to store coordinates and densities */
        double *data = coordHost.data();
        std::vector<double> coordVector(data, data + 3 * nnodes);

        data = vortexHost.data();
        std::vector<double> vortexVector(data, data + 3 * nnodes);

        /* Initialize the PvFMM tree with the coordinates provided */
        size_t max_pts = 500;
        auto comm = _pm.mesh().localGrid()->globalGrid().comm();
        std::cout << "Creating PVFMM tree.\n";
        auto * tree = PtFMM_CreateTree(coordVector, vortexVector, coordVector, comm, max_pts, pvfmm::PXY);
        tree->SetupFMM(_pvfmm_matricies.get());
        std::cout << "Created PVFMM tree.\n";

        /* Evaluate the FMM to compute forces */
        std::vector<double> results;
        std::cout << "Evaluating PVFMM tree.\n";
        PtFMM_Evaluate(tree, results, nnodes);
        std::cout << "Evaluated PVFMM tree.\n";

        /* Copy the force array back to the device. Reuse vortexHost for this */
        std::memcpy(vortexHost.data(), results.data(), nnodes * sizeof(double) * 3);
        
        /* Copy the computed forces into the zwdot view, also changing the sign of the results
         * to account for the difference between the scale factor in the PvFmm BiotSavart kernel 
         * and computes and the scale factor in the high-order model. */
        Kokkos::deep_copy(vortex, vortexHost);
        Kokkos::parallel_for("Copy back FMM Results",
            Cajita::createExecutionPolicy(node_space, ExecutionSpace()),
            KOKKOS_LAMBDA(int i, int j) {
            for (int n = 0; n < 3; n++) {
                zdot(i + xmin, j + ymin, n) = -1.0 * vortex(i, j, n);
	    } 
        });

        /* Clean up and destroy FMM solver state */
        delete tree;
    } 

    const pm_type & _pm;
    const BoundaryCondition & _bc;
    double _dx, _dy;
    std::unique_ptr<pvfmm::mem::MemoryManager> _pvfmm_mem_mgr;
    std::unique_ptr<pvfmm::PtFMM<double>> _pvfmm_matricies;
    const pvfmm::Kernel<double>& _pvfmm_kernel_fn;

};

}; // namespace Beatnik

#endif // BEATNIK_PVFMMBRSOLVER_HPP
