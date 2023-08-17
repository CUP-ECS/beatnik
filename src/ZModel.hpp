/****************************************************************************
 * Copyright (c) 2021-2022 by the Beatnik authors                           *
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
 * ZModel class that handles computing derivatives of interface position and
 * velocity, using external classses for different velocity calculation
 * strategies
 */

#ifndef BEATNIK_ZMODEL_HPP
#define BEATNIK_ZMODEL_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Mesh.hpp>

#include <BoundaryCondition.hpp>
#include <Operators.hpp>

namespace Beatnik
{


// Type tags designating difference orders of the zmodel
namespace Order
{
    struct Low {};

    struct Medium {};

    struct High {};
} // namespace Order

/**
 * The ZModel Class
 * @class ZModel
 * @brief ZModel class handles the specific of the various ZModel versions, 
 * invoking an external class to solve far-field forces if necessary.
 **/
template <class ExecutionSpace, class MemorySpace, class MethodOrder, class BRSolver>
class ZModel
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

    ZModel( const pm_type & pm, const BoundaryCondition &bc,
            const BRSolver *br, /* pointer because could be null */
            const double dx, const double dy, 
            const double A, const double g, const double mu)
        : _pm( pm )
        , _bc( bc )
        , _br( br )
        , _dx( dx )
        , _dy( dy )
        , _A( A )
        , _g( g )
        , _mu( mu )
    {
        // Need the node triple layout for storing vector normals and the 
        // node double layout for storing x and y surface derivative
        auto node_double_layout =
            Cajita::createArrayLayout( _pm.mesh().localGrid(), 2, Cajita::Node() );
        auto node_triple_layout =
            Cajita::createArrayLayout( _pm.mesh().localGrid(), 3, Cajita::Node() );
        auto node_scalar_layout =
            Cajita::createArrayLayout( _pm.mesh().localGrid(), 1, Cajita::Node() );

        // Temporary used for central differencing of vorticities along the 
        // surface in calculating the vorticity derivative
        _V = Cajita::createArray<double, device_type>(
            "V", node_scalar_layout );

        /* We need a halo for _V so that we can do fourth-order central differencing on
         * it. This requires a depth 2 stencil with adjacent faces */
        int halo_depth = 2; 
        _v_halo = Cajita::createHalo( Cajita::FaceHaloPattern<2>(),
                            halo_depth, *_V );

        /* Storage for the reisz transform of the vorticity. In the low and 
         * medium order models, it is used to calculate the vorticity 
         * derivative. In the low order model, it is also projected onto the 
         * surface normal to compute the interface velocity.  
         * XXX Make this conditional on the model we run. */
        _reisz = Cajita::createArray<double, device_type>( "reisz", node_double_layout );
        Cajita::ArrayOp::assign( *_reisz, 0.0, Cajita::Ghost() );

        /* If we're not the hgh order model, initialize the FFT solver and 
         * the working space it will need. 
         * XXX figure out how to make this conditional on model order. */
        Cajita::Experimental::FastFourierTransformParams params;
        _C1 = Cajita::createArray<double, device_type>("C1", node_double_layout);
        _C2 = Cajita::createArray<double, device_type>("C2", node_double_layout);

        params.setAllToAll(true);
        params.setPencils(true);
        params.setReorder(false);
        _fft = Cajita::Experimental::createHeffteFastFourierTransform<double, device_type>(*node_double_layout, params);
    }

    double computeMinTimestep(double atwood, double g)
    {
        return 1.0/(25.0*sqrt(atwood * g));
    }

    /* Compute the velocities needed by the relevant Z-Model. Both the full 
     * velocity vector and the magnitude of the normal velocity to the surface. 
     * A reisz transform can be used to directly compute the the the magnitude 
     * of the normal velocity of the interface, and a full field method for 
     * solving Birchoff-Rott integral can be used to compute the full interface
     * velocity. Based on this, we end up with three models: a low-order model 
     * that uses only the reisz transform but would need many more points to 
     * accurately resolve the interface, a high-order model that directly 
     * calculated the BR integral to compute both the full velocity and normal 
     * magnitude but needs smaller timesteps, and a mixed method that uses the 
     * reisz transform for the magnitude of the velocity to calculate vorticity
     * and the BR solver for the fine-grain velocity. 
     *
     * This process is done in three passes - 
     * 1. An initial bulk-sychronous pass to calculate to calculate global 
     *    information such as the reisz transform and/or far-field solve
     * 2. A separate pass over each mesh point that computes normals and 
     *    determinants, call method-specific routine to finish calculation 
     *    of the full velocity and normal velocity magnitude, and then 
     *    calculate "V", an intermediate per-point value needed in calculating
     *    the vorticity derivative. 
     * 3. A final pass that central differences V and combines it with a 
     *    a laplacian-based artificial viscosity to calculate the final
     *    vorticity derivative. Note that this requires haloing V prior
     *    to the parallel loop that performs this calculation. 
     */

    KOKKOS_INLINE_FUNCTION
    static double reiszWeight(double i, int numnodes)
    {
        if (numnodes % 2 == 0) {
	    if (i < 0) {
                return numnodes/2 + i;
            } else if (i > 0)  {
                return i - numnodes/2;
            } else {
                return i;
            }
        } else {
	    if (i <= 0) {
                return (numnodes - 1)/2 + i;
            } else {
                return i - (numnodes - 1)/2 - 1;
            }
        }
    }

    template <class VorticityView>
    void computeReiszTransform(VorticityView w) const
    {
        /* Construct the temporary arrays C1 and C2 */
        auto local_grid = _pm.mesh().localGrid();
        auto & global_grid = local_grid->globalGrid();
        auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );
        auto local_nodes = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());

        /* Get the views we'll be computing with in parallel loops */
        auto C1 = _C1->view();
        auto C2 = _C2->view();
        auto reisz = _reisz->view();

        /* First put w into the real parts of C1 and C2 (and zero out
         * any imaginary parts left from previous runs!) */
        Kokkos::parallel_for("Build FFT Arrays", 
                     Cajita::createExecutionPolicy(local_nodes, ExecutionSpace()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            C1(i, j, 0) = w(i, j, 0);
            C1(i, j, 1) = 0;
            C2(i, j, 0) = w(i, j, 1);
            C2(i, j, 1) = 0;
        });

        /* Do we need to halo C1 and C2 now? We shouldn't, since the FFT should take
         * care of that. */

        /* Now do the FFTs of vorticity */
        _fft->forward(*_C1, Cajita::Experimental::FFTScaleNone());
        _fft->forward(*_C2, Cajita::Experimental::FFTScaleNone());

        int nx = global_grid.globalNumEntity(Cajita::Node(), 0);
        int ny = global_grid.globalNumEntity(Cajita::Node(), 1);

        /* Now construct reisz from the weighted sum of those FFTs to take the inverse FFT. */
        parallel_for("Combine FFTs", 
                     Cajita::createExecutionPolicy(local_nodes, ExecutionSpace()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            int indicies[2] = {i, j};
            double location[2];
            local_mesh.coordinates( Cajita::Node(), indicies, location );

            double k1 = reiszWeight(location[0], nx);
            double k2 = reiszWeight(location[1], ny);

            if ((k1 != 0) || (k2 != 0)) {
                /* real part = -i * M1 * imag(C1) + -i * M2 * imag(C2)
                 *           = M1 * imag(C1) + M2 * imag(C2)
                 * imag part = -i * M1 * real(C1) - i * M2 * real(C2)
                 *           = -M1 * real(C1) - M2 * real(C2)
                 */
                double len = sqrt(k1 * k1 + k2 * k2);
                double M1 = k1 / len;
                double M2 = k2 / len;

                reisz(i, j, 0) = M1 * C1(i, j, 1) + M2 * C2(i, j, 1);
                reisz(i, j, 1) = -M1 * C1(i, j, 0) - M2 * C2(i, j, 0);
            } else {
                reisz(i, j, 0) = 0.0; 
                reisz(i, j, 1) = 0.0;
            }
        });

        /* We then do the reverse transform to finish the reisz transform,
         * which is used later to calculate final interface velocity */
        _fft->reverse(*_reisz, Cajita::Experimental::FFTScaleFull());
    }

    /* For low order, we calculate the reisz transform used to compute the magnitude
     * of the interface velocity. This will be projected onto surface normals later 
     * once we have the normals */
    template <class PositionView, class VorticityView>
    void prepareVelocities(Order::Low, [[maybe_unused]] PositionView zdot,
                           [[maybe_unused]] PositionView z, VorticityView w) const
    {
        computeReiszTransform(w);
    }

    /* For medium order, we calculate the fourier velocity that we later 
     * normalize for vorticity calculations and directly compute the 
     * interface velocity (zdot) using a far field method. */
    template <class PositionView, class VorticityView>
    void prepareVelocities(Order::Medium, PositionView zdot, PositionView z, VorticityView w) const
    {
        computeReiszTransform(w);
        _br->computeInterfaceVelocity(zdot, z, w);
    }

    /* For high order, we just directly compute the interface velocity (zdot)
     * using a far field method and later normalize that for use in the vorticity 
     * calculation. */
    template <class PositionView, class VorticityView>
    void prepareVelocities(Order::High, PositionView zdot, PositionView z, VorticityView w) const
    {
        _br->computeInterfaceVelocity(zdot, z, w);
    }

    // Compute the final interface velocities and normalized BR velocities
    // from the previously computed Fourier and/or Birkhoff-Rott velocities and the surface
    // normal based on  the order of technique we're using.
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION 
    static void finalizeVelocity(Order::Low, double &zndot, ViewType zdot, 
        int i, int j, ViewType reisz, double norm[3], double deth) 
    {
        zndot = -0.5 * reisz(i, j, 0) / deth;
        for (int d = 0; d < 3; d++)
            zdot(i, j, d) = zndot * norm[d];
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    static void finalizeVelocity(Order::Medium, double &zndot, 
        [[maybe_unused]] ViewType zdot, 
        [[maybe_unused]] int i, [[maybe_unused]] int j,
        ViewType reisz, [[maybe_unused]] double norm[3], double deth) 
    {
        zndot = -0.5 * reisz(i, j, 0) / deth;
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    static void finalizeVelocity(Order::High, double &zndot, ViewType zdot, 
        int i, int j, [[maybe_unused]] ViewType reisz, 
        [[maybe_unused]] double norm[3], [[maybe_unused]] double deth)
    {
        double interface_velocity[3] = {zdot(i, j, 0), zdot(i, j, 1), zdot(i, j, 2)};
        zndot = sqrt(Operators::dot(interface_velocity, interface_velocity));
    }
 
    // External entry point from the TimeIntegration object that uses the
    // problem manager state.
    template <class PositionView, class VorticityView>
    void computeDerivatives( PositionView zdot, VorticityView wdot ) const
    {
       _pm.gather();
       auto z_orig = _pm.get( Cajita::Node(), Field::Position() );
       auto w_orig = _pm.get( Cajita::Node(), Field::Vorticity() );
       computeHaloedDerivatives( z_orig, w_orig, zdot, wdot );
    } 

    // External entry point from the TimeIntegration object that uses the
    // passed-in state
    template <class PositionView, class VorticityView>
    void computeDerivatives( node_array &z, node_array &w,
                             PositionView zdot, VorticityView wdot ) const
    {
        _pm.gather( z, w );
	computeHaloedDerivatives( z.view(), w.view(), zdot, wdot );
    }

    // Shared internal entry point from the external points from the
    // TimeIntegration object
    template <class PositionView, class VorticityView>
    void computeHaloedDerivatives( PositionView z_view, VorticityView w_view,
                                   PositionView zdot, VorticityView wdot ) const
    {
        // External calls to this object work on Cajita arrays, but internal
        // methods mostly work on the views, with the entry points responsible
        // for handling the halos.
	double dx = _dx, dy = _dy;
 
        // Phase 1: Globally-dependent bulk synchronous calculations that 
        // namely the reisz transform and/or far-field force solve to calculate
        // interface velocity and velocity normal magnitudes, using the
        // appropriate method. We do not attempt to overlap this with the 
        // mostly-local parallel calculations in phase 2
        prepareVelocities(MethodOrder(), zdot, z_view, w_view);

        auto reisz = _reisz->view();
        double g = _g;
        double A = _A;

        // Phase 2: Process the globally-dependent velocity information into 
        // into final interface position derivatives and the information 
        // needed for calculating the vorticity derivative
        auto V_view = _V->view();

        auto local_grid = _pm.mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        Kokkos::parallel_for( "Interface Velocity",  
            createExecutionPolicy(own_node_space, ExecutionSpace()), 
            KOKKOS_LAMBDA(int i, int j) {
            //  2.1 Compute Dx and Dy of z and w by fourth-order central differencing. 
            double dx_z[3], dy_z[3];

            for (int n = 0; n < 3; n++) {
               dx_z[n] = Operators::Dx(z_view, i, j, n, dx);
               dy_z[n] = Operators::Dy(z_view, i, j, n, dy);
            }

            //  2.2 Compute h11, h12, h22, and det_h from Dx and Dy
            double h11 = Operators::dot(dx_z, dx_z);
            double h12 = Operators::dot(dx_z, dy_z);
            double h22 = Operators::dot(dy_z, dy_z);
            double deth = h11*h22 - h12*h12;

            //  2.3 Compute the surface normal as (Dx \cross Dy)/sqrt(deth)
            double N[3];
            Operators::cross(N, dx_z, dy_z);
            for (int n = 0; n < 3; n++)
		N[n] /= sqrt(deth);

            //  2.4 Compute zdot and zndot as needed using specialized helper functions
            double zndot;
            finalizeVelocity(MethodOrder(), zndot, zdot, i, j, 
                             reisz, N, deth );

            //  2.5 Compute V from zndot and vorticity 
	    double w1 = w_view(i, j, 0); 
            double w2 = w_view(i, j, 1);

	    V_view(i, j, 0) = zndot * zndot 
                         - 0.25*(h22*w1*w1 - 2.0*h12*w1*w2 + h11*w2*w2)/deth 
                         - 2*g*z_view(i, j, 2);
        });

        // 3. Phase 3: Halo V and apply boundary condtions on it, then calculate
        // central differences of V, laplacians for artificial viscosity, and
        // put it all together to calcualte the final vorticity derivative.

        // Halo V and correct any boundary condition corrections so that we can 
        // compute finite differences correctly.
        _v_halo->gather( ExecutionSpace(), *_V);
        _bc.applyField( _pm.mesh(), *_V, 1 );

        double mu = _mu;
        Kokkos::parallel_for( "Interface Vorticity",
            createExecutionPolicy(own_node_space, ExecutionSpace()), 
            KOKKOS_LAMBDA(int i, int j) {
            double dx_v = Operators::Dx(V_view, i, j, 0, dx);
            double dy_v = Operators::Dy(V_view, i, j, 0, dy);
            double lap_w0 = Operators::laplace(w_view, i, j, 0, dx, dy);
            double lap_w1 = Operators::laplace(w_view, i, j, 1, dx, dy);
            wdot(i, j, 0) = A * dx_v + mu * lap_w0;
            wdot(i, j, 1) = A * dy_v + mu * lap_w1;
        });

    }

  private:
    const pm_type & _pm;
    const BoundaryCondition & _bc;
    const BRSolver *_br;
    double _dx, _dy;
    double _A, _g, _mu;
    std::shared_ptr<node_array> _V;
    std::shared_ptr<halo_type> _v_halo;

    /* XXX Make this conditional on not being the high-order model */ 
    std::shared_ptr<node_array> _reisz;
    std::shared_ptr<node_array> _C1, _C2; 
    std::shared_ptr<Cajita::Experimental::HeffteFastFourierTransform<Cajita::Node, mesh_type, double, device_type, Cajita::Experimental::Impl::FFTBackendDefault>> _fft;
}; // class ZModel

} // namespace Beatnik

#endif // BEATNIK_ZMODEL_HPP
