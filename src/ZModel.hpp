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
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include <Beatnik_Types.hpp>
#include <StructuredMesh.hpp>
#include <CreateBRSolver.hpp>
#include <Beatnik_ArrayUtils.hpp>

#include <BoundaryCondition.hpp>
#include <Operators.hpp>

int di = 4, dj = 3;

namespace Beatnik
{

/**
 * The ZModel Class
 * @class ZModel
 * @brief ZModel class handles the specific of the various ZModel versions, 
 * invoking an external class to solve far-field forces if necessary.
 **/

// NodeArray would be the passed-on array type
// Not template zmodel on exec, mem space, pull it out of arraytype instead
// Implement our own arrayops in the strucutre currently in place
/*Then see if it's worth moving everything to templates
 * 
 */
template <class ProblemManagerType, class BRSolverType, class ModelOrderTag>
class ZModel
{
  public:
    using execution_space = typename ProblemManagerType::execution_space;
    using memory_space = typename ProblemManagerType::memory_space;
    using mesh_type_tag = typename ProblemManagerType::mesh_type_tag;
    using mesh_type = typename ProblemManagerType::mesh_type::mesh_type; // This is a Cabana::Grid::Mesh type
    using mesh_array_type = typename ProblemManagerType::mesh_array_type;
    using value_type = typename mesh_array_type::value_type;
    using entity_type = typename ProblemManagerType::entity_type;
    using halo_type = Cabana::Grid::Halo<memory_space>;

    ZModel( const ProblemManagerType & pm, const BoundaryCondition &bc,
            const BRSolverType *br, /* pointer because could be null */
            const double dx, const double dy, 
            const double A, const double g, const double mu,
            const int heffte_configuration)
        : _pm( pm )
        , _bc( bc )
        , _br( br )
        , _dx( dx )
        , _dy( dy )
        , _A( A )
        , _g( g )
        , _mu( mu )
        , _heffte_configuration( heffte_configuration )
    {
        // Need the node triple layout for storing vector normals and the 
        // node double layout for storing x and y surface derivative
        auto node_double_layout =
            ArrayUtils::createArrayLayout<value_type>( _pm.mesh().layoutObj(), 2, entity_type() );
        auto node_triple_layout =
            ArrayUtils::createArrayLayout<value_type>( _pm.mesh().layoutObj(), 3, entity_type() );
        auto node_scalar_layout =
            ArrayUtils::createArrayLayout<value_type>( _pm.mesh().layoutObj(), 1, entity_type() );

        
        // Initize omega view
        _omega = ArrayUtils::createArray<memory_space>("omega", node_triple_layout);

        // Temporary used for central differencing of vorticities along the 
        // surface in calculating the vorticity derivative
        _V = ArrayUtils::createArray<memory_space>("V", node_scalar_layout);

        /* Storage for the reisz transform of the vorticity. In the low and 
         * medium order models, it is used to calculate the vorticity 
         * derivative. In the low order model, it is also projected onto the 
         * surface normal to compute the interface velocity.  
         * XXX Make this conditional on the model we run. */
        _reisz = ArrayUtils::createArray<memory_space>("reisz", node_double_layout);
        ArrayUtils::ArrayOp::assign( *_reisz, 0.0 );

        /* We need a halo for _V so that we can do fourth-order central differencing on
         * it. This requires a depth 2 stencil with adjacent faces.
         *
         * We also only need fft variables for structured meshes
         * 
         * Only initialize for structured meshes
         */
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            int halo_depth = 2; 
             _v_halo = Cabana::Grid::createHalo( Cabana::Grid::FaceHaloPattern<2>(),
                            halo_depth, *_V->array() );

            /* If we're not the hgh order model, initialize the FFT solver and 
            * the working space it will need. 
            * XXX figure out how to make this conditional on model order. */
            Cabana::Grid::Experimental::FastFourierTransformParams fft_params;
            _C1 = ArrayUtils::createArray<memory_space>("C1", node_double_layout);
            _C2 = ArrayUtils::createArray<memory_space>("C2", node_double_layout);

            switch (_heffte_configuration) {
                case 0:
                    fft_params.setAllToAll(false);
                    fft_params.setPencils(false);
                    fft_params.setReorder(false);
                    break;
                case 1:
                    fft_params.setAllToAll(false);
                    fft_params.setPencils(false);
                    fft_params.setReorder(true);
                    break;
                case 2:
                    fft_params.setAllToAll(false);
                    fft_params.setPencils(true);
                    fft_params.setReorder(false);
                    break;
                case 3:
                    fft_params.setAllToAll(false);
                    fft_params.setPencils(true);
                    fft_params.setReorder(true);
                    break;
                case 4:
                    fft_params.setAllToAll(true);
                    fft_params.setPencils(false);
                    fft_params.setReorder(false);
                    break;
                case 5:
                    fft_params.setAllToAll(true);
                    fft_params.setPencils(false);
                    fft_params.setReorder(true);
                    break;
                case 6:
                    fft_params.setAllToAll(true);
                    fft_params.setPencils(true);
                    fft_params.setReorder(false);
                    break;
                case 7:
                    fft_params.setAllToAll(true);
                    fft_params.setPencils(true);
                    fft_params.setReorder(true);
                    break;
            }
            _fft = Cabana::Grid::Experimental::createHeffteFastFourierTransform<double, memory_space>(*node_double_layout->layout(), fft_params);
        }
        else 
        {
            printf("WARNING: ZModel constructor: Unfinished implementation for unstructured meshes.\n");
            // throw std::invalid_argument("ZModel constructor: Unfinished implementation for unstructured meshes.");
        }
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

    template <class VorticityArray>
    void computeReiszTransform(VorticityArray w, Cabana::Grid::Node) const
    {
        /* Construct the temporary arrays C1 and C2 */
        auto local_grid = w.clayout()->layout()->localGrid();
        auto& global_grid = local_grid->globalGrid();
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *local_grid );
        auto local_nodes = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        /* Get the views we'll be computing with in parallel loops */
        auto C1 = _C1->array()->view();
        auto C2 = _C2->array()->view();
        auto reisz = _reisz->array()->view();
        auto w_view = w.array()->view();

        /* First put w into the real parts of C1 and C2 (and zero out
         * any imaginary parts left from previous runs!) */
        Kokkos::parallel_for("Build FFT Arrays", 
                     Cabana::Grid::createExecutionPolicy(local_nodes, execution_space()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            C1(i, j, 0) = w_view(i, j, 0);
            C1(i, j, 1) = 0;
            C2(i, j, 0) = w_view(i, j, 1);
            C2(i, j, 1) = 0;
        });

        /* Do we need to halo C1 and C2 now? We shouldn't, since the FFT should take
         * care of that. */

        /* Now do the FFTs of vorticity */
        _fft->forward(*_C1->array(), Cabana::Grid::Experimental::FFTScaleNone());
        _fft->forward(*_C2->array(), Cabana::Grid::Experimental::FFTScaleNone());

        int nx = global_grid.globalNumEntity(Cabana::Grid::Node(), 0);
        int ny = global_grid.globalNumEntity(Cabana::Grid::Node(), 1);

        /* Now construct reisz from the weighted sum of those FFTs to take the inverse FFT. */
        parallel_for("Combine FFTs", 
                     Cabana::Grid::createExecutionPolicy(local_nodes, execution_space()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            int indicies[2] = {i, j};
            double location[2];
            local_mesh.coordinates( Cabana::Grid::Node(), indicies, location );

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
        _fft->reverse(*_reisz->array(), Cabana::Grid::Experimental::FFTScaleFull());
    }

    /* For low order, we calculate the reisz transform used to compute the magnitude
     * of the interface velocity. This will be projected onto surface normals later 
     * once we have the normals */
    template <class EntityTag>
    void prepareVelocities(Order::Low, [[maybe_unused]] mesh_array_type& zdot,
                           [[maybe_unused]] mesh_array_type& z, mesh_array_type& w,
                           [[maybe_unused]] mesh_array_type& omega,
                           EntityTag etag) const
    {
        if constexpr (std::is_same_v<EntityTag, Cabana::Grid::Node>)
        {
            computeReiszTransform(w, etag);
        } 
        else if constexpr (std::is_same_v<EntityTag, NuMesh::Face>)
        {
            // XXX - Perform unstructured equaivalent
            throw std::invalid_argument("ZModel::prepareVelocities: Not yet implemented for unstructured meshes.");
        } 
    }

    /* For medium order, we calculate the fourier velocity that we later 
     * normalize for vorticity calculations and directly compute the 
     * interface velocity (zdot) using a far field method. */
    template <class EntityTag>
    void prepareVelocities(Order::Medium, mesh_array_type& zdot, mesh_array_type& z, mesh_array_type& w,
                           mesh_array_type& omega, EntityTag etag) const
    {
        if constexpr (std::is_same_v<EntityTag, Cabana::Grid::Node>)
        {
            computeReiszTransform(w, etag);
            _br->computeInterfaceVelocity(zdot.array()->view(), z.array()->view(), omega.array()->view());
        } 
        else if constexpr (std::is_same_v<EntityTag, NuMesh::Face>)
        {
            // XXX - Perform unstructured equaivalent
            throw std::invalid_argument("ZModel::prepareVelocities: Not yet implemented for unstructured meshes.");
        }
    }

    /* For high order, we just directly compute the interface velocity (zdot)
     * using a far field method and later normalize that for use in the vorticity 
     * calculation. */
    template <class EntityTag>
    void prepareVelocities(Order::High, mesh_array_type& zdot, mesh_array_type& z,
                           [[maybe_unused]] mesh_array_type& w, mesh_array_type& omega,
                           [[maybe_unused]] EntityTag etag) const
    {
        if constexpr (std::is_same_v<EntityTag, Cabana::Grid::Node>)
        {
            _br->computeInterfaceVelocity(zdot.array()->view(), z.array()->view(), omega.array()->view());
        } 
        else if constexpr (std::is_same_v<EntityTag, NuMesh::Face>)
        {
            // XXX - Perform unstructured equivalent
            throw std::invalid_argument("ZModel::prepareVelocities: Not yet implemented for unstructured meshes.");
        }
    }

    // Compute the final interface velocities and normalized BR velocities
    // from the previously computed Fourier and/or Birkhoff-Rott velocities and the surface
    // normal based on  the order of technique we're using.
    template <class DecompositionTag>
    void finalizeVelocity(Order::Low, mesh_array_type& zndot, mesh_array_type& zdot, 
        mesh_array_type& reisz, mesh_array_type& surface_norm, mesh_array_type& inv_deth, DecompositionTag tag) const 
    {
        /*
         * zndot = (i, j, 1)
         * norm = surface_normal array: (i, j, 1)
         * reisz = array 
         *     The reisz array is (i, j, 2), but we only use the first of the two third dimensions.
         * deth = array: (i, j, 1)
         * zdot = array: (i, j, 1)
         * 
         *  Compute:
         *     zndot = -0.5 * reisz(i, j, 0) / deth;
         *     zdot = zndot * surface_norm;
         */

        // Step 1: (reisz) -> (-0.5 * reisz)
        // We can edit the reisz view because it is not used again after this calculation.
        // We cannot edit inv_deth.
        auto reisz_D1 = ArrayUtils::ArrayOp::copyDim(reisz, 0, tag); 
        ArrayUtils::ArrayOp::scale(*reisz_D1, -0.5, tag);

        // Step 2: zndot = -0.5 * reisz(i, j, 0) / deth;
        auto zndot_1 = ArrayUtils::ArrayOp::element_multiply(inv_deth, *reisz_D1, tag);
        ArrayUtils::ArrayOp::copy(zndot, *zndot_1, tag);
       
        // Step 3: zdot = zndot * surface_norm
        auto zdot_1 = ArrayUtils::ArrayOp::element_multiply(zndot, surface_norm, tag);

        // Step 4: Copy back into zdot
        ArrayUtils::ArrayOp::copy(zdot, *zdot_1, tag);
    }

    template <class DecompositionTag>
    void finalizeVelocity(Order::Medium, mesh_array_type& zndot, 
        [[maybe_unused]] mesh_array_type& zdot, 
        mesh_array_type& reisz, [[maybe_unused]] mesh_array_type& surface_norm, mesh_array_type& inv_deth,
        DecompositionTag tag) const
    {
        /**
         * Compute: zndot = -0.5 * reisz(i, j, 0) / deth;
         */

        // Step 1: (1/deth) -> (-0.5/deth)
        ArrayUtils::ArrayOp::scale(inv_deth, -0.5, tag);

        // Step 2: zndot = (-0.5/deth) * reisz
        auto reisz_D1 = ArrayUtils::ArrayOp::copyDim(reisz, 0, tag);
        auto result = ArrayUtils::ArrayOp::element_multiply(inv_deth, *reisz_D1, tag);

        // Step 3: Copy back into zndot
        ArrayUtils::ArrayOp::copy(zndot, *result, tag);
    }

    template <class DecompositionTag>
    void finalizeVelocity(Order::High, mesh_array_type& zndot, mesh_array_type& zdot, 
        [[maybe_unused]] mesh_array_type& reisz, 
        [[maybe_unused]] mesh_array_type& surface_norm, [[maybe_unused]] mesh_array_type& inv_deth,
        DecompositionTag tag) const
    {
        /**
         * Compute:
         *  double interface_velocity[3] = {zdot(i, j, 0), zdot(i, j, 1), zdot(i, j, 2)};
         *  zndot = sqrt(Operators::dot(interface_velocity, interface_velocity));
         */
        Operators::SqrtFunctor sqrt_func;
        auto interface_velocity = ArrayUtils::ArrayOp::element_dot(zdot, zdot, tag);
        ArrayUtils::ArrayOp::apply(*interface_velocity, sqrt_func, tag);

        // Copy back into zndot
        ArrayUtils::ArrayOp::copy(zndot, *interface_velocity, tag);
    }
 
    // External entry point from the TimeIntegration object that uses the
    // problem manager state.
    template <class EntityTag, class DecompositionTag>
    void computeDerivatives( mesh_array_type& zdot_ptr, mesh_array_type& wdot_ptr,
                             EntityTag etag, DecompositionTag dtag ) const
    {
       _pm.gather();
       auto z_orig = _pm.get( Field::Position() );
       auto w_orig = _pm.get( Field::Vorticity() );
       computeHaloedDerivatives( *z_orig, *w_orig, zdot_ptr, wdot_ptr, etag, dtag );
    } 

    // External entry point from the TimeIntegration object that uses the
    // passed-in state
    template <class EntityTag, class DecompositionTag>
    void computeDerivatives( mesh_array_type& z, mesh_array_type& w,
                             mesh_array_type& zdot, mesh_array_type& wdot,
                             EntityTag etag, DecompositionTag dtag ) const
    {
        _pm.gather( z, w );
	    computeHaloedDerivatives( z, w, zdot, wdot, etag, dtag );
    }

    /**  
     * Shared internal entry point from the external points from the TimeIntegration object
     * etag = Cabana::Grid::Node or NuMesh variant,
     * dtag = Cabana::Grid::Own or NuMesh variant
     */
    template <class EntityTag, class DecompositionTag>
    void computeHaloedDerivatives( mesh_array_type& z_array, mesh_array_type& w_array,
                                   mesh_array_type& zdot_array, mesh_array_type& wdot_array,
                                   EntityTag etag, DecompositionTag dtag ) const
    {
        // External calls to this object work on Cabana::Grid arrays, but internal
        // methods mostly work on the views, with the entry points responsible
        // for handling the halos.
 
        // Phase 1: Globally-dependent bulk synchronous calculations that 
        // namely the reisz transform and/or far-field force solve to calculate
        // interface velocity and velocity normal magnitudes, using the
        // appropriate method. We do not attempt to overlap this with the 
        // mostly-local parallel calculations in phase 2

        // Get dx and dy arrays
        auto z_dx = _pm.mesh().Dx(z_array, _dx);
        auto z_dy = _pm.mesh().Dy(z_array, _dy);
        /**
         * In the unstructured formulation, dx and dy are only needed by vorticity,
         * and if we keep omega constant for now then dx and dy are not needed.
         * Can just return an array of zeros of the correct shape of dx and dy
         * 
         * Give some facility to refine a single face and some facility to refine all faces
         * We can figure out WHICH faces to refine later
         * 
         * Later: de-refinement
         * Python repo: 
         * 
         * Ask Patrick:
         * How should we distribute the mesh? Keep the top layers globally to avoid communication
         * Ian isn't sure how many internal particles there are yet.
         * Or comminucate the mesh each time
         */

        // Phase 1.a: Calcuate the omega value for each point
        auto out = _pm.mesh().omega(w_array, *z_dx, *z_dy);
        // Keep omega constant for now, quadratic in terms of the initial vertical position
        Beatnik::ArrayUtils::ArrayOp::copy(*_omega, *out, dtag);

        // Phase 1.b: Compute zdot
        prepareVelocities(ModelOrderTag(), zdot_array, z_array, w_array, *_omega, etag);

        /*
        Part 2.2:
            double h11 = Operators::dot(dx_z, dx_z);
            double h12 = Operators::dot(dx_z, dy_z);
            double h22 = Operators::dot(dy_z, dy_z);
            double deth = h11*h22 - h12*h12;
         */
        auto h11 = ArrayUtils::ArrayOp::element_dot(*z_dx, *z_dx, dtag);
        auto h12 = ArrayUtils::ArrayOp::element_dot(*z_dx, *z_dy, dtag);
        auto h22 = ArrayUtils::ArrayOp::element_dot(*z_dy, *z_dy, dtag);
        auto deth = ArrayUtils::ArrayOp::clone(*h11);
        ArrayUtils::ArrayOp::assign(*deth, 0.0, dtag);
        ArrayUtils::ArrayOp::update(*deth, 1.0,
            *ArrayUtils::ArrayOp::element_multiply(*h11, *h22, dtag), 1.0,
            *ArrayUtils::ArrayOp::element_multiply(*h12, *h12, dtag), -1.0,
            dtag);

        /*
        Part 2.3: Compute the surface normal as (Dx \cross Dy)/sqrt(deth)
            double N[3];
            Operators::cross(N, dx_z, dy_z);
            for (int n = 0; n < 3; n++)
		        N[n] = N[n] * (1/sqrt(deth));
         */
        Operators::SqrtFunctor sqrt_func;
        Operators::InverseFunctor inverse_func;
        auto inv_deth = ArrayUtils::ArrayOp::cloneCopy(*deth, dtag);
        ArrayUtils::ArrayOp::apply(*inv_deth, inverse_func, dtag); // deth -> 1/deth
        // Copy 1/deth because it is needed in finalizeVelocity functions
        auto inv_sqrt_deth = ArrayUtils::ArrayOp::cloneCopy(*inv_deth, dtag); 
        ArrayUtils::ArrayOp::apply(*inv_sqrt_deth, sqrt_func, dtag); // 1/deth -> 1/sqrt(deth)
        auto surface_normal = 
            ArrayUtils::ArrayOp::element_multiply(
                *inv_sqrt_deth, // 1/sqrt(deth)
                *ArrayUtils::ArrayOp::element_cross(*z_dx, *z_dy, dtag), // Dx \cross Dy
                dtag
            );

        /*
         * Part 2.4: Compute zdot and zndot as needed using specialized helper functions
         */
        // Clone another 1D array to create zndot, which is also 1D.
        auto zndot = ArrayUtils::ArrayOp::clone(*h11); 
        ArrayUtils::ArrayOp::assign(*zndot, 0.0, dtag); // XXX - Should not be needed
        finalizeVelocity(ModelOrderTag(), *zndot, zdot_array, *_reisz, *surface_normal, *inv_deth, dtag);

        /*
         * Part 2.5: Compute V from zndot and vorticity:
         * w1 = w_view(i, j, 0); 
         * w2 = w_view(i, j, 1);
         * V_view(i, j, 0) = zndot * zndot 
         *                     - 0.25*(h22*w1*w1 - 2.0*h12*w1*w2 + h11*w2*w2)/deth 
         *                     - 2*g*z_view(i, j, 2);
         */
        auto w1 = ArrayUtils::ArrayOp::copyDim(w_array, 0, dtag);
        auto w2 = ArrayUtils::ArrayOp::copyDim(w_array, 1, dtag);
        auto z2 = ArrayUtils::ArrayOp::copyDim(z_array, 2, dtag);
        auto zndot_squared = ArrayUtils::ArrayOp::element_multiply(*zndot, *zndot, dtag);
        auto h22_w1_w1 = ArrayUtils::ArrayOp::element_multiply(
            *h22,
            *ArrayUtils::ArrayOp::element_multiply(*w1, *w1, dtag),
            dtag
        );
        auto h12_w1_w2 = ArrayUtils::ArrayOp::element_multiply(
            *h12,
            *ArrayUtils::ArrayOp::element_multiply(*w1, *w2, dtag),
            dtag
        );
        auto h11_w2_w2 = ArrayUtils::ArrayOp::element_multiply(
            *h11,
            *ArrayUtils::ArrayOp::element_multiply(*w2, *w2, dtag),
            dtag
        );

        // Compute (h22*w1*w1 - 2.0*h12*w1*w2 + h11*w2*w2)/deth
        ArrayUtils::ArrayOp::update(*h22_w1_w1, 1.0, *h12_w1_w2, -2.0, *h11_w2_w2, 1.0, dtag);
        auto inner_part = ArrayUtils::ArrayOp::element_multiply(*h22_w1_w1, *inv_deth, dtag);

        // Compute V_view(i, j, 0) = zndot * zndot 
        //                     - 0.25*(h22*w1*w1 - 2.0*h12*w1*w2 + h11*w2*w2)/deth 
        //                     - 2*g*z_view(i, j, 2);
        // Result stored in zndot_squared
        ArrayUtils::ArrayOp::update(*zndot_squared, 1.0, *inner_part, -0.25, *z2, -2.0*_g, dtag);
        // Copy result into _V
        ArrayUtils::ArrayOp::copy(*_V, *zndot_squared, dtag);


        // 3. Phase 3: Halo V and apply boundary condtions on it, then calculate
        // central differences of V, laplacians for artificial viscosity, and
        // put it all together to calcualte the final vorticity derivative.

        // Halo V and correct any boundary condition corrections so that we can 
        // compute finite differences correctly.
        if constexpr (std::is_same_v<mesh_type_tag, Mesh::Structured>)
        {
            _v_halo->gather( execution_space(), *_V->array());
            _bc.applyField( _pm.mesh(), *_V->array(), 1 );
        }
        else if constexpr (std::is_same_v<mesh_type_tag, Mesh::Unstructured>)
        {
            // XXX - Unstructured gather operation
            throw std::invalid_argument("ZModel::computeHaloedDerivatives: Not yet implemented for unstructured meshes.");
        }

        /**
         * Compute:
         * 
         * Kokkos::parallel_for( "Interface Vorticity", policy, 
              KOKKOS_LAMBDA(int i, int j) {
                double dx_v = Operators::Dx(V_view, i, j, 0, dx);
                double dy_v = Operators::Dy(V_view, i, j, 0, dy);
                double lap_w0 = Operators::laplace(w_view, i, j, 0, dx, dy);
                double lap_w1 = Operators::laplace(w_view, i, j, 1, dx, dy);
                wdot_view(i, j, 0) = A * dx_v + mu * lap_w0;
                wdot_view(i, j, 1) = A * dy_v + mu * lap_w1;
            });
         */
        auto dx_v = _pm.mesh().Dx(*_V, _dx);
        auto dy_v = _pm.mesh().Dy(*_V, _dy);
        auto lap_w = _pm.mesh().laplace(w_array, _dx, _dy);

        // Compute wdot0 and wdot1
        auto wdot0 = ArrayUtils::ArrayOp::copyDim(*lap_w, 0, dtag);
        ArrayUtils::ArrayOp::update(*wdot0, _mu, *dx_v, _A, dtag);
        auto wdot1 = ArrayUtils::ArrayOp::copyDim(*lap_w, 1, dtag);
        ArrayUtils::ArrayOp::update(*wdot1, _mu, *dy_v, _A, dtag);

        // Copy results into wdot
        ArrayUtils::ArrayOp::copyDim(wdot_array, 0, *wdot0, 0, dtag);
        ArrayUtils::ArrayOp::copyDim(wdot_array, 1, *wdot1, 0, dtag);
    }

    std::shared_ptr<mesh_array_type> getOmega()
    {
        return _omega;
    }

    template <class l2g_type, class View>
    void printView(l2g_type local_L2G, View z, int option, int DEBUG_X, int DEBUG_Y) const
    {
        int dims = z.extent(2);

        std::array<long, 2> rmin, rmax;
        for (int d = 0; d < 2; d++) {
            rmin[d] = local_L2G.local_own_min[d];
            rmax[d] = local_L2G.local_own_max[d];
        }
	    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

        using execution_space = typename ProblemManagerType::execution_space;
        Kokkos::parallel_for("print views",
            Cabana::Grid::createExecutionPolicy(remote_space, execution_space()),
            KOKKOS_LAMBDA(int i, int j) {
            
            int local_li[2] = {i, j};
            int local_gi[2] = {0, 0};   // global i, j
            local_L2G(local_li, local_gi);
            if (option == 1){
                if (dims == 3) {
                    printf("%d %d %.12lf %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }
                else if (dims == 2) {
                    printf("%d %d %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                }
                else if (dims == 1) {
                    printf("%d %d %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0));
                }
            }
            else if (option == 2) {
                if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                    if (dims == 3) {
                    printf("%d %d %.12lf %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1), z(i, j, 2));
                    }
                    else if (dims == 2) {
                        printf("%d %d %.12lf %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0), z(i, j, 1));
                    }
                    else if (dims == 1) {
                    printf("%d %d %.12lf\n", local_gi[0], local_gi[1], z(i, j, 0));
                }
                }
            }
        });
    }

  private:
    const ProblemManagerType & _pm;
    const BoundaryCondition & _bc;
    const BRSolverType *_br;
    double _dx, _dy;
    double _A, _g, _mu;
    const int _heffte_configuration;
    std::shared_ptr<mesh_array_type> _V;
    std::shared_ptr<halo_type> _v_halo;
    std::shared_ptr<mesh_array_type> _omega;

    /* XXX Make this conditional on not being the high-order model */ 
    std::shared_ptr<mesh_array_type> _reisz;
    std::shared_ptr<mesh_array_type> _C1, _C2; 
    std::shared_ptr<Cabana::Grid::Experimental::HeffteFastFourierTransform<Cabana::Grid::Node, mesh_type, double, memory_space, execution_space, Cabana::Grid::Experimental::Impl::FFTBackendDefault>> _fft;
}; // class ZModel

} // namespace Beatnik

#endif // BEATNIK_ZMODEL_HPP
