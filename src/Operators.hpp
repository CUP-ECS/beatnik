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
 * @author Jason Stewart <jastewart@unm.edu>
 *
 * @section DESCRIPTION
 * Supporting functions for Z-Model calculations, primarily Simple differential 
 * and other mathematical operators but also some utility functions that 
 * we may want to later contribute back to Cabana_Grid or other supporting libraries.
 */

#ifndef BEATNIK_OPERATORS_HPP
#define BEATNIK_OPERATORS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace Beatnik
{

/**
 * Operators that work on Cabana::Arrays assuming an array 
 * derived from a structured, 2D mesh
 */

/* Simple vector and finite difference operators needed by the ZModel code.
 * Note that we use higher-order difference operators as the highly-variable
 * curvature of surface can make lower-order operators inaccurate */
namespace Operators
{
    //---------------------------------------------------------------------------//
    //! Square root functor
    struct SqrtFunctor
    {
        //! Constructor.
        SqrtFunctor( ) {}

        KOKKOS_INLINE_FUNCTION
        double operator()( double val ) const
        {
            return std::sqrt(val);
        }
    };

    /* Fourth order central difference calculation for derivatives along the 
     * interface surface */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    double Dx(ViewType f, int i, int j, int d, double dx)
    {
        return (f(i - 2, j, d) - 8.0*f(i - 1, j, d) + 8.0*f(i + 1, j, d) - f(i + 2, j, d)) / (12.0 * dx);
        //return (f(i + 1, j, d) - f(i - 1, j, d)) / (2.0 * dx);
    } 

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    void Dx(double out[3], ViewType f, int i, int j, double dx) 
    {
        for (int d = 0; d < 3; d++) {
            out[d] = Dx(f, i, j, d, dx);
        }
    } 

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    double Dy(ViewType f, int i, int j, int d, double dy)
    {  
        return (f(i, j - 2, d) - 8.0*f(i, j - 1, d) + 8.0*f(i, j + 1, d) - f(i, j + 2, d)) / (12.0 * dy);
        //return (f(i, j+1, d) - f(i, j-1, d)) / (2.0 * dy);
    }
 
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    void Dy(double out[3], ViewType f, int i, int j, double dy) 
    {
        for (int d = 0; d < 3; d++) {
            out[d] = Dy(f, i, j, d, dy);
        }
    } 

    /* 9-point laplace stencil operator for computing artificial viscosity */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    double laplace(ViewType f, int i, int j, int d, double dx, double dy) 
    {
        return (0.5*f(i+1, j, d) + 0.5*f(i-1, j, d) + 0.5*f(i, j+1, d) + 0.5*f(i, j-1, d)
                + 0.25*f(i+1, j+1, d) + 0.25*f(i+1, j-1, d) + 0.25*f(i-1, j+1, d) + 0.25*f(i-1, j-1, d)
                - 3*f(i, j, d))/(dx*dy);
//        return (f(i + 1, j, d) + f(i -1, j, d) + f(i, j+1, d) + f(i, j-1,d) - 4.0 * f(i, j, d)) / (dx * dy);
    }

    KOKKOS_INLINE_FUNCTION
    double dot(double u[3], double v[3]) 
    {
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    }

    KOKKOS_INLINE_FUNCTION
    void cross(double N[3], double u[3], double v[3]) 
    {
        N[0] = u[1]*v[2] - u[2]*v[1];
        N[1] = u[2]*v[0] - u[0]*v[2];
        N[2] = u[0]*v[1] - u[1]*v[0];
    }

    /* Compute the Birchorff Rott force exerted on an i/j point with given location
     * by a k/l point with given vorticity, with an additional position offset 
     * (to * take care of periodic boundary contitions) */
    template <class PositionView>
    KOKKOS_INLINE_FUNCTION
    void BR(double out[3], PositionView z, PositionView z2, PositionView omega_view,
            double epsilon, double dx, double dy, double weight, int i, int j, int k, int l,
            double offset[3]) 
    {
        double omega[3], zdiff[3], zsize;
        zsize = 0.0;
        for (int d = 0; d < 3; d++) {
            // omega[d] = w2(k, l, 1) * Dx(z2, k, l, d, dx) - w2(k, l, 0) * Dy(z2, k, l, d, dy);
            omega[d] = omega_view(k, l, d);
            zdiff[d] = z(i, j, d) - (z2(k, l, d) + offset[d]);
            zsize += zdiff[d] * zdiff[d];
        }  
        zsize = pow(zsize + epsilon, 1.5); // matlab code doesn't square epsilon
        for (int d = 0; d < 3; d++) {
            zdiff[d] /= zsize;
        }
        cross(out, omega, zdiff);
        for (int d = 0; d < 3; d++) {  
            out[d] *= (dx * dy * weight) / (-4.0 * Kokkos::numbers::pi_v<double>);
        }
    }

    template <class PositionSlice, class OmegaSlice, class WeightSlice>
    KOKKOS_INLINE_FUNCTION
    void BR(double out[3], int my_id, int neighbor_id,
            PositionSlice p, OmegaSlice o, WeightSlice w, 
            double epsilon, double dx, double dy)
    {
        double omega[3], zdiff[3], zsize;
        zsize = 0.0;
        for (int d = 0; d < 3; d++) {
            omega[d] = o(neighbor_id, d);
            zdiff[d] = p(my_id, d) - p(neighbor_id, d);
            zsize += zdiff[d] * zdiff[d];
        } 
          
        zsize = pow(zsize + epsilon, 1.5); // matlab code doesn't square epsilon
        
        for (int d = 0; d < 3; d++) {
            zdiff[d] /= zsize;
        }
        cross(out, omega, zdiff);
        for (int d = 0; d < 3; d++) {  
            out[d] *= (dx * dy * w(neighbor_id)) / (-4.0 * Kokkos::numbers::pi_v<double>);
        }
    } 

    template <long M, long N>
    Cabana::Grid::IndexSpace<M + N> crossIndexSpace(
            const Cabana::Grid::IndexSpace<M>& index_space1,
            const Cabana::Grid::IndexSpace<N>& index_space2)
    {
        std::array<long, M + N> range_min;
        std::array<long, M + N> range_max;
        for ( int d = 0; d < M; ++d ) {
            range_min[d] = index_space1.min( d );
            range_max[d] = index_space1.max( d );
        }

        for ( int d = M; d < M + N; ++d ) {
            range_min[d] = index_space2.min( d - M );
            range_max[d] = index_space2.max( d - M );
        }

        return Cabana::Grid::IndexSpace<M + N>( range_min, range_max );
    }
    
}; // namespace operator

}; // namespace beatnik

#endif // BEATNIK_OPERATORS_HPP
