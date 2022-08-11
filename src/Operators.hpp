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

#ifndef BEATNIK_OPERATORS_HPP
#define BEATNIK_OPERATORS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace Beatnik
{

/* Simple vector and finite difference operators needed by the ZModel code.
 * Note that we use higher-order difference operators as the highly-variable
 * curvature of surface can make lower-order operators inaccurate */
namespace Operators
{
    /* Fourth order central difference calculation for derivatives along the 
     * interface surface */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    double Dx(ViewType f, int i, int j, int d, double dx) 
    {
        return (f(i - 2, j, d) - 8.0*f(i - 1, j, d) + 8.0*f(i + 1, j, d) - f(i + 2, j, d)) / (12.0 * dx);
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
        //return (f(i + 1, j, d) + f(i -1, j, d) + f(i, j+1, d) + f(i, j-1,d) - 4.0 * f(i, j, d)) / (dx * dy);
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

    template <class VorticityView, class PositionView>
    KOKKOS_INLINE_FUNCTION
    void BR(double out[3], VorticityView w, PositionView z, double epsilon,
            double dx, double dy, int i, int j, int k, int l)
    {
        double omega[3], zdiff[3], zsize;
        zsize = 0.0;
        for (int d = 0; d < 3; d++) {
            omega[d] = w(k, l, 1) * Dx(z, k, l, d, dx) - w(k, l, 0) * Dy(z, k, l, d, dy);
            zdiff[d] = z(i, j, d) - z(k, l, d);
            zsize += zdiff[d] * zdiff[d];
        }  
        zsize = pow(zsize * zsize + epsilon * epsilon, 1.5);
        for (int d = 0; d < 3; d++) {
            zdiff[d] /= zsize;
        }
        cross(out, omega, zdiff);
        for (int d = 0; d < 3; d++) {  
            out[d] /= (-4.0 * Kokkos::Experimental::pi_v<double>);
        }
    }

    
}; // namespace operator

}; // namespace beatnik

#endif // BEATNIK_OPERATORS_HPP

