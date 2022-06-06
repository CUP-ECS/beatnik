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
    double Dy(ViewType f, int i, int j, int d, double dy)
    {
        return (f(i, j - 2, d) - 8.0*f(i, j - 1, d) + 8.0*f(i, j + 1, d) - f(i, j + 2, d)) / (12.0 * dy);
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
}; // namespace operator

}; // namespace beatnik

#endif // BEATNIK_OPERATORS_HPP

