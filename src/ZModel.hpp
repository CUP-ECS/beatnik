/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 *
 * @section DESCRIPTION
 * ZModel class that handles computing derivatives of ionterface position and
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
#include <ArtificialViscosity.hpp>

namespace Beatnik
{


// Type tags designating difference orders of the 
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

//class BRSolver> 
//
template <class ExecutionSpace, class MemorySpace, class MethodOrder>
class ZModel
{
  public:
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using Node = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      MemorySpace>;

    // Meaningless type for now until we have 3D support in.
    using halo_type = Cajita::Halo<MemorySpace>;
    using mesh_type = Mesh<ExecutionSpace, MemorySpace>;

    ZModel( const std::unique_ptr<pm_type>& pm,
            BoundaryCondition &bc,
            ArtificialViscosity &av,
            double A,
            double g )
        : _pm( pm )
        , _bc( bc )
        , _av( av )
        , _A( A )
        , _g( g )
    {
        // Need the node triple layout for storing vector normals and the 
        // node double layout for storing x and y surface derivative
        auto node_pair_layout =
            Cajita::createArrayLayout( _pm->mesh()->localGrid(), 2, Cajita::Node() );
        auto node_triple_layout =
            Cajita::createArrayLayout( _pm->mesh()->localGrid(), 3, Cajita::Node() );
        auto node_scalar_layout =
            Cajita::createArrayLayout( _pm->mesh()->localGrid(), 3, Cajita::Node() );

        // We do lots of calculations with these derivatives, but they're only
        // used after the velocity calculations are done so they can probably be
        // cached in local variables.

        // XXX make this condtional about having been given a BR type
        _ubar = Cajita::createArray<double, MemorySpace>(
            "ubar", node_triple_layout );

        // XXX make this condtional about having been given a BR type
        _ueps = Cajita::createArray<double, MemorySpace>(
            "ueps", node_triple_layout );

        // XXX make this condtional about having been given a BR type
        _V = Cajita::createArray<double, MemorySpace>(
            "ueps", node_scalar_layout );

        Cajita::ArrayOp::assign( *_ubar, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *_ueps, 0.0, Cajita::Ghost() );
    }

    // This depends on the 
    double computeMinTimestep(double delta_t)
    {
        return delta_t;
    }

    void computeVelocities(Order::Low)
    {
    }

    void computeVelocities(Order::Medium)
    {
    }

    void computeVelocities(Order::High)
    {
    }

    // Compute the final interface velocities and normalized BR velocities
    // from the previously computed Fourier and/or BR velocities and the surface
    // normal based on  the order of technique we're using.
    KOKKOS_INLINE_FUNCTION
    void finalizeVelocity(Order::Low, int i, int j, double norm[3],
                          node_array zdot, double &un_local)
    {
    }

    KOKKOS_INLINE_FUNCTION
    void finalizeVelocity(Order::Medium, int i, int j, double norm[3],
                          node_array zdot, double &un_local)
    {
    }

    KOKKOS_INLINE_FUNCTION
    void finalizeVelocity(Order::High, int i, int j, double norm[3],
                          node_array zdot, double &zndot)
    {
    }

    void computeDerivatives( node_array zdot, node_array wdot )
    { 
        // 1. Compute the interface and vorticity velocities using 
        // the supplied methods.
        computeVelocities(MethodOrder());

        // 2. Halo the positions and vorticity so we can compute surface normals
        // and vorticity laplacians.
        _pm.gather();

        double g = _g;
        double A = _A;
        // 3. Now process those into final interface position derivatives
        //    and the information needed for calculating the vorticity derivative
#if 0
        parallel_for( ExecutionSpace, IndexSpace, KOKKOS_LABMDA(int i, int j) {
            //  3.1 Compute Dx and Dy of z and w by fourth-order central 
            //      differencing
            double dx_z[3], dx_y[3];
            for (int n = 0; n < 3; n++) {
                dx_z[n] = Dx(z(i, j, n), dx);
                dy_z[n] = Dy(z(i, j, n), dy);
	    }

            //  3.2 Compute h11, h12, h22, and det_h from Dx and Dy
            double h11 = dot(dx_z, dx_z);
            double h12 = dot(dx_z, dy_z);
            double h22 = dot(dy_z, dy_y);
            double deth = h11*h22 - h12*h12;

            //  3.3 Compute the surface normal as (Dx \cross Dy)/sqrt(deth)
            double N[3];
            cross(N, dx_z, dy_z);
            for (int n = 0; n < 3; n++)
		N[n] /= fsqrt(deth);

            //  3.4 Compute zdot and zndot using specialized helper functions
            finalizeVelocity(MethodOrder(), i, j, N, zdot, zndot);

            //  3.5 Compute V from zndot and vorticity 
	    w1 = w(i, j, 0); w2 = w(i, j, 1);
	    V(i, j, 0) = zndot * zndot 
                         - 0.25*(h22*w1*w1 - 2.0*h12*w1*w2 + h11.*w2.^2)./deth 
                         - 2*g*z(i, j, 2);   
        });
#endif
        // 4. Halo V and apply boundary condtions, since we need it for central
        //    differencing
        
        // 5. Compute the final vorticity derivative
#if 0
        parallel_for( ExecutionSpace, IndexSpace, KOKKOS_LABMDA(int i, int j) {
            wdot(i, i, 0) = Dx(V(i, j, 0), dx) + av(i, j, w, 0);
            wdot(i, i, 1) = Dy(V(i, j, 0), dy) + av(i, j, w, 1);
        }
#endif
    }

  private:
    BoundaryCondition & _bc;
    ArtificialViscosity & _av;
    double _g, _A;
    std::unique_ptr<pm_type> & _pm;
    std::shared_ptr<node_array> _ubar, _ueps, _V; // intermediate state for 
                                                  // calculation of derivatives
}; // class ZModel

} // namespace Beatnik

#endif // BEATNIK_ZMODEL_HPP
