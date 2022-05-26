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

namespace Beatnik
{


// Type tags designating difference orders of the 
namespace Order
{
    struct Low {};

    struct Medium {};

    struct High {};
} // namespace Order

/* Simple vector and finite difference operators needed by the ZModel code.
 * Note that we use higher-order difference operators as the highly-variable
 * curvature of surface can make lower-order operators inaccurate */
namespace Operator
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
    double Dy(ViewType f, int i, int j, int d, double dx)
    {
        return (f(i, j - 2, d) - 8.0*f(i, j - 1, d) + 8.0*f(i, j + 1, d) - f(i, j + 2, d)) / (12.0 * dx);
    }
 
    /* 9-point laplace stencil operator for computing artificial viscosity */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    double laplace(ViewType f, int i, int j, int d, double h) 
    {
        return (0.5*f(i+1, j, d) + 0.5*f(i-1, j, d) + 0.5*f(i, j+1, d) + 0.5*f(i, j-1, d) 
            + 0.25*f(i+1, j+1, d) + 0.25*f(i+1, j-1, d) + 0.5*f(i-1, j+1, d) + 0.5*f(i-1, j-1, d)
            - 3*f(i, j, d))/(h*h);
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
    using exec_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using pm_type = ProblemManager<ExecutionSpace, MemorySpace>;
    using device_type = typename pm_type::device_type; // Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cajita::UniformMesh<double, 2>; 

    using Node = Cajita::Node;

    using node_array =
        Cajita::Array<double, Cajita::Node, Cajita::UniformMesh<double, 2>,
                      MemorySpace>;

    using halo_type = Cajita::Halo<MemorySpace>;

    ZModel( const pm_type & pm, const BoundaryCondition &bc,
            double A, double g, double mu)
        : _pm( pm )
        , _bc( bc )
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
            Cajita::createArrayLayout( _pm.mesh().localGrid(), 3, Cajita::Node() );

        // Temporary used for central differencing of vorticities along the 
        // surface in calculating the vorticity derivative/
        _V = Cajita::createArray<double, MemorySpace>(
            "V", node_scalar_layout );

        /* Storage for the reisz transform of the vorticity. In the low and medium 
         * order models,
         * it is used to calculate the vorticity derivative. In the low order model,
         * it is also projected onto the surface normal to compute the interface velocity.
         * XXX Make this conditional on the model we run. */
        _reisz = Cajita::createArray<double, MemorySpace>( "reisz", node_double_layout );
        Cajita::ArrayOp::assign( *_reisz, 0.0, Cajita::Ghost() );

        /* We need a halo for _V so that we can do fourth-order central differencing on
         * it. This requires a depth 2 stencil with adjacent faces */
        int halo_depth = 2;
        _v_halo = Cajita::createHalo( Cajita::FaceHaloPattern<2>(),
                            halo_depth, *_V );

        /* If we're not the hgh order model, initialize the FFT solver and the working space
         * it will need. XXX figure out how to make this conditional on model order. */
        Cajita::Experimental::FastFourierTransformParams params;
        _M1 = Cajita::createArray<double, MemorySpace>("M1", node_double_layout);
        _M2 = Cajita::createArray<double, MemorySpace>("M2", node_double_layout);

        params.setAllToAll(true);
        params.setPencils(true);
        params.setReorder(false);
        _fft = Cajita::Experimental::createHeffteFastFourierTransform<double, device_type>(*node_double_layout, params);


    }

    double computeMinTimestep(double atwood, double g)
    {
        return 1.0/(25.0*sqrt(atwood * g));
    }

    /* Compute the velocities needed by the relevant Z-Model */
    void computeFourierVelocity() const
    {
        /* This is two FFTs on temporary arrays and an inverse FFT */

        /* Construct the temporary arrays M1 and M2 */
        auto local_grid = _pm.mesh().localGrid();
        auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );
        auto local_nodes = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());

        /* Get the views we'll be computing with in parallel loops */
        auto w = _pm.get( Cajita::Node(), Field::Vorticity() );
        auto M1 = _M1->view();
        auto M2 = _M1->view();
        auto reisz = _reisz->view();

        /* First put w into the real parts of M1 and M2 */
        Kokkos::parallel_for("Build FFT Arrays", 
                     Cajita::createExecutionPolicy(local_nodes, ExecutionSpace()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            M1(i, j, 0) = w(i, j, 0);
            M2(i, j, 0) = w(i, j, 1);
        });

        /* Now do the FFTs of vorticity */
        _fft->forward(*_M1, Cajita::Experimental::FFTScaleNone());
        _fft->forward(*_M2, Cajita::Experimental::FFTScaleNone());

        /* Now construct reisz from the weighted sum of those FFTs to take the inverse FFT. */
        parallel_for("Combine FFTs", 
                     Cajita::createExecutionPolicy(local_nodes, ExecutionSpace()), 
                     KOKKOS_LAMBDA(const int i, const int j) {
            int indicies[2] = {i, j};
            double location[2];
            local_mesh.coordinates( Cajita::Node(), indicies, location );
            double len = sqrt(location[0] * location[0] + location[1] * location[1]);
            if ((location[0] != 0) || (location[1]  != 0)) {
                reisz(i, j, 0) = (-M1(i, j, 1) * location[0] - M2(i, j, 1) * location[1]) / len;
                reisz(i, j, 1) = ( M1(i, j, 0) * location[0] + M2(i, j, 1) * location[1]) / len;
            } else {
                reisz(i, j, 0) = 0.0; 
                reisz(i, j, 1) = 0.0;
            }
        });
        /* Finally do the reverse transform to get the non-normalized fourier velocity.
         * We'll drop the imaginary part and project the real part onto the interface 
         * velocity later */
        _fft->reverse(*_reisz, Cajita::Experimental::FFTScaleNone());
    }

    /* Compute the velocities needed by the relevant Z-Model */
    void computeInterfaceVelocity() const
    {
    }

    /* For low order, we calculate the fourier velocity and then 
     * later finalize that once we have the normals into the interface
     * velocity */
    void computeVelocities(Order::Low) const
    {
        computeFourierVelocity();
    }

    /* For medium order, we calculate the fourier velocity that we later 
     * normalize for vorticity calculations and directly compute the 
     * interface velocity (zdot) using a fast multipole method. */
    void computeVelocities(Order::Medium) const
    {
        computeFourierVelocity();
        computeInterfaceVelocity();
    }

    /* For high order, we just directly compute the interface velocity (zdot)
     * using a fast multipole method and later normalize that for use in the
     * vorticity calculation. */
    void computeVelocities(Order::High) const
    {
        computeInterfaceVelocity();
    }

    // Compute the final interface velocities and normalized BR velocities
    // from the previously computed Fourier and/or BR velocities and the surface
    // normal based on  the order of technique we're using.
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION 
    static void finalizeVelocity(Order::Low, double &zndot, ViewType zdot, int i, int j, 
                          double reisz, double norm[3], double deth) 
    {
        zndot = reisz / deth;
        for (int d = 0; d < 3; d++)
            zdot(i, j, d) = zndot * norm[d];
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    static void finalizeVelocity(Order::Medium, double &zndot, 
        [[maybe_unused]] ViewType zdot, [[maybe_unused]] int i, [[maybe_unused]] int j,
        double reisz, [[maybe_unused]] double norm[3], double deth) 
    {
        zndot = reisz / deth;
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    static void finalizeVelocity(Order::High, double &zndot, ViewType zdot, int i, int j,
                         [[maybe_unused]] double reisz, double norm[3], [[maybe_unused]] double deth)
    {
        double interface_velocity[3] = {zdot(i, j, 0), zdot(i, j, 1), zdot(i, j, 2)};
        zndot = Operator::dot(norm, interface_velocity);
    }
    
    template <class PositionView, class VorticityView>
    void computeDerivatives( PositionView z, VorticityView w, 
        PositionView zdot, VorticityView wdot) const
    { 
        // 1. Compute the interface and vorticity velocities using 
        // the supplied methods in terms of the unit mesh.
        computeVelocities(MethodOrder());

        // 2. Halo the positions and vorticity so we can compute surface normals
        // and vorticity laplacians.
        _pm.gather();

        double g = _g;
        double A = _A;

        // 3. Now process those into final interface position derivatives
        //    and the information needed for calculating the vorticity derivative
        auto reisz = _reisz->view();
        auto V = _V->view();

        auto local_grid = _pm.mesh().localGrid();
        auto own_node_space = local_grid->indexSpace(Cajita::Own(), Cajita::Node(), Cajita::Local());
        Kokkos::parallel_for( "Interface Velocity",  
            createExecutionPolicy(own_node_space, ExecutionSpace()), 
            KOKKOS_LAMBDA(int i, int j) {
            //  3.1 Compute Dx and Dy of z and w by fourth-order central 
            //      differencing. Because we're on a unit mesh, dx and dy are 1.
            double dx_z[3], dy_z[3];

            for (int n = 0; n < 3; n++) {
               dx_z[n] = Operator::Dx(z, i, j, n, 1.0);
               dy_z[n] = Operator::Dy(z, i, j, n, 1.0);
            }

            //  3.2 Compute h11, h12, h22, and det_h from Dx and Dy
            double h11 = Operator::dot(dx_z, dx_z);
            double h12 = Operator::dot(dx_z, dy_z);
            double h22 = Operator::dot(dy_z, dy_z);
            double deth = h11*h22 - h12*h12;

            //  3.3 Compute the surface normal as (Dx \cross Dy)/sqrt(deth)
            double N[3];
            Operator::cross(N, dx_z, dy_z);
            for (int n = 0; n < 3; n++)
		N[n] /= sqrt(deth);

            //  3.4 Compute zdot and zndot as needed using specialized helper functions
            double zndot;
            finalizeVelocity(MethodOrder(), zndot, zdot, i, j, reisz(i, j, 0), N, deth );

            //  3.5 Compute V from zndot and vorticity 
	    double w1 = w(i, j, 0); 
            double w2 = w(i, j, 1);

	    V(i, j, 0) = zndot * zndot 
                         - 0.25*(h22*w1*w1 - 2.0*h12*w1*w2 + h11*w2*w2)/deth 
                         - 2*g*z(i, j, 2);   
        });
        // 4. Halo V and apply boundary condtions, since we need it for central
        //    differencing of V.
        _v_halo->gather( ExecutionSpace(), *_V );

        // 5. Compute the final vorticity derivative
        double mu = _mu;
        Kokkos::parallel_for( "Interface Vorticity",
            createExecutionPolicy(own_node_space, ExecutionSpace()), 
            KOKKOS_LAMBDA(int i, int j) {
            wdot(i, i, 0) = A * Operator::Dx(V, i, j, 0, 1.0) + mu * Operator::laplace(w, i, j, 0, 1.0);
            wdot(i, i, 1) = A * Operator::Dy(V, i, j, 0, 1.0) + mu * Operator::laplace(w, i, j, 1, 1.0);
        });
    }

  private:
    const BoundaryCondition & _bc;
    double _g, _A, _mu;
    const pm_type & _pm;
    std::shared_ptr<node_array> _V;
    std::shared_ptr<halo_type> _v_halo;

    /* XXX Make this conditional on not being the high-order model */ 
    std::shared_ptr<node_array> _reisz;
    std::shared_ptr<node_array> _M1, _M2; 
    std::shared_ptr<Cajita::Experimental::HeffteFastFourierTransform<Cajita::Node, mesh_type, double, device_type, Cajita::Experimental::Impl::FFTBackendDefault>> _fft;

    /* XXX Conditional declarations for medium/high-order models */

}; // class ZModel

} // namespace Beatnik

#endif // BEATNIK_ZMODEL_HPP
