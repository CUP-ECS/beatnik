/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 *
 * @section DESCRIPTION
 * Boundary Conditions for Beatnik prototype interface model
 */

#ifndef BEATNIK_ARTIFICIALVISCOSITY_HPP
#define BEATNIK_ARTIFICIALVISCOSITY_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace Beatnik
{

/**
 * @class ArtificialViscosity
 * @brief Functor that computes artificial viscosity in the z-model calculation
 * using Kokkos inline Functions
 */

struct ArtificialViscosity
{
    using node_array = Cajita::Array<Cajita::Node, 2>;
    using Position = Field::Position;

    template <class PType>
    KOKKOS_INLINE_FUNCTION void operator()( int i, int j, Node, Position, PType& p, 
                                            const int gi, const int gj,
                                            const int i, const int j ) const
    {
        
    }

    ArtificialViscosity(double mu) 
    : _mu(mu)
    {}

    double _mu;
};

} // namespace Beatnik

#endif // BEATNIK_ARTIFICIALVISCOSITY_HPP
