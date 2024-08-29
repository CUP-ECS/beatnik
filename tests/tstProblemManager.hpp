#ifndef _TSTPROBLEMMANGER_HPP_
#define _TSTPROBLEMMANGER_HPP_

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstMesh.hpp"

template <std::size_t Dim>
class NullInitFunctor
{
  public:
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Position,
                     [[maybe_unused]] const int index[Dim],
                     [[maybe_unused]] const double x[Dim],
                     [[maybe_unused]] double& z1, 
                     [[maybe_unused]] double& z2, 
                     [[maybe_unused]] double& z3) const
    {
        return true;
    };

    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, Beatnik::Field::Vorticity,
                     [[maybe_unused]] const int index[Dim],
                     [[maybe_unused]] const double x[Dim],
                     [[maybe_unused]] double& w1,
                     [[maybe_unused]] double& w2 ) const
    {
        return true;
    };
};

template <class T>
class ProblemManagerTest : public MeshTest<T>
{

    using pm_type = Beatnik::ProblemManager<typename T::ExecutionSpace,
                                            typename T::MemorySpace>;

    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;

  public:
    NullInitFunctor<2> createFunctor_;
    std::shared_ptr<pm_type> testPM_;

    virtual void SetUp() override
    {
        MeshTest<T>::SetUp();
        this->testPM_ =
            std::make_shared<pm_type>( *this->testMesh_, createFunctor_ );
    }

    virtual void TearDown() override
    {
        this->testPM_ = NULL;
        MeshTest<T>::TearDown();
    }
};

#endif // _TSTPROBLEMMANAGER_HPP_
