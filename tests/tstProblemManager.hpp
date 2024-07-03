#ifndef _TSTPROBLEMMANGER_HPP_
#define _TSTPROBLEMMANGER_HPP_

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <ProblemManager.hpp>

#include <mpi.h>

#include "tstMesh.hpp"

namespace BeatnikTest
{

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

  protected:
    NullInitFunctor<2> createFunctor_;
    std::shared_ptr<pm_type> testPM_Periodic_;
    std::shared_ptr<pm_type> testPM_NonPeriodic_;

    void SetUp() override
    {
        MeshTest<T>::SetUp();
        this->testPM_Periodic_ =
            std::make_shared<pm_type>( *this->testMeshPeriodic_, BoundaryConditionTest::bc_periodic_, createFunctor_ );
        this->testPM_NonPeriodic_ =
            std::make_shared<pm_type>( *this->testMeshNonPeriodic_, BoundaryConditionTest::bc_non_periodic, createFunctor_ );
    }

    void TearDown() override
    {
        this->testMeshPeriodic_ = NULL;
        this->testMeshNonPeriodic_ = NULL;
        MeshTest<T>::TearDown();
    }

  public:
    template<class ProblemManager>
    void initializePositions(ProblemManager &pm)
    {
        auto mesh = pm.mesh();
        auto rank = mesh.rank();
        auto z = pm.get( Cabana::Grid::Node(), Beatnik::Field::Position() );
        // auto zspace = mesh->localGrid()->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
        //                                             Cabana::Grid::Local() );
        // Kokkos::parallel_for(
        //     "InitializePositions",
        //     createExecutionPolicy( zspace, ExecutionSpace ),
        //     KOKKOS_LAMBDA( const int i, const int j ) {
        //         for (int d = 0; d < 3; d++)
        //             z( i, j, d ) = rank * 1000 + i * 100 + j * 10 + d;
        //     } );
    }

};

} // end namespace BeatnikTest

#endif // _TSTPROBLEMMANAGER_HPP_
