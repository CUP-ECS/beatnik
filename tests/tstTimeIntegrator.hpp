#ifndef _TSTTIMEINTEGRATOR_HPP_
#define _TSTTIMEINTEGRATOR_HPP_

#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <Solver.hpp>

#include <mpi.h>

#include "TestingBase.hpp"

namespace BeatnikTest
{

template <class T>
class TimeIntegratorTest : public TestingBase<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    using pm_type = Beatnik::ProblemManager<ExecutionSpace, MemorySpace>;
    using zm_type_h = Beatnik::ZModel<ExecutionSpace, MemorySpace, Beatnik::Order::High, Beatnik::Params>;
    using ti_type = Beatnik::TimeIntegrator<ExecutionSpace, MemorySpace, zm_type_h>;

  protected:
    std::shared_ptr<pm_type> pm_correct_;
    //std::shared_ptr<>

    void SetUp() override
    {
        TestingBase<T>::SetUp();
    }

    void TearDown() override
    { 
        TestingBase<T>::TearDown();
    }

  public:
    
};

} // end namespace BeatnikTest

#endif // _TSTTIMEINTEGRATOR_HPP_
