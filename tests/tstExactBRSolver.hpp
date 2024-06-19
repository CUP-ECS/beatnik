#ifndef TSTEXACTBRSOLVER_HPP
#define TSTEXACTBRSOLVER_HPP

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <Mesh.hpp>

#include <mpi.h>

#include "tstDriver.hpp"


class CalcPi {
 public:
  int mult(int a, int b);
};

TEST_F(CalcPi, multTest)
{
  ASSERT_EQ(6, CalcPi::mult(2, 3));
}



#endif // TSTEXACTBRSOLVER_HPP
