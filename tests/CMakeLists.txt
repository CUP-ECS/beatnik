############################################################################
# Copyright (c) 2025 by the Beatnik authors                                #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Beatnik library. Beatnik is distributed under a #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

include(${TEST_HARNESS_DIR}/test_harness.cmake)

# set(NOBACKEND_TESTS
#   Kernels
#   )

set(SERIAL_TESTS
  # None currently
  )

set(MPI_TESTS
  BoundaryCondition
  Mesh
  )

# Beatnik_add_tests_nobackend(PACKAGE Beatnik NAMES ${NOBACKEND_TESTS})

Beatnik_add_tests(PACKAGE Beatnik NAMES ${SERIAL_TESTS})

Beatnik_add_tests(MPI PACKAGE Beatnik NAMES ${MPI_TESTS})
