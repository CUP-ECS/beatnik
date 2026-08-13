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
#
# unit tier -- SELF-VALIDATING STANDALONE TESTS
# =============================================
#
# These are NOT gtest tests and are deliberately not registered through
# cmake/test_harness/test_harness.cmake. That harness generates one binary per
# Kokkos backend from a `tst<Name>.hpp` and links gtest, which is right for a
# test that wants a backend sweep. The tests here instead:
#
#   * run on Kokkos::DefaultExecutionSpace (one binary, no `_SERIAL`/`_HIP`
#     name suffix), and
#   * decide their own pass/fail and RETURN NON-ZERO ON FAILURE.
#
# The second property is the load-bearing one. In spack mode there is no build
# tree, so there is no ctest to interpret a test's output -- the gate wrapper
# already walks an installed manifest and launches binaries directly, and the
# unit tier has to work the same way. A binary that is authoritative about its
# own verdict works in BOTH modes at once: ctest's default success criterion is
# exit code zero, so `add_test` below needs no PASS_REGULAR_EXPRESSION and no
# other special handling. See tests/unit_tests/Beatnik_TestAssert.hpp.
#
# Both registration styles can coexist; a gtest-harness `unit` test added via
# `Beatnik_add_tests(... LABEL unit ...)` in tests/CMakeLists.txt lands in the
# same tier and the same manifest.
#
# To add a test: drop `Beatnik_Test_<Name>.cpp` in this directory and add it to
# the list below. The target name is the file stem, which is the existing
# `Beatnik_Test_<Name>` convention (cf. Beatnik_Test_PythonCompare, which is
# likewise registered by hand rather than through the harness).
############################################################################

set(BEATNIK_UNIT_TEST_SOURCES
  # T1b: icosphere generation, mesh geometry, adjacency consistency.
  Beatnik_Test_MeshGeometry.cpp
  # T2b: the seven surface differential operators, and updateSheetVector.
  Beatnik_Test_T2bOperators.cpp
  )

# The unit tier runs at ONE rank. A multi-rank unit test would have to reduce
# its failure count before reporting a verdict (the recorder is rank-local by
# design), which is a decision for the task that first needs one -- and the
# rank sweep is the `regression` tier's job, not this one's.
set(_beatnik_unit_ranks 1)

foreach(_src ${BEATNIK_UNIT_TEST_SOURCES})
  get_filename_component(_target ${_src} NAME_WE)

  add_executable(${_target} ${_src})
  target_link_libraries(${_target} PRIVATE Beatnik MPI::MPI_CXX)
  target_include_directories(${_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # Launched under mpiexec even at one rank, so the test exercises the same
  # MPI_Init path the real runs do rather than a special-cased serial one.
  add_test(NAME ${_target} COMMAND
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_beatnik_unit_ranks}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${_target} PROPERTIES
    LABELS unit
    PROCESSORS ${_beatnik_unit_ranks}
    ENVIRONMENT OMP_NUM_THREADS=1)

  # Record the target for the installed-binary path's manifest (see
  # tests/CMakeLists.txt). Single-sourced from the same loop that applied the
  # label, so the two cannot drift.
  set_property(GLOBAL APPEND PROPERTY BEATNIK_UNIT_TARGETS ${_target})

  # +testing installs the tier AS A TIER, so a later task's unit test inherits
  # this for free. share/Beatnik/tests is the harness convention and is what
  # the spack package prepends to PATH.
  if(Beatnik_INSTALL_TEST_EXECUTABLES)
    install(TARGETS ${_target}
            RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
  endif()
endforeach()
