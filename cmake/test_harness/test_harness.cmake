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

##--------------------------------------------------------------------------##
## Test tiers, backends, and rank sweep
##--------------------------------------------------------------------------##
# Every registered ctest case carries exactly one tier label:
#
#   regression  Full end-to-end runs that compose the whole pipeline.
#               THIS TIER IS THE SHIP GATE.
#   unit        Utilities, kernels, single-component or single-phase tests.
#               Diagnostic — informative, but does not gate a change.
#
# Test *names* additionally carry the Kokkos backend as a suffix (e.g.
# Beatnik_Test_Foo_MPI_SERIAL_np_4), so a backend is selected with -R:
#
#   ctest -L regression -R SERIAL     # the project-wide gate
#   ctest -L unit                     # diagnostics, all backends
#
# Pass LABEL <tier> to Beatnik_add_tests / Beatnik_add_tests_nobackend. It is
# required; there is deliberately no default, so nothing lands in or out of the
# gate by accident.
#
# MPI tests are parameterized over BEATNIK_TEST_MPI_RANKS. The gate requires
# ranks 1-6, which is the default. Lower it only for a machine that cannot
# oversubscribe, and say so in that system's systems/<system>/claude.md.
set(BEATNIK_TEST_MPI_RANKS "1;2;3;4;5;6" CACHE STRING
    "MPI rank counts each MPI test is registered at (gate requires 1;2;3;4;5;6)")

# Accumulates the target names of regression-tier tests so tests/CMakeLists.txt
# can emit a manifest for the installed-binary gate path, which has no build
# tree and therefore no ctest. Keeps the gate single-sourced.
#
# BEATNIK_UNIT_TARGETS does the same for the `unit` tier, which needs it for the
# same reason: scripts/<system>/unit_tests.* runs the whole tier from an install
# prefix and must discover the binaries rather than hard-code them.
# tests/unit_tests/CMakeLists.txt appends to it too, so one manifest covers both
# registration styles.
#
# Deliberately left UNSET rather than initialized to "": APPEND on a property
# already set to the empty string yields a leading empty list element, which
# would become a blank first line in the manifest.
get_property(_beatnik_gate_init GLOBAL PROPERTY BEATNIK_GATE_TARGETS SET)
if(_beatnik_gate_init)
  set_property(GLOBAL PROPERTY BEATNIK_GATE_TARGETS)
endif()
get_property(_beatnik_unit_init GLOBAL PROPERTY BEATNIK_UNIT_TARGETS SET)
if(_beatnik_unit_init)
  set_property(GLOBAL PROPERTY BEATNIK_UNIT_TARGETS)
endif()

include(FindPackageHandleStandardArgs)
find_program(VALGRIND_EXECUTABLE valgrind)
find_package_handle_standard_args(VALGRIND REQUIRED_VARS VALGRIND_EXECUTABLE)
if(VALGRIND_FOUND)
  set(VALGRIND_ARGS --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes --error-exitcode=1)
endif()

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
macro(Beatnik_add_tests_nobackend)
  cmake_parse_arguments(BEATNIK_UNIT_TEST "" "PACKAGE;LABEL" "NAMES" ${ARGN})
  if(BEATNIK_UNIT_TEST_NAMES AND NOT BEATNIK_UNIT_TEST_LABEL)
    message(FATAL_ERROR
      "Beatnik_add_tests_nobackend: LABEL is required (regression or unit)")
  endif()
  foreach(_test ${BEATNIK_UNIT_TEST_NAMES})
    set(_target ${BEATNIK_UNIT_TEST_PACKAGE}_${_test}_test)
    add_executable(${_target} tst${_test}.cpp ${TEST_HARNESS_DIR}/unit_test_main.cpp)
    target_link_libraries(${_target} PRIVATE ${BEATNIK_UNIT_TEST_PACKAGE} ${gtest_target})
    add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args})
    set_property(TEST ${_target} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
    set_property(TEST ${_target} PROPERTY LABELS ${BEATNIK_UNIT_TEST_LABEL})
    if(VALGRIND_FOUND)
      add_test(NAME ${_target}_valgrind COMMAND ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args})
      set_property(TEST ${_target}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
      # Valgrind runs are always diagnostic, never part of the gate.
      set_property(TEST ${_target}_valgrind PROPERTY LABELS unit)
    endif()
    if(BEATNIK_UNIT_TEST_LABEL STREQUAL regression)
      set_property(GLOBAL APPEND PROPERTY BEATNIK_GATE_TARGETS ${_target})
    elseif(BEATNIK_UNIT_TEST_LABEL STREQUAL unit)
      set_property(GLOBAL APPEND PROPERTY BEATNIK_UNIT_TARGETS ${_target})
    endif()
    if(Beatnik_INSTALL_TEST_EXECUTABLES)
      install(TARGETS ${_target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
    endif()
  endforeach()
endmacro()

##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
set(BEATNIK_TEST_DEVICES)
foreach(_device ${BEATNIK_SUPPORTED_DEVICES})
  if(Kokkos_ENABLE_${_device})
    list(APPEND BEATNIK_TEST_DEVICES ${_device})
    if(_device STREQUAL CUDA)
      list(APPEND BEATNIK_TEST_DEVICES CUDA_UVM)
    endif()
  endif()
endforeach()

macro(Beatnik_add_tests)
  cmake_parse_arguments(BEATNIK_UNIT_TEST "MPI" "PACKAGE;LABEL" "NAMES" ${ARGN})
  if(BEATNIK_UNIT_TEST_NAMES AND NOT BEATNIK_UNIT_TEST_LABEL)
    message(FATAL_ERROR
      "Beatnik_add_tests: LABEL is required (regression or unit)")
  endif()
  # Rank sweep comes from BEATNIK_TEST_MPI_RANKS, not from MPIEXEC_MAX_NUMPROCS.
  # The gate is defined as ranks 1-6, so it must not silently shrink on a host
  # that reports fewer slots — an under-parameterized gate is a false green.
  set(BEATNIK_UNIT_TEST_MPIEXEC_NUMPROCS ${BEATNIK_TEST_MPI_RANKS})
  set(BEATNIK_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND BEATNIK_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  if(BEATNIK_UNIT_TEST_MPI)
    set(BEATNIK_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/mpi_unit_test_main.cpp)
  else()
    set(BEATNIK_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/unit_test_main.cpp)
  endif()
  foreach(_device ${BEATNIK_TEST_DEVICES})
    set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
    file(MAKE_DIRECTORY ${_dir})
    foreach(_test ${BEATNIK_UNIT_TEST_NAMES})
      set(_file ${_dir}/tst${_test}_${_device}.cpp)
      file(WRITE ${_file}
        "#include <Test${_device}_Category.hpp>\n"
        "#include <tst${_test}.hpp>\n"
      )
      if(BEATNIK_UNIT_TEST_MPI)
        set(_target ${BEATNIK_UNIT_TEST_PACKAGE}_Test_${_test}_MPI_${_device})
      else()
        set(_target ${BEATNIK_UNIT_TEST_PACKAGE}_Test_${_test}_${_device})
      endif()
      add_executable(${_target} ${_file} ${BEATNIK_UNIT_TEST_MAIN})
      target_include_directories(${_target} PRIVATE ${_dir}
        ${TEST_HARNESS_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${_target} PRIVATE ${BEATNIK_UNIT_TEST_PACKAGE} ${gtest_target})
      if(BEATNIK_UNIT_TEST_MPI)
        foreach(_np ${BEATNIK_UNIT_TEST_MPIEXEC_NUMPROCS})
          add_test(NAME ${_target}_np_${_np} COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS} ${gtest_args})
          set_property(TEST ${_target}_np_${_np} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          set_property(TEST ${_target}_np_${_np} PROPERTY PROCESSORS ${_np})
          set_property(TEST ${_target}_np_${_np} PROPERTY LABELS ${BEATNIK_UNIT_TEST_LABEL})
        endforeach()
      else()
        if(_device STREQUAL THREADS OR _device STREQUAL OPENMP)
          foreach(_thread ${BEATNIK_UNIT_TEST_NUMTHREADS})
            add_test(NAME ${_target}_nt_${_thread} COMMAND
                    ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
            if(_device STREQUAL OPENMP)
              set_property(TEST ${_target}_nt_${_thread} PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
            endif()
            set_property(TEST ${_target}_nt_${_thread} PROPERTY LABELS ${BEATNIK_UNIT_TEST_LABEL})
            if(VALGRIND_FOUND)
              add_test(NAME ${_target}_nt_${_thread}_valgrind COMMAND
                ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
              if(_device STREQUAL OPENMP)
                set_property(TEST ${_target}_nt_${_thread}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
              endif()
            endif()
          endforeach()
        else()
          add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args})
          set_property(TEST ${_target} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          set_property(TEST ${_target} PROPERTY LABELS ${BEATNIK_UNIT_TEST_LABEL})
          if(VALGRIND_FOUND)
            add_test(NAME ${_target}_valgrind COMMAND ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args})
            set_property(TEST ${_target}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          endif()
        endif()
      endif()
      # Valgrind variants are always diagnostic, never gating.
      if(VALGRIND_FOUND)
        get_property(_all_tests DIRECTORY PROPERTY TESTS)
        foreach(_t ${_all_tests})
          if(_t MATCHES "^${_target}.*_valgrind$")
            set_property(TEST ${_t} PROPERTY LABELS unit)
          endif()
        endforeach()
      endif()
      if(BEATNIK_UNIT_TEST_LABEL STREQUAL regression)
        set_property(GLOBAL APPEND PROPERTY BEATNIK_GATE_TARGETS ${_target})
      elseif(BEATNIK_UNIT_TEST_LABEL STREQUAL unit)
        set_property(GLOBAL APPEND PROPERTY BEATNIK_UNIT_TARGETS ${_target})
      endif()
      if(Beatnik_INSTALL_TEST_EXECUTABLES)
        install(TARGETS ${_target}
                RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
      endif()
    endforeach()
  endforeach()
endmacro()
