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
# TEST TIERS, BACKENDS, AND THE SHIP GATE
# =======================================
#
# Every test lands in exactly one tier, chosen by the LABEL argument below.
# The tier is what decides whether a test gates a change:
#
#   regression  Full end-to-end runs that compose the whole pipeline
#               (problem setup -> time integration -> solve -> result check).
#               THIS TIER IS THE SHIP GATE. Adding a test here changes what
#               must pass before any code change ships, so it requires the
#               user's explicit confirmation.
#
#   unit        Utilities, kernels, single-component and single-phase tests.
#               Diagnostic: useful for locating a fault, but not gating.
#               A test that is failing or flaky belongs HERE plus an entry in
#               README "Known Issues" — never silently dropped from the lists
#               to make the gate green.
#
# Test names carry the active Kokkos backend as a suffix, e.g.
# `Beatnik_Test_Foo_MPI_SERIAL_np_4`, so a backend is selected with `-R`:
#
#   ctest -L regression -R SERIAL     # the project-wide ship gate
#   ctest -L regression -R HIP        # the extra gate on tuolumne
#   ctest -L unit                     # all diagnostics, every backend
#   ctest -L unit -R SERIAL           # diagnostics on one backend
#
# MPI tests are registered once per rank count in BEATNIK_TEST_MPI_RANKS
# (default "1;2;3;4;5;6" — see cmake/test_harness/test_harness.cmake).
#
# THE GATE, single-sourced. It must read identically here, in CLAUDE.md
# "Minimum test set", in systems/<system>/claude.md "Backends", and in
# scripts/<system>/run_regression_minset.*:
#
#     label `regression`  x  backend SERIAL (+ HIP on tuolumne)  x  ranks 1-6
#
# There is deliberately no CI; the gate is operator-run. See CLAUDE.md.
############################################################################

include(${TEST_HARNESS_DIR}/test_harness.cmake)

##--------------------------------------------------------------------------##
## regression tier — THE SHIP GATE. Confirm with the user before adding here.
##--------------------------------------------------------------------------##
# End-to-end tests that exercise the whole solver pipeline.
#
# EMPTY BY DESIGN, FOR NOW. The pre-redesign solver and its only end-to-end
# test (tstFmmVsExact.hpp, an FMM-vs-Exact BR comparison) were removed in
# 89ec015 to build a new solver from scratch. Until the new solver lands its
# first end-to-end test the gate is structurally correct but vacuous — a green
# gate proves nothing. Tracked in README "Known Issues".
set(REGRESSION_MPI_TESTS
  # None yet — see the note above.
  )

set(REGRESSION_SERIAL_TESTS
  # None yet — end-to-end Beatnik runs are inherently MPI.
  )

##--------------------------------------------------------------------------##
## unit tier — diagnostic. Add freely; these do not gate a change.
##--------------------------------------------------------------------------##
# Backend-independent: pure host utilities, no Kokkos execution space.
set(UNIT_NOBACKEND_TESTS
  # None currently
  )

# Backend-parameterized, single process.
set(UNIT_SERIAL_TESTS
  # None currently
  )

# Backend-parameterized, multi-process.
set(UNIT_MPI_TESTS
  # None currently
  )

##--------------------------------------------------------------------------##
## Registration
##--------------------------------------------------------------------------##
Beatnik_add_tests_nobackend(PACKAGE Beatnik LABEL unit
  NAMES ${UNIT_NOBACKEND_TESTS})

Beatnik_add_tests(PACKAGE Beatnik LABEL unit
  NAMES ${UNIT_SERIAL_TESTS})

Beatnik_add_tests(MPI PACKAGE Beatnik LABEL unit
  NAMES ${UNIT_MPI_TESTS})

Beatnik_add_tests(PACKAGE Beatnik LABEL regression
  NAMES ${REGRESSION_SERIAL_TESTS})

Beatnik_add_tests(MPI PACKAGE Beatnik LABEL regression
  NAMES ${REGRESSION_MPI_TESTS})

##--------------------------------------------------------------------------##
## Regression-comparison harness (Python)
##--------------------------------------------------------------------------##
# tests/regression_tests/compare_output.py compares a Beatnik HDF5 checkpoint
# against a zmodel3d-amr `.npz` gold file. It is a standalone CLI, so it is
# registered directly rather than through the gtest harness — it has no Kokkos
# backend and no rank sweep.
#
# The two cases below run it on the COMMITTED synthetic fixtures in
# tests/regression_tests/fixtures/ and pass today, before any solver exists:
#
#   ..._python_compare           positive: the same surface in a different row
#                                order, which must MATCH. This is the case that
#                                actually exercises the comparator's job —
#                                recovering a correspondence the files do not
#                                record.
#   ..._python_compare_negative  negative: one vertex displaced, which must
#                                FAIL. A comparator that has only ever seen
#                                matching data has not been tested, so this is
#                                registered with WILL_FAIL.
#
# Tier `unit`: diagnostic, not gating. These test the HARNESS, not Beatnik, and
# promoting anything into the `regression` tier changes the ship gate and needs
# the user's confirmation (see the tier comment at the top of this file).
#
# Skipped, with a status message, when Python or its numpy/h5py dependencies are
# missing — a missing interpreter must not fail the build on a machine where the
# C++ tests are the point.
find_package(Python3 COMPONENTS Interpreter QUIET)

set(_beatnik_compare_dir ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests)
set(_beatnik_compare_py ${_beatnik_compare_dir}/compare_output.py)
set(_beatnik_fixture_dir ${_beatnik_compare_dir}/fixtures)

if(NOT Python3_Interpreter_FOUND)
  message(STATUS
    "Beatnik: no Python 3 interpreter found; skipping the compare_output.py "
    "regression-harness tests")
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy, h5py"
    RESULT_VARIABLE _beatnik_compare_deps
    OUTPUT_QUIET ERROR_QUIET)

  if(NOT _beatnik_compare_deps EQUAL 0)
    message(STATUS
      "Beatnik: ${Python3_EXECUTABLE} lacks numpy and/or h5py; skipping the "
      "compare_output.py regression-harness tests")
  else()
    add_test(NAME Beatnik_Test_PythonCompare
      COMMAND ${Python3_EXECUTABLE} ${_beatnik_compare_py}
              ${_beatnik_fixture_dir}/synthetic_match.h5
              ${_beatnik_fixture_dir}/synthetic_gold.npz
              --rtol 1e-10 --atol 1e-12)
    set_tests_properties(Beatnik_Test_PythonCompare PROPERTIES LABELS unit)

    add_test(NAME Beatnik_Test_PythonCompare_Negative
      COMMAND ${Python3_EXECUTABLE} ${_beatnik_compare_py}
              ${_beatnik_fixture_dir}/synthetic_perturbed.h5
              ${_beatnik_fixture_dir}/synthetic_gold.npz
              --rtol 1e-10 --atol 1e-12)
    set_tests_properties(Beatnik_Test_PythonCompare_Negative PROPERTIES
      LABELS unit
      WILL_FAIL TRUE)
  endif()
endif()

##--------------------------------------------------------------------------##
## Gate manifest for the installed-binary path
##--------------------------------------------------------------------------##
# In spack/installed mode there is no build tree and therefore no ctest, so the
# gate wrapper cannot ask ctest which tests carry the `regression` label. Emit
# the regression-tier target names here instead, from the same registration
# that applied the labels, so the two cannot drift. The wrapper finds this file
# by scanning PATH (the spack package prepends share/Beatnik/tests).
get_property(_beatnik_gate_targets GLOBAL PROPERTY BEATNIK_GATE_TARGETS)
if(NOT _beatnik_gate_targets)
  set(_beatnik_gate_targets "")
endif()
string(REPLACE ";" "\n" _beatnik_gate_manifest "${_beatnik_gate_targets}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/beatnik_gate_manifest.txt
"# Beatnik ship-gate manifest — GENERATED by tests/CMakeLists.txt, do not edit.
# One regression-tier test target per line. Backend is the name suffix.
# Gate ranks: ${BEATNIK_TEST_MPI_RANKS}
${_beatnik_gate_manifest}
")

if(Beatnik_INSTALL_TEST_EXECUTABLES)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/beatnik_gate_manifest.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
endif()
