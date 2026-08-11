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
## unit tier -- standalone self-validating tests
##--------------------------------------------------------------------------##
# tests/unit_tests/ holds `unit`-tier tests that are NOT gtest tests: each is a
# single binary on Kokkos::DefaultExecutionSpace that decides its own pass/fail
# and returns non-zero on failure. That is what makes the tier runnable in spack
# mode, where there is no build tree and therefore no ctest. See that
# directory's CMakeLists.txt for the full reasoning, and
# tests/unit_tests/Beatnik_TestAssert.hpp for the assertion helper.
#
# Both styles land in the same tier and the same manifest, so
# `ctest -L unit` and scripts/<system>/unit_tests.* both see everything.
add_subdirectory(unit_tests)

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

# Paths RELATIVE TO THE INSTALLED share/Beatnik/tests directory, used both by
# the install rules below and by the manifest the installed-binary path reads.
# Keeping the repo's own directory layout under one install root is what lets
# the same relative arguments work in both places -- see the install section.
set(_beatnik_rel_compare_py regression_tests/compare_output.py)
set(_beatnik_rel_fixtures regression_tests/fixtures)

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
    set(_beatnik_compare_args --rtol 1e-10 --atol 1e-12)

    add_test(NAME Beatnik_Test_PythonCompare
      COMMAND ${Python3_EXECUTABLE} ${_beatnik_compare_py}
              ${_beatnik_fixture_dir}/synthetic_match.h5
              ${_beatnik_fixture_dir}/synthetic_gold.npz
              ${_beatnik_compare_args})
    set_tests_properties(Beatnik_Test_PythonCompare PROPERTIES LABELS unit)

    add_test(NAME Beatnik_Test_PythonCompare_Negative
      COMMAND ${Python3_EXECUTABLE} ${_beatnik_compare_py}
              ${_beatnik_fixture_dir}/synthetic_perturbed.h5
              ${_beatnik_fixture_dir}/synthetic_gold.npz
              ${_beatnik_compare_args})
    set_tests_properties(Beatnik_Test_PythonCompare_Negative PROPERTIES
      LABELS unit
      WILL_FAIL TRUE)

    # The same two invocations, for the installed-binary path. `py-pass` /
    # `py-fail` encode the expected exit status, which is where ctest's
    # WILL_FAIL property has to go when there is no ctest -- a runner that
    # treated the negative case as an ordinary test would report the suite red
    # exactly when the comparator is working.
    string(REPLACE ";" " " _beatnik_compare_args_str "${_beatnik_compare_args}")
    list(APPEND _beatnik_unit_manifest_lines
      "py-pass Beatnik_Test_PythonCompare ${_beatnik_rel_compare_py} ${_beatnik_rel_fixtures}/synthetic_match.h5 ${_beatnik_rel_fixtures}/synthetic_gold.npz ${_beatnik_compare_args_str}")
    list(APPEND _beatnik_unit_manifest_lines
      "py-fail Beatnik_Test_PythonCompare_Negative ${_beatnik_rel_compare_py} ${_beatnik_rel_fixtures}/synthetic_perturbed.h5 ${_beatnik_rel_fixtures}/synthetic_gold.npz ${_beatnik_compare_args_str}")
  endif()
endif()

##--------------------------------------------------------------------------##
## Installing the two test tiers under +testing
##--------------------------------------------------------------------------##
# WHY THIS EXISTS. Beatnik is built via spack with `+testing`, which sets
# Beatnik_ENABLE_TESTING and Beatnik_INSTALL_TEST_EXECUTABLES (see the spack
# package's cmake_args) and prepends share/Beatnik/tests to PATH (see its
# setup_run_environment). A test binary or data file that exists only in the
# discarded build tree therefore cannot be run in spack mode AT ALL -- that is
# the root of the no-ctest wrinkle the manifests below work around, not a side
# effect of it.
#
# The C++ test executables of both tiers are installed by the loops that build
# them (cmake/test_harness/test_harness.cmake and tests/unit_tests/
# CMakeLists.txt). What is left is the regression tier's Python script and its
# data, which no target owns.
#
# STRUCTURE IS PRESERVED, NOT MADE RELOCATABLE-BY-LOOKUP. The installed tree
# mirrors the repo's tests/ layout under share/Beatnik/tests, so
# `regression_tests/fixtures/synthetic_gold.npz` names the same thing relative
# to either root. That is the cheaper of the two options the task offered and it
# is the one that keeps compare_output.py unchanged: the script takes both files
# as explicit CLI arguments and performs no lookup of its own, so there is
# nothing to make relocatable -- only a caller's relative paths, which one
# shared root already handles. Adding a search path inside the script would be
# inventing a lookup in order to relocate it.
#
# make_fixtures.py is deliberately NOT installed: it regenerates the fixtures
# from scratch and is a development tool, not something a test run needs.
if(Beatnik_INSTALL_TEST_EXECUTABLES)
  set(_beatnik_test_install ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)

  # The comparator itself. PROGRAMS, not FILES, so it keeps its exec bit.
  install(PROGRAMS ${_beatnik_compare_dir}/compare_output.py
          DESTINATION ${_beatnik_test_install}/regression_tests)

  # The committed synthetic fixtures the two comparator cases load. A
  # regression test installed without its gold file is not installed.
  install(FILES
            ${_beatnik_fixture_dir}/synthetic_gold.npz
            ${_beatnik_fixture_dir}/synthetic_match.h5
            ${_beatnik_fixture_dir}/synthetic_perturbed.h5
          DESTINATION ${_beatnik_test_install}/regression_tests/fixtures)

  # The T1a gold checkpoint, which regression test 1 (T1c) compares against.
  install(FILES
            ${_beatnik_compare_dir}/initial_conditions/gold.npz
            ${_beatnik_compare_dir}/initial_conditions/README.md
          DESTINATION ${_beatnik_test_install}/regression_tests/initial_conditions)
endif()

##--------------------------------------------------------------------------##
## Manifests for the installed-binary path
##--------------------------------------------------------------------------##
# In spack/installed mode there is no build tree and therefore no ctest, so a
# wrapper script cannot ask ctest which tests carry a given label. Emit the
# target names here instead, from the same registrations that applied the
# labels, so the two cannot drift. A wrapper finds these files by scanning PATH
# (the spack package prepends share/Beatnik/tests).

# -- regression tier: the ship gate --------------------------------------------
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

# -- unit tier -----------------------------------------------------------------
# Three line kinds, because the tier is not homogeneous:
#
#   exe     <target>                      a binary; run it, expect exit 0
#   py-pass <name> <script> <args...>     python; expect exit 0
#   py-fail <name> <script> <args...>     python; expect NON-ZERO (ctest
#                                         WILL_FAIL, which has no ctest here)
#
# Script paths are relative to the directory holding this manifest, so the file
# and the data it names move together.
get_property(_beatnik_unit_targets GLOBAL PROPERTY BEATNIK_UNIT_TARGETS)
foreach(_t ${_beatnik_unit_targets})
  list(APPEND _beatnik_unit_manifest_lines "exe ${_t}")
endforeach()
if(NOT _beatnik_unit_manifest_lines)
  set(_beatnik_unit_manifest_lines "")
endif()
string(REPLACE ";" "\n" _beatnik_unit_manifest "${_beatnik_unit_manifest_lines}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/beatnik_unit_manifest.txt
"# Beatnik unit-tier manifest — GENERATED by tests/CMakeLists.txt, do not edit.
# Line kinds:  exe <target> | py-pass <name> <script> <args...>
#              py-fail <name> <script> <args...>   (expects a NON-ZERO exit)
# Script paths are relative to this file's directory.
# The unit tier is DIAGNOSTIC: it does not gate a change. See CLAUDE.md.
${_beatnik_unit_manifest}
")

if(Beatnik_INSTALL_TEST_EXECUTABLES)
  install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/beatnik_gate_manifest.txt
            ${CMAKE_CURRENT_BINARY_DIR}/beatnik_unit_manifest.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
endif()
