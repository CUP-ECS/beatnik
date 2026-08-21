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
#   milestone   Long end-to-end runs against a multi-thousand-step reference
#               gold set, at ranks 1 and 4. Structurally identical to
#               `regression` -- same standalone self-validating binaries, same
#               manifest line format -- and deliberately NOT part of the ship
#               gate: these runs are far too long to sit in front of every
#               change. Created by task M0-T1. A milestone test that fails is a
#               real failure and gets a README "Known Issues" entry; it is not
#               a reason to touch the gate.
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
#   ctest -L milestone                # the long milestone runs, NOT the gate
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
# The `milestone` tier is single-sourced the same way and is NOT the gate:
#
#     label `milestone`   x  backend SERIAL (+ HIP on tuolumne)  x  ranks 1, 4
#
# There is deliberately no CI; the gate is operator-run. See docs/testing.md.
############################################################################

include(${TEST_HARNESS_DIR}/test_harness.cmake)

##--------------------------------------------------------------------------##
## regression tier — THE SHIP GATE. Confirm with the user before adding here.
##--------------------------------------------------------------------------##
# End-to-end tests that exercise the whole solver pipeline.
#
# The gtest-harness lists below are still empty; regression test 1 (T1c) is
# registered further down, in the "regression tier -- standalone end-to-end
# tests" section, for the same reason tests/unit_tests/ exists: in spack mode
# there is no build tree and therefore no ctest, so a gate test must be
# authoritative about its own exit code. See that section.
set(REGRESSION_MPI_TESTS
  # None registered through the gtest harness. See the standalone section below.
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
## Shared paths for the regression tier and the comparator harness
##--------------------------------------------------------------------------##
# Defined once, above BOTH consumers: the standalone gate tests below and the
# Python comparator cases further down name the same script and the same
# fixtures, and a second copy of either path is a way for the two to drift.
set(_beatnik_compare_dir ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests)
set(_beatnik_compare_py ${_beatnik_compare_dir}/compare_output.py)
set(_beatnik_fixture_dir ${_beatnik_compare_dir}/fixtures)

# Paths RELATIVE TO THE INSTALLED share/Beatnik/tests directory, used both by
# the install rules below and by the manifest the installed-binary path reads.
# Keeping the repo's own directory layout under one install root is what lets
# the same relative arguments work in both places -- see the install section.
set(_beatnik_rel_compare_py regression_tests/compare_output.py)
set(_beatnik_rel_fixtures regression_tests/fixtures)

##--------------------------------------------------------------------------##
## regression tier -- standalone end-to-end tests. THE SHIP GATE.
##--------------------------------------------------------------------------##
# ADDED BY TASK T1c, WITH THE USER'S EXPLICIT AUTHORIZATION. Before this the
# `regression` tier was EMPTY and a green gate proved nothing (see the note on
# REGRESSION_MPI_TESTS above and README "Known Issues"). It now has exactly one
# member:
#
#   Beatnik_Test_InitialConditions   regression test 1 -- the whole driver path
#                                    at 0 timesteps, compared against the T1a
#                                    Python gold checkpoint by
#                                    compare_output.py.
#
# GREW TO TWO MEMBERS AT TASK T2c, pre-authorized in that task's exit criterion:
#
#   Beatnik_Test_BirkhoffRott        T2c -- the vertex source quadrature and the
#                                    direct O(N^2) Birkhoff-Rott sum, against
#                                    hard-coded reference numbers computed from
#                                    the read-only Python. Takes NO arguments;
#                                    everything it needs it computes.
#
# GREW TO THREE MEMBERS AT TASK T2d, authorized by the user for that task:
#
#   Beatnik_Test_DirectSolve10Steps  regression test 2 -- ten TVD-RK3 timesteps
#                                    with the direct BR solver, compared against
#                                    the T2a Python gold set at EVERY step by
#                                    compare_output.py. This is the first gate
#                                    member that says the solver integrates
#                                    anything.
#
# GREW TO FOUR MEMBERS AT TASK T4a, authorized in that task's exit criterion:
#
#   Beatnik_Test_RefineSplitEdges    regression test 4 -- twenty timesteps with
#                                    indicator-driven refinement through
#                                    Tessera::splitEdges(). Takes NO arguments:
#                                    it compares global counts, the projection
#                                    identity and risk R12's shape signals, not
#                                    field tables, because R13's diagonal
#                                    difference makes a field-for-field
#                                    comparison against the Python meaningless
#                                    after the first pass. See that file's
#                                    header.
#
# GREW TO FIVE MEMBERS AT TASK T4b, authorized in that task's exit criterion:
#
#   Beatnik_Test_DynamicRemeshSplit  regression test 5 -- twenty timesteps of
#                                    metric-driven dynamic remeshing, split
#                                    third only, with the collapse, flip and
#                                    smoothing thirds configured off through the
#                                    REFERENCE's own knobs rather than through a
#                                    Beatnik-only switch. Takes NO arguments.
#                                    See that file's header.
#
# The gate on tuolumne is therefore 5 x {SERIAL, HIP} x ranks 1-6 = 60 launches.
#
# HOW A MULTI-RANK REGRESSION ENTRY IS REPRESENTED, AND WHY THIS SHAPE
# --------------------------------------------------------------------
# T1c's exit criterion names ranks 1, 2 and 4; the GATE is defined at ranks
# 1-6. Those are not in tension and the difference is not encoded here:
#
#   **The rank set is a property of the gate, not of the test.** So the test is
#   registered once per backend and the rank sweep stays where it already lives
#   -- BEATNIK_TEST_MPI_RANKS for ctest, BEATNIK_GATE_RANKS in
#   scripts/<system>/run_regression_minset.*. The test reads its own comm size
#   and adapts. T1c's three rank counts are therefore a verified SUBSET of what
#   the gate runs, and the gate definition (label x backend x ranks 1-6) is
#   unchanged and still single-sourced.
#
# The alternative -- a per-test rank list on the manifest line -- was rejected
# because it would let one test quietly gate at fewer ranks than the gate
# claims, which is precisely the "gate silently shrinks" failure CLAUDE.md
# forbids.
#
# What DID have to change is that a gate manifest line now carries ARGUMENTS
# after the target name (the gold file, the comparator, the mismatched fixture),
# resolved relative to the manifest's own directory. The line format is
#
#     <target> [args...]
#
# and the runner splits on whitespace, filters the BACKEND SUFFIX ON FIELD 1,
# and runs from the manifest directory. A bare target name -- every line the
# manifest could previously hold -- is still valid, so this widens the format
# rather than replacing it. This is the same convention the unit manifest's
# `py-pass` / `py-fail` lines already use, and it exists for the same reason:
# a test's data must travel with the manifest that names it.
#
# ONE BINARY PER BACKEND, NOT ONE ON THE DEFAULT SPACE
# ---------------------------------------------------
# tests/unit_tests/ builds one binary on Kokkos::DefaultExecutionSpace with no
# backend suffix, which is right for a diagnostic. It is WRONG here, and the
# reason is mechanical: the gate selects a backend with `-R <backend>` (ctest)
# or by matching the target's `_<BACKEND>` suffix (the manifest runner), so a
# target without a suffix is skipped by the installed gate path ENTIRELY -- a
# silent zero-test pass. And on tuolumne the default space is HIP, so a single
# binary could not honestly answer for SERIAL. Each enabled backend therefore
# gets its own translation unit, generated below, that pins
# BEATNIK_TEST_EXEC_SPACE.
set(BEATNIK_REGRESSION_TEST_SOURCES
  # T1c: regression test 1 -- initial conditions at 0 timesteps.
  Beatnik_Test_InitialConditions.cpp
  # T2c: the vertex quadrature and the direct Birkhoff-Rott sum.
  Beatnik_Test_BirkhoffRott.cpp
  # T2d: regression test 2 -- ten timesteps against the T2a gold set.
  Beatnik_Test_DirectSolve10Steps.cpp
  # T4a: regression test 4 -- twenty timesteps with splitEdges() refinement.
  Beatnik_Test_RefineSplitEdges.cpp
  # T4b: regression test 5 -- twenty timesteps of split-only dynamic remeshing.
  Beatnik_Test_DynamicRemeshSplit.cpp
  )

# Kokkos execution-space class per BEATNIK_SUPPORTED_DEVICES name. Spelled out
# rather than derived, so an unsupported spelling is a configure-time error and
# not a name that silently fails to compile.
set(_beatnik_space_SERIAL        Kokkos::Serial)
set(_beatnik_space_THREADS       Kokkos::Threads)
set(_beatnik_space_OPENMP        Kokkos::OpenMP)
set(_beatnik_space_CUDA          Kokkos::Cuda)
set(_beatnik_space_CUDA_UVM      Kokkos::Cuda)
set(_beatnik_space_HIP           Kokkos::HIP)
set(_beatnik_space_SYCL          Kokkos::Experimental::SYCL)
set(_beatnik_space_OPENMPTARGET  Kokkos::Experimental::OpenMPTarget)

# Arguments PER TEST, twice over: absolute for ctest (which runs from the build
# tree), manifest-relative for the installed path. Each pair must name the same
# files -- that is what keeps `ctest -L regression` and the manifest testing the
# same thing.
#
# Keyed by the source stem rather than shared across the tier: T2c's test takes
# none, and handing it three paths it ignores would make the manifest line claim
# a dependency that does not exist. An entry is REQUIRED for every source in
# BEATNIK_REGRESSION_TEST_SOURCES -- a missing one is a configure-time error
# below, not a test silently launched without its data.
set(_beatnik_args_Beatnik_Test_InitialConditions_abs
  ${_beatnik_compare_dir}/initial_conditions/gold.npz
  ${_beatnik_compare_dir}/compare_output.py
  ${_beatnik_fixture_dir}/synthetic_gold.npz)
set(_beatnik_args_Beatnik_Test_InitialConditions_rel
  regression_tests/initial_conditions/gold.npz
  ${_beatnik_rel_compare_py}
  ${_beatnik_rel_fixtures}/synthetic_gold.npz)

# T2c takes no arguments. Declared explicitly (and as DEFINED, which an empty
# list still is) so the "did the author forget?" check below stays meaningful.
set(_beatnik_args_Beatnik_Test_BirkhoffRott_abs "")
set(_beatnik_args_Beatnik_Test_BirkhoffRott_rel "")

# T2d takes the gold DIRECTORY (not a single file): it compares one checkpoint
# per step, and finds each step's gold file by its `_step%07d.npz` suffix rather
# than by rebuilding the name from a time -- the time is what is under test.
set(_beatnik_args_Beatnik_Test_DirectSolve10Steps_abs
  ${_beatnik_compare_dir}/direct-solve-10-steps/gold
  ${_beatnik_compare_dir}/compare_output.py)
set(_beatnik_args_Beatnik_Test_DirectSolve10Steps_rel
  regression_tests/direct-solve-10-steps/gold
  ${_beatnik_rel_compare_py})

# T4a takes no arguments and reads no gold file: the Python reference enters it
# as compiled-in literals (global face and vertex counts per refinement pass),
# because those are the only quantities R13 leaves comparable. Declared
# explicitly -- and as DEFINED, which an empty list still is -- so the "did the
# author forget?" check below stays meaningful.
set(_beatnik_args_Beatnik_Test_RefineSplitEdges_abs "")
set(_beatnik_args_Beatnik_Test_RefineSplitEdges_rel "")

# T4b takes no arguments and reads no gold file either, for the same reason as
# T4a and one more: its masks are partial from the first pass, so R13's diagonal
# difference makes the two codes integrate different meshes from step 2 and only
# the pass-1 counts and the shape statistics are comparable. Those enter as
# compiled-in literals.
set(_beatnik_args_Beatnik_Test_DynamicRemeshSplit_abs "")
set(_beatnik_args_Beatnik_Test_DynamicRemeshSplit_rel "")

foreach(_src ${BEATNIK_REGRESSION_TEST_SOURCES})
  get_filename_component(_stem ${_src} NAME_WE)

  if(NOT DEFINED _beatnik_args_${_stem}_abs
     OR NOT DEFINED _beatnik_args_${_stem}_rel)
    message(FATAL_ERROR
      "tests/CMakeLists.txt: no argument lists recorded for regression test "
      "'${_stem}'. Set _beatnik_args_${_stem}_abs and _beatnik_args_${_stem}_rel "
      "above (to \"\" if the test takes none) in the same change that added it.")
  endif()

  foreach(_device ${BEATNIK_TEST_DEVICES})
    if(NOT DEFINED _beatnik_space_${_device})
      message(FATAL_ERROR
        "tests/CMakeLists.txt: no Kokkos execution space recorded for backend "
        "'${_device}'. Add it to the _beatnik_space_* table above in the same "
        "change that added the backend.")
    endif()

    set(_target ${_stem}_MPI_${_device})
    set(_gendir ${CMAKE_CURRENT_BINARY_DIR}/regression_${_device})
    file(MAKE_DIRECTORY ${_gendir})
    set(_genfile ${_gendir}/${_stem}_${_device}.cpp)
    file(WRITE ${_genfile}
      "// GENERATED by tests/CMakeLists.txt -- do not edit.\n"
      "#include <Kokkos_Core.hpp>\n"
      "#define BEATNIK_TEST_EXEC_SPACE ${_beatnik_space_${_device}}\n"
      "#include \"${_beatnik_compare_dir}/${_src}\"\n")

    add_executable(${_target} ${_genfile})
    target_link_libraries(${_target} PRIVATE Beatnik MPI::MPI_CXX)
    target_include_directories(${_target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)

    # One ctest case per gate rank count, exactly as the gtest harness does, so
    # `ctest -L regression -R ${_device}` covers the whole sweep.
    foreach(_np ${BEATNIK_TEST_MPI_RANKS})
      add_test(NAME ${_target}_np_${_np} COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS}
        ${_beatnik_args_${_stem}_abs})
      set_tests_properties(${_target}_np_${_np} PROPERTIES
        LABELS regression
        PROCESSORS ${_np}
        ENVIRONMENT OMP_NUM_THREADS=1)
    endforeach()

    # The gate manifest line, from the same loop that applied the label. A test
    # with no arguments contributes a bare target name, which is the format the
    # manifest already accepted before T1c widened it.
    set(_manifest_line "${_target}")
    if(_beatnik_args_${_stem}_rel)
      string(REPLACE ";" " " _args_str "${_beatnik_args_${_stem}_rel}")
      set(_manifest_line "${_target} ${_args_str}")
    endif()
    set_property(GLOBAL APPEND PROPERTY BEATNIK_GATE_TARGETS
      "${_manifest_line}")

    if(Beatnik_INSTALL_TEST_EXECUTABLES)
      install(TARGETS ${_target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
    endif()
  endforeach()
endforeach()

##--------------------------------------------------------------------------##
## milestone tier -- long end-to-end runs. NOT THE SHIP GATE.
##--------------------------------------------------------------------------##
# ADDED BY TASK M0-T1, which created the tier. It shipped EMPTY and M0-T3 filled
# it: the tier now has exactly TWO members, both milestone 0's, and
# milestone1.md's M1-T1 registers the third.
#
#   Beatnik_Test_Milestone0Frozen    2000 TVD-RK3 timesteps of the reference's
#                                    default physics on a frozen mesh at
#                                    --icosphere-subdivisions 3, compared against
#                                    the M0-G1 gold set at all 81 checkpointed
#                                    steps at --rtol 1e-10 --atol 1e-12.
#                                    M0-A1's PRIMARY member.
#   Beatnik_Test_Milestone0FrozenL4  the same test at subdivisions 4 against the
#                                    M0-G2 gold set. Three lines of source: it
#                                    #defines BEATNIK_M0_LEVEL 4 and #includes
#                                    the file above, so the assertion body exists
#                                    once and every per-level literal is
#                                    re-derived inside it.
#
# TWO SOURCE STEMS, NOT ONE TARGET WITH TWO ARGUMENT LISTS. The loop below keys
# arguments by source stem, one list per source, and each member needs its own
# gold DIRECTORY. Two stems give two honest _abs / _rel pairs and leave this loop
# exactly as M0-T1 wrote it; one stem with two lists would have to teach the loop
# to carry more than one per source, and an argument list naming the wrong
# level's gold set is precisely what its FATAL_ERROR exists to prevent.
#
# Structurally this is the regression loop above, with three differences and no
# others; keep it that way, because the two tiers must stay comparable:
#
#   1. LABELS milestone, and the manifest line goes to BEATNIK_MILESTONE_TARGETS
#      -> beatnik_milestone_manifest.txt.
#   2. Ranks 1 and 4, not the gate's 1-6. As with the gate, the rank set is a
#      property of the TIER, not of a test: it lives in
#      BEATNIK_MILESTONE_MPI_RANKS here and in BEATNIK_MILESTONE_RANKS in the
#      runner, never in a per-test list on a manifest line.
#   3. The gold data is two 81-file sets, installed below.
#
# WHY A SEPARATE TIER AND NOT MORE REGRESSION MEMBERS. A milestone member is a
# 2000-step run at 642 or 2562 vertices; putting one in `regression` would add
# it to every change's 60-launch gate. The gate stays at five members and 60
# launches (CLAUDE.md "Minimum test set"), and this tier is run on demand.
set(BEATNIK_MILESTONE_TEST_SOURCES
  # M0-T3: milestone 0 at subdivision 3 -- the primary member.
  Beatnik_Test_Milestone0Frozen.cpp
  # M0-T3: the same test at subdivision 4 -- the second member.
  Beatnik_Test_Milestone0FrozenL4.cpp
  )

# The tier's rank sweep, overridable exactly as BEATNIK_TEST_MPI_RANKS is.
set(BEATNIK_MILESTONE_MPI_RANKS "1;4" CACHE STRING
  "Rank counts the milestone tier is registered at (default 1;4)")

# Argument lists per test, absolute for ctest and manifest-relative for the
# installed path, exactly as the regression tier keys them by source stem. An
# entry is REQUIRED for every source above -- the FATAL_ERROR in the loop is
# what stops a milestone test from being launched without its gold set.

# Gold directories for the tier, named once and used by both the install rules
# and the argument lists below, so the two cannot drift.
set(_beatnik_m0_sub3_dir
  ${_beatnik_compare_dir}/milestone0-sub3-2000-steps)
set(_beatnik_m0_sub4_dir
  ${_beatnik_compare_dir}/milestone0-sub4-2000-steps)

# M0-T3, both members. Each takes the gold DIRECTORY for ITS OWN LEVEL (not a
# single file): it compares one checkpoint per checkpointed step and finds each
# step's gold file by its `_step%07d.npz` suffix rather than by rebuilding the
# name from a time -- the time is what is under test. Same shape as T2d's pair
# above. **The level in the path must match the level the source compiles at**
# (BEATNIK_M0_LEVEL, 3 by default and 4 in the L4 stem): a mismatch is a
# structural comparator failure at step 0, not a subtle one, but it is still the
# mistake this table exists to keep visible.
set(_beatnik_args_Beatnik_Test_Milestone0Frozen_abs
  ${_beatnik_m0_sub3_dir}/gold
  ${_beatnik_compare_dir}/compare_output.py)
set(_beatnik_args_Beatnik_Test_Milestone0Frozen_rel
  regression_tests/milestone0-sub3-2000-steps/gold
  ${_beatnik_rel_compare_py})

set(_beatnik_args_Beatnik_Test_Milestone0FrozenL4_abs
  ${_beatnik_m0_sub4_dir}/gold
  ${_beatnik_compare_dir}/compare_output.py)
set(_beatnik_args_Beatnik_Test_Milestone0FrozenL4_rel
  regression_tests/milestone0-sub4-2000-steps/gold
  ${_beatnik_rel_compare_py})

foreach(_src ${BEATNIK_MILESTONE_TEST_SOURCES})
  get_filename_component(_stem ${_src} NAME_WE)

  if(NOT DEFINED _beatnik_args_${_stem}_abs
     OR NOT DEFINED _beatnik_args_${_stem}_rel)
    message(FATAL_ERROR
      "tests/CMakeLists.txt: no argument lists recorded for milestone test "
      "'${_stem}'. Set _beatnik_args_${_stem}_abs and _beatnik_args_${_stem}_rel "
      "above (to \"\" if the test takes none) in the same change that added it.")
  endif()

  foreach(_device ${BEATNIK_TEST_DEVICES})
    if(NOT DEFINED _beatnik_space_${_device})
      message(FATAL_ERROR
        "tests/CMakeLists.txt: no Kokkos execution space recorded for backend "
        "'${_device}'. Add it to the _beatnik_space_* table above in the same "
        "change that added the backend.")
    endif()

    set(_target ${_stem}_MPI_${_device})
    set(_gendir ${CMAKE_CURRENT_BINARY_DIR}/milestone_${_device})
    file(MAKE_DIRECTORY ${_gendir})
    set(_genfile ${_gendir}/${_stem}_${_device}.cpp)
    file(WRITE ${_genfile}
      "// GENERATED by tests/CMakeLists.txt -- do not edit.\n"
      "#include <Kokkos_Core.hpp>\n"
      "#define BEATNIK_TEST_EXEC_SPACE ${_beatnik_space_${_device}}\n"
      "#include \"${_beatnik_compare_dir}/${_src}\"\n")

    add_executable(${_target} ${_genfile})
    target_link_libraries(${_target} PRIVATE Beatnik MPI::MPI_CXX)
    target_include_directories(${_target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)

    foreach(_np ${BEATNIK_MILESTONE_MPI_RANKS})
      add_test(NAME ${_target}_np_${_np} COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS}
        ${_beatnik_args_${_stem}_abs})
      set_tests_properties(${_target}_np_${_np} PROPERTIES
        LABELS milestone
        PROCESSORS ${_np}
        ENVIRONMENT OMP_NUM_THREADS=1)
    endforeach()

    set(_manifest_line "${_target}")
    if(_beatnik_args_${_stem}_rel)
      string(REPLACE ";" " " _args_str "${_beatnik_args_${_stem}_rel}")
      set(_manifest_line "${_target} ${_args_str}")
    endif()
    set_property(GLOBAL APPEND PROPERTY BEATNIK_MILESTONE_TARGETS
      "${_manifest_line}")

    if(Beatnik_INSTALL_TEST_EXECUTABLES)
      install(TARGETS ${_target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
    endif()
  endforeach()
endforeach()

##--------------------------------------------------------------------------##
## Measurement drivers -- IN NO TIER. Not the gate, not the milestone tier.
##--------------------------------------------------------------------------##
# ADDED BY TASK M0-D1. These are per-backend binaries that RUN a configuration
# and write checkpoints; every comparison they feed is made offline in Python.
# They exist because a per-backend binary is only obtainable from a test source
# -- the generated translation unit below is what pins BEATNIK_TEST_EXEC_SPACE
# -- and examples/02_adaptive_mesh_bubble cannot answer for SERIAL at all
# (adaptive_mesh_bubble.cpp:209 fixes the space to
# Kokkos::DefaultExecutionSpace, and the installed binary is +rocm, so it is
# HIP-only). M0-D1's matrix is (level 3, level 4) x (SERIAL, HIP) x (ranks 1, 4)
# and half of it has no other driver.
#
# THIS LOOP IS THE MILESTONE LOOP ABOVE, STOPPED SHORT. Same source list shape,
# same per-backend generated TU, same install destination -- and then it does
# NOT:
#
#   * apply LABELS, so no `ctest -L <tier>` selects it;
#   * call add_test at all, so it has no ctest case at any rank count;
#   * append to BEATNIK_GATE_TARGETS or BEATNIK_MILESTONE_TARGETS, so neither
#     manifest names it and neither run_regression_minset.* nor run_milestone.*
#     can pick it up.
#
# It follows that a driver here gates NOTHING and is never run by accident: it
# is launched only by the batch script that measures with it, by name through
# `beatnik_exe`. Keep it that way. A measurement that becomes a test acquires a
# pass criterion, and the whole reason M0-D1 has a driver rather than a test is
# that the criterion is what it is measuring.
#
# There is deliberately no argument-list table here and no FATAL_ERROR guard
# demanding one: a driver's arguments come from the batch script that measures
# with it, not from a manifest line, because there is no manifest line. What a
# driver's arguments ARE is documented in its own file header.
set(BEATNIK_DRIVER_SOURCES
  # M0-D1: the milestone-0 frozen-mesh run. Writes checkpoints, compares
  # nothing; milestone0_ladder.py does the comparing, offline.
  Beatnik_Test_Milestone0Run.cpp
  )

foreach(_src ${BEATNIK_DRIVER_SOURCES})
  get_filename_component(_stem ${_src} NAME_WE)

  foreach(_device ${BEATNIK_TEST_DEVICES})
    if(NOT DEFINED _beatnik_space_${_device})
      message(FATAL_ERROR
        "tests/CMakeLists.txt: no Kokkos execution space recorded for backend "
        "'${_device}'. Add it to the _beatnik_space_* table above in the same "
        "change that added the backend.")
    endif()

    set(_target ${_stem}_MPI_${_device})
    set(_gendir ${CMAKE_CURRENT_BINARY_DIR}/driver_${_device})
    file(MAKE_DIRECTORY ${_gendir})
    set(_genfile ${_gendir}/${_stem}_${_device}.cpp)
    file(WRITE ${_genfile}
      "// GENERATED by tests/CMakeLists.txt -- do not edit.\n"
      "#include <Kokkos_Core.hpp>\n"
      "#define BEATNIK_TEST_EXEC_SPACE ${_beatnik_space_${_device}}\n"
      "#include \"${_beatnik_compare_dir}/${_src}\"\n")

    add_executable(${_target} ${_genfile})
    target_link_libraries(${_target} PRIVATE Beatnik MPI::MPI_CXX)
    target_include_directories(${_target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests)

    # Installed so `beatnik_exe <target>` resolves it in spack mode, where there
    # is no build tree. This is the ONLY thing this loop shares with the two
    # tier loops past the build step.
    if(Beatnik_INSTALL_TEST_EXECUTABLES)
      install(TARGETS ${_target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
    endif()
  endforeach()
endforeach()

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

  # The T2a gold SET -- eleven files, steps 0-10 -- which regression test 2
  # (T2d) compares against one step at a time. Globbed rather than enumerated
  # because the filenames encode the simulation time and would have to be
  # transcribed by hand otherwise; the glob runs at configure time, so a gold
  # file added later needs a reconfigure, which regenerating a gold set implies
  # anyway.
  file(GLOB _beatnik_t2a_gold
       ${_beatnik_compare_dir}/direct-solve-10-steps/gold/*.npz)
  if(NOT _beatnik_t2a_gold)
    message(FATAL_ERROR
      "tests/CMakeLists.txt: no .npz gold files under "
      "regression_tests/direct-solve-10-steps/gold. Regression test 2 installed "
      "without its gold set is not installed.")
  endif()
  install(FILES ${_beatnik_t2a_gold}
          DESTINATION ${_beatnik_test_install}/regression_tests/direct-solve-10-steps/gold)
  install(FILES ${_beatnik_compare_dir}/direct-solve-10-steps/README.md
          DESTINATION ${_beatnik_test_install}/regression_tests/direct-solve-10-steps)

  # -- the milestone tier's gold sets (M0-G1, M0-G2) --------------------------
  # Installed HERE, by M0-T1, and not by M0-T3 which adds the test source: the
  # data is already committed, installing it now is what makes the FATAL_ERROR
  # guard below meaningful, and it leaves M0-T3 a test-source task only.
  #
  # 81 numbered checkpoints each, steps 0-2000 every 25, found by their
  # `_step%07d.npz` suffix at run time. `checkpoint_latest.npz` sits beside them
  # in the repo and is DELIBERATELY EXCLUDED from the glob: it duplicates the
  # final numbered step, it is inert to goldForStep because it carries no
  # `_step%07d.npz` suffix, and shipping it would put a second copy of the
  # largest file in every install. Each set's README.md records the per-step
  # time, minimum-quality and volume-drift series and is installed with it.
  #
  # NOTE THE LAYOUT DIFFERENCE, which is why this is two rules and not a reuse
  # of the T2a one: these READMEs live INSIDE gold/, beside the .npz, whereas
  # direct-solve-10-steps/README.md sits one level up. Both layouts are
  # preserved as they are in the repo rather than unified, because the installed
  # tree mirrors tests/ exactly (see the STRUCTURE IS PRESERVED note above).
  foreach(_set sub3 sub4)
    file(GLOB _beatnik_m0_gold
         ${_beatnik_m0_${_set}_dir}/gold/*_step*.npz)
    if(NOT _beatnik_m0_gold)
      message(FATAL_ERROR
        "tests/CMakeLists.txt: no numbered .npz gold files under "
        "regression_tests/milestone0-${_set}-2000-steps/gold. A milestone test "
        "installed without its gold set is not installed.")
    endif()
    install(FILES ${_beatnik_m0_gold}
                  ${_beatnik_m0_${_set}_dir}/gold/README.md
            DESTINATION
              ${_beatnik_test_install}/regression_tests/milestone0-${_set}-2000-steps/gold)
  endforeach()
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
#
# One regression-tier test per line:
#
#     <target> [args...]
#
# Field 1 is the target; the backend is its name suffix and is what the runner
# filters on. Remaining fields are arguments passed to the binary, and any path
# among them is relative to THIS FILE'S DIRECTORY, so the manifest and the data
# it names move together. (Same convention as the unit manifest's py-pass /
# py-fail lines, and the same reason.)
#
# THE RANK SWEEP IS NOT IN THIS FILE. It is a property of the gate, not of a
# test: the runner applies BEATNIK_GATE_RANKS to every line. A per-test rank
# list here would let one test gate at fewer ranks than the gate claims.
#
# Gate ranks: ${BEATNIK_TEST_MPI_RANKS}
${_beatnik_gate_manifest}
")

# -- milestone tier: the long runs, NOT the gate --------------------------------
# Same line format and the same "paths are relative to THIS FILE'S DIRECTORY"
# convention as the gate manifest, so scripts/<system>/run_milestone.* is the
# gate runner with a different label, manifest name and rank list -- not a
# different mechanism. Emitted unconditionally, so an EMPTY tier produces a
# manifest with zero non-comment lines rather than no file at all: a missing
# manifest and an empty one are different failures and the runner reports them
# differently.
get_property(_beatnik_milestone_targets GLOBAL PROPERTY BEATNIK_MILESTONE_TARGETS)
if(NOT _beatnik_milestone_targets)
  set(_beatnik_milestone_targets "")
endif()
string(REPLACE ";" "\n" _beatnik_milestone_manifest
       "${_beatnik_milestone_targets}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/beatnik_milestone_manifest.txt
"# Beatnik milestone-tier manifest — GENERATED by tests/CMakeLists.txt, do not edit.
#
# One milestone-tier test per line, in the gate manifest's format:
#
#     <target> [args...]
#
# Field 1 is the target; the backend is its name suffix and is what the runner
# filters on. Remaining fields are arguments passed to the binary, and any path
# among them is relative to THIS FILE'S DIRECTORY.
#
# THIS TIER IS NOT THE SHIP GATE. It holds long end-to-end runs against
# multi-thousand-step reference gold sets and is run on demand, by
# scripts/<system>/run_milestone.*. The gate is beatnik_gate_manifest.txt.
#
# THE RANK SWEEP IS NOT IN THIS FILE, for the same reason it is not in the gate
# manifest: it is a property of the tier, and the runner applies
# BEATNIK_MILESTONE_RANKS to every line.
#
# Milestone ranks: ${BEATNIK_MILESTONE_MPI_RANKS}
${_beatnik_milestone_manifest}
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
# The unit tier is DIAGNOSTIC: it does not gate a change. See docs/testing.md.
${_beatnik_unit_manifest}
")

if(Beatnik_INSTALL_TEST_EXECUTABLES)
  install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/beatnik_gate_manifest.txt
            ${CMAKE_CURRENT_BINARY_DIR}/beatnik_milestone_manifest.txt
            ${CMAKE_CURRENT_BINARY_DIR}/beatnik_unit_manifest.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/Beatnik/tests)
endif()
