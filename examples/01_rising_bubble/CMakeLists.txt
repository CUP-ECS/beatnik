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
# Example 01 — rising bubble.
#
# WORK IN PROGRESS, NOT BUILT BY DEFAULT.
#
# rocketrig.cpp and InputFile.hpp target the PRE-REDESIGN solver API: they
# include Solver.hpp and BoundaryCondition.hpp, which 89ec015 removed along with
# the rest of the old solver. They are kept here as the starting material for
# the rising-bubble driver (InputFile.hpp in particular is a complete key=value
# deck parser worth carrying forward), but they cannot compile until the new
# solver exposes an equivalent entry point.
#
# So the target is registered only under an explicit opt-in, which keeps
# Beatnik_ENABLE_EXAMPLES=ON from breaking the build:
#
#     cmake -DBeatnik_ENABLE_EXAMPLES=ON -DBeatnik_EXAMPLES_ALLOW_BROKEN=ON ...
#
# When the driver is ported to the new API: delete the option and the guard,
# register the target and its test unconditionally, and update the accepted
# arguments in README.md in the same change.
############################################################################

# The next four lines are just so that this can be used as a standalone
# CMakeLists.txt for testing of the Target export.
cmake_minimum_required(VERSION 3.16)
project(Beatnik_RisingBubble)
if(NOT TARGET Beatnik)
  find_package(Beatnik)
endif()

option(Beatnik_EXAMPLES_ALLOW_BROKEN
  "Build example drivers that still target the pre-redesign solver API" OFF)

if(NOT Beatnik_EXAMPLES_ALLOW_BROKEN)
  message(STATUS
    "Example 01_rising_bubble skipped: rocketrig.cpp targets the removed "
    "pre-redesign solver API. Set Beatnik_EXAMPLES_ALLOW_BROKEN=ON to attempt "
    "it anyway.")
  return()
endif()

add_executable(rocketrig rocketrig.cpp)
target_link_libraries(rocketrig Beatnik)

# rocketrig takes a single positional argument: the input deck. multi_mode.in is
# the shipped sample. Labeled `regression` as an end-to-end run — it is gating,
# which is a further reason it stays behind the opt-in until it builds.
add_test(NAME Beatnik_Example_01_rising_bubble
  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:rocketrig> ${MPIEXEC_POSTFLAGS}
          ${CMAKE_CURRENT_SOURCE_DIR}/multi_mode.in)
set_tests_properties(Beatnik_Example_01_rising_bubble PROPERTIES
  PROCESSORS ${MPIEXEC_MAX_NUMPROCS}
  LABELS regression)

if(Beatnik_INSTALL_EXAMPLES)
  install(TARGETS rocketrig
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/multi_mode.in
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/Beatnik/examples/01_rising_bubble)
endif()
