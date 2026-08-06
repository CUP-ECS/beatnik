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
# Example 02 — adaptive-mesh rising bubble.
#
# C++/Kokkos/MPI port of zmodel3d-amr's run_adaptive_mesh_bubble.py. Unlike
# example 01, this one targets the CURRENT solver API and builds unconditionally
# under Beatnik_ENABLE_EXAMPLES=ON.
#
# It BUILDS but does not yet RUN to completion: every library body is a stub
# that throws std::logic_error. `--help` works and a real invocation parses its
# arguments and then dies in the first stub. That is why no ctest case is
# registered here — a test asserting "throws logic_error" would have to be
# deleted the moment the solver starts working, and a test asserting success
# cannot pass yet.
#
# WHEN THE SOLVER LANDS: register the end-to-end regression cases here, against
# the committed gold files under tests/regression_tests/gold/. Those cases go in
# the `regression` tier, which is the ship gate, so adding them requires the
# user's explicit confirmation (CLAUDE.md "Minimum test set").
#
# The accepted arguments are documented in the top-level README.md and in
# InputFile.hpp::printSchema, and all three must be updated together.
############################################################################

# The next four lines let this directory be configured standalone, for testing
# the exported Beatnik target.
cmake_minimum_required(VERSION 3.16)
project(Beatnik_AdaptiveMeshBubble)
if(NOT TARGET Beatnik)
  find_package(Beatnik)
endif()

add_executable(adaptive_mesh_bubble adaptive_mesh_bubble.cpp)
target_link_libraries(adaptive_mesh_bubble Beatnik)

# InputFile.hpp is included as "InputFile.hpp" from the driver, so the source
# directory must be on the include path. Named explicitly rather than relying on
# the compiler's quoted-include search, which differs between compilers.
target_include_directories(adaptive_mesh_bubble
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# A cheap, always-passing smoke check: `--help` must exit 0 and print the
# schema. It exercises option-table construction (every setter binds a real
# field) without touching a stub. Tier `unit` — diagnostic, not gating.
add_test(NAME Beatnik_Example_02_adaptive_mesh_bubble_help
  COMMAND adaptive_mesh_bubble --help)
set_tests_properties(Beatnik_Example_02_adaptive_mesh_bubble_help PROPERTIES
  LABELS unit
  PASS_REGULAR_EXPRESSION "Usage: adaptive_mesh_bubble")

if(Beatnik_INSTALL_EXAMPLES)
  install(TARGETS adaptive_mesh_bubble
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
