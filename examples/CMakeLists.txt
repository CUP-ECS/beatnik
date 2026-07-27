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
# Example aggregator. One example per ordered `NN_name/` subdirectory, each
# with its own CMakeLists.txt and a single driver source; shared helpers go in
# examples/common/. Add every new subdirectory here.
#
# Reached only when Beatnik_ENABLE_EXAMPLES=ON (spack: `+examples`), so
# examples never bloat a default build. They must build in both build modes
# (manual out-of-tree and spack-installed).
#
# Each example's accepted arguments are documented in README.md and must be
# kept in sync with the code in the same change.
############################################################################

add_subdirectory(01_rising_bubble)
