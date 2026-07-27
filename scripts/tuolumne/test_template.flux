#!/bin/bash
# flux: --job-name=[JOB_NAME]
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
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
# Template flux batch script for tuolumne. Copy to
# scripts/tuolumne/<your_script>.flux, fill in the [BRACKETED] parts, and
# submit with:
#
#     flux batch scripts/tuolumne/<your_script>.flux
#
# The --output line writes stdout/stderr to <job-name>.<jobid>.log in the
# submitting directory; read that log to harvest results.
#
# Do NOT export MPICH_*/GTL_*/FI_*/HSA_*/OMP_* here. Those live in
# scripts/tuolumne/runtime_env.sh and the resolver sources them for you.
# Do NOT hardcode a `spack env activate` line either — the resolver activates
# the dev env, or the production env when BEATNIK_USE_PROD=1. For a long or
# large run set BEATNIK_USE_PROD=1 below, so a dev rebuild cannot break the
# queued job. Ask the user which env a new script should target.
############################################################################

set -u

# Any `module load` goes HERE, before the resolver source.

export BEATNIK_REPO="${BEATNIK_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Uncomment for large/long jobs to pin the production environment:
# export BEATNIK_USE_PROD=1

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary

# Resolve the binary for whichever build mode is active: on PATH in
# spack/installed mode, under BEATNIK_BUILD_DIR in manual/tree mode.
EXE="$(beatnik_exe [EXECUTABLE_NAME])" || exit 1

# NOTE: --ntasks must equal --nodes * 4 (tuolumne runs 4 ranks per node).
# Update both together, here and in the flux header above.
flux run \
    --ntasks=4 \
    --nodes=1 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${EXE}" [EXTRA_ARGS]
