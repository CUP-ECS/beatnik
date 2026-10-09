#!/bin/bash
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
# THE MILESTONE TIER AS EIGHT JOBS, one per (member, backend), each running
# ranks 1 and 4 through run_milestone.flux. Runs on the LOGIN node: it only
# calls `flux batch` and launches nothing itself.
#
# Why split (tasks/add-canopy-t6.md T9b): one serial 16-launch job holds every
# result until the slowest launch ends, and its walltime is the SUM of all
# sixteen. Eight jobs run concurrently, so the tier's wall-clock is bounded by
# the slowest pair (level-4 FMM SERIAL), and each requests only the walltime
# its own two launches need, which a backfilling scheduler starts sooner. A
# bare `flux batch scripts/tuolumne/run_milestone.flux` still runs the whole
# tier as one job.
#
# Each row's -t is about 1.5x that pair's measured np1 + np4 launch time
# (T9b's table: the T6 tier run's per-launch costs, T9a/T9e for HIP). It
# overrides the runner's own `# flux: -t` directive.
#
# Usage:  scripts/tuolumne/submit_milestone_split.sh
#
# Prints one `<jobid> <member> <backend> <-t>` line per submission. Each job
# writes beatnik_milestone_<member-short>_<backend>.<jobid>.log to the repo
# root, and its `[milestone] label=... members=... backends=...` line and
# provenance block show the filter it received: flux copies the submitting
# environment into the job, which is how the two variables reach the runner.
############################################################################

set -euo pipefail

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${_repo}"

# The tier is ranks 1 and 4 on the dev env; a stray override inherited from
# the submitting shell would reach every job silently, so refuse it here.
if [ -n "${BEATNIK_MILESTONE_RANKS:-}" ] && [ "${BEATNIK_MILESTONE_RANKS}" != "1 4" ]; then
    echo "submit_milestone_split: BEATNIK_MILESTONE_RANKS='${BEATNIK_MILESTONE_RANKS}'" \
         "is set in this shell; unset it (the tier is ranks 1 and 4)." >&2
    exit 1
fi
if [ -n "${BEATNIK_USE_PROD:-}" ]; then
    echo "submit_milestone_split: BEATNIK_USE_PROD is set; the milestone tier" \
         "runs on the dev env. Unset it." >&2
    exit 1
fi

# member  backend  -t
_rows=(
    "Beatnik_Test_Milestone0Frozen    SERIAL  15m"
    "Beatnik_Test_Milestone0Frozen    HIP     10m"
    "Beatnik_Test_Milestone0FrozenL4  SERIAL  45m"
    "Beatnik_Test_Milestone0FrozenL4  HIP     10m"
    "Beatnik_Test_Milestone0Fmm       SERIAL  225m"
    "Beatnik_Test_Milestone0Fmm       HIP     20m"
    "Beatnik_Test_Milestone0FmmL4     SERIAL  390m"
    "Beatnik_Test_Milestone0FmmL4     HIP     105m"
)

for _row in "${_rows[@]}"; do
    read -r _member _backend _t <<<"${_row}"
    _short="${_member#Beatnik_Test_}"
    _jobid="$(BEATNIK_MILESTONE_MEMBERS="${_member}" \
              BEATNIK_MILESTONE_BACKENDS="${_backend}" \
              flux batch -t "${_t}" \
                  --job-name="beatnik_milestone_${_short}_${_backend}" \
                  scripts/tuolumne/run_milestone.flux)"
    echo "${_jobid} ${_member} ${_backend} ${_t}"
done
