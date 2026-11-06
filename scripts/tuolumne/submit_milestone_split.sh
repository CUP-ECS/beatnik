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
# THE WAY TO RUN THE MILESTONE TIER: eight jobs, one per (member, backend),
# each running ranks 1 and 4 through run_milestone.flux. Runs on the LOGIN node: it only
# calls `flux batch` and launches nothing itself.
#
# Why split (tasks/add-canopy-t6.md T9b): one serial 16-launch job holds every
# result until the slowest launch ends, and its walltime is the SUM of all
# sixteen. Eight jobs run concurrently, so the tier's wall-clock is bounded by
# the slowest pair (level-4 FMM SERIAL), and each requests only the walltime
# its own two launches need, which a backfilling scheduler starts sooner. A
# bare `flux batch scripts/tuolumne/run_milestone.flux` still runs the whole
# tier as one job, but only as a fallback; run the tier through this script.
#
# Each row's -t is about 1.5x that job's MEASURED wall in the first split run
# (T9b, 2026-10-09, jobs f3d7zNR4jrtF..f3d7zPLmZqxj, all green; the measured
# wall is the comment beside each row). It overrides the runner's own
# `# flux: -t` directive. The tier's wall-clock is the slowest row, level-4
# FMM SERIAL at 4.2 h; the eight walls sum to 8.69 h.
#
# QUEUE: a row whose -t is 45 minutes or less goes to `pdebug`, any longer row
# to `pbatch` (systems/tuolumne/claude.md, "Choosing a queue"). Every job is
# one node (the runner's `# flux: --nodes=1`), inside pdebug's 1-2 node rule.
# The `-q` given here overrides the runner's own `# flux: -q` directive.
#
# A row may carry a fourth field, `pending=<task ID>`, for a member whose code
# that task has not yet landed. A pending row is submitted only when its
# `<member>_MPI_<backend>` line is in the installed beatnik_milestone_manifest.txt;
# otherwise it prints a PENDING line to stderr and is not submitted. A row
# without the field is always submitted, so a missing established member still
# fails in the runner, exactly as before. A pending row's -t is an estimate, not
# a measured wall: re-time it from the task's first run and drop the field.
#
# Usage:  scripts/tuolumne/submit_milestone_split.sh
#
# Prints one `<jobid> <member> <backend> <-t> <queue>` line per submission. Each job
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

# member                          backend -t       measured wall (T9b)
_rows=(
    "Beatnik_Test_Milestone0Frozen    SERIAL  6m"    #   236 s
    "Beatnik_Test_Milestone0Frozen    HIP     4m"    #   144 s
    "Beatnik_Test_Milestone0FrozenL4  SERIAL  45m"   #  1781 s
    "Beatnik_Test_Milestone0FrozenL4  HIP     5m"    #   168 s
    "Beatnik_Test_Milestone0Fmm       SERIAL  240m"  #  9581 s
    "Beatnik_Test_Milestone0Fmm       HIP     16m"   #   613 s
    "Beatnik_Test_Milestone0FmmL4     SERIAL  380m"  # 15130 s
    "Beatnik_Test_Milestone0FmmL4     HIP     91m"   #  3614 s
    # Not yet measured. ~1.5x the M1-G2 Python bound for two 600-step launches
    # (2 x 707 s); tasks/milestone1.md M1-T2j adds the member, M1-T2 times it.
    "Beatnik_Test_Milestone1Bubble    SERIAL  40m  pending=M1-T2j"
    "Beatnik_Test_Milestone1Bubble    HIP     40m  pending=M1-T2j"
)

# The installed milestone manifest, looked up only if a row is pending. The
# resolver is sourced in a subshell so the spack env it activates does not leak
# into the environment flux batch copies into each job.
_manifest=""
for _row in "${_rows[@]}"; do
    case "${_row}" in
        *pending=*)
            _manifest="$(
                source "${_repo}/scripts/lib/beatnik_env.sh" >/dev/null 2>&1 || exit 1
                IFS=':'
                for _p in ${PATH}; do
                    if [ -f "${_p}/beatnik_milestone_manifest.txt" ]; then
                        echo "${_p}/beatnik_milestone_manifest.txt"
                        break
                    fi
                done
            )" || true
            if [ -z "${_manifest}" ]; then
                echo "submit_milestone_split: a row is pending, and" \
                     "beatnik_milestone_manifest.txt was not found on the dev env's" \
                     "PATH to check it against. Is the dev env installed?" >&2
                exit 1
            fi
            break
            ;;
    esac
done

for _row in "${_rows[@]}"; do
    read -r _member _backend _t _pending <<<"${_row}"
    if [ -n "${_pending:-}" ]; then
        if ! grep -q "^${_member}_MPI_${_backend}\([[:space:]]\|\$\)" "${_manifest}"; then
            echo "PENDING ${_member} ${_backend}: not in ${_manifest}" \
                 "(${_pending}); not submitted" >&2
            continue
        fi
    fi
    case "${_t}" in
        *h) _minutes=$(( ${_t%h} * 60 )) ;;
        *m) _minutes=${_t%m} ;;
        *)  echo "submit_milestone_split: row -t '${_t}' is not <N>m or <N>h" >&2
            exit 1 ;;
    esac
    if [ "${_minutes}" -le 45 ]; then _queue=pdebug; else _queue=pbatch; fi
    _short="${_member#Beatnik_Test_}"
    _jobid="$(BEATNIK_MILESTONE_MEMBERS="${_member}" \
              BEATNIK_MILESTONE_BACKENDS="${_backend}" \
              flux batch -t "${_t}" -q "${_queue}" \
                  --job-name="beatnik_milestone_${_short}_${_backend}" \
                  scripts/tuolumne/run_milestone.flux)"
    echo "${_jobid} ${_member} ${_backend} ${_t} ${_queue}"
done
