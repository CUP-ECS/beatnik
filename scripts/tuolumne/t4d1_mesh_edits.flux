#!/bin/bash
# flux: --job-name=beatnik_t4d1_mesh_edits
# flux: --nodes=2
# flux: --exclusive
# flux: -t 15m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
# flux: --bank=unm
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
# T4d1 exit criterion (tasks/milestone1.md): Beatnik_Test_MeshEdits at EVERY
# ONE of ranks 1, 2, 3, 4, 5 and 6, in one job.
#
# WHY THIS SCRIPT EXISTS RATHER THAN THE TIER RUNNER OR ctest. The tier
# registers every test at one rank (tests/unit_tests/CMakeLists.txt), spack
# mode has no build tree and so no ctest (CLAUDE.md "Build mode"), and
# scripts/tuolumne/unit_tests.flux allocates ONE node while each launch asks for
# ceil(n/4) nodes, so 5 and 6 ranks cannot launch there. So this allocates TWO
# nodes -- tuolumne runs 4 ranks per node -- and loops the one binary over the
# six rank counts, exactly as scripts/tuolumne/t4_fmm_vs_direct.flux does. The
# member still lands in the tier runner's one-rank run, which discovers tests.
#
# THE TEST WRITES NO FILES, so BEATNIK_TEST_SCRATCH is not set.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set).
#
# Usage:  flux batch scripts/tuolumne/t4d1_mesh_edits.flux [RANKS...]
#
#   RANKS...  rank counts to run; default `1 2 3 4 5 6`, which is the exit
#             criterion. Overriding it is for DEBUGGING A FAILURE, never for
#             reporting T4d1: a narrowed sweep changes what DONE means.
#
# Then read beatnik_t4d1_mesh_edits.<jobid>.log in the submitting directory.
# The job's exit status is non-zero if any rank count fails.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root. `flux batch` copies the script into a per-job spool
# directory before running it, so BASH_SOURCE points at /var/tmp/... and cannot
# find the checkout. It does preserve the submitting working directory, so walk
# up from PWD as the fallback; BASH_SOURCE still covers a direct `bash` run.
beatnik_find_repo() {
    local _d
    for _d in "$(dirname "${BASH_SOURCE[0]}")/../.." "${PWD}"; do
        _d="$(cd "${_d}" 2>/dev/null && pwd)" || continue
        while [ -n "${_d}" ] && [ "${_d}" != "/" ]; do
            if [ -f "${_d}/scripts/lib/beatnik_env.sh" ]; then
                printf '%s\n' "${_d}"
                return 0
            fi
            _d="$(dirname "${_d}")"
        done
    done
    return 1
}
export BEATNIK_REPO="${BEATNIK_REPO:-$(beatnik_find_repo)}"
if [ -z "${BEATNIK_REPO}" ]; then
    echo "[t4d1] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[t4d1] spack env status:"
spack env status 2>&1 | sed 's/^/[t4d1]   /'
echo "[t4d1] compiler:"
{ hipcc --version 2>/dev/null || c++ --version 2>/dev/null || echo "unknown"; } \
    | sed 's/^/[t4d1]   /'
echo "[t4d1] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t4d1] git status --short:"
git -C "${BEATNIK_REPO}" status --short 2>&1 | sed 's/^/[t4d1]   /'

EXE="$(beatnik_exe Beatnik_Test_MeshEdits)" || exit 1
echo "[t4d1] exe     = ${EXE}"
echo "[t4d1] exe mtime = $(date -r "${EXE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"

if [ "$#" -gt 0 ]; then
    _ranks=( "$@" )
    echo "[t4d1] *** RANK SET OVERRIDDEN: ${_ranks[*]} -- this is NOT the T4d1" \
         "exit criterion, which is 1 2 3 4 5 6 ***"
else
    _ranks=( 1 2 3 4 5 6 )
fi
echo "[t4d1] ranks   = ${_ranks[*]}"

_rc=0
_pass=0
_fail=0
_names_failed=""

for _np in "${_ranks[@]}"; do
    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/t4_fmm_vs_direct.flux and must not be
    # simplified: a wrong binding does not fail, it oversubscribes one device.
    _nodes=$(( (_np + 3) / 4 ))

    echo "[t4d1] === Beatnik_Test_MeshEdits at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[t4d1] command: flux run --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1" \
         "${EXE}"

    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${EXE}"
    _obs=$?

    if [ "${_obs}" -eq 0 ]; then
        echo "[t4d1] PASS np=${_np}"
        _pass=$(( _pass + 1 ))
    else
        echo "[t4d1] FAIL np=${_np}: rc=${_obs}"
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} np${_np}"
        _rc=1
    fi
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
if [ "${_total}" -eq 0 ]; then
    echo "[t4d1] FAIL: no rank count was run at all, which is not a pass."
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t4d1] SUMMARY: PASS (${_pass}/${_total} rank counts)"
else
    echo "[t4d1] SUMMARY: FAIL (${_pass}/${_total} rank counts);" \
         "failed:${_names_failed}"
fi
exit "${_rc}"
