#!/bin/bash
# flux: --job-name=beatnik_t4_fmm_vs_direct
# flux: --nodes=2
# flux: --exclusive
# flux: -t 25m
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
# T4 exit criterion (tasks/canopy/add-canopy.md): Beatnik_Test_FmmVsDirect at
# EVERY ONE of ranks 1, 2, 3, 4, 5 and 6.
#
# WHY THIS SCRIPT EXISTS RATHER THAN THE TIER RUNNER OR ctest. Neither existing
# path can run a `unit`-tier member at ranks 1-6:
#
#   * the tier registers every test at exactly one rank
#     (tests/unit_tests/CMakeLists.txt, `set(_beatnik_unit_ranks 1)`), so a
#     ctest entry never sweeps;
#   * in spack mode there is no build tree and therefore no ctest at all
#     (CLAUDE.md "Build mode");
#   * scripts/tuolumne/unit_tests.flux pins its allocation to ONE node while
#     deriving each launch's width from BEATNIK_UNIT_RANKS as ceil(n/4), so
#     BEATNIK_UNIT_RANKS=5 or 6 asks for two nodes inside a one-node
#     allocation.
#
# So this allocates TWO nodes -- tuolumne runs 4 ranks per node and the sweep
# reaches 6 -- and loops the one binary over the six rank counts. The tier
# runner is left alone: it stays the one-rank diagnostic path, and this member
# lands in it for free because it discovers its tests.
#
# THE TEST WRITES NO CHECKPOINTS, so BEATNIK_TEST_SCRATCH is deliberately not
# set: `makeSpinUpParams` leaves the checkpoint directory empty and CheckpointIO
# is a no-op without one. Nothing here touches a filesystem at all.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# iterative test work at 2562 vertices, not a large queued run, and
# t3_fmm_velocity.flux does the same.
#
# Usage:  flux batch scripts/tuolumne/t4_fmm_vs_direct.flux [RANKS...]
#
#   RANKS...  rank counts to run; default `1 2 3 4 5 6`, which is the exit
#             criterion. Overriding it is for DEBUGGING A FAILURE, never for
#             reporting T4: a narrowed sweep silently changes what DONE means
#             and the substitution is invisible in the diff. In particular
#             4 stays in -- Canopy's three-component gradient bodies are known
#             wrong at exactly 4 ranks under LaplaceKernel
#             (canopy/README.md:562-588, risk R4), so if this fails at 4 and
#             nowhere else that is the finding, to be raised upstream rather
#             than accommodated.
#
# Then read beatnik_t4_fmm_vs_direct.<jobid>.log in the submitting directory.
# The job's own exit status is meaningful: the test decides its own verdict and
# returns non-zero on failure (tests/unit_tests/Beatnik_TestAssert.hpp), and
# this script aggregates.
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
    echo "[t4] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance -- a number in the progress log is only reusable if a later
## session can tell which toolchain produced it.
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[t4] spack env status:"
spack env status 2>&1 | sed 's/^/[t4]   /'
echo "[t4] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t4] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"

EXE="$(beatnik_exe Beatnik_Test_FmmVsDirect)" || exit 1
echo "[t4] exe     = ${EXE}"
echo "[t4] exe mtime = $(date -r "${EXE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"

if [ "$#" -gt 0 ]; then
    _ranks=( "$@" )
    echo "[t4] *** RANK SET OVERRIDDEN: ${_ranks[*]} -- this is NOT the T4 exit" \
         "criterion, which is 1 2 3 4 5 6 ***"
else
    _ranks=( 1 2 3 4 5 6 )
fi
echo "[t4] ranks   = ${_ranks[*]}"

_rc=0
_pass=0
_fail=0
_names_failed=""

for _np in "${_ranks[@]}"; do
    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/unit_tests.flux and must not be
    # simplified: a wrong binding does not fail, it oversubscribes one device
    # and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))

    echo "[t4] === Beatnik_Test_FmmVsDirect at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[t4] command: flux run --ntasks=${_np} --nodes=${_nodes} --exclusive" \
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
        echo "[t4] PASS np=${_np}"
        _pass=$(( _pass + 1 ))
    else
        echo "[t4] FAIL np=${_np}: rc=${_obs}" >&2
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
    echo "[t4] FAIL: no rank count was run at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t4] SUMMARY: PASS (${_pass}/${_total} rank counts)"
else
    echo "[t4] SUMMARY: FAIL (${_pass}/${_total} rank counts);" \
         "failed:${_names_failed}" >&2
    # A failure at np=4 ALONE is a decomposition-bug signature and not a budget
    # signature (R4). Read the per-check detail lines in this log before
    # touching any tolerance.
fi
exit "${_rc}"
