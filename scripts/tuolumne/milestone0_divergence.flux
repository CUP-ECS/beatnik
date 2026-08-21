#!/bin/bash
# flux: --job-name=beatnik_m0div
# flux: --nodes=1
# flux: --exclusive
# flux: -t 1h
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
# M0-D1's MEASUREMENT SWEEP. Not a test run: no tier, no gate, no pass/fail
# criterion beyond "every launch completed its step budget".
#
# It launches Beatnik_Test_Milestone0Run over M0-D1's matrix and leaves 2000-step
# checkpoint series on lustre. Every COMPARISON is made afterwards, offline, by
# tests/regression_tests/milestone0_ladder.py -- nothing here compares anything,
# which is why this script has no notion of a tolerance.
#
#     matrix : (level 3, level 4) x (SERIAL, HIP) x (ranks 1, 4)
#     plus   : level 2 at --steps 0, both backends at 1 rank, for the step-0
#              generator gate of M0-D1 step 1 (risk M0-R5). Levels 3 and 4 get
#              their step-0 files from the full runs above, so only level 2
#              needs a launch of its own.
#
# ENVIRONMENT AND QUEUE, per M0-D1: the DEV spack env -- BEATNIK_USE_PROD is NOT
# set here and must not be -- submitted `-q pdebug -t 1h`. The reference's own
# single-core Python produced both 2000-step gold sets in about 110 minutes
# total, so one hour is the budget for this whole sweep and a sweep that does
# not fit it is a PERFORMANCE FINDING for the progress log, not a reason to move
# queues, lengthen the walltime, or quietly run fewer steps.
#
# THE DEADLINE GUARD IS THE POINT OF THE ESTIMATES BELOW. pdebug kills the job
# at the wall, and a job killed at the wall leaves the queue looking exactly
# like one that passed -- while having written a truncated checkpoint series
# that milestone0_ladder.py would happily tabulate. So each row carries a
# measured wall-time estimate, the runs are ordered cheapest-first, and a launch
# whose estimate does not fit the remaining budget is SKIPPED with a loud
# message and a non-zero exit rather than started and killed halfway.
#
# Submit with:  flux batch scripts/tuolumne/milestone0_divergence.flux
# Then read the beatnik_m0div.<jobid>.log it writes to cwd. Attach to wait:
#   jobid=$(flux batch scripts/tuolumne/milestone0_divergence.flux)
#   flux job attach "$jobid"
#
# BEATNIK_M0_MODE=probe runs the same matrix at 25 steps instead of 2000. That
# is how the estimates in the table below were measured, and re-measuring them
# is what to do after any change that could move the per-step cost.
#
# --nodes=1 covers the 4-rank case at tuolumne's 4-ranks-per-node.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs
# none today.

# Pin the repo root. `flux batch` copies this script into a per-job spool
# directory, so BASH_SOURCE points at /var/tmp/... and cannot find the checkout.
# It does preserve the submitting working directory, so walk up from PWD;
# BASH_SOURCE still covers a direct `bash` run. Copied in shape from
# run_milestone.flux, which already paid for this.
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
    echo "[m0d1] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
# The numbers this job prints are only reusable if a later session can tell
# which toolchain produced them, so this block is not optional decoration.
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[m0d1] spack env status:"
spack env status 2>&1 || echo "[m0d1] (spack env status unavailable)"
echo "[m0d1] compiler:"
( cd "${BEATNIK_REPO}" && spack find --format '{name}{@version}{%compiler}' beatnik 2>&1 ) \
    || echo "[m0d1] (spack find unavailable)"
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[m0d1] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3
fi
echo "[m0d1] commit: $(cd "${BEATNIK_REPO}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m0d1] git status --short:"
( cd "${BEATNIK_REPO}" && git status --short 2>/dev/null | head -20 )
echo "[m0d1] submit command: flux batch scripts/tuolumne/milestone0_divergence.flux"
echo "[m0d1] mode: ${BEATNIK_M0_MODE:-full}"
echo "[m0d1] hostname: $(hostname)"
echo "[m0d1] date: $(date -Is)"
echo "=================================================================="

##--------------------------------------------------------------------------##
## Parameters
##--------------------------------------------------------------------------##
BEATNIK_M0_MODE="${BEATNIK_M0_MODE:-full}"
BEATNIK_M0_DRIVER="${BEATNIK_M0_DRIVER:-Beatnik_Test_Milestone0Run}"
BEATNIK_M0_CKPT_EVERY="${BEATNIK_M0_CKPT_EVERY:-25}"

# Parent of the per-run output directories. MUST be on a PARALLEL filesystem:
# the checkpoints go through MPI-IO and a node-local scratch fails every launch
# that spans more than one node (CLAUDE.md "Minimum test set").
BEATNIK_M0_SCRATCH_ROOT="${BEATNIK_M0_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/milestone0}"

# Seconds of the 1h wall this sweep may spend launching. The remainder is slack
# for startup, the final flush and the report below.
BEATNIK_M0_BUDGET="${BEATNIK_M0_BUDGET:-3300}"

# The sweep. One row per launch: `level backend ranks steps estimate_seconds`.
#
# ORDERED CHEAPEST-FIRST, deliberately. If the budget runs out the runs that
# are missing are the expensive ones, the guard names them, and the exit status
# is non-zero -- rather than the sweep dying mid-write on an arbitrary row.
#
# The estimates are MEASURED, by this same script under BEATNIK_M0_MODE=probe,
# and are recorded in tasks/milestone0-progress-log.md under `## M0-D1` beside
# the wall times they predicted. They are padded above the measurement, because
# a low estimate spends the budget and a high one only skips a run early.
if [ "${BEATNIK_M0_MODE}" = "probe" ]; then
    BEATNIK_M0_STEPS="${BEATNIK_M0_STEPS:-25}"
    SWEEP="
2 SERIAL 1 0 60
2 HIP    1 0 60
3 HIP    1 ${BEATNIK_M0_STEPS} 180
3 HIP    4 ${BEATNIK_M0_STEPS} 180
3 SERIAL 1 ${BEATNIK_M0_STEPS} 300
3 SERIAL 4 ${BEATNIK_M0_STEPS} 300
4 HIP    1 ${BEATNIK_M0_STEPS} 300
4 HIP    4 ${BEATNIK_M0_STEPS} 300
4 SERIAL 1 ${BEATNIK_M0_STEPS} 600
4 SERIAL 4 ${BEATNIK_M0_STEPS} 600
"
else
    BEATNIK_M0_STEPS="${BEATNIK_M0_STEPS:-2000}"
    # Estimates below are 2000 x the per-step solve cost the probe measured
    # (job f3TSuF7DFxAB, 25 steps per row), padded ~1.4-2x and rounded up:
    #
    #   L3 HIP np1 0.005385  np4 0.014058   L4 HIP np1 0.008792  np4 0.019211
    #   L3 SER np1 0.043030  np4 0.021891   L4 SER np1 0.644068  np4 0.187007
    #
    # which predicts 1887 s of solve for the eight 2000-step rows. Note the two
    # SERIAL rows where 4 ranks are FASTER than 1 -- the direct BR sum is
    # O(N^2/P) with no GPU to saturate -- and the two HIP rows where they are
    # slower, which is 2562 vertices not filling one MI300A.
    SWEEP="
2 SERIAL 1 0 60
2 HIP    1 0 60
3 HIP    1 ${BEATNIK_M0_STEPS} 120
3 HIP    4 ${BEATNIK_M0_STEPS} 150
4 HIP    1 ${BEATNIK_M0_STEPS} 150
4 HIP    4 ${BEATNIK_M0_STEPS} 180
3 SERIAL 4 ${BEATNIK_M0_STEPS} 180
3 SERIAL 1 ${BEATNIK_M0_STEPS} 240
4 SERIAL 4 ${BEATNIK_M0_STEPS} 600
4 SERIAL 1 ${BEATNIK_M0_STEPS} 1800
"
fi

echo "[m0d1] driver=${BEATNIK_M0_DRIVER} steps=${BEATNIK_M0_STEPS}" \
     "checkpoint_every=${BEATNIK_M0_CKPT_EVERY}" \
     "scratch_root=${BEATNIK_M0_SCRATCH_ROOT}" \
     "budget=${BEATNIK_M0_BUDGET}s"

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[m0d1] FAIL: BEATNIK_USE_PROD=1. M0-D1 measures the DEV env; a" >&2
    echo "  production-env number is not comparable with the rest of the" >&2
    echo "  sweep and the prod env must not be rebuilt under a live job." >&2
    exit 1
fi

_t_job0="$(date +%s)"
_rc=0
_launched=0
_skipped=0

##--------------------------------------------------------------------------##
## The sweep
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin, so with
# the row list on stdin the first launched binary swallows every remaining row
# and the sweep silently runs only its first member -- reporting a plausible
# partial table. run_milestone.flux already paid for this; copied in shape.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _level="$1"
    _backend="$2"
    _np="$3"
    _steps="$4"
    _estimate="$5"

    _tag="sub${_level}_${_backend}_np${_np}_steps${_steps}"
    _elapsed=$(( $(date +%s) - _t_job0 ))
    _remaining=$(( BEATNIK_M0_BUDGET - _elapsed ))

    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[m0d1] === SKIPPED ${_tag}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. THE SWEEP DOES NOT FIT pdebug's 1h cap." >&2
        echo "  This is a performance finding for tasks/milestone0-progress-log.md," >&2
        echo "  not a reason to run fewer steps or a coarser checkpoint interval." >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    _exe="$(beatnik_exe "${BEATNIK_M0_DRIVER}_MPI_${_backend}")" || {
        echo "[m0d1] FAIL: cannot resolve ${BEATNIK_M0_DRIVER}_MPI_${_backend}" >&2
        _rc=1
        continue
    }

    # One output directory PER RUN, on lustre, deleted and recreated immediately
    # before that run -- so a stale checkpoint from an earlier sweep cannot be
    # read back as this run's output and tabulated as a divergence.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_M0_SCRATCH_ROOT}/${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
        echo "[m0d1] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        continue
    }

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from run_milestone.flux:195-203 and must not be simplified:
    # a wrong binding does not fail, it oversubscribes one device and returns a
    # plausible wall time that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[m0d1] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[m0d1] binding: --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1"
    echo "[m0d1] scratch = ${BEATNIK_TEST_SCRATCH}"
    echo "[m0d1] exe     = ${_exe}"

    # /usr/bin/time -v is INSIDE flux run, so each rank reports its own peak
    # resident set size rather than the launcher's. GPU-side memory is out of
    # scope for M0-D1 -- there is no mechanism for it here.
    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        /usr/bin/time -v "${_exe}" "${_level}" "${_steps}" \
        "${BEATNIK_M0_CKPT_EVERY}" || {
            echo "[m0d1] FAIL: ${_tag} exited non-zero" >&2
            _rc=1
        }
    _t1="$(date +%s)"
    _launched=$(( _launched + 1 ))

    _wall=$(( _t1 - _t0 ))
    if [ "${_steps}" -gt 0 ]; then
        echo "[m0d1] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s" \
             "s_per_step=$(awk -v w="${_wall}" -v s="${_steps}" 'BEGIN{printf "%.6f", w/s}')"
    else
        echo "[m0d1] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s"
    fi
    echo "[m0d1] checkpoints written:" \
         "$(find "${BEATNIK_TEST_SCRATCH}" -name '*_step*.h5' | wc -l)"
done 3<<EOF
${SWEEP}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( $(date +%s) - _t_job0 ))
echo "[m0d1] launched=${_launched} skipped=${_skipped} total=${_total}s" \
     "budget=${BEATNIK_M0_BUDGET}s"
if [ "${_launched}" -eq 0 ]; then
    echo "[m0d1] FAIL: the sweep launched nothing." >&2
    _rc=1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[m0d1] PASS: every launch in the matrix completed its step budget."
    echo "[m0d1] Next: tabulate offline with"
    echo "  tests/regression_tests/milestone0_ladder.py pair --run <dir> --ref <dir>"
    echo "  tests/regression_tests/milestone0_ladder.py series --dir <dir>"
else
    echo "[m0d1] FAIL: see the messages above. A partial sweep is NOT a" >&2
    echo "  measurement -- do not tabulate one without saying so." >&2
fi
exit "${_rc}"
