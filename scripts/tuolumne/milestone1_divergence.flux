#!/bin/bash
# flux: --job-name=beatnik_m1d1
# flux: --nodes=1
# flux: --exclusive
# flux: -t 20m
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
# M1-D1's MEASUREMENT SWEEP (tasks/milestone1.md). Not a test run: no tier, no
# gate, no pass/fail criterion beyond "every launch completed its step budget".
#
# It launches Beatnik_Test_Milestone0Run at subdivision level 2 for 600 steps
# with a checkpoint every 10 -- with those positionals its makeParams() is the
# M1-G1 command field for field -- and leaves four 61-checkpoint series on
# lustre. Every COMPARISON is made afterwards, offline, by
# tests/regression_tests/milestone0_ladder.py against
# tests/regression_tests/direct-solve-600-steps/gold/. Nothing here compares
# anything, which is why this script has no notion of a tolerance.
#
#     matrix : level 2 x (SERIAL, HIP) x (ranks 1, 4), 600 steps, every 10
#
# ENVIRONMENT AND QUEUE: the DEV spack env -- BEATNIK_USE_PROD is NOT set here
# and must not be -- submitted to pdebug. Level 2 is 162 vertices, and M0-D1
# measured level 3 (642 vertices) at 2000 steps on SERIAL np1 at 86 s, so the
# four launches take minutes and `-t 20m` makes a hang fail fast.
#
# THE DEADLINE GUARD. pdebug kills the job at the wall, and a job killed at the
# wall leaves the queue looking exactly like one that passed while having
# written a truncated checkpoint series. So each row carries a wall-time
# estimate, and a launch whose estimate does not fit the remaining budget is
# SKIPPED with a loud message and a non-zero exit rather than started and
# killed halfway.
#
# Submit through the task loop's wrapper (or `flux batch` by hand) from the
# repo root, then read the beatnik_m1d1.<jobid>.log it writes to cwd. Wait with
# `flux job status <jobid>`, never `flux job attach`: attach forwards SIGTERM to
# the job, and a killed waiter kills the job with it.
#
# --nodes=1 covers the 4-rank case at tuolumne's 4-ranks-per-node.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs
# none today.

# Pin the repo root. `flux batch` copies this script into a per-job spool
# directory, so BASH_SOURCE points at /var/tmp/... and cannot find the checkout.
# It does preserve the submitting working directory, so walk up from PWD;
# BASH_SOURCE still covers a direct `bash` run.
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
    echo "[m1d1] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[m1d1] FAIL: BEATNIK_USE_PROD=1. M1-D1 measures the DEV env; a" >&2
    echo "  production-env number is not comparable with the rest of the" >&2
    echo "  sweep and the prod env must not be rebuilt under a live job." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[m1d1] spack env status:"
spack env status 2>&1 || echo "[m1d1] (spack env status unavailable)"
echo "[m1d1] installed beatnik:"
( cd "${BEATNIK_REPO}" && spack find -l --format '{hash:7} {name}{@version}{%compiler}' beatnik 2>&1 ) \
    || echo "[m1d1] (spack find unavailable)"
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[m1d1] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3
fi
echo "[m1d1] commit: $(cd "${BEATNIK_REPO}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1d1] git status --short:"
( cd "${BEATNIK_REPO}" && git status --short 2>/dev/null | head -20 )
echo "[m1d1] submit command: flux batch scripts/tuolumne/milestone1_divergence.flux"
echo "[m1d1] hostname: $(hostname)"
echo "[m1d1] date: $(date -Is)"
echo "=================================================================="

##--------------------------------------------------------------------------##
## Parameters
##--------------------------------------------------------------------------##
BEATNIK_M1_DRIVER="Beatnik_Test_Milestone0Run"
BEATNIK_M1_LEVEL=2
BEATNIK_M1_STEPS=600
BEATNIK_M1_CKPT_EVERY=10

# Parent of the per-run output directories. MUST be on a PARALLEL filesystem:
# the checkpoints go through MPI-IO and a node-local scratch fails every launch
# that spans more than one node (CLAUDE.md "Minimum test set").
BEATNIK_M1_SCRATCH_ROOT="${BEATNIK_M1_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/M1-D1/scratch}"

# Seconds of the 20m wall this sweep may spend launching. The remainder is
# slack for startup, the final flush and the report below.
BEATNIK_M1_BUDGET="${BEATNIK_M1_BUDGET:-1080}"

# The sweep. One row per launch: `backend ranks estimate_seconds`. The
# estimates scale M0-D1's measured level-3 per-step costs (job f3TT4psJ8it7)
# down to 600 steps without crediting level 2's smaller N, so they are padded.
SWEEP="
HIP    1 60
HIP    4 60
SERIAL 4 90
SERIAL 1 120
"

echo "[m1d1] driver=${BEATNIK_M1_DRIVER} level=${BEATNIK_M1_LEVEL}" \
     "steps=${BEATNIK_M1_STEPS} checkpoint_every=${BEATNIK_M1_CKPT_EVERY}" \
     "scratch_root=${BEATNIK_M1_SCRATCH_ROOT} budget=${BEATNIK_M1_BUDGET}s"

_t_job0="$(date +%s)"
_rc=0
_launched=0
_skipped=0

##--------------------------------------------------------------------------##
## The sweep
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin, so with
# the row list on stdin the first launched binary swallows every remaining row
# and the sweep silently runs only its first member.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _backend="$1"
    _np="$2"
    _estimate="$3"

    _tag="sub${BEATNIK_M1_LEVEL}_${_backend}_np${_np}_steps${BEATNIK_M1_STEPS}"
    _elapsed=$(( $(date +%s) - _t_job0 ))
    _remaining=$(( BEATNIK_M1_BUDGET - _elapsed ))

    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[m1d1] === SKIPPED ${_tag}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. THE SWEEP DOES NOT FIT the walltime." >&2
        echo "  This is a performance finding for the progress log, not a" >&2
        echo "  reason to run fewer steps or a coarser checkpoint interval." >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    _exe="$(beatnik_exe "${BEATNIK_M1_DRIVER}_MPI_${_backend}")" || {
        echo "[m1d1] FAIL: cannot resolve ${BEATNIK_M1_DRIVER}_MPI_${_backend}" >&2
        _rc=1
        continue
    }

    # One output directory PER RUN, on lustre, deleted and recreated immediately
    # before that run, so a stale checkpoint from an earlier sweep cannot be
    # read back as this run's output.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_M1_SCRATCH_ROOT}/${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
        echo "[m1d1] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        continue
    }

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from milestone0_divergence.flux and must not be simplified:
    # a wrong binding does not fail, it oversubscribes one device and returns a
    # plausible wall time that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[m1d1] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[m1d1] binding: --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1"
    echo "[m1d1] scratch = ${BEATNIK_TEST_SCRATCH}"
    echo "[m1d1] exe     = ${_exe}"

    # /usr/bin/time -v is INSIDE flux run, so each rank reports its own peak
    # resident set size rather than the launcher's.
    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        /usr/bin/time -v "${_exe}" "${BEATNIK_M1_LEVEL}" "${BEATNIK_M1_STEPS}" \
        "${BEATNIK_M1_CKPT_EVERY}" || {
            echo "[m1d1] FAIL: ${_tag} exited non-zero" >&2
            _rc=1
        }
    _t1="$(date +%s)"
    _launched=$(( _launched + 1 ))

    _wall=$(( _t1 - _t0 ))
    echo "[m1d1] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s" \
         "s_per_step=$(awk -v w="${_wall}" -v s="${BEATNIK_M1_STEPS}" 'BEGIN{printf "%.6f", w/s}')"
    echo "[m1d1] checkpoints written:" \
         "$(find "${BEATNIK_TEST_SCRATCH}" -name '*_step*.h5' | wc -l)"
done 3<<EOF
${SWEEP}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( $(date +%s) - _t_job0 ))
echo "[m1d1] launched=${_launched} skipped=${_skipped} total=${_total}s" \
     "budget=${BEATNIK_M1_BUDGET}s"
if [ "${_launched}" -eq 0 ]; then
    echo "[m1d1] FAIL: the sweep launched nothing." >&2
    _rc=1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[m1d1] PASS: every launch in the matrix completed its step budget."
    echo "[m1d1] Next: tabulate offline with"
    echo "  tests/regression_tests/milestone0_ladder.py pair --run <dir> --ref <gold>"
    echo "  tests/regression_tests/milestone0_ladder.py growth --run <dir> --ref <gold>"
else
    echo "[m1d1] FAIL: see the messages above. A partial sweep is NOT a" >&2
    echo "  measurement -- do not tabulate one without saying so." >&2
fi
exit "${_rc}"
