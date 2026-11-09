#!/bin/bash
# flux: --job-name=beatnik_m1_milestone
# flux: --nodes=1
# flux: --exclusive
# flux: -t 10m
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
# THE MILESTONE-1 COMPARISON JOB (tasks/milestone1.md, M1-T2j and M1-A1j).
# Runs only the `milestone` member Beatnik_Test_Milestone1Bubble, from a
# STAGED directory, in six launches, in this order:
#
#   compare          SERIAL np1, SERIAL np4, HIP np1, HIP np4 -- the
#                    1610-step run: the step-0 field-wise comparison against
#                    the M1-G2 gold, the time-matched statistics checks
#                    against the M1-G3 steps 0-1610 gold set (compiled into
#                    the test), and the liveness assertions
#   fail-step0       HIP np1 -- failure direction (a): the step-10 checkpoint
#                    against the STEP-0 gold must exit exactly 1
#   fail-no-cleanup  HIP np1 -- failure direction (b): --no-isotropic-cleanup
#                    forced, a statistics check must fail
#
# Each launch prints `[milestone] <PASS|FAIL> <target> np=<N> in <S>s` (the
# two failure directions add `mode=<mode>` before `in`), and the job ends
# `[milestone] SUMMARY: <PASS|FAIL> (<passed>/<run> launches)`, the shapes
# run_milestone.flux prints. A job killed at its walltime prints no SUMMARY.
# A compare launch that FAILS because a statistics band missed is a
# measurement, not a defect: the test checks every gold checkpoint and its
# `[m1stat]`, `[m1statworst]` and `[m1final]` lines say where and by how much
# (see the test's file header).
#
# WHY NOT run_milestone.flux: it resolves binaries from the live dev view and
# runs every member it is given, including the hours-long FMM members.
#
# THE BINARIES ARE STAGED, NOT RESOLVED. $1 (required) is a directory holding
#   Beatnik_Test_Milestone1Bubble_MPI_{SERIAL,HIP}
#   regression_tests/compare_output.py
#   regression_tests/milestone1-bubble/gold/*_step*.npz  (only step 0 is read)
#   COMMIT   (`git rev-parse HEAD` and the `git status --porcelain` count,
#            written at staging time)
# copied from the dev view's share/Beatnik/tests/ in their manifest-relative
# layout. The job reads nothing else from the stage's surroundings, never
# calls beatnik_exe, and echoes stage/COMMIT, never `git rev-parse HEAD`:
# the tree may move between submission and start (tasks/milestone1.md
# Conventions, "Long jobs"). Beatnik, Tessera and Canopy are header-only in
# these binaries, so the copied executable freezes the code under test.
#
# OUTPUT: each launch writes its checkpoints and a launch.log under
# $2 (default /p/lustre5/stewartj/beatnik/<stage's parent name>/out)/<tag>/,
# with <tag> = <target>_np<N> for a comparison and <target>_np<N>_<mode> for
# a failure direction. On lustre: the checkpoints go through MPI-IO, so a
# node-local scratch fails every launch that spans more than one node.
#
# ENVIRONMENT AND QUEUE: the DEV spack env's runtime libraries, through the
# resolver -- BEATNIK_USE_PROD is refused -- submitted to pdebug (one node,
# under 45 minutes: systems/tuolumne/claude.md, "Choosing a queue").
#
# WALLTIME AND THE DEADLINE GUARD. -t 10m is about 2x the expected job of
# about 4.5 min. M1-T2 measured run walls per 600 steps of 18.9 s (SERIAL,
# either rank count), 6.8 s (HIP np1) and 9.8 s (HIP np4), launch overhead
# 5-9 s, and 26 s of job setup (tasks/milestone1-progress-log.md, ## M1-T2).
# Scaled to 1610 steps and by 1.13 for the gold's late mesh growth to 1216
# faces (## M1-G3), a compare launch is about 70 s (SERIAL), 31 s (HIP np1)
# and 44 s (HIP np4); the comparator runs once per launch. Each row carries an
# estimate of about 2x that, and a launch whose estimate does not fit the
# remaining budget is SKIPPED with a loud message and a non-zero exit rather
# than started and killed at the wall.
#
# Submit through the task loop's wrapper from the repo root:
#
#   "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/m1_milestone_run.flux \
#       -- /p/lustre5/stewartj/beatnik/M1-A1j/stage
#
# then read the beatnik_m1_milestone.<jobid>.log it writes to cwd. Wait with
# `flux job status <jobid>`, never `flux job attach`: attach forwards SIGTERM
# to the job, and a killed waiter kills the job with it.
#
# --nodes=1 covers the 4-rank case at tuolumne's 4 ranks per node.
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
    echo "[milestone] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[milestone] FAIL: BEATNIK_USE_PROD=1. This job runs the DEV env's" >&2
    echo "  staged binaries; the prod env must not be mixed in, and must not" >&2
    echo "  be rebuilt under a live job." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Parameters
##--------------------------------------------------------------------------##
if [ $# -lt 1 ] || [ -z "${1}" ]; then
    echo "[milestone] FAIL: usage: m1_milestone_run.flux <stage-dir> [out-root]" >&2
    exit 1
fi
M1_STAGE="$(cd "${1}" 2>/dev/null && pwd)" || {
    echo "[milestone] FAIL: stage directory '${1}' does not exist." >&2
    exit 1
}
M1_OUT_ROOT="${2:-$(dirname "${M1_STAGE}")/out}"
M1_MEMBER="Beatnik_Test_Milestone1Bubble"
M1_GOLD="${M1_STAGE}/regression_tests/milestone1-bubble/gold"
M1_COMPARE="${M1_STAGE}/regression_tests/compare_output.py"

# Seconds of the 10m wall the launches may spend. The remainder is slack for
# the resolver, the provenance block and the report below.
M1_BUDGET="${M1_BUDGET:-540}"

# The launches. One row per launch:
#   mode backend ranks estimate_seconds
# The estimates sum to 510 s, inside the budget.
SWEEP="
compare         SERIAL 1 140
compare         SERIAL 4 140
compare         HIP    1 70
compare         HIP    4 90
fail-step0      HIP    1 30
fail-no-cleanup HIP    1 40
"

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[milestone] spack env status:"
spack env status 2>&1 | sed 's/^/[milestone]   /'
echo "[milestone] stage: ${M1_STAGE}"
if [ -f "${M1_STAGE}/COMMIT" ]; then
    echo "[milestone] stage/COMMIT:"
    sed 's/^/  /' "${M1_STAGE}/COMMIT"
else
    echo "[milestone] FAIL: ${M1_STAGE}/COMMIT is missing; the stage is not" \
         "a staged directory." >&2
    echo "[milestone] SUMMARY: FAIL (0/0 launches)"
    exit 1
fi
for _b in SERIAL HIP; do
    _exe="${M1_STAGE}/${M1_MEMBER}_MPI_${_b}"
    echo "[milestone] ${_exe}:"
    ls -l "${_exe}" 2>&1 | sed 's/^/  /'
    sha256sum "${_exe}" 2>&1 | sed 's/^/  /'
    echo "[milestone] ldd (non-system):"
    ldd "${_exe}" 2>&1 | grep -E "spack|not found" | sed 's/^/  /'
done
echo "[milestone] comparator: ${M1_COMPARE}"
sha256sum "${M1_COMPARE}" 2>&1 | sed 's/^/  /'
echo "[milestone] gold: ${M1_GOLD}" \
     "($(find "${M1_GOLD}" -name '*_step*.npz' 2>/dev/null | wc -l) numbered .npz)"
_python="${BEATNIK_PYTHON:-python3}"
echo "[milestone] python: $(command -v "${_python}" || echo "${_python} NOT FOUND")"
"${_python}" -c "import sys, numpy, h5py; print('  python', sys.version.split()[0], 'numpy', numpy.__version__, 'h5py', h5py.__version__)" 2>&1
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[milestone] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3
fi
echo "[milestone] submit command: \"\$TASKLOOP_SUBMIT\" scripts/tuolumne/m1_milestone_run.flux -- ${*}"
echo "[milestone] rows:"
printf '%s\n' "${SWEEP}" | sed '/^$/d; s/^/  /'
echo "[milestone] hostname: $(hostname)"
echo "[milestone] date: $(date -Is)"
echo "[milestone] out_root=${M1_OUT_ROOT} budget=${M1_BUDGET}s"
echo "=================================================================="

_t_job0="$(date +%s)"
_rc=0
_run=0
_passed=0
_skipped=0

##--------------------------------------------------------------------------##
## The launches
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin, so with
# the row list on stdin the first launched binary swallows every remaining row
# and the sweep silently runs only its first member.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _mode="$1"
    _backend="$2"
    _np="$3"
    _estimate="$4"

    _target="${M1_MEMBER}_MPI_${_backend}"
    if [ "${_mode}" = "compare" ]; then
        _tag="${_target}_np${_np}"
        _label="np=${_np}"
    else
        _tag="${_target}_np${_np}_${_mode}"
        _label="np=${_np} mode=${_mode}"
    fi

    _elapsed=$(( $(date +%s) - _t_job0 ))
    _remaining=$(( M1_BUDGET - _elapsed ))
    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[milestone] === SKIPPED ${_tag}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. THE SWEEP DOES NOT FIT the walltime." >&2
        echo "  This is a performance finding for the progress log, not a" >&2
        echo "  reason to run fewer steps." >&2
        echo "[milestone] FAIL ${_target} ${_label} in 0s (SKIPPED)"
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    _exe="${M1_STAGE}/${_target}"
    if [ ! -x "${_exe}" ]; then
        echo "[milestone] FAIL: ${_exe} is not an executable in the stage" >&2
        echo "[milestone] FAIL ${_target} ${_label} in 0s (no binary)"
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    # One output directory PER LAUNCH, on lustre, deleted and recreated
    # immediately before that launch, so stale output from an earlier
    # submission cannot be read back as this one's.
    _out="${M1_OUT_ROOT}/${_tag}"
    export BEATNIK_TEST_SCRATCH="${_out}"
    rm -rf "${_out}"
    mkdir -p "${_out}" || {
        echo "[milestone] FAIL: cannot create ${_out}" >&2
        echo "[milestone] FAIL ${_target} ${_label} in 0s (no output dir)"
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    }

    _args=( "${M1_GOLD}" "${M1_COMPARE}" )
    [ "${_mode}" = "compare" ] || _args+=( "${_mode}" )

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from t4d6_gid_cadence.flux and must not be simplified: a
    # wrong binding does not fail, it oversubscribes one device and returns a
    # plausible wall time that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[milestone] === ${_target} at ${_np} rank(s) / ${_nodes} node(s):" \
         "mode=${_mode} ==="
    echo "[milestone] binding: --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1"
    echo "[milestone] out = ${_out}"
    echo "[milestone] exe = ${_exe} ${_args[*]}"

    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "${_args[@]}" \
        2>&1 | tee "${_out}/launch.log"
    _launch_rc="${PIPESTATUS[0]}"
    _t1="$(date +%s)"
    _run=$(( _run + 1 ))
    if [ "${_launch_rc}" -eq 0 ]; then
        _passed=$(( _passed + 1 ))
        echo "[milestone] PASS ${_target} ${_label} in $(( _t1 - _t0 ))s"
    else
        _rc=1
        echo "[milestone] FAIL ${_target} ${_label} in $(( _t1 - _t0 ))s" \
             "(rc=${_launch_rc})"
    fi
    echo "[milestone] LAUNCH_WALL ${_tag} wall=$(( _t1 - _t0 ))s" \
         "estimate=${_estimate}s rc=${_launch_rc}"
    echo "[milestone] checkpoints written:" \
         "$(find "${_out}" -name '*.h5' | wc -l)"
done 3<<EOF
${SWEEP}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( $(date +%s) - _t_job0 ))
echo "[milestone] launched=${_run} skipped=${_skipped} total=${_total}s" \
     "budget=${M1_BUDGET}s"
if [ "${_run}" -eq 0 ]; then
    echo "[milestone] FAIL: the sweep launched nothing." >&2
    _rc=1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[milestone] SUMMARY: PASS (${_passed}/$(( _run + _skipped )) launches)"
else
    echo "[milestone] SUMMARY: FAIL (${_passed}/$(( _run + _skipped )) launches)"
fi
exit "${_rc}"
