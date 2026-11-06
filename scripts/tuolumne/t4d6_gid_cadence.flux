#!/bin/bash
# flux: --job-name=beatnik_t4d6
# flux: --nodes=1
# flux: --exclusive
# flux: -t 60m
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
# T4d6's MEASUREMENT JOB (tasks/milestone1.md, submitted by T4d6j). Not a test
# run: no tier, no gate, no pass/fail criterion beyond "every launch completed
# its step budget". Every judgment is T4d6's, made afterwards from the
# `[m1run] key=value` lines Beatnik_Test_Milestone1Run prints every step.
#
# Three groups of launches, in this order:
#
#   cadence   600 steps at the milestone defaults, gid renumbering at the
#             RemeshParams default (every 50 remesh calls), HIP, ranks 1 and 4
#   control   the same at rank 1 on HIP with renumbering OFF (cadence 0), so
#             the gid-space ratio must grow
#   umbrella  50 steps at the milestone defaults, SERIAL and HIP, ranks 1 and
#             4 -- the T4d exit criterion's volume drift, minimum quality and
#             cumulative splits / collapses / flips / smooth_steps
#
# THE SELECTOR, $2. `measure` (the default) runs exactly the seven rows
# above and writes under /p/lustre5/stewartj/beatnik/T4d6j/out. `diag` runs
# only the two diag rows -- 600 steps, 4 ranks, HIP, NO checkpoints,
# renumbering every 50 (diag) and never (diag0) -- and writes under
# /p/lustre5/stewartj/beatnik/T4d6/out. T4d6 uses them to attribute per-rank
# RSS growth at constant face count: to the gid space or its renumbering, or
# to neither, and apart from the 4-rank checkpoint writes. The
# selector is a positional because the task loop's submit wrapper passes
# script arguments after `--` and takes no options:
#
#   "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/t4d6_gid_cadence.flux \
#       -- /p/lustre5/stewartj/beatnik/T4d6/stage diag
#
# THE BINARIES ARE STAGED, NOT RESOLVED. The job runs
# Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP} from the staged directory given
# as $1 (default /p/lustre5/stewartj/beatnik/T4d6j/stage), never through
# beatnik_exe, which resolves the live dev view: a dev-env `spack install`
# during the job must not change what it runs (tasks/milestone1.md
# Conventions, "Long jobs"). Provenance is the stage's COMMIT file, not
# `git rev-parse HEAD`. Beatnik, Tessera and Canopy are header-only in these
# binaries, so the copied executable freezes the code under test.
#
# ENVIRONMENT AND QUEUE: the DEV spack env's runtime libraries, through the
# resolver -- BEATNIK_USE_PROD is refused -- submitted to pdebug.
#
# THE DEADLINE GUARD. pdebug kills the job at the wall, and a job killed at the
# wall leaves the queue looking exactly like one that passed. So each row
# carries a wall-time estimate, and a launch whose estimate does not fit the
# remaining budget is SKIPPED with a loud message and a non-zero exit rather
# than started and killed halfway. The estimates take M1-G2's Python wall of
# 707 s for 600 steps (tasks/milestone1-progress-log.md, ## M1-G2) as the
# bound for each 600-step launch, padded; no Beatnik dynamic-remesh run had
# been timed when they were written.
#
# Submit through the task loop's wrapper from the repo root, then read the
# beatnik_t4d6.<jobid>.log it writes to cwd. Wait with `flux job status
# <jobid>`, never `flux job attach`: attach forwards SIGTERM to the job, and a
# killed waiter kills the job with it.
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
    echo "[t4d6] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[t4d6] FAIL: BEATNIK_USE_PROD=1. T4d6 measures the DEV env's" >&2
    echo "  staged binaries; the prod env must not be mixed in, and must not" >&2
    echo "  be rebuilt under a live job." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Parameters
##--------------------------------------------------------------------------##
BEATNIK_T4D6_STAGE="${1:-/p/lustre5/stewartj/beatnik/T4d6j/stage}"
BEATNIK_T4D6_SELECT="${2:-measure}"
BEATNIK_T4D6_DRIVER="Beatnik_Test_Milestone1Run"

# Parent of the per-launch output directories. MUST be on a PARALLEL
# filesystem: the checkpoints go through MPI-IO and a node-local scratch fails
# every launch that spans more than one node (CLAUDE.md "Minimum test set").
# Each selector has its own root, so the diag launches never overwrite the
# measurement's output.
case "${BEATNIK_T4D6_SELECT}" in
    measure) _default_out=/p/lustre5/stewartj/beatnik/T4d6j/out ;;
    diag)    _default_out=/p/lustre5/stewartj/beatnik/T4d6/out ;;
    *)
        echo "[t4d6] FAIL: unknown selector '${BEATNIK_T4D6_SELECT}'" \
             "(expected 'measure' or 'diag')." >&2
        exit 1
        ;;
esac
BEATNIK_T4D6_OUT_ROOT="${BEATNIK_T4D6_OUT_ROOT:-${_default_out}}"

# Seconds of the 60m wall the launches may spend. The remainder is slack for
# startup, the final flush and the report below.
BEATNIK_T4D6_BUDGET="${BEATNIK_T4D6_BUDGET:-3420}"

# The launches. One row per launch:
#   group backend ranks steps checkpoint_every renumber_every estimate_seconds
# Renumber cadence 50 is RemeshParams::renumber_gids_every's default, the
# value a milestone run uses; 0 is never. Checkpoints every 10, as the
# milestone run writes them. The measurement rows' estimates sum to 3300 s,
# inside the budget. The two diag rows (renumbering every 50, and never) run
# only under the `diag` selector, and the measurement rows only under
# `measure`.
SWEEP_ALL="
cadence  HIP    1 600 10 50 900
cadence  HIP    4 600 10 50 900
control  HIP    1 600 10 0  900
umbrella SERIAL 1 50  10 50 150
umbrella SERIAL 4 50  10 50 150
umbrella HIP    1 50  10 50 150
umbrella HIP    4 50  10 50 150
diag     HIP    4 600 0  50 300
diag0    HIP    4 600 0  0  300
"
if [ "${BEATNIK_T4D6_SELECT}" = "diag" ]; then
    SWEEP="$(printf '%s\n' "${SWEEP_ALL}" | awk '$1 ~ /^diag/')"
else
    SWEEP="$(printf '%s\n' "${SWEEP_ALL}" | awk 'NF && $1 !~ /^diag/')"
fi

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[t4d6] stage: ${BEATNIK_T4D6_STAGE}"
if [ -f "${BEATNIK_T4D6_STAGE}/COMMIT" ]; then
    echo "[t4d6] stage/COMMIT:"
    sed 's/^/  /' "${BEATNIK_T4D6_STAGE}/COMMIT"
else
    echo "[t4d6] FAIL: ${BEATNIK_T4D6_STAGE}/COMMIT is missing; the stage is" \
         "not a staged directory." >&2
    exit 1
fi
for _b in SERIAL HIP; do
    _exe="${BEATNIK_T4D6_STAGE}/${BEATNIK_T4D6_DRIVER}_MPI_${_b}"
    echo "[t4d6] ${_exe}:"
    ls -l "${_exe}" 2>&1 | sed 's/^/  /'
    sha256sum "${_exe}" 2>&1 | sed 's/^/  /'
    echo "[t4d6] ldd (non-system):"
    ldd "${_exe}" 2>&1 | grep -E "spack|not found" | sed 's/^/  /'
done
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[t4d6] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3
fi
echo "[t4d6] submit command: \"\$TASKLOOP_SUBMIT\" scripts/tuolumne/t4d6_gid_cadence.flux -- ${*}"
echo "[t4d6] selector: ${BEATNIK_T4D6_SELECT}"
echo "[t4d6] rows:"
printf '%s\n' "${SWEEP}" | sed 's/^/  /'
echo "[t4d6] hostname: $(hostname)"
echo "[t4d6] date: $(date -Is)"
echo "[t4d6] out_root=${BEATNIK_T4D6_OUT_ROOT} budget=${BEATNIK_T4D6_BUDGET}s"
echo "=================================================================="

_t_job0="$(date +%s)"
_rc=0
_launched=0
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
    _group="$1"
    _backend="$2"
    _np="$3"
    _steps="$4"
    _ckpt="$5"
    _renum="$6"
    _estimate="$7"

    _tag="${_group}_${_backend}_np${_np}"
    _elapsed=$(( $(date +%s) - _t_job0 ))
    _remaining=$(( BEATNIK_T4D6_BUDGET - _elapsed ))

    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[t4d6] === SKIPPED ${_tag}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. THE SWEEP DOES NOT FIT the walltime." >&2
        echo "  This is a performance finding for the progress log, not a" >&2
        echo "  reason to run fewer steps." >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    _exe="${BEATNIK_T4D6_STAGE}/${BEATNIK_T4D6_DRIVER}_MPI_${_backend}"
    if [ ! -x "${_exe}" ]; then
        echo "[t4d6] FAIL: ${_exe} is not an executable in the stage" >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    # One output directory PER LAUNCH, on lustre, deleted and recreated
    # immediately before that launch, so stale output from an earlier
    # submission cannot be read back as this one's.
    _out="${BEATNIK_T4D6_OUT_ROOT}/${_tag}"
    export BEATNIK_TEST_SCRATCH="${_out}"
    rm -rf "${_out}"
    mkdir -p "${_out}" || {
        echo "[t4d6] FAIL: cannot create ${_out}" >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    }

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from milestone1_divergence.flux and must not be
    # simplified: a wrong binding does not fail, it oversubscribes one device
    # and returns a plausible wall time that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t4d6] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s):" \
         "steps=${_steps} checkpoint_every=${_ckpt} renumber_every=${_renum} ==="
    echo "[t4d6] binding: --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1"
    echo "[t4d6] out = ${_out}"
    echo "[t4d6] exe = ${_exe}"

    # /usr/bin/time -v is INSIDE flux run, so each rank reports its own peak
    # resident set size rather than the launcher's. The launch's output also
    # goes to launch.log in its own directory.
    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        /usr/bin/time -v "${_exe}" "${_steps}" "${_ckpt}" "${_renum}" \
        2>&1 | tee "${_out}/launch.log"
    _launch_rc="${PIPESTATUS[0]}"
    _t1="$(date +%s)"
    _launched=$(( _launched + 1 ))
    if [ "${_launch_rc}" -ne 0 ]; then
        echo "[t4d6] FAIL: ${_tag} exited ${_launch_rc}" >&2
        _rc=1
    fi

    _wall=$(( _t1 - _t0 ))
    echo "[t4d6] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s" \
         "rc=${_launch_rc}" \
         "s_per_step=$(awk -v w="${_wall}" -v s="${_steps}" 'BEGIN{printf "%.6f", w/s}')"
    echo "[t4d6] checkpoints written:" \
         "$(find "${_out}" -name '*_step*.h5' | wc -l)"
done 3<<EOF
${SWEEP}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( $(date +%s) - _t_job0 ))
echo "[t4d6] launched=${_launched} skipped=${_skipped} total=${_total}s" \
     "budget=${BEATNIK_T4D6_BUDGET}s"
if [ "${_launched}" -eq 0 ]; then
    echo "[t4d6] FAIL: the sweep launched nothing." >&2
    _rc=1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t4d6] PASS: every launch completed its step budget. T4d6 judges"
    echo "  the [m1run] lines; nothing here has judged them."
else
    echo "[t4d6] FAIL: see the messages above. A partial sweep is NOT a" >&2
    echo "  measurement -- do not judge one without saying so." >&2
fi
exit "${_rc}"
