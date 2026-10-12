#!/bin/bash
# flux: --job-name=beatnik_cprof_knob_arms
# flux: --nodes=1
# flux: --exclusive
# flux: -t 58m
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
# canopy_profiling, the L5/L6 accuracy question, part 2: which knob closes
# Beatnik's FMM error at levels 5 and 6, and what each costs.
# Beatnik_Probe_FmmCost on HIP, FMM-driven, 50 steps, sample_every 25 (so
# [acc] lands on steps 1, 25 and 50, the reference's states in part 1):
#   L4 np1 and L5/L6 np4 at the baseline (ncrit 8, order 3, theta 0.3), and
#   one knob changed at a time: ncrit 64, order 4, mac_theta 0.2.
#
# Usage:  flux batch scripts/tuolumne/cprof_knob_arms.flux
############################################################################
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root; see t6b_key_demand.flux for why BASH_SOURCE is not enough.
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
    echo "[cps] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[cps] FAIL: BEATNIK_USE_PROD=1. canopy_profiling measures the DEV" >&2
    echo "  env: the production binary has no level-2 timers to read." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
CANOPY_SRC="${CANOPY_SRC:-$(cd "${BEATNIK_REPO}/../canopy" 2>/dev/null && pwd)}"
beatnik_env_summary
echo "[cps] spack env status:"
spack env status 2>&1 | sed 's/^/[cps]   /'
echo "[cps] compiler = $(spack find --format '{name}@{version} {compiler}' beatnik 2>/dev/null | head -1)"
echo "[cps] beatnik commit = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[cps] beatnik dirty  = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s)"
git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | sed 's/^/[cps]   beatnik: /'
export BEATNIK_CANOPY_HASH="$(git -C "${CANOPY_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[cps] canopy src    = ${CANOPY_SRC}"
echo "[cps] canopy commit = ${BEATNIK_CANOPY_HASH}"
echo "[cps] canopy porcelain ($(git -C "${CANOPY_SRC}" status --porcelain 2>/dev/null | wc -l) line(s)):"
git -C "${CANOPY_SRC}" status --porcelain 2>/dev/null | sed 's/^/[cps]   canopy: /'
echo "[cps] canopy spec   = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^%]*' | head -1)"
echo "[cps] submit = flux batch scripts/tuolumne/cprof_knob_arms.flux"


# The binding, copied EXACTLY from t6b_key_demand.flux:231-245.
launch() {
    local _exe="$1" _np="$2"; shift 2
    local _nodes=$(( (_np + 3) / 4 ))
    echo "[cps] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
         "--exclusive --gpus-per-task=1 --cores-per-task=24" \
         "--setopt=mpibind=verbose:1 ${_exe} $*"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "$@"
}
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
_rc=0
_t_job0="$(date +%s)"

_exe="$(beatnik_exe Beatnik_Probe_FmmCost_MPI_HIP)" || exit 1
BEATNIK_CPS_BUDGET="${BEATNIK_CPS_BUDGET:-3300}"
# label level ranks extra-key estimate_seconds   (cheapest first)
SWEEP="
l4base   4 1 ncrit=8      60
l4nc64   4 1 ncrit=64     60
l4o4     4 1 order=4      90
l4th02   4 1 mac_theta=0.2 90
l5base   5 4 ncrit=8      120
l5nc64   5 4 ncrit=64     120
l5o4     5 4 order=4      200
l5th02   5 4 mac_theta=0.2 200
l6base   6 4 ncrit=8      300
l6nc64   6 4 ncrit=64     300
l6o4     6 4 order=4      500
l6th02   6 4 mac_theta=0.2 500
"
[ -n "${BEATNIK_CPS_SWEEP:-}" ] && SWEEP="${BEATNIK_CPS_SWEEP}"
printf '%s\n' "${SWEEP}" | sed '/^$/d; s/^/[cps]   /'
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _label="$1"; _level="$2"; _np="$3"; _knob="$4"; _estimate="$5"
    _run="${_label}_L${_level}_HIP_np${_np}"
    _remaining=$(( BEATNIK_CPS_BUDGET - ( $(date +%s) - _t_job0 ) ))
    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[cps] === SKIPPED ${_run}: ${_remaining}s left, estimate ${_estimate}s ===" >&2
        _rc=1
        continue
    fi
    echo "[cps] === ${_run}: ${_knob} ==="
    _t0="$(date +%s)"
    launch "${_exe}" "${_np}" "${_level}" 50 25 "${_knob}" "run=${_run}" \
        || { echo "[cps] FAIL: ${_run}" >&2; _rc=1; }
    echo "[cps] LAUNCH ${_run} wall=$(( $(date +%s) - _t0 ))s"
done 3<<< "${SWEEP}"
echo "[cps] total wall $(( $(date +%s) - _t_job0 ))s rc=${_rc}"
exit "${_rc}"
