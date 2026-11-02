#!/bin/bash
# flux: --job-name=beatnik_cpb
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
# canopy_profiling T3: the baseline cost model, and the standard measurement
# set (SMS) every O task re-runs. Beatnik_Probe_FmmCost, FMM-driven, at the
# probe's defaults (ncrit 8, order 3, mac_theta 0.3, m2l_op_count_cap 65536)
# unless a row names a knob. One named set per pdebug job; BEATNIK_CPB_SET
# picks it (tasks/canopy_profiling.md, T3 and "SMS definition").
#
# Usage (from the checkout; one job per set, submitted together):
#   BEATNIK_CPB_SET=<set> flux batch --job-name=beatnik_cpb_<set> \
#       scripts/tuolumne/canopy_profiling_baseline.flux
# BEATNIK_CPS_SWEEP replaces the set's rows; BEATNIK_CPS_BUDGET (default
# 3300 s) is the remaining-budget guard. A row the guard skips is printed as
# SKIPPED and makes the job exit 1.
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
    echo "[cpb] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[cpb] FAIL: BEATNIK_USE_PROD=1. canopy_profiling measures the DEV" >&2
    echo "  env: the production binary has no level-2 timers to read." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

BEATNIK_CPB_SET="${BEATNIK_CPB_SET:-}"

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
CANOPY_SRC="${CANOPY_SRC:-$(cd "${BEATNIK_REPO}/../canopy" 2>/dev/null && pwd)}"
beatnik_env_summary
echo "[cpb] spack env status:"
spack env status 2>&1 | sed 's/^/[cpb]   /'
echo "[cpb] compiler = $(spack find --format '{name}@{version} {compiler}' beatnik 2>/dev/null | head -1)"
echo "[cpb] beatnik commit = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[cpb] beatnik dirty  = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s)"
git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | sed 's/^/[cpb]   beatnik: /'
export BEATNIK_CANOPY_HASH="$(git -C "${CANOPY_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[cpb] canopy src    = ${CANOPY_SRC}"
echo "[cpb] canopy commit = ${BEATNIK_CANOPY_HASH}"
echo "[cpb] canopy porcelain ($(git -C "${CANOPY_SRC}" status --porcelain 2>/dev/null | wc -l) line(s)):"
git -C "${CANOPY_SRC}" status --porcelain 2>/dev/null | sed 's/^/[cpb]   canopy: /'
echo "[cpb] canopy spec   = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^%]*' | head -1)"
echo "[cpb] submit = BEATNIK_CPB_SET=${BEATNIK_CPB_SET} flux batch scripts/tuolumne/canopy_profiling_baseline.flux"
echo "[cpb] node = $(hostname)"


# The binding, copied EXACTLY from t6b_key_demand.flux:217-231.
launch() {
    local _exe="$1" _np="$2"; shift 2
    local _nodes=$(( (_np + 3) / 4 ))
    echo "[cpb] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
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

BEATNIK_CPS_BUDGET="${BEATNIK_CPS_BUDGET:-3300}"
# label level ranks steps sample_every space knob estimate_seconds
# (cheapest first; knob '-' means the probe's defaults, and a knob may be a
# comma-separated list of key=value pairs). Estimates are from
# T0's level-4 trend bins and T2's early rates, with margin.
case "${BEATNIK_CPB_SET}" in
    # ---- T3 baseline matrix -------------------------------------------------
    hip_np1_l4) SWEEP="
l4       4 1 2000 50 HIP    -        2400
" ;;
    hip_np1_l5l6) SWEEP="
l6       6 1 250  25 HIP    -        1100
l5       5 1 800  50 HIP    -        1400
" ;;
    hip_np4_l4) SWEEP="
l4       4 4 2000 50 HIP    -        1800
" ;;
    hip_np4_l5l6) SWEEP="
l6       6 4 250  25 HIP    -        700
l5       5 4 800  50 HIP    -        1200
" ;;
    hip_np4_ncrit) SWEEP="
l5nc64   5 4 800  50 HIP    ncrit=64 500
l5nc32   5 4 800  50 HIP    ncrit=32 700
" ;;
    serial_np1) SWEEP="
l4       4 1 500  25 SERIAL -        1500
" ;;
    serial_np4) SWEEP="
l4       4 4 500  25 SERIAL -        2600
" ;;
    # ---- The SMS: every O task re-runs these four sets ----------------------
    # (tasks/canopy_profiling.md, "SMS definition"). Estimates are T3's
    # measured launch walls plus margin.
    sms_hip_np1) SWEEP="
smsl5    5 1 300  25 HIP    -        500
smsl4    4 1 1200 50 HIP    -        1250
" ;;
    sms_hip_np4) SWEEP="
smsl5    5 4 300  25 HIP    -        400
smsl4    4 4 1200 50 HIP    -        900
" ;;
    sms_serial_np1) SWEEP="
smsl4    4 1 500  25 SERIAL -        1100
" ;;
    sms_serial_np4) SWEEP="
smsl4    4 4 500  25 SERIAL -        2300
" ;;
    # ---- Run-to-run spread (R2): the SMS's HIP np4 rows, twice -------------
    spread_hip_np4) SWEEP="
smsl5rep0 5 4 300  25 HIP   -        450
smsl5rep1 5 4 300  25 HIP   -        450
smsl4rep0 4 4 1200 50 HIP   -        900
smsl4rep1 4 4 1200 50 HIP   -        900
" ;;
    *)
        if [ -z "${BEATNIK_CPS_SWEEP:-}" ]; then
            echo "[cpb] FAIL: unknown BEATNIK_CPB_SET '${BEATNIK_CPB_SET}'" \
                 "and no BEATNIK_CPS_SWEEP" >&2
            exit 1
        fi
        SWEEP="" ;;
esac
[ -n "${BEATNIK_CPS_SWEEP:-}" ] && SWEEP="${BEATNIK_CPS_SWEEP}"
echo "[cpb] set = ${BEATNIK_CPB_SET:-<override>}, budget ${BEATNIK_CPS_BUDGET}s"
printf '%s\n' "${SWEEP}" | sed '/^$/d; s/^/[cpb]   /'
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _label="$1"; _level="$2"; _np="$3"; _steps="$4"; _every="$5"
    _space="$6"; _knob="$7"; _estimate="$8"
    _run="${_label}_L${_level}_${_space}_np${_np}"
    _remaining=$(( BEATNIK_CPS_BUDGET - ( $(date +%s) - _t_job0 ) ))
    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[cpb] === SKIPPED ${_run}: ${_remaining}s left, estimate ${_estimate}s ===" >&2
        echo "[cpb] === SKIPPED ${_run}: ${_remaining}s left, estimate ${_estimate}s ==="
        _rc=1
        continue
    fi
    _exe="$(beatnik_exe "Beatnik_Probe_FmmCost_MPI_${_space}")" || { _rc=1; continue; }
    _args=( "${_level}" "${_steps}" "${_every}" )
    # A knob column may list several key=value pairs, comma-separated.
    [ "${_knob}" != "-" ] && _args+=( ${_knob//,/ } )
    echo "[cpb] === ${_run}: steps ${_steps} sample_every ${_every} knob ${_knob} ==="
    _t0="$(date +%s)"
    launch "${_exe}" "${_np}" "${_args[@]}" "run=${_run}" \
        || { echo "[cpb] FAIL: ${_run}" >&2; _rc=1; }
    echo "[cpb] LAUNCH ${_run} wall=$(( $(date +%s) - _t0 ))s"
done 3<<< "${SWEEP}"
echo "[cpb] total wall $(( $(date +%s) - _t_job0 ))s rc=${_rc}"
exit "${_rc}"
