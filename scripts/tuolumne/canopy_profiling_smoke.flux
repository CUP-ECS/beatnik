#!/bin/bash
# flux: --job-name=beatnik_cprof_smoke
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
# canopy_profiling T2's SMOKE RUN (`tasks/canopy_profiling.md`):
# Beatnik_Probe_FmmCost on HIP, one node, pdebug.
#
#   1. `bogus=1` -- the failure direction: the probe must exit non-zero
#      naming `bogus`.
#   2. level 4, np1, 60 steps, sample_every 20, canopy_report=0.
#   3. the same with canopy_report=1, for the reporting-overhead ratio; its
#      log must carry [Canopy Diagnostics] lines, row 2's must carry none.
#   4. level 5 and level 6 sizing, 10 steps, sample_every 5, at np4 then np1.
#      Their s/step is what T3's walltimes are sized from. R8: level 6 at np1
#      may not fit one MI300A; a launch that dies or is skipped is a finding.
#
# Shape copied from t6b_key_demand.flux (preamble, resolver, provenance,
# binding) and t5_divergence.flux (budget guard, row override, PROD refusal).
# Rows are ordered cheapest-first, so a budget overrun skips the expensive
# sizing rows loudly rather than killing a launch mid-run.
#
# DEV ENV ONLY: the probe reads the level-2 timers, which only the dev env's
# `canopy +profiling profiling_level=2` / `beatnik profiling_level=2` compile.
#
# Usage:  flux batch --flags=waitable scripts/tuolumne/canopy_profiling_smoke.flux
#
# Then read beatnik_cprof_smoke.<jobid>.log in the submitting directory, and
# reduce it with
#   python3 tests/regression_tests/fmm_profile_reduce.py --format probe <log>
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
echo "[cps] submit = flux batch --flags=waitable scripts/tuolumne/canopy_profiling_smoke.flux"

_target="Beatnik_Probe_FmmCost_MPI_HIP"
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[cps] exe = ${_exe}"

# Per-launch output copies, so the diagnostic-line count is per launch. Lustre,
# per job: nothing here is read back across jobs.
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
BEATNIK_CPS_OUT_ROOT="${BEATNIK_CPS_OUT_ROOT:-/p/lustre5/stewartj/beatnik/canopy_profiling/smoke}"
_outdir="${BEATNIK_CPS_OUT_ROOT}/job${_jobid}"
mkdir -p "${_outdir}" || { echo "[cps] FAIL: cannot create ${_outdir}" >&2; exit 1; }
echo "[cps] per-launch output = ${_outdir}"

# Seconds of wall the rows may spend. -t is 58m (3480 s); 3300 leaves three
# minutes for startup and the report.
BEATNIK_CPS_BUDGET="${BEATNIK_CPS_BUDGET:-3300}"

# One row per launch:
#   label level ranks steps sample_every canopy_report estimate_seconds
#
# Estimates are padded guesses from T9b's level-4 HIP wall (about 0.35 s/step
# early in the trajectory at np1) scaled by N for levels 5 and 6, plus process
# start and the [acc] samples' standalone direct sum. They only gate the
# budget; the measured s/step is what this job is for.
SWEEP="
l4rep0   4 1 60 20 0 240
l4rep1   4 1 60 20 1 300
l5np4    5 4 10  5 0 300
l5np1    5 1 10  5 0 420
l6np4    6 4 10  5 0 700
l6np1    6 1 10  5 0 1100
"
if [ -n "${BEATNIK_CPS_SWEEP:-}" ]; then
    SWEEP="${BEATNIK_CPS_SWEEP}"
    echo "[cps] *** SWEEP OVERRIDDEN -- this job runs a SUBSET of T2's smoke" \
         "matrix; say which in the log entry. ***"
fi
echo "[cps] budget=${BEATNIK_CPS_BUDGET}s sweep:"
printf '%s\n' "${SWEEP}" | sed '/^$/d; s/^/[cps]   /'

# The binding, copied EXACTLY from t6b_key_demand.flux:217-231. A wrong
# binding does not fail; it oversubscribes one device and returns a plausible
# number.
launch() {
    local _np="$1"; shift
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

_t_job0="$(date +%s)"
_rc=0

##--------------------------------------------------------------------------##
## 1. Failure direction: an unknown key must be refused by name.
##--------------------------------------------------------------------------##
echo "[cps] === bogus: expect a non-zero exit naming 'bogus' ==="
_f="${_outdir}/bogus.out"
launch 1 4 2 1 bogus=1 > "${_f}" 2>&1
_obs=$?
sed 's/^/[cps:bogus] /' "${_f}"
if [ "${_obs}" -ne 0 ] && grep -q "unknown key 'bogus'" "${_f}"; then
    echo "[cps] BOGUS_CHECK PASS rc=${_obs} (refused, naming bogus)"
else
    echo "[cps] BOGUS_CHECK FAIL rc=${_obs}: an unknown key was not refused by name" >&2
    _rc=1
fi

##--------------------------------------------------------------------------##
## 2-4. The measured rows.
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _label="$1"; _level="$2"; _np="$3"; _steps="$4"; _every="$5"
    _report="$6"; _estimate="$7"

    _run="${_label}_L${_level}_HIP_np${_np}_rep${_report}"
    _remaining=$(( BEATNIK_CPS_BUDGET - ( $(date +%s) - _t_job0 ) ))
    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[cps] === SKIPPED ${_run}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. A finding for the T2 log entry, not a" \
             "reason to run fewer steps. ===" >&2
        _rc=1
        continue
    fi

    echo "[cps] === ${_run}: level ${_level}, np ${_np}, ${_steps} steps," \
         "sample_every ${_every}, canopy_report=${_report} ==="
    _f="${_outdir}/${_run}.out"
    _t0="$(date +%s)"
    launch "${_np}" "${_level}" "${_steps}" "${_every}" \
        "canopy_report=${_report}" "run=${_run}" > "${_f}" 2>&1
    _obs=$?
    _wall=$(( $(date +%s) - _t0 ))
    cat "${_f}"
    _ndiag="$(grep -c '\[Canopy Diagnostics\]' "${_f}")"
    _ndiag2="$(grep -c '\[Canopy diag\]' "${_f}")"
    echo "[cps] LAUNCH ${_run} rc=${_obs} wall=${_wall}s estimate=${_estimate}s" \
         "canopy_diagnostics_lines=${_ndiag} canopy_diag_lines=${_ndiag2}" \
         "s_per_step_incl_startup=$(awk -v w="${_wall}" -v s="${_steps}" 'BEGIN{printf "%.4f", w/s}')"
    if [ "${_obs}" -ne 0 ]; then
        echo "[cps] FAIL: ${_run} exited ${_obs}" >&2
        _rc=1
    fi
    if [ "${_report}" = "0" ] && [ $(( _ndiag + _ndiag2 )) -ne 0 ]; then
        echo "[cps] FAIL: ${_run} has canopy_report=0 but printed Canopy" \
             "diagnostic lines" >&2
        _rc=1
    fi
    if [ "${_report}" = "1" ] && [ "${_ndiag}" -eq 0 ]; then
        echo "[cps] FAIL: ${_run} has canopy_report=1 but printed no" \
             "[Canopy Diagnostics] line" >&2
        _rc=1
    fi
done 3<<< "${SWEEP}"

echo "[cps] total wall $(( $(date +%s) - _t_job0 ))s"
if [ "${_rc}" -eq 0 ]; then
    echo "[cps] SUMMARY: PASS -- every row ran. Reduce the [cost] / [acc] /"
    echo "[cps] [cost-total] rows with fmm_profile_reduce.py --format probe."
else
    echo "[cps] SUMMARY: FAIL -- read the FAIL / SKIPPED lines above." >&2
fi
exit "${_rc}"
