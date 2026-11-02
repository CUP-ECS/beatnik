#!/bin/bash
# flux: --job-name=beatnik_cprof_ref_levels
# flux: --nodes=1
# flux: --exclusive
# flux: -t 50m
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
# canopy_profiling, the L5/L6 accuracy question, part 1: does the Python
# reference treecode lose the same accuracy at levels 5 and 6?
#
#   1. Beatnik_Test_Milestone0Run, HIP np1, DIRECT-driven, 50 steps, a
#      checkpoint every 25, at levels 4, 5 and 6 (claim A's trajectory kind).
#   2. tests/regression_tests/reference_treecode_levels.py on the step-25 and
#      step-50 checkpoints: the reference's direct sum against its treecode
#      (theta 0.3, order 2) at ncrit 64 (its default) and 8 (Beatnik's
#      milestone value). Level 4 first, as the calibration against T9r.
#
# Usage:  flux batch scripts/tuolumne/cprof_ref_levels.flux
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
echo "[cps] submit = flux batch scripts/tuolumne/cprof_ref_levels.flux"


# The binding, copied EXACTLY from t6b_key_demand.flux:217-231.
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

_ref="${ZMODEL3D_REPO:-${HOME}/research-bridges/zmodel-steve/zmodel3d-amr}"
_python=/usr/tce/bin/python3
echo "[cps] reference repo   = ${_ref}"
echo "[cps] reference commit = $(git -C "${_ref}" rev-parse HEAD 2>/dev/null || echo unknown)"
_ref_dirty="$(git -C "${_ref}" status --porcelain -- zmodel3d 2>&1)"
echo "[cps] reference zmodel3d/ status = '${_ref_dirty}'"
echo "[cps] python = ${_python} ($(${_python} --version 2>&1))"
if [ -n "${_ref_dirty}" ]; then
    echo "[cps] FAIL: the reference's zmodel3d/ package is not clean." >&2
    exit 1
fi

_exe="$(beatnik_exe Beatnik_Test_Milestone0Run_MPI_HIP)" || exit 1
export BEATNIK_TEST_SCRATCH="/p/lustre5/stewartj/beatnik/canopy_profiling/ref_levels/job${_jobid}"
rm -rf "${BEATNIK_TEST_SCRATCH}"; mkdir -p "${BEATNIK_TEST_SCRATCH}" || exit 1
echo "[cps] scratch = ${BEATNIK_TEST_SCRATCH}"

for _level in 4 5 6; do
    echo "[cps] === checkpoints: level ${_level}, direct, 50 steps every 25 ==="
    _t0="$(date +%s)"
    launch "${_exe}" 1 "${_level}" 50 25 direct || { echo "[cps] FAIL: L${_level} run" >&2; _rc=1; }
    echo "[cps] CKPT_WALL L${_level} $(( $(date +%s) - _t0 ))s"
done
find "${BEATNIK_TEST_SCRATCH}" -name '*.h5' | sort | sed 's/^/[cps] h5: /'

for _level in 4 5 6; do
    _files="$(find "${BEATNIK_TEST_SCRATCH}/sub${_level}_"* -name '*_step00000[25]*.h5' 2>/dev/null | sort)"
    _n="$(printf '%s\n' "${_files}" | grep -c .)"
    if [ "${_n}" -ne 2 ]; then
        echo "[cps] FAIL: level ${_level}: ${_n} step-25/50 checkpoints, expected 2" >&2
        _rc=1
        continue
    fi
    echo "[cps] === reference at level ${_level} ==="
    _t0="$(date +%s)"
    # shellcheck disable=SC2086
    PYTHONPATH="${_ref}" "${_python}" \
        "${BEATNIK_REPO}/tests/regression_tests/reference_treecode_levels.py" \
        ${_files} --ncrit 64 8 || { echo "[cps] FAIL: reference L${_level}" >&2; _rc=1; }
    echo "[cps] REF_WALL L${_level} $(( $(date +%s) - _t0 ))s"
done

echo "[cps] total wall $(( $(date +%s) - _t_job0 ))s rc=${_rc}"
exit "${_rc}"
