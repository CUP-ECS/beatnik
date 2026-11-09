#!/bin/bash
# flux: --job-name=large_rollup_P1
# flux: --nodes=1
# flux: --exclusive
# flux: -t 45m
# flux: --output=/p/lustre5/stewartj/beatnik/large_rollup/P1/{{name}}.{{jobid}}.log
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
# large_rollup P1 (`tasks/large_rollup.md`): cost, memory and early
# stability of the configuration-(a) bubble at icosphere levels 6 and 7.
#
# Six 60-step launches on one node, dev env, each under its own 6-minute cap.
# Each rank's executable is wrapped in `/usr/bin/time -f "rss_kb=%M"`, so
# every rank prints its peak RSS on stderr. Every stdout/stderr line of a
# launch is prefixed with its `date +%s.%N` arrival time and written to
# <out>/<tag>/launch.log, which also holds the launch's start time
# (`[large_rollup] <tag> start=<epoch>`) so the setup wall (start to the
# `step=    1` line) can be computed afterwards.
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/large_rollup_probe.flux
# Output: /p/lustre5/stewartj/beatnik/large_rollup/P1/<tag>/launch.log
#
# Every launch prints `[large_rollup] <tag> rc=<rc> wall=<s>`; a launch
# killed by its cap shows a non-zero rc and no solver stop line.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

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
    echo "[large_rollup] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

OUT=/p/lustre5/stewartj/beatnik/large_rollup/P1
mkdir -p "${OUT}" || exit 1

# --- provenance ------------------------------------------------------------
beatnik_env_summary
EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[large_rollup] exe       = ${EXE}"
echo "[large_rollup] spack     ="
spack find -lv beatnik 2>&1 | sed 's/^/[large_rollup]   /'
echo "[large_rollup] compiler  = $(hipcc --version 2>/dev/null | head -1 || echo unknown)"
echo "[large_rollup] HEAD      = $(git -C "${BEATNIK_REPO}" rev-parse HEAD)"
echo "[large_rollup] porcelain = $(git -C "${BEATNIK_REPO}" status --porcelain | wc -l)"
echo "[large_rollup] submit    = \"\$TASKLOOP_SUBMIT\" --purpose \"large_rollup P1: cost, memory and early stability at levels 6 and 7\" scripts/tuolumne/large_rollup_probe.flux"
echo "[large_rollup] host      = $(hostname)  jobid = ${FLUX_JOB_ID:-unknown}"

BASE=( --A 0.3 --g 1.0 --mu 0.002
       --viscosity-mode laplace-beltrami
       --br-approximation direct
       --source-quadrature vertex
       --adaptive-dt
       --dynamic-remesh
       --isotropic-cleanup
       --eps 0.025
       --steps 60
       --progress-time-interval 0 )

# tag|ranks|extra arguments
LAUNCHES=(
    "L6_np1|1|--icosphere-subdivisions 6 --remesh-h-max 4.1e-3 --dt 0.003"
    "L6_np4|4|--icosphere-subdivisions 6 --remesh-h-max 4.1e-3 --dt 0.003"
    "L7_np1|1|--icosphere-subdivisions 7 --remesh-h-max 2.05e-3 --dt 0.003"
    "L7_np4|4|--icosphere-subdivisions 7 --remesh-h-max 2.05e-3 --dt 0.003"
    "L7_np4_dt1e-3|4|--icosphere-subdivisions 7 --remesh-h-max 2.05e-3 --dt 0.001"
    "L7_np4_hmax_default|4|--icosphere-subdivisions 7 --dt 0.003"
)

for _l in "${LAUNCHES[@]}"; do
    _tag="${_l%%|*}"
    _rest="${_l#*|}"
    _np="${_rest%%|*}"
    read -r -a _extra <<< "${_rest#*|}"
    _dir="${OUT}/${_tag}"
    mkdir -p "${_dir}"
    echo "[large_rollup] === ${_tag}: np=${_np} ${_extra[*]}"
    _t0=$(date +%s.%N)
    echo "${_t0} [large_rollup] ${_tag} start=${_t0} np=${_np} args: ${BASE[*]} ${_extra[*]}" > "${_dir}/launch.log"
    flux run -t 6m \
        --ntasks="${_np}" \
        --nodes=1 \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        /usr/bin/time -f "rss_kb=%M" \
        "${EXE}" "${BASE[@]}" "${_extra[@]}" \
        --checkpoint-dir "${_dir}" --checkpoint-every-steps 60 2>&1 \
        | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done \
        >> "${_dir}/launch.log"
    _rc=${PIPESTATUS[0]}
    _wall=$(awk -v a="${_t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
    # Echo the launch's progress, stop and rss lines into the job log too.
    grep -E 'step=|nonfinite|rss_kb=|[Ss]top|rror' "${_dir}/launch.log" | tail -n 40
    echo "[large_rollup] ${_tag} rc=${_rc} wall=${_wall}"
done
