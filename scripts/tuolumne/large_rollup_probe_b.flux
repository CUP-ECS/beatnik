#!/bin/bash
# flux: --job-name=large_rollup_P1b
# flux: --nodes=1
# flux: --exclusive
# flux: -t 30m
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
# large_rollup P1b (`tasks/large_rollup.md` P1, follow-up probe): does an
# explicit viscous step limit, dt <~ 0.5 h^2 / mu, explain the level-7
# blow-up of the P1 probe (`large_rollup_probe.flux`)?
#
# Four 60-step launches at icosphere level 7 on 4 ranks, dev env, each under
# its own 6-minute cap, with the same base arguments, launch line and
# per-launch logging as `large_rollup_probe.flux`:
#   L7_np4_dt4e-4              --dt 4e-4, h_max 2.05e-3 (dt mu/h^2 ~ 0.2-0.3)
#   L7_np4_dt8e-4_pow2         --dt 8e-4 --dt-edge-power 2 (dt mu/h^2 ~ 0.41,
#                              held constant as edges shrink)
#   L7_np4_hmax_default_dt4e-4 the control at --dt 4e-4, default h_max
#   L7_np4_dt1e-3_mu0          the P1 dt1e-3 arm with --mu 0: if viscosity
#                              is the cause, this one stays finite
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/large_rollup_probe_b.flux
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
echo "[large_rollup] submit    = \"\$TASKLOOP_SUBMIT\" --purpose \"large_rollup P1b: level-7 viscous dt limit and control direction\" scripts/tuolumne/large_rollup_probe_b.flux"
echo "[large_rollup] host      = $(hostname)  jobid = ${FLUX_JOB_ID:-unknown}"

BASE=( --A 0.3 --g 1.0
       --viscosity-mode laplace-beltrami
       --br-approximation direct
       --source-quadrature vertex
       --adaptive-dt
       --dynamic-remesh
       --isotropic-cleanup
       --eps 0.025
       --steps 60
       --progress-time-interval 0
       --icosphere-subdivisions 7 )

# tag|ranks|extra arguments
LAUNCHES=(
    "L7_np4_dt4e-4|4|--mu 0.002 --remesh-h-max 2.05e-3 --dt 4e-4"
    "L7_np4_dt8e-4_pow2|4|--mu 0.002 --remesh-h-max 2.05e-3 --dt 8e-4 --dt-edge-power 2"
    "L7_np4_hmax_default_dt4e-4|4|--mu 0.002 --dt 4e-4"
    "L7_np4_dt1e-3_mu0|4|--mu 0 --remesh-h-max 2.05e-3 --dt 0.001"
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
