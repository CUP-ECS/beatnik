#!/bin/bash
# flux: --job-name=large_rollup_P1c
# flux: --nodes=2
# flux: --exclusive
# flux: -t 45m
# flux: --output=/p/lustre5/stewartj/beatnik/large_rollup/P1c/{{name}}.{{jobid}}.log
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
# large_rollup P1c (`tasks/large_rollup.md` P1c): stability and cost at an
# edge-tied eps (8e-3, about four level-7 edges) on 4 and 8 ranks, plus the
# level-6 pilot candidate.
#
# Four 60-step launches, dev env, each under its own 8-minute cap, with the
# launch line and per-launch logging of `large_rollup_probe_b.flux`. Every
# launch carries D1's step-size rule (--dt-edge-power 2) and edit caps
# (--remesh-max-splits 1000 --remesh-max-collapses 400):
#   L7_np4_eps0.025   baseline: P1b's L7_np4_dt8e-4_pow2 with the raised caps
#                     and --min-dt 5e-5
#   L7_np4_eps8e-3    the baseline at eps 8e-3
#   L7_np8_eps8e-3    the same on 8 ranks over 2 nodes
#   L6_np4_eps1.6e-2  pilot candidate: level-7 lengths doubled, dt x4
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/large_rollup_probe_c.flux
# Output: /p/lustre5/stewartj/beatnik/large_rollup/P1c/<tag>/launch.log
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

OUT=/p/lustre5/stewartj/beatnik/large_rollup/P1c
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
echo "[large_rollup] submit    = \"\$TASKLOOP_SUBMIT\" --purpose \"large_rollup P1c: stability and cost at edge-tied eps 8e-3, np4/np8\" scripts/tuolumne/large_rollup_probe_c.flux"
echo "[large_rollup] host      = $(hostname)  jobid = ${FLUX_JOB_ID:-unknown}"

BASE=( --A 0.3 --g 1.0 --mu 0.002
       --viscosity-mode laplace-beltrami
       --br-approximation direct
       --source-quadrature vertex
       --adaptive-dt
       --dynamic-remesh
       --isotropic-cleanup
       --steps 60
       --progress-time-interval 0
       --dt-edge-power 2
       --remesh-max-splits 1000
       --remesh-max-collapses 400 )

# tag|ranks|nodes|extra arguments
LAUNCHES=(
    "L7_np4_eps0.025|4|1|--icosphere-subdivisions 7 --eps 0.025 --remesh-h-max 2.05e-3 --remesh-h-min 1.5e-3 --remesh-sagitta-tolerance 0.004 --dt 8e-4 --min-dt 5e-5"
    "L7_np4_eps8e-3|4|1|--icosphere-subdivisions 7 --eps 8e-3 --remesh-h-max 2.05e-3 --remesh-h-min 1.5e-3 --remesh-sagitta-tolerance 0.004 --dt 8e-4 --min-dt 5e-5"
    "L7_np8_eps8e-3|8|2|--icosphere-subdivisions 7 --eps 8e-3 --remesh-h-max 2.05e-3 --remesh-h-min 1.5e-3 --remesh-sagitta-tolerance 0.004 --dt 8e-4 --min-dt 5e-5"
    "L6_np4_eps1.6e-2|4|1|--icosphere-subdivisions 6 --eps 1.6e-2 --remesh-h-max 4.1e-3 --remesh-h-min 3.0e-3 --remesh-sagitta-tolerance 0.016 --dt 3.2e-3 --min-dt 2e-4"
)

for _l in "${LAUNCHES[@]}"; do
    _tag="${_l%%|*}"
    _rest="${_l#*|}"
    _np="${_rest%%|*}"
    _rest="${_rest#*|}"
    _nodes="${_rest%%|*}"
    read -r -a _extra <<< "${_rest#*|}"
    _dir="${OUT}/${_tag}"
    mkdir -p "${_dir}"
    echo "[large_rollup] === ${_tag}: np=${_np} nodes=${_nodes} ${_extra[*]}"
    _t0=$(date +%s.%N)
    echo "${_t0} [large_rollup] ${_tag} start=${_t0} np=${_np} nodes=${_nodes} args: ${BASE[*]} ${_extra[*]}" > "${_dir}/launch.log"
    flux run -t 8m \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
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
