#!/bin/bash
# flux: --job-name=rb_amr_sweep
# flux: --nodes=1
# flux: --exclusive
# flux: -t 60m
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
# Which knobs keep the level-3 split-only AMR bubble alive longer?
#
# The 16-rank AMR run of rising_bubble_l3.flux (--remesh-h-max 0.03) stopped
# at step 2756, t = 1.4765, "nonfinite mesh-RHS state", with dt pinned at the
# --min-dt floor 2.5e-4 since ~step 1500, minEdge refined to 0.0011 (eps is
# 0.025) and minQ down to 0.09. Each variant below changes one of those, and
# runs on 4 ranks of one node to --t-end 4.0 (the frozen run reached 3.49)
# or until it stops, under a per-variant 13-minute cap. No checkpoints: the
# answer is in the progress lines and the stop line.
#
#   flux batch --output=/p/lustre5/stewartj/beatnik/rising_bubble_amr_sweep/{{name}}.{{jobid}}.log \
#       scripts/tuolumne/rising_bubble_amr_sweep.flux
#
# Every variant prints `[sweep] <name> rc=<rc> wall=<s>`, preceded by the
# solver's own stop line if it stopped; a variant killed by its cap prints
# rc=143 or similar and no stop line.
############################################################################

set -u

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
    echo "[sweep] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi

export BEATNIK_USE_PROD=1

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary
EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[sweep] exe = ${EXE}"

BASE=( --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
       --viscosity-mode laplace-beltrami
       --br-approximation direct
       --source-quadrature vertex
       --adaptive-dt
       --icosphere-subdivisions 3
       --dynamic-remesh
       --remesh-h-max 0.03
       --remesh-collapse-factor 0
       --remesh-smooth-iters 0
       --remesh-flip-min-gain 1e12
       --no-isotropic-cleanup
       --steps 100000000
       --t-end 4.0
       --progress-time-interval 0.1 )

VARIANTS=(
    "A_hmin005_mindt5e-5|--remesh-h-min 0.005 --min-dt 5e-5"
    "B_A_redist10|--remesh-h-min 0.005 --min-dt 5e-5 --redistribute-every 10"
    "C_B_hmin01|--remesh-h-min 0.01 --min-dt 5e-5 --redistribute-every 10"
    "D_B_sag008|--remesh-h-min 0.005 --min-dt 5e-5 --redistribute-every 10 --remesh-sagitta-tolerance 0.008"
)

for _v in "${VARIANTS[@]}"; do
    _name="${_v%%|*}"
    read -r -a _extra <<< "${_v#*|}"
    echo "[sweep] === ${_name}: ${_extra[*]}"
    _t0=$(date +%s)
    flux run -t 13m \
        --ntasks=4 \
        --nodes=1 \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        "${EXE}" "${BASE[@]}" "${_extra[@]}"
    _rc=$?
    echo "[sweep] ${_name} rc=${_rc} wall=$(( $(date +%s) - _t0 ))"
done
