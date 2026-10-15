#!/bin/bash
# flux: --job-name=rising_bubble_l3
# flux: --nodes=4
# flux: --exclusive
# flux: -t 240m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
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
# Level-3 rising bubble (examples/02_adaptive_mesh_bubble) run to collapse,
# for presentation output. One script, two modes, chosen at submit time:
#
#   RB_MODE=no_amr  frozen mesh: --no-dynamic-remesh --refine-every 0
#   RB_MODE=amr     split-only dynamic remeshing: --dynamic-remesh with the
#                   T4d passes (collapse, flips, smoothing, isotropic cleanup)
#                   switched off, since they still throw
#
#   RB_MODE=no_amr flux batch --job-name=rb_l3_no_amr \
#       --output=/p/lustre5/stewartj/beatnik/rising_bubble_no_amr/{{name}}.{{jobid}}.log \
#       scripts/tuolumne/rising_bubble_l3.flux
#
# Optional overrides, all read from the submitting environment:
#
#   RB_OUT_DIR  checkpoint directory (default
#               /p/lustre5/stewartj/beatnik/rising_bubble_<RB_MODE>); with
#               more than one mode, each mode writes to <RB_OUT_DIR>/<mode>
#   RB_STEPS    step cap (default 100000000, i.e. run to collapse)
#   RB_LEVEL    --icosphere-subdivisions (default 3)
#   RB_USE_PROD 1 = production env (default), 0 = dev env, e.g. to try a fix
#               before it is installed in production
#   RB_NODES    nodes to launch on; ranks = 4 x nodes (default 4). Must match
#               the allocation, so pass the same -N to flux batch.
#   RB_EXTRA_ARGS  extra solver flags appended last, word-split, e.g.
#               "--dt-edge-power 2 --min-dt 2.5e-5" to keep dt under the
#               h^2/mu viscous limit at higher levels
#
# RB_MODE may name both modes, run one after the other in one allocation. The
# pre-flight smoke test is exactly that, short and on one pdebug node:
#
#   RB_MODE="no_amr amr" RB_STEPS=200 RB_NODES=1 \
#   RB_OUT_DIR=/p/lustre5/stewartj/beatnik/rising_bubble_smoke \
#   flux batch -N1 -q pdebug -t 10m --job-name=rb_l3_smoke \
#       scripts/tuolumne/rising_bubble_l3.flux
#
# "Until collapse" is --steps set far past any reachable count: the solver
# stops on its own at the first nonfinite state and prints
# `stopping at step=... t=...: nonfinite ...`. Otherwise the run ends at the
# 240m walltime. Checkpoints are written every 0.01 time units, so a run killed
# at the wall still leaves its trajectory on disk.
#
# Physics is the milestone-0 configuration (README configuration (a) of the
# Python reference) with the direct Birkhoff-Rott sum.
#
# Targets the PRODUCTION env by default, so a dev rebuild cannot break a
# queued job.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs
# none today.

# `flux batch` spools the script to /var/tmp, so BASH_SOURCE cannot find the
# checkout; walk up from PWD, which flux preserves.
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
    echo "[rising_bubble] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi

export BEATNIK_USE_PROD="${RB_USE_PROD:-1}"

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

RB_MODE="${RB_MODE:-}"
RB_STEPS="${RB_STEPS:-100000000}"
RB_LEVEL="${RB_LEVEL:-3}"
RB_NODES="${RB_NODES:-4}"
RB_NTASKS=$(( RB_NODES * 4 ))
read -r -a _extra <<< "${RB_EXTRA_ARGS:-}"
read -r -a _modes <<< "${RB_MODE}"
if [ "${#_modes[@]}" -eq 0 ]; then
    echo "[rising_bubble] FAIL: set RB_MODE=no_amr, RB_MODE=amr, or both." >&2
    exit 1
fi
for _m in "${_modes[@]}"; do
    case "${_m}" in
        no_amr|amr) ;;
        *)
            echo "[rising_bubble] FAIL: unknown RB_MODE '${_m}'." >&2
            exit 1
            ;;
    esac
done

beatnik_env_summary
_src_env="${BEATNIK_ACTIVE_SPACK_ENV}"
echo "[rising_bubble] modes   = ${RB_MODE}"
echo "[rising_bubble] level   = ${RB_LEVEL}"
echo "[rising_bubble] extra   = ${RB_EXTRA_ARGS:-<none>}"
echo "[rising_bubble] ranks   = ${RB_NTASKS} on ${RB_NODES} node(s)"
echo "[rising_bubble] job     = $(flux getattr jobid 2>/dev/null || echo "${FLUX_JOB_ID:-<none>}")"
for _r in beatnik canopy tessera; do
    echo "[rising_bubble] src ${_r} = $(git -C "${_src_env}/${_r}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" \
         "$(git -C "${_src_env}/${_r}" rev-parse HEAD 2>/dev/null || echo unknown)"
done

EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[rising_bubble] exe     = ${EXE}"

_rc_all=0
for _m in "${_modes[@]}"; do
    case "${_m}" in
        no_amr)
            AMR_ARGS=( --no-dynamic-remesh --refine-every 0 )
            ;;
        amr)
            # h-max 0.03 (default 0.05) so splitting starts near t=0: at
            # 0.05 the split threshold is 1.35 x 0.05 = 0.0675, about twice
            # L3's initial edge length, and nothing splits until roll-up.
            # Sweep variant D (job f3e92SU62uew, longest-lived: t = 2.53),
            # with redistribution every 5 steps rather than D's 10: a
            # floor on h and dt, tangential redistribution, and a looser
            # sagitta tolerance so curvature drives fewer splits.
            AMR_ARGS=( --dynamic-remesh
                       --remesh-h-max 0.03
                       --remesh-h-min 0.005
                       --min-dt 5e-5
                       --redistribute-every 5
                       --remesh-sagitta-tolerance 0.008
                       --remesh-collapse-factor 0
                       --remesh-smooth-iters 0
                       --remesh-flip-min-gain 1e12
                       --no-isotropic-cleanup )
            ;;
    esac
    if [ -n "${RB_OUT_DIR:-}" ] && [ "${#_modes[@]}" -gt 1 ]; then
        _out="${RB_OUT_DIR}/${_m}"
    else
        _out="${RB_OUT_DIR:-/p/lustre5/stewartj/beatnik/rising_bubble_${_m}}"
    fi
    mkdir -p "${_out}" || exit 1

    ARGS=( --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
           --viscosity-mode laplace-beltrami
           --br-approximation direct
           --source-quadrature vertex
           --adaptive-dt
           --icosphere-subdivisions "${RB_LEVEL}"
           "${AMR_ARGS[@]}"
           --steps "${RB_STEPS}"
           --progress-time-interval 0.05
           --checkpoint-dir "${_out}"
           --checkpoint-every-time 0.01
           ${_extra[@]+"${_extra[@]}"} )
    echo "[rising_bubble] === mode ${_m} -> ${_out}"
    echo "[rising_bubble] args    = ${ARGS[*]}"

    # Tuolumne runs 4 ranks per node; --nodes must match the allocation.
    _t0=$(date +%s)
    flux run \
        --ntasks="${RB_NTASKS}" \
        --nodes="${RB_NODES}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${EXE}" "${ARGS[@]}"
    _rc=$?
    echo "[rising_bubble] mode ${_m} exit rc=${_rc} after $(( $(date +%s) - _t0 )) s"
    [ "${_rc}" -ne 0 ] && _rc_all=${_rc}
done
exit ${_rc_all}
