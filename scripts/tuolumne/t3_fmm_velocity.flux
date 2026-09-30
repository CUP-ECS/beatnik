#!/bin/bash
# flux: --job-name=beatnik_t3_fmm
# flux: --nodes=1
# flux: --exclusive
# flux: -t 10m
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
# T3 exit criterion (tasks/canopy/add-canopy.md): a two-step run of
# examples/02_adaptive_mesh_bubble at the milestone-0 configuration, at 1 and
# 4 ranks, under a selectable --br-approximation.
#
# The FMM half asks only "does it complete without throwing" -- there is NO
# accuracy claim here, and none may be read out of a passing run. At
# --icosphere-subdivisions 3 (642 vertices) the far field is not even live at
# the default ncrit, so an FMM run that agrees with the direct one agrees
# because it IS a direct sum, not because the expansion is right. T4 is where
# anything is first asserted.
#
# The direct half is a REGRESSION check and is why this script is
# mode-parameterized rather than FMM-only: the same script run before and
# after the source change emits byte-identical command lines, so h5diff over
# the two checkpoint trees sees only what the change did.
#
# Usage:  flux batch scripts/tuolumne/t3_fmm_velocity.flux LABEL [MODE...]
#
#   LABEL    subdirectory of the scratch root holding this submission's
#            checkpoints, e.g. `baseline` or `after`.
#   MODE...  --br-approximation values to run; default `direct fmm`.
#
#   flux batch scripts/tuolumne/t3_fmm_velocity.flux baseline direct
#   flux batch scripts/tuolumne/t3_fmm_velocity.flux after direct fmm
#
# Then read beatnik_t3_fmm.<jobid>.log in the submitting directory.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root. `flux batch` copies the script into a per-job spool
# directory before running it, so BASH_SOURCE points at /var/tmp/... and cannot
# find the checkout. It does preserve the submitting working directory, so walk
# up from PWD as the fallback; BASH_SOURCE still covers a direct `bash` run.
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
    echo "[t3] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance -- everything needed to tell two submissions apart
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[t3] spack env status:"
spack env status 2>&1 | sed 's/^/[t3]   /'
echo "[t3] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t3] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"

EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[t3] exe     = ${EXE}"
echo "[t3] exe mtime = $(date -r "${EXE}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"

_label="${1:-run}"
shift || true
if [ "$#" -gt 0 ]; then
    _modes=( "$@" )
else
    _modes=( direct fmm )
fi
echo "[t3] label   = ${_label}"
echo "[t3] modes   = ${_modes[*]}"

# MPI-IO writes these, so the root must be a PARALLEL filesystem: a node-local
# path fails every launch that spans more than one node.
_root="/p/lustre5/stewartj/beatnik/fmm/debug/${_label}"
echo "[t3] ckpt root = ${_root}"

##--------------------------------------------------------------------------##
## The milestone-0 configuration, at level 3 for 2 steps
##--------------------------------------------------------------------------##
# Every flag below is the CLI equivalent of an assignment in
# tests/regression_tests/Beatnik_Test_Milestone0Frozen.cpp::makeParams, whose
# comments name them line by line. Two deliberate departures from that
# function, both required by what T3 is checking rather than by preference:
#   --icosphere-subdivisions 3 / --steps 2   (makeParams: kSubdivisions, 2000)
#   --checkpoint-every-steps 1               (makeParams: 25)
# `--source-quadrature vertex` is MANDATORY, not decorative: ZModelParams
# defaults to Face, whose generate() throws, and the FMM adapter rejects
# anything but Vertex regardless. `--bernoulli-scalar-mode normal-speed` keeps
# computeSurfaceRieszScalar -- still a throwing stub, and T7's -- out of the
# call path.
_common_args=(
    --state-model potential --mesh-kind icosphere --icosphere-subdivisions 3
    --radius 0.25 --center-z 0.25 --initial-shape sphere
    --initial-potential-strength 0 --polar-amp 0
    --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 --sigma 0
    --forcing-sign 1 --br-sign 1 --kernel-blob-mode length
    --viscosity-mode laplace-beltrami --velocity-mode full
    --bernoulli-scalar-mode normal-speed --source-quadrature vertex
    --steps 2 --dt 0.003 --adaptive-dt --min-dt 2.5e-4 --dt-edge-power 1.0
    --max-sheet-dt-product 0 --dt-switch-time -1
    --no-dynamic-remesh --refine-every 0 --isotropic-cleanup
    --checkpoint-every-steps 1
    --no-video
)

_rc=0
_launched=0

for _mode in "${_modes[@]}"; do
    for _np in 1 4; do
        _tag="${_mode}_np${_np}"

        # One output directory PER RUN, removed and recreated immediately
        # before that run -- so a stale checkpoint from an earlier submission
        # cannot be read back as this run's output and h5diff'd as a pass.
        _ckpt="${_root}/${_tag}"
        rm -rf "${_ckpt}"
        mkdir -p "${_ckpt}" || {
            echo "[t3] FAIL: cannot create ${_ckpt}" >&2
            _rc=1
            continue
        }

        # Tuolumne packs 4 ranks per node; round the node count up. The binding
        # is copied EXACTLY from milestone0_divergence.flux:255-272 and must not
        # be simplified: a wrong binding does not fail, it oversubscribes one
        # device and returns a plausible result.
        _nodes=$(( (_np + 3) / 4 ))

        _args=( "${_common_args[@]}" --checkpoint-dir "${_ckpt}"
                --br-approximation "${_mode}" )

        echo "[t3] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s) ==="
        echo "[t3] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
             "--exclusive --gpus-per-task=1 --cores-per-task=24" \
             "--setopt=mpibind=verbose:1 ${EXE} ${_args[*]}"

        flux run \
            --ntasks="${_np}" \
            --nodes="${_nodes}" \
            --exclusive \
            --gpus-per-task=1 \
            --cores-per-task=24 \
            --setopt=mpibind=verbose:1 \
            "${EXE}" "${_args[@]}"
        _launch_rc=$?
        _launched=$(( _launched + 1 ))
        echo "[t3] ${_tag} rc=${_launch_rc}"
        if [ "${_launch_rc}" -ne 0 ]; then
            echo "[t3] FAIL: ${_tag} exited ${_launch_rc}" >&2
            _rc=1
        fi
        echo "[t3] ${_tag} wrote: $(ls -1 "${_ckpt}" 2>/dev/null | wc -l) file(s)"
    done
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
echo "[t3] launches = ${_launched}"
if [ "${_rc}" -eq 0 ]; then
    echo "[t3] ALL LAUNCHES rc=0"
else
    echo "[t3] AT LEAST ONE LAUNCH FAILED -- read the rc lines above" >&2
fi
# A non-zero rc from an `fmm` launch is the interesting case and needs the log
# read, not just the status: BEATNIK_NOT_IMPLEMENTED (the stub still in place,
# i.e. the build did not pick up the change) and the adapter's `~canopy`
# runtime_error both look like "fmm ran and failed" from out here.
exit "${_rc}"
