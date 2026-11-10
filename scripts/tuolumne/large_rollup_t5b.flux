#!/bin/bash
# flux: --job-name=large_rollup_T5b
# flux: --nodes=1
# flux: --exclusive
# flux: -t 30m
# flux: --output=/p/lustre5/stewartj/beatnik/large_rollup/T5b/{{name}}.{{jobid}}.log
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
# large_rollup T5b (`tasks/large_rollup.md` T5b): restart from a Beatnik
# checkpoint, end to end, dev env, 1 node.
#
#   FROZEN MESH, against the Python reference (exact restart check):
#     frz_base   configuration (a) with --no-dynamic-remesh --refine-every 0,
#                level 2, 4 ranks, 10 steps, a checkpoint every 5 steps
#     frz_r4     restart from frz_base's step-5 .h5, --steps 5, 4 ranks
#     frz_r1     the same at 1 rank
#     ref_conv   m1d3_refrun.py convert of the step-5 .h5
#     ref_run    the reference restarted from it for 5 steps
#                (m1d3_refrun.py run, stage /p/lustre5/stewartj/beatnik/M1-G3j/stage)
#     compare_output.py of frz_r4's and frz_r1's step-10 .h5 against the
#     reference's step-10 .npz, at its defaults (`RESULT: match` expected)
#   DYNAMIC REMESH (configuration (a), level 2, 4 ranks):
#     dyn_full     70 steps, a checkpoint every 50 steps
#     dyn_restart  restart from dyn_full's step-50 .h5, --steps 20
#     each run's final progress line, and the checks: dyn_restart reaches
#     step 70, |volRel| <= 1e-12, F within 10 % of dyn_full's step-70 F
#     (no equality is asserted: a restarted dynamic-remesh run is not the
#     uninterrupted run, `tasks/large_rollup.md` Deliberate deviations)
#   SERIES ACROSS A RESTART (frozen, 4 ranks):
#     ser_seg1  5 steps into series/
#     ser_seg2  restart from series/checkpoint_latest.h5 into series/, 5 steps
#     the master series.xmf and the .xmfindex both list steps 0, 5, 10
#   FAILURE DIRECTION:
#     nonexist  --restart-from /nonexistent.h5 exits non-zero naming the path
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/large_rollup_t5b.flux
# Output: /p/lustre5/stewartj/beatnik/large_rollup/T5b/<jobid>/<tag>/
#
# Every launch prints `[large_rollup] <tag> rc=<rc> wall=<s>`, and every check
# prints `[large_rollup] CHECK <name>: PASS|FAIL ...`. The job exits non-zero
# if any check fails.
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
if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[large_rollup] FAIL: BEATNIK_USE_PROD=1; T5b runs the DEV env." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

JOBID="$(flux getattr jobid 2>/dev/null || echo unknown)"
OUT="/p/lustre5/stewartj/beatnik/large_rollup/T5b/${JOBID}"
mkdir -p "${OUT}" || exit 1

STAGE=/p/lustre5/stewartj/beatnik/M1-G3j/stage
PYTHON=/usr/tce/bin/python3
REFRUN="${BEATNIK_REPO}/scripts/tuolumne/m1d3_refrun.py"
CMP="${BEATNIK_REPO}/tests/regression_tests/compare_output.py"
export PYTHONDONTWRITEBYTECODE=1

# --- provenance ------------------------------------------------------------
beatnik_env_summary
EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[large_rollup] exe       = ${EXE}"
echo "[large_rollup] spack     ="
spack find -lv beatnik 2>&1 | sed 's/^/[large_rollup]   /'
echo "[large_rollup] compiler  = $(hipcc --version 2>/dev/null | head -1 || echo unknown)"
echo "[large_rollup] HEAD      = $(git -C "${BEATNIK_REPO}" rev-parse HEAD)"
echo "[large_rollup] porcelain = $(git -C "${BEATNIK_REPO}" status --porcelain | wc -l)"
echo "[large_rollup] submit    = \"\$TASKLOOP_SUBMIT\" --purpose \"T5b exit criterion: restart vs reference, dynamic restart, series, nonexistent path\" scripts/tuolumne/large_rollup_t5b.flux"
echo "[large_rollup] host      = $(hostname)  jobid = ${JOBID}"
echo "[large_rollup] python    = ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "[large_rollup] ref stage = ${STAGE}; stage/COMMIT:"
sed 's/^/[large_rollup]   /' "${STAGE}/COMMIT"
echo "[large_rollup] output    = ${OUT}"

_fail=0
check() {   # name, condition-status (0 = pass), detail
    if [ "$2" -eq 0 ]; then
        echo "[large_rollup] CHECK $1: PASS $3"
    else
        echo "[large_rollup] CHECK $1: FAIL $3"
        _fail=1
    fi
}

# launch <tag> <ranks> <beatnik arguments...>: one Beatnik launch with the
# rank-to-GPU binding of large_rollup_probe_c.flux, its own 5-minute cap, a
# timestamped launch.log, and one summary line. Sets _rc.
launch() {
    local _tag="$1" _np="$2"
    shift 2
    local _nodes=$(( (_np + 3) / 4 ))
    local _dir="${OUT}/${_tag}"
    mkdir -p "${_dir}"
    echo "[large_rollup] === ${_tag}: np=${_np} nodes=${_nodes} $*"
    local _t0
    _t0=$(date +%s.%N)
    echo "${_t0} [large_rollup] ${_tag} np=${_np} args: $*" > "${_dir}/launch.log"
    flux run -t 5m \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${EXE}" "$@" 2>&1 \
        | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done \
        >> "${_dir}/launch.log"
    _rc=${PIPESTATUS[0]}
    local _wall
    _wall=$(awk -v a="${_t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
    grep -E 'restarted from|stopping at|rror|nonfinite' "${_dir}/launch.log" | tail -n 10
    grep -E 'step=' "${_dir}/launch.log" | tail -n 1
    echo "[large_rollup] ${_tag} rc=${_rc} wall=${_wall}"
}

# pylaunch <tag> <python arguments...>: one CPU-only Python launch, 5-minute
# cap, output to <tag>/run.log. Sets _rc.
pylaunch() {
    local _tag="$1"
    shift
    local _dir="${OUT}/${_tag}"
    mkdir -p "${_dir}"
    echo "[large_rollup] === ${_tag}: ${PYTHON} $*"
    local _t0
    _t0=$(date +%s.%N)
    ( cd "${STAGE}" && flux run -t 5m -N 1 -n 1 "${PYTHON}" "$@" ) \
        > "${_dir}/run.log" 2>&1
    _rc=$?
    local _wall
    _wall=$(awk -v a="${_t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
    grep -E 'restarted from|stopping at|\[m1d3conv\]|\[m1d3ref\] step=|Error' "${_dir}/run.log" | tail -n 6
    echo "[large_rollup] ${_tag} rc=${_rc} wall=${_wall}"
}

# The last progress line of a launch.log, without its timestamp.
last_progress() {
    grep -E ' step= *[0-9]+ t=' "$1" | tail -n 1 | cut -d' ' -f2-
}

# A field `name=value` of a progress line.
field() {
    printf '%s\n' "$1" | tr ' ' '\n' | sed -n "s/^$2=//p" | head -n 1
}

PHYS=( --A 0.3 --g 1.0 --mu 0.002
       --viscosity-mode laplace-beltrami
       --br-approximation direct
       --source-quadrature vertex
       --adaptive-dt
       --isotropic-cleanup
       --icosphere-subdivisions 2 )
FROZEN=( "${PHYS[@]}" --no-dynamic-remesh --refine-every 0 )
DYNAMIC=( "${PHYS[@]}" --dynamic-remesh --progress-time-interval 0 )
REFARGS=( --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
          --viscosity-mode laplace-beltrami --br-approximation direct
          --source-quadrature vertex --adaptive-dt --isotropic-cleanup
          --no-dynamic-remesh --refine-every 0 --no-video )

##--------------------------------------------------------------------------##
## Frozen mesh, against the reference
##--------------------------------------------------------------------------##
launch frz_base 4 "${FROZEN[@]}" --steps 10 --checkpoint-every-steps 5 \
    --checkpoint-dir "${OUT}/frz_base"
check frz_base_rc "${_rc}" "(rc=${_rc})"
STEP5="$(ls "${OUT}"/frz_base/checkpoint_t*_step0000005.h5 2>/dev/null | head -n 1)"
[ -n "${STEP5}" ]; check frz_base_step5_exists $? "(${STEP5:-none})"

launch frz_r4 4 "${FROZEN[@]}" --restart-from "${STEP5}" --steps 5 \
    --checkpoint-every-steps 5 --checkpoint-dir "${OUT}/frz_r4"
check frz_r4_rc "${_rc}" "(rc=${_rc})"
launch frz_r1 1 "${FROZEN[@]}" --restart-from "${STEP5}" --steps 5 \
    --checkpoint-every-steps 5 --checkpoint-dir "${OUT}/frz_r1"
check frz_r1_rc "${_rc}" "(rc=${_rc})"

mkdir -p "${OUT}/ref"
pylaunch ref_conv "${REFRUN}" convert "${STEP5}" "${OUT}/ref/restart_step5.npz"
check ref_conv_rc "${_rc}" "(rc=${_rc})"
pylaunch ref_run "${REFRUN}" run "${STAGE}" -- "${REFARGS[@]}" \
    --restart-from "${OUT}/ref/restart_step5.npz" --steps 5 \
    --checkpoint-every-steps 5 --checkpoint-dir "${OUT}/ref/out"
check ref_run_rc "${_rc}" "(rc=${_rc})"
REF10="$(ls "${OUT}"/ref/out/checkpoint_t*_step0000010.npz 2>/dev/null | head -n 1)"
[ -n "${REF10}" ]; check ref_step10_exists $? "(${REF10:-none})"

for _tag in frz_r4 frz_r1 frz_base; do
    _h5="$(ls "${OUT}/${_tag}"/checkpoint_t*_step0000010.h5 2>/dev/null | head -n 1)"
    echo "[large_rollup] === compare ${_tag}: ${_h5:-none} vs ${REF10:-none}"
    "${PYTHON}" "${CMP}" "${_h5}" "${REF10}" > "${OUT}/${_tag}/compare_step10.log" 2>&1
    _crc=$?
    _res="$(grep -E '^RESULT:' "${OUT}/${_tag}/compare_step10.log" | tail -n 1)"
    echo "[large_rollup] compare ${_tag} rc=${_crc} ${_res}"
    if [ "${_tag}" != frz_base ]; then
        [ "${_crc}" -eq 0 ] && [ "${_res}" = "RESULT: match" ]
        check "compare_${_tag}_step10" $? "(${_res:-no RESULT line})"
    fi
done

##--------------------------------------------------------------------------##
## Dynamic remesh
##--------------------------------------------------------------------------##
launch dyn_full 4 "${DYNAMIC[@]}" --steps 70 --checkpoint-every-steps 50 \
    --checkpoint-dir "${OUT}/dyn_full"
check dyn_full_rc "${_rc}" "(rc=${_rc})"
STEP50="$(ls "${OUT}"/dyn_full/checkpoint_t*_step0000050.h5 2>/dev/null | head -n 1)"
[ -n "${STEP50}" ]; check dyn_full_step50_exists $? "(${STEP50:-none})"
launch dyn_restart 4 "${DYNAMIC[@]}" --restart-from "${STEP50}" --steps 20 \
    --checkpoint-every-steps 50 --checkpoint-dir "${OUT}/dyn_restart"
check dyn_restart_rc "${_rc}" "(rc=${_rc})"

_full="$(last_progress "${OUT}/dyn_full/launch.log")"
_rest="$(last_progress "${OUT}/dyn_restart/launch.log")"
echo "[large_rollup] dyn_full    final: ${_full}"
echo "[large_rollup] dyn_restart final: ${_rest}"
_rstep="$(printf '%s\n' "${_rest}" | sed -n 's/^step= *\([0-9]*\) .*/\1/p')"
_vol="$(field "${_rest}" volRel)"
_F_full="$(field "${_full}" F)"
_F_rest="$(field "${_rest}" F)"
[ "${_rstep:-0}" = 70 ]; check dyn_restart_reaches_70 $? "(last step ${_rstep:-none})"
awk -v v="${_vol:-nan}" 'BEGIN { x = v + 0; if (v == "" || v ~ /nan|inf/) exit 1; exit ((x < 0 ? -x : x) <= 1e-12) ? 0 : 1 }'
check dyn_restart_volRel $? "(volRel ${_vol:-none})"
awk -v a="${_F_rest:-0}" -v b="${_F_full:-0}" 'BEGIN { if (b <= 0) exit 1; d = (a - b) / b; if (d < 0) d = -d; exit (d <= 0.10) ? 0 : 1 }'
check dyn_restart_F_within_10pct $? "(F restart ${_F_rest:-none} vs uninterrupted ${_F_full:-none})"

##--------------------------------------------------------------------------##
## The checkpoint series across a restart into the same directory
##--------------------------------------------------------------------------##
launch ser_seg1 4 "${FROZEN[@]}" --steps 5 --checkpoint-every-steps 5 \
    --checkpoint-dir "${OUT}/series"
check ser_seg1_rc "${_rc}" "(rc=${_rc})"
launch ser_seg2 4 "${FROZEN[@]}" --restart-from "${OUT}/series/checkpoint_latest.h5" \
    --steps 5 --checkpoint-every-steps 5 --checkpoint-dir "${OUT}/series"
check ser_seg2_rc "${_rc}" "(rc=${_rc})"
echo "[large_rollup] series index:"
sed 's/^/[large_rollup]   /' "${OUT}/series/checkpoint.xmfindex"
_nidx="$(grep -c . "${OUT}/series/checkpoint.xmfindex" 2>/dev/null || echo 0)"
_nmaster="$(grep -c '<Time Value=' "${OUT}/series/checkpoint.xmf" 2>/dev/null || echo 0)"
_steps="$(sed -n 's/.*_step\([0-9]*\) .*/\1/p' "${OUT}/series/checkpoint.xmfindex" | tr '\n' ' ')"
[ "${_nidx}" = 3 ] && [ "${_nmaster}" = 3 ] && [ "${_steps}" = "0000000 0000005 0000010 " ]
check series_continued $? "(index lines ${_nidx}, master timesteps ${_nmaster}, steps ${_steps})"

##--------------------------------------------------------------------------##
## Failure direction
##--------------------------------------------------------------------------##
launch nonexist 1 "${FROZEN[@]}" --restart-from /nonexistent.h5 --steps 5 \
    --checkpoint-dir "${OUT}/nonexist"
_nrc="${_rc}"
grep -E 'error' "${OUT}/nonexist/launch.log" | head -n 3
[ "${_nrc}" -ne 0 ] && grep -q "/nonexistent.h5" "${OUT}/nonexist/launch.log"
check nonexistent_path_fails $? "(rc=${_nrc})"

echo "[large_rollup] T5b overall: $([ "${_fail}" -eq 0 ] && echo PASS || echo FAIL)"
exit "${_fail}"
