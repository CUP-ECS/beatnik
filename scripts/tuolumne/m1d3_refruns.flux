#!/bin/bash
# flux: --job-name=beatnik_m1d3
# flux: --nodes=1
# flux: --exclusive
# flux: -t 45m
# flux: --output={{name}}.{{jobid}}.log
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
# M1-D3 (`tasks/milestone1.md`): three runs of the Python reference, each
# changing one variable against the M1-G3 gold, to separate the causes of
# Beatnik's late drift from the gold. All three run concurrently, one core
# each, to `time` 1.2382 (just past gold step 1610's 1.238122), a checkpoint
# every 10 steps:
#
#   ctl    restarted from the gold's own step-10 checkpoint. Changes nothing
#          but the restart path; its distance from the gold is the floor
#          against which the other two are read.
#   bk10   restarted from Beatnik's step-10 checkpoint (M1-A1j, HIP np1,
#          converted by `m1d3_refrun.py convert`). Changes only the state the
#          reference starts from: Beatnik's 518/1032 mesh, selected by
#          Tessera at steps 1-3, against the gold's 547/1090. Reference solver,
#          rules and step-size rule throughout.
#   dtpin  from step 0 with the reference's own command, its n-th step size
#          replaced by Beatnik's n-th (M1-A1j HIP np1 `[m1traj]`). Changes only
#          the step-size sequence.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. The reference is the stage M1-G3j
# generated the gold from (/p/lustre5/stewartj/beatnik/M1-G3j/stage, read
# only), wrapped by `scripts/tuolumne/m1d3_refrun.py`, never edited.
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/m1d3_refruns.flux
# Output: /p/lustre5/stewartj/beatnik/M1-D3/ref/<run>/ (checkpoints, run.log);
# then `m1d3_drift.py` reads them on the login node.
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
    echo "[m1d3] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi
if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[m1d3] FAIL: BEATNIK_USE_PROD=1; M1-D3 measures the DEV env." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_stage=/p/lustre5/stewartj/beatnik/M1-G3j/stage
_gold=/p/lustre5/stewartj/beatnik/M1-G3j/out
_root=/p/lustre5/stewartj/beatnik/M1-D3
_out="${_root}/ref"
_bk10="${_root}/inputs/beatnik_step0000010.npz"
_g10="${_gold}/checkpoint_t00000p011017_step0000010.npz"
_bklog=/p/lustre5/stewartj/beatnik/M1-A1j/out/Beatnik_Test_Milestone1Bubble_MPI_HIP_np1/launch.log
_python=/usr/tce/bin/python3
_wrap="${BEATNIK_REPO}/scripts/tuolumne/m1d3_refrun.py"
_tend=1.2382
export PYTHONDONTWRITEBYTECODE=1

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[m1d3] commit: $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1d3] git status --short:"
git -C "${BEATNIK_REPO}" status --short 2>&1 | sed 's/^/[m1d3]   /'
echo "[m1d3] beatnik binaries: none (reference-only job)"
for _f in "${_wrap}" "${_bk10}" "${_g10}" "${_bklog}"; do
    echo "[m1d3] sha256 $(sha256sum "${_f}" | cut -c1-64) ${_f}"
done
echo "[m1d3] reference stage = ${_stage}; stage/COMMIT:"
sed 's/^/[m1d3]   /' "${_stage}/COMMIT"
_ref_dirty="$(sed -n '/^reference_zmodel3d_status_begin$/,/^reference_zmodel3d_status_end$/p' \
                  "${_stage}/COMMIT" | sed '1d;$d')"
if ! grep -qx 'reference_zmodel3d_status_end' "${_stage}/COMMIT"; then
    echo "[m1d3] FAIL: stage/COMMIT carries no reference zmodel3d/ status block." >&2
    exit 1
fi
echo "[m1d3] reference zmodel3d/ status (from COMMIT) = '${_ref_dirty}'"
echo "[m1d3] python = ${_python} ($(${_python} --version 2>&1)), numpy $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1), scipy $(${_python} -c 'import scipy; print(scipy.__version__)' 2>&1)"
echo "[m1d3] host = $(hostname)  date = $(date -Is)"
echo "[m1d3] output = ${_out}  t_end = ${_tend}"
echo "[m1d3] submit = \"\$TASKLOOP_SUBMIT\" --purpose ... scripts/tuolumne/m1d3_refruns.flux"
echo "=================================================================="
if [ -n "${_ref_dirty}" ]; then
    echo "[m1d3] FAIL: the staged reference's zmodel3d/ package was not clean." >&2
    exit 1
fi
for _f in "${_bk10}" "${_g10}" "${_bklog}"; do
    [ -f "${_f}" ] || { echo "[m1d3] FAIL: missing input ${_f}" >&2; exit 1; }
done
if [ -e "${_out}" ] && [ -n "$(ls -A "${_out}" 2>/dev/null)" ]; then
    echo "[m1d3] FAIL: ${_out} is not empty; refusing to mix runs." >&2
    exit 1
fi
mkdir -p "${_out}/ctl" "${_out}/bk10" "${_out}/dtpin" || exit 1

_common=( --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
          --viscosity-mode laplace-beltrami --br-approximation direct
          --adaptive-dt --dynamic-remesh --isotropic-cleanup
          --source-quadrature vertex
          --steps 3000 --t-end "${_tend}" --checkpoint-every-steps 10 --no-video )

cd "${_stage}" || exit 1
_t_job0="$(date +%s)"
launch() {   # name, wrapper options..., -- , extra reference options...
    local _name="$1"; shift
    echo "[m1d3] === ${_name}: flux run -N 1 -n 1 ${_python} ${_wrap} run ${_stage} $* --checkpoint-dir ${_out}/${_name}"
    flux run -N 1 -n 1 "${_python}" "${_wrap}" run "${_stage}" "$@" \
        --checkpoint-dir "${_out}/${_name}" > "${_out}/${_name}/run.log" 2>&1 &
}
launch ctl   -- "${_common[@]}" --restart-from "${_g10}"
_p_ctl=$!
launch bk10  -- "${_common[@]}" --restart-from "${_bk10}"
_p_bk10=$!
launch dtpin --dt-from "${_bklog}" -- "${_common[@]}"
_p_dtpin=$!

_rc=0
for _name in ctl bk10 dtpin; do
    _pv="_p_${_name}"
    wait "${!_pv}"
    _r=$?
    echo "[m1d3] ${_name} rc=${_r} wall_since_start=$(( $(date +%s) - _t_job0 ))s" \
         "checkpoints=$(ls "${_out}/${_name}"/*_step???????.npz 2>/dev/null | wc -l)"
    grep -E '^(restarted from|stopping at|\[m1d3ref\] (reference argv|dt pinned))' "${_out}/${_name}/run.log" \
        | sed "s/^/[m1d3] ${_name} /"
    tail -n 2 "${_out}/${_name}/run.log" | sed "s/^/[m1d3] ${_name} tail: /"
    [ "${_r}" -eq 0 ] || _rc=1
done
echo "[m1d3] total=$(( $(date +%s) - _t_job0 ))s rc=${_rc}"
exit "${_rc}"
