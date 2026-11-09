#!/bin/bash
# flux: --job-name=beatnik_m1d3_probe
# flux: --nodes=1
# flux: --exclusive
# flux: -t 10m
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
# M1-D3 (`tasks/milestone1.md`): is the late mesh of either code still a
# fixed point of the reference's remeshing rules? Applies one reference
# remesh plus cleanup (`m1d2_refhooks.py probe`, M1-D2's run parameters
# `/p/lustre5/stewartj/beatnik/M1-D2/ref_params.json`) to Beatnik's M1-A1j
# HIP np1 checkpoints at steps 800..1610 every 50 and to the M1-G3 gold at
# steps 900..1610 every 50, and prints one `[m1d2probe]` line per state.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. The probe runs in this script's body, on
# the allocated node, against the reference stage M1-G3j generated the gold
# from (/p/lustre5/stewartj/beatnik/M1-G3j/stage, read only, never edited).
#
# Usage:  "$TASKLOOP_SUBMIT" --purpose "..." scripts/tuolumne/m1d3_probe.flux
# Output: /p/lustre5/stewartj/beatnik/M1-D3/probe/probe.txt
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
    echo "[m1d3p] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi
if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[m1d3p] FAIL: BEATNIK_USE_PROD=1; M1-D3 measures the DEV env." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_stage=/p/lustre5/stewartj/beatnik/M1-G3j/stage
_gold=/p/lustre5/stewartj/beatnik/M1-G3j/out
_bk=/p/lustre5/stewartj/beatnik/M1-A1j/out/Beatnik_Test_Milestone1Bubble_MPI_HIP_np1/beatnik_milestone1/HIP_np1_compare
_params=/p/lustre5/stewartj/beatnik/M1-D2/ref_params.json
_out=/p/lustre5/stewartj/beatnik/M1-D3/probe
_python=/usr/tce/bin/python3
_hooks="${BEATNIK_REPO}/scripts/tuolumne/m1d2_refhooks.py"
export PYTHONDONTWRITEBYTECODE=1

echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[m1d3p] commit: $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1d3p] git status --short:"
git -C "${BEATNIK_REPO}" status --short 2>&1 | sed 's/^/[m1d3p]   /'
echo "[m1d3p] beatnik binaries: none (reference-only job)"
echo "[m1d3p] sha256 $(sha256sum "${_hooks}" | cut -c1-64) ${_hooks}"
echo "[m1d3p] sha256 $(sha256sum "${_params}" | cut -c1-64) ${_params}"
echo "[m1d3p] reference stage = ${_stage}; stage/COMMIT:"
sed 's/^/[m1d3p]   /' "${_stage}/COMMIT"
_ref_dirty="$(sed -n '/^reference_zmodel3d_status_begin$/,/^reference_zmodel3d_status_end$/p' \
                  "${_stage}/COMMIT" | sed '1d;$d')"
if ! grep -qx 'reference_zmodel3d_status_end' "${_stage}/COMMIT"; then
    echo "[m1d3p] FAIL: stage/COMMIT carries no reference zmodel3d/ status block." >&2
    exit 1
fi
echo "[m1d3p] reference zmodel3d/ status (from COMMIT) = '${_ref_dirty}'"
echo "[m1d3p] python = ${_python} ($(${_python} --version 2>&1)), numpy $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[m1d3p] host = $(hostname)  date = $(date -Is)"
echo "[m1d3p] output = ${_out}"
echo "[m1d3p] submit = \"\$TASKLOOP_SUBMIT\" --purpose ... scripts/tuolumne/m1d3_probe.flux"
echo "=================================================================="
if [ -n "${_ref_dirty}" ]; then
    echo "[m1d3p] FAIL: the staged reference's zmodel3d/ package was not clean." >&2
    exit 1
fi
if [ -e "${_out}" ] && [ -n "$(ls -A "${_out}" 2>/dev/null)" ]; then
    echo "[m1d3p] FAIL: ${_out} is not empty; refusing to mix runs." >&2
    exit 1
fi
mkdir -p "${_out}" || exit 1

_items=()
for _s in $(seq 800 50 1600) 1610; do
    _f="$(ls "${_bk}"/checkpoint_t*_step$(printf '%07d' "${_s}").h5 2>/dev/null)"
    [ -n "${_f}" ] && _items+=( "beatnik=${_f}" )
done
for _s in $(seq 900 50 1600) 1610; do
    _f="$(ls "${_gold}"/checkpoint_t*_step$(printf '%07d' "${_s}").npz 2>/dev/null)"
    [ -n "${_f}" ] && _items+=( "gold=${_f}" )
done
echo "[m1d3p] === probe: ${#_items[@]} states"
_t0="$(date +%s)"
cd /tmp || exit 1
"${_python}" "${_hooks}" probe "${_stage}" "${_params}" "${_items[@]}" \
    | tee "${_out}/probe.txt"
_rc=${PIPESTATUS[0]}
echo "[m1d3p] probe rc=${_rc} wall=$(( $(date +%s) - _t0 ))s"
exit "${_rc}"
