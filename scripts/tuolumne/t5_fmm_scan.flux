#!/bin/bash
# flux: --job-name=beatnik_t5_scan
# flux: --nodes=1
# flux: --exclusive
# flux: -t 45m
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
# T5 STEPS 1-6 (tasks/canopy/add-canopy.md): the far-field fidelity scan.
#
# Not a test run: no tier, no gate, and no pass/fail criterion beyond "every
# launch completed its scan". Beatnik_Test_FmmScan compiles no tolerance --
# T5 produces the numbers T6 compiles -- so this script's exit status says the
# MEASUREMENT completed, never that Beatnik agreed with anything.
#
# EACH LAUNCH IS ONE SCAN AGAINST ONE SHARED STATE, which is the constraint the
# driver is shaped around and the reason the scan is not spread over launches.
# The spin-up is downstream of five timesteps and the timestep is not bitwise
# reproducible (T3: 1.15e-15 np1 / 1.88e-15 np4 of field RMS run-to-run on an
# identical binary), and T4 measured the consequence downstream: the same arm at
# np=1 gave 5.00765e-4 on one job and 5.00819e-4 on another, a floor of about
# 1e-4 OF THE ERROR. Within one launch every arm sees byte-identical state and
# arm-to-arm differences are exact. ACROSS the launches below -- different
# level, different rank count, different backend -- they are not, and any figure
# read across two rows of this sweep carries that floor.
#
#     matrix : (level 4, level 3) x (HIP np1, HIP np4) plus SERIAL np1
#
# WHY LEVEL 3 IS IN IT AT ALL. At theta = 0.3 the near field reaches
# sqrt3/theta = 5.77 cell widths, covering pi*(sqrt3/theta)^2 ~ 105 occupied
# leaves of a 2-manifold, and 642 vertices at ncrit = 8 occupy only 80. NO ncrit
# makes the far field live at level 3 -- lower values degenerate the tree -- so
# the level-3 rows are there to MEASURE that, as the finding for that level,
# rather than to be tuned toward liveness. The far-field accuracy claim rests on
# the level-4 rows.
#
# WHY BOTH BACKENDS. HIP is tuolumne's production path and is what the numbers
# are quoted at. SERIAL is the cross-check: it shares no reduction order with
# HIP, so a figure that agrees across the two is not an artifact of GPU
# summation order.
#
# THE SCAN WRITES NO CHECKPOINTS, so BEATNIK_TEST_SCRATCH is deliberately not
# set -- makeScanParams leaves the checkpoint directory empty and CheckpointIO
# is a no-op without one. Nothing here touches a filesystem, exactly as
# t4_fmm_vs_direct.flux does.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# iterative measurement work at 642 and 2562 vertices, not a large queued run.
#
# Usage:  flux batch scripts/tuolumne/t5_fmm_scan.flux
#
#   BEATNIK_T5_SPINUP     spin-up steps, default 5. Changing it changes the
#                         state and therefore every number, so it belongs in the
#                         log entry beside the results if it is ever moved.
#   BEATNIK_T5_MATRIX     override the row list, for DEBUGGING A FAILURE only.
#                         One row per line: `level backend ranks`.
#
# Then read beatnik_t5_scan.<jobid>.log in the submitting directory. The table
# is every `[t5arm]` line; each is self-describing key=value and carries its own
# full qualification list, so no row needs the header to be read.
#
# --nodes=1 covers the 4-rank rows at tuolumne's 4 ranks per node.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root. `flux batch` copies this script into a per-job spool
# directory, so BASH_SOURCE points at /var/tmp/... and cannot find the checkout.
# It does preserve the submitting working directory, so walk up from PWD;
# BASH_SOURCE still covers a direct `bash` run.
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
    echo "[t5] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
# These are the numbers T6, T7 and T8 key off. They are only reusable if a later
# session can tell which toolchain produced them, so this block is not optional.
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[t5] spack env status:"
spack env status 2>&1 | sed 's/^/[t5]   /'
echo "[t5] spack find beatnik:"
( cd "${BEATNIK_REPO}" && spack find --format '{name}{@version}{%compiler}' beatnik 2>&1 ) \
    | sed 's/^/[t5]   /' || echo "[t5]   (spack find unavailable)"
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[t5] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3 | sed 's/^/[t5]   /'
fi
echo "[t5] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t5] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t5] git status --short:"
( cd "${BEATNIK_REPO}" && git status --short 2>/dev/null | head -20 | sed 's/^/[t5]   /' )
echo "[t5] hostname = $(hostname)"
echo "[t5] date     = $(date -Is)"
echo "=================================================================="

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[t5] FAIL: BEATNIK_USE_PROD=1. T5 measures the DEV env; a" >&2
    echo "  production-env number is not comparable with T4's, and the prod" >&2
    echo "  env must not be rebuilt under a live job." >&2
    exit 1
fi

BEATNIK_T5_DRIVER="${BEATNIK_T5_DRIVER:-Beatnik_Test_FmmScan}"
BEATNIK_T5_SPINUP="${BEATNIK_T5_SPINUP:-5}"

# One row per launch: `level backend ranks`. Ordered cheapest-first so a
# truncated sweep is missing its most expensive rows rather than an arbitrary
# set of them.
if [ -n "${BEATNIK_T5_MATRIX:-}" ]; then
    MATRIX="${BEATNIK_T5_MATRIX}"
    echo "[t5] *** MATRIX OVERRIDDEN -- this is NOT T5's scan matrix ***"
else
    MATRIX="
3 HIP    1
3 HIP    4
4 HIP    1
4 HIP    4
3 SERIAL 1
4 SERIAL 1
"
fi

echo "[t5] driver=${BEATNIK_T5_DRIVER} spin_up=${BEATNIK_T5_SPINUP}"
echo "[t5] matrix:"
printf '%s\n' "${MATRIX}" | sed '/^$/d; s/^/[t5]   /'

_rc=0
_pass=0
_fail=0
_failed_rows=""
_t_job0="$(date +%s)"

##--------------------------------------------------------------------------##
## The sweep
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin, so with
# the row list on stdin the first launched binary swallows every remaining row
# and the sweep silently runs only its first member, reporting a plausible
# partial table. milestone0_divergence.flux already paid for this.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _level="$1"
    _backend="$2"
    _np="$3"

    _tag="sub${_level}_${_backend}_np${_np}"

    _exe="$(beatnik_exe "${BEATNIK_T5_DRIVER}_MPI_${_backend}")" || {
        echo "[t5] FAIL: cannot resolve ${BEATNIK_T5_DRIVER}_MPI_${_backend}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _failed_rows="${_failed_rows} ${_tag}(unresolved)"
        continue
    }

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from t4_fmm_vs_direct.flux:138-145 and must not be
    # simplified: a wrong binding does not fail, it oversubscribes one device
    # and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))

    echo "[t5] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[t5] exe     = ${_exe}"
    echo "[t5] exe mtime = $(date -r "${_exe}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"
    echo "[t5] command: flux run --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1" \
         "${_exe} ${_level} ${BEATNIK_T5_SPINUP}"

    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "${_level}" "${BEATNIK_T5_SPINUP}"
    _obs=$?
    _t1="$(date +%s)"

    echo "[t5] LAUNCH_WALL ${_tag} wall=$(( _t1 - _t0 ))s rc=${_obs}"
    if [ "${_obs}" -eq 0 ]; then
        echo "[t5] PASS ${_tag}"
        _pass=$(( _pass + 1 ))
    else
        echo "[t5] FAIL ${_tag}: rc=${_obs}" >&2
        _fail=$(( _fail + 1 ))
        _failed_rows="${_failed_rows} ${_tag}"
        _rc=1
    fi
done 3<<EOF
${MATRIX}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t5] total wall $(( $(date +%s) - _t_job0 ))s"
if [ "${_total}" -eq 0 ]; then
    echo "[t5] FAIL: no row was run at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t5] SUMMARY: PASS (${_pass}/${_total} rows)"
    echo "[t5] Next: the table is every '[t5arm]' line above. A row whose"
    echo "  p2p_frac is 1.0 measured a direct sum, not an expansion (R1)."
else
    echo "[t5] SUMMARY: FAIL (${_pass}/${_total} rows); failed:${_failed_rows}" >&2
    echo "[t5] A partial scan is NOT the scan -- do not tabulate one without" >&2
    echo "  saying which rows are missing." >&2
fi
exit "${_rc}"
