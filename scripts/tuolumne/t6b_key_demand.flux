#!/bin/bash
# flux: --job-name=beatnik_t6b_key_demand
# flux: --nodes=1
# flux: --exclusive
# flux: -t 30m
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
# T3's VALIDATION LAUNCH (`tasks/add-canopy-t6.md`): Beatnik_Probe_FmmKeyDemand
# at LEVEL 3 on HIP, at ranks 1 and 4.
#
# WHAT IT IS FOR. T3 does not measure the demand series -- T5 does, at level 4.
# This script exists to show that the probe RUNS: that both per-backend targets
# resolve, that the rank-local fields are printed once per rank rather than
# reduced, and -- in the failure direction -- that against the `canopy
# ~profiling` this env still concretizes the demand column reads the `-1`
# SENTINEL and says so loudly, while the two ungated depth columns beside it
# carry real non-zero counts. A probe that reported `-1` for THOSE would be
# reporting its own bug (risk R7).
#
# **RUN THIS BEFORE T4.** Turning `+profiling` on makes the `-1` unreachable in
# this environment and the observation cannot be retaken.
#
# WHY np1 AND np4. np4 is the only launch that exercises the per-rank unreduced
# printing at all: at np1 "per rank" and "reduced" are the same four lines. The
# operator-key cap is PER RANK, so np4 carries four times the key budget and
# overflows LESS than np1, not four times as much (T0 measured 13 274 fallback
# pairs at np1 against 5 300 at np4, same step, same level). Never average the
# rank-local columns across ranks.
#
# WHY LEVEL 3 AND WHY HIP. Cost. Level-3 claim A -- which drives the same 2000
# direct steps and does strictly MORE work per state, since it runs the direct
# comparator too -- measured 166 s on HIP, so two probe launches sit far inside
# `pdebug`'s 1 h cap and `-t 30m` is generous. A short limit makes a hang fail
# fast. The `_MPI_SERIAL` target must build and `beatnik_exe` must resolve it,
# but it is NOT launched here and nothing is claimed for it.
#
# **T5 EXTENDS THIS FILE** with the level-4 matrix rather than creating its own.
# The level-4 series is hours, not minutes (T0 measured one level-4 FMM
# trajectory at 2 373 s at HIP np1), so T5 owns the queue and `-t` decision that
# comes with it.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# measurement work, not a large production run.
#
# Usage:  flux batch scripts/tuolumne/t6b_key_demand.flux
#
# Then read beatnik_t6b_key_demand.<jobid>.log in the submitting directory.
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
    echo "[t6b] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance -- a number in the progress log is only reusable if a later
## session can tell which toolchain produced it.
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[t6b] spack env status:"
spack env status 2>&1 | sed 's/^/[t6b]   /'
echo "[t6b] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t6b] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t6b] submit  = flux batch scripts/tuolumne/t6b_key_demand.flux"
echo "[t6b] canopy  = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^ ]*' | head -1)"

_target_stem="Beatnik_Probe_FmmKeyDemand"
_backend="HIP"
_level=3
_ranks=( 1 4 )
echo "[t6b] target = ${_target_stem}_MPI_${_backend}  level = ${_level}  ranks = ${_ranks[*]}"

# THE PROBE IS IN NO TIER, so there is no manifest line to read its arguments
# out of -- unlike `t6_l3_member.flux`, which reads the member's gold paths out
# of beatnik_milestone_manifest.txt. A driver's arguments come from the script
# that measures with it (tests/CMakeLists.txt, the driver loop), and the
# probe's is one positional: the level. `beatnik_exe` still resolves the
# binary, which in installed mode is `command -v <basename>` on the installed
# share/Beatnik/tests directory.
_target="${_target_stem}_MPI_${_backend}"
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[t6b] exe   = ${_exe}"

# The SERIAL target is not launched here and nothing is claimed for it, but it
# must build and resolve -- the driver loop names its targets
# `<stem>_MPI_<BACKEND>` and a probe that exists on only one backend would
# silently narrow what T5 can measure. Resolution only; no launch.
_serial_exe="$(beatnik_exe "${_target_stem}_MPI_SERIAL")" || {
    echo "[t6b] FAIL: ${_target_stem}_MPI_SERIAL does not resolve." >&2
    exit 1
}
echo "[t6b] serial exe (resolved, NOT launched) = ${_serial_exe}"

# MUST be on a PARALLEL filesystem: the probe writes an 81-checkpoint series
# through MPI-IO, and a node-local scratch fails every launch spanning more
# than one node.
BEATNIK_T6B_SCRATCH_ROOT="${BEATNIK_T6B_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t6b_keydemand}"
echo "[t6b] scratch root = ${BEATNIK_T6B_SCRATCH_ROOT}"

_rc=0
_pass=0
_fail=0
_names_failed=""
_t0_all=$(date +%s)

for _np in "${_ranks[@]}"; do
    # One I/O directory PER (target, level, rank), on lustre, deleted and
    # recreated immediately before the launch so a stale checkpoint from an
    # earlier run cannot be read back as this run's output.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T6B_SCRATCH_ROOT}/${_target}_L${_level}_np${_np}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
        echo "[t6b] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_np${_np}(scratch)"
        continue
    fi
    echo "[t6b] scratch = ${BEATNIK_TEST_SCRATCH}"

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/t6_l3_member.flux:200-208 and must
    # not be simplified: a wrong binding does not fail, it oversubscribes one
    # device and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t6b] === ${_target} level ${_level} at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[t6b] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
         "--exclusive --gpus-per-task=1 --cores-per-task=24" \
         "--setopt=mpibind=verbose:1 ${_exe} ${_level}"

    _t0=$(date +%s)
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "${_level}"
    _obs=$?
    _dt=$(( $(date +%s) - _t0 ))

    if [ "${_obs}" -eq 0 ]; then
        echo "[t6b] PASS ${_backend} L${_level} np=${_np} in ${_dt}s"
        _pass=$(( _pass + 1 ))
    else
        echo "[t6b] FAIL ${_backend} L${_level} np=${_np}: rc=${_obs} after ${_dt}s" >&2
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_np${_np}"
        _rc=1
    fi
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t6b] total wall $(( $(date +%s) - _t0_all ))s over ${_total} launch(es)"
if [ "${_total}" -eq 0 ]; then
    echo "[t6b] FAIL: no launch ran at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t6b] SUMMARY: PASS (${_pass}/${_total} launches)"
    echo "[t6b] Read the [t6probe] header line for demand_available, then the"
    echo "[t6b] per-rank rows. Under this env's ~profiling canopy the demand"
    echo "[t6b] column is the -1 SENTINEL and is NOT a measurement of zero"
    echo "[t6b] demand; occupied_depths and cells_at_max_depth ARE live."
else
    echo "[t6b] SUMMARY: FAIL (${_pass}/${_total} launches);" \
         "failed:${_names_failed}" >&2
    # The probe asserts on NO measured value, so a non-zero rc here is a run
    # that could not proceed -- a throw, a non-finite velocity, an early stop,
    # a round-trip mismatch, or the R5 parameter check finding that the probe
    # and Beatnik_Test_Milestone0Fmm.cpp have drifted apart. Read the [FAIL]
    # detail lines; none of them is a number to widen.
fi
exit "${_rc}"
