#!/bin/bash
# flux: --job-name=beatnik_t6_l3_member
# flux: --nodes=1
# flux: --exclusive
# flux: -t 58m
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
# T6's PRE-SUBMISSION CHECK: Beatnik_Test_Milestone0Fmm -- the LEVEL-3 member,
# alone -- at all four (backend, rank) combinations of the milestone tier, at
# FULL step count.
#
# WHY THIS EXISTS. The full tier run with T6's two members in it is hours long
# and goes to `-q pbatch`. A registration typo, an off-by-one in the ladder JSON
# parsing or a mis-derived volume-drift rtol should not be discovered five hours
# into that job, and the tier runner has no per-member filter -- only
# BEATNIK_MILESTONE_BACKENDS, _RANKS and _SCRATCH_ROOT -- so the member binaries
# are invoked directly here.
#
# **THIS IS NOT A SHORTENED SUBSTITUTE.** It is the real member at full
# fidelity: 2000 direct steps with 81 same-state FMM evaluations, then 2000
# FMM-driven steps, then the divergence ladder. It exercises every assertion,
# both of claim A's negative cases, claim B's negative case, the ladder JSON
# parsing and the volume-drift bound. There is no step-count or
# checkpoint-interval override anywhere in this path, deliberately (risk R9):
# a knob that can silently shorten a 2000-step run is how a truncated run reads
# as a shorter pass.
#
# WHY LEVEL 3 AND NOT LEVEL 4. Cost, and nothing else. T5 measured a 2000-step
# level-3 FMM trajectory at 256 s at HIP np1 against level 4's 2377 s, so the
# four level-3 launches plausibly fit `pdebug`'s 1 h cap and the four level-4
# ones certainly do not. The level-4 member is checked by the tier run itself.
#
# Estimated from T5's and M0-D1's measured rows: HIP np1 ~364 s, HIP np4 ~282 s,
# SERIAL np1 ~702 s, SERIAL np4 ~482 s -- about 31 minutes against this script's
# `-t 58m`. The SERIAL FMM cost is EXTRAPOLATED (T5 measured only HIP
# trajectories) at roughly 2x HIP, from the T5 scan's 29 s level-4 Serial row
# against its HIP rows. If that extrapolation is badly wrong this script is what
# finds out, which is the second reason to run it before the tier.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is test
# work, not a large production run.
#
# Usage:  flux batch scripts/tuolumne/t6_l3_member.flux [BACKEND...]
#
#   BACKEND...  backends to run; default `SERIAL HIP`, which with ranks 1 and 4
#               is the member's full share of the tier. Overriding it is for
#               DEBUGGING A FAILURE and is announced loudly in the log.
#
# Then read beatnik_t6_l3_member.<jobid>.log in the submitting directory.
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
    echo "[t6l3] FAIL: cannot locate the Beatnik checkout." >&2
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
echo "[t6l3] spack env status:"
spack env status 2>&1 | sed 's/^/[t6l3]   /'
echo "[t6l3] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t6l3] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t6l3] submit  = flux batch scripts/tuolumne/t6_l3_member.flux"

_target_stem="Beatnik_Test_Milestone0Fmm"

if [ "$#" -gt 0 ]; then
    _backends=( "$@" )
    echo "[t6l3] *** BACKEND SET OVERRIDDEN: ${_backends[*]} -- this is NOT the" \
         "member's full share of the tier, which is SERIAL HIP ***"
else
    _backends=( SERIAL HIP )
fi
_ranks=( 1 4 )
echo "[t6l3] backends = ${_backends[*]}  ranks = ${_ranks[*]}"

# The member's arguments are MANIFEST-RELATIVE, so the launch runs from the
# manifest directory -- the same convention, and the same reasoning, as the
# tier runner's. Locate the manifest the way that runner does.
_manifest=""
_saved_ifs="${IFS}"
IFS=':'
for _p in ${PATH}; do
    if [ -f "${_p}/beatnik_milestone_manifest.txt" ]; then
        _manifest="${_p}/beatnik_milestone_manifest.txt"
        break
    fi
done
IFS="${_saved_ifs}"
if [ -z "${_manifest}" ]; then
    echo "[t6l3] FAIL: beatnik_milestone_manifest.txt not found on PATH." >&2
    echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing?" >&2
    exit 1
fi
_manifest_dir="$(cd "$(dirname "${_manifest}")" && pwd)"
echo "[t6l3] manifest = ${_manifest}"

# MUST be on a PARALLEL filesystem: this member writes TWO 2000-step checkpoint
# series through MPI-IO, and a node-local scratch fails every launch spanning
# more than one node.
BEATNIK_T6_SCRATCH_ROOT="${BEATNIK_T6_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t6_l3check}"
echo "[t6l3] scratch root = ${BEATNIK_T6_SCRATCH_ROOT}"

_rc=0
_pass=0
_fail=0
_names_failed=""
_t0_all=$(date +%s)

for _backend in "${_backends[@]}"; do
    _target="${_target_stem}_MPI_${_backend}"
    # The arguments this target takes, read from the manifest rather than
    # retyped: a hand-copied gold path that names the wrong level is exactly the
    # mistake the registration loop's FATAL_ERROR exists to prevent, and
    # retyping it here would route around that.
    _line="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
             awk -v t="${_target}" '$1 == t' || true)"
    if [ -z "${_line}" ]; then
        echo "[t6l3] FAIL: the manifest names no ${_target}." >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}(absent)"
        continue
    fi
    # shellcheck disable=SC2086
    set -- ${_line}
    shift
    _args=( "$@" )
    _exe="$(beatnik_exe "${_target}")" || { _rc=1; continue; }
    echo "[t6l3] exe   = ${_exe}"
    echo "[t6l3] args  = ${_args[*]}"

    for _np in "${_ranks[@]}"; do
        # One I/O directory PER (target, rank), on lustre, deleted and recreated
        # so a stale checkpoint from an earlier run cannot be read back as this
        # run's output -- which for claim B's ladder would be a directory
        # holding two files for one step, i.e. a measurement of a mixture.
        export BEATNIK_TEST_SCRATCH="${BEATNIK_T6_SCRATCH_ROOT}/${_target}_np${_np}"
        rm -rf "${BEATNIK_TEST_SCRATCH}"
        if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
            echo "[t6l3] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
            _rc=1
            _fail=$(( _fail + 1 ))
            _names_failed="${_names_failed} ${_backend}_np${_np}(scratch)"
            continue
        fi
        echo "[t6l3] scratch = ${BEATNIK_TEST_SCRATCH}"

        # Tuolumne packs 4 ranks per node; round the node count up. The binding
        # is copied EXACTLY from scripts/tuolumne/run_milestone.flux:211-218 and
        # must not be simplified: a wrong binding does not fail, it
        # oversubscribes one device and returns a plausible number that reads
        # like a real measurement.
        _nodes=$(( (_np + 3) / 4 ))
        echo "[t6l3] === ${_target} at ${_np} rank(s) / ${_nodes} node(s) ==="
        echo "[t6l3] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
             "--exclusive --gpus-per-task=1 --cores-per-task=24" \
             "--setopt=mpibind=verbose:1 ${_exe} ${_args[*]}"

        _t0=$(date +%s)
        ( cd "${_manifest_dir}" && flux run \
            --ntasks="${_np}" \
            --nodes="${_nodes}" \
            --exclusive \
            --gpus-per-task=1 \
            --cores-per-task=24 \
            --setopt=mpibind=verbose:1 \
            "${_exe}" "${_args[@]}" )
        _obs=$?
        _dt=$(( $(date +%s) - _t0 ))

        if [ "${_obs}" -eq 0 ]; then
            echo "[t6l3] PASS ${_backend} np=${_np} in ${_dt}s"
            _pass=$(( _pass + 1 ))
        else
            echo "[t6l3] FAIL ${_backend} np=${_np}: rc=${_obs} after ${_dt}s" >&2
            _fail=$(( _fail + 1 ))
            _names_failed="${_names_failed} ${_backend}_np${_np}"
            _rc=1
        fi
    done
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t6l3] total wall $(( $(date +%s) - _t0_all ))s over ${_total} launch(es)"
if [ "${_total}" -eq 0 ]; then
    echo "[t6l3] FAIL: no launch ran at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t6l3] SUMMARY: PASS (${_pass}/${_total} launches)"
else
    echo "[t6l3] SUMMARY: FAIL (${_pass}/${_total} launches);" \
         "failed:${_names_failed}" >&2
    # Read the per-check detail lines in this log before touching any
    # tolerance. A claim-A state above tau_A, a horizon earlier than the
    # envelope and a volume drift outside the derived rtol are all FINDINGS to
    # be recorded with their step -- not numbers to widen.
fi
exit "${_rc}"
