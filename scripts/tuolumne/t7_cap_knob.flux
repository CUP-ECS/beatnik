#!/bin/bash
# flux: --job-name=beatnik_t7_cap_knob
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
# T7's FAILURE-DIRECTION PAIR (`tasks/add-canopy-t6.md`):
# Beatnik_Probe_FmmKeyDemand at LEVEL 3 on HIP np1, run TWICE -- once with no
# `argv[2]`, once with `argv[2] = 1024` -- to show that
# `FmmParams::m2l_op_count_cap` reaches Canopy rather than being accepted and
# dropped.
#
# WHAT IT PROVES, AND WHY IT TAKES TWO LAUNCHES. T7 plumbs a new knob through
# `FmmParams` to `FmmConfig::m2l_op_count_cap`. A single default-valued launch
# cannot distinguish "the knob is routed" from "the member exists and nothing
# reads it", because the default IS today's constant: a knob that is accepted
# and silently dropped produces a byte-identical run. So the pair is the test.
#
#   launch A  no argv[2]  -> header op_count_cap=32768 op_cap=32768,
#                            global_m2l_fallback == 0 at EVERY row
#   launch B  argv[2]=1024 -> header op_count_cap=1024  op_cap=1024,
#                            global_m2l_fallback NON-ZERO in its rows
#
# `op_count_cap` is what was CONFIGURED (`FmmParams`, echoed back out of the
# solver) and `op_cap` is what is IN FORCE
# (`DownwardSweep::m2l_effective_op_cap()`, the smaller of the count cap and
# what the byte budget buys). Both must move for the routing claim to hold:
# op_count_cap alone moving would mean Beatnik stored the value and Canopy
# never saw it, and op_cap alone moving is not reachable at all.
#
# WHY 1024 BINDS WITH CERTAINTY AT LEVEL 3. Measured peak `unique_ops` there is
# about 6 400 (`## T3`) and measured demand 5 938 - 6 624 over five draws
# (`## T4`, `## T5`), so 1024 refuses columns at every state by a factor of
# roughly six. This is the one thing level 3 CAN say about the overflow path:
# the level-3 member declares `kFarFieldIsLive = false` and never overflows at
# the DEFAULT cap, which is precisely why launch A's fallback is exactly zero
# and launch B's is not. No level-4 run is needed or wanted here -- T7 ships
# the cap at 32768 byte-for-byte and T8 is the task that changes its value.
#
# WHY LEVEL 3 AND np1, AND WHY THIS FITS `pdebug`. Cost, and nothing else: the
# probe at level 3 is 24 s at HIP np1 (`## T3`), so the pair is about a minute
# and `-t 30m` is already generous. A short limit is deliberate -- it makes a
# hang fail fast instead of holding a `pdebug` allocation. np4 would add the
# per-rank unreduced printing, which T3 and T5 have already exercised and which
# the routing claim does not depend on.
#
# NOT A REPLACEMENT FOR THE OTHER HALF OF T7's EXIT CRITERION. The default
# direction -- that the level-3 FMM member still passes unchanged at HIP np1 --
# is run through `scripts/tuolumne/t6_l3_member.flux HIP`, which already exists
# and reads the member's gold paths out of the milestone manifest. This script
# does not duplicate it.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# measurement work, not a large production run.
#
# Usage:  flux batch scripts/tuolumne/t7_cap_knob.flux
#
# Then read beatnik_t7_cap_knob.<jobid>.log in the submitting directory.
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
    echo "[t7] FAIL: cannot locate the Beatnik checkout." >&2
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
echo "[t7] spack env status:"
spack env status 2>&1 | sed 's/^/[t7]   /'
echo "[t7] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t7] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t7] submit  = flux batch scripts/tuolumne/t7_cap_knob.flux"
echo "[t7] canopy  = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^ ]*' | head -1)"
echo "[t7] beatnik = $(spack find --variants beatnik 2>/dev/null | grep -o 'beatnik@[^ ]*' | head -1)"

_target_stem="Beatnik_Probe_FmmKeyDemand"
_backend="HIP"
_level=3
_np=1

# THE PAIR, as `<tag>:<cap argument>`. An EMPTY cap field means the argument is
# omitted entirely, which is the `FmmParams` default and the control; `1024` is
# the failure direction. The default launch runs FIRST so that an apparatus
# problem shows against the known-good configuration before the capped launch
# is read for anything.
_pair=( "default:" "cap1024:1024" )
echo "[t7] target = ${_target_stem}_MPI_${_backend}  level = ${_level}  ranks = ${_np}"
echo "[t7] pair   = ${_pair[*]}  (tag:argv[2], empty = argument omitted)"

# THE PROBE IS IN NO TIER, so there is no manifest line to read its arguments
# out of -- unlike `t6_l3_member.flux`, which reads the member's gold paths out
# of beatnik_milestone_manifest.txt. A driver's arguments come from the script
# that measures with it, and the probe's are one required positional (the
# level) and one optional one (the M2L operator column-count cap). `beatnik_exe`
# still resolves the binary, which in installed mode is `command -v <basename>`
# on the installed share/Beatnik/tests directory.
_target="${_target_stem}_MPI_${_backend}"
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[t7] exe   = ${_exe}"

# MUST be on a PARALLEL filesystem: the probe writes an 81-checkpoint series
# through MPI-IO, and a node-local scratch fails every launch spanning more
# than one node.
BEATNIK_T7_SCRATCH_ROOT="${BEATNIK_T7_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t7_cap_knob}"
# Per-job subdirectory, so two submissions of this script cannot collide in the
# filesystem. `FLUX_JOB_ID` is NOT set for a batch script -- flux sets it for
# the tasks the shell launches, not for the batch instance's init program -- so
# ask the batch instance itself, and fall back to the PID for a direct `bash`
# run, which has no job id at all. The PID is enough for collision safety; the
# jobid is what makes the directory traceable back to a log.
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
echo "[t7] scratch root = ${BEATNIK_T7_SCRATCH_ROOT}"
echo "[t7] scratch job  = job${_jobid}"

_rc=0
_pass=0
_fail=0
_names_failed=""
_t0_all=$(date +%s)

for _entry in "${_pair[@]}"; do
    _tag="${_entry%%:*}"
    _cap="${_entry##*:}"

    # One I/O directory PER (JOB, target, level, rank, TAG), on lustre, deleted
    # and recreated immediately before the launch. The TAG is in the path as
    # well as the job id: the probe keys its own subdirectory by level, space
    # and rank count only, so the two launches here would otherwise share one
    # series and the second would read back the first's checkpoints.
    #
    # THE JOB ID IN THE PATH IS LOAD-BEARING, not tidiness. T5 lost a launch to
    # two overlapping submissions writing the same checkpoint files, with
    # `H5FD__sec2_lock(): unable to lock file, errno = 11` at step 1325 and a
    # `collective tags 14 and 1 do not match` MPI abort behind it. A collision
    # need not abort to be harmful -- it could also round-trip another job's
    # particle counts -- so the path is per job whether submissions overlap or
    # not.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T7_SCRATCH_ROOT}/job${_jobid}/${_target}_L${_level}_np${_np}_${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
        echo "[t7] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}(scratch)"
        continue
    fi
    echo "[t7] scratch = ${BEATNIK_TEST_SCRATCH}"

    # The probe's argument vector. An empty `_cap` means the OPTIONAL argument
    # is omitted, not passed as an empty string -- `std::atoi("")` is 0, which
    # is a LEGAL cap meaning "admit no column", so an empty argv[2] would
    # measure the wrong configuration and still exit 0.
    _args=( "${_level}" )
    if [ -n "${_cap}" ]; then
        _args+=( "${_cap}" )
    fi

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/t6_l3_member.flux:202-209 and must
    # not be simplified: a wrong binding does not fail, it oversubscribes one
    # device and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t7] === ${_target} level ${_level} at ${_np} rank(s) / ${_nodes} node(s), cap=${_cap:-default} ==="
    echo "[t7] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
         "--exclusive --gpus-per-task=1 --cores-per-task=24" \
         "--setopt=mpibind=verbose:1 ${_exe} ${_args[*]}"

    _t0=$(date +%s)
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "${_args[@]}"
    _obs=$?
    _dt=$(( $(date +%s) - _t0 ))

    if [ "${_obs}" -eq 0 ]; then
        echo "[t7] PASS ${_backend} L${_level} np=${_np} ${_tag} in ${_dt}s"
        _pass=$(( _pass + 1 ))
    else
        echo "[t7] FAIL ${_backend} L${_level} np=${_np} ${_tag}: rc=${_obs} after ${_dt}s" >&2
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}"
        _rc=1
    fi
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t7] total wall $(( $(date +%s) - _t0_all ))s over ${_total} launch(es)"
if [ "${_total}" -eq 0 ]; then
    echo "[t7] FAIL: no launch ran at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t7] SUMMARY: PASS (${_pass}/${_total} launches)"
    echo "[t7] A PASS HERE IS NOT THE RESULT -- both launches exit 0 whatever"
    echo "[t7] the cap does, because the probe asserts on no measured value."
    echo "[t7] Read the two [t6probe] header lines and compare FOUR things:"
    echo "[t7]   default launch  op_count_cap=32768  op_cap=32768"
    echo "[t7]   cap1024 launch  op_count_cap=1024   op_cap=1024"
    echo "[t7] then global_m2l_fallback over the 81 rows of each: EXACTLY ZERO"
    echo "[t7] in every default row, NON-ZERO in the capped rows. Both halves"
    echo "[t7] are needed. If op_count_cap moves and op_cap does not, Beatnik"
    echo "[t7] stored the value and Canopy never saw it -- the knob is accepted"
    echo "[t7] and dropped, which is the exact failure this pair exists to"
    echo "[t7] catch. If the capped launch's fallback is zero, the cap did not"
    echo "[t7] bind and the comparison says nothing; peak unique_ops at level 3"
    echo "[t7] is about 6 400, so 1024 must refuse columns at every state."
    echo "[t7] Count the states: 81 per launch, or the series is incomplete"
    echo "[t7] whatever the rc says."
else
    echo "[t7] SUMMARY: FAIL (${_pass}/${_total} launches);" \
         "failed:${_names_failed}" >&2
    # The probe asserts on NO measured value, so a non-zero rc here is a run
    # that could not proceed -- a throw, a non-finite velocity, an early stop,
    # a round-trip mismatch, or the R5 parameter check finding that the probe
    # and Beatnik_Test_Milestone0Fmm.cpp have drifted apart. A negative argv[2]
    # also lands here, by design, rather than reaching Canopy's own throw.
    # Read the [FAIL] detail lines; none of them is a number to widen.
fi
exit "${_rc}"
