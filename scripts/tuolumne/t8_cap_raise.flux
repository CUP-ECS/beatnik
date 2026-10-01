#!/bin/bash
# flux: --job-name=beatnik_t8_cap_raise
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
# T8's CAP-RAISE MEASUREMENT (`tasks/add-canopy-t6.md`):
# Beatnik_Probe_FmmKeyDemand at LEVEL 4 on HIP, at np1 and np4, run TWICE at
# each rank count -- once at `argv[2] = 32768` (the control, which must
# reproduce T5's reading) and once at `argv[2] = 65536` (the candidate) -- to
# show that the raised cap admits EVERY demanded key at every state.
#
# WHY FOUR LAUNCHES AND NOT TWO. The candidate alone cannot be read: a launch
# in which `unique_ops == demand` everywhere is equally consistent with "the
# cap was raised and now admits everything" and with "this draw's demand
# happened to stay under 32 768", which at np4 is in fact what happens --
# T5 measured no np4 rank ever reaching the cap. The control pins the
# apparatus to T5's numbers in the SAME job and at the SAME allocation:
#
#   32768 at np1 -> unique_ops CLAMPED at exactly 32 768 at the over-cap
#                   states, first_exceed_step = 900, demand peak ~37 500
#   65536 at np1 -> unique_ops == demand at ALL 81 states,
#                   first_exceed_step = -1, keys_built_delta == demand
#   32768 at np4 -> no rank reaches the cap; first_exceed_step = -1 already
#   65536 at np4 -> identical demand series, unique_ops == demand
#
# The np4 pair is therefore a NULL result by construction and is run anyway:
# the exit criterion asserts `unique_ops == demand` at both rank counts, and a
# raise that perturbed the np4 series would be evidence of something the cap
# is not supposed to touch.
#
# THE CONTROL RUNS FIRST at each rank count, so an apparatus problem shows
# against T5's known numbers before the candidate is read for anything.
#
# NO REBUILD IS NEEDED FOR THIS SCRIPT. T7 made the cap the probe's optional
# `argv[2]`, so both caps are measurable against the INSTALLED binary. The
# source change T8 makes -- the per-level 65536 constant in
# `Beatnik_Test_Milestone0Fmm.cpp`'s level-4 arm -- moves the MEMBER, not the
# probe, and is verified separately.
#
# WHAT THIS SCRIPT CANNOT SHOW. Fallback. Raising the count cap cannot drive
# `global_m2l_fallback` to zero: T5 measured non-zero fallback at 71 of 81
# states at np4, where NO rank ever reaches the cap, so those refusals are not
# cap-driven. The fallback column here is RECORDED, not asserted on, and
# identifying the other refusal path is T8b.
#
# `canopy` MUST READ `+profiling` in the provenance block below. In a
# `~profiling` build the demand column is the `-1` sentinel and nothing in the
# run is a measurement.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# measurement work, not a large production run.
#
# Cost: T5 measured level 4 at 56-57 s (np1) and 69-75 s (np4) per launch, so
# four launches is about 4.5 minutes. `-t 30m` is deliberately generous and
# still makes a hang fail fast rather than hold a `pdebug` allocation.
#
# Usage:  flux batch scripts/tuolumne/t8_cap_raise.flux
#
# Then read beatnik_t8_cap_raise.<jobid>.log in the submitting directory.
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
    echo "[t8] FAIL: cannot locate the Beatnik checkout." >&2
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
echo "[t8] spack env status:"
spack env status 2>&1 | sed 's/^/[t8]   /'
echo "[t8] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t8] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t8] submit  = flux batch scripts/tuolumne/t8_cap_raise.flux"
echo "[t8] canopy  = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^ ]*' | head -1)"
echo "[t8] beatnik = $(spack find --variants beatnik 2>/dev/null | grep -o 'beatnik@[^ ]*' | head -1)"
echo "[t8] canopy variants (MUST read +profiling):"
spack find --variants canopy 2>&1 | grep -i canopy | sed 's/^/[t8]   /'

_target_stem="Beatnik_Probe_FmmKeyDemand"
_backend="HIP"
_level=4

# THE MATRIX, as `<ranks>:<tag>:<cap argument>`. An EMPTY cap field would mean
# the argument is omitted; no entry here uses it, because T8 compares two
# EXPLICIT caps rather than a default against a value -- the installed binary's
# default is still 32768 and naming it explicitly is what makes the control
# readable beside the candidate. Each rank count's CONTROL runs first.
_matrix=( "1:cap32768:32768" "1:cap65536:65536" "4:cap32768:32768" "4:cap65536:65536" )
echo "[t8] target = ${_target_stem}_MPI_${_backend}  level = ${_level}"
echo "[t8] matrix = ${_matrix[*]}  (ranks:tag:argv[2])"

# THE PROBE IS IN NO TIER, so there is no manifest line to read its arguments
# out of. A driver's arguments come from the script that measures with it, and
# the probe's are one required positional (the level) and one optional one (the
# M2L operator column-count cap). `beatnik_exe` still resolves the binary,
# which in installed mode is `command -v <basename>` on the installed
# share/Beatnik/tests directory.
_target="${_target_stem}_MPI_${_backend}"
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[t8] exe   = ${_exe}"

# MUST be on a PARALLEL filesystem: the probe writes an 81-checkpoint series
# through MPI-IO, and a node-local scratch fails every launch spanning more
# than one node.
BEATNIK_T8_SCRATCH_ROOT="${BEATNIK_T8_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t8_cap_raise}"
# Per-job subdirectory, so two submissions of this script cannot collide in the
# filesystem. `FLUX_JOB_ID` is NOT set for a batch script -- flux sets it for
# the tasks the shell launches, not for the batch instance's init program -- so
# ask the batch instance itself, and fall back to the PID for a direct `bash`
# run, which has no job id at all. The PID is enough for collision safety; the
# jobid is what makes the directory traceable back to a log.
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
echo "[t8] scratch root = ${BEATNIK_T8_SCRATCH_ROOT}"
echo "[t8] scratch job  = job${_jobid}"

_rc=0
_pass=0
_fail=0
_names_failed=""
_t0_all=$(date +%s)

for _entry in "${_matrix[@]}"; do
    _np="${_entry%%:*}"
    _rest="${_entry#*:}"
    _tag="${_rest%%:*}"
    _cap="${_rest##*:}"

    # One I/O directory PER (JOB, target, level, rank, TAG), on lustre, deleted
    # and recreated immediately before the launch. The TAG is in the path as
    # well as the job id: the probe keys its own subdirectory by level, space
    # and rank count only, so the two launches at one rank count would
    # otherwise share one series and the second would read back the first's
    # checkpoints.
    #
    # THE JOB ID IN THE PATH IS LOAD-BEARING, not tidiness. T5 lost a launch to
    # two overlapping submissions writing the same checkpoint files, with
    # `H5FD__sec2_lock(): unable to lock file, errno = 11` at step 1325 and a
    # `collective tags 14 and 1 do not match` MPI abort behind it. A collision
    # need not abort to be harmful -- it could also round-trip another job's
    # particle counts -- so the path is per job whether submissions overlap or
    # not.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T8_SCRATCH_ROOT}/job${_jobid}/${_target}_L${_level}_np${_np}_${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
        echo "[t8] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}(scratch)"
        continue
    fi
    echo "[t8] scratch = ${BEATNIK_TEST_SCRATCH}"

    # The probe's argument vector. An empty `_cap` means the OPTIONAL argument
    # is omitted, not passed as an empty string -- `std::atoi("")` is 0, which
    # is a LEGAL cap meaning "admit no column", so an empty argv[2] would
    # measure the wrong configuration and still exit 0.
    _args=( "${_level}" )
    if [ -n "${_cap}" ]; then
        _args+=( "${_cap}" )
    fi

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/t6b_key_demand.flux:238-243 and must
    # not be simplified: a wrong binding does not fail, it oversubscribes one
    # device and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t8] === ${_target} level ${_level} at ${_np} rank(s) / ${_nodes} node(s), cap=${_cap:-default} ==="
    echo "[t8] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
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
        echo "[t8] PASS ${_backend} L${_level} np=${_np} ${_tag} in ${_dt}s"
        _pass=$(( _pass + 1 ))
    else
        echo "[t8] FAIL ${_backend} L${_level} np=${_np} ${_tag}: rc=${_obs} after ${_dt}s" >&2
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}"
        _rc=1
    fi
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t8] total wall $(( $(date +%s) - _t0_all ))s over ${_total} launch(es)"
if [ "${_total}" -eq 0 ]; then
    echo "[t8] FAIL: no launch ran at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t8] SUMMARY: PASS (${_pass}/${_total} launches)"
    echo "[t8] A PASS HERE IS NOT THE RESULT -- every launch exits 0 whatever"
    echo "[t8] the cap does, because the probe asserts on no measured value."
    echo "[t8] Read, per launch, 81 rows x ranks and the [t6probe] header:"
    echo "[t8]   cap65536 launches  header op_count_cap=65536 op_cap=65536,"
    echo "[t8]                      unique_ops == demand at ALL 81 states on"
    echo "[t8]                      EVERY rank, first_exceed_step = -1,"
    echo "[t8]                      demand_saturated = 0 everywhere, and"
    echo "[t8]                      keys_built_delta == demand at the np1 peak"
    echo "[t8]                      state rather than 32768."
    echo "[t8]   cap32768 launches  header op_count_cap=32768 op_cap=32768, and"
    echo "[t8]                      at np1 unique_ops CLAMPED at exactly 32768"
    echo "[t8]                      with first_exceed_step = 900 -- T5's"
    echo "[t8]                      reading, reproduced. Without this half the"
    echo "[t8]                      candidate's clean series could be a draw in"
    echo "[t8]                      which demand simply stayed low."
    echo "[t8] FALLBACK IS RECORDED, NOT ASSERTED ON. global_m2l_fallback does"
    echo "[t8] NOT reach zero at either cap or either rank count; T5 measured"
    echo "[t8] 71 of 81 np4 states non-zero with no rank anywhere near the cap,"
    echo "[t8] so those refusals are not cap-driven and T8b owns them. A"
    echo "[t8] non-zero fallback column here is not a T8 failure."
    echo "[t8] Count the states: 81 per rank per launch, or the series is"
    echo "[t8] incomplete whatever the rc says."
else
    echo "[t8] SUMMARY: FAIL (${_pass}/${_total} launches);" \
         "failed:${_names_failed}" >&2
    # The probe asserts on NO measured value, so a non-zero rc here is a run
    # that could not proceed -- a throw, a non-finite velocity, an early stop,
    # a round-trip mismatch, or the R5 parameter check finding that the probe
    # and Beatnik_Test_Milestone0Fmm.cpp have drifted apart. A negative argv[2]
    # also lands here, by design, rather than reaching Canopy's own throw.
    # Read the [FAIL] detail lines; none of them is a number to widen.
fi
exit "${_rc}"
