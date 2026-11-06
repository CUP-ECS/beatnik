#!/bin/bash
# flux: --job-name=beatnik_t8b_fallback_reasons
# flux: --nodes=1
# flux: --exclusive
# flux: -t 15m
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
# T8b's PER-REASON FALLBACK BREAKDOWN (`tasks/add-canopy-t6.md`):
# Beatnik_Probe_FmmKeyDemand at LEVEL 4 on HIP, at np1 and np4, both at the
# POST-T8 CAP `argv[2] = 65536`, to identify which refusal path routes pairs to
# Canopy's per-pair M2L fallback by counting PAIRS PER REASON.
#
# WHY 65536 AND NOT 32768. At 65536 nothing seen here is cap-driven -- T8
# measured `unique_ops == demand` at all 81 states on every rank at that cap,
# so the count-cap counter should read 0 at every state and the non-cap counter
# should carry the whole remaining fallback total. That is the measurement;
# running the control again at 32768 would only reproduce T8's own table.
#
# THE QUESTION. T5 found that at np4 NO rank's demand ever reaches even the old
# 32768 cap (`first_exceed_step = -1` for all four ranks in all three draws),
# yet `global_m2l_fallback` is non-zero at 71 of 81 states. T8 then raised the
# cap and removed 44.3 % of np1 fallback PAIRS at the 37 over-cap states while
# moving the non-zero STATE count by exactly zero -- 71 of 81 at both caps and
# both rank counts. So a second refusal path carries 100 % of np4 fallback and
# 56 % of the np1 pairs, and no cap value can reach it.
#
# WHAT THE TWO NEW COLUMNS ARE. Every pair that reaches the per-pair
# m2l_translate path does so for exactly one of two reasons, and they are only
# distinguishable at Canopy's local->global remap:
#
#   fb_range_guard  the classify pass never hashed a key for the pair, because
#                   max_d fell outside [0, max_depth] or one of dd, ii, jj, kk
#                   exceeded its bound. A REPRESENTABILITY limit of the key
#                   encoding (KernelType::m2l_key_dd_max and
#                   M2L_KEY_OFFSET_MAX), which no cap can raise.
#   fb_count_cap    the key WAS hashed and the merge then refused it a column.
#                   A BUDGET, and the one T8 already moved.
#
# plus fb_dropped, which MUST be 0: a refused pair whose target depth is
# outside [0, max_depth] is placed in neither an operator column nor the
# fallback table, so it is invisible to total_fallback_pair_count() and its
# contribution is never evaluated -- a wrong velocity, not a slow one.
#
# THE IDENTITY IS ASSERTED, NOT INSPECTED. The probe checks
# `fb_range_guard + fb_count_cap == global_m2l_fallback` and `fb_dropped == 0`
# at every one of the 81 states, so this script's rc is meaningful for the
# FIRST time among the probe's runners: a breakdown that does not account for
# every fallback pair fails the launch rather than printing a plausible number.
# Every other column the probe prints is still asserted on by nothing.
#
# WHAT TO EXPECT, from T8's measurements:
#
#   np1 65536  fb_count_cap = 0 at all 81 states (nothing is cap-driven at
#              this cap), fb_range_guard = global_m2l_fallback at the 71
#              non-zero states, fb_total about 216 288 over the series.
#   np4 65536  the same shape, with fb_total about 100 % non-cap by
#              construction -- no np4 rank has ever reached either cap.
#   both       fb_dropped = 0 everywhere, 71 of 81 states with non-zero
#              fallback, and the ten zero-fallback states at steps 0 through
#              225 at occupied_depths 5 or 6.
#
# IF fb_range_guard DOMINATES, the finding is that the remaining fallback is a
# representability limit of the basis's key encoding and not a budget. That is
# recorded and raised; M2L_KEY_DD_MAX and M2L_KEY_OFFSET_MAX are NOT changed
# here, and T9a's precondition of zero fallback at all 81 states may then be
# unreachable without a new task at the key encoding.
#
# THE ~profiling FAILURE DIRECTION IS NOT TAKEN HERE. This env concretizes
# `canopy +profiling` and has since T4, so no Beatnik binary in it can observe
# the -1 sentinel. It is taken in Canopy, by
# `canopy/scripts/tuolumne/run_t8b_fallback_reasons.flux`, in both of T1's
# cmake trees.
#
# `canopy` MUST READ `+profiling` in the provenance block below. In a
# `~profiling` build all three per-reason columns are the `-1` sentinel, the
# identity check is SKIPPED rather than passed, and nothing in the run is a
# measurement.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# measurement work, not a large production run.
#
# Cost: T8 measured level 4 at 56-60 s (np1) and 69-75 s (np4) per launch, so
# two launches is about 2.5 minutes. `-t 15m` is generous against that and
# still makes a hang fail fast rather than hold a `pdebug` allocation.
#
# Usage:  flux batch scripts/tuolumne/t8b_fallback_reasons.flux
#
# Then read beatnik_t8b_fallback_reasons.<jobid>.log in the submitting
# directory.
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
    echo "[t8b] FAIL: cannot locate the Beatnik checkout." >&2
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
echo "[t8b] spack env status:"
spack env status 2>&1 | sed 's/^/[t8b]   /'
echo "[t8b] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t8b] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t8b] submit  = flux batch scripts/tuolumne/t8b_fallback_reasons.flux"
echo "[t8b] canopy  = $(spack find --variants canopy 2>/dev/null | grep -o 'canopy@[^ ]*' | head -1)"
echo "[t8b] beatnik = $(spack find --variants beatnik 2>/dev/null | grep -o 'beatnik@[^ ]*' | head -1)"
echo "[t8b] canopy variants (MUST read +profiling):"
spack find --variants canopy 2>&1 | grep -i canopy | sed 's/^/[t8b]   /'

_target_stem="Beatnik_Probe_FmmKeyDemand"
_backend="HIP"
_level=4

# THE MATRIX, as `<ranks>:<tag>:<cap argument>`. An EMPTY cap field would mean
# the argument is omitted; neither entry uses it -- the cap is named
# EXPLICITLY, because the whole point of measuring at 65536 is that nothing
# seen is cap-driven, and a run that silently took the installed binary's
# 32768 default would mix the two reasons back together.
#
# NO 32768 CONTROL. T8's job f3bafaXSEZT5 already ran that pair in the same
# shape and its table is in the progress log; repeating it would cost another
# 2.5 minutes to reproduce a number that is not what T8b is asking.
_matrix=( "1:cap65536:65536" "4:cap65536:65536" )
echo "[t8b] target = ${_target_stem}_MPI_${_backend}  level = ${_level}"
echo "[t8b] matrix = ${_matrix[*]}  (ranks:tag:argv[2])"

# THE PROBE IS IN NO TIER, so there is no manifest line to read its arguments
# out of. A driver's arguments come from the script that measures with it, and
# the probe's are one required positional (the level) and one optional one (the
# M2L operator column-count cap). `beatnik_exe` still resolves the binary,
# which in installed mode is `command -v <basename>` on the installed
# share/Beatnik/tests directory.
_target="${_target_stem}_MPI_${_backend}"
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[t8b] exe   = ${_exe}"

# MUST be on a PARALLEL filesystem: the probe writes an 81-checkpoint series
# through MPI-IO, and a node-local scratch fails every launch spanning more
# than one node.
BEATNIK_T8B_SCRATCH_ROOT="${BEATNIK_T8B_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t8b_fallback_reasons}"
# Per-job subdirectory, so two submissions of this script cannot collide in the
# filesystem. `FLUX_JOB_ID` is NOT set for a batch script -- flux sets it for
# the tasks the shell launches, not for the batch instance's init program -- so
# ask the batch instance itself, and fall back to the PID for a direct `bash`
# run, which has no job id at all. The PID is enough for collision safety; the
# jobid is what makes the directory traceable back to a log.
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
echo "[t8b] scratch root = ${BEATNIK_T8B_SCRATCH_ROOT}"
echo "[t8b] scratch job  = job${_jobid}"

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
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T8B_SCRATCH_ROOT}/job${_jobid}/${_target}_L${_level}_np${_np}_${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
        echo "[t8b] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}(scratch)"
        continue
    fi
    echo "[t8b] scratch = ${BEATNIK_TEST_SCRATCH}"

    # The probe's argument vector. An empty `_cap` means the OPTIONAL argument
    # is omitted, not passed as an empty string -- `std::atoi("")` is 0, which
    # is a LEGAL cap meaning "admit no column", so an empty argv[2] would
    # measure the wrong configuration and still exit 0.
    _args=( "${_level}" )
    if [ -n "${_cap}" ]; then
        _args+=( "${_cap}" )
    fi

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from scripts/tuolumne/t6b_key_demand.flux:224-229 and must
    # not be simplified: a wrong binding does not fail, it oversubscribes one
    # device and returns a plausible number that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t8b] === ${_target} level ${_level} at ${_np} rank(s) / ${_nodes} node(s), cap=${_cap:-default} ==="
    echo "[t8b] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
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
        echo "[t8b] PASS ${_backend} L${_level} np=${_np} ${_tag} in ${_dt}s"
        _pass=$(( _pass + 1 ))
    else
        echo "[t8b] FAIL ${_backend} L${_level} np=${_np} ${_tag}: rc=${_obs} after ${_dt}s" >&2
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}_${_tag}"
        _rc=1
    fi
done

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
echo "[t8b] total wall $(( $(date +%s) - _t0_all ))s over ${_total} launch(es)"
if [ "${_total}" -eq 0 ]; then
    echo "[t8b] FAIL: no launch ran at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t8b] SUMMARY: PASS (${_pass}/${_total} launches)"
    echo "[t8b] A PASS HERE IS PART OF THE RESULT, unlike every earlier"
    echo "[t8b] runner of this probe: T8b adds the SUM IDENTITY as a real"
    echo "[t8b] assertion, so a non-zero rc can mean the breakdown failed to"
    echo "[t8b] account for every fallback pair. Everything else the probe"
    echo "[t8b] prints is still asserted on by nothing."
    echo "[t8b] Read, per launch, 81 rows x ranks plus the two trailers:"
    echo "[t8b]   [t6probe] header   fallback_breakdown_available=1 (a 0 means"
    echo "[t8b]                      this is a ~profiling canopy and NO column"
    echo "[t8b]                      below is a measurement), op_count_cap and"
    echo "[t8b]                      op_cap both 65536."
    echo "[t8b]   [t6probe] row      fb_range_guard + fb_count_cap ==="
    echo "[t8b]                      global_m2l_fallback at every state, and"
    echo "[t8b]                      fb_dropped=0. At cap 65536 fb_count_cap"
    echo "[t8b]                      should be 0 at EVERY state -- nothing is"
    echo "[t8b]                      cap-driven there (T8)."
    echo "[t8b]   [t8bprobe] trailer the series totals: fb_total,"
    echo "[t8b]                      fb_range_guard_total, fb_count_cap_total,"
    echo "[t8b]                      fb_dropped_total and the state counts."
    echo "[t8b]                      fb_states should be 71 of 81 at BOTH rank"
    echo "[t8b]                      counts -- T8 measured that at both caps."
    echo "[t8b] WHICH REASON DOMINATES IS THE FINDING. If it is fb_range_guard,"
    echo "[t8b] the remaining fallback is a representability limit of the"
    echo "[t8b] basis's key encoding rather than a budget, and no cap value"
    echo "[t8b] reaches it. M2L_KEY_DD_MAX and M2L_KEY_OFFSET_MAX are NOT"
    echo "[t8b] changed by this task; that is a new task and a raised risk."
    echo "[t8b] Count the states: 81 per rank per launch, or the series is"
    echo "[t8b] incomplete whatever the rc says."
else
    echo "[t8b] SUMMARY: FAIL (${_pass}/${_total} launches);" \
         "failed:${_names_failed}" >&2
    # Two classes of non-zero rc here, and they are not the same finding.
    # A run that could not proceed -- a throw, a non-finite velocity, an early
    # stop, a round-trip mismatch, or the R5 parameter check finding that the
    # probe and Beatnik_Test_Milestone0Fmm.cpp have drifted apart. A negative
    # argv[2] also lands here, by design, rather than reaching Canopy's throw.
    # OR the SUM IDENTITY failed, which T8b added: either some pair reaches
    # the fallback path for a reason neither counter counts, or fb_dropped is
    # non-zero and pairs are being DROPPED -- a wrong velocity, not a slow
    # one. Read the [FAIL] detail lines; none of them is a number to widen,
    # and the identity in particular is not a tolerance.
fi
exit "${_rc}"
