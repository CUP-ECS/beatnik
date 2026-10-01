#!/bin/bash
# flux: --job-name=beatnik_t6b_key_demand
# flux: --nodes=1
# flux: --exclusive
# flux: -t 60m
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
# T5's MEASUREMENT RUN (`tasks/add-canopy-t6.md`): Beatnik_Probe_FmmKeyDemand
# at LEVEL 4 on HIP at ranks 1 and 4, with a LEVEL 3 np1 control beside it.
#
# WHAT IT IS FOR. T5 measures the level-4 M2L operator-key demand series, so
# T8 can size the count cap from a number instead of a guess. T3 and T4 ran
# this same script at level 3 only, to show that the probe RUNS -- that both
# per-backend targets resolve, and that the rank-local fields are printed once
# per rank rather than reduced. That is now established, and the level-4 matrix
# below is the measurement itself.
#
# ONE SUBMISSION IS NOT THE NUMBER. The level-3 control was measured twice and
# the two runs disagree in 369 of 405 rows of `unique_ops`, with np1 peak
# demand 6 198 and 5 938 -- about a 4 % spread (`## T4` in the progress log).
# The peak is a draw from a distribution, so T5 submits this script THREE
# separate times and reports the WORST OBSERVED peak per (rank count, rank)
# with the run-to-run spread stated. Three independent allocations also
# separate a run-to-run effect from an allocation-fixed one. Never a mean, and
# never a single draw.
#
# WHAT THE LEVEL-3 LAUNCH IS FOR NOW. It is a REPRODUCIBILITY CHECK against a
# measured band -- np1 peak demand 6 198 at step 1600 and 5 938 at step 1575 --
# not a test of whether level-3 demand is under the cap, which is already
# measured (zero fallback in all 405 rows of both runs, `demand_saturated`
# never set). **T5 WIDENED THAT BAND BY MEASURING IT THREE MORE TIMES**: the
# five draws now on record span 5 586 to 6 624 -- about +/-8.5 % around 6 100,
# not 4 % -- and the peak's STEP moves from 400 to 1925 across them, so the
# original two-point 5 938-6 198 range was an under-sampled min/max and not a
# bound. Compare the SUBSTANTIVE columns instead, which are stable to the point
# of being identical: zero fallback and zero over-cap rows at every state, a
# demand minimum of 828 in every draw, `demand == unique_ops` in 81 of 81, and
# `occupied_depths` in 4-6. One of those moving means the measurement apparatus
# moved; a peak outside 5 586-6 624 alone does not. Level 3 declares
# `kFarFieldIsLive = false` and cannot exercise the overflow path at all, so it
# proves NOTHING about level 4. Note also that `demand == unique_ops` at
# level 3 is an artefact of the cap not binding there: a level-4
# `demand > realized` is the EXPECTED reading, not a regression against the
# control.
#
# THE `-1` OBSERVATION HAS BEEN TAKEN AND IS RECORDED. T3 ran this script
# against the `canopy ~profiling` the env concretized then (job `f3bQSGSss6RD`)
# and read the `-1` SENTINEL in 405 of 405 rows, with the loud
# `*** DEMAND UNAVAILABLE ***` header line -- see `## T3` in
# `tasks/add-canopy-t6-progress-log.md`. T4 then added `+profiling` to the
# canopy spec, so the sentinel is no longer reachable in this environment and
# nothing here needs to preserve it.
#
# WHAT THIS SCRIPT SHOWS NOW. Under `canopy +profiling` the header reports
# `demand_available=1` and the demand column carries a REAL non-negative count,
# beside the two ungated depth columns that were live all along. A probe that
# reported `-1` for the depth columns would still be reporting its own bug
# (risk R7), and a `-1` in the demand column now would mean the binary that ran
# was not built against the `+profiling` canopy -- check the `spack find
# --variants canopy` line this script echoes before trusting any row.
#
# WHY np1 AND np4. np4 is the only launch that exercises the per-rank unreduced
# printing at all: at np1 "per rank" and "reduced" are the same four lines. The
# operator-key cap is PER RANK, so np4 carries four times the key budget and
# overflows LESS than np1, not four times as much (T0 measured 13 274 fallback
# pairs at np1 against 5 300 at np4, same step, same level). Never average the
# rank-local columns across ranks.
#
# WHY HIP, AND WHY THIS FITS `pdebug`. Cost. The probe drives claim A's
# trajectory and drops the direct comparator and the per-state Python
# comparator subprocess, so it costs well under claim A itself: level-4 claim A
# measured 84.543 s at HIP np1 and 100.154 s at HIP np4, and level-3 claim A's
# 166 s came back as a 24 s probe launch. What is hours rather than minutes is
# claim B -- the 2000-step FMM-DRIVEN trajectory, 2 373 s at level 4 HIP np1 --
# and the probe runs no part of it. Three launches therefore sit far inside
# `pdebug`'s 1 h cap; `-t 60m` is the cap itself and also covers the optional
# SERIAL level-4 pair (SERIAL np1 1432 s, np4 523 s) if the HIP result comes
# back ambiguous and that pair has to be added. The `_MPI_SERIAL` target must
# build and `beatnik_exe` must resolve it, but it is NOT launched here and
# nothing is claimed for it.
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

# THE MATRIX, as `<level>:<ranks>`. Level 4 at np1 and np4 is the measurement;
# level 3 at np1 is the control, and it runs FIRST so that an apparatus problem
# shows up against its known band before the expensive pair spends any time.
# The optional SERIAL level-4 pair (Do step 4) is added here only if the HIP
# result is ambiguous; it is not here now, and adding it must not raise `-t`
# past the 60m already set -- split the submission instead.
_matrix=( "3:1" "4:1" "4:4" )
echo "[t6b] target = ${_target_stem}_MPI_${_backend}  matrix = ${_matrix[*]}  (level:ranks)"

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
# Per-job subdirectory, so two submissions of this script cannot collide in the
# filesystem. `FLUX_JOB_ID` is NOT set for a batch script -- flux sets it for
# the tasks the shell launches, not for the batch instance's init program -- so
# ask the batch instance itself, and fall back to the PID for a direct `bash`
# run, which has no job id at all. The PID is enough for collision safety; the
# jobid is what makes the directory traceable back to a log.
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
echo "[t6b] scratch root = ${BEATNIK_T6B_SCRATCH_ROOT}"
echo "[t6b] scratch job  = job${_jobid}"

_rc=0
_pass=0
_fail=0
_names_failed=""
_t0_all=$(date +%s)

for _entry in "${_matrix[@]}"; do
    _level="${_entry%%:*}"
    _np="${_entry##*:}"

    # One I/O directory PER (JOB, target, level, rank), on lustre, deleted and
    # recreated immediately before the launch so a stale checkpoint from an
    # earlier run cannot be read back as this run's output.
    #
    # THE JOB ID IN THE PATH IS LOAD-BEARING, not tidiness. T5 submits this
    # script three times, and two submissions that overlap in the queue would
    # otherwise write the SAME checkpoint files: that is exactly how T5's first
    # attempt lost a launch, with
    # `H5FD__sec2_lock(): unable to lock file, errno = 11` at step 1325 and a
    # `collective tags 14 and 1 do not match` MPI abort behind it. A collision
    # need not abort to be harmful -- it could also round-trip another job's
    # particle counts -- so the path is per job whether the submissions overlap
    # or not.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T6B_SCRATCH_ROOT}/job${_jobid}/${_target}_L${_level}_np${_np}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
        echo "[t6b] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        _fail=$(( _fail + 1 ))
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}(scratch)"
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
        _names_failed="${_names_failed} ${_backend}_L${_level}_np${_np}"
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
    echo "[t6b] per-rank rows. Under this env's +profiling canopy the demand"
    echo "[t6b] column is a REAL count (0 is a legal measurement); a -1 there"
    echo "[t6b] means the binary was not built against +profiling, not zero"
    echo "[t6b] demand. occupied_depths and cells_at_max_depth are ungated and"
    echo "[t6b] were live before +profiling as well."
    echo "[t6b] Then read the [t6probe] trailer lines, ONE PER RANK: they carry"
    echo "[t6b] first_exceed_step, peak_demand and its step, and the peak"
    echo "[t6b] keys_built_delta and its step, already derived from the demand"
    echo "[t6b] and op_cap columns. Do NOT re-derive them from the rows, and do"
    echo "[t6b] NOT average a rank-local column across ranks -- the cap is per"
    echo "[t6b] rank and a mean describes a tree no rank has. Count the states:"
    echo "[t6b] 81 per rank, or the series is incomplete whatever the rc says."
    echo "[t6b] If demand_saturated=1 appears in ANY row, the peak is not a"
    echo "[t6b] measurement: it is a lower bound of 1048576 and must be"
    echo "[t6b] reported as one. The [Canopy] cap warning is NOT an overflow"
    echo "[t6b] test -- it was absent from both np4 launches of the failing"
    echo "[t6b] tier run despite non-zero fallback at 71 states. Use"
    echo "[t6b] global_m2l_fallback and demand against op_cap."
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
