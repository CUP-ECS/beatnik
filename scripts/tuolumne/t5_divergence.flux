#!/bin/bash
# flux: --job-name=beatnik_t5_div
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
# T5 STEP 7 (tasks/canopy/add-canopy.md): the DIVERGENCE HORIZON of an
# FMM-driven milestone-0 trajectory, and the volume-drift bound claim B needs.
#
# This is M0-D1's measurement with a per-evaluation perturbation as the seed
# instead of a one-ulp initial condition. It rebuilds none of M0-D1's
# machinery: the driver is Beatnik_Test_Milestone0Run with T5's argv[4..6], the
# ladder is tests/regression_tests/milestone0_ladder.py, and the budget-guard
# shape is milestone0_divergence.flux's. Nothing here compares anything -- every
# comparison is offline in Python afterwards, which is why this script has no
# notion of a tolerance.
#
# THE PUBLISHED ENVELOPE IS FMM-DRIVEN AGAINST THE IN-TREE PYTHON GOLD SET,
# because that is the quantity T6 step 3 asserts and an envelope measured
# against anything else does not transfer to it. The direct-driven row is
# reported ALONGSIDE as attribution: it separates the FMM's per-evaluation
# perturbation from the Beatnik-versus-Python drift that is present either way.
# The two ladders nearly coincide and the reason is already measured -- the
# direct path tracks the gold set to 8.5e-13 at step 2000
# (tasks/completed/milestone0-progress-log.md), so at any rung at or above 1e-10 they
# agree.
#
# R8 BINDS HERE: Zoltan2's partition is non-deterministic across runs, so two
# FMM-driven runs of the same deck take different summation orders and diverge
# from each other as well as from the direct path. An envelope measured once is
# tripped by that noise. So the matrix carries TWO INDEPENDENT np=1 FMM RUNS of
# the identical deck (rows `fmmA` and `fmmB`), the envelope is set from the
# EARLIEST observed horizon with margin, and the run-to-run spread is recorded
# separately from the direct-versus-FMM gap.
#
#     matrix (full, 2000 steps, checkpoint every 25):
#       L3 HIP np1 fmm  ncrit 8 p3   -- CONTROL: the far field is NOT live at
#                                       642 vertices under any ncrit at
#                                       theta = 0.3, so this row is an
#                                       FMM-bookkeeping direct sum and its
#                                       horizon should match the direct row's.
#                                       It is the discriminator that says the
#                                       level-4 horizon is the EXPANSION's.
#       L4 HIP np1 direct            -- attribution
#       L4 HIP np1 fmm  ncrit 8 p3   -- run A, the published envelope's
#       L4 HIP np1 fmm  ncrit 8 p3   -- run B, the R8 repeat
#       L4 HIP np4 fmm  ncrit 8 p3   -- rank attribution
#
# ncrit = 8 at level 4 is T4's measured-live pair: 320 occupied leaves against a
# 105-leaf near field, realized P2P fraction 0.2537. The DEFAULT ncrit = 64
# would make every row a direct sum with FMM bookkeeping around it and the
# horizon would read as a success at every order (R1).
#
# BUDGET. The FMM-driven per-step cost is UNMEASURED -- T8 owns it -- so it
# cannot be extrapolated from M0-D1's direct figures (L3 HIP np1 0.005385
# s/step, np4 0.014058; L4 HIP np1 0.008792, np4 0.019211). Run
# BEATNIK_T5_MODE=probe FIRST, at 25 steps, and set the per-row estimates from
# that measurement. pdebug caps at ONE HOUR and a job killed at the wall leaves
# the queue looking exactly like one that passed -- while having written a
# TRUNCATED checkpoint series that milestone0_ladder.py would happily tabulate.
# So each row carries an estimate, rows are ordered cheapest-first, and a row
# whose estimate does not fit the remaining budget is SKIPPED loudly with a
# non-zero exit rather than started and killed halfway. A sweep that does not
# fit is a FINDING FOR THE LOG, not a reason to lengthen the walltime, change
# queue, or quietly run fewer steps.
#
# Submit with:
#   jobid=$(flux batch scripts/tuolumne/t5_divergence.flux)
#   flux job status "$jobid"; echo "status rc=$?"
# Never `flux job attach`: it forwards SIGTERM and cancels the job on two
# SIGINTs, so anything that kills the waiting command kills the job with it.
#
# Then tabulate offline:
#   tests/regression_tests/milestone0_ladder.py pair --run <dir> --ref <gold>
#   tests/regression_tests/milestone0_ladder.py series --dir <dir>
#
# --nodes=1 covers the np=4 row at tuolumne's 4 ranks per node.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root. `flux batch` copies this script into a per-job spool
# directory, so BASH_SOURCE points at /var/tmp/... and cannot find the checkout.
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
    echo "[t5d] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance, BEFORE any work
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[t5d] spack env status:"
spack env status 2>&1 | sed 's/^/[t5d]   /'
echo "[t5d] spack find beatnik:"
( cd "${BEATNIK_REPO}" && spack find --format '{name}{@version}{%compiler}' beatnik 2>&1 ) \
    | sed 's/^/[t5d]   /' || echo "[t5d]   (spack find unavailable)"
CC_BIN="$(command -v CC || command -v amdclang++ || command -v hipcc || true)"
if [ -n "${CC_BIN}" ]; then
    echo "[t5d] ${CC_BIN} --version:"
    "${CC_BIN}" --version 2>&1 | head -3 | sed 's/^/[t5d]   /'
fi
echo "[t5d] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t5d] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t5d] hostname = $(hostname)"
echo "[t5d] date     = $(date -Is)"
echo "=================================================================="

if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[t5d] FAIL: BEATNIK_USE_PROD=1. T5 measures the DEV env, and the" >&2
    echo "  prod env must not be rebuilt under a live job." >&2
    exit 1
fi

##--------------------------------------------------------------------------##
## Parameters
##--------------------------------------------------------------------------##
BEATNIK_T5_MODE="${BEATNIK_T5_MODE:-full}"
BEATNIK_T5_DRIVER="${BEATNIK_T5_DRIVER:-Beatnik_Test_Milestone0Run}"
BEATNIK_T5_CKPT_EVERY="${BEATNIK_T5_CKPT_EVERY:-25}"

# Parent of the per-run output directories. MUST be on a PARALLEL filesystem:
# the checkpoints go through MPI-IO and a node-local scratch fails every launch
# that spans more than one node (CLAUDE.md "Minimum test set").
BEATNIK_T5_SCRATCH_ROOT="${BEATNIK_T5_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t5_divergence}"

# Seconds of the wall this sweep may spend launching. The remainder is slack for
# startup, the final flush and the report below. pdebug caps at 1h; -t is 58m
# (3480 s), so 3300 leaves three minutes of wall past the budget on top of the
# ~640 s the padded estimates already hold back from the measured 2656 s.
BEATNIK_T5_BUDGET="${BEATNIK_T5_BUDGET:-3300}"

# One row per launch:
#   label level backend ranks approx ncrit order steps estimate_seconds
#
# `ncrit` and `order` are passed to the driver only when approx is `fmm`; under
# `direct` they are placeholders the driver ignores. The LABEL is what makes the
# two identical FMM rows distinguishable -- it names the output directory, so
# run A and run B do not alias inside one scratch and get read back as one
# series.
#
# ORDERED CHEAPEST-FIRST, deliberately. If the budget runs out the rows that are
# missing are the expensive ones, the guard names them, and the exit status is
# non-zero -- rather than the sweep dying mid-write on an arbitrary row.
if [ "${BEATNIK_T5_MODE}" = "probe" ]; then
    BEATNIK_T5_STEPS="${BEATNIK_T5_STEPS:-25}"
    # The probe exists to MEASURE the per-step cost the full estimates below are
    # built from. Its own estimates are padded guesses and only have to be large
    # enough not to skip a row: at 25 steps every row is seconds of solve.
    SWEEP="
l3fmm   3 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 300
l4dir   4 HIP 1 direct 0 0 ${BEATNIK_T5_STEPS} 300
l4fmmA  4 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 300
l4fmmB  4 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 300
l4fmm4  4 HIP 4 fmm    8 3 ${BEATNIK_T5_STEPS} 300
"
else
    BEATNIK_T5_STEPS="${BEATNIK_T5_STEPS:-2000}"
    # Estimates are 2000 x the per-step cost the probe measured, padded and
    # rounded up: a low estimate spends the budget, a high one only skips a row
    # early. THE PROBE'S JOB ID AND ITS MEASURED s/step BELONG IN THE LOG ENTRY
    # beside the wall times they predicted, as M0-D1's do.
    #
    # PROBE MEASUREMENT. These estimates are 2000 x the per-step solve cost
    # this same script measured under BEATNIK_T5_MODE=probe, job f3aPSQfNG7C3,
    # 25 steps per row, padded and rounded up -- a low estimate spends the
    # budget and the row killed at the wall is the one that leaves a truncated
    # series behind, while a high one only skips a row early.
    #
    #   l4dir  (L4 HIP np1 direct) 0.009899 s/step
    #   l3fmm  (L3 HIP np1 fmm)    0.068442
    #   l4fmmA (L4 HIP np1 fmm)    0.434082
    #   l4fmmB (L4 HIP np1 fmm)    0.454472
    #   l4fmm4 (L4 HIP np4 fmm)    0.361115
    #
    # THE FMM IS 44x THE DIRECT PER-STEP COST AT LEVEL 4 (0.434 against
    # 0.009899), which is why this sweep needs most of the hour that M0-D1's
    # whole matrix fitted inside. That ratio is a T8 number, not a T5 one --
    # recorded here only because it is what the budget is built from.
    #
    # 2000 steps therefore predicts 20 s, 137 s, 868 s, 909 s and 722 s of
    # solve: 2656 s for the five rows, against a 3300 s budget inside a 58 m
    # wall. The two envelope rows (fmmA, fmmB) run before the rank row, so a
    # budget overrun costs the attribution and not the measurement R8 needs.
    SWEEP="
l4dir   4 HIP 1 direct 0 0 ${BEATNIK_T5_STEPS} 60
l3fmm   3 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 180
l4fmmA  4 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 1000
l4fmmB  4 HIP 1 fmm    8 3 ${BEATNIK_T5_STEPS} 1060
l4fmm4  4 HIP 4 fmm    8 3 ${BEATNIK_T5_STEPS} 880
"
fi

# ROW OVERRIDE. One row per line, same nine fields as the tables above.
#
# ADDED because the measurement showed the five-row sweep does NOT fit pdebug's
# one-hour cap: a 25-step probe under-predicts the FMM per-step cost by about
# 1.9x, because the cost GROWS as the bubble deforms and the tree deepens (the
# L3 row measured 0.128 s/step over 2000 steps against the probe's 0.068). The
# response is to run the rows that do not fit as SEPARATE JOBS, each well inside
# the cap and each running its full 2000 steps -- not to lengthen the walltime,
# change queue, or run fewer steps, all three of which are forbidden and none of
# which this is. The budget guard still applies inside each job.
if [ -n "${BEATNIK_T5_SWEEP:-}" ]; then
    SWEEP="${BEATNIK_T5_SWEEP}"
    echo "[t5d] *** SWEEP OVERRIDDEN -- this job runs a SUBSET of T5 step 7's" \
         "matrix. The full matrix is the union of this job and its siblings;" \
         "say which is which in the log entry. ***"
fi

echo "[t5d] driver=${BEATNIK_T5_DRIVER} mode=${BEATNIK_T5_MODE}" \
     "steps=${BEATNIK_T5_STEPS} checkpoint_every=${BEATNIK_T5_CKPT_EVERY}"
echo "[t5d] scratch_root=${BEATNIK_T5_SCRATCH_ROOT} budget=${BEATNIK_T5_BUDGET}s"
echo "[t5d] sweep:"
printf '%s\n' "${SWEEP}" | sed '/^$/d; s/^/[t5d]   /'

_t_job0="$(date +%s)"
_rc=0
_launched=0
_skipped=0
_dirs=""

##--------------------------------------------------------------------------##
## The sweep
##--------------------------------------------------------------------------##
# FD 3, NOT STDIN: `flux run` inherits and CONSUMES the loop's stdin.
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _label="$1"
    _level="$2"
    _backend="$3"
    _np="$4"
    _approx="$5"
    _ncrit="$6"
    _order="$7"
    _steps="$8"
    _estimate="$9"

    _tag="${_label}_sub${_level}_${_backend}_np${_np}_${_approx}_steps${_steps}"
    _elapsed=$(( $(date +%s) - _t_job0 ))
    _remaining=$(( BEATNIK_T5_BUDGET - _elapsed ))

    if [ "${_remaining}" -lt "${_estimate}" ]; then
        echo "[t5d] === SKIPPED ${_tag}: ${_remaining}s of budget left," \
             "estimate ${_estimate}s. THE SWEEP DOES NOT FIT pdebug's 1h cap." >&2
        echo "  This is a finding for tasks/canopy/add-canopy-progress-log.md," >&2
        echo "  not a reason to run fewer steps or a coarser checkpoint" >&2
        echo "  interval. R8 needs more than one FMM run, so a skipped fmmB" >&2
        echo "  row means the envelope has NO measured spread beside it and" >&2
        echo "  is not usable by T6." >&2
        _skipped=$(( _skipped + 1 ))
        _rc=1
        continue
    fi

    _exe="$(beatnik_exe "${BEATNIK_T5_DRIVER}_MPI_${_backend}")" || {
        echo "[t5d] FAIL: cannot resolve ${BEATNIK_T5_DRIVER}_MPI_${_backend}" >&2
        _rc=1
        continue
    }

    # One output directory PER RUN, on lustre, deleted and recreated immediately
    # before that run -- so a stale checkpoint from an earlier sweep cannot be
    # read back as this run's output and tabulated as a divergence.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_T5_SCRATCH_ROOT}/${_tag}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
        echo "[t5d] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        _rc=1
        continue
    }

    # The driver's argv. Under `direct` the trailing two are omitted entirely
    # rather than passed as zeros, so the run is byte-identical in argument
    # shape to an M0-D1 invocation.
    if [ "${_approx}" = "direct" ]; then
        set -- "${_level}" "${_steps}" "${BEATNIK_T5_CKPT_EVERY}" direct
    else
        set -- "${_level}" "${_steps}" "${BEATNIK_T5_CKPT_EVERY}" \
               "${_approx}" "${_ncrit}" "${_order}"
    fi

    # Tuolumne packs 4 ranks per node; round the node count up. The binding is
    # copied EXACTLY from t4_fmm_vs_direct.flux:138-145 and must not be
    # simplified: a wrong binding does not fail, it oversubscribes one device
    # and returns a plausible wall time that reads like a real measurement.
    _nodes=$(( (_np + 3) / 4 ))
    echo "[t5d] === ${_tag} at ${_np} rank(s) / ${_nodes} node(s) ==="
    echo "[t5d] binding: --ntasks=${_np} --nodes=${_nodes} --exclusive" \
         "--gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1"
    echo "[t5d] scratch = ${BEATNIK_TEST_SCRATCH}"
    echo "[t5d] exe     = ${_exe}"
    echo "[t5d] command: flux run ... ${_exe} $*"

    _t0="$(date +%s)"
    flux run \
        --ntasks="${_np}" \
        --nodes="${_nodes}" \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${_exe}" "$@" || {
            echo "[t5d] FAIL: ${_tag} exited non-zero" >&2
            _rc=1
        }
    _t1="$(date +%s)"
    _launched=$(( _launched + 1 ))

    _wall=$(( _t1 - _t0 ))
    if [ "${_steps}" -gt 0 ]; then
        echo "[t5d] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s" \
             "s_per_step=$(awk -v w="${_wall}" -v s="${_steps}" 'BEGIN{printf "%.6f", w/s}')"
    else
        echo "[t5d] LAUNCH_WALL ${_tag} wall=${_wall}s estimate=${_estimate}s"
    fi
    _written="$(find "${BEATNIK_TEST_SCRATCH}" -name '*_step*.h5' | wc -l)"
    echo "[t5d] checkpoints written: ${_written}"
    # A truncated series is the failure mode this whole script is shaped around,
    # and it is checkable here rather than left for the ladder to tabulate: 2000
    # steps every 25, plus the step-0 file setup() always writes, is 81 files.
    _expect=$(( _steps / BEATNIK_T5_CKPT_EVERY + 1 ))
    if [ "${_written}" -ne "${_expect}" ]; then
        echo "[t5d] FAIL: ${_tag} wrote ${_written} checkpoints, expected" \
             "${_expect}. A TRUNCATED SERIES IS NOT A SHORTER MEASUREMENT --" \
             "do not tabulate it." >&2
        _rc=1
    fi
    _dirs="${_dirs} ${_tag}"
done 3<<EOF
${SWEEP}
EOF

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( $(date +%s) - _t_job0 ))
echo "[t5d] launched=${_launched} skipped=${_skipped} total=${_total}s" \
     "budget=${BEATNIK_T5_BUDGET}s"
echo "[t5d] run directories under ${BEATNIK_T5_SCRATCH_ROOT}:${_dirs}"
if [ "${_launched}" -eq 0 ]; then
    echo "[t5d] FAIL: the sweep launched nothing." >&2
    _rc=1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[t5d] PASS: every row completed its step budget and wrote a full series."
    echo "[t5d] Next, offline:"
    echo "  milestone0_ladder.py pair --run <run>/sub4_HIP_np1_fmm_ncrit8_p3 \\"
    echo "      --ref tests/regression_tests/milestone0-sub4-2000-steps/gold"
    echo "  milestone0_ladder.py series --dir <run>/..."
else
    echo "[t5d] FAIL: see the messages above. A partial sweep is NOT a" >&2
    echo "  measurement -- do not tabulate one without saying so." >&2
fi
exit "${_rc}"
