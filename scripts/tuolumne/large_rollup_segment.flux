#!/bin/bash
# flux: --job-name=large_rollup_segment
# flux: --nodes=1
# flux: --exclusive
# flux: -t 240m
# flux: --output=/p/lustre5/stewartj/beatnik/large_rollup/V1/{{name}}.{{jobid}}.log
# flux: -q pbatch
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
# large_rollup segment job (`tasks/large_rollup.md` V1j, V1c, R1j): one
# segment of a long adaptive_mesh_bubble run, launched from a STAGED
# directory. A long run is a chain of these jobs, each restarting from the
# previous segment's latest checkpoint in the same output directory.
#
# Usage (a copy of this file in the stage directory is what gets submitted;
# segment N>1 adds `# flux: --dependency=afterok:<previous id>` to its copy):
#   "$TASKLOOP_SUBMIT" --purpose "..." <stage>/segment1.flux -- \
#       <stage>/args.txt <out> <t_end> seg1
#   "$TASKLOOP_SUBMIT" --purpose "..." <stage>/segment2.flux -- \
#       <stage>/args.txt <out> <t_end> seg2 <out>/checkpoint_latest.h5
#
# Arguments:
#   $1  arguments file: the base solver arguments, whitespace separated. Its
#       directory is the stage, which holds
#         bin/adaptive_mesh_bubble  copied from the production view
#         COMMIT                    the commits the stage was built from
#   $2  output directory; the run's --checkpoint-dir
#   $3  this segment's --t-end
#   $4  tag (seg1, seg2, ...): names the launch log and the summary line
#   $5  optional --restart-from path; absent means a cold start
#
# The script appends `--checkpoint-dir <out> --t-end <t_end>` and, when $5
# is given, `--restart-from <$5>` to the arguments file's contents.
#
# Environment: the PRODUCTION spack env's runtime libraries through the
# resolver (BEATNIK_USE_PROD=1). The binary is the staged copy, never
# `beatnik_exe`, and the job echoes <stage>/COMMIT, never `git rev-parse`:
# the checkout and the env may move between submission and start.
#
# Output, all under <out>:
#   <tag>.launch.log  the launch's output, every line prefixed `date +%s.%N`
#   rss.log           `<epoch> <pid> <rss_kb> ...` every 60 s from the first
#                     node, appended across segments
# and in the job log, one summary line per launch:
#   [large_rollup] <tag> rc=<rc> wall=<s>
# The `--output` directive names the V1 directory; a copy staged for another
# task edits that line to its own directory, which must exist before submit.
#
# How a segment ends. Before launching, the script reads the job's remaining
# time with `flux job timeleft` and caps the launch 10 minutes under it:
# `flux run -t <cap>s` with cap = timeleft - 600 s, about 230 minutes in a
# -t 240m job. A launch wall within 60 s of the cap counts as a cap kill. If
# `flux job timeleft` fails or the cap is not a positive integer, the segment
# prints `[large_rollup] FAIL:` and its summary line and exits 1 without
# launching. The binary writes a final checkpoint when it reaches --t-end or
# stops on a non-finite state; a launch killed by its cap writes none, so the
# newest checkpoint is the last one written on the arguments file's
# checkpoint cadence. Exit status:
#   0        the launch exited 0 (reached --t-end), or the cap killed it and
#            <out>/checkpoint_latest.h5 exists for the next segment;
#   non-zero no launch cap, a solver stop (a `nonfinite` stop line), a
#            missing restart file, any other failure, or no checkpoint to
#            continue from. This breaks an afterok chain on purpose.
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
export BEATNIK_USE_PROD=1
export BEATNIK_REPO="${BEATNIK_REPO:-$(beatnik_find_repo)}"
if [ -z "${BEATNIK_REPO}" ]; then
    echo "[large_rollup] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "[large_rollup] FAIL: usage: <args file> <out> <t_end> <tag> [restart path]" >&2
    exit 2
fi
ARGS_FILE="$1"
OUT="$2"
T_END="$3"
TAG="$4"
RESTART="${5:-}"
STAGE="$(cd "$(dirname "${ARGS_FILE}")" && pwd)" || exit 1
EXE="${STAGE}/bin/adaptive_mesh_bubble"
LAUNCH_LOG="${OUT}/${TAG}.launch.log"

for _f in "${ARGS_FILE}" "${EXE}" "${STAGE}/COMMIT"; do
    if [ ! -f "${_f}" ]; then
        echo "[large_rollup] FAIL: staged file missing: ${_f}" >&2
        exit 1
    fi
done
mkdir -p "${OUT}" || exit 1

# --- provenance ------------------------------------------------------------
beatnik_env_summary
echo "[large_rollup] exe       = ${EXE}"
echo "[large_rollup] spack     ="
spack find -lv beatnik 2>&1 | sed 's/^/[large_rollup]   /'
echo "[large_rollup] compiler  = $(hipcc --version 2>/dev/null | head -1 || echo unknown)"
echo "[large_rollup] COMMIT    ="
sed 's/^/[large_rollup]   /' "${STAGE}/COMMIT"
echo "[large_rollup] jobid     = $(flux getattr jobid 2>/dev/null || echo unknown)  host = $(hostname)"
echo "[large_rollup] submit    = \"\$TASKLOOP_SUBMIT\" --purpose \"...\" ${STAGE}/<segment file> -- $*"

# Launch cap: the job's remaining time less 600 s. A launch wall at or past
# CAP_DETECT (the cap less 60 s) counts as a cap kill.
cap_fail() {
    echo "[large_rollup] FAIL: $1" >&2
    echo "[large_rollup] ${TAG} rc=1 wall=0.0"
    exit 1
}
_left="$(flux job timeleft 2>&1)" || cap_fail "flux job timeleft failed: ${_left}"
_left="${_left%%.*}"
[[ "${_left}" =~ ^[0-9]+$ ]] || cap_fail "flux job timeleft gave a non-integer: '${_left}'"
CAP=$(( _left - 600 ))
[ "${CAP}" -gt 0 ] || cap_fail "launch cap ${CAP} s is not positive (timeleft ${_left} s)"
CAP_DETECT=$(( CAP - 60 ))
echo "[large_rollup] cap       = ${CAP}s (flux job timeleft ${_left}s less 600 s), cap kill at wall >= ${CAP_DETECT}s"

read -r -a BASE <<< "$(tr '\n' ' ' < "${ARGS_FILE}")"
SEG=( --checkpoint-dir "${OUT}" --t-end "${T_END}" )
if [ -n "${RESTART}" ]; then
    if [ ! -e "${RESTART}" ]; then
        echo "[large_rollup] FAIL: restart file missing: ${RESTART}" >&2
        echo "[large_rollup] ${TAG} rc=1 wall=0.0"
        exit 1
    fi
    SEG+=( --restart-from "${RESTART}" )
fi

# --- RSS sampler (the batch script runs on the job's first node) -----------
(
    while true; do
        echo "$(date +%s) $(ps -C adaptive_mesh_bubble -o pid=,rss= | tr '\n' ' ')" >> "${OUT}/rss.log"
        sleep 60
    done
) &
RSS_PID=$!

# --- launch ----------------------------------------------------------------
echo "[large_rollup] === ${TAG}: np=4 nodes=1 ${BASE[*]} ${SEG[*]}"
_t0=$(date +%s.%N)
echo "${_t0} [large_rollup] ${TAG} start=${_t0} np=4 nodes=1 args: ${BASE[*]} ${SEG[*]}" > "${LAUNCH_LOG}"
flux run -t "${CAP}s" \
    --ntasks=4 \
    --nodes=1 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    /usr/bin/time -f "rss_kb=%M" \
    "${EXE}" "${BASE[@]}" "${SEG[@]}" 2>&1 \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done \
    >> "${LAUNCH_LOG}"
_rc=${PIPESTATUS[0]}
_wall=$(awk -v a="${_t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')

kill "${RSS_PID}" 2>/dev/null
wait "${RSS_PID}" 2>/dev/null

# Echo the launch's last progress, stop and rss lines into the job log too.
grep -E 'step=|nonfinite|rss_kb=|[Ss]top|rror|timeout|expired' "${LAUNCH_LOG}" | tail -n 40
echo "[large_rollup] ${TAG} rc=${_rc} wall=${_wall}"

# --- exit status for the afterok chain -------------------------------------
if [ "${_rc}" -eq 0 ]; then
    echo "[large_rollup] ${TAG} end=t_end"
    exit 0
fi
if grep -q 'nonfinite' "${LAUNCH_LOG}"; then
    echo "[large_rollup] ${TAG} end=solver_stop"
    exit 1
fi
# A cap kill: Flux's timeout exception in the log, or a wall within a minute
# of the launch cap (CAP_DETECT), whichever text Flux printed.
_capped=0
if grep -qE 'type=timeout|expired' "${LAUNCH_LOG}" \
   || awk -v w="${_wall}" -v c="${CAP_DETECT}" 'BEGIN { exit !(w >= c) }'; then
    _capped=1
fi
if [ "${_capped}" -eq 1 ] && [ -e "${OUT}/checkpoint_latest.h5" ]; then
    echo "[large_rollup] ${TAG} end=cap checkpoint=$(readlink "${OUT}/checkpoint_latest.h5")"
    exit 0
fi
echo "[large_rollup] ${TAG} end=failure (no cap kill with a checkpoint to continue from)"
exit "${_rc}"
