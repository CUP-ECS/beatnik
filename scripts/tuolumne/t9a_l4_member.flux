#!/bin/bash
# flux: --job-name=beatnik_t9a_l4_member
# flux: --nodes=1
# flux: --exclusive
# flux: -t 55m
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
# T9a's MEASUREMENT (`tasks/add-canopy-t6.md`): Beatnik_Test_Milestone0FmmL4
# -- the LEVEL-4 FMM member, alone -- on HIP, at ONE rank count per
# submission, at FULL step count.
#
# WHY ONE RANK COUNT PER JOB. The member measured 2 450 s at HIP np1 and
# 1 598 s at HIP np4 (T6 cost table) -- 67 minutes together, over `pdebug`'s
# 60-minute cap. So np1 and np4 are two submissions of this script:
#
#   flux batch scripts/tuolumne/t9a_l4_member.flux 1           (-t 55m default)
#   flux batch -t 40m scripts/tuolumne/t9a_l4_member.flux 4
#
# **THIS IS NOT A SHORTENED SUBSTITUTE.** It is the real member at full
# fidelity: 2000 direct steps with 81 same-state FMM evaluations (claim A),
# then 2000 FMM-driven steps (claim B), then the divergence ladder. There is
# no step-count override anywhere in this path, deliberately.
#
# THE LAUNCH FAILS BY CONSTRUCTION, and that is not the result. The member
# asserts `p.m2l_fallback == 0` per claim-A state (`:1500`) and the same on
# claim B's final state (`:1967`); the classify pass's range guard carries
# 100 % of level-4 fallback (T8b) and no configuration removes it, so both
# fail. T9b, not T9a, replaces those checks. What T9a reads is the
# `[t6] CLAIM A SERIES` block and the `CLAIM A: worst relative velocity
# error` note -- the per-state error against `tau_A` -- and it confirms that
# every `[FAIL]` line is at one of :1500, :1505, :1506 or :1967. Any other
# failed check, a walltime kill or a missing claim-A state is a real failure.
#
# Targets the DEVELOPMENT spack env (BEATNIK_USE_PROD is not set): this is
# measurement work, not a large production run.
#
# Usage:  flux batch [-t <limit>] scripts/tuolumne/t9a_l4_member.flux <NP>
#
#   NP  the rank count, 1 or 4. Required: running both in one job does not
#       fit pdebug.
#
# Then read beatnik_t9a_l4_member.<jobid>.log in the submitting directory.
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
    echo "[t9a] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

if [ "$#" -ne 1 ] || { [ "$1" != "1" ] && [ "$1" != "4" ]; }; then
    echo "[t9a] FAIL: exactly one argument, the rank count 1 or 4, is required." >&2
    exit 1
fi
_np="$1"

##--------------------------------------------------------------------------##
## Provenance -- a number in the progress log is only reusable if a later
## session can tell which toolchain produced it.
##--------------------------------------------------------------------------##
_canopy_src="${BEATNIK_REPO}/../canopy"
beatnik_env_summary
echo "[t9a] spack env status:"
spack env status 2>&1 | sed 's/^/[t9a]   /'
echo "[t9a] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t9a] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t9a] canopy branch = $(git -C "${_canopy_src}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "[t9a] canopy commit = $(git -C "${_canopy_src}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t9a] canopy dirty  = $(git -C "${_canopy_src}" status --porcelain --untracked-files=no 2>/dev/null | wc -l) file(s) modified"
echo "[t9a] submit  = flux batch scripts/tuolumne/t9a_l4_member.flux ${_np}"
echo "[t9a] canopy variants (MUST read +profiling):"
spack find --variants canopy 2>&1 | grep -i canopy | sed 's/^/[t9a]   /'
echo "[t9a] beatnik spec (compiler after the %):"
spack find --variants beatnik 2>&1 | grep -i beatnik | sed 's/^/[t9a]   /'
echo "[t9a] compiler version:"
CC --version 2>&1 | head -2 | sed 's/^/[t9a]   /'

_target="Beatnik_Test_Milestone0FmmL4_MPI_HIP"
echo "[t9a] target = ${_target}  np = ${_np}"

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
    echo "[t9a] FAIL: beatnik_milestone_manifest.txt not found on PATH." >&2
    echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing?" >&2
    exit 1
fi
_manifest_dir="$(cd "$(dirname "${_manifest}")" && pwd)"
echo "[t9a] manifest = ${_manifest}"

# The arguments this target takes, read from the manifest rather than retyped:
# a hand-copied gold path that names the wrong level is exactly the mistake the
# registration loop's FATAL_ERROR exists to prevent.
_line="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
         awk -v t="${_target}" '$1 == t' || true)"
if [ -z "${_line}" ]; then
    echo "[t9a] FAIL: the manifest names no ${_target}." >&2
    exit 1
fi
# shellcheck disable=SC2086
set -- ${_line}
shift
_args=( "$@" )
_exe="$(beatnik_exe "${_target}")" || exit 1
echo "[t9a] exe   = ${_exe}"
echo "[t9a] args  = ${_args[*]}"

# MUST be on a PARALLEL filesystem: this member writes TWO 2000-step checkpoint
# series through MPI-IO, and a node-local scratch fails every launch spanning
# more than one node. Per JOB, so two submissions cannot share a series (T5
# lost a launch to exactly that collision). `FLUX_JOB_ID` is not set for a
# batch script's init program, so ask the batch instance itself.
BEATNIK_T9A_SCRATCH_ROOT="${BEATNIK_T9A_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/t9a_l4_member}"
_jobid="${FLUX_JOB_ID:-$(flux getattr jobid 2>/dev/null)}"
_jobid="${_jobid:-pid$$}"
export BEATNIK_TEST_SCRATCH="${BEATNIK_T9A_SCRATCH_ROOT}/job${_jobid}/${_target}_np${_np}"
rm -rf "${BEATNIK_TEST_SCRATCH}"
if ! mkdir -p "${BEATNIK_TEST_SCRATCH}"; then
    echo "[t9a] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
    exit 1
fi
echo "[t9a] scratch = ${BEATNIK_TEST_SCRATCH}"

# Tuolumne packs 4 ranks per node; round the node count up. The binding is
# copied EXACTLY from scripts/tuolumne/t6_l3_member.flux:197-207 and must not
# be simplified: a wrong binding does not fail, it oversubscribes one device
# and returns a plausible number that reads like a real measurement.
_nodes=$(( (_np + 3) / 4 ))
echo "[t9a] === ${_target} at ${_np} rank(s) / ${_nodes} node(s) ==="
echo "[t9a] command: flux run --ntasks=${_np} --nodes=${_nodes}" \
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

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
echo "[t9a] HIP np=${_np}: rc=${_obs} after ${_dt}s"
echo "[t9a] A NON-ZERO rc IS EXPECTED: the :1500 and :1967 zero-fallback"
echo "[t9a] checks fail by construction at level 4 (T8b). Read, in order:"
echo "[t9a]   [t6] CLAIM A SERIES   81 rows; the relative-error column against"
echo "[t9a]                         tau_A, and the p2p_frac column."
echo "[t9a]   [note] CLAIM A: worst relative velocity error ... at step ..."
echo "[t9a]   [FAIL] lines          every 'at:' must be :1500, :1505, :1506 or"
echo "[t9a]                         :1967. Anything else is a real failure."
exit "${_obs}"
