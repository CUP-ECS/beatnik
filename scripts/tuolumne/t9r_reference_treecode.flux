#!/bin/bash
# flux: --job-name=beatnik_t9r_reference_treecode
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
# T9r's MEASUREMENT (`tasks/add-canopy-t6.md`): the reference treecode's own
# claim-A error -- `zmodel3d`'s treecode (order 2, theta 0.3, ncrit 64)
# against its own direct sum -- on the 81 states of the level-4 member's gold
# set, through tests/regression_tests/reference_treecode_error.py.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. Python runs directly in this script's
# body, which already executes on the allocated node; there is no `flux run`
# and so no rank-to-GPU binding. The resolver is still sourced first, as every
# batch script here must, and targets the DEVELOPMENT env (BEATNIK_USE_PROD is
# not set): this is measurement work.
#
# The reference repository is read, never edited. Its `zmodel3d/` package
# must be clean; this script refuses to measure otherwise.
#
# Usage:  flux batch scripts/tuolumne/t9r_reference_treecode.flux
#   Override the reference location with ZMODEL3D_REPO=<path>.
# Then read beatnik_t9r_reference_treecode.<jobid>.log in the submitting
# directory: 81 `[t9r] row` lines and one `[t9r] worst` line.
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
    echo "[t9r] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_ref="${ZMODEL3D_REPO:-${HOME}/research-bridges/zmodel-steve/zmodel3d-amr}"
_python=/usr/tce/bin/python3

##--------------------------------------------------------------------------##
## Provenance -- a number in the progress log is only reusable if a later
## session can tell which code produced it.
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[t9r] spack env status:"
spack env status 2>&1 | sed 's/^/[t9r]   /'
echo "[t9r] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[t9r] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[t9r] reference repo   = ${_ref}"
echo "[t9r] reference commit = $(git -C "${_ref}" rev-parse HEAD 2>/dev/null || echo unknown)"
_ref_dirty="$(git -C "${_ref}" status --porcelain -- zmodel3d 2>&1)"
echo "[t9r] reference zmodel3d/ status = '${_ref_dirty}'"
echo "[t9r] python  = ${_python} ($(${_python} --version 2>&1))"
echo "[t9r] host    = $(hostname)"
echo "[t9r] submit  = flux batch scripts/tuolumne/t9r_reference_treecode.flux"
if [ -n "${_ref_dirty}" ]; then
    echo "[t9r] FAIL: the reference's zmodel3d/ package is not clean." >&2
    exit 1
fi

##--------------------------------------------------------------------------##
## Measure
##--------------------------------------------------------------------------##
_t0=$(date +%s)
PYTHONPATH="${_ref}" "${_python}" \
    "${BEATNIK_REPO}/tests/regression_tests/reference_treecode_error.py"
_rc=$?
_dt=$(( $(date +%s) - _t0 ))
echo "[t9r] rc=${_rc} after ${_dt}s"
exit "${_rc}"
