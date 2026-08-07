#!/bin/bash
# flux: --job-name=beatnik_v0_smoke
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
# V0 smoke test (tasks/framework.md, task V0 steps 3-4).
#
# Proves the framework binary is *runnable*, not that it is correct:
#
#   step 3: `--help` exits 0 and prints the option schema.
#   step 4: a real invocation parses its arguments, echoes the resolved
#           configuration, and dies inside a documented stub -- i.e. a
#           std::logic_error from BEATNIK_NOT_IMPLEMENTED, NOT an
#           argument-parsing error and NOT a segfault.
#
# The step-4 command line is the SAME one that generated the T1a gold file
# (tests/regression_tests/initial_conditions/README.md), so when the stubs are
# filled in this script becomes the direct precursor to regression test 1.
#
# Submit with:   flux batch scripts/tuolumne/run_v0_smoke.flux
# Then read the beatnik_v0_smoke.<jobid>.log it writes to cwd.
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
    echo "[v0] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary

EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1
echo "[v0] exe = ${EXE}"

##--------------------------------------------------------------------------##
## Step 3 -- --help must exit 0
##--------------------------------------------------------------------------##
# Run on the batch node rather than at submit time: --help still calls
# Kokkos::initialize, which needs a GPU in a HIP build.
echo "[v0] === step 3: --help ==="
flux run --ntasks=1 --nodes=1 --exclusive --gpus-per-task=1 --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${EXE}" --help
_help_rc=$?
echo "[v0] --help rc=${_help_rc} (want 0)"

##--------------------------------------------------------------------------##
## Step 4 -- a real invocation reaches a stub
##--------------------------------------------------------------------------##
# The T1a gold-file command line. --source-quadrature vertex is mandatory: the
# Python default is `face` and only `vertex` is ported (see risk R11).
echo "[v0] === step 4: real invocation ==="
flux run --ntasks=1 --nodes=1 --exclusive --gpus-per-task=1 --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${EXE}" \
        --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
        --viscosity-mode laplace-beltrami \
        --br-approximation direct --isotropic-cleanup \
        --checkpoint-every-steps 1 --no-video \
        --steps 0 --source-quadrature vertex
_run_rc=$?
echo "[v0] real invocation rc=${_run_rc}"
echo "[v0] Expected at this stage: a NONZERO rc whose log above shows the echoed"
echo "[v0] configuration followed by a std::logic_error naming a stub. A parse"
echo "[v0] error, or a signal (rc>=128), is a FAILURE of step 4."

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
if [ "${_help_rc}" -eq 0 ]; then
    echo "[v0] step 3 PASS"
else
    echo "[v0] step 3 FAIL (--help rc=${_help_rc})" >&2
fi
# Step 4's verdict needs a human/agent reading the message above -- the script
# only flags the two mechanically-detectable failures.
if [ "${_run_rc}" -eq 0 ]; then
    echo "[v0] step 4 UNEXPECTED PASS: the run exited 0, so it never hit a stub." >&2
elif [ "${_run_rc}" -ge 128 ]; then
    echo "[v0] step 4 FAIL: died on signal $(( _run_rc - 128 )), not a logic_error." >&2
else
    echo "[v0] step 4: inspect the log for the stub name (see note above)."
fi

exit "${_help_rc}"
