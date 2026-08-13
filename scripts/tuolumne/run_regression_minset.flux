#!/bin/bash
# flux: --job-name=beatnik_regression_minset
# flux: --nodes=2
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
# THE SHIP GATE for tuolumne.
#
# Gate definition (single-sourced — must stay identical to CLAUDE.md
# "Minimum test set", the lists in tests/CMakeLists.txt, and
# systems/tuolumne/claude.md "Backends"):
#
#     tier label : regression
#     backends   : SERIAL (project-wide) + HIP (tuolumne-specific)
#     ranks      : 1 2 3 4 5 6
#
# Submit with:   flux batch scripts/tuolumne/run_regression_minset.flux
# Then read the beatnik_regression_minset.<jobid>.log it writes to cwd.
#
# --nodes=2 covers the 6-rank case at tuolumne's 4-ranks-per-node.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source, so the resolver
# and the spack env see the final module state. Tuolumne needs none today.

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
    echo "[gate] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary

##--------------------------------------------------------------------------##
## Gate parameters
##--------------------------------------------------------------------------##
BEATNIK_GATE_LABEL="regression"
BEATNIK_GATE_BACKENDS="${BEATNIK_GATE_BACKENDS:-SERIAL HIP}"
BEATNIK_GATE_RANKS="${BEATNIK_GATE_RANKS:-1 2 3 4 5 6}"

echo "[gate] label=${BEATNIK_GATE_LABEL} backends='${BEATNIK_GATE_BACKENDS}'" \
     "ranks='${BEATNIK_GATE_RANKS}'"

_gate_rc=0

##--------------------------------------------------------------------------##
## manual / tree mode: ctest inside the build directory
##--------------------------------------------------------------------------##
# The harness already registered one ctest case per (backend, rank), so
# `-L <label> -R <backend>` selects the whole rank sweep.
if [ "${BEATNIK_BIN_MODE}" = "tree" ]; then
    if [ ! -d "${BEATNIK_BUILD_DIR}" ]; then
        echo "[gate] FAIL: build dir ${BEATNIK_BUILD_DIR} does not exist." >&2
        exit 1
    fi
    for _backend in ${BEATNIK_GATE_BACKENDS}; do
        echo "[gate] ctest -L ${BEATNIK_GATE_LABEL} -R ${_backend}"
        ( cd "${BEATNIK_BUILD_DIR}" &&
          ctest --output-on-failure --no-tests=ignore \
                -L "${BEATNIK_GATE_LABEL}" -R "${_backend}" ) || _gate_rc=1
    done

##--------------------------------------------------------------------------##
## spack / installed mode: loop the installed gate binaries over the ranks
##--------------------------------------------------------------------------##
# No build tree exists, so there is no ctest to drive. The gate binaries and a
# manifest naming them were installed by `spack install`; walk the manifest and
# launch each one at every required rank count through flux.
else
    # Locate the manifest CMake generated and the package installed. The spack
    # package prepends share/Beatnik/tests to PATH, so scan PATH for it.
    _manifest=""
    _saved_ifs="${IFS}"
    IFS=':'
    for _p in ${PATH}; do
        if [ -f "${_p}/beatnik_gate_manifest.txt" ]; then
            _manifest="${_p}/beatnik_gate_manifest.txt"
            break
        fi
    done
    IFS="${_saved_ifs}"

    if [ -z "${_manifest}" ]; then
        echo "[gate] FAIL: beatnik_gate_manifest.txt not found on PATH." >&2
        echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing" >&2
        echo "  (and Beatnik_INSTALL_TEST_EXECUTABLES=ON)?" >&2
        exit 1
    fi
    _manifest_dir="$(cd "$(dirname "${_manifest}")" && pwd)"
    echo "[gate] manifest = ${_manifest}"

    # The tests run FROM the manifest directory (see below), which lives inside a
    # spack install prefix and is read-only, so a test that writes output needs an
    # explicit writable place. Give it one under the submitting directory --
    # absolute, because the working directory changes below -- so its artifacts
    # land beside the job's .log rather than in a temp directory nobody looks at.
    export BEATNIK_TEST_SCRATCH="${BEATNIK_TEST_SCRATCH:-${PWD}/beatnik_gate_scratch}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
        echo "[gate] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        exit 1
    }
    echo "[gate] scratch = ${BEATNIK_TEST_SCRATCH}"

    # A manifest line is `<target> [args...]`. Field 1 carries the backend
    # suffix and is what selects; the remaining fields are the binary's
    # arguments and any path among them is MANIFEST-RELATIVE, so the whole
    # invocation runs from the manifest directory. Prefixing only some paths is
    # the bug unit_tests.flux already paid for: a missing fixture then makes a
    # negative case pass for the wrong reason. Here the negative case lives
    # inside the test binary, which demands the comparator's exit status be
    # exactly 1 (a detected mismatch) and not 2 (a load error) precisely so a
    # mis-resolved path cannot masquerade as a pass -- but running from the
    # right directory is still what makes that check about the mesh rather than
    # about the paths.
    _ranks_run=0
    for _backend in ${BEATNIK_GATE_BACKENDS}; do
        _lines="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
                  awk -v b="_${_backend}" \
                      'substr($1, length($1) - length(b) + 1) == b' || true)"
        if [ -z "${_lines}" ]; then
            echo "[gate] no ${BEATNIK_GATE_LABEL} binaries for ${_backend}"
            continue
        fi
        # FD 3, NOT STDIN: `flux run` below inherits and CONSUMES the loop's
        # stdin, so with the line list on stdin the first launched binary
        # swallows every remaining line and the gate silently runs only its
        # first member -- reporting PASS while covering less than it claims,
        # which is exactly the "gate silently shrinks" failure CLAUDE.md's
        # minimum-test-set rule exists to prevent. Invisible while the tier had
        # one member per backend; found in unit_tests.flux the moment T2b added
        # a second test, and fixed here at the same time rather than left for
        # T2d's regression test 2 to rediscover.
        while IFS= read -r _line <&3; do
            [ -n "${_line}" ] || continue
            # shellcheck disable=SC2086
            set -- ${_line}
            _target="$1"
            shift
            _exe="$(beatnik_exe "${_target}")" || { _gate_rc=1; continue; }
            for _np in ${BEATNIK_GATE_RANKS}; do
                # Tuolumne packs 4 ranks per node; round the node count up.
                _nodes=$(( (_np + 3) / 4 ))
                echo "[gate] === ${_target} at ${_np} ranks / ${_nodes} node(s) ==="
                ( cd "${_manifest_dir}" && flux run \
                    --ntasks="${_np}" \
                    --nodes="${_nodes}" \
                    --exclusive \
                    --gpus-per-task=1 \
                    --cores-per-task=24 \
                    --setopt=mpibind=verbose:1 \
                    "${_exe}" "$@" ) || _gate_rc=1
                _ranks_run=$(( _ranks_run + 1 ))
            done
        done 3<<EOF
${_lines}
EOF
    done

    # A manifest that named nothing runnable is not a pass. The gate was vacuous
    # by design until T1c landed regression test 1; now that it has a member,
    # zero launches means the manifest, the backend filter or the install is
    # broken, and reporting PASS for that is the failure mode this whole file
    # exists to avoid.
    if [ "${_ranks_run}" -eq 0 ]; then
        echo "[gate] FAIL: the manifest named no runnable ${BEATNIK_GATE_LABEL}" \
             "tests for backends '${BEATNIK_GATE_BACKENDS}'." >&2
        echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing, and do" >&2
        echo "  the manifest's target names carry the expected _<BACKEND> suffix?" >&2
        _gate_rc=1
    fi
fi

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
# The regression tier has ONE member as of T1c (2026-08-12):
# Beatnik_Test_InitialConditions, regression test 1 -- the whole driver path at
# 0 timesteps against the T1a Python gold checkpoint. It is no longer vacuous,
# but it covers only what exists: mesh generation, the initial condition, and
# the checkpoint write. There is still no timestep (T2d) and no adaptivity (T4),
# so a green result here does not say the solver integrates anything.
if [ "${_gate_rc}" -eq 0 ]; then
    echo "[gate] PASS (label=${BEATNIK_GATE_LABEL})"
else
    echo "[gate] FAIL (label=${BEATNIK_GATE_LABEL})" >&2
fi
exit "${_gate_rc}"
