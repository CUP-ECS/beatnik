#!/bin/bash
# flux: --job-name=beatnik_regression_minset
# flux: --nodes=2
# flux: --exclusive
# flux: --time=30
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

# Pin the repo root: a scheduler may launch this from a spool copy, so
# BASH_SOURCE is not a reliable way back to the checkout.
export BEATNIK_REPO="${BEATNIK_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

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
    echo "[gate] manifest = ${_manifest}"

    for _backend in ${BEATNIK_GATE_BACKENDS}; do
        # Manifest lines are target names, one per line, e.g.
        # Beatnik_Test_Foo_MPI_SERIAL. Comments and blanks are skipped.
        _targets="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
                    grep -- "_${_backend}\$" || true)"
        if [ -z "${_targets}" ]; then
            echo "[gate] no ${BEATNIK_GATE_LABEL} binaries for ${_backend}"
            continue
        fi
        for _target in ${_targets}; do
            _exe="$(beatnik_exe "${_target}")" || { _gate_rc=1; continue; }
            for _np in ${BEATNIK_GATE_RANKS}; do
                # Tuolumne packs 4 ranks per node; round the node count up.
                _nodes=$(( (_np + 3) / 4 ))
                echo "[gate] ${_target} at ${_np} ranks / ${_nodes} node(s)"
                flux run \
                    --ntasks="${_np}" \
                    --nodes="${_nodes}" \
                    --exclusive \
                    --gpus-per-task=1 \
                    --cores-per-task=24 \
                    --setopt=mpibind=verbose:1 \
                    "${_exe}" || _gate_rc=1
            done
        done
    done
fi

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
# NOTE: the regression tier is currently EMPTY — the pre-redesign solver and its
# only test were removed in 89ec015, and the new solver has not landed tests
# yet. Until it does this gate is structurally correct but vacuous, and a green
# result here proves nothing. See README "Known Issues".
if [ "${_gate_rc}" -eq 0 ]; then
    echo "[gate] PASS (label=${BEATNIK_GATE_LABEL})"
else
    echo "[gate] FAIL (label=${BEATNIK_GATE_LABEL})" >&2
fi
exit "${_gate_rc}"
