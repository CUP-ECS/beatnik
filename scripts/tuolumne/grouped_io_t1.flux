#!/bin/bash
# flux: --job-name=grouped_io_t1
# flux: --nodes=1
# flux: --exclusive
# flux: -t 5m
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
# T1 exit-criterion run for tasks/grouped-io.md: four steps of
# adaptive_mesh_bubble checkpointing every two steps, so the run takes
# checkpoints at steps 0, 2 and 4 plus finalize()'s repeat of the last one --
# the shape that makes MeshSeries' strictly-increasing-time rule fire if the
# equal-time branch in CheckpointIO::write is missing.
#
# The four remesh flags are not a weakened run and not optional: the example
# defaults to dynamic remeshing with collapse, flips, smoothing and isotropic
# cleanup all on, and all four are task T4d and still throw, so without them the
# solver aborts before the first step and only the startup checkpoint is ever
# written. This is the SAME split-only configuration gate member T4b uses
# (tests/regression_tests/Beatnik_Test_DynamicRemeshSplit.cpp:31-34), reached
# through the reference's own knobs rather than a new switch.
# `--source-quadrature vertex --br-approximation direct` are required for the
# same reason: the example defaults to `face` quadrature and `fmm`, and both
# `FaceQuadrature::generate` and `BRSolverFMM::computeInterfaceVelocity` are
# unimplemented, so the defaults abort on the first step. Together the six flags
# are exactly T4b's configuration
# (Beatnik_Test_DynamicRemeshSplit.cpp:179-186). None of them touches the I/O
# path under test.
#
# Targets the DEVELOPMENT env (BEATNIK_USE_PROD left commented out).
# The checkpoint directory is on lustre: the checkpoints go through MPI-IO, so
# a node-local path fails any multi-node launch.
#
#     flux batch scripts/tuolumne/grouped_io_t1.flux
############################################################################

set -u

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
    echo "[grouped_io_t1] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# Uncomment for large/long jobs to pin the production environment:
# export BEATNIK_USE_PROD=1

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary

OUT_DIR="/p/lustre5/stewartj/beatnik/grouped_io/grouped_io_t1"

echo "[provenance] commit    = $(cd "${BEATNIK_REPO}" && git rev-parse HEAD)"
echo "[provenance] submitted = flux batch scripts/tuolumne/grouped_io_t1.flux"
echo "[provenance] out dir   = ${OUT_DIR}"

# Start from an empty directory, or a leftover master/.xmfindex from an earlier
# run makes the frame and <Time Value=> counts unreadable.
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}" || exit 1

EXE="$(beatnik_exe adaptive_mesh_bubble)" || exit 1

# NOTE: --ntasks must equal --nodes * 4 (tuolumne runs 4 ranks per node).
# Update both together, here and in the flux header above.
flux run \
    --ntasks=4 \
    --nodes=1 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${EXE}" --steps 4 --checkpoint-every-steps 2 --checkpoint-dir "${OUT_DIR}" \
        --source-quadrature vertex --br-approximation direct \
        --remesh-collapse-factor 0 --remesh-smooth-iters 0 \
        --remesh-flip-min-gain 1e12 --no-isotropic-cleanup
run_rc=$?
echo "[result] adaptive_mesh_bubble rc=${run_rc}"

echo "[result] directory listing:"
ls -l "${OUT_DIR}"

# The exit criterion, measured rather than eyeballed: exactly one temporal
# collection, and one <Time Value=> per DISTINCT frame stem -- the duplicate
# final frame must appear once, not twice.
MASTER="${OUT_DIR}/checkpoint.xmf"
if [ ! -f "${MASTER}" ]; then
    echo "[result] FAIL: no master ${MASTER}"
    exit 1
fi
collections=$(grep -c 'CollectionType="Temporal"' "${MASTER}")
times=$(grep -c '<Time Value=' "${MASTER}")
frames=$(ls "${OUT_DIR}" | grep -c '^checkpoint_t.*\.h5$')
echo "[result] CollectionType=\"Temporal\" count = ${collections} (expect 1)"
echo "[result] <Time Value= count            = ${times}"
echo "[result] distinct frame .h5 count      = ${frames}"
echo "[result] master text:"
cat "${MASTER}"
echo "[result] xmfindex:"
cat "${OUT_DIR}/checkpoint.xmfindex"

if [ "${run_rc}" -ne 0 ] || [ "${collections}" -ne 1 ] || \
   [ "${times}" -ne "${frames}" ]; then
    echo "[result] FAIL"
    exit 1
fi
echo "[result] PASS"
