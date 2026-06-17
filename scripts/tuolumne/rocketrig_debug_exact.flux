#!/bin/bash
# flux: --job-name=rocketrig_debug_exact
# flux: --nodes=4
# flux: --exclusive
# flux: --time=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
#
# Overnight debug run: EXACT BR solver on the shared debug deck
#   /p/lustre5/stewartj/beatnik/fmm/debug/single_mode_debug.in
# with output isolated in the exact/ subdir. Pairs with
# rocketrig_debug_fmm.flux (same deck, fmm solver) for a head-to-head
# comparison; both keep the deck's write_frequency so output frames align.
#
# Uses the DEVELOPMENT spack env (iterative debug work, not a production
# job). rocketrig takes only the input-file path (no CLI solver override),
# so the deck is copied into exact/ and its br_solver line is forced to
# 'exact' there -- the base deck is left untouched.

set -euo pipefail

spack env activate ~/spack_envs/tuolumne_beatnik

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

BASE_DIR="/p/lustre5/stewartj/beatnik/fmm/debug"
RUN_DIR="${BASE_DIR}/exact"
SRC_INPUT="${BASE_DIR}/single_mode_debug.in"

mkdir -p "${RUN_DIR}"
# Force the BR solver to exact in this run's copy (only the 'br_solver'
# line; the 'fmm_*' tunable lines are left as-is and simply ignored).
sed 's/^br_solver.*/br_solver            = exact/' "${SRC_INPUT}" \
    > "${RUN_DIR}/single_mode_debug.in"
cd "${RUN_DIR}"

ROCKETRIG="$(command -v rocketrig)"
INPUT="${RUN_DIR}/single_mode_debug.in"

echo ":::"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT} (br_solver forced to exact)"
echo "::: run directory   : ${RUN_DIR}"
echo ":::"

flux run \
    --ntasks=16 \
    --nodes=4 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"
