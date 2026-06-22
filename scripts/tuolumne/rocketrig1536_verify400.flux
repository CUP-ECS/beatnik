#!/bin/bash
# flux: --job-name=rk1536_verify400
# flux: --nodes=64
# flux: --exclusive
# flux: --time=60
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
#
# Large single-mode sech2 rollup driven by the FMM BR solver, sized to reach
# FULL rollup in one job:
#   mesh   : 1536x1536 (B=24)
#   ranks  : 256 (64 nodes x 4 GPUs/tasks, 1 GPU/task)
#   deck   : /p/lustre5/stewartj/beatnik/fmm/n1536_p256/single_mode_verify400.in
#
# Sizing (CALIBRATED on this exact geometry, jobs f3Fp373YbYRu/f3Fp37BBsqh1):
#   measured early-regime per-step (50-step calib, ncrit=64) = 3.802 s
#   ncrit=64 beat ncrit=128 (3.802 vs 4.114 s) -> deck uses ncrit=64.
#   full-rollup average applies the 256^2 reference growth factor 2.24x
#   (early 1.303 -> run-avg 2.917): 3.802 * 8400 * 2.24 ~= 19.9 h.
#   --time=1440 (24 h, pbatch max) leaves ~20% margin; the rebalance-heavy
#   tail may be relatively costlier at 256 ranks, so the max wall is used.
# rocketrig has NO restart/checkpoint, so the whole rollup must fit one wall.
#
# Uses the PRODUCTION spack env so dev-env rebuilds cannot disturb a job that
# may sit in the queue for hours. rocketrig takes only the input-file path
# (no CLI solver override), so the deck is copied into fmm/ and its br_solver
# line is forced to 'fmm' there -- the base deck is left untouched.

set -euo pipefail

spack env activate ~/spack_envs/tuolumne_beatnik_production

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export GTL_DREG_CACHE_SIZE=262144   # default 10000 exhausted at 256 ranks (GTL dreg_evict NO_SPACE during Rebalance)
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

BASE_DIR="/p/lustre5/stewartj/beatnik/fmm/n1536_p256"
RUN_DIR="${BASE_DIR}/verify400"
SRC_INPUT="${BASE_DIR}/single_mode_verify400.in"

mkdir -p "${RUN_DIR}"
# Force the BR solver to fmm in this run's copy (only the 'br_solver' line).
sed 's/^br_solver.*/br_solver            = fmm/' "${SRC_INPUT}" \
    > "${RUN_DIR}/single_mode_verify400.in"
cd "${RUN_DIR}"

ROCKETRIG="$(command -v rocketrig)"
INPUT="${RUN_DIR}/single_mode_verify400.in"

echo ":::"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT} (br_solver forced to fmm)"
echo "::: run directory   : ${RUN_DIR}"
echo "::: mesh 1536^2 (B=24), 256 ranks / 64 nodes, 400-step GTL_DREG_CACHE_SIZE verification (must clear step ~307 Rebalance)"
echo ":::"

flux run \
    --ntasks=256 \
    --nodes=64 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"
