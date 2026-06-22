#!/bin/bash
# flux: --job-name=rk1536_calib_n128
# flux: --nodes=64
# flux: --exclusive
# flux: --time=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
#
# CALIBRATION run for the 1536^2 (B=24) full-rollup job: 50 steps at the exact
# production geometry (256 ranks / 64 nodes) to measure the early-regime
# per-step wallclock. fmm_ncrit = 128 (GPU-throughput experiment). Pairs with the ncrit=128
# variant for a head-to-head perf comparison. Short --time => good backfill.
# pdebug caps at 16 nodes, so this must run on pbatch.
#
# Uses the PRODUCTION spack env (same binary the big run will use).

set -euo pipefail

spack env activate ~/spack_envs/tuolumne_beatnik_production

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

BASE_DIR="/p/lustre5/stewartj/beatnik/fmm/n1536_p256"
RUN_DIR="${BASE_DIR}/calib_ncrit128"
SRC_INPUT="${BASE_DIR}/single_mode_calib_ncrit128.in"

mkdir -p "${RUN_DIR}"
cp "${SRC_INPUT}" "${RUN_DIR}/single_mode.in"
cd "${RUN_DIR}"

ROCKETRIG="$(command -v rocketrig)"
INPUT="${RUN_DIR}/single_mode.in"

echo ":::"
echo "::: CALIBRATION fmm_ncrit=128  | 1536^2 B=24 | 256 ranks / 64 nodes | 50 steps"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT}"
echo "::: run directory   : ${RUN_DIR}"
echo ":::"

flux run \
    --ntasks=256 \
    --nodes=64 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"
