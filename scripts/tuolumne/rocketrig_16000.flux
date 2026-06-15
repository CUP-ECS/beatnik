#!/bin/bash
# flux: --job-name=rocketrig_large
# flux: --nodes=1
# flux: --exclusive
# flux: --time=60
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
#
# Large-scale single-mode rocketrig run driven by Canopy's FMM BR solver.
# Uses examples/01_rocketrig/single_mode_large.in (16000x16000 mesh, 300
# timesteps).
#
# NOTE: This script uses the *production* spack environment
# (~/spack_envs/tuolumne_beatnik_production) rather than the development
# environment (~/spack_envs/tuolumne_beatnik). Large-scale jobs sit in
# queue for a long time; pinning them to the production env means
# ongoing development rebuilds in tuolumne_beatnik cannot break a job
# that is already queued or running.

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

NUM_PROCS=16
MESH_SIZE=4000
RUN_DIR="/p/lustre5/stewartj/beatnik/fmm/${NUM_PROCS}p_${MESH_SIZE}mesh"

ROCKETRIG="$(command -v rocketrig)"
INPUT_SRC="~/spack_envs/tuolumne_beatnik/beatnik/examples/01_rocketrig/single_mode_4000.in"

mkdir -p "${RUN_DIR}"
cp "${INPUT_SRC}" "${RUN_DIR}/single_mode_4000.in"
cd "${RUN_DIR}"

INPUT="${RUN_DIR}/single_mode_4000.in"

echo ":::"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT} (copied from ${INPUT_SRC})"
echo "::: run directory   : ${RUN_DIR}"
echo ":::"

flux run \
    --ntasks=${NUM_PROCS} \
    --nodes=4 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"
