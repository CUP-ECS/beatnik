#!/bin/bash
# flux: --job-name=BeatnikTest 
# flux: --nodes=2
# flux: --exclusive 
# flux: --time=120            
# flux: --output={{name}}.{{jobid}}.log 

# FIXME change these to point to the right directories
SPACK_INSTALL=${HOME}/spack
BEATNIK_ENV=${HOME}/beatnik-env-tioga
BEATNIK_SCRATCH=/p/lustre1/${USER}/beatnik-tioga

echo "Loading spack environment"
source ${SPACK_INSTALL}/share/spack/setup-env.sh
spack env activate ${BEATNIK_ENV} 

echo "Creating output directory"
mkdir -p ${BEATNIK_SCRATCH}/data/raw
cd ${BEATNIK_SCRATCH}

# Make sure cray mpich supports GPU-aware communication
export MPICH_GPU_SUPPORT_ENABLED=1

echo "Starting MPI Run with ${SLURM_NTASKS} processes"
flux run --ntasks=16 --gpus-per-task=1 --cores-per-task=8 --setopt=mpibind=verbose:1 rocketrig -n 16384 -w 16 -F 0

echo "Finished MPI Run. Output in ${BEATNIK_SCRATCH}/data"
