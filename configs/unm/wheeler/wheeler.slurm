#!/bin/bash
#SBATCH --job-name=BeatnikTest # Job name
#SBATCH --nodes=2
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00              # Time limit hrs:min:sec
#SBATCH --output=BeatnikTest.%j.log # Standard output and error log

# FIXME change these to poiint to the right directories
SPACK_INSTALL=${HOME}/spack
BEATNIK_SCRATCH=/carc/scratch/users/${USER}/beatnik-wheeler

echo "Loading spack and beatnik"
source ${SPACK_INSTALL}/share/spack/setup-env.sh
spack load beatnik

mkdir -p ${BEATNIK_SCRATCH}/data/raw

cd ${BEATNIK_SCRATCH}
echo "Starting MPI Run with ${SLURM_NTASKS} processes"
srun -n ${SLURM_NTASKS} rocketrig -n 512 -F 0 -w ${SLURM_NTASKS}

echo "Finished MPI Run. Output in ${BEATNIK_SCRATCH}/data"
