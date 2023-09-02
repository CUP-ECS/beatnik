#!/bin/bash
#SBATCH --job-name=BeatnikTest # Job name
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=BeatnikTest.%j.log # Standard output and error log
#SBATCH --partition=cup-ecs

# FIXME change these to poiint to the right directories
SPACK_INSTALL=/users/${USER}/spack
BEATNIK_BUILD=/users/${USER}/workspace/beatnik/build-hopper
BEATNIK_SCRATCH=/carc/scratch/${USER}/beatnik-hopper

echo "Loading compiler"
module load gcc/11.2.0

echo "Loading spack development environment"
source ${SPACK_INSTALL}/share/spack/setup-env.sh
spack env activate -d ${BEATNIK_BUILD}

mkdir -p ${BEATNIK_SCRATCH}/data/raw
cp ${BEATNIK_BUILD}/bin/rocketrig ${BEATNIK_SCRATCH}

cd ${BEATNIK_SCRATCH}
echo "Starting MPI Run with ${SLURM_NTASKS}"
srun -n ${SLURM_NTASKS} ${BEATNIK_SCRATCH}/rocketrig -x cuda -n 8192 -w ${SLURM_NTASKS}

echo "Finished MPI Run. Output in ${BEATNIK_SCRATCH}/data"
