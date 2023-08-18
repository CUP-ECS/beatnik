#!/bin/bash
#SBATCH --job-name=BeatnikTest # Job name
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=BeatnikTest.%j.log # Standard output and error log
#SBATCH --partition=cup-ecs

echo "Loading compiler"
module load gcc/11.2.0

echo "Loading spack development environment"
source /users/bridges/spack/share/spack/setup-env.sh
spack env activate -d /users/bridges/workspace/beatnik/build-hopper

mkdir -p /hopper/scratch/bridges/beatnik/data/raw
cd /hopper/scratch/bridges/beatnik
cp /users/bridges/workspace/beatnik/build-hopper/bin/rocketrig .

echo "Starting MPI Run with ${SLURM_NTASKS}"
srun -n ${SLURM_NTASKS} /users/bridges/workspace/beatnik/build-hopper/bin/rocketrig -x cuda -n 8192 -w ${SLURM_NTASKS}

echo "Finished MPI Run"
