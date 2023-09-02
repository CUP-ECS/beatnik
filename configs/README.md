# Build Configuraitons for diffrent HPC systems

This directory contains build information and scripts for a variety of systems, mainly using the spack build system.  In general, these builds are designed to work in a self-contained spack environment (spack documentation goes here XXX) with appropriate external compilers. Each directory describes the compiler used on each system and how to access/configure it, along with the spack environment to create in which to build beatnik.  For systems where spack is not pre-configured by the system maintainers, the build directory also contains a link to a repository with a relevant spack packages.yaml file, as well as a spack environment specification the can fully configure and build on that system. 

## General build requirements
Beatnik depends on the following pacakges to build in all configurations:
  1. Cabana version 0.5.0 or newer
  1. A heffte version compatible with Cabana (2.1.0 for Cabana 0.5.0, 2.3.0 will be needed startign with Cabana 0.6.0)
  1. An MPI implementation - note that MPI must be GPU-aware when running on GPU systems.
  1. Kokkos 4.0 or newer
  1. LLNL Silo 4.11 or newer configured with MPI support
  1. LLNL BLT (available as a git submodule)

## Installing and Building Beatnik with Spack

The beatnik spack package should enforce its build requirements appropriately; we strongly suggest that you use the spack package for both installation (via `spack install`) and development (via `spack develop`). We strive to keep the spack package spec up to date to enable this, as well as to maintain spack package.yaml descriptions for key systems in a [separate github repository](https://github.com/CUP-ECS/spack-configs). 

When you need more control over the build configuration (e.g. complex options for building spack or the packages on which it depends), we suggest using an explicit spack environment for building Beatnik's dependencies and using, 

### Current list of tested systems targets and suggested installation method

We have tested beatnik installation via spack install on the following systems with the provided spack install flags. An example run script and (if necessary) `setup.sh` file to set up the system environment is also provided for each of these systems in the appropriate subdirectory.
  * University of New Mexico
    * Hopper V100/A100 GPU cluster system - `spack install beatnik +cuda cuda_arch=80` (or `cuda_arch=70` for the V100 nodes).
    * General UNM (Wheeler/Hopper) CPU systems - `spack install beatnik`
  * Lawrence Livermore National Laboratory - Needed spack package.yaml file and documentation for LLNL systems available at (https://github.com/CUP-ECS/spack-configs)
    * Lassen V100 GPU system - `spack install beatnik +cuda cuda_arch=70 %gcc@8.3.1`; other compilers untested.
    * Quartz CPU system - `spack install beatnik`; with included package.yaml configuration
    * Tioga MX250X GPU system - `spack install beatnik +rocm amdgpu_arch=gfx90a`
  * Los Alamos National Laboratoru
    * Chicoma Cray A100 GPU system - `spack install beatnik +cuda cuda_arch=70 %XXX`
If you use `spack install` to build beatnik, you'll then need to run `spack load beatnik` to get access to the test executables to run, for example the `rocketrig` benchmark.

### Developing Beatnik and its dependencies using a Spack package description

If you want to develop Beatnik, we recommend using and environment along with the `spack develop` command to setup the development environment. In addition to allowing you to use spack to install dependecies, this will also let you tweak the package specification to control the details of the build environment, directly modify packages that you're developing (e.g. beatnik *and* its dependencies if you want!) and still use spack to build it. 

For example, to work on the current development branch of beatnik on the LLNL lassen system, you might do the following:
```
# Create a local spack environment in which to develop
[bridges7@lassen708:~]$ mkdir devel-env
[bridges7@lassen708:~]$ cd devel-env
[bridges7@lassen708:devel-env]$ spack env create -d .
==> Created environment in /g/g16/bridges7/devel-env
==> You can activate this environment with:
==>   spack env activate /g/g16/bridges7/devel-env
[bridges7@lassen708:devel-env]$ spack env activate .
[bridges7@lassen708:devel-env]$ spack add beatnik @develop +cuda cuda_arch=70 %gcc@8.3.1
==> Adding beatnik@develop%gcc@8.3.1+cuda cuda_arch=70 to environment /g/g16/bridges7/devel-env
# Mark beatnik as a package to develop from source locally in this environment
# and set up all dependencies
> spack develop beatnik @develop
> spack concretize 
# Now start developing in a git subbranch
> cd spack
> git checkout release-1.0-cleanup
# Install from 
> spack install
```

Importantly, you can also develop beatnik's //dependencies// this way. For example, if you want to modify cabana to better support beatnik, you would also run `spack develop cabana && spack concretize -f` to add cabana to the list of packages to develop locally. You can find more information on the spack develop workflow at the [Spack documentation webpages](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html).

## Known Spack build problems
  * Cabana versions prior to 0.6.0 (which is as unreleased as of August, 2023) request a HeFFTe version (2.1.0) that will not build using the HeFFTe spack specification in Spack versions 0.20.0 and before. The HeFFTe spack specification has been patched to address these problems, but this requires using the develop version of spack (or modifying the HeFFTe spack configuration in the spack repo) at least until Spack 0.21.0 or Cabana 0.6.0 are available.
