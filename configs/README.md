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
    * Darwin GPU system - `spack instal 

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
[bridges7@lassen708:devel-env]$ spack develop beatnik @develop +cuda cuda_arch=70 %gcc@8.3.1
==> Configuring spec beatnik@=develop%gcc@8.3.1+cuda cuda_arch=70 for development at path beatnik
[bridges7@lassen708:devel-env]$ spack concretize
==> Concretized beatnik@develop%gcc@8.3.1+cuda cuda_arch=70
...
# Now start developing in a git subbranch
[bridges7@lassen708:devel-env] cd beatnik
[bridges7@lassen708:beatnik]$ git checkout release-1.0-cleanup
Branch 'release-1.0-cleanup' set up to track remote branch 'release-1.0-cleanup' from 'origin'.
Switched to a new branch 'release-1.0-cleanup'
[bridges7@lassen708:beatnik]$ spack install
[+] /usr/tce/packages/cmake/cmake-3.23.1 (external cmake-3.23.1-gg5jrvmou665ssxt4wucmgij3b7nknou)
[+] /usr/tce/packages/cuda/cuda-11.7.0 (external cuda-11.7.0-u5mvzngv5t3r5s7em4bqmsupkkkrt7qu)
[+] /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1 (external spectrum-mpi-rolling-release-cms5gg7vjfr6ez4i6g6v2d72yxpd4apx)
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/pkgconf-1.9.5-2c2v5nm34o53bqmfesem4p7oyfqvisra
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/zlib-1.2.13-ibqzhjkqlgbzuer47g26qazsmqkete5y
[+] /usr (external ncurses-5.9-umdjlknnhlhps46qxuwkuvv2mqfotyhb)
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/gmake-4.4.1-bt4sn3lkoi26rkpzpnhk3suoeevhlgrf
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/blt-0.5.3-yd22z7pcyvdwzjd6de4fpgegj5gdtnck
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/kokkos-nvcc-wrapper-4.0.01-taoq622vaz2tem3e2afagwic5a7zhgfw
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/fftw-3.3.10-2rwicdyovmekxe4pwf4hgivtkttxozoh
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/readline-8.2-nlvfz33jvxohsdw5b3223xhwe44fjjot
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/hdf5-1.14.2-fjtpyalmdxaoyh4ea7yxy66ka2okvtoi
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/kokkos-4.1.00-kvn3hvnouc545lvxet73z3xhpdyhsxcn
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/heffte-2.1.0-rzqlejfwrnp34ihemu7tfzt5a2nkdqsh
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/silo-4.11-gaci6mdypwmuhepf6zsqqnupilug2jsl
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/cabana-0.5.0-fybm456ddrcnchkxcof53q7j427ef2to
==> Installing beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa [17/17]
==> No binary for beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa found: installing from source
==> No patches needed for beatnik
==> beatnik: Executing phase: 'cmake'
==> beatnik: Executing phase: 'build'
==> beatnik: Executing phase: 'install'
==> beatnik: Successfully installed beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa
  Stage: 0.00s.  Cmake: 11.20s.  Build: 1m 21.66s.  Install: 0.86s.  Post-install: 0.34s.  Total: 1m 35.20s
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa
```

Importantly, you can also develop beatnik's //dependencies// this way. For example, if you want to modify cabana to better support beatnik, you would similarly run `spack develop cabana && spack concretize -f` to add cabana to the list of packages to develop locally. You can find more information on the spack develop workflow at the [Spack documentation webpages](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html).

## Known Spack build problems
  * Cabana versions prior to 0.6.0 (which is as unreleased as of August, 2023) request a HeFFTe version (2.1.0) that will not build using the HeFFTe spack specification in Spack versions 0.20.0 and before. The HeFFTe spack specification has been patched to address these problems, but this requires using the develop version of spack (or modifying the HeFFTe spack configuration in the spack repo) at least until Spack 0.21.0 or Cabana 0.6.0 are available.
