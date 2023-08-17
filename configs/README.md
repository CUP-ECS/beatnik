# Build Configuraitons for diffrent HPC systems

This directory contains build information and scripts for a variety of systems, mainly using the spack build system.  In general, these builds are designed to work in a self-contained spack environment (spack documentation goes here XXX) with appropriate external compilers. Each directory describes the compiler used on each system and how to access/configure it, along with the spack environment to create in which to build beatnik.  For systems where spack is not pre-configured by the system maintainers, the build directory also contains a link to a repository with a relevant spack packages.yaml file, as well as a spack environment specification the can fully configure and build on that system. 

## Current list of tested build targets and suggested build method

  * University of New Mexico
    * Hopper V100/A100 GPU cluster system - Spack package 
    * Xena K40m GPU cluster system - Spack package
    * General UNM (Wheeler/Hopper) CPU systems - Spack package
  * Lawrence Livermore National Laboratory
    * Lassen V100 GPU system
    * Quartz CPU system
    * Tioga MX250X GPU system - WORK IN PROGRESS
  * Los Alamos National Laboratoru
    * Chicoma Cray A100 GPU system
    * Darwin GPU system

## Building Beatnik using Spack

Assuming that you have Spack already installed on your HPC systems (as described in https://spack.readthedocs.io), you can use it to either install Beatnik directly, develop Beatnik from the Spack specification, or create a spack environment for building Beatnik. Note that there are some version dependencies here, and using a Spack version newer than 0.20.0 is helpful. To build a Beatnik in a spack environment, do the following:

### Using the builtin Spack package description

Beatnik has a package specification in the well-known Spack installation package that can be used to directly build it on systems where spack is well-configured. On these systems, you can simply install the appropriate beatnik configuration. In particular, `spack install beatnik` should work on these systems. To get GPU support with this method, however, you will need to force the appropriate flags through to the packages on which Beatnik depends, for example by running `spack install beatnik ++cuda cuda_arch=80` if your machine is an NVIDIA A100 system.

### Developing Beatnik using a Spack package description

XXX 

### Building Beatnik in a Spack Environment

If you want fine-grain control of the build environment in which Beatnik is built, you can create a Spack environment from a specification and use that to install the various Spack pacakges on which Beatnik depends. This will allow you to more easily tweak build options for these dependencies, for example. To build a Beatnik in a spack environment, do the following:

  1. If not checked out from git recursively, checkout all needed Beatnik submodules, e.g. `git submodule init && git submodule update --recursive`
  1. Create a build directory for housing the Spack environment and housing the out-of-source build, e.g. `mkdir build.
  1. Copy the appropriate spack.yaml file from configs/[systemname]/ to spack-env.yaml in the newly-created build directory, e.g. `cp configs/unm-hopper/spack.yaml build/`
  1. Perform any compiler setup needed using the system module system, as spack environments do not necessarily configure the compiler. This could include installing appropriate development tools on Linux or MacOS systems (e.g. using apt or homebrew), or loading the proper compiler module on an HPC system. This compiler should be compatible with one used in the spack.yaml file chosen, and ideally described in a README.md file in the associated configs/ directory
  1. Change directory to the created build directory and create a spack environment in which to build Beatnik in that directory; we often create the environmetn in the build directory (e.g. `cd build; spack env create -d . spack-env.yaml`) but using named Spack environments is also possible
  1. Activate, concretize, and install the resulting environment, e.g. `spack env activate -d . && spack concretize && spack install`
  1. Run cmake and make to create the appropriate Makefiles and build using them, e.g. `cmake .. && make`.
  
### Beatnik Build-Time Configuration Options

XXX

### Known Spack build problems
  * Cabana versions prior to 0.6.0 (which is as unreleased as of August, 2023) request a HeFFTe version (2.1.0) that will not build using the HeFFTe spack specification in Spack versions 0.20.0 and before. The HeFFTe spack specification has been patched to address these problems, but this requires using the develop version of spack (or modifying the HeFFTe spack configuration in the spack repo) at least until Spack 0.21.0 or Cabana 0.6.0 are available.
