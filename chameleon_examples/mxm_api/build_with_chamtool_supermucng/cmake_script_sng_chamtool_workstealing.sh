# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

# load intel_compiler & itac, ...
echo "Loading intel-compiler & mlpack, chameleon-lib, ..."
module use ~/.modules
module load cmake
module load hwloc
module load boost/1.75.0-gcc11
module load libffi-3.3-lrz-spack
module load cereal-1.3.0-gcc-8.4.0-iu6fggf-lrz-spack
module load hdf5/1.10.7-intel21-impi
module load armadillo-10.6.2
module load ensmallen-2.17.0
module load mlpack-3.4.2        # built with oneapi/2021
module load chamtool_workstealing

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# set parameters for mxm_api
export NUM_ITERATIONS=5

# run cmake
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        -DMXM_ITERATIVE_VERSION=1 -DMXM_NUM_ITERATIONS=1 -DMXM_PARALLEL_INIT=1 \
        -DMXM_COMPILE_CHAMELEON=1 -DMXM_COMPILE_TASKING=1 \
        -DCMAKE_CXX_FLAGS="-DNUM_ITERATIONS=${NUM_ITERATIONS}" \
        ..