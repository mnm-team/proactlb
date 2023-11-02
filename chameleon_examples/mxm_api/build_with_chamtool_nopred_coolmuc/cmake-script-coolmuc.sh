# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

echo "Loading intel-compiler, chameleon-lib, ..."
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load cmake-3.20.1-oneapi-2021.1-enbi76g
module load boost-1.76.0-gcc-7.5.0-fromrfo # built with local-spack
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load libffi-3.3-oneapi-2021.1-3vomiwb
module load chamtool_nopred

# set parameters for mxm_api
export NUM_ITERATIONS=2

# run cmake
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        -DMXM_ITERATIVE_VERSION=1 -DMXM_NUM_ITERATIONS=1 -DMXM_PARALLEL_INIT=1 \
        -DMXM_COMPILE_CHAMELEON=1 -DMXM_COMPILE_TASKING=1 \
        -DCMAKE_CXX_FLAGS="-DNUM_ITERATIONS=${NUM_ITERATIONS}" \
        ..
