# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

# load intel_compiler & itac, ...
module use ~/.module
module use ~/loc-libs/spack/share/spack/lmod/linux-sles15-zen2
module load loc-util-loggers
# module load hdf5-1.10.4-intel-2021.2.0-akmloob
module load hdf5-1.10.7-gcc-10.2.1-3c4s4ai
module load intel-oneapi-compilers-2021.2.0-gcc-10.2.1-rrubbme
module load intel-oneapi-mpi-2021.2.0-gcc-10.2.1-7hbelev
module load intel-mkl-2020.4.304-gcc-10.2.1-3szhsyk
module load netcdf-c-4.6.1-gcc-10.2.1-s5bm5ra
module load cmake-3.20.3-gcc-10.2.1-crj7py7
module load hwloc-2.4.1-gcc-10.2.1-bu232a3
module load libffi-3.3-gcc-10.2.1-pp5dq3s
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load boost-1.76.0-gcc-10.2.1-hotyooi
module load armadillo-10.4.0
module load ensmallen-2.16.2
module load mlpack-3.4.2
module load chamtool_pred3_mig1_off1
# module load chamtool_pred3_mig1_off2

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# set parameters for mxm_api
export NUM_ITERATIONS=5

# run cmake
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        -DMXM_ITERATIVE_VERSION=1 -DMXM_NUM_ITERATIONS=1 -DMXM_PARALLEL_INIT=0 \
        -DMXM_COMPILE_CHAMELEON=1 -DMXM_COMPILE_TASKING=1 \
        -DCMAKE_CXX_FLAGS="-DNUM_ITERATIONS=${NUM_ITERATIONS}" \
        ..
