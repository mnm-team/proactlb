#!/bin/bash
echo "!!!!!!! Compiling Chameleon-with-Tool on SNG !!!!!!!"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon_tool_dev/}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# copy chameleon_patched files to merge with the tool
echo "2. Copy/Replace chameleon_patched files to integrate with the tool..."
cp ~/chameleon-scripts/cham_tools/load_pred_tool/chameleon_patch/* ~/chameleon_tool_dev/src/

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on SNG, e.g., hwloc, cmake
echo "3. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.modules
module load cmake/3.21.4
module load boost/1.75.0-gcc11
module load hwloc/2.6.0-gcc11
module load libffi-3.3-lrz-spack

export INCLUDE=$INCLUDE:/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/hwloc/2.6.0-gcc-cprcxhk/include
export CPATH=$CPATH:/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/hwloc/2.6.0-gcc-cprcxhk/include
export LIBRARY_PATH=$LIBRARY_PATH:/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/hwloc/2.6.0-gcc-cprcxhk/lib

# option: load itac packages on CoolMUC2
# module load itac
# source /lrz/sys/intel/studio2019_u5/itac/2019.5.041/bin/itacvars.sh intel64

# setting compilers and flags
# note: intel oneapi/2021 is already loaded on coolmuc2
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx
export FORT_COMPILER=mpiifort
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export KMP_AFFINITY=verbose
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto

# move to Chameleon folder
cd ${DIR_CH_SRC}

# remove some old compiled files
rm -r cmake_install.cmake CMakeCache.txt Makefile CMakeFiles install_manifest.txt

# compile chameleon
if [ "${BUILD_CHAM}" = "1" ]
then

# ##################### WITH ITAC #########################


# ##################### WITHOUT ITAC #########################
make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_mig" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL" \
    cmake .
make install

fi

# move back to the current folder
cd ${CUR_DIR}
