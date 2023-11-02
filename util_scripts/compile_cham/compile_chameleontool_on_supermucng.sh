#!/bin/bash
echo "!!!!!!! Compiling Chameleon-with-Tool on SNG !!!!!!!"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon_tool_dev/}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# copy chameleon_patched files to merge with the tool
echo "2. Copy/Replace chameleon_patched files to integrate with the tool..."
cp ~/chameleon-scripts/cham_tools/load_pred_tool/chameleon_patch/* ~/chameleon_tool_dev/src/

# turn on CHAMELEON_TOOL flag and declare where is the tool
echo "3. Turn on CHAMELEON_TOOL flag, and declare the tool path..."
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_LIBRARIES=/dss/dsshome1/00/di46nig/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on SNG, e.g., hwloc, cmake
echo "4. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.modules
module load cmake/3.21.4
module load boost/1.75.0-gcc11
module load hwloc/2.6.0-gcc11
module load libffi-3.3-lrz-spack

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
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred3_mig1_off1" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=3 -DCHAM_PROACT_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1" \
    cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred3_mig1_off2" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=3 -DCHAM_PROACT_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL" \
    cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred2_mig2_off1" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=2 -DCHAM_PROACT_MIGRATION=2 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1" \
    cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred2_mig2_off2" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=2 -DCHAM_PROACT_MIGRATION=2 -DCHAM_STATS_PER_SYNC_INTERVAL" \
    cmake .
make install

fi

# move back to the current folder
cd ${CUR_DIR}