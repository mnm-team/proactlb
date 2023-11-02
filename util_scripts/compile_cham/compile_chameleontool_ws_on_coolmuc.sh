#!/bin/bash
echo "!!!!!!! Compiling Chameleon-with-Workstealing on CoolMUC2 !!!!!!!"

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon_tool_dev/}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# copy chameleon_patched files to merge with the tool
echo "2. Copy/Replace chameleon_patched files to integrate with the cham-src..."
cp ~/chameleon-scripts/cham_tools/load_pred_tool/chameleon_patch/* ~/chameleon_tool_dev/src/

# turn on CHAMELEON_TOOL flag and declare where is the tool
echo "3. Turn off CHAMELEON_TOOL flag, and declare the tool path..."
# export CHAMELEON_TOOL=1
# export CHAMELEON_TOOL_LIBRARIES=/dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on CoolMUC2, e.g., hwloc, cmake
echo "4. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load cmake-3.20.1-oneapi-2021.1-enbi76g
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load libffi-3.3-oneapi-2021.1-3vomiwb

# option: load itac packages on CoolMUC2
module load itac
module load itac-oneapi-2021-lrz
# source /lrz/sys/intel/studio2019_u5/itac/2019.5.041/bin/itacvars.sh intel64

# setting compilers and flags
# note: intel oneapi/2021 is already loaded on coolmuc2
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc
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

# make clean
# cmake -DCMAKE_INSTALL_PREFIX="/dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/chamtool_pred3_mig1_itac" \
#     -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
#     -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
#     -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=3 -DCHAM_PROACT_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1 -DCHAMELEON_TRACE_ITAC=1 -DTRACE" \
#     cmake .
# make install

# ##################### WITHOUT ITAC #########################

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_workstealing" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=0 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DWORK_STEALING=2 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1" \
    cmake .
make install

fi

# move back to the current folder
cd ${CUR_DIR}