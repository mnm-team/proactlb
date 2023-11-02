#!/bin/bash
echo "!!!!!!! Compiling Chameleon-with-Tool on BEAST !!!!!!!"

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon_tool_dev/}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# copy chameleon_patched files to merge with the tool
echo "2. Copy/Replace chameleon_patched files to integrate with the tool..."
cp ~/chameleon-scripts/cham_tools/load_pred_tool/chameleon_patch/* ~/chameleon_tool_dev/src/

# turn on CHAMELEON_TOOL flag and declare where is the tool
echo "3. Turn on CHAMELEON_TOOL flag, and declare the tool path..."
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_LIBRARIES=/home/ra56kop/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on CoolMUC2, e.g., hwloc, cmake
echo "4. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.module
module use ~/loc-libs/spack/share/spack/lmod/linux-sles15-zen2
module load loc-util-loggers
# module load hdf5-1.10.4-intel-2021.2.0-akmloob
module load hdf5-1.10.7-gcc-10.2.1-3c4s4ai
module load intel-oneapi-compilers-2021.2.0-gcc-10.2.1-rrubbme
module load intel-oneapi-mpi-2021.2.0-gcc-10.2.1-7hbelev
module load netcdf-c-4.6.1-gcc-10.2.1-s5bm5ra
module load cmake-3.20.3-gcc-10.2.1-crj7py7
module load hwloc-2.4.1-gcc-10.2.1-bu232a3
module load libffi-3.3-gcc-10.2.1-pp5dq3s

# setting compilers and flags
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


# ##################### WITHOUT ITAC #########################

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred3_mig1_off1_dbg" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=3 -DCHAM_PROACT_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1 -DDEBUG_PROACTIVE_MIGRATION=1" \
    cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred3_mig1_off2_dbg" \
    -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
    -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
    -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=3 -DCHAM_PROACT_MIGRATION=1 -DCHAM_STATS_PER_SYNC_INTERVAL -DDEBUG_PROACTIVE_MIGRATION=1" \
    cmake .
make install

# make clean
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred2_mig2_off1" \
#     -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
#     -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
#     -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=2 -DCHAM_PROACT_MIGRATION=2 -DCHAM_STATS_PER_SYNC_INTERVAL -DOFFLOAD_SEND_TASKS_SEPARATELY=1" \
#     cmake .
# make install

# make clean
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred2_mig2_off2" \
#     -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
#     -DCHAMELEON_PRINT_CONFIG_VALUES=1 -DCHAMELEON_STATS_RECORD=1 -DCHAMELEON_STATS_PRINT=1 -DCHAMELEON_TOOL_SUPPORT=1 \
#     -DCMAKE_CXX_FLAGS="-DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_PREDICTION_MODE=2 -DCHAM_PROACT_MIGRATION=2 -DCHAM_STATS_PER_SYNC_INTERVAL" \
#     cmake .
# make install

fi

# move back to the current folder
cd ${CUR_DIR}