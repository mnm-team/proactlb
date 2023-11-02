#!/bin/bash
#SBATCH -J promig1_imb3
#SBATCH -o job_output/out_%j_mxm_cham_proactmig_off1_case3.txt
#SBATCH -e job_output/err_%j_mxm_cham_proactmig_off1_case3.txt
#SBATCH -D ./
##SBATCH --mail-type=ALL
##SBATCH --mail-user=minh.thanh.chung@ifi.lmu.de
#SBATCH --no-requeue
#SBATCH --account=pn73yo
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --nodes=4
##SBATCH --ntasks={{NTASKS}}
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=23
#SBATCH --exclusive
##SBATCH --mem=MaxMemPerNode
##SBATCH --ear=off

source /etc/profile.d/modules.sh
module load slurm_setup

module use ~/.modules
module load cmake/3.16.5
module load boost/1.73.0-gcc8
module load hwloc/2.2.0-gcc8
module load libffi-3.3-lrz-spack
module load cereal-1.3.0-gcc-8.4.0-iu6fggf-lrz-spack
module load hdf5/1.10.7-intel19-impi
module load armadillo-10.6.2
module load ensmallen-2.17.0
module load mlpack-3.4.2        # built with oneapi/2021
module load chamtool_pred3_mig1_off1

## do not have permission
## module use -a /lrz/sys/share/modules/extfiles
## module add likwid/modified

# export KMP_AFFINITY=verbose
unset KMP_AFFINITY
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto

ulimit -s unlimited

## for new mpi version which does not support a sufficient number of tags
export MPIR_CVAR_CH4_OFI_TAG_BITS=30
export MPIR_CVAR_CH4_OFI_RANK_BITS=8

## for chameleon tool configs
export OMP_NUM_THREADS=23
export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MXM_EXAMPLE=1
export EST_NUM_ITERS=5
export TIME2TRAIN=1
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_SUPPORT=1
export CHAMELEON_TOOL_LIBRARIES=/dss/dsshome1/00/di46nig/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so
export LOG_DIR=/dss/dsshome1/00/di46nig/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_supermucng/run_experiments/uniform_load/logs/imb_case3

export TASKS_PER_RANK="800,100,50,50,50,50,90,90"
NUM_TASKS="800 100 50 50 50 50 90 90"
MATRIX_SIZE=384

echo "mpirun -n ${SLURM_NTASKS} /dss/dsshome1/00/di46nig/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_supermucng/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}"
mpirun -n ${SLURM_NTASKS} /dss/dsshome1/00/di46nig/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_supermucng/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}
