#!/bin/bash
#SBATCH -J cham_mxm_api
#SBATCH -o job_output/out_%j_mxm_cham_mig_case1.txt
#SBATCH -e job_output/err_%j_mxm_cham_mig_case1.txt
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
module load hwloc
module load libffi-3.3-lrz-spack
module load intel_mig

## unset KMP_AFFINITY
export KMP_AFFINITY=verbose
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto
## export OMP_PROC_BIND=true

ulimit -s unlimited

## for new mpi version which does not support a sufficient number of tags
export MPIR_CVAR_CH4_OFI_TAG_BITS=30
export MPIR_CVAR_CH4_OFI_RANK_BITS=8

## for chameleon configs
export OMP_NUM_THREADS=23
export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MXM_EXAMPLE=1

MATRIX_SIZE=512
NUM_TASKS="400 300 200 200 200 100 100 100"

echo "mpirun -n ${SLURM_NTASKS} /dss/dsshome1/00/di46nig/chameleon-scripts/cham_examples/mxm_api/build_with_cham_supermucng/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}"
mpirun -n ${SLURM_NTASKS} /dss/dsshome1/00/di46nig/chameleon-scripts/cham_examples/mxm_api/build_with_cham_supermucng/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}
