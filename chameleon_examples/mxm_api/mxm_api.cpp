#ifndef NR_TASKS
#define NR_TASKS 200
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
#endif

#ifndef RANDOMDIST
#define RANDOMDIST 1
#endif

#ifndef PARALLEL_INIT
#define PARALLEL_INIT 0
#endif

#ifndef VERBOSE_MSG
#define VERBOSE_MSG 0
#endif

#ifndef VERBOSE_MATRIX
#define VERBOSE_MATRIX 0
#endif

#ifndef CHECK_GENERATED_TASK_ID
#define CHECK_GENERATED_TASK_ID 0
#endif

#ifndef SIMULATE_CONST_WORK
#define SIMULATE_CONST_WORK 0
#endif

#ifndef COMPILE_CHAMELEON
#define COMPILE_CHAMELEON 1
#endif

#ifndef COMPILE_TASKING
#define COMPILE_TASKING 1
#endif

#ifndef USE_TASK_ANNOTATIONS
#define USE_TASK_ANNOTATIONS 0
#endif

#ifndef USE_REPLICATION
#define USE_REPLICATION 0
#endif

#ifndef ITERATIVE_VERSION
#define ITERATIVE_VERSION 1
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1
#endif

#ifndef NUM_REPETITIONS
#define NUM_REPETITIONS 2
#endif

#ifndef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#if !COMPILE_CHAMELEON
#undef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif

#define LOG(rank, str) printf("#R%d: %s\n", rank, str)
#define SPEC_RESTRICT __restrict__

#include "util_string.h"
#include "util_init.h"

#if CHECK_GENERATED_TASK_ID
#include <mutex>
#endif

#if COMPILE_CHAMELEON
#include "chameleon.h"
#include "chameleon_pre_init.h"
#endif

// ------------------------------------------------------------------------
// Global variables
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// Main function
// ------------------------------------------------------------------------
int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
    my_rank_id = iMyRank;
    num_procs = iNumProcs;
	double fTimeStart, fTimeEnd;
	double wTimeCham, wTimeHost;
	bool pass = true;

#if COMPILE_CHAMELEON
    chameleon_pre_init();
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);
#endif

    int ret_code = parse_command_line_args(argc, argv);
    if (ret_code != 0) {
        return ret_code;
    }

    if(iMyRank == 0) {
        if(matrix_size_mode == matrix_size_mode_normal) {
            printf("Mode: Normal Task Distribution\n");
        } else if (matrix_size_mode == matrix_size_mode_non_uniform) {
            printf("Mode: Non-Uniform Task Distribution\n");
        } else if (matrix_size_mode == matrix_size_mode_non_uniform_imb){
            printf("Mode: Non-Uniform-Imbalance Task Distribution\n");
        }
    }

    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());


    // ---------------------------------------------------
    // Allocate and init the matrices
    // ---------------------------------------------------
    double **matrices_a, **matrices_b, **matrices_c;
	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];

    // to check the list of matrix sizes
    std::string str_mat_size_list = "";

#if PARALLEL_INIT
    if(iMyRank == 0) {
        printf("Init in parallel...\n");
    }
    #pragma omp parallel for
#endif
	for(int i = 0; i < numberOfTasks; i++) {

        int cur_size = matrixSize;
        if(matrix_size_mode == matrix_size_mode_non_uniform || matrix_size_mode == matrix_size_mode_non_uniform_imb) {
            cur_size = non_uniform_full_array_matrix_sizes[i];
        }
        str_mat_size_list += std::to_string(cur_size) + " ";

 		matrices_a[i] = new double[(long)cur_size*cur_size];
    	matrices_b[i] = new double[(long)cur_size*cur_size];
    	matrices_c[i] = new double[(long)cur_size*cur_size];

    	if(RANDOMINIT) {
    		initialize_matrix_rnd(matrices_a[i], cur_size);
    		initialize_matrix_rnd(matrices_b[i], cur_size);
    		initialize_matrix_zero(matrices_c[i], cur_size);
    	}
    	else {
    		initialize_matrix_test_A(matrices_a[i], cur_size);
    		initialize_matrix_test_A(matrices_b[i], cur_size);
    		initialize_matrix_zero(matrices_c[i], cur_size);
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Rank%d: task sizes=%s\n", my_rank_id, str_mat_size_list.c_str());

    // ---------------------------------------------------
    // Execute the tasks with Chameleon
    // ---------------------------------------------------

#if COMPILE_CHAMELEON
    fTimeStart=MPI_Wtime();
    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
#endif
            #pragma omp for
            for(int i = 0; i < numberOfTasks; i++) {
                int cur_size = matrixSize;
                if(matrix_size_mode == matrix_size_mode_non_uniform || matrix_size_mode == matrix_size_mode_non_uniform_imb) {
                    cur_size = non_uniform_full_array_matrix_sizes[i];
                }
                
                // printf("R#%d Iter%d: Chameleon task %d with matrix_size=%d\n", iMyRank, iter, i, cur_size);

                double * SPEC_RESTRICT A = matrices_a[i];
                double * SPEC_RESTRICT B = matrices_b[i];
                double * SPEC_RESTRICT C = matrices_c[i];

                // here we need to call library function to add task entry point and parameters by hand
                void* literal_matrix_size   = *(void**)(&cur_size);
                void* literal_i             = *(void**)(&i);

                chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[5];
                args[0] = chameleon_map_data_entry_create(A, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[1] = chameleon_map_data_entry_create(B, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
                args[2] = chameleon_map_data_entry_create(C, cur_size*cur_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
                args[3] = chameleon_map_data_entry_create(literal_matrix_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
                args[4] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

                // create opaque task here
                cham_migratable_task_t *cur_task = chameleon_create_task((void *)&matrixMatrixKernel, 5, args);

                // get the id of the last task added
                TYPE_TASK_ID last_t_id = chameleon_get_task_id(cur_task);

                // add tasks to the queue
                int32_t res = chameleon_add_task(cur_task);

                // clean up again
                delete[] args;
            }

            // call chameleon distributed taskwait to execute all the tasks
    	    int res = chameleon_distributed_taskwait(0);

            #pragma omp single
            MPI_Barrier(MPI_COMM_WORLD);

#if ITERATIVE_VERSION
        }
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    fTimeEnd=MPI_Wtime();
    wTimeCham = fTimeEnd-fTimeStart;
    if( iMyRank == 0 ) {
        printf("#R%d: Computations with chameleon took %.5f\n", iMyRank, wTimeCham);
    }

    LOG(iMyRank, "Validation:");
    if(numberOfTasks > 0) {
        for(int t = 0; t < numberOfTasks; t++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform || matrix_size_mode == matrix_size_mode_non_uniform_imb) {
                cur_size = non_uniform_full_array_matrix_sizes[t];
            }
            pass &= check_test_matrix(matrices_c[t], t, cur_size, cur_size);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }
#endif /* COMPILE_CHAMELEON */

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------
    // Execute the tasks with OpenMP-tasking
    // ---------------------------------------------------
#if COMPILE_TASKING
    fTimeStart=MPI_Wtime();
    #pragma omp parallel
    {
#if ITERATIVE_VERSION
        for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
#endif
		#pragma omp for
        for(int i = 0; i < numberOfTasks; i++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform || matrix_size_mode == matrix_size_mode_non_uniform_imb) {
                cur_size = non_uniform_full_array_matrix_sizes[i];
            }
            
            // uses normal tasks to have a fair comparison
            #pragma omp task default(shared) firstprivate(i,cur_size)
            {
                compute_matrix_matrix(matrices_a[i], matrices_b[i], matrices_c[i], cur_size);
            }
        }

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

#if ITERATIVE_VERSION
        }
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fTimeEnd=MPI_Wtime();
    wTimeHost = fTimeEnd-fTimeStart;

    if( iMyRank == 0 ) {
        printf("#R%d: Computations with normal tasking took %.5f\n", iMyRank, wTimeHost);
    }

    LOG(iMyRank, "Validation:");
    pass = true;
    if(numberOfTasks > 0) {
        for(int t = 0; t < numberOfTasks; t++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform || matrix_size_mode == matrix_size_mode_non_uniform_imb) {
                cur_size = non_uniform_full_array_matrix_sizes[t];
            }

            pass &= check_test_matrix(matrices_c[t], t, cur_size, cur_size);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }
#endif /* COMPILE_TASKING */

#if COMPILE_TASKING && COMPILE_CHAMELEON
    if( iMyRank==0 ) {
        printf("#R%d: This corresponds to a speedup of %.5f!\n", iMyRank, wTimeHost/wTimeCham);
    }
#endif /* COMPILE_TASKING && COMPILE_CHAMELEON */ 

    // deallocate matrices
    for(int i = 0; i < numberOfTasks; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }
    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;
    

    MPI_Barrier(MPI_COMM_WORLD);
#if COMPILE_CHAMELEON
    #pragma omp parallel
    {
        chameleon_thread_finalize();
    }
    chameleon_finalize();
#endif
    MPI_Finalize();

    return 0;
}