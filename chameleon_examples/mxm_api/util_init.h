#ifndef __UTIL_INIT__
#define __UTIL_INIT__

#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <inttypes.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>
#include <atomic>
#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include <mpi.h>
#include "math.h"

// static rank id that can also be used in other functions except main
static int my_rank_id = 0;
static int num_procs = -1;

typedef enum matrix_size_mode_t {
    matrix_size_mode_normal = 0,
    matrix_size_mode_normal_imb = 1,
    matrix_size_mode_non_uniform = 2,
    matrix_size_mode_non_uniform_imb = 3
} matrix_size_mode_t;

matrix_size_mode_t matrix_size_mode = matrix_size_mode_normal;
int numberOfTasks = 0;

// mode: normal (default size)
int matrixSize = 100;

// mode: non-uniform
typedef enum non_uniform_ordering_t {
    non_uniform_ordering_high_to_low = 0,
    non_uniform_ordering_low_to_high = 1
} non_uniform_ordering_t;

typedef struct non_uniform_matrix_settings_t {
    int matrix_size;
    int number_tasks;
} non_uniform_matrix_settings_t;

non_uniform_ordering_t non_uniform_ordering = non_uniform_ordering_high_to_low;
std::vector<non_uniform_matrix_settings_t> non_uniform_matrix_settings;
std::vector<int> non_uniform_full_array_matrix_sizes;

#if USE_EXTERNAL_CALLBACK
typedef struct my_custom_params_t {
    int val1;
    TYPE_TASK_ID task_id;
} my_custom_params_t;

void print_finish_message(void *param) {
    // parse parameter again to regular data type
    my_custom_params_t *mydata = (my_custom_params_t *) param;
    printf("#R%d (OS_TID:%ld): External task finish callback for task with id %d with value %d\n", my_rank_id, syscall(SYS_gettid), mydata->task_id, mydata->val1);
    // clean up
    free(mydata);
}
#endif

void initialize_matrix_rnd(double *mat, int matrixSize) {
	double lower_bound = 0;
	double upper_bound = 10000;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= unif(re);
	}
}

void initialize_matrix_zero(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
		mat[i]= 0;
	}
}

void initialize_matrix_test_A(double *mat, int matrixSize) {
	for(int i=0; i<matrixSize*matrixSize; i++) {
			mat[i]= 1;
    }
}

void compute_matrix_matrix(double * SPEC_RESTRICT a, double * SPEC_RESTRICT b, double * SPEC_RESTRICT c, int matrixSize) {
    // make the tasks more computational expensive by repeating this operation several times to better see effects 
    for(int iter=0;iter<NUM_REPETITIONS;iter++) {
        for(int i=0;i<matrixSize;i++) {
            for(int j=0;j<matrixSize;j++) {
                c[i*matrixSize+j]=0;
                for(int k=0;k<matrixSize;k++) {
                    c[i*matrixSize+j] += a[i*matrixSize+k] * b[k*matrixSize+j];
                }
            }
        }
    }
}

bool check_test_matrix(double *c, int matrix_idx, double val, int matrixSize) {
	for(int i=0;i<matrixSize;i++) {
		for(int j=0;j<matrixSize;j++) {
			if(fabs(c[i*matrixSize+j] - val) > 1e-3) {
				printf("#R%d (OS_TID:%ld): Error in matrix %03d entry (%d,%d) expected:%f but value is %f\n", my_rank_id, syscall(SYS_gettid),matrix_idx,i,j,val,c[i*matrixSize+j]);
				return false;
			}
		}
	}
	return true;
}

void compute_random_task_distribution(int *dist, int nRanks) {
	double *weights = new double[nRanks];
	
	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;
	double sum = 0;

	for(int i=0; i<nRanks; i++) {
		weights[i]= unif(re);
		sum += weights[i];
	}

	for(int i=0; i<nRanks; i++) {
		weights[i]= weights[i]/sum;
		dist[i] = weights[i]*NR_TASKS;
	}

	delete[] weights;
}

void printHelpMessage() {
    if(my_rank_id == 0) {
        std::cout << "Usage (mode=normal): mpiexec -n np ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] " << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        matrixSize:   Number of elements of the matrixSize x matrixSize matrices" << std::endl;
        std::cout << "        nt_(i):       Number of tasks for process i " << std::endl;
        std::cout << "    If the number of tasks is not specified for every process, the application will generate an initial task distribution" << std::endl << std::endl;

        std::cout << "Usage (mode=normal-imb): mpiexec -n np ./matrixExample uniform-imb numtasks matrixSize(0) ... matrixSize(N) " << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        numtasks:      Number of tasks per rank" << std::endl;
        std::cout << "        matrixSize(i): Matrix size for process i " << std::endl;

        std::cout << "Usage (mode=non-uniform): mpiexec -n np ./matrixExample non-uniform matrixSizes numberTasks [order_(0) ... order_(np-1)] " << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        matrixSizes:  Comma separated list of different matrix sizes for non-uniform task creation" << std::endl;
        std::cout << "        numberTasks:  Comma separated list defining number of tasks for each matrix size" << std::endl;
        std::cout << "        order_(i):    Ordering of tasks using matrix sizes for rank/process i; 0=\"high to low\" (default); 1=\"low to high\"" << std::endl << std::endl;

        std::cout << "Usage (mode=non-uniform-imb): mpiexec -n np ./matrixExample non-uniform-imb matrixSizes numberTasks <imb_ratio> <std_value>" << std::endl;
        std::cout << "    Arguments: " << std::endl;
        std::cout << "        matrixSizes:  Comma separated list of different matrix sizes for non-uniform task creation" << std::endl;
        std::cout << "        numberTasks:  Comma separated list defining number of tasks for each matrix size" << std::endl;
        std::cout << "        imb_ratio:    The imbalance ratio is calculate by max_load and avg_load" << std::endl << std::endl;
        std::cout << "        std_value:    The approximate standard deviation among Ranks" << std::endl << std::endl;
    }
}

void printArray(int rank, double * SPEC_RESTRICT array, char* arr_name, int n) {
    printf("#R%d (OS_TID:%ld): %s[0-%d] at (" DPxMOD "): ", rank, syscall(SYS_gettid), arr_name, n, DPxPTR(&array[0]));
    for(int i = 0; i < n; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void matrixMatrixKernel(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int matrixSize, int i) {
#if VERBOSE_MATRIX
    int iMyRank2;
    MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank2);
    printArray(iMyRank2, A, "A", 10);
    printArray(iMyRank2, B, "B", 10);
    printArray(iMyRank2, C, "C", 10);
#endif

#if SIMULATE_CONST_WORK
    // simulate work by just touching the arrays and wait for 50 ms here
    C[matrixSize] = A[matrixSize] * B[matrixSize];
    usleep(50000);
#else
    compute_matrix_matrix(A, B, C, matrixSize);
#endif
}

int estimate_num_tasks_by_imb_ratio(double imb_ratio, double std_devia){

    int N = 0;
    int NTASKS = 0;
    // std::string sizes_str = "";
    // std::string tasks_str = "";
    for (int i = 0; i < non_uniform_matrix_settings.size(); i++){
        // sizes_str += std::to_string(non_uniform_matrix_settings[i].matrix_size) + " ";
        // tasks_str += std::to_string(non_uniform_matrix_settings[i].number_tasks) + " ";
        int ntasks = non_uniform_matrix_settings[i].number_tasks;
        int load = ntasks * non_uniform_matrix_settings[i].matrix_size;
        // N += non_uniform_matrix_settings[i].number_tasks;
        N += load;
        NTASKS += ntasks;
    }
    // printf("Estimate: sizes=%s\n", sizes_str.c_str());
    // printf("Estimate: amount=%s\n", tasks_str.c_str());

    // printf("Estimate R%d: num_ranks=%d, N=%d\n", my_rank_id, num_procs, N);

    /* Assume we have: imb_ratio (I) = [(n-1)N - R] / (N + R)
       where, N is max load at Rank 0, R is the total load of the remaining ranks
       n is the number of ranks (processes)
       Therefore, R = (n - 1 - I) * N / (I + 1) */
    if (((num_procs - 1 - imb_ratio) * N) <= 0){
        std::cout << "Error: num_procs and imb_ration cannot generate the list of msizes" << std::endl;
        return 1;
    }
    
    // calculate the remaining load of other ranks
    double R = (double)((num_procs - 1 - imb_ratio) * N / (imb_ratio + 1));
    double avg_load_per_rank = R / (num_procs - 1);
    // printf("Estimate R%d: Load of the remaining ranks=%f\n", my_rank_id, R);

    // the first rank, always set the max-overloaded rank
    if (my_rank_id == 0){
        numberOfTasks = NTASKS;
    }
    else if (my_rank_id != 0 && my_rank_id < num_procs-1){
        double r = avg_load_per_rank / N;
        int count = 0;
        for (int i = 0; i < non_uniform_matrix_settings.size()-1; i++){
            int amount = non_uniform_matrix_settings[i].number_tasks;
            if (i == non_uniform_matrix_settings.size()-1){
                int tmp_ntasks = (int)(r * NTASKS) - count;
                non_uniform_matrix_settings[i].number_tasks = tmp_ntasks;
                count += tmp_ntasks;
            }
            else {
                non_uniform_matrix_settings[i].number_tasks = (int)(r * amount);
                count += non_uniform_matrix_settings[i].number_tasks;
            }
        }
        numberOfTasks = count;
    }
    // the last rank, get the rest
    else{
        double r = (R - avg_load_per_rank*(num_procs-2)) / N;
        int count = 0;
        for (int i = 0; i < non_uniform_matrix_settings.size(); i++){
            int amount = non_uniform_matrix_settings[i].number_tasks;
            if (i == non_uniform_matrix_settings.size()-1){
                non_uniform_matrix_settings[i].number_tasks = (int)(r*NTASKS) - count;
                count += non_uniform_matrix_settings[i].number_tasks;
            }
            else {
                non_uniform_matrix_settings[i].number_tasks = (int)(r * amount);
                count += non_uniform_matrix_settings[i].number_tasks;
            }
        }
        numberOfTasks = count;
    }

    return 0;
}

int parse_command_line_args(int argc, char **argv) {

    if (argc >= 2 && strcmp(argv[1], "uniform-imb") == 0) {
        matrix_size_mode = matrix_size_mode_normal_imb;

        // check the number of argurments
        if (argc != num_procs+3){
            std::cout << "Error: Insufficient number parameters" << std::endl;
            printHelpMessage();
            return 1;
        }

        // parse the arguments
        if(my_rank_id == 0) {
            LOG(my_rank_id, "using user-defined initial load for uniform-imb...");    
        }
        numberOfTasks = atoi( argv[2] ); 
        matrixSize = atoi( argv[my_rank_id+3] ); 
    }

    else if (argc >= 2 && strcmp(argv[1], "non-uniform-imb") == 0) {

        matrix_size_mode = matrix_size_mode_non_uniform_imb;

        // at least the number of arguments > 4
        if (argc < 4){
            std::cout << "Error: Insufficient number parameters" << std::endl;
            printHelpMessage();
            return 1;
        }

        // parse matrix sizes and the amount of tasks per size
        std::string str_msizes(argv[2]);
        std::list<std::string> cur_split_msizes = split(str_msizes, ',');
        std::string str_ntasks(argv[3]);
        std::list<std::string> cur_split_ntasks = split(str_ntasks, ',');
        if(cur_split_msizes.size() != cur_split_ntasks.size()) {
            std::cout << "Error: Number of matrix sizes and number of tasks does not match!" << std::endl;
            return 1;
        }

        std::string msizes_str = "";
        for (std::string s : cur_split_msizes) {
            non_uniform_matrix_settings_t new_obj;
            new_obj.matrix_size = std::atoi(s.c_str());
            non_uniform_matrix_settings.push_back(new_obj);
            msizes_str += s + " ";
        }
        // printf("List sizes: %s\n", msizes_str.c_str());

        std::string ntasks_str = "";
        numberOfTasks = 0;
        int count = 0;
        for (std::string s : cur_split_ntasks) {
            int tmp_num = std::atoi(s.c_str());
            non_uniform_matrix_settings[count].number_tasks = tmp_num;
            numberOfTasks += tmp_num;
            count++;
            ntasks_str += s + " ";
        }
        // printf("Amount tasks: %s\n", ntasks_str.c_str());

        // parse the imbalance ratio and standard deviation
        double imb_ratio, std_devia;
        imb_ratio = std::atof(argv[4]);
        std_devia = std::atof(argv[5]);
        // printf("Imb_ration & Std_dev: %f & %f\n", imb_ratio, std_devia);

        // estimate the number of tasks and their sizes on each rank
        estimate_num_tasks_by_imb_ratio(imb_ratio, std_devia);

        // assign tasks to the array
        non_uniform_full_array_matrix_sizes.clear();
        for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
            for (int i = 0; i < s.number_tasks; i++) {
                non_uniform_full_array_matrix_sizes.push_back(s.matrix_size);
            }
        }

        // shuffle the vector of sizes
        std::random_device rd("/dev/urandom"); 
        std::default_random_engine eng(rd());
        std::shuffle(std::begin(non_uniform_full_array_matrix_sizes), std::end(non_uniform_full_array_matrix_sizes), eng);
    }
    else if(argc >= 2 && strcmp(argv[1], "non-uniform") == 0) {
        matrix_size_mode = matrix_size_mode_non_uniform;

        if(argc < 4) {
            std::cout << "Error: Insufficient number parameters" << std::endl;
            printHelpMessage();
            return 1;
        }

        // parse matrix sizes and number of tasks
        std::string str_msizes(argv[2]);
        std::list<std::string> cur_split_msizes = split(str_msizes, ',');
        std::string str_ntasks(argv[3]);
        std::list<std::string> cur_split_ntasks = split(str_ntasks, ',');
        if(cur_split_msizes.size() != cur_split_ntasks.size()) {
            std::cout << "Error: Number of matrix sizes and number of tasks does not match!" << std::endl;
            return 1;
        }

        for (std::string s : cur_split_msizes) {
            non_uniform_matrix_settings_t new_obj;
            new_obj.matrix_size = std::atoi(s.c_str());
            non_uniform_matrix_settings.push_back(new_obj);
        }

        numberOfTasks = 0;
        int count = 0;
        for (std::string s : cur_split_ntasks) {
            int tmp_num = std::atoi(s.c_str());
            non_uniform_matrix_settings[count].number_tasks = tmp_num;
            numberOfTasks += tmp_num;
            count++;
        }

        // parse ordering
        if(argc > 4) {
            if(argc != 4+num_procs) {
                std::cout << "Error: Number of matrix ordering values does not match number of processes/ranks!" << std::endl;
                return 1;
            }
            int tmp_order = std::atoi(argv[4+my_rank_id]);
            non_uniform_ordering = (non_uniform_ordering_t)tmp_order;
        }

        // apply ordering
        if (non_uniform_ordering == non_uniform_ordering_high_to_low) {
            std::sort(
                non_uniform_matrix_settings.begin(), 
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                { 
                    return a.matrix_size > b.matrix_size;
                }
            );
        } else {
            std::sort(
                non_uniform_matrix_settings.begin(), 
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                { 
                    return b.matrix_size > a.matrix_size;
                }
            );
        }

        non_uniform_full_array_matrix_sizes.clear();
        for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
            for (int i = 0; i < s.number_tasks; i++) {
                non_uniform_full_array_matrix_sizes.push_back(s.matrix_size);
            }
        }

        // ===== DEBUG
        // printf("Rank#%d - Ordering: %d\n", my_rank_id, non_uniform_ordering);
        // for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
        //     printf("Rank#%d - MatrixSize: %d, NumTasks: %d\n", my_rank_id, s.matrix_size, s.number_tasks);
        // }
        // printf("Rank#%d - Size Array: ", my_rank_id);
        // for (int s : non_uniform_full_array_matrix_sizes) {
        //     printf("%d,", s);
        // }
        // printf("\n");
        // ===== DEBUG

    } else if(argc == 2) {
        matrix_size_mode = matrix_size_mode_normal;
        matrixSize = atoi( argv[1] );
        if(RANDOMDIST) {
            int *dist = new int[num_procs];
            if( my_rank_id==0 ) {
                compute_random_task_distribution(dist, num_procs);
            }
            MPI_Bcast(dist, num_procs, MPI_INTEGER, 0, MPI_COMM_WORLD);
            numberOfTasks = dist[my_rank_id];
            delete[] dist;
        } else {
            numberOfTasks = NR_TASKS;
        }
    } else if(argc == num_procs+2) {
        matrix_size_mode = matrix_size_mode_normal;
        if(my_rank_id == 0) {
            LOG(my_rank_id, "using user-defined initial load distribution...");    
        }
        matrixSize = atoi( argv[1] ); 
        numberOfTasks = atoi( argv[my_rank_id+2] ); 
    } else { 
        printHelpMessage();
        return 1;
    }

    return 0;
}

#endif