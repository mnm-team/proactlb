#include <algorithm>
#include <iostream>
#include <functional>
#include <cstring>
#include <chrono>
#include <random>
#include <math.h>
#include <omp.h>

#include "Utility.hpp"

#if COMPILE_CHAMELEON
#include "chameleon.h"
#include "chameleon_pre_init.h"
#else
#include <mpi.h>
#endif

#define MAX_MPI_PROCESSES 32
#define MAX_THREADS 8 // on mnm laptop
#define NUM_THREADS 2
#define DEPTH 5

#define DEBUG 1

//================================================================
// Tree-related Util Funtions
//================================================================

float Point::distance_squared(Point &a, Point &b){
    if(a.dimension != b.dimension){
        std::cout << "Dimensions do not match!" << std::endl;
        exit(1);
    }
    float dist = 0;
    for(int i = 0; i < a.dimension; ++i){
        float tmp = a.coordinates[i] - b.coordinates[i];
        dist += tmp * tmp;
    }
    return dist;
}

Node* build_tree_rec(Point** point_list, int num_points, int depth){
    if (num_points <= 0){
        return nullptr;
    }

    if (num_points == 1){
        return new Node(point_list[0], nullptr, nullptr); 
    }

    int dim = point_list[0]->dimension;

    // sort list of points based on axis
    int axis = depth % dim;
    using std::placeholders::_1;
    using std::placeholders::_2;

    std::sort(
        point_list, point_list + (num_points - 1), 
        std::bind(Point::compare, _1, _2, axis));

    // select median
    Point** median = point_list + (num_points / 2);
    Point** left_points = point_list;
    Point** right_points = median + 1;

    int num_points_left = num_points / 2;
    int num_points_right = num_points - (num_points / 2) - 1; 

    Node* left_node;
    Node* right_node;

    // left subtree
    #pragma omp task if(depth < DEPTH) shared(left_node) firstprivate(left_points, num_points_left, depth)
    {
        left_node = build_tree_rec(left_points, num_points_left, depth + 1);
    }

    // right subtree
    #pragma omp task if(depth < DEPTH) shared(right_node) firstprivate(right_points, num_points_right, depth)
    {
        right_node = build_tree_rec(right_points, num_points_right, depth + 1);
    }

    #pragma omp taskwait

    // return median node
    return new Node(*median, left_node, right_node); 
}

Node* build_tree(Point** point_list, int num_points){
    return build_tree_rec(point_list, num_points, 0);
}

Node *nearest(Node *root, Point *query, int depth, Node *best, float &best_dist){
    if (root == nullptr)
        return nullptr;

    int dim = query->dimension;
    int axis = depth % dim;

    Node *best_local = best;
    float best_dist_local = best_dist;

    float d_euclidian = root->point->distance_squared(*query);
    float d_axis = query->coordinates[axis] - root->point->coordinates[axis];
    float d_axis_square = d_axis * d_axis;

    if (d_euclidian < best_dist_local){
        best_local = root;
        best_dist_local = d_euclidian;
    }

    Node *visit_branch;
    Node *other_branch;

    if (d_axis < 0){
        visit_branch = root->left;
        other_branch = root->right;
    } else {
        visit_branch = root->right;
        other_branch = root->left;
    }

    Node *further;
    further = nearest(visit_branch, query, depth+1, best_local, best_dist_local);
    if (further != nullptr){
        float dist_further = further->point->distance_squared(*query);
        if (dist_further < best_dist_local){
            best_dist_local = dist_further;
            best_local = further;
        }
    }

    if (d_axis_square < best_dist_local){
        further = nearest(other_branch, query, depth+1, best_local, best_dist_local);
        if (further != nullptr){
            float dist_further = further->point->distance_squared(*query);
            if (dist_further < best_dist_local){
                best_local = further;
            }
        }
    }

    return best_local;
}

Node *nearest_neighbor(Node *root, Point *query){
    float best_dist = root->point->distance_squared(*query);
    return nearest(root, query, 0, root, best_dist);
}

int target_to_process(float x){
    return x < 0 ? 0 : 1;
}

int _ttp_d1_(float* x){
    return target_to_process(x[0]);
}

int _ttp_d2_(float* x){
    return target_to_process(x[0]) * 2 + target_to_process(x[1]);
}

int _ttp_d3_(float* x){
    return target_to_process(x[0]) * 4 + target_to_process(x[1]) * 2 + target_to_process(x[2]);
}

int _ttp_d4_(float* x){
    return target_to_process(x[0]) * 8 + target_to_process(x[1]) * 4 + target_to_process(x[2]) * 2 + target_to_process(x[3]);
}

int assign_target_process(float *x, int dim, int mpi_size){
    int process = -1;
    
    // check dimension
    if (dim <= 0){
        exit(1);
    }

    // assign by dim-limit with num of processes
    if(mpi_size >= 16){
        if(dim >= 4){
            process = _ttp_d4_(x);
        } else if (dim == 3) {
            process = _ttp_d3_(x);
        } else if (dim == 2) {
            process = _ttp_d2_(x);
        } else {
            process = _ttp_d1_(x);
        }
    } else if(mpi_size >= 8){
        if(dim >= 3) {
            process = _ttp_d3_(x);
        } else if(dim == 2) {
            process = _ttp_d2_(x);
        } else {
            process = _ttp_d1_(x);
        }
    } else if(mpi_size >= 4){
        if (dim >= 2) {
            process = _ttp_d2_(x);
        } else {
            process = _ttp_d1_(x);
        }
    } else if (mpi_size >= 2){
       process = _ttp_d1_(x);
    } else {
        exit(1);
    }

    return process;
}

void assign_target_processes(float *x, int *target_process, int dim, int num_points, int mpi_size){
    for (int n = 0; n < num_points; ++n){
        int tar = assign_target_process(x + n*dim, dim, mpi_size);
        target_process[n] = tar;
        // std::cout << "[DEBUG] assign_target_process: point " << n << ", tar = " << tar << std::endl;
    }
}

//================================================================
// Main Function
//================================================================
int main(int argc, char **argv){

    // configure omp: disable dynamic adjustment of the number of threads
    // to use for executing parallel regions
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    /* 
    * Init chameleon to determine base addresses
    */
#if COMPILE_CHAMELEON
    chameleon_pre_init();
    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);

    int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);

#else
    int provided_threading = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_threading);
    /* 
     * where, MPI_THREAD_FUNNELED is the second level that informs MPI the application
     * is multithreaded, but all MPI call will be issued from the master thread only.
     */
    if (provided_threading != MPI_THREAD_FUNNELED){
        exit(1);
    }

#endif

    // for mpi ranks and initialization
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (!((mpi_size%2 == 0) || (mpi_size%4==0))){
        std::cerr << "Expected number of processes for data decomposition in 2, 4, ..." << std::endl;
        MPI_Finalize();
        exit(1);
    }

    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 10;

#if DEBUG
    // for measuring the local runtime
    auto start_time = std::chrono::high_resolution_clock::now();

    // for specifying the problem
    Utility::specify_problem(argc, argv, &seed, &dim, &num_points);
#else
    // for specifying the problem
    Utility::specify_problem(&seed, &dim, &num_points);
#endif

    // broadcast the number of dimensions and points
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* 
     * data decomposition and problem generation on rank 0
     * the decomposing method is based on hypercubes,
     *  + first, find nearest neighbor in these cubes
     *  + later, find nearest neighbor overall
     */
    float *x_queries = (float *)calloc(num_queries, sizeof(float) * dim);
    float *x_global = nullptr;
    int *id_global = nullptr;
    int *target_processes = nullptr;
    const int NUM_MPI_RANKS = mpi_size;
    int displacements[NUM_MPI_RANKS];
    int num_local_points;

    if (mpi_rank == 0){

        int num_local_points_arr[NUM_MPI_RANKS];

        // generate the query problem
        float *x_tmp = Utility::generate_problem(seed, dim, num_points + num_queries);
        x_global = (float *)calloc(num_points + num_queries, dim*sizeof(float));
        id_global = (int *)calloc(num_points + num_queries, sizeof(int));

        // copy the points belonging to query-list
        std::memcpy(x_queries, x_tmp + num_points*dim, num_queries*dim*sizeof(float));

        // assign target processes based on coordinates
        target_processes = (int *)calloc(num_points, sizeof(int));
        assign_target_processes(x_tmp, target_processes, dim, num_points, mpi_size);
        for (int i = 0; i < mpi_size; i++) {
            num_local_points_arr[i] = 0;
        }
        for (int n = 0; n < num_points; ++n){
            int target = target_processes[n];
            num_local_points_arr[target] += 1;
        }

        // check num_local_points_arr
        // for (int i = 0; i < mpi_size; i++){
        //     std::cout << "[DEBUG] R" << i << ": num_local_points_arr=" << num_local_points_arr[i] << std::endl;
        // }

        // extract displacements, disp[rank_i] = disp[rank_i-1] + num_points(rank_i-1);
        displacements[0] = 0;
        // std::cout << "[DEBUG] R0: displacement[0]=0" << std::endl;
        for (int i = 1; i < mpi_size; i++){
            displacements[i] = displacements[i-1] + num_local_points_arr[i-1];
            // std::cout << "[DEBUG] R" << i << ": displacement[" << i << "]=" << displacements[i-1] + num_points_local[i-1] << std::endl;
        }

        // re-sort array based on target process
        int *tmp_indices = (int *)calloc(mpi_size, sizeof(int));
        for (int n = 0; n < num_points; ++n){
            int target_id = target_processes[n];
            int ind = tmp_indices[target_id];
            id_global[displacements[target_id] + ind] = n;
            std::memcpy(x_global + (displacements[target_id] + ind)*dim, x_tmp + n*dim, dim * sizeof(float));
            tmp_indices[target_id] += 1;

            // std::cout << "[DEBUG] id_global[displacements[target_id] + ind] = " \
            //           << "id_global[displacements[" << target_id << "] + " << ind << "]" \
            //           << "=" << n << std::endl;
            // std::cout << "[DEBUG] copy Point " << n << " to x_global at (displacements[" << target_id \
            //           << "] + " << ind << ")" << std::endl;
        }

        MPI_Scatter(num_local_points_arr, 1, MPI_INT, &num_local_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // local clean-up
        free(tmp_indices);
        free(x_tmp);
        free(target_processes);
    } else {
        MPI_Scatter(NULL, 1, MPI_INT, &num_local_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // for (int i = 0; i < mpi_size; ++i){
        // std::cout << "[ADDRESS] R" << mpi_rank << " num_points_master[" << i << "]: " << &num_points_master[i] << std::endl;
        // std::cout << "[ADDRESS] R" << mpi_rank << " num_points_local[" << i << "]: " << &num_points_local[i] << std::endl;
        // std::cout << "[ADDRESS] R" << mpi_rank << " displacements[" << i << "]: " << &displacements[i] << std::endl;
    // }

    /* 
     * Scatter information about the number of local points
     * which each rank will hold
     */
    
    std::cout << "[DEBUG] After Scatter: R" << mpi_rank << ", num_points_local=" << num_local_points << std::endl;

    /*
     * Clean up everything global in main
     */
    if(mpi_rank == 0){
        free(id_global);
        free(x_global);
    }

    // Utility::free_tree(tree);

    // free(id);
    // free(x);
    free(x_queries);

    // free(min_distances);
    // free(min_distances_global);

    // for(int n = 0; n < num_local_points; ++n){
    //     delete points[n];
    // }
    // free(points);

    MPI_Finalize();

    return 0;
}