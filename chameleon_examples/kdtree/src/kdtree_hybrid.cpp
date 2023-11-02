#include <algorithm>
#include <iostream>
#include <functional>
#include <cstring>
#include <chrono>
#include <random>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#include "Utility.hpp"

#define MAX_MPI_PROCESSES 32
#define MAX_THREADS 8 // on mnm laptop
#define NUM_THREADS 2
#define DEPTH 5

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

    // for mpi ranks and initialization
    int mpi_rank, mpi_size;
    int provided_threading = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_threading);
    /* 
     * where, MPI_THREAD_FUNNELED is the second level that informs MPI the application
     * is multithreaded, but all MPI call will be issued from the master thread only.
     */
    if (provided_threading != MPI_THREAD_FUNNELED){
        exit(1);
    }
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
    int num_points_local[MAX_MPI_PROCESSES];
    int displacements[MAX_MPI_PROCESSES];

    if (mpi_rank == 0){

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
            num_points_local[i] = 0;
        }
        for (int n = 0; n < num_points; ++n){
            int target = target_processes[n];
            num_points_local[target] += 1;
        }

        // check num_points_local
        // for (int i = 0; i < mpi_size; i++){
        //     std::cout << "[DEBUG] R" << i << ": num_points_local=" << num_points_local[i] << std::endl;
        // }

        // extract displacements, disp[rank_i] = disp[rank_i-1] + num_points(rank_i-1);
        displacements[0] = 0;
        std::cout << "[DEBUG] R0: displacement[0]=0" << std::endl;
        for (int i = 1; i < mpi_size; i++){
            displacements[i] = displacements[i-1] + num_points_local[i-1];
            std::cout << "[DEBUG] R" << i << ": displacement[" << i << "]=" << displacements[i-1] + num_points_local[i-1] << std::endl;
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

        // local clean-up
        free(tmp_indices);
        free(x_tmp);
        free(target_processes);
    }

    /* 
     * Scatter information about the number of local points
     * which each rank will hold
     */
    MPI_Scatter(num_points_local, 1, MPI_INT, num_points_local, 1, MPI_INT, 0, MPI_COMM_WORLD);


    /* 
     * Scatter information about x_global and id_global pointers
     * to each rank
     */
    float *x = (float *)calloc(num_points_local[0], dim*sizeof(float));
    int *id = (int *)calloc(num_points_local[0], sizeof(int));
    MPI_Datatype dt_point;
    MPI_Type_contiguous(dim, MPI_FLOAT, &dt_point);
    MPI_Type_commit(&dt_point);
    MPI_Scatterv(x_global, num_points_local, displacements,
                 dt_point, x, num_points_local[0],
                 dt_point, 0, MPI_COMM_WORLD);
    MPI_Scatterv(id_global, num_points_local, displacements,
                 MPI_INT, id, num_points_local[0],
                 MPI_INT, 0, MPI_COMM_WORLD);
    
    /*
     * Create points for building kdtree
     * each rank has its own points and build a separate kdtree
     */
    Point **points = (Point **)calloc(num_points_local[0], sizeof(Point *));
    for (int n = 0; n < num_points_local[0]; ++n){
        points[n] = new Point(dim, id[n] + 1, x + n*dim);
        // std::cout << "[DEBUG] point " << points[n]->ID << ", rank " << mpi_rank << std::endl;
    }

    // build the tree
    Node *tree;
    #pragma omp parallel // num_threads(3)
    {
        #pragma omp single
        tree = build_tree(points, num_points_local[0]);
    }

    /* 
     * Query on each process, return the nearest neighbor,
     * the result is nearest in overall
     */

    // broadcast the query
    MPI_Bcast(x_queries, num_queries * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float *min_distances = (float *)calloc(num_queries, sizeof(float));
    float *min_distances_global = (float *)calloc(num_queries, sizeof(float));

    #pragma omp parallel for schedule(dynamic, 1) num_threads(NUM_THREADS)
    for (int q = 0; q < num_queries; q++){
        float *x_query = x_queries + q * dim;
        Point query(dim, num_points+q, x_query); // create the query point
        Node *res = nearest_neighbor(tree, &query); // search the query point

        // return the output of min distance (i.e, to the query point)
        // but, this is just the local result, after this we have to do reduce
        min_distances[q] = query.distance(*res->point);
    }

    // reduce the min_distances in overall
    MPI_Reduce(min_distances, min_distances_global, num_queries, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    if(mpi_rank == 0){
        for(int q = 0; q < num_queries; ++q){
            Utility::print_result_line(num_points + q, min_distances_global[q]);
        }
    }

#if DEBUG
    auto end_time = std::chrono::high_resolution_clock::now();
    if (mpi_rank == 0){
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        std::cout << "elapsed_time = " << elapsed_time.count() << " (second)" << std::endl;
    }
#endif

    if(mpi_rank == 0){
        std::cout << "DONE" << std::endl;
    }

    /*
     * Clean up everything global in main
     */
    if(mpi_rank == 0){
        free(id_global);
        free(x_global);
    }

    // Utility::free_tree(tree);

    free(id);
    free(x);
    free(x_queries);

    // free(min_distances);
    // free(min_distances_global);

    for(int n = 0; n < num_points_local[0]; ++n){
        delete points[n];
    }
    free(points);

    MPI_Finalize();

    return 0;
}