#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <math.h>
#include <omp.h>

#include "Utility.hpp"

#define MAX_THREADS 8
#define NUM_THREADS 4
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

Node *build_tree_rec(Point **point_list, int num_points, int depth){
    if (num_points <= 0){
        return nullptr;
    }

    if (num_points == 1){
        return new Node(point_list[0], nullptr, nullptr);
    }

    // get the dimension
    int dim = point_list[0]->dimension;

    // sort the list of points
    int axis = depth % dim;
    using std::placeholders::_1;
    using std::placeholders::_2;
    // sort point_list from begin to end, then bind the results to place 1, 2
    std::sort(point_list, point_list+(num_points-1), std::bind(Point::compare, _1, _2, axis));

    // select median
    Point **median = point_list + (num_points/2);
    Point **left_points = point_list;
    Point **right_points = median + 1;

    // calculate child points on the left and the right
    int num_points_left = num_points / 2;
    int num_points_right = num_points - num_points_left - 1;

    Node *left_node, *right_node;

    // determine the left subtree, and build it by recursive calling
    #pragma omp task if(depth < DEPTH) shared(left_node) firstprivate(left_points, num_points_left, depth)
    {
        left_node = build_tree_rec(left_points, num_points_left, depth+1);
    }

    // determine the right subtree, and build it by recursive calling
    #pragma omp task if(depth < DEPTH) shared(right_node) firstprivate(right_points, num_points_right, depth)
    {
        right_node = build_tree_rec(right_points, num_points_right, depth+1);
    }

    #pragma omp taskwait

    // return median node
    return new Node(*median, left_node, right_node);
}

Node *build_tree(Point **point_list, int num_nodes){
    return build_tree_rec(point_list, num_nodes, 0);
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

    // the query point is smaller than root in axis dimension, then go left
    if (d_axis < 0){
        visit_branch = root->left;
        other_branch = root->right;

    // the query point is larger than root in axis dimension, then go right
    } else {
        visit_branch = root->right;
        other_branch = root->left;
    }

    // after that, go to searching further nodes
    Node *further = nearest(visit_branch, query, depth+1, best_local, best_dist_local);
    if (further != nullptr){
        float dist_further = further->point->distance_squared(*query);
        if (dist_further < best_dist_local){
            best_local = further;
            best_dist_local = dist_further;
        }
    }

    // ???
    if (d_axis_square < best_dist_local){
        further = nearest(other_branch, query, depth+1, best_local, best_dist_local);
        if (further != nullptr){
            float dist_further = further->point->distance_squared(*query);
            if (dist_further < best_dist_local){
                best_local = further;
                // best_dist_local = dist_further;
            }
        }
    }

    return best_local;
}

Node *nearest_neighbor(Node *root, Point *query){
    float best_dist = root->point->distance_squared(*query);
    return nearest(root, query, 0, root, best_dist);
}

//================================================================
// Main Function
//================================================================

int main(int argc, char **argv){
    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 5;

#if DEBUG
    // for measuring the local runtime
    auto start_time = std::chrono::high_resolution_clock::now();

    // for specifying the problem
    Utility::specify_problem(argc, argv, &seed, &dim, &num_points);
#else
    // for specifying the problem
    Utility::specify_problem(&seed, &dim, &num_points);
#endif

    // the last points are query
    float *x = Utility::generate_problem(seed, dim, num_points + num_queries);
    Point **points = (Point **)calloc(num_points, sizeof(Point *));

    // create points
    for (int n = 0; n < num_points; ++n){
        points[n] = new Point(dim, n+1, x+n*dim);
    }

    // show the points
    for (int n = 0; n < num_points; ++n){
        std::cout << points[n][0] << std::endl;
    }

    // build the tree | just need 1 thread to build the tree
    Node *tree;
    #pragma omp parallel
    {
        #pragma omp single
        tree = build_tree(points, num_points); 
    }

    // print the tree
    // Utility::print_tree_rec(tree, 0);

    // -------------------------------------------------------------
    // query and find nearest neighbor in parallel with multiple threads
    // -------------------------------------------------------------
    #pragma omp parallel for ordered schedule(static, 1) num_threads(NUM_THREADS)
    for (int q = 1; q <= num_queries; ++q){

        // a query pointer by x axis
        float *x_query = x + (num_points + q) * dim;

        // create a query point
        Point query(dim, num_points + q, x_query);

        // find the nearest node with the query point
        Node *res = nearest_neighbor(tree, &query);

        // get and print the result
        float min_distance = query.distance(*res->point);
        // Utility::print_result_line(query.ID, min_distance);

        // debug - print query points
        #pragma omp ordered 
        std::cout << "T" << omp_get_thread_num() << "|Query point: " << query << std::endl;

#if DEBUG
        // in case you want to have further debug information about
        // the query point and the nearest neighbor
        // std::cout << "Query: " << query << std::endl;
        // std::cout << "NN: " << *res->point << std::endl << std::endl;
#endif
    }

#if DEBUG
    // for measuring your local runtime
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
#endif

    std::cout << "DONE" << std::endl;

    // clean-up
    Utility::free_tree(tree);
    for(int n = 0; n < num_points; ++n){
        delete points[n];
    }
    free(points);
    free(x);

    return 0;
}