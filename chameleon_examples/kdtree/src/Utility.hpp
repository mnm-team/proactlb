#include <iostream>
#include <random>
#include <math.h>

#include "Node.hpp"

//================================================================
// Namespace for Utility
//================================================================ 

namespace Utility {

    // generate random vector based on seed
    float *generate_problem(int seed, int dim, int num_points);

    // print head and left-most / right-most leafs of node
    void print_head_and_leaves(Node *tree);

    // free the tree which is built dynamically
    void free_tree(Node* root);

    // help functions for displaying the tree
    void print_tree_rec(Node *root, int depth);
    void print_tree(Node *root);

    // prompt to specify problem details and validate them
    void validate_input(int seed, int dim, int num_points);
    void specify_problem(int *seed, int *dim, int *num_points);
    void specify_problem(int argc, char **argv, int *seed, int *dim, int *num_points);

    // print the results
    void print_result_line(int ID, float distance);
}