#include <iostream>
#include <random>
#include <math.h>

#define MAX_PRINT_DIMENSION 5

//================================================================
// Class Point
//================================================================ 
/**
 * @attribute ID
 * @attribute dimension
 * @attribute coordinates
 */

class Point {
    public:
        int ID;
        int dimension;
        float *coordinates;

        // Constructors
        Point();
        Point(int dim, int ID, float *coord);

        // Destructor
        ~Point();
        void free_point();

        // For estimating Euclidian Distance between 2 points
        static float distance_squared(Point &a, Point &b);
        float distance_squared(Point &b);

        static float distance(Point &a, Point &b);
        float distance(Point &b);

        // For comparing 2 points based on one axis of their coordinates
        static bool compare(Point *a, Point *b, int axis);

        // For representing the points
        friend std::ostream& operator<<(std::ostream&, const Point &point);
};

//================================================================
// Class Node
//================================================================ 
/**
 * @attribute *point
 * @attribute *left
 * @attribute *right
 */

class Node {
    public:
        Point *point;
        Node *left;
        Node *right;

        // Constructors
        Node() = default;
        Node(Point *p, Node *l, Node *r) : point{p}, left{l}, right{r} {};

        // Destructor
        ~Node() = default;
};