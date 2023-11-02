CXX=c++
CXX_FLAGS=-O3 -std=c++17 -lm -Wall -Wextra -mavx
OPENMP=-fopenmp
DEBUG=-DDEBUG

MPICXX=mpicxx
MPICXX_FLAGS = --std=c++17 -mavx -O3 -Wall -Wextra -g -DOMPI_SKIP_MPICXX

SRC=./src
UTIL=./src/Node.cpp ./src/Node.hpp ./src/Utility.cpp ./src/Utility.hpp

all: sequential debug	# omp_cpu omp_gpu

#-----------------------------------------------------------------------------------------#
sequential: $(SRC)/kdtree_seq.cpp
	$(CXX) $(CXX_FLAGS) $(SRC)/kdtree_seq.cpp $(UTIL) -o sequential

debug: $(SRC)/kdtree_seq.cpp
	$(CXX) $(CXX_FLAGS) $(DEBUG) $(SRC)/kdtree_seq.cpp $(UTIL) -o sequential

omp: $(SRC)/kdtree_omp.cpp
	$(CXX) $(CXX_FLAGS) $(DEBUG) $(OPENMP) $(SRC)/kdtree_omp.cpp $(UTIL) -o omp

hybrid: $(SRC)/kdtree_hybrid.cpp
	$(MPICXX) $(MPICXX_FLAGS) $(DEBUG) $(OPENMP) $(SRC)/kdtree_hybrid.cpp $(UTIL) -o hybrid

#-----------------------------------------------------------------------------------------#

run_hybrid_debug_2:
	mpirun -np 2 ./hybrid 7 3 20

run_hybrid_debug_4:
	mpirun -np 4 ./hybrid 14 3 30

#-----------------------------------------------------------------------------------------#

clean:
	rm -f sequential omp hybrid