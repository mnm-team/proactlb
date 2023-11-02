
# option for compiling with intel tracer
option(CHAMELEON_TRACE_ITAC "Enable tracing with Intel Trace Analyzer " off)
option(KDTREE_ITERATIVE_VERSION "Enable iterative computation " off)
option(KDTREE_NUM_ITERATIONS "Set num of iterations " off)
option(KDTREE_COMPILE_CHAMELEON "Enable compiling with Chameleon " on)
option(KDTREE_NUM_REPETITIONS "Enable repeat in kdtree kernels " off)
option(KDTREE_DEBUG_MODE "Enable debug mode in kdtree main" off)

# link VTune
if (CHAMELEON_TRACE_ITAC)
    include_directories($ENV{VT_ROOT}/include)
    link_directories($ENV{VT_ROOT}/lib)
    set (ADDITIONAL_VTUNE_FLAGS ${VT_ADD_LIBS} "-DTRACE -trace")
endif()

# find MPI
find_package(MPI REQUIRED)

# find OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(DEFAULT_OPENMP_FLAG OpenMP::OpenMP_CXX)
endif()

# linking options of the mxm_api
if (KDTREE_ITERATIVE_VERSION)
    message("KDTREE_ITERATIVE_VERSION: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DITERATIVE_VERSION=1")
endif()
if (KDTREE_PARALLEL_INIT)
    message("KDTREE_PARALLEL_INIT: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DPARALLEL_INIT=1")
endif()
if (KDTREE_COMPILE_CHAMELEON)
    message("KDTREE_COMPILE_CHAMELEON: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DCOMPILE_CHAMELEON=1")
endif()
if (KDTREE_DEBUG_MODE)
    message("KDTREE_DEBUG_MODE: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DDEBUG")
endif()
if (KDTREE_NUM_ITERATIONS)
    message("KDTREE_NUM_ITERATIONS: ON")
endif()
if (KDTREE_NUM_REPETITIONS)
    message("KDTREE_NUM_REPETITIONS: ON")
endif()

# link Chameleon
set(CHAMELEON_LINK_FLAGS "-debug -g -std=c++17 -mavx -O3 -Wall -Wextra -DOMPI_SKIP_MPICXX -lm -lstdc++ -lchameleon")

# compile the code
add_executable(kdtree kdtree_chameleon.cpp Node.cpp Node.hpp Utility.cpp Utility.hpp)
target_link_libraries(kdtree ${MPI_CXX_LIBRARIES} ${DEFAULT_OPENMP_FLAG} ${ADDITIONAL_VTUNE_FLAGS} ${APP_COMPILE_FLAGS} ${CHAMELEON_LINK_FLAGS})

