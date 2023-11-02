# require a minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# declare the project name
project(mxm_api_chamtool)

# option for compiling with intel tracer
option(CHAMELEON_TRACE_ITAC "Enable tracing with Intel Trace Analyzer " off)
option(MXM_ITERATIVE_VERSION "Enable iterative computation " off)
option(MXM_NUM_ITERATIONS "Set num of iterations " off)
option(MXM_PARALLEL_INIT "Enable parallel init for mxm " off)
option(MXM_COMPILE_CHAMELEON "Enable compiling with Chameleon " on)
option(MXM_COMPILE_TASKING "Enable compiling with OpenMP Tasking " on)
option(MXM_NUM_REPETITIONS "Enable repeat in mxm kernels " off)
option(MXM_USE_REPLICATION "Enable using replication with Chameleon " off)
option(MXM_USE_EXTERNAL_CALLBACK "Enable using external callback tools " off)

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
if (MXM_ITERATIVE_VERSION)
    message("MXM_ITERATIVE_VERSION: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DITERATIVE_VERSION=1")
endif()
if (MXM_PARALLEL_INIT)
    message("MXM_PARALLEL_INIT: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DPARALLEL_INIT=1")
endif()
if (MXM_COMPILE_CHAMELEON)
    message("MXM_COMPILE_CHAMELEON: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DCOMPILE_CHAMELEON=1")
endif()
if (MXM_COMPILE_TASKING)
    message("MXM_COMPILE_TASKING: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DCOMPILE_TASKING=1")
endif()
if (MXM_USE_REPLICATION)
    message("MXM_USE_REPLICATION: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DUSE_REPLICATION=1")
endif()
if (MXM_USE_EXTERNAL_CALLBACK)
    message("MXM_USE_EXTERNAL_CALLBACK: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DUSE_EXTERNAL_CALLBACK=1")
endif()
if (MXM_NUM_ITERATIONS)
    message("MXM_NUM_ITERATIONS: ON")
endif()
if (MXM_NUM_REPETITIONS)
    message("MXM_NUM_REPETITIONS: ON")
endif()

# link Chameleon
set(CHAMELEON_LINK_FLAGS "-debug -g -O3 -std=c++11 -lm -lstdc++ -lchameleon")

# compile the code
add_executable(mxm_api mxm_api.cpp)
target_link_libraries(mxm_api ${MPI_CXX_LIBRARIES} ${DEFAULT_OPENMP_FLAG} ${ADDITIONAL_VTUNE_FLAGS} ${APP_COMPILE_FLAGS} ${CHAMELEON_LINK_FLAGS})
