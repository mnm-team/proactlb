
# option for compiling with intel tracer
option(CHAMELEON_TRACE_ITAC "Enable tracing with Intel Trace Analyzer " off)
option(NBODY_COMPILE_CHAMELEON "Enable compiling with Chameleon " on)
option(NBODY_DEBUG_MODE "Enable debug mode in Nbody main" off)

# linking with VTune
if (CHAMELEON_TRACE_ITAC)
    message("CHAMELEON_TRACE_ITAC: ON")
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
if (NBODY_COMPILE_CHAMELEON)
    message("NBODY_COMPILE_CHAMELEON: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DCOMPILE_CHAMELEON=1")
endif()
if (NBODY_DEBUG_MODE)
    message("NBODY_DEBUG_MODE: ON")
    set(APP_COMPILE_FLAGS ${APP_COMPILE_FLAGS} "-DDEBUG")
endif()

# link Chameleon
set(CHAMELEON_LINK_FLAGS "-debug -g -std=c++17 -mavx -O3 -Wall -Wextra -DOMPI_SKIP_MPICXX -lm -lstdc++ -lchameleon")

# compile the code
add_executable(nbody main.cpp nbody_cham_solver.cpp common.cpp common.h nbody.cpp nbody.h)
target_link_libraries(nbody ${MPI_CXX_LIBRARIES} ${DEFAULT_OPENMP_FLAG} ${ADDITIONAL_VTUNE_FLAGS} ${APP_COMPILE_FLAGS} ${CHAMELEON_LINK_FLAGS})

