# require a minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# declare the project name
project(fib-bot-suits)

# option for compiling with intel tracer
option(CHAMELEON_TRACE_ITAC "Enable tracing with Intel Trace Analyzer " off)

# link VTune
if (CHAMELEON_TRACE_ITAC)
    include_directories($ENV{VT_ROOT}/include)
    link_directories($ENV{VT_ROOT}/lib)
    set (ADDITIONAL_VTUNE_FLAGS "${ADDITIONAL_VTUNE_FLAGS} -DTRACE -trace")
endif()

# find MPI lib
find_package(MPI REQUIRED)

# link Chameleon
set(CHAMELEON_LINK_FLAGS "-debug -O3 -lstdc++ ${ADDITIONAL_VTUNE_FLAGS} -lm -fopenmp -lchameleon")


# compile the code
add_executable(fib fib.cpp)
target_link_libraries(fib ${MPI_CXX_LIBRARIES} ${CHAMELEON_LINK_FLAGS})
# set_property(TARGET fib PROPERTY CXX_STANDARD 17)
