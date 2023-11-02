# Declaration for target and src
TARGET	:= main
SRC		:= main.cpp
LIBS_NAME := armadillo boost_serialization mlpack

# Compilers and linking libs
CXX := g++
CXXFLAGS += -std=c++14 -Wall -Wextra -O3 -DNDEBUG 
# LDFLAGS += -L/home/ctminh/projects/loc-libs/mlpack-4.0.0/install/lib
# INCLFLAGS := -I/home/ctminh/projects/loc-libs/mlpack-4.0.0/install/include
LIBS = -larmadillo -lboost_serialization -lmlpack -fopenmp

CLEAN_LIST := $(TARGET) $(OBJS)

default: all

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)