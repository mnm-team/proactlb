# Compilers
CC=gcc
MPICC=mpicc

ifdef I_MPI_ROOT
MPICC=mpiicc
endif

# Nbody parameters
BIGO?=N2
BS?=2048

# Preprocessor flags
CPPFLAGS=-Isrc -DBIGO=$(BIGO) -D_BIGO_$(BIGO) -DBLOCK_SIZE=$(BS)

# Compiler flags
CFLAGS=-O3 -std=gnu11

# Linker flags
LDFLAGS=-lrt -lm

SEQ_NONBLK_SOURCES=		\
    ./src/seq/common.c	\
    ./src/seq/utils.c	\
    ./src/seq/main.c

MPI_BLK_SOURCES=		\
    ./src/mpi/common.c	\
    ./src/mpi/utils.c	\
    ./src/mpi/main.c

BIN=	\
	01.nbody_seq_nonblk.$(BIGO).bin \
	03.nbody_mpi_blk.$(BIGO).bin

all: $(BIN)

01.nbody_seq_nonblk.$(BIGO).bin: $(SEQ_NONBLK_SOURCES) ./src/seq/nbody_seq_nonblk.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o ./bin/$@ $^ $(LDFLAGS)

03.nbody_mpi_blk.$(BIGO).bin: $(MPI_BLK_SOURCES) ./src/mpi/nbody_mpi_blk.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o ./bin/$@ $^ $(LDFLAGS)

# Run test cases
run_seq_nonblk:
	./bin/01.nbody_seq_nonblk.$(BIGO).bin -p 4096 -t 10

run_mpi_blk:
	mpirun -np 2 ./bin/03.nbody_mpi_blk.$(BIGO).bin -p 4096 -t 10

clean:
	rm -f *.o ./bin/*.bin

