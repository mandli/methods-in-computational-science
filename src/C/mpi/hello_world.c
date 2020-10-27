
// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int num_procs, rank;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total number of processes and this processes' rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Hellow world from process %d of %d.\n", rank, num_procs);

    MPI_Finalize();
    
    return 0;
}
