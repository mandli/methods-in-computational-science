// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
    int num_procs, rank;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total number of processes and this processes' rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << "Hello world from process " << rank << " of " << num_procs << ".\n";

    MPI_Finalize();
    
    return 0;
}
