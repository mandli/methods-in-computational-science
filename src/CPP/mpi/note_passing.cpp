// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
    int num_procs, rank;
    MPI_Status status;

    int tag;
    double message;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total number of processes and this processes' rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // If we only have one process then we are alone and cannot message anyone :(
    if (num_procs == 1)
    {
        cout << "Only one process used, no messages passed.\n";
        MPI_Finalize();
        return 0;
    }

    // Not super important for us
    tag = 42;

    // Start with rank 0 and start passing
    if (rank == 0)
    {
        message = 2.718281828;
        cout << "Process " << rank << " sending message = " << message << "\n";

        // We just send from the first rank
        MPI_Send(&message, 1, MPI_DOUBLE_PRECISION, 1, tag, MPI_COMM_WORLD);
    }
    else if (rank < num_procs - 1)
    {

        // We first recieve and then send from these middle ranks
        MPI_Recv(&message, 1, MPI_DOUBLE_PRECISION, rank-1, tag, MPI_COMM_WORLD, &status);
        cout << "Process " << rank << " recieved message = " << message << "\n";
        cout << "Process " << rank << " sending message = " << message << "\n";
        MPI_Send(&message, 1, MPI_DOUBLE_PRECISION, rank+1, tag, MPI_COMM_WORLD);
        
    }
    else if (rank == num_procs - 1)
    {
        // We just recieve in this last rank
        MPI_Recv(&message, 1, MPI_DOUBLE_PRECISION, rank-1, tag, MPI_COMM_WORLD, &status);

        cout << "Process " << rank << " recieved message = " << message << "\n";
    }

    MPI_Finalize();
    
    return 0;
}
