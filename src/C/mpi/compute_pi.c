// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

int main(int argc, char* argv[])
{
    int num_procs, rank, tag;
    MPI_Status status;

    double const pi = 3.1415926535897932384626433832795;
    
    int num_points, points_per_proc, start_index, end_index;
    double x, dx, pi_sum, pi_sum_proc, difference;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total number of processes and this processes' rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Rank 0 will ask for number of points
    if (rank == 0)
    {
        printf("How many points to use?\n");
        scanf("%d", &num_points);
    }
    // Broadcast the number of poitns
    MPI_Bcast(&num_points, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    dx = 1.0 / (double)num_points;

    // Determine how many points to handle with each proc
    points_per_proc = (num_points + num_procs - 1) / num_procs;
    // Only print out the number of points per proc by rank 0
    if (rank == 0)
        printf("Points per proc = %d\n.", points_per_proc);

    // Determine start and end indices for this rank's points
    start_index = rank * points_per_proc;
    end_index = (int)fmin((rank + 1) * points_per_proc, num_points) - 1;

    // Diagnostic - Print the intervals handled by each rank
    printf("Rank %d - (%d, %d)\n", rank, start_index, end_index);

    pi_sum_proc = 0.0;
    for (int i = start_index; i <= end_index; ++i)
    {
        x = (i - 0.5) * dx;
        pi_sum_proc += 1.0 / (1.0 + pow(x, 2));
    }

    MPI_Reduce(&pi_sum_proc, &pi_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, 
                MPI_COMM_WORLD);

    if (rank == 0)
    {
        pi_sum *= 4.0 * dx;
        difference = fabs(pi - pi_sum);
        printf("The approximation to pi is %f.\n", pi_sum);
        printf("Difference = %f", difference);
    }

    MPI_Finalize();
    
    return 0;
}
