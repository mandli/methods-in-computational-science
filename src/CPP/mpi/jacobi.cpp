/*
    Solve the Poisson problem
        u_{xx} = f(x)   x \in [a, b]
    with 
        u(a) = alpha, u(b) = beta
    using Jacobi iterations and MPI.

    Note that the stencil has the following layout
        [* 0, 1, 2, ... n-2, n-1, *]
    Since the local indices into the arrays will always be
    the same, the [start_index, end_index] will actually refer
    to the global index space so that x[i] can be computed.
*/

// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

int main(int argc, char* argv[])
{
    // MPI Variables
    int num_procs, rank, tag, rank_num_points, start_index, end_index;
    MPI_Status status;
    MPI_Request request;

    // Problem paramters
    double const alpha = 0.0, beta = 3.0, a = 0.0, b = 1.0;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2, 16), PRINT_INTERVAL = 10;
    int N, num_points;
    double x, dx, tolerance, du_max, du_max_proc;

    // IO
    bool serial_output = true;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Discretization
    num_points = 19;
    dx = (b - a) / ((double)(num_points + 1));
    tolerance = 0.1 * pow(dx, 2);

    // Organization of local process (rank) data
    rank_num_points = (num_points + num_procs - 1) / num_procs;
    start_index = rank * rank_num_points + 1;
    end_index = fmin((rank + 1) * rank_num_points, num_points);
    rank_num_points = end_index - start_index + 1;

    // Diagnostic - Print the intervals handled by each rank
    cout << "Rank " << rank << ": " << rank_num_points << " - (" << start_index << ", " << end_index << ")\n";

    // Allocate memory for work space - allocate extra two points for halo and
    // boundaries
    double *u = new double[rank_num_points + 2];
    double *u_old = new double[rank_num_points + 2];
    double *f = new double[rank_num_points + 2];

    // Initialize arrays - fill boundaries
    for (int i = 0; i < rank_num_points + 2; ++i)
    {
        x = dx * (double) (i + start_index - 1) + a;
        f[i] = exp(x);                     // RHS function
        u[i] = alpha + x * (beta - alpha); // Initial guess
    }

    /* Jacobi Iterations */
    N = 0;
    while (N < MAX_ITERATIONS)
    {
        // Copy u into u_old
        for (int i = 0; i < rank_num_points + 2; ++i)
            u_old[i] = u[i];


        // Send data to the right (tag = 1)
        if (rank < num_procs - 1)
            MPI_Isend(&u_old[rank_num_points], 1, MPI_DOUBLE_PRECISION, rank + 1, 1, MPI_COMM_WORLD, &request);
        // Send data to the left (tag = 2)
        if (rank > 0)
            MPI_Isend(&u_old[1], 1, MPI_DOUBLE_PRECISION, rank - 1, 2, MPI_COMM_WORLD, &request);

        // Receive data from the right (tag = 1)
        if (rank < num_procs - 1)
            MPI_Recv(&u_old[rank_num_points + 1], 1, MPI_DOUBLE_PRECISION, rank + 1, 2, MPI_COMM_WORLD, &status);
        // Receive data from the left (tag = 2)
        if (rank > 0)
            MPI_Recv(&u_old[0], 1, MPI_DOUBLE_PRECISION, rank - 1, 1, MPI_COMM_WORLD, &status);

        /* Apply Jacobi */
        du_max_proc = 0.0;
        for (int i = 1; i < rank_num_points + 1; ++i)
        {
            u[i] = 0.5 * (u_old[i-1] + u_old[i+1] - pow(dx, 2) * f[i]);
            du_max_proc = fmax(du_max_proc, fabs(u[i] - u_old[i]));
        }
        /* ------------ */

        // Find global maximum change in solution - acts as an implicit barrier
        MPI_Allreduce(&du_max_proc, &du_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

        // Periodically report progress
        if (rank == 0)
            if (N%PRINT_INTERVAL == 0)
                printf("After %d iterations, du_max = %f\n", N, du_max);

        // All processes have the same du_max and should check for convergence
        if (du_max < tolerance)
            break;
        N++;
    }

    cout << "Rank " << rank << " finished after " << N << " iterations, du_max = " << du_max << ".\n";

    // Output Results
    // Check for failure
    if (N >= MAX_ITERATIONS)
    {
        if (rank == 0)
        {
            cout << "*** Jacobi failed to converge!\n";
            cout << "***   Reached du_max = " << du_max << "\n";
            cout << "***   Tolerance = " << tolerance << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Synchronize here before output
    MPI_Barrier(MPI_COMM_WORLD);

    // Demonstration of two approaches to writing out files:
    if (serial_output)
    {
        // Each rank will open the same file for writing in turn waiting for
        // the previous rank to tell it to go.
        if (rank == 0)
        {
            // Setup file for writing and let rank + 1 to go
            ofstream fp("jacobi_0.txt");

            for (int i = 1; i < rank_num_points + 1; ++i)
            {
                x = (double) (i + start_index - 1) * dx  + a;
                fp << x << " " << u[i] << "\n";
            }
            // Record right boundary if this is only process
            if (num_procs == 1)
                fp << b << " " << u[rank_num_points + 1];   

            fp.close();

            // Notify next rank
            if (num_procs > 1)
                MPI_Send(MPI_BOTTOM, 0, MPI_INTEGER, 1, 4, MPI_COMM_WORLD);
        }
        else
        {
            // Wait to go for the previous rank, write out, and let the next rank know to go.
            MPI_Recv(MPI_BOTTOM, 0, MPI_INTEGER, rank - 1, 4, MPI_COMM_WORLD, &status);

            // Begin writing out
            ofstream fp("jacobi_0.txt", ofstream::app);
            for (int i = 1; i < rank_num_points + 1; ++i)
            {
                x = (double) (i + start_index - 1) * dx  + a;
                fp << x << " " << u[i] << "\n";
            }

            // This is the last process, write out boundary
            if (rank == num_procs - 1)
                fp << b << " " << u[rank_num_points + 1];  

            fp.close();

            // Send signal to next rank if necessary
            if (rank < num_procs - 1)
                MPI_Send(MPI_BOTTOM, 0, MPI_INTEGER, rank + 1, 4, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Each rank writes out to it's own file and post-processing will handle opening
        // up all the files - rank determines the file names
        string file_name = "jacobi_" + to_string(rank) + ".txt";
        ofstream fp(file_name);

        if (rank == 0)
            fp << 0.0 << " " << u[0] << "\n";

        for (int i = 1; i < rank_num_points + 1; ++i)
        {
            x = (double) (i + start_index - 1) * dx  + a;
            fp << x << " " << u[i] << "\n";
        }

        if (rank == num_procs - 1)
            fp << b << " " << u[rank_num_points + 1];

        fp.close();
    }

    MPI_Finalize();

    return 0;
}
