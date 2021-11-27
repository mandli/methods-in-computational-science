/*
    Solve the Poisson problem
        u_{xx} + u_{yy} = f(x, y)   x \in \Omega = [0, pi] x [0, pi]
    with 
        u(0, y) = u(pi, y) = 0
        u(x, 0) = 2 sin x
        u(x, pi) = -2 sin x
    and
        f(x, y) = -20 sin x cos 3 y
    using Jacobi iterations and MPI.  For simplicity we will assume that we 
    will use a uniform discretization.
*/

// MPI Library
#include "mpi.h"

// Standard IO libraries
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

int main(int argc, char *argv[])
{

    // Problem paramters
    double const pi = 3.141592654;
    double const a = 0.0, b = pi;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2, 16), PRINT_INTERVAL = 100;
    int N, k;
    double x, y, dx, dy, tolerance, du_max;

    // MPI Variables
    int num_procs, rank, tag, rank_N, start_index, end_index;
    double du_max_proc;
    MPI_Status status;
    MPI_Request request;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TODO: Handle the case of only one process
    if (num_procs == 1)
    {
        cout << "This program only handles more than one process.\n";
        MPI_Finalize();
        return 0;
    }

    // Discretization
    N = 100;
    dx = (pi - 0) / ((double)(N + 1));
    dy = dx;
    tolerance = 0.1 * pow(dx, 2);

    // Organization of local process (rank) data
    rank_N = (N + num_procs - 1) / num_procs;
    start_index = rank * rank_N + 1;
    end_index = fmin((rank + 1) * rank_N, N);
    rank_N = end_index - start_index + 1;

    // Diagnostic - Print the intervals handled by each rank
    cout << "Rank " << rank << ": " << rank_N << " - (" << start_index << ", " << end_index << ")\n";

    // Allocate work arrays
    double *send_buffer = new double[N];
    double *recv_buffer = new double[N];
    double **u = new double*[N + 2];
    double **u_old = new double*[N + 2];
    double **f = new double*[N + 2];
    for (int i = 0; i < N + 2; ++i)
    {
        u[i] = new double[rank_N + 2];
        u_old[i] = new double[rank_N + 2];
        f[i] = new double[rank_N + 2];
    }

    // For reference, (x_i, y_j) u[i][j] 
    // so that i references columns and j rows
    // Initialize arrays - fill boundaries
    for (int i = 0; i < N + 2; ++i)
    {
        x = dx * (double) i + a;
        for (int j = 0; j < rank_N + 2; ++j)
        {
            y = dy * (double) (j + start_index) + a;
            f[i][j] = -20.0 * sin(x) * cos(3.0 * y);
            u[i][j] = 1.0;
        }
    }

    // Set boundaries
    // Set bottom (rank 0)
    if (rank == 0)
    {
        for (int i = 0; i < N + 2; ++i)
        {
            x = dx * (double) i + a;
            u[i][0] = 2.0 * sin(x);
        }
    }
    // Set top (rank num_procs - 1)
    if (rank == num_procs - 1)
    {
        for (int i = 0; i < N + 2; ++i)
        {
            x = dx * (double) i + a;
            u[i][rank_N + 1] = -2.0 * sin(x);
        }
    }
    // Set left and right (all ranks)
    for (int j = 0; j < rank_N + 2; ++j)
    {   
        u[0][j] = 0.0;
        u[N + 1][j] = 0.0;
    }

    // Inital copy into u_old - note that this does not require communication
    // as we know all values on each process at this point
    for (int i = 0; i < N + 2; ++i)
        for (int j = 0; j < rank_N + 2; ++j)
            u_old[i][j] = u[i][j];

    /* Jacobi Iterations */
    k = 0;
    while (k < MAX_ITERATIONS)
    {
        k++;

        du_max_proc = 0.0;
        for (int i = 1; i < N + 1; ++i)
        {
            for (int j = 1; j < rank_N + 1; ++j)
            {
                u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1] - pow(dx, 2) * f[i][j]);
                du_max_proc = fmax(du_max_proc, fabs(u[i][j] - u_old[i][j]));
            }
        }

        // Final global max change in solution
        MPI_Allreduce(&du_max_proc, &du_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0)
            if (N%PRINT_INTERVAL == 0)
                cout << "After " << k << " iterations, du_max = " << du_max << "\n";

        if (du_max < tolerance)
            break;

        // Copy into old data that we have
        for (int i = 1; i < N + 1; ++i)
            for (int j = 1; j < rank_N + 1; ++j)
                u_old[i][j] = u[i][j];

        // Communicate data
        // Send data up (tag = 1)
        if (rank < num_procs - 1)
        {
            for (int i = 1; i < N + 1; ++i)
                send_buffer[i - 1] = u_old[i][rank_N];
            MPI_Isend(send_buffer, N, MPI_DOUBLE_PRECISION, rank + 1, 1, MPI_COMM_WORLD, &request);
        }
        // Send data down (tag = 2)
        if (rank > 0)
        {
            for (int i = 1; i < N + 1; ++i)
                send_buffer[i - 1] = u_old[i][1];
            MPI_Isend(send_buffer, N, MPI_DOUBLE_PRECISION, rank - 1, 2, MPI_COMM_WORLD, &request);
        }

        // Receive data from above (tag = 2)
        if (rank < num_procs - 1)
        {
            MPI_Recv(recv_buffer, N, MPI_DOUBLE_PRECISION, rank + 1, 2, MPI_COMM_WORLD, &status);
            for (int i = 1; i < N + 1; ++i)
                u_old[i][rank_N + 1] = recv_buffer[i - 1];
        }

        // Receive data from below (tag = 1)
        if (rank > 0)
        {
            MPI_Recv(recv_buffer, N, MPI_DOUBLE_PRECISION, rank - 1, 1, MPI_COMM_WORLD, &status);
            for (int i = 1; i < N + 1; ++i)
                u_old[i][0] = recv_buffer[i - 1];
        }
    }

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

    // Write each row from bottom to top
    // Each rank writes out to it's own file and post-processing will handle opening
    // up all the files - rank determines the file names
    string file_name = "jacobi_" + to_string(rank) + ".txt";
    ofstream fp(file_name);

    if (rank == 0)
    {
        // Write out bottom boundary
        for (int i = 0; i < N + 2; ++i)
            fp << u[i][0] << " ";
        fp << "\n";
    }

    for (int j = 1; j < rank_N + 1; ++j)
    {
        for (int i = 0; i < N + 2; ++i)
            fp << u[i][j] << " ";
        fp << "\n";
    }
    
    if (rank == num_procs - 1)
    {
        // Write out top boundary
        for (int i = 0; i < N + 2; ++i)
            fp << u[i][rank_N + 1] << " ";
    }

    fp.close();

    MPI_Finalize();

    return 0;
}