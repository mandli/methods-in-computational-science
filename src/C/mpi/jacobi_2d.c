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
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

int main(int argc, char *argv[])
{

    // Problem paramters
    double const pi = 3.141592654;
    double const a = 0.0, b = pi;

    // MPI Variables
    int num_procs, rank, tag, rank_num_points, start_index, end_index;
    MPI_Status status;
    MPI_Request request;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2, 16), PRINT_INTERVAL = 1000;
    int N, num_points;
    double x, y, dx, dy, tolerance, du_max;

    // Work arrays
    double **u, **u_old, **f;

    // IO
    FILE *fp;
    char file_name[20];

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        // Discretization
        num_points = 100;
        dx = (pi - 0) / ((double)(num_points + 1));
        dy = dx;
        tolerance = 0.1 * pow(dx, 2);

        // Diagnostic - Print the intervals handled by each rank
        printf("Rank %d: %d - (%d, %d)\n", rank, rank_num_points, start_index, end_index);

        // Allocate memory for work space
        u = (double**)malloc((num_points + 2) * sizeof(double * ));
        u_old = (double**)malloc((num_points + 2) * sizeof(double));
        f = (double**)malloc((num_points + 2) * sizeof(double));
        for (int i = 0; i < num_points + 2; ++i)
        {
            u[i] = (double *)malloc((num_points + 2) * sizeof(double));
            u_old[i] = (double *)malloc((num_points + 2) * sizeof(double));
            f[i] = (double *)malloc((num_points + 2) * sizeof(double));
        }

        // Initialize arrays - fill boundaries
        for (int i = 0; i < num_points + 2; ++i)
        {
            x = dx * (double) i + a;
            for (int j = 0; j < num_points + 2; ++j)
            {
                y = dy * (double) j + a;
                f[i][j] = -20.0 * sin(x) * cos(3.0 * y);
                u[i][j] = -20.0;
            }
        }

        // Set boundaries
        for (int i = 0; i < num_points + 2; ++i)
        {   
            x = dx * (double) i + a;
            u[i][0] = 2.0 * sin(x);
            u[i][num_points + 1] = -2.0 * sin(x);
        }
        for (int j = 0; j < num_points + 2; ++j)
        {
            u[0][j] = 0.0;
            u[num_points + 1][j] = 0.0;
        }

        while (N < MAX_ITERATIONS)
        {
            for (int i = 0; i < num_points + 2; ++i)
                for (int j = 0; j < num_points + 2; ++j)
                    u_old[i][j] = u[i][j];

            du_max = 0.0;
            for (int i = 1; i < num_points + 1; ++i)
                for (int j = 1; j < num_points + 1; ++j)
                {
                    u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1] - pow(dx, 2) * f[i][j]);
                    du_max = fmax(du_max, fabs(u[i][j] - u_old[i][j]));
                }

            if (N%PRINT_INTERVAL == 0)
                printf("After %d iterations, du_max = %f\n", N, du_max);


            if (du_max < tolerance)
                break;

            N++;
        }

        // Output Results
        // Check for failure
        if (N >= MAX_ITERATIONS)
        {
        
            if (rank == 0)
            {
                printf("*** Jacobi failed to converge!\n");
                printf("***   Reached du_max = %f\n", du_max);
                printf("***   Tolerance = %f\n", tolerance);
                
                MPI_Finalize();
                return 1;
            }
        }

        // Write each row from bottom to top
        sprintf(file_name, "jacobi_%d.txt", rank);
        fp = fopen(file_name, "w");

        for (int i = 0; i < num_points + 2; ++i)
        {
            for (int j = 0; j < num_points + 2; ++j)
                fprintf(fp, "%f ", u[i][j]);
            fprintf(fp, "\n");
        }
        
        fclose(fp);
    }

    MPI_Finalize();

    return 0;
}