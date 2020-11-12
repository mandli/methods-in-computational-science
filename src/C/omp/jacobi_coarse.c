/*
    Solve the Poisson problem
        u_{xx} = f(x)   x \in [a, b]
    with 
        u(a) = alpha, u(b) = beta
    using Jacobi iterations and OpenMP using fine grain parallelism.
*/

// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

int main(int argc, char* argv[])
{
    // Problem parameters
    double const a = 0.0, b = 1.0, alpha = 0.0, beta = 3.0;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2,16), PRINT_INTERVAL = 100;
    int num_points, N;
    double dx, tolerance, du_max, du_max_thread;

    // Work arrays
    double *x, *f, *u_old, *u;

    // OpenMP
    int num_threads, thread_ID, thread_num_points;
    int i, n, start_index, end_index;

    // File IO
    FILE *fp;

    num_threads = 1;
    #ifdef _OPENMP
    num_threads = 4;
    omp_set_num_threads(num_threads);
    printf("Using OpenMP with %d threads.\n", num_threads);
    #endif

    // Numerical discretization
    num_points = 100;
    dx = (b - a) / (num_points + 1);
    tolerance = 0.1 * pow(dx, 2);

    // Create work arrays
    x = malloc((num_points + 2) * sizeof(double));
    u = malloc((num_points + 2) * sizeof(double));
    u_old = malloc((num_points + 2) * sizeof(double));
    f = malloc((num_points + 2) * sizeof(double));

    // Set boundary conditions here before threads
    u[0] = alpha;
    u[num_points + 1] = beta;

    #pragma omp parallel private(thread_num_points, start_index, end_index, du_max_thread)
    {
        thread_ID = omp_get_thread_num();

        thread_num_points = (num_points + num_threads - 1) / num_threads;
        start_index = thread_ID * thread_num_points + 1;
        end_index = fmin((thread_ID + 1) * thread_num_points, num_points);
        thread_num_points = end_index - start_index + 1;

        for (int i = start_index; i < end_index + 1; ++i)
        {
            x[i] = (double) i * dx + a;
            f[i] = exp(x[i]);
            u[i] = alpha + x[i] * (beta - alpha);   
        }

        while (N < MAX_ITERATIONS)
        {
            for (int i = start_index; i < end_index + 1 + 2; ++i)
                u_old[i] = u[i];

            du_max_thread = 0.0;
            for (int i = start_index; i < end_index + 1; ++i)
            {
                u[i] = 0.5 * (u_old[i-1] + u_old[i+1] - pow(dx, 2) * f[i]);
                du_max_thread = fmax(du_max_thread, fabs(u[i] - u_old[i]));
            }
            
            if (N%PRINT_INTERVAL == 0)
                printf("After %d iterations, du_max = %f\n", N, du_max);

            #pragma omp critical
                du_max = fmax(du_max, du_max_thread);

            #pragma omp barrier

            if (du_max < tolerance)
                break;

            N++;
        }
    }

    // Output Results
    // Check for failure
    if (N >= MAX_ITERATIONS)
    {
        printf("*** Jacobi failed to converge!\n");
        printf("***   Reached du_max = %f\n", du_max);
        printf("***   Tolerance = %f\n", tolerance);
        return 1;
    }

    fp = fopen("jacobi_0.txt", "w");
    for (int i = 0; i < num_points + 2; ++i)
        fprintf(fp, "%f %f\n", x[i], u[i]);
    fclose(fp);

    return 0;
}