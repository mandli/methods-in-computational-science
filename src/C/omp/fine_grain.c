
// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

// Math library
#include <math.h>

int main(int argc, char* argv[])
{

    int const n = pow(2, 10) + 1;
    int i, thread_ID, num_threads;

    double x[n], y[n];
    double norm, true_x_norm, y_norm;

    // Handle setting the number of threads
    num_threads = 1;
    #ifdef _OPENMP
        num_threads = 8;
        omp_set_num_threads(num_threads);
        printf("Using OpenMP with %d threads.\n", num_threads);
    #endif

    // Initialize x vector
    #pragma parallel for
    for (i = 0; i < n; ++i)
        x[i] = (double)i;

    norm = 0.0;
    y_norm = 0.0;

    #pragma omp parallel
    {
        #pragma omp for reduction(+ : norm)
        for (i = 0; i < n; ++i)
            norm = norm + fabs(x[i]);

        #pragma omp barrier  // Not srtictly needed

        #pragma omp for reduction(+ : y_norm)
        for (i = 0; i < n; ++i)
        {
            y[i] = x[i] / norm;
            y_norm = y_norm + fabs(y[i]);
        }
    }

    true_x_norm = n * (n - 1) / 2;
    printf("Norm of x = %f, n (n-1) / 2 = %f.\n", norm, true_x_norm);
    printf("Norm of y should be 1, is %f.\n", y_norm);

    return 0;
}