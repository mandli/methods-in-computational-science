
// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

// Math library
#include <math.h>

int main(int argc, char* argv[])
{

    int const n = pow(2, 10);
    int i, thread_ID, num_threads;

    double x[n], y[n];
    double norm, y_norm;

    // Handle setting the number of threads
    num_threads = 1;
    #ifdef _OPENMP
        num_threads = 8;
        omp_set_num_threads(num_threads);
        printf("Using OpenMP with %d threads.", num_threads);
    #endif

    // Initialize x vector
    #pragma parallel for
    for (i = 0; i < n; ++i)
        x[i] = (double)i;

    norm = 0.0;
    y_norm = 0.0;

    // Fork into num_threads threads
    #pragma omp parallel private(i)
    {
        #pragma omp parallel for reduction(+ : norm)
        for (i = 0; i < n; ++i)
            norm += abs(x[i]);

        #pragma omp barrier // Not entirely needed (implicit)

        #pragma omp parallel for reduction(+ : y_norm)
        for (i = 0; i < n; ++i)
            y_norm += abs(y[i]);
    }

    printf("Norm of x = %f, n (n+1) / 2 = %f.", norm, n * (n + 1) / 2);
    printf("Norm of y should be 1, is %f.", y_norm);

    return 0;
}