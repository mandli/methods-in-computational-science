
// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

// Math library
#include <math.h>

int main(int argc, char* argv[])
{
    int const n = pow(2, 16);
    int num_threads, i, n_temp;
    double x, dx, y[n];

    #ifdef _OPENMP
        printf("How many threads to use? ");
        scanf("%d", &num_threads);
        omp_set_num_threads(num_threads);
        printf("Using OpenMP with %d threads.\n", num_threads);
    #endif

    dx = 1.0 / (double)(n + 1);
    #pragma parallel for private(x)
    for (i=0 ; i < n ; i ++)
    {
        x = i * dx;
        y[i] = exp(x) * cos(x) * sin(x) * sqrt(5.0 * x + 6.0);
    }

    printf("Filled vector y of length %d.\n", n);

    return 0;
}