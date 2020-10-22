
// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

// Math library
#include <math.h>

int main(int argc, char* argv[])
{
    int const n = pow(2, 3);
    double x[n], y[n];
    double norm, norm_thread, y_norm, y_norm_thread, true_norm;
    int num_threads, points_per_thread, thread_ID;
    int start_index, end_index;

    num_threads = 1;

    points_per_thread = n;
    printf("Points per thread = %d / %d\n", points_per_thread, n);

    // Initialize x
    for (int i = 0; i < n; ++i)
        x[i] = (double)i;

    norm = 0.0;
    y_norm = 0.0;

    thread_ID = 0;
    start_index = thread_ID * points_per_thread;
    end_index = (int)fmin((thread_ID + 1) * points_per_thread, n);

    printf("Thread %d will take i = (%d, %d).\n", thread_ID, start_index, end_index);

    norm_thread = 0.0;
    for (int i = start_index; i < end_index; ++i)
        norm_thread += fabs(x[i]);

    norm += norm_thread;
    printf("norm updated to %f\n", norm);

    y_norm_thread = 0.0;
    for (int i = start_index; i < end_index; ++i)
    {
        y[i] = x[i] / norm;
        y_norm_thread += fabs(y[i]);
    }

    y_norm += y_norm_thread;
    printf("norm updated to %f\n", y_norm);

    true_norm = n * (n - 1) / 2;
    printf("Norm of x = %f, n (n - 1) / 2 = %f.\n", norm, true_norm);
    printf("Norm of y = %f.\n", y_norm);

    return 0;
}