
// OpenMP library header
#include <omp.h>

// Standard io stream and namespace
#include <iostream>
using namespace std;

// Math library
#include <math.h>

int main()
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
        cout << "Using OpenMP with " << num_threads << ".\n";
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
    cout << "Norm of x = " << norm << ", n (n-1) / 2 = " << true_x_norm << ".\n";
    cout << "Norm of y should be 1, is " << y_norm << ".\n";

    return 0;
}