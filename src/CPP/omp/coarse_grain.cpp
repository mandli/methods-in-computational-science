
// OpenMP library header
#include <omp.h>

// Standard io stream and namespace
#include <iostream>
#include <string>
using namespace std;

// Math library
#include <math.h>

int main(int argc, char* argv[])
{
    int const n = pow(2, 10);
    double x[n], y[n];
    double norm, norm_thread, y_norm, y_norm_thread, true_norm;
    int num_threads, points_per_thread, thread_ID;
    int start_index, end_index;

    num_threads = 1;
    #ifdef _OPENMP
        num_threads = 15;
        omp_set_num_threads(num_threads);
        cout << "Using OpenMP with " << num_threads << " threads.\n";
    #endif

    points_per_thread = n / num_threads;
    cout << "Points per thread = " << points_per_thread << " / " << n << "\n";

    // Initialize x
    for (int i = 0; i < n; ++i)
        x[i] = (double)i;

    norm = 0.0;
    y_norm = 0.0;

    #pragma omp parallel private(norm_thread, start_index, end_index, thread_ID, y_norm_thread)
    {

        thread_ID = omp_get_thread_num();

        start_index = thread_ID * points_per_thread;
        end_index = (int)fmin((thread_ID + 1) * points_per_thread, n);
        // For only the last thread, give it all the extraneous points
        if (thread_ID == num_threads - 1)
            end_index += n % num_threads;

        string output = "Thread " + to_string(thread_ID) + " will take i = [";
        output += to_string(start_index) + ", " + to_string(end_index-1) + "].\n";    
        cout << output;

        norm_thread = 0.0;
        for (int i = start_index; i < end_index; ++i)
            norm_thread += fabs(x[i]);

        #pragma omp critical
            norm += norm_thread;

        #pragma omp barrier

        y_norm_thread = 0.0;
        for (int i = start_index; i < end_index; ++i)
        {
            y[i] = x[i] / norm;
            y_norm_thread += fabs(y[i]);
        }

        #pragma omp critical
            y_norm += y_norm_thread;

        #pragma omp barrier
    }

    true_norm = n * (n - 1) / 2;
    cout << "Norm of x = " << norm << ", n (n - 1) / 2 = " << true_norm << ".\n";
    cout << "Norm of y = " << y_norm << ".\n";

    return 0;
}