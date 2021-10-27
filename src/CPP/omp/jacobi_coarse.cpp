/*
    Solve the Poisson problem
        u_{xx} = f(x)   x \in [a, b]
    with 
        u(a) = alpha, u(b) = beta
    using Jacobi iterations and OpenMP using fine grain parallelism.
*/

// OpenMP library header
#include <omp.h>

#include <iostream>
#include <fstream>
using namespace std;

// Math library
#include <math.h>

int main()
{
    // Problem parameters
    double const a = 0.0, b = 1.0, alpha = 0.0, beta = 3.0;

    // Numerical parameters
    long int const MAX_ITERATIONS = pow(2,32);
    int const PRINT_INTERVAL = 1000;

    // Numerical discretization
    int N, k;
    double dx, tolerance, du_max;
    N = 1000;
    dx = (b - a) / (N + 1);
    tolerance = 0.1 * pow(dx, 2);

    // Work arrays
    double *x = new double[N + 2];
    double *u = new double[N + 2];
    double *u_old = new double[N + 2];
    double *f = new double[N + 2];

    // OpenMP
    int num_threads, thread_N, start_index, end_index, thread_ID;
    double du_max_thread;
    string output;
    num_threads = 1;
    #ifdef _OPENMP
        num_threads = 8;
        omp_set_num_threads(num_threads);
        cout << "Using OpenMP with " << num_threads << " threads.\n";
    #endif

    // Parallel section
    k = 0;
    #pragma omp parallel private(output, thread_ID, du_max_thread, thread_N, start_index, end_index)
    {
        thread_ID = omp_get_thread_num();

        thread_N = (N + num_threads - 1) / num_threads;
        start_index = thread_ID * thread_N + 1;
        end_index = fmin((thread_ID + 1) * thread_N, N);
        thread_N = end_index - start_index + 1;

        output = "Thread " + to_string(thread_ID) + " will take (";
        output += to_string(start_index) + ", " + to_string(end_index) + ")\n";
        cout << output;

        // Initialize arrays including initial guess
        for (int i = start_index; i < end_index + 1; ++i)
        {
            x[i] = (double) i * dx + a;
            f[i] = exp(x[i]);
            u[i] = alpha + x[i] * (beta - alpha);   
        }

        // Fix up end points
        #pragma omp single nowait
        {
            x[0] = a;
            x[N + 1] = b;
            f[0] = exp(x[0]);
            f[N + 1] = exp(x[N + 1]);
            u[0] = alpha;
            u_old[0] = u[0];
            u[N + 1] = beta;
            u_old[N + 1] = u[N + 1];
        }

        // Primary algorithm loop
        while (k < MAX_ITERATIONS)
        {
            for (int i = start_index; i < end_index + 1; ++i)
                u_old[i] = u[i];

            #pragma omp barrier

            du_max_thread = 0.0;
            for (int i = start_index; i < end_index + 1; ++i)
            {
                u[i] = 0.5 * (u_old[i-1] + u_old[i+1] - pow(dx, 2) * f[i]);
                du_max_thread = fmax(du_max_thread, fabs(u[i] - u_old[i]));
            }
            
            du_max = 0.0;
            #pragma omp critical
            {
                du_max = fmax(du_max, du_max_thread);
            }

            #pragma omp barrier

            #pragma omp single nowait
            {
                if (k%PRINT_INTERVAL == 0)
                {
                    output = "After " + to_string(k + 1);
                    output += " iterations, du_max = " + to_string(du_max) + "\n"; 
                    cout << output;
                }
                k++;
            }

            if (du_max < tolerance)
                break;
        }

    }

    // Check for failure
    if (k >= MAX_ITERATIONS)
    {
        cout << "*** Jacobi failed to converge!\n";
        cout << "***   Reached du_max = " << du_max << "\n";
        cout << "***   Tolerance = " << tolerance << "\n";
        return 1;
    }

    // Output Results
    ofstream fp("jacobi_0.txt");
    for (int i = 0; i < N + 2; ++i)
        fp << x[i] << " " << u[i] << "\n";
    fp.close();

    return 0;
}