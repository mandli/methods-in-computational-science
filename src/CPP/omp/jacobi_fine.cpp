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
    int const MAX_ITERATIONS = pow(2,16), PRINT_INTERVAL = 100;

    // Numerical discretization
    int N, k;
    double dx, tolerance, du_max;
    N = 100;
    dx = (b - a) / (N + 1);
    tolerance = 0.1 * pow(dx, 2);

    // Work arrays
    double *x = new double[N + 2];
    double *u = new double[N + 2];
    double *u_old = new double[N + 2];
    double *f = new double[N + 2];

    // OpenMP setup
    int num_threads, thread_ID;
    num_threads = 1;
    #ifdef _OPENMP
        num_threads = 8;
        omp_set_num_threads(num_threads);
        cout << "Using OpenMP with " << num_threads << " threads.\n";
    #endif

    // Initialize arrays including initial guess
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < N + 2; ++i)
    {
        x[i] = (double) i * dx + a;
        f[i] = exp(x[i]);
        u[i] = alpha + x[i] * (beta - alpha);
    }

    // Primary algorithm loop
    k = 0;
    while (k < MAX_ITERATIONS)
    {
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < N + 2; ++i)
            u_old[i] = u[i];

        du_max = 0.0;
        #pragma omp parallel for schedule(dynamic, 10) reduction(max : du_max)
        for (int i = 1; i < N + 1; ++i)
        {
            u[i] = 0.5 * (u_old[i-1] + u_old[i+1] - pow(dx, 2) * f[i]);
            du_max = fmax(du_max, fabs(u[i] - u_old[i]));
        }
        
        if (k%PRINT_INTERVAL == 0)
            cout << "After " << k + 1 << " iterations, du_max = " << du_max << ".\n";

        if (du_max < tolerance)
            break;

        k++;
    }

    // Check for failure
    if (N >= MAX_ITERATIONS)
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