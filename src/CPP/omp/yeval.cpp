// OpenMP library header
#include <omp.h>

// Standard io stream and namespace
#include <iostream>
using namespace std;

#include <math.h>

int main()
{
    int const n = pow(2, 8);
    int num_threads, i, n_temp;
    double x, dx, y[n], sum;

    #ifdef _OPENMP
        cout << "How many threads to use? ";
        cin >> num_threads;
        omp_set_num_threads(num_threads);
        cout << "Using OpneMP with " << num_threads << " threads.\n";
    #endif

    dx = 1.0 / (double)(n + 1);
    #pragma omp parallel for private(x)
    for (i=0 ; i < n ; i ++)
    {
        x = i * dx;
        y[i] = exp(x) * cos(x) * sin(x) * sqrt(5.0 * x + 6.0);
    }
    cout << "Filled vector y of length " << n << ".\n";

    #pragma omp parallel for reduction(+ : sum)
    for (i=0 ; i < n ; i ++)
        sum = sum + y[i];
    cout << sum;

    return 0;
}