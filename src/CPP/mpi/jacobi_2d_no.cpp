// Standard IO libraries
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

int main()
{
    // Problem paramters
    double const pi = 3.141592654;
    double const a = 0.0, b = pi;

    // Numerical parameters
    int const MAX_ITERATIONS = pow(2, 16), PRINT_INTERVAL = 10;
    int N, k;
    double x, y, dx, dy, tolerance, du_max;

    // Discretization
    N = 100;
    dx = (pi - 0) / ((double)(N + 1));
    dy = dx;
    tolerance = 0.01 * pow(dx, 2);

    // Allocate work arrays
    double *buffer = new double[N + 2];
    double **u = new double*[N + 2];
    double **u_old = new double*[N + 2];
    double **f = new double*[N + 2];
    for (int i = 0; i < N + 2; ++i)
    {
        u[i] = new double[N + 2];
        u_old[i] = new double[N + 2];
        f[i] = new double[N + 2];
    }

    // For reference, (x_i, y_j) u[i][j] 
    // so that i references columns and j rows
    // Initialize arrays - fill boundaries
    for (int i = 0; i < N + 2; ++i)
    {
        x = dx * (double) i + a;
        for (int j = 0; j < N + 2; ++j)
        {
            y = dy * (double) j + a;
            f[i][j] = -20.0 * sin(x) * cos(3.0 * y);
            u[i][j] = 1.0;
        }
    }

    // Set boundaries
    // Top and Bottom boundary
    for (int i = 0; i < N+2; ++i)
    {
        x = dx * (double) i + a;
        u[i][0] = 2.0 * sin(x);
        u[i][N+1] = -2.0 * sin(x);
    }
    // Left and Right boundary
    for (int j = 0; j < N+2; ++j)
    {
        u[0][j] = 0.0;
        u[N+1][j] = 0.0;
    }

    // Inital copy into u_old
    for (int i = 0; i < N + 2; ++i)
        for (int j = 0; j < N + 2; ++j)
            u_old[i][j] = u[i][j];

    /* Jacobi Iterations */
    k = 0;
    du_max = 0.0;
    for (int i = 1; i < N + 1; ++i)
        for (int j = 1; j < N + 1; ++j)
            du_max = fmax(du_max, fabs(u[i][j] - u_old[i][j]));
    cout << "After " << k << " iterations, du_max = " << du_max << "\n";
    while (k < MAX_ITERATIONS)
    {
        k++;

        du_max = 0.0;
        for (int i = 1; i < N + 1; ++i)
        {
            for (int j = 1; j < N + 1; ++j)
            {
                u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1] - pow(dx, 2) * f[i][j]);
                du_max = fmax(du_max, fabs(u[i][j] - u_old[i][j]));
            }
        }

        if (k%PRINT_INTERVAL == 0)
            cout << "After " << k << " iterations, du_max = " << du_max << "\n";

        if (du_max < tolerance)
            break;

        // Copy into old data for next loop
        for (int i = 1; i < N + 1; ++i)
            for (int j = 1; j < N + 1; ++j)
                u_old[i][j] = u[i][j];
    }

    // Output Results
    // Check for failure
    if (k >= MAX_ITERATIONS)
    {
        cout << "*** Jacobi failed to converge!\n";
        cout << "***   Reached du_max = " << du_max << "\n";
        cout << "***   Tolerance = " << tolerance << "\n";
        return 1;
    }

    // Write each row from bottom to top
    // Each rank writes out to it's own file and post-processing will handle opening
    // up all the files - rank determines the file names
    string file_name = "jacobi_" + to_string(0) + ".txt";
    ofstream fp(file_name);

    for (int i = 0; i < N + 2; ++i)
    {
        for (int j = 0; j < N + 2; ++j)
            fp << u[i][j] << " ";
        fp << "\n";
    }

    fp.close();

    return 0;
}