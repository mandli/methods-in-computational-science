
// OpenMP library header
#include <omp.h>

// Timing support
#include <time.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

double matrix_multiply_test(int N, int method)
{
    int i, j, k;
    double A[N][N], B[N][N], C[N][N];

    clock_t start, end;

    // Create randomized matrices of the requested size
    for (i=1 ; i < N ; i++)
        for (j=1 ; j < N ; j++)
            A[i][j] = rand();
            B[i][j] = rand();
            C[i][j] = 0.0;

    // Start timer and compute matrix product
    start = clock();
    switch(method)
    {
        case 1:
            printf("No intrinsic version of matrix multiplication in C99,\n");
            printf("defaulting to triple loop.\n");
        case 2:
            for (i=0 ; i < N ; i++)
                for (j=0 ; j < N ; j++)
                    for (k=0 ; k < N ; k++)
                        C[i][j] += A[i][k] * B[k][j];
            break;
        case 3:
            for (i=0 ; i < N ; i++)
                for (j=0 ; j < N ; j++)
                    break;
                    // C[i, j] = DDOT(); //A[i, :] * B[:, j];
            break;
        case 4:
            // C = DGEMM()
            break;
        default:
            printf("*** ERROR *** Invalid multiplication method chosen!\n");
            return -1;
    }
    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
    int N, method, threads;
    N = 1000; method = 1, threads = 1;

    switch(argc)
    {
        case 4:
            threads = atoi(argv[3]);
        case 3:
            method = atoi(argv[2]);
        case 2:
            N = atoi(argv[1]);
            break;
    }

    printf("Timing for %dx%d matrices: %d s.", N, N, matrix_multiply_test(N, method));

    return 0;
}