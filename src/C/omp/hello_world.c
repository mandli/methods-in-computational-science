
// OpenMP library header
#include <omp.h>

// Standard IO libraries
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int total_threads, thread_ID;

    // Fork into threads
    #pragma omp parallel
    {
        total_threads = omp_get_num_threads();
        thread_ID = omp_get_thread_num();
        printf("Hello world from %d of %d!\n", thread_ID, total_threads);
    }
    // Join all threads
    printf("Finalizing with %d of %d\n", thread_ID, total_threads);

    return 0;
}