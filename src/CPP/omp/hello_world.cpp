
// OpenMP library header
#include <omp.h>

// Standard io stream and namespace
#include <iostream>
#include <string>
using namespace std;

int main() 
{
    int total_threads;

    // Fork into threads
    #pragma omp parallel
    {
        int thread_ID = omp_get_thread_num();

        if (thread_ID == 0)
        {
            total_threads = omp_get_num_threads();
        }
        #pragma omp barrier


        // Concatenate the string and output as one operation to avoid race
        // condition
        string output = "Hello, World from " + to_string(thread_ID) + " of " + to_string(total_threads) + "!\n";
        cout << output;
    }

    return 0;
}