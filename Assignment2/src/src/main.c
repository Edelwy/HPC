#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lenia.h"

#define N 512
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

#define PRINT_STATE 0

void final_state(double* world, int n)
{
    FILE* fp = fopen("final_state.txt", "w");
    if (fp == NULL) 
        return;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%f ", world[i * n + j]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    // Get the grid size from the command line.
    int n = N;
    if (argc > 1)
        n = atoi(argv[1]);

    // Place two orbiums in the world with different angles. (y, x, angle)
    // Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
    struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, n / 3, 0}, {n / 3, 0, 180}};

    double start = omp_get_wtime();
    double *world = evolve_lenia(n, n, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    double stop = omp_get_wtime();
    printf("Execution time: %.3f\n", stop - start);

#if PRINT_STATE // Export the final state to a text file.
    final_state(world, n);
#endif

    free(world);
    return 0;
}
