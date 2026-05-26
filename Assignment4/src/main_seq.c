#define _POSIX_C_SOURCE 199309L /* CLOCK_MONOTONIC for std=c99 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lenia.h"

#define DEFAULT_N      128
#define DEFAULT_STEPS  100
#define DT             0.1
#define KERNEL_SIZE    26
#define NUM_ORBIUMS    2

int main(int argc, char *argv[])
{
    int n = DEFAULT_N;
    int steps = DEFAULT_STEPS;
    struct lenia_opts opts = { NULL, NULL, 1 };

    int i = 1;
    if (i < argc && argv[i][0] != '-') { n = atoi(argv[i]); i++; }
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--steps"))      steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gif"))   opts.gif_path = argv[++i];
        else if (!strcmp(argv[i], "--final")) opts.final_path = argv[++i];
        else if (!strcmp(argv[i], "--halo"))  opts.halo_steps = atoi(argv[++i]);
    }

    struct orbium_coo orbiums[NUM_ORBIUMS] = { {0, n / 3, 0}, {n / 3, 0, 180} };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double *world = evolve_lenia(n, n, steps, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS, &opts);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Execution time: %.3f\n", elapsed);

    free(world);
    return 0;
}
