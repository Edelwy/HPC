#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "lenia.h"

#define DEFAULT_N      128
#define DEFAULT_STEPS  100
#define DT             0.1
#define KERNEL_SIZE    26
#define NUM_ORBIUMS    2

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

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

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double *world = evolve_lenia(n, n, steps, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS, &opts);
    double t1 = MPI_Wtime();

    /* We want the slowest time, so all ranks send time to rank 0, max returned. */
    double local = t1 - t0, elapsed;
    MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Execution time: %.3f\n", elapsed);

    free(world);
    MPI_Finalize();
    return 0;
}
