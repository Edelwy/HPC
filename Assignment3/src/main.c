#define _POSIX_C_SOURCE 199309L /* CLOCK_MONOTONIC for std=c99 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lennard_jones.h"
#include "lennard_jones_common.h"

#define DEFAULT_N     1000
#define DEFAULT_STEPS 5000
#define DENSITY       0.95
#define TEMPERATURE   0.5
#define SEED          42

int main(int argc, char **argv) {
    unsigned int n = DEFAULT_N;
    unsigned int nsteps = DEFAULT_STEPS;
    int track_energy = 0;
    const char *gif_path = NULL;
    const char *final_path = NULL;

    /* usage: ./lj.out [N] [--steps S] [--energy] [--gif path] [--final path] */
    int i = 1;
    if (i < argc && argv[i][0] != '-') {
        n = (unsigned int)strtoul(argv[i], NULL, 10);
        i++;
    }
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--steps"))       nsteps = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--energy")) track_energy = 1;
        else if (!strcmp(argv[i], "--gif"))    gif_path = argv[++i];
        else if (!strcmp(argv[i], "--final"))  final_path = argv[++i];
    }

    /* box size follows from particle count and target density */
    double particle_box_size = ceil(sqrt((double)n / DENSITY));
    double box_size = (4.0 / 3.0) * particle_box_size;
    double box_fraction = particle_box_size / box_size;

    Particle *particles = (Particle *)calloc(n, sizeof(Particle));
    initialize_particles(particles, n, box_size, box_fraction, SEED, TEMPERATURE);

    SimOptions opts;
    opts.n = n;
    opts.nsteps = nsteps;
    opts.box_size = box_size;
    opts.track_energy = track_energy;
    opts.gif_path = gif_path;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    SimulationResult result = run_simulation(particles, &opts);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* final-state visualisation is written after timing so it is excluded */
    if (final_path) dump_final_state(final_path, particles, n);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Final KE: %10.4f | delta: %+.4f\n", result.final_kinetic, result.final_kinetic - result.start_kinetic);
    printf("Final PE: %10.4f | delta: %+.4f\n", result.final_potential, result.final_potential - result.start_potential);
    printf("Final E:  %10.4f | delta: %+.4f\n", result.final_total, result.final_total - result.start_total);
    printf("Execution time: %.3f\n", elapsed);

    free(particles);
    return 0;
}
