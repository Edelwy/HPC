#ifndef LENNARD_JONES_H
#define LENNARD_JONES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Physical / integration constants. Shared by every variant so the
 * simulations are numerically identical and directly comparable. */
#define DT      0.002
#define SIGMA   1.0
#define EPSILON 1.0
#define R_CUT   2.5
#define JITTER  0.05

/* Animation (GIF) parameters. Only used when an output path is requested. */
#define FRAME_WIDTH           800
#define FRAME_HEIGHT          800
#define FRAME_EVERY           5
#define FRAME_PARTICLE_RADIUS 2
#define FRAME_DELAY           3

typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    double fx;
    double fy;
} Particle;

typedef struct {
    unsigned int n;
    const Particle *particles;
    double start_kinetic;
    double start_potential;
    double start_total;
    double final_kinetic;
    double final_potential;
    double final_total;
} SimulationResult;

/* Run-time settings shared by every variant. */
typedef struct {
    unsigned int n;       /* number of particles */
    unsigned int nsteps;  /* simulation steps */
    double box_size;      /* periodic box side */
    int track_energy;     /* if non-zero, print KE/PE/E every step */
    const char *gif_path; /* if non-NULL, write an animation (untimed use) */
} SimOptions;

/* Advances the system for opts->nsteps steps in place and returns the start
 * and final energies. Each variant implements this differently. */
SimulationResult run_simulation(Particle *particles, const SimOptions *opts);

#ifdef __cplusplus
}
#endif

#endif
