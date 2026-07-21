#ifndef LENNARD_JONES_COMMON_H
#define LENNARD_JONES_COMMON_H

#include "gifenc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Constants shared by every variant so the
 * simulations are directly comparable. */
#define DT      0.002   // Time step.
#define SIGMA   1.0     // Distance between particles.
#define EPSILON 1.0     // Depth of the potential well.
#define R_CUT   2.5     // Cutoff radius for the potential.
#define JITTER  0.05    // Random jitter for particle placement.

/* GIF parameters. Only used when an output path is requested. */
#define FRAME_WIDTH           800
#define FRAME_HEIGHT          800
#define FRAME_EVERY           5
#define FRAME_PARTICLE_RADIUS 2
#define FRAME_DELAY           3

typedef struct {
    double x;   // x-coordinate of the particle.
    double y;   // y-coordinate of the particle.
    double vx;  // x-velocity of the particle.
    double vy;  // y-velocity of the particle.
    double fx;  // x-force on the particle.
    double fy;  // y-force on the particle.
} Particle; 

typedef struct {
    unsigned int n;             // Number of particles.
    const Particle *particles;  // Pointer to the particles array.
    double start_kinetic;       // Initial kinetic energy.
    double start_potential;     // Initial potential energy.
    double start_total;         // Initial total energy.
    double final_kinetic;       // Final kinetic energy.
    double final_potential;     // Final potential energy.
    double final_total;         // Final total energy.
} SimulationResult; 

/* Run-time settings shared by every variant. */
typedef struct {
    unsigned int n;       // Number of particles.
    unsigned int nsteps;  // Simulation steps.
    double box_size;      // Periodic box side.
    int track_energy;     // If non-zero, print KE/PE/E every step.
    const char *gif_path; // If non-NULL, write an GIF.
} SimOptions;

/* Each variant implements this differently. */
SimulationResult run_simulation(Particle *particles, const SimOptions *opts);

/* Places particles on a jittered grid. */
int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature);

/* Periodic boundary wrap of all positions into. */
void wrap_positions(Particle *particles, unsigned int n, double box_size);

/* Shift potential to ensure it goes to zero at the cutoff distance. */
double compute_v_shift(void);

/* Kinetic energy of the system. */
double compute_ke(const Particle *particles, unsigned int n);

/* Potential energy of the system (half-matrix, r_cut cutoff). Host helper used
 * by the GPU variants, which keep positions on the device during the run. */
double compute_pe(const Particle *particles, unsigned int n, double box_size);

/* GIF helpers. */
ge_GIF *open_gif(const char *path);
void render_frame(ge_GIF *gif, const Particle *particles, unsigned int n,
                     double box_size);

/* Dumps final particle positions to a text file. */
void dump_final_state(const char *path, const Particle *particles,
                      unsigned int n);

#ifdef __cplusplus
}
#endif

#endif
