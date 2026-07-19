#ifndef LENNARD_JONES_COMMON_H
#define LENNARD_JONES_COMMON_H

#include "lennard_jones.h"
#include "gifenc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Places particles on a jittered grid, zeroes net momentum and scales the
 * velocities to the requested temperature. Same seed => identical start. */
int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature);

/* Periodic boundary wrap of all positions into [0, box_size). */
void wrap_positions(Particle *particles, unsigned int n, double box_size);

/* Potential shift so V(r_cut) = 0 (improves energy conservation). */
double compute_v_shift(void);

/* Kinetic energy of the system. */
double compute_ke(const Particle *particles, unsigned int n);

/* Potential energy of the system (half-matrix, r_cut cutoff). Host helper used
 * by the GPU variants, which keep positions on the device during the run. */
double compute_pe(const Particle *particles, unsigned int n, double box_size);

/* Animation helpers (no-ops unless a path is given by the caller). */
ge_GIF *lj_open_gif(const char *path);
void lj_render_frame(ge_GIF *gif, const Particle *particles, unsigned int n,
                     double box_size);

/* Dumps final particle positions (x y per line) to a text file for plotting. */
void dump_final_state(const char *path, const Particle *particles,
                      unsigned int n);

#ifdef __cplusplus
}
#endif

#endif
