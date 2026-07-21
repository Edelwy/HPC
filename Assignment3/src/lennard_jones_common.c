#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lennard_jones_common.h"

static double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size,
                         double placement_fraction, unsigned int seed,
                         double temperature) {
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // Place particles on a grid with jitter and random velocities.
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // Remove net momentum and accumulate kinetic energy.
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx +
                     particles[k].vy * particles[k].vy);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // Scale velocities to the requested temperature.
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);
        if (wx < 0.0) wx += box_size;
        if (wy < 0.0) wy += box_size;
        p->x = wx;
        p->y = wy;
    }
}

double compute_v_shift(void) {
    const double r_cut = R_CUT * SIGMA;
    const double sr2 = (SIGMA * SIGMA) / (r_cut * r_cut);
    const double sr6 = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;
    return 4.0 * EPSILON * (sr12 - sr6);
}

// Mainly used by the GPU variants, forces run on the device.
// Potential energy is not computed there, so when we report energy on every step,
// we use this helper function to compute it on the host.
// Also used for start and end potential energy calculations.
double compute_pe(const Particle *particles, unsigned int n, double box_size) {
    double pe = 0.0;
    double v_shift = compute_v_shift();
    const double rc2 = (R_CUT * SIGMA) * (R_CUT * SIGMA);
    for (unsigned int i = 0; i < n; ++i) {
        double xi = particles[i].x;
        double yi = particles[i].y;
        for (unsigned int j = i + 1; j < n; ++j) {
            double dx = xi - particles[j].x;
            double dy = yi - particles[j].y;
            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);
            double r2 = dx * dx + dy * dy;
            if (r2 >= rc2) continue;
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
        }
    }
    return pe;
}

// Black background, yellow particles.
static uint8_t palette[] = {0, 0, 0, 255, 255, 0};

static void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    img[(size_t)y * (size_t)w + (size_t)x] = index;
}

ge_GIF *open_gif(const char *path) {
    return ge_new_gif(path, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT,
                      palette, 8, -1, 0);
}

void render_frame(ge_GIF *gif, const Particle *particles, unsigned int n,
                     double box_size) {
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);
    for (unsigned int i = 0; i < n; ++i) {
        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;
        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
    ge_add_frame(gif, FRAME_DELAY);
}

void dump_final_state(const char *path, const Particle *particles,
                      unsigned int n) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    for (unsigned int i = 0; i < n; ++i) {
        fprintf(fp, "%.6f %.6f\n", particles[i].x, particles[i].y);
    }
    fclose(fp);
}
