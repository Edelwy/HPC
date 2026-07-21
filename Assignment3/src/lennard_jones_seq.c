#include <math.h>
#include <stdio.h>

#include "lennard_jones_common.h"

// Newton's 3rd law: each pair is visited once and its force is applied to both particles. 
// This halves the work of the naive N^2 loop.
static double compute_forces(Particle *particles, unsigned int n, double box_size,
                             double v_shift) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    double pe = 0.0;
    const double rc2 = (R_CUT * SIGMA) * (R_CUT * SIGMA);

    for (unsigned int i = 0; i < n; ++i) {
        double xi = particles[i].x;
        double yi = particles[i].y;
        // Used for fewer writes into array.
        double fxi = 0.0;
        double fyi = 0.0;
        for (unsigned int j = i + 1; j < n; ++j) {
            double dx = xi - particles[j].x;
            double dy = yi - particles[j].y;
            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // Only compute force if the distance is less than the cutoff.
            double r2 = dx * dx + dy * dy;
            if (r2 >= rc2) continue;

            // Precompute the powers.
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            // Compared to original we already divided with r here.
            double fmag = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2; 
            double fxij = fmag * dx; 
            double fyij = fmag * dy;

            fxi += fxij;
            fyi += fyij;
            particles[j].fx -= fxij; // Opposite to i.
            particles[j].fy -= fyij; // Opposite to i.

            // In the original version we divided with 1/2 since every pair appears twice.
            pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
        }
        // Write to i at the end to avoid extra writes.
        particles[i].fx += fxi;
        particles[i].fy += fyi;
    }
    return pe;
}

// Same as in the original version.
static double leapfrog_step(Particle *particles, unsigned int n, double box_size,
                            double v_shift) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }
    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, n, box_size, v_shift);
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }
    return pe;
}


SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;
    double v_shift = compute_v_shift();

    SimulationResult out;
    out.start_potential = compute_forces(particles, n, box_size, v_shift);
    out.start_kinetic = compute_ke(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

    ge_GIF *gif = NULL;
    if (opts->gif_path) {
        gif = open_gif(opts->gif_path);
        render_frame(gif, particles, n, box_size);
    }

    out.final_potential = out.start_potential;
    out.final_kinetic = out.start_kinetic;
    out.final_total = out.start_total;

    for (unsigned int step = 0; step < opts->nsteps; step++) {
        out.final_potential = leapfrog_step(particles, n, box_size, v_shift);
        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (opts->track_energy) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, 
                   out.final_kinetic, 
                   out.final_potential, 
                   out.final_total);
        }
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame(gif, particles, n, box_size);
        }
    }

    if (gif) ge_close_gif(gif);

    out.n = n;
    out.particles = particles;
    return out;
}
