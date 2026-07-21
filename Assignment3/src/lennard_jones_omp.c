#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lennard_jones_common.h"

// Same Newton's 3rd law loop as the sequential version, parallelised with an
// OpenMP array-section reduction.
static double compute_forces(const Particle *particles, double *fx, double *fy,
                             unsigned int n, double box_size, double v_shift) {
    // Initialize forces to 0.
    for (unsigned int i = 0; i < n; ++i) {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }
    double pe = 0.0;
    const double rc2 = (R_CUT * SIGMA) * (R_CUT * SIGMA);

    #pragma omp parallel for reduction(+:pe) reduction(+:fx[:n]) reduction(+:fy[:n]) schedule(dynamic, 64)
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

            double fmag = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            double fxij = fmag * dx;
            double fyij = fmag * dy;

            fx[i] += fxij;
            fy[i] += fyij;
            fx[j] -= fxij;
            fy[j] -= fyij;

            pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
        }
    }
    return pe;
}

static double leapfrog_step(Particle *particles, double *fx, double *fy,
                            unsigned int n, double box_size, double v_shift) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * fx[i];
        p->vy += 0.5 * DT * fy[i];
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }
    wrap_positions(particles, n, box_size);
    double pe = compute_forces(particles, fx, fy, n, box_size, v_shift);
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * fx[i];
        particles[i].vy += 0.5 * DT * fy[i];
    }
    return pe;
}

SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;
    double v_shift = compute_v_shift();

    double *fx = (double *)malloc(n * sizeof(double));
    double *fy = (double *)malloc(n * sizeof(double));

    SimulationResult out;
    out.start_potential = compute_forces(particles, fx, fy, n, box_size, v_shift);
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
        out.final_potential = leapfrog_step(particles, fx, fy, n, box_size, v_shift);
        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (opts->track_energy) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                   step, out.final_kinetic, out.final_potential, out.final_total);
        }
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame(gif, particles, n, box_size);
        }
    }

    if (gif) ge_close_gif(gif);

    free(fx);
    free(fy);
    out.n = n;
    out.particles = particles;
    return out;
}
