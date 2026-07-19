#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "lennard_jones.h"
#include "lennard_jones_common.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 128
#endif

/* Base CUDA port: Array-of-Structs layout, one thread per particle, full N^2
 * force evaluation (Newton's 3rd law is dropped because the f[j] -= write would
 * race across threads). Leapfrog is split into three kernels. */

__global__ void kick_drift_kernel(Particle *p, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += 0.5 * DT * p[i].fx;
    p[i].vy += 0.5 * DT * p[i].fy;
    p[i].x += DT * p[i].vx;
    p[i].y += DT * p[i].vy;
    double wx = fmod(p[i].x, box_size);
    double wy = fmod(p[i].y, box_size);
    if (wx < 0.0) wx += box_size;
    if (wy < 0.0) wy += box_size;
    p[i].x = wx;
    p[i].y = wy;
}

__global__ void forces_kernel(Particle *p, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = p[i].x;
    double yi = p[i].y;
    double fxi = 0.0;
    double fyi = 0.0;
    const double rc2 = R_CUT * R_CUT;
    for (unsigned int j = 0; j < n; ++j) {
        if (j == i) continue;
        double dx = xi - p[j].x;
        double dy = yi - p[j].y;
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        double r2 = dx * dx + dy * dy;
        if (r2 >= rc2) continue;
        double sr2 = 1.0 / r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double fmag = 24.0 * EPSILON * (2.0 * sr12 - sr6) * sr2;
        fxi += fmag * dx;
        fyi += fmag * dy;
    }
    p[i].fx = fxi;
    p[i].fy = fyi;
}

__global__ void kick_kernel(Particle *p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += 0.5 * DT * p[i].fx;
    p[i].vy += 0.5 * DT * p[i].fy;
}

extern "C" SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;

    SimulationResult out;
    out.start_kinetic = compute_ke(particles, n);
    out.start_potential = compute_pe(particles, n, box_size);
    out.start_total = out.start_kinetic + out.start_potential;

    Particle *d_p;
    checkCudaErrors(cudaMalloc(&d_p, n * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(d_p, particles, n * sizeof(Particle), cudaMemcpyHostToDevice));

    unsigned int block = BLOCKSIZE;
    unsigned int grid = (n + block - 1) / block;

    forces_kernel<<<grid, block>>>(d_p, n, box_size);
    checkCudaErrors(cudaGetLastError());

    ge_GIF *gif = NULL;
    if (opts->gif_path) {
        gif = lj_open_gif(opts->gif_path);
        lj_render_frame(gif, particles, n, box_size);
    }

    out.final_kinetic = out.start_kinetic;
    out.final_potential = out.start_potential;
    out.final_total = out.start_total;

    for (unsigned int step = 0; step < opts->nsteps; step++) {
        kick_drift_kernel<<<grid, block>>>(d_p, n, box_size);
        forces_kernel<<<grid, block>>>(d_p, n, box_size);
        kick_kernel<<<grid, block>>>(d_p, n);

        /* Per-step energy / animation need host-side data; only used outside
         * the timed benchmark, so the extra copy-back is acceptable there. */
        if (opts->track_energy || (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)) {
            checkCudaErrors(cudaMemcpy(particles, d_p, n * sizeof(Particle), cudaMemcpyDeviceToHost));
            if (opts->track_energy) {
                double ke = compute_ke(particles, n);
                double pe = compute_pe(particles, n, box_size);
                printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n", step, ke, pe, ke + pe);
            }
            if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
                lj_render_frame(gif, particles, n, box_size);
            }
        }
    }

    checkCudaErrors(cudaMemcpy(particles, d_p, n * sizeof(Particle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_p));

    if (gif) ge_close_gif(gif);

    out.final_kinetic = compute_ke(particles, n);
    out.final_potential = compute_pe(particles, n, box_size);
    out.final_total = out.final_kinetic + out.final_potential;

    out.n = n;
    out.particles = particles;
    return out;
}
