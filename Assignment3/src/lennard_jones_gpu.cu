#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "lennard_jones_common.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 128
#endif

// First part of the leapfrog step using current forces and wrapping.
__global__ void leapfrog_current_kernel(Particle *p, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += 0.5 * DT * p[i].fx;
    p[i].vy += 0.5 * DT * p[i].fy;
    p[i].x += DT * p[i].vx;
    p[i].y += DT * p[i].vy;

    // This is just the wrapping of the positions function.
    double wx = fmod(p[i].x, box_size);
    double wy = fmod(p[i].y, box_size);
    if (wx < 0.0) wx += box_size;
    if (wy < 0.0) wy += box_size;
    p[i].x = wx;
    p[i].y = wy;
}

// Same as the sequential version but with the loop over all particles.
__global__ void forces_kernel(Particle *p, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return; // Threads with no particle.
    double xi = p[i].x; // Particle i's x coordinate.
    double yi = p[i].y; // Particle i's y coordinate.
    double fxi = 0.0;
    double fyi = 0.0;
    const double rc2 = (R_CUT * SIGMA) * (R_CUT * SIGMA);
    for (unsigned int j = 0; j < n; ++j) {
        if (j == i) continue;
        double dx = xi - p[j].x;
        double dy = yi - p[j].y;
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        double r2 = dx * dx + dy * dy;
        if (r2 >= rc2) continue;
        double sr2 = (SIGMA * SIGMA) / r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double fmag = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
        fxi += fmag * dx;
        fyi += fmag * dy;
    }
    p[i].fx = fxi;
    p[i].fy = fyi;
}

// Second part of the leapfrog step using new forces.
__global__ void leapfrog_new_kernel(Particle *p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += 0.5 * DT * p[i].fx;
    p[i].vy += 0.5 * DT * p[i].fy;
}

// CUDA files are compiled with `nvcc` which is a CPP compiler.
// The `extern "C"` is used to prevent name mangling so that it is properly linked.
extern "C" SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;

    SimulationResult out;
    out.start_kinetic = compute_ke(particles, n);
    out.start_potential = compute_pe(particles, n, box_size);
    out.start_total = out.start_kinetic + out.start_potential;

    
    Particle *d_p; // Device pointer, holds the GPU address.
    checkCudaErrors(cudaMalloc(&d_p, n * sizeof(Particle))); // Allocate memory on device.
    checkCudaErrors(cudaMemcpy(d_p, particles, n * sizeof(Particle), cudaMemcpyHostToDevice)); // Copy data from host to device.

    unsigned int block = BLOCKSIZE;
    unsigned int grid = (n + block - 1) / block;

    // Calculates forces for all particles.
    forces_kernel<<<grid, block>>>(d_p, n, box_size);
    checkCudaErrors(cudaGetLastError());

    ge_GIF *gif = NULL;
    if (opts->gif_path) {
        gif = open_gif(opts->gif_path);
        render_frame(gif, particles, n, box_size);
    }

    out.final_kinetic = out.start_kinetic;
    out.final_potential = out.start_potential;
    out.final_total = out.start_total;

    // Performs the leapfrog step.
    for (unsigned int step = 0; step < opts->nsteps; step++) {
        leapfrog_current_kernel<<<grid, block>>>(d_p, n, box_size);
        forces_kernel<<<grid, block>>>(d_p, n, box_size);
        leapfrog_new_kernel<<<grid, block>>>(d_p, n);

        if (opts->track_energy || (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)) {
            checkCudaErrors(cudaMemcpy(particles, d_p, n * sizeof(Particle), cudaMemcpyDeviceToHost));
            if (opts->track_energy) {
                double ke = compute_ke(particles, n);
                double pe = compute_pe(particles, n, box_size);
                printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n", step, ke, pe, ke + pe);
            }
            if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
                render_frame(gif, particles, n, box_size);
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
