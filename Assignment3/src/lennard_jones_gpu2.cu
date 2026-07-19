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

/* Task 141: two GPUs. The particle array is split in half; device 0 owns
 * [0, n0), device 1 owns [n0, n). Each device keeps a full copy of all
 * positions but only updates (forces / integrates) its own half. After the
 * drift each device sends its owned positions to the other (staged through the
 * host), so both hold the full up-to-date positions before the force pass.
 * Kernels take a (start, count) range; otherwise they mirror the base variant.
 * Assumption: exactly two visible GPUs. */

__global__ void kick_drift_kernel(Particle *p, unsigned int start,
                                   unsigned int count, double box_size) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    unsigned int i = start + t;
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

__global__ void forces_kernel(Particle *p, unsigned int n, unsigned int start,
                              unsigned int count, double box_size) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    unsigned int i = start + t;
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

__global__ void kick_kernel(Particle *p, unsigned int start, unsigned int count) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    unsigned int i = start + t;
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

    /* device 0 owns [0, n0), device 1 owns [n0, n) */
    unsigned int n0 = n / 2;
    unsigned int n1 = n - n0;
    unsigned int block = BLOCKSIZE;
    unsigned int grid0 = (n0 + block - 1) / block;
    unsigned int grid1 = (n1 + block - 1) / block;

    Particle *d_p[2];
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc(&d_p[0], n * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(d_p[0], particles, n * sizeof(Particle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaSetDevice(1));
    checkCudaErrors(cudaMalloc(&d_p[1], n * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(d_p[1], particles, n * sizeof(Particle), cudaMemcpyHostToDevice));

    /* initial forces on each device for its own range */
    checkCudaErrors(cudaSetDevice(0));
    forces_kernel<<<grid0, block>>>(d_p[0], n, 0, n0, box_size);
    checkCudaErrors(cudaSetDevice(1));
    forces_kernel<<<grid1, block>>>(d_p[1], n, n0, n1, box_size);

    ge_GIF *gif = NULL;
    if (opts->gif_path) {
        gif = lj_open_gif(opts->gif_path);
        lj_render_frame(gif, particles, n, box_size);
    }

    out.final_kinetic = out.start_kinetic;
    out.final_potential = out.start_potential;
    out.final_total = out.start_total;

    for (unsigned int step = 0; step < opts->nsteps; step++) {
        /* half-kick + drift on each device (owned range) */
        checkCudaErrors(cudaSetDevice(0));
        kick_drift_kernel<<<grid0, block>>>(d_p[0], 0, n0, box_size);
        checkCudaErrors(cudaSetDevice(1));
        kick_drift_kernel<<<grid1, block>>>(d_p[1], n0, n1, box_size);

        /* exchange updated owned positions through the host */
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaMemcpy(particles, d_p[0], n0 * sizeof(Particle), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaSetDevice(1));
        checkCudaErrors(cudaMemcpy(particles + n0, d_p[1] + n0, n1 * sizeof(Particle), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaMemcpy(d_p[0] + n0, particles + n0, n1 * sizeof(Particle), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaSetDevice(1));
        checkCudaErrors(cudaMemcpy(d_p[1], particles, n0 * sizeof(Particle), cudaMemcpyHostToDevice));

        /* force pass (needs all positions) then second half-kick */
        checkCudaErrors(cudaSetDevice(0));
        forces_kernel<<<grid0, block>>>(d_p[0], n, 0, n0, box_size);
        kick_kernel<<<grid0, block>>>(d_p[0], 0, n0);
        checkCudaErrors(cudaSetDevice(1));
        forces_kernel<<<grid1, block>>>(d_p[1], n, n0, n1, box_size);
        kick_kernel<<<grid1, block>>>(d_p[1], n0, n1);

        if (opts->track_energy || (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)) {
            checkCudaErrors(cudaSetDevice(0));
            checkCudaErrors(cudaMemcpy(particles, d_p[0], n0 * sizeof(Particle), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaSetDevice(1));
            checkCudaErrors(cudaMemcpy(particles + n0, d_p[1] + n0, n1 * sizeof(Particle), cudaMemcpyDeviceToHost));
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

    /* gather final owned ranges from both devices */
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpy(particles, d_p[0], n0 * sizeof(Particle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_p[0]));
    checkCudaErrors(cudaSetDevice(1));
    checkCudaErrors(cudaMemcpy(particles + n0, d_p[1] + n0, n1 * sizeof(Particle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_p[1]));

    if (gif) ge_close_gif(gif);

    out.final_kinetic = compute_ke(particles, n);
    out.final_potential = compute_pe(particles, n, box_size);
    out.final_total = out.final_kinetic + out.final_potential;

    out.n = n;
    out.particles = particles;
    return out;
}
