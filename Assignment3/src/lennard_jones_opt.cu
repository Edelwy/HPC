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

/* Tasks 139 + 140 on top of the base full-N^2 GPU variant:
 *  - Structure-of-Arrays layout (x,y,vx,vy,fx,fy) so global-memory accesses by
 *    consecutive threads coalesce.
 *  - Shared-memory tiling: each block cooperatively stages BLOCKSIZE neighbour
 *    positions into shared memory and reuses them across the whole block,
 *    cutting global-memory traffic in the force loop (NVIDIA n-body pattern).
 *  - BLOCKSIZE is a compile-time knob (-DBLOCKSIZE) for block-size tuning. */

__global__ void kick_drift_kernel(double *x, double *y, double *vx, double *vy,
                                   const double *fx, const double *fy,
                                   unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
    x[i] += DT * vx[i];
    y[i] += DT * vy[i];
    double wx = fmod(x[i], box_size);
    double wy = fmod(y[i], box_size);
    if (wx < 0.0) wx += box_size;
    if (wy < 0.0) wy += box_size;
    x[i] = wx;
    y[i] = wy;
}

__global__ void forces_tiled_kernel(const double *x, const double *y,
                                     double *fx, double *fy,
                                     unsigned int n, double box_size) {
    extern __shared__ double sh[];
    double *sx = sh;
    double *sy = &sh[blockDim.x];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double xi = (i < n) ? x[i] : 0.0;
    double yi = (i < n) ? y[i] : 0.0;
    double fxi = 0.0;
    double fyi = 0.0;
    const double rc2 = R_CUT * R_CUT;

    for (unsigned int base = 0; base < n; base += blockDim.x) {
        unsigned int jid = base + threadIdx.x;
        sx[threadIdx.x] = (jid < n) ? x[jid] : 0.0;
        sy[threadIdx.x] = (jid < n) ? y[jid] : 0.0;
        __syncthreads();

        unsigned int rem = n - base;
        unsigned int jmax = (rem < blockDim.x) ? rem : blockDim.x;
        for (unsigned int k = 0; k < jmax; ++k) {
            unsigned int j = base + k;
            if (j == i) continue;
            double dx = xi - sx[k];
            double dy = yi - sy[k];
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
        __syncthreads();
    }

    if (i < n) {
        fx[i] = fxi;
        fy[i] = fyi;
    }
}

__global__ void kick_kernel(double *vx, double *vy, const double *fx,
                            const double *fy, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += 0.5 * DT * fx[i];
    vy[i] += 0.5 * DT * fy[i];
}

/* AoS host particles <-> SoA host staging buffers. */
static void aos_to_soa(const Particle *p, unsigned int n, double *x, double *y,
                       double *vx, double *vy) {
    for (unsigned int i = 0; i < n; ++i) {
        x[i] = p[i].x;   y[i] = p[i].y;
        vx[i] = p[i].vx; vy[i] = p[i].vy;
    }
}

static void soa_to_aos(Particle *p, unsigned int n, const double *x,
                       const double *y, const double *vx, const double *vy) {
    for (unsigned int i = 0; i < n; ++i) {
        p[i].x = x[i];   p[i].y = y[i];
        p[i].vx = vx[i]; p[i].vy = vy[i];
    }
}

extern "C" SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;

    SimulationResult out;
    out.start_kinetic = compute_ke(particles, n);
    out.start_potential = compute_pe(particles, n, box_size);
    out.start_total = out.start_kinetic + out.start_potential;

    double *hx = (double *)malloc(n * sizeof(double));
    double *hy = (double *)malloc(n * sizeof(double));
    double *hvx = (double *)malloc(n * sizeof(double));
    double *hvy = (double *)malloc(n * sizeof(double));
    aos_to_soa(particles, n, hx, hy, hvx, hvy);

    double *dx, *dy, *dvx, *dvy, *dfx, *dfy;
    checkCudaErrors(cudaMalloc(&dx, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dy, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dvx, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dvy, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dfx, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dfy, n * sizeof(double)));
    checkCudaErrors(cudaMemcpy(dx, hx, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dy, hy, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dvx, hvx, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dvy, hvy, n * sizeof(double), cudaMemcpyHostToDevice));

    unsigned int block = BLOCKSIZE;
    unsigned int grid = (n + block - 1) / block;
    size_t shmem = 2 * block * sizeof(double);

    forces_tiled_kernel<<<grid, block, shmem>>>(dx, dy, dfx, dfy, n, box_size);
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
        kick_drift_kernel<<<grid, block>>>(dx, dy, dvx, dvy, dfx, dfy, n, box_size);
        forces_tiled_kernel<<<grid, block, shmem>>>(dx, dy, dfx, dfy, n, box_size);
        kick_kernel<<<grid, block>>>(dvx, dvy, dfx, dfy, n);

        if (opts->track_energy || (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0)) {
            checkCudaErrors(cudaMemcpy(hx, dx, n * sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hy, dy, n * sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hvx, dvx, n * sizeof(double), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hvy, dvy, n * sizeof(double), cudaMemcpyDeviceToHost));
            soa_to_aos(particles, n, hx, hy, hvx, hvy);
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

    checkCudaErrors(cudaMemcpy(hx, dx, n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hy, dy, n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hvx, dvx, n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hvy, dvy, n * sizeof(double), cudaMemcpyDeviceToHost));
    soa_to_aos(particles, n, hx, hy, hvx, hvy);

    checkCudaErrors(cudaFree(dx));
    checkCudaErrors(cudaFree(dy));
    checkCudaErrors(cudaFree(dvx));
    checkCudaErrors(cudaFree(dvy));
    checkCudaErrors(cudaFree(dfx));
    checkCudaErrors(cudaFree(dfy));
    free(hx); free(hy); free(hvx); free(hvy);

    if (gif) ge_close_gif(gif);

    out.final_kinetic = compute_ke(particles, n);
    out.final_potential = compute_pe(particles, n, box_size);
    out.final_total = out.final_kinetic + out.final_potential;

    out.n = n;
    out.particles = particles;
    return out;
}
