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

/* Fixed per-cell capacity. With density ~0.95 and cells of side ~r_cut the
 * expected occupancy is a handful of particles, so this is comfortably large.
 * Overflow is intentionally NOT handled (script-level assumption). */
#define MAX_PER_CELL 64

/* Task 138: neighbourhood cell lists. The box is split into cells of side
 * >= r_cut, so a particle can only interact with the 3x3 block of cells around
 * its own. The lists are rebuilt every step (particles move) with atomics.
 * Kick/drift/kick kernels are identical to the base GPU variant. */

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

__global__ void kick_kernel(Particle *p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += 0.5 * DT * p[i].fx;
    p[i].vy += 0.5 * DT * p[i].fy;
}

__global__ void bin_kernel(const Particle *p, unsigned int n, int ncell,
                           double cell_size, int *cell_count, int *cell_list) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx = (int)(p[i].x / cell_size);
    int cy = (int)(p[i].y / cell_size);
    if (cx >= ncell) cx = ncell - 1;
    if (cy >= ncell) cy = ncell - 1;
    int cell = cy * ncell + cx;
    int slot = atomicAdd(&cell_count[cell], 1);
    if (slot < MAX_PER_CELL) {
        cell_list[cell * MAX_PER_CELL + slot] = (int)i;
    }
}

__global__ void forces_cells_kernel(Particle *p, unsigned int n, double box_size,
                                     int ncell, double cell_size,
                                     const int *cell_count, const int *cell_list) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = p[i].x;
    double yi = p[i].y;
    double fxi = 0.0;
    double fyi = 0.0;
    const double rc2 = R_CUT * R_CUT;

    int cx = (int)(xi / cell_size);
    int cy = (int)(yi / cell_size);
    if (cx >= ncell) cx = ncell - 1;
    if (cy >= ncell) cy = ncell - 1;

    for (int dcy = -1; dcy <= 1; ++dcy) {
        int ncy = (cy + dcy + ncell) % ncell;
        for (int dcx = -1; dcx <= 1; ++dcx) {
            int ncx = (cx + dcx + ncell) % ncell;
            int cell = ncy * ncell + ncx;
            int cnt = cell_count[cell];
            if (cnt > MAX_PER_CELL) cnt = MAX_PER_CELL;
            for (int s = 0; s < cnt; ++s) {
                unsigned int j = (unsigned int)cell_list[cell * MAX_PER_CELL + s];
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
        }
    }
    p[i].fx = fxi;
    p[i].fy = fyi;
}

extern "C" SimulationResult run_simulation(Particle *particles, const SimOptions *opts) {
    unsigned int n = opts->n;
    double box_size = opts->box_size;

    SimulationResult out;
    out.start_kinetic = compute_ke(particles, n);
    out.start_potential = compute_pe(particles, n, box_size);
    out.start_total = out.start_kinetic + out.start_potential;

    int ncell = (int)(box_size / R_CUT);
    double cell_size = box_size / (double)ncell;
    int ncell2 = ncell * ncell;

    Particle *d_p;
    int *d_count;
    int *d_list;
    checkCudaErrors(cudaMalloc(&d_p, n * sizeof(Particle)));
    checkCudaErrors(cudaMalloc(&d_count, ncell2 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_list, (size_t)ncell2 * MAX_PER_CELL * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_p, particles, n * sizeof(Particle), cudaMemcpyHostToDevice));

    unsigned int block = BLOCKSIZE;
    unsigned int grid = (n + block - 1) / block;

    checkCudaErrors(cudaMemset(d_count, 0, ncell2 * sizeof(int)));
    bin_kernel<<<grid, block>>>(d_p, n, ncell, cell_size, d_count, d_list);
    forces_cells_kernel<<<grid, block>>>(d_p, n, box_size, ncell, cell_size, d_count, d_list);
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
        checkCudaErrors(cudaMemset(d_count, 0, ncell2 * sizeof(int)));
        bin_kernel<<<grid, block>>>(d_p, n, ncell, cell_size, d_count, d_list);
        forces_cells_kernel<<<grid, block>>>(d_p, n, box_size, ncell, cell_size, d_count, d_list);
        kick_kernel<<<grid, block>>>(d_p, n);

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
    checkCudaErrors(cudaFree(d_count));
    checkCudaErrors(cudaFree(d_list));

    if (gif) ge_close_gif(gif);

    out.final_kinetic = compute_ke(particles, n);
    out.final_potential = compute_pe(particles, n, box_size);
    out.final_total = out.final_kinetic + out.final_potential;

    out.n = n;
    out.particles = particles;
    return out;
}
