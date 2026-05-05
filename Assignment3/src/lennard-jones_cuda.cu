#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA headers
#include <cuda_runtime.h>

#include "gifenc.h"
#include "lennard-jones.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 16
#endif
#define MAX_KERNEL_SIZE 64

// Internal pair-list data used by the OpenMP implementation.
// This avoids pointer-based pairs and makes the structure CUDA-friendly later.
typedef struct {
    unsigned int i;
    unsigned int j;
    double dx;
    double dy;
    double r2;
} ForcePair;

typedef struct {
    double fx;
    double fy;
} PairForce;

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {
                             0, 0, 0,
                             255, 255, 0};

void set_pixel(uint8_t *img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}


void render_frame_gif(ge_GIF *gif, const Particle *particles, unsigned int n, double box_size) {

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
}
#endif
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double compute_ke(const Particle *particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle *p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle *particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {

    srand(seed);
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
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
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (
            particles[k].vx * particles[k].vx +
            particles[k].vy * particles[k].vy
        );
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positions(Particle *particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}



#define CUDA_CHECK(call) do {                                                \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(err__));              \
        exit(1);                                                             \
    }                                                                        \
} while (0)


__global__ void build_pairs_kernel(const Particle *particles,
                                   unsigned int n,
                                   double box_size,
                                   double radius2,
                                   ForcePair *pairs,
                                   unsigned int *pair_count)
{
    // One GPU thread considers one (i, j) candidate from an n x n grid.
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n || j <= i) {
        return;
    }

    double dx = particles[i].x - particles[j].x;
    double dy = particles[i].y - particles[j].y;

    // Minimum-image convention for periodic boundaries.
    dx -= box_size * nearbyint(dx / box_size);
    dy -= box_size * nearbyint(dy / box_size);

    double r2 = dx * dx + dy * dy;

    if (r2 >= radius2 || r2 == 0.0) {
        return;
    }

    unsigned int idx = atomicAdd(pair_count, 1u);

    pairs[idx].i = i;
    pairs[idx].j = j;
    pairs[idx].dx = dx;
    pairs[idx].dy = dy;
    pairs[idx].r2 = r2;
}


__global__ void compute_pair_forces_kernel(const ForcePair *pairs,
                                           PairForce *pair_forces,
                                           unsigned int pair_count,
                                           double sigma2,
                                           double coeff,
                                           double v_shift,
                                           double *block_pe)
{
    extern __shared__ double sh_pe[];

    unsigned int tid = threadIdx.x;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    double pe = 0.0;

    if (k < pair_count) {
        double dx = pairs[k].dx;
        double dy = pairs[k].dy;
        double r2 = pairs[k].r2;

        double inv_r2 = 1.0 / r2;
        double sig2_over_r2 = sigma2 * inv_r2;
        double sr6 = sig2_over_r2 * sig2_over_r2 * sig2_over_r2;
        double sr12 = sr6 * sr6;

        double force_factor = coeff * (2.0 * sr12 - sr6) * inv_r2;

        pair_forces[k].fx = force_factor * dx;
        pair_forces[k].fy = force_factor * dy;

        pe = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
    }

    sh_pe[tid] = pe;
    __syncthreads();

    // Block reduction for potential energy.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_pe[tid] += sh_pe[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_pe[blockIdx.x] = sh_pe[0];
    }
}


__global__ void reset_forces_kernel(Particle *particles, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
}


__global__ void accumulate_forces_kernel(Particle *particles,
                                         const ForcePair *pairs,
                                         const PairForce *pair_forces,
                                         unsigned int pair_count)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) {
        return;
    }

    unsigned int i = pairs[k].i;
    unsigned int j = pairs[k].j;
    double fx = pair_forces[k].fx;
    double fy = pair_forces[k].fy;

    // This is the unavoidable conflict step in this simple pair-parallel CUDA version.
    // It is correct and mirrors the OpenMP atomic accumulation stage.
    atomicAdd(&particles[i].fx,  fx);
    atomicAdd(&particles[i].fy,  fy);
    atomicAdd(&particles[j].fx, -fx);
    atomicAdd(&particles[j].fy, -fy);
}


__global__ void first_half_kick_and_drift_kernel(Particle *particles,
                                                 unsigned int n,
                                                 double box_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    Particle *p = &particles[i];

    p->vx += 0.5 * DT * p->fx;
    p->vy += 0.5 * DT * p->fy;

    p->x += DT * p->vx;
    p->y += DT * p->vy;

    // Periodic wrap, equivalent to the CPU wrap_positions function.
    double wx = fmod(p->x, box_size);
    double wy = fmod(p->y, box_size);

    if (wx < 0.0) wx += box_size;
    if (wy < 0.0) wy += box_size;

    p->x = wx;
    p->y = wy;
}


__global__ void second_half_kick_kernel(Particle *particles, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].vx += 0.5 * DT * particles[i].fx;
        particles[i].vy += 0.5 * DT * particles[i].fy;
    }
}


unsigned int build_pairs_cuda(Particle *d_particles,
                              unsigned int n,
                              double box_size,
                              ForcePair *d_pairs,
                              unsigned int *d_pair_count)
{
    const double radius2 = R_CUT * R_CUT;

    CUDA_CHECK(cudaMemset(d_pair_count, 0, sizeof(unsigned int)));

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);

    build_pairs_kernel<<<blocks, threads>>>(
        d_particles, n, box_size, radius2, d_pairs, d_pair_count
    );
    CUDA_CHECK(cudaGetLastError());

    unsigned int pair_count = 0;
    CUDA_CHECK(cudaMemcpy(&pair_count, d_pair_count,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return pair_count;
}


double compute_pair_forces_cuda(const ForcePair *d_pairs,
                                PairForce *d_pair_forces,
                                unsigned int pair_count)
{
    if (pair_count == 0) {
        return 0.0;
    }

    const double sigma2 = SIGMA * SIGMA;
    const double coeff = 24.0 * EPSILON;
    const double v_shift = compute_v_shift();

    const int threads = BLOCKSIZE*BLOCKSIZE;
    const int blocks = (pair_count + threads - 1) / threads;

    double *d_block_pe = NULL;
    double *h_block_pe = (double*) malloc((size_t)blocks * sizeof(double));
    if (!h_block_pe) {
        fprintf(stderr, "Failed to allocate host PE reduction buffer\n");
        exit(1);
    }

    CUDA_CHECK(cudaMalloc((void**)&d_block_pe, (size_t)blocks * sizeof(double)));

    compute_pair_forces_kernel<<<blocks, threads, threads * sizeof(double)>>>(
        d_pairs, d_pair_forces, pair_count, sigma2, coeff, v_shift, d_block_pe
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_block_pe, d_block_pe,
                          (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double pe = 0.0;
    for (int b = 0; b < blocks; ++b) {
        pe += h_block_pe[b];
    }

    CUDA_CHECK(cudaFree(d_block_pe));
    free(h_block_pe);

    return pe;
}


void accumulate_forces_cuda(Particle *d_particles,
                            unsigned int n,
                            const ForcePair *d_pairs,
                            const PairForce *d_pair_forces,
                            unsigned int pair_count)
{
    const int threads = BLOCKSIZE*BLOCKSIZE;
    int particle_blocks = (n + threads - 1) / threads;
    int pair_blocks = (pair_count + threads - 1) / threads;

    reset_forces_kernel<<<particle_blocks, threads>>>(d_particles, n);
    CUDA_CHECK(cudaGetLastError());

    if (pair_count > 0) {
        accumulate_forces_kernel<<<pair_blocks, threads>>>(
            d_particles, d_pairs, d_pair_forces, pair_count
        );
        CUDA_CHECK(cudaGetLastError());
    }
}


double compute_forces_cuda(Particle *d_particles,
                           unsigned int n,
                           double box_size,
                           ForcePair *d_pairs,
                           PairForce *d_pair_forces,
                           unsigned int *d_pair_count)
{
    unsigned int pair_count = build_pairs_cuda(d_particles, n, box_size,
                                               d_pairs, d_pair_count);
    double pe = compute_pair_forces_cuda(d_pairs, d_pair_forces, pair_count);
    accumulate_forces_cuda(d_particles, n, d_pairs, d_pair_forces, pair_count);
    return pe;
}


double leapfrog_step_cuda(Particle *d_particles,
                          unsigned int n,
                          double box_size,
                          ForcePair *d_pairs,
                          PairForce *d_pair_forces,
                          unsigned int *d_pair_count)
{
    const int threads = BLOCKSIZE*BLOCKSIZE;
    const int blocks = (n + threads - 1) / threads;

    first_half_kick_and_drift_kernel<<<blocks, threads>>>(d_particles, n, box_size);
    CUDA_CHECK(cudaGetLastError());

    double pe = compute_forces_cuda(d_particles, n, box_size,
                                    d_pairs, d_pair_forces, d_pair_count);

    second_half_kick_kernel<<<blocks, threads>>>(d_particles, n);
    CUDA_CHECK(cudaGetLastError());

    return pe;
}

SimulationResult run_simulation(Particle *particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {

    size_t ParticlePairCountMax = (size_t)n * (size_t)(n - 1) / 2;

    Particle *d_particles = NULL;
    ForcePair *d_pairs = NULL;
    PairForce *d_pair_forces = NULL;
    unsigned int *d_pair_count = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_particles, (size_t)n * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc((void**)&d_pairs, ParticlePairCountMax * sizeof(ForcePair)));
    CUDA_CHECK(cudaMalloc((void**)&d_pair_forces, ParticlePairCountMax * sizeof(PairForce)));
    CUDA_CHECK(cudaMalloc((void**)&d_pair_count, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_particles, particles,
                          (size_t)n * sizeof(Particle), cudaMemcpyHostToDevice));

    SimulationResult out;

    out.start_potential = compute_forces_cuda(d_particles, n, box_size,
                                              d_pairs, d_pair_forces, d_pair_count);
    CUDA_CHECK(cudaMemcpy(particles, d_particles,
                          (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));

    out.start_kinetic = compute_ke(particles, n);
    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF *gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif

    for (unsigned int step = 0; step < nsteps; step++) {

        out.final_potential = leapfrog_step_cuda(d_particles, n, box_size,
                                                 d_pairs, d_pair_forces, d_pair_count);

        // Copy back for KE/logging/GIF. This is simple and correct.
        // Later, KE can also be moved to a CUDA reduction to avoid this copy every step.
        CUDA_CHECK(cudaMemcpy(particles, d_particles,
                              (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));

        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf(
                "step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n",
                step,
                out.final_kinetic,
                out.final_potential,
                out.final_total
            );
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    CUDA_CHECK(cudaMemcpy(particles, d_particles,
                          (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_pair_forces));
    CUDA_CHECK(cudaFree(d_pair_count));

    out.n = n;
    out.particles = particles;
    return out;
}
