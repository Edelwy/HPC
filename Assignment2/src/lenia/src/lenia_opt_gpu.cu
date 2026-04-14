#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>

#include "helper_cuda.h"

// Uncomment to generate gif animation
#define GENERATE_GIF

// For prettier indexing syntax
#define w_opt(r, c) (w[(r) * kernel_size + (c)])
#define input_opt(r, c) (world[((r) % rows) * cols + ((c) % cols)])

#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Cuda sizes
#define BLOCKSIZE 16 // 16 or 32
#define MAX_KERNEL_SIZE 64

// Assumption: kernel goes into constant memory
__constant__ double d_w[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Function to calculate Gaussian
//binline double gauss(double x, double mu, double sigma)
// Called from device (GPU)
__host__ __device__ __forceinline__ double gauss(double x, double mu, double sigma)
{
//    return exp(-0.5 * pow((x - mu) / sigma, 2));
    double t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}


// modulo is slooow
__device__ __forceinline__ int wrap(int x, int max)
{
    if (x < 0) return x + max;
    if (x >= max) return x - max;
    return x;
}

// Function for growth criteria
//double growth_lenia(double u)
// Also alled from device (GPU)
__device__ __forceinline__ double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1)
                {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}


// Cuda convolution
__global__ void convolutionStep(double* world, double* world_b, int rows, int cols, int kernel_size, int R, double dt)
{
    extern __shared__ double tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    const int tile_w = blockDim.x + kernel_size - 1;
    const int tile_h = blockDim.y + kernel_size - 1;

    for (int dy = ty; dy < tile_h; dy += blockDim.y)
    {
        for (int dx = tx; dx < tile_w; dx += blockDim.x)
        {
            int gx = wrap(bx + dx - R, cols);
            int gy = wrap(by + dy - R, rows);
            tile[dy * tile_w + dx] = world[gy * cols + gx];
        }
    }

    __syncthreads();

    if (x < cols && y < rows)
    {
        double sum = 0.0;

        for (int kri = 0; kri < kernel_size; kri++)
        {
            for (int kcj = 0; kcj < kernel_size; kcj++)
            {
                const int wi = kernel_size - 1 - kri;
                const int wj = kernel_size - 1 - kcj;
                sum += d_w[wi * kernel_size + wj] * tile[(ty + kri) * tile_w + (tx + kcj)];
            }
        }

        int cell = y * cols + x;

        double val = world[cell] + dt * growth_lenia(sum);

        val = fmin(1.0, fmax(0.0, val));

        world_b[cell] = val;
    }
}


// Function to evolve Lenia
double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",     /* file name */
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    // Allocate memory
    double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    //double *world_b = (double *)calloc(rows * cols, sizeof(double));
    //double *tmp = (double *)calloc(rows * cols, sizeof(double));
    
    int R = kernel_size / 2;
    
    // Timing. TODO: decide where to time.
    cudaEvent_t start, stop, startKernel, stopKernel;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&startKernel));
    checkCudaErrors(cudaEventCreate(&stopKernel));

    checkCudaErrors(cudaEventRecord(start));

    // Generate convolution kernel
    w=generate_kernel(w,kernel_size);

    // Copy to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_w, w, kernel_size * kernel_size * sizeof(double)));

    // Place orbiums
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // Copy world to device memory
    double *d_world;
    double *d_world_b;

    checkCudaErrors(cudaMalloc(&d_world, rows * cols * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_world_b, rows * cols * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_world, world, rows * cols * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((cols + BLOCKSIZE - 1) / BLOCKSIZE, (rows + BLOCKSIZE - 1) / BLOCKSIZE);

    const int tile_w_host = (int)BLOCKSIZE + (int)kernel_size - 1;
    const int tile_h_host = (int)BLOCKSIZE + (int)kernel_size - 1;
    int sharedMemSize = tile_w_host * tile_h_host * (int)sizeof(double);
    
    // Lenia Simulation
    checkCudaErrors(cudaEventRecord(startKernel));
    for (unsigned int step = 0; step < steps; step++)
    {
        convolutionStep<<<grid, block, sharedMemSize>>>(d_world, d_world_b, rows, cols, kernel_size, R, dt);
        checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();

        double* tmp = d_world;
        d_world = d_world_b;
        d_world_b = tmp;

        #ifdef GENERATE_GIF
        checkCudaErrors(cudaMemcpy(world, d_world, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
        for (int cel = 0; cel < rows * cols; cel++) {
            gif->frame[cel] = world[cel] * 255;
        }
        ge_add_frame(gif, 5);
        #endif
    }
    checkCudaErrors(cudaEventRecord(stopKernel));
    checkCudaErrors(cudaEventSynchronize(stopKernel));

    checkCudaErrors(cudaMemcpy(world, d_world, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_world_b));

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float time, timeKernel;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&timeKernel, startKernel, stopKernel));
	printf("Time: device = %f s\n", time/1000.0);
    printf("Time: kernel = %f s\n\n", timeKernel/1000.0);

    #ifdef GENERATE_GIF
    ge_close_gif(gif);
    #endif
    free(w);
    return world;
}
