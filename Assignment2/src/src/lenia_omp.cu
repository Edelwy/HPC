#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"
#include <omp.h>

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>

// Uncomment to generate gif animation
#define GENERATE_GIF 1

// For prettier indexing syntax
#define w_opt(r, c) (w[(r) * kernel_size + (c)])
#define input_opt(r, c) (world[((r) % rows) * cols + ((c) % cols)])

#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma)
{
//    return exp(-0.5 * pow((x - mu) / sigma, 2));
    double t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}


// modulo is slooow
inline int wrap(int x, int max)
{
    if (x < 0) return x + max;
    if (x >= max) return x - max;
    return x;
}

// Function for growth criteria
double growth_lenia(double u)
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
    double *world_b = (double *)calloc(rows * cols, sizeof(double));
    double *tmp = (double *)calloc(rows * cols, sizeof(double));

    int R = kernel_size / 2;

    // Generate convolution kernel
    w=generate_kernel(w,kernel_size);

    // Place orbiums
    #pragma omp parallel for
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++)
    {
        
        // Evolution
        #pragma omp parallel for collapse(2)
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
            
                // from convolve2d
                double sum = 0;
                for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        int ni = wrap(i - R + kri, rows);
			int nj = wrap(j - R + kcj, cols);

			sum += w_opt(ki, kj) * world[ni * cols + nj];
                    }
                }
                int cell = i * cols + j;
                tmp[cell] = sum;
            
                world_b[cell] = world[cell] + dt * growth_lenia(tmp[cell]);
                world_b[cell] = fmin(1, fmax(0, world_b[cell])); // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[cell] = world_b[cell] * 255;
#endif
            }
        }
        
        double *tmp_ptr = world;
	world = world_b;
	world_b = tmp_ptr;
        
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    free(w);
    free(tmp);
    return world;
}
