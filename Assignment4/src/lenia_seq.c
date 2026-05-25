#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "lenia_common.h"
#include "orbium.h"
#include "gifenc.h"

double *evolve_lenia(unsigned int rows, unsigned int cols, unsigned int steps,
                     double dt, unsigned int kernel_size,
                     const struct orbium_coo *orbiums, unsigned int num_orbiums,
                     const struct lenia_opts *opts)
{
    int R = (int)kernel_size / 2;

    double *w        = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world    = (double *)calloc(rows * cols, sizeof(double));
    double *world_b  = (double *)calloc(rows * cols, sizeof(double));

    generate_kernel(w, kernel_size);
    for (unsigned int o = 0; o < num_orbiums; o++)
        place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    ge_GIF *gif = NULL;
    if (opts && opts->gif_path) {
        gif = ge_new_gif(opts->gif_path, cols, rows, inferno_pallete, 8, -1, 0);
    }

    for (unsigned int step = 0; step < steps; step++) {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < (int)kernel_size; ki++) {
                    int ni = wrap((int)i + ki - R, (int)rows);
                    for (int kj = 0; kj < (int)kernel_size; kj++) {
                        int nj = wrap((int)j + kj - R, (int)cols);
                        sum += w[(kernel_size - 1 - ki) * kernel_size + (kernel_size - 1 - kj)]
                             * world[ni * cols + nj];
                    }
                }
                double v = world[i * cols + j] + dt * growth_lenia(sum);
                if (v < 0.0) v = 0.0; else if (v > 1.0) v = 1.0;
                world_b[i * cols + j] = v;
                if (gif) gif->frame[i * cols + j] = (uint8_t)(v * 255);
            }
        }
        if (gif) ge_add_frame(gif, 5);
        double *tmp = world; world = world_b; world_b = tmp;
    }

    if (gif) ge_close_gif(gif);
    if (opts && opts->final_path) dump_final_state(opts->final_path, world, rows, cols);

    free(w);
    free(world_b);
    return world;
}
