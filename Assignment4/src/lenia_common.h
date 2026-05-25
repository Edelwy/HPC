#ifndef LENIA_COMMON_H
#define LENIA_COMMON_H

#include <stdio.h>

/* Lenia growth parameters. Centralised so every variant agrees. */
#define LENIA_GROWTH_MU    0.15
#define LENIA_GROWTH_SIGMA 0.015
#define LENIA_KERNEL_MU    0.5
#define LENIA_KERNEL_SIGMA 0.15

double gauss(double x, double mu, double sigma);
double growth_lenia(double u);

/* Builds a normalised ring kernel of side `size` into K (size*size doubles). */
void generate_kernel(double *K, unsigned int size);

/* Branch-free toroidal wrap, valid for x in [-max, 2*max). */
static inline int wrap(int x, int max) {
    if (x < 0) return x + max;
    if (x >= max) return x - max;
    return x;
}

/* Dump a contiguous rows*cols double world to a plain text file (one row per line). */
void dump_final_state(const char *path, const double *world,
                      unsigned int rows, unsigned int cols);

#endif
