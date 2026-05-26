#include <math.h>
#include <stdio.h>
#include "lenia_common.h"

double gauss(double x, double mu, double sigma)
{
    double t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}

double growth_lenia(double u)
{
    return -1.0 + 2.0 * gauss(u, LENIA_GROWTH_MU, LENIA_GROWTH_SIGMA);
}

void generate_kernel(double *K, unsigned int size)
{
    int r = (int)size / 2;
    double sum = 0.0;
    for (int y = -r; y < r; y++) {
        for (int x = -r; x < r; x++) {
            double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
            double v = (distance > 1.0) ? 0.0
                                        : gauss(distance, LENIA_KERNEL_MU, LENIA_KERNEL_SIGMA);
            K[(y + r) * size + (x + r)] = v;
            sum += v;
        }
    }
    for (unsigned int i = 0; i < size * size; i++) K[i] /= sum;
}

void dump_final_state(const char *path, const double *world,
                      unsigned int rows, unsigned int cols)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            fprintf(fp, "%.6f ", world[i * cols + j]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}
