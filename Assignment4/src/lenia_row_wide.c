#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include "lenia.h"
#include "lenia_common.h"
#include "orbium.h"
#include "gifenc.h"

double *evolve_lenia(unsigned int rows, unsigned int cols, unsigned int steps,
                     double dt, unsigned int kernel_size,
                     const struct orbium_coo *orbiums, unsigned int num_orbiums,
                     const struct lenia_opts *opts)
{
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    const int R = (int)kernel_size / 2;
    const int K = (opts && opts->halo_steps > 0) ? opts->halo_steps : 1;
    const int H = K * R;

    int *counts       = (int *)calloc(procs, sizeof(int));
    int *offsets      = (int *)calloc(procs, sizeof(int));
    int *cell_counts  = (int *)calloc(procs, sizeof(int));
    int *cell_offsets = (int *)calloc(procs, sizeof(int));
    {
        int base = (int)rows / procs, extra = (int)rows % procs, off = 0;
        for (int r = 0; r < procs; r++) {
            counts[r]       = base + (r < extra ? 1 : 0);
            offsets[r]      = off;
            cell_counts[r]  = counts[r] * (int)cols;
            cell_offsets[r] = off * (int)cols;
            off += counts[r];
        }
    }
    const int local_rows = counts[rank];

    int min_local = counts[0];
    for (int r = 1; r < procs; r++) if (counts[r] < min_local) min_local = counts[r];
    if (min_local < H) {
        if (rank == 0) fprintf(stderr,
            "lenia_row_wide: strip %d < halo width %d (K=%d, R=%d). "
            "Pick (N, P) such that local_rows >= K*R.\n", min_local, H, K, R);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int padded = (local_rows + 2 * H) * (int)cols;
    double *world   = (double *)calloc(padded, sizeof(double));
    double *world_b = (double *)calloc(padded, sizeof(double));

    double *w = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    generate_kernel(w, kernel_size);

    double *world_full = NULL;
    if (rank == 0) {
        world_full = (double *)calloc(rows * cols, sizeof(double));
        for (unsigned int o = 0; o < num_orbiums; o++)
            place_orbium(world_full, rows, cols,
                         orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    MPI_Scatterv(world_full, cell_counts, cell_offsets, MPI_DOUBLE,
                 world + H * (int)cols, local_rows * (int)cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    if (rank == 0) { free(world_full); world_full = NULL; }
    memcpy(world_b, world, padded * sizeof(double));

    const int up         = (rank - 1 + procs) % procs;
    const int down       = (rank + 1) % procs;
    const int halo_cells = H * (int)cols;

    int do_gif   = (opts && opts->gif_path)   ? 1 : 0;
    int do_final = (opts && opts->final_path) ? 1 : 0;
    ge_GIF *gif = NULL;
    double *frame_buf = NULL;
    if (rank == 0 && do_gif) {
        gif = ge_new_gif(opts->gif_path, cols, rows, inferno_pallete, 8, -1, 0);
        frame_buf = (double *)malloc(rows * cols * sizeof(double));
    }

    unsigned int step = 0;
    while (step < steps) {
        MPI_Sendrecv(world + H * (int)cols,                    halo_cells, MPI_DOUBLE, up,   0,
                     world + (H + local_rows) * (int)cols,     halo_cells, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(world + local_rows * (int)cols,           halo_cells, MPI_DOUBLE, down, 1,
                     world + 0,                                halo_cells, MPI_DOUBLE, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        unsigned int substeps = K;
        if (step + substeps > steps) substeps = steps - step;

        for (unsigned int k = 1; k <= substeps; k++) {
            const int from = k * R;
            const int to   = (local_rows + 2 * H) - (int)k * R;
            for (int gi = from; gi < to; gi++) {
                for (int j = 0; j < (int)cols; j++) {
                    double sum = 0.0;
                    for (int ki = 0; ki < (int)kernel_size; ki++) {
                        int ni = gi + ki - R;
                        for (int kj = 0; kj < (int)kernel_size; kj++) {
                            int nj = wrap(j + kj - R, (int)cols);
                            sum += w[(kernel_size - 1 - ki) * kernel_size + (kernel_size - 1 - kj)]
                                 * world[ni * (int)cols + nj];
                        }
                    }
                    double v = world[gi * (int)cols + j] + dt * growth_lenia(sum);
                    if (v < 0.0) v = 0.0; else if (v > 1.0) v = 1.0;
                    world_b[gi * (int)cols + j] = v;
                }
            }
            double *tmp = world; world = world_b; world_b = tmp;

            if (do_gif) {
                MPI_Gatherv(world + H * (int)cols, local_rows * (int)cols, MPI_DOUBLE,
                            frame_buf, cell_counts, cell_offsets, MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
                if (rank == 0) {
                    for (unsigned int i = 0; i < rows * cols; i++)
                        gif->frame[i] = (uint8_t)(frame_buf[i] * 255);
                    ge_add_frame(gif, 5);
                }
            }
        }
        step += substeps;
    }

    double *result = NULL;
    if (rank == 0) result = (double *)malloc(rows * cols * sizeof(double));
    MPI_Gatherv(world + H * (int)cols, local_rows * (int)cols, MPI_DOUBLE,
                result, cell_counts, cell_offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0 && do_final) dump_final_state(opts->final_path, result, rows, cols);

    if (gif) ge_close_gif(gif);
    free(frame_buf);
    free(world); free(world_b); free(w);
    free(counts); free(offsets); free(cell_counts); free(cell_offsets);
    return result;
}
