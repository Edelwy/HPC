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

/* Convolve+grow+clip a single owned cell using a halo-padded extended buffer.
 * No wrap is needed: the halos already contain the correct toroidal neighbours. */
static inline double evolve_cell_block(const double *src, const double *w,
                                       int ext_i, int ext_j, int ext_cols,
                                       int kernel_size, int R, double dt)
{
    double sum = 0.0;
    for (int ki = 0; ki < kernel_size; ki++) {
        int ni = ext_i + ki - R;
        for (int kj = 0; kj < kernel_size; kj++) {
            int nj = ext_j + kj - R;
            sum += w[(kernel_size - 1 - ki) * kernel_size + (kernel_size - 1 - kj)]
                 * src[ni * ext_cols + nj];
        }
    }
    double v = src[ext_i * ext_cols + ext_j] + dt * growth_lenia(sum);
    if (v < 0.0) v = 0.0; else if (v > 1.0) v = 1.0;
    return v;
}

/* Helper: copy a packed block (local_rows * local_cols doubles) into world_full
 * at the position determined by the rank's Cartesian coords. */
static void unpack_block(double *world_full, const double *gather_pack,
                         MPI_Comm cart, int procs, int local_rows, int local_cols,
                         unsigned int cols)
{
    for (int r = 0; r < procs; r++) {
        int rc[2]; MPI_Cart_coords(cart, r, 2, rc);
        int row0 = rc[0] * local_rows, col0 = rc[1] * local_cols;
        for (int i = 0; i < local_rows; i++) {
            memcpy(world_full + (row0 + i) * cols + col0,
                   gather_pack + r * local_rows * local_cols + i * local_cols,
                   local_cols * sizeof(double));
        }
    }
}

double *evolve_lenia(unsigned int rows, unsigned int cols, unsigned int steps,
                     double dt, unsigned int kernel_size,
                     const struct orbium_coo *orbiums, unsigned int num_orbiums,
                     const struct lenia_opts *opts)
{
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    /* 2-D Cartesian topology with toroidal wrap. */
    int dims[2] = {0, 0};
    MPI_Dims_create(procs, 2, dims);
    int periods[2] = {1, 1};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);
    int north, south, east, west;
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west,  &east);

    const int R = (int)kernel_size / 2;

    if ((int)rows % dims[0] != 0 || (int)cols % dims[1] != 0) {
        if (rank == 0) fprintf(stderr,
            "lenia_block: grid %ux%u must divide evenly by process grid %dx%d\n",
            rows, cols, dims[0], dims[1]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int local_rows = (int)rows / dims[0];
    const int local_cols = (int)cols / dims[1];
    const int my_row0    = coords[0] * local_rows;
    const int my_col0    = coords[1] * local_cols;

    if (local_rows < R || local_cols < R) {
        if (rank == 0) fprintf(stderr,
            "lenia_block: local block %dx%d smaller than kernel radius %d; "
            "use lenia_row for this (N,P) combination\n",
            local_rows, local_cols, R);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int ext_rows = local_rows + 2 * R;
    const int ext_cols = local_cols + 2 * R;

    double *w = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    generate_kernel(w, kernel_size);

    double *world   = (double *)calloc(ext_rows * ext_cols, sizeof(double));
    double *world_b = (double *)calloc(ext_rows * ext_cols, sizeof(double));

    /* --- Distribute initial world. Rank 0 builds it, sends each rank its block. --- */
    {
        double *world_full = NULL;
        if (rank == 0) {
            world_full = (double *)calloc(rows * cols, sizeof(double));
            for (unsigned int o = 0; o < num_orbiums; o++)
                place_orbium(world_full, rows, cols,
                             orbiums[o].row, orbiums[o].col, orbiums[o].angle);
        }
        double *recv_pack = (double *)malloc(local_rows * local_cols * sizeof(double));
        if (rank == 0) {
            /* Place own block. */
            for (int i = 0; i < local_rows; i++)
                memcpy(world + (R + i) * ext_cols + R,
                       world_full + (my_row0 + i) * cols + my_col0,
                       local_cols * sizeof(double));
            /* Send everyone else's block. */
            double *send_pack = (double *)malloc(local_rows * local_cols * sizeof(double));
            for (int r = 1; r < procs; r++) {
                int rc[2]; MPI_Cart_coords(cart, r, 2, rc);
                int row0 = rc[0] * local_rows, col0 = rc[1] * local_cols;
                for (int i = 0; i < local_rows; i++)
                    memcpy(send_pack + i * local_cols,
                           world_full + (row0 + i) * cols + col0,
                           local_cols * sizeof(double));
                MPI_Send(send_pack, local_rows * local_cols, MPI_DOUBLE, r, 0, cart);
            }
            free(send_pack);
            free(world_full);
        } else {
            MPI_Recv(recv_pack, local_rows * local_cols, MPI_DOUBLE, 0, 0,
                     cart, MPI_STATUS_IGNORE);
            for (int i = 0; i < local_rows; i++)
                memcpy(world + (R + i) * ext_cols + R,
                       recv_pack + i * local_cols,
                       local_cols * sizeof(double));
        }
        free(recv_pack);
    }
    memcpy(world_b, world, ext_rows * ext_cols * sizeof(double));

    /* --- Derived datatype for E/W column halos. --- */
    MPI_Datatype col_type;
    MPI_Type_vector(local_rows, R, ext_cols, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);
    const int ns_count = R * ext_cols;  /* contiguous after E/W pass populates corners */

    int do_gif   = (opts && opts->gif_path)   ? 1 : 0;
    int do_final = (opts && opts->final_path) ? 1 : 0;

    ge_GIF *gif = NULL;
    double *gather_pack = NULL;  /* rank 0 only: concatenated blocks */
    double *frame_buf   = NULL;  /* rank 0 only: rectangular world buffer */
    double *send_pack   = NULL;  /* every rank: packed own block */
    if (do_gif || do_final) send_pack = (double *)malloc(local_rows * local_cols * sizeof(double));
    if (rank == 0 && (do_gif || do_final)) {
        gather_pack = (double *)malloc((size_t)procs * local_rows * local_cols * sizeof(double));
        frame_buf   = (double *)malloc(rows * cols * sizeof(double));
    }
    if (rank == 0 && do_gif)
        gif = ge_new_gif(opts->gif_path, cols, rows, inferno_pallete, 8, -1, 0);

    for (unsigned int step = 0; step < steps; step++) {
        /* E/W column halo (derived MPI_Type_vector). */
        MPI_Sendrecv(world + R * ext_cols + R,                 1, col_type, west, 0,
                     world + R * ext_cols + R + local_cols,    1, col_type, east, 0,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(world + R * ext_cols + local_cols,        1, col_type, east, 1,
                     world + R * ext_cols + 0,                 1, col_type, west, 1,
                     cart, MPI_STATUS_IGNORE);
        /* N/S full-width halo (contiguous; corners arrive via the just-filled E/W halos). */
        MPI_Sendrecv(world + R * ext_cols,                     ns_count, MPI_DOUBLE, north, 2,
                     world + (R + local_rows) * ext_cols,      ns_count, MPI_DOUBLE, south, 2,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(world + local_rows * ext_cols,            ns_count, MPI_DOUBLE, south, 3,
                     world + 0,                                ns_count, MPI_DOUBLE, north, 3,
                     cart, MPI_STATUS_IGNORE);

        for (int li = 0; li < local_rows; li++) {
            int ext_i = li + R;
            for (int lj = 0; lj < local_cols; lj++) {
                int ext_j = lj + R;
                world_b[ext_i * ext_cols + ext_j] =
                    evolve_cell_block(world, w, ext_i, ext_j, ext_cols,
                                      (int)kernel_size, R, dt);
            }
        }
        double *tmp = world; world = world_b; world_b = tmp;

        if (do_gif) {
            for (int i = 0; i < local_rows; i++)
                memcpy(send_pack + i * local_cols,
                       world + (R + i) * ext_cols + R,
                       local_cols * sizeof(double));
            MPI_Gather(send_pack, local_rows * local_cols, MPI_DOUBLE,
                       gather_pack, local_rows * local_cols, MPI_DOUBLE, 0, cart);
            if (rank == 0) {
                unpack_block(frame_buf, gather_pack, cart, procs, local_rows, local_cols, cols);
                for (unsigned int i = 0; i < rows * cols; i++)
                    gif->frame[i] = (uint8_t)(frame_buf[i] * 255);
                ge_add_frame(gif, 5);
            }
        }
    }

    /* Final gather to rank 0. */
    double *result = NULL;
    if (rank == 0) result = (double *)malloc(rows * cols * sizeof(double));
    if (!send_pack) send_pack = (double *)malloc(local_rows * local_cols * sizeof(double));
    if (rank == 0 && !gather_pack)
        gather_pack = (double *)malloc((size_t)procs * local_rows * local_cols * sizeof(double));
    for (int i = 0; i < local_rows; i++)
        memcpy(send_pack + i * local_cols,
               world + (R + i) * ext_cols + R,
               local_cols * sizeof(double));
    MPI_Gather(send_pack, local_rows * local_cols, MPI_DOUBLE,
               gather_pack, local_rows * local_cols, MPI_DOUBLE, 0, cart);
    if (rank == 0) {
        unpack_block(result, gather_pack, cart, procs, local_rows, local_cols, cols);
        if (do_final) dump_final_state(opts->final_path, result, rows, cols);
    }

    if (gif) ge_close_gif(gif);
    free(send_pack);
    free(gather_pack);
    free(frame_buf);
    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart);
    free(world);
    free(world_b);
    free(w);
    return result;
}
