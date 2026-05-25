#ifndef LENIA_H
#define LENIA_H

#ifdef __cplusplus
extern "C" {
#endif

struct orbium_coo {
    int row;
    int col;
    int angle;
};

/* Optional run-time settings shared by every variant. NULL paths disable I/O. */
struct lenia_opts {
    const char *gif_path;    /* if non-NULL, write per-step GIF frames */
    const char *final_path;  /* if non-NULL, dump final world to a text file */
    int halo_steps;          /* exchange every K steps (only lenia_row_wide: default 1) */
};

/* Returns the final world buffer (rows*cols doubles) on rank 0; NULL on other ranks.
 * For lenia_seq.c the buffer is always returned. Caller takes ownership and frees. */
double *evolve_lenia(unsigned int rows, unsigned int cols, unsigned int steps,
                     double dt, unsigned int kernel_size,
                     const struct orbium_coo *orbiums, unsigned int num_orbiums,
                     const struct lenia_opts *opts);

#ifdef __cplusplus
}
#endif
#endif
