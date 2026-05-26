# **LENIA:** *Game of Life with MPI*

Our scripts and modified source code can be found in our [GitHub repository](https://github.com/Edelwy/HPC). [**Lenia**](https://chakazul.github.io/lenia.html) is a continuous cellular automaton generalising the **Game of Life**. [Instructions](https://github.com/Edelwy/HPC/blob/3e0556dc028157fbecde8640b65139431db06004/Assignment4/Assignment4.md) are available in this repository as well.

## Project structure and running

The `lenia.h` header is included in both `main_seq.c` and `main_mpi.c`, which function as separate entry points depending on whether MPI is used or not.

```text
make METHOD=seq  : gcc   + main_seq.c + lenia_seq.c
make METHOD=row  : mpicc + main_mpi.c + lenia_row.c
make METHOD=block
make METHOD=row_wide
```

We have one sequential version `lenia_seq.c` and three MPI versions:

- `lenia_row.c` - row-wise decomposition
- `lenia_block.c` - block-wise decomposition
- `lenia_row_wide.c` - row-wise decomposition with wider halos and less frequent communication

All common functions are implemented in `lenia_common.c`, so the main decomposition-specific logic lives in `evolve_lenia`.

For **time measurement** in the MPI case, each rank measures its local elapsed time and rank 0 reports the **maximum** across ranks, i.e. the slowest rank determines the step time.

## Implementation

### Sequential implementation

First we improved the initial **sequential version**, analogously to what we did in [Assignment 2](https://github.com/Edelwy/HPC/blob/main/Assignment2/src/src/lenia_opt.cu). The improved version was around three times faster than the one provided and we use the improved one as the basis for all our speed comparisons. Like before, we replaced expensive `pow` and modulo operations where possible, fused two full grid passes into one, and used a double-buffer instead of in-place update.

```c
double *tmp = world; world = world_b; world_b = tmp;
```

We read all values from `world`, write all values to `world_b`, swap pointers, and repeat. This eliminates the read-after-write hazard, allows loop fusion, and replaces `fmin` / `fmax` calls with a cheaper if-else structure.

### MPI implementations

#### Row implementation

First we implemented the **row-rank** version. We compute the row partition from the number of rows divided by the number of processes. If the division is uneven, the lowest ranks get one additional row until the remainder is distributed.

The grid is stored by rows, so each cell `(i, j)` is accessed via `i * cols + j`. We use:

```c
int *counts       = (int *)calloc(procs, sizeof(int));
int *offsets      = (int *)calloc(procs, sizeof(int));
int *cell_counts  = (int *)calloc(procs, sizeof(int));
int *cell_offsets = (int *)calloc(procs, sizeof(int));
```

Here `counts` gives the number of rows per rank, `offsets` the row offset, and `cell_counts` / `cell_offsets` the same information measured in elements instead of rows.

Each rank generates its own kernel. Rank 0 builds the full starting world and scatters the owned row strips to all ranks. Each rank allocates a padded local buffer with top and bottom halo rows:

```c
const int padded = (local_rows + 2 * R) * (int)cols;
double *world   = (double *)calloc(padded, sizeof(double));
```

The strips are distributed with `MPI_Scatterv`, skipping the halo rows:

```c
MPI_Scatterv(
    world_full,
    cell_counts,
    cell_offsets,
    MPI_DOUBLE,
    world + R * cols,
    local_rows * cols,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD
);
```

Neighbouring ranks and halo size are:

```c
const int up   = (rank - 1 + procs) % procs;
const int down = (rank + 1) % procs;
const int halo_cells = R * (int)cols;
```

We then exchange top and bottom halos using `MPI_Sendrecv`, evolve the owned strip, and gather the final unpadded strips back on rank 0 with `MPI_Gatherv`.

#### Block implementation

For the **block-rank** version, we use a 2D Cartesian process grid:

```c
MPI_Dims_create(procs, 2, dims);
```

This factorises the number of processes into two dimensions and stores the result in `dims`. We then create a periodic Cartesian communicator:

```c
int periods[2] = {1, 1};
MPI_Comm cart;
MPI_Cart_create(
    MPI_COMM_WORLD,
    2,
    dims,
    periods,
    0,
    &cart
);
```

Each rank obtains its 2D coordinates and its neighbours:

```c
int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);

int north, south, east, west;
MPI_Cart_shift(cart, 0, 1, &north, &south);
MPI_Cart_shift(cart, 1, 1, &west,  &east);
```

The implementation assumes uniform block sizes and also rejects decompositions where a local block is smaller than the kernel radius. The local block shape and its starting coordinates are:

```c
const int local_rows = (int)rows / dims[0];
const int local_cols = (int)cols / dims[1];
const int my_row0    = coords[0] * local_rows;
const int my_col0    = coords[1] * local_cols;
```

Each rank allocates a halo-padded local block:

```c
const int ext_rows = local_rows + 2 * R;
const int ext_cols = local_cols + 2 * R;
```

The global world is distributed manually rather than with `MPI_Scatterv`, because each rank owns a 2D block rather than a contiguous row strip. Rank 0 packs each block and sends it to the corresponding rank with `MPI_Send`, while non-root ranks receive with `MPI_Recv`.

For east/west halo exchange we use an MPI derived datatype:

```c
MPI_Datatype col_type;
MPI_Type_vector(local_rows, R, ext_cols, MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);
```

This is needed because vertical halo strips are not contiguous in row-major memory. North/south halo rows are contiguous, so a plain count is enough:

```c
const int ns_count = R * ext_cols;
```

The block method then performs four `MPI_Sendrecv` calls per iteration: west, east, north, and south. At the end, rank 0 gathers packed local blocks and reconstructs the full matrix with an `unpack_block` helper.

#### Row implementation with wider halos

This version is very similar to the original row version. We compute a wider halo width `H = K * R`, where `K` is the number of local steps between communications. The method aborts if the smallest owned strip is thinner than the wide halo width.

Communication uses `H` instead of `R`, and the evolution loop contains an additional **substeps** loop which performs up to `K` local updates before the next halo exchange.

## Results

Below are the execution times in seconds averaged over 5 runs for the **row-rank** and **block-rank** implementations. We did not compute size `128 × 128` on `32` ranks for the row decomposition because each rank needs 13 rows of halo above and below, and the owned strip would be thinner than the halo width, causing overlap.

### Average execution time (s) - row

| Size | seq | 1 | 2 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 128  | 1.18 | 1.14 | 0.60 | 0.36 | 0.14 | — |
| 512  | 19.49 | 19.31 | 9.22 | 5.56 | 1.24 | 0.71 |
| 1024 | 76.95 | 75.05 | 37.43 | 22.24 | 7.24 | 2.93 |
| 2048 | 312.09 | 294.96 | 147.92 | 73.45 | 19.31 | 10.29 |
| 4096 | 1557.65 | 1587.70 | 790.32 | 392.97 | 96.22 | 48.49 |

### Average execution time (s) - block

| Size | seq | 1 | 2 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 128  | 1.18 | 1.01 | 0.51 | 0.26 | 0.07 | 0.04 |
| 512  | 19.49 | 16.13 | 8.07 | 4.15 | 1.03 | 0.53 |
| 1024 | 76.95 | 64.49 | 32.26 | 16.40 | 4.11 | 2.04 |
| 2048 | 312.09 | 258.11 | 129.10 | 64.59 | 16.38 | 8.11 |
| 4096 | 1557.65 | 1032.91 | 516.59 | 258.57 | 66.63 | 33.29 |

As one can see from the execution times and speed-ups, the **32-rank version was fastest** in all cases. It outperformed the other alternatives even on small grids, which was not necessarily obvious due to communication overhead. However, the speed-up increased with grid size.

### Speed-up relative to sequential - row

| Size | 1 | 2 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 128  | 1.03 | 1.98 | 3.26 | 8.68 | — |
| 512  | 1.01 | 2.11 | 3.51 | 15.69 | 27.37 |
| 1024 | 1.03 | 2.06 | 3.46 | 10.62 | 26.25 |
| 2048 | 1.06 | 2.11 | 4.25 | 16.16 | 30.33 |
| 4096 | 0.98 | 1.97 | 3.96 | 16.19 | 32.12 |

### Speed-up relative to sequential - block

| Size | 1 | 2 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 128  | 1.16 | 2.32 | 4.49 | 16.94 | 30.24 |
| 512  | 1.21 | 2.41 | 4.70 | 18.89 | 36.99 |
| 1024 | 1.19 | 2.39 | 4.69 | 18.71 | 37.63 |
| 2048 | 1.21 | 2.42 | 4.83 | 19.06 | 38.47 |
| 4096 | 1.51 | 3.02 | 6.02 | 23.38 | 46.79 |

The MPI overhead was too small to make the 1-rank version slower than the sequential one; in fact, it was slightly faster. This is likely due to minor differences in the inner loop, but the speed-up is negligible.

The **block** grid division slightly outperformed the **row** division in all measured cases. The best speed-up from the row method to the block method was **1.95** for the `128 × 128` grid on `16` ranks. The derived data types used in the block-rank method also contribute to the speed-up.

### Speed-up: block vs row

| Size | 1 | 2 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 128  | 1.13 | 1.17 | 1.38 | 1.95 | — |
| 512  | 1.20 | 1.14 | 1.34 | 1.20 | 1.35 |
| 1024 | 1.16 | 1.16 | 1.36 | 1.76 | 1.43 |
| 2048 | 1.14 | 1.15 | 1.14 | 1.18 | 1.27 |
| 4096 | 1.54 | 1.53 | 1.52 | 1.44 | 1.46 |

All the previous versions were tested on **one node**. We also tested both the row and block implementations on **two nodes** by assigning each node half the processes. For size `1024 × 1024` and `4096 × 4096` on `16` and `32` processes, this barely produced any difference. The only difference that was not negligible was for the `1024 × 1024` grid on `16` processes, which was roughly **12% slower** on 2 nodes for the row-rank method.

### Speed-up: 2 nodes vs 1 node - block

| Size | 16 | 32 |
|---|---:|---:|
| 1024 | 1.008 | 0.982 |
| 4096 | 1.032 | 1.024 |

### Speed-up: 2 nodes vs 1 node - row

| Size | 16 | 32 |
|---|---:|---:|
| 1024 | 0.889 | 1.004 |
| 4096 | 0.993 | 1.030 |

Finally, we tried **reducing communication overhead** by taking a larger padding and updating it every `K` steps. We tested this for the two larger cases where it makes the most sense, due to the halo becoming quite large for larger `K`, and on the two best-performing rank sizes. This actually made the algorithm slower, with the slow-down increasing with the size of `K`.

### Mean time (s) per K - 1 node

| Size | Procs | K=1 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|---:|
| 2048 | 16 | 19.49 | 21.12 | 24.96 | 32.68 |
| 4096 | 32 | 49.98 | 54.86 | 68.48 | 108.59 |

### Mean time (s) per K - 2 nodes

| Size | Procs | K=1 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|---:|
| 2048 | 16 | 21.21 | 22.56 | 25.30 | 33.04 |
| 4096 | 32 | 50.73 | 53.94 | 68.50 | 90.17 |
