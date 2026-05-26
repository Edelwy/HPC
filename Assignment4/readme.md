# **LENIA:** *Game of Life with MPI*

Our scripts and modified source code can be found in our [GitHub repository](https://github.com/Edelwy/HPC). [**Lenia**](https://chakazul.github.io/lenia.html) is a continuous cellular automaton generalising the **Game of Life**. [Instructions](https://github.com/Edelwy/HPC/blob/3e0556dc028157fbecde8640b65139431db06004/Assignment4/Assignment4.md) are available in this repository as well. 

## Project structure and running

The **project structure** is the following: the `lenia.h` header is included in both `main_seq.c` and `main_mpi.c` files, which function as separate starting points depending on whether *MPI* is used or not.

```
make METHOD=seq  : gcc  + main_seq.c  + lenia_seq.c
make METHOD=row  : mpicc + main_mpi.c  + lenia_row.c
```

Then we have one sequential version `lenia_seq.c` and three *MPI* versions with default one being `lenia_row.c`. Every rank runs the same main, but inside `evolve_lenia` each rank owns a piece of the grid and communicates halos. In the default version the piece of the grid is a **row**. In the `lenia_block.c` we have blocks instead of rows. In the `lenia_row_wide.c` the grids are rows, however the halos are larger in order to minimize the number of transactions. All the common functions are implemented in `lenia_common.c` these remain the same throughout versions. Therefore the only thing really implemented separatly is the `evolve_lenia` function.

For **time measurment** in the *MPI* case, we send all times from all ranks to rank zero, the one we use for communication in our case, and take the **slowest time** as elapsed time.

## Implementation

### Sequential implementation

First we improved the initial **sequential version**, analogously to what we did [Assignment 2](https://github.com/Edelwy/HPC/blob/main/Assignment2/src/src/lenia_opt.cu). The improved verison was three times faster than the one provided and we use the improved one as the basis for all our speed comparisons. Like before we replaced the `pow` and modulo operations, we fused two full grid passes into one and used a double-buffer instead of in-place update.

```
double *tmp = world; world = world_b; world_b = tmp;
```

We read all values from `world`, write all values to `world_b`, swap pointers, and repeat. This eliminates the read-after-write hazard so the inner loop is allowed to be fused, and lets us replace `fmin` and `fmax`, two libm calls, with a cheaper if-else structure.

### MPI implementations

#### Row implementation

First we implemented the **row-rank** version. First we compute the row partition based on the number of rows divided by the number of processes. The remainder of this division is assigned as one extra row for the first ranks until needed. 

The grid is stored by rows, so each cell $(i, j)$ can be accessed via index `i * cols + j`. Then we have the following setup:

```c
    int *counts       = (int *)calloc(procs, sizeof(int));
    int *offsets      = (int *)calloc(procs, sizeof(int));
    int *cell_counts  = (int *)calloc(procs, sizeof(int));
    int *cell_offsets = (int *)calloc(procs, sizeof(int));
```

So `counts` gives us the number of rows for rank, `offsets` gives us the offset where the first row starts. Similarly for `cell_counts` and `cell_offsets` but using cell indices.

Afterwards we generate the kernel for each rank has its own. Only the zero rank builds the full starting world a piece of which is then sent to each rank. The others have it as `NULL` so the sending is ignored. We free this after the pieces are scattered. We use a row plus bottom and top halo:

```c
    /* R is the kernel radius. */
    const int padded = (local_rows + 2 * R) * (int)cols;
    double *world   = (double *)calloc(padded, sizeof(double));
```

Meaning each rank has its own **padded world** which it fills.
To actually send the pieces of the world to each rank we use `MPI_Scatterv`. We skip the upper halo and bottom halo, they should remain empty.

```c
MPI_Scatterv(
    world_full,        // Send buffer on root.
    cell_counts,       // How many doubles to send to each rank.
    cell_offsets,      // Where each rank's piece starts.
    MPI_DOUBLE,        // Datatype.
    world + R*cols,    // Recieve buffer.
    local_rows * cols, // How many doubles this rank receives.
    MPI_DOUBLE,        // Datatype.
    0,                 // Root, which is rank zero.
    MPI_COMM_WORLD
);
```

These three following lines define who each rank talks to for halo exchange, and how much data each message carries:

```c
    const int up   = (rank - 1 + procs) % procs;
    const int down = (rank + 1) % procs;
    const int halo_cells = R * (int)cols;
```
We don't just use `rank - 1` and `rank + 1` because this is calculated via modulo. 

Now that we finished the setup we can do the main part of the evolution. First we send the current rank's top owned rows to the rank above and receive from the rank below into current rank's bottom halo. We do this via `MPI_Sendrecv`. 

```c
MPI_Sendrecv(
    world + R * (int)cols,                // Start at first unpadded cell.
    halo_cells,                           // Send the halo number of cells.
    MPI_DOUBLE,                           // Datatype.
    up,                                   // Send to upper rank.
    0,                                    // Sender tag.
    world + (R + local_rows) * (int)cols, // Recieve in bottom halo.
    halo_cells,                           // Halo size.
    MPI_DOUBLE,                           // Datatype.
    down,                                 // Recieve from bottom rank.
    0,                                    // Reciever tag.
    MPI_COMM_WORLD,     
    MPI_STATUS_IGNORE
);

MPI_Sendrecv(
    world + local_rows * (int)cols,       // Start at last R unpadded cells.
    halo_cells,                           // Send the halo number of cells.
    MPI_DOUBLE,                           // Datatype. 
    down,                                 // Send to bottom rank.
    1,                                    // Sender tag.
    world,                                // Recieve in upper halo.
    halo_cells,                           // Halo size.
    MPI_DOUBLE,                           // Datatype.
    up,                                   // Recieve from upper rank.
    1,                                    // Reciever tag.
    MPI_COMM_WORLD, 
    MPI_STATUS_IGNORE
);
```

Than we do the same for the reversed bottom-up halo situation. The rest is just cell evolution. Now at the end we have to **gather** all the information from all ranks to our root rank zero. We do this via `MPI_Gatherv`.

```c
MPI_Gatherv(
    world + R * (int)cols,  // Each rank sends its unpadded strip.
    local_rows * (int)cols, // Size of the strip.
    MPI_DOUBLE,             // Datatype.
    result,                 // Recieve buffer only on root elsewhere NULL.
    cell_counts,            // How many doubles each rank sends. 
    cell_offsets,           // Where each piece starts.
    MPI_DOUBLE,             // Datatype.
    0,                      // Root, which is zero.
    MPI_COMM_WORLD
);
```

#### Block implementation

We want a 2-dimensional process grid. We do this via:

```c
MPI_Dims_create(procs, 2, dims);
```

This gives us a nice factorization into two dimensions based on the number of processes, the dimensions are saved to `dims`. Since the dimensions wrap around, we use a new communicator since usually all ranks are in a flat list. The new communicator where the same ranks are organized as a 2D Cartesian grid is `cart`.

```c
int periods[2] = {1, 1};
MPI_Comm cart;
MPI_Cart_create(
    MPI_COMM_WORLD, 
    2,                  // 2 dimensions.
    dims,               // Shape of the grid.
    periods,            // Which dimensions wrap around (all).
    0,                  // Do not reoder ranks.
    &cart               // Output communicator.
);
```

We then obtain the coordinates of the current rank:
```c
    int coords[2]; MPI_Cart_coords(cart, rank, 2, coords);
```

We also want the neighbouring ranks:
```c
    int north, south, east, west;
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west,  &east);
```

We abort the process if the blocks are **not of uniform size**, because our algorithm assumes this in general, since for our grids this is usually the case. Similarly if the block is **not big enough** for the kernel radius. 

We calculate the sizes of the current rank and the beggining row and column:

```c
    const int local_rows = (int)rows / dims[0];
    const int local_cols = (int)cols / dims[1];
    const int my_row0    = coords[0] * local_rows;
    const int my_col0    = coords[1] * local_cols;
```

Here we also need a padding in each size, so each rank will have their buffer of the world as the padded rows and columns in each direction - east, west, north and south.

```c
    const int ext_rows = local_rows + 2 * R;
    const int ext_cols = local_cols + 2 * R;
```

Then we have to **distribute the world** to the ranks like we did in the row version, however here it was more complicated, so we did not use `MPI_Scatterv`. Again the root rank 0 creates the full world. Each rank also allocates a flat buffer big enough to hold exactly each owned block, this is used when receaving data from root rank. Root rank zero then **copies** its own block directly into the interior of world. Then it loops over all processes and sends each rank its part using `MPI_Send`.

```c
MPI_Send(
    send_pack,                  // Buffer being sent.
    local_rows * local_cols,    // Size of the packet being sent.
    MPI_DOUBLE,                 // Datatype.
    r,                          // Destination rank.
    0,                          // Message tag.
    cart                        // Communicator in which this happens.
);
```

The other ranks however do not do the sending, only recieving via `MPI_Recv`. This is then copied into the interior of their world (skipping the padding like before).

```c
MPI_Recv(
    recv_pack,                  // Buffer to which data is written.
    local_rows * local_cols,    // Size of the data recieved  
    MPI_DOUBLE,                 // Datatype.
    0,                          // Root rank is zero.
    0,                          // Message tag.
    cart,                       // Communicator.
    MPI_STATUS_IGNORE           // Ignore messages.
);
```

We then also use a **custom MPI datatype variable**. It describes a vertical block of columns inside the extended 2D local array. That is needed because east and west halos are not contiguous in memory.

```c
    MPI_Datatype col_type;          // Custom type.
    MPI_Type_vector(
        local_rows,                 // How many rows the column halo spans.
        R,                          // How many doubles per row to include.
        ext_cols,                   // From one row to the next in memory.
        MPI_DOUBLE,                 // Base datatype.
        &col_type
    );
    MPI_Type_commit(&col_type); 
```
We don't have this problem with east and west memory though since these are continous in memory, so we can use:

```c
    const int ns_count = R * ext_cols;  
```

Then we we go into the the main evolution loop. Here everything is similar to the process before, only that now we have 4 send and recieve calls to `MPI_Sendrecv` one for each halo. First we have the west, then east, then top, and finally bottom.

At the end do the same as before. Rank zero saves the result using `MPI_Gather`. We then implemented an `unpack_block` function that actually computes where that block belongs in the full global matrix depending on the rank coordinates, since the gathering just concatinates the chunks. We don't need this in the row-rank version.

#### Row implementation with wider halos

Very similar implementation to the original row version. Firslty, we abort this method if the smallest row strip is smaller than the wide halo width. Here the padding is calculated using `H = K * R` where `K` is the number of steps we wish to skip. 

Everywhere in the sending and recieving `H` is used instead of the radius. Then for the evolution loop we create another **substeps loop** where we evolve it like in the sequential version and only update it after `K` substeps have passed.

## Results