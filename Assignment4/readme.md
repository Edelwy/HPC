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

Below are the execution times in seconds averaged over 5 runs for the **row-rank** implementation and **block-rank** implementation. We did not compute size `128 × 128` on `32` ranks because each rank needs 13 rows of halo above and below, and the owned strip would be thinner than the halo width, causing overlap.

<table><tr><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Average execution time (s) for method: row</div><style type="text/css">
#T_0c72e_row0_col0, #T_0c72e_row1_col0, #T_0c72e_row2_col0, #T_0c72e_row3_col0, #T_0c72e_row4_col1 {
  background-color: darkred;
}
#T_0c72e_row0_col4, #T_0c72e_row1_col5, #T_0c72e_row2_col5, #T_0c72e_row3_col5, #T_0c72e_row4_col5 {
  background-color: darkgreen;
}
</style>
<table id="T_0c72e">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0c72e_level0_col0" class="col_heading level0 col0" >seq</th>
      <th id="T_0c72e_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_0c72e_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_0c72e_level0_col3" class="col_heading level0 col3" >4</th>
      <th id="T_0c72e_level0_col4" class="col_heading level0 col4" >16</th>
      <th id="T_0c72e_level0_col5" class="col_heading level0 col5" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0c72e_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_0c72e_row0_col0" class="data row0 col0" >1.18</td>
      <td id="T_0c72e_row0_col1" class="data row0 col1" >1.14</td>
      <td id="T_0c72e_row0_col2" class="data row0 col2" >0.6</td>
      <td id="T_0c72e_row0_col3" class="data row0 col3" >0.36</td>
      <td id="T_0c72e_row0_col4" class="data row0 col4" >0.14</td>
      <td id="T_0c72e_row0_col5" class="data row0 col5" ></td>
    </tr>
    <tr>
      <th id="T_0c72e_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_0c72e_row1_col0" class="data row1 col0" >19.49</td>
      <td id="T_0c72e_row1_col1" class="data row1 col1" >19.31</td>
      <td id="T_0c72e_row1_col2" class="data row1 col2" >9.22</td>
      <td id="T_0c72e_row1_col3" class="data row1 col3" >5.56</td>
      <td id="T_0c72e_row1_col4" class="data row1 col4" >1.24</td>
      <td id="T_0c72e_row1_col5" class="data row1 col5" >0.71</td>
    </tr>
    <tr>
      <th id="T_0c72e_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_0c72e_row2_col0" class="data row2 col0" >76.95</td>
      <td id="T_0c72e_row2_col1" class="data row2 col1" >75.05</td>
      <td id="T_0c72e_row2_col2" class="data row2 col2" >37.43</td>
      <td id="T_0c72e_row2_col3" class="data row2 col3" >22.24</td>
      <td id="T_0c72e_row2_col4" class="data row2 col4" >7.24</td>
      <td id="T_0c72e_row2_col5" class="data row2 col5" >2.93</td>
    </tr>
    <tr>
      <th id="T_0c72e_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_0c72e_row3_col0" class="data row3 col0" >312.09</td>
      <td id="T_0c72e_row3_col1" class="data row3 col1" >294.96</td>
      <td id="T_0c72e_row3_col2" class="data row3 col2" >147.92</td>
      <td id="T_0c72e_row3_col3" class="data row3 col3" >73.45</td>
      <td id="T_0c72e_row3_col4" class="data row3 col4" >19.31</td>
      <td id="T_0c72e_row3_col5" class="data row3 col5" >10.29</td>
    </tr>
    <tr>
      <th id="T_0c72e_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_0c72e_row4_col0" class="data row4 col0" >1557.65</td>
      <td id="T_0c72e_row4_col1" class="data row4 col1" >1587.7</td>
      <td id="T_0c72e_row4_col2" class="data row4 col2" >790.32</td>
      <td id="T_0c72e_row4_col3" class="data row4 col3" >392.97</td>
      <td id="T_0c72e_row4_col4" class="data row4 col4" >96.22</td>
      <td id="T_0c72e_row4_col5" class="data row4 col5" >48.49</td>
    </tr>
  </tbody>
</table>
</td><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Average execution time (s) for method: block</div><style type="text/css">
#T_f924c_row0_col0, #T_f924c_row1_col0, #T_f924c_row2_col0, #T_f924c_row3_col0, #T_f924c_row4_col0 {
  background-color: darkred;
}
#T_f924c_row0_col5, #T_f924c_row1_col5, #T_f924c_row2_col5, #T_f924c_row3_col5, #T_f924c_row4_col5 {
  background-color: darkgreen;
}
</style>
<table id="T_f924c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f924c_level0_col0" class="col_heading level0 col0" >seq</th>
      <th id="T_f924c_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_f924c_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_f924c_level0_col3" class="col_heading level0 col3" >4</th>
      <th id="T_f924c_level0_col4" class="col_heading level0 col4" >16</th>
      <th id="T_f924c_level0_col5" class="col_heading level0 col5" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f924c_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_f924c_row0_col0" class="data row0 col0" >1.18</td>
      <td id="T_f924c_row0_col1" class="data row0 col1" >1.01</td>
      <td id="T_f924c_row0_col2" class="data row0 col2" >0.51</td>
      <td id="T_f924c_row0_col3" class="data row0 col3" >0.26</td>
      <td id="T_f924c_row0_col4" class="data row0 col4" >0.07</td>
      <td id="T_f924c_row0_col5" class="data row0 col5" >0.04</td>
    </tr>
    <tr>
      <th id="T_f924c_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_f924c_row1_col0" class="data row1 col0" >19.49</td>
      <td id="T_f924c_row1_col1" class="data row1 col1" >16.13</td>
      <td id="T_f924c_row1_col2" class="data row1 col2" >8.07</td>
      <td id="T_f924c_row1_col3" class="data row1 col3" >4.15</td>
      <td id="T_f924c_row1_col4" class="data row1 col4" >1.03</td>
      <td id="T_f924c_row1_col5" class="data row1 col5" >0.53</td>
    </tr>
    <tr>
      <th id="T_f924c_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_f924c_row2_col0" class="data row2 col0" >76.95</td>
      <td id="T_f924c_row2_col1" class="data row2 col1" >64.49</td>
      <td id="T_f924c_row2_col2" class="data row2 col2" >32.26</td>
      <td id="T_f924c_row2_col3" class="data row2 col3" >16.4</td>
      <td id="T_f924c_row2_col4" class="data row2 col4" >4.11</td>
      <td id="T_f924c_row2_col5" class="data row2 col5" >2.04</td>
    </tr>
    <tr>
      <th id="T_f924c_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_f924c_row3_col0" class="data row3 col0" >312.09</td>
      <td id="T_f924c_row3_col1" class="data row3 col1" >258.11</td>
      <td id="T_f924c_row3_col2" class="data row3 col2" >129.1</td>
      <td id="T_f924c_row3_col3" class="data row3 col3" >64.59</td>
      <td id="T_f924c_row3_col4" class="data row3 col4" >16.38</td>
      <td id="T_f924c_row3_col5" class="data row3 col5" >8.11</td>
    </tr>
    <tr>
      <th id="T_f924c_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_f924c_row4_col0" class="data row4 col0" >1557.65</td>
      <td id="T_f924c_row4_col1" class="data row4 col1" >1032.91</td>
      <td id="T_f924c_row4_col2" class="data row4 col2" >516.59</td>
      <td id="T_f924c_row4_col3" class="data row4 col3" >258.57</td>
      <td id="T_f924c_row4_col4" class="data row4 col4" >66.63</td>
      <td id="T_f924c_row4_col5" class="data row4 col5" >33.29</td>
    </tr>
  </tbody>
</table>
</td></tr></table>

As one can see from the execution times and speed-ups, the **32-rank version was fastest** in all cases. It outperformed the other alternatives even on small grids, which was not necessarily obvious due to communication overhead. However, the speed-up increased with grid size.

<table><tr><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Speed-up for method: row</div><style type="text/css">
#T_19b19_row0_col0, #T_19b19_row1_col0, #T_19b19_row2_col0, #T_19b19_row3_col0, #T_19b19_row4_col0 {
  background-color: darkred;
}
#T_19b19_row0_col3, #T_19b19_row1_col4, #T_19b19_row2_col4, #T_19b19_row3_col4, #T_19b19_row4_col4 {
  background-color: darkgreen;
}
</style>
<table id="T_19b19">
  <thead>
    <tr>
      <th class="index_name level0" >Procs</th>
      <th id="T_19b19_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_19b19_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_19b19_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_19b19_level0_col3" class="col_heading level0 col3" >16</th>
      <th id="T_19b19_level0_col4" class="col_heading level0 col4" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_19b19_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_19b19_row0_col0" class="data row0 col0" >1.03</td>
      <td id="T_19b19_row0_col1" class="data row0 col1" >1.98</td>
      <td id="T_19b19_row0_col2" class="data row0 col2" >3.26</td>
      <td id="T_19b19_row0_col3" class="data row0 col3" >8.68</td>
      <td id="T_19b19_row0_col4" class="data row0 col4" ></td>
    </tr>
    <tr>
      <th id="T_19b19_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_19b19_row1_col0" class="data row1 col0" >1.01</td>
      <td id="T_19b19_row1_col1" class="data row1 col1" >2.11</td>
      <td id="T_19b19_row1_col2" class="data row1 col2" >3.51</td>
      <td id="T_19b19_row1_col3" class="data row1 col3" >15.69</td>
      <td id="T_19b19_row1_col4" class="data row1 col4" >27.37</td>
    </tr>
    <tr>
      <th id="T_19b19_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_19b19_row2_col0" class="data row2 col0" >1.03</td>
      <td id="T_19b19_row2_col1" class="data row2 col1" >2.06</td>
      <td id="T_19b19_row2_col2" class="data row2 col2" >3.46</td>
      <td id="T_19b19_row2_col3" class="data row2 col3" >10.62</td>
      <td id="T_19b19_row2_col4" class="data row2 col4" >26.25</td>
    </tr>
    <tr>
      <th id="T_19b19_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_19b19_row3_col0" class="data row3 col0" >1.06</td>
      <td id="T_19b19_row3_col1" class="data row3 col1" >2.11</td>
      <td id="T_19b19_row3_col2" class="data row3 col2" >4.25</td>
      <td id="T_19b19_row3_col3" class="data row3 col3" >16.16</td>
      <td id="T_19b19_row3_col4" class="data row3 col4" >30.33</td>
    </tr>
    <tr>
      <th id="T_19b19_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_19b19_row4_col0" class="data row4 col0" >0.98</td>
      <td id="T_19b19_row4_col1" class="data row4 col1" >1.97</td>
      <td id="T_19b19_row4_col2" class="data row4 col2" >3.96</td>
      <td id="T_19b19_row4_col3" class="data row4 col3" >16.19</td>
      <td id="T_19b19_row4_col4" class="data row4 col4" >32.12</td>
    </tr>
  </tbody>
</table>
</td><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Speed-up for method: block</div><style type="text/css">
#T_56055_row0_col0, #T_56055_row1_col0, #T_56055_row2_col0, #T_56055_row3_col0, #T_56055_row4_col0 {
  background-color: darkred;
}
#T_56055_row0_col4, #T_56055_row1_col4, #T_56055_row2_col4, #T_56055_row3_col4, #T_56055_row4_col4 {
  background-color: darkgreen;
}
</style>
<table id="T_56055">
  <thead>
    <tr>
      <th class="index_name level0" >Procs</th>
      <th id="T_56055_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_56055_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_56055_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_56055_level0_col3" class="col_heading level0 col3" >16</th>
      <th id="T_56055_level0_col4" class="col_heading level0 col4" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_56055_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_56055_row0_col0" class="data row0 col0" >1.16</td>
      <td id="T_56055_row0_col1" class="data row0 col1" >2.32</td>
      <td id="T_56055_row0_col2" class="data row0 col2" >4.49</td>
      <td id="T_56055_row0_col3" class="data row0 col3" >16.94</td>
      <td id="T_56055_row0_col4" class="data row0 col4" >30.24</td>
    </tr>
    <tr>
      <th id="T_56055_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_56055_row1_col0" class="data row1 col0" >1.21</td>
      <td id="T_56055_row1_col1" class="data row1 col1" >2.41</td>
      <td id="T_56055_row1_col2" class="data row1 col2" >4.7</td>
      <td id="T_56055_row1_col3" class="data row1 col3" >18.89</td>
      <td id="T_56055_row1_col4" class="data row1 col4" >36.99</td>
    </tr>
    <tr>
      <th id="T_56055_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_56055_row2_col0" class="data row2 col0" >1.19</td>
      <td id="T_56055_row2_col1" class="data row2 col1" >2.39</td>
      <td id="T_56055_row2_col2" class="data row2 col2" >4.69</td>
      <td id="T_56055_row2_col3" class="data row2 col3" >18.71</td>
      <td id="T_56055_row2_col4" class="data row2 col4" >37.63</td>
    </tr>
    <tr>
      <th id="T_56055_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_56055_row3_col0" class="data row3 col0" >1.21</td>
      <td id="T_56055_row3_col1" class="data row3 col1" >2.42</td>
      <td id="T_56055_row3_col2" class="data row3 col2" >4.83</td>
      <td id="T_56055_row3_col3" class="data row3 col3" >19.06</td>
      <td id="T_56055_row3_col4" class="data row3 col4" >38.47</td>
    </tr>
    <tr>
      <th id="T_56055_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_56055_row4_col0" class="data row4 col0" >1.51</td>
      <td id="T_56055_row4_col1" class="data row4 col1" >3.02</td>
      <td id="T_56055_row4_col2" class="data row4 col2" >6.02</td>
      <td id="T_56055_row4_col3" class="data row4 col3" >23.38</td>
      <td id="T_56055_row4_col4" class="data row4 col4" >46.79</td>
    </tr>
  </tbody>
</table>
</td></tr></table>

The MPI overhead was too small to make the 1-rank version slower than the sequential one; in fact, it was slightly faster. This is likely due to minor differences in the inner loop, but the speed-up is negligible.

The **block** grid division slightly outperformed the **row** division in all measured cases. The best speed-up from the row method to the block method was **1.95** for the `128 × 128` grid on `16` ranks. The derived data types used in the block-rank method also contribute to the speed-up.

<table><tr><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Speed-up: block vs row</div><style type="text/css">
#T_33fc4_row0_col0, #T_33fc4_row1_col1, #T_33fc4_row2_col1, #T_33fc4_row3_col2, #T_33fc4_row4_col3 {
  background-color: darkred;
}
#T_33fc4_row0_col3, #T_33fc4_row1_col4, #T_33fc4_row2_col3, #T_33fc4_row3_col4, #T_33fc4_row4_col0 {
  background-color: darkgreen;
}
</style>
<table id="T_33fc4">
  <thead>
    <tr>
      <th class="index_name level0" >Procs</th>
      <th id="T_33fc4_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_33fc4_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_33fc4_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_33fc4_level0_col3" class="col_heading level0 col3" >16</th>
      <th id="T_33fc4_level0_col4" class="col_heading level0 col4" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_33fc4_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_33fc4_row0_col0" class="data row0 col0" >1.13</td>
      <td id="T_33fc4_row0_col1" class="data row0 col1" >1.17</td>
      <td id="T_33fc4_row0_col2" class="data row0 col2" >1.38</td>
      <td id="T_33fc4_row0_col3" class="data row0 col3" >1.95</td>
      <td id="T_33fc4_row0_col4" class="data row0 col4" ></td>
    </tr>
    <tr>
      <th id="T_33fc4_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_33fc4_row1_col0" class="data row1 col0" >1.2</td>
      <td id="T_33fc4_row1_col1" class="data row1 col1" >1.14</td>
      <td id="T_33fc4_row1_col2" class="data row1 col2" >1.34</td>
      <td id="T_33fc4_row1_col3" class="data row1 col3" >1.2</td>
      <td id="T_33fc4_row1_col4" class="data row1 col4" >1.35</td>
    </tr>
    <tr>
      <th id="T_33fc4_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_33fc4_row2_col0" class="data row2 col0" >1.16</td>
      <td id="T_33fc4_row2_col1" class="data row2 col1" >1.16</td>
      <td id="T_33fc4_row2_col2" class="data row2 col2" >1.36</td>
      <td id="T_33fc4_row2_col3" class="data row2 col3" >1.76</td>
      <td id="T_33fc4_row2_col4" class="data row2 col4" >1.43</td>
    </tr>
    <tr>
      <th id="T_33fc4_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_33fc4_row3_col0" class="data row3 col0" >1.14</td>
      <td id="T_33fc4_row3_col1" class="data row3 col1" >1.15</td>
      <td id="T_33fc4_row3_col2" class="data row3 col2" >1.14</td>
      <td id="T_33fc4_row3_col3" class="data row3 col3" >1.18</td>
      <td id="T_33fc4_row3_col4" class="data row3 col4" >1.27</td>
    </tr>
    <tr>
      <th id="T_33fc4_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_33fc4_row4_col0" class="data row4 col0" >1.54</td>
      <td id="T_33fc4_row4_col1" class="data row4 col1" >1.53</td>
      <td id="T_33fc4_row4_col2" class="data row4 col2" >1.52</td>
      <td id="T_33fc4_row4_col3" class="data row4 col3" >1.44</td>
      <td id="T_33fc4_row4_col4" class="data row4 col4" >1.46</td>
    </tr>
  </tbody>
</table>
</td></tr></table>

All the previous versions were tested on **one node**. We also tested both the row and block implementations on **two nodes** by assigning each node half the processes. For size `1024 × 1024` and `4096 × 4096` on `16` and `32` processes, this barely produced any difference. The only difference that was not negligible was for the `1024 × 1024` grid on `16` processes, which was roughly **12% slower** on 2 nodes for the row-rank method.

<table><tr><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Speed-up: 2 nodes vs 1 node, method: block</div><style type="text/css">
#T_28b78_row2_col3, #T_28b78_row4_col3 {
  background-color: darkgreen;
}
#T_28b78_row2_col4, #T_28b78_row4_col4 {
  background-color: darkred;
}
</style>
<table id="T_28b78">
  <thead>
    <tr>
      <th class="index_name level0" >Procs</th>
      <th id="T_28b78_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_28b78_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_28b78_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_28b78_level0_col3" class="col_heading level0 col3" >16</th>
      <th id="T_28b78_level0_col4" class="col_heading level0 col4" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_28b78_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_28b78_row0_col0" class="data row0 col0" ></td>
      <td id="T_28b78_row0_col1" class="data row0 col1" ></td>
      <td id="T_28b78_row0_col2" class="data row0 col2" ></td>
      <td id="T_28b78_row0_col3" class="data row0 col3" ></td>
      <td id="T_28b78_row0_col4" class="data row0 col4" ></td>
    </tr>
    <tr>
      <th id="T_28b78_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_28b78_row1_col0" class="data row1 col0" ></td>
      <td id="T_28b78_row1_col1" class="data row1 col1" ></td>
      <td id="T_28b78_row1_col2" class="data row1 col2" ></td>
      <td id="T_28b78_row1_col3" class="data row1 col3" ></td>
      <td id="T_28b78_row1_col4" class="data row1 col4" ></td>
    </tr>
    <tr>
      <th id="T_28b78_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_28b78_row2_col0" class="data row2 col0" ></td>
      <td id="T_28b78_row2_col1" class="data row2 col1" ></td>
      <td id="T_28b78_row2_col2" class="data row2 col2" ></td>
      <td id="T_28b78_row2_col3" class="data row2 col3" >1.008</td>
      <td id="T_28b78_row2_col4" class="data row2 col4" >0.982</td>
    </tr>
    <tr>
      <th id="T_28b78_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_28b78_row3_col0" class="data row3 col0" ></td>
      <td id="T_28b78_row3_col1" class="data row3 col1" ></td>
      <td id="T_28b78_row3_col2" class="data row3 col2" ></td>
      <td id="T_28b78_row3_col3" class="data row3 col3" ></td>
      <td id="T_28b78_row3_col4" class="data row3 col4" ></td>
    </tr>
    <tr>
      <th id="T_28b78_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_28b78_row4_col0" class="data row4 col0" ></td>
      <td id="T_28b78_row4_col1" class="data row4 col1" ></td>
      <td id="T_28b78_row4_col2" class="data row4 col2" ></td>
      <td id="T_28b78_row4_col3" class="data row4 col3" >1.032</td>
      <td id="T_28b78_row4_col4" class="data row4 col4" >1.024</td>
    </tr>
  </tbody>
</table>
</td><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Speed-up: 2 nodes vs 1 node, method: row</div><style type="text/css">
#T_82163_row2_col3, #T_82163_row4_col3 {
  background-color: darkred;
}
#T_82163_row2_col4, #T_82163_row4_col4 {
  background-color: darkgreen;
}
</style>
<table id="T_82163">
  <thead>
    <tr>
      <th class="index_name level0" >Procs</th>
      <th id="T_82163_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_82163_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_82163_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_82163_level0_col3" class="col_heading level0 col3" >16</th>
      <th id="T_82163_level0_col4" class="col_heading level0 col4" >32</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_82163_level0_row0" class="row_heading level0 row0" >128</th>
      <td id="T_82163_row0_col0" class="data row0 col0" ></td>
      <td id="T_82163_row0_col1" class="data row0 col1" ></td>
      <td id="T_82163_row0_col2" class="data row0 col2" ></td>
      <td id="T_82163_row0_col3" class="data row0 col3" ></td>
      <td id="T_82163_row0_col4" class="data row0 col4" ></td>
    </tr>
    <tr>
      <th id="T_82163_level0_row1" class="row_heading level0 row1" >512</th>
      <td id="T_82163_row1_col0" class="data row1 col0" ></td>
      <td id="T_82163_row1_col1" class="data row1 col1" ></td>
      <td id="T_82163_row1_col2" class="data row1 col2" ></td>
      <td id="T_82163_row1_col3" class="data row1 col3" ></td>
      <td id="T_82163_row1_col4" class="data row1 col4" ></td>
    </tr>
    <tr>
      <th id="T_82163_level0_row2" class="row_heading level0 row2" >1024</th>
      <td id="T_82163_row2_col0" class="data row2 col0" ></td>
      <td id="T_82163_row2_col1" class="data row2 col1" ></td>
      <td id="T_82163_row2_col2" class="data row2 col2" ></td>
      <td id="T_82163_row2_col3" class="data row2 col3" >0.889</td>
      <td id="T_82163_row2_col4" class="data row2 col4" >1.004</td>
    </tr>
    <tr>
      <th id="T_82163_level0_row3" class="row_heading level0 row3" >2048</th>
      <td id="T_82163_row3_col0" class="data row3 col0" ></td>
      <td id="T_82163_row3_col1" class="data row3 col1" ></td>
      <td id="T_82163_row3_col2" class="data row3 col2" ></td>
      <td id="T_82163_row3_col3" class="data row3 col3" ></td>
      <td id="T_82163_row3_col4" class="data row3 col4" ></td>
    </tr>
    <tr>
      <th id="T_82163_level0_row4" class="row_heading level0 row4" >4096</th>
      <td id="T_82163_row4_col0" class="data row4 col0" ></td>
      <td id="T_82163_row4_col1" class="data row4 col1" ></td>
      <td id="T_82163_row4_col2" class="data row4 col2" ></td>
      <td id="T_82163_row4_col3" class="data row4 col3" >0.993</td>
      <td id="T_82163_row4_col4" class="data row4 col4" >1.03</td>
    </tr>
  </tbody>
</table>
</td></tr></table>

Finally, we tried **reducing communication overhead** by taking a larger padding and updating it every `K` steps. We tested this for the two larger cases where it makes the most sense, due to the halo becoming quite large for larger `K`, and on the two best-performing rank sizes. This actually made the algorithm slower, with the slow-down increasing with the size of `K`.

<table><tr><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Mean time (s) per K for 1 node</div><style type="text/css">
#T_79ed4_row0_col0, #T_79ed4_row1_col0 {
  background-color: darkgreen;
}
#T_79ed4_row0_col3, #T_79ed4_row1_col3 {
  background-color: darkred;
}
</style>
<table id="T_79ed4">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level0" >K</th>
      <th id="T_79ed4_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_79ed4_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_79ed4_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_79ed4_level0_col3" class="col_heading level0 col3" >8</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="index_name level1" >Procs</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_79ed4_level0_row0" class="row_heading level0 row0" >2048</th>
      <th id="T_79ed4_level1_row0" class="row_heading level1 row0" >16</th>
      <td id="T_79ed4_row0_col0" class="data row0 col0" >19.49</td>
      <td id="T_79ed4_row0_col1" class="data row0 col1" >21.12</td>
      <td id="T_79ed4_row0_col2" class="data row0 col2" >24.96</td>
      <td id="T_79ed4_row0_col3" class="data row0 col3" >32.68</td>
    </tr>
    <tr>
      <th id="T_79ed4_level0_row1" class="row_heading level0 row1" >4096</th>
      <th id="T_79ed4_level1_row1" class="row_heading level1 row1" >32</th>
      <td id="T_79ed4_row1_col0" class="data row1 col0" >49.98</td>
      <td id="T_79ed4_row1_col1" class="data row1 col1" >54.86</td>
      <td id="T_79ed4_row1_col2" class="data row1 col2" >68.48</td>
      <td id="T_79ed4_row1_col3" class="data row1 col3" >108.59</td>
    </tr>
  </tbody>
</table>
</td><td style='vertical-align: top; padding: 5px;'><div style='text-align: left; font-weight: bold;'>Mean time (s) per K for 2 nodes</div><style type="text/css">
#T_c7354_row0_col0, #T_c7354_row1_col0 {
  background-color: darkgreen;
}
#T_c7354_row0_col3, #T_c7354_row1_col3 {
  background-color: darkred;
}
</style>
<table id="T_c7354">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="index_name level0" >K</th>
      <th id="T_c7354_level0_col0" class="col_heading level0 col0" >1</th>
      <th id="T_c7354_level0_col1" class="col_heading level0 col1" >2</th>
      <th id="T_c7354_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_c7354_level0_col3" class="col_heading level0 col3" >8</th>
    </tr>
    <tr>
      <th class="index_name level0" >Size</th>
      <th class="index_name level1" >Procs</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c7354_level0_row0" class="row_heading level0 row0" >2048</th>
      <th id="T_c7354_level1_row0" class="row_heading level1 row0" >16</th>
      <td id="T_c7354_row0_col0" class="data row0 col0" >21.21</td>
      <td id="T_c7354_row0_col1" class="data row0 col1" >22.56</td>
      <td id="T_c7354_row0_col2" class="data row0 col2" >25.3</td>
      <td id="T_c7354_row0_col3" class="data row0 col3" >33.04</td>
    </tr>
    <tr>
      <th id="T_c7354_level0_row1" class="row_heading level0 row1" >4096</th>
      <th id="T_c7354_level1_row1" class="row_heading level1 row1" >32</th>
      <td id="T_c7354_row1_col0" class="data row1 col0" >50.73</td>
      <td id="T_c7354_row1_col1" class="data row1 col1" >53.94</td>
      <td id="T_c7354_row1_col2" class="data row1 col2" >68.5</td>
      <td id="T_c7354_row1_col3" class="data row1 col3" >90.17</td>
    </tr>
  </tbody>
</table>
</td></tr></table>