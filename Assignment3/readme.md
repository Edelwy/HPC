# **Molecular dynamics**: *2D Lennard-Jones Simulation* 

Our scripts and modified source code can be found in our [GitHub repository](https://github.com/Edelwy/HPC). The point is to design a CUDA implementation of the molecular dynamic simulation.  [Instructions](https://github.com/laspp/HPC/blob/2472513da1348693fd3adbfefac1a6728f4fba4e/labs/04-Assignment3-CUDA/Assignment3.md) are available in this repository as well.

## Project structure and running

```sh
Assignment3/
├── Assignment3.md          # The assignment instructions.
├── readme.md               # The detailed report on code, running and structure.
├── latex/                  # The report which documents results, markdown summery.
├── results/                # CSV's from benchmark scripts.
└── src/
    ├── main.c                    # The main program file, entry point.
    ├── lennard_jones_common.h    # Common to all implementations.
    ├── lennard_jones_common.c    
    ├── gifenc.c / gifenc.h       # GIF library.
    ├── helper_cuda.h             # CUDA error macros (NVIDIA SDK).
    ├── Makefile                  # METHOD: seq, omp, gpu, cells, opt, gpu2.
    │
    ├── lennard_jones_seq.c       # improved sequential with Newton's 3rd law.
    ├── lennard_jones_omp.c       # OpenMP and Newton via array reduction.
    ├── lennard_jones_gpu.cu      # base CUDA does full N².
    ├── lennard_jones_cells.cu    # CUDA with GPU cell lists, biggest optimization.
    ├── lennard_jones_opt.cu      # CUDA with memory and GPU usage optimizations.
    ├── lennard_jones_gpu2.cu     # CUDA with two GPUs.
    │
    └── benchmarks/
        ├── bench.sh              # One config.
        └── submit_all.sh         # Dependency chain of all configs.
```

We have different methods for building different implementations. First we have the sequential version which has mainly been optimized via Newton's 3rd law. Then the OpenMP version and **four** different CUDA versions.

```
make METHOD=seq   : gcc   + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_seq.c
make METHOD=omp   : gcc   + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_omp.c
make METHOD=gpu   : nvcc  + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_gpu.cu
make METHOD=cells : nvcc  + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_cells.cu
make METHOD=opt   : nvcc  + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_opt.cu
make METHOD=gpu2  : nvcc  + main.c + lennard_jones_common.c + gifenc.c + lennard_jones_gpu2.cu
```

Different flags support different featrues: 
`[N] [--steps S] [--energy] [--gif PATH] [--final PATH]`

```sh
  N              # Number of particles.                       Default: 1000
  --steps S      # Simulation steps.                          Default: 5000
  --energy       # Print KE / PE / E every step.              Default: OFF
  --gif PATH     # Write animation GIF.                       Default: OFF           
  --final PATH   # Write final positions after timer stops.   Default: OFF

# Fixed: density 0.95, temperature 0.5, seed 42.
```

The `lennard_jones_common.h` contains all the functions used throughout the implementations. Most of the functions are the same as in the [original sequential version](https://github.com/laspp/HPC/blob/2472513da1348693fd3adbfefac1a6728f4fba4e/labs/04-Assignment3-CUDA/src/lennard-jones/src/lennard-jones.cu). Added was the standalone `compute_pe` function which was inside `compute_forces` before. It loops over the Netwon half-matrix and returns the total potential energy without updating forces.

Here are a few examples on how to run:
```bash
  make && ./lj.out
  make && ./lj.out 4000 --steps 5000
  make && ./lj.out 1000 --gif simulation.gif
  make && ./lj.out 1000 --final final_state.txt
  make && ./lj.out 1000 --steps 1000 --energy
  make METHOD=omp && OMP_NUM_THREADS=8 ./lj.out 2000    
  make METHOD=gpu BLOCKSIZE=256 && ./lj.out 4000
  ```

## Implementations

### Sequential implementation

We replaced the full loop over all ordered pairs with a loop over unordered pairs, since we can count on Newton's third law. This changes what is returned in the original:

```c
double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
pe += 0.5 * vij;
```
Versus the new version:
```c
pe += 4.0 * EPSILON * (sr12 - sr6) - v_shift;
```

We also replaced `r` with its squared value and precomputed some things. Here is the original:

```c
double r = sqrt(dx*dx + dy*dy);
double sr = SIGMA / r;
double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
double fx = fij * dx / r;
double fy = fij * dy / r;
```

Versus the new version, where we avoid `sqrt` and `pow`:
```c
double r2 = dx * dx + dy * dy;
double sr2 = (SIGMA * SIGMA) / r2;
double sr6 = sr2 * sr2 * sr2;
double sr12 = sr6 * sr6;
double fmag = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2; 
double fxij = fmag * dx; 
double fyij = fmag * dy;
```
### OpenMP implementation

We wanted to keep the same optimization as the sequential version but that means changing how forces are stored and how the loops are parallelized. We used to separate **flat arrays** for the forces, since array reduction only works on array sections not struct members.

```c
double *fx = (double *)malloc(n * sizeof(double));
double *fy = (double *)malloc(n * sizeof(double));
```

An **array reduction** is an OpenMP feature where each thread gets a private copy of a whole array or its slice and updates that copy independantly. Afterwards OpenMP adds all copies together as a point-wise sum. We do this to whole `fx` and `fy` arrays therefore from $0$ to $n$. Since `pe` is not an array this is just normal accumulation over the loop. Scheduling is used because the outer loop is uneven, early $i$'s are long, while last ones are short. The `dynamic` is used so that when a thread finishes its chunk, it asks for the next free chunk, the chunks are not fixed upfront. Chunks are of size $64$ which was an empirical choice.

```c
#pragma omp parallel for reduction(+:pe) reduction(+:fx[:n]) reduction(+:fy[:n]) schedule(dynamic, 64)
```

We also parallelized the `leapfrog_step` function with:
```c
#pragma omp parallel for
```

### Basic CUDA implementation

In the basic implementation we reverted back to the full ordered pair loop, therefore we have loop of size $n^2$. Data lives on the device and potential energy is not computed on GPU, it only writes forces. Potential energy is computed on host using `compute_pe` after the copy-back. We experimented with different `BLOCKSIZE` variables, which decide the number of threads per block. Tested were $32$, $64$, $128$, $256$ and $512$. Below is the code that initializes it with `checkCudaErrors` excluded for readability.

```c
Particle *d_p;                                                           // Device pointer.
cudaMalloc(&d_p, n * sizeof(Particle));                                  // Allocate memory on device.
cudaMemcpy(d_p, particles, n * sizeof(Particle),cudaMemcpyHostToDevice); // Copy from host to device.

unsigned int block = BLOCKSIZE;              
unsigned int grid  = (n + block - 1) / block; // Blocks launched: enough to cover N particles.     
```

The functions `leapfrog_new_kernel`, `leapfrog_current_kernel` and `forces_kernel` are global functions which run on the device *(GPU)* and are launched from the host *(CPU)*. First we launch `forces_kernel` which calculates the force per particle:

```c
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
```

In this naive version every thread *(particle)* loops through all other particles same as the sequential version. Then we have the leapfrog steps on GPU which use `leapfrog_current_kernel` followed by `forces_kernel` and `leapfrog_new_kernel`. So in the OpenMP version we have:

```c
static double leapfrog_step(Particle *particles, double *fx, double *fy,
                            unsigned int n, double box_size, double v_shift) {
    #pragma omp parallel for // Current leapfrog kernel.
    for (unsigned int i = 0; i < n; ++i) {
        Particle *p = &particles[i];
        p->vx += 0.5 * DT * fx[i];
        p->vy += 0.5 * DT * fy[i];
        p->x += DT * p->vx;
        p->y += DT * p->vy;
    }
    wrap_positions(particles, n, box_size); // Still in current kernel.
    double pe = compute_forces(particles, fx, fy, n, box_size, v_shift);
    
    #pragma omp parallel for // New leapfrog kernel.
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].vx += 0.5 * DT * fx[i];
        particles[i].vy += 0.5 * DT * fy[i];
    }
    return pe;
}
```

The functions in kernel are basically the same with the wrapping explicitly written as it happens on GPU. Finally we copy the result back to host. The entire result can always be read from particles array.

```c
cudaMemcpy(particles, d_p, n * sizeof(Particle), cudaMemcpyDeviceToHost); // Copy from device to host.
cudaFree(d_p);                                                            // Free device pointer.
```

### Neighbourhood CUDA implementation

The majority of functions remained the same including the leapfrog step functions that do not calculate the forces. Since forces are limited by the cut-off which se checked in the sequential version as:

```c
const double rc2 = (R_CUT * SIGMA) * (R_CUT * SIGMA);
double r2 = dx * dx + dy * dy;
if (r2 >= rc2) continue;
```

we can only look at the particles that matter which are at `R_CUT * SIGMA`. We named this small square tile of the simulation box which matters a **cell**. The entire box is split into square cells of side at least af cut-off size. Since every particle has the particles that matter inside the disk of radious cut-off around the particle, we only need to look at the neighbouring $8$ cells to find all of them. The optimal solution would be cells of size `R_CUT * SIGMA`, however the box size might not allow this completly:

```c
int ncell = (int)(box_size / (R_CUT * SIGMA)); // How many cells fit on the edge of the box.
double cell_size = box_size / (double)ncell;   // Cell size which is larger than cut-off by design.
int ncell2 = ncell * ncell;                    // The number of cells.
```

Then we have `d_count` which counts the number of particles in a cell, and `d_list` which has at most `ncell2 * MAX_PER_CELL` number of indices, so this is used as the particle index.

```c
cudaMalloc(&d_p, n * sizeof(Particle));                                     // GPU particle array.
cudaMalloc(&d_count, ncell2 * sizeof(int));                                 // One counter per cell.
cudaMalloc(&d_list, (size_t)ncell2 * MAX_PER_CELL * sizeof(int));           // Cell particle indices.
cudaMemcpy(d_p, particles, n * sizeof(Particle), cudaMemcpyHostToDevice);   // Copy host to device.
```

Then instead of one `forces_kernel` we fist compute `bin_kernel` which bins together the particles in cells. We calculate the column and row position and put them into the appropriate cell. So after this call we get the `d_count` list filled by the number of particles for each cell and the `d_list` has all particle indices of the particles in that cell. Then we call the `forces_cells_kernel` which is similar to before but only on the neighbouring cells. This gives us the cell of the particle $i$:

```c
int cx = (int)(xi / cell_size);
int cy = (int)(yi / cell_size);
if (cx >= ncell) cx = ncell - 1;
if (cy >= ncell) cy = ncell - 1;
```

Then we only check the neighbours:
```c
for (int dcy = -1; dcy <= 1; ++dcy) {
    int ncy = (cy + dcy + ncell) % ncell;
    for (int dcx = -1; dcx <= 1; ++dcx) {
        int ncx = (cx + dcx + ncell) % ncell;
            // Code...
```

### Memory optimizations CUDA implementation

We exchanged the array of structs *(AoS)* for separate arrays *(SoA)* for each field. We did this via `soa_to_aos` function so the signature is still the same.

```c
Particle p[n];                                 // Before
double x[n], y[n], vx[n], vy[n], fx[n], fy[n]; // After
```

## Results

Can be found in the [latex report](https://github.com/Edelwy/HPC/blob/3a6c459ccaec52d97bcd7c4595b12c051d74e0f5/Assignment3/latex/main.tex) or the *Jupyter notebook* [analysis](https://github.com/Edelwy/HPC/blob/3a6c459ccaec52d97bcd7c4595b12c051d74e0f5/Assignment3/results/analiza.ipynb).