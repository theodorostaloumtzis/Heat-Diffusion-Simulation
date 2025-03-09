## Heat Diffusion Simulation Using Only MPI

This project implements a heat diffusion simulation using only MPI (Message Passing Interface). Below is an explanation of the code and its main features.

#### Constants and Setup

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

#define GRID_SIZE 100
#define TIMESTEPS 100000
#define DT 0.1
#define DX 1.0
#define ALPHA 0.01
```

Here, the constants and the included libraries are defined. We use `mpi.h` to parallelize the simulation with MPI functions.

#### Grid Initialization

```c
// Initialize the grid
void initialize(double *grid, int rank, int chunk_size, int remainder, int size) {
    int i, j, start_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Adjust chunk size for the last process
    }

    start_row = rank * chunk_size;

    for (i = 0; i < chunk_size; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            int global_i = start_row + i;
            if (global_i >= GRID_SIZE / 2 - 24 && global_i < GRID_SIZE / 2 + 24 &&
                j >= GRID_SIZE / 2 - 24 && j < GRID_SIZE / 2 + 24) {
                grid[(i + 1) * GRID_SIZE + j] = 100.0; // i+1 to account for the extra row at the beginning
            } else {
                grid[(i + 1) * GRID_SIZE + j] = 0.0;   // i+1 to account for the extra row at the beginning
            }
        }
    }
}
```

The `initialize` function initializes the grid. The grid is divided among processes, and each process is responsible for initializing its own portion. The `if` statement ensures that the heat source is correctly placed at the center of the grid.

#### Updating the Grid

```c
// Update the grid
void update(double *grid, double *temp, int chunk_size) {
    int i, j;

    for (i = 1; i <= chunk_size; i++) { // Update from 1 to chunk_size to avoid halo rows
        for (j = 1; j < GRID_SIZE - 1; j++) {
            double up = grid[(i-1) * GRID_SIZE + j];
            double down = grid[(i+1) * GRID_SIZE + j];
            double left = grid[i * GRID_SIZE + j-1];
            double right = grid[i * GRID_SIZE + j+1];
            temp[i * GRID_SIZE + j] =
                grid[i * GRID_SIZE + j] +
                ALPHA * DT / (DX * DX) * (up + down + left + right - 4 * grid[i * GRID_SIZE + j]);
        }
    }
}
```

The `update` function computes the new state of the grid using the finite difference method. Boundary conditions are handled in such a way as to ensure stability and correctness.

#### Writing Results to a File

```c
// Function to write the results to a file
void writeToFile(double *grid, char *filename) {
    FILE *file = fopen(filename, "w");
    int i, j;
    for (i = 0; i < GRID_SIZE; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            fprintf(file, "%f ", grid[i * GRID_SIZE + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
```

The `writeToFile` function writes the final state of the grid to a file for analysis and visualization.

#### Main Function

```c
int main(int argc, char *argv[]) {
    int rank, size;
    int chunk_size, remainder;
    double *grid, *temp_grid;
    struct timespec start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = GRID_SIZE / size;
    remainder = GRID_SIZE % size;

    // Allocate memory for each process's chunk, plus two extra rows for halo exchange
    grid = (double *)malloc((chunk_size + 2) * GRID_SIZE * sizeof(double));
    temp_grid = (double *)malloc((chunk_size + 2) * GRID_SIZE * sizeof(double));

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    // Initialize the grid
    initialize(grid, rank, chunk_size, remainder, size);

    MPI_Request send_req[2], recv_req[2];
    int t;
    for (t = 0; t < TIMESTEPS; t++) {
        // Non-blocking sends and receives for halo rows
        if (rank > 0) {
            MPI_Irecv(grid, GRID_SIZE, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_req[0]);
            MPI_Isend(grid + GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_req[0]);
        }
        if (rank < size - 1) {
            MPI_Irecv(grid + (chunk_size + 1) * GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_req[1]);
            MPI_Isend(grid + chunk_size * GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_req[1]);
        }

        // Wait for halo exchanges to complete
        if (rank > 0) {
            MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&send_req[0], MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);
            MPI_Wait(&send_req[1], MPI_STATUS_IGNORE);
        }

        // Update interior cells
        update(grid, temp_grid, chunk_size);

        // Swap grids
        double *swap = grid;
        grid = temp_grid;
        temp_grid = swap;
    }

    // Gather results at rank 0
    double *full_grid = NULL;
    if (rank == 0) {
        full_grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    }

    MPI_Gather(
        grid + GRID_SIZE,           // Send buffer
        chunk_size * GRID_SIZE,     // Number of elements to send
        MPI_DOUBLE,                 // Data type
        full_grid,                  // Receive buffer
        chunk_size * GRID_SIZE,     // Number of elements each process sends
        MPI_DOUBLE,                 // Data type
        0,                          // Root process
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) +
                             ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
        printf("Time taken: %fs\n", elapsed_time);
        writeToFile(full_grid, "heatmap_parallel_mpi.txt");
        free(full_grid);
    }

    free(grid);
    free(temp_grid);
    MPI_Finalize();
    return 0;
}
```

### Explanation

1. **Setup and Initialization:**
   - The constants defining the grid size, the number of time steps, the time interval, and the diffusion parameter are declared at the start.
   - The MPI processes are initialized with `MPI_Init`, and their roles are determined via `MPI_Comm_rank` and `MPI_Comm_size`.
   - The grid is divided into chunks, and each process is responsible for initializing its own chunk using the `initialize` function.

2. **Updating the Grid:**
   - The `update` function calculates the new state of the grid using the heat diffusion equation.
   - Each process computes its portion of the grid in parallel.

3. **Communication Between Processes:**
   - The processes communicate with each other using non-blocking sends and receives (`MPI_Irecv`, `MPI_Isend`) to exchange boundary (halo) rows.

4. **Collecting and Saving Results:**
   - The grid values from all processes are gathered at process 0 using `MPI_Gather`.
   - Process 0 stores the results in a file and prints out the total execution time of the simulation.

5. **Memory Allocation:**
   - Memory for the grids is dynamically allocated using `malloc` and freed at the end of the program.

In this way, the heat diffusion simulation is parallelized using the MPI library, leveraging the parallel processing capabilities of modern computing systems.
