## Hybrid Parallel Heat Diffusion Simulation Using MPI and OpenMP

This project implements a hybrid parallel heat diffusion simulation using MPI (Message Passing Interface) and OpenMP (Open Multi-Processing). Below is an explanation of how the code works and its main features.

### Constants and Setup

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define GRID_SIZE 100
#define TIMESTEPS 100000
#define DT 0.1
#define DX 1.0
#define ALPHA 0.01
```

Here, the constants and included libraries are defined. We use `mpi.h` for MPI functions and `omp.h` to leverage OpenMP for thread-level parallelism.

### Grid Initialization

```c
void initialize(double *grid, int rank, int chunk_size, int remainder, int size) {
    int i, j, start_row, end_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Adjust the chunk for the last process
    }

    start_row = rank * chunk_size;
    end_row = start_row + chunk_size;

    #pragma omp parallel for private(i, j)
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            if (i >= GRID_SIZE / 2 - 24 && i < GRID_SIZE / 2 + 24 && j >= GRID_SIZE / 2 - 24 && j < GRID_SIZE / 2 + 24) {
                grid[i * GRID_SIZE + j] = 100.0;
            } else {
                grid[i * GRID_SIZE + j] = 0.0;
            }
        }
    }
}
```

The `initialize` function initializes the grid. The grid is divided among processes, with each process responsible for initializing a portion of it. The use of OpenMP (`#pragma omp parallel for`) enables thread-level parallelization for the initialization.

### Updating the Grid

```c
void update(double *grid, double *temp, int rank, int size, int chunk_size, int remainder) {
    int i, j, start_row, end_row;
    
    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Adjust the chunk for the last process
    }

    start_row = rank * chunk_size;
    end_row = start_row + chunk_size;
    
    #pragma omp parallel for private(i, j)
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            double up = (i > 0) ? grid[(i-1) * GRID_SIZE + j] : 0.0;
            double down = (i < GRID_SIZE-1) ? grid[(i+1) * GRID_SIZE + j] : 0.0;
            double left = (j > 0) ? grid[i * GRID_SIZE + j-1] : 0.0;
            double right = (j < GRID_SIZE-1) ? grid[i * GRID_SIZE + j+1] : 0.0;
            temp[i * GRID_SIZE + j] = grid[i * GRID_SIZE + j] + ALPHA * DT / (DX * DX) * (up + down + left + right - 4 * grid[i * GRID_SIZE + j]);
        }
    }
}
```

The `update` function computes the new state of the grid using the finite difference method. OpenMP is used to allow thread-level parallelism of the computation.

### Writing Results to a File

```c
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

### Main Function

```c
int main(int argc, char *argv[]) {
    int rank, size;
    double *grid, *temp_grid;
    struct timespec start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = GRID_SIZE / size;
    int remainder = GRID_SIZE % size;

    // Allocate memory for each process's chunk
    grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    temp_grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    initialize(grid, rank, chunk_size, remainder, size);

    MPI_Request send_req, recv_req;
    int t;
    for (t = 0; t < TIMESTEPS; t++) {
        update(grid, temp_grid, rank, size, chunk_size, remainder);

        // Exchange boundary rows with neighbors
        if (rank > 0) {
            MPI_Isend(temp_grid + rank * chunk_size * GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(temp_grid + (rank-1) * chunk_size * GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        }
        if (rank < size-1) {
            MPI_Isend(temp_grid + (rank+1) * chunk_size * GRID_SIZE - GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(temp_grid + (rank+1) * chunk_size * GRID_SIZE, GRID_SIZE, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        }

        // Gather updated grids
        MPI_Gather(temp_grid + rank * chunk_size * GRID_SIZE, chunk_size * GRID_SIZE, MPI_DOUBLE,
                   grid, chunk_size * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Broadcast the updated grid to all processes
        MPI_Bcast(grid, GRID_SIZE * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
        printf("Time required: %fs\n", elapsed_time);
        writeToFile(grid, "heatmap_parallel_hybrid.txt");
    }

    free(grid);
    free(temp_grid);
    MPI_Finalize();
    return 0;
}
```

### Features

1. **Hybrid Parallelization**: The code uses MPI for process-level parallelization and OpenMP for thread-level parallelization.
2. **Initialization and Update**: The initialization and update functions are parallelized with OpenMP to speed up execution.
3. **Boundary Exchange**: Processes exchange their boundary rows with neighboring processes using non-blocking MPI communication.

### Improvements

1. **Code Optimization**: Consider using more advanced optimization techniques to reduce communication overhead.
2. **Alternative Parallelization Approaches**: Experiment with other parallelization strategies to improve performance on specific platforms.

### Conclusion

This MPI and OpenMP-based implementation significantly improves the performance of the heat diffusion simulation by parallelizing the computation across multiple processes and threads. However, further optimizations and advanced techniques can be applied for even better performance and scalability.
