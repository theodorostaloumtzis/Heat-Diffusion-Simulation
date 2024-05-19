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

// Initialize the grid
void initialize(double *grid, int rank, int chunk_size, int remainder, int size) {
    int i, j, start_row, end_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Adjust chunk size for the last process
    }

    start_row = rank * chunk_size;
    end_row = start_row + chunk_size;

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

// Update the grid
void update(double *grid, double *temp, int rank, int size, int chunk_size, int remainder) {
    int i, j, start_row, end_row;
    
    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Adjust chunk size for the last process
    }

    start_row = rank * chunk_size;
    end_row = start_row + chunk_size;

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

int main(int argc, char *argv[]) {
    int rank, size;
    double *grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    double *temp_grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    struct timespec start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Process %d of %d initialized.\n", rank, size);

    int chunk_size = GRID_SIZE / size;
    int remainder = GRID_SIZE % size;

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    initialize(grid, rank, chunk_size, remainder, size);
    printf("Process %d: Grid initialized.\n", rank);

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
        MPI_Gather(temp_grid + rank * chunk_size * GRID_SIZE, chunk_size * GRID_SIZE, MPI_DOUBLE, grid, chunk_size * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Broadcast the updated grid to all processes
        MPI_Bcast(grid, GRID_SIZE * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
        printf("Time taken: %fs\n", elapsed_time);
        writeToFile(grid, "heathmap_parallel_mpi.txt");
    }

    free(grid);
    free(temp_grid);
    MPI_Finalize();
    return 0;
}