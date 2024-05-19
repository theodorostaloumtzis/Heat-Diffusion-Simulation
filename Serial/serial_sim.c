#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define GRID_SIZE 100
#define TIMESTEPS 100000
#define DT 0.1
#define DX 1.0
#define ALPHA 0.01

// Initialize the grid
void initialize(double grid[GRID_SIZE][GRID_SIZE]) {
    int i, j;
    for (i = 0; i < GRID_SIZE; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            if (i >= GRID_SIZE / 2 - 24 && i < GRID_SIZE / 2 + 24 && j >= GRID_SIZE / 2 - 24 && j < GRID_SIZE / 2 + 24) {
                grid[i][j] = 100.0;
            } else {
                grid[i][j] = 0.0;
            }
        }
    }
}

// Update the grid
void update(double grid[GRID_SIZE][GRID_SIZE]) {
    double temp[GRID_SIZE][GRID_SIZE];
    int i, j;
    for (i = 0; i < GRID_SIZE; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
           if (i == 0 || i == GRID_SIZE - 1 || j == 0 || j == GRID_SIZE - 1) {
                temp[i][j] = grid[i][j];
            } else {
                temp[i][j] = grid[i][j] + ALPHA * DT / (DX * DX) * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1] - 4 * grid[i][j]);
            }
        }
    }

    for (i = 0; i < GRID_SIZE; i++) {
        for (j = 0; j < GRID_SIZE; j++) {
            grid[i][j] = temp[i][j];
        }
    }
}

// Write the grid to a file
void writeToFile(double grid[GRID_SIZE][GRID_SIZE]) {
    FILE *file = fopen("heatmapserial.txt", "w");
    // Write the grid to the file
    int i,j;
    for (i = 0; i < GRID_SIZE; i++) {
        // Write the temperature of each cell to the file
        for (j = 0; j < GRID_SIZE; j++) {
            fprintf(file, "%f ", grid[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Main function
int main() {
    double grid[GRID_SIZE][GRID_SIZE];
    clock_t start, end;

    start = clock();
    // Initialize the grid
    initialize(grid);
    // Update the grid for a number of timesteps
    int t;
    for (t = 0; t < TIMESTEPS; t++) {
        update(grid);
    }
    end = clock();
    printf("Time taken: %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    writeToFile(grid);

    return 0;
}