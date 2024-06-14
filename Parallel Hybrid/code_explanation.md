### Υβριδική Παράλληλη Προσομοίωση Διάχυσης Θερμότητας Χρησιμοποιώντας MPI και OpenMP

Αυτό το έργο υλοποιεί μια υβριδική παράλληλη προσομοίωση διάχυσης θερμότητας χρησιμοποιώντας MPI (Message Passing Interface) και OpenMP (Open Multi-Processing). Παρακάτω εξηγείται πώς λειτουργεί ο κώδικας και τα κύρια χαρακτηριστικά του.

### Σταθερές και Προετοιμασία

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

Οι σταθερές και οι βιβλιοθήκες που περιλαμβάνονται ορίζονται εδώ. Χρησιμοποιούμε το `mpi.h` για τις λειτουργίες του MPI και το `omp.h` για την αξιοποίηση του OpenMP για παραλληλοποίηση σε επίπεδο νήματος.

### Αρχικοποίηση του Πλέγματος

```c
void initialize(double *grid, int rank, int chunk_size, int remainder, int size) {
    int i, j, start_row, end_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Προσαρμογή του chunk για την τελευταία διεργασία
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

Η συνάρτηση `initialize` αρχικοποιεί το πλέγμα. Το πλέγμα διαιρείται μεταξύ των διεργασιών, με κάθε διεργασία να είναι υπεύθυνη για την αρχικοποίηση ενός τμήματος. Η χρήση του OpenMP (#pragma omp parallel for) επιτρέπει την παραλληλοποίηση της αρχικοποίησης σε επίπεδο νήματος.

### Ενημέρωση του Πλέγματος

```c
void update(double *grid, double *temp, int rank, int size, int chunk_size, int remainder) {
    int i, j, start_row, end_row;
    
    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Προσαρμογή του chunk για την τελευταία διεργασία
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

Η συνάρτηση `update` υπολογίζει τη νέα κατάσταση του πλέγματος χρησιμοποιώντας τη μέθοδο πεπερασμένων διαφορών. Η χρήση του OpenMP επιτρέπει την παραλληλοποίηση του υπολογισμού σε επίπεδο νήματος.

### Εγγραφή Αποτελεσμάτων σε Αρχείο

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

Η συνάρτηση `writeToFile` γράφει την τελική κατάσταση του πλέγματος σε ένα αρχείο για ανάλυση και οπτικοποίηση.

### Κύρια Συνάρτηση

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

        // Ανταλλαγή ορίων γραμμών με γείτονες
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

        // Συγκέντρωση ενημερωμένων πλεγμάτων
        MPI_Gather(temp_grid + rank * chunk_size * GRID_SIZE, chunk_size * GRID_SIZE, MPI_DOUBLE, grid, chunk_size * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Μετάδοση του ενημερωμένου πλέγματος σε όλες τις διεργασίες
        MPI_Bcast(grid, GRID_SIZE * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
        printf("Χρόνος που χρειάστηκε: %fs\n", elapsed_time);
        writeToFile(grid, "heatmap_parallel_hybrid.txt");
    }

    free(grid);
    free(temp_grid);
    MPI_Finalize();
    return 0;
}
```

### Χαρακτηριστικά

1. **Υβριδική Παραλληλοποίηση**: Ο κώδικας χρησιμοποιεί το MPI για παραλληλοποίηση σε επίπεδο διεργασίας και το OpenMP για παραλληλοποίηση σε επίπεδο νήματος.
2. **Αρχικοποίηση και Ενημέρωση**: Οι λειτουργίες αρχικοποίησης και ενημέρωσης είναι παραλληλοποιημένες χρησιμοποιώντας το OpenMP για να επιταχυνθεί η εκτέλεση.
3. **Ανταλλαγή Συνόρων**: Οι διεργασίες ανταλλάσσουν τις γραμμές συνόρων τους με τις γειτονικές διεργασίες χρησιμοποιώντας μη-μπλοκάρισμα MPI επικοινωνία.

### Βελτιώσεις

1. **Βελτιστοποίηση Κώδικα**: Εξετάστε τη χρήση πιο εξελιγμένων τεχνικών βελτιστοποίησης για να μειώσετε το κόστος επικοινωνίας.
2. **Διαφορετικές Προσεγγίσεις Παραλληλοποίησης**: Πειραματιστείτε με άλλες προσεγγίσεις παραλληλοποίησης για να βελτιώσετε την απόδοση σε συγκεκριμένες πλατφόρμες.

###  Συμπέρασμα

Αυτή η υλοποίηση με βάση το MPI και OMP βελτιώνει σημαντικά την απόδοση της προσομοίωσης διάχυσης θερμότητας παραλληλοποιώντας τον υπολογισμό σε πολλαπλές διεργασίες και νήματα. Ωστόσο, περαιτέρω βελτιστοποιήσεις και προηγμένες τεχνικές μπορούν να εφαρμοστούν για ακόμα καλύτερη απόδοση και κλιμάκωση.
