### Παράλληλη Προσομοίωση Διάχυσης Θερμότητας Χρησιμοποιώντας MPI

Αυτό το έργο υλοποιεί μια παράλληλη προσομοίωση διάχυσης θερμότητας χρησιμοποιώντας MPI (Message Passing Interface). Παρακάτω εξηγείται πώς λειτουργεί ο κώδικας και τα κύρια χαρακτηριστικά του.

#### Σταθερές και Προετοιμασία

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

Οι σταθερές και οι βιβλιοθήκες που περιλαμβάνονται ορίζονται εδώ. Χρησιμοποιούμε το `mpi.h` για τις λειτουργίες του MPI ώστε να παραλληλοποιήσουμε την προσομοίωση.

#### Αρχικοποίηση του Πλέγματος

```c
void initialize(double *grid, int rank, int chunk_size, int remainder, int size) {
    int i, j, start_row, end_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Προσαρμογή του chunk για την τελευταία διεργασία
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
```

Η συνάρτηση `initialize` αρχικοποιεί το πλέγμα. Το πλέγμα διαιρείται μεταξύ των διεργασιών, με κάθε διεργασία να είναι υπεύθυνη για την αρχικοποίηση ενός τμήματος. Η συνθήκη `if` διασφαλίζει ότι η πηγή θερμότητας τοποθετείται σωστά στο κέντρο του πλέγματος.

#### Ενημέρωση του Πλέγματος

```c
void update(double *grid, double *temp, int rank, int size, int chunk_size, int remainder) {
    int i, j, start_row, end_row;

    if (remainder != 0 && rank == size - 1) {
        chunk_size += remainder; // Προσαρμογή του chunk για την τελευταία διεργασία
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
```

Η συνάρτηση `update` υπολογίζει τη νέα κατάσταση του πλέγματος χρησιμοποιώντας τη μέθοδο πεπερασμένων διαφορών. Οι συνθήκες ορίων διαχειρίζονται για να διασφαλιστεί η σταθερότητα και η ορθότητα.

#### Εγγραφή Αποτελεσμάτων σε Αρχείο

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

#### Κύρια Συνάρτηση

```c
int main(int argc, char *argv[]) {
    int rank, size;
    double *grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    double *temp_grid = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    struct timespec start, end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Διεργασία %d από %d αρχικοποιήθηκε.\n", rank, size);

    int chunk_size = GRID_SIZE / size;
    int remainder = GRID_SIZE % size;

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    initialize(grid, rank, chunk_size, remainder, size);
    printf("Διεργασία %d: Το πλέγμα αρχικοποιήθηκε.\n", rank);

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
        
        // Αναμετάδοση του ενημερωμένου πλέγματος σε όλες τις διεργασίες
        MPI_Bcast(grid, GRID_SIZE * GRID_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1000000000.0);
        printf("Χρόνος εκτέλεσης: %fs\n", elapsed_time);
        writeToFile(grid, "heatmap_parallel_mpi.txt");
    }

    free(grid);
    free(temp_grid);
    MPI_Finalize();
    return 0;
}
```

Αυτή είναι η κύρια συνάρτηση του προγράμματος:
- Αρχικοποιεί το MPI και το πλέγμα.
- Εκτελεί το βήμα ενημέρωσης για κάθε χρονικό βήμα, ανταλλάσσοντας γραμμές ορίων μεταξύ γειτονικών διεργασιών.
- Συγκεντρώνει και αναμεταδίδει το ενημερωμένο πλέγμα σε όλες τις διεργασίες.
- Τέλος, γράφει τα αποτελέσματα σε ένα αρχείο και εκτυπώνει το χρόνο εκτέλεσης.

#### Περαιτέρω Βελτιστοποιήσεις και Σκέψεις

1. **Εξισορρόπηση Φορτίου**: Βεβαιωθείτε ότι κάθε διεργασία λαμβάνει περίπου ίσο όγκο εργασίας για να αποφύγετε την αδράνεια και να εξασφαλίσετε μέγιστη εκμετάλλευση των πόρων.
2. **Μη-αποκλειστική Επικοινωνία**: Χρησιμοποιήστε μη-αποκλειστικές κλήσεις MPI για να επικαλύψετε τον υπολογισμό με την επικ

οινωνία για βελτιωμένη απόδοση.
3. **Αποδόμηση Περιοχών**: Εφαρμόστε πιο εξελιγμένες στρατηγικές αποδόμησης περιοχών για να βελτιστοποιήσετε τη διανομή του πλέγματος μεταξύ των διεργασιών.
4. **Προηγμένα Χαρακτηριστικά του MPI**: Αξιοποιήστε τα προηγμένα χαρακτηριστικά του MPI, όπως οι προσαρμοσμένοι τύποι δεδομένων και οι συλλογικές λειτουργίες, για καλύτερη απόδοση και κλιμάκωση.
5. **Υβριδικός Παραλληλισμός**: Συνδυάστε το MPI με άλλα μοντέλα παραλληλισμού, όπως το OpenMP, για υβριδικό παραλληλισμό, ιδιαίτερα χρήσιμο για συστοιχίες πολυπύρηνων επεξεργαστών.

### Συμπέρασμα

Αυτή η υλοποίηση με βάση το MPI βελτιώνει σημαντικά την απόδοση της προσομοίωσης διάχυσης θερμότητας παραλληλοποιώντας τον υπολογισμό σε πολλαπλές διεργασίες. Ωστόσο, περαιτέρω βελτιστοποιήσεις και προηγμένες τεχνικές μπορούν να εφαρμοστούν για ακόμα καλύτερη απόδοση και κλιμάκωση.
