# Heat-Diffusion-Simulation (Προσομοίωση διάχυσης θερμότητας)

Αυτό το αποθετήριο περιλαμβάνει μια προσομοίωση διάχυσης θερμότητας χρησιμοποιώντας δύο διαφορετικές προσεγγίσεις: σειριακή και παράλληλη με OpenMP. Ο κώδικας εκτελείται σε επεξεργαστές πολλαπλών πυρήνων για να επιτύχει ταχύτερους υπολογισμούς.

## Περιγραφή του Προβλήματος

Η διάχυση θερμότητας είναι μια φυσική διεργασία που περιγράφει τη μεταφορά θερμικής ενέργειας μέσω αγωγής. Η εξίσωση διάχυσης θερμότητας περιγράφει αυτή τη διαδικασία μαθηματικά:

\[ \frac{\partial u}{\partial t} = \alpha \nabla^2 u \]

όπου:
- \( u \) είναι η θερμοκρασία,
- \( t \) είναι ο χρόνος,
- \( \alpha \) είναι ο συντελεστής θερμικής διάχυσης,
- \( \nabla^2 \) είναι ο τελεστής Λαπλασιανού.

Αυτή η προσομοίωση χρησιμοποιεί τη μέθοδο πεπερασμένων διαφορών για την επίλυση αυτής της εξίσωσης σε ένα 2D πλέγμα.

## Έναρξη

1. **Προαπαιτήσεις:**
    - Ένας μεταγλωττιστής C (π.χ., GCC)
    - Βιβλιοθήκη OpenMP (συνήθως συμπεριλαμβάνεται στους περισσότερους σύγχρονους μεταγλωττιστές)

2. **Κλωνοποίηση του αποθετηρίου:**

    ```bash
    git clone https://github.com/theodorostaloumtzis/Heat-Diffusion-Simulation.git
    ```

## Μεταγλώττιση και Εκτέλεση της Προσομοίωσης Διάχυσης Θερμότητας

1. **Μετακίνηση στον κατάλογο του έργου:**

    ```bash
    cd Heat-Diffusion-Simulation
    ```

2. **Μεταγλώττιση του κώδικα:**

    Για τον σειριακό κώδικα:
    ```bash
    gcc -o heat_diffusion_serial heat_diffusion_serial.c -lm -Wall
    ```

    Για τον κώδικα με OpenMP:
    ```bash
    gcc -o heat_diffusion_omp heat_diffusion_omp.c -lm -fopenmp -Wall
    ```

    - **Επεξήγηση σημαιών:**
        - `-o heat_diffusion_serial` / `-o heat_diffusion_omp`: Ορίζει το όνομα του εκτελέσιμου αρχείου εξόδου.
        - `heat_diffusion_serial.c` / `heat_diffusion_omp.c`: Καθορίζει το αρχείο του πηγαίου κώδικα.
        - `-lm`: Συνδέει τη βασική βιβλιοθήκη μαθηματικών.
        - `-fopenmp`: Ενεργοποιεί την υποστήριξη OpenMP για παράλληλη εκτέλεση (μόνο για τον κώδικα OpenMP).
        - `-Wall`: Ενεργοποιεί τις προειδοποιήσεις του μεταγλωττιστή (προαιρετικό, αλλά συνιστάται).

3. **Εκτέλεση της προσομοίωσης:**

    Για τον σειριακό κώδικα:
    ```bash
    ./heat_diffusion_serial
    ```

    Για τον κώδικα με OpenMP:
    ```bash
    ./heat_diffusion_omp
    ```

## Κατανόηση του Κώδικα

Η βασική λειτουργικότητα βρίσκεται στα αρχεία `heat_diffusion_serial.c` και `heat_diffusion_omp.c`. Ακολουθεί μια ανάλυση των βασικών βημάτων:

* **Αρχικοποίηση:**
    - Ορίζονται οι διαστάσεις του πλέγματος και τα χρονικά βήματα για την προσομοίωση.
    - Διατίθεται μνήμη για το πλέγμα θερμοκρασίας.
    - Το πλέγμα αρχικοποιείται με κατάλληλες τιμές θερμοκρασίας.

* **Παράλληλοποίηση (για τον κώδικα OpenMP):**
    - Χρήση των οδηγιών OpenMP για τον παράλληλο υπολογισμό μέσα σε κάθε τμήμα του πλέγματος.

* **Προσομοίωση Διάχυσης Θερμότητας:**
    - Ο κύκλος του κύριου προγράμματος επαναλαμβάνεται για τα χρονικά βήματα.
    - Μέσα σε κάθε χρονικό βήμα:
        - Η συνάρτηση `update` υπολογίζει νέες τιμές θερμοκρασίας για κάθε σημείο του πλέγματος με βάση τους γείτονές του και την εξίσωση διάχυσης θερμότητας.

* **Ολοκλήρωση:**
    - Απελευθερώνεται η μνήμη που έχει διατεθεί για το πλέγμα.

## Οδηγίες Δοκιμών και Αποτελεσμάτων

* Μετά την εκτέλεση της προσομοίωσης, τα αποτελέσματα θα αποθηκευτούν σε ένα αρχείο `heatmap.txt`.
* Μπορείτε να χρησιμοποιήσετε ένα εργαλείο οπτικοποίησης (π.χ., Gnuplot, Matplotlib) για να δημιουργήσετε γραφήματα από τα δεδομένα και να δείτε την κατανομή της θερμότητας.

## Αντιμετώπιση Προβλημάτων

* **Σφάλματα Μεταγλώττισης:**
    - Βεβαιωθείτε ότι οι βιβλιοθήκες και τα αρχεία κεφαλίδας είναι σωστά εγκατεστημένα και προσβάσιμα από τον μεταγλωττιστή σας.
* **Εκτέλεση:**
    - Ελέγξτε ότι έχετε τα απαραίτητα δικαιώματα εκτέλεσης για το εκτελέσιμο αρχείο (`chmod +x heat_diffusion_serial` / `chmod +x heat_diffusion_omp`).

## Συγγραφέας

Theodoros Taloumtzis ([https://github.com/theodorostaloumtzis](https://github.com/theodorostaloumtzis))
