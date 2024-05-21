***Heat-Diffusion-Simulation***

This repository implements a parallel simulation of heat diffusion using a hybrid MPI and OpenMP approach. The code leverages the Message Passing Interface (MPI) for communication and memory management between processes, and OpenMP for parallel processing on multi-core processors.

**Getting Started**

1. **Prerequisites:**
   - A C compiler (e.g., GCC)
   - MPI library (e.g., Open MPI) or MPI compiler (mpicc)
   - OpenMP library (typically included with most modern compilers)

2. **Clone the repository:**

   ```bash
   git clone https://github.com/theodorostaloumtzis/Heat-Diffusion-Simulation.git
   ```

**Compiling and Running the Heat Diffusion Simulation**

1. **Navigate to the project directory:**

   ```bash
   cd Heat-Diffusion-Simulation
   ```

2. **Compile the code:**

   Command for the serial and omp code:
   ```bash
   gcc -o heat_diffusion heat_diffusion.c -lm -fopenmp -Wall
   ```
   Command for the mpi and hybrid code:
   ```bash
   mpicc -o heat_diffusion heat_diffusion.c -lm -fopenmp -Wall
   ```

   - **Explanation of flags:**
     - `-o heat_diffusion`: Sets the output executable name to "heat_diffusion".
     - `heat_diffusion.c`: Specifies the source code file.
     - `-lm`: Links the standard math library (required for mathematical functions).
     - `-fopenmp`: Enables OpenMP support for parallel execution.
     - `-Wall`: Enables compiler warnings (optional, but recommended for code quality).

3. **Run the simulation:**

   ```bash
   mpirun -np <num_processes> ./heat_diffusion
   ```

   - **Explanation:**
     - `mpirun`: The MPI launcher command.
     - `-np <num_processes>`: Specifies the desired number of MPI processes to use for parallel execution. Replace `<num_processes>` with the desired number.
     - `./heat_diffusion`: The compiled executable file.

**Additional Notes:**

* Ensure you have MPI libraries installed and configured on your system for `mpirun` to function correctly.
* Adjust the compilation flags (`-lm`, `-fopenmp`, `-Wall`) if necessary based on your system and compiler.
* This approach assumes all necessary header files (e.g., `mpi.h`, `omp.h`) are included in the `heat_diffusion.c` file or are located in standard system directories.

By following these steps, you can compile and run the heat diffusion simulation code without a Makefile. Remember to replace `<num_processes>` with the desired number of MPI processes to leverage parallel processing capabilities.

**Understanding the Code**

The core functionality resides in the `heat_diffusion.c` file. Here's a breakdown of the key steps:

* **Initialization:**
   - MPI environment is initialized.
   - Grid dimensions (size) and timesteps for the simulation are defined.
   - Memory is allocated for the temperature grid.
   - The grid is initialized with appropriate temperature values (optional).

* **Parallelization:**
   - The grid is divided into chunks, and each MPI process is assigned a chunk.
   - OpenMP threads are used to parallelize calculations within each process's chunk.

* **Heat Diffusion Simulation:**
   - The main simulation loop iterates over timesteps.
   - Within each timestep:
     - The `update` function (not explicitly shown) calculates new temperature values for each grid point based on its neighbors and the heat diffusion equation.
     - MPI processes exchange boundary data with their neighbors to ensure accurate calculations at the edges of each process's chunk.
     - `MPI_Gather` is used to collect temperature data from all processes onto the root process (usually process 0).
     - The root process broadcasts the complete, updated temperature grid back to all processes.

* **Finalization:**
   - MPI environment is finalized.
   - Memory allocated for the grid is freed.

**Author**

Theodoros Taloumtzis ([https://github.com/theodorostaloumtzis](https://github.com/theodorostaloumtzis))

