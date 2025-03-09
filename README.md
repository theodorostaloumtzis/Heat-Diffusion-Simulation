# Heat-Diffusion-Simulation

This repository includes a heat diffusion simulation using two different approaches: a serial version and a parallel version with OpenMP and MPI. The code runs on multi-core processors and distributed environments for faster computations.

## Problem Description

Heat diffusion is a physical process that describes the transfer of thermal energy through conduction. The heat diffusion equation mathematically describes this process:

## The Heat Diffusion Equation: Overview

The heat diffusion equation describes how temperature in a material changes over time.

**Mathematical Form:**

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

* $u(x, t)$: temperature at point $x$ and time $t$
* $t$: time
* $\alpha$: thermal diffusivity (how easily heat diffuses)
* $\nabla^2 u$: Laplacian operator (the second spatial derivative)

This simulation uses the finite difference method to solve this equation on a 2D grid.

## Simulation Result

<img src="Serial/heatmap_serial.png" alt="heatmap">

## Getting Started

1. **Prerequisites:**
    - A C compiler (e.g., GCC, MPICC)
    - OpenMP library (usually included in most modern compilers)
    - MPI (Message Passing Interface) library

2. **Clone the repository:**

    ```bash
    git clone https://github.com/theodorostaloumtzis/Heat-Diffusion-Simulation.git
    ```

## Compiling and Running the Heat Diffusion Simulation

1. **Navigate to the project directory:**

    ```bash
    cd Heat-Diffusion-Simulation
    ```

2. **Compile the code:**

    For the serial code:
    ```bash
    gcc -o serial_sim serial_sim.c -lm -Wall
    ```

    For the OpenMP code:
    ```bash
    gcc -o parallel_sim_omp parallel_sim_omp.c -lm -fopenmp -Wall
    ```

    For the MPI code:
    ```bash
    mpicc -o parallel_sim_mpi parallel_sim_mpi.c -lm -lmpi -Wall
    ```

    For the hybrid code (MPI + OpenMP):
    ```bash
    mpicc -o parallel_sim_hybrid parallel_sim_hybrid.c -lm -fopenmp -lmpi -Wall
    ```

    - **Explanation of flags:**
        - `-o serial_sim`, `-o parallel_sim_omp`, etc.: Specifies the name of the output executable.
        - `serial_sim.c`, `parallel_sim_omp.c`, etc.: Specifies the source file.
        - `-lm`: Links the standard math library.
        - `-fopenmp`: Enables OpenMP support for parallel execution (only for the OpenMP code).
        - `-lmpi`: Links the MPI library for parallel execution (only for the MPI code).
        - `-Wall`: Enables compiler warnings (optional, but recommended).

3. **Run the simulation:**

    For the serial code:
    ```bash
    ./serial_sim
    ```

    For the OpenMP code:
    ```bash
    ./parallel_sim_omp
    ```

    For the MPI code:
    ```bash
    mpirun -n <number_of_processes> ./parallel_sim_mpi
    ```

    For the hybrid code (MPI + OpenMP):
    ```bash
    mpirun -n <number_of_processes> ./parallel_sim_hybrid
    ```

## Understanding the Code

The core functionality resides in the `serial_sim.c`, `parallel_sim_omp.c`, `parallel_sim_mpi.c`, and `parallel_sim_hybrid.c` files. Below is an outline of the main steps:

* **Initialization:**
    - The grid dimensions and the time steps for the simulation are defined.
    - Memory is allocated for the temperature grid.
    - The grid is initialized with appropriate temperature values.

* **Parallelization (for OpenMP and MPI code):**
    - OpenMP directives are used for parallel computation in the OpenMP code.
    - The MPI library is used for parallel computation in the MPI code.
    - Both approaches are used together for the hybrid code.

* **Heat Diffusion Simulation:**
    - The main program loop runs for the specified time steps.
    - Within each time step:
        - The `update` function calculates new temperature values for each grid point based on its neighbors and the heat diffusion equation.

* **Completion:**
    - The memory allocated for the grid is freed.

## Testing and Results

* After running the simulation, the results will be stored in a `heatmap.txt` file.
* You can use any visualization tool (e.g., Gnuplot, Matplotlib) to plot the data and observe how the temperature distribution in the grid evolves over time.

This repository provides a complete codebase for heat diffusion simulation in different execution environments, enabling the examination and evaluation of serial, parallel, and hybrid implementations’ performance.

## Author

Theodoros Taloumtzis ([https://github.com/theodorostaloumtzis](https://github.com/theodorostaloumtzis))
