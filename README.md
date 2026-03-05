# DepOp Seminar – Experiment Repository

## Project Description

This repository contains the implementation and experimental environment used for the seminar paper titled:

**"Word embedding for semantic SQL in RDBMS"**

The repository provides the code required to reproduce the experiments, analyses, and visualizations discussed in the paper.

The project consists of a Python package implementing the core functionality, several Jupyter notebooks used to run the experiments, and a containerized environment to ensure reproducibility.

All experiments can be executed through the provided notebooks within the Docker-based environment.

---

# Repository Structure

### Python Package

The repository includes a Python package that implements the computational logic required for the experiments.

This package performs tasks such as:

* data processing
* model computation
* experiment orchestration
* generation of intermediate results

The package is installed automatically inside the Docker environment.

---

### Jupyter Notebooks

The repository contains **five Jupyter notebooks** that reproduce the experiments described in the seminar paper.

The notebooks use the functions provided by the Python package to:

* perform experimental computations
* generate datasets and intermediate results
* produce the visualizations shown in the seminar paper

Each notebook represents a specific experiment or analysis step.

---

### Docker Environment

The project uses **Docker Compose** to create a reproducible environment consisting of two containers:

1. **PostgreSQL database container**

   * stores experimental data
   * provides database access for the experiments

2. **Python / Jupyter Notebook container**

   * installs the project Python package
   * contains all required dependencies
   * runs the Jupyter Notebook server used to execute the experiments

---

# Setup and Execution

## Requirements

To run the project locally, the following software must be installed:

* Docker

No local Python installation is required.

---

## Starting the Environment

To build and start the experiment environment, run:

```bash
docker compose up --build
```

This command performs the following steps:

* builds the Python container
* starts both containers
* initializes the PostgreSQL database
* installs the Python package inside the container
* launches the Jupyter Notebook server

---

## Accessing the Notebooks

After the containers have started, the terminal will display a URL for accessing the Jupyter Notebook interface.

Open this URL in a browser to access the notebooks.

All experiments can be executed from the notebooks located in the `notebooks/` directory.

---

# Running the Experiments

All relevant project functionality is already integrated into the notebooks.

Executing notebook cells will:

* generate experimental data
* run the computational models
* produce the plots and results described in the seminar paper

Note that executing the notebooks may trigger computationally expensive operations.

Depending on the available hardware, generating the full set of experimental results may take a considerable amount of time.

---

# Reproducibility

This repository is intended to allow reproduction of the experimental results presented in the seminar paper.

The Docker-based environment ensures that all required dependencies and services are configured consistently.

