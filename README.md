# MNIST Experiment Tracker

## Project overview

This repository provides a system for tracking machine learning experiments. It includes a modified version of the experiments.py file and a tracking module to log key aspects of each experiment. A dashboard module allows users to visualize and compare experiment results, while a docker-compose is used for setting up a PostgreSQL database for data storage and retrieval.

### Features:

#### Tracking Module

Logs critical experiment details, including:

- Configurations
- Training Metrics
- Evaluation Metrics
- Model Checkpoints
- Dataset Metadata

#### Dashboard Module

Provides an interface to view, compare, and analyze experiment results.

#### Database Integration

Uses PostgreSQL (via docker-compose) for storage and retrieval of experiment data.

## Setting Up the Environment

### 1️. Create a Virtual Environment

This project requires Python **3.13**. To set up a virtual environment under `.venv`, run:

```sh
python -m venv .venv
```

### 2️. Activate the Virtual Environment

#### On macOS/Linux:

```sh
source .venv/bin/activate
```

#### On Windows (cmd):

```sh
.venv\Scripts\activate
```

#### On Windows (PowerShell):

```sh
.venv\Scripts\Activate.ps1
```

### 3️. Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```sh
pip install -r requirements.txt
```

---

## Setting Up the Database

### 4️. Start PostgreSQL with Docker

Ensure you have **Docker** installed. Start the database using:

```sh
docker compose up -d
```

This will run the PostgreSQL database as a detached process.

### 5️. Apply Database Migrations

Run Alembic to migrate the database schema:

```sh
alembic upgrade head
```

---

## Running Experiments

### 6️. Run the Experiment Script

To execute the experiment, run:

```sh
python src/experiments.py
```

---

## Running the Dashboard

### 7️. Start the Streamlit Dashboard

To start the dashboard, run:

```sh
python -m streamlit run src/app.py
```

This will launch a web-based dashboard for visualization and interaction.

---

## Summary of Commands

| Step                         | Command                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| Create Virtual Environment   | `python -m venv .venv`                                                          |
| Activate Virtual Environment | `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows) |
| Install Dependencies         | `pip install -r requirements.txt`                                               |
| Start PostgreSQL (Docker)    | `docker compose up -d`                                                          |
| Run Database Migrations      | `alembic upgrade head`                                                          |
| Run Experiments              | `python src/experiments.py`                                                     |
| Run Streamlit Dashboard      | `python -m streamlit run src/app.py`                                            |

---

## Notes

- Ensure **Python 3.13** is installed before proceeding.
- Install **Docker** if you haven't already.
- If you encounter permission errors with PowerShell, try running:
  ```sh
  Set-ExecutionPolicy Unrestricted -Scope Process
  ```
