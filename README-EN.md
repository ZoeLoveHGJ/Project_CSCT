1# Project-CSCT: A Simulation Framework for Mobile RFID Anti-Collision Algorithms

This repository contains the simulation framework for **CSCT**, a novel anti-collision algorithm designed for mobile RFID tags under the EPC C1G2 standard. The framework is developed to simulate and evaluate the performance of CSCT against several baseline algorithms in various mobile environments.

## Key Features

- **EPC C1G2 Compliance:** The simulation core is built upon the specifications of the EPCglobal Class 1 Generation 2 standard.
- **Mobile Tag Simulation:** Specifically designed to model scenarios with mobile RFID tags, considering factors like tag movement and residence time.
- **Multiple Algorithm Implementations:** Includes the proposed CSCT algorithm and a variety of well-known baseline algorithms for comprehensive comparison.
- **Extensive Experiment Scripts:** Provides scripts to easily replicate experiments that evaluate algorithms against different metrics like arrival rate, tag residence time, and burstiness.
- **Data Analysis and Visualization:** Contains tools and scripts to process raw simulation data into final figures and KPIs.

## Algorithms Implemented

### Proposed Algorithm
- **CSCT**: The core algorithm proposed in this research project.
- **CSCT_Estimator**: A version of CSCT using a backlog estimation technique.
- **CSCT_Oracle**: A version of CSCT with access to perfect backlog information, serving as a theoretical upper bound.

### Baseline Algorithms
- **ABP** (Adaptive Binary Splitting)
- **ACDQT** (Adaptive-Correlative Dynamic Query Tree)
- **DCT** (Dynamic-Class-based Traversal)
- **EDRCT** (Efficient Dynamic RFID Tag Collection)
- **MRSMBA** (Mobility-Resistant Slotted-Aloha-based)
- **TAD** (Tag-Amount-Difference-based)

## Project Structure

```
Project-CSCT/
│
├── Framework.py                # Core simulation framework engine
├── CSCT.py                     # Implementation of the CSCT algorithm
├── MRSMBA.py, DCT.py, etc.     # Implementations of baseline algorithms
│
├── exp0.py, exp1.py, etc.      # Main experiment execution scripts
├── Analysis.py                 # Helper script for analyzing experiment results
├── Tool.py                     # Utility functions for experiments
│
├── exp_final_*/                # Directories containing final experiment results
├── exp0_results_*/             # Directories for specific experiment results (data and charts)
│   ├── kpi_*.csv               # CSV files with Key Performance Indicators
│   └── *.png, *.pdf            # Figures and plots generated from results
│
└── README.md                   # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Project-CSCT
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. The project requires common scientific computing libraries.
    > **Note:** A `requirements.txt` file is not present. You may need to manually install libraries like `pandas`, `matplotlib`, and `numpy`. It is recommended to create a `requirements.txt` file for easier dependency management.
    ```bash
    pip install pandas matplotlib numpy scipy
    ```

## How to Run Experiments

1.  **Configure an experiment:**
    Open one of the experiment files (e.g., `exp1.py`, `exp2.py`).
    Modify parameters such as `TAG_ARRIVAL_RATE`, `SIMULATION_TIME`, or algorithm-specific settings as needed.

2.  **Execute the script:**
    Run the desired experiment script from the terminal.
    ```bash
    python exp1.py
    ```

3.  **Find the results:**
    After the simulation completes, the results (CSV data files and plots) will be saved in the corresponding results directory (e.g., `exp1_results_vs_arrival_rate/`).
