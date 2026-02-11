# CSCT: Context-Scheduled Collision Tree

> **A Reliable Anti-collision Protocol for Mobile RFID Tags Identification**

[![Journal](https://img.shields.io/badge/Journal-Computer%20Networks-blue)](https://www.sciencedirect.com/journal/computer-networks)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.comnet.2026.112091-green)](https://doi.org/10.1016/j.comnet.2026.112091)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

This repository contains the source code and simulation framework for the paper:

**"Context-Scheduled Collision Tree: A Reliable Anti-collision Protocol for Mobile RFID Tags Identification"**

Published in *Computer Networks* (Online: February 10, 2026)

---

## Table of Contents

- [Research Background](#research-background)
- [Proposed Method](#proposed-method)
- [Project Structure](#project-structure)
- [Implemented Algorithms](#implemented-algorithms)
- [Key Performance Indicators](#key-performance-indicators)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Key Results](#key-results)
- [Citation](#citation)

---

## Research Background

In traditional RFID systems, tags are assumed to remain stationary within the reader's interrogation zone until they are identified. However, in real-world applications such as **warehouse logistics**, **conveyor-belt inventory**, and **vehicular access control**, RFID tags are attached to mobile objects that **continuously enter and leave** the reader's field. This mobility poses significant challenges:

- **Tag Loss**: Tags may leave the field before being successfully identified.
- **Dynamic Backlog**: The number of unidentified tags fluctuates unpredictably over time.
- **Time Urgency**: Each tag has a limited residence time, creating a deadline for identification.

Existing tree-based and ALOHA-based anti-collision protocols are designed for static scenarios and fail to adequately address these challenges, resulting in high tag loss rates and low throughput under mobile conditions.

## Proposed Method

We propose **CSCT (Context-Scheduled Collision Tree)**, a novel anti-collision protocol specifically designed for mobile RFID tag identification. CSCT operates through a **two-phase architecture**:

### Phase A — Discovery
A slotted-ALOHA frame is used to quickly discover the set of active tags in the field. Fresh tags respond based on a hash of their ID, and collisions are recorded as **Collision Contexts** — structured metadata capturing the set of colliding tags and the associated prefix.

### Phase B — Resolution
Collision Contexts are managed in a **Context-Collision Queue (CCQ)**, a priority queue that schedules the resolution order based on a configurable priority function:

```
Priority = α × EstimatedSize − β × PrefixDepth
```

This scheduling strategy prioritizes larger collision groups (which contain more tags at risk of being lost), while preferring shallower tree nodes (which can be resolved more efficiently).

### Key Innovations
1. **Context-aware scheduling** — Collision resolution is dynamically prioritized based on the estimated urgency and size of each collision context.
2. **Two-phase decoupled architecture** — Separates the discovery of new tags from the resolution of known collisions, enabling continuous processing of incoming traffic.
3. **Physical layer robustness** — Incorporates simulation of random channel errors and capture effects for realistic evaluation.

## Project Structure

```
Project_CSCT/
│
├── Framework.py               # Core simulation engine (FTRSS)
│                               #   - EPC C1G2 physical layer parameters
│                               #   - Mobile tag stream generation (Poisson & Burst/LIFO)
│                               #   - Event-driven simulation loop
│                               #   - Channel error & capture effect modeling
│
├── CSCT.py                    # ★ Proposed CSCT algorithm
├── CSCT_Estimator.py          # CSCT variant with backlog estimation
├── CSCT_Oracle.py             # CSCT variant with perfect backlog info (upper bound)
│
├── ABP.py                     # Baseline: Adaptive Binary Splitting Protocol
├── ACDQT.py                   # Baseline: Adaptive-Correlative Dynamic Query Tree
├── DCT.py                     # Baseline: Dynamic Collision Tree
├── MRSMBA.py                  # Baseline: Mobility-Resistant Slotted-ALOHA-based
├── TAD.py                     # Baseline: Tag-Amount-Difference-based
│
├── algorithm_config.py        # Algorithm registry, configs & plot styles
├── Tool.py                    # KPI calculation & visualization utilities
├── Analysis.py                # Cross-experiment result comparator
│
├── exp1.py                    # Exp 1: Performance vs. tag arrival rate
├── exp1_B.py                  # Exp 1B: Performance vs. burstiness
├── exp1_C.py                  # Exp 1C: Additional arrival rate analysis
├── exp2.py                    # Exp 2: Performance vs. tag residence time
├── exp4.py                    # Exp 4: Full dynamic comparison
├── exp0.py                    # Exp 0: Parameter tuning & sensitivity analysis
├── Exp_Sup_1/2/3.py           # Supplementary experiments
├── exp_final_*.py             # Final polished experiment scripts
│
├── exp*_results_*/            # Output directories (CSV data + figures)
├── Test_Run.py                # Quick single-run debugging script
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Implemented Algorithms

### Proposed

| Algorithm | Description |
|---|---|
| **CSCT** | Context-Scheduled Collision Tree (this work) |
| **CSCT_Estimator** | CSCT with online backlog estimation mechanism |
| **CSCT_Oracle** | CSCT with oracle (perfect) backlog information — theoretical upper bound |

### Baselines

| Algorithm | Full Name | Reference |
|---|---|---|
| **DCT** | Dynamic Collision Tree | 2019 |
| **MRSMBA** | Mobility-Resistant Slotted Multiple Binary-split Algorithm | 2019 |
| **TAD** | Tag-Amount-Difference-based | 2020 |
| **ABP** | Adaptive Binary-splitting Protocol | 2023 |
| **ACDQT** | Adaptive-Correlative Dynamic Query Tree | 2023 |

## Key Performance Indicators

The framework evaluates all algorithms on the following KPIs:

| KPI | Symbol | Description |
|---|---|---|
| **Tag Loss Rate** | RLO | Fraction of tags that leave the field without being identified |
| **System Throughput** | THR | Number of tags successfully identified per second (tags/s) |
| **Average Identification Delay** | DLY | Mean time from a tag's arrival to its successful identification (ms) |
| **Time to Clearance** | TCL | Time required to identify a batch of tags that entered simultaneously (ms) |
| **Ratio of Busyness** | RBY | Fraction of time the reader channel is occupied |
| **Communication Cost** | — | Total reader and tag bits transmitted |

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/ZoeLoveHGJ/Project_CSCT.git
cd Project_CSCT

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data processing & CSV export |
| `matplotlib` | Result visualization |
| `scipy` | Scientific computing utilities |

## Running Experiments

### Quick Validation

For a quick sanity check of the framework:

```bash
python Test_Run.py
```

### Main Experiments

Each experiment script is self-contained and can be run independently:

```bash
# Experiment 1: Tag arrival rate impact
python exp1.py

# Experiment 2: Tag residence time impact
python exp2.py

# Experiment 4: Full dynamic comparison
python exp4.py

# Experiment 1B: Burstiness impact
python exp1_B.py
```

### Parameter Tuning

To explore CSCT's parameter sensitivity:

```bash
python exp0.py
```

### Output

After each experiment completes, results are saved to the corresponding output directory:
- **CSV files**: Raw KPI data (e.g., `kpi_tag_loss_rate.csv`, `kpi_system_throughput_tps.csv`)
- **PNG/PDF figures**: Automatically generated comparison charts

### Cross-Experiment Analysis

To compare CSCT's performance against baselines across all experiments:

```bash
python Analysis.py
```

## Key Results

Under representative mobile RFID scenarios, CSCT demonstrates:

- **Significantly lower tag loss rate** — CSCT's context-aware scheduling ensures high-priority collision groups are resolved before tags leave the field.
- **Higher system throughput** — The two-phase architecture maximizes channel utilization by overlapping discovery and resolution.
- **Robust performance under bursty traffic** — CSCT adapts well to sudden surges in tag arrival thanks to its dynamic Q-value adjustment and priority scheduling.
- **Near-oracle performance** — The gap between CSCT (with estimation) and CSCT_Oracle (with perfect information) is small, validating the effectiveness of the scheduling heuristic.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ZHOU2026112091,
title = {Context-Scheduled Collision Tree: A Reliable Anti-collision Protocol for Mobile RFID Tags Identification},
journal = {Computer Networks},
pages = {112091},
year = {2026},
doi = {https://doi.org/10.1016/j.comnet.2026.112091},
author = {Hongquan Zhou and Xiaolin Jia and Zhong Du and Yajun Gu and Hong Yang},
}
```

## License

This project is open-sourced for academic and research purposes. Please refer to the [LICENSE](LICENSE) file for details.

---

**[中文版 README](README-ZH.md)**
