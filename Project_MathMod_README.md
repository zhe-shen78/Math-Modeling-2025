# 2025 National Math Modeling Contest (Problem A) - Smoke Interference Strategy

## 📖 Project Overview
This repository contains the solution for the **2025 National Undergraduate Mathematical Contest in Modeling (Problem A)**. Our team secured the **Provincial 1st Prize** (Shandong Province).

### 🎯 The Challenge: Smoke Interference Strategy
The task involved optimizing the deployment of smoke interference bombs by drones to protect a ground target from incoming air-to-ground missiles.

**Key Constraints & Variables:**
- **Target**: A cylindrical fixed target (Radius: 7m, Height: 10m) protected by a decoy.
- **Threat**: 3 incoming missiles ($M_1, M_2, M_3$) flying at 300 m/s.
- **Defense**: 5 Drones ($FY_1$ to $FY_5$) carrying smoke bombs.
- **Physics**:
  - Drones fly at 70-140 m/s.
  - Smoke bombs explode after a delay, forming a cloud that sinks at 3 m/s.
  - Effective shielding radius: 10m; Duration: 20s.

**Objective**: Maximize the total effective shielding time for the true target against the missiles by calculating the optimal:
- Drone flight paths (Direction & Speed).
- Bomb release points.
- Detonation timing.

## 💡 Our Approach
We developed a multi-stage optimization model to solve this dynamic pursuit-evasion problem:

1.  **Geometric Modeling**: Established a 3D coordinate system to track the trajectories of missiles, drones, and sinking smoke clouds.
2.  **Optimization Algorithms**:
    - **Single Drone/Single Missile**: Calculated the precise interception window using kinematic equations.
    - **Multi-Drone/Multi-Missile**: Utilized **Genetic Algorithms (GA)** and **Simulated Annealing** to coordinate multiple drones for continuous coverage.
3.  **Simulation**: Built a Python simulation to verify the shielding coverage over time.

## 📂 Files
- `Paper.pdf`: The final submitted paper detailing our methodology and results.
- `Code/`: Python scripts for simulation and optimization.
  - `Problem 1`: Single drone interception analysis.
  - `Problem 2-5`: Multi-agent cooperative strategy algorithms.

## 🏆 Achievement
- **Award**: Provincial First Prize.
- **Role**: Team Leader & Modeler. Responsible for algorithm design and Python implementation.
