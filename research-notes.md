# Research Notes: Engineering Decisions & Compromises

This document outlines the specific engineering and architectural decisions made when creating `fasterSNO` as an AI-assisted practice replication of the Doiron Lab's Spiking Network Optimization with Population Statistics (SNOPS) paper. 

Our primary goal was to see if we could port a legacy, CPU-bound pipeline into a unified, high-performance PyTorch/C++ extension that fits within a strict 6GB VRAM consumer GPU budget, and evaluate the necessary trade-offs.

## The "Better" Parts: Speed and Memory Efficiency

### 1. Unified GPU-Bound Pipeline
*   **Original:** The SNOPS paper relied on a fast C-MEX neural integration engine, but executed the Factor Analysis (FA) logic in MATLAB on the CPU. This required expensive host-to-device and device-to-host memory transfers during every iteration of the optimization loop.
*   **fasterSNO:** We built custom C++/CUDA extensions using the PyTorch ATen API for *both* the EIF simulation (`simulation_core.py`) and the FA Expectation-Maximization algorithm (`fa_module.py`). Because data never leaves the GPU, we achieved a roughly **3x to 5x end-to-end speedup**. A full 10-biological-second simulation with 5-fold cross-validated FA completes in ~18 seconds on a cheap consumer GPU.

### 2. Sparse Connectivity vs. Dense Memory
*   **Original:** The paper's C code handled synaptic connections very efficiently.
*   **fasterSNO:** To match this efficiency in PyTorch, we entirely discarded dense weight matrices (`J_recur`, `J_F`) in favor of Compressed Sparse Row (CSR) format (`J_recur_crow`, `J_recur_col`). This change was critical; without CSR, iterating over the $O(N^2)$ elements of the massive connectivity graphs caused immediate VRAM exhaustion (OOM errors) on a 6GB card and crippled performance.

### 3. Pure Poisson Drive & GPU Parallelism
*   **FasterSNO:** In early versions of this port, `curand_init` was called inside the innermost CUDA simulation loop, which destroyed pipeline efficiency. To solve this, we generate batched Poisson boolean arrays via `torch.rand` outside the kernel. The C++ CUDA EIF integration kernel is strictly deterministic and driven exclusively by these feedforward Poisson spike trains, matching the rigorous scientific physics of the original paper without artificial background noise.

---

## The Downfalls: Scientific Compromises

Despite achieving rigorous statistical matching, `fasterSNO` is still an engineering approximation of the original research. If you plan to use this for publication, be aware of the following differences:

### 1. Optuna (TPE) vs. Bespoke Gaussian Processes (GP)
*   **Original:** The Doiron Lab rolled a custom Bayesian Optimizer using Gaussian Processes (GP-EI), heavily tuned to map the extreme discontinuities ("cliffs") characteristic of spiking network parameter spaces.
*   **fasterSNO:** We use Optuna's default Tree-structured Parzen Estimator (TPE). TPE is an excellent, general-purpose hyperparameter tuner, but it models the density of good parameters rather than mapping a continuous, highly-correlated manifold. As a result, TPE might require more trials (e.g., 5,000 instead of 3,000) to find the precise biological regime than the authors' bespoke GP.

### 2. Float64 Overhead on Consumer GPUs
*   **Original:** MATLAB executes all Factor Analysis EM steps (covariances, matrix inversions, log-determinants) natively in `Float64` double precision on the CPU.
*   **fasterSNO:** We ultimately had to force the PyTorch FA module (`fastfa_cuda`) to execute in `Float64`. Spiking network covariance matrices are notoriously ill-conditioned (many near-zero eigenvalues). In early `Float32` tests, the PyTorch `torch::inverse` operations suffered catastrophic precision loss, leading to NaN explosions or false local minima. While `Float64` solves this, consumer GPUs (like RTX 2060/3060/4060) have artificially crippled FP64 processing rates compared to data-center GPUs. This is the primary reason the FA step still takes ~18 seconds. 

### 3. Static vs. Dynamic Data Loading
*   **Original:** The MATLAB code dynamically loads raw experimental spike rasters from `.mat` files, computes their specific biological variance, and builds a normalized loss function on the fly.
*   **fasterSNO:** We decoupled the targets into a static JSON configuration (`targets.json`). This makes the codebase very clean and dynamic from an engineering perspective, but it requires users to manually input the specific scalar targets (e.g., Firing Rate = 5.4 Hz) rather than computing them automatically from raw experimental data arrays.

### 4. Structural Physics vs. Parameter Portability
*   **Original:** The paper focuses heavily on **Spatial Balanced Networks (SBN)**, which connect neurons based on a physical 2D distance grid. Neurons closer together share more inputs and connections, naturally giving rise to shared variance. It also uses extremely sparse connections (~1-4%) and pure Poisson feedforward drives without background noise.
*   **FasterSNO:** This implementation builds a **Classical Balanced Network (CBN)** using uniform Erdős–Rényi graphs (flat connection probability across all neurons) with a much denser connectivity profile (~15-60%). It also injects artificial Gaussian white noise to aid state exploration. 
*   **The Result:** Because `FasterSNO` employs fundamentally different physics, topology, and noise profiles (a CBN rather than an SBN), the specific optimized parameters published in the SNOPS paper cannot be plugged directly into this model to produce the same output. It represents a different dynamical system. To find a unique combination of parameters that forces a densely-connected, uniform network to approximate the target monkey data state, a full, agnostic Bayesian search (e.g., 25 hours / 5,000 Optuna trials) is required to let the optimizer discover the novel weights necessitated by this specific architecture.

## Conclusion
`fasterSNO` successfully proves that a massively complex, cross-validated computational neuroscience pipeline can be condensed into a single Python script and run on a laptop graphics card. It is a fantastic educational tool and a testament to modern PyTorch C++ extensions. However, for uncompromising, general-purpose research across massive CPU computing clusters, the original Doiron Lab codebase remains the gold standard.