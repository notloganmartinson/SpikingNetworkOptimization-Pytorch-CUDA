# FasterSNO (Spiking Network Optimization)

**DISCLAIMER & CREDITS:** This repository is a personal engineering exercise and is NOT intended to claim superiority over the original research it replicates. The core scientific methodology, equations, and Factor Analysis algorithms used here are derived entirely from the groundbreaking paper:

*“Automated customization of large-scale spiking network models to neuronal population activity”*
by S. Wu, C. Huang, A. C. Synder, M. A. Smith, B. Doiron, B. M. Yu. (Nat Comput Sci, 2024).

I was reading their fascinating paper, noticed the Factor Analysis pipeline in their original MATLAB codebase was heavily CPU-bound, and wanted to see if I could port the entire simulation and optimization loop to a consumer GPU using modern PyTorch/C++ extensions. This project was built to test my own AI-assisted coding capabilities using Gemini CLI. The original authors' MATLAB codebase remains the scientifically rigorous, general-purpose standard for their methodology, and all credit for the Spiking Network Optimization with Population Statistics (SNOPS) framework goes to them. You can find their original repository and code here: [SpikingNetworkOptimization](https://github.com/ShenghaoWu/SpikingNetworkOptimization).

---

### Overview

`fasterSNO` is an experimental, high-performance PyTorch/CUDA port of the SNOPS algorithm. It simulates large-scale Exponential Integrate-and-Fire (EIF) spiking networks and uses Bayesian optimization (Optuna) to fit network parameters (like connection strengths and decay times) to target biological Population Statistics, such as Firing Rate, Coefficient of Variation (CV), and Factor Analysis (FA) shared variance metrics.

By moving both the sparse neural integration and the Expectation-Maximization (EM) Factor Analysis algorithms entirely into custom C++/CUDA kernels, `fasterSNO` eliminates CPU-GPU data transfer bottlenecks. It achieves a 3x to 5x speedup over the original CPU-bound FA pipeline, allowing a 10-biological-second simulation and 5-fold cross-validated FA to run in roughly ~18 seconds on a 6GB VRAM GPU.

### Prerequisites

*   Linux operating system.
*   NVIDIA GPU (tested within a 6GB VRAM constraint).
*   Python 3.12+
*   PyTorch (with CUDA support)
*   Optuna
*   Ninja (for compiling PyTorch C++ extensions)

### Installation

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install torch optuna matplotlib numpy
   ```

### Usage

**1. Define Targets**
Edit `targets.json` to define the target biological metrics (e.g., V4 visual cortex data) and their respective penalty weights for the cost function.

**2. Run the Optimizer**
To begin searching the 9-dimensional parameter space for a network state that matches your targets:
```bash
python optimizer.py
```
This will create a local SQLite database (`v4_optimization.db`) to safely persist all trial data. You can visualize the optimization history using Optuna's dashboard:
```bash
optuna-dashboard sqlite:///v4_optimization.db
```

**3. Verify and Plot Results**
To run a 10-second simulation using the best parameters found by the optimizer and generate a detailed metrics report and raster plot:
```bash
python verify_replica.py
```
This script will output `final_metrics_report.txt` and `best_params_raster.png`.

**4. Random Parameter Sweep**
To test random biological parameters without Bayesian guidance:
```bash
python sweep.py
```

### Architecture
*   `simulation_core.py`: The unified PyTorch C++ extension that handles spatial 2D distance-based sparse CSR weight generation, feedforward Poisson drives, and the EIF CUDA integration kernels.
*   `fa_module.py`: A custom C++ extension implementing the Factor Analysis EM algorithm and K-Fold cross-validation entirely in `Float64` on the GPU.
*   `metrics.py`: Data binning, pairwise correlation, and FA metric extraction.
*   `optimizer.py`: The Optuna-based dynamic objective function.

### Acknowledgments
All core scientific methodologies, equations, and Factor Analysis approaches are derived from the original SNOPS paper and codebase. See `research-notes.md` for a detailed breakdown of the engineering decisions and known scientific compromises made during this port.
