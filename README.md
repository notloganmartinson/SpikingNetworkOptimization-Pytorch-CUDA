# FasterSNO (Spiking Network Optimization)

**DISCLAIMER & CREDITS:** This repository is a personal engineering exercise and is NOT intended to claim superiority over the original research it replicates. The core scientific methodology, equations, and Factor Analysis algorithms used here are derived entirely from the groundbreaking paper:

*“Automated customization of large-scale spiking network models to neuronal population activity”*
by S. Wu, C. Huang, A. C. Synder, M. A. Smith, B. Doiron, B. M. Yu. (Nat Comput Sci, 2024).

I was reading their fascinating paper, noticed the Factor Analysis pipeline in their original MATLAB codebase was heavily CPU-bound, and wanted to see if I could port the entire simulation and optimization loop to a consumer GPU using modern PyTorch/C++ extensions. This project was built to test my own AI-assisted coding capabilities using Gemini CLI. The original authors' MATLAB codebase remains the scientifically rigorous, general-purpose standard for their methodology, and all credit for the Spiking Network Optimization with Population Statistics (SNOPS) framework goes to them. You can find their original repository and code here: [SpikingNetworkOptimization](https://github.com/ShenghaoWu/SpikingNetworkOptimization).

---

### Overview

`FasterSNO` is a high-performance PyTorch/CUDA port of the SNOPS algorithm. It simulates large-scale **Spatial Balanced Networks (SBN)** using Exponential Integrate-and-Fire (EIF) neurons and uses Bayesian optimization (Optuna) to fit network parameters to target biological Population Statistics, such as Firing Rate, Coefficient of Variation (CV), and Factor Analysis (FA) shared variance metrics.

By moving both the sparse spatial neural integration and the 64-bit Cross-Validated Factor Analysis algorithms entirely into custom C++/CUDA kernels, `FasterSNO` eliminates CPU-GPU data transfer bottlenecks. It achieves a significant speedup over the original CPU-bound pipeline, allowing a 10-biological-second simulation and 5-fold cross-validated FA to run in roughly ~18 seconds on a 6GB VRAM GPU.

### Key Scientific Upgrades
*   **True Spatial Topology:** Connections are generated based on 2D Euclidean distance using Gaussian probability profiles, matching the structural physics of the paper.
*   **64-Bit Precision:** All Factor Analysis math (Expectation-Maximization) is executed in `Float64` to ensure numerical stability for ill-conditioned covariance matrices.
*   **True 5-Fold Cross-Validation:** Latent dimensionality (`zDim`) is selected by maximizing the Log-Likelihood on held-out test data, preventing overfitting.

### Prerequisites

*   Linux operating system.
*   NVIDIA GPU (tested within a 6GB VRAM constraint).
*   Python 3.12+
*   PyTorch (with CUDA support)
*   Optuna & Ninja

### Installation

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**1. Define Targets**
Edit `targets.json` to dynamically set the target biological metrics and their respective penalty weights.

**2. Run the Optimizer**
To begin searching the 11-dimensional parameter space:
```bash
./run_optimizer_bg.sh
```
This runs the optimizer in the background using `nohup`. You can monitor progress with:
```bash
tail -f optimization_run.log
```
Trials are safely persisted in `v4_optimization.db`. Visualize your results using the dashboard:
```bash
optuna-dashboard sqlite:///v4_optimization.db
```

**3. Verify and Plot Results**
To generate a final metrics report and optimal raster plot:
```bash
python verify_replica.py
```

### Architecture
*   `simulation_core.py`: Unified PyTorch C++ extension handling 2D distance-based sparse CSR weight generation and EIF CUDA integration.
*   `fa_module.py`: Custom C++ extension implementing the FA EM algorithm and K-Fold cross-validation entirely in `Float64` on the GPU.
*   `metrics.py`: Data binning, pairwise correlation, and population statistic extraction.
*   `optimizer.py`: Optuna-based dynamic objective function class.

### Acknowledgments
All scientific methodologies are derived from the SNOPS paper and Dr. Shenghao Wu's repository. Detailed engineering trade-offs are documented in `research-notes.md`.