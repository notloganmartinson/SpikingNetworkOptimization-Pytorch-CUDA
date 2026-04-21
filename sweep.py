import random
import time
import warnings
from simulation_core import run_simulation
from metrics import calculate_metrics

warnings.filterwarnings("ignore")

# --- THE 5000-DART SWEEP (TABLE 1 PARAMETERS) ---
if __name__ == "__main__":
    N_TRIALS = 2
    print(f"Starting {N_TRIALS} random parameter sweeps from Table 1 ranges...")
    
    valid_states = []

    for i in range(N_TRIALS):
        # Exact parameter boundaries from Table 1
        tau_ed = random.uniform(1.0, 25.0)
        tau_id = random.uniform(1.0, 25.0)
        jei    = random.uniform(-150.0, 0.0)
        jie    = random.uniform(0.0, 150.0)
        jii    = random.uniform(-150.0, 0.0)
        jee    = random.uniform(0.0, 150.0)
        jef    = random.uniform(0.0, 150.0)
        jif    = random.uniform(0.0, 150.0)
        sigmaRR_E = random.uniform(0.0, 0.25)
        sigmaRR_I = random.uniform(0.0, 0.25)
        sigmaRX = random.uniform(0.0, 0.25)

        # Run Simulation
        spike_times, spike_ids, N_total = run_simulation(
            tau_ed, tau_id, jee, jie, jei, jii, jef, jif, 
            sigmaRR_E=sigmaRR_E, sigmaRR_I=sigmaRR_I, sigmaRX=sigmaRX, t_steps=200000, device='cuda'
        )

        # Compute Metrics (Rate, CV, and FA Population Stats)
        rate, cv, pshared, dshared, mean_corr = calculate_metrics(
            spike_times, spike_ids, N_total, t_max_ms=10000.0, calc_fa=True
        )

        # Filtering rule directly from paper: 0.5 < fr < 60.0
        if 0.5 <= rate <= 60.0:
            print(f"[{i+1}/{N_TRIALS}] VALID HIT! Rate: {rate:.2f} Hz | CV: {cv:.4f} | %sh: {pshared:.3f} | d_sh: {dshared} | m_corr: {mean_corr:.4f}")
            valid_states.append((rate, cv, pshared, dshared, mean_corr, tau_ed, tau_id, jee, jie, jei, jii, jef, jif, sigmaRR_E, sigmaRR_I, sigmaRX))
        else:
            print(f"[{i+1}/{N_TRIALS}] Invalid (Rate: {rate:.2f} Hz)")

    print(f"\nSweep Complete. Found {len(valid_states)} biologically valid states.")
