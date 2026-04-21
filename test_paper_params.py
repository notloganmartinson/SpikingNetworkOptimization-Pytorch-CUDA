import torch
import warnings
from simulation_core import run_simulation
from metrics import calculate_metrics

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Parameters from simulator_short_demo.m (ground truth)
    # taudsynI=10.0 (guess), mean_sigmaRRIs=0.1 (guess)
    paper_params = {
        'tau_ed': 22.102, 
        'tau_id': 10.0, 
        'jee': 2.7119, 
        'jie': 27.965, 
        'jei': -66.265, 
        'jii': -74.641, 
        'jef': 101.54, 
        'jif': 35.767, 
        'sigmaRR_E': 0.13862, 
        'sigmaRR_I': 0.1, 
        'sigmaRX': 0.063017
    }
    
    print(f"Running simulation with Paper params: {paper_params}")
    spike_times, spike_ids, N_total = run_simulation(
        **paper_params, t_steps=200000, device='cuda'
    )
    
    rate, cv, pshared, dshared, mean_corr = calculate_metrics(
        spike_times, spike_ids, N_total, t_max_ms=10000.0, calc_fa=True
    )
    
    print("\n=== REPLICA METRICS ===")
    print(f"Firing Rate : {rate:.2f} Hz")
    print(f"CV          : {cv:.4f}")
    print(f"% Shared    : {pshared:.4f}")
    print(f"Shared Dim  : {dshared}")
    print(f"Mean Corr   : {mean_corr:.4f}")
