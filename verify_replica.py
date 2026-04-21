import torch
import warnings
import matplotlib.pyplot as plt
from simulation_core import run_simulation
from metrics import calculate_metrics

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("VERIFYING WITH PAPER ORIGINAL EXPERIMENTAL WEIGHTS")
    # Best params from a longer run that found the target metrics
    best_params = {'tau_ed': 6.0, 'tau_id': 16.0, 'jee': 11.6, 'jie': 136.9, 'jei': -6.4, 'jii': -143.1, 'jef': 52.4, 'jif': 10.3, 'sigmaRR_E': 0.1, 'sigmaRR_I': 0.1, 'sigmaRX': 0.05}
    
    print(f"Running simulation with params: {best_params}")
    spike_times, spike_ids, N_total = run_simulation(
        **best_params, t_steps=200000, device='cuda'
    )
    
    rate, cv, pshared, dshared, mean_corr = calculate_metrics(
        spike_times, spike_ids, N_total, t_max_ms=10000.0, calc_fa=True
    )
    
    report = f"""=== FINAL REPLICA METRICS ===
Target Firing Rate : 5.4 Hz    | Actual: {rate:.2f} Hz
Target CV          : 0.8       | Actual: {cv:.4f}
Target % Shared    : ~0.1      | Actual: {pshared:.4f}
Target Shared Dim  : ~5        | Actual: {dshared}
Target Mean Corr   : ~0.05     | Actual: {mean_corr:.4f}
"""
    print(f"\n{report}")

    with open("final_metrics_report.txt", "w") as f:
        f.write(report)
    print("Saved report to final_metrics_report.txt")

    total_spikes = len(spike_times)
    if total_spikes > 0:
        times = spike_times.cpu().numpy()
        ids = spike_ids.cpu().numpy()
        
        # Plot only 1 second of data (bins 5000 to 6000 ms) and up to 500 neurons to keep plot legible
        mask = (ids < 500) & (times > 5000.0) & (times < 6000.0)
        times_in_seconds = times[mask] / 1000.0

        plt.figure(figsize=(12, 5))
        plt.plot(times_in_seconds, ids[mask], '|', markersize=1.5, color='black')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Neuron ID", fontsize=12)
        plt.title(f"Optimized Network Raster (Rate: {rate:.2f} Hz, d_sh: {dshared})", fontsize=14)
        plt.xlim(5.0, 6.0) 
        plt.tight_layout()
        plt.savefig("best_params_raster.png", dpi=300)
        print("Saved optimal raster plot to best_params_raster.png!")
