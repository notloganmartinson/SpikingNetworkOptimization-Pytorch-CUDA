import optuna
import warnings
import json
import os
from simulation_core import run_simulation
from metrics import calculate_metrics

warnings.filterwarnings("ignore")

class DynamicObjective:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
    def __call__(self, trial):
        # Table 1 Parameters
        tau_ed = trial.suggest_float('tau_ed', 1.0, 25.0)
        tau_id = trial.suggest_float('tau_id', 1.0, 25.0)
        jee = trial.suggest_float('jee', 0.0, 150.0)
        jie = trial.suggest_float('jie', 0.0, 150.0)
        jei = trial.suggest_float('jei', -150.0, 0.0)
        jii = trial.suggest_float('jii', -150.0, 0.0)
        jef = trial.suggest_float('jef', 0.0, 150.0)
        jif = trial.suggest_float('jif', 0.0, 150.0)
        sigmaRR_E = trial.suggest_float('sigmaRR_E', 0.0, 0.25)
        sigmaRR_I = trial.suggest_float('sigmaRR_I', 0.0, 0.25)
        sigmaRX = trial.suggest_float('sigmaRX', 0.0, 0.25)

        # Run the unified sparse engine
        spike_times, spike_ids, N_total = run_simulation(
            tau_ed, tau_id, jee, jie, jei, jii, jef, jif, 
            sigmaRR_E=sigmaRR_E, sigmaRR_I=sigmaRR_I, sigmaRX=sigmaRX, t_steps=200000, device='cuda'
        )
        
        # Calculate all metrics including Factor Analysis population statistics
        rate, cv, pshared, dshared, mean_corr = calculate_metrics(
            spike_times, spike_ids, N_total, t_max_ms=10000.0, calc_fa=True
        )

        t = self.config["targets"]
        w = self.config["weights"]
        bounds = self.config["rate_bounds"]

        # Dynamic early stopping bounds
        if rate < bounds[0] or rate > bounds[1]:
            # Heavy penalty for falling completely outside biologically plausible bounds
            return 2000.0 + (rate - t["rate"])**2

        # Dynamic cost function
        cost = (
            ((rate - t["rate"])**2 * w["rate"]) +
            ((cv - t["cv"])**2 * w["cv"]) +
            ((pshared - t["pshared"])**2 * w["pshared"]) +
            ((dshared - t["dshared"])**2 * w["dshared"]) +
            ((mean_corr - t["mean_corr"])**2 * w["mean_corr"])
        )
        
        return cost

if __name__ == "__main__":
    config_path = 'targets.json'
    print(f"STARTING STOCHASTIC BAYESIAN SEARCH (Dynamic Configuration: {config_path})")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['TMPDIR'] = os.path.join(current_dir, 'tmp')
    
    objective_func = DynamicObjective(config_path)
    
    study = optuna.create_study(
        study_name=objective_func.config.get('experiment_name', 'v4_optimization'),
        storage='sqlite:///v4_optimization.db',
        load_if_exists=True,
        direction='minimize'
    )
    study.optimize(objective_func, n_trials=100)

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Experiment     : {objective_func.config.get('experiment_name', 'Unknown')}")
    print(f"Best Parameters: {study.best_params}")
    print(f"Best Cost      : {study.best_value:.4f}")