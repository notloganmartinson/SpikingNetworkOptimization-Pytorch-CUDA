import torch
import numpy as np
from fa_module import compute_population_statistics_cv

def get_spike_counts(spike_times, spike_ids, n_neurons, t_max_ms, bin_size_ms=50.0):
    num_bins = int(t_max_ms / bin_size_ms)
    bins = (spike_times / bin_size_ms).long()
    
    valid_mask = (bins >= 0) & (bins < num_bins)
    valid_bins = bins[valid_mask]
    valid_ids = spike_ids[valid_mask]
    
    spike_counts = torch.zeros((n_neurons, num_bins), device=spike_times.device, dtype=torch.float64)
    # Must use index_put with accumulate=True, but PyTorch 1D index_put_ doesn't handle duplicate 2D indices properly if not flattened
    for b, idx in zip(valid_bins, valid_ids):
        spike_counts[idx, b] += 1.0
    return spike_counts

def calculate_mean_corr(spike_counts):
    if spike_counts.shape[1] < 2:
        return 0.0
    # Center counts
    counts_centered = spike_counts - spike_counts.mean(dim=1, keepdim=True)
    # Normalize
    std = counts_centered.std(dim=1, unbiased=True, keepdim=True)
    std[std == 0] = 1e-6
    counts_norm = counts_centered / std
    
    # Compute correlation matrix
    N_bins = spike_counts.shape[1]
    corr_matrix = (counts_norm @ counts_norm.T) / (N_bins - 1)
    
    # Extract mean of upper triangle
    N_neurons = corr_matrix.shape[0]
    triu_indices = torch.triu_indices(N_neurons, N_neurons, offset=1)
    mean_corr = corr_matrix[triu_indices[0], triu_indices[1]].mean().item()
    
    # Handle NaN if any
    if np.isnan(mean_corr):
        return 0.0
    return mean_corr

def calculate_metrics(spike_times, spike_ids, N_total, t_max_ms=1000.0, calc_fa=True, n_sample_fa=100):
    total_spikes = len(spike_times)
    rate = (total_spikes / N_total) / (t_max_ms / 1000.0)

    cv = 0.0
    if total_spikes > 100:
        times_cpu = spike_times.cpu().numpy()
        ids_cpu = spike_ids.cpu().numpy()
        cvs = []
        for n in range(min(100, N_total)): # Match paper sample size of 100
            n_spikes = times_cpu[ids_cpu == n]
            if len(n_spikes) >= 2:
                isi = np.diff(n_spikes)
                if np.mean(isi) > 0:
                    cvs.append(np.std(isi) / np.mean(isi))
        if cvs:
            cv = np.mean(cvs)

    pshared, dshared, mean_corr = 0.0, 0, 0.0
    if calc_fa and total_spikes > 100:
        # Sample a subset of E neurons for FA to avoid memory/compute blowup, matching paper
        # Paper uses sample_e_neurons, N=100 typically
        N_E = 2500
        # Choose `n_sample_fa` excitatory neurons randomly
        sample_ids = torch.randperm(N_E, device=spike_times.device)[:n_sample_fa]
        
        # Get spike counts for all neurons
        spike_counts = get_spike_counts(spike_times, spike_ids, N_total, t_max_ms, bin_size_ms=50.0)
        
        # Slice for the sample
        sample_counts = spike_counts[sample_ids]
        
        # Compute mean pairwise correlation
        mean_corr = calculate_mean_corr(sample_counts)
        
        try:
            pshared, dshared, _, _, _ = compute_population_statistics_cv(sample_counts, candidate_zDims=[3, 5, 8, 10, 15], num_folds=5)
        except Exception as e:
            pass # Skip failed FA models
            
    return rate, cv, pshared, dshared, mean_corr