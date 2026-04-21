import torch
import time
import argparse
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE CUDA BACKEND (SPARSE AND BIOLOGICALLY ACCURATE) ---
cuda_source = """
#include <torch/extension.h>
#include <math.h>

// KERNEL 1: Feedforward Poisson Drive using PyTorch generated boolean array and CSR
__global__ void feedforward_poisson_kernel(
    const bool* __restrict__ F_spikes,
    float* __restrict__ g_F_rise, 
    const int* __restrict__ J_F_crow,
    const int* __restrict__ J_F_col,
    const float* __restrict__ J_F_val,
    const int N_total, const int N_F
) {
    int source = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int s = source; s < N_F; s += stride) {
        if (F_spikes[s]) {
            int start = J_F_crow[s];
            int end = J_F_crow[s+1];
            for (int i = start; i < end; ++i) {
                int target = J_F_col[i];
                float weight = J_F_val[i];
                atomicAdd(&g_F_rise[target], weight / sqrtf((float)N_total));
            }
        }
    }
}

// KERNEL 2: Exponential Integrate-and-Fire & Double-Exponential Synapses
__global__ void eif_update_kernel(
    float* __restrict__ V, float* __restrict__ cooldown,
    float* __restrict__ g_E_rise, float* __restrict__ g_E_decay,
    float* __restrict__ g_I_rise, float* __restrict__ g_I_decay,
    float* __restrict__ g_F_rise, float* __restrict__ g_F_decay,
    const float* __restrict__ tau_m, const float* __restrict__ Delta_T, const float* __restrict__ tau_ref,
    const float E_L, const float V_T, const float V_th, const float V_re,
    const float dt, const int N, 
    const float tau_er, const float tau_ir, const float tau_Fr,
    const float tau_ed, const float tau_id, const float tau_Fd,
    float* __restrict__ spike_buffer_times, int* __restrict__ spike_buffer_ids,
    int* __restrict__ spike_count, const float current_time, const int max_spikes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        // 1. Double-Exponential Synaptic Updates
        float exp_er = expf(-dt / tau_er); float exp_ed = expf(-dt / tau_ed);
        float exp_ir = expf(-dt / tau_ir); float exp_id = expf(-dt / tau_id);
        float exp_Fr = expf(-dt / tau_Fr); float exp_Fd = expf(-dt / tau_Fd);

        g_E_decay[i] = g_E_decay[i] * exp_ed + g_E_rise[i] * dt;
        g_E_rise[i] = g_E_rise[i] * exp_er;

        g_I_decay[i] = g_I_decay[i] * exp_id + g_I_rise[i] * dt;
        g_I_rise[i] = g_I_rise[i] * exp_ir;

        g_F_decay[i] = g_F_decay[i] * exp_Fd + g_F_rise[i] * dt;
        g_F_rise[i] = g_F_rise[i] * exp_Fr;

        float I_syn = g_E_decay[i] + g_I_decay[i] + g_F_decay[i];

        // 2. Exponential Integrate-and-Fire Voltage Update
        float v = V[i];
        float c = cooldown[i];

        if (c <= 0.0f) {
            float dV = (-(v - E_L) / tau_m[i]) + 
                       (Delta_T[i] / tau_m[i]) * expf((v - V_T) / Delta_T[i]) + 
                       I_syn;
            v += dV * dt;
        }

        // 3. Spike Generation
        if (v >= V_th) {
            v = V_re;
            c = tau_ref[i];
            int slot = atomicAdd(spike_count, 1);
            if (slot < max_spikes) {
                spike_buffer_times[slot] = current_time;
                spike_buffer_ids[slot] = i;
            }
        } else {
            c -= dt;
        }

        V[i] = v;
        cooldown[i] = c;
    }
}

// KERNEL 3: Recurrent Spike Propagation using CSR
__global__ void propagate_kernel(
    const float* __restrict__ V, const float V_re, 
    float* __restrict__ g_E_rise, float* __restrict__ g_I_rise,
    const int* __restrict__ J_recur_crow,
    const int* __restrict__ J_recur_col,
    const float* __restrict__ J_recur_val,
    const int N_total, const int N_E
) {
    int source = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int s = source; s < N_total; s += stride) {
        if (V[s] == V_re) {
            int start = J_recur_crow[s];
            int end = J_recur_crow[s+1];
            for (int i = start; i < end; ++i) {
                int target = J_recur_col[i];
                float weight = J_recur_val[i];
                float scaled_weight = weight / sqrtf((float)N_total);
                if (s < N_E) {
                    atomicAdd(&g_E_rise[target], scaled_weight);
                } else {
                    atomicAdd(&g_I_rise[target], scaled_weight);
                }
            }
        }
    }
}

void step(
    torch::Tensor V, torch::Tensor cooldown, 
    torch::Tensor g_E_rise, torch::Tensor g_E_decay,
    torch::Tensor g_I_rise, torch::Tensor g_I_decay,
    torch::Tensor g_F_rise, torch::Tensor g_F_decay,
    torch::Tensor J_recur_crow, torch::Tensor J_recur_col, torch::Tensor J_recur_val,
    torch::Tensor J_F_crow, torch::Tensor J_F_col, torch::Tensor J_F_val,
    torch::Tensor tau_m, torch::Tensor Delta_T, torch::Tensor tau_ref,
    float E_L, float V_T, float V_th, float V_re, float dt,
    float tau_er, float tau_ir, float tau_Fr,
    float tau_ed, float tau_id, float tau_Fd,
    torch::Tensor spike_times, torch::Tensor spike_ids, torch::Tensor spike_count,
    float current_time, int max_spikes, int N_F,
    torch::Tensor F_spikes
) {
    int N_total = V.size(0);
    int N_E = 2500; // Hardcoded to match paper
    int threads = 256;
    
    int blocks_F = (N_F + threads - 1) / threads;
    feedforward_poisson_kernel<<<blocks_F, threads>>>(
        F_spikes.data_ptr<bool>(),
        g_F_rise.data_ptr<float>(), 
        J_F_crow.data_ptr<int>(), J_F_col.data_ptr<int>(), J_F_val.data_ptr<float>(),
        N_total, N_F
    );

    int blocks_N = (N_total + threads - 1) / threads;
    eif_update_kernel<<<blocks_N, threads>>>(
        V.data_ptr<float>(), cooldown.data_ptr<float>(),
        g_E_rise.data_ptr<float>(), g_E_decay.data_ptr<float>(),
        g_I_rise.data_ptr<float>(), g_I_decay.data_ptr<float>(),
        g_F_rise.data_ptr<float>(), g_F_decay.data_ptr<float>(),
        tau_m.data_ptr<float>(), Delta_T.data_ptr<float>(), tau_ref.data_ptr<float>(),
        E_L, V_T, V_th, V_re, dt, N_total,
        tau_er, tau_ir, tau_Fr, tau_ed, tau_id, tau_Fd,
        spike_times.data_ptr<float>(), spike_ids.data_ptr<int>(),
        spike_count.data_ptr<int>(), current_time, max_spikes
    );

    propagate_kernel<<<blocks_N, threads>>>(
        V.data_ptr<float>(), V_re, 
        g_E_rise.data_ptr<float>(), g_I_rise.data_ptr<float>(), 
        J_recur_crow.data_ptr<int>(), J_recur_col.data_ptr<int>(), J_recur_val.data_ptr<float>(),
        N_total, N_E
    );
}
"""

cpp_source = "void step(torch::Tensor V, torch::Tensor cooldown, torch::Tensor g_E_rise, torch::Tensor g_E_decay, torch::Tensor g_I_rise, torch::Tensor g_I_decay, torch::Tensor g_F_rise, torch::Tensor g_F_decay, torch::Tensor J_recur_crow, torch::Tensor J_recur_col, torch::Tensor J_recur_val, torch::Tensor J_F_crow, torch::Tensor J_F_col, torch::Tensor J_F_val, torch::Tensor tau_m, torch::Tensor Delta_T, torch::Tensor tau_ref, float E_L, float V_T, float V_th, float V_re, float dt, float tau_er, float tau_ir, float tau_Fr, float tau_ed, float tau_id, float tau_Fd, torch::Tensor spike_times, torch::Tensor spike_ids, torch::Tensor spike_count, float current_time, int max_spikes, int N_F, torch::Tensor F_spikes);"
snn_cuda = load_inline(name='snn_v10_sparse', cpp_sources=cpp_source, cuda_sources=cuda_source, functions=['step'], with_cuda=True)

# --- 2. CLI ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--tau_ed', type=float, default=5.0, help='Excitatory decay time (1-25 ms)')
parser.add_argument('--tau_id', type=float, default=8.0, help='Inhibitory decay time (1-25 ms)')
parser.add_argument('--jee', type=float, default=20.0, help='E to E connection (0 to 150 mV)')
parser.add_argument('--jie', type=float, default=10.0, help='E to I connection (0 to 150 mV)')
parser.add_argument('--jei', type=float, default=-60.0, help='I to E connection (-150 to 0 mV)')
parser.add_argument('--jii', type=float, default=-75.0, help='I to I connection (-150 to 0 mV)')
parser.add_argument('--jef', type=float, default=60.0, help='Feedforward to E (0 to 150 mV)')
parser.add_argument('--jif', type=float, default=25.0, help='Feedforward to I (0 to 150 mV)')
args = parser.parse_args()

# --- 3. INITIALIZATION ---
device = 'cuda'
N_e, N_i, N_F = 2500, 625, 625  # Exact sizes from paper
N_total = N_e + N_i

print(f"Constructing Doiron Lab Replica (Sparse CSR) | N_E: {N_e}, N_I: {N_i}, N_F: {N_F}")

# Recurrent Probabilities from paper
P_ee, P_ei, P_ie, P_ii = 0.15, 0.6, 0.45, 0.6
J_recur_dense = torch.zeros((N_total, N_total), device=device)
J_recur_dense[:N_e, :N_e] = (torch.rand((N_e, N_e), device=device) < P_ee).float() * args.jee
J_recur_dense[:N_e, N_e:] = (torch.rand((N_e, N_i), device=device) < P_ie).float() * args.jie
J_recur_dense[N_e:, :N_e] = (torch.rand((N_i, N_e), device=device) < P_ei).float() * args.jei
J_recur_dense[N_e:, N_e:] = (torch.rand((N_i, N_i), device=device) < P_ii).float() * args.jii

J_recur_sparse = J_recur_dense.to_sparse_csr()
J_recur_crow = J_recur_sparse.crow_indices().int()
J_recur_col = J_recur_sparse.col_indices().int()
J_recur_val = J_recur_sparse.values().float()
del J_recur_dense # free memory

# Feedforward Probabilities from paper
P_eF, P_iF = 0.1, 0.05
J_F_dense = torch.zeros((N_F, N_total), device=device)
J_F_dense[:, :N_e] = (torch.rand((N_F, N_e), device=device) < P_eF).float() * args.jef
J_F_dense[:, N_e:] = (torch.rand((N_F, N_i), device=device) < P_iF).float() * args.jif

J_F_sparse = J_F_dense.to_sparse_csr()
J_F_crow = J_F_sparse.crow_indices().int()
J_F_col = J_F_sparse.col_indices().int()
J_F_val = J_F_sparse.values().float()
del J_F_dense # free memory

# State Variables
V = torch.empty(N_total, device=device).uniform_(-65, -50)
cooldown = torch.zeros(N_total, device=device)
g_E_rise = torch.zeros(N_total, device=device)
g_E_decay = torch.zeros(N_total, device=device)
g_I_rise = torch.zeros(N_total, device=device)
g_I_decay = torch.zeros(N_total, device=device)
g_F_rise = torch.zeros(N_total, device=device)
g_F_decay = torch.zeros(N_total, device=device)

# Spike Buffers
max_spikes = 2_000_000
spike_times = torch.zeros(max_spikes, device=device)
spike_ids = torch.zeros(max_spikes, device=device, dtype=torch.int32)
spike_count = torch.zeros(1, device=device, dtype=torch.int32)

# Biological Constants
tau_m = torch.zeros(N_total, device=device); tau_m[:N_e] = 15.0; tau_m[N_e:] = 10.0
Delta_T = torch.zeros(N_total, device=device); Delta_T[:N_e] = 2.0; Delta_T[N_e:] = 0.5
tau_ref = torch.zeros(N_total, device=device); tau_ref[:N_e] = 1.5; tau_ref[N_e:] = 0.5
dt = 0.05
tau_er, tau_ir, tau_Fr = 1.0, 1.0, 1.0
tau_Fd = 5.0
fire_prob = 10.0 * (dt / 1000.0) # 10Hz Poisson drive

# --- 4. EXECUTION ---
print("Simulating 1.0 Biological Second...")
start = time.time()
for t in range(20000):
    # Pre-generate random noise for feedforward Poisson spikes
    F_spikes = torch.rand(N_F, device=device) < fire_prob
    
    snn_cuda.step(
        V, cooldown, 
        g_E_rise, g_E_decay, g_I_rise, g_I_decay, g_F_rise, g_F_decay,
        J_recur_crow, J_recur_col, J_recur_val,
        J_F_crow, J_F_col, J_F_val,
        tau_m, Delta_T, tau_ref, 
        -60.0, -50.0, -10.0, -65.0, dt, 
        tau_er, tau_ir, tau_Fr, args.tau_ed, args.tau_id, tau_Fd,
        spike_times, spike_ids, spike_count, t*dt, max_spikes, N_F,
        F_spikes
    )

torch.cuda.synchronize()
end = time.time()
total_spikes = int(spike_count.item())
print(f"Done in {end-start:.2f}s! Spikes: {total_spikes} | Rate: {total_spikes/N_total:.2f} Hz")

# --- 5. RASTER PLOT ---
if total_spikes > 0:
    times = spike_times[:total_spikes].cpu().numpy()
    ids = spike_ids[:total_spikes].cpu().numpy()
    mask = ids < 500
    times_in_seconds = times[mask] / 1000.0

    plt.figure(figsize=(12, 5))
    plt.plot(times_in_seconds, ids[mask], '|', markersize=1.5, color='black')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Neuron ID", fontsize=12)
    plt.title(f"Classical Balanced Network (Rate: {total_spikes/N_total:.2f} Hz)", fontsize=14)
    plt.xlim(0.5, 1.0) 
    plt.tight_layout()
    plt.savefig("paper_replica_raster.png", dpi=300)
    print("Saved exact paper replica plot to paper_replica_raster.png!")
