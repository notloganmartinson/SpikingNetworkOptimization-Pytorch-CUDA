import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

# --- THE UNIFIED CUDA BACKEND (SPARSE) ---
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

        float v = V[i];
        float c = cooldown[i];

        if (c <= 0.0f) {
            float dV = (-(v - E_L) / tau_m[i]) + 
                       (Delta_T[i] / tau_m[i]) * expf((v - V_T) / Delta_T[i]) + 
                       I_syn;
            // Deterministic update (No artificial noise)
            v += (dV * dt);
        }

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
    int N_E = 2500; 
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

snn_cuda = load_inline(
    name='snn_v12_sbn', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    functions=['step'], 
    with_cuda=True,
    extra_cflags=['-O3']
)

def build_sparse_connectivity(jee, jie, jei, jii, jef, jif, sigmaRR_E, sigmaRR_I, sigmaRX, device='cuda'):
    N_e, N_i, N_F = 2500, 625, 625
    N_total = N_e + N_i
    
    grid_E = int(np.sqrt(N_e))
    grid_I = int(np.sqrt(N_i))
    grid_F = int(np.sqrt(N_F))
    
    x_E, y_E = torch.meshgrid(torch.arange(grid_E, device=device).float() / grid_E, torch.arange(grid_E, device=device).float() / grid_E, indexing='ij')
    coords_E = torch.stack([x_E.flatten(), y_E.flatten()], dim=1)
    
    x_I, y_I = torch.meshgrid(torch.arange(grid_I, device=device).float() / grid_I, torch.arange(grid_I, device=device).float() / grid_I, indexing='ij')
    coords_I = torch.stack([x_I.flatten(), y_I.flatten()], dim=1)
    
    x_F, y_F = torch.meshgrid(torch.arange(grid_F, device=device).float() / grid_F, torch.arange(grid_F, device=device).float() / grid_F, indexing='ij')
    coords_F = torch.stack([x_F.flatten(), y_F.flatten()], dim=1)
    
    coords = torch.cat([coords_E, coords_I], dim=0)
    
    def periodic_dist_sq(c1, c2):
        diff = torch.abs(c1.unsqueeze(1) - c2.unsqueeze(0))
        diff = torch.min(diff, 1.0 - diff)
        return torch.sum(diff**2, dim=-1)
        
    dist_sq_recur = periodic_dist_sq(coords, coords)
    
    def calc_prob(p_mean, sigma, dist_sq):
        if sigma < 1e-5:
            return torch.zeros_like(dist_sq)
        p_base = p_mean / (2.0 * np.pi * sigma**2)
        P = p_base * torch.exp(-dist_sq / (2.0 * sigma**2))
        return torch.clamp(P, 0.0, 1.0)
    
    P_recur = torch.zeros((N_total, N_total), device=device)
    P_recur[:N_e, :N_e] = calc_prob(0.01, sigmaRR_E, dist_sq_recur[:N_e, :N_e])
    P_recur[:N_e, N_e:] = calc_prob(0.04, sigmaRR_E, dist_sq_recur[:N_e, N_e:])
    P_recur[N_e:, :N_e] = calc_prob(0.03, sigmaRR_I, dist_sq_recur[N_e:, :N_e])
    P_recur[N_e:, N_e:] = calc_prob(0.04, sigmaRR_I, dist_sq_recur[N_e:, N_e:])
    
    J_recur_dense = torch.zeros((N_total, N_total), device=device)
    J_recur_dense[:N_e, :N_e] = (torch.rand((N_e, N_e), device=device) < P_recur[:N_e, :N_e]).float() * jee
    J_recur_dense[:N_e, N_e:] = (torch.rand((N_e, N_i), device=device) < P_recur[:N_e, N_e:]).float() * jie
    J_recur_dense[N_e:, :N_e] = (torch.rand((N_i, N_e), device=device) < P_recur[N_e:, :N_e]).float() * jei
    J_recur_dense[N_e:, N_e:] = (torch.rand((N_i, N_i), device=device) < P_recur[N_e:, N_e:]).float() * jii

    J_recur_sparse = J_recur_dense.to_sparse_csr()
    J_recur_crow = J_recur_sparse.crow_indices().int()
    J_recur_col = J_recur_sparse.col_indices().int()
    J_recur_val = J_recur_sparse.values().float()
    del J_recur_dense, P_recur

    # Feedforward
    dist_sq_F = periodic_dist_sq(coords_F, coords)
    P_F = torch.zeros((N_F, N_total), device=device)
    P_F[:, :N_e] = calc_prob(0.1, sigmaRX, dist_sq_F[:, :N_e])
    P_F[:, N_e:] = calc_prob(0.05, sigmaRX, dist_sq_F[:, N_e:])
    
    J_F_dense = torch.zeros((N_F, N_total), device=device)
    J_F_dense[:, :N_e] = (torch.rand((N_F, N_e), device=device) < P_F[:, :N_e]).float() * jef
    J_F_dense[:, N_e:] = (torch.rand((N_F, N_i), device=device) < P_F[:, N_e:]).float() * jif

    J_F_sparse = J_F_dense.to_sparse_csr()
    J_F_crow = J_F_sparse.crow_indices().int()
    J_F_col = J_F_sparse.col_indices().int()
    J_F_val = J_F_sparse.values().float()
    del J_F_dense, P_F

    return J_recur_crow, J_recur_col, J_recur_val, J_F_crow, J_F_col, J_F_val

def run_simulation(tau_ed, tau_id, jee, jie, jei, jii, jef, jif, sigmaRR_E=0.1, sigmaRR_I=0.1, sigmaRX=0.05, t_steps=200000, device='cuda'):
    N_e, N_i, N_F = 2500, 625, 625
    N_total = N_e + N_i
    dt = 0.05

    J_recur_crow, J_recur_col, J_recur_val, J_F_crow, J_F_col, J_F_val = build_sparse_connectivity(
        jee, jie, jei, jii, jef, jif, sigmaRR_E, sigmaRR_I, sigmaRX, device
    )

    V = torch.empty(N_total, device=device).uniform_(-65, -50)
    cooldown = torch.zeros(N_total, device=device)
    g_E_rise = torch.zeros(N_total, device=device)
    g_E_decay = torch.zeros(N_total, device=device)
    g_I_rise = torch.zeros(N_total, device=device)
    g_I_decay = torch.zeros(N_total, device=device)
    g_F_rise = torch.zeros(N_total, device=device)
    g_F_decay = torch.zeros(N_total, device=device)

    max_spikes = 10_000_000
    spike_times = torch.zeros(max_spikes, device=device)
    spike_ids = torch.zeros(max_spikes, device=device, dtype=torch.int32)
    spike_count = torch.zeros(1, device=device, dtype=torch.int32)

    tau_m = torch.zeros(N_total, device=device)
    tau_m[:N_e] = 15.0
    tau_m[N_e:] = 10.0
    Delta_T = torch.zeros(N_total, device=device)
    Delta_T[:N_e] = 2.0
    Delta_T[N_e:] = 0.5
    tau_ref = torch.zeros(N_total, device=device)
    tau_ref[:N_e] = 1.5
    tau_ref[N_e:] = 0.5

    fire_prob = 10.0 * (dt / 1000.0) # 10Hz Poisson drive

    for t in range(t_steps):
        F_spikes = torch.rand(N_F, device=device) < fire_prob

        snn_cuda.step(
            V, cooldown, 
            g_E_rise, g_E_decay, g_I_rise, g_I_decay, g_F_rise, g_F_decay,
            J_recur_crow, J_recur_col, J_recur_val,
            J_F_crow, J_F_col, J_F_val,
            tau_m, Delta_T, tau_ref, 
            -60.0, -50.0, -10.0, -65.0, dt, 
            1.0, 1.0, 1.0, tau_ed, tau_id, 5.0,
            spike_times, spike_ids, spike_count, t*dt, max_spikes, N_F,
            F_spikes
        )

    torch.cuda.synchronize()
    total_spikes = int(spike_count.item())
    
    return spike_times[:total_spikes], spike_ids[:total_spikes], N_total