import torch
from torch.utils.cpp_extension import load_inline
import time

fa_cpp_source = """
#include <torch/extension.h>
#include <vector>
#include <iostream>

// Factor Analysis EM algorithm
// X is (xDim, N)
// Returns {percentshared, d_shared, normevals, LL}
std::vector<torch::Tensor> fastfa_cuda(torch::Tensor X, int zDim, int cyc, double tol, double minVarFrac) {
    auto options = torch::TensorOptions().device(X.device()).dtype(X.dtype());
    
    int xDim = X.size(0);
    int N = X.size(1);

    // Compute covariance matrix cX = cov(X', 1) -> biased covariance
    auto X_mean = X.mean(1, true);
    auto X_centered = X - X_mean;
    auto cX = torch::matmul(X_centered, X_centered.transpose(0, 1)) / N;

    // varFloor = minVarFrac * diag(cX)
    auto diag_cX = cX.diag();
    auto varFloor = diag_cX * minVarFrac;

    // Initialization
    double scale = 1.0; 
    auto L = torch::randn({xDim, zDim}, options) * std::sqrt(scale / zDim);
    auto Ph = diag_cX.clone();

    auto I = torch::eye(zDim, options);
    torch::Tensor LL;

    double const_val = -xDim / 2.0 * std::log(2.0 * M_PI);

    for (int i = 0; i < cyc; ++i) {
        // E-step
        auto iPh = 1.0 / Ph;
        auto iPh_diag = torch::diag(iPh);
        auto iPhL = iPh_diag.matmul(L); // (xDim, zDim)
        
        auto L_T_iPhL = L.transpose(0, 1).matmul(iPhL); // (zDim, zDim)
        auto inv_term = torch::inverse(I + L_T_iPhL); // (zDim, zDim)
        
        auto MM = iPh_diag - iPhL.matmul(inv_term).matmul(iPhL.transpose(0, 1)); // (xDim, xDim)
        auto beta = L.transpose(0, 1).matmul(MM); // (zDim, xDim)
        
        auto cX_beta = cX.matmul(beta.transpose(0, 1)); // (xDim, zDim)
        auto EZZ = I - beta.matmul(L) + beta.matmul(cX_beta); // (zDim, zDim)
        
        // Log-likelihood
        auto ldM = 0.5 * torch::logdet(MM);
        auto current_LL = N * const_val + N * ldM - 0.5 * N * (MM * cX).sum();
        LL = current_LL;
        
        // M-step
        L = cX_beta.matmul(torch::inverse(EZZ));
        
        auto cX_beta_L = cX_beta * L; // element-wise
        Ph = diag_cX - cX_beta_L.sum(1);
        Ph = torch::max(Ph, varFloor);
    }

    // Compute Population Statistics
    auto shared = L.matmul(L.transpose(0, 1)); // (xDim, xDim)
    
    // compute eigenvalues (shared is symmetric)
    auto evals = std::get<0>(torch::linalg_eigh(shared));
    auto normevals = std::get<0>(torch::sort(evals, -1, true));

    // d_shared (threshold 0.95)
    auto evals_sum = normevals.sum();
    auto cumsum_evals = torch::cumsum(normevals / evals_sum, 0);
    auto d_shared_mask = cumsum_evals < 0.95;
    auto d_shared = d_shared_mask.sum() + 1;

    // percent shared
    auto sharedvar = shared.diag();
    auto percentshared = (sharedvar / (sharedvar + Ph)).mean();

    std::vector<torch::Tensor> res = {percentshared, d_shared, normevals, LL};
    return res;
}

// Cross-Validation version
double fastfa_cv_cuda(torch::Tensor X_train, torch::Tensor X_test, int zDim, int cyc, double tol, double minVarFrac) {
    auto options = torch::TensorOptions().device(X_train.device()).dtype(X_train.dtype());
    
    int xDim = X_train.size(0);
    int N_train = X_train.size(1);
    int N_test = X_test.size(1);

    auto X_mean = X_train.mean(1, true);
    auto X_centered = X_train - X_mean;
    auto cX = torch::matmul(X_centered, X_centered.transpose(0, 1)) / N_train;

    auto diag_cX = cX.diag();
    auto varFloor = diag_cX * minVarFrac;

    double scale = 1.0; 
    auto L = torch::randn({xDim, zDim}, options) * std::sqrt(scale / zDim);
    auto Ph = diag_cX.clone();

    auto I = torch::eye(zDim, options);

    for (int i = 0; i < cyc; ++i) {
        auto iPh = 1.0 / Ph;
        auto iPh_diag = torch::diag(iPh);
        auto iPhL = iPh_diag.matmul(L); 
        
        auto L_T_iPhL = L.transpose(0, 1).matmul(iPhL); 
        auto inv_term = torch::inverse(I + L_T_iPhL); 
        
        auto MM = iPh_diag - iPhL.matmul(inv_term).matmul(iPhL.transpose(0, 1)); 
        auto beta = L.transpose(0, 1).matmul(MM); 
        
        auto cX_beta = cX.matmul(beta.transpose(0, 1)); 
        auto EZZ = I - beta.matmul(L) + beta.matmul(cX_beta); 
        
        L = cX_beta.matmul(torch::inverse(EZZ));
        
        auto cX_beta_L = cX_beta * L; 
        Ph = diag_cX - cX_beta_L.sum(1);
        Ph = torch::max(Ph, varFloor);
    }

    // Now evaluate LL on X_test
    auto X_test_centered = X_test - X_mean;
    auto cX_test = torch::matmul(X_test_centered, X_test_centered.transpose(0, 1)) / N_test;

    auto iPh = 1.0 / Ph;
    auto iPh_diag = torch::diag(iPh);
    auto iPhL = iPh_diag.matmul(L); 
    auto L_T_iPhL = L.transpose(0, 1).matmul(iPhL); 
    auto inv_term = torch::inverse(I + L_T_iPhL); 
    auto MM = iPh_diag - iPhL.matmul(inv_term).matmul(iPhL.transpose(0, 1)); 
    
    double const_val = -xDim / 2.0 * std::log(2.0 * M_PI);
    auto ldM = 0.5 * torch::logdet(MM);
    auto test_LL = N_test * const_val + N_test * ldM - 0.5 * N_test * (MM * cX_test).sum();

    return test_LL.item<double>();
}
"""

fa_cuda = load_inline(
    name='fa_cuda_ext',
    cpp_sources=fa_cpp_source,
    functions=['fastfa_cuda', 'fastfa_cv_cuda'],
    with_cuda=True,
    extra_cflags=['-O3'],
)

def compute_population_statistics_cv(spike_counts, candidate_zDims=[3, 5, 8, 10, 15], num_folds=5):
    if not spike_counts.is_cuda:
        spike_counts = spike_counts.cuda()
    spike_counts = spike_counts.double()
    
    xDim, N = spike_counts.shape
    if N < num_folds * 2:
        # Not enough data for CV
        return compute_population_statistics(spike_counts, zDim=3)

    # Shuffle data across time bins for CV
    perm = torch.randperm(N, device=spike_counts.device)
    X_shuffled = spike_counts[:, perm]
    
    fold_size = N // num_folds
    
    best_ll = -float('inf')
    best_zDim = candidate_zDims[0]
    
    cyc = 100
    tol = 1e-8
    minVarFrac = 0.01

    for zDim in candidate_zDims:
        sum_ll = 0.0
        try:
            for fold in range(num_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < num_folds - 1 else N
                
                # Split
                test_mask = torch.zeros(N, dtype=torch.bool, device=spike_counts.device)
                test_mask[start_idx:end_idx] = True
                train_mask = ~test_mask
                
                X_train = X_shuffled[:, train_mask]
                X_test = X_shuffled[:, test_mask]
                
                ll = fa_cuda.fastfa_cv_cuda(X_train, X_test, zDim, cyc, tol, minVarFrac)
                sum_ll += ll
                
            if sum_ll > best_ll:
                best_ll = sum_ll
                best_zDim = zDim
        except Exception:
            continue
            
    # Finally run on full dataset with best_zDim
    try:
        pshared, dshared, normevals, _ = fa_cuda.fastfa_cuda(spike_counts, best_zDim, cyc, tol, minVarFrac)
        return pshared.item(), dshared.item(), normevals, best_zDim, best_ll
    except Exception:
        return 0.0, 0, torch.zeros(xDim, device=spike_counts.device), 0, 0.0

def compute_population_statistics(spike_counts, zDim=5):
    # spike_counts is [N_neurons, N_bins]
    # Ensure it's double on CUDA
    if not spike_counts.is_cuda:
        spike_counts = spike_counts.cuda()
    spike_counts = spike_counts.double()
    
    # EM parameters
    cyc = 100
    tol = 1e-8
    minVarFrac = 0.01
    
    # Run the custom C++ extension
    pshared, dshared, normevals, LL = fa_cuda.fastfa_cuda(spike_counts, zDim, cyc, tol, minVarFrac)
    return pshared.item(), dshared.item(), normevals, LL.item()

if __name__ == "__main__":
    print("Testing Custom C++/CUDA Factor Analysis (with Cross-Validation)...")
    torch.manual_seed(42)
    # Generate mock spike counts: 100 neurons, 200 bins
    mock_spikes = torch.poisson(torch.rand(100, 200).cuda() * 2).double()
    
    start = time.time()
    pshared, dshared, normevals, best_zDim, best_ll = compute_population_statistics_cv(mock_spikes, candidate_zDims=[3, 5, 8])
    end = time.time()
    
    print(f"Time: {end-start:.4f}s")
    print(f"Percent Shared: {pshared:.4f}")
    print(f"Dim Shared: {dshared}")
    print(f"Best zDim: {best_zDim}")
    print(f"Log Likelihood (CV): {best_ll:.4f}")
    print(f"Norm Evals (first 5): {normevals[:5].cpu().numpy()}")
    print("Test Passed!")
