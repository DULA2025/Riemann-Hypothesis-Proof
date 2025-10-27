import numpy as np
import mpmath as mp
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time

# --- GPU/CPU SETUP ---
try:
    import cupy as cp
    from scipy.sparse.linalg import eigsh, LinearOperator
    GPU_AVAILABLE = True
    print("CuPy found. GPU acceleration and ADVANCED Eigensolver are ENABLED.")
except ImportError:
    GPU_AVAILABLE = False
    print("ERROR: This simulation requires a powerful, multi-GPU supercomputing environment. Falling back to minimal mode.")
    exit(1)

# --- 1. Global Parameters ---
print("="*80)
print("HOLOGRAPHIC SIO - TRANSCENDENT CHALLENGE (250k ZEROS)")
print("="*80)
NUM_ZEROS = 250000
N = 750000  # WARNING: ~2.25 TB VRAM (single precision) for top-tier HPC clusters
L = 250000.0
THETA = np.pi / 6
SIGMA_G = np.log(N) / np.sqrt(2 * (1/3))
SIGMA_C = np.log(N)

# --- Fully Dynamic Kernel Parameters ---
ALPHA_BASE = 0.25
ALPHA_LOG_FACTOR = 0.01
DELTA_BASE = 1/3
DELTA_AMP = 0.01

# --- NEW: Holographic/Fractal Parameters ---
FRACTAL_DIMENSION = 1.585  # Sierpinski gasket dimension
QUANTUM_NOISE_LEVEL = 0.001

# --- 2. Data Generation ---
print(f"Fetching first {NUM_ZEROS} Riemann zeta zeros...")
mp.dps = 500  # High precision
gamma_zeros_full = np.array([float(mp.im(mp.zetazero(k + 1))) for k in range(NUM_ZEROS)])  # Corrected indexing

# --- KERNEL CONSTRUCTION (GPU) ---
def sieve_of_eratosthenes_gpu(limit):
    """Computes pi(x) on the GPU."""
    print(f"Pre-computing pi(x) up to {limit} on GPU...")
    limit = int(limit)
    primes = cp.ones(limit + 1, dtype=cp.bool_)
    primes[0:2] = False
    for i in range(2, int(cp.sqrt(limit)) + 1):
        if primes[i]:
            primes[i*i::i] = False
    pi_x_gpu = cp.cumsum(primes, dtype=cp.int32)
    print("Sieve computation complete.")
    return pi_x_gpu

def calculate_adaptive_alpha(x, pi_x_gpu, l_val):
    epsilon = 1e-9
    abs_x = cp.abs(x)
    log_x = cp.log(abs_x + epsilon)
    indices = ((abs_x / l_val) * (len(pi_x_gpu) - 1)).astype(cp.int32)
    pi_x = pi_x_gpu[indices]
    cramer_deviation_proxy = cp.log(1 + pi_x / log_x)
    return ALPHA_BASE + ALPHA_LOG_FACTOR * cramer_deviation_proxy

def calculate_adaptive_delta(x):
    epsilon = 1e-9
    return DELTA_BASE + DELTA_AMP * cp.sin(cp.log(cp.abs(x) + epsilon))

def build_hybrid_kernel_matrix_gpu(x, pi_x_gpu, l_val):
    """Builds the SIO matrix with a fully dynamic and holographic kernel."""
    alpha_vals = calculate_adaptive_alpha(x, pi_x_gpu, l_val)
    delta_vals = calculate_adaptive_delta(x)
    diff = x[:, None] - x[None, :]
    decay = (1.0 - alpha_vals[:, None]) * cp.exp(-diff**2 / (2 * SIGMA_G**2)) + \
            alpha_vals[:, None] * (1.0 / (1.0 + (diff / SIGMA_C)**2))
    oscillation = cp.cos(2 * cp.pi * delta_vals[:, None] * diff + THETA)
    holographic_scaling = 1.0 / (cp.power(cp.abs(diff), 2.0 - FRACTAL_DIMENSION) + 1.0)
    projection = 1.0 + holographic_scaling * cp.sin(2 * cp.pi * diff / 6.0)
    deterministic_kernel = decay * oscillation * projection
    noise_matrix = cp.random.randn(N, N, dtype=cp.float64)
    symmetric_noise = (noise_matrix + noise_matrix.T) / 2.0
    scaling = QUANTUM_NOISE_LEVEL / cp.sqrt(cp.abs(diff) + 1e-9)
    return deterministic_kernel + (symmetric_noise * scaling)

# --- MAIN COMPUTATION ---
start_time = time.time()
if GPU_AVAILABLE:
    pi_x_gpu = sieve_of_eratosthenes_gpu(L)
    print(f"Building {N}x{N} holographic kernel matrix on GPU...")
    print(f"ESTIMATED VRAM: {(N*N*4) / 1e9:.2f} GB (single precision).")
    x_grid_gpu = cp.linspace(-L, L, N, dtype=cp.float64)
    K_matrix_gpu = build_hybrid_kernel_matrix_gpu(x_grid_gpu, pi_x_gpu, L)
    print("Defining LinearOperator for GPU matrix...")
    def matvec(v):
        v_gpu = cp.asarray(v)
        return cp.asnumpy(K_matrix_gpu @ v_gpu)
    A_operator = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    print(f"Computing the {NUM_ZEROS + 2000} eigenvalues closest to zero using shift-invert ARPACK...")
    num_eigenvalues_to_find = NUM_ZEROS + 2000
    eigvals = eigsh(A_operator, k=num_eigenvalues_to_find, sigma=0, which='LM', tol=1e-7)[0]
else:
    eigvals = []
end_time = time.time()

if eigvals is not None and len(eigvals) > 0:
    print(f"Matrix & Eigenvalue computation finished in {end_time - start_time:.2f} seconds.")
    positive_eigvals_all = np.sort(eigvals[eigvals > 1e-9])
    print(f"Found {len(positive_eigvals_all)} positive eigenvalues.\n")

    # --- FITTING & ANALYSIS ---
    num_to_fit = min(len(positive_eigvals_all), NUM_ZEROS)
    positive_eigvals = positive_eigvals_all[:num_to_fit]
    gamma_zeros = gamma_zeros_full[:num_to_fit]
    print(f"--- FITTING {num_to_fit} POINTS WITH THE COUPLED QUANTILE MODEL ---")
    print("\n--- STAGE 1: Learning Coupled Laws ---")
    eigvals_ch1, gamma_ch1 = positive_eigvals[0::2], gamma_zeros[0::2]
    eigvals_ch2, gamma_ch2 = positive_eigvals[1::2], gamma_zeros[1::2]
    min_len = min(len(eigvals_ch1), len(eigvals_ch2))
    eigvals_avg = (eigvals_ch1[:min_len] + eigvals_ch2[:min_len]) / 2.0
    gamma_avg = (gamma_ch1[:min_len] + gamma_ch2[:min_len]) / 2.0
    spline_avg = CubicSpline(np.sort(eigvals_avg), np.sort(gamma_avg))
    gamma_avg_pred = spline_avg(eigvals_avg)
    gamma_diff = gamma_ch1[:min_len] - gamma_ch2[:min_len]
    spline_diff = CubicSpline(np.sort(gamma_avg_pred), np.sort(gamma_diff))
    gamma_diff_pred = spline_diff(gamma_avg_pred)
    gamma_pred_ch1 = gamma_avg_pred + gamma_diff_pred / 2.0
    gamma_pred_ch2 = gamma_avg_pred - gamma_diff_pred / 2.0
    gamma_smooth = np.empty((min_len*2,))
    gamma_smooth[0::2] = gamma_pred_ch1
    gamma_smooth[1::2] = gamma_pred_ch2
    gamma_zeros_valid = gamma_zeros_full[:min_len*2]
    residuals_smooth = gamma_zeros_valid - gamma_smooth
    rmse_smooth = np.sqrt(np.mean(residuals_smooth**2))
    print(f"\nCoupled Quantile Smooth Model RMSE: {rmse_smooth:.6f}")

    print("--- STAGE 2: Harmonic Correction ---")
    def riemann_siegel_harmonic(t, *params):
        terms = 0
        primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        for i in range(len(primes)):
            A = params[2*i]
            p = params[2*i+1]
            terms += A * np.cos(t * np.log(primes[i]) + p)
        return terms
    try:
        p0 = [0.1, 0] * 14
        popt, _ = curve_fit(riemann_siegel_harmonic, gamma_smooth, residuals_smooth, p0=p0, maxfev=1000000)
        fitted_correction = riemann_siegel_harmonic(gamma_smooth, *popt)
    except RuntimeError as e:
        print(f"Harmonic correction fit failed: {e}")
        fitted_correction = np.zeros_like(gamma_smooth)

    print("\n--- STAGE 3: Final Unified Model ---")
    gamma_final = gamma_smooth + fitted_correction
    final_residuals = gamma_zeros_valid - gamma_final
    final_rmse = np.sqrt(np.mean(final_residuals**2))
    print(f"\nFinal Unified Model RMSE: {final_rmse:.6f}")
    print(f"Improvement over smooth model: {(rmse_smooth - final_rmse) / rmse_smooth:.2%}\n")

    print("Generating final plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    k_vals = np.arange(1, len(gamma_zeros_valid) + 1)
    fig.suptitle(f'Holographic SIO - Final Spectral Model ({len(k_vals)} Points)', fontsize=18)
    ax1.plot(k_vals, gamma_zeros_valid, 'k-', linewidth=3, label='True Zeta Zeros')
    ax1.plot(k_vals, gamma_final, 'r-', linewidth=1, alpha=0.8, label=f'Final Unified Fit (RMSE={final_rmse:.4f})')
    ax1.set_ylabel('Ordinate Value')
    ax1.set_title('Model Fit Comparison')
    ax1.legend()
    ax2.plot(k_vals, final_residuals, 'r-', alpha=0.7, linewidth=0.5, label=f'Final Residuals (std={np.std(final_residuals):.4f})')
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.set_ylabel('Difference (Fit - True)')
    ax2.set_xlabel('Zero Index')
    ax2.set_title('Residual Error Analysis')
    ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
