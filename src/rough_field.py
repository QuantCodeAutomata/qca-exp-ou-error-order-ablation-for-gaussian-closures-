"""
Rough Gaussian field: fractional Brownian motion covariance and sampling.

Provides:
  fbm_covariance        — exact fBm covariance kernel
  build_rough_covariance — PSD-regularised N×N covariance on uniform grid
  sample_rough_field     — Cholesky Monte Carlo sampler
  sample_rough_field_sobol — PCA-ordered randomised quasi-Monte Carlo sampler
"""

import numpy as np
from scipy.stats import norm


def fbm_covariance(t_grid, H):
    """
    Fractional Brownian motion covariance matrix.

    K_{ij} = 0.5 * (t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})

    Parameters
    ----------
    t_grid : 1-D array of time points (positive, increasing)
    H      : Hurst exponent in (0, 0.5)

    Returns
    -------
    K : (N, N) symmetric covariance matrix
    """
    ti = t_grid[:, None]
    tj = t_grid[None, :]
    K = 0.5 * (ti ** (2 * H) + tj ** (2 * H) - np.abs(ti - tj) ** (2 * H))
    return K


def build_rough_covariance(T, N, H):
    """
    Build an N×N fBm covariance matrix on a uniform grid in (0, T].

    Grid points are t_i = i*T/N for i=1..N (excluding 0).
    Applies eigenvalue flooring at 1e-12 to ensure positive semi-definiteness.

    Parameters
    ----------
    T : maturity
    N : number of time steps
    H : Hurst exponent

    Returns
    -------
    K      : (N, N) PSD covariance matrix
    t_grid : (N,) array of time points
    """
    t_grid = np.linspace(T / N, T, N)
    K = fbm_covariance(t_grid, H)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 1e-12)
    K_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return K_psd, t_grid


def sample_rough_field(K, n_paths, seed=42):
    """
    Sample paths of a mean-zero Gaussian field via Cholesky decomposition.

    Parameters
    ----------
    K       : (N, N) PSD covariance matrix
    n_paths : number of sample paths
    seed    : random seed

    Returns
    -------
    Y : (N, n_paths) array of field samples
    """
    N = K.shape[0]
    L = np.linalg.cholesky(K + 1e-10 * np.eye(N))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, n_paths))
    return L @ Z


def sample_rough_field_sobol(K, n_paths, n_scrambles=4):
    """
    Sample paths of a mean-zero Gaussian field via PCA-ordered randomised Sobol sequences.

    PCA ordering concentrates variance in the leading components, improving
    the effective dimensionality for QMC convergence on rough fields.

    Parameters
    ----------
    K          : (N, N) PSD covariance matrix
    n_paths    : number of Sobol points per scramble
    n_scrambles: number of independent scrambles (output is n_paths*n_scrambles paths)

    Returns
    -------
    Y : (N, n_paths * n_scrambles) array of field samples
    """
    from scipy.stats.qmc import Sobol

    N = K.shape[0]
    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals_sorted = np.maximum(eigvals[order], 1e-12)
    eigvecs_sorted = eigvecs[:, order]

    # PCA transform matrix: columns are sqrt-scaled eigenvectors
    A = eigvecs_sorted * np.sqrt(eigvals_sorted)  # (N, N)

    all_paths = []
    for scr in range(n_scrambles):
        sampler = Sobol(d=N, scramble=True, seed=scr * 17 + 42)
        u = sampler.random(n_paths)
        z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))  # (n_paths, N)
        Y = A @ z.T  # (N, n_paths)
        all_paths.append(Y)
    return np.concatenate(all_paths, axis=1)  # (N, n_paths * n_scrambles)
