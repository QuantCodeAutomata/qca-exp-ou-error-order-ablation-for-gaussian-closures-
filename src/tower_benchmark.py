"""
Exact tower recursion benchmark for the exp-OU stochastic-volatility model.

The characteristic function phi(u) = E[exp(-w(u) * integral_0^T exp(X_t) dt)] is
computed by backward recursion on a dense equally-spaced spatial grid using
scipy.ndimage.map_coordinates (cubic spline) for interpolation.

Key design choices
------------------
* Dense spatial grid (n_grid ≈ 512): separates spatial accuracy from temporal step
  size so that v_dt << grid-spacing² / 12 (linear interpolation failure mode) is
  never reached even at N → 400.  Cubic interpolation error is O(h⁴).
* Batch multi-frequency evaluation: all COS cosine-term frequencies are propagated
  simultaneously through a single (n_freq, n_grid) state array, avoiding repeated
  tower runs.  The frequency-independent interpolation coordinates are precomputed.
* Right-endpoint Riemann rule: vertex exp(-w*dt*exp(X')) applied at the next state.

Reference: Blueprint paper, Section on benchmark conventions.
  - 2PI methods benchmarked against N=100-step tower.
  - SCG / continuous-time methods benchmarked against N=400-step tower.
"""

import numpy as np
import scipy.sparse
from scipy.ndimage import map_coordinates
from typing import Optional

# ---------------------------------------------------------------------------
# GH quadrature for conditional expectation and for stationary averaging
# ---------------------------------------------------------------------------
_GH_N = 32
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(_GH_N)

_GH_N_STAT = 64
_GH_NODES_STAT, _GH_WEIGHTS_STAT = np.polynomial.hermite.hermgauss(_GH_N_STAT)


def _catmullrom_weights(t: float) -> np.ndarray:
    """Return the four Catmull-Rom weights for fractional offset t in [0,1)."""
    return np.array([
        0.5 * (-t**3 + 2*t**2 - t),
        0.5 * (3*t**3 - 5*t**2 + 2),
        0.5 * (-3*t**3 + 4*t**2 + t),
        0.5 * (t**3 - t**2),
    ])


def _build_catmullrom_matrix(
    xi_flat: np.ndarray, n_grid: int
) -> scipy.sparse.csr_matrix:
    """
    Build sparse Catmull-Rom interpolation matrix M of shape (len(xi_flat), n_grid).

    Vectorised construction: 4 nonzeros per row, boundary-clamped ('nearest').
    """
    xq = np.clip(xi_flat, 0.0, float(n_grid - 1))
    i_arr = np.floor(xq).astype(int)
    t = xq - i_arr                                                    # (nq,)

    # Catmull-Rom basis for l in {-1, 0, 1, 2} ↔ l_offset in {0,1,2,3}
    t2, t3 = t * t, t * t * t
    w0 =  0.5 * (-t3 + 2*t2 - t)
    w1 =  0.5 * ( 3*t3 - 5*t2 + 2)
    w2 =  0.5 * (-3*t3 + 4*t2 + t)
    w3 =  0.5 * (  t3 -   t2)

    nq = len(xq)
    row_idx = np.tile(np.arange(nq), 4)
    col_idx = np.concatenate([
        np.clip(i_arr - 1, 0, n_grid - 1),
        np.clip(i_arr,     0, n_grid - 1),
        np.clip(i_arr + 1, 0, n_grid - 1),
        np.clip(i_arr + 2, 0, n_grid - 1),
    ])
    data = np.concatenate([w0, w1, w2, w3])

    return scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(nq, n_grid), dtype=np.float64
    )


def _tower_run_batch(
    u_arr: np.ndarray,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    n_grid: int,
    x_halfwidth: float,
) -> np.ndarray:
    """
    Core batch tower solver. Returns phi(u) for all u in u_arr simultaneously.

    Uses a precomputed sparse Catmull-Rom interpolation matrix to replace
    per-frequency map_coordinates calls with two dense matmuls per step:
        psi_at_xp.real = (M @ psi.real.T).T
        psi_at_xp.imag = (M @ psi.imag.T).T

    Parameters
    ----------
    u_arr      : (n_freq,) array of Fourier frequencies
    theta, mu, sigma, T, N : model / discretisation parameters
    n_grid     : spatial grid size
    x_halfwidth: grid half-width in units of sqrt(gamma)

    Returns
    -------
    phi_arr : (n_freq,) complex array
    """
    n_freq = len(u_arr)
    dt = T / N
    gamma = sigma ** 2 / (2.0 * theta)
    w_arr = 0.5 * (1j * u_arr + u_arr ** 2)   # (n_freq,)

    # OU exact discrete transition parameters
    m_factor = np.exp(-theta * dt)
    v_dt = gamma * (1.0 - np.exp(-2.0 * theta * dt))
    sqrt2v_dt = np.sqrt(2.0 * v_dt)

    # Dense equally-spaced spatial grid on [mu-hw*sigma_stat, mu+hw*sigma_stat]
    sigma_stat = np.sqrt(gamma)
    x_lo = mu - x_halfwidth * sigma_stat
    x_hi = mu + x_halfwidth * sigma_stat
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    dx = x_grid[1] - x_grid[0]

    # ── Precompute GH query coordinates in grid-index units ──────────────────
    mu_conds = mu + m_factor * (x_grid - mu)                        # (n_grid,)
    x_prime = mu_conds[:, None] + sqrt2v_dt * _GH_NODES[None, :]   # (n_grid, n_GH)
    xi_prime = (x_prime - x_lo) / dx                                # fractional indices
    xi_flat = xi_prime.ravel()                                       # (n_grid*n_GH,)

    # ── Build sparse interpolation matrix (precomputed once) ─────────────────
    M_interp = _build_catmullrom_matrix(xi_flat, n_grid)             # (n_grid*n_GH, n_grid)

    # ── Precompute vertex (frequency-specific, time-independent) ─────────────
    EX = np.exp(x_prime)                                             # (n_grid, n_GH)
    # vertex[k, i, j] = exp(-w_k * dt * exp(x_prime[i,j]))
    vertex = np.exp(-dt * w_arr[:, None, None] * EX[None, :, :])    # (n_freq, n_grid, n_GH)
    # Pre-weight by GH: vertex_w[k, i, j] = vertex[k,i,j] * gh_w[j]
    gh_w = _GH_WEIGHTS / np.sqrt(np.pi)                              # (n_GH,)
    vertex_w = vertex * gh_w[None, None, :]                          # (n_freq, n_grid, n_GH)

    # ── Stationary averaging nodes ────────────────────────────────────────────
    x_stat_avg = mu + np.sqrt(2.0 * gamma) * _GH_NODES_STAT         # (n_GH_stat,)
    xi_stat_avg = (x_stat_avg - x_lo) / dx
    M_stat = _build_catmullrom_matrix(xi_stat_avg, n_grid)           # (n_GH_stat, n_grid)

    # ── Initialise psi (n_freq, n_grid) = 1 ──────────────────────────────────
    psi = np.ones((n_freq, n_grid), dtype=complex)

    # ── Backward recursion using sparse matmul ────────────────────────────────
    for _ in range(N):
        # Interpolate all frequencies at once:
        # psi.T shape: (n_grid, n_freq)
        # M_interp @ psi.T: (n_grid*n_GH, n_freq)
        psi_T = psi.T                                                 # (n_grid, n_freq)
        interp_re = M_interp @ psi_T.real                             # (n_grid*n_GH, n_freq)
        interp_im = M_interp @ psi_T.imag
        # psi_at_xp[k, i, j] = psi_interp at x_prime[i,j] for frequency k
        # shape needed: (n_freq, n_grid, n_GH)
        psi_at_xp_re = interp_re.T.reshape(n_freq, n_grid, _GH_N)   # (n_freq, n_grid, n_GH)
        psi_at_xp_im = interp_im.T.reshape(n_freq, n_grid, _GH_N)
        psi_at_xp = psi_at_xp_re + 1j * psi_at_xp_im

        # GH quadrature: psi_new[k, i] = sum_j vertex_w[k,i,j] * psi_at_xp[k,i,j]
        psi = np.einsum("kij,kij->ki", vertex_w, psi_at_xp)         # (n_freq, n_grid)

    # ── Final expectation over stationary distribution ────────────────────────
    # M_stat: (n_GH_stat, n_grid), psi.T: (n_grid, n_freq)
    psi_stat_re = M_stat @ psi.T.real                                # (n_GH_stat, n_freq)
    psi_stat_im = M_stat @ psi.T.imag
    psi_stat = psi_stat_re + 1j * psi_stat_im                        # (n_GH_stat, n_freq)
    gh_stat_w = _GH_WEIGHTS_STAT / np.sqrt(np.pi)
    phi_arr = gh_stat_w @ psi_stat                                    # (n_freq,)

    return phi_arr


def tower_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 100,
    n_grid: int = 128,
    x_halfwidth: float = 6.0,
) -> complex:
    """
    Single-frequency tower CF evaluation.

    Parameters
    ----------
    u : Fourier frequency
    theta, mu, sigma, T : exp-OU parameters
    N : time steps  (100 for 2PI benchmark, 400 for SCG benchmark)
    n_grid     : spatial grid points
    x_halfwidth: grid half-width in units of sqrt(gamma)

    Returns
    -------
    phi : complex CF value
    """
    result = _tower_run_batch(
        np.array([float(u)]), theta, mu, sigma, T, N, n_grid, x_halfwidth
    )
    return complex(result[0])


def tower_cf_vec(
    u_arr: np.ndarray,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 100,
    n_grid: int = 128,
    x_halfwidth: float = 6.0,
) -> np.ndarray:
    """
    Batch tower CF: all frequencies in a single backward pass.

    Parameters
    ----------
    u_arr : (n_freq,) array of Fourier frequencies
    theta, mu, sigma, T, N, n_grid, x_halfwidth : same as tower_cf

    Returns
    -------
    phi_arr : (n_freq,) complex array
    """
    return _tower_run_batch(
        np.asarray(u_arr, dtype=float), theta, mu, sigma, T, N, n_grid, x_halfwidth
    )


def tower_call_price(
    F: float,
    K: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 100,
    cos_L: float = 6.0,
    cos_N: int = 128,
) -> float:
    """
    Price a European call using the tower benchmark + COS inversion.

    Evaluates all COS frequencies in a single batched tower run.

    Parameters
    ----------
    F, K     : forward and strike
    theta, mu, sigma, T : exp-OU model parameters
    N        : tower time steps
    cos_L    : COS half-width (sets integration range [log(K/F)-L, log(K/F)+L])
    cos_N    : number of COS cosine terms

    Returns
    -------
    call_price : float
    """
    from src.cos_pricer import cos_call_price_batched
    return cos_call_price_batched(
        lambda u_arr: tower_cf_vec(u_arr, theta, mu, sigma, T, N),
        F, K, T, L=cos_L, N=cos_N,
    )


def tower_smile(
    F: float,
    strikes: np.ndarray,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 100,
    cos_L: float = 6.0,
    cos_N: int = 128,
) -> np.ndarray:
    """
    Compute a full implied-vol smile via tower benchmark.

    Runs the tower ONCE for all COS frequencies, then evaluates all strikes.

    Parameters
    ----------
    F, strikes : forward price and array of strikes
    theta, mu, sigma, T : model parameters
    N, cos_L, cos_N : tower and COS settings

    Returns
    -------
    ivols : array of Black implied vols (NaN for failures)
    """
    from src.cos_pricer import cos_smile_from_cf_batch
    from src.metrics import implied_vol

    cf_batch = lambda u_arr: tower_cf_vec(u_arr, theta, mu, sigma, T, N)
    prices = cos_smile_from_cf_batch(cf_batch, F, strikes, T, L=cos_L, N=cos_N)
    return np.array([implied_vol(p, F, K, T) for p, K in zip(prices, strikes)])


def tower_implied_vol(
    F: float,
    K: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 100,
) -> float:
    """
    Convenience wrapper: tower benchmark implied vol for one strike.

    Returns NaN on failure.
    """
    from src.metrics import implied_vol
    price = tower_call_price(F, K, theta, mu, sigma, T, N)
    return implied_vol(price, F, K, T)
