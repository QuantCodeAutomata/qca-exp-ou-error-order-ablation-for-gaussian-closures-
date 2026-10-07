"""
2PI effective-action gap-equation solver for the exp-OU model.

Implements the Hartree and melon levels of the 2PI skeleton expansion.
The gap equations are solved by damped fixed-point iteration.

Custom — Context7 found no library equivalent (paper 2PI formalism).

Reference: Blueprint paper, Section on 2PI Hartree and melon closures.

Gap equations:
  m    = -K @ D1         (mean equation)
  G^-1 = K^-1 + diag(D2) - Sigma_melon  (covariance equation)

Gaussian derivative moments (a=1):
  D0_k = c_k * exp(m_k + 0.5*G_kk)
  D1_k = D0_k * (m_k + G_kk)
  D2_k = D0_k * ((m_k + G_kk)^2 + G_kk)

where c_k = w * dt is the complex vertex coefficient at time step k.

Melon exchange diagram:
  Sigma_melon[i,j] = D1_i * D1_j * (exp(G_ij) - 1 - G_ij - 0.5*G_ij^2)
"""

import numpy as np
from typing import Literal


def _ou_covariance(t_grid: np.ndarray, gamma: float, theta: float) -> np.ndarray:
    """OU stationary covariance matrix K_{ij} = gamma * exp(-theta*|t_i - t_j|)."""
    diff = np.abs(t_grid[:, None] - t_grid[None, :])
    return gamma * np.exp(-theta * diff)


def _derivative_moments(
    m: np.ndarray, G: np.ndarray, c_vec: np.ndarray, mu: float
) -> tuple:
    """
    Compute Gaussian derivative moments D0, D1, D2 for the exp-OU vertex.

    The vertex at step k is  s_k(phi) = c_k * exp(mu + phi_k),  where phi_k is
    the centred log-vol field.  The Gaussian integrals under N(m_k, G_kk) give:

      D0_k = c_k * exp(mu + m_k + 0.5*G_kk)           [zeroth moment]
      D1_k = D0_k * (m_k + G_kk)                        [first moment]
      D2_k = D0_k * ((m_k + G_kk)^2 + G_kk)            [second moment]
    """
    Gdiag = np.diag(G)
    D0 = c_vec * np.exp(mu + m + 0.5 * Gdiag)
    D1 = D0 * (m + Gdiag)
    D2 = D0 * ((m + Gdiag) ** 2 + Gdiag)
    return D0, D1, D2


def two_pi_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 20,
    method: Literal["hartree", "melon"] = "hartree",
    max_iter: int = 400,
    damping: float = 0.2,
    tol: float = 1e-7,
    reg: float = 1e-10,
) -> complex:
    """
    Compute the exp-OU log-price CF via 2PI Hartree or melon gap equations.

    Parameters
    ----------
    u       : Fourier frequency
    theta   : OU mean-reversion speed
    mu      : OU long-run log-vol level
    sigma   : OU vol-of-vol
    T       : time to maturity
    N       : number of time steps (20 recommended for speed)
    method  : 'hartree' or 'melon'
    max_iter: maximum iteration count
    damping : fixed-point damping factor in (0, 1)
    tol     : convergence tolerance on (m, G) update norm
    reg     : regularisation added to K and G^-1 for numerical stability

    Returns
    -------
    phi : complex characteristic function value
    """
    gamma = sigma ** 2 / (2.0 * theta)

    # For u=0, phi(0)=1 exactly.
    if u == 0.0:
        return 1.0 + 0j

    # Free-field (stationary Gaussian) log-CF:
    #   log phi_free = -w * T * exp(mu + gamma/2)
    w = 0.5 * (1j * u + u ** 2)
    v_avg = np.exp(mu + 0.5 * gamma)          # E[v_t] under stationary Gaussian
    log_phi_free = -w * T * v_avg

    # For large u the CF is negligibly small (|phi| < 1e-10).
    # Return the free-field value directly to avoid overflow in the iteration.
    if abs(log_phi_free.real) > 25.0:         # |phi_free| < exp(-25) ≈ 1.4e-11
        return np.exp(log_phi_free)

    dt = T / N

    # Time grid (midpoints)
    t_grid = np.linspace(dt / 2.0, T - dt / 2.0, N)

    # OU covariance and precision matrices
    K = _ou_covariance(t_grid, gamma, theta)
    K_reg = K + reg * np.eye(N)
    Kinv = np.linalg.inv(K_reg)

    # Vertex coefficients: c_k = w * dt (mu is folded into exp in _derivative_moments)
    c_vec = w * dt * np.ones(N, dtype=complex)

    # Initialise: m=0 (centred field), G=K (free covariance)
    m = np.zeros(N, dtype=complex)
    G = K.astype(complex)

    for iteration in range(max_iter):
        with np.errstate(over='ignore', invalid='ignore'):
            D0, D1, D2 = _derivative_moments(m, G, c_vec, mu)

        # Guard against overflow/NaN in moments — fall back to free-field CF
        if not np.all(np.isfinite(D0)):
            return np.exp(log_phi_free)

        # Mean gap equation: m_new = -K @ D1
        m_new = -K_reg @ D1

        # Covariance gap equation: G^{-1} = K^{-1} + diag(D2) [- sigma_melon]
        Ginv_new = Kinv + np.diag(D2)

        if method == "melon":
            # Melon exchange self-energy:
            #   Sigma_ij = D1_i * D1_j * (exp(G_ij) - 1 - G_ij - 0.5*G_ij^2)
            # Clip real part to avoid exp overflow in the strongly-coupled regime.
            G_clip = np.clip(G.real, -15.0, 15.0) + 1j * G.imag
            with np.errstate(over='ignore'):
                exp_G = np.exp(G_clip)
            sigma_melon = np.outer(D1, D1) * (exp_G - 1.0 - G_clip - 0.5 * G_clip ** 2)
            Ginv_new = Ginv_new - sigma_melon

        # Solve for G_new
        try:
            G_new = np.linalg.inv(Ginv_new + reg * np.eye(N))
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(G_new)):
            break

        # Check convergence
        delta_m = np.max(np.abs(m_new - m))
        delta_G = np.max(np.abs(G_new - G))

        # Damped update
        m = (1.0 - damping) * m + damping * m_new
        G = (1.0 - damping) * G + damping * G_new

        if delta_m < tol and delta_G < tol:
            break

    # Re-compute moments at the converged saddle
    with np.errstate(over='ignore', invalid='ignore'):
        D0, D1, D2 = _derivative_moments(m, G, c_vec, mu)

    if not np.all(np.isfinite(D0)) or not np.all(np.isfinite(G)):
        return np.exp(log_phi_free)

    # ---------------------------------------------------------------------------
    # Log-CF from the 2PI effective action (Cornwall-Jackiw-Tomboulis):
    #
    #   Gamma[m,G] = 0.5 m^T K^{-1} m
    #              + 0.5 Tr(K^{-1} G)  -  0.5 log det(K^{-1} G)  -  N/2
    #              + sum_k D_k(0)
    #              - Phi_2[m, G]
    #
    # log phi = -Gamma[m(w), G(w)]  +  Gamma[0, K]
    #         = -Gamma[m(w), G(w)]       (since Gamma at u=0 is 0 by construction)
    #
    # Using:  -0.5 log det(K^{-1} G) = 0.5 (log det K - log det G)
    # ---------------------------------------------------------------------------
    log_phi = (
        - 0.5 * m @ Kinv @ m
        - 0.5 * np.trace(Kinv @ G)
        + 0.5 * N
        - np.sum(D0)
    )

    # Add  +0.5*(logdet G - logdet K)
    try:
        _, ldet_G = np.linalg.slogdet(G)
        _, ldet_K = np.linalg.slogdet(K_reg)
        log_phi += 0.5 * (ldet_G - ldet_K)
    except Exception:
        pass

    # Melon skeleton Phi_2 contributes with a + sign to log phi (= -Gamma):
    #   Phi_2 = -0.5 * sum_{i,j} D1_i D1_j (exp(G_ij) - 1 - G_ij - 0.5 G_ij^2)
    if method == "melon":
        G_clip = np.clip(G.real, -15.0, 15.0) + 1j * G.imag
        exp_G = np.exp(G_clip)
        phi2 = -0.5 * np.sum(
            np.outer(D1, D1) * (exp_G - 1.0 - G_clip - 0.5 * G_clip ** 2)
        )
        log_phi += phi2

    return np.exp(log_phi)


def two_pi_call_price(
    F: float,
    K: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    N: int = 20,
    method: Literal["hartree", "melon"] = "hartree",
    cos_L: float = 6.0,
    cos_N: int = 256,
) -> float:
    """
    Price a European call using the 2PI CF + COS inversion.

    Parameters
    ----------
    F, K    : forward and strike
    theta, mu, sigma, T : exp-OU model parameters
    N       : 2PI discretisation steps
    method  : 'hartree' or 'melon'
    cos_L   : COS truncation half-width
    cos_N   : COS cosine term count

    Returns
    -------
    call_price : float
    """
    from src.cos_pricer import cos_call_price

    def phi(u: float) -> complex:
        return two_pi_cf(u, theta, mu, sigma, T, N=N, method=method)

    return cos_call_price(phi, F, K, T, L=cos_L, N=cos_N)
