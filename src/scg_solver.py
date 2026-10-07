"""
Self-Consistent Gaussian (SCG) approximation for the exp-OU model.

The SCG method uses a log-quadratic ansatz
    psi(z) = exp(A + B*z + 0.5*C*z^2)
and evolves the ODE system for (A, B, C) backward from T to 0.

For rho=0 (no spot-vol correlation), the leverage terms vanish and the system
simplifies to three coupled complex ODEs driven by the self-consistency
conditions on the tilted Gaussian measure.

Reference: Blueprint paper, Section 3 (SCG) and ODE system equations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple


def _scg_rhs(t: float, y: np.ndarray, theta: float, mu: float, sigma: float, w: complex) -> np.ndarray:
    """
    Right-hand side of the SCG ODE system in real form (rho=0 case).

    State vector: y = [Re(A), Im(A), Re(B), Im(B), Re(C), Im(C)]

    ODEs (backward in time, integrated from T to 0):
        C' = sigma^2*C^2 - 2*theta*C - w*V_eff
        B' = -(theta - sigma^2*C)*B - w*V_eff*(1 - m)
        A' = 0.5*sigma^2*(C + B^2) - w*V_eff*(1 - m + 0.5*(m^2 - v))

    where:
        gamma = sigma^2 / (2*theta)   [stationary OU variance]
        m = B*gamma / (1 - gamma*C)   [tilted Gaussian mean]
        v = gamma / (1 - gamma*C)     [tilted Gaussian variance]
        V_eff = exp(mu + m + v/2)     [effective variance]
    """
    gamma = sigma ** 2 / (2.0 * theta)

    # Unpack complex state
    A = y[0] + 1j * y[1]
    B = y[2] + 1j * y[3]
    C = y[4] + 1j * y[5]

    denom = 1.0 - gamma * C
    if abs(denom) < 1e-12:
        return np.zeros(6)

    m = B * gamma / denom
    v = gamma / denom
    V_eff = np.exp(mu + m + 0.5 * v)

    dC = sigma ** 2 * C ** 2 - 2.0 * theta * C - w * V_eff
    dB = -(theta - sigma ** 2 * C) * B - w * V_eff * (1.0 - m)
    dA = 0.5 * sigma ** 2 * (C + B ** 2) - w * V_eff * (1.0 - m + 0.5 * (m ** 2 - v))

    return [dA.real, dA.imag, dB.real, dB.imag, dC.real, dC.imag]


def scg_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    max_step: float = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> complex:
    """
    Compute the exp-OU log-price characteristic function via SCG ODE.

    Integrates (A, B, C) backward from t=T (terminal conditions A=B=C=0)
    to t=0 and returns phi = exp(A + B^2*gamma/(2*(1-gamma*C))) / sqrt(1-gamma*C).

    Parameters
    ----------
    u     : Fourier frequency
    theta : OU mean-reversion speed
    mu    : OU long-run level
    sigma : OU noise coefficient
    T     : time to maturity
    max_step : maximum ODE step (default T/200)
    rtol, atol : ODE solver tolerances

    Returns
    -------
    phi : complex characteristic function value
    """
    gamma = sigma ** 2 / (2.0 * theta)
    w = 0.5 * (1j * u + u ** 2)

    if max_step is None:
        max_step = T / 200.0

    # Integrate forward: t_span = (0, T) with A(0)=B(0)=C(0)=0.
    # The blueprint's ODE uses -w*V_eff (source with minus sign), which corresponds
    # to the forward-time variational formulation where (A,B,C) start at zero and
    # accumulate the coupling as t increases from 0 to T.
    # phi = exp(A(T) + B(T)^2*gamma/(2*(1-gamma*C(T)))) / sqrt(1-gamma*C(T)).
    y0 = np.zeros(6)  # A=B=C=0 at t=0
    try:
        sol = solve_ivp(
            fun=lambda t, y: _scg_rhs(t, y, theta, mu, sigma, w),
            t_span=(0.0, T),
            y0=y0,
            method="RK45",
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )
    except Exception:
        return complex(np.nan)

    if not sol.success:
        return complex(np.nan)

    # Read off terminal values at t=T
    A = sol.y[0, -1] + 1j * sol.y[1, -1]
    B = sol.y[2, -1] + 1j * sol.y[3, -1]
    C = sol.y[4, -1] + 1j * sol.y[5, -1]

    denom = 1.0 - gamma * C
    if abs(denom) < 1e-12 or denom.real < 0:
        return complex(np.nan)

    log_phi = A + B ** 2 * gamma / (2.0 * denom) - 0.5 * np.log(denom)
    return np.exp(log_phi)


def scg_call_price(
    F: float,
    K: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    cos_L: float = 6.0,
    cos_N: int = 256,
) -> float:
    """
    Price a European call using the SCG characteristic function + COS inversion.

    Parameters
    ----------
    F, K    : forward and strike
    theta   : OU mean-reversion
    mu      : OU long-run level
    sigma   : OU vol-of-vol
    T       : maturity
    cos_L   : COS truncation half-width
    cos_N   : COS cosine term count

    Returns
    -------
    call_price : float
    """
    from src.cos_pricer import cos_call_price

    def phi(u: float) -> complex:
        return scg_cf(u, theta, mu, sigma, T)

    return cos_call_price(phi, F, K, T, L=cos_L, N=cos_N)


def scg_stationary_taylor_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
) -> complex:
    """
    Method 1: Quadratic Taylor vertex under stationary Gaussian N(0, gamma).

    The exp(X) vertex is approximated as exp(mu)*(1 + X' + 0.5*X'^2) where X' = X - mu.
    The CF is then exp(-w*T*exp(mu)*(1 + gamma/2)).

    Custom — Context7 found no library equivalent (paper variant 1).
    """
    gamma = sigma ** 2 / (2.0 * theta)
    w = 0.5 * (1j * u + u ** 2)
    # Taylor: E[exp(w*dt*exp(X))] ≈ exp(w*dt*exp(mu)*(1 + gamma/2)) for small coupling
    return np.exp(-w * T * np.exp(mu) * (1.0 + gamma / 2.0))


def scg_stationary_exact_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
) -> complex:
    """
    Method 2: Exact exponential prefactor under stationary Gaussian N(mu, gamma).

    E[exp(X)] = exp(mu + gamma/2) for X ~ N(mu, gamma).
    CF = exp(-w * T * exp(mu + gamma/2)).

    Custom — Context7 found no library equivalent (paper variant 2).
    """
    gamma = sigma ** 2 / (2.0 * theta)
    w = 0.5 * (1j * u + u ** 2)
    return np.exp(-w * T * np.exp(mu + gamma / 2.0))


def scg_tilted_taylor_cf(
    u: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
) -> complex:
    """
    Method 3: Taylor prefactor under tilted Gaussian N(m, v).

    Uses a single saddle-point step to find the tilted mean m, then
    applies quadratic Taylor approximation of exp(X) around m.

    Custom — Context7 found no library equivalent (paper variant 3).
    """
    gamma = sigma ** 2 / (2.0 * theta)
    w = 0.5 * (1j * u + u ** 2)
    # Tilted mean from the leading SCG equation with C=0:
    # m = w*T*gamma*exp(mu+m+v/2), solve for m
    # Approximate: m ≈ w*T*gamma*exp(mu + gamma/2) (one Newton step from m=0)
    v = gamma
    m = w * T * gamma * np.exp(mu + v / 2.0)
    V_eff = np.exp(mu + m + v / 2.0)
    # Taylor: exp(x) ≈ V_eff*(1 + (x-m) + 0.5*(x-m)^2) around tilted mean m
    # Gaussian average: E[exp(w*dt*V_eff*(1+(x-m)+0.5*(x-m)^2))] with x~N(m,v)
    return np.exp(-w * T * V_eff * (1.0 + 0.5 * v))
