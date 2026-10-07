"""
COS method (Fang & Oosterlee 2008) for option pricing via characteristic-function inversion.

The COS formula prices a European call by expanding the payoff in a cosine basis
over the truncated log-price domain [a, b].
"""

import numpy as np
from typing import Callable


def _chi_psi(a: float, b: float, c: float, d: float, k_arr: np.ndarray):
    """
    Compute COS payoff coefficients chi_k and psi_k for the integration interval [c, d].

    chi_k = Re[ integral_c^d e^x cos(k*pi*(x-a)/(b-a)) dx ]
    psi_k = integral_c^d cos(k*pi*(x-a)/(b-a)) dx
    """
    L = b - a
    kp = k_arr * np.pi / L  # k*pi/(b-a)

    # chi_k: see Fang & Oosterlee (2008) Appendix A
    # chi_0 = e^d - e^c
    # chi_k = [e^d*(cos(k*pi*(d-a)/L) + kp*sin(k*pi*(d-a)/L))
    #          - e^c*(cos(k*pi*(c-a)/L) + kp*sin(k*pi*(c-a)/L))] / (1 + kp^2)
    denom = 1.0 + kp ** 2
    denom[0] = 1.0  # avoid divide-by-zero for k=0

    chi = (
        np.exp(d) * (np.cos(kp * (d - a)) + kp * np.sin(kp * (d - a)))
        - np.exp(c) * (np.cos(kp * (c - a)) + kp * np.sin(kp * (c - a)))
    ) / denom
    chi[0] = np.exp(d) - np.exp(c)

    # psi_k: integral of cos term
    # psi_0 = d - c
    # psi_k = L/(k*pi) * [sin(k*pi*(d-a)/L) - sin(k*pi*(c-a)/L)]
    psi = np.zeros_like(chi)
    psi[0] = d - c
    psi[1:] = L / (k_arr[1:] * np.pi) * (
        np.sin(kp[1:] * (d - a)) - np.sin(kp[1:] * (c - a))
    )

    return chi, psi


def cos_call_price(
    phi: Callable[[float], complex],
    F: float,
    K: float,
    T: float,
    L: float = 6.0,
    N: int = 256,
    r: float = 0.0,
) -> float:
    """
    Price a European call using the COS method.

    Parameters
    ----------
    phi : characteristic function of log-price. phi(u) = E[exp(iu*log(S_T/F))]
    F   : forward price
    K   : strike
    T   : time to maturity
    L   : half-width of the truncation interval (domain = [-L, L] in log-moneyness)
    N   : number of cosine terms
    r   : risk-free rate (default 0)

    Returns
    -------
    call_price : float
    """
    a, b = -L, L
    x0 = np.log(K / F)  # log-moneyness (typically slightly negative for ATM)

    k_arr = np.arange(N, dtype=float)
    kp = k_arr * np.pi / (b - a)

    # Characteristic function evaluated at cosine frequencies
    # Fang-Oosterlee: phi_k = phi(k*pi/(b-a)) * exp(-ik*pi*a/(b-a))
    phi_vals = np.array([phi(kp[k]) * np.exp(-1j * kp[k] * a) for k in range(N)],
                        dtype=complex)
    phi_vals[0] *= 0.5  # half-weight for k=0

    # Payoff coefficients for call: integration over [0, b] (intrinsic positive region)
    c, d = max(a, x0), b          # call payoff region in log-price space
    chi, psi = _chi_psi(a, b, c, d, k_arr)

    # V_k = 2/(b-a) * (chi_k - K/F * psi_k)   [from Fang & Oosterlee eq. 29]
    V_k = 2.0 / (b - a) * (chi - K / F * psi)

    # COS pricing formula (Fang & Oosterlee 2008, Eq. 10)
    # At t=0, the "current log-price" relative to F is log(F/F)=0, so:
    # C = F * e^{-rT} * Re[sum_k phi(k*pi/L) * exp(-ik*pi*a/L) * V_k]
    # phi_vals already encodes the exp(-ik*pi*a/L) phase factor.
    price = F * np.real(np.sum(phi_vals * V_k)) * np.exp(-r * T)

    # Enforce no-arbitrage lower bound
    price = max(price, max(F * np.exp(-r * T) - K * np.exp(-r * T), 0.0))
    return float(price)


def cos_call_prices(
    phi: Callable[[float], complex],
    F: float,
    strikes: np.ndarray,
    T: float,
    L: float = 6.0,
    N: int = 256,
    r: float = 0.0,
) -> np.ndarray:
    """
    Price a vector of European calls using the COS method (single phi evaluation per freq).

    Parameters
    ----------
    phi     : characteristic function
    F       : forward price
    strikes : array of strike prices
    T       : time to maturity
    L       : truncation half-width
    N       : number of cosine terms
    r       : risk-free rate

    Returns
    -------
    prices : array of call prices (same shape as strikes)
    """
    a, b = -L, L
    k_arr = np.arange(N, dtype=float)
    kp = k_arr * np.pi / (b - a)

    # Evaluate CF at all frequencies once
    phi_vals = np.array([phi(kp[k]) * np.exp(-1j * kp[k] * a) for k in range(N)],
                        dtype=complex)
    phi_vals[0] *= 0.5

    prices = np.zeros(len(strikes))
    for i, K in enumerate(strikes):
        x0 = np.log(K / F)
        c, d = max(a, x0), b
        chi, psi = _chi_psi(a, b, c, d, k_arr)
        V_k = 2.0 / (b - a) * (chi - K / F * psi)
        price = F * np.real(np.sum(phi_vals * V_k)) * np.exp(-r * T)
        price = max(price, max((F - K) * np.exp(-r * T), 0.0))
        prices[i] = price

    return prices


def cos_call_price_batched(
    phi_batch: Callable[[np.ndarray], np.ndarray],
    F: float,
    K: float,
    T: float,
    L: float = 6.0,
    N: int = 128,
    r: float = 0.0,
) -> float:
    """
    Price a single European call using a BATCHED characteristic function.

    phi_batch(u_arr) must accept an array of frequencies and return a complex array.
    This avoids the overhead of N separate phi() calls.

    Parameters
    ----------
    phi_batch : callable(u_arr) -> complex ndarray
    F, K      : forward and strike
    T         : maturity
    L         : COS half-width
    N         : number of cosine terms
    r         : risk-free rate

    Returns
    -------
    call_price : float
    """
    a, b = -L, L
    k_arr = np.arange(N, dtype=float)
    kp = k_arr * np.pi / (b - a)

    # One batched CF call for all frequencies
    raw_phi = np.asarray(phi_batch(kp), dtype=complex)
    phi_vals = raw_phi * np.exp(-1j * kp * a)
    phi_vals[0] *= 0.5

    x0 = np.log(K / F)
    c, d = max(a, x0), b
    chi, psi = _chi_psi(a, b, c, d, k_arr)
    V_k = 2.0 / (b - a) * (chi - K / F * psi)
    price = F * float(np.real(np.dot(phi_vals, V_k))) * np.exp(-r * T)
    return max(price, max((F - K) * np.exp(-r * T), 0.0))


def cos_smile_from_cf_batch(
    phi_batch: Callable[[np.ndarray], np.ndarray],
    F: float,
    strikes: np.ndarray,
    T: float,
    L: float = 6.0,
    N: int = 128,
    r: float = 0.0,
) -> np.ndarray:
    """
    Compute call prices for a smile using a BATCHED characteristic function.

    Evaluates phi at all COS frequencies ONCE, then computes all strike prices.

    Parameters
    ----------
    phi_batch : callable(u_arr) -> complex ndarray
    F         : forward price
    strikes   : array of strikes
    T         : maturity
    L         : COS half-width
    N         : number of cosine terms
    r         : risk-free rate

    Returns
    -------
    prices : array of call prices
    """
    a, b = -L, L
    k_arr = np.arange(N, dtype=float)
    kp = k_arr * np.pi / (b - a)

    raw_phi = np.asarray(phi_batch(kp), dtype=complex)
    phi_vals = raw_phi * np.exp(-1j * kp * a)
    phi_vals[0] *= 0.5

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        x0 = np.log(K / F)
        c, d = max(a, x0), b
        chi, psi = _chi_psi(a, b, c, d, k_arr)
        V_k = 2.0 / (b - a) * (chi - K / F * psi)
        price = F * float(np.real(np.dot(phi_vals, V_k))) * np.exp(-r * T)
        prices[i] = max(price, max((F - K) * np.exp(-r * T), 0.0))

    return prices
