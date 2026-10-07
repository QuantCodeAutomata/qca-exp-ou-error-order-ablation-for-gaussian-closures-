"""
Model parameter containers for all stochastic-volatility models.

Provides dataclasses for:
  ExpOUParams         — exponential-OU (Stein-Stein with exp transform)
  SABRParams          — SABR (CEV forward + lognormal vol)
  RoughHestonParams   — rough Heston (fractional Riccati)
  RoughBergomiParams  — rough Bergomi (Riemann-Liouville variance field)
  RoughSABRParams     — rough SABR (rough vol + CEV forward)
"""

from dataclasses import dataclass, field


@dataclass
class ExpOUParams:
    """
    Exponential-OU stochastic-volatility model.

    dX_t = -theta*(X_t - mu)*dt + sigma*dW_t^X
    v_t  = exp(X_t)

    Attributes
    ----------
    theta : mean-reversion speed
    mu    : stationary log-vol level
    sigma : vol-of-vol (OU noise coefficient)
    rho   : spot–vol correlation
    T     : time to maturity
    """
    theta: float = 3.0
    mu: float = -2.0
    sigma: float = 1.0
    rho: float = 0.0
    T: float = 1.0

    @property
    def gamma(self) -> float:
        """Effective coupling: sigma^2 / (2*theta) = stationary OU variance."""
        return self.sigma ** 2 / (2.0 * self.theta)


@dataclass
class SABRParams:
    """
    SABR stochastic-volatility model.

    dF_t   = alpha_t * F_t^beta * dW_t^F
    dalpha = nu * alpha_t * dW_t^alpha
    corr(dW^F, dW^alpha) = rho

    Attributes
    ----------
    F0    : initial forward level
    alpha0: initial (SABR) vol
    beta  : CEV exponent in (0, 1]
    nu    : vol-of-vol
    rho   : spot–vol correlation
    T     : time to maturity
    """
    F0: float = 100.0
    alpha0: float = 0.2
    beta: float = 0.5
    nu: float = 0.4
    rho: float = 0.0
    T: float = 1.0


@dataclass
class RoughHestonParams:
    """
    Rough Heston model (El Euch & Rosenbaum 2019).

    Variance follows a rough kernel with Hurst index H < 0.5.

    Attributes
    ----------
    kappa : mean-reversion speed
    theta : long-run variance level
    lam   : vol-of-vol (lambda)
    H     : Hurst index (roughness); H=0.5 recovers standard Heston
    v0    : initial variance
    rho   : spot–vol correlation
    T     : time to maturity
    """
    kappa: float = 1.0
    theta: float = 0.04
    lam: float = 0.5
    H: float = 0.1
    v0: float = 0.04
    rho: float = 0.0
    T: float = 1.0


@dataclass
class RoughBergomiParams:
    """
    Rough Bergomi model (Bayer, Friz & Gatheral 2016).

    v_t = v0 * exp(eta*W_hat_t - 0.5*eta^2*t^{2H})
    where W_hat is a Riemann-Liouville fBm.

    Attributes
    ----------
    v0  : initial variance level
    eta : vol-of-vol scaling
    rho : spot–vol correlation
    H   : Hurst index (roughness)
    T   : time to maturity
    """
    v0: float = 0.04
    eta: float = 1.0
    rho: float = -0.4
    H: float = 0.1
    T: float = 1.0


@dataclass
class RoughSABRParams:
    """
    Rough SABR model.

    alpha_t = alpha0 * exp(eta*W_hat_t - 0.5*eta^2*t^{2H})
    dF_t    = alpha_t * F_t^beta * dW_t

    Attributes
    ----------
    F0    : initial forward
    alpha0: initial vol level
    beta  : CEV exponent
    eta   : vol-of-vol
    rho   : spot–vol correlation
    H     : Hurst index
    T     : time to maturity
    """
    F0: float = 100.0
    alpha0: float = 0.2
    beta: float = 0.5
    eta: float = 1.0
    rho: float = 0.0
    H: float = 0.1
    T: float = 1.0
