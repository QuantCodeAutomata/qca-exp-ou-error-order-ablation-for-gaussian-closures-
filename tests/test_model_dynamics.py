"""
Tests for model parameter dataclasses (src/model_dynamics.py).
"""

import pytest
from src.model_dynamics import (
    ExpOUParams,
    SABRParams,
    RoughHestonParams,
    RoughBergomiParams,
    RoughSABRParams,
)


def test_expou_defaults():
    p = ExpOUParams()
    assert p.theta == 3.0
    assert p.mu == -2.0
    assert p.sigma == 1.0
    assert p.rho == 0.0
    assert p.T == 1.0


def test_expou_gamma_property():
    p = ExpOUParams(theta=3.0, sigma=1.0)
    expected = 1.0 ** 2 / (2.0 * 3.0)
    assert abs(p.gamma - expected) < 1e-12


def test_expou_gamma_increases_with_sigma():
    p1 = ExpOUParams(theta=3.0, sigma=1.0)
    p2 = ExpOUParams(theta=3.0, sigma=2.0)
    assert p2.gamma > p1.gamma


def test_expou_custom_params():
    p = ExpOUParams(theta=1.0, mu=-3.0, sigma=1.8, rho=-0.5, T=2.0)
    assert p.theta == 1.0
    assert p.mu == -3.0
    assert p.T == 2.0


def test_sabr_defaults():
    p = SABRParams()
    assert p.F0 == 100.0
    assert p.beta == 0.5
    assert p.nu == 0.4
    assert p.rho == 0.0


def test_rough_heston_defaults():
    p = RoughHestonParams()
    assert p.H == 0.1
    assert p.v0 == 0.04
    assert 0.0 < p.H < 0.5, "Rough Heston should have H < 0.5"


def test_rough_bergomi_defaults():
    p = RoughBergomiParams()
    assert p.eta == 1.0
    assert p.v0 == 0.04
    assert p.H < 0.5


def test_rough_sabr_defaults():
    p = RoughSABRParams()
    assert p.F0 == 100.0
    assert 0.0 < p.beta <= 1.0


def test_all_models_instantiable():
    """All model containers should be instantiable with default params."""
    models = [ExpOUParams(), SABRParams(), RoughHestonParams(),
              RoughBergomiParams(), RoughSABRParams()]
    for m in models:
        assert m is not None
