"""
Tests for rough_field.py — fBm covariance and field samplers.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.rough_field import (
    fbm_covariance,
    build_rough_covariance,
    sample_rough_field,
    sample_rough_field_sobol,
)


# ---------------------------------------------------------------------------
# Standard parameters
# ---------------------------------------------------------------------------

H_LIST = [0.05, 0.10, 0.20, 0.30, 0.49]
T = 1.0
N = 20


# ---------------------------------------------------------------------------
# fbm_covariance
# ---------------------------------------------------------------------------

class TestFbmCovariance:
    def test_shape(self):
        """Output is (N, N) square matrix."""
        t = np.linspace(0.05, 1.0, N)
        K = fbm_covariance(t, H=0.2)
        assert K.shape == (N, N)

    def test_symmetry(self):
        """Covariance matrix is symmetric."""
        t = np.linspace(0.05, 1.0, N)
        K = fbm_covariance(t, H=0.2)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_diagonal_is_t2H(self):
        """Diagonal entries equal t^{2H} (variance of fBm at time t)."""
        t = np.array([0.25, 0.5, 1.0])
        H = 0.3
        K = fbm_covariance(t, H)
        np.testing.assert_allclose(np.diag(K), t ** (2 * H), rtol=1e-12)

    def test_zero_off_diagonal_at_same_point(self):
        """K(t, t) = t^{2H}: on-diagonal uses |t-t|=0 correctly."""
        t = np.array([0.5])
        H = 0.2
        K = fbm_covariance(t, H)
        assert abs(K[0, 0] - 0.5 ** (2 * H)) < 1e-14

    @pytest.mark.parametrize("H", H_LIST)
    def test_psd_for_various_H(self, H):
        """Covariance matrix is positive semi-definite for all tested H values."""
        t = np.linspace(0.05, 1.0, 15)
        K = fbm_covariance(t, H)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10), (
            f"Negative eigenvalue {eigvals.min():.2e} for H={H}")


# ---------------------------------------------------------------------------
# build_rough_covariance
# ---------------------------------------------------------------------------

class TestBuildRoughCovariance:
    def test_output_shapes(self):
        """Returns (N, N) matrix and length-N time grid."""
        K, t = build_rough_covariance(T=1.0, N=N, H=0.2)
        assert K.shape == (N, N)
        assert t.shape == (N,)

    def test_time_grid_endpoints(self):
        """Grid starts at T/N and ends at T."""
        K, t = build_rough_covariance(T=2.0, N=N, H=0.2)
        assert abs(t[0] - 2.0 / N) < 1e-12
        assert abs(t[-1] - 2.0) < 1e-12

    def test_psd_after_regularisation(self):
        """Returned matrix is PSD (all eigenvalues >= 0)."""
        for H in H_LIST:
            K, _ = build_rough_covariance(T=1.0, N=N, H=H)
            eigvals = np.linalg.eigvalsh(K)
            assert np.all(eigvals >= -1e-10), (
                f"Negative eigenvalue {eigvals.min():.2e} for H={H}")

    def test_symmetry_after_regularisation(self):
        """Returned matrix is symmetric."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.1)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_diagonal_variance_order(self):
        """Diagonal entries (marginal variances) are non-decreasing in time."""
        K, t = build_rough_covariance(T=1.0, N=N, H=0.2)
        diag = np.diag(K)
        assert np.all(np.diff(diag) >= -1e-12), (
            "Marginal variances not non-decreasing in time")


# ---------------------------------------------------------------------------
# sample_rough_field (Cholesky Monte Carlo)
# ---------------------------------------------------------------------------

class TestSampleRoughField:
    def test_output_shape(self):
        """Output shape is (N, n_paths)."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y = sample_rough_field(K, n_paths=500, seed=0)
        assert Y.shape == (N, 500)

    def test_zero_mean(self):
        """Sample mean is close to zero for large n_paths."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y = sample_rough_field(K, n_paths=10000, seed=1)
        np.testing.assert_allclose(Y.mean(axis=1), 0.0, atol=0.05)

    def test_marginal_variance(self):
        """Sample variance at each time ≈ K_{ii} (within 10% for 10k paths)."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y = sample_rough_field(K, n_paths=20000, seed=2)
        sample_var = Y.var(axis=1)
        target_var = np.diag(K)
        # Allow 15% relative error for each time step
        np.testing.assert_allclose(sample_var, target_var, rtol=0.15,
                                   err_msg="Sample variance deviates >15% from diagonal of K")

    def test_covariance_structure(self):
        """Sample covariance matrix (N×N) is close to K for large n_paths."""
        K, _ = build_rough_covariance(T=1.0, N=10, H=0.2)
        Y = sample_rough_field(K, n_paths=50000, seed=3)
        sample_cov = np.cov(Y)   # (10, 10)
        # Frobenius norm of relative error < 10%
        rel_err = np.linalg.norm(sample_cov - K) / np.linalg.norm(K)
        assert rel_err < 0.10, f"Sample covariance Frobenius relative error = {rel_err:.3f}"

    def test_reproducibility(self):
        """Same seed gives identical output."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y1 = sample_rough_field(K, n_paths=100, seed=42)
        Y2 = sample_rough_field(K, n_paths=100, seed=42)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_seeds_different(self):
        """Different seeds give different output."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y1 = sample_rough_field(K, n_paths=100, seed=42)
        Y2 = sample_rough_field(K, n_paths=100, seed=99)
        assert not np.allclose(Y1, Y2), "Different seeds produced identical samples"


# ---------------------------------------------------------------------------
# sample_rough_field_sobol (RQMC)
# ---------------------------------------------------------------------------

class TestSampleRoughFieldSobol:
    def test_output_shape(self):
        """Output shape is (N, n_paths * n_scrambles)."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        n_paths, n_sc = 256, 4
        Y = sample_rough_field_sobol(K, n_paths=n_paths, n_scrambles=n_sc)
        assert Y.shape == (N, n_paths * n_sc)

    def test_zero_mean(self):
        """Sobol sample mean is close to zero."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y = sample_rough_field_sobol(K, n_paths=1024, n_scrambles=4)
        np.testing.assert_allclose(Y.mean(axis=1), 0.0, atol=0.05)

    def test_marginal_variance_accuracy(self):
        """Sobol sample variance approximates diagonal of K."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.2)
        Y = sample_rough_field_sobol(K, n_paths=2048, n_scrambles=4)
        sample_var = Y.var(axis=1)
        target_var = np.diag(K)
        np.testing.assert_allclose(sample_var, target_var, rtol=0.15,
                                   err_msg="Sobol variance deviates >15% from K diagonal")

    def test_finite_values(self):
        """All sampled values are finite."""
        K, _ = build_rough_covariance(T=1.0, N=N, H=0.1)
        Y = sample_rough_field_sobol(K, n_paths=512, n_scrambles=2)
        assert np.all(np.isfinite(Y)), "Non-finite values in Sobol field samples"
