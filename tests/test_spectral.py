"""Calibrate the tools on signals where we KNOW the answer, before trusting
them on market data (like checking a spectrum analyser with a known source)."""
import numpy as np

from signaldesk.spectral import acf, ljung_box, shuffled_null_band, welch_psd


def _ar1(n, phi, rng):
    # AR(1): x[n] = phi*x[n-1] + e[n] -> a one-pole low-pass filter on white noise.
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_white_noise_psd_level():
    # One-sided PSD of white noise with variance s^2 at fs=1 is flat at 2*s^2.
    x = np.random.default_rng(1).normal(scale=0.02, size=3000)
    _, p = welch_psd(x)
    assert np.isclose(np.median(p), 2 * 0.02**2, rtol=0.15)


def test_white_noise_is_accepted():
    x = np.random.default_rng(2).standard_normal(3000)
    _, p = welch_psd(x)
    _, lo, hi = shuffled_null_band(x, n_shuffles=200)
    outside = np.mean((p < lo) | (p > hi))
    assert outside < 0.15          # ~5% expected by chance
    assert ljung_box(x)[1] > 0.01


def test_coloured_noise_is_detected():
    x = _ar1(3000, 0.3, np.random.default_rng(3))
    _, p = welch_psd(x)
    _, lo, hi = shuffled_null_band(x, n_shuffles=200)
    assert np.mean((p < lo) | (p > hi)) > 0.5
    assert ljung_box(x)[1] < 1e-6
    assert abs(acf(x, 1)[1] - 0.3) < 0.06


def test_robust_band_widens_under_volatility_bursts():
    # White noise whose power jumps 10x for 5% of the time: still uncorrelated,
    # but the naive 1.96/sqrt(n) band is too narrow. The robust band must be wider.
    from signaldesk.spectral import robust_acf_band
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    x[1000:1150] *= 10
    assert robust_acf_band(x, 5)[0] > 1.5 * 1.96 / np.sqrt(len(x))
