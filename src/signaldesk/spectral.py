"""Spectral tools: PSD estimate, a white-noise reference band, autocorrelation.

The question for step 4: are daily returns white noise?
White noise = uncorrelated samples = flat PSD = autocorrelation that is zero
at every lag except lag 0. If returns are white, today's return tells you
nothing (linearly) about tomorrow's -- no predictable signal.
"""
import numpy as np
from scipy import signal, stats

FS = 1.0          # sample rate: 1 sample per trading day -> Nyquist = 0.5 cycles/day
NPERSEG = 256     # Welch segment length ~ 1 trading year -> resolution 1/256 cycles/day


def welch_psd(x: np.ndarray, nperseg: int = NPERSEG):
    """One-sided Welch PSD (Hann window, 50% overlap, mean removed per segment).

    Averaging ~22 segments trades frequency resolution for a much lower
    variance estimate -- a raw periodogram's bins fluctuate by ~100%.
    """
    return signal.welch(x, fs=FS, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, detrend="constant", scaling="density")


def shuffled_null_band(x: np.ndarray, n_shuffles: int = 500, nperseg: int = NPERSEG,
                       q=(2.5, 97.5), seed: int = 0):
    """White-noise reference band built from the data itself.

    Randomly permuting the samples keeps the exact amplitude distribution
    (including the heavy tails) but destroys all time ordering -- so every
    shuffle is, by construction, white noise with the same power. Running the
    same Welch estimator on 500 shuffles gives the spread a white signal would
    show *with this estimator*, so we don't need an analytic chi-squared model.

    Returns (median, low, high) per frequency bin.
    """
    rng = np.random.default_rng(seed)
    psds = np.array([welch_psd(rng.permutation(x), nperseg)[1] for _ in range(n_shuffles)])
    lo, hi = np.percentile(psds, q, axis=0)
    return np.median(psds, axis=0), lo, hi


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation for lags 0..max_lag (the time-domain twin of the PSD:
    Wiener-Khinchin says PSD = Fourier transform of the autocorrelation)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def ljung_box(x: np.ndarray, lags: int = 10):
    """Ljung-Box test: one number summarising 'are the first `lags`
    autocorrelations jointly zero?'. Small p-value (< 0.05) = NOT white.

    Q = n(n+2) * sum_k rho_k^2 / (n-k)  ~  chi-squared(lags) under white noise.
    Caveat: assumes constant variance; heavy-tailed, volatility-clustered data
    makes it reject a bit too often, so treat borderline p-values with care.
    """
    n = len(x)
    rho = acf(x, lags)[1:]
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)


def robust_acf_band(x: np.ndarray, max_lag: int, z: float = 1.96) -> np.ndarray:
    """95% band for each sample autocorrelation that stays valid when the
    noise power changes over time (volatility clustering).

    The textbook band +-1.96/sqrt(n) assumes constant variance. When loud
    bursts occur, a handful of huge samples dominate the lag products, so the
    sample ACF scatters much more than 1/sqrt(n). This band uses the actual
    lag products: se_k = sqrt(sum x[t]^2 x[t-k]^2) / sum x[t]^2.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([z * np.sqrt(np.sum(x[k:]**2 * x[:-k]**2)) / denom
                     for k in range(1, max_lag + 1)])
