"""震源时间函数。"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey


def time_axis(nt: int, dt: float, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(nt, dtype=np.float64) * dt


def ricker(t: np.ndarray, f0: float, t0: float | None = None) -> np.ndarray:
    """Ricker 子波，峰值频率 ``f0``，峰值位于 ``t0``（默认 1.5/f0）。"""
    t = np.asarray(t, dtype=np.float64)
    if t0 is None:
        t0 = 1.5 / f0
    tau = t - t0
    a = (np.pi * f0 * tau) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gabor(t: np.ndarray, f0: float, sigma: float | None = None, t0: float | None = None) -> np.ndarray:
    """Gabor 子波（高斯包络余弦），用于宽频带测试。"""
    t = np.asarray(t, dtype=np.float64)
    if t0 is None:
        t0 = 3.0 * (sigma if sigma is not None else 1.0 / f0)
    if sigma is None:
        sigma = 1.0 / (2.0 * np.pi * f0) * 2.0
    tau = t - t0
    return np.exp(-(tau / sigma) ** 2) * np.cos(2.0 * np.pi * f0 * tau)


def gaussian_derivative(t: np.ndarray, f0: float, t0: float | None = None) -> np.ndarray:
    """高斯一阶导数子波。"""
    t = np.asarray(t, dtype=np.float64)
    if t0 is None:
        t0 = 2.0 / f0
    sigma = 0.5 / f0
    tau = t - t0
    return -tau / sigma ** 2 * np.exp(-(tau / sigma) ** 2)


def make_wavelet(
    nt: int,
    dt: float,
    f0: float,
    kind: str = "ricker",
    *,
    t0: float | None = None,
    taper: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """生成并归一化震源子波。"""
    t = time_axis(nt, dt)
    kind = kind.lower()
    if kind == "ricker":
        w = ricker(t, f0, t0)
    elif kind == "gabor":
        w = gabor(t, f0, t0=t0)
    elif kind in ("gaussian", "gaussian_derivative", "gauss"):
        w = gaussian_derivative(t, f0, t0)
    else:
        raise ValueError(f"未知子波类型 {kind!r}")
    if taper:
        w = w * tukey(nt, alpha=0.02)
    if normalize:
        peak = np.max(np.abs(w))
        if peak > 0:
            w = w / peak
    return w.astype(np.float64)


def dominant_frequency(wavelet: np.ndarray, dt: float) -> float:
    """由振幅谱峰值估计主频。"""
    spec = np.abs(np.fft.rfft(wavelet))
    freqs = np.fft.rfftfreq(wavelet.size, d=dt)
    if spec.size <= 2:
        return 0.0
    spec[0] = 0.0
    return float(freqs[int(np.argmax(spec))])
