"""Wiener 带通滤波与多尺度频带管理。

文档要求“炮集做 Wiener 带通滤波，由低频到高频分段”。这里实现的是
零相位 Wiener 最优滤波器：

1. 以当前炮集为参考，构造理想 Butterworth 带通目标 ``y``；
2. 在频域最小化 ``||h*x - y||^2``，得到 ``H(f)=Pxy/(Pxx+eps)``；
3. 对 H 做平滑和频带掩膜，再作用到所有炮集。

与直接做 Butterworth 滤波相比，Wiener 设计会自适应数据谱，在信噪比
低的频段自动压低响应；同时保持零相位，不会引入走时偏差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import tukey
from scipy.fft import next_fast_len

from .config import StageConfig


def _safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray, axis: int = 0) -> np.ndarray:
    nt = data.shape[axis]
    # filtfilt 的默认 padlen 约为 3*(2*len(sos)+1)；太短时退回单向滤波
    padlen = 3 * (2 * sos.shape[0] + 1)
    if nt > padlen + 1:
        return signal.sosfiltfilt(sos, data, axis=axis)
    return signal.sosfilt(sos, data, axis=axis)


def butter_bandpass(
    data: np.ndarray,
    dt: float,
    f_low: float,
    f_high: float,
    *,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    """零相位 Butterworth 带通（用于构造 Wiener 目标）。"""
    data = np.asarray(data, dtype=np.float64)
    fs = 1.0 / dt
    nyq = fs / 2.0
    f_high = min(float(f_high), 0.95 * nyq)
    f_low = max(float(f_low), 0.0)
    if f_low <= 0:
        sos = signal.butter(order, f_high, btype="lowpass", fs=fs, output="sos")
    else:
        if f_low >= f_high:
            raise ValueError("f_low 必须小于 f_high")
        sos = signal.butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return _safe_sosfiltfilt(sos, data, axis=axis)


@dataclass
class WienerFilter:
    """零相位 Wiener 带通滤波器（频率响应已离散到 ``nfft`` 点）。"""

    response: np.ndarray
    dt: float
    nt: int
    nfft: int
    f_low: float
    f_high: float
    window: np.ndarray
    meta: dict[str, Any]

    def apply(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        moved = np.moveaxis(arr, axis, 0)
        nt = moved.shape[0]
        if nt != self.nt:
            raise ValueError(f"数据时间长度 {nt} 与滤波器 nt={self.nt} 不一致")
        shape = moved.shape
        flat = moved.reshape(nt, -1)
        padded = np.zeros((self.nfft, flat.shape[1]), dtype=np.float64)
        padded[:nt] = flat
        spec = np.fft.rfft(padded, axis=0)
        spec *= self.response[:, None]
        out = np.fft.irfft(spec, n=self.nfft, axis=0)[:nt]
        out *= self.window[:, None]
        return np.moveaxis(out.reshape(shape), 0, axis)

    def amplitude_response(self, freqs: np.ndarray) -> np.ndarray:
        f = np.fft.rfftfreq(self.nfft, d=self.dt)
        return np.interp(np.asarray(freqs), f, self.response, left=0.0, right=0.0)

    # ------------------------------------------------------------------
    @classmethod
    def design(
        cls,
        reference: np.ndarray,
        dt: float,
        f_low: float,
        f_high: float,
        *,
        time_axis: int = 0,
        order: int = 4,
        eps: float = 1e-3,
        mask_octaves: float = 0.5,
        smooth: float = 1.0,
        taper: float = 0.05,
    ) -> "WienerFilter":
        """从参考炮集设计零相位 Wiener 带通。

        ``reference`` 可以是 ``(nt, ntraces)`` 或更高维；沿 ``time_axis``
        是时间。低截频以下/高截频以上各 ``mask_octaves`` 个倍频程做余弦
        掩膜，防止噪声频段被放大。
        """
        ref = np.asarray(reference, dtype=np.float64)
        moved = np.moveaxis(ref, time_axis, 0)
        nt = moved.shape[0]
        if nt < 8:
            raise ValueError("时间采样点太少")
        flat = moved.reshape(nt, -1)
        # 去掉直流，加 Tukey 窗抑制周期延拓振铃
        flat = flat - np.mean(flat, axis=0, keepdims=True)
        win = tukey(nt, alpha=0.05)
        x = flat * win[:, None]
        y = butter_bandpass(x, dt, f_low, f_high, order=order, axis=0)

        nfft = int(next_fast_len(2 * nt))
        xp = np.zeros((nfft, x.shape[1]), dtype=np.float64)
        yp = np.zeros((nfft, y.shape[1]), dtype=np.float64)
        xp[:nt] = x
        yp[:nt] = y
        X = np.fft.rfft(xp, axis=0)
        Y = np.fft.rfft(yp, axis=0)
        pxx = np.mean(np.abs(X) ** 2, axis=1)
        pxy = np.mean(np.conj(X) * Y, axis=1).real
        h = pxy / (pxx + eps * max(float(np.max(pxx)), 1e-30))
        h = np.clip(h, 0.0, 1.5)
        h = gaussian_filter1d(h, sigma=max(smooth, 0.0))

        freqs = np.fft.rfftfreq(nfft, d=dt)
        nyq = 0.5 / dt
        hi = min(f_high * (2.0 ** mask_octaves), 0.95 * nyq)
        lo = max(f_low / (2.0 ** mask_octaves), 0.0)
        mask = np.ones_like(freqs)
        if lo > 0:
            idx = freqs < lo
            mask[idx] = 0.0
            # 过渡带余弦
            trans = (freqs >= lo) & (freqs < f_low)
            if np.any(trans):
                t = (freqs[trans] - lo) / max(f_low - lo, 1e-12)
                mask[trans] = 0.5 - 0.5 * np.cos(np.pi * t)
        trans_hi = (freqs > f_high) & (freqs <= hi)
        if np.any(trans_hi):
            t = (freqs[trans_hi] - f_high) / max(hi - f_high, 1e-12)
            mask[trans_hi] = 0.5 + 0.5 * np.cos(np.pi * t)
        mask[freqs > hi] = 0.0
        h = h * mask
        h[0] = 0.0
        h[-1] = 0.0

        return cls(
            response=h.astype(np.float64),
            dt=float(dt),
            nt=int(nt),
            nfft=nfft,
            f_low=float(f_low),
            f_high=float(f_high),
            window=tukey(nt, alpha=taper),
            meta={"order": order, "eps": eps, "mask_octaves": mask_octaves},
        )

    def copy(self) -> "WienerFilter":
        return WienerFilter(
            self.response.copy(),
            self.dt,
            self.nt,
            self.nfft,
            self.f_low,
            self.f_high,
            self.window.copy(),
            dict(self.meta),
        )


def wiener_bandpass(
    data: np.ndarray,
    dt: float,
    f_low: float,
    f_high: float,
    *,
    time_axis: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """一次性完成“设计 + 应用”的便捷函数。"""
    filt = WienerFilter.design(data, dt, f_low, f_high, time_axis=time_axis, **kwargs)
    return filt.apply(data, axis=time_axis)


def filter_dataset(
    data: np.ndarray,
    filt: WienerFilter,
    *,
    time_axis: int = 1,
) -> np.ndarray:
    """对炮集 ``(ns, nt, nr, ncomp)`` 沿时间轴做 Wiener 滤波。"""
    return filt.apply(data, axis=time_axis)


def make_stage_filter(
    reference: np.ndarray,
    dt: float,
    stage: StageConfig,
    *,
    time_axis: int = 1,
    **kwargs: Any,
) -> WienerFilter:
    return WienerFilter.design(
        reference, dt, stage.f_low, stage.f_high, time_axis=time_axis, **kwargs
    )


def frequency_spectrum(data: np.ndarray, dt: float, *, time_axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """平均振幅谱，便于检查多尺度频带。"""
    arr = np.moveaxis(np.asarray(data, dtype=np.float64), time_axis, 0)
    flat = arr.reshape(arr.shape[0], -1)
    spec = np.mean(np.abs(np.fft.rfft(flat, axis=0)), axis=1)
    freqs = np.fft.rfftfreq(arr.shape[0], d=dt)
    return freqs, spec
