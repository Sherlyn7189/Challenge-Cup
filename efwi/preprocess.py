"""实际资料预处理：直达波/面波、振幅/相位、子波估计。

EFWI 内部数据格式为 ``(ns, nt, nr, ncomp)``。本模块提供可组合的实际资料
预处理步骤，并且保留缺失道 mask：

* 直达波切除（按炮检距/速度预测到时）；
* FK 视速度滤波（压制低速面波/线性干扰）；
* 球面扩散/道均衡振幅校正；
* 相位旋转（Hilbert 解析信号）；
* 尖峰压制；
* 基于“观测直达波 / 平滑模型合成直达波”的 Wiener 子波估计。

所有函数默认沿时间轴 ``axis=0``，支持单炮 ``(nt, nr)`` 或多炮
``(ns, nt, nr, ncomp)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import hilbert

from .io import SegyGathers


@dataclass
class PreprocessConfig:
    direct_mute_velocity: float | None = 1500.0
    direct_mute_factor: float = 0.80
    direct_mute_taper: float = 0.02

    fk_filter: bool = True
    fk_vmin: float = 900.0
    fk_vmax: float = 7000.0
    fk_taper: float = 0.15

    trace_balance: bool = True
    balance_max_gain: float = 8.0

    phase_deg: float = 0.0
    spherical_gain_power: float = 0.0
    despike: bool = False
    despike_threshold: float = 6.0


def _as_4d(data: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, :, :, None], True
    if arr.ndim == 3:
        # (ns, nt, nr) -> 单分量；或 (nt, nr, ncomp)
        if arr.shape[0] < arr.shape[1]:
            return arr[:, :, :, None], False
        return arr[None, :, :, :], False
    if arr.ndim == 4:
        return arr, False
    raise ValueError("data 形状应为 (nt,nr)、(ns,nt,nr) 或 (ns,nt,nr,ncomp)")


def offsets_from_geometry(shots: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    shots = np.asarray(shots, dtype=np.float64)
    receivers = np.asarray(receivers, dtype=np.float64)
    d = shots[:, None, :] - receivers[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def mute_direct_wave(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    *,
    velocity: float = 1500.0,
    factor: float = 0.8,
    taper: float = 0.02,
) -> np.ndarray:
    """按炮检距预测直达波到时，平滑切除到时前的能量。

    ``data`` 支持 ``(nt,nr)``、``(ns,nt,nr)``、``(ns,nt,nr,ncomp)``；
    ``offsets`` 形状 ``(ns,nr)``。返回同形状数组。
    """
    arr, was_2d = _as_4d(data)
    ns, nt, nr, nc = arr.shape
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (ns, nr):
        raise ValueError(f"offsets 形状应为 {(ns, nr)}")
    t = np.arange(nt, dtype=np.float64) * dt
    t_mute = offsets / max(velocity, 1e-9) * factor
    weight = 0.5 * (1.0 + np.tanh((t[None, :, None] - t_mute[:, None, :]) / max(taper, 1e-9)))
    out = arr * weight[:, :, :, None]
    return out[0, :, :, 0] if was_2d else out


def fk_filter(
    data: np.ndarray,
    dt: float,
    dx: float,
    *,
    vmin: float = 900.0,
    vmax: float = 7000.0,
    taper: float = 0.15,
) -> np.ndarray:
    """二维 FK 视速度滤波。

    在 ``(t, x)`` 平面做 rFFT/FFT，保留视速度
    ``v = f / |k|`` 位于 ``[vmin, vmax]`` 的分量；``vmin`` 以下通常是低速
    面波/线性干扰，``vmax`` 以上是空间假频/噪声。边界用余弦过渡。
    """
    arr, was_2d = _as_4d(data)
    ns, nt, nr, nc = arr.shape
    if nr < 2 or nt < 8:
        return arr[0, :, :, 0] if was_2d else arr
    k = np.fft.fftfreq(nr, d=dx)
    f = np.fft.rfftfreq(nt, d=dt)
    v_app = f[:, None] / (np.abs(k)[None, :] + 1e-12)
    vmax = max(float(vmax), float(vmin) + 1e-9)
    lo, hi = float(vmin), float(vmax)
    width = max((hi - lo) * max(taper, 1e-6), 1e-9)

    def smooth_step(x, a, b):
        u = np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * u)

    mask = smooth_step(v_app, lo - width, lo) * (1.0 - smooth_step(v_app, hi, hi + width))
    mask[f <= 0.0, :] = 0.0
    mask[:, np.abs(k) <= 0.0] = 0.0
    out = np.empty_like(arr)
    for s in range(ns):
        for c in range(nc):
            spec = np.fft.rfft(arr[s, :, :, c], axis=0)
            spec = np.fft.fft(spec, axis=1)
            spec *= mask
            rec = np.fft.ifft(spec, axis=1)
            out[s, :, :, c] = np.fft.irfft(rec, n=nt, axis=0)
    return out[0, :, :, 0] if was_2d else out


def trace_balance(
    data: np.ndarray,
    *,
    max_gain: float = 8.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """道均衡：把每道 RMS 拉到所有有效道的中位 RMS，并限制最大增益。"""
    arr, was_2d = _as_4d(data)
    rms = np.sqrt(np.mean(arr * arr, axis=1))  # (ns, nr, nc)
    valid = rms > eps
    if not np.any(valid):
        return data
    target = float(np.median(rms[valid]))
    gain = np.ones_like(rms)
    gain[valid] = target / rms[valid]
    gain = np.clip(gain, 1.0 / max(max_gain, 1.0), max(max_gain, 1.0))
    out = arr * gain[:, None, :, :]
    return out[0, :, :, 0] if was_2d else out


def phase_rotate(data: np.ndarray, degrees: float) -> np.ndarray:
    """用 Hilbert 解析信号做常相位旋转（正角度为相位超前）。"""
    arr, was_2d = _as_4d(data)
    angle = np.deg2rad(float(degrees))
    analytic = hilbert(arr, axis=1)
    out = np.real(analytic * np.exp(1j * angle))
    return out[0, :, :, 0] if was_2d else out


def spherical_divergence_gain(
    data: np.ndarray,
    dt: float,
    *,
    power: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """球面扩散补偿：乘 ``(t+t0)^power`` 后按最大绝对值归一化。"""
    arr, was_2d = _as_4d(data)
    t = np.arange(arr.shape[1], dtype=np.float64) * dt
    gain = (t + eps) ** float(power)
    out = arr * gain[None, :, None, None]
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out /= peak
    return out[0, :, :, 0] if was_2d else out


def despike(data: np.ndarray, *, threshold: float = 6.0, size: int = 5) -> np.ndarray:
    """沿时间的中值滤波压制尖峰，只替换偏离中值超过 threshold×MAD 的点。"""
    arr, was_2d = _as_4d(data)
    med = median_filter(arr, size=(1, size, 1, 1), mode="nearest")
    mad = np.median(np.abs(arr - med), axis=1, keepdims=True) + 1e-12
    bad = np.abs(arr - med) > threshold * mad
    out = np.where(bad, med, arr)
    return out[0, :, :, 0] if was_2d else out


# ----------------------------------------------------------------------
# 子波估计
# ----------------------------------------------------------------------
@dataclass
class WaveletEstimate:
    wavelet: np.ndarray
    transfer: np.ndarray
    coherence: float
    dt: float
    meta: dict[str, Any] = field(default_factory=dict)


def estimate_source_wavelet(
    observed: np.ndarray,
    synthetic: np.ndarray,
    reference_wavelet: np.ndarray,
    dt: float,
    *,
    time_window: tuple[float, float] | None = None,
    eps: float = 1e-3,
    taper: float = 0.05,
    normalize: bool = True,
) -> WaveletEstimate:
    """用观测/合成直达波估计震源子波。

    设观测 ``y = w_true * G``，合成 ``x = w_ref * G``（G 近似相同），
    频域 Wiener 传递函数 ``H = Pxy / (Pxx + eps)``，则估计子波
    ``w_est = IFFT(FFT(w_ref) * H)``。

    ``observed/synthetic`` 可为 ``(nt, ntraces)`` 或 ``(nt,)``；时间窗
    以秒为单位，缺省使用整段。返回的 ``coherence`` 是
    ``|Pxy|²/(Pxx*Pyy)`` 的频带平均值，用于 QC。
    """
    y = np.asarray(observed, dtype=np.float64)
    x = np.asarray(synthetic, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError(f"observed/synthetic 形状不一致: {y.shape} vs {x.shape}")
    if y.ndim == 1:
        y = y[:, None]
        x = x[:, None]
    nt = y.shape[0]
    if time_window is not None:
        t = np.arange(nt) * dt
        m = (t >= time_window[0]) & (t <= time_window[1])
        if not np.any(m):
            raise ValueError("time_window 内没有采样")
        y = y[m]
        x = x[m]
        nt = y.shape[0]
    w = np.hanning(nt)[:, None]
    y = y * w
    x = x * w
    Y = np.fft.rfft(y, axis=0)
    X = np.fft.rfft(x, axis=0)
    pxx = np.mean(np.abs(X) ** 2, axis=1)
    pxy = np.mean(np.conj(X) * Y, axis=1)
    pyy = np.mean(np.abs(Y) ** 2, axis=1)
    H = pxy / (pxx + eps * max(float(np.max(pxx)), 1e-30))
    coh = np.abs(pxy) ** 2 / (pxx * pyy + 1e-30)
    ref = np.asarray(reference_wavelet, dtype=np.float64).reshape(-1)
    if ref.size < nt:
        ref = np.pad(ref, (0, nt - ref.size))
    elif ref.size > nt:
        ref = ref[:nt]
    W = np.fft.rfft(ref)
    est = np.fft.irfft(W * H, n=nt)
    if taper > 0:
        win = np.hanning(nt) ** 0.5
        est *= win
    if normalize:
        peak = float(np.max(np.abs(est)))
        if peak > 0:
            est /= peak
    band = (np.fft.rfftfreq(nt, dt) > 0.0)
    return WaveletEstimate(
        wavelet=est,
        transfer=H,
        coherence=float(np.mean(coh[band])) if np.any(band) else 0.0,
        dt=float(dt),
        meta={"nt": nt, "window": time_window},
    )


# ----------------------------------------------------------------------
# 完整预处理流程
# ----------------------------------------------------------------------
def preprocess_gathers(
    gathers: SegyGathers,
    config: PreprocessConfig | None = None,
    *,
    receiver_spacing: float | None = None,
) -> tuple[SegyGathers, dict[str, Any]]:
    """对 ``SegyGathers`` 依次做切除/FK/均衡/相位/尖峰处理。"""
    cfg = config or PreprocessConfig()
    data = gathers.data.copy()
    mask = gathers.mask.copy()
    offsets = offsets_from_geometry(gathers.shots, gathers.receivers)
    dx = receiver_spacing
    if dx is None and gathers.n_receivers > 1:
        dx = float(np.median(np.abs(np.diff(gathers.receivers[:, 0]))))
    dx = max(float(dx or 1.0), 1e-6)
    mask_t = np.broadcast_to(mask[:, None, :, :], data.shape)
    qc: dict[str, Any] = {
        "input_rms": float(np.sqrt(np.mean(data[mask_t] ** 2))) if mask.any() else 0.0
    }

    if cfg.direct_mute_velocity:
        data = mute_direct_wave(
            data, gathers.dt, offsets,
            velocity=cfg.direct_mute_velocity,
            factor=cfg.direct_mute_factor,
            taper=cfg.direct_mute_taper,
        )
    if cfg.fk_filter:
        data = fk_filter(
            data, gathers.dt, dx,
            vmin=cfg.fk_vmin, vmax=cfg.fk_vmax, taper=cfg.fk_taper,
        )
    if cfg.trace_balance:
        data = trace_balance(data, max_gain=cfg.balance_max_gain)
    if cfg.spherical_gain_power:
        data = spherical_divergence_gain(data, gathers.dt, power=cfg.spherical_gain_power)
    if cfg.phase_deg:
        data = phase_rotate(data, cfg.phase_deg)
    if cfg.despike:
        data = despike(data, threshold=cfg.despike_threshold)

    data = np.where(mask_t, data, 0.0)
    qc.update(
        {
            "output_rms": float(np.sqrt(np.mean(data[mask_t] ** 2))) if mask.any() else 0.0,
            "max_amplitude": float(np.max(np.abs(data[mask_t]))) if mask.any() else 0.0,
            "receiver_spacing": dx,
            "n_missing": int((~mask).sum()),
        }
    )
    out = SegyGathers(
        shots=gathers.shots.copy(),
        receivers=gathers.receivers.copy(),
        data=data,
        dt=gathers.dt,
        mask=mask,
        shot_ids=gathers.shot_ids.copy(),
        receiver_ids=gathers.receiver_ids.copy(),
        text_header=gathers.text_header,
        meta=dict(gathers.meta),
        component_ids=gathers.component_ids,
    )
    return out, qc
