"""黏弹性本构：广义标准线性体（GSLS）与常 Q 拟合。

物理模型
--------
技术方案 2.3 选择 GSLS：L 个标准线性体并联，松弛函数为::

    K(t) = K_R [1 - Σ_l Δ_l e^{-t/τ^σ_l}],   Δ_l = 1 - τ^ε_l/τ^σ_l

对应频域复模量（``ω = 2πf``）::

    K(ω) = K_R [1 - Σ_l Δ_l / (1 + i ω τ^σ_l)]

品质因子 ``Q(ω) = Re K / Im K``。松弛模量 ``K_R`` 由参考频率处的相速度
标定：``K_R = ρ v_p² / Re K_norm(ω_ref)``，因此模型里给的 ``vp`` 表示
``f_ref`` 处的相速度；未松弛模量 ``K_U = K_R (1 + Σ Δ_l)``。

离散形式（Robertsson 等 1994，技术方案 2.4.1）::

    r_l^{n+1} = a_l r_l^n + b_l (ε^{n+1} - ε^n)
    p^{n+1}   = K_U ε^{n+1} - Σ_l r_l^{n+1}

    a_l = exp(-Δt/τ^σ_l),  b_l = K_R Δ_l (1 - a_l) τ^σ_l / Δt

其中 ``ε`` 为体应变，``r_l`` 为记忆变量。``Δ_l > 0`` 为衰减，``Δ_l < 0``
为增益（用于 Q 补偿，见 :mod:`ertm.imaging` 的反传）。

常 Q 拟合
---------
给定 ``Q``、``f_ref`` 与频带，用最小二乘同时拟合 ``L`` 个机制的
``τ^σ_l`` 与 ``Δ_l``，使 ``Q(ω)`` 在频带内逼近常数；``L=3`` 在
0.5–1.5 倍频带内误差通常小于 5%。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .exceptions import ERTMError
from .logging_utils import get_logger

logger = get_logger(__name__)


def normalized_modulus(
    tau_sigma: Iterable[float],
    delta: Iterable[float],
    freqs: Iterable[float],
) -> np.ndarray:
    """归一化复模量 ``K(ω)/K_R = 1 - Σ Δ_l/(1 + iωτ_l)``。"""
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    result = np.ones_like(omega, dtype=np.complex128)
    for tau, dlt in zip(tau_sigma, delta):
        result -= dlt / (1.0 + 1j * omega * tau)
    return result


@dataclass(frozen=True)
class GSLSParameters:
    """一组 GSLS 机制（同一 Q 值对应一套参数）。"""

    q: float
    f_ref: float
    tau_sigma: tuple[float, ...]
    delta: tuple[float, ...]
    k_relaxed: float
    k_unrelaxed: float
    f_band: tuple[float, float] = (0.0, 0.0)
    fit_error: float = 0.0

    # ------------------------------------------------------------------ 频响
    def modulus(self, freqs: Iterable[float]) -> np.ndarray:
        """归一化复模量 ``K(ω)/K_R``（不含 ``K_R``）。"""
        return normalized_modulus(self.tau_sigma, self.delta, freqs)

    def quality(self, freqs: Iterable[float]) -> np.ndarray:
        """``Q(ω) = Re K / Im K``（衰减时为正）。"""
        modulus = self.modulus(freqs)
        real = np.real(modulus)
        imag = np.imag(modulus)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(np.abs(imag) > 1e-12, real / imag, np.inf)
        return q

    def phase_velocity(self, freqs: Iterable[float], vp: float) -> np.ndarray:
        """相速度 ``sqrt(Re K / ρ)``，已按 ``f_ref`` 的 ``vp`` 标定。"""
        modulus = np.real(self.modulus(freqs))
        reference = float(np.real(self.modulus([self.f_ref])[0]))
        return vp * np.sqrt(np.maximum(modulus, 1e-9) / max(reference, 1e-12))

    def summary(self) -> dict[str, float]:
        return {
            "q": float(self.q),
            "f_ref": float(self.f_ref),
            "n_mechanisms": len(self.tau_sigma),
            "k_relaxed": float(self.k_relaxed),
            "k_unrelaxed": float(self.k_unrelaxed),
            "fit_error": float(self.fit_error),
        }


# ----------------------------------------------------------------------------
# 拟合
# ----------------------------------------------------------------------------
def _pack(q: float, f_ref: float, tau: np.ndarray, delta: np.ndarray, band, error) -> GSLSParameters:
    return GSLSParameters(
        q=float(q),
        f_ref=float(f_ref),
        tau_sigma=tuple(float(t) for t in tau),
        delta=tuple(float(d) for d in delta),
        k_relaxed=1.0,
        k_unrelaxed=float(1.0 + np.sum(delta)),
        f_band=(float(band[0]), float(band[1])),
        fit_error=float(error),
    )


def _single_mechanism(q: float, f_ref: float) -> tuple[float, float]:
    """单机制解析解：峰值 Q 落在 ``f_ref``。

    由 ``Q_min = 2√(1-Δ)/Δ`` 反解 ``Δ``，再由峰值频率
    ``ω_p = 1/√(τ^ε τ^σ)`` 定 ``τ^σ``。
    """
    u = (-1.0 + np.sqrt(1.0 + q * q)) / max(q, 1e-12)  # = √(1-Δ)
    delta = 1.0 - u * u
    tau_sigma = u / (2.0 * np.pi * f_ref)
    return float(tau_sigma), float(delta)


def fit_gsls(
    q: float,
    f_ref: float,
    *,
    n_mechanisms: int = 3,
    band_ratio: float = 1.5,
    min_q: float = 2.0,
) -> GSLSParameters:
    """拟合常 Q 的 GSLS 参数。

    ``band_ratio`` 表示频带为 ``[f_ref/band_ratio, f_ref*band_ratio]``。
    ``Q`` 很大（> 1e4）时退化为近弹性：``Δ=0``，避免无意义的拟合。
    """
    if q <= min_q:
        raise ERTMError(f"GSLS 拟合要求 Q > {min_q}，当前 {q}")
    if f_ref <= 0:
        raise ERTMError(f"f_ref 必须为正，当前 {f_ref}")
    f_lo, f_hi = f_ref / band_ratio, f_ref * band_ratio
    freqs = np.geomspace(f_lo, f_hi, 24)
    if q > 1.0e4 or int(n_mechanisms) <= 1:
        tau, delta = _single_mechanism(q, f_ref)
        return _pack(q, f_ref, np.array([tau]), np.array([delta]),
                     (f_lo, f_hi), 0.0)

    try:
        from scipy.optimize import least_squares
    except Exception:  # pragma: no cover - scipy 在 requirements 中
        logger.warning("缺少 scipy，GSLS 退化为单机制解析解")
        tau, delta = _single_mechanism(q, f_ref)
        return _pack(q, f_ref, np.array([tau]), np.array([delta]), (f_lo, f_hi), 0.0)

    n = int(n_mechanisms)
    # 初值：τ 在对数频率上均布，每个机制承担 2/Q 的衰减
    tau0 = 1.0 / (2.0 * np.pi * np.geomspace(f_lo, f_hi, n))
    delta0 = np.full(n, 2.0 / q)

    def residual(params: np.ndarray) -> np.ndarray:
        tau = np.exp(params[:n])
        delta = np.exp(params[n:])
        model = GSLSParameters(
            q=q, f_ref=f_ref, tau_sigma=tuple(tau), delta=tuple(delta),
            k_relaxed=1.0, k_unrelaxed=1.0 + float(delta.sum()),
        )
        q_model = model.quality(freqs)
        log_error = np.log(np.maximum(q_model, 1e-6)) - np.log(q)
        # 约束：优先保证 f_ref 附近的拟合精度
        weights = 1.0 + 0.5 * np.exp(-0.5 * (np.log(freqs / f_ref) / 0.25) ** 2)
        return log_error * weights

    try:
        solution = least_squares(
            residual,
            np.concatenate([np.log(tau0), np.log(delta0)]),
            bounds=(
                np.concatenate([np.log(tau0) - 3.0, np.log(delta0) - 3.0]),
                np.concatenate([np.log(tau0) + 3.0, np.log(delta0) + 3.0]),
            ),
            max_nfev=400,
            xtol=1e-10,
            ftol=1e-10,
        )
    except Exception as exc:  # pragma: no cover - 数值退化时兜底
        logger.warning("GSLS 拟合失败（%s），退化为单机制解析解", exc)
        tau, delta = _single_mechanism(q, f_ref)
        return _pack(q, f_ref, np.array([tau]), np.array([delta]), (f_lo, f_hi), 0.0)

    tau = np.exp(solution.x[:n])
    delta = np.exp(solution.x[n:])
    fitted = GSLSParameters(
        q=q, f_ref=f_ref, tau_sigma=tuple(tau), delta=tuple(delta),
        k_relaxed=1.0, k_unrelaxed=1.0 + float(delta.sum()),
    )
    error = float(np.max(np.abs(np.log(np.maximum(fitted.quality(freqs), 1e-6)) - np.log(q))))
    return _pack(q, f_ref, tau, delta, (f_lo, f_hi), error)


def gsls_table(
    q_values: Iterable[float],
    f_ref: float,
    *,
    n_mechanisms: int = 3,
    band_ratio: float = 1.5,
) -> dict[float, GSLSParameters]:
    """为一组 Q 值建立参数表（模型里通常只有几种 Q 值）。"""
    table: dict[float, GSLSParameters] = {}
    for value in sorted({float(q) for q in np.asarray(list(q_values), dtype=float)}):
        if not np.isfinite(value) or value <= 2.0:
            value = 1.0e4  # 视为弹性
        table[value] = fit_gsls(
            value, f_ref, n_mechanisms=n_mechanisms, band_ratio=band_ratio
        )
    return table


# ----------------------------------------------------------------------------
# 验证辅助
# ----------------------------------------------------------------------------
def estimate_q_spectral_ratio(
    reference: np.ndarray,
    attenuated: np.ndarray,
    freqs: np.ndarray,
    travel_time: float,
    *,
    min_relative: float = 0.05,
) -> float:
    """谱比法反演 Q：``ln(A_ref/A_att) = π f t / Q``。

    ``reference`` / ``attenuated`` 为同一接收点的两道振幅谱（或道集），
    ``travel_time`` 为路径差（秒）。只在两道都高于峰值 ``min_relative`` 的
    频带内拟合，避免带外噪声把斜率拉平。返回最小二乘拟合的 Q。
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    att = np.asarray(attenuated, dtype=np.float64).ravel()
    freq = np.asarray(freqs, dtype=np.float64).ravel()
    if ref.size != att.size or freq.size != ref.size:
        raise ERTMError("谱比法要求参考道/衰减道/频率轴长度一致")
    ratio = np.log(np.maximum(np.abs(ref), 1e-30) / np.maximum(np.abs(att), 1e-30))
    ref_max = float(np.max(np.abs(ref))) if ref.size else 0.0
    att_max = float(np.max(np.abs(att))) if att.size else 0.0
    band = (
        (freq > 0)
        & (np.abs(ref) > min_relative * ref_max)
        & (np.abs(att) > min_relative * att_max)
        & (ratio > 0)
    )
    if np.count_nonzero(band) < 3 or travel_time <= 0:
        return float("inf")
    slope = float(np.polyfit(freq[band], ratio[band], 1)[0])
    return float(np.pi * travel_time / max(slope, 1e-12))
