"""Q 衰减建模与 Q 补偿（振幅衰减与相位频散解耦）。

物理与实现约定
--------------
技术方案 6.4 / 7.4 要求 Q 补偿进入反传算子，并且必须同时给出三组对照：
``Q=∞``、含 Q 正演但不补偿、含 Q 正演且补偿。本模块提供这三组对照所需的
两个算子：

- :func:`q_response`：常 Q 介质的**振幅衰减**与**相位频散**分离表示
  ``H(f,t) = exp(-π f t/Q) · exp(i·φ(f,t))``，其中
  ``φ = -(2 f t/Q)·ln(f/f_ref)`` 是与振幅项满足 Kramers-Kronig 关系的一阶
  近似；两项可以分别开关，对应“振幅/相位解耦”；
- :func:`compensation_response`：反传侧的补偿响应，按补偿强度 ``α_c`` 缩放
  指数、按 ``gain_cap`` 限幅，并在高频段做时变低通（技术方案 7.4.4/7.4.5
  的稳定化策略：增益上限 + 时变低通）。

补偿作用在**反传边界条件**（即观测记录）上：把记录按 Q 的逆过程放大后再
注入检波点位置，等价于在接收侧做补偿；源侧保持物理衰减，符合 6.4.4 的
规定。真正把 Q 写进一阶黏弹方程（GSLS 记忆变量）属于第二阶段，本模块的
``response`` 形式便于届时替换而不改动成像流程。

稳定性
------
增益随时长指数增长，因此默认 ``alpha_c=0.8``、``gain_cap=30`` 并叠加时变
低通；三者共同保证补偿后不出现指数发散（7.4.2 的“伴随增益指数增长”）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import QConfig
from .exceptions import DataError
from .logging_utils import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 频响
# ----------------------------------------------------------------------------
def q_response(
    freqs: np.ndarray,
    travel_time: float,
    q: float,
    f_ref: float,
    *,
    strength: float = 1.0,
    amplitude: bool = True,
    dispersion: bool = True,
) -> np.ndarray:
    """常 Q 介质的复频响（``freqs`` 单位 Hz，``travel_time`` 单位 s）。"""
    if q <= 0:
        raise DataError(f"Q 必须为正，当前 {q}")
    f = np.abs(np.asarray(freqs, dtype=np.float64))
    f_safe = np.where(f > 1e-9, f, 1e-9)
    t_star = float(travel_time) * float(strength)
    response = np.ones_like(f, dtype=np.complex128)
    if amplitude:
        response *= np.exp(-np.pi * f * t_star / float(q))
    if dispersion:
        phase = -2.0 * f * t_star / float(q) * np.log(f_safe / float(f_ref))
        response *= np.exp(1j * phase)
    return response


def compensation_response(
    freqs: np.ndarray,
    travel_time: float,
    q: float,
    f_ref: float,
    *,
    alpha_c: float = 0.8,
    gain_cap: float = 30.0,
    lowpass: np.ndarray | None = None,
) -> np.ndarray:
    """补偿频响：增益指数按 ``alpha_c`` 缩放并限幅，可叠加低通。"""
    if not 0.0 <= alpha_c <= 1.0:
        raise DataError(f"alpha_c 必须在 [0,1] 内，当前 {alpha_c}")
    f = np.abs(np.asarray(freqs, dtype=np.float64))
    f_safe = np.where(f > 1e-9, f, 1e-9)
    t_star = float(travel_time) * float(alpha_c)
    gain = np.exp(np.pi * f * t_star / float(q))
    gain = np.minimum(gain, float(gain_cap))
    phase = 2.0 * f * t_star / float(q) * np.log(f_safe / float(f_ref))
    response = gain * np.exp(1j * phase)
    if lowpass is not None:
        response = response * np.asarray(lowpass, dtype=np.float64)
    return response


# ----------------------------------------------------------------------------
# 时变滤波
# ----------------------------------------------------------------------------
def time_varying_filter(
    data: np.ndarray,
    dt: float,
    response_fn: Callable[[float, np.ndarray], np.ndarray],
    *,
    nperseg: int = 128,
    overlap: float = 0.5,
) -> np.ndarray:
    """短时傅里叶域的分块时变滤波（重叠相加）。

    ``response_fn(t_center, freqs) -> (n_freq,)`` 给出该时刻的复频响。
    分块处理使增益/截止频率可以随时间变化，这正是 Q 补偿需要的：越晚的
    时刻增益越大、通带越窄。
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise DataError(f"仅支持 1D/2D 数据，当前 {array.ndim}")
    if dt <= 0:
        raise DataError(f"dt 必须为正，当前 {dt}")

    n = array.shape[0]
    nperseg = int(min(max(nperseg, 8), max(n, 8)))
    hop = max(int(nperseg * (1.0 - overlap)), 1)
    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=dt)

    out = np.zeros_like(array)
    weight = np.zeros(n, dtype=np.float64)
    for start in range(0, n, hop):
        stop = min(start + nperseg, n)
        length = stop - start
        if length < 4:
            break
        segment = np.zeros((nperseg, array.shape[1]), dtype=np.float64)
        segment[:length] = array[start:stop]
        spectrum = np.fft.rfft(segment * window[:, None], axis=0)
        t_center = (start + 0.5 * nperseg) * dt
        response = np.asarray(response_fn(t_center, freqs), dtype=np.complex128)
        if response.shape[0] != freqs.size:
            raise DataError(
                f"频响长度 {response.shape[0]} 与频率点数 {freqs.size} 不一致"
            )
        filtered = np.fft.irfft(spectrum * response[:, None], n=nperseg, axis=0)
        out[start:stop] += filtered[:length] * window[:length, None]
        weight[start:stop] += window[:length] ** 2
    weight = np.where(weight > 1e-12, weight, 1.0)
    result = out / weight[:, None]
    return result[:, 0] if squeeze else result


# ----------------------------------------------------------------------------
# 面向炮记录的算子
# ----------------------------------------------------------------------------
@dataclass
class QCompensator:
    """反传前的 Q 补偿算子（可直接作为 ``migrate(preprocess=...)`` 传入）。

    ``wave`` 选择用 Qp 还是 Qs（PP 用 Qp、PS 的 S 腿用 Qs，见 6.4.5）。
    """

    config: QConfig
    q_value: float = 100.0
    wave: str = "P"
    nperseg: int = 128

    def __post_init__(self) -> None:
        self.config.validate()
        if self.q_value <= 0:
            raise DataError(f"q_value 必须为正，当前 {self.q_value}")

    def __call__(self, data: np.ndarray, dt: float) -> np.ndarray:
        if not self.config.enabled:
            return np.asarray(data)

        q = float(self.q_value)
        alpha = float(self.config.alpha_c)
        cap = float(self.config.gain_cap)
        f_ref = float(self.config.f_ref)
        nyquist = 0.5 / dt

        def response(t_center: float, freqs: np.ndarray) -> np.ndarray:
            travel = max(t_center, 1e-6)
            # 时变低通：越晚的时刻可恢复的高频越少，避免增益在噪声频段发散
            cutoff_ratio = 0.95 - 0.35 * min(travel / 0.5, 1.0)
            cutoff = max(nyquist * cutoff_ratio, 2.0 * f_ref)
            lowpass = 1.0 / (1.0 + (np.abs(freqs) / cutoff) ** 4)
            return compensation_response(
                freqs,
                travel,
                q,
                f_ref,
                alpha_c=alpha,
                gain_cap=cap,
                lowpass=lowpass,
            )

        return time_varying_filter(data, dt, response, nperseg=self.nperseg)


def attenuate_record(
    data: np.ndarray,
    dt: float,
    q: float,
    *,
    f_ref: float = 30.0,
    strength: float = 1.0,
    nperseg: int = 128,
) -> np.ndarray:
    """对记录施加常 Q 衰减（振幅 + 频散），用于合成“含 Q 观测”。"""
    def response(t_center: float, freqs: np.ndarray) -> np.ndarray:
        return q_response(
            freqs,
            max(t_center, 1e-6),
            q,
            f_ref,
            strength=strength,
        )

    return time_varying_filter(data, dt, response, nperseg=nperseg)


def build_compensator(
    config: QConfig,
    model_qp: float,
    model_qs: float,
    *,
    wave: str = "P",
) -> QCompensator:
    """按波型选择 Q 值构造补偿算子。"""
    q_value = float(model_qp if str(wave).upper().startswith("P") else model_qs)
    return QCompensator(config=config, q_value=q_value, wave=str(wave))
