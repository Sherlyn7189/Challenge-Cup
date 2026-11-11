"""原生刚度梯度 -> (vp, vs, rho[, epsilon, delta]) 的链式转换。

正演/伴随在 ``(C11, C13, C33, C44, rho)`` 空间给出原生梯度；本模块按
刚度对物理参数的偏导做链式求和，并把伪 Hessian 对角做同样的对角近似
（忽略交叉项，即高频/无限孔径近似下的对角块）。

各向异性参数默认固定；当 ``invert_anisotropy=True`` 时额外给出
``epsilon``（VTI 的 C11 / HTI 的 C33）与 ``delta``（C13）的梯度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from .fd.operator import GradientResult
from .model import ElasticModel


@dataclass
class PhysicalGradient:
    g: dict[str, np.ndarray]
    h: dict[str, np.ndarray]
    g_native: dict[str, np.ndarray]
    h_native: dict[str, np.ndarray]

    @property
    def g_vp(self) -> np.ndarray:
        return self.g["vp"]

    @property
    def g_vs(self) -> np.ndarray:
        return self.g["vs"]

    @property
    def g_rho(self) -> np.ndarray:
        return self.g["rho"]


def _stiffness_derivatives(
    model: ElasticModel,
) -> dict[str, dict[str, np.ndarray]]:
    """返回 ``dC_ij/dp``，键为 c11/c13/c33/c44，子键为参数名。"""
    vp = np.asarray(model.vp, dtype=np.float64)
    vs = np.asarray(model.vs, dtype=np.float64)
    rho = np.asarray(model.rho, dtype=np.float64)
    vp2 = vp * vp
    vs2 = vs * vs
    eps = np.asarray(model.epsilon, dtype=np.float64)
    delta = np.asarray(model.delta, dtype=np.float64)

    if model.symmetry == "iso":
        return {
            "c11": {"vp": 2 * rho * vp, "vs": np.zeros_like(vp), "rho": vp2},
            "c13": {"vp": 2 * rho * vp, "vs": -4 * rho * vs, "rho": vp2 - 2 * vs2},
            "c33": {"vp": 2 * rho * vp, "vs": np.zeros_like(vp), "rho": vp2},
            "c44": {"vp": np.zeros_like(vp), "vs": 2 * rho * vs, "rho": vs2},
        }

    q = (vp2 - vs2) * (vp2 * (1.0 + 2.0 * delta) - vs2)
    q = np.maximum(q, 1e-10 * np.maximum(vp2 * vp2, 1.0))
    sq = np.sqrt(q)
    dsq_dvp = (vp * (vp2 * (1.0 + 2.0 * delta) - vs2) + (vp2 - vs2) * vp * (1.0 + 2.0 * delta)) / sq
    dsq_dvs = (-vs * (vp2 * (1.0 + 2.0 * delta) - vs2) - (vp2 - vs2) * vs) / sq
    dsq_ddelta = (vp2 - vs2) * vp2 / sq

    if model.symmetry == "vti":
        c11 = {
            "vp": 2 * rho * vp * (1.0 + 2.0 * eps),
            "vs": np.zeros_like(vp),
            "rho": vp2 * (1.0 + 2.0 * eps),
            "epsilon": 2 * rho * vp2,
            "delta": np.zeros_like(vp),
        }
        c33 = {
            "vp": 2 * rho * vp,
            "vs": np.zeros_like(vp),
            "rho": vp2,
            "epsilon": np.zeros_like(vp),
            "delta": np.zeros_like(vp),
        }
    else:  # hti
        c11 = {
            "vp": 2 * rho * vp,
            "vs": np.zeros_like(vp),
            "rho": vp2,
            "epsilon": np.zeros_like(vp),
            "delta": np.zeros_like(vp),
        }
        c33 = {
            "vp": 2 * rho * vp * (1.0 + 2.0 * eps),
            "vs": np.zeros_like(vp),
            "rho": vp2 * (1.0 + 2.0 * eps),
            "epsilon": 2 * rho * vp2,
            "delta": np.zeros_like(vp),
        }

    c13 = {
        "vp": rho * dsq_dvp,
        "vs": rho * (dsq_dvs - 2.0 * vs),
        "rho": sq - vs2,
        "epsilon": np.zeros_like(vp),
        "delta": rho * dsq_ddelta,
    }
    c44 = {
        "vp": np.zeros_like(vp),
        "vs": 2 * rho * vs,
        "rho": vs2,
        "epsilon": np.zeros_like(vp),
        "delta": np.zeros_like(vp),
    }
    return {"c11": c11, "c13": c13, "c33": c33, "c44": c44}


def convert_gradient(
    result: GradientResult,
    model: ElasticModel,
    params: Iterable[str] = ("vp", "vs", "rho"),
) -> PhysicalGradient:
    params = tuple(params)
    dc = _stiffness_derivatives(model)
    keys = ("c11", "c13", "c33", "c44")

    g = {p: np.zeros_like(model.vp, dtype=np.float64) for p in params}
    h = {p: np.zeros_like(model.vp, dtype=np.float64) for p in params}

    # 密度原生项（直接来自动量方程）
    if "rho" in params:
        g["rho"] += result.g["rho"]
        h["rho"] += result.h["rho"]

    for ckey in keys:
        g_c = result.g[ckey]
        h_c = result.h[ckey]
        for p in params:
            if p not in dc[ckey]:
                continue
            d = dc[ckey][p]
            g[p] += g_c * d
            h[p] += h_c * d * d

    # vp 的参数空间伪 Hessian 用虚拟源 (2*rho*vp*theta) 直接累积：
    # H_vp = 2 * ∫(2*rho*vp*theta)^2 dt = 8*rho^2*vp^2*H_theta。
    # 这比 C11/C13/C33 的链式平方多了 2*exx*ezz 交叉项，更接近
    # Shin et al. (2001) 的 pseudo-Hessian 定义。各向异性目前保留
    # 刚度空间的对角链式近似。
    if model.symmetry == "iso" and "vp" in params and "theta" in result.h:
        rho = np.asarray(model.rho, dtype=np.float64)
        vp = np.asarray(model.vp, dtype=np.float64)
        h["vp"] = 8.0 * rho * rho * vp * vp * np.asarray(result.h["theta"], dtype=np.float64)

    return PhysicalGradient(
        g=g,
        h=h,
        g_native={k: result.g[k] for k in result.g},
        h_native={k: result.h[k] for k in result.h},
    )


def precondition(
    g: dict[str, np.ndarray],
    h: dict[str, np.ndarray],
    *,
    damping: float = 1e-3,
    smooth: float = 2.0,
    normalize: bool = True,
    water_level: float = 0.0,
    max_gain: float = 20.0,
    min_h: float = 0.0,
) -> dict[str, np.ndarray]:
    """伪 Hessian 对角预条件。

    定义（参数空间，Shin 型对角近似）::

        H_p(x) = sum_s ∫ || S_p(x,t) ||^2 dt
        g_pre  = g / (H_p + eta*max(H_p) + water_level*max(H_p) + min_h)

    其中 ``S_p`` 是参数 p 的虚拟源（散射源），不是对梯度做平滑。``smooth``
    只用于稳定 **H 对角估计**（高频/有限孔径下的估计噪声），不替代 H 的
    物理定义。

    数值稳定措施：

    * ``damping`` / ``water_level`` / ``min_h``：相对/绝对水层；
    * ``max_gain``：把预条件增益限制在 ``max_gain × 典型增益`` 以内，防止
      深部极弱照明点被放大成噪声；
    * 非有限值清零；可选按原始梯度 RMS 归一化，保持参数间尺度一致。

    物理意义：H 越大表示该点照明越强；除以 H 补偿中深部照明不足，使梯度
    能量在空间上更均衡。
    """
    out = {}
    for key, gk in g.items():
        gk = np.asarray(gk, dtype=np.float64)
        hk = np.asarray(h[key], dtype=np.float64)
        if smooth and smooth > 0:
            hk = gaussian_filter(hk, sigma=smooth, mode="nearest")
        hmax = float(np.max(hk))
        if hmax <= 0 or not np.isfinite(hmax):
            out[key] = np.zeros_like(gk)
            continue
        floor = max(damping * hmax, water_level * hmax, min_h)
        h_eff = hk + floor
        gain = 1.0 / h_eff
        if max_gain and max_gain > 0:
            # 以中位增益为“典型增益”，限制极端照明点的放大倍数
            typical = float(np.median(gain[np.isfinite(gain)]))
            if typical > 0:
                gain = np.minimum(gain, max_gain * typical)
        pre = gk * gain
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
        if normalize:
            scale = float(np.sqrt(np.mean(pre * pre)))
            ref = float(np.sqrt(np.mean(gk * gk)))
            if scale > 0 and ref > 0:
                pre *= ref / scale
        out[key] = pre
    return out


def illumination(result: GradientResult, model: ElasticModel) -> dict[str, np.ndarray]:
    """仅返回链式转换后的伪 Hessian 照明场，便于可视化。"""
    return convert_gradient(result, model).h
