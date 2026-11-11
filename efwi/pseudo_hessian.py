"""伪 Hessian 对角照明补偿（独立模块，便于单独测试/可视化）。

全 Hessian 不可行；在高频近似 + 无限孔径假设下只保留对角元。对角元
物理上表示“该点被震源/散射能量照明的程度”，因此以阻尼后的倒数作为
梯度预条件算子::

    g_pre = g / (diag(H) + eps * max(diag(H)))

正演核在 ``EFWI`` 计算梯度时会顺带累积 ``diag(H)``（见
``Elastic2DOperator.compute_gradient``），本模块只负责预条件与诊断。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .gradient import precondition


@dataclass
class PseudoHessian:
    diagonal: dict[str, np.ndarray]
    params: tuple[str, ...] = ("vp", "vs", "rho")

    @classmethod
    def from_gradient_result(cls, result, model, params=("vp", "vs", "rho")) -> "PseudoHessian":
        from .gradient import convert_gradient

        pg = convert_gradient(result, model, params)
        return cls(diagonal=pg.h, params=tuple(params))

    def smoothed(self, sigma: float = 2.0) -> dict[str, np.ndarray]:
        return {
            k: gaussian_filter(np.asarray(v, dtype=np.float64), sigma=sigma, mode="nearest")
            for k, v in self.diagonal.items()
        }

    def precondition(
        self,
        gradient: dict[str, np.ndarray],
        *,
        damping: float = 1e-3,
        smooth: float = 2.0,
        normalize: bool = True,
        water_level: float = 0.0,
        max_gain: float = 20.0,
        min_h: float = 0.0,
    ) -> dict[str, np.ndarray]:
        return precondition(
            gradient,
            self.diagonal,
            damping=damping,
            smooth=smooth,
            normalize=normalize,
            water_level=water_level,
            max_gain=max_gain,
            min_h=min_h,
        )

    def illumination(self) -> np.ndarray:
        """归一化照明场，用于绘图（多参数取平均）。"""
        fields = [np.asarray(v, dtype=np.float64) for v in self.diagonal.values()]
        out = np.mean(fields, axis=0) if fields else np.zeros((1, 1))
        m = float(np.max(out))
        return out / (m + 1e-30)

    def stats(self, gradient: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
        """伪 Hessian / 梯度的统计量，用于测试报告和稳定性检查。"""
        out: dict[str, dict[str, float]] = {}
        for key, hk in self.diagonal.items():
            hk = np.asarray(hk, dtype=np.float64)
            finite = hk[np.isfinite(hk)]
            row = {
                "h_min": float(np.min(finite)) if finite.size else 0.0,
                "h_max": float(np.max(finite)) if finite.size else 0.0,
                "h_median": float(np.median(finite)) if finite.size else 0.0,
                "h_zero_fraction": float(np.mean(finite <= 0.0)) if finite.size else 1.0,
                "h_dynamic_range_db": float(
                    20.0 * np.log10((np.max(finite) + 1e-30) / (np.median(finite) + 1e-30))
                ) if finite.size else 0.0,
            }
            if gradient is not None and key in gradient:
                gk = np.asarray(gradient[key], dtype=np.float64)
                row["g_rms"] = float(np.sqrt(np.mean(gk * gk)))
                row["g_max"] = float(np.max(np.abs(gk)))
            out[key] = row
        return out
