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
    ) -> dict[str, np.ndarray]:
        return precondition(gradient, self.diagonal, damping=damping, smooth=smooth)

    def illumination(self) -> np.ndarray:
        """归一化照明场，用于绘图（多参数取平均）。"""
        fields = [np.asarray(v, dtype=np.float64) for v in self.diagonal.values()]
        out = np.mean(fields, axis=0) if fields else np.zeros((1, 1))
        m = float(np.max(out))
        return out / (m + 1e-30)
