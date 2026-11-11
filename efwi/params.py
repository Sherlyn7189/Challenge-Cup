"""参数空间打包/解包与重参数化。

正演与约束在物理参数空间完成；优化器在“相对对数”空间迭代::

    x_vp = log(vp / vp_ref),  x_vs = log(vs / vs_ref),  x_rho = log(rho / rho_ref)

这样三个量纲不同的参数具有可比尺度，且天然保证正性。各向异性参数
``epsilon/delta`` 可正可负，使用线性参数化。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .model import ElasticModel

LOG_KEYS = ("vp", "vs", "rho")


class ParameterSpace:
    def __init__(
        self,
        keys: Iterable[str] = ("vp", "vs", "rho"),
        refs: dict[str, float] | None = None,
    ) -> None:
        self.keys = tuple(keys)
        if not self.keys:
            raise ValueError("至少需要一个参数")
        for key in self.keys:
            if key not in ("vp", "vs", "rho", "epsilon", "delta"):
                raise ValueError(f"未知参数 {key!r}")
        self.refs: dict[str, float] = dict(refs or {})

    # ------------------------------------------------------------------
    def set_reference(self, model: ElasticModel) -> None:
        for key in self.keys:
            arr = getattr(model, key)
            if key in LOG_KEYS:
                self.refs[key] = float(np.median(arr))
                if self.refs[key] <= 0:
                    self.refs[key] = 1.0

    def ensure_reference(self, model: ElasticModel) -> None:
        missing = [k for k in self.keys if k not in self.refs]
        if missing:
            self.set_reference(model)

    @property
    def log_keys(self) -> tuple[str, ...]:
        return tuple(k for k in self.keys if k in LOG_KEYS)

    @property
    def linear_keys(self) -> tuple[str, ...]:
        return tuple(k for k in self.keys if k not in LOG_KEYS)

    def sizes(self, shape: tuple[int, int]) -> dict[str, int]:
        n = int(shape[0] * shape[1])
        return {k: n for k in self.keys}

    @property
    def total_size(self) -> int:
        raise AttributeError("总长度依赖模型网格，请使用 total_size_for(shape)")

    def total_size_for(self, shape: tuple[int, int]) -> int:
        return int(shape[0] * shape[1] * len(self.keys))

    def slices(self, shape: tuple[int, int]) -> dict[str, slice]:
        n = int(shape[0] * shape[1])
        out = {}
        start = 0
        for key in self.keys:
            out[key] = slice(start, start + n)
            start += n
        return out

    # ------------------------------------------------------------------
    def pack(self, model: ElasticModel) -> np.ndarray:
        self.ensure_reference(model)
        parts = []
        for key in self.keys:
            arr = np.asarray(getattr(model, key), dtype=np.float64)
            if key in LOG_KEYS:
                parts.append(np.log(arr / self.refs[key]).ravel())
            else:
                parts.append(arr.ravel())
        return np.concatenate(parts)

    def unpack(self, x: np.ndarray, template: ElasticModel) -> ElasticModel:
        self.ensure_reference(template)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        shape = template.shape
        n = shape[0] * shape[1]
        sl = self.slices(shape)
        data = {}
        for key in self.keys:
            seg = x[sl[key]].reshape(shape)
            if key in LOG_KEYS:
                data[key] = self.refs[key] * np.exp(seg)
            else:
                data[key] = seg
        # 未参与反演的参数保持模板值
        for key in ("vp", "vs", "rho", "epsilon", "delta"):
            if key not in data:
                data[key] = np.asarray(getattr(template, key), dtype=np.float64)
        vp = data["vp"]
        vs = data["vs"]
        if np.any(vs >= vp):
            data["vs"] = np.minimum(vs, 0.98 * vp)
        return template.with_params(**data)

    def gradient_to_space(self, grads: dict[str, np.ndarray], model: ElasticModel) -> np.ndarray:
        """物理空间梯度 -> 优化空间梯度（log 参数乘 m）。"""
        self.ensure_reference(model)
        parts = []
        for key in self.keys:
            g = np.asarray(grads[key], dtype=np.float64)
            if key in LOG_KEYS:
                g = g * np.asarray(getattr(model, key), dtype=np.float64)
            parts.append(g.ravel())
        return np.concatenate(parts)

    def zero(self, shape: tuple[int, int]) -> dict[str, np.ndarray]:
        return {k: np.zeros(shape, dtype=np.float64) for k in self.keys}
