"""跨模块共享的数据契约。

这些 dataclass 是模块之间的“最小公分母”：模型、记录、图像和指标都用
统一字段表达，避免各模块各自定义一套字典格式。

数组轴序约定
------------
二维数组统一为 ``(nz, nx)``，索引 ``[iz, ix]``；三维为 ``(nz, ny, nx)``。
时间序列统一为 ``(nt,)`` 或 ``(nt, nrec)``。长度单位米，时间单位秒。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """规则二维网格。"""

    nx: int
    nz: int
    dx: float
    dz: float
    x0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.nz < 2:
            raise ValueError(f"网格至少需要 2x2，当前为 {self.nx}x{self.nz}")
        if self.dx <= 0 or self.dz <= 0:
            raise ValueError(f"网格间距必须为正，当前 dx={self.dx}, dz={self.dz}")

    @property
    def shape(self) -> tuple[int, int]:
        """数组形状 ``(nz, nx)``。"""
        return (self.nz, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.nz

    @property
    def x_min(self) -> float:
        return float(self.x0)

    @property
    def x_max(self) -> float:
        return float(self.x0 + (self.nx - 1) * self.dx)

    @property
    def z_min(self) -> float:
        return float(self.z0)

    @property
    def z_max(self) -> float:
        return float(self.z0 + (self.nz - 1) * self.dz)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``imshow`` 用范围：``(x_min, x_max, z_max, z_min)``（z 向下）。"""
        return (self.x_min, self.x_max, self.z_max, self.z_min)

    def x_at(self, ix: int) -> float:
        return float(self.x0 + ix * self.dx)

    def z_at(self, iz: int) -> float:
        return float(self.z0 + iz * self.dz)

    def index_of(self, x: float, z: float, *, clip: bool = True) -> tuple[int, int]:
        """物理坐标转最近网格索引 ``(ix, iz)``。"""
        ix = int(round((float(x) - self.x0) / self.dx))
        iz = int(round((float(z) - self.z0) / self.dz))
        if clip:
            ix = min(max(ix, 0), self.nx - 1)
            iz = min(max(iz, 0), self.nz - 1)
        elif not (0 <= ix < self.nx and 0 <= iz < self.nz):
            raise ValueError(f"坐标 ({x}, {z}) 超出网格范围")
        return ix, iz


@dataclass
class ElasticModel:
    """二维地质模型。

    所有数组形状必须为 ``grid.shape``，且满足 ``vp > vs > 0``、``rho > 0``、
    ``qp > 0``、``qs > 0``。首期只承载弹性/黏弹性参数，不在此处做 Q 拟合。
    """

    grid: GridSpec
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    qp: np.ndarray
    qs: np.ndarray
    name: str = "model"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def validate(self) -> "ElasticModel":
        """检查形状、有限值和物理合理性，返回自身便于链式调用。"""
        fields = {"vp": self.vp, "vs": self.vs, "rho": self.rho, "qp": self.qp, "qs": self.qs}
        for key, value in fields.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"模型字段 {key} 必须是 numpy 数组")
            if value.shape != self.shape:
                raise ValueError(
                    f"模型字段 {key} 形状 {value.shape} 与网格 {self.shape} 不一致"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"模型字段 {key} 含 NaN/Inf")
            if np.any(value <= 0):
                raise ValueError(f"模型字段 {key} 必须全为正数")
        if np.any(self.vs >= self.vp):
            raise ValueError("模型必须满足 vs < vp")
        return self

    def summary(self) -> dict[str, dict[str, float]]:
        """返回各字段的 min/max/mean，用于日志与报告。"""
        result: dict[str, dict[str, float]] = {}
        for key in ("vp", "vs", "rho", "qp", "qs"):
            arr = getattr(self, key)
            result[key] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
        return result


@dataclass(frozen=True)
class SourceSpec:
    """点源/力源/矩张量源的通用描述。"""

    x: float
    z: float
    f0: float = 20.0
    kind: str = "ricker"
    amplitude: float = 1.0
    delay: float = 0.0


@dataclass(frozen=True)
class ReceiverSpec:
    """检波点集合。"""

    x: np.ndarray
    z: np.ndarray
    component: str = "vz"

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).reshape(-1)
        z = np.asarray(self.z, dtype=float).reshape(-1)
        if x.size == 0 or z.size == 0:
            raise ValueError("检波点不能为空")
        if x.size != z.size:
            raise ValueError(f"检波点 x/z 数量不一致: {x.size} vs {z.size}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "z", z)

    @property
    def nrec(self) -> int:
        return int(self.x.size)


@dataclass
class ShotRecord:
    """单炮记录。"""

    data: np.ndarray
    dt: float
    t: np.ndarray
    receivers: ReceiverSpec
    source: SourceSpec
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ShotRecord":
        data = np.asarray(self.data)
        if data.ndim != 2:
            raise ValueError(f"炮记录必须是二维 (nt, nrec)，当前 {data.shape}")
        if data.shape[0] != self.t.size:
            raise ValueError(f"记录时间维 {data.shape[0]} 与 t 长度 {self.t.size} 不一致")
        if data.shape[1] != self.receivers.nrec:
            raise ValueError(
                f"记录道数 {data.shape[1]} 与检波点 {self.receivers.nrec} 不一致"
            )
        if self.dt <= 0:
            raise ValueError("dt 必须为正")
        if not np.all(np.isfinite(data)):
            raise ValueError("炮记录含 NaN/Inf")
        return self


@dataclass
class ImageResult:
    """偏移/成像结果。"""

    image: np.ndarray
    grid: GridSpec
    mode: str = "pp"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ImageResult":
        image = np.asarray(self.image)
        if image.shape != self.grid.shape:
            raise ValueError(f"图像形状 {image.shape} 与网格 {self.grid.shape} 不一致")
        if not np.all(np.isfinite(image)):
            raise ValueError("图像含 NaN/Inf")
        return self


@dataclass
class Metrics:
    """评价指标集合。"""

    values: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "meta": dict(self.meta)}
