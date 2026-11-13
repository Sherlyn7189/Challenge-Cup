"""三维弹性波基础模块（实验性）。

包含：

* ``Grid3D``：三维规则网格与 CFL；
* ``ElasticModel3D``：各向同性 vp/vs/rho；
* ``kernels3d``：三维速度—应力正演/伴随/梯度；
* ``Elastic3DOperator``：正演、单炮梯度、多炮梯度累积。

当前三维模块是 P0 的**基础实现**：各向同性、海绵吸收层、规则网格；
三维 VTI/HTI、自由表面地形、三维多尺度/约束反演仍属于后续工作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Grid3D:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    pad: int = 2
    sponge: int = 6

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 2 * self.pad + 8:
            raise ValueError("三维网格太小")
        if self.sponge < 3:
            raise ValueError("sponge 至少 3")
        if self.sponge >= min(self.nx, self.ny, self.nz) // 3:
            raise ValueError("sponge 过宽")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx

    @property
    def y(self) -> np.ndarray:
        return np.arange(self.ny) * self.dy

    @property
    def z(self) -> np.ndarray:
        return np.arange(self.nz) * self.dz

    def cfl_dt(self, vmax: float, safety: float = 0.35) -> float:
        h = min(self.dx, self.dy, self.dz)
        return float(safety * h / (vmax * (3.0 ** 0.5)))


@dataclass
class ElasticModel3D:
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None

    def __post_init__(self) -> None:
        self.vp = np.asarray(self.vp, dtype=np.float64)
        self.vs = np.asarray(self.vs, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        if self.vp.ndim != 3 or self.vp.shape != self.vs.shape or self.vp.shape != self.rho.shape:
            raise ValueError("vp/vs/rho 必须是同形状三维数组")
        if np.any(self.vp <= 0) or np.any(self.vs <= 0) or np.any(self.rho <= 0):
            raise ValueError("vp/vs/rho 必须为正")
        if np.any(self.vs >= self.vp):
            raise ValueError("必须满足 vs < vp")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.vp.shape

    def copy(self) -> "ElasticModel3D":
        return ElasticModel3D(self.vp.copy(), self.vs.copy(), self.rho.copy(), self.dx, self.dy, self.dz)

    def stiffness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c11 = self.rho * self.vp ** 2
        c13 = self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)
        c44 = self.rho * self.vs ** 2
        return c11, c13, c44

    def max_velocity(self) -> float:
        return float(np.max(self.vp))

    def describe(self) -> str:
        return (
            f"ElasticModel3D(shape={self.shape}, "
            f"vp=[{self.vp.min():.0f},{self.vp.max():.0f}], "
            f"vs=[{self.vs.min():.0f},{self.vs.max():.0f}], "
            f"rho=[{self.rho.min():.0f},{self.rho.max():.0f}])"
        )


@dataclass
class Survey3D:
    shots: np.ndarray
    receivers: np.ndarray
    nt: int
    dt: float
    wavelet: np.ndarray
    source_kind: str = "explosive"
    amplitude: float = 1e6

    def __post_init__(self) -> None:
        self.shots = np.asarray(self.shots, dtype=np.float64)
        self.receivers = np.asarray(self.receivers, dtype=np.float64)
        if self.shots.ndim != 2 or self.shots.shape[1] != 3:
            raise ValueError("shots 形状应为 (ns,3)")
        if self.receivers.ndim != 2 or self.receivers.shape[1] != 3:
            raise ValueError("receivers 形状应为 (nr,3)")
        self.nt = int(self.nt)
        self.dt = float(self.dt)
        self.wavelet = np.asarray(self.wavelet, dtype=np.float64).reshape(-1)
        if self.wavelet.size != self.nt:
            raise ValueError("wavelet 长度与 nt 不一致")
        if self.source_kind not in ("explosive", "force"):
            raise ValueError("source_kind 只能是 explosive/force")

    @property
    def n_shots(self) -> int:
        return int(self.shots.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])


@dataclass
class TrilinearWeights:
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    w: np.ndarray

    @property
    def n(self) -> int:
        return self.i.shape[1]


def _trilinear(grid: Grid3D, pos: np.ndarray, offsets: tuple[float, float, float]) -> TrilinearWeights:
    pad = grid.pad
    x = np.asarray(pos, dtype=np.float64)[:, 0]
    y = np.asarray(pos, dtype=np.float64)[:, 1]
    z = np.asarray(pos, dtype=np.float64)[:, 2]
    fi = x / grid.dx + pad + offsets[0]
    fj = y / grid.dy + pad + offsets[1]
    fk = z / grid.dz + pad + offsets[2]
    NX, NY, NZ = grid.nx + 2 * pad, grid.ny + 2 * pad, grid.nz + 2 * pad
    i0 = np.clip(np.floor(fi).astype(np.int64), 1, NX - 3)
    j0 = np.clip(np.floor(fj).astype(np.int64), 1, NY - 3)
    k0 = np.clip(np.floor(fk).astype(np.int64), 1, NZ - 3)
    wx = fi - i0
    wy = fj - j0
    wz = fk - k0
    idx = []
    w = []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                idx.append((i0 + di, j0 + dj, k0 + dk))
                w.append(
                    (wx if di else 1 - wx) * (wy if dj else 1 - wy) * (wz if dk else 1 - wz)
                )
    ii = np.stack([a[0] for a in idx], axis=0)
    jj = np.stack([a[1] for a in idx], axis=0)
    kk = np.stack([a[2] for a in idx], axis=0)
    ww = np.stack(w, axis=0)
    wsum = ww.sum(axis=0)
    ww = ww / np.where(wsum > 1e-12, wsum, 1.0)
    return TrilinearWeights(ii, jj, kk, ww)
