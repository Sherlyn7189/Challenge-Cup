"""观测系统、双线性插值权重与震源/检波器布置。

交错网格上的场量位置：

* ``vx``：``(i+1/2, j)``
* ``vz``：``(i, j+1/2)``
* ``sxx/szz``：``(i, j)``
* ``sxz``：``(i+1/2, j+1/2)``

震源/检波器坐标以米为单位，双线性权重在初始化时算好，正演循环里只做
加权求和/散加，避免重复的坐标查找。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GridConfig
from .wavelet import make_wavelet


@dataclass
class BilinearWeights:
    """四个相邻网格点的索引与权重，形状均为 ``(4, n)``。"""

    i: np.ndarray
    j: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        self.i = np.asarray(self.i, dtype=np.int64)
        self.j = np.asarray(self.j, dtype=np.int64)
        self.w = np.asarray(self.w, dtype=np.float64)
        if self.i.shape != self.j.shape or self.i.shape != self.w.shape:
            raise ValueError("插值权重形状不一致")
        if self.i.ndim != 2 or self.i.shape[0] != 4:
            raise ValueError("插值权重形状应为 (4, n)")

    @property
    def n(self) -> int:
        return self.i.shape[1]


def _bilinear_weights(
    grid: GridConfig,
    positions: np.ndarray,
    x_offset: float,
    z_offset: float,
) -> BilinearWeights:
    x = np.asarray(positions, dtype=np.float64)[:, 0]
    z = np.asarray(positions, dtype=np.float64)[:, 1]
    pad = grid.pad
    nx_p = grid.nx + 2 * pad
    nz_p = grid.nz + 2 * pad

    fi = x / grid.dx + pad + x_offset
    fj = z / grid.dz + pad + z_offset
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    # 保证四邻点都落在有效更新区域（1..N-2）
    i0 = np.clip(i0, 1, nx_p - 3)
    j0 = np.clip(j0, 1, nz_p - 3)
    wx = fi - i0
    wz = fj - j0

    ii = np.stack([i0, i0 + 1, i0, i0 + 1], axis=0)
    jj = np.stack([j0, j0, j0 + 1, j0 + 1], axis=0)
    ww = np.stack(
        [(1 - wx) * (1 - wz), wx * (1 - wz), (1 - wx) * wz, wx * wz], axis=0
    )
    # 边缘位置被裁剪后权重和可能略微偏离 1，重新归一化
    wsum = ww.sum(axis=0)
    ww = ww / np.where(wsum > 1e-12, wsum, 1.0)
    return BilinearWeights(ii, jj, ww)


@dataclass
class Survey:
    """炮集观测系统。

    ``wavelet`` 长度为 ``nt``；正演时第 n 步注入 ``wavelet[n]``。
    """

    shots: np.ndarray
    receivers: np.ndarray
    nt: int
    dt: float
    wavelet: np.ndarray
    source_kind: str = "explosive"
    amplitude: float = 1e6
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shots = np.asarray(self.shots, dtype=np.float64)
        self.receivers = np.asarray(self.receivers, dtype=np.float64)
        if self.shots.ndim != 2 or self.shots.shape[1] != 2:
            raise ValueError("shots 形状应为 (ns, 2)")
        if self.receivers.ndim != 2 or self.receivers.shape[1] != 2:
            raise ValueError("receivers 形状应为 (nr, 2)")
        self.nt = int(self.nt)
        self.dt = float(self.dt)
        if self.nt <= 0 or self.dt <= 0:
            raise ValueError("nt/dt 必须为正")
        self.wavelet = np.asarray(self.wavelet, dtype=np.float64).reshape(-1)
        if self.wavelet.size != self.nt:
            raise ValueError(f"wavelet 长度 {self.wavelet.size} 与 nt={self.nt} 不一致")
        self.source_kind = str(self.source_kind).lower()
        if self.source_kind not in ("explosive", "force"):
            raise ValueError("source_kind 只能是 explosive/force")
        if self.amplitude <= 0:
            raise ValueError("amplitude 必须为正")

    @property
    def n_shots(self) -> int:
        return int(self.shots.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    @property
    def duration(self) -> float:
        return (self.nt - 1) * self.dt

    def validate_against_grid(self, grid: GridConfig, *, allow_sponge: bool = False) -> None:
        for name, pos in (("shots", self.shots), ("receivers", self.receivers)):
            x, z = pos[:, 0], pos[:, 1]
            if np.any(x < 0) or np.any(x > (grid.nx - 1) * grid.dx):
                raise ValueError(f"{name} 的 x 超出模型范围")
            if np.any(z < 0) or np.any(z > (grid.nz - 1) * grid.dz):
                raise ValueError(f"{name} 的 z 超出模型范围")
            if not allow_sponge:
                sx = grid.sponge * grid.dx
                sz = grid.sponge * grid.dz
                inside = (
                    (x >= sx - 1e-9)
                    & (x <= (grid.nx - 1) * grid.dx - sx + 1e-9)
                    & (z >= sz - 1e-9)
                    & (z <= (grid.nz - 1) * grid.dz - sz + 1e-9)
                )
                if not np.all(inside):
                    raise ValueError(
                        f"{name} 落在吸收层内，会被强烈衰减；请移到 sponge 以内，"
                        "或显式设置 allow_sponge=True"
                    )

    # ------------------------------------------------------------------
    def source_weights(self, grid: GridConfig) -> dict[str, BilinearWeights]:
        if self.source_kind == "explosive":
            return {"stress": _bilinear_weights(grid, self.shots, 0.0, 0.0)}
        # 力源作用在速度分量上：vx 在 (i+1/2,j)，vz 在 (i,j+1/2)
        return {
            "vx": _bilinear_weights(grid, self.shots, -0.5, 0.0),
            "vz": _bilinear_weights(grid, self.shots, 0.0, -0.5),
        }

    def receiver_weights(self, grid: GridConfig) -> dict[str, BilinearWeights]:
        return {
            "vx": _bilinear_weights(grid, self.receivers, -0.5, 0.0),
            "vz": _bilinear_weights(grid, self.receivers, 0.0, -0.5),
        }

    def subset_shots(self, indices: Any) -> "Survey":
        idx = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        return Survey(
            self.shots[idx].copy(),
            self.receivers.copy(),
            self.nt,
            self.dt,
            self.wavelet.copy(),
            self.source_kind,
            self.amplitude,
            dict(self.meta),
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_positions(
        cls,
        grid: GridConfig,
        shots: np.ndarray,
        receivers: np.ndarray,
        *,
        f0: float = 10.0,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        dt: float | None = None,
        nt: int | None = None,
        record_time: float | None = None,
        source_kind: str = "explosive",
        amplitude: float = 1e6,
        wavelet_kind: str = "ricker",
        allow_sponge: bool = False,
    ) -> "Survey":
        """根据最大炮检距自动设计记录长度。

        记录长度取 ``1.5 * max_offset / vmin + 2/f0``，保证最深/最远
        反射体到时能被记录到。
        """
        shots = np.asarray(shots, dtype=np.float64)
        receivers = np.asarray(receivers, dtype=np.float64)
        if dt is None:
            dt = grid.cfl_dt(vmax)
        if nt is None:
            if record_time is None:
                dx = shots[:, None, 0] - receivers[None, :, 0]
                dz = shots[:, None, 1] - receivers[None, :, 1]
                max_offset = float(np.max(np.sqrt(dx * dx + dz * dz)))
                record_time = 1.5 * max_offset / max(vmin, 1e-6) + 2.0 / f0
            nt = int(np.ceil(record_time / dt)) + 1
        wavelet = make_wavelet(nt, dt, f0, wavelet_kind)
        survey = cls(shots, receivers, nt, dt, wavelet, source_kind, amplitude)
        survey.validate_against_grid(grid, allow_sponge=allow_sponge)
        return survey
