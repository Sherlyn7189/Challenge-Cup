"""地形、高程静校正与自由表面辅助。

实际陆上资料通常有起伏地表。真正的不规则自由表面需要在每个 x 列改变
差分模板；本模块提供两条工程上常用、可验证的路径：

1. ``Topography``：由地表高程计算炮点/检波点高程静校正，并对炮集做
   亚采样时移（``apply_elevation_statics``）；
2. ``fill_air_with_replacement``：把地表以上填充为替换速度，在规则网格上
   做浮动 datum 正演/反演。

配合 ``Elastic2DOperator(free_surface=True)`` 的平自由表面边界，可覆盖
“平地表自由表面 + 起伏地形静校正”的实际资料流程。真正的逐列不规则
自由表面仍属于后续工作（见 docs）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .io import SegyGathers
from .model import ElasticModel


@dataclass
class Topography:
    x: np.ndarray
    z: np.ndarray  # 地表深度，正向下；也可传高程，只要与 datum 一致

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        self.z = np.asarray(self.z, dtype=np.float64).reshape(-1)
        if self.x.size != self.z.size or self.x.size < 2:
            raise ValueError("地形 x/z 长度必须一致且至少 2 个点")
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.z = self.z[order]
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("地形 x 必须严格递增")

    @classmethod
    def flat(cls, x: np.ndarray, z0: float) -> "Topography":
        x = np.asarray(x, dtype=np.float64)
        return cls(x, np.full_like(x, float(z0)))

    def at(self, x: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=np.float64), self.x, self.z,
                         left=self.z[0], right=self.z[-1])

    def statics(
        self,
        positions: np.ndarray,
        *,
        datum: float,
        replacement_velocity: float = 1500.0,
    ) -> np.ndarray:
        """高程静校正量（秒），正值为延迟。

        ``statics = (datum - z_surface) / v_replacement``
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError("positions 形状应为 (n,2+)")
        zs = self.at(pos[:, 0])
        return (float(datum) - zs) / max(float(replacement_velocity), 1e-9)

    def surface_indices(self, grid: Any) -> np.ndarray:
        """每个 x 列的自由表面 z 索引（物理网格，向下）。"""
        z = self.at(np.asarray(grid.x, dtype=np.float64))
        idx = np.round(z / grid.dz).astype(np.int64)
        return np.clip(idx, 0, grid.nz - 1)


def apply_elevation_statics(
    gathers: SegyGathers,
    topography: Topography,
    *,
    datum: float,
    replacement_velocity: float = 1500.0,
    source_sign: float = 1.0,
    receiver_sign: float = 1.0,
) -> tuple[SegyGathers, dict[str, Any]]:
    """对炮集做炮点+检波点高程静校正（亚采样时移）。

    正静校正量表示向下移动（延迟）。每条道的总静校正为
    ``source_sign*t_s + receiver_sign*t_r``；用线性插值实现小数采样移动，
    移出记录范围的部分补零。
    """
    s_static = topography.statics(gathers.shots, datum=datum, replacement_velocity=replacement_velocity)
    r_static = topography.statics(gathers.receivers, datum=datum, replacement_velocity=replacement_velocity)
    total = source_sign * s_static[:, None] + receiver_sign * r_static[None, :]
    shift_samples = total / gathers.dt

    data = gathers.data.copy()
    ns, nt, nr, nc = data.shape
    t = np.arange(nt, dtype=np.float64)
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                if not gathers.mask[s, r, c]:
                    continue
                # output(t) = input(t - shift)：shift>0 表示延迟
                src = t - shift_samples[s, r]
                data[s, :, r, c] = np.interp(src, t, data[s, :, r, c], left=0.0, right=0.0)
    out = SegyGathers(
        shots=gathers.shots.copy(),
        receivers=gathers.receivers.copy(),
        data=data,
        dt=gathers.dt,
        mask=gathers.mask.copy(),
        shot_ids=gathers.shot_ids.copy(),
        receiver_ids=gathers.receiver_ids.copy(),
        text_header=gathers.text_header,
        meta=dict(gathers.meta),
        component_ids=gathers.component_ids,
    )
    qc = {
        "source_static_range_s": [float(s_static.min()), float(s_static.max())],
        "receiver_static_range_s": [float(r_static.min()), float(r_static.max())],
        "total_static_range_s": [float(total.min()), float(total.max())],
        "datum": float(datum),
        "replacement_velocity": float(replacement_velocity),
    }
    return out, qc


def fill_air_with_replacement(
    model: ElasticModel,
    topography: Topography,
    *,
    replacement_velocity: float = 1500.0,
    replacement_vs: float | None = None,
    replacement_rho: float | None = None,
) -> ElasticModel:
    """把地表以上单元填充为替换速度（浮动 datum 建模）。

    对规则网格逐列使用地表深度；仅用于静校正/浮动 datum 近似，不是真正的
    不规则自由表面。
    """
    z = np.arange(model.nz, dtype=np.float64) * (model.dz or 1.0)
    x = np.arange(model.nx, dtype=np.float64) * (model.dx or 1.0)
    surf = topography.at(x)
    air = z[None, :] < surf[:, None]  # (nx,nz)
    vp = np.where(air, replacement_velocity, model.vp)
    vs0 = replacement_vs if replacement_vs is not None else replacement_velocity / 1.8
    rho0 = replacement_rho if replacement_rho is not None else 310.0 * replacement_velocity ** 0.25
    vs = np.where(air, vs0, model.vs)
    rho = np.where(air, rho0, model.rho)
    return model.with_params(vp=vp, vs=np.minimum(vs, 0.95 * vp), rho=rho)
