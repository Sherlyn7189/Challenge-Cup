"""合成模型与演示观测系统。

模型包含：速度随深度压实、三个倾斜界面、一个低速透镜体、一条断层，
以及可选的 VTI 各向异性层。用于 EFWI/RTM 的端到端示例和回归测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridConfig
from .geometry import Survey
from .model import ElasticModel, gardner_density


@dataclass
class DemoSetup:
    grid: GridConfig
    true_model: ElasticModel
    initial_model: ElasticModel
    survey: Survey


def demo_true_model(
    grid: GridConfig,
    *,
    anisotropy: str = "iso",
    epsilon: float = 0.0,
    delta: float = 0.0,
    gamma: float = 0.0,
) -> ElasticModel:
    x = grid.x
    z = grid.z
    X, Z = np.meshgrid(x, z, indexing="ij")

    vp = 1850.0 + 1.65 * Z

    # 三个界面，带倾角和断层错断
    def depth(z0: float, dip: float) -> np.ndarray:
        d = z0 + dip * (X - 0.5 * x[-1])
        return d + 45.0 * np.tanh((X - 0.62 * x[-1]) / 25.0)

    d1 = depth(330.0, 0.18)
    d2 = depth(560.0, -0.12)
    d3 = depth(820.0, 0.06)
    vp += 170.0 * np.tanh((Z - d1) / 18.0)
    vp -= 140.0 * np.tanh((Z - d2) / 20.0)
    vp += 230.0 * np.tanh((Z - d3) / 16.0)

    # 低速透镜体
    vp -= 210.0 * np.exp(-(((X - 0.45 * x[-1]) ** 2 + (Z - 0.58 * z[-1]) ** 2) / (2 * 75.0 ** 2)))
    vp = np.maximum(vp, 1500.0)

    vs = vp / 1.85
    rho = gardner_density(vp)
    eps = np.full(grid.shape, float(epsilon))
    dlt = np.full(grid.shape, float(delta))
    gam = np.full(grid.shape, float(gamma))
    if anisotropy != "iso":
        # 深层页岩强各向异性：SH/SV 速度差约 15%
        shale = 0.5 * (1.0 + np.tanh((Z - d2) / 40.0))
        eps = eps + 0.16 * shale
        dlt = dlt + 0.08 * shale
        gam = gam + 0.15 * shale
    return ElasticModel(vp, vs, rho, eps, dlt, gam, anisotropy, grid.dx, grid.dz)


def demo_initial_model(true_model: ElasticModel, *, sigma: float = 9.0) -> ElasticModel:
    """平滑真模型 + 一维梯度趋势，作为 FWI 初始模型（不含界面细节）。"""
    smooth = true_model.smoothed(sigma=sigma, keys=("vp", "vs", "rho"))
    z = np.arange(true_model.nz) * (true_model.dz or 1.0)
    vp1d = 1850.0 + 1.65 * z
    vp1d = np.repeat(vp1d[None, :], true_model.nx, axis=0)
    vp = 0.55 * smooth.vp + 0.45 * vp1d
    vs = vp / 1.85
    rho = gardner_density(vp)
    vs = np.minimum(vs, 0.95 * vp)
    return ElasticModel(
        vp,
        vs,
        rho,
        np.asarray(true_model.epsilon).copy(),
        np.asarray(true_model.delta).copy(),
        np.asarray(true_model.gamma).copy(),
        true_model.symmetry,
        true_model.dx,
        true_model.dz,
    )


def demo_survey(
    grid: GridConfig,
    *,
    n_shots: int = 5,
    n_receivers: int = 61,
    f0: float = 10.0,
    source_depth: float | None = None,
    vmin: float = 1600.0,
    vmax: float = 4200.0,
    dt: float | None = None,
    nt: int | None = None,
) -> Survey:
    """沿一条水平测线布置震源和检波器（位于吸收层以内）。"""
    margin = 2.0 * grid.sponge * grid.dx
    x_min = margin
    x_max = (grid.nx - 1) * grid.dx - margin
    if x_max <= x_min:
        raise ValueError("模型太窄，无法布置测线")
    if source_depth is None:
        source_depth = grid.sponge * grid.dz + 2.0 * grid.dz
    shots = np.stack(
        [np.linspace(x_min, x_max, n_shots), np.full(n_shots, source_depth)], axis=1
    )
    receivers = np.stack(
        [np.linspace(x_min, x_max, n_receivers), np.full(n_receivers, source_depth)], axis=1
    )
    return Survey.from_positions(
        grid, shots, receivers, f0=f0, vmin=vmin, vmax=vmax, dt=dt, nt=nt,
        source_kind="explosive",
    )


def demo_setup(
    *,
    quick: bool = True,
    anisotropy: str = "iso",
    n_shots: int | None = None,
    n_receivers: int | None = None,
) -> DemoSetup:
    if quick:
        grid = GridConfig(81, 61, 15.0, 15.0, sponge=10)
        n_shots = 4 if n_shots is None else n_shots
        n_receivers = 41 if n_receivers is None else n_receivers
        f0 = 12.0
    else:
        grid = GridConfig(121, 101, 10.0, 10.0, sponge=16)
        n_shots = 6 if n_shots is None else n_shots
        n_receivers = 81 if n_receivers is None else n_receivers
        f0 = 10.0
    true = demo_true_model(grid, anisotropy=anisotropy)
    initial = demo_initial_model(true, sigma=8.0 if quick else 9.0)
    survey = demo_survey(
        grid, n_shots=n_shots, n_receivers=n_receivers, f0=f0,
        vmin=1600.0, vmax=4300.0,
    )
    return DemoSetup(grid, true, initial, survey)
