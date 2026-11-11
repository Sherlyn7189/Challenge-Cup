"""“反演—成像—地质建模”闭环。

文档 2.2.6 第 5 点给出的闭环::

    初始速度模型
      -> 正演模拟
      -> 与实测波形求残差
      -> EFWI 更新模型
      -> 用新模型做 RTM 保幅成像
      -> 从成像结果提取反射点与倾角
      -> 约束下三维建模
      -> 建模结果再反馈为正演模型

本模块把该流程串起来；三维建模采用“倾角引导的走向外推”：沿 y 方向
按每层平均倾角对二维剖面做横向错动，得到符合成像倾角趋势的三维体积，
再取中心剖面反馈给下一轮 EFWI。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import shift as nd_shift

from .config import FWIConfig, GridConfig
from .imaging import (
    DipField,
    RTMResult,
    depth_taper,
    extract_dips,
    reverse_time_migration,
    top_mute,
)
from .inversion import ElasticFWI, FWIResult, save_result
from .model import ElasticModel
from .viz import plot_dip, plot_history, plot_model_panel, plot_rtm


# ----------------------------------------------------------------------
# 三维模型
# ----------------------------------------------------------------------
@dataclass
class Model3D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    epsilon: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    symmetry: str = "iso"

    def central_slice(self) -> ElasticModel:
        iy = self.y.size // 2
        dx = float(self.x[1] - self.x[0]) if self.x.size > 1 else None
        dz = float(self.z[1] - self.z[0]) if self.z.size > 1 else None
        return ElasticModel(
            self.vp[:, iy, :],
            self.vs[:, iy, :],
            self.rho[:, iy, :],
            self.epsilon[:, iy, :],
            self.delta[:, iy, :],
            self.gamma[:, iy, :],
            self.symmetry,
            dx,
            dz,
        )

    def save_npz(self, path: str) -> str:
        np.savez_compressed(
            path,
            x=self.x, y=self.y, z=self.z,
            vp=self.vp, vs=self.vs, rho=self.rho,
            epsilon=self.epsilon, delta=self.delta, gamma=self.gamma,
            symmetry=np.array(self.symmetry),
        )
        return path

    def save_vtk(self, path: str) -> str | None:
        try:
            import pyvista as pv
        except Exception:
            return None
        grid = pv.ImageData(
            dimensions=(self.x.size, self.y.size, self.z.size),
            spacing=(
                float(self.x[1] - self.x[0]) if self.x.size > 1 else 1.0,
                float(self.y[1] - self.y[0]) if self.y.size > 1 else 1.0,
                float(self.z[1] - self.z[0]) if self.z.size > 1 else 1.0,
            ),
            origin=(float(self.x[0]), float(self.y[0]), float(self.z[0])),
        )
        for key in ("vp", "vs", "rho", "epsilon", "delta", "gamma"):
            grid.point_data[key] = np.asarray(getattr(self, key)).ravel(order="F")
        grid.save(path)
        return path


def _weighted_dip_by_depth(dip: DipField) -> np.ndarray:
    """按深度对成像倾角做相干加权平均，作为三维外推的错动趋势。"""
    z = np.arange(dip.dip.shape[1])
    w = dip.coherence * dip.reflection_mask.astype(np.float64)
    out = np.zeros(dip.dip.shape[1], dtype=np.float64)
    for j in range(dip.dip.shape[1]):
        wj = w[:, j]
        if np.sum(wj) > 1e-9:
            out[j] = float(np.sum(dip.dip[:, j] * wj) / np.sum(wj))
    return out


def build_3d_model(
    model: ElasticModel,
    dip: np.ndarray | DipField,
    *,
    ny: int = 21,
    dy: float = 20.0,
    max_shift_cells: float = 12.0,
    lateral_variation: float = 0.0,
    seed: int = 0,
) -> Model3D:
    """由二维模型 + 倾角场构建三维模型（倾角引导的走向外推）。"""
    dip_field = dip if isinstance(dip, DipField) else None
    if dip_field is not None:
        dip_z = _weighted_dip_by_depth(dip_field)
    else:
        arr = np.asarray(dip, dtype=np.float64)
        dip_z = np.mean(arr, axis=0) if arr.ndim == 2 else np.zeros(model.nz)
    x = np.arange(model.nx) * (model.dx or 1.0)
    z = np.arange(model.nz) * (model.dz or 1.0)
    y = (np.arange(ny) - ny // 2) * dy
    dx = model.dx or 1.0

    rng = np.random.default_rng(seed)
    vp = np.empty((model.nx, ny, model.nz), dtype=np.float64)
    vs = np.empty_like(vp)
    rho = np.empty_like(vp)
    eps = np.empty_like(vp)
    dlt = np.empty_like(vp)
    gam = np.empty_like(vp)

    tan_dip = np.tan(np.clip(dip_z, -1.2, 1.2))
    for iy, yy in enumerate(y):
        # 沿 y 方向按倾角做横向错动：模拟三维构造的侧向变化
        shift_cells_z = np.clip(tan_dip * yy / dx, -max_shift_cells, max_shift_cells)
        for key, dst in (
            ("vp", vp), ("vs", vs), ("rho", rho),
            ("epsilon", eps), ("delta", dlt), ("gamma", gam),
        ):
            src = np.asarray(getattr(model, key), dtype=np.float64)
            out = np.empty_like(src)
            for j in range(model.nz):
                out[:, j] = np.interp(
                    np.arange(model.nx) - shift_cells_z[j],
                    np.arange(model.nx),
                    src[:, j],
                    left=src[0, j],
                    right=src[-1, j],
                )
            if lateral_variation > 0:
                mod = 1.0 + lateral_variation * np.sin(
                    2.0 * np.pi * yy / max(y[-1] - y[0], 1e-9) + rng.uniform(0, 2 * np.pi)
                )
                if key in ("vp", "vs", "rho"):
                    out = out * mod
            dst[:, iy, :] = out

    vs = np.minimum(vs, 0.95 * vp)
    rho = np.clip(rho, 1000.0, 3500.0)
    return Model3D(x, y, z, vp, vs, rho, eps, dlt, gam, model.symmetry)


# ----------------------------------------------------------------------
# 闭环
# ----------------------------------------------------------------------
@dataclass
class WorkflowConfig:
    fwi: FWIConfig = field(default_factory=FWIConfig)
    n_outer: int = 1
    rtm_wavefield: str = "p"
    rtm_imaging: str = "source_normalized"
    rtm_storage_mb: float = 256.0
    top_mute_velocity: float | None = 1600.0
    depth_mute: tuple[float, float] | None = None
    build_3d: bool = True
    ny: int = 21
    dy: float = 20.0
    outdir: str = "runs/closed_loop"
    save_figures: bool = True
    save_vtk: bool = True


@dataclass
class OuterRecord:
    outer: int
    fwi: FWIResult
    rtm: RTMResult | None
    dip: DipField | None
    model3d: Model3D | None
    paths: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    final_model: ElasticModel
    records: list[OuterRecord]
    outdir: str
    metrics: dict[str, Any] = field(default_factory=dict)


class ClosedLoopWorkflow:
    def __init__(
        self,
        initial_model: ElasticModel,
        survey,
        *,
        true_model: ElasticModel | None = None,
        observed: np.ndarray | None = None,
        grid: GridConfig | None = None,
        config: WorkflowConfig | None = None,
        verbose: bool = True,
    ) -> None:
        self.initial_model = initial_model.copy()
        self.survey = survey
        self.true_model = true_model.copy() if true_model is not None else None
        self.observed = observed
        self.config = config or WorkflowConfig()
        self.verbose = verbose
        if grid is None:
            if initial_model.dx is None or initial_model.dz is None:
                raise ValueError("缺少网格信息")
            grid = GridConfig(initial_model.nx, initial_model.nz, initial_model.dx, initial_model.dz)
        self.grid = grid

    def run(self) -> WorkflowResult:
        outdir = self.config.outdir
        os.makedirs(outdir, exist_ok=True)
        records: list[OuterRecord] = []
        model = self.initial_model.copy()
        observed = self.observed

        for outer in range(self.config.n_outer):
            if self.verbose:
                print(f"\n########## 闭环第 {outer + 1}/{self.config.n_outer} 轮 ##########")
            fwi = ElasticFWI(
                model,
                self.survey,
                true_model=self.true_model if observed is None else None,
                observed=observed,
                grid=self.grid,
                config=self.config.fwi,
                verbose=self.verbose,
            )
            if observed is None:
                observed = fwi.observed
            fwi_result = fwi.run()
            model = fwi_result.model
            outer_dir = os.path.join(outdir, f"outer_{outer + 1:02d}")
            paths = {"fwi_dir": save_result(fwi_result, os.path.join(outer_dir, "fwi"))}
            if self.config.save_figures:
                paths["model_panel"] = plot_model_panel(
                    [self.initial_model, model] + ([self.true_model] if self.true_model else []),
                    self.grid,
                    os.path.join(outer_dir, "models.png"),
                    titles=["initial", "FWI"] + (["true"] if self.true_model else []),
                )
                paths["history"] = plot_history(fwi_result.history, os.path.join(outer_dir, "convergence.png"))

            rtm = None
            dip = None
            model3d = None
            data_rtm = observed
            if self.config.top_mute_velocity:
                data_rtm = top_mute(
                    data_rtm,
                    self.survey.dt,
                    self.survey.shots,
                    self.survey.receivers,
                    self.config.top_mute_velocity,
                )
            rtm = reverse_time_migration(
                model,
                self.survey,
                data_rtm,
                grid=self.grid,
                wavefield=self.config.rtm_wavefield,
                imaging=self.config.rtm_imaging,
                storage_mb=self.config.rtm_storage_mb,
                verbose=self.verbose,
            )
            if self.config.depth_mute is not None:
                z0, z1 = self.config.depth_mute
                rtm.image = depth_taper(rtm.image, self.grid.z, z0, z1)
            dip = extract_dips(rtm.image, self.grid.dx, self.grid.dz, sigma=2.0)
            np.savez_compressed(
                os.path.join(outer_dir, "rtm.npz"),
                image=rtm.image,
                illumination=rtm.illumination,
                dip=dip.dip,
                coherence=dip.coherence,
                reflection_mask=dip.reflection_mask,
            )
            if self.config.save_figures:
                paths["rtm"] = plot_rtm(
                    rtm.image, self.grid, os.path.join(outer_dir, "rtm.png"),
                    title=f"RTM (outer {outer + 1})",
                )
                paths["dip"] = plot_dip(
                    np.degrees(dip.dip), dip.coherence, self.grid,
                    os.path.join(outer_dir, "dips.png"),
                )

            if self.config.build_3d:
                model3d = build_3d_model(
                    model, dip,
                    ny=self.config.ny, dy=self.config.dy,
                    seed=self.config.fwi.seed,
                )
                paths["model3d"] = model3d.save_npz(os.path.join(outer_dir, "model3d.npz"))
                if self.config.save_vtk:
                    vtk_path = model3d.save_vtk(os.path.join(outer_dir, "model3d.vti"))
                    if vtk_path:
                        paths["model3d_vtk"] = vtk_path
                # 反馈：用三维模型的中心剖面作为下一轮初始模型
                model = model3d.central_slice()

            records.append(OuterRecord(outer, fwi_result, rtm, dip, model3d, paths))
            if self.verbose:
                print(f"  输出目录: {outer_dir}")

        metrics = {
            "n_outer": len(records),
            "final_misfit": records[-1].fwi.history[-1]["misfit"] if records else float("nan"),
            "final_vp_range": [float(model.vp.min()), float(model.vp.max())],
            "final_vs_range": [float(model.vs.min()), float(model.vs.max())],
            "final_rho_range": [float(model.rho.min()), float(model.rho.max())],
        }
        with open(os.path.join(outdir, "workflow_metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        return WorkflowResult(model, records, outdir, metrics)
