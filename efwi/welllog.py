"""测井数据、初始模型构建与尺度换算。

测井场景的有利条件（文档 2.2.6 第 4 点）：

* 信噪比高、径向探测范围仅数十米，计算规模可控；
* 可由时差曲线延拓得到较准确初始速度模型，回避地面地震 FWI 的
  初始模型难题；
* 测井主频 kHz 量级，地面地震约 30 Hz，网格/时间步长必须按尺度换算；
* 四川深层页岩强各向异性，需要 VTI/HTI 扩展。

本模块提供 LAS 解析（无第三方依赖）、曲线延拓/平滑、井间横向插值、
VTI/HTI 参数估计和网格设计。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .config import GridConfig
from .model import ElasticModel, gardner_density

_NULL_VALUES = {-999.25, -999.0, -9999.0, -9999.25, 1e30}


# ----------------------------------------------------------------------
# LAS 解析
# ----------------------------------------------------------------------
def _parse_curve_header(line: str) -> tuple[str, str]:
    """解析 ``DTC .US/F : description`` 形式的曲线头。"""
    head = line.split(":")[0]
    m = re.match(r"\s*([A-Za-z0-9_\-]+)\s*(?:\.\s*([^\s]+))?", head)
    if not m:
        return "", ""
    return m.group(1).upper(), (m.group(2) or "").upper()


def read_las(path: str) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    """读取 LAS 2.0/3.0 的 ASCII 段，返回 ``(curves, units, header)``。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    curves: dict[str, list[float]] = {}
    units: dict[str, str] = {}
    header: dict[str, str] = {}
    in_curves = False
    in_ascii = False
    null_value = -999.25
    order: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("~C"):
            in_curves, in_ascii = True, False
            continue
        if upper.startswith("~A"):
            in_curves, in_ascii = False, True
            continue
        if upper.startswith("~"):
            in_curves, in_ascii = False, False
            continue
        if in_curves:
            if ":" in stripped and not stripped.startswith("#"):
                key_part, _, desc = stripped.partition(":")
                key_part = key_part.strip()
                if "." in key_part:
                    name, unit = _parse_curve_header(stripped)
                    if name:
                        order.append(name)
                        units[name] = unit
                        curves.setdefault(name, [])
                        continue
                if key_part.upper().startswith("NULL"):
                    try:
                        null_value = float(desc.strip().split()[0])
                    except Exception:
                        pass
                if key_part.upper().startswith("STRT") or key_part.upper().startswith("STOP"):
                    header[key_part.upper()] = desc.strip()
            else:
                name, unit = _parse_curve_header(stripped)
                if name:
                    order.append(name)
                    units[name] = unit
                    curves.setdefault(name, [])
            continue
        if in_ascii:
            if stripped.startswith("#") or stripped.startswith("~"):
                continue
            parts = re.split(r"[\s,]+", stripped)
            vals = []
            ok = True
            for p in parts:
                if not p:
                    continue
                try:
                    vals.append(float(p))
                except ValueError:
                    ok = False
                    break
            if not ok or not vals:
                continue
            for name, val in zip(order, vals):
                curves[name].append(val)

    out: dict[str, np.ndarray] = {}
    for name, vals in curves.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr[~np.isfinite(arr)] = np.nan
        for null in _NULL_VALUES:
            arr[arr == null] = np.nan
        out[name] = arr
    return out, units, header


def _unit_multiplier(unit: str, values: np.ndarray, kind: str) -> float:
    unit = unit.upper().replace(" ", "")
    if kind == "slowness":
        if "/FT" in unit or "/F" in unit:
            return 1.0  # us/ft
        if "/M" in unit:
            return 0.3048  # us/m -> us/ft
        med = np.nanmedian(values) if values.size else 100.0
        return 1.0 if med < 220 else 0.3048
    if kind == "density":
        if "K/M3" in unit or "KG/M3" in unit:
            return 1.0
        if "G/C" in unit or "G/CC" in unit:
            return 1000.0
        med = np.nanmedian(values) if values.size else 2.3
        return 1000.0 if med < 10 else 1.0
    return 1.0


@dataclass
class WellLog:
    depth: np.ndarray
    vp: np.ndarray | None = None
    vs: np.ndarray | None = None
    rho: np.ndarray | None = None
    dtc: np.ndarray | None = None
    dts: np.ndarray | None = None
    rhob: np.ndarray | None = None
    x: float = 0.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float64).reshape(-1)
        if self.depth.size < 3:
            raise ValueError("测井曲线点数太少")
        order = np.argsort(self.depth)
        self.depth = self.depth[order]
        for key in ("vp", "vs", "rho", "dtc", "dts", "rhob"):
            val = getattr(self, key, None)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).reshape(-1)
                if arr.size != self.depth.size:
                    raise ValueError(f"{key} 长度与 depth 不一致")
                setattr(self, key, arr[order])
        self._derive()

    def _derive(self) -> None:
        dtc = self.dtc
        dts = self.dts
        rhob = self.rhob
        if self.vp is None and dtc is not None:
            # 统一换算成 us/ft
            dtc_ft = dtc * _unit_multiplier(
                str(self.meta.get("dtc_unit", "")), dtc, "slowness"
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                self.vp = np.where(dtc_ft > 0, 304800.0 / dtc_ft, np.nan)
        if self.vs is None and dts is not None:
            dts_ft = dts * _unit_multiplier(
                str(self.meta.get("dts_unit", "")), dts, "slowness"
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                self.vs = np.where(dts_ft > 0, 304800.0 / dts_ft, np.nan)
        if self.rho is None and rhob is not None:
            self.rho = rhob * _unit_multiplier(
                str(self.meta.get("rhob_unit", "")), rhob, "density"
            )
        if self.vp is not None:
            self.vp = _fill_nan(self.depth, self.vp)
        if self.vs is None and self.vp is not None:
            self.vs = self.vp / 1.8
        if self.rho is None and self.vp is not None:
            self.rho = gardner_density(self.vp)
        if self.vs is not None:
            self.vs = _fill_nan(self.depth, self.vs)
        if self.rho is not None:
            self.rho = _fill_nan(self.depth, self.rho)
            self.rho = np.clip(self.rho, 1000.0, 3500.0)
        if self.vp is not None and self.vs is not None:
            self.vs = np.minimum(self.vs, 0.95 * self.vp)

    @classmethod
    def from_las(cls, path: str, *, x: float = 0.0, name: str | None = None) -> "WellLog":
        curves, units, header = read_las(path)
        if not curves:
            raise ValueError(f"LAS 文件没有有效曲线: {path}")
        depth_key = None
        for key in ("DEPT", "DEPTH", "DEP", "MD"):
            if key in curves:
                depth_key = key
                break
        if depth_key is None:
            depth_key = next(iter(curves))
        depth = curves[depth_key]
        dtc = _first(curves, ("DTC", "DT", "AC", "DTC1"))
        dts = _first(curves, ("DTS", "DTSM", "DTSF", "DT4S"))
        rhob = _first(curves, ("RHOB", "RHOZ", "DEN", "RHOB1"))
        meta = {
            "dtc_unit": _first_unit(units, ("DTC", "DT", "AC")),
            "dts_unit": _first_unit(units, ("DTS", "DTSM", "DTSF", "DT4S")),
            "rhob_unit": _first_unit(units, ("RHOB", "RHOZ", "DEN")),
            "header": header,
            "path": os.path.abspath(path),
        }
        obj = cls(
            depth=depth,
            dtc=dtc,
            dts=dts,
            rhob=rhob,
            x=x,
            name=name or os.path.splitext(os.path.basename(path))[0],
            meta=meta,
        )
        return obj

    def resampled(self, z: np.ndarray) -> "WellLog":
        z = np.asarray(z, dtype=np.float64)
        data = {}
        for key in ("vp", "vs", "rho"):
            arr = getattr(self, key)
            if arr is not None:
                data[key] = np.interp(z, self.depth, np.asarray(arr, dtype=np.float64))
        return WellLog(z, x=self.x, name=self.name, meta=dict(self.meta), **data)

    def thomsen(self, *, vsh: np.ndarray | None = None, shale_ratio: float = 1.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """由纵/横波速度估计 Thomsen 参数。

        默认假设 SH 速度比 SV 快 ``shale_ratio``（深层页岩约 15%），
        给出 ``gamma``；``epsilon`` 与 ``gamma`` 同量级，``delta`` 取
        经验比例。
        """
        if self.vs is None:
            raise ValueError("缺少横波速度")
        if vsh is None:
            vsh = self.vs * shale_ratio
        gamma = (vsh ** 2 - self.vs ** 2) / (2.0 * self.vs ** 2)
        epsilon = 0.9 * gamma
        delta = 0.5 * gamma
        return epsilon, delta, gamma

    def describe(self) -> str:
        return (
            f"WellLog({self.name or 'unnamed'}, n={self.depth.size}, "
            f"depth=[{self.depth.min():.1f},{self.depth.max():.1f}] m, "
            f"vp=[{np.nanmin(self.vp):.0f},{np.nanmax(self.vp):.0f}] m/s)"
            if self.vp is not None else f"WellLog({self.name}, n={self.depth.size})"
        )


def _fill_nan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    if mask.sum() == 0:
        raise ValueError("曲线全部为无效值")
    if mask.sum() == 1:
        return np.full_like(y, y[mask][0])
    return np.interp(x, x[mask], y[mask])


def _first(curves: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray | None:
    for n in names:
        if n in curves:
            return curves[n]
    return None


def _first_unit(units: dict[str, str], names: Iterable[str]) -> str:
    for n in names:
        if n in units:
            return units[n]
    return ""


# ----------------------------------------------------------------------
# 初始模型构建
# ----------------------------------------------------------------------
def extrapolate_curve(
    depth: np.ndarray,
    values: np.ndarray,
    z: np.ndarray,
    *,
    positive_trend: bool = True,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """把测井曲线延拓到目标深度网格 ``z``。

    测井段内线性插值；浅于/深于测井段用整体压实趋势线性外推，并保证
    速度/密度为正、随深度不出现负梯度（除非原曲线本身如此）。
    """
    depth = np.asarray(depth, dtype=np.float64)
    values = _fill_nan(depth, values)
    z = np.asarray(z, dtype=np.float64)
    out = np.interp(z, depth, values)
    slope = 0.0
    if depth.size >= 2:
        slope = float(np.polyfit(depth, values, 1)[0])
    if positive_trend:
        slope = max(slope, 0.0)
    above = z < depth[0]
    below = z > depth[-1]
    if np.any(above):
        out[above] = values[0] + slope * (z[above] - depth[0])
    if np.any(below):
        out[below] = values[-1] + slope * (z[below] - depth[-1])
    out = np.maximum(out, 1.0)
    if smooth_sigma > 0:
        out = gaussian_filter1d(out, sigma=smooth_sigma, mode="nearest")
    return out


def build_initial_model(
    logs: list[WellLog],
    grid: GridConfig,
    *,
    datum: float = 0.0,
    smooth_sigma: float = 1.5,
    sigma_xy: float = 2.0,
    anisotropy: str = "iso",
    shale_ratio: float = 1.15,
    epsilon: float | None = None,
    delta: float | None = None,
    gamma: float | None = None,
) -> ElasticModel:
    """由一口或多口测井构建二维初始模型。

    ``datum`` 为模型 z=0 对应的测井深度（缺省 0）。多井时在井间沿 x
    线性插值，井外取最近井曲线。
    """
    if not logs:
        raise ValueError("至少需要一口测井")
    z = np.asarray(grid.z, dtype=np.float64)
    wells = sorted(logs, key=lambda w: w.x)
    vp_curves = []
    vs_curves = []
    rho_curves = []
    for w in wells:
        d = w.depth - datum
        vp_curves.append(extrapolate_curve(d, w.vp, z, smooth_sigma=smooth_sigma))
        vs_curves.append(extrapolate_curve(d, w.vs, z, smooth_sigma=smooth_sigma))
        rho_curves.append(extrapolate_curve(d, w.rho, z, positive_trend=False, smooth_sigma=smooth_sigma))
    xw = np.array([w.x for w in wells], dtype=np.float64)
    xg = np.asarray(grid.x, dtype=np.float64)

    def lateral(curves: list[np.ndarray]) -> np.ndarray:
        arr = np.stack(curves, axis=0)  # (nwell, nz)
        if arr.shape[0] == 1:
            return np.repeat(arr[0][None, :], grid.nx, axis=0)
        out = np.empty((grid.nx, grid.nz), dtype=np.float64)
        for j in range(grid.nz):
            out[:, j] = np.interp(xg, xw, arr[:, j])
        return out

    vp = lateral(vp_curves)
    vs = lateral(vs_curves)
    rho = lateral(rho_curves)
    vs = np.minimum(vs, 0.95 * vp)
    rho = np.clip(rho, 1000.0, 3500.0)
    if sigma_xy > 0:
        vp = gaussian_filter(vp, sigma=(sigma_xy, 0.0), mode="nearest")
        vs = gaussian_filter(vs, sigma=(sigma_xy, 0.0), mode="nearest")
        rho = gaussian_filter(rho, sigma=(sigma_xy, 0.0), mode="nearest")

    if anisotropy == "iso":
        eps = np.full(grid.shape, 0.0 if epsilon is None else epsilon)
        dlt = np.full(grid.shape, 0.0 if delta is None else delta)
        gam = np.full(grid.shape, 0.0 if gamma is None else gamma)
    else:
        e0, d0, g0 = wells[0].thomsen(shale_ratio=shale_ratio)
        eps = np.full(grid.shape, float(np.median(e0)) if epsilon is None else epsilon)
        dlt = np.full(grid.shape, float(np.median(d0)) if delta is None else delta)
        gam = np.full(grid.shape, float(np.median(g0)) if gamma is None else gamma)
    return ElasticModel(
        vp, vs, rho, eps, dlt, gam, anisotropy, grid.dx, grid.dz
    )


# ----------------------------------------------------------------------
# 尺度设计
# ----------------------------------------------------------------------
def design_scale(
    vmin: float,
    vmax: float,
    fmax: float,
    *,
    ppw: float = 8.0,
    cfl: float = 0.45,
    record_time: float | None = None,
) -> dict[str, float]:
    """按“每波长最少网格点数 + CFL”设计网格与时间步。

    返回 ``dx, dz, dt, wavelength_min, ppw, cfl, nt (若给 record_time)``。
    """
    lam_min = vmin / max(fmax, 1e-9)
    dx = lam_min / max(ppw, 1.0)
    dt = cfl * dx / (vmax * (2.0 ** 0.5))
    out = {
        "dx": dx,
        "dz": dx,
        "dt": dt,
        "wavelength_min": lam_min,
        "ppw": ppw,
        "cfl": cfl,
    }
    if record_time is not None:
        out["nt"] = float(np.ceil(record_time / dt)) + 1
    return out


def logging_seismic_scales(
    vp: float = 3000.0,
    f_logging: float = 10000.0,
    f_seismic: float = 30.0,
    *,
    ppw: float = 8.0,
    cfl: float = 0.45,
) -> dict[str, dict[str, float]]:
    """对比测井尺度与地面地震尺度的网格/时间步，说明为何必须换算。"""
    out = {}
    for name, f in (("logging", f_logging), ("seismic", f_seismic)):
        out[name] = design_scale(vp, vp, f, ppw=ppw, cfl=cfl)
        out[name]["frequency"] = f
        out[name]["wavelength"] = vp / f
    out["ratio"] = {
        "frequency": f_logging / f_seismic,
        "dx": out["logging"]["dx"] / out["seismic"]["dx"],
        "dt": out["logging"]["dt"] / out["seismic"]["dt"],
    }
    return out


def scale_matrix(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    *,
    dx: float,
    dz: float,
    symmetry: str = "iso",
    epsilon: float = 0.0,
    delta: float = 0.0,
    gamma: float = 0.0,
) -> ElasticModel:
    """从测井矩阵直接构造模型（便于把井旁道转为二维剖面）。"""
    return ElasticModel(vp, vs, rho, epsilon, delta, gamma, symmetry, dx, dz)
