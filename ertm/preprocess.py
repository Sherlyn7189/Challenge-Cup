"""观测系统转换与采集预处理。

四个必要步骤（对应技术方案“井中场景四项关键技术”的第 ① 项）：

1. **四分量 COG → 共炮点道集 CSG**：井中四分量（水听器 p + 三分量检波器）
   以共接收点道集（COG）记录，RTM 需要 CSG 作为反传边界条件，因此先按
   检波器方位角把水平分量旋转到炮点—检波点平面；
2. **时间采样插值**：实测 36 μs、40 μs 采样往往与差分稳定条件不一致，
   用 Kaiser 窗 sinc 插值到求解器要求的 ``dt``（插值方向由 ``dt`` 大小决定，
   既支持升采样也支持降采样）；
3. **直达波切除**：近源强直达波会淹没浅部成像，按“直达波走时 + 子波长度”
   做顶部切除，并加余弦过渡；
4. **炮集组织**：按炮点坐标生成 CSG 列表，供多炮叠加使用。

本模块只做“数据形态”的变换，不做任何波场延拓。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from . import numerics as num
from .exceptions import DataError
from .logging_utils import get_logger
from .types import ElasticModel, ReceiverSpec, ShotRecord

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 四分量旋转：COG → CSG
# ----------------------------------------------------------------------------
def rotate_components(
    components: dict[str, np.ndarray],
    angle_deg: float,
) -> dict[str, np.ndarray]:
    """把水平分量旋转到炮点—检波点平面（二维简化）。

    井中四分量通常是 ``p``（水听器/压力）+ ``vx, vy, vz``（三分量检波器）。
    二维 P-SV 情形下只需 ``vx, vz`` 两个水平/垂直分量：按方位角 ``θ`` 旋转::

        vx' =  cosθ·vx + sinθ·vz
        vz' = -sinθ·vx + cosθ·vz

    旋转保持能量（正交变换），因此不会改变总能量；``p`` 不变。三维时还需
    绕 z 轴旋转 ``vy``，本期只做二维，保留接口。
    """
    if not components:
        raise DataError("四分量旋转需要至少一个分量")
    angle = np.deg2rad(float(angle_deg))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    rotated = {name: np.array(array, copy=True) for name, array in components.items()}
    if "vx" in rotated and "vz" in rotated:
        vx = np.asarray(components["vx"], dtype=np.float64)
        vz = np.asarray(components["vz"], dtype=np.float64)
        rotated["vx"] = (cos * vx + sin * vz).astype(np.float32)
        rotated["vz"] = (-sin * vx + cos * vz).astype(np.float32)
    else:
        logger.warning("缺少 vx/vz 分量，四分量旋转被跳过")
    return rotated


def rotate_shot(shot: ShotRecord, angle_deg: float) -> ShotRecord:
    """对整炮记录做 COG→CSG 旋转，返回新记录。"""
    if angle_deg == 0.0:
        return shot
    components = rotate_components(shot.components, angle_deg)
    primary_name = "p" if "p" in components else ("vz" if "vz" in components else None)
    data = components[primary_name] if primary_name else shot.data
    return replace(shot, data=data, components=components, meta={**shot.meta, "rotation_deg": angle_deg})


# ----------------------------------------------------------------------------
# 时间采样插值
# ----------------------------------------------------------------------------
def resample_shot(shot: ShotRecord, dt_target: float) -> ShotRecord:
    """把记录重采样到 ``dt_target``（sinc 插值，保幅保相位）。"""
    if abs(dt_target - shot.dt) < 1e-12:
        return shot
    if dt_target <= 0:
        raise DataError(f"dt_target 必须为正，当前 {dt_target}")
    components = dict(shot.components) or {"primary": shot.data}
    converted = {
        name: num.resample_sinc(np.asarray(array, dtype=np.float64), shot.dt, dt_target)
        for name, array in components.items()
    }
    primary = "primary" if "primary" in converted else next(iter(converted))
    nt = converted[primary].shape[0]
    t = np.arange(nt, dtype=np.float64) * dt_target
    data = converted[primary].astype(np.float32)
    new_components = {} if "primary" in converted else {
        name: array.astype(np.float32) for name, array in converted.items()
    }
    return replace(
        shot,
        data=data,
        dt=float(dt_target),
        t=t,
        components=new_components,
        meta={
            **shot.meta,
            "input_dt": float(shot.dt),
            "resampled_dt": float(dt_target),
            "resampled": True,
        },
    )


def required_dt(model: ElasticModel, dx: float | None = None, *, order: int = 8) -> float:
    """返回满足 CFL 的建议 ``dt``（供采集设计/重采样使用）。"""
    dx = model.grid.dx if dx is None else dx
    return num.cfl_dt(float(np.max(model.vp)), dx, model.grid.dz, order, safety=0.5)


# ----------------------------------------------------------------------------
# 直达波切除
# ----------------------------------------------------------------------------
def mute_direct_wave(
    shot: ShotRecord,
    v_reference: float,
    *,
    margin_periods: float = 1.2,
    offset_factor: float = 1.05,
    taper_periods: float = 0.5,
) -> ShotRecord:
    """按直达波走时切除近源强能量。

    切除时间::

        t_mute(x) = delay + margin_periods/f0 + offset_factor·|x-x_s|/v_ref

    其中 ``v_reference`` 取近地表速度（``vmax`` 会切掉有效反射，``vmin`` 会
    残留直达波）；切除边界用余弦过渡，避免阶跃产生的高频假象。
    """
    if v_reference <= 0:
        raise DataError(f"v_reference 必须为正，当前 {v_reference}")
    data = np.array(shot.data, dtype=np.float64, copy=True)
    t = np.asarray(shot.t, dtype=np.float64)
    offset = np.abs(shot.receivers.x - shot.source.x)
    period = 1.0 / max(float(shot.source.f0), 1e-6)
    t_mute = (
        float(shot.source.delay)
        + margin_periods * period
        + offset_factor * offset / v_reference
    )
    taper = max(float(taper_periods) * period, 1e-9)

    weights = np.empty_like(data)
    for index in range(data.shape[1]):
        start = t_mute[index] - taper
        end = t_mute[index] + taper
        ramp = np.clip((t - start) / (end - start), 0.0, 1.0)
        weights[:, index] = 0.5 - 0.5 * np.cos(np.pi * ramp)

    data *= weights
    components = {
        name: (np.asarray(array, dtype=np.float64) * weights).astype(np.float32)
        for name, array in shot.components.items()
    }
    primary = components.get("p", data.astype(np.float32)) if components else data.astype(np.float32)
    return replace(
        shot,
        data=primary,
        components=components,
        meta={**shot.meta, "direct_mute": True, "mute_velocity": float(v_reference)},
    )


def taper_record_ends(shot: ShotRecord, fraction: float = 0.05) -> ShotRecord:
    """对记录首尾做余弦 taper，压制截断引起的边界反射。"""
    if not 0 < fraction < 0.5:
        raise DataError(f"fraction 必须在 (0, 0.5) 内，当前 {fraction}")

    def _taper(array: np.ndarray) -> np.ndarray:
        out = np.array(array, dtype=np.float64, copy=True)
        n = out.shape[0]
        w = max(int(fraction * n), 1)
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(w) + 0.5) / w)
        out[:w] *= ramp[:, None]
        out[-w:] *= ramp[::-1][:, None]
        return out

    data = _taper(shot.data)
    components = {name: _taper(array).astype(np.float32) for name, array in shot.components.items()}
    primary = components.get("p", data.astype(np.float32)) if components else data.astype(np.float32)
    return replace(shot, data=primary, components=components, meta={**shot.meta, "taper_fraction": fraction})


# ----------------------------------------------------------------------------
# 组合流程
# ----------------------------------------------------------------------------
def prepare_shot(
    shot: ShotRecord,
    *,
    model: ElasticModel,
    record_dt: float | None = None,
    rotation_deg: float = 0.0,
    mute: bool = True,
    v_reference: float | None = None,
    taper_fraction: float = 0.05,
) -> ShotRecord:
    """一次完成“旋转 → 重采样 → 切除 → 首尾 taper”。"""
    prepared = shot
    if rotation_deg:
        prepared = rotate_shot(prepared, rotation_deg)
    if record_dt is not None and abs(record_dt - prepared.dt) > 1e-12:
        prepared = resample_shot(prepared, record_dt)
    if mute:
        velocity = float(v_reference) if v_reference is not None else float(
            np.percentile(model.vp, 25)
        )
        prepared = mute_direct_wave(prepared, velocity)
    if taper_fraction > 0:
        prepared = taper_record_ends(prepared, taper_fraction)
    logger.info(
        "炮集预处理完成: dt=%.2e, 道数=%d, 分量=%s",
        prepared.dt,
        prepared.receivers.nrec,
        prepared.component_names,
    )
    return prepared


def build_shot_sequence(
    base_shot: ShotRecord,
    source_positions: Sequence[float],
) -> list[ShotRecord]:
    """按炮点位置复制炮集（占位：真实数据按炮点分文件读入）。"""
    shots: list[ShotRecord] = []
    for x in source_positions:
        shots.append(
            replace(
                base_shot,
                source=replace(base_shot.source, x=float(x)),
                meta={**base_shot.meta, "shot_x": float(x)},
            )
        )
    return shots
