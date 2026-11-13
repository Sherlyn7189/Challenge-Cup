"""SEG-Y 实际资料读取、观测系统映射与写回。

当前 EFWI 的观测数据内部格式为 ``(ns, nt, nr, ncomp)``，本模块负责从
SEG-Y 道头恢复：

* 炮点 ``(x, z)``（默认 SourceX / SourceDepth）；
* 检波点 ``(x, z)``（默认 GroupX / GroupDepth）；
* 每炮的接收道与分量（默认单分量；可指定 TraceIdentificationCode 等）；
* 时间采样 ``dt/nt``；
* 缺失道 mask。

SEG-Y 坐标单位由 SourceGroupScalar（71）控制：负值表示除以 |scalar|，
正值表示乘以 scalar。坐标默认使用米；如资料用其他单位，请通过
``coordinate_scale`` 显式换算。

示例::

    from efwi.io import read_segy_gathers
    gathers = read_segy_gathers("data/line1.sgy")
    print(gathers.data.shape, gathers.dt, gathers.shots[:3])

    survey = gathers.to_survey(wavelet=wavelet)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

try:  # segyio 是可选依赖；没有实际数据时核心算法仍可运行
    import segyio
except Exception:  # pragma: no cover - 取决于环境
    segyio = None


def _require_segyio() -> Any:
    if segyio is None:
        raise ImportError("需要安装 segyio 才能读写 SEG-Y：pip install segyio")
    return segyio


def _scaled(value: float, scalar: float) -> float:
    """应用 SEG-Y 坐标 scalar。"""
    if scalar is None or not np.isfinite(scalar) or scalar == 0:
        return float(value)
    if scalar < 0:
        return float(value) / abs(float(scalar))
    return float(value) * float(scalar)


@dataclass
class SegyShot:
    shot_id: int
    source: np.ndarray
    receivers: np.ndarray
    data: np.ndarray
    receiver_ids: np.ndarray
    component_ids: np.ndarray | None = None


@dataclass
class SegyGathers:
    """解编后的炮集。

    ``data`` 形状 ``(ns, nt, nr, ncomp)``；``mask`` 形状
    ``(ns, nr, ncomp)``，表示该炮/道/分量是否实际存在。缺失道在 data 中
    为 0，避免破坏固定接收点排列的 FWI 接口。
    """

    shots: np.ndarray
    receivers: np.ndarray
    data: np.ndarray
    dt: float
    mask: np.ndarray
    shot_ids: np.ndarray
    receiver_ids: np.ndarray
    text_header: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    component_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.shots = np.asarray(self.shots, dtype=np.float64)
        self.receivers = np.asarray(self.receivers, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.data.ndim != 4:
            raise ValueError("data 形状应为 (ns, nt, nr, ncomp)")
        if self.data.shape[:3] != (self.shots.shape[0], self.data.shape[1], self.receivers.shape[0]):
            raise ValueError("data/shots/receivers 形状不一致")
        expected_mask = (self.data.shape[0], self.data.shape[2], self.data.shape[3])
        if self.mask.shape != expected_mask:
            raise ValueError(f"mask 形状应为 {expected_mask}")

    @property
    def n_shots(self) -> int:
        return int(self.shots.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    @property
    def nt(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_components(self) -> int:
        return int(self.data.shape[3])

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.nt, dtype=np.float64) * self.dt

    def component_slice(self, index: int) -> np.ndarray:
        return self.data[..., index]

    def to_survey(self, wavelet: np.ndarray, **kwargs: Any) -> Any:
        """转成 :class:`efwi.geometry.Survey`（固定检波点排列）。"""
        from .geometry import Survey

        return Survey(
            self.shots,
            self.receivers,
            self.nt,
            self.dt,
            wavelet,
            kwargs.pop("source_kind", "explosive"),
            kwargs.pop("amplitude", 1e6),
            kwargs.pop("meta", dict(self.meta)),
        )

    def summary(self) -> dict[str, Any]:
        present = self.mask.sum()
        total = self.mask.size
        return {
            "n_shots": self.n_shots,
            "n_receivers": self.n_receivers,
            "nt": self.nt,
            "dt": self.dt,
            "n_components": self.n_components,
            "traces_present": int(present),
            "traces_total": int(total),
            "missing_fraction": float(1.0 - present / max(total, 1)),
            "shot_x_range": [float(self.shots[:, 0].min()), float(self.shots[:, 0].max())],
            "receiver_x_range": [float(self.receivers[:, 0].min()), float(self.receivers[:, 0].max())],
        }


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------
def read_segy_gathers(
    path: str,
    *,
    shot_field: str = "FieldRecord",
    source_x_field: str = "SourceX",
    source_z_field: str = "SourceDepth",
    group_x_field: str = "GroupX",
    group_z_field: str = "ReceiverGroupElevation",
    scalar_field: str = "SourceGroupScalar",
    component_field: str | None = None,
    component_values: Sequence[int] | None = None,
    dt: float | None = None,
    t0: float | None = None,
    coordinate_scale: float = 1.0,
    common_receivers: bool = True,
    receiver_tolerance: float = 1e-3,
    max_shots: int | None = None,
) -> SegyGathers:
    """读取 SEG-Y 并解编为固定接收点排列的炮集。

    默认按 ``FieldRecord`` 分炮，用 ``SourceX/SourceDepth`` 与
    ``GroupX/GroupDepth`` 建立二维 ``(x,z)`` 坐标。三维数据可把
    ``source_x_field="SourceX"``、``source_z_field="SourceY"`` 读成平面坐标，
    或自行扩展为三维 Survey。
    """
    sio = _require_segyio()
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with sio.open(path, "r", ignore_geometry=True) as f:
        ntraces = int(f.tracecount)
        samples = np.asarray(f.samples, dtype=np.float64)
        nt = int(samples.size)
        if dt is None:
            interval_us = float(f.bin[sio.BinField.Interval])
            dt = interval_us * 1e-6 if interval_us > 0 else float(samples[1] - samples[0]) * 1e-3
        if t0 is None:
            t0 = float(samples[0]) * 1e-3 if samples.size else 0.0
        try:
            text_header = sio.tools.wrap(f.text[0])
        except Exception:
            text_header = ""

        header_fields = {name: getattr(sio.TraceField, name) for name in (
            shot_field, source_x_field, source_z_field,
            group_x_field, group_z_field, scalar_field,
        )}
        comp_field = getattr(sio.TraceField, component_field) if component_field else None

        records = []
        for it in range(ntraces):
            h = f.header[it]
            shot_key = int(h[header_fields[shot_field]])
            sx = _scaled(h[header_fields[source_x_field]], h[header_fields[scalar_field]]) * coordinate_scale
            sz = _scaled(h[header_fields[source_z_field]], h[header_fields[scalar_field]]) * coordinate_scale
            gx = _scaled(h[header_fields[group_x_field]], h[header_fields[scalar_field]]) * coordinate_scale
            gz = _scaled(h[header_fields[group_z_field]], h[header_fields[scalar_field]]) * coordinate_scale
            rec_id = int(h[sio.TraceField.TraceNumber])
            comp = int(h[comp_field]) if comp_field is not None else 0
            records.append(
                {
                    "index": it,
                    "shot": shot_key,
                    "source": (sx, sz),
                    "receiver": (gx, gz),
                    "receiver_id": rec_id,
                    "component": comp,
                    "trace": np.asarray(f.trace[it], dtype=np.float64),
                }
            )

    if max_shots is not None:
        keep = sorted({r["shot"] for r in records})[: int(max_shots)]
        records = [r for r in records if r["shot"] in set(keep)]
    if not records:
        raise ValueError("SEG-Y 中没有有效道")

    # 分炮
    shot_keys = sorted({r["shot"] for r in records})
    shots = []
    per_shot = []
    for key in shot_keys:
        rs = [r for r in records if r["shot"] == key]
        # 同一炮内 source 取平均，避免道头轻微不一致
        src = np.mean(np.asarray([r["source"] for r in rs], dtype=np.float64), axis=0)
        shots.append(src)
        per_shot.append(rs)

    comps = sorted({r["component"] for r in records})
    if component_values is not None:
        comps = [int(c) for c in component_values]
        comp_index = {c: i for i, c in enumerate(comps)}
    else:
        comp_index = {c: i for i, c in enumerate(comps)}
    ncomp = len(comps)

    # 固定接收点排列：默认取所有炮的并集，按 x,z 排序
    def key_of(pos: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round(pos[0] / receiver_tolerance)),
            int(round(pos[1] / receiver_tolerance)),
        )

    if common_receivers:
        receiver_map: dict[tuple[int, int], tuple[float, float]] = {}
        for rs in per_shot:
            for r in rs:
                receiver_map.setdefault(key_of(r["receiver"]), r["receiver"])
        receiver_keys = sorted(receiver_map, key=lambda k: (receiver_map[k][0], receiver_map[k][1]))
        receivers = np.asarray([receiver_map[k] for k in receiver_keys], dtype=np.float64)
        receiver_index = {k: i for i, k in enumerate(receiver_keys)}
    else:
        rs0 = per_shot[0]
        uniq = {}
        for r in rs0:
            uniq.setdefault(key_of(r["receiver"]), r["receiver"])
        receiver_keys = sorted(uniq, key=lambda k: (uniq[k][0], uniq[k][1]))
        receivers = np.asarray([uniq[k] for k in receiver_keys], dtype=np.float64)
        receiver_index = {k: i for i, k in enumerate(receiver_keys)}

    data = np.zeros((len(shots), nt, len(receivers), ncomp), dtype=np.float64)
    mask = np.zeros((len(shots), len(receivers), ncomp), dtype=bool)
    for si, rs in enumerate(per_shot):
        for r in rs:
            ri = receiver_index.get(key_of(r["receiver"]))
            ci = comp_index.get(r["component"])
            if ri is None or ci is None:
                continue
            tr = r["trace"]
            if tr.size != nt:
                raise ValueError(f"第 {r['index']} 道长度 {tr.size} 与 nt={nt} 不一致")
            data[si, :, ri, ci] = tr
            mask[si, ri, ci] = True

    return SegyGathers(
        shots=np.asarray(shots, dtype=np.float64),
        receivers=receivers,
        data=data,
        dt=float(dt),
        mask=mask,
        shot_ids=np.asarray(shot_keys, dtype=np.int64),
        receiver_ids=np.asarray([receiver_index[k] for k in receiver_keys], dtype=np.int64),
        text_header=text_header,
        meta={"path": os.path.abspath(path), "t0": float(t0), "n_traces": ntraces,
              "component_values": comps},
        component_ids=np.asarray(comps, dtype=np.int64),
    )


# ----------------------------------------------------------------------
# 写回（测试、导出、QC 用）
# ----------------------------------------------------------------------
def write_segy_gathers(
    path: str,
    gathers: SegyGathers,
    *,
    coordinate_scalar: int = 1,
    format: int = 5,
) -> str:
    """把 ``SegyGathers`` 写成单分量/多分量 SEG-Y。

    每个 (shot, receiver, component) 一道；用 ``FieldRecord`` 分炮。
    ``format=5`` 为 IEEE float，``format=1`` 为 IBM float。
    """
    sio = _require_segyio()
    spec = sio.spec()
    spec.format = int(format)
    spec.sorting = 2
    spec.samples = np.arange(gathers.nt, dtype=np.float32) * gathers.dt * 1000.0
    spec.tracecount = gathers.n_shots * gathers.n_receivers * gathers.n_components

    with sio.create(path, spec) as f:
        f.bin[sio.BinField.Interval] = int(round(gathers.dt * 1e6))
        f.bin[sio.BinField.Format] = int(format)
        it = 0
        for si in range(gathers.n_shots):
            sx, sz = gathers.shots[si]
            for ri in range(gathers.n_receivers):
                gx, gz = gathers.receivers[ri]
                for ci in range(gathers.n_components):
                    comp = int(gathers.component_ids[ci]) if gathers.component_ids is not None else 1
                    f.trace[it] = gathers.data[si, :, ri, ci].astype(np.float32)
                    f.header[it] = {
                        sio.TraceField.TraceNumber: it + 1,
                        sio.TraceField.FieldRecord: int(gathers.shot_ids[si]),
                        sio.TraceField.SourceX: int(round(sx / coordinate_scalar)),
                        sio.TraceField.SourceY: 0,
                        sio.TraceField.SourceDepth: int(round(sz / coordinate_scalar)),
                        sio.TraceField.GroupX: int(round(gx / coordinate_scalar)),
                        sio.TraceField.GroupY: 0,
                        sio.TraceField.ReceiverGroupElevation: int(round(gz / coordinate_scalar)),
                        sio.TraceField.SourceGroupScalar: int(coordinate_scalar),
                        sio.TraceField.CoordinateUnits: 1,
                        sio.TraceField.TraceIdentificationCode: comp,
                        sio.TraceField.offset: int(round(abs(gx - sx) / max(abs(coordinate_scalar), 1))),
                    }
                    it += 1
    return path


# ----------------------------------------------------------------------
def read_segy_shot_list(path: str, **kwargs: Any) -> list[SegyShot]:
    """按炮返回原始接收点列表（不做共同接收点合并）。"""
    gathers = read_segy_gathers(path, common_receivers=False, **kwargs)
    out = []
    for si in range(gathers.n_shots):
        out.append(
            SegyShot(
                shot_id=int(gathers.shot_ids[si]),
                source=gathers.shots[si].copy(),
                receivers=gathers.receivers.copy(),
                data=gathers.data[si].copy(),
                receiver_ids=gathers.receiver_ids.copy(),
                component_ids=gathers.component_ids,
            )
        )
    return out


def gather_quality_control(gathers: SegyGathers) -> dict[str, Any]:
    """基本 QC：缺失道、振幅、主频、死道。"""
    from .wavelet import dominant_frequency

    mask = gathers.mask
    data = gathers.data
    mask_t = np.broadcast_to(mask[:, None, :, :], data.shape)
    present = data[mask_t] if mask.any() else np.asarray([])
    dead = []
    for si in range(gathers.n_shots):
        for ri in range(gathers.n_receivers):
            tr = data[si, :, ri, 0]
            if mask[si, ri, 0] and np.max(np.abs(tr)) <= 1e-12:
                dead.append([si, ri])
    return {
        **gathers.summary(),
        "max_amplitude": float(np.max(np.abs(present))) if present.size else 0.0,
        "rms_amplitude": float(np.sqrt(np.mean(present ** 2))) if present.size else 0.0,
        "dead_traces": int(len(dead)),
        "dominant_frequency_hz": float(dominant_frequency(data[0, :, 0, 0], gathers.dt))
        if gathers.nt > 8 else 0.0,
    }
