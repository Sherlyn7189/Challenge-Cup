"""SEG-Y 输入输出。

为什么需要这一层
----------------
在此之前，数据只能来自"配置 + 正演"，无法接入实测资料。本模块把 SEG-Y 的
道头信息映射成项目的数据契约 :class:`~ertm.types.ShotRecord`，让流水线
（预处理 → RTM → 评价 → 输出）可以直接处理野外/处理中心给的数据。

读取约定
--------
- 采样间隔取自二进制头 ``Interval``（微秒），**不依赖** ``segyio`` 的
  ``samples`` 标签（不同软件对单位约定不一致）；
- 坐标取自道头 ``SourceX/GroupX``（可扩展到 Y），并按 ``SourceGroupScalar``
  换算（正数乘、负数除，SEG-Y 标准）；
- 按炮分组：``group_by="source"``（默认，按 SourceX/SourceY）或
  ``"field_record"``（按 ``FieldRecord`` 道头）或 ``"file"``（整文件一炮）；
- 组内按 ``GroupX`` 排序（``sort_by_offset`` 可按偏移距排序），井中数据把
  ``receiver_depth_from="group_y"`` 打开即可用 ``GroupY`` 当深度。

写出一致性
----------
``write_segy_shot`` 写出的文件必须能被 :func:`read_segy_shots` 原样读回
（有往返测试）；``write_segy_image`` 把深度域图像写成"道 = x、样点 = 深度"
的 SEG-Y，方便交给第三方软件查看（约定期刊/软件约定见函数文档）。

依赖
----
``segyio``（已在 requirements 中）。缺少时给出明确错误而不是 ImportError 堆栈。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .exceptions import DataError
from .logging_utils import get_logger
from .types import ElasticModel, GridSpec, ImageResult, ReceiverSpec, ShotRecord, SourceSpec

logger = get_logger(__name__)

try:  # pragma: no cover - 环境相关
    import segyio
except Exception:  # pragma: no cover
    segyio = None


def _require_segyio():
    if segyio is None:  # pragma: no cover
        raise DataError(
            "未安装 segyio，无法读写 SEG-Y；请执行 "
            "`.venv\\Scripts\\python.exe -m pip install segyio`"
        )
    return segyio


# ----------------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SegYReadOptions:
    """SEG-Y 读取选项。"""

    component: str = "p"
    group_by: str = "source"  # source | field_record | file
    receiver_depth: float = 0.0
    source_depth: float = 0.0
    receiver_depth_from: str = "constant"  # constant | group_y
    coordinate_scalar: int | None = None
    shot_limit: int = 0
    sort_by_offset: bool = True
    f0: float = 20.0
    delay: float = 0.0

    def validate(self) -> None:
        if self.group_by not in {"source", "field_record", "file"}:
            raise DataError(f"group_by 必须是 source/field_record/file，当前 {self.group_by!r}")
        if self.receiver_depth_from not in {"constant", "group_y"}:
            raise DataError(
                f"receiver_depth_from 必须是 constant/group_y，当前 {self.receiver_depth_from!r}"
            )
        if self.shot_limit < 0:
            raise DataError("shot_limit 不能为负")


def _apply_scalar(value: int, scalar: int | None) -> float:
    """SEG-Y 坐标标量：正数乘、负数除。"""
    if scalar is None or scalar == 0:
        return float(value)
    return float(value) / abs(scalar) if scalar < 0 else float(value) * abs(scalar)


def _header_int(header: Any, field_key: str, default: int = 0) -> int:
    try:
        return int(header[field_key])
    except Exception:  # pragma: no cover - 非法头
        return int(default)


def _read_headers(path: Path, options: SegYReadOptions) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 ``(data[n_samp, n_trace], 坐标数组, dt)``。"""
    segy = _require_segyio()
    try:
        with segy.open(str(path), "r", ignore_geometry=True) as handle:
            ns = int(handle.bin[segy.BinField.Samples])
            interval = int(handle.bin[segy.BinField.Interval]) or 1000
            n_trace = int(handle.tracecount)
            data = np.asarray(handle.trace.raw[:], dtype=np.float64).T  # (ns, ntr)
            sources = np.array(
                [
                    _apply_scalar(
                        _header_int(handle.header[i], segy.TraceField.SourceX),
                        options.coordinate_scalar
                        or _header_int(handle.header[i], segy.TraceField.SourceGroupScalar),
                    )
                    for i in range(n_trace)
                ],
                dtype=np.float64,
            )
            groups = np.array(
                [
                    _apply_scalar(
                        _header_int(handle.header[i], segy.TraceField.GroupX),
                        options.coordinate_scalar
                        or _header_int(handle.header[i], segy.TraceField.SourceGroupScalar),
                    )
                    for i in range(n_trace)
                ],
                dtype=np.float64,
            )
            group_y = np.array(
                [
                    _apply_scalar(
                        _header_int(handle.header[i], segy.TraceField.GroupY),
                        options.coordinate_scalar
                        or _header_int(handle.header[i], segy.TraceField.SourceGroupScalar),
                    )
                    for i in range(n_trace)
                ],
                dtype=np.float64,
            )
            field = np.array(
                [_header_int(handle.header[i], segy.TraceField.FieldRecord) for i in range(n_trace)],
                dtype=np.int64,
            )
    except (OSError, ValueError) as exc:
        raise DataError(f"读取 SEG-Y 失败: {path}: {exc}") from exc

    if data.shape[0] != ns:
        ns = data.shape[0]
    dt = interval * 1e-6
    return data, (sources, groups, group_y, field), dt


def expand_segy_paths(path: str | Path) -> list[Path]:
    """把"文件 / 目录 / 通配符"展开成按名称排序的文件列表。

    真实项目常见"一炮一文件"或"一条测线一目录"，因此输入允许
    ``data/*.sgy``、``data/line1`` 这类写法。
    """
    import glob as glob_module

    text = str(path)
    if any(char in text for char in "*?["):
        matches = sorted(Path(item) for item in glob_module.glob(text))
        files = [item for item in matches if item.is_file()]
        if not files:
            raise DataError(f"通配符没有匹配到 SEG-Y 文件: {text}")
        return files
    candidate = Path(text)
    if candidate.is_dir():
        files = sorted(
            item
            for item in candidate.iterdir()
            if item.is_file() and item.suffix.lower() in {".sgy", ".segy"}
        )
        if not files:
            raise DataError(f"目录中没有 SEG-Y 文件: {candidate}")
        return files
    if not candidate.is_file():
        raise DataError(f"SEG-Y 文件不存在: {candidate}")
    return [candidate]


def summarize_segy(path: str | Path) -> dict[str, Any]:
    """只读道头，给出 QC 摘要（不加载道数据，适合先检查大文件）。

    返回文件数、道数、采样、炮数、坐标范围与偏移距范围，用于在正式成像前
    判断"数据是否可用、坐标是否合理、dt 是否被写错"。
    """
    segy = _require_segyio()
    files = expand_segy_paths(path)
    total_traces = 0
    total_shots = 0
    dt_set: set[float] = set()
    ns_set: set[int] = set()
    src_min = grp_min = float("inf")
    src_max = grp_max = float("-inf")
    offset_min = float("inf")
    offset_max = float("-inf")
    for file_path in files:
        with segy.open(str(file_path), "r", ignore_geometry=True) as handle:
            ns = int(handle.bin[segy.BinField.Samples])
            interval = int(handle.bin[segy.BinField.Interval]) or 1000
            n_trace = int(handle.tracecount)
            sources = set()
            for index in range(n_trace):
                header = handle.header[index]
                scalar = _header_int(header, segy.TraceField.SourceGroupScalar)
                sx = _apply_scalar(_header_int(header, segy.TraceField.SourceX), scalar)
                gx = _apply_scalar(_header_int(header, segy.TraceField.GroupX), scalar)
                offset = abs(_header_int(header, segy.TraceField.offset))
                sources.add(sx)
                src_min, src_max = min(src_min, sx), max(src_max, sx)
                grp_min, grp_max = min(grp_min, gx), max(grp_max, gx)
                offset_min, offset_max = min(offset_min, offset), max(offset_max, offset)
            total_traces += n_trace
            total_shots += len(sources)
            dt_set.add(interval * 1e-6)
            ns_set.add(ns)
    return {
        "files": [str(item) for item in files],
        "n_files": len(files),
        "n_traces": int(total_traces),
        "n_shots": int(total_shots),
        "dt": sorted(dt_set)[0] if len(dt_set) == 1 else sorted(dt_set),
        "ns": sorted(ns_set)[0] if len(ns_set) == 1 else sorted(ns_set),
        "source_x_range": [float(src_min), float(src_max)],
        "receiver_x_range": [float(grp_min), float(grp_max)],
        "offset_range": [float(offset_min), float(offset_max)],
    }


def read_segy_shots(
    path: str | Path,
    options: SegYReadOptions | None = None,
) -> list[ShotRecord]:
    """读取 SEG-Y（文件 / 目录 / 通配符），返回按炮分组的记录列表。"""
    opts = options or SegYReadOptions()
    opts.validate()
    files = expand_segy_paths(path)
    shots: list[ShotRecord] = []
    for file_path in files:
        shots.extend(_read_single_segy(file_path, opts))
    if opts.shot_limit:
        shots = shots[: int(opts.shot_limit)]
    if len(files) > 1:
        logger.info("多文件 SEG-Y 读取: %d 个文件, 共 %d 炮", len(files), len(shots))
    return shots


def _read_single_segy(file_path: Path, opts: SegYReadOptions) -> list[ShotRecord]:
    """读单个 SEG-Y 文件（路径已校验）。"""

    data, headers, dt = _read_headers(file_path, opts)
    sources, groups, group_y, field = headers
    n_trace = data.shape[1]

    if opts.group_by == "file":
        keys = np.zeros(n_trace, dtype=np.int64)
    elif opts.group_by == "field_record":
        keys = field
    else:
        # 按源坐标分组（同一炮的多道 SourceX 相同）
        _, keys = np.unique(sources, return_inverse=True)

    shots: list[ShotRecord] = []
    for key in np.unique(keys):
        index = np.where(keys == key)[0]
        if opts.sort_by_offset:
            index = index[np.argsort(groups[index])]
        receiver_x = groups[index]
        if opts.receiver_depth_from == "group_y":
            receiver_z = group_y[index]
        else:
            receiver_z = np.full(index.size, float(opts.receiver_depth), dtype=float)
        receivers = ReceiverSpec(x=receiver_x, z=receiver_z, component=opts.component)
        source = SourceSpec(
            x=float(sources[index[0]]),
            z=float(opts.source_depth),
            f0=float(opts.f0),
            kind="ricker",
            amplitude=1.0,
            delay=float(opts.delay),
        )
        trace_data = data[:, index].astype(np.float32)  # data 已是 (nt, n_trace)
        t = np.arange(trace_data.shape[0], dtype=np.float64) * dt
        record = ShotRecord(
            data=trace_data,
            dt=dt,
            t=t,
            receivers=receivers,
            source=source,
            meta={
                "input": "segy",
                "file": str(file_path),
                "component": opts.component,
                "group_by": opts.group_by,
                "group_key": str(key),
                "n_traces": int(index.size),
                "placeholder": False,
            },
            components={opts.component: trace_data},
        )
        record.validate()
        shots.append(record)

    if opts.shot_limit:
        shots = shots[: int(opts.shot_limit)]
    logger.info(
        "SEG-Y 读取完成: %s, %d 炮, dt=%.3e s, 首炮 %d 道",
        file_path,
        len(shots),
        dt,
        shots[0].receivers.nrec if shots else 0,
    )
    return shots


def read_segy_components(
    component_files: dict[str, str | Path],
    options: SegYReadOptions | None = None,
) -> list[ShotRecord]:
    """读多分量 SEG-Y（每个分量一个文件），按炮序对齐后合并。"""
    if not component_files:
        raise DataError("component_files 不能为空")
    per_component: dict[str, list[ShotRecord]] = {}
    for component, path in component_files.items():
        opts = options or SegYReadOptions()
        opts = SegYReadOptions(
            component=component,
            group_by=opts.group_by,
            receiver_depth=opts.receiver_depth,
            source_depth=opts.source_depth,
            receiver_depth_from=opts.receiver_depth_from,
            coordinate_scalar=opts.coordinate_scalar,
            shot_limit=opts.shot_limit,
            sort_by_offset=opts.sort_by_offset,
            f0=opts.f0,
            delay=opts.delay,
        )
        per_component[component] = read_segy_shots(path, opts)

    counts = {name: len(items) for name, items in per_component.items()}
    if len(set(counts.values())) != 1:
        raise DataError(f"各分量炮数不一致: {counts}")
    merged: list[ShotRecord] = []
    for index in range(next(iter(counts.values()))):
        base = next(iter(per_component.values()))[index]
        components = {name: items[index].data for name, items in per_component.items()}
        primary = components.get("p", base.data)
        merged.append(
            ShotRecord(
                data=primary,
                dt=base.dt,
                t=base.t,
                receivers=base.receivers,
                source=base.source,
                meta={**base.meta, "components": list(components)},
                components=components,
            )
        )
    logger.info("多分量 SEG-Y 合并完成: %d 炮, 分量 %s", len(merged), list(component_files))
    return merged


# ----------------------------------------------------------------------------
# 写出
# ----------------------------------------------------------------------------
def write_segy_shot(
    shot: ShotRecord,
    path: str | Path,
    *,
    fmt: int = 5,
    component: str | None = None,
) -> Path:
    """把一炮写成 SEG-Y（道 = 检波点，样点 = 时间）。

    ``fmt=5`` 为 IEEE 浮点（推荐，兼容性好），``fmt=1`` 为 IBM 浮点。
    道头写入 ``SourceX``、``GroupX``、``SourceGroupScalar=1``、偏移距与道号，
    二进制头写入采样间隔（微秒）。
    """
    shot.validate()
    segy = _require_segyio()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(shot.data, dtype=np.float32)
    ns, n_trace = data.shape
    interval_us = int(round(shot.dt * 1e6))

    spec = segy.spec()
    spec.format = int(fmt)
    spec.samples = np.arange(ns, dtype=np.float32) * (interval_us / 1000.0)  # ms 标签
    spec.tracecount = n_trace
    spec.sorting = 1
    try:
        with segy.create(str(target), spec) as handle:
            handle.text[0] = f"ERTM shot {shot.source.x:.1f}/{shot.source.z:.1f}"
            handle.bin[segy.BinField.Interval] = interval_us
            handle.bin[segy.BinField.Samples] = ns
            for index in range(n_trace):
                handle.header[index] = {
                    segy.TraceField.SourceX: int(round(shot.source.x)),
                    segy.TraceField.SourceY: int(round(shot.source.z)),
                    segy.TraceField.GroupX: int(round(float(shot.receivers.x[index]))),
                    segy.TraceField.GroupY: int(round(float(shot.receivers.z[index]))),
                    segy.TraceField.SourceGroupScalar: 1,
                    segy.TraceField.offset: int(
                        round(abs(float(shot.receivers.x[index]) - shot.source.x))
                    ),
                    segy.TraceField.FieldRecord: 1,
                    segy.TraceField.TraceNumber: index + 1,
                    segy.TraceField.TRACE_SAMPLE_COUNT: ns,
                    segy.TraceField.TRACE_SAMPLE_INTERVAL: interval_us,
                }
                handle.trace[index] = np.ascontiguousarray(data[:, index])
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataError(f"写出 SEG-Y 失败: {target}: {exc}") from exc
    logger.info("炮记录已写出 SEG-Y: %s (%d 道)", target, n_trace)
    return target


def write_segy_shots(
    shots: Sequence[ShotRecord],
    path: str | Path,
    *,
    fmt: int = 5,
) -> Path:
    """把多炮写进同一个 SEG-Y（道头 ``FieldRecord`` = 炮号，``SourceX`` = 炮点）。

    与 :func:`read_segy_shots` 的 ``group_by="source"`` 配对，可直接往返。
    """
    if not shots:
        raise DataError("至少需要一炮才能写 SEG-Y")
    for shot in shots:
        shot.validate()
    segy = _require_segyio()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ns = int(shots[0].data.shape[0])
    if any(int(shot.data.shape[0]) != ns for shot in shots):
        raise DataError("多炮写出要求所有炮的记录长度一致")
    n_trace = int(sum(shot.data.shape[1] for shot in shots))
    interval_us = int(round(shots[0].dt * 1e6))

    spec = segy.spec()
    spec.format = int(fmt)
    spec.samples = np.arange(ns, dtype=np.float32) * (interval_us / 1000.0)
    spec.tracecount = n_trace
    spec.sorting = 1
    try:
        with segy.create(str(target), spec) as handle:
            handle.text[0] = f"ERTM shots x{len(shots)}"
            handle.bin[segy.BinField.Interval] = interval_us
            handle.bin[segy.BinField.Samples] = ns
            index = 0
            for shot_index, shot in enumerate(shots):
                data = np.asarray(shot.data, dtype=np.float32)
                for trace in range(data.shape[1]):
                    handle.header[index] = {
                        segy.TraceField.SourceX: int(round(shot.source.x)),
                        segy.TraceField.SourceY: int(round(shot.source.z)),
                        segy.TraceField.GroupX: int(round(float(shot.receivers.x[trace]))),
                        segy.TraceField.GroupY: int(round(float(shot.receivers.z[trace]))),
                        segy.TraceField.SourceGroupScalar: 1,
                        segy.TraceField.offset: int(
                            round(abs(float(shot.receivers.x[trace]) - shot.source.x))
                        ),
                        segy.TraceField.FieldRecord: shot_index + 1,
                        segy.TraceField.TraceNumber: index + 1,
                        segy.TraceField.TRACE_SAMPLE_COUNT: ns,
                        segy.TraceField.TRACE_SAMPLE_INTERVAL: interval_us,
                    }
                    handle.trace[index] = np.ascontiguousarray(data[:, trace])
                    index += 1
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataError(f"写出 SEG-Y 失败: {target}: {exc}") from exc
    logger.info("已写出 SEG-Y: %s (%d 炮, %d 道)", target, len(shots), n_trace)
    return target


def write_segy_image(
    image: ImageResult,
    path: str | Path,
    *,
    fmt: int = 5,
) -> Path:
    """把深度域图像写成 SEG-Y：道 = x，样点 = 深度。

    约定：``Interval`` 与样点标签用**米**（深度采样 ``dz``），道头
    ``GroupX`` 写 x 坐标。这不是勘探地震常见的"时间剖面"，而是一种
    通用的"道 × 样点"容器，便于在第三方软件中按深度查看。
    """
    image.validate()
    segy = _require_segyio()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(image.image, dtype=np.float32).T  # (nx, nz)
    n_trace, ns = data.shape
    dz = float(image.grid.dz)
    dz_us = int(round(dz * 1_000))  # 米 → "微秒"槽位，仅作容器用

    spec = segy.spec()
    spec.format = int(fmt)
    spec.samples = np.arange(ns, dtype=np.float32) * dz
    spec.tracecount = n_trace
    spec.sorting = 1
    try:
        with segy.create(str(target), spec) as handle:
            handle.text[0] = f"ERTM image {image.mode.upper()} (depth domain)"
            handle.bin[segy.BinField.Interval] = max(dz_us, 1)
            handle.bin[segy.BinField.Samples] = ns
            for index in range(n_trace):
                x = float(image.grid.x_at(index))
                handle.header[index] = {
                    segy.TraceField.SourceX: int(round(x)),
                    segy.TraceField.GroupX: int(round(x)),
                    segy.TraceField.SourceGroupScalar: 1,
                    segy.TraceField.CDP: index + 1,
                    segy.TraceField.TRACE_SAMPLE_COUNT: ns,
                    segy.TraceField.TRACE_SAMPLE_INTERVAL: max(dz_us, 1),
                }
                handle.trace[index] = np.ascontiguousarray(data[index])
    except (OSError, ValueError, RuntimeError) as exc:
        raise DataError(f"写出图像 SEG-Y 失败: {target}: {exc}") from exc
    logger.info("成像结果已写出 SEG-Y: %s (%d 道 × %d 样点)", target, n_trace, ns)
    return target


def read_segy_image(path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """读回 ``write_segy_image`` 写的图像，返回 ``(image[nz,nx], dz, x)``。"""
    segy = _require_segyio()
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"SEG-Y 文件不存在: {file_path}")
    try:
        with segy.open(str(file_path), "r", ignore_geometry=True) as handle:
            ns = int(handle.bin[segy.BinField.Samples])
            interval_us = int(handle.bin[segy.BinField.Interval]) or 1000
            data = np.asarray(handle.trace.raw[:], dtype=np.float32).T
            x = np.array(
                [
                    float(handle.header[i][segy.TraceField.GroupX])
                    for i in range(int(handle.tracecount))
                ],
                dtype=np.float64,
            )
    except (OSError, ValueError) as exc:
        raise DataError(f"读取 SEG-Y 图像失败: {file_path}: {exc}") from exc
    dz = interval_us / 1_000.0
    return data.reshape(ns, -1), dz, x
