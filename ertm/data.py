"""数据处理与持久化模块。

首期只支持 NumPy ``.npz``（模型、炮记录、图像）和 JSON（指标、报告），
不实现 SEG-Y 解析和大型数据分块。这样可以在不引入额外依赖的前提下，
让流水线具备“可保存、可加载、可追溯”的能力。

所有文件都带 ``meta`` 字段（JSON 字符串），包含网格、单位、占位标记等
信息，避免“只有数组、不知道为什么”的黑盒文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DataError
from .logging_utils import get_logger
from .types import (
    ElasticModel,
    GridSpec,
    ImageResult,
    ReceiverSpec,
    ShotRecord,
    SourceSpec,
)

logger = get_logger(__name__)

OUTPUT_SUBDIRS = ("logs", "models", "shots", "images", "reports")


def ensure_output_dirs(root: str | Path) -> dict[str, Path]:
    """创建标准输出目录并返回路径表。"""
    base = Path(root).expanduser()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataError(f"无法创建输出根目录: {base}: {exc}") from exc
    if not base.is_dir():
        raise DataError(f"输出根路径不是目录: {base}")

    result: dict[str, Path] = {"root": base}
    for name in OUTPUT_SUBDIRS:
        path = base / name
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DataError(f"无法创建输出子目录: {path}: {exc}") from exc
        result[name] = path
    return result


def _dump_meta(meta: dict[str, Any]) -> np.ndarray:
    try:
        text = json.dumps(meta, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise DataError(f"元数据无法 JSON 序列化: {exc}") from exc
    return np.array(text)


def _load_meta(array: np.ndarray) -> dict[str, Any]:
    try:
        value = json.loads(str(array.item()))
    except (ValueError, TypeError) as exc:
        raise DataError(f"元数据不是合法 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DataError("元数据必须是 JSON 对象")
    return value


class NumpyDataStore:
    """基于 ``.npz`` 的数据仓储。"""

    name = "numpy-data-store"

    def __init__(self, root: str | Path) -> None:
        self.dirs = ensure_output_dirs(root)
        self.root = self.dirs["root"]

    # ------------------------------------------------------------------ model
    def save_model(self, model: ElasticModel, name: str = "model") -> Path:
        model.validate()
        path = self.dirs["models"] / f"{name}.npz"
        meta = {
            "kind": "elastic_model",
            "name": model.name,
            "grid": {
                "nx": model.grid.nx,
                "nz": model.grid.nz,
                "dx": model.grid.dx,
                "dz": model.grid.dz,
                "x0": model.grid.x0,
                "z0": model.grid.z0,
            },
            "model_meta": model.meta,
            "units": {"length": "m", "velocity": "m/s", "density": "kg/m3"},
        }
        try:
            np.savez_compressed(
                path,
                vp=model.vp,
                vs=model.vs,
                rho=model.rho,
                qp=model.qp,
                qs=model.qs,
                meta=_dump_meta(meta),
            )
        except OSError as exc:
            raise DataError(f"保存模型失败: {path}: {exc}") from exc
        logger.info("模型已保存: %s", path)
        return path

    def load_model(self, path: str | Path) -> ElasticModel:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = _load_meta(data["meta"])
                grid_meta = meta["grid"]
                grid = GridSpec(
                    nx=int(grid_meta["nx"]),
                    nz=int(grid_meta["nz"]),
                    dx=float(grid_meta["dx"]),
                    dz=float(grid_meta["dz"]),
                    x0=float(grid_meta.get("x0", 0.0)),
                    z0=float(grid_meta.get("z0", 0.0)),
                )
                model = ElasticModel(
                    grid=grid,
                    vp=np.asarray(data["vp"]),
                    vs=np.asarray(data["vs"]),
                    rho=np.asarray(data["rho"]),
                    qp=np.asarray(data["qp"]),
                    qs=np.asarray(data["qs"]),
                    name=str(meta.get("name", path.stem)),
                    meta=dict(meta.get("model_meta", {})),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise DataError(f"加载模型失败: {path}: {exc}") from exc
        try:
            model.validate()
        except ValueError as exc:
            raise DataError(f"模型文件内容无效: {path}: {exc}") from exc
        return model

    # ------------------------------------------------------------------- shot
    def save_shot(self, shot: ShotRecord, name: str = "shot") -> Path:
        shot.validate()
        path = self.dirs["shots"] / f"{name}.npz"
        meta = {
            "kind": "shot_record",
            "dt": shot.dt,
            "component": shot.receivers.component,
            "components": list(shot.component_names),
            "source": {
                "x": shot.source.x,
                "z": shot.source.z,
                "f0": shot.source.f0,
                "kind": shot.source.kind,
                "amplitude": shot.source.amplitude,
                "delay": shot.source.delay,
                "mechanism": shot.source.mechanism,
            },
            "shot_meta": shot.meta,
            "units": {"time": "s", "amplitude": "a.u."},
        }
        payload: dict[str, Any] = {
            "data": shot.data,
            "t": shot.t,
            "rx": shot.receivers.x,
            "rz": shot.receivers.z,
            "meta": _dump_meta(meta),
        }
        for index, (name_, array) in enumerate(shot.components.items()):
            payload[f"component_{index}"] = array
        payload["component_names"] = np.array(list(shot.components), dtype=str)
        try:
            np.savez_compressed(path, **payload)
        except OSError as exc:
            raise DataError(f"保存炮记录失败: {path}: {exc}") from exc
        logger.info("炮记录已保存: %s (%d 分量)", path, len(shot.components))
        return path

    def load_shot(self, path: str | Path) -> ShotRecord:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = _load_meta(data["meta"])
                source_meta = meta["source"]
                source = SourceSpec(
                    x=float(source_meta["x"]),
                    z=float(source_meta["z"]),
                    f0=float(source_meta["f0"]),
                    kind=str(source_meta["kind"]),
                    amplitude=float(source_meta["amplitude"]),
                    delay=float(source_meta["delay"]),
                    mechanism=str(source_meta.get("mechanism", "explosive")),
                )
                receivers = ReceiverSpec(
                    x=np.asarray(data["rx"], dtype=float),
                    z=np.asarray(data["rz"], dtype=float),
                    component=str(meta.get("component", "vz")),
                )
                components: dict[str, np.ndarray] = {}
                if "component_names" in data:
                    names = [str(item) for item in np.asarray(data["component_names"])]
                    for index, name_ in enumerate(names):
                        key = f"component_{index}"
                        if key in data:
                            components[name_] = np.asarray(data[key], dtype=np.float32)
                shot = ShotRecord(
                    data=np.asarray(data["data"]),
                    dt=float(meta["dt"]),
                    t=np.asarray(data["t"], dtype=float),
                    receivers=receivers,
                    source=source,
                    meta=dict(meta.get("shot_meta", {})),
                    components=components,
                )
        except (OSError, KeyError, ValueError) as exc:
            raise DataError(f"加载炮记录失败: {path}: {exc}") from exc
        try:
            shot.validate()
        except ValueError as exc:
            raise DataError(f"炮记录文件内容无效: {path}: {exc}") from exc
        return shot

    # ------------------------------------------------------------------ image
    def save_image(self, image: ImageResult, name: str = "image") -> Path:
        image.validate()
        path = self.dirs["images"] / f"{name}.npz"
        meta = {
            "kind": "image_result",
            "mode": image.mode,
            "grid": {
                "nx": image.grid.nx,
                "nz": image.grid.nz,
                "dx": image.grid.dx,
                "dz": image.grid.dz,
                "x0": image.grid.x0,
                "z0": image.grid.z0,
            },
            "image_meta": image.meta,
            "units": {"length": "m", "amplitude": "a.u."},
        }
        try:
            np.savez_compressed(path, image=image.image, meta=_dump_meta(meta))
        except OSError as exc:
            raise DataError(f"保存图像失败: {path}: {exc}") from exc
        logger.info("图像已保存: %s", path)
        return path

    def load_image(self, path: str | Path) -> ImageResult:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = _load_meta(data["meta"])
                grid_meta = meta["grid"]
                grid = GridSpec(
                    nx=int(grid_meta["nx"]),
                    nz=int(grid_meta["nz"]),
                    dx=float(grid_meta["dx"]),
                    dz=float(grid_meta["dz"]),
                    x0=float(grid_meta.get("x0", 0.0)),
                    z0=float(grid_meta.get("z0", 0.0)),
                )
                image = ImageResult(
                    image=np.asarray(data["image"]),
                    grid=grid,
                    mode=str(meta.get("mode", "pp")),
                    meta=dict(meta.get("image_meta", {})),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise DataError(f"加载图像失败: {path}: {exc}") from exc
        try:
            image.validate()
        except ValueError as exc:
            raise DataError(f"图像文件内容无效: {path}: {exc}") from exc
        return image

    # ------------------------------------------------------------------- json
    def save_json(self, payload: Any, name: str, subdir: str = "reports") -> Path:
        """保存任意可 JSON 序列化对象。"""
        if subdir not in self.dirs:
            raise DataError(f"未知输出子目录: {subdir}")
        path = self.dirs[subdir] / f"{name}.json"
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise DataError(f"保存 JSON 失败: {path}: {exc}") from exc
        logger.info("JSON 已保存: %s", path)
        return path
