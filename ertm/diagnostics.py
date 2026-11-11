"""项目结构自检。

用途：
- ``python main.py --check``：不跑完整流程，快速检查目录、导入、配置和
  各模块的最小契约；
- ``tests/test_structure.py``：把同样的检查固化为单元测试。

检查只做“能不能组装起来”，不做数值精度判断。
"""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (
    PROJECT_ROOT,
    AppConfig,
    GridConfig,
    ReceiverConfig,
    SolverConfig,
    load_config,
)
from .data import ensure_output_dirs
from .evaluation import BasicEvaluator
from .models import LayeredModelBuilder


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        return f"[{status}] {self.name}" + (f": {self.detail}" if self.detail else "")


REQUIRED_FILES = (
    "main.py",
    "config/default.json",
    "ertm/__init__.py",
    "ertm/config.py",
    "ertm/types.py",
    "ertm/interfaces.py",
    "ertm/exceptions.py",
    "ertm/logging_utils.py",
    "ertm/numerics.py",
    "ertm/propagation.py",
    "ertm/models.py",
    "ertm/solver.py",
    "ertm/imaging.py",
    "ertm/preprocess.py",
    "ertm/qcomp.py",
    "ertm/data.py",
    "ertm/visualization.py",
    "ertm/evaluation.py",
    "ertm/pipeline.py",
    "ertm/diagnostics.py",
    "tests/__init__.py",
    "tests/test_structure.py",
    "tests/test_rtm.py",
)

REQUIRED_DIRS = (
    "config",
    "ertm",
    "tests",
    "output",
    "output/logs",
    "output/models",
    "output/shots",
    "output/images",
    "output/reports",
)

MODULES = (
    "ertm",
    "ertm.config",
    "ertm.types",
    "ertm.interfaces",
    "ertm.exceptions",
    "ertm.logging_utils",
    "ertm.numerics",
    "ertm.propagation",
    "ertm.models",
    "ertm.solver",
    "ertm.imaging",
    "ertm.preprocess",
    "ertm.qcomp",
    "ertm.data",
    "ertm.visualization",
    "ertm.evaluation",
    "ertm.pipeline",
    "ertm.diagnostics",
)


def _tiny_config(config: AppConfig) -> AppConfig:
    """构造一个 41x31、120 步的微型配置，用于快速契约检查。"""
    anomaly = config.model.anomaly
    if anomaly is not None:
        anomaly = replace(anomaly, x=200.0, z=180.0, rx=30.0, rz=20.0)
    model = replace(config.model, anomaly=anomaly)
    return replace(
        config,
        grid=GridConfig(nx=41, nz=31, dx=10.0, dz=10.0, x0=0.0, z0=0.0),
        model=model,
        source=replace(config.source, x=200.0, z=20.0, f0=50.0, delay=0.01),
        receivers=ReceiverConfig(count=9, x_start=0.0, x_end=400.0, z=20.0),
        solver=SolverConfig(
            nt=400,
            dt=2.5e-4,
            order=8,
            boundary="free",
            dtype="float32",
            engine="acoustic",
            npml=6,
        ),
    )


def _check_paths(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    results.append(
        CheckResult(
            "必需文件",
            not missing,
            "全部存在" if not missing else f"缺少: {', '.join(missing)}",
        )
    )
    missing_dirs = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    results.append(
        CheckResult(
            "必需目录",
            not missing_dirs,
            "全部存在" if not missing_dirs else f"缺少: {', '.join(missing_dirs)}",
        )
    )
    return results


def _check_imports() -> list[CheckResult]:
    results: list[CheckResult] = []
    failures: list[str] = []
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - 需要把所有导入错误汇总
            failures.append(f"{module_name}: {exc}")
    results.append(
        CheckResult(
            "模块导入",
            not failures,
            "全部可导入" if not failures else "; ".join(failures),
        )
    )
    return results


def _check_config() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        config = load_config()
        results.append(CheckResult("默认配置", True, f"{config.project.name} v{config.project.version}"))
        tiny = _tiny_config(config)
        results.append(
            CheckResult(
                "微型配置",
                tiny.grid.shape == (31, 41),
                f"shape={tiny.grid.shape}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("默认配置", False, str(exc)))
    return results


def _check_components() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        from .imaging import build_imager
        from .solver import build_solver

        config = _tiny_config(load_config())
        model = LayeredModelBuilder().build(config)
        results.append(
            CheckResult("模型构建", model.shape == config.grid.shape, f"shape={model.shape}")
        )

        from .pipeline import _build_receivers, _build_source

        source = _build_source(config)
        receivers = _build_receivers(config)
        solver = build_solver(config.solver)
        shot = solver.forward(model, source, receivers)
        results.append(
            CheckResult(
                "正演接口",
                shot.data.shape == (config.solver.nt, receivers.nrec)
                and np.all(np.isfinite(shot.data)),
                f"shape={shot.data.shape}, 峰值={np.abs(shot.data).max():.3e}",
            )
        )

        imager = build_imager(config.imaging, config.solver)
        image = imager.migrate(shot, model)
        results.append(
            CheckResult(
                "成像接口",
                image.image.shape == model.shape and np.all(np.isfinite(image.image)),
                f"shape={image.image.shape}, 峰值={np.abs(image.image).max():.3e}",
            )
        )

        metrics = BasicEvaluator().evaluate(image, model, shot)
        results.append(
            CheckResult(
                "评价接口",
                bool(metrics.values) and np.isfinite(list(metrics.values.values())).all(),
                f"{len(metrics.values)} 项指标",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("模块契约", False, f"{type(exc).__name__}: {exc}"))
    return results


def _check_output() -> list[CheckResult]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dirs = ensure_output_dirs(Path(tmp) / "output")
            missing = [name for name in ("logs", "models", "shots", "images", "reports")
                       if not dirs[name].is_dir()]
        return [
            CheckResult(
                "输出目录",
                not missing,
                "可创建" if not missing else f"缺少: {', '.join(missing)}",
            )
        ]
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("输出目录", False, str(exc))]


def run_checks(root: str | Path | None = None) -> list[CheckResult]:
    """运行全部结构检查，返回结果列表。"""
    base = Path(root).expanduser().resolve() if root is not None else PROJECT_ROOT
    results: list[CheckResult] = []
    results.extend(_check_paths(base))
    results.extend(_check_imports())
    results.extend(_check_config())
    results.extend(_check_components())
    results.extend(_check_output())
    return results


def format_checks(results: Iterable[CheckResult]) -> str:
    return "\n".join(item.line() for item in results)
