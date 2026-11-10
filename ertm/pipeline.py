"""流水线编排。

``ERTMPipeline`` 是模块之间的唯一“组装点”：
配置 -> 模型 -> 观测系统 -> 正演 -> 成像 -> 评价 -> 输出。

它只负责调用接口、组织产物和写报告，不实现任何数值算法。这样替换
占位求解器/成像器时，``main.py`` 和测试都不需要改。
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, AppConfig, save_config
from .data import NumpyDataStore
from .evaluation import build_evaluator
from .exceptions import PipelineError
from .imaging import build_imager
from .interfaces import DataStore, Evaluator, ForwardSolver, Imager, ModelBuilder, Visualizer
from .logging_utils import get_logger
from .models import LayeredModelBuilder
from .solver import build_solver
from .types import (
    ElasticModel,
    ImageResult,
    Metrics,
    ReceiverSpec,
    ShotRecord,
    SourceSpec,
)
from .visualization import build_visualizer

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """一次完整运行的产物与元数据。"""

    config: AppConfig
    model: ElasticModel
    shot: ShotRecord
    image: ImageResult
    metrics: Metrics
    artifacts: list[Path] = field(default_factory=list)
    report_path: Path | None = None
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "project": self.config.project.name,
            "elapsed_s": round(self.elapsed_s, 3),
            "model_shape": list(self.model.shape),
            "n_receivers": self.shot.receivers.nrec,
            "n_time_samples": int(self.shot.t.size),
            "image_mode": self.image.mode,
            "placeholder": bool(self.image.meta.get("placeholder", False)),
            "artifacts": [str(p) for p in self.artifacts],
            "report": None if self.report_path is None else str(self.report_path),
            "metrics": dict(self.metrics.values),
        }


def _resolve_output_root(config: AppConfig) -> Path:
    root = Path(config.output.root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def _build_receivers(config: AppConfig) -> ReceiverSpec:
    x = np.linspace(
        config.receivers.x_start,
        config.receivers.x_end,
        int(config.receivers.count),
        dtype=float,
    )
    z = np.full_like(x, float(config.receivers.z))
    return ReceiverSpec(x=x, z=z, component=config.receivers.component)


def _build_source(config: AppConfig) -> SourceSpec:
    return SourceSpec(
        x=float(config.source.x),
        z=float(config.source.z),
        f0=float(config.source.f0),
        kind=config.source.kind,
        amplitude=float(config.source.amplitude),
        delay=float(config.source.delay),
    )


class ERTMPipeline:
    """ERTM 首期流水线（占位算法 + 完整工程链路）。"""

    def __init__(
        self,
        config: AppConfig,
        *,
        model_builder: ModelBuilder | None = None,
        solver: ForwardSolver | None = None,
        imager: Imager | None = None,
        data_store: DataStore | None = None,
        visualizer: Visualizer | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.config = config
        self.output_root = _resolve_output_root(config)

        self.data_store = data_store or NumpyDataStore(self.output_root)
        self.model_builder = model_builder or LayeredModelBuilder()
        self.solver = solver or build_solver(config.solver)
        self.imager = imager or build_imager(config.imaging)
        self.visualizer = visualizer or build_visualizer(
            self.data_store.dirs["images"],
            enabled=config.output.save_figures,
        )
        self.evaluator = evaluator or build_evaluator()

    def run(self) -> PipelineResult:
        """执行完整流水线并返回产物。"""
        started = time.perf_counter()
        artifacts: list[Path] = []

        logger.info("=" * 72)
        logger.info("ERTM 流水线启动: %s", self.config.project.name)
        logger.info(
            "网格: nx=%d, nz=%d, dx=%.2f, dz=%.2f, nt=%d, dt=%.4f",
            self.config.grid.nx,
            self.config.grid.nz,
            self.config.grid.dx,
            self.config.grid.dz,
            self.config.solver.nt,
            self.config.solver.dt,
        )

        # 1) 配置快照：无论是否保存数据，都保留一份可追溯配置。
        snapshot = save_config(
            self.config,
            self.data_store.dirs["reports"] / "config_snapshot.json",
        )
        artifacts.append(snapshot)

        # 2) 模型
        try:
            model = self.model_builder.build(self.config)
        except Exception as exc:
            raise PipelineError(f"模型构建阶段失败: {exc}") from exc

        if self.config.output.save_data:
            artifacts.append(self.data_store.save_model(model, "model"))
        if self.config.output.save_figures:
            figure = self.visualizer.plot_model(model, "model")
            if figure is not None:
                artifacts.append(figure)

        # 3) 观测系统
        source = _build_source(self.config)
        receivers = _build_receivers(self.config)

        # 4) 正演
        try:
            shot = self.solver.forward(model, source, receivers)
        except Exception as exc:
            raise PipelineError(f"正演阶段失败: {exc}") from exc

        if self.config.output.save_data:
            artifacts.append(self.data_store.save_shot(shot, "shot"))
        if self.config.output.save_figures:
            figure = self.visualizer.plot_shot(shot, "shot")
            if figure is not None:
                artifacts.append(figure)

        # 5) 成像
        try:
            image = self.imager.migrate(shot, model)
        except Exception as exc:
            raise PipelineError(f"成像阶段失败: {exc}") from exc

        if self.config.output.save_data:
            artifacts.append(self.data_store.save_image(image, f"image_{image.mode}"))
        if self.config.output.save_figures:
            figure = self.visualizer.plot_image(image, f"image_{image.mode}")
            if figure is not None:
                artifacts.append(figure)

        # 6) 评价
        try:
            metrics = self.evaluator.evaluate(image, model, shot)
        except Exception as exc:
            raise PipelineError(f"评价阶段失败: {exc}") from exc

        artifacts.append(self.data_store.save_json(metrics.to_dict(), "metrics"))
        if self.config.output.save_figures:
            figure = self.visualizer.plot_metrics(metrics, "metrics")
            if figure is not None:
                artifacts.append(figure)

        # 7) 运行报告
        elapsed = time.perf_counter() - started
        report = {
            "project": self.config.project.name,
            "version": self.config.project.version,
            "placeholder": True,
            "note": "首期占位实现：正演/成像为占位算法，不可用于物理解释。",
            "elapsed_s": round(elapsed, 3),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "config": self.config.to_dict(),
            "model_summary": model.summary(),
            "shot_meta": shot.meta,
            "image_meta": image.meta,
            "metrics": metrics.to_dict(),
            "artifacts": [str(path) for path in artifacts],
        }
        report_path = self.data_store.save_json(report, "run_report")

        result = PipelineResult(
            config=self.config,
            model=model,
            shot=shot,
            image=image,
            metrics=metrics,
            artifacts=[*artifacts, report_path],
            report_path=report_path,
            elapsed_s=elapsed,
        )
        logger.info("ERTM 流水线完成: %.3f s, 产物 %d 个", elapsed, len(result.artifacts))
        logger.info("运行报告: %s", report_path)
        logger.info("=" * 72)
        return result


def run_pipeline(config: AppConfig) -> PipelineResult:
    """模块级便捷函数。"""
    return ERTMPipeline(config).run()
