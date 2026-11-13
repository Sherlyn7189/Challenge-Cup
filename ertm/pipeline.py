"""流水线编排。

``ERTMPipeline`` 是模块之间的唯一“组装点”，顺序为::

    配置 → 模型 → 观测系统 → 正演（多炮） → 采集效应/预处理
         → RTM 成像（可选 Q 补偿） → 叠加 → 评价 → 输出

它只负责调用接口、组织产物和写报告，不实现任何数值算法。这样替换
求解器/成像器/评价器时，``main.py`` 与测试都不需要改。

多炮处理
--------
每炮独立成像，按炮做振幅归一化后叠加（技术方案 6.3.5）；单炮时等价于直接
输出。回归对比（Q=∞ / 含 Q 不补偿 / 含 Q 补偿）由 ``q`` 配置段控制，并在
报告里记录实际使用的补偿强度与增益上限。
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import qc
from .config import PROJECT_ROOT, AppConfig, save_config
from .data import NumpyDataStore
from .evaluation import build_evaluator
from .exceptions import PipelineError
from .imaging import build_imager
from .interfaces import DataStore, Evaluator, ForwardSolver, Imager, ModelBuilder, Visualizer
from .logging_utils import get_logger
from .models import LayeredModelBuilder
from .preprocess import prepare_shot, resample_shot
from .qcomp import QCompensator, attenuate_record, build_compensator
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
    shots: list[ShotRecord] = field(default_factory=list)
    shot_images: list[ImageResult] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    report_path: Path | None = None
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "project": self.config.project.name,
            "elapsed_s": round(self.elapsed_s, 3),
            "model_shape": list(self.model.shape),
            "n_receivers": self.shot.receivers.nrec,
            "n_shots": len(self.shots) or 1,
            "n_time_samples": int(self.shot.t.size),
            "image_mode": self.image.mode,
            "placeholder": bool(self.image.meta.get("placeholder", False)),
            "engine": self.image.meta.get("engine"),
            "condition": self.image.meta.get("condition"),
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
    """按观测系统生成检波点（地面排列或井中排列）。"""
    acquisition = config.acquisition
    count = int(config.receivers.count)
    if acquisition.geometry == "borehole":
        z = np.linspace(
            float(acquisition.borehole_z_start),
            float(acquisition.borehole_z_end),
            count,
            dtype=float,
        )
        x = np.full_like(z, float(acquisition.borehole_x))
    else:
        x = np.linspace(
            float(config.receivers.x_start),
            float(config.receivers.x_end),
            count,
            dtype=float,
        )
        z = np.full_like(x, float(config.receivers.z))
    return ReceiverSpec(x=x, z=z, component=config.receivers.component)


def _build_source(config: AppConfig, x: float | None = None) -> SourceSpec:
    return SourceSpec(
        x=float(config.source.x if x is None else x),
        z=float(config.source.z),
        f0=float(config.source.f0),
        kind=config.source.kind,
        amplitude=float(config.source.amplitude),
        delay=float(config.source.delay),
        mechanism=config.source.mechanism,
    )


def _build_sources(config: AppConfig) -> list[SourceSpec]:
    positions = tuple(config.acquisition.shots) or (float(config.source.x),)
    return [_build_source(config, x=float(x)) for x in positions]


def stack_images(
    images: list[ImageResult],
    *,
    normalize: bool = True,
) -> ImageResult:
    """按炮归一化后叠加（技术方案 6.3.5）。"""
    if not images:
        raise PipelineError("没有可叠加的成像结果")
    if len(images) == 1:
        return images[0]
    stacked = np.zeros(images[0].image.shape, dtype=np.float64)
    for image in images:
        data = np.asarray(image.image, dtype=np.float64)
        if normalize:
            scale = float(np.max(np.abs(data)))
            if scale > 0:
                data = data / scale
        stacked += data
    stacked /= len(images)
    meta = dict(images[0].meta)
    meta.update(
        {"stacked": True, "n_shots": len(images), "stack_normalize": bool(normalize)}
    )
    return ImageResult(
        image=stacked.astype(np.float32),
        grid=images[0].grid,
        mode=images[0].mode,
        meta=meta,
    )


class ERTMPipeline:
    """ERTM 流水线（真实正演 + 逆时偏移）。"""

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
        self.solver = solver or build_solver(config.solver, config.q)
        self.imager = imager or build_imager(config.imaging, config.solver, config.q)
        self.visualizer = visualizer or build_visualizer(
            self.data_store.dirs["images"],
            enabled=config.output.save_figures,
        )
        self.evaluator = evaluator or build_evaluator()

    # ------------------------------------------------------------------ 主流程
    def run(self) -> PipelineResult:
        """执行完整流水线并返回产物。"""
        started = time.perf_counter()
        artifacts: list[Path] = []
        config = self.config

        logger.info("=" * 72)
        logger.info("ERTM 流水线启动: %s", config.project.name)
        logger.info(
            "网格 nx=%d nz=%d dx=%.1f dz=%.1f | 引擎=%s | nt=%d dt=%.2e | 观测=%s",
            config.grid.nx,
            config.grid.nz,
            config.grid.dx,
            config.grid.dz,
            config.solver.engine,
            config.solver.nt,
            config.solver.dt,
            config.acquisition.geometry,
        )

        # 1) 配置快照
        snapshot = save_config(
            config,
            self.data_store.dirs["reports"] / "config_snapshot.json",
        )
        artifacts.append(snapshot)

        # 2) 模型
        try:
            model = self.model_builder.build(config)
        except Exception as exc:
            raise PipelineError(f"模型构建阶段失败: {exc}") from exc
        if config.output.save_data:
            artifacts.append(self.data_store.save_model(model, "model"))
        if config.output.save_figures:
            figure = self.visualizer.plot_model(model, "model")
            if figure is not None:
                artifacts.append(figure)

        # 3) 观测系统 / 数据输入
        if config.input.format == "segy":
            shots = self._load_segy_shots()
            if not shots:
                raise PipelineError("SEG-Y 输入没有读到任何炮记录")
            receivers = shots[0].receivers
            sources = [shot.source for shot in shots]
            logger.info(
                "观测系统（SEG-Y 输入）: %d 炮 × %d 道, 来源=%s",
                len(shots),
                receivers.nrec,
                config.input.segy_path or list(config.input.component_files),
            )
            shots = [
                self._apply_acquisition(shot, model, index)
                for index, shot in enumerate(shots)
            ]
        else:
            receivers = _build_receivers(config)
            sources = _build_sources(config)
            logger.info("观测系统: %d 炮 × %d 道", len(sources), receivers.nrec)
            # 4) 正演 + 采集效应
            shots = []
            for index, source in enumerate(sources):
                try:
                    shot = self.solver.forward(
                        model,
                        source,
                        receivers,
                        components=self._record_components(),
                    )
                    shot = self._apply_acquisition(shot, model, index)
                except Exception as exc:
                    raise PipelineError(f"第 {index} 炮正演阶段失败: {exc}") from exc
                shots.append(shot)

        if config.output.save_data:
            artifacts.append(self.data_store.save_shot(shots[0], "shot"))
        if config.output.save_figures:
            figure = self.visualizer.plot_shot(shots[0], "shot")
            if figure is not None:
                artifacts.append(figure)

        # 5) 成像（可选 Q 补偿）
        compensator = self._build_compensator(model)
        shot_images: list[ImageResult] = []
        for index, shot in enumerate(shots):
            try:
                image = self.imager.migrate(shot, model, preprocess=compensator)
            except Exception as exc:
                raise PipelineError(f"第 {index} 炮成像阶段失败: {exc}") from exc
            image.meta["shot_index"] = index
            shot_images.append(image)

        image = stack_images(
            shot_images,
            normalize=bool(config.imaging.stack_normalize),
        )

        # 5.5) Q 补偿对比：同一观测、同一成像参数，只改反传是否补偿
        if self._should_compare_q():
            artifacts.extend(self._run_q_comparison(shots, model, image))

        if config.output.save_data:
            artifacts.append(self.data_store.save_image(image, f"image_{image.mode}"))
        if config.output.segy_image:
            from .segy import write_segy_image

            artifacts.append(
                write_segy_image(
                    image,
                    self.data_store.dirs["images"] / f"image_{image.mode}.sgy",
                )
            )
        if config.output.save_figures:
            figure = self.visualizer.plot_image(image, f"image_{image.mode}")
            if figure is not None:
                artifacts.append(figure)

        # 6) 评价
        try:
            metrics = self.evaluator.evaluate(image, model, shots[0])
        except Exception as exc:
            raise PipelineError(f"评价阶段失败: {exc}") from exc
        artifacts.append(self.data_store.save_json(metrics.to_dict(), "metrics"))
        if config.output.save_figures:
            figure = self.visualizer.plot_metrics(metrics, "metrics")
            if figure is not None:
                artifacts.append(figure)

        # 7) 运行报告
        elapsed = time.perf_counter() - started
        report = {
            "project": config.project.name,
            "version": config.project.version,
            "placeholder": False,
            "method": "reverse-time migration (two-way wave equation)",
            "elapsed_s": round(elapsed, 3),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "config": config.to_dict(),
            "model_summary": model.summary(),
            "acquisition": {
                "geometry": config.acquisition.geometry,
                "n_shots": len(sources),
                "n_receivers": receivers.nrec,
                "record_dt": config.acquisition.record_dt,
                "components": list(self._record_components()),
                "direct_mute": bool(config.acquisition.mute_direct),
            },
            "input": {
                "format": config.input.format,
                "segy_path": config.input.segy_path,
                "component_files": dict(config.input.component_files),
                "n_shots_loaded": len(shots),
            },
            "shot_meta": shots[0].meta,
            "image_meta": image.meta,
            "metrics": metrics.to_dict(),
            "q_comparison": getattr(self, "_q_comparison", None),
            "artifacts": [str(path) for path in artifacts],
        }
        report_path = self.data_store.save_json(report, "run_report")

        result = PipelineResult(
            config=config,
            model=model,
            shot=shots[0],
            image=image,
            metrics=metrics,
            shots=shots,
            shot_images=shot_images,
            artifacts=[*artifacts, report_path],
            report_path=report_path,
            elapsed_s=elapsed,
        )
        logger.info("ERTM 流水线完成: %.3f s, 产物 %d 个", elapsed, len(result.artifacts))
        logger.info("运行报告: %s", report_path)
        logger.info("=" * 72)
        return result

    # ---------------------------------------------------------------- 辅助
    def _record_components(self) -> tuple[str, ...]:
        """按观测系统选择记录分量（1 或 4）。"""
        n = int(self.config.acquisition.components)
        if self.config.solver.engine == "acoustic":
            return ("p",)
        return ("p", "vx", "vz", "sxz") if n >= 4 else ("vz",)

    def _apply_acquisition(
        self,
        shot: ShotRecord,
        model: ElasticModel,
        index: int,
    ) -> ShotRecord:
        """施加采集效应：采样率 → 旋转 → 切除 → taper → 可选 Q 衰减。"""
        config = self.config
        processed = shot
        if config.acquisition.record_dt is not None:
            processed = resample_shot(processed, float(config.acquisition.record_dt))
        processed = prepare_shot(
            processed,
            model=model,
            rotation_deg=float(config.acquisition.rotation_deg),
            mute=bool(config.acquisition.mute_direct),
        )
        if (
            config.q.enabled
            and config.q.apply_attenuation
            and config.q.model == "filter"
            and config.input.format == "synthetic"
        ):
            q_value = float(np.mean(model.qp))
            processed.data = attenuate_record(
                np.asarray(processed.data, dtype=np.float64),
                processed.dt,
                q=q_value,
                f_ref=float(config.q.f_ref),
                strength=float(config.q.attenuation_strength),
            ).astype(np.float32)
            if processed.components:
                processed.components = {
                    name: attenuate_record(
                        np.asarray(array, dtype=np.float64),
                        processed.dt,
                        q=q_value,
                        f_ref=float(config.q.f_ref),
                        strength=float(config.q.attenuation_strength),
                    ).astype(np.float32)
                    for name, array in processed.components.items()
                }
            processed.meta = {**processed.meta, "q_attenuated": True}
        processed.meta = {**processed.meta, "shot_index": index}
        return processed

    def _load_segy_shots(self) -> list[ShotRecord]:
        """按 ``input`` 配置读取 SEG-Y（单文件或多分量多文件）。"""
        from .segy import SegYReadOptions, read_segy_components, read_segy_shots

        cfg = self.config.input
        options = SegYReadOptions(
            component=cfg.component,
            group_by=cfg.group_by,
            receiver_depth=cfg.receiver_depth,
            source_depth=cfg.source_depth,
            receiver_depth_from=cfg.receiver_depth_from,
            coordinate_scalar=cfg.coordinate_scalar,
            shot_limit=cfg.shot_limit,
            sort_by_offset=cfg.sort_by_offset,
            f0=float(self.config.source.f0),
            delay=float(self.config.source.delay),
        )
        try:
            if cfg.component_files:
                return read_segy_components(dict(cfg.component_files), options)
            return read_segy_shots(cfg.segy_path, options)
        except Exception as exc:
            raise PipelineError(f"SEG-Y 读取失败: {exc}") from exc

    def _should_compare_q(self) -> bool:
        config = self.config
        return bool(
            config.q.enabled
            and config.q.compare
            and config.solver.engine == "viscoacoustic"
        )

    def _run_q_comparison(
        self,
        shots: list[ShotRecord],
        model: ElasticModel,
        compensated: ImageResult,
    ) -> list[Path]:
        """用同一批记录、同一成像参数跑一遍“不补偿”的 RTM，输出对比与 Q 模型。"""
        config = self.config
        baseline_q = replace(config.q, enabled=False, compare=False)
        baseline_imager = build_imager(config.imaging, config.solver, baseline_q)
        baseline_images: list[ImageResult] = []
        for shot in shots:
            baseline_images.append(baseline_imager.migrate(shot, model))
        baseline = stack_images(
            baseline_images, normalize=bool(config.imaging.stack_normalize)
        )
        metrics = qc.comparison_metrics(baseline, compensated, model)
        self._q_comparison = metrics
        logger.info(
            "Q 补偿对比: 深部窗口增益 %.2fx（%.0f–%.0f m）",
            metrics["band_amplitude_gain"],
            metrics["band_top_m"],
            metrics["band_bottom_m"],
        )
        artifacts: list[Path] = []
        png, json_path = qc.save_comparison(
            baseline,
            compensated,
            model,
            self.data_store.dirs["images"],
        )
        artifacts.extend([png, json_path])
        q_npz, q_png = qc.save_q_model(
            model,
            self.data_store.dirs["models"],
        )
        artifacts.extend([q_npz, q_png])
        meta = compensated.meta.get("q", {})
        gain_curve = qc.save_gain_curve(
            meta,
            self.data_store.dirs["images"],
            dt=float(compensated.meta.get("dt", config.solver.dt)),
            nt=int(compensated.meta.get("nt", config.solver.nt)),
        )
        if gain_curve is not None:
            artifacts.append(gain_curve)
        return artifacts

    def _build_compensator(self, model: ElasticModel) -> QCompensator | None:
        config = self.config
        if not config.q.enabled:
            return None
        # GSLS 模型的补偿在反传算子内部完成（imaging 负责），数据域不再重复补偿
        if config.q.model == "gsls":
            return None
        wave = "P" if config.imaging.mode in {"pp", "ps"} else "S"
        compensator = build_compensator(
            config.q,
            float(np.mean(model.qp)),
            float(np.mean(model.qs)),
            wave=wave,
        )
        logger.info(
            "Q 补偿: alpha_c=%.2f, Q=%s%.1f, 增益上限=%.1f",
            config.q.alpha_c,
            wave,
            compensator.q_value,
            config.q.gain_cap,
        )
        return compensator


def run_pipeline(config: AppConfig) -> PipelineResult:
    """模块级便捷函数。"""
    return ERTMPipeline(config).run()
