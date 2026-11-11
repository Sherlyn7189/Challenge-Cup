"""算法评价模块。

提供两类指标：

- **图像统计**：幅值范围、均值、标准差、非零占比；
- **成像质量代理指标**（不依赖真实反射率模型）：
  ``image_peak_x/z``（最强成像点位置，米）、``image_energy``（总能量）、
  ``deep_shallow_energy_ratio``（深部/浅部能量比，用于观察归一化与 Q 补偿
  对深部照明的影响）、``image_focusing``（最强点能量与周围环带能量之比）。

真正的定量评价（界面归位误差、AVA、道集平坦度）需要反射率参考模型，
属于第二阶段；本模块不读文件、不画图、不做物理计算。
"""

from __future__ import annotations

import numpy as np

from .exceptions import EvaluationError
from .logging_utils import get_logger
from .types import ElasticModel, ImageResult, Metrics, ShotRecord

logger = get_logger(__name__)


class BasicEvaluator:
    """基础统计 + 成像质量代理指标评价器。"""

    name = "basic-evaluator"

    def evaluate(
        self,
        image: ImageResult,
        model: ElasticModel,
        shot: ShotRecord | None = None,
        reference: ImageResult | None = None,
    ) -> Metrics:
        """计算基础指标；若给定参考图像，附加误差指标。"""
        try:
            image.validate()
            model.validate()
            if shot is not None:
                shot.validate()
            if reference is not None:
                reference.validate()
        except ValueError as exc:
            raise EvaluationError(f"评价输入无效: {exc}") from exc

        data = image.image.astype(np.float64, copy=False)
        values: dict[str, float] = {
            "image_min": float(np.min(data)),
            "image_max": float(np.max(data)),
            "image_mean": float(np.mean(data)),
            "image_std": float(np.std(data)),
            "image_nonzero_fraction": float(np.count_nonzero(data) / data.size),
            "model_vp_min": float(np.min(model.vp)),
            "model_vp_mean": float(np.mean(model.vp)),
            "model_vp_max": float(np.max(model.vp)),
            "model_vs_vp_ratio_mean": float(np.mean(model.vs / model.vp)),
        }
        values.update(self.image_quality(data, image, model))

        if shot is not None:
            shot_data = shot.data.astype(np.float64, copy=False)
            values["shot_abs_max"] = float(np.max(np.abs(shot_data)))
            values["shot_rms"] = float(np.sqrt(np.mean(shot_data**2)))

        if reference is not None:
            values.update(self.compare_images(image, reference))

        metrics = Metrics(
            values=values,
            meta={
                "placeholder": bool(image.meta.get("placeholder", False)),
                "evaluator": self.name,
                "image_mode": image.mode,
                "engine": image.meta.get("engine"),
                "condition": image.meta.get("condition"),
                "has_shot": shot is not None,
                "has_reference": reference is not None,
            },
        )
        logger.info("基础评价完成: %d 项指标", len(metrics.values))
        return metrics

    @staticmethod
    def image_quality(
        data: np.ndarray,
        image: ImageResult,
        model: ElasticModel,
    ) -> dict[str, float]:
        """成像质量代理指标（量纲统一到米/无量纲）。"""
        magnitude = np.abs(np.asarray(data, dtype=np.float64))
        total = float(np.sum(magnitude**2))
        if magnitude.size == 0 or total <= 0:
            return {
                "image_peak_x": 0.0,
                "image_peak_z": 0.0,
                "image_energy": 0.0,
                "deep_shallow_energy_ratio": 0.0,
                "image_focusing": 0.0,
            }
        iz, ix = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        grid = image.grid
        nz = magnitude.shape[0]
        half = max(nz // 2, 1)
        shallow = float(np.sum(magnitude[:half] ** 2))
        deep = float(np.sum(magnitude[half:] ** 2))
        # 聚焦度：以 3×3 邻域能量占 11×11 邻域能量的比例衡量“点状/成层”程度
        iz0, iz1 = max(iz - 1, 0), min(iz + 2, nz)
        ix0, ix1 = max(ix - 1, 0), min(ix + 2, magnitude.shape[1])
        local = float(np.sum(magnitude[iz0:iz1, ix0:ix1] ** 2))
        iz0w, iz1w = max(iz - 5, 0), min(iz + 6, nz)
        ix0w, ix1w = max(ix - 5, 0), min(ix + 6, magnitude.shape[1])
        window = float(np.sum(magnitude[iz0w:iz1w, ix0w:ix1w] ** 2))
        return {
            "image_peak_x": float(grid.x_at(ix)),
            "image_peak_z": float(grid.z_at(iz)),
            "image_energy": total,
            "deep_shallow_energy_ratio": float(deep / shallow) if shallow > 0 else 0.0,
            "image_focusing": float(local / window) if window > 0 else 0.0,
        }

    @staticmethod
    def compare_images(predicted: ImageResult, reference: ImageResult) -> dict[str, float]:
        """计算两幅图像的 RMSE、相对 L2 误差和相关系数。"""
        if predicted.image.shape != reference.image.shape:
            raise EvaluationError(
                f"图像形状不一致: {predicted.image.shape} vs {reference.image.shape}"
            )
        pred = predicted.image.astype(np.float64, copy=False).ravel()
        ref = reference.image.astype(np.float64, copy=False).ravel()
        diff = pred - ref
        rmse = float(np.sqrt(np.mean(diff**2)))
        ref_norm = float(np.linalg.norm(ref))
        relative_l2 = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else float("inf")
        pred_std = float(np.std(pred))
        ref_std = float(np.std(ref))
        if pred_std > 0 and ref_std > 0:
            correlation = float(np.corrcoef(pred, ref)[0, 1])
        else:
            correlation = 0.0
        return {
            "rmse": rmse,
            "relative_l2": relative_l2,
            "correlation": correlation,
        }


def build_evaluator() -> BasicEvaluator:
    """评价器工厂。"""
    return BasicEvaluator()
