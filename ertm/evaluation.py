"""算法评价模块。

首期只提供“结构/统计型”指标，用于确认数据形状、量级和占位流程是否正常；
真正的成像质量评价（界面归位、AVA、道集平坦度、Q 补偿频谱比等）在后续
阶段实现。

本模块不读文件、不画图、不做物理计算，只接收内存中的数组/对象。
"""

from __future__ import annotations

import numpy as np

from .exceptions import EvaluationError
from .logging_utils import get_logger
from .types import ElasticModel, ImageResult, Metrics, ShotRecord

logger = get_logger(__name__)


class BasicEvaluator:
    """基础统计评价器。"""

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
        if shot is not None:
            shot_data = shot.data.astype(np.float64, copy=False)
            values["shot_abs_max"] = float(np.max(np.abs(shot_data)))
            values["shot_rms"] = float(np.sqrt(np.mean(shot_data**2)))

        if reference is not None:
            values.update(self.compare_images(image, reference))

        metrics = Metrics(
            values=values,
            meta={
                "placeholder": True,
                "evaluator": self.name,
                "image_mode": image.mode,
                "has_shot": shot is not None,
                "has_reference": reference is not None,
            },
        )
        logger.info("基础评价完成: %d 项指标", len(metrics.values))
        return metrics

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
