"""ERTM 成像算法模块（首期为占位实现）。

职责边界
--------
- 对外提供统一的 ``migrate`` 接口：炮记录 + 模型 -> 成像结果；
- 预留 ``image_condition`` 接口，后续实现 PP/PS 互相关、归一化和
  Laplacian 滤波；
- 当前返回零图像，并在 ``meta`` 中标记 ``placeholder=True``，避免被
  误认为真实偏移结果。

注意：本模块不负责正演、不负责 Q 补偿、不负责绘图。这些分别属于
``solver.py``、后续 Q 模块和 ``visualization.py``。
"""

from __future__ import annotations

import numpy as np

from .config import ImagingConfig
from .exceptions import ImagingError
from .logging_utils import get_logger
from .types import ElasticModel, ImageResult, ShotRecord

logger = get_logger(__name__)


class ERTMImager:
    """ERTM 成像器占位实现。"""

    name = "ertm-imager-placeholder"

    def __init__(self, config: ImagingConfig) -> None:
        self.config = config
        self.config.validate()

    def migrate(self, shot: ShotRecord, model: ElasticModel) -> ImageResult:
        """把单炮记录偏移为图像。

        当前实现只做输入校验和占位输出，不进行波场反传。真实实现将：
        1. 反传检波点残差；
        2. 与源波场做零延迟互相关；
        3. 归一化、Laplacian、叠加。
        """
        try:
            shot.validate()
            model.validate()
        except ValueError as exc:
            raise ImagingError(f"成像输入无效: {exc}") from exc

        image = np.zeros(model.shape, dtype=np.float32)
        result = ImageResult(
            image=image,
            grid=model.grid,
            mode=self.config.mode,
            meta={
                "placeholder": True,
                "method": "zero-field",
                "imager": self.name,
                "normalize": self.config.normalize,
                "laplacian": self.config.laplacian,
                "shot_placeholder": bool(shot.meta.get("placeholder", False)),
            },
        )
        result.validate()
        logger.info(
            "占位成像完成: mode=%s, shape=%s, 输入道数=%d",
            self.config.mode,
            image.shape,
            shot.receivers.nrec,
        )
        return result

    def image_condition(
        self,
        source_wavefield: np.ndarray,
        receiver_wavefield: np.ndarray,
    ) -> np.ndarray:
        """占位成像条件：返回零场。

        真实实现将对源/检波点波场做 P/S 分离后互相关，并按源照明归一化。
        """
        source = np.asarray(source_wavefield)
        receiver = np.asarray(receiver_wavefield)
        if source.shape != receiver.shape:
            raise ImagingError(
                f"成像条件输入形状不一致: {source.shape} vs {receiver.shape}"
            )
        return np.zeros_like(source, dtype=np.float32)


def build_imager(config: ImagingConfig) -> ERTMImager:
    """成像器工厂。"""
    return ERTMImager(config)
