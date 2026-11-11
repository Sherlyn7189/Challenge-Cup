"""模块之间的统一接口（Protocol）。

使用 ``typing.Protocol`` 而不是强制继承，目的是：

- 各模块可以独立替换（占位实现 / 正式实现 / 测试替身）；
- 流水线只依赖接口，不依赖具体类；
- 不引入额外的基类层次，保持低耦合。

所有接口的输入/输出都使用 :mod:`ertm.types` 中的数据契约。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import AppConfig
from .types import (
    ElasticModel,
    ImageResult,
    Metrics,
    ReceiverSpec,
    ShotRecord,
    SourceSpec,
)


@runtime_checkable
class ModelBuilder(Protocol):
    """地质模型构建接口。"""

    name: str

    def build(self, config: AppConfig) -> ElasticModel:
        ...


@runtime_checkable
class ForwardSolver(Protocol):
    """正演求解器接口（双程波延拓，多分量记录）。"""

    name: str

    def forward(
        self,
        model: ElasticModel,
        source: SourceSpec,
        receivers: ReceiverSpec,
        *,
        components: tuple[str, ...] | None = ...,
    ) -> ShotRecord:
        ...


@runtime_checkable
class Imager(Protocol):
    """成像算法接口。

    ``migrate`` 内部完成“正传 + 反传 + 成像条件”，是 RTM 的唯一入口；
    ``image_condition`` 只提供逐时刻的被积函数，供 QC 与自定义叠加使用。
    """

    name: str

    def migrate(
        self,
        shot: ShotRecord,
        model: ElasticModel,
        *,
        wavelet: np.ndarray | None = ...,
        preprocess: Any = ...,
    ) -> ImageResult:
        ...

    def image_condition(
        self,
        source_wavefield: np.ndarray,
        receiver_wavefield: np.ndarray,
    ) -> np.ndarray:
        ...


@runtime_checkable
class DataStore(Protocol):
    """数据持久化接口。"""

    name: str
    root: Path
    dirs: dict[str, Path]

    def save_model(self, model: ElasticModel, name: str = ...) -> Path:
        ...

    def load_model(self, path: str | Path) -> ElasticModel:
        ...

    def save_shot(self, shot: ShotRecord, name: str = ...) -> Path:
        ...

    def load_shot(self, path: str | Path) -> ShotRecord:
        ...

    def save_image(self, image: ImageResult, name: str = ...) -> Path:
        ...

    def load_image(self, path: str | Path) -> ImageResult:
        ...

    def save_json(self, payload: Any, name: str, subdir: str = ...) -> Path:
        ...


@runtime_checkable
class Visualizer(Protocol):
    """可视化接口。"""

    name: str

    def plot_model(self, model: ElasticModel, name: str = ...) -> Path | None:
        ...

    def plot_shot(self, shot: ShotRecord, name: str = ...) -> Path | None:
        ...

    def plot_image(self, image: ImageResult, name: str = ...) -> Path | None:
        ...

    def plot_metrics(self, metrics: Metrics, name: str = ...) -> Path | None:
        ...


@runtime_checkable
class Evaluator(Protocol):
    """评价接口。"""

    name: str

    def evaluate(
        self,
        image: ImageResult,
        model: ElasticModel,
        shot: ShotRecord | None = ...,
        reference: ImageResult | None = ...,
    ) -> Metrics:
        ...
