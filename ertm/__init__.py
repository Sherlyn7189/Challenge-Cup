"""ERTM：弹性波逆时偏移项目骨架。

首期只提供可运行的工程结构、统一接口和占位算法；真正的黏弹性正演、
P/S 分离、PP/PS 成像和 Q 补偿在后续阶段实现。

快速入口::

    from ertm.config import load_config
    from ertm.pipeline import ERTMPipeline

    result = ERTMPipeline(load_config()).run()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import AppConfig, load_config, save_config
from .exceptions import (
    ConfigError,
    DataError,
    ERTMError,
    EvaluationError,
    ImagingError,
    ModelError,
    PipelineError,
    SolverError,
    VisualizationError,
)
from .types import (
    ElasticModel,
    GridSpec,
    ImageResult,
    Metrics,
    ReceiverSpec,
    ShotRecord,
    SourceSpec,
)

__all__ = [
    "__version__",
    "AppConfig",
    "load_config",
    "save_config",
    "ElasticModel",
    "GridSpec",
    "ImageResult",
    "Metrics",
    "ReceiverSpec",
    "ShotRecord",
    "SourceSpec",
    "ERTMError",
    "ConfigError",
    "ModelError",
    "SolverError",
    "ImagingError",
    "DataError",
    "VisualizationError",
    "EvaluationError",
    "PipelineError",
]
