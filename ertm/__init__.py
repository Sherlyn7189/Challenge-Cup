"""ERTM：弹性波逆时偏移（双程波 RTM）项目。

当前版本实现：

- 二维一阶速度—应力（声波/弹性）有限差分正演，交错网格八阶、CFS-PML；
- 双程波逆时偏移：源波场正传 + 观测波场反传 + 互相关成像条件，
  支持零延迟、归一化（源照明）、激发振幅三种条件，Laplacian 与低通后处理；
- 井中场景支撑：四分量 COG→CSG 旋转、时间采样 sinc 插值、直达波切除、
  多分量弹性 RTM（PP/PS，散度/旋度 P/S 分离）；
- GSLS 记忆变量黏声正演与伴随算子内的 Q 补偿（负衰减 + 振幅上限 + 时变低通）；
- 数据域 Q 滤波与补偿（频域常 Q 算子，与 GSLS 二选一）；
- 检查点存储与按预算重算，多炮按炮归一化叠加。

快速入口::

    from ertm.config import load_config
    from ertm.pipeline import ERTMPipeline

    result = ERTMPipeline(load_config()).run()
    print(result.image.meta["condition"], result.metrics.values["image_peak_z"])
"""

from __future__ import annotations

__version__ = "0.3.0"

from .config import (
    AppConfig,
    AcquisitionConfig,
    ImagingConfig,
    QConfig,
    SolverConfig,
    load_config,
    save_config,
)
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
    "AcquisitionConfig",
    "ImagingConfig",
    "QConfig",
    "SolverConfig",
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
