"""ERTM 统一异常体系。

所有业务异常都继承 :class:`ERTMError`，便于入口层统一捕获、记录日志并
返回稳定的退出码。异常对象可以序列化为字典，用于写入运行报告。
"""

from __future__ import annotations

from typing import Any


class ERTMError(Exception):
    """ERTM 所有可预期业务异常的基类。"""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self.message = str(message)
        self.context = dict(context or {})
        super().__init__(self.message)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({detail})"

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的异常描述。"""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "context": self.context,
        }


class ConfigError(ERTMError):
    """配置缺失、类型错误或参数越界。"""


class ModelError(ERTMError):
    """地质模型构建或模型参数校验失败。"""


class SolverError(ERTMError):
    """正演/伴随求解器输入或运行失败。"""


class ImagingError(ERTMError):
    """成像输入或成像条件失败。"""


class DataError(ERTMError):
    """数据读写、格式或元数据失败。"""


class VisualizationError(ERTMError):
    """绘图或图像输出失败。"""


class EvaluationError(ERTMError):
    """评价指标计算失败。"""


class PipelineError(ERTMError):
    """流程编排失败（通常是多个模块的组合错误）。"""
