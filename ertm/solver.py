"""数值求解器模块（首期为占位实现）。

真正的交错网格黏弹性正演/伴随在后续阶段实现。本模块当前只提供：

- 统一的 ``forward`` / ``adjoint`` 接口；
- 一个可解释的占位炮记录：按常速度计算直达波走时，叠加 Ricker 子波；
- 输入校验和日志。

占位记录**不是**波动方程数值解，不能用于物理结论；它的作用是让
I/O、可视化、评价和流水线在早期就能端到端跑通。
"""

from __future__ import annotations

import numpy as np

from .config import SolverConfig
from .exceptions import SolverError
from .logging_utils import get_logger
from .types import ElasticModel, ReceiverSpec, ShotRecord, SourceSpec

logger = get_logger(__name__)


def _ricker(t: np.ndarray, t0: float, f0: float) -> np.ndarray:
    """Ricker 子波（无量纲），用于占位直达波。"""
    arg = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def _gaussian(t: np.ndarray, t0: float, f0: float) -> np.ndarray:
    """Gaussian 子波（无量纲），用于占位直达波。"""
    arg = np.pi * f0 * (t - t0)
    return np.exp(-0.5 * arg**2)


class PlaceholderForwardSolver:
    """占位正演求解器。

    只实现接口和运动学直达波，不做波场延拓。后续用真正的
    ``StaggeredGridSolver`` 替换本类即可，流水线无需改动。
    """

    name = "placeholder-forward-solver"

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.config.validate()

    def forward(
        self,
        model: ElasticModel,
        source: SourceSpec,
        receivers: ReceiverSpec,
    ) -> ShotRecord:
        """生成占位炮记录，形状 ``(nt, nrec)``。"""
        try:
            model.validate()
        except ValueError as exc:
            raise SolverError(f"正演输入模型无效: {exc}") from exc

        if source.kind == "custom":
            raise SolverError("custom 源需要外部时间序列，占位求解器暂不支持")
        if source.kind not in {"ricker", "gaussian"}:
            raise SolverError(f"不支持的源类型: {source.kind}")

        nt = int(self.config.nt)
        dt = float(self.config.dt)
        t = np.arange(nt, dtype=np.float64) * dt
        data = np.zeros((nt, receivers.nrec), dtype=np.float32)

        # 占位运动学：用模型平均速度估算直达波走时。
        v_mean = float(np.mean(model.vp))
        if v_mean <= 0:
            raise SolverError("模型平均速度非正，无法生成占位记录")

        wavelet = _ricker if source.kind == "ricker" else _gaussian
        for i, (x, z) in enumerate(zip(receivers.x, receivers.z)):
            distance = float(np.hypot(x - source.x, z - source.z))
            t0 = float(source.delay) + distance / v_mean
            data[:, i] = np.float32(source.amplitude * wavelet(t, t0, source.f0))

        record = ShotRecord(
            data=data,
            dt=dt,
            t=t,
            receivers=receivers,
            source=source,
            meta={
                "placeholder": True,
                "method": "constant-velocity-direct-arrival",
                "solver": self.name,
                "v_mean": v_mean,
                "nt": nt,
                "dt": dt,
            },
        )
        record.validate()
        logger.info(
            "占位正演完成: nt=%d, nrec=%d, v_mean=%.1f m/s",
            nt,
            receivers.nrec,
            v_mean,
        )
        return record

    def adjoint(self, record: ShotRecord, model: ElasticModel) -> np.ndarray:
        """占位伴随：返回与模型同形状的零场。

        真正的离散伴随在后续阶段实现；保留接口是为了让成像模块和
        测试可以提前对接。
        """
        record.validate()
        logger.info("占位伴随完成（零场）: shape=%s", model.shape)
        return np.zeros(model.shape, dtype=np.float32)


def build_solver(config: SolverConfig) -> PlaceholderForwardSolver:
    """求解器工厂：后续可在这里按 ``config`` 选择 CPU/GPU 后端。"""
    return PlaceholderForwardSolver(config)
