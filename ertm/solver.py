"""正演求解器（一阶速度—应力有限差分）。

本模块是 :mod:`ertm.propagation` 的“对接口的一层薄封装”：把
:class:`~ertm.config.SolverConfig` 翻译成 :class:`~ertm.propagation.PropagationSpec`，
按引擎选择声波/弹性波传播器，返回统一的 :class:`~ertm.types.ShotRecord`
（含多分量记录）。

之所以保留这一层而不是让流水线直接调用传播器：
- 流水线只依赖 ``ForwardSolver`` 协议，便于替换成解析解、GPU 后端或测试替身；
- ``dt``/``nt``/``order``/``npml`` 的校验集中在一处；
- 多分量记录的统一封装（主分量 + ``components`` 字典）只在这里做一次。
"""

from __future__ import annotations

import numpy as np

from .config import QConfig, SolverConfig
from .exceptions import SolverError
from .logging_utils import get_logger
from .propagation import (
    PropagationSpec,
    QOptions,
    check_stability,
    forward_shot,
    source_wavelet,
    stability_report,
)
from .types import ElasticModel, ReceiverSpec, ShotRecord, SourceSpec

logger = get_logger(__name__)


class FiniteDifferenceSolver:
    """一阶速度—应力方程的正演求解器。"""

    name = "fd-two-way"

    def __init__(self, config: SolverConfig, q_config: QConfig | None = None) -> None:
        self.config = config
        self.config.validate()
        self.q_config = q_config or QConfig()

    def q_options(self, *, compensation: bool = False) -> QOptions | None:
        """黏声引擎的 GSLS 选项；其他引擎返回 None。"""
        if self.config.engine != "viscoacoustic":
            return None
        return QOptions(
            enabled=True,
            f_ref=float(self.q_config.f_ref),
            n_mechanisms=int(self.q_config.n_mechanisms),
            band_ratio=float(self.q_config.band_ratio),
            compensation=bool(compensation),
            alpha_c=float(self.q_config.alpha_c),
            amplitude_cap=float(self.q_config.amplitude_cap),
            lowpass_interval=int(self.q_config.lowpass_interval),
        )

    # ---------------------------------------------------------------- 规格
    def spec(self, model: ElasticModel) -> PropagationSpec:
        spec = PropagationSpec.from_config(self.config, model, npml=self.config.npml)
        check_stability(model, spec)
        return spec

    def components(self, n_components: int = 1) -> tuple[str, ...]:
        """按“单分量/四分量”选择记录分量。

        二维 P-SV 系统只有 ``v_x, v_z`` 两个位移分量，四分量中的水平横向
        分量（SH）在二维近似下不参与，``p`` 由应力合成；三维扩展时再补
        ``v_y``。因此这里的四分量 = ``p, vx, vz`` + 记录元数据，用于演示
        井中四分量→共炮点道集的转换链路。
        """
        if self.config.engine in ("acoustic", "viscoacoustic"):
            return ("p",)
        if int(n_components) >= 4:
            return ("p", "vx", "vz", "sxz")
        return ("vz",)

    def report(self, model: ElasticModel) -> dict[str, float]:
        return stability_report(model, self.spec(model))

    # ---------------------------------------------------------------- 正演
    def forward(
        self,
        model: ElasticModel,
        source: SourceSpec,
        receivers: ReceiverSpec,
        *,
        components: tuple[str, ...] | None = None,
        wavelet: np.ndarray | None = None,
    ) -> ShotRecord:
        """正演一炮，返回多分量 :class:`ShotRecord`。"""
        try:
            model.validate()
        except ValueError as exc:
            raise SolverError(f"正演输入模型无效: {exc}") from exc

        spec = self.spec(model)
        names = components or self.components()
        logger.info(
            "正演: engine=%s, nt=%d, dt=%.2e, 分量=%s",
            self.config.engine,
            spec.nt,
            spec.dt,
            list(names),
        )
        record = forward_shot(
            model,
            source,
            receivers,
            spec,
            engine=self.config.engine,
            components=names,
            q_options=self.q_options(),
        )
        record.meta.update(
            {
                "placeholder": False,
                "solver": self.name,
                "engine": self.config.engine,
                "order": self.config.order,
                "boundary": self.config.boundary,
                "npml": self.config.npml,
                "dt": spec.dt,
                "nt": spec.nt,
                "components": list(names),
            }
        )
        return record

    def wavelet(self, source: SourceSpec, model: ElasticModel) -> np.ndarray:
        """生成与求解器时间网格一致的源子波（供成像复用）。"""
        spec = self.spec(model)
        return source_wavelet(source, spec.dt, spec.nt)

    def adjoint(self, record: ShotRecord, model: ElasticModel) -> np.ndarray:
        """离散伴随（保留接口）。

        真正的伴随成像由 :mod:`ertm.imaging` 的反传延拓完成；这里不再提供
        独立的“伴随场”返回，避免出现两条互不一致的实现路径。
        """
        raise SolverError(
            "伴随传播请使用 ertm.imaging.RTMImager.migrate（内部完成反传与成像）"
        )


def build_solver(
    config: SolverConfig,
    q_config: QConfig | None = None,
) -> FiniteDifferenceSolver:
    """求解器工厂（后续可在此按配置选择 GPU/分布式后端）。"""
    return FiniteDifferenceSolver(config, q_config)
