"""双程波逆时偏移（RTM）成像。

与首期占位实现的区别
--------------------
本模块实现完整的 RTM 链路：

1. **正传**：源子波在声波（``P``）或弹性波解耦的 ``P/S`` 系统中正向延拓，
   按内存预算保存**检查点**（技术方案 6.2.4），反传时分块重算源波场；
2. **反传**：观测记录按时间倒序注入检波点位置，反向延拓得到接收波场；
3. **成像条件**（技术方案 6.3）：

   - ``zero_lag``：原始零延迟互相关（式 6.7）；
   - ``normalized``：以源波场能量归一化（式 6.10），可选接收照明归一化；
   - ``excitation``：激发振幅/波至时间约束，只保留源波场的激发时刻振幅，
     不必保存完整源波场轨迹（“免存完整震源波场”的工程实现）；

4. **后处理**：Laplacian 压制低波数假象、波数域低通抑制高频噪声。

多炮叠加、按炮归一化由 :mod:`ertm.pipeline` 负责；本模块只处理单炮，
保持“一个函数一件事”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import numerics as num
from .config import ImagingConfig, QConfig, SolverConfig
from .exceptions import ImagingError
from .logging_utils import get_logger
from .propagation import (
    PropagationSpec,
    Propagator2D,
    QOptions,
    build_propagator,
    check_stability,
    source_wavelet,
)
from .types import ElasticModel, GridSpec, ImageResult, ShotRecord

logger = get_logger(__name__)

#: 成像模式 → (源波型, 接收波型)
WAVE_MODES: dict[str, tuple[str, str]] = {
    "pp": ("P", "P"),
    "ps": ("P", "S"),
    "sp": ("S", "P"),
    "ss": ("S", "S"),
}

#: 声波/黏声引擎只支持 PP
_ACOUSTIC_MODES = {"pp"}


# ----------------------------------------------------------------------------
# 成像条件（逐时刻项）
# ----------------------------------------------------------------------------
def zero_lag_condition(
    source_wavefield: np.ndarray,
    receiver_wavefield: np.ndarray,
) -> np.ndarray:
    """零延迟互相关的被积函数 ``W_s·W_r``（式 6.7）。"""
    return np.asarray(source_wavefield) * np.asarray(receiver_wavefield)


def normalized_terms(
    source_wavefield: np.ndarray,
    receiver_wavefield: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """归一化互相关的分解项 ``(W_s·W_r, W_s², W_r²)``（式 6.10）。

    逐时刻只做乘加、沿时间累加后再相除，避免逐点除法放大数值噪声。
    """
    source = np.asarray(source_wavefield)
    receiver = np.asarray(receiver_wavefield)
    return source * receiver, source * source, receiver * receiver


def excitation_terms(
    source_amplitude: np.ndarray,
    receiver_wavefield: np.ndarray,
) -> np.ndarray:
    """激发振幅成像条件的被积函数 ``A_s(x)·W_r(x, t_exc)``。"""
    return np.asarray(source_amplitude) * np.asarray(receiver_wavefield)


# ----------------------------------------------------------------------------
# 源波场检查点
# ----------------------------------------------------------------------------
@dataclass
class CheckpointStore:
    """源波场检查点：保存间隔由内存预算决定。"""

    states: list[tuple[int, dict[str, np.ndarray]]] = field(default_factory=list)
    interval: int = 1
    bytes_per_state: int = 0

    @property
    def memory_mb(self) -> float:
        return len(self.states) * self.bytes_per_state / (1024.0**2)

    def summary(self) -> dict[str, Any]:
        return {
            "interval": int(self.interval),
            "n_states": int(len(self.states)),
            "memory_mb": round(self.memory_mb, 3),
        }


def record_source_wavefield(
    prop: Propagator2D,
    wavelet: np.ndarray,
    *,
    budget_mb: float,
) -> CheckpointStore:
    """正向延拓并保存检查点状态。"""
    nt = int(len(wavelet))
    per_state = max(int(prop.state_bytes()), 1)
    budget = max(float(budget_mb), 1.0) * 1024.0**2
    max_states = max(1, int(budget // per_state))
    interval = max(1, int(np.ceil(nt / max_states)))
    store = CheckpointStore(interval=interval, bytes_per_state=per_state)
    for step in range(nt):
        prop.step(float(wavelet[step]))
        if step % interval == 0:
            store.states.append((step, prop.state()))
    logger.debug(
        "源波场检查点: nt=%d, interval=%d, states=%d, 约 %.2f MB",
        nt,
        interval,
        len(store.states),
        store.memory_mb,
    )
    return store


def replay_block(
    prop: Propagator2D,
    wavelet: np.ndarray,
    store: CheckpointStore,
    block_index: int,
    separation: str,
) -> list[dict[str, np.ndarray]]:
    """从第 ``block_index`` 个检查点重算，返回该块内逐步的成像场。"""
    start, state = store.states[block_index]
    prop.load_state(state)
    block_end = (
        store.states[block_index + 1][0]
        if block_index + 1 < len(store.states)
        else len(wavelet)
    )
    fields: list[dict[str, np.ndarray]] = []
    for step in range(start, block_end):
        prop.step(float(wavelet[step]))
        # 必须复制：``image_fields`` 返回的是波场视图，重放会原地覆盖。
        fields.append(
            {name: np.array(field, copy=True) for name, field in prop.image_fields(separation).items()}
        )
    return fields


# ----------------------------------------------------------------------------
# 偏移器
# ----------------------------------------------------------------------------
class RTMImager:
    """双程波逆时偏移成像器（单炮）。"""

    name = "rtm-imager"

    def __init__(
        self,
        imaging_config: ImagingConfig,
        solver_config: SolverConfig | None = None,
        q_config: QConfig | None = None,
    ) -> None:
        self.config = imaging_config
        self.config.validate()
        self.solver_config = solver_config or SolverConfig()
        self.solver_config.validate()
        self.q_config = q_config or QConfig()
        self.q_config.validate()
        self._validate_engine_mode()

    def q_options(self, *, compensation: bool = False) -> QOptions | None:
        """黏声引擎的 GSLS 选项（反传时 ``compensation=True``）。"""
        if self.solver_config.engine != "viscoacoustic":
            return None
        return QOptions(
            enabled=True,
            f_ref=float(self.q_config.f_ref),
            n_mechanisms=int(self.q_config.n_mechanisms),
            band_ratio=float(self.q_config.band_ratio),
            compensation=bool(compensation and self.q_config.enabled),
            alpha_c=float(self.q_config.alpha_c),
            amplitude_cap=float(self.q_config.amplitude_cap),
            lowpass_interval=int(self.q_config.lowpass_interval),
        )

    # ------------------------------------------------------------- public API
    def migrate(
        self,
        shot: ShotRecord,
        model: ElasticModel,
        *,
        wavelet: np.ndarray | None = None,
        preprocess: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> ImageResult:
        """把单炮记录偏移为深度域图像。

        ``preprocess`` 可选，签名为 ``(data, dt) -> data``，用于反传前对记录
        做 Q 补偿（:mod:`ertm.qcomp`）；默认不改变数据。
        """
        try:
            shot.validate()
            model.validate()
        except ValueError as exc:
            raise ImagingError(f"成像输入无效: {exc}") from exc

        spec = PropagationSpec.from_config(self.solver_config, model)
        try:
            check_stability(model, spec)
        except Exception as exc:  # SolverError → ImagingError
            raise ImagingError(str(exc)) from exc

        engine = self.solver_config.engine
        separation = self.config.separation
        source_wave, receiver_wave = self._resolve_mode()
        condition = str(self.config.condition).lower()

        record = self._prepare_record(shot, spec, preprocess)
        if not record:
            raise ImagingError("炮记录为空，无法反传")
        wavelet = self._prepare_wavelet(shot, spec, wavelet)

        logger.info(
            "RTM 开始: engine=%s, mode=%s, 条件=%s, nt=%d, dt=%.2e, 检波点=%d",
            engine,
            self.config.mode,
            condition,
            spec.nt,
            spec.dt,
            shot.receivers.nrec,
        )

        forward_q = self.q_options(compensation=False)
        backward_q = self.q_options(compensation=True)
        prop = build_propagator(model, spec, engine=engine, q_options=forward_q)
        prop.set_source(shot.source)
        prop.set_receivers(shot.receivers)
        store = record_source_wavefield(
            prop, wavelet, budget_mb=float(self.config.checkpoint_mb)
        )

        back = build_propagator(model, spec, engine=engine, q_options=backward_q)
        back.set_source(shot.source)
        back.set_receivers(shot.receivers)

        numerator = np.zeros(model.shape, dtype=np.float64)
        illumination = np.zeros(model.shape, dtype=np.float64)
        receiver_illumination = np.zeros(model.shape, dtype=np.float64)
        # 近源强能量（直达波残留、自由表面虚反射）会污染浅部成像，按子波周期
        # 做时间切除：只累计 t > delay + start_periods/f0 的时刻（式 6.3.5）。
        start_step = int(
            np.floor(
                (
                    float(shot.source.delay)
                    + float(self.config.start_periods) / max(float(shot.source.f0), 1e-6)
                )
                / spec.dt
            )
        )
        start_step = int(np.clip(start_step, 0, spec.nt))

        if condition == "excitation":
            peak_amplitude, peak_step = self._excitation_maps(
                prop, store, wavelet, separation, model
            )
            for block_index in range(len(store.states) - 1, -1, -1):
                start = store.states[block_index][0]
                block_end = (
                    store.states[block_index + 1][0]
                    if block_index + 1 < len(store.states)
                    else spec.nt
                )
                for step in range(block_end - 1, start - 1, -1):
                    self._inject_record(back, record, step, spec)
                    back.step(0.0)
                    if step < start_step:
                        continue
                    mask = peak_step == step
                    if np.any(mask):
                        receiver = back.image_fields(separation)[receiver_wave]
                        numerator += np.where(
                            mask, peak_amplitude * np.asarray(receiver), 0.0
                        )
                        receiver_illumination += np.where(
                            mask, np.asarray(receiver) ** 2, 0.0
                        )
            illumination = peak_amplitude**2
        else:
            for block_index in range(len(store.states) - 1, -1, -1):
                fields = replay_block(prop, wavelet, store, block_index, separation)
                start = store.states[block_index][0]
                for offset in range(len(fields) - 1, -1, -1):
                    step = start + offset
                    self._inject_record(back, record, step, spec)
                    back.step(0.0)
                    source = np.asarray(fields[offset][source_wave])
                    receiver = np.asarray(
                        back.image_fields(separation)[receiver_wave]
                    )
                    # 照明项按全时段累计（源照明的物理定义），互相关项只累计
                    # 切除之后的时间，避免近源强能量被“小分母”放大。
                    if condition == "normalized" or self.config.receiver_normalization:
                        illumination += source * source
                        receiver_illumination += receiver * receiver
                    if step < start_step:
                        continue
                    numerator += source * receiver

        image = numerator
        eps = float(self.config.eps)
        if condition == "normalized" and self.config.illumination:
            image = image / (illumination + eps)
        if self.config.receiver_normalization:
            image = image / (receiver_illumination + eps)

        image = self._postprocess(np.asarray(image, dtype=np.float64), model, spec)
        result = ImageResult(
            image=np.asarray(image, dtype=np.float32),
            grid=model.grid,
            mode=self.config.mode,
            meta=self._meta(spec, engine, shot, store, condition, start_step),
        )
        result.validate()
        logger.info(
            "RTM 完成: engine=%s, mode=%s, 条件=%s, 范围=[%.3e, %.3e]",
            engine,
            self.config.mode,
            condition,
            float(np.min(result.image)),
            float(np.max(result.image)),
        )
        return result

    def image_condition(
        self,
        source_wavefield: np.ndarray,
        receiver_wavefield: np.ndarray,
    ) -> np.ndarray:
        """逐时刻的成像条件项（接口兼容）。

        时间积分与归一化在 :meth:`migrate` 中完成，本方法只返回被积函数，
        便于 QC 与自定义叠加。
        """
        source = np.asarray(source_wavefield)
        receiver = np.asarray(receiver_wavefield)
        if source.shape != receiver.shape:
            raise ImagingError(
                f"成像条件输入形状不一致: {source.shape} vs {receiver.shape}"
            )
        return zero_lag_condition(source, receiver).astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def _validate_engine_mode(self) -> None:
        if (
            self.solver_config.engine in ("acoustic", "viscoacoustic")
            and self.config.mode not in _ACOUSTIC_MODES
        ):
            raise ImagingError(
                f"声波/黏声引擎只支持 PP 成像，当前 imaging.mode={self.config.mode!r}；"
                "多波型成像请设置 solver.engine='elastic'"
            )

    def _resolve_mode(self) -> tuple[str, str]:
        mode = str(self.config.mode).lower()
        if mode not in WAVE_MODES:
            raise ImagingError(f"不支持的成像模式 {mode!r}")
        return WAVE_MODES[mode]

    def _prepare_wavelet(
        self,
        shot: ShotRecord,
        spec: PropagationSpec,
        wavelet: np.ndarray | None,
    ) -> np.ndarray:
        if wavelet is not None:
            return np.asarray(wavelet, dtype=np.float32)
        return source_wavelet(shot.source, spec.dt, spec.nt)

    def _prepare_record(
        self,
        shot: ShotRecord,
        spec: PropagationSpec,
        preprocess: Callable[[np.ndarray, float], np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """把记录对齐到延拓时间网格，并施加可选的 Q 补偿。"""
        components = dict(shot.components)
        if not components:
            components["primary"] = shot.data
        dt_in = float(shot.dt)
        aligned: dict[str, np.ndarray] = {}
        for name, array in components.items():
            data = np.asarray(array, dtype=np.float64)
            if abs(dt_in - spec.dt) > 1e-12 and data.shape[0] > 1:
                data = num.resample_sinc(data, dt_in, spec.dt)
            if preprocess is not None:
                data = np.asarray(preprocess(data, spec.dt), dtype=np.float64)
            aligned[name] = self._fit_length(data, int(spec.nt))
        return aligned

    @staticmethod
    def _fit_length(data: np.ndarray, nt: int) -> np.ndarray:
        if data.shape[0] >= nt:
            return data[:nt]
        padded = np.zeros((nt, data.shape[1]), dtype=data.dtype)
        padded[: data.shape[0]] = data
        return padded

    def _inject_record(
        self,
        back: Propagator2D,
        record: dict[str, np.ndarray],
        step: int,
        spec: PropagationSpec,
    ) -> None:
        """在反传第 ``step`` 步注入时间倒序的记录。

        反传第一步（``step=nt-1``）注入最晚时刻的样点，之后逐样点倒退，
        与正向延拓的时间轴严格镜像。
        """
        index = int(step)
        for name, data in record.items():
            if name in ("primary", "p") and self.solver_config.engine == "acoustic":
                target = "p"
            elif name == "primary":
                target = "vz"
            else:
                target = name
            values = np.asarray(data[index], dtype=np.float32)
            if not np.any(values):
                continue
            back.inject(target, values)

    def _excitation_maps(
        self,
        prop: Propagator2D,
        store: CheckpointStore,
        wavelet: np.ndarray,
        separation: str,
        model: ElasticModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        """激发振幅成像条件的“峰值时刻 + 峰值振幅”两张图。"""
        shape = model.shape
        peak_step = np.full(shape, -1, dtype=np.int32)
        peak_amplitude = np.zeros(shape, dtype=np.float64)
        running = np.zeros(shape, dtype=np.float64)
        for block_index in range(len(store.states)):
            fields = replay_block(prop, wavelet, store, block_index, separation)
            start = store.states[block_index][0]
            for offset, field in enumerate(fields):
                amplitude = np.abs(np.asarray(field["P"]))
                better = amplitude > running
                if not np.any(better):
                    continue
                running = np.where(better, amplitude, running)
                peak_amplitude = np.where(better, amplitude, peak_amplitude)
                peak_step = np.where(better, start + offset, peak_step)
        return peak_amplitude, peak_step

    def _postprocess(
        self,
        image: np.ndarray,
        model: ElasticModel,
        spec: PropagationSpec,
    ) -> np.ndarray:
        data = np.asarray(image, dtype=np.float64)
        if self.config.laplacian:
            data = num.laplacian(data, model.grid.dx, model.grid.dz)
        if self.config.lowpass_fraction < 1.0:
            data = num.lowpass_wavenumber(
                data, model.grid.dx, model.grid.dz, self.config.lowpass_fraction
            )
        # PML 区域不满足伴随对称性，成像值不可信，直接置零（技术方案 7.4.7）。
        return data * self._pml_mask(model, spec)

    @staticmethod
    def _pml_mask(model: ElasticModel, spec: PropagationSpec) -> np.ndarray:
        nz, nx = model.shape
        n = int(spec.npml)
        mask = np.ones((nz, nx), dtype=np.float64)
        mask[:, :n] = 0.0
        mask[:, -n:] = 0.0
        mask[-n:, :] = 0.0
        if spec.boundary == "cpml":
            mask[:n, :] = 0.0
        return mask

    def _meta(
        self,
        spec: PropagationSpec,
        engine: str,
        shot: ShotRecord,
        store: CheckpointStore,
        condition: str,
        start_step: int = 0,
    ) -> dict[str, Any]:
        return {
            "placeholder": False,
            "method": "reverse-time migration (two-way wave equation)",
            "imager": self.name,
            "engine": engine,
            "condition": condition,
            "separation": self.config.separation,
            "modes": [f"{a}{b}" for a, b in [self._resolve_mode()]],
            "illumination": bool(self.config.illumination),
            "receiver_normalization": bool(self.config.receiver_normalization),
            "laplacian": bool(self.config.laplacian),
            "lowpass_fraction": float(self.config.lowpass_fraction),
            "checkpoints": store.summary(),
            "start_step": int(start_step),
            "start_periods": float(self.config.start_periods),
            "nt": int(spec.nt),
            "dt": float(spec.dt),
            "order": int(spec.order),
            "npml": int(spec.npml),
            "source_x": float(shot.source.x),
            "source_z": float(shot.source.z),
            "n_receivers": int(shot.receivers.nrec),
        }


def build_imager(
    config: ImagingConfig,
    solver_config: SolverConfig | None = None,
    q_config: QConfig | None = None,
) -> RTMImager:
    """成像器工厂。"""
    return RTMImager(config, solver_config, q_config)
