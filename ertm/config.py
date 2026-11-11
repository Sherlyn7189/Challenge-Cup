"""配置模型与加载。

配置采用 JSON（标准库即可解析，避免额外依赖）。每个配置段对应一个
frozen dataclass，加载时一次性完成类型转换和范围校验，后续模块只读。

约定
----
- 所有长度/深度单位为米，时间单位为秒，频率单位为赫兹；
- 缺省配置位于 ``config/default.json``；
- 用户配置可以是完整配置，也可以只写需要覆盖的字段（缺失字段用默认值）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_ORDERS = {2, 4, 6, 8}
_VALID_BOUNDARIES = {"cpml", "free", "none"}
_VALID_MODES = {"pp", "ps", "sp", "ss"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_ENGINES = {"acoustic", "elastic", "viscoacoustic"}
_VALID_CONDITIONS = {"zero_lag", "normalized", "excitation"}
_VALID_SEPARATIONS = {"divcurl", "stress"}
_VALID_GEOMETRIES = {"surface", "borehole"}


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"配置段 {name} 必须是对象，当前为 {type(value).__name__}")
    return value


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是数字，当前为 {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"{name} 必须为正数，当前为 {number}")
    return number


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "ERTM Demo"
    version: str = "0.1.0"

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ConfigError("project.name 不能为空")
        if not str(self.version).strip():
            raise ConfigError("project.version 不能为空")


@dataclass(frozen=True)
class GridConfig:
    nx: int = 201
    nz: int = 151
    dx: float = 5.0
    dz: float = 5.0
    x0: float = 0.0
    z0: float = 0.0

    def validate(self) -> None:
        if self.nx < 2 or self.nz < 2:
            raise ConfigError(f"grid.nx/nz 至少为 2，当前 {self.nx}x{self.nz}")
        if self.dx <= 0 or self.dz <= 0:
            raise ConfigError(f"grid.dx/dz 必须为正，当前 {self.dx}/{self.dz}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def x_max(self) -> float:
        return float(self.x0 + (self.nx - 1) * self.dx)

    @property
    def z_max(self) -> float:
        return float(self.z0 + (self.nz - 1) * self.dz)


@dataclass(frozen=True)
class LayerConfig:
    z_top: float
    vp: float
    vs: float
    rho: float
    qp: float = 80.0
    qs: float = 40.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "LayerConfig":
        data = _as_mapping(data, f"model.layers[{index}]")
        try:
            layer = cls(
                z_top=float(data["z_top"]),
                vp=_positive(data["vp"], f"model.layers[{index}].vp"),
                vs=_positive(data["vs"], f"model.layers[{index}].vs"),
                rho=_positive(data["rho"], f"model.layers[{index}].rho"),
                qp=_positive(data.get("qp", 80.0), f"model.layers[{index}].qp"),
                qs=_positive(data.get("qs", 40.0), f"model.layers[{index}].qs"),
            )
        except KeyError as exc:
            raise ConfigError(f"model.layers[{index}] 缺少字段 {exc.args[0]!r}") from exc
        layer.validate()
        return layer

    def validate(self) -> None:
        if self.z_top < 0:
            raise ConfigError(f"layer.z_top 不能为负，当前 {self.z_top}")
        if self.vs >= self.vp:
            raise ConfigError(f"layer 必须满足 vs < vp，当前 vs={self.vs}, vp={self.vp}")
        if self.rho <= 0 or self.qp <= 0 or self.qs <= 0:
            raise ConfigError("layer 的 rho/qp/qs 必须为正")


@dataclass(frozen=True)
class AnomalyConfig:
    kind: str = "cave"
    x: float = 500.0
    z: float = 350.0
    rx: float = 40.0
    rz: float = 30.0
    dip: float = 0.0
    vp: float | None = None
    vs: float | None = None
    rho: float | None = None
    qp: float | None = None
    qs: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnomalyConfig | None":
        if data is None:
            return None
        data = _as_mapping(data, "model.anomaly")
        anomaly = cls(
            kind=str(data.get("kind", "cave")),
            x=float(data.get("x", 500.0)),
            z=float(data.get("z", 350.0)),
            rx=_positive(data.get("rx", 40.0), "model.anomaly.rx"),
            rz=_positive(data.get("rz", 30.0), "model.anomaly.rz"),
            dip=float(data.get("dip", 0.0)),
            vp=None if data.get("vp") is None else _positive(data["vp"], "model.anomaly.vp"),
            vs=None if data.get("vs") is None else _positive(data["vs"], "model.anomaly.vs"),
            rho=None if data.get("rho") is None else _positive(data["rho"], "model.anomaly.rho"),
            qp=None if data.get("qp") is None else _positive(data["qp"], "model.anomaly.qp"),
            qs=None if data.get("qs") is None else _positive(data["qs"], "model.anomaly.qs"),
        )
        anomaly.validate()
        return anomaly

    def validate(self) -> None:
        if self.kind not in {"cave", "fracture", "fault"}:
            raise ConfigError(
                f"model.anomaly.kind 必须是 cave/fracture/fault，当前 {self.kind!r}"
            )
        if self.rx <= 0 or self.rz <= 0:
            raise ConfigError("model.anomaly.rx/rz 必须为正")
        if not -90.0 <= float(self.dip) <= 90.0:
            raise ConfigError("model.anomaly.dip 必须在 [-90, 90] 度内")
        if self.vp is not None and self.vs is not None and self.vs >= self.vp:
            raise ConfigError("anomaly 必须满足 vs < vp")


@dataclass(frozen=True)
class ModelConfig:
    background_vp: float = 3000.0
    background_vs: float = 1700.0
    background_rho: float = 2400.0
    background_qp: float = 100.0
    background_qs: float = 50.0
    layers: tuple[LayerConfig, ...] = ()
    anomaly: AnomalyConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        data = _as_mapping(data, "model")
        raw_layers = data.get("layers", []) or []
        if not isinstance(raw_layers, list):
            raise ConfigError("model.layers 必须是数组")
        layers = tuple(LayerConfig.from_dict(item, i) for i, item in enumerate(raw_layers))
        model = cls(
            background_vp=_positive(data.get("background_vp", 3000.0), "model.background_vp"),
            background_vs=_positive(data.get("background_vs", 1700.0), "model.background_vs"),
            background_rho=_positive(data.get("background_rho", 2400.0), "model.background_rho"),
            background_qp=_positive(data.get("background_qp", 100.0), "model.background_qp"),
            background_qs=_positive(data.get("background_qs", 50.0), "model.background_qs"),
            layers=layers,
            anomaly=AnomalyConfig.from_dict(data.get("anomaly")),
        )
        model.validate()
        return model

    def validate(self) -> None:
        if self.background_vs >= self.background_vp:
            raise ConfigError("model 必须满足 background_vs < background_vp")
        tops = [layer.z_top for layer in self.layers]
        if tops != sorted(tops):
            raise ConfigError("model.layers 必须按 z_top 从小到大排列")
        if len(set(tops)) != len(tops):
            raise ConfigError("model.layers 存在重复的 z_top")


@dataclass(frozen=True)
class SourceConfig:
    x: float = 500.0
    z: float = 20.0
    f0: float = 20.0
    kind: str = "ricker"
    amplitude: float = 1.0
    delay: float = 0.08
    mechanism: str = "explosive"

    def validate(self) -> None:
        if self.f0 <= 0:
            raise ConfigError("source.f0 必须为正")
        if self.amplitude <= 0:
            raise ConfigError("source.amplitude 必须为正")
        if self.kind not in {"ricker", "gaussian", "custom"}:
            raise ConfigError("source.kind 必须是 ricker/gaussian/custom")
        if self.delay < 0:
            raise ConfigError("source.delay 不能为负")
        if self.mechanism not in {"explosive", "shear", "force_z"}:
            raise ConfigError("source.mechanism 必须是 explosive/shear/force_z")


@dataclass(frozen=True)
class ReceiverConfig:
    count: int = 101
    x_start: float = 0.0
    x_end: float = 1000.0
    z: float = 10.0
    component: str = "vz"

    def validate(self) -> None:
        if self.count < 1:
            raise ConfigError("receivers.count 至少为 1")
        if self.x_end <= self.x_start:
            raise ConfigError("receivers.x_end 必须大于 x_start")
        if self.component not in {"vx", "vz", "vy", "p", "pressure"}:
            raise ConfigError("receivers.component 必须是 vx/vz/vy/p")


@dataclass(frozen=True)
class SolverConfig:
    nt: int = 1200
    dt: float = 5.0e-4
    order: int = 8
    boundary: str = "cpml"
    dtype: str = "float32"
    engine: str = "acoustic"
    npml: int = 20

    def validate(self) -> None:
        if self.nt < 1:
            raise ConfigError("solver.nt 至少为 1")
        if self.dt <= 0:
            raise ConfigError("solver.dt 必须为正")
        if self.order not in _VALID_ORDERS:
            raise ConfigError(f"solver.order 必须是 {sorted(_VALID_ORDERS)} 之一")
        if self.boundary not in _VALID_BOUNDARIES:
            raise ConfigError(f"solver.boundary 必须是 {sorted(_VALID_BOUNDARIES)} 之一")
        if self.dtype not in _VALID_DTYPES:
            raise ConfigError(f"solver.dtype 必须是 {sorted(_VALID_DTYPES)} 之一")
        if self.engine not in _VALID_ENGINES:
            raise ConfigError(f"solver.engine 必须是 {sorted(_VALID_ENGINES)} 之一")
        if self.npml < 3:
            raise ConfigError(f"solver.npml 至少为 3，当前 {self.npml}")


@dataclass(frozen=True)
class ImagingConfig:
    """成像参数。

    ``condition`` 对应技术方案 6.3：``zero_lag``（式 6.7）、``normalized``
    （式 6.10，源照明归一化）、``excitation``（激发振幅/波至时间约束，
    免存完整源波场）。``separation`` 选择 P/S 分离方式：
    ``divcurl``（式 3.7，散度/旋度）或 ``stress``（应力代数组合）。
    """

    mode: str = "pp"
    condition: str = "normalized"
    separation: str = "divcurl"
    normalize: bool = True
    illumination: bool = True
    receiver_normalization: bool = False
    laplacian: bool = True
    lowpass_fraction: float = 0.85
    checkpoint_mb: float = 256.0
    eps: float = 1.0e-10
    stack_normalize: bool = True
    start_periods: float = 1.5

    def validate(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ConfigError(f"imaging.mode 必须是 {sorted(_VALID_MODES)} 之一")
        if self.condition not in _VALID_CONDITIONS:
            raise ConfigError(
                f"imaging.condition 必须是 {sorted(_VALID_CONDITIONS)} 之一"
            )
        if self.separation not in _VALID_SEPARATIONS:
            raise ConfigError(
                f"imaging.separation 必须是 {sorted(_VALID_SEPARATIONS)} 之一"
            )
        if not 0.0 < self.lowpass_fraction <= 1.0:
            raise ConfigError("imaging.lowpass_fraction 必须在 (0, 1] 内")
        if self.checkpoint_mb <= 0:
            raise ConfigError("imaging.checkpoint_mb 必须为正")
        if self.eps <= 0:
            raise ConfigError("imaging.eps 必须为正")
        if self.start_periods < 0:
            raise ConfigError("imaging.start_periods 不能为负")


@dataclass(frozen=True)
class AcquisitionConfig:
    """观测系统与采集参数。

    ``geometry`` 支持 ``surface``（地面排列）与 ``borehole``（井中接收）；
    ``components`` 为 1 或 4，4 表示四分量共接收点道集（COG）；
    ``record_dt`` 模拟实测采样率（36 μs、40 μs），需要先插值到稳定 dt；
    ``rotation_deg`` 为 COG→CSG 的分量旋转角；``shots`` 为空表示单炮。
    """

    geometry: str = "surface"
    components: int = 1
    record_dt: float | None = None
    rotation_deg: float = 0.0
    mute_direct: bool = True
    shots: tuple[float, ...] = ()
    borehole_x: float = 0.0
    borehole_z_start: float = 0.0
    borehole_z_end: float = 1000.0

    def validate(self) -> None:
        if self.geometry not in _VALID_GEOMETRIES:
            raise ConfigError(
                f"acquisition.geometry 必须是 {sorted(_VALID_GEOMETRIES)} 之一"
            )
        if int(self.components) not in (1, 4):
            raise ConfigError("acquisition.components 只能是 1 或 4")
        if self.record_dt is not None and self.record_dt <= 0:
            raise ConfigError("acquisition.record_dt 必须为正")
        if not -180.0 <= self.rotation_deg <= 180.0:
            raise ConfigError("acquisition.rotation_deg 必须在 [-180, 180] 内")
        for value in self.shots:
            if float(value) < 0:
                raise ConfigError("acquisition.shots 的炮点坐标不能为负")


@dataclass(frozen=True)
class QConfig:
    """衰减与 Q 补偿参数（技术方案 6.4、7.4）。

    ``alpha_c`` 为补偿强度（0 不补偿、1 完全补偿），``gain_cap`` 为增益上限，
    ``f_ref`` 为参考频率；``apply_attenuation`` 决定正演记录是否叠加衰减
    （用于合成“含 Q 观测”，便于对照补偿效果）。
    """

    enabled: bool = False
    model: str = "filter"
    alpha_c: float = 0.8
    f_ref: float = 30.0
    gain_cap: float = 30.0
    apply_attenuation: bool = False
    attenuation_strength: float = 1.0
    n_mechanisms: int = 3
    band_ratio: float = 1.5
    amplitude_cap: float = 50.0
    lowpass_interval: int = 20

    def validate(self) -> None:
        if not 0.0 <= self.alpha_c <= 1.0:
            raise ConfigError("q.alpha_c 必须在 [0, 1] 内")
        if self.f_ref <= 0:
            raise ConfigError("q.f_ref 必须为正")
        if self.gain_cap < 1.0:
            raise ConfigError("q.gain_cap 必须 ≥ 1")
        if not 0.0 <= self.attenuation_strength <= 1.0:
            raise ConfigError("q.attenuation_strength 必须在 [0, 1] 内")
        if self.model not in {"filter", "gsls"}:
            raise ConfigError(f"q.model 必须是 filter 或 gsls，当前 {self.model!r}")
        if self.n_mechanisms < 1:
            raise ConfigError("q.n_mechanisms 至少为 1")
        if self.band_ratio <= 1.0:
            raise ConfigError("q.band_ratio 必须 > 1")
        if self.amplitude_cap < 1.0:
            raise ConfigError("q.amplitude_cap 必须 ≥ 1")
        if self.lowpass_interval < 1:
            raise ConfigError("q.lowpass_interval 至少为 1")


@dataclass(frozen=True)
class OutputConfig:
    root: str = "output"
    save_figures: bool = True
    save_data: bool = True

    def validate(self) -> None:
        if not str(self.root).strip():
            raise ConfigError("output.root 不能为空")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file_name: str = "ertm.log"

    def validate(self) -> None:
        if str(self.level).upper() not in _VALID_LEVELS:
            raise ConfigError(f"logging.level 必须是 {sorted(_VALID_LEVELS)} 之一")
        if not str(self.file_name).strip():
            raise ConfigError("logging.file_name 不能为空")


@dataclass(frozen=True)
class AppConfig:
    """应用总配置。"""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    receivers: ReceiverConfig = field(default_factory=ReceiverConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    q: QConfig = field(default_factory=QConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        self.project.validate()
        self.grid.validate()
        self.model.validate()
        self.source.validate()
        self.receivers.validate()
        self.acquisition.validate()
        self.solver.validate()
        self.imaging.validate()
        self.q.validate()
        self.output.validate()
        self.logging.validate()
        self._validate_geometry()
        self._validate_coupling()

    def _validate_coupling(self) -> None:
        """跨配置段的组合约束。"""
        if self.q.enabled and self.q.model == "gsls" and self.solver.engine != "viscoacoustic":
            raise ConfigError(
                "q.model='gsls' 要求 solver.engine='viscoacoustic'（黏声 GSLS 传播器），"
                f"当前 solver.engine={self.solver.engine!r}"
            )

    def _validate_geometry(self) -> None:
        """检查源/检波点是否在网格范围内（允许 2 个网格的边界余量）。"""
        margin = 2.0 * max(self.grid.dx, self.grid.dz)
        if not (self.grid.x0 - margin <= self.source.x <= self.grid.x_max + margin):
            raise ConfigError(
                f"source.x={self.source.x} 超出网格 x 范围 "
                f"[{self.grid.x0}, {self.grid.x_max}]"
            )
        if not (self.grid.z0 - margin <= self.source.z <= self.grid.z_max + margin):
            raise ConfigError(
                f"source.z={self.source.z} 超出网格 z 范围 "
                f"[{self.grid.z0}, {self.grid.z_max}]"
            )
        if self.receivers.x_start < self.grid.x0 - margin or self.receivers.x_end > self.grid.x_max + margin:
            raise ConfigError(
                f"receivers 范围 [{self.receivers.x_start}, {self.receivers.x_end}] "
                f"超出网格 x 范围 [{self.grid.x0}, {self.grid.x_max}]"
            )
        if not (self.grid.z0 - margin <= self.receivers.z <= self.grid.z_max + margin):
            raise ConfigError(
                f"receivers.z={self.receivers.z} 超出网格 z 范围 "
                f"[{self.grid.z0}, {self.grid.z_max}]"
            )
        if self.model.anomaly is not None:
            anomaly = self.model.anomaly
            if not (self.grid.x0 <= anomaly.x <= self.grid.x_max):
                raise ConfigError("model.anomaly.x 超出网格范围")
            if not (self.grid.z0 <= anomaly.z <= self.grid.z_max):
                raise ConfigError("model.anomaly.z 超出网格范围")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        data = _as_mapping(data, "root")
        return cls(
            project=ProjectConfig(**data.get("project", {})),
            grid=GridConfig(**data.get("grid", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            source=SourceConfig(**data.get("source", {})),
            receivers=ReceiverConfig(**data.get("receivers", {})),
            acquisition=AcquisitionConfig(**data.get("acquisition", {})),
            solver=SolverConfig(**data.get("solver", {})),
            imaging=ImagingConfig(**data.get("imaging", {})),
            q=QConfig(**data.get("q", {})),
            output=OutputConfig(**data.get("output", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的普通字典（tuple -> list）。"""
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载 JSON 配置；``None`` 表示使用 ``config/default.json``。"""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件不是合法 JSON: {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件: {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置文件根节点必须是 JSON 对象")
    try:
        return AppConfig.from_dict(raw)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"配置字段不匹配: {exc}") from exc


def save_config(config: AppConfig, path: str | Path) -> Path:
    """保存配置快照，返回实际路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
