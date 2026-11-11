"""二维一阶速度—应力方程的正演与反传延拓。

本模块实现 RTM 需要的“引擎”，不含成像条件（后者在 :mod:`ertm.imaging`）：

- :class:`AcousticPropagator`：声波一阶速度—压力系统，交错网格八阶、
  CFS-PML 吸收边界，可注入震源、采样单分量/压力记录，并支持反传
  （把记录作为边界源注入，时间倒序）；
- :class:`ElasticPropagator`：弹性波一阶速度—应力系统（Virieux 交错网格），
  变量位置与 ``docs/parts/03`` 表 3.1 一致，八阶空间、二阶时间，CFS-PML；
  支持爆炸源（P）与剪切源（S），并提供两种 P/S 分离方式：

  * ``stress``：``P=(σ_xx+σ_zz)/2``、``S=σ_xz`` —— 局部代数组合，无数值
    微分、不改变相位，是各向同性介质下的精确分解；
  * ``divcurl``：``P=∇·v``、``S=(∇×v)_y`` —— 技术方案式 (3.7) 的矢量成像
    条件形式，便于与文献对照。

时间更新严格遵循技术方案 3.2.3 的顺序：先速度、再应力，源注入在应力层，
反传时数据注入在对应采样量的同一层，保证正演/反传的转置关系。

所有内部数组都带 ``pad = ghost_width(order)`` 的填充，填充区数值恒为零，
因此交错模板在边界处自然退化，不需要额外的分支判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import numerics as num
from .viscoelastic import normalized_modulus
from .config import SolverConfig
from .exceptions import SolverError
from .logging_utils import get_logger
from .types import ElasticModel, ReceiverSpec, ShotRecord, SourceSpec

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 规格与稳定性
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PropagationSpec:
    """一次延拓的离散规格（与具体物理系统无关）。"""

    dt: float
    nt: int
    order: int = 8
    npml: int = 20
    dtype: str = "float32"
    boundary: str = "cpml"

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt 必须为正，当前 {self.dt}")
        if self.nt < 1:
            raise ValueError(f"nt 至少为 1，当前 {self.nt}")
        if self.order not in num.SUPPORTED_ORDERS:
            raise ValueError(f"order 必须是 {num.SUPPORTED_ORDERS}，当前 {self.order}")
        if self.npml < 1:
            raise ValueError(f"npml 必须 ≥ 1，当前 {self.npml}")
        if self.boundary not in {"cpml", "free", "none"}:
            raise ValueError(f"boundary 必须是 cpml/free/none，当前 {self.boundary!r}")

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.nt, dtype=np.float64) * self.dt

    @classmethod
    def from_config(
        cls,
        config: SolverConfig,
        model: ElasticModel | None = None,
        *,
        npml: int | None = None,
    ) -> "PropagationSpec":
        """由配置生成规格；``npml`` 缺省时按模型尺寸自适应。"""
        if model is None:
            cells = 20
        else:
            cells = int(min(20, max(8, min(model.grid.nx, model.grid.nz) // 8)))
        return cls(
            dt=float(config.dt),
            nt=int(config.nt),
            order=int(config.order),
            npml=int(npml if npml is not None else cells),
            dtype=str(config.dtype),
            boundary=str(config.boundary),
        )


def stability_report(model: ElasticModel, spec: PropagationSpec) -> dict[str, float]:
    """返回 CFL 相关诊断量（不做抛错处理，便于报告/日志）。"""
    vmax = float(np.max(model.vp))
    vmin = float(np.min(model.vp))
    limit = num.cfl_dt(vmax, model.grid.dx, model.grid.dz, spec.order)
    return {
        "vmax": vmax,
        "vmin": vmin,
        "dt": spec.dt,
        "dt_max": limit,
        "cfl_ratio": float(spec.dt / limit) if limit > 0 else float("inf"),
        "ppw_vmin": num.points_per_wavelength(vmin, 1.0 / (4.0 * spec.dt), model.grid.dx),
    }


def check_stability(model: ElasticModel, spec: PropagationSpec, *, safety: float = 0.55) -> None:
    """检查 ``dt`` 是否满足 CFL；不满足时给出可执行的建议。

    技术方案 2.5.3/5.5 要求实测采样（36 μs、40 μs）先做时间插值再延拓，
    因此这里不做“自动放大 dt”的隐式处理，而是明确报错并给出建议值。
    """
    vmax = float(np.max(model.vp))
    limit = num.cfl_dt(vmax, model.grid.dx, model.grid.dz, spec.order, safety=safety)
    if spec.dt > limit:
        suggested = num.cfl_dt(vmax, model.grid.dx, model.grid.dz, spec.order, safety=0.5)
        raise SolverError(
            f"时间步 dt={spec.dt:.3e} s 超过 CFL 稳定上限 {limit:.3e} s"
            f"（vmax={vmax:.0f} m/s, dx={model.grid.dx}, dz={model.grid.dz}, order={spec.order}）。"
            f"请先把数据插值到 dt ≤ {suggested:.3e} s 再延拓。"
        )


@dataclass(frozen=True)
class QOptions:
    """黏声传播（GSLS）与反传补偿的选项。"""

    enabled: bool = False
    f_ref: float = 30.0
    n_mechanisms: int = 3
    band_ratio: float = 1.5
    compensation: bool = False
    alpha_c: float = 1.0
    amplitude_cap: float = 50.0
    lowpass_interval: int = 20
    f_max: float | None = None
    gain_budget_db: float = 40.0

    def validate(self) -> None:
        if self.f_ref <= 0:
            raise ValueError("q.f_ref 必须为正")
        if int(self.n_mechanisms) < 1:
            raise ValueError("q.n_mechanisms 至少为 1")
        if self.band_ratio <= 1.0:
            raise ValueError("q.band_ratio 必须 > 1")
        if not 0.0 <= self.alpha_c <= 1.0:
            raise ValueError("q.alpha_c 必须在 [0, 1] 内")
        if self.amplitude_cap < 1.0:
            raise ValueError("q.amplitude_cap 必须 ≥ 1")
        if self.f_max is not None and self.f_max <= 0:
            raise ValueError("q.f_max 必须为正或 None")
        if not 0.0 < self.gain_budget_db <= 200.0:
            raise ValueError("q.gain_budget_db 必须在 (0, 200] 内")


# ----------------------------------------------------------------------------
# 公共基类
# ----------------------------------------------------------------------------
def _pad(array: np.ndarray, pad: int, dtype: np.dtype) -> np.ndarray:
    return np.pad(array.astype(dtype, copy=False), pad, mode="edge")


def _mean_x(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """把节点量平均到 x 方向面心 ``(j+1/2)``。"""
    out = np.empty_like(array)
    out[:, :-1] = 0.5 * (array[:, :-1] + array[:, 1:])
    out[:, -1] = array[:, -1]
    return out.astype(dtype, copy=False)


def _mean_z(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """把节点量平均到 z 方向面心 ``(i+1/2)``。"""
    out = np.empty_like(array)
    out[:-1, :] = 0.5 * (array[:-1, :] + array[1:, :])
    out[-1, :] = array[-1, :]
    return out.astype(dtype, copy=False)


def _mean_xz(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """把节点量平均到单元角点 ``(i+1/2, j+1/2)``。"""
    return _mean_z(_mean_x(array, dtype), dtype)


class Propagator2D:
    """延拓器公共部分：网格映射、源/检位置、采样与场管理。"""

    name = "propagator-2d"
    #: 该传播器可采样的分量名
    components: tuple[str, ...] = ()

    def __init__(self, model: ElasticModel, spec: PropagationSpec) -> None:
        model.validate()
        self.model = model
        self.spec = spec
        self.grid = model.grid
        self.dt = float(spec.dt)
        self.nt = int(spec.nt)
        self.order = int(spec.order)
        self.coeffs = num.staggered_coefficients(self.order)
        self.pad = num.ghost_width(self.order)
        self.dtype = spec.numpy_dtype
        self.shape = (self.grid.nz + 2 * self.pad, self.grid.nx + 2 * self.pad)
        self.dx = float(self.grid.dx)
        self.dz = float(self.grid.dz)
        self.x0 = float(self.grid.x0)
        self.z0 = float(self.grid.z0)
        self.step_index = 0

        vmax = float(np.max(model.vp))
        self.pml = self._make_pml(vmax)
        self._source_index: tuple[int, int] | None = None
        self._receivers: ReceiverSpec | None = None
        self._recv_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _make_pml(self, vmax: float | None = None) -> num.CPML2D:
        """构造 PML：``free`` 边界时顶部不设 PML，留给自由表面。"""
        speed = float(np.max(self.model.vp)) if vmax is None else float(vmax)
        free_surface = self.spec.boundary == "free"
        return num.CPML2D(
            self.shape,
            self.dx,
            self.dz,
            self.spec.npml,
            self.dt,
            speed,
            f0=25.0,
            z_sides=(not free_surface, True),
            enabled=self.spec.boundary != "none",
        )

    def _apply_boundary(self) -> None:
        """自由表面条件（子类实现）；PML 已在导数项中处理。"""
        if self.spec.boundary == "free":
            self._free_surface()

    def _free_surface(self) -> None:
        """默认不做自由表面处理。"""

    # ------------------------------------------------------------- geometry
    def node_index(self, x: float, z: float) -> tuple[int, int]:
        """物理坐标 → 数组索引 ``(iz, ix)``。"""
        ix, iz = self.grid.index_of(x, z)
        return iz + self.pad, ix + self.pad

    def fractional_index(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        x_offset: float = 0.0,
        z_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """物理坐标 → 小数数组索引（``offset`` 以格点为单位，±0.5 表示半格）。"""
        xf = (np.asarray(x, dtype=np.float64) - self.x0) / self.dx + self.pad + x_offset
        zf = (np.asarray(z, dtype=np.float64) - self.z0) / self.dz + self.pad + z_offset
        return zf, xf

    def set_source(self, source: SourceSpec) -> None:
        self._source_index = self.node_index(source.x, source.z)
        self._source = source
        self._mechanism = str(getattr(source, "mechanism", "explosive"))

    def set_receivers(self, receivers: ReceiverSpec) -> None:
        self._receivers = receivers
        self._recv_cache.clear()

    # ------------------------------------------------------------ state I/O
    def _field_names(self) -> tuple[str, ...]:
        """参与检查点保存的场名（子类实现）。"""
        raise NotImplementedError

    def _crop(self, field: np.ndarray) -> np.ndarray:
        """去掉填充区，返回物理域视图 ``(nz, nx)``。"""
        lo = self.pad
        return field[lo : lo + self.grid.nz, lo : lo + self.grid.nx]

    def state(self) -> dict[str, np.ndarray]:
        """返回可缓存的完整状态（场 + PML 记忆变量 + 步数）。"""
        state = {name: getattr(self, name).copy() for name in self._field_names()}
        for key, value in self.pml._memory.items():
            state[f"pml:{key}"] = value.copy()
        state["_step"] = np.array(self.step_index, dtype=np.int64)
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        """从 :meth:`state` 的副本恢复状态（用于检查点重算）。"""
        for name in self._field_names():
            getattr(self, name)[...] = state[name]
        for key, memory in self.pml._memory.items():
            cached = state.get(f"pml:{key}")
            if cached is not None:
                memory[...] = cached
        self.step_index = int(np.asarray(state["_step"]).item())

    def state_bytes(self) -> int:
        """单个检查点状态的字节数（用于内存预算）。"""
        return int(
            sum(getattr(self, name).nbytes for name in self._field_names())
            + self.pml.memory_bytes()
        )

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """清零所有场与 PML 记忆变量。"""
        raise NotImplementedError

    # ------------------------------------------------------------ interface
    def step(self, source_amplitude: float = 0.0) -> None:
        """推进一步，``source_amplitude`` 为当前时刻震源子波值。"""
        raise NotImplementedError

    def sample(self, components: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """在当前时刻采样检波点，返回 ``{分量: (nrec,)}``。"""
        raise NotImplementedError

    def inject(self, component: str, values: np.ndarray) -> None:
        """把数据注入到采样该分量的场位置（反传用）。"""
        raise NotImplementedError

    def image_fields(self, separation: str = "stress") -> dict[str, np.ndarray]:
        """返回成像用的标量场字典（至少含 ``P``）。"""
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    def _add_at_nodes(
        self,
        field: np.ndarray,
        values: np.ndarray,
        component: str,
        scale: np.ndarray | float = 1.0,
    ) -> None:
        """把道数据按双线性权重累加到场的检波点位置（反传注入用）。"""
        if self._receivers is None:
            raise SolverError("注入数据前必须设置检波点")
        x_offset, z_offset = {
            "vx": (-0.5, 0.0),
            "vz": (0.0, -0.5),
            "sxz": (-0.5, -0.5),
        }.get(component, (0.0, 0.0))
        z_index, x_index = self.fractional_index(
            self._receivers.x, self._receivers.z, x_offset=x_offset, z_offset=z_offset
        )
        z0 = np.floor(z_index).astype(np.int64)
        x0 = np.floor(x_index).astype(np.int64)
        tz = z_index - z0
        tx = x_index - x0
        nz, nx = self.shape
        valid = (z0 >= 0) & (z0 < nz - 1) & (x0 >= 0) & (x0 < nx - 1)
        zc = np.clip(z0, 0, nz - 2)
        xc = np.clip(x0, 0, nx - 2)
        values = np.asarray(values, dtype=self.dtype)
        if np.ndim(scale):
            factor = np.asarray(scale, dtype=self.dtype)[zc[valid], xc[valid]]
        else:
            factor = np.float32(scale)
        weight00 = (1 - tz[valid]) * (1 - tx[valid]) * factor
        weight01 = (1 - tz[valid]) * tx[valid] * factor
        weight10 = tz[valid] * (1 - tx[valid]) * factor
        weight11 = tz[valid] * tx[valid] * factor
        np.add.at(field, (zc[valid], xc[valid]), values[valid] * weight00)
        np.add.at(field, (zc[valid], xc[valid] + 1), values[valid] * weight01)
        np.add.at(field, (zc[valid] + 1, xc[valid]), values[valid] * weight10)
        np.add.at(field, (zc[valid] + 1, xc[valid] + 1), values[valid] * weight11)

    def _sample_field(
        self,
        field: np.ndarray,
        component: str,
        receivers: ReceiverSpec,
    ) -> np.ndarray:
        key = f"{component}:{receivers.nrec}"
        cached = self._recv_cache.get(key)
        if cached is None:
            x_offset, z_offset = {
                "vx": (-0.5, 0.0),
                "vz": (0.0, -0.5),
                "sxz": (-0.5, -0.5),
            }.get(component, (0.0, 0.0))
            zf, xf = self.fractional_index(
                receivers.x, receivers.z, x_offset=x_offset, z_offset=z_offset
            )
            self._recv_cache[key] = (zf, xf)
            cached = (zf, xf)
        zf, xf = cached
        return num.bilinear_sample(field, zf, xf).astype(self.dtype, copy=False)


# ----------------------------------------------------------------------------
# 声波：一阶速度—压力
# ----------------------------------------------------------------------------
class AcousticPropagator(Propagator2D):
    """声波一阶速度—压力系统。

    方程（ρ 为密度，κ = ρ·v_p² 为体积模量）::

        ρ ∂v/∂t = ∇p,     (1/κ) ∂p/∂t = ∇·v

    交错位置：``p`` 在单元中心 ``(i, j)``，``v_x`` 在 ``(i, j+1/2)``，
    ``v_z`` 在 ``(i+1/2, j)``。
    """

    name = "acoustic-velocity-pressure"
    components = ("p", "vx", "vz")

    def __init__(self, model: ElasticModel, spec: PropagationSpec) -> None:
        super().__init__(model, spec)
        rho = _pad(model.rho, self.pad, self.dtype)
        vp = _pad(model.vp, self.pad, self.dtype)
        self.kappa = (rho * vp * vp).astype(self.dtype)
        inv_rho = (1.0 / rho).astype(self.dtype)
        self.inv_rho_x = _mean_x(inv_rho, self.dtype)
        self.inv_rho_z = _mean_z(inv_rho, self.dtype)
        self.p = np.zeros(self.shape, dtype=self.dtype)
        self.vx = np.zeros(self.shape, dtype=self.dtype)
        self.vz = np.zeros(self.shape, dtype=self.dtype)

    def reset(self) -> None:
        self.p.fill(0.0)
        self.vx.fill(0.0)
        self.vz.fill(0.0)
        self.pml = self._make_pml()
        self.step_index = 0

    def step(self, source_amplitude: float = 0.0) -> None:
        dt = self.dt
        order, dx, dz = self.order, self.dx, self.dz
        self.vx += dt * self.inv_rho_x * self.pml.derivative(
            "vx_dp_dx", self.p, axis=1, forward=True, order=order, spacing=dx
        )
        self.vz += dt * self.inv_rho_z * self.pml.derivative(
            "vz_dp_dz", self.p, axis=0, forward=True, order=order, spacing=dz
        )
        divergence = self.pml.derivative(
            "p_dvx_dx", self.vx, axis=1, forward=False, order=order, spacing=dx
        )
        divergence += self.pml.derivative(
            "p_dvz_dz", self.vz, axis=0, forward=False, order=order, spacing=dz
        )
        self.p += dt * self.kappa * divergence
        if source_amplitude != 0.0 and self._source_index is not None:
            iz, ix = self._source_index
            self.p[iz, ix] += self.dtype.type(source_amplitude)
        self._apply_boundary()
        self.step_index += 1

    def _free_surface(self) -> None:
        """自由表面（压力释放）：``p = 0``。"""
        self.p[self.pad, :] = 0.0

    def sample(self, components: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        if self._receivers is None:
            return {}
        names = components or ("p",)
        fields = {"p": self.p, "vx": self.vx, "vz": self.vz}
        out: dict[str, np.ndarray] = {}
        for name in names:
            if name == "sxz":  # 声波系统没有剪切分量，返回零便于统一接口
                out[name] = np.zeros(self._receivers.nrec, dtype=self.dtype)
                continue
            field = fields.get(name)
            if field is None:
                raise SolverError(f"声波传播器不支持分量 {name!r}，可选 {self.components}")
            out[name] = self._sample_field(field, name, self._receivers)
        return out

    def inject(self, component: str, values: np.ndarray) -> None:
        if self._receivers is None:
            raise SolverError("注入数据前必须设置检波点")
        field = {"p": self.p, "vx": self.vx, "vz": self.vz}.get(component)
        if field is None:
            raise SolverError(f"声波传播器无法注入分量 {component!r}")
        x_offset, z_offset = {"vx": (-0.5, 0.0), "vz": (0.0, -0.5)}.get(component, (0.0, 0.0))
        z_index, x_index = self.fractional_index(
            self._receivers.x, self._receivers.z, x_offset=x_offset, z_offset=z_offset
        )
        z0 = np.floor(z_index).astype(np.int64)
        x0 = np.floor(x_index).astype(np.int64)
        tz = z_index - z0
        tx = x_index - x0
        nz, nx = self.shape
        valid = (z0 >= 0) & (z0 < nz - 1) & (x0 >= 0) & (x0 < nx - 1)
        zc = np.clip(z0, 0, nz - 2)
        xc = np.clip(x0, 0, nx - 2)
        values = np.asarray(values, dtype=self.dtype)
        np.add.at(field, (zc[valid], xc[valid]), values[valid] * (1 - tz[valid]) * (1 - tx[valid]))
        np.add.at(field, (zc[valid], xc[valid] + 1), values[valid] * (1 - tz[valid]) * tx[valid])
        np.add.at(field, (zc[valid] + 1, xc[valid]), values[valid] * tz[valid] * (1 - tx[valid]))
        np.add.at(field, (zc[valid] + 1, xc[valid] + 1), values[valid] * tz[valid] * tx[valid])

    def image_fields(self, separation: str = "stress") -> dict[str, np.ndarray]:
        return {"P": self._crop(self.p)}

    def _field_names(self) -> tuple[str, ...]:
        return ("p", "vx", "vz")

    @property
    def max_amplitude(self) -> float:
        return float(np.max(np.abs(self.p)))


# ----------------------------------------------------------------------------
# 黏声波：一阶速度—压力 + GSLS 记忆变量
# ----------------------------------------------------------------------------
class ViscoacousticPropagator(Propagator2D):
    """黏声（GSLS）一阶速度—压力系统。

    与 :class:`AcousticPropagator` 的唯一区别是**本构关系**：压力不再由
    ``p += Δt κ ∇·v`` 直接积分，而是按技术方案 2.4.1 的记忆变量形式::

        ε^{n+1} = ε^n + Δt ∇·v
        r_l^{n+1} = a_l r_l^n + b_l (ε^{n+1} - ε^n)
        p^{n+1} = K_R ε^{n+1} + Σ_l r_l^{n+1}

    参数 ``a_l, b_l, K_U`` 由 :func:`ertm.viscoelastic.fit_gsls` 按模型中的
    每个 Q 值拟合得到，因此同一模型里不同层可以有不同 Q。

    反传补偿
    --------
    ``q_options.compensation=True`` 时把 ``Δ_l`` 乘以 ``-α_c``：本构变成
    “增益”，即 Q 的逆过程（技术方案 6.4.1）。指数增长由两道保险控制：
    时变低通（每 ``lowpass_interval`` 步一次）与振幅上限 ``amplitude_cap``。
    """

    name = "viscoacoustic-gsls"
    components = ("p", "vx", "vz")

    def __init__(
        self,
        model: ElasticModel,
        spec: PropagationSpec,
        *,
        q_options: QOptions | None = None,
    ) -> None:
        super().__init__(model, spec)
        self.q_options = q_options or QOptions(enabled=True)
        self.q_options.validate()

        rho = _pad(model.rho, self.pad, self.dtype)
        vp = _pad(model.vp, self.pad, self.dtype)
        qp = _pad(model.qp, self.pad, self.dtype)
        self.kappa_relaxed = (rho * vp * vp).astype(self.dtype)
        inv_rho = (1.0 / rho).astype(self.dtype)
        self.inv_rho_x = _mean_x(inv_rho, self.dtype)
        self.inv_rho_z = _mean_z(inv_rho, self.dtype)
        self.rho = rho

        self._build_relaxation(qp)
        zeros = lambda: np.zeros(self.shape, dtype=self.dtype)  # noqa: E731
        self.p = zeros()
        self.vx = zeros()
        self.vz = zeros()
        self.eps = zeros()
        self.r = np.zeros((self.n_mechanisms,) + self.shape, dtype=self.dtype)
        self._reference_amplitude = 0.0

    # ------------------------------------------------------------------ 参数
    def _build_relaxation(self, qp: np.ndarray) -> None:
        """按模型中出现的 Q 值拟合 GSLS，并装配成逐点系数数组。"""
        from .viscoelastic import fit_gsls

        n = int(self.q_options.n_mechanisms)
        self.n_mechanisms = n
        values = np.clip(
            np.asarray(qp, dtype=np.float64).ravel(), 2.5, 1.0e5
        )
        unique = np.unique(values)
        if unique.size > 32:  # 极端非均质 Q：用中位数拟合一套参数
            unique = np.array([float(np.median(values))])
        fit_error = 0.0
        index = np.zeros(values.shape, dtype=np.int64)
        tau_sigma = np.zeros((unique.size, n))
        delta = np.zeros((unique.size, n))
        modulus_ref = np.zeros(unique.size)
        for i, value in enumerate(unique):
            if value > 1.0e4:
                tau_sigma[i, :] = 1.0e-3
                delta[i, :] = 0.0
                modulus_ref[i] = 1.0
            else:
                params = fit_gsls(
                    float(value),
                    self.q_options.f_ref,
                    n_mechanisms=n,
                    band_ratio=self.q_options.band_ratio,
                )
                fit_error = max(fit_error, params.fit_error)
                tau = np.asarray(params.tau_sigma, dtype=np.float64)
                dlt = np.asarray(params.delta, dtype=np.float64)
                tau_sigma[i, : tau.size] = tau
                if tau.size < n:
                    tau_sigma[i, tau.size :] = tau[-1]
                delta[i, : dlt.size] = dlt
                modulus_ref[i] = float(
                    np.real(
                        normalized_modulus(tau, dlt, [self.q_options.f_ref])[0]
                    )
                )
        index = np.searchsorted(unique, values).reshape(self.shape)
        tau_node = tau_sigma[index]  # (nz, nx, n)

        if self.q_options.compensation:
            # ---- 带限 + 增益预算正则化（逐机制、逐 Q 值）------------------
            # 1) 频带上限：峰值频率高于 f_max 的机制不参与补偿（保留其正向衰减）
            # 2) 增益预算：机制 l 在记录末端 T 的累积增益不得超过 G_max，
            #    因此把强度缩为 alpha_l = min(alpha_c, ln(G_max)·Q/(π·f_l·T))
            time_total = max(self.nt * self.dt, self.dt)
            budget = 10.0 ** (float(self.q_options.gain_budget_db) / 20.0)
            f_max = self.q_options.f_max
            alpha_table = np.zeros((unique.size, n))
            f_peak_table = np.zeros((unique.size, n))
            for i, value in enumerate(unique):
                tau = tau_sigma[i]
                f_peak = np.sqrt(np.maximum(1.0 - delta[i], 1e-6)) / (2.0 * np.pi * tau)
                f_peak_table[i] = f_peak
                keep = np.ones(n, dtype=bool)
                if f_max is not None:
                    keep = f_peak <= float(f_max)
                needed = np.log(budget) * max(float(value), 1.0) / (
                    np.pi * np.maximum(f_peak, 1e-6) * time_total
                )
                alpha = np.minimum(float(self.q_options.alpha_c), needed)
                alpha = np.where(keep, alpha, 0.0)
                alpha_table[i] = alpha
                # 只翻转松弛强度，K_R 仍按“物理介质”标定：
                # 补偿算子 = 同一 GSLS 模型、同一 K_R、同一 τ，仅 Δ→−αΔ
                # （满足“补偿模型与正演物理模型一致”的约束）
                delta[i] = -alpha * delta[i]
            self.alpha_mechanisms = tuple(
                float(v) for v in np.mean(alpha_table, axis=0)
            )
            self.f_peak_mechanisms = tuple(
                float(v) for v in np.mean(f_peak_table, axis=0)
            )
        else:
            self.alpha_mechanisms = tuple(float(self.q_options.alpha_c) for _ in range(n))
            self.f_peak_mechanisms = tuple(
                float(v)
                for v in np.mean(
                    np.sqrt(np.maximum(1.0 - delta, 1e-6))
                    / (2.0 * np.pi * tau_sigma),
                    axis=0,
                )
            )

        delta_node = delta[index]
        k_relaxed = (self.kappa_relaxed / modulus_ref[index]).astype(self.dtype)

        self.tau_sigma = np.moveaxis(tau_node, -1, 0)
        self.delta = np.moveaxis(delta_node, -1, 0)
        self.k_relaxed = k_relaxed
        self.k_unrelaxed = (
            k_relaxed * (1.0 + np.sum(np.moveaxis(delta_node, -1, 0), axis=0))
        ).astype(self.dtype)
        self.a = np.exp(-self.dt / self.tau_sigma).astype(self.dtype)
        self.b = (
            k_relaxed[None, :, :]
            * self.delta
            * (1.0 - self.a)
            * self.tau_sigma
            / self.dt
        ).astype(self.dtype)
        self.q_fit_error = float(fit_error)
        logger.debug(
            "黏声参数装配: L=%d, unique Q=%d, k_u/k_r=%.3f, 拟合误差=%.2e",
            n,
            unique.size,
            float(np.mean(self.k_unrelaxed / np.maximum(k_relaxed, 1e-30))),
            fit_error,
        )

    def reset(self) -> None:
        for name in ("p", "vx", "vz", "eps"):
            getattr(self, name).fill(0.0)
        self.r.fill(0.0)
        self._reference_amplitude = 0.0
        self.pml = self._make_pml()
        self.step_index = 0

    # ------------------------------------------------------------------ 步进
    def step(self, source_amplitude: float = 0.0) -> None:
        dt = self.dt
        order, dx, dz = self.order, self.dx, self.dz
        pml = self.pml

        self.vx += dt * self.inv_rho_x * pml.derivative(
            "vx_dp_dx", self.p, axis=1, forward=True, order=order, spacing=dx
        )
        self.vz += dt * self.inv_rho_z * pml.derivative(
            "vz_dp_dz", self.p, axis=0, forward=True, order=order, spacing=dz
        )
        divergence = pml.derivative(
            "p_dvx_dx", self.vx, axis=1, forward=False, order=order, spacing=dx
        )
        divergence += pml.derivative(
            "p_dvz_dz", self.vz, axis=0, forward=False, order=order, spacing=dz
        )
        increment = (dt * divergence).astype(self.dtype, copy=False)
        self.eps += increment

        # 记忆变量与应力（顺序不可交换，技术方案 3.2.3）
        self.r *= self.a
        self.r += self.b * increment[None, :, :]
        total = self.r.sum(axis=0)
        np.add(self.k_relaxed * self.eps, total, out=self.p)

        if source_amplitude != 0.0 and self._source_index is not None:
            iz, ix = self._source_index
            # 压力源等价于注入体应变，保证与声波求解器的软源一致
            self.eps[iz, ix] += self.dtype.type(
                source_amplitude / float(self.k_relaxed[iz, ix])
            )
            self.p[iz, ix] += self.dtype.type(source_amplitude)

        self._apply_boundary()
        if self.q_options.compensation:
            self._stabilize()
        self.step_index += 1

    # --------------------------------------------------------------- 稳定化
    def _stabilize(self) -> None:
        peak = float(np.max(np.abs(self.p)))
        if self.step_index < 30:
            self._reference_amplitude = max(self._reference_amplitude, peak)
            return
        reference = max(self._reference_amplitude, 1e-30)
        limit = self.q_options.amplitude_cap * reference
        if peak > limit:
            scale = self.dtype.type(limit / peak)
            self.p *= scale
            self.eps *= scale
            self.vx *= scale
            self.vz *= scale
            self.r *= scale
            self._clipped = getattr(self, "_clipped", 0) + 1
        interval = int(self.q_options.lowpass_interval)
        if interval > 1 and self.step_index % interval == 0:
            self._lowpass_steps = getattr(self, "_lowpass_steps", 0) + 1
            time = self.step_index * self.dt
            ratio = max(0.9 - 0.4 * min(time / 1.0, 1.0), 0.35)
            self.p[:] = num.lowpass_wavenumber(self.p, dx=self.dx, dz=self.dz,
                                               cutoff_fraction=ratio)
            self.eps[:] = num.lowpass_wavenumber(self.eps, dx=self.dx, dz=self.dz,
                                                 cutoff_fraction=ratio)

    def sample(self, components: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        if self._receivers is None:
            return {}
        names = components or ("p",)
        fields = {"p": self.p, "vx": self.vx, "vz": self.vz}
        out: dict[str, np.ndarray] = {}
        for name in names:
            field = fields.get(name)
            if field is None:
                raise SolverError(f"黏声传播器不支持分量 {name!r}，可选 {self.components}")
            out[name] = self._sample_field(field, name, self._receivers)
        return out

    def inject(self, component: str, values: np.ndarray) -> None:
        if self._receivers is None:
            raise SolverError("注入数据前必须设置检波点")
        field = {"p": self.p, "vx": self.vx, "vz": self.vz}.get(component)
        if field is None:
            raise SolverError(f"黏声传播器无法注入分量 {component!r}")
        self._add_at_nodes(field, values, component)
        if component == "p":
            # 压力注入必须同时写入体应变，否则下一步 p 由 (ε, r) 重算时会被抹掉
            scale = 1.0 / np.maximum(np.abs(self.k_relaxed), 1e-20)
            self._add_at_nodes(self.eps, values, component, scale=scale)

    def image_fields(self, separation: str = "stress") -> dict[str, np.ndarray]:
        return {"P": self._crop(self.p)}

    def _field_names(self) -> tuple[str, ...]:
        return ("p", "vx", "vz", "eps", "r")

    @property
    def quality_summary(self) -> dict[str, Any]:
        return {
            "mechanisms": int(self.n_mechanisms),
            "q_fit_error": float(self.q_fit_error),
            "compensation": bool(self.q_options.compensation),
            "alpha_c": float(self.q_options.alpha_c),
            "alpha_mechanisms": list(getattr(self, "alpha_mechanisms", ())),
            "f_peak_mechanisms_hz": [
                round(v, 1) for v in getattr(self, "f_peak_mechanisms", ())
            ],
            "f_max": None if self.q_options.f_max is None else float(self.q_options.f_max),
            "gain_budget_db": float(self.q_options.gain_budget_db),
            "clipped": int(getattr(self, "_clipped", 0)),
            "lowpass_steps": int(getattr(self, "_lowpass_steps", 0)),
        }


# ----------------------------------------------------------------------------
# 弹性波：一阶速度—应力（P-SV）
# ----------------------------------------------------------------------------
class ElasticPropagator(Propagator2D):
    """弹性波一阶速度—应力系统（二维 P-SV）。

    变量位置遵循技术方案表 3.1：``σ_xx, σ_zz`` 在单元中心，``v_x`` 在
    ``(i, j+1/2)``，``v_z`` 在 ``(i+1/2, j)``，``σ_xz`` 在角点。
    """

    name = "elastic-velocity-stress"
    components = ("vx", "vz", "sxx", "szz", "sxz", "p")

    def __init__(self, model: ElasticModel, spec: PropagationSpec) -> None:
        super().__init__(model, spec)
        rho = _pad(model.rho, self.pad, self.dtype)
        vp = _pad(model.vp, self.pad, self.dtype)
        vs = _pad(model.vs, self.pad, self.dtype)
        mu = (rho * vs * vs).astype(self.dtype)
        lam = (rho * (vp * vp - 2.0 * vs * vs)).astype(self.dtype)
        self.lam = lam
        self.mu = mu
        self.lam_2mu = (lam + 2.0 * mu).astype(self.dtype)
        self.mu_xz = _mean_xz(mu, self.dtype)
        inv_rho = (1.0 / rho).astype(self.dtype)
        self.inv_rho_x = _mean_x(inv_rho, self.dtype)
        self.inv_rho_z = _mean_z(inv_rho, self.dtype)
        zeros = lambda: np.zeros(self.shape, dtype=self.dtype)  # noqa: E731
        self.vx = zeros()
        self.vz = zeros()
        self.sxx = zeros()
        self.szz = zeros()
        self.sxz = zeros()

    def reset(self) -> None:
        for name in ("vx", "vz", "sxx", "szz", "sxz"):
            getattr(self, name).fill(0.0)
        self.pml = self._make_pml()
        self.step_index = 0

    def step(self, source_amplitude: float = 0.0) -> None:
        dt = self.dt
        order, dx, dz = self.order, self.dx, self.dz
        pml = self.pml

        # 1) 速度更新：v^{n+1/2} = v^{n-1/2} + dt/ρ (∇·σ)^n
        self.vx += dt * self.inv_rho_x * (
            pml.derivative("vx_dsxx", self.sxx, axis=1, forward=True, order=order, spacing=dx)
            + pml.derivative("vx_dsxz", self.sxz, axis=0, forward=False, order=order, spacing=dz)
        )
        self.vz += dt * self.inv_rho_z * (
            pml.derivative("vz_dsxz", self.sxz, axis=1, forward=False, order=order, spacing=dx)
            + pml.derivative("vz_dszz", self.szz, axis=0, forward=True, order=order, spacing=dz)
        )

        # 2) 应变率与应力更新：σ^{n+1} = σ^n + dt·C:ε(v^{n+1/2})
        dvx_dx = pml.derivative("sxx_dvx", self.vx, axis=1, forward=False, order=order, spacing=dx)
        dvz_dz = pml.derivative("szz_dvz", self.vz, axis=0, forward=False, order=order, spacing=dz)
        self.sxx += dt * (self.lam_2mu * dvx_dx + self.lam * dvz_dz)
        self.szz += dt * (self.lam * dvx_dx + self.lam_2mu * dvz_dz)
        self.sxz += dt * self.mu_xz * (
            pml.derivative("sxz_dvx", self.vx, axis=0, forward=True, order=order, spacing=dz)
            + pml.derivative("sxz_dvz", self.vz, axis=1, forward=True, order=order, spacing=dx)
        )

        # 3) 源注入：按机制激发（爆炸源为纯 P 辐射，剪切源主要产生 S）
        if source_amplitude != 0.0 and self._source_index is not None:
            iz, ix = self._source_index
            value = self.dtype.type(source_amplitude)
            mechanism = getattr(self, "_mechanism", "explosive")
            if mechanism == "shear":
                self.sxz[iz, ix] += value
            elif mechanism == "force_z":
                self.vz[iz, ix] += value
            else:
                self.sxx[iz, ix] += value
                self.szz[iz, ix] += value
        self._apply_boundary()
        self.step_index += 1

    def _free_surface(self) -> None:
        """自由表面：法向与切向应力为零。"""
        self.szz[self.pad, :] = 0.0
        self.sxz[self.pad, :] = 0.0

    def sample(self, components: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        if self._receivers is None:
            return {}
        names = components or ("vx", "vz")
        out: dict[str, np.ndarray] = {}
        for name in names:
            if name == "p":
                out[name] = self._sample_field(self.pressure_field(), "sxx", self._receivers)
                continue
            field = getattr(self, name, None)
            if field is None:
                raise SolverError(f"弹性传播器不支持分量 {name!r}，可选 {self.components}")
            out[name] = self._sample_field(field, name, self._receivers)
        return out

    def inject(self, component: str, values: np.ndarray) -> None:
        """把记录注入到对应分量（速度分量注入速度场，应力分量注入应力场）。

        ``p`` 是应力合成量，反传时等量注入两个正应力（与爆炸源的可逆关系一致）。
        """
        if self._receivers is None:
            raise SolverError("注入数据前必须设置检波点")
        if component == "p":
            self._spread_into(self.sxx, values, "sxx")
            self._spread_into(self.szz, values, "sxx")
            return
        field = getattr(self, component, None)
        if field is None:
            raise SolverError(f"弹性传播器无法注入分量 {component!r}")
        self._spread_into(field, values, component)

    def _spread_into(
        self,
        field: np.ndarray,
        values: np.ndarray,
        component: str,
    ) -> None:
        x_offset, z_offset = {
            "vx": (-0.5, 0.0),
            "vz": (0.0, -0.5),
            "sxz": (-0.5, -0.5),
        }.get(component, (0.0, 0.0))
        z_index, x_index = self.fractional_index(
            self._receivers.x, self._receivers.z, x_offset=x_offset, z_offset=z_offset
        )
        z0 = np.floor(z_index).astype(np.int64)
        x0 = np.floor(x_index).astype(np.int64)
        tz = z_index - z0
        tx = x_index - x0
        nz, nx = self.shape
        valid = (z0 >= 0) & (z0 < nz - 1) & (x0 >= 0) & (x0 < nx - 1)
        zc = np.clip(z0, 0, nz - 2)
        xc = np.clip(x0, 0, nx - 2)
        values = np.asarray(values, dtype=self.dtype)
        np.add.at(field, (zc[valid], xc[valid]), values[valid] * (1 - tz[valid]) * (1 - tx[valid]))
        np.add.at(field, (zc[valid], xc[valid] + 1), values[valid] * (1 - tz[valid]) * tx[valid])
        np.add.at(field, (zc[valid] + 1, xc[valid]), values[valid] * tz[valid] * (1 - tx[valid]))
        np.add.at(field, (zc[valid] + 1, xc[valid] + 1), values[valid] * tz[valid] * tx[valid])

    # ------------------------------------------------------------- P/S 分离
    def pressure_field(self) -> np.ndarray:
        """``(σ_xx + σ_zz)/2``：各向同性介质的 P 应力（无相位畸变）。"""
        return 0.5 * (self.sxx + self.szz)

    def shear_field(self) -> np.ndarray:
        """``σ_xz``：S 应力分量。"""
        return self.sxz

    def divergence_field(self) -> np.ndarray:
        """``∇·v``，用普通交错差分（不更新 PML 记忆变量）。"""
        return num.dminus(self.vx, 1, self.coeffs, self.dx) + num.dminus(
            self.vz, 0, self.coeffs, self.dz
        )

    def curl_field(self) -> np.ndarray:
        """``(∇×v)_y = ∂v_x/∂z - ∂v_z/∂x``。"""
        return num.dminus(self.vx, 0, self.coeffs, self.dz) - num.dminus(
            self.vz, 1, self.coeffs, self.dx
        )

    def image_fields(self, separation: str = "stress") -> dict[str, np.ndarray]:
        mode = str(separation).lower()
        if mode == "stress":
            return {
                "P": self._crop(self.pressure_field()),
                "S": self._crop(self.shear_field()),
            }
        if mode == "divcurl":
            return {
                "P": self._crop(self.divergence_field()),
                "S": self._crop(self.curl_field()),
            }
        raise SolverError(f"未知的 P/S 分离方式 {separation!r}，可选 stress/divcurl")

    def _field_names(self) -> tuple[str, ...]:
        return ("vx", "vz", "sxx", "szz", "sxz")


# ----------------------------------------------------------------------------
# 工厂与正演驱动
# ----------------------------------------------------------------------------
def build_propagator(
    model: ElasticModel,
    spec: PropagationSpec,
    *,
    engine: str = "acoustic",
    q_options: QOptions | None = None,
) -> Propagator2D:
    """按 ``engine`` 构造传播器（``acoustic`` / ``viscoacoustic`` / ``elastic``）。"""
    engine = str(engine).lower()
    if engine == "acoustic":
        return AcousticPropagator(model, spec)
    if engine == "viscoacoustic":
        return ViscoacousticPropagator(model, spec, q_options=q_options)
    if engine == "elastic":
        return ElasticPropagator(model, spec)
    raise SolverError(f"未知的延拓引擎 {engine!r}，可选 acoustic/viscoacoustic/elastic")


def source_wavelet(source: SourceSpec, dt: float, nt: int) -> np.ndarray:
    """按源规格生成时间序列（Ricker / Gaussian）。"""
    t = np.arange(nt, dtype=np.float64) * dt
    if source.kind == "ricker":
        wave = num.ricker(t, source.f0, source.delay)
    elif source.kind == "gaussian":
        wave = num.gaussian_pulse(t, source.f0, source.delay)
    elif source.kind == "custom":
        raise SolverError("custom 源需要外部时间序列，请直接传入数组")
    else:
        raise SolverError(f"不支持的源类型: {source.kind}")
    return (float(source.amplitude) * wave).astype(np.float32)


def run_forward(
    model: ElasticModel,
    source: SourceSpec,
    receivers: ReceiverSpec,
    spec: PropagationSpec,
    *,
    engine: str = "acoustic",
    components: tuple[str, ...] | None = None,
    wavelet: np.ndarray | None = None,
    propagator: Propagator2D | None = None,
    record_every: int = 1,
    q_options: QOptions | None = None,
) -> dict[str, np.ndarray]:
    """执行一次正演，返回 ``{分量: (nt, nrec)}``。

    ``wavelet`` 给定时按其长度决定步数（用于与实测记录对齐）；否则用
    :func:`source_wavelet` 生成 ``spec.nt`` 个采样。
    """
    prop = propagator or build_propagator(
        model, spec, engine=engine, q_options=q_options
    )
    prop.set_source(source)
    prop.set_receivers(receivers)
    if wavelet is None:
        wavelet = source_wavelet(source, spec.dt, spec.nt)
    nt = int(len(wavelet))
    names = components or prop.components[:1]
    data = {name: np.zeros((nt, receivers.nrec), dtype=np.float32) for name in names}
    for step in range(nt):
        prop.step(float(wavelet[step]))
        if record_every <= 1 or step % record_every == 0:
            sampled = prop.sample(names)
            for name in names:
                data[name][step] = sampled[name]
    return data


def forward_shot(
    model: ElasticModel,
    source: SourceSpec,
    receivers: ReceiverSpec,
    spec: PropagationSpec,
    *,
    engine: str = "acoustic",
    components: tuple[str, ...] | None = None,
    q_options: QOptions | None = None,
) -> ShotRecord:
    """正演并封装为 :class:`~ertm.types.ShotRecord`（主分量为第一个分量）。"""
    names = components or (("p",) if engine in ("acoustic", "viscoacoustic") else ("vx", "vz", "sxz"))
    data = run_forward(
        model, source, receivers, spec, engine=engine, components=names, q_options=q_options
    )
    primary = names[0]
    t = np.arange(data[primary].shape[0], dtype=np.float64) * spec.dt
    meta: dict[str, Any] = {
        "engine": engine,
        "solver": f"{engine}-velocity-stress",
        "dt": spec.dt,
        "nt": int(t.size),
        "order": spec.order,
        "npml": spec.npml,
        "components": list(names),
        "placeholder": False,
        "method": "finite-difference two-way wave equation",
    }
    shot = ShotRecord(
        data=data[primary].astype(np.float32),
        dt=float(spec.dt),
        t=t,
        receivers=receivers,
        source=source,
        meta=meta,
        components={name: data[name].astype(np.float32) for name in names},
    )
    shot.validate()
    return shot
