"""EFWI 伴随源、伴随波场与梯度的显式接口。

本模块把 ``Elastic2DOperator`` 内部的“正演 -> 残差 -> 伴随源 -> 反向传播 ->
梯度累积”流程整理成可复用、可逐炮检查的 API：

* :func:`data_residual` / :func:`objective`：残差与归一化目标函数；
* :func:`adjoint_source`：数据空间伴随源 ``P^T r``；
* :class:`ShotGradientAccumulator`：多炮梯度/伪 Hessian 累积；
* :func:`compute_adjoint_gradients`：一次性得到 vp/vs/rho 梯度；
* :func:`directional_derivative` / :func:`check_gradient`：有限差分方向导数检验。

离散约定与正演算子完全一致（见 ``docs/EFWI_梯度推导.md``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .fd.operator import Elastic2DOperator, ShotGradient
from .gradient import PhysicalGradient, convert_gradient
from .model import ElasticModel


# ----------------------------------------------------------------------
# 残差 / 目标函数 / 伴随源
# ----------------------------------------------------------------------
@dataclass
class Residual:
    r: np.ndarray
    energy: float
    n_samples: int

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.r * self.r)))


def data_residual(synthetic: np.ndarray, observed: np.ndarray) -> Residual:
    """``r = d_syn - d_obs``，并统计观测能量 ``||d_obs||^2``。"""
    syn = np.asarray(synthetic, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if syn.shape != obs.shape:
        raise ValueError(f"合成/观测数据形状不一致: {syn.shape} vs {obs.shape}")
    r = syn - obs
    energy = float(np.sum(obs * obs))
    return Residual(r=r, energy=energy if energy > 0 else 1.0, n_samples=r.size)


def objective(residual: Residual, *, normalize: bool = True) -> float:
    """``J = 0.5 * ||r||^2 / ||d_obs||^2``（默认归一化）。"""
    val = 0.5 * float(np.sum(residual.r * residual.r))
    return val / residual.energy if normalize else val


def adjoint_source(residual: Residual | np.ndarray, *, gain: float = 1.0) -> np.ndarray:
    """数据空间伴随源。

    对 ``J=0.5||Pv-d||²`` 有 ``∂J/∂v = P^T r``。这里返回 ``gain * r``；
    真正的 ``P^T``（检波点双线性散加）由正演算子在每个时间步完成。
    若目标函数做了 ``1/||d||²`` 归一化，``gain`` 可取 ``1/||d||²``。
    """
    r = residual.r if isinstance(residual, Residual) else np.asarray(residual, dtype=np.float64)
    return gain * r


# ----------------------------------------------------------------------
# 逐炮梯度累积
# ----------------------------------------------------------------------
@dataclass
class ShotAdjoint:
    shot: int
    misfit_raw: float
    g: dict[str, np.ndarray]       # 物理参数空间（vp/vs/rho/...），未归一化
    h: dict[str, np.ndarray]       # 对应伪 Hessian
    g_native: dict[str, np.ndarray]
    h_native: dict[str, np.ndarray]
    synthetic: np.ndarray
    residual: np.ndarray


@dataclass
class AdjointResult:
    misfit: float
    obs_energy: float
    g: dict[str, np.ndarray]
    h: dict[str, np.ndarray]
    g_native: dict[str, np.ndarray]
    h_native: dict[str, np.ndarray]
    shots: list[ShotAdjoint] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def gradient_norm(self) -> float:
        return float(np.sqrt(sum(np.sum(v * v) for v in self.g.values())))

    @property
    def g_vp(self) -> np.ndarray:
        return self.g["vp"]

    @property
    def g_vs(self) -> np.ndarray:
        return self.g["vs"]

    @property
    def g_rho(self) -> np.ndarray:
        return self.g["rho"]

    def as_physical(self) -> PhysicalGradient:
        return PhysicalGradient(
            g=self.g, h=self.h, g_native=self.g_native, h_native=self.h_native
        )


class ShotGradientAccumulator:
    """多炮原始梯度/伪 Hessian 的累加器。

    ``misfit`` 与梯度在 ``finalize`` 时统一按**全局**观测能量归一化，
    与 :meth:`Elastic2DOperator.compute_gradient` 完全一致。
    """

    def __init__(
        self,
        model: ElasticModel,
        params: Iterable[str] = ("vp", "vs", "rho"),
        *,
        obs_energy: float = 1.0,
        retain_shots: bool = True,
    ) -> None:
        self.model = model
        self.params = tuple(params)
        self.obs_energy = float(obs_energy) if obs_energy > 0 else 1.0
        self.retain_shots = bool(retain_shots)
        shape = model.shape
        self._g = {k: np.zeros(shape, dtype=np.float64) for k in self.params}
        self._h = {k: np.zeros(shape, dtype=np.float64) for k in self.params}
        self._g_native = {
            k: np.zeros(shape, dtype=np.float64)
            for k in ("c11", "c13", "c33", "c44", "rho")
        }
        self._h_native = {
            k: np.zeros(shape, dtype=np.float64)
            for k in ("c11", "c13", "c33", "c44", "rho", "theta")
        }
        self._misfit_raw = 0.0
        self.shots: list[ShotAdjoint] = []
        self.meta: dict[str, Any] = {}

    def add_shot(self, shot: ShotGradient) -> ShotAdjoint:
        pg = convert_gradient(shot, self.model, self.params)
        self._misfit_raw += shot.misfit_raw
        for key in self.params:
            self._g[key] += pg.g[key]
            self._h[key] += pg.h[key]
        for key in self._g_native:
            self._g_native[key] += shot.g[key]
        for key in self._h_native:
            self._h_native[key] += shot.h[key]
        item = ShotAdjoint(
            shot=shot.shot,
            misfit_raw=shot.misfit_raw,
            g={k: v.copy() for k, v in pg.g.items()},
            h={k: v.copy() for k, v in pg.h.items()},
            g_native={k: v.copy() for k, v in shot.g.items()},
            h_native={k: v.copy() for k, v in shot.h.items()},
            synthetic=shot.synthetic,
            residual=shot.residuals,
        )
        if self.retain_shots:
            self.shots.append(item)
        return item

    def finalize(self) -> AdjointResult:
        e = self.obs_energy
        return AdjointResult(
            misfit=self._misfit_raw / e,
            obs_energy=e,
            g={k: v / e for k, v in self._g.items()},
            h={k: v.copy() for k, v in self._h.items()},
            g_native={k: v / e for k, v in self._g_native.items()},
            h_native={k: v.copy() for k, v in self._h_native.items()},
            shots=self.shots,
            meta=dict(self.meta),
        )


def compute_adjoint_gradients(
    operator: Elastic2DOperator,
    model: ElasticModel,
    observed: np.ndarray,
    *,
    params: Iterable[str] = ("vp", "vs", "rho"),
    syn_filter: Any | None = None,
    keep_shots: bool = True,
) -> AdjointResult:
    """逐炮计算并累积 vp/vs/rho 梯度与伪 Hessian。

    ``observed`` 形状 ``(ns, nt, nr, 2)``。若给了 ``syn_filter``，``observed``
    必须已经用同一个滤波器处理过。``keep_shots=False`` 时不保留逐炮数据，
    可显著降低大炮集的内存占用。
    """
    obs = np.asarray(observed, dtype=np.float64)
    expected = (operator.survey.n_shots, operator.survey.nt, operator.survey.n_receivers, 2)
    if obs.shape != expected:
        raise ValueError(f"observed 形状应为 {expected}，实际 {obs.shape}")
    operator.set_model(model)
    energy = float(np.sum(obs * obs))
    acc = ShotGradientAccumulator(
        model, params, obs_energy=energy, retain_shots=keep_shots
    )
    for s in range(operator.survey.n_shots):
        shot = operator.compute_shot_gradient(s, model, obs[s], syn_filter=syn_filter)
        acc.add_shot(shot)
    acc.meta.update(operator.info())
    return acc.finalize()


# ----------------------------------------------------------------------
# 有限差分方向导数检验
# ----------------------------------------------------------------------
def _perturb(
    model: ElasticModel,
    direction: dict[str, np.ndarray],
    eps: float,
) -> tuple[ElasticModel, ElasticModel]:
    plus = model.copy()
    minus = model.copy()
    for key, d in direction.items():
        arr = np.asarray(getattr(model, key), dtype=np.float64)
        step = eps * np.asarray(d, dtype=np.float64)
        setattr(plus, key, arr + step)
        setattr(minus, key, arr - step)
    plus.vs = np.minimum(plus.vs, 0.95 * plus.vp)
    minus.vs = np.minimum(minus.vs, 0.95 * minus.vp)
    return plus, minus


def directional_derivative(
    operator: Elastic2DOperator,
    model: ElasticModel,
    observed: np.ndarray,
    direction: dict[str, np.ndarray],
    *,
    eps: float,
    syn_filter: Any | None = None,
) -> float:
    """中心差分的真实方向导数 ``[J(m+eps d)-J(m-eps d)]/(2 eps)``。"""
    plus, minus = _perturb(model, direction, eps)
    fp = operator.misfit(plus, observed, syn_filter=syn_filter)
    fm = operator.misfit(minus, observed, syn_filter=syn_filter)
    return float((fp - fm) / (2.0 * eps))


def check_gradient(
    operator: Elastic2DOperator,
    model: ElasticModel,
    observed: np.ndarray,
    directions: dict[str, np.ndarray],
    *,
    eps: dict[str, float] | float = 1.0,
    params: Iterable[str] = ("vp", "vs", "rho"),
    syn_filter: Any | None = None,
) -> dict[str, dict[str, float]]:
    """逐参数比较伴随梯度与中心差分方向导数。

    返回 ``{param: {"adjoint", "finite_difference", "abs_error", "rel_error"}}``。
    每次调用只做一次伴随，然后对每个参数做两次正演。
    """
    adj = compute_adjoint_gradients(
        operator, model, observed, params=params, syn_filter=syn_filter
    )
    out: dict[str, dict[str, float]] = {}
    for key, d in directions.items():
        if key not in adj.g:
            continue
        if not np.any(np.asarray(d)):
            continue
        e = eps[key] if isinstance(eps, dict) else float(eps)
        fd = directional_derivative(
            operator, model, observed, {key: d}, eps=e, syn_filter=syn_filter
        )
        a = float(np.sum(adj.g[key] * d))
        out[key] = {
            "adjoint": a,
            "finite_difference": fd,
            "abs_error": abs(a - fd),
            "rel_error": abs(a - fd) / max(abs(fd), 1e-30),
        }
    return out


# ----------------------------------------------------------------------
def shot_gradient_table(result: AdjointResult) -> list[dict[str, float]]:
    """每炮 misfit 与梯度范数，便于诊断炮集贡献。"""
    rows = []
    for s in result.shots:
        rows.append(
            {
                "shot": float(s.shot),
                "misfit_raw": float(s.misfit_raw),
                "vp_norm": float(np.linalg.norm(s.g["vp"])) if "vp" in s.g else 0.0,
                "vs_norm": float(np.linalg.norm(s.g["vs"])) if "vs" in s.g else 0.0,
                "rho_norm": float(np.linalg.norm(s.g["rho"])) if "rho" in s.g else 0.0,
            }
        )
    return rows
