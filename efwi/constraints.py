"""地质与岩石物理约束：凸集、Dykstra 交替投影、ADMM 投影。

文档要求把以下认识表达为若干凸集，并投影到交集：

* 参数上下界；
* 侧向/垂向梯度上限；
* Gardner 密度—速度经验关系距离。

实现要点：

* 上下界、Gardner 管道（log-log 空间中是线性带）都有解析投影；
* 梯度上限集合 ``{m: |Dx m|<=gx, |Dz m|<=gz}`` 的投影用 ADMM +
  DCT-II（Neumann 边界）求解，内部迭代若干次即可；
* 外层用 Dykstra 交替投影（保持交集投影的收敛性）或 consensus ADMM。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .config import ConstraintConfig
from .model import ElasticModel


class ConstraintSet(Protocol):
    keys: tuple[str, ...]

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ...


# ----------------------------------------------------------------------
# 基本凸集
# ----------------------------------------------------------------------
@dataclass
class BoxSet:
    keys: tuple[str, ...]
    lower: dict[str, float]
    upper: dict[str, float]

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for k in self.keys:
            out[k] = np.clip(values[k], self.lower[k], self.upper[k])
        return out


def _d_dx(a: np.ndarray, dx: float) -> np.ndarray:
    """Neumann 边界下的前向差分（最后一行无梯度）。"""
    out = np.zeros_like(a)
    out[:-1, :] = (a[1:, :] - a[:-1, :]) / dx
    return out


def _d_dz(a: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(a)
    out[:, :-1] = (a[:, 1:] - a[:, :-1]) / dz
    return out


def _d_dxT(a: np.ndarray, dx: float) -> np.ndarray:
    """前向差分的转置。"""
    out = np.zeros_like(a)
    out[0, :] = -a[0, :] / dx
    out[1:, :] = (a[:-1, :] - a[1:, :]) / dx
    return out


def _d_dzT(a: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(a)
    out[:, 0] = -a[:, 0] / dz
    out[:, 1:] = (a[:, :-1] - a[:, 1:]) / dz
    return out


def project_gradient_bound(
    m: np.ndarray,
    gx: float,
    gz: float,
    dx: float,
    dz: float,
    *,
    n_iter: int = 20,
    rho: float = 10.0,
) -> np.ndarray:
    """投影到 ``|dm/dx|<=gx, |dm/dz|<=gz``。

    使用 Neumann 边界（DCT-II 对角化前向差分的法方程），避免周期 FFT
    在模型边缘把左右/上下边界错误地“缝”在一起。
    """
    from scipy.fft import dctn, idctn

    m = np.asarray(m, dtype=np.float64)
    if gx <= 0 and gz <= 0:
        return m.copy()
    nx, nz = m.shape
    kx = np.arange(nx)
    kz = np.arange(nz)
    lam_x = (2.0 * np.sin(np.pi * kx / (2.0 * nx)) / dx) ** 2
    lam_z = (2.0 * np.sin(np.pi * kz / (2.0 * nz)) / dz) ** 2
    denom = 1.0 + rho * (lam_x[:, None] + lam_z[None, :])

    u = m.copy()
    dux = _d_dx(u, dx) if gx > 0 else np.zeros_like(u)
    duz = _d_dz(u, dz) if gz > 0 else np.zeros_like(u)
    px = np.clip(dux, -gx, gx) if gx > 0 else dux
    pz = np.clip(duz, -gz, gz) if gz > 0 else duz
    qx = np.zeros_like(u)
    qz = np.zeros_like(u)

    for _ in range(max(int(n_iter), 1)):
        rhs = m.copy()
        if gx > 0:
            rhs += rho * _d_dxT(px - qx, dx)
        if gz > 0:
            rhs += rho * _d_dzT(pz - qz, dz)
        u = idctn(dctn(rhs, type=2, norm="ortho") / denom, type=2, norm="ortho")
        if gx > 0:
            dux = _d_dx(u, dx)
            px = np.clip(dux + qx, -gx, gx)
            qx += dux - px
        if gz > 0:
            duz = _d_dz(u, dz)
            pz = np.clip(duz + qz, -gz, gz)
            qz += duz - pz
    return u


@dataclass
class GradientBoundSet:
    key: str
    gx: float
    gz: float
    dx: float
    dz: float
    n_iter: int = 12
    rho: float = 1.0

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key,)

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            self.key: project_gradient_bound(
                values[self.key], self.gx, self.gz, self.dx, self.dz,
                n_iter=self.n_iter, rho=self.rho,
            )
        }


@dataclass
class GardnerTube:
    a: float = 310.0
    b: float = 0.25
    tol: float = 0.10
    vp_key: str = "vp"
    rho_key: str = "rho"

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.vp_key, self.rho_key)

    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        vp = np.asarray(values[self.vp_key], dtype=np.float64)
        rho = np.asarray(values[self.rho_key], dtype=np.float64)
        lvp = np.log(np.maximum(vp, 1e-6))
        lrho = np.log(np.maximum(rho, 1e-6))
        g = lrho - (np.log(self.a) + self.b * lvp)
        delta = g - np.clip(g, -self.tol, self.tol)
        denom = 1.0 + self.b * self.b
        lvp_new = lvp + self.b * delta / denom
        lrho_new = lrho - delta / denom
        return {
            self.vp_key: np.exp(lvp_new),
            self.rho_key: np.exp(lrho_new),
        }

    def residual(self, values: dict[str, np.ndarray]) -> np.ndarray:
        vp = np.asarray(values[self.vp_key], dtype=np.float64)
        rho = np.asarray(values[self.rho_key], dtype=np.float64)
        return np.log(np.maximum(rho, 1e-6)) - (np.log(self.a) + self.b * np.log(np.maximum(vp, 1e-6)))


# ----------------------------------------------------------------------
# 组合投影
# ----------------------------------------------------------------------
def dykstra_projection(
    init: dict[str, np.ndarray],
    sets: Iterable[ConstraintSet],
    n_iter: int = 8,
) -> dict[str, np.ndarray]:
    sets = list(sets)
    x = {k: np.asarray(v, dtype=np.float64).copy() for k, v in init.items()}
    # Dykstra 的关键：每个集合各自维护一份增量，而不是所有集合共用一份
    p = [{k: np.zeros_like(x[k]) for k in S.keys} for S in sets]
    for _ in range(max(int(n_iter), 1)):
        for i, S in enumerate(sets):
            keys = S.keys
            y = {k: x[k] + p[i][k] for k in keys}
            z = S.project(y)
            for k in keys:
                p[i][k] = y[k] - z[k]
                x[k] = z[k]
    return x


def admm_projection(
    init: dict[str, np.ndarray],
    sets: Iterable[ConstraintSet],
    n_iter: int = 8,
    rho: float = 1.0,
) -> dict[str, np.ndarray]:
    sets = list(sets)
    x0 = {k: np.asarray(v, dtype=np.float64).copy() for k, v in init.items()}
    x = {k: v.copy() for k, v in x0.items()}
    z = [{k: x[k].copy() for k in S.keys} for S in sets]
    u = [{k: np.zeros_like(x[k]) for k in S.keys} for S in sets]
    count = {k: 0 for k in x}
    for S in sets:
        for k in S.keys:
            count[k] += 1

    for _ in range(max(int(n_iter), 1)):
        for k in x:
            acc = x0[k].copy()
            for i, S in enumerate(sets):
                if k in S.keys:
                    acc += rho * (z[i][k] - u[i][k])
            x[k] = acc / (1.0 + rho * max(count[k], 1))
        for i, S in enumerate(sets):
            inp = {k: x[k] + u[i][k] for k in S.keys}
            z_new = S.project(inp)
            for k in S.keys:
                u[i][k] += x[k] - z_new[k]
                z[i][k] = z_new[k]
    return x


@dataclass
class ModelConstraints:
    """把配置翻译成若干凸集，并对物理模型参数做投影。"""

    config: ConstraintConfig
    dx: float
    dz: float
    params: tuple[str, ...]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.sets: list[ConstraintSet] = []
        p = set(self.params)
        bounds = {
            "vp": self.config.vp_bounds,
            "vs": self.config.vs_bounds,
            "rho": self.config.rho_bounds,
            "epsilon": self.config.epsilon_bounds,
            "delta": self.config.delta_bounds,
        }
        active_bounds = [k for k in self.params if k in bounds]
        if active_bounds:
            self.sets.append(
                BoxSet(
                    tuple(active_bounds),
                    {k: float(bounds[k][0]) for k in active_bounds},
                    {k: float(bounds[k][1]) for k in active_bounds},
                )
            )
        grads = {
            "vp": self.config.vp_grad_max,
            "vs": self.config.vs_grad_max,
            "rho": self.config.rho_grad_max,
            "epsilon": self.config.epsilon_grad_max,
            "delta": self.config.delta_grad_max,
        }
        for k in self.params:
            if k in grads:
                gx, gz = grads[k]
                if gx > 0 or gz > 0:
                    self.sets.append(
                        GradientBoundSet(
                            k, float(gx), float(gz), self.dx, self.dz,
                            n_iter=self.config.grad_proj_iter,
                            rho=self.config.grad_proj_rho,
                        )
                    )
        if self.config.use_gardner and {"vp", "rho"}.issubset(p):
            self.sets.append(
                GardnerTube(
                    a=self.config.gardner_a,
                    b=self.config.gardner_b,
                    tol=self.config.gardner_tol,
                )
            )

    # ------------------------------------------------------------------
    def project(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.config.method == "none" or not self.sets:
            return {k: np.asarray(v, dtype=np.float64).copy() for k, v in values.items()}
        if self.config.method == "dykstra":
            out = dykstra_projection(values, self.sets, self.config.n_iter)
        elif self.config.method == "admm":
            out = admm_projection(values, self.sets, self.config.n_iter, self.config.admm_rho)
        else:
            raise ValueError(f"未知约束投影方法 {self.config.method!r}")
        if {"vp", "vs"}.issubset(out):
            out["vs"] = np.minimum(out["vs"], 0.98 * out["vp"])
        return out

    def project_model(self, model: ElasticModel) -> ElasticModel:
        values = {k: np.asarray(getattr(model, k), dtype=np.float64) for k in self.params}
        projected = self.project(values)
        return model.with_params(**projected)

    def violations(self, model: ElasticModel) -> dict[str, float]:
        """返回各约束的最大违反量，便于诊断。"""
        out = {}
        values = {k: np.asarray(getattr(model, k), dtype=np.float64) for k in self.params}
        for i, S in enumerate(self.sets):
            if isinstance(S, BoxSet):
                for k in S.keys:
                    v = values[k]
                    out[f"box_{k}"] = float(
                        max(np.max(S.lower[k] - v), np.max(v - S.upper[k]), 0.0)
                    )
            elif isinstance(S, GradientBoundSet):
                v = values[S.key]
                gx = np.max(np.abs(_d_dx(v, S.dx))) if S.gx > 0 else 0.0
                gz = np.max(np.abs(_d_dz(v, S.dz))) if S.gz > 0 else 0.0
                out[f"grad_{S.key}"] = float(max(gx - S.gx, gz - S.gz, 0.0))
            elif isinstance(S, GardnerTube):
                r = S.residual(values)
                out["gardner"] = float(max(np.max(np.abs(r)) - S.tol, 0.0))
        return out
