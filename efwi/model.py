"""弹性参数模型与刚度矩阵。

支持三类对称性：

* ``iso``：各向同性，参数 ``(vp, vs, rho)``；
* ``vti``：垂直对称轴横向各向同性（页岩最常见的近似），
  参数 ``(vp, vs, rho, epsilon, delta, gamma)``；
* ``hti``：水平对称轴横向各向同性（垂直裂缝），在二维 x-z 平面内
  由 VTI 刚度绕 z 轴旋转 90° 得到（C11 <-> C33，C13/C44 不变）。

二维 P-SV 问题只需要 ``C11, C13, C33, C44`` 四个刚度分量，SH 分量
（由 ``gamma`` 控制）不参与正演，但保留在模型中供三维扩展和测井解释。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter


def _as_field(value: float | np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} 形状应为 {shape}，实际 {arr.shape}")
    return arr.copy()


@dataclass
class ElasticModel:
    """二维弹性参数模型（数组形状 ``(nx, nz)``，z 向下为正）。"""

    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    epsilon: float | np.ndarray = 0.0
    delta: float | np.ndarray = 0.0
    gamma: float | np.ndarray = 0.0
    symmetry: str = "iso"
    dx: float | None = None
    dz: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vp = np.asarray(self.vp, dtype=np.float64)
        self.vs = np.asarray(self.vs, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        if self.vp.ndim != 2:
            raise ValueError("vp/vs/rho 必须是二维数组")
        shape = self.vp.shape
        if self.vs.shape != shape or self.rho.shape != shape:
            raise ValueError("vp/vs/rho 形状必须一致")
        self.epsilon = _as_field(self.epsilon, shape, "epsilon")
        self.delta = _as_field(self.delta, shape, "delta")
        self.gamma = _as_field(self.gamma, shape, "gamma")
        self.symmetry = str(self.symmetry).lower()
        if self.symmetry not in ("iso", "vti", "hti"):
            raise ValueError("symmetry 必须是 iso/vti/hti")
        self.validate()

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.vp.shape

    @property
    def nx(self) -> int:
        return self.vp.shape[0]

    @property
    def nz(self) -> int:
        return self.vp.shape[1]

    @property
    def anisotropy(self) -> bool:
        return self.symmetry != "iso"

    def validate(self) -> "ElasticModel":
        for name in ("vp", "vs", "rho"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} 含 NaN/Inf")
            if np.any(arr <= 0):
                raise ValueError(f"{name} 必须为正")
        if np.any(self.vs >= self.vp):
            raise ValueError("必须满足 vs < vp")
        for name in ("epsilon", "delta", "gamma"):
            arr = getattr(self, name)
            if np.any(arr < -0.5) or np.any(arr > 1.0):
                raise ValueError(f"{name} 超出合理范围 [-0.5, 1]")
        return self

    def copy(self) -> "ElasticModel":
        return ElasticModel(
            self.vp.copy(),
            self.vs.copy(),
            self.rho.copy(),
            self.epsilon.copy(),
            self.delta.copy(),
            self.gamma.copy(),
            self.symmetry,
            self.dx,
            self.dz,
            dict(self.meta),
        )

    def with_params(self, **kwargs) -> "ElasticModel":
        data = dict(
            vp=self.vp,
            vs=self.vs,
            rho=self.rho,
            epsilon=self.epsilon,
            delta=self.delta,
            gamma=self.gamma,
            symmetry=self.symmetry,
            dx=self.dx,
            dz=self.dz,
            meta=dict(self.meta),
        )
        data.update(kwargs)
        return ElasticModel(**data)

    # ------------------------------------------------------------------
    # 刚度与拉梅参数
    # ------------------------------------------------------------------
    def stiffness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 ``(C11, C13, C33, C44)``（二维 P-SV 所需分量）。"""
        vp2 = self.vp ** 2
        vs2 = self.vs ** 2
        rho = self.rho
        eps = self.epsilon
        delta = self.delta

        if self.symmetry == "iso":
            c11 = rho * vp2
            c33 = rho * vp2
            c13 = rho * (vp2 - 2.0 * vs2)
            c44 = rho * vs2
            return c11, c13, c33, c44

        # VTI 与 HTI 共用的 C13：由 Thomsen 参数定义的垂向/水平速度组合
        q = (vp2 - vs2) * (vp2 * (1.0 + 2.0 * delta) - vs2)
        q = np.maximum(q, 1e-10 * np.maximum(vp2 * vp2, 1.0))
        c13 = rho * (np.sqrt(q) - vs2)
        c44 = rho * vs2
        if self.symmetry == "vti":
            c11 = rho * vp2 * (1.0 + 2.0 * eps)
            c33 = rho * vp2
        else:  # hti：绕 z 轴旋转 90°，快方向水平
            c11 = rho * vp2
            c33 = rho * vp2 * (1.0 + 2.0 * eps)
        return c11, c13, c33, c44

    def lame(self) -> tuple[np.ndarray, np.ndarray]:
        """各向同性拉梅参数；各向异性时返回等效 λ、μ。"""
        mu = self.rho * self.vs ** 2
        lam = self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)
        return lam, mu

    def max_velocity(self) -> float:
        """用于 CFL 的最大相速度估计。"""
        c11, _, c33, c44 = self.stiffness()
        vals = [np.max(np.sqrt(c11 / self.rho)), np.max(np.sqrt(c33 / self.rho))]
        vals.append(np.max(np.sqrt(c44 / self.rho)))
        return float(max(vals))

    def min_velocity(self) -> float:
        c11, _, c33, c44 = self.stiffness()
        vals = [np.min(np.sqrt(c11 / self.rho)), np.min(np.sqrt(c33 / self.rho))]
        vals.append(np.min(np.sqrt(c44 / self.rho)))
        return float(min(vals))

    def thomsen_velocities(self) -> dict[str, np.ndarray]:
        """按 Thomsen 近似给出垂向/水平 P、SV、SH 速度，便于测井对照。"""
        eps, delta, gamma = self.epsilon, self.delta, self.gamma
        vp = self.vp
        vs = self.vs
        return {
            "vp_vertical": vp,
            "vp_horizontal": vp * np.sqrt(1.0 + 2.0 * eps),
            "vs_vertical": vs,
            "vs_horizontal": vs * np.sqrt(1.0 + 2.0 * gamma),
            "vp_nmo": vp * np.sqrt(1.0 + 2.0 * delta),
        }

    # ------------------------------------------------------------------
    # 平滑 / 统计 / 保存
    # ------------------------------------------------------------------
    def smoothed(self, sigma: float, keys: Iterable[str] = ("vp", "vs", "rho")) -> "ElasticModel":
        data = {}
        for key in keys:
            arr = getattr(self, key)
            data[key] = gaussian_filter(arr, sigma=sigma, mode="nearest")
        if "vp" in data or "vs" in data:
            vp = data.get("vp", self.vp)
            vs = data.get("vs", self.vs)
            if np.any(vs >= vp):
                data["vs"] = np.minimum(vs, 0.95 * vp)
        return self.with_params(**data)

    def rms(self, key: str) -> float:
        return float(np.sqrt(np.mean(getattr(self, key) ** 2)))

    def relative_error(self, other: "ElasticModel", key: str) -> float:
        a = getattr(self, key)
        b = getattr(other, key)
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

    def describe(self) -> str:
        return (
            f"ElasticModel(shape={self.shape}, symmetry={self.symmetry}, "
            f"vp=[{self.vp.min():.0f},{self.vp.max():.0f}] m/s, "
            f"vs=[{self.vs.min():.0f},{self.vs.max():.0f}] m/s, "
            f"rho=[{self.rho.min():.0f},{self.rho.max():.0f}] kg/m3)"
        )

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            vp=self.vp,
            vs=self.vs,
            rho=self.rho,
            epsilon=np.asarray(self.epsilon),
            delta=np.asarray(self.delta),
            gamma=np.asarray(self.gamma),
            symmetry=np.array(self.symmetry),
            dx=np.array(self.dx if self.dx is not None else np.nan),
            dz=np.array(self.dz if self.dz is not None else np.nan),
        )

    @classmethod
    def load(cls, path: str) -> "ElasticModel":
        data = np.load(path, allow_pickle=False)
        dx = float(data["dx"])
        dz = float(data["dz"])
        return cls(
            data["vp"],
            data["vs"],
            data["rho"],
            data["epsilon"],
            data["delta"],
            data["gamma"],
            str(data["symmetry"]),
            None if np.isnan(dx) else dx,
            None if np.isnan(dz) else dz,
        )


# ----------------------------------------------------------------------
# 便于测试/示例的构造器
# ----------------------------------------------------------------------
def constant_model(
    nx: int,
    nz: int,
    vp: float = 3000.0,
    vs: float = 1600.0,
    rho: float = 2200.0,
    *,
    dx: float = 10.0,
    dz: float = 10.0,
    symmetry: str = "iso",
    epsilon: float = 0.0,
    delta: float = 0.0,
    gamma: float = 0.0,
) -> ElasticModel:
    shape = (nx, nz)
    return ElasticModel(
        np.full(shape, vp),
        np.full(shape, vs),
        np.full(shape, rho),
        epsilon,
        delta,
        gamma,
        symmetry,
        dx,
        dz,
    )


def gardner_density(vp: np.ndarray | float, a: float = 310.0, b: float = 0.25) -> np.ndarray:
    """Gardner 经验关系 ``rho = a * vp**b``（rho: kg/m3, vp: m/s）。"""
    return a * np.power(np.asarray(vp, dtype=np.float64), b)


def model_from_1d(
    vp: np.ndarray,
    vs: np.ndarray | None,
    rho: np.ndarray | None,
    nz: int,
    nx: int,
    *,
    dx: float = 10.0,
    dz: float = 10.0,
    symmetry: str = "iso",
    epsilon: float = 0.0,
    delta: float = 0.0,
    gamma: float = 0.0,
) -> ElasticModel:
    """把一维测井曲线横向复制成二维初始模型。"""
    vp = np.asarray(vp, dtype=np.float64).reshape(-1)
    if vp.size != nz:
        vp = np.interp(np.linspace(0, 1, nz), np.linspace(0, 1, vp.size), vp)
    vs = vp / 1.73 if vs is None else np.asarray(vs, dtype=np.float64).reshape(-1)
    if vs.size != nz:
        vs = np.interp(np.linspace(0, 1, nz), np.linspace(0, 1, vs.size), vs)
    rho = gardner_density(vp) if rho is None else np.asarray(rho, dtype=np.float64).reshape(-1)
    if rho.size != nz:
        rho = np.interp(np.linspace(0, 1, nz), np.linspace(0, 1, rho.size), rho)
    return ElasticModel(
        np.repeat(vp[None, :], nx, axis=0),
        np.repeat(vs[None, :], nx, axis=0),
        np.repeat(rho[None, :], nx, axis=0),
        epsilon,
        delta,
        gamma,
        symmetry,
        dx,
        dz,
    )
