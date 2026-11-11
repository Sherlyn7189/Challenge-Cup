"""全局配置对象。

所有数值参数都集中在这里，便于把文档中的实现规格逐条落地：

* 多尺度：``StageConfig`` 默认 8 Hz -> 15 Hz，各 25 次迭代；
* 伪 Hessian：``ph_damping`` / ``ph_smooth``；
* 拟牛顿：``lbfgs_memory`` / ``max_line_search``；
* 地质与岩石物理约束：``ConstraintConfig``（上下界、梯度上限、Gardner 关系）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class GridConfig:
    """二维规则网格（x 为水平方向，z 为深度方向，单位 m）。

    模型数组形状为 ``(nx, nz)``，网格节点坐标为 ``x = i*dx``、``z = j*dz``。
    正演时在四周各加 ``pad`` 个虚拟节点，并在物理边界内设置 ``sponge`` 个
    节点的吸收层。震源/检波器应放在吸收层以内。
    """

    nx: int
    nz: int
    dx: float
    dz: float
    pad: int = 2
    sponge: int = 16

    def __post_init__(self) -> None:
        if self.nx < 2 * self.pad + 8 or self.nz < 2 * self.pad + 8:
            raise ValueError("网格太小：nx/nz 至少为 2*pad+8")
        if self.dx <= 0 or self.dz <= 0:
            raise ValueError("dx/dz 必须为正")
        if self.sponge < 4:
            raise ValueError("sponge 至少为 4 个网格")
        if self.sponge >= min(self.nx, self.nz) // 3:
            raise ValueError("sponge 过宽，会吃掉有效模型区域")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.nz)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """给 imshow 使用的 ``(x0, x1, z1, z0)``，z 轴向下。"""
        return (0.0, (self.nx - 1) * self.dx, (self.nz - 1) * self.dz, 0.0)

    @property
    def x(self) -> Any:
        import numpy as np

        return np.arange(self.nx) * self.dx

    @property
    def z(self) -> Any:
        import numpy as np

        return np.arange(self.nz) * self.dz

    def cfl_dt(self, vmax: float, safety: float = 0.45) -> float:
        """二阶交错网格弹性波方程的 CFL 时间步长。

        ``dt <= safety * h_min / (vmax * sqrt(2))``；默认 safety=0.45，
        对强各向异性/强反差模型偏保守。
        """
        h = min(self.dx, self.dz)
        return float(safety * h / (vmax * (2.0 ** 0.5)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageConfig:
    """一个多尺度频带。``f_low``/``f_high`` 为 Wiener 带通的低/高截频。"""

    f_low: float
    f_high: float
    n_iter: int
    name: str = ""

    def __post_init__(self) -> None:
        if not (0 < self.f_low < self.f_high):
            raise ValueError("需要 0 < f_low < f_high")
        if self.n_iter < 0:
            raise ValueError("n_iter 不能为负")


def default_stages() -> list[StageConfig]:
    """文档规定：先 8 Hz、再 15 Hz，各约 25 次迭代。"""
    return [
        StageConfig(f_low=2.0, f_high=8.0, n_iter=25, name="band_8Hz"),
        StageConfig(f_low=3.0, f_high=15.0, n_iter=25, name="band_15Hz"),
    ]


@dataclass
class ConstraintConfig:
    """地质/岩石物理约束（均为凸集）。

    参数单位：速度 m/s，密度 kg/m^3，梯度单位 (m/s)/m 或 (kg/m^3)/m。
    ``vp_grad_max=(横向, 垂向)``。Gardner 关系在 log-log 空间是线性关系，
    因此其“管道”是凸集，可精确投影。
    """

    vp_bounds: tuple[float, float] = (1200.0, 6500.0)
    vs_bounds: tuple[float, float] = (250.0, 3800.0)
    rho_bounds: tuple[float, float] = (1400.0, 3200.0)
    epsilon_bounds: tuple[float, float] = (-0.05, 0.45)
    delta_bounds: tuple[float, float] = (-0.05, 0.45)

    vp_grad_max: tuple[float, float] = (8.0, 12.0)
    vs_grad_max: tuple[float, float] = (5.0, 8.0)
    rho_grad_max: tuple[float, float] = (4.0, 6.0)
    epsilon_grad_max: tuple[float, float] = (0.05, 0.06)
    delta_grad_max: tuple[float, float] = (0.05, 0.06)

    use_gardner: bool = True
    gardner_a: float = 310.0
    gardner_b: float = 0.25
    gardner_tol: float = 0.10  # log 密度空间内的半宽，约 10%

    method: str = "dykstra"  # "dykstra" | "admm" | "none"
    n_iter: int = 6
    admm_rho: float = 1.0
    grad_proj_iter: int = 20
    grad_proj_rho: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FWIConfig:
    """EFWI 主循环配置。"""

    stages: list[StageConfig] = field(default_factory=default_stages)
    params: tuple[str, ...] = ("vp", "vs", "rho")
    invert_anisotropy: bool = False

    # 拟牛顿
    lbfgs_memory: int = 8
    max_line_search: int = 8
    c1: float = 1e-4
    c2: float = 0.9
    max_rel_step: float = 0.05
    alpha_init: float = 1.0
    min_alpha: float = 1e-6

    # 伪 Hessian 预处理
    pseudo_hessian: bool = True
    ph_damping: float = 1e-3
    ph_smooth: int = 2

    # 正演/存储
    storage_mb: float = 512.0
    dtype: str = "float32"
    workers: int = 1
    verbose: bool = True

    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        allowed = {"vp", "vs", "rho", "epsilon", "delta"}
        for key in self.params:
            if key not in allowed:
                raise ValueError(f"未知参数 {key!r}")
        if self.invert_anisotropy:
            for key in ("epsilon", "delta"):
                if key not in self.params:
                    self.params = self.params + (key,)
        if self.ph_damping <= 0:
            raise ValueError("ph_damping 必须为正")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype 只能是 float32/float64")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = [asdict(s) for s in self.stages]
        return d
