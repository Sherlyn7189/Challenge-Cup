"""弹性波逆时偏移（RTM）与地质属性提取。

RTM 使用 EFWI 更新后的模型作为背景模型：源波场正演、接收波场反传，
在成像点做互相关。默认采用“源照明归一化”成像条件（保幅成像）::

    I(x) = sum_t s(x,t) r(x,t) / (sum_t s(x,t)^2 + eps)

弹性波场先做 P/S 分离：P 用速度散度，S 用速度旋度。这样 PP/SS 成像
不会把不同波型混在一起，也更接近保幅偏移的要求。

之后从成像结果提取反射点与倾角（结构张量），供三维地质建模和下一轮
EFWI 的地质约束使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .config import GridConfig
from .fd import Elastic2DOperator
from .geometry import Survey
from .model import ElasticModel


@dataclass
class RTMResult:
    image: np.ndarray
    illumination: np.ndarray
    model: ElasticModel
    wavefield: str
    imaging: str
    shot_images: np.ndarray | None = None

    def normalized(self, eps: float = 1e-6) -> np.ndarray:
        return self.image / (np.max(np.abs(self.image)) + eps)


def reverse_time_migration(
    model: ElasticModel,
    survey: Survey,
    data: np.ndarray,
    *,
    grid: GridConfig | None = None,
    wavefield: str = "p",
    imaging: str = "source_normalized",
    storage_mb: float = 256.0,
    eps: float = 1e-3,
    dtype: str = "float32",
    keep_shot_images: bool = False,
    verbose: bool = False,
) -> RTMResult:
    if grid is None:
        if model.dx is None or model.dz is None:
            raise ValueError("模型缺少 dx/dz")
        grid = GridConfig(model.nx, model.nz, model.dx, model.dz)
    op = Elastic2DOperator(model, survey, grid, dtype=dtype, storage_mb=storage_mb, verbose=verbose)
    image = np.zeros((op.NX, op.NZ), dtype=np.float64)
    illum = np.zeros((op.NX, op.NZ), dtype=np.float64)
    shot_images = [] if keep_shot_images else None

    for s in range(survey.n_shots):
        if verbose:
            print(f"  RTM shot {s + 1}/{survey.n_shots}")
        img_s, illum_s = op.rtm_shot(
            s, data[s], wavefield=wavefield, imaging=imaging,
            storage_mb=storage_mb, eps=eps,
        )
        image += img_s
        illum += illum_s
        if shot_images is not None:
            shot_images.append(img_s)

    sl = (slice(op.pad, op.pad + grid.nx), slice(op.pad, op.pad + grid.nz))
    out = RTMResult(
        image=image[sl].copy(),
        illumination=illum[sl].copy(),
        model=model,
        wavefield=wavefield,
        imaging=imaging,
        shot_images=np.asarray(shot_images) if shot_images is not None else None,
    )
    return out


# ----------------------------------------------------------------------
# 倾角 / 反射点
# ----------------------------------------------------------------------
@dataclass
class DipField:
    dip: np.ndarray          # 反射面倾角，弧度，[-pi/2, pi/2]
    coherence: np.ndarray    # 结构张量相干性 [0,1]
    reflection_mask: np.ndarray
    image: np.ndarray

    @property
    def dip_deg(self) -> np.ndarray:
        return np.degrees(self.dip)

    def reflector_points(self, min_amplitude_percentile: float = 75.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回反射点坐标/倾角/振幅。"""
        amp = np.abs(self.image)
        thr = np.percentile(amp, min_amplitude_percentile)
        mask = self.reflection_mask & (amp >= thr)
        ix, iz = np.nonzero(mask)
        return (
            ix.astype(np.float64),
            iz.astype(np.float64),
            self.dip[ix, iz],
            self.image[ix, iz],
        )


def extract_dips(
    image: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    *,
    sigma: float = 3.0,
    envelope_sigma: float = 1.5,
) -> DipField:
    """结构张量法估计反射面倾角与相干性。

    图像梯度方向垂直于反射面；结构张量主特征向量给出法向，减去 90°
    即反射面相对水平方向的倾角。
    """
    image = np.asarray(image, dtype=np.float64)
    gx = gaussian_filter(image, sigma=sigma, order=(1, 0), mode="nearest") / max(dx, 1e-12)
    gz = gaussian_filter(image, sigma=sigma, order=(0, 1), mode="nearest") / max(dz, 1e-12)
    jxx = gaussian_filter(gx * gx, sigma=sigma, mode="nearest")
    jzz = gaussian_filter(gz * gz, sigma=sigma, mode="nearest")
    jxz = gaussian_filter(gx * gz, sigma=sigma, mode="nearest")
    theta = 0.5 * np.arctan2(2.0 * jxz, jxx - jzz)
    dip = theta - 0.5 * np.pi
    # 折叠到 [-pi/2, pi/2]
    dip = (dip + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    tr = jxx + jzz
    coh = np.sqrt((jxx - jzz) ** 2 + 4.0 * jxz ** 2) / (tr + 1e-30)
    coh = np.clip(coh, 0.0, 1.0)

    env = gaussian_filter(np.abs(image), sigma=envelope_sigma, mode="nearest")
    local_max = maximum_filter(env, size=(3, 5), mode="nearest")
    thr = np.percentile(env, 60.0)
    mask = (env >= local_max - 1e-15) & (env > thr)
    return DipField(dip=dip, coherence=coh, reflection_mask=mask, image=image.copy())


def apply_laplacian_filter(image: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """可选的拉普拉斯锐化（会改变振幅，保幅流程默认不启用）。"""
    from scipy.ndimage import laplace

    return -laplace(image) / (dx * dz)


def top_mute(
    data: np.ndarray,
    dt: float,
    shots: np.ndarray,
    receivers: np.ndarray,
    velocity: float,
    *,
    factor: float = 0.9,
    taper: float = 0.02,
) -> np.ndarray:
    """压制直达波/近场：按炮检距预测直达波到时，在到时前做平滑切除。

    ``data`` 形状 ``(ns, nt, nr, ncomp)``。直接波能量通常比反射波强 1~2 个
    数量级，不切除会在 RTM 中形成浅部强假象。
    """
    data = np.asarray(data, dtype=np.float64)
    ns, nt, nr = data.shape[:3]
    t = np.arange(nt, dtype=np.float64) * dt
    dx = shots[:, None, 0] - receivers[None, :, 0]
    dz = shots[:, None, 1] - receivers[None, :, 1]
    offset = np.sqrt(dx * dx + dz * dz)
    t_mute = offset / max(velocity, 1e-6) * factor
    weight = 0.5 * (1.0 + np.tanh((t[None, :, None] - t_mute[:, None, :]) / max(taper, 1e-9)))
    return data * weight[..., None]


def depth_taper(image: np.ndarray, z: np.ndarray, z_start: float, z_end: float) -> np.ndarray:
    """对成像结果做深度域平滑切除，去掉震源附近/浅部采集脚印。"""
    image = np.asarray(image, dtype=np.float64).copy()
    z = np.asarray(z, dtype=np.float64)
    w = np.ones_like(z)
    w[z <= z_start] = 0.0
    band = (z > z_start) & (z < z_end)
    if np.any(band):
        u = (z[band] - z_start) / max(z_end - z_start, 1e-12)
        w[band] = 0.5 - 0.5 * np.cos(np.pi * u)
    return image * w[None, :]
