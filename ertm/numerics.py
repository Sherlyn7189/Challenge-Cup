"""有限差分、吸收边界与信号处理基础算子。

本模块只提供“与物理无关”的数值工具，方便正演（:mod:`ertm.propagation`）、
成像（:mod:`ertm.imaging`）和预处理（:mod:`ertm.preprocess`）共用同一套
离散格式，避免各模块各自实现一份差分而产生不一致。

包含内容
--------
- 交错网格八阶（可降到 2/4/6 阶）空间导数算子，系数见
  ``docs/parts/03_decoupling_dimension_grid.md`` 式 (3.4)；
- CFS-PML（复杂频率偏移完全匹配层）系数与记忆变量；
- CFL 稳定性上限与主频采样点数（PPW）估计；
- Kaiser 窗 sinc 时间插值（用于 36/40 μs 实测采样到稳定 dt 的重采样）；
- Laplacian、波数域低通、余弦 taper、Ricker 子波、双线性采样等工具。

约定
----
- 二维数组统一 ``(nz, nx)``，``axis=0`` 对应 z，``axis=1`` 对应 x；
- ``dplus`` 给出 ``i+1/2`` 处的导数，``dminus`` 给出 ``i-1/2`` 处的导数，
  这样一阶速度—应力系统的交错位置无需任何插值；
- 所有算子都在“已带填充”的数组上工作（填充宽度 ≥ ``ghost_width(order)``），
  边界外数值恒为零，配合 PML 使用。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# ----------------------------------------------------------------------------
# 差分系数
# ----------------------------------------------------------------------------
#: 交错网格一阶导数系数（Levander 1988；八阶见技术方案式 3.4）。
STAGGERED_COEFFICIENTS: dict[int, tuple[float, ...]] = {
    2: (1.0,),
    4: (9.0 / 8.0, -1.0 / 24.0),
    6: (75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0),
    8: (1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0),
}

SUPPORTED_ORDERS = tuple(sorted(STAGGERED_COEFFICIENTS))


def staggered_coefficients(order: int) -> tuple[float, ...]:
    """返回交错网格差分系数；不支持的阶数抛出 ``ValueError``。"""
    try:
        return STAGGERED_COEFFICIENTS[int(order)]
    except (KeyError, TypeError) as exc:  # pragma: no cover - 配置层已先校验
        raise ValueError(
            f"不支持的空间阶数 {order!r}，可选 {SUPPORTED_ORDERS}"
        ) from exc


def coefficient_l1(order: int) -> float:
    """差分系数绝对值之和，用于 CFL 估计。"""
    return float(sum(abs(c) for c in staggered_coefficients(order)))


def ghost_width(order: int) -> int:
    """差分模板触及的最远格点距离（含安全余量），即数组填充宽度。"""
    return len(staggered_coefficients(order)) + 1


# ----------------------------------------------------------------------------
# 交错网格导数
# ----------------------------------------------------------------------------
def _staggered_derivative(
    field: np.ndarray,
    axis: int,
    coeffs: tuple[float, ...],
    spacing: float,
    *,
    forward: bool,
) -> np.ndarray:
    """交错网格导数核心实现（原地累加 + 预分配输出，避免大规模临时数组）。"""
    array = np.asarray(field)
    if array.ndim != 2:
        raise ValueError(f"导数算子只支持二维数组，当前 {array.ndim} 维")
    if axis not in (0, 1):
        raise ValueError(f"axis 必须是 0 或 1，当前 {axis}")
    if spacing <= 0:
        raise ValueError(f"网格间距必须为正，当前 {spacing}")

    out = np.zeros_like(array, dtype=np.result_type(array.dtype, np.float32))
    n = array.shape[axis]
    k_max = len(coeffs)
    lo, hi = k_max, n - k_max
    if lo >= hi:
        raise ValueError(
            f"数组在 axis={axis} 上长度 {n} 不足以支持 {k_max} 阶差分模板"
        )

    if axis == 0:
        target = out[lo:hi, :]
        for k, coeff in enumerate(coeffs, start=1):
            if forward:
                target += coeff * (
                    array[lo + k : hi + k, :] - array[lo - k + 1 : hi - k + 1, :]
                )
            else:
                target += coeff * (
                    array[lo + k - 1 : hi + k - 1, :] - array[lo - k : hi - k, :]
                )
    else:
        target = out[:, lo:hi]
        for k, coeff in enumerate(coeffs, start=1):
            if forward:
                target += coeff * (
                    array[:, lo + k : hi + k] - array[:, lo - k + 1 : hi - k + 1]
                )
            else:
                target += coeff * (
                    array[:, lo + k - 1 : hi + k - 1] - array[:, lo - k : hi - k]
                )
    out /= spacing
    return out


def dplus(
    field: np.ndarray,
    axis: int,
    coeffs: Iterable[float],
    spacing: float,
) -> np.ndarray:
    """``i+1/2`` 处的交错导数 ``∂f/∂x``（或 ``∂f/∂z``）。"""
    return _staggered_derivative(field, axis, tuple(coeffs), spacing, forward=True)


def dminus(
    field: np.ndarray,
    axis: int,
    coeffs: Iterable[float],
    spacing: float,
) -> np.ndarray:
    """``i-1/2`` 处的交错导数 ``∂f/∂x``（或 ``∂f/∂z``）。"""
    return _staggered_derivative(field, axis, tuple(coeffs), spacing, forward=False)


# ----------------------------------------------------------------------------
# CFS-PML
# ----------------------------------------------------------------------------
class CPMLAxis:
    """单轴 CFS-PML 的衰减剖面与递推系数。

    记忆变量递推（Komatitsch & Martin 2007）::

        ψ^{n} = b·ψ^{n-1} + a·D f
        ∂f/∂x|_PML = D f / κ + ψ

    其中 ``b = exp(-(σ/κ + α)Δt)``，``a = σ/(σ + κ·α)·(b-1)``，仅在 PML 层内
    ``σ>0``；层外 ``b=1, a=0``，等价于不加边界处理。
    """

    def __init__(
        self,
        length: int,
        spacing: float,
        npml: int,
        dt: float,
        vmax: float,
        *,
        axis: int,
        profile_order: int = 2,
        reflection: float = 1e-5,
        kappa_max: float = 1.0,
        alpha_max: float | None = None,
        sides: tuple[bool, bool] = (True, True),
        enabled: bool = True,
    ) -> None:
        if length < 2:
            raise ValueError(f"PML 轴长度至少为 2，当前 {length}")
        self.length = int(length)
        self.spacing = float(spacing)
        self.npml = int(max(1, min(npml, length // 3)))
        self.axis = axis
        self.dt = float(dt)
        self.enabled = bool(enabled)
        self.sides = (bool(sides[0]), bool(sides[1]))

        sigma = np.zeros(length, dtype=np.float64)
        kappa = np.ones(length, dtype=np.float64)
        alpha = np.zeros(length, dtype=np.float64)

        depth = max(self.npml * spacing, spacing)
        sigma_max = (
            -(profile_order + 1)
            * vmax
            * np.log(max(reflection, 1e-12))
            / (2.0 * depth)
        )
        alpha_peak = float(alpha_max) if alpha_max is not None else float(
            max(2.0 * np.pi * 5.0, 1.0)
        )

        cells = np.arange(self.npml, dtype=np.float64)
        # d 从 1（最外层）衰减到 0（PML 内边界）
        d = (self.npml - cells) / self.npml
        profile = d**profile_order
        if self.enabled and self.sides[0]:
            sigma[: self.npml] = sigma_max * profile
            kappa[: self.npml] = 1.0 + (kappa_max - 1.0) * profile
            alpha[: self.npml] = alpha_peak * (1.0 - d)
        if self.enabled and self.sides[1]:
            sigma[length - self.npml :] = sigma_max * profile[::-1]
            kappa[length - self.npml :] = 1.0 + (kappa_max - 1.0) * profile[::-1]
            alpha[length - self.npml :] = alpha_peak * (1.0 - d[::-1])

        denom = sigma * kappa + kappa * kappa * alpha
        safe = denom > 0
        b = np.exp(-(sigma / kappa + alpha) * self.dt)
        a = np.zeros_like(sigma)
        a[safe] = sigma[safe] / denom[safe] * (b[safe] - 1.0)
        a[~safe] = 0.0

        profile_shape = (1, length) if axis == 1 else (length, 1)
        self.b = b.reshape(profile_shape).astype(np.float32)
        self.a = a.reshape(profile_shape).astype(np.float32)
        self.kappa = kappa.reshape(profile_shape).astype(np.float32)

    def apply(self, derivative: np.ndarray, memory: np.ndarray) -> np.ndarray:
        """更新记忆变量并返回带 PML 修正的导数（不改变 ``derivative``）。"""
        memory *= self.b
        memory += self.a * derivative
        if float(np.min(self.kappa)) == 1.0:
            return derivative + memory
        return derivative / self.kappa + memory


class CPML2D:
    """二维 CFS-PML：按“导数项名称”管理各自的记忆变量。"""

    def __init__(
        self,
        shape: tuple[int, int],
        dx: float,
        dz: float,
        npml: int,
        dt: float,
        vmax: float,
        f0: float,
        *,
        profile_order: int = 2,
        reflection: float = 1e-5,
        kappa_max: float = 1.0,
        z_sides: tuple[bool, bool] = (True, True),
        enabled: bool = True,
    ) -> None:
        nz, nx = int(shape[0]), int(shape[1])
        alpha_max = 2.0 * np.pi * float(max(f0, 1.0))
        self.x = CPMLAxis(
            nx,
            dx,
            npml,
            dt,
            vmax,
            axis=1,
            profile_order=profile_order,
            reflection=reflection,
            kappa_max=kappa_max,
            alpha_max=alpha_max,
            enabled=enabled,
        )
        self.z = CPMLAxis(
            nz,
            dz,
            npml,
            dt,
            vmax,
            axis=0,
            profile_order=profile_order,
            reflection=reflection,
            kappa_max=kappa_max,
            alpha_max=alpha_max,
            sides=z_sides,
            enabled=enabled,
        )
        self._memory: dict[str, np.ndarray] = {}

    def memory_bytes(self) -> int:
        return int(sum(item.nbytes for item in self._memory.values()))

    def derivative(
        self,
        name: str,
        field: np.ndarray,
        *,
        axis: int,
        forward: bool,
        order: int,
        spacing: float,
    ) -> np.ndarray:
        """计算带 PML 修正的交错导数；``name`` 唯一标识该导数项。"""
        raw = (
            dplus(field, axis, staggered_coefficients(order), spacing)
            if forward
            else dminus(field, axis, staggered_coefficients(order), spacing)
        )
        memory = self._memory.get(name)
        if memory is None:
            memory = np.zeros_like(raw, dtype=np.float32)
            self._memory[name] = memory
        elif memory.shape != raw.shape:
            raise ValueError(
                f"PML 记忆变量 {name!r} 形状 {memory.shape} 与导数 {raw.shape} 不一致"
            )
        pml_axis = self.x if axis == 1 else self.z
        return pml_axis.apply(raw, memory)


# ----------------------------------------------------------------------------
# 稳定性与精度估计
# ----------------------------------------------------------------------------
def cfl_dt(vmax: float, dx: float, dz: float, order: int, safety: float = 0.5) -> float:
    """二维交错网格 CFL 稳定上限 ``Δt_max``。

    一阶速度—应力系统的最严格条件为::

        Δt ≤ safety / (vmax · sqrt((C/dx)² + (C/dz)²))

    其中 ``C`` 为差分系数绝对值之和。``safety`` 默认 0.5，兼顾频散误差。
    """
    if vmax <= 0:
        raise ValueError(f"vmax 必须为正，当前 {vmax}")
    c = coefficient_l1(order)
    return float(safety / (vmax * np.hypot(c / dx, c / dz)))


def points_per_wavelength(vmin: float, f0: float, spacing: float) -> float:
    """最小波长内的网格点数（PPW），工程上建议 ≥ 8。"""
    return float(vmin / (f0 * spacing)) if vmin > 0 and f0 > 0 else 0.0


# ----------------------------------------------------------------------------
# 时间重采样（36/40 μs 实测采样 → 稳定 Δt）
# ----------------------------------------------------------------------------
def resample_sinc(
    data: np.ndarray,
    dt_in: float,
    dt_out: float,
    *,
    axis: int = 0,
    half_length: int = 8,
    beta: float = 8.6,
) -> np.ndarray:
    """Kaiser 窗 sinc 重采样。

    地震道的频谱集中在低频段，sinc 插值比线性/三次插值更能保幅保相位，
    因此用它把实测采样（36 μs、40 μs）插值到满足 CFL 的 ``dt_out``。
    输入 ``data`` 沿 ``axis`` 为时间维，其余维度视为道并向量化处理。

    返回新数组，长度 ``n_out = floor((n_in-1)·dt_in/dt_out) + 1``。
    """
    array = np.asarray(data, dtype=np.float64)
    if dt_in <= 0 or dt_out <= 0:
        raise ValueError("dt_in / dt_out 必须为正")
    if array.ndim == 1:
        array = array[:, None]
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"仅支持 1D/2D 数据，当前 {array.ndim} 维")
    if axis == 1:
        array = array.T
    n_in = array.shape[0]
    if n_in < 3:
        raise ValueError(f"时间采样点太少（{n_in}），无法插值")

    duration = (n_in - 1) * dt_in
    n_out = int(np.floor(duration / dt_out + 1e-9)) + 1
    positions = np.arange(n_out, dtype=np.float64) * (dt_out / dt_in)

    taps = np.arange(-half_length + 1, half_length + 1)
    base = np.floor(positions).astype(np.int64)
    offsets = base[:, None] + taps[None, :]
    delta = positions[:, None] - offsets
    window = np.kaiser(2 * half_length, beta)[None, :]
    weights = np.sinc(delta) * window
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum == 0] = 1.0
    weights /= weight_sum

    clipped = np.clip(offsets, 0, n_in - 1)
    gathered = array[clipped]  # (n_out, n_taps, n_trace)
    out = np.einsum("ot,ots->os", weights, gathered)
    result = np.ascontiguousarray(out)
    if squeeze:
        result = result[:, 0]
    if axis == 1:
        result = result.T
    return result


# ----------------------------------------------------------------------------
# 滤波与图像处理
# ----------------------------------------------------------------------------
def laplacian(field: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """五点 Laplacian（式 6.13），用于压制零延迟互相关的低波数假象。"""
    array = np.asarray(field, dtype=np.float64)
    padded = np.pad(array, 1, mode="edge")
    d2x = (padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / (dx * dx)
    d2z = (padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / (dz * dz)
    return d2x + d2z


def _gaussian_kernel1d(sigma_samples: float, radius: int | None = None) -> np.ndarray:
    sigma = float(max(sigma_samples, 1e-6))
    radius = int(radius if radius is not None else max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth(field: np.ndarray, sigma_z: float, sigma_x: float) -> np.ndarray:
    """可分离高斯平滑（sigma 以“网格点数”为单位）。"""
    array = np.asarray(field, dtype=np.float64)
    kz = _gaussian_kernel1d(sigma_z)
    kx = _gaussian_kernel1d(sigma_x)
    temp = np.apply_along_axis(lambda col: np.convolve(col, kz, mode="same"), 0, array)
    return np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), 1, temp)


def lowpass_wavenumber(
    field: np.ndarray,
    dx: float,
    dz: float,
    cutoff_fraction: float = 0.85,
) -> np.ndarray:
    """波数域低通：``cutoff_fraction`` 为 Nyquist 的比例（0,1]。"""
    if not 0.0 < cutoff_fraction <= 1.0:
        raise ValueError(f"cutoff_fraction 必须在 (0,1]，当前 {cutoff_fraction}")
    array = np.asarray(field, dtype=np.float64)
    nz, nx = array.shape
    kz = np.fft.fftfreq(nz, d=dz)
    kx = np.fft.fftfreq(nx, d=dx)
    kz_max = 0.5 / dz
    kx_max = 0.5 / dx
    radial = np.sqrt((kz[:, None] / max(kz_max, 1e-12)) ** 2 + (kx[None, :] / max(kx_max, 1e-12)) ** 2)
    response = np.clip(1.0 - (radial - cutoff_fraction) / 0.15, 0.0, 1.0)
    spectrum = np.fft.fft2(array)
    return np.real(np.fft.ifft2(spectrum * response))


def cosine_taper(field: np.ndarray, width: int, axes: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """对数组边缘做余弦 taper，避免 FFT/滤波的截断振铃。"""
    array = np.array(field, dtype=np.float64, copy=True)
    width = int(max(0, width))
    if width == 0:
        return array
    for axis in axes:
        n = array.shape[axis]
        w = min(width, n // 2)
        if w <= 0:
            continue
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(w) + 0.5) / w)
        view = np.moveaxis(array, axis, 0)
        view[:w] *= ramp[:, None] if view.ndim > 1 else ramp
        view[-w:] *= ramp[::-1][:, None] if view.ndim > 1 else ramp[::-1]
    return array


# ----------------------------------------------------------------------------
# 子波与采样
# ----------------------------------------------------------------------------
def ricker(t: np.ndarray, f0: float, t0: float = 0.0) -> np.ndarray:
    """Ricker 子波（峰值 1，位于 ``t0``）。"""
    arg = np.pi * float(f0) * (np.asarray(t) - float(t0))
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def gaussian_pulse(t: np.ndarray, f0: float, t0: float = 0.0) -> np.ndarray:
    """高斯包络脉冲（用于需要宽带激发源的测试）。"""
    arg = np.pi * float(f0) * (np.asarray(t) - float(t0))
    return np.exp(-0.5 * arg**2).astype(np.float32)


def bilinear_sample(
    field: np.ndarray,
    z_index: np.ndarray,
    x_index: np.ndarray,
) -> np.ndarray:
    """按（可能为小数的）数组索引双线性采样，越界返回 0。

    ``z_index``/``x_index`` 是“数组索引坐标”（不是物理坐标），调用方负责
    加上填充偏移。返回形状与输入索引一致的数组。
    """
    array = np.asarray(field)
    zi = np.asarray(z_index, dtype=np.float64)
    xi = np.asarray(x_index, dtype=np.float64)
    z0 = np.floor(zi).astype(np.int64)
    x0 = np.floor(xi).astype(np.int64)
    tz = zi - z0
    tx = xi - x0

    nz, nx = array.shape
    valid = (z0 >= 0) & (z0 < nz - 1) & (x0 >= 0) & (x0 < nx - 1)
    zc = np.clip(z0, 0, nz - 2)
    xc = np.clip(x0, 0, nx - 2)

    v00 = array[zc, xc]
    v01 = array[zc, xc + 1]
    v10 = array[zc + 1, xc]
    v11 = array[zc + 1, xc + 1]
    value = (
        v00 * (1 - tz) * (1 - tx)
        + v01 * (1 - tz) * tx
        + v10 * tz * (1 - tx)
        + v11 * tz * tx
    )
    return np.where(valid, value, 0.0)
