"""Numba 加速的交错网格弹性波正演/伴随核。

二维 P-SV 速度—应力一阶方程组（Virieux 1986 型交错网格）::

    rho dvx/dt = d(sxx)/dx + d(sxz)/dz
    rho dvz/dt = d(sxz)/dx + d(szz)/dz
    dsxx/dt = C11 dvx/dx + C13 dvz/dz
    dszz/dt = C13 dvx/dx + C33 dvz/dz
    dsxz/dt = C44 (dvx/dz + dvz/dx)

各向同性时 ``C11=C33=rho*vp^2``、``C13=rho*(vp^2-2vs^2)``、
``C44=rho*vs^2``；VTI/HTI 时由模型模块给出对应刚度。

伴随方程是离散正演算子的转置：空间一阶差分转置为负的散度/梯度，
时间方向反向推进，数据残差作为伴随源注入接收点。这样得到的梯度与
有限差分梯度在离散意义下一致（见 tests/test_gradient.py）。
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ----------------------------------------------------------------------
# 震源
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def inject_source_stress(sxx, szz, si, sj, sw, amp, dt):
    """爆炸源：同时加到 sxx 与 szz（各向同性震源）。"""
    for k in range(si.size):
        v = dt * amp * sw[k]
        i = si[k]
        j = sj[k]
        sxx[i, j] += v
        szz[i, j] += v


@njit(cache=True, fastmath=True)
def inject_source_force_x(vx, rho_x, si, sj, sw, amp, dt):
    """x 方向力源：作为加速度加到 vx。"""
    for k in range(si.size):
        i = si[k]
        j = sj[k]
        vx[i, j] += dt * amp * sw[k] / rho_x[i, j]


@njit(cache=True, fastmath=True)
def inject_source_force_z(vz, rho_z, si, sj, sw, amp, dt):
    """z 方向力源：作为加速度加到 vz。"""
    for k in range(si.size):
        i = si[k]
        j = sj[k]
        vz[i, j] += dt * amp * sw[k] / rho_z[i, j]


# ----------------------------------------------------------------------
# 正演一步
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def update_velocity(
    vx, vz, sxx, szz, sxz, rho_x, rho_z,
    dt, dx, dz, ax, az, h_rho, accumulate_h,
):
    nx, nz = vx.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            bx = (sxx[i + 1, j] - sxx[i, j]) * inv_dx + (sxz[i, j] - sxz[i, j - 1]) * inv_dz
            bz = (sxz[i, j] - sxz[i - 1, j]) * inv_dx + (szz[i, j + 1] - szz[i, j]) * inv_dz
            rx = rho_x[i, j]
            rz = rho_z[i, j]
            axv = bx / rx
            azv = bz / rz
            vx[i, j] += dt * axv
            vz[i, j] += dt * azv
            ax[i, j] = axv
            az[i, j] = azv
            if accumulate_h:
                h_rho[i, j] += dt * ((axv / rx) ** 2 + (azv / rz) ** 2)


@njit(cache=True, fastmath=True)
def update_stress(
    vx, vz, sxx, szz, sxz, c11, c13, c33, c44c,
    dt, dx, dz, exx, ezz, exz,
    h_c11, h_c13, h_c33, h_c44, accumulate_h,
):
    nx, nz = vx.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            ex = (vx[i, j] - vx[i - 1, j]) * inv_dx
            ez = (vz[i, j] - vz[i, j - 1]) * inv_dz
            exz_v = (vx[i, j + 1] - vx[i, j]) * inv_dz + (vz[i + 1, j] - vz[i, j]) * inv_dx
            sxx[i, j] += dt * (c11[i, j] * ex + c13[i, j] * ez)
            szz[i, j] += dt * (c13[i, j] * ex + c33[i, j] * ez)
            sxz[i, j] += dt * c44c[i, j] * exz_v
            exx[i, j] = ex
            ezz[i, j] = ez
            exz[i, j] = exz_v
            if accumulate_h:
                h_c11[i, j] += dt * ex * ex
                h_c13[i, j] += dt * (ex * ex + ez * ez)
                h_c33[i, j] += dt * ez * ez
                h_c44[i, j] += dt * exz_v * exz_v


@njit(cache=True, fastmath=True)
def apply_damping(field, damp):
    nx, nz = field.shape
    for i in range(nx):
        for j in range(nz):
            field[i, j] *= damp[i, j]


# ----------------------------------------------------------------------
# 接收 / 伴随源注入
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def record_data(vx, vz, rix, rjx, rwx, riz, rjz, rwz, out):
    """把 vx/vz 双线性插值到检波点，``out`` 形状 ``(nr, 2)``。"""
    nr = out.shape[0]
    for r in range(nr):
        sx = 0.0
        sz = 0.0
        for k in range(4):
            sx += rwx[k, r] * vx[rix[k, r], rjx[k, r]]
            sz += rwz[k, r] * vz[riz[k, r], rjz[k, r]]
        out[r, 0] = sx
        out[r, 1] = sz


@njit(cache=True, fastmath=True)
def inject_adjoint_source(vax, vaz, rix, rjx, rwx, riz, rjz, rwz, residual):
    """把数据残差（或 RTM 的接收数据）散加到伴随速度场。"""
    nr = residual.shape[0]
    for r in range(nr):
        rx = residual[r, 0]
        rz = residual[r, 1]
        if rx != 0.0 or rz != 0.0:
            for k in range(4):
                vax[rix[k, r], rjx[k, r]] += rwx[k, r] * rx
                vaz[riz[k, r], rjz[k, r]] += rwz[k, r] * rz


# ----------------------------------------------------------------------
# 伴随一步（离散转置）
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def adjoint_update_velocity(vax, vaz, saxx, sazz, saxz, c11, c13, c33, c44c, dt, dx, dz):
    nx, nz = vax.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            pxx = c11[i, j] * saxx[i, j] + c13[i, j] * sazz[i, j]
            pxx_next = c11[i + 1, j] * saxx[i + 1, j] + c13[i + 1, j] * sazz[i + 1, j]
            pxz = c44c[i, j] * saxz[i, j]
            pxz_prev = c44c[i, j - 1] * saxz[i, j - 1]
            vax[i, j] += -dt * ((pxx_next - pxx) * inv_dx + (pxz - pxz_prev) * inv_dz)

            pzz = c13[i, j] * saxx[i, j] + c33[i, j] * sazz[i, j]
            pzz_next = c13[i, j + 1] * saxx[i, j + 1] + c33[i, j + 1] * sazz[i, j + 1]
            qxz = c44c[i, j] * saxz[i, j]
            qxz_prev = c44c[i - 1, j] * saxz[i - 1, j]
            vaz[i, j] += -dt * ((pzz_next - pzz) * inv_dz + (qxz - qxz_prev) * inv_dx)


@njit(cache=True, fastmath=True)
def adjoint_update_stress(vax, vaz, saxx, sazz, saxz, rho_x, rho_z, dt, dx, dz):
    nx, nz = vax.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            vxr = vax[i, j] / rho_x[i, j]
            vzr = vaz[i, j] / rho_z[i, j]
            vxr_xm = vax[i - 1, j] / rho_x[i - 1, j]
            vzr_zm = vaz[i, j - 1] / rho_z[i, j - 1]
            saxx[i, j] += -dt * (vxr - vxr_xm) * inv_dx
            sazz[i, j] += -dt * (vzr - vzr_zm) * inv_dz

            vxr_zp = vax[i, j + 1] / rho_x[i, j + 1]
            vzr_xp = vaz[i + 1, j] / rho_z[i + 1, j]
            saxz[i, j] += -dt * ((vxr_zp - vxr) * inv_dz + (vzr_xp - vzr) * inv_dx)


@njit(cache=True, fastmath=True)
def adjoint_step(
    vax, vaz, saxx, sazz, saxz,
    vax_new, vaz_new, saxx_new, sazz_new, saxz_new,
    c11, c13, c33, c44c, rho_x, rho_z, dt, dx, dz,
):
    """伴随场反向推进一步（离散转置），使用双缓冲避免原地更新污染。

    对应正演映射::

        v^{n+1/2} = v^{n-1/2} + dt/rho * D s^n
        s^{n+1}   = s^n + dt * C E v^{n+1/2}

    的转置::

        v~^{n-1/2}= v~^{n+1/2} + dt * E^T C s~^{n+1}
        s~^n      = s~^{n+1} + dt * D^T (v~^{n-1/2}/rho)
    """
    nx, nz = vax.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz

    # 双缓冲的边界必须先清零：第二遍在 i=1/j=1 处会读取 vax_new[0]/[nz-1]，
    # 若沿用两步前的旧值会污染梯度。
    for i in range(nx):
        vax_new[i, 0] = 0.0
        vax_new[i, nz - 1] = 0.0
        vaz_new[i, 0] = 0.0
        vaz_new[i, nz - 1] = 0.0
        saxx_new[i, 0] = 0.0
        saxx_new[i, nz - 1] = 0.0
        sazz_new[i, 0] = 0.0
        sazz_new[i, nz - 1] = 0.0
        saxz_new[i, 0] = 0.0
        saxz_new[i, nz - 1] = 0.0
    for j in range(nz):
        vax_new[0, j] = 0.0
        vax_new[nx - 1, j] = 0.0
        vaz_new[0, j] = 0.0
        vaz_new[nx - 1, j] = 0.0
        saxx_new[0, j] = 0.0
        saxx_new[nx - 1, j] = 0.0
        sazz_new[0, j] = 0.0
        sazz_new[nx - 1, j] = 0.0
        saxz_new[0, j] = 0.0
        saxz_new[nx - 1, j] = 0.0

    # 第一遍：v~^{n-1/2} = v~^{n+1/2} + dt * E^T C^T s~^{n+1}
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            pxx = c11[i, j] * saxx[i, j] + c13[i, j] * sazz[i, j]
            pxx_next = c11[i + 1, j] * saxx[i + 1, j] + c13[i + 1, j] * sazz[i + 1, j]
            pxz = c44c[i, j] * saxz[i, j]
            pxz_prev = c44c[i, j - 1] * saxz[i, j - 1]
            vax_new[i, j] = vax[i, j] - dt * (
                (pxx_next - pxx) * inv_dx + (pxz - pxz_prev) * inv_dz
            )

            pzz = c13[i, j] * saxx[i, j] + c33[i, j] * sazz[i, j]
            pzz_next = c13[i, j + 1] * saxx[i, j + 1] + c33[i, j + 1] * sazz[i, j + 1]
            qxz = c44c[i, j] * saxz[i, j]
            qxz_prev = c44c[i - 1, j] * saxz[i - 1, j]
            vaz_new[i, j] = vaz[i, j] - dt * (
                (pzz_next - pzz) * inv_dz + (qxz - qxz_prev) * inv_dx
            )

    # 第二遍：s~^n = s~^{n+1} + dt * D^T (v~^{n-1/2}/rho)
    # 必须等 v~ 全部更新完，否则相邻单元的 s~ 会用到尚未写入的值。
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            vxr = vax_new[i, j] / rho_x[i, j]
            vzr = vaz_new[i, j] / rho_z[i, j]
            vxr_xm = vax_new[i - 1, j] / rho_x[i - 1, j]
            vzr_zm = vaz_new[i, j - 1] / rho_z[i, j - 1]
            saxx_new[i, j] = saxx[i, j] - dt * (vxr - vxr_xm) * inv_dx
            sazz_new[i, j] = sazz[i, j] - dt * (vzr - vzr_zm) * inv_dz

            vxr_zp = vax_new[i, j + 1] / rho_x[i, j + 1]
            vzr_xp = vaz_new[i + 1, j] / rho_z[i + 1, j]
            saxz_new[i, j] = saxz[i, j] - dt * (
                (vxr_zp - vxr) * inv_dz + (vzr_xp - vzr) * inv_dx
            )

@njit(cache=True, fastmath=True)
def adjoint_stiffness_gradient(
    saxx, sazz, saxz, exx, ezz, exz, dt, g_c11, g_c13, g_c33, g_c44c,
):
    """累积刚度梯度（用本步的伴随应力与本步正演应变率）。"""
    nx, nz = saxx.shape
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            g_c11[i, j] += dt * saxx[i, j] * exx[i, j]
            g_c13[i, j] += dt * (saxx[i, j] * ezz[i, j] + sazz[i, j] * exx[i, j])
            g_c33[i, j] += dt * sazz[i, j] * ezz[i, j]
            g_c44c[i, j] += dt * saxz[i, j] * exz[i, j]


@njit(cache=True, fastmath=True)
def adjoint_density_gradient(
    vax, vaz, ax, az, rho_x, rho_z, dt, g_rho_x, g_rho_z,
):
    """累积密度梯度。

    注意使用的是**本步推进后的**速度伴随 ``v~^{n-1/2}``（即速度更新方程
    的 Lagrange 乘子），而不是 ``v~^{n+1/2}``；否则时间循环的离散梯度会
    有 O(dt) 的系统偏差。见 docs/EFWI_梯度推导.md。
    """
    nx, nz = vax.shape
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            g_rho_x[i, j] += -dt * vax[i, j] * ax[i, j] / rho_x[i, j]
            g_rho_z[i, j] += -dt * vaz[i, j] * az[i, j] / rho_z[i, j]


# ----------------------------------------------------------------------
# 波场分离（RTM 用）
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def divergence(vx, vz, out, dx, dz):
    """P 波场：速度散度，位于单元中心。"""
    nx, nz = vx.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            out[i, j] = (vx[i, j] - vx[i - 1, j]) * inv_dx + (vz[i, j] - vz[i, j - 1]) * inv_dz


@njit(cache=True, fastmath=True)
def curl(vx, vz, out, dx, dz):
    """S 波场：速度旋度（二维只有一个分量），位于单元中心。"""
    nx, nz = vx.shape
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            out[i, j] = (vz[i + 1, j] - vz[i, j]) * inv_dx - (vx[i, j + 1] - vx[i, j]) * inv_dz


@njit(cache=True, fastmath=True)
def accumulate_image(image, illum, source_field, receiver_field, dt):
    """RTM 互相关成像条件（源照明归一化在 Python 侧完成）。"""
    nx, nz = image.shape
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            s = source_field[i, j]
            image[i, j] += dt * s * receiver_field[i, j]
            illum[i, j] += dt * s * s


@njit(cache=True, fastmath=True)
def accumulate_illumination(illum, source_field, dt):
    nx, nz = illum.shape
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            illum[i, j] += dt * source_field[i, j] * source_field[i, j]
