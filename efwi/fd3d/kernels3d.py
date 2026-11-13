"""三维各向同性弹性波正演/伴随 Numba 核（实验性）。

速度—应力一阶方程，交错网格：vx(i+1/2,j,k)、vy(i,j+1/2,k)、
vz(i,j,k+1/2)；正应力在中心，剪应力在棱边。海绵吸收层。
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def inject_source_stress3d(sxx, syy, szz, si, sj, sk, sw, amp, dt):
    for m in range(si.size):
        v = dt * amp * sw[m]
        i, j, k = si[m], sj[m], sk[m]
        sxx[i, j, k] += v
        syy[i, j, k] += v
        szz[i, j, k] += v


@njit(cache=True, fastmath=True)
def inject_source_force3d(vx, vy, vz, rho_x, rho_y, rho_z, si, sj, sk, sw, amp, dt):
    for m in range(si.size):
        v = dt * amp * sw[m]
        i, j, k = si[m], sj[m], sk[m]
        vx[i, j, k] += v / rho_x[i, j, k]
        vy[i, j, k] += v / rho_y[i, j, k]
        vz[i, j, k] += v / rho_z[i, j, k]


@njit(cache=True, fastmath=True)
def update_velocity3d(
    vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
    rho_x, rho_y, rho_z, dt, dx, dy, dz, ax, ay, az,
):
    nx, ny, nz = vx.shape
    idx, idy, idz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                bx = (
                    (sxx[i + 1, j, k] - sxx[i, j, k]) * idx
                    + (sxy[i, j, k] - sxy[i, j - 1, k]) * idy
                    + (sxz[i, j, k] - sxz[i, j, k - 1]) * idz
                )
                by = (
                    (sxy[i, j, k] - sxy[i - 1, j, k]) * idx
                    + (syy[i, j + 1, k] - syy[i, j, k]) * idy
                    + (syz[i, j, k] - syz[i, j, k - 1]) * idz
                )
                bz = (
                    (sxz[i, j, k] - sxz[i - 1, j, k]) * idx
                    + (syz[i, j, k] - syz[i, j - 1, k]) * idy
                    + (szz[i, j, k + 1] - szz[i, j, k]) * idz
                )
                vx[i, j, k] += dt * bx / rho_x[i, j, k]
                vy[i, j, k] += dt * by / rho_y[i, j, k]
                vz[i, j, k] += dt * bz / rho_z[i, j, k]
                ax[i, j, k] = bx / rho_x[i, j, k]
                ay[i, j, k] = by / rho_y[i, j, k]
                az[i, j, k] = bz / rho_z[i, j, k]


@njit(cache=True, fastmath=True)
def update_stress3d(
    vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
    c11, c13, mu_xy, mu_xz, mu_yz, dt, dx, dy, dz,
    exx, eyy, ezz, exy, exz, eyz,
):
    nx, ny, nz = vx.shape
    idx, idy, idz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                ex = (vx[i, j, k] - vx[i - 1, j, k]) * idx
                ey = (vy[i, j, k] - vy[i, j - 1, k]) * idy
                ez = (vz[i, j, k] - vz[i, j, k - 1]) * idz
                sxx[i, j, k] += dt * (c11[i, j, k] * ex + c13[i, j, k] * (ey + ez))
                syy[i, j, k] += dt * (c13[i, j, k] * ex + c11[i, j, k] * ey + c13[i, j, k] * ez)
                szz[i, j, k] += dt * (c13[i, j, k] * ex + c13[i, j, k] * ey + c11[i, j, k] * ez)

                exy_v = (vx[i, j + 1, k] - vx[i, j, k]) * idy + (vy[i + 1, j, k] - vy[i, j, k]) * idx
                exz_v = (vx[i, j, k + 1] - vx[i, j, k]) * idz + (vz[i + 1, j, k] - vz[i, j, k]) * idx
                eyz_v = (vy[i, j, k + 1] - vy[i, j, k]) * idz + (vz[i, j + 1, k] - vz[i, j, k]) * idy
                sxy[i, j, k] += dt * mu_xy[i, j, k] * exy_v
                sxz[i, j, k] += dt * mu_xz[i, j, k] * exz_v
                syz[i, j, k] += dt * mu_yz[i, j, k] * eyz_v

                exx[i, j, k] = ex
                eyy[i, j, k] = ey
                ezz[i, j, k] = ez
                exy[i, j, k] = exy_v
                exz[i, j, k] = exz_v
                eyz[i, j, k] = eyz_v


@njit(cache=True, fastmath=True)
def apply_damping3d(field, damp):
    nx, ny, nz = field.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                field[i, j, k] *= damp[i, j, k]


@njit(cache=True, fastmath=True)
def record_data3d(
    vx, vy, vz,
    ix, jx, kx, wx, iy, jy, ky, wy, iz, jz, kz, wz, out,
):
    nr = out.shape[0]
    for r in range(nr):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for m in range(8):
            sx += wx[m, r] * vx[ix[m, r], jx[m, r], kx[m, r]]
            sy += wy[m, r] * vy[iy[m, r], jy[m, r], ky[m, r]]
            sz += wz[m, r] * vz[iz[m, r], jz[m, r], kz[m, r]]
        out[r, 0] = sx
        out[r, 1] = sy
        out[r, 2] = sz


@njit(cache=True, fastmath=True)
def inject_adjoint_source3d(
    vax, vay, vaz,
    ix, jx, kx, wx, iy, jy, ky, wy, iz, jz, kz, wz, residual,
):
    nr = residual.shape[0]
    for r in range(nr):
        rx, ry, rz = residual[r, 0], residual[r, 1], residual[r, 2]
        for m in range(8):
            vax[ix[m, r], jx[m, r], kx[m, r]] += wx[m, r] * rx
            vay[iy[m, r], jy[m, r], ky[m, r]] += wy[m, r] * ry
            vaz[iz[m, r], jz[m, r], kz[m, r]] += wz[m, r] * rz


@njit(cache=True, fastmath=True)
def adjoint_step3d(
    vax, vay, vaz, saxx, sayy, sazz, saxy, saxz, sayz,
    vax2, vay2, vaz2, saxx2, sayy2, sazz2, saxy2, saxz2, sayz2,
    c11, c13, mu_xy, mu_xz, mu_yz, rho_x, rho_y, rho_z,
    dt, dx, dy, dz,
):
    nx, ny, nz = vax.shape
    idx, idy, idz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    # 边界清零
    for i in range(nx):
        for j in range(ny):
            vax2[i, j, 0] = 0.0; vax2[i, j, nz - 1] = 0.0
            vay2[i, j, 0] = 0.0; vay2[i, j, nz - 1] = 0.0
            vaz2[i, j, 0] = 0.0; vaz2[i, j, nz - 1] = 0.0
            saxx2[i, j, 0] = 0.0; saxx2[i, j, nz - 1] = 0.0
            sayy2[i, j, 0] = 0.0; sayy2[i, j, nz - 1] = 0.0
            sazz2[i, j, 0] = 0.0; sazz2[i, j, nz - 1] = 0.0
            saxy2[i, j, 0] = 0.0; saxy2[i, j, nz - 1] = 0.0
            saxz2[i, j, 0] = 0.0; saxz2[i, j, nz - 1] = 0.0
            sayz2[i, j, 0] = 0.0; sayz2[i, j, nz - 1] = 0.0
    for i in range(nx):
        for k in range(nz):
            vax2[i, 0, k] = 0.0; vax2[i, ny - 1, k] = 0.0
            vay2[i, 0, k] = 0.0; vay2[i, ny - 1, k] = 0.0
            vaz2[i, 0, k] = 0.0; vaz2[i, ny - 1, k] = 0.0
    for j in range(ny):
        for k in range(nz):
            vax2[0, j, k] = 0.0; vax2[nx - 1, j, k] = 0.0
            vay2[0, j, k] = 0.0; vay2[nx - 1, j, k] = 0.0
            vaz2[0, j, k] = 0.0; vaz2[nx - 1, j, k] = 0.0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                pxx = c11[i, j, k] * saxx[i, j, k] + c13[i, j, k] * (sayy[i, j, k] + sazz[i, j, k])
                pyy = c13[i, j, k] * saxx[i, j, k] + c11[i, j, k] * sayy[i, j, k] + c13[i, j, k] * sazz[i, j, k]
                pzz = c13[i, j, k] * (saxx[i, j, k] + sayy[i, j, k]) + c11[i, j, k] * sazz[i, j, k]
                pxx1 = c11[i + 1, j, k] * saxx[i + 1, j, k] + c13[i + 1, j, k] * (sayy[i + 1, j, k] + sazz[i + 1, j, k])
                pyy1 = c13[i, j + 1, k] * saxx[i, j + 1, k] + c11[i, j + 1, k] * sayy[i, j + 1, k] + c13[i, j + 1, k] * sazz[i, j + 1, k]
                pzz1 = c13[i, j, k + 1] * (saxx[i, j, k + 1] + sayy[i, j, k + 1]) + c11[i, j, k + 1] * sazz[i, j, k + 1]
                pxy = mu_xy[i, j, k] * saxy[i, j, k]
                pxy_ym = mu_xy[i, j - 1, k] * saxy[i, j - 1, k]
                pxy_xm = mu_xy[i - 1, j, k] * saxy[i - 1, j, k]
                pxz = mu_xz[i, j, k] * saxz[i, j, k]
                pxz_zm = mu_xz[i, j, k - 1] * saxz[i, j, k - 1]
                pxz_xm = mu_xz[i - 1, j, k] * saxz[i - 1, j, k]
                pyz = mu_yz[i, j, k] * sayz[i, j, k]
                pyz_zm = mu_yz[i, j, k - 1] * sayz[i, j, k - 1]
                pyz_ym = mu_yz[i, j - 1, k] * sayz[i, j - 1, k]

                vax2[i, j, k] = vax[i, j, k] - dt * (
                    (pxx1 - pxx) * idx + (pxy - pxy_ym) * idy + (pxz - pxz_zm) * idz
                )
                vay2[i, j, k] = vay[i, j, k] - dt * (
                    (pxy - pxy_xm) * idx + (pyy1 - pyy) * idy + (pyz - pyz_zm) * idz
                )
                vaz2[i, j, k] = vaz[i, j, k] - dt * (
                    (pxz - pxz_xm) * idx + (pyz - pyz_ym) * idy + (pzz1 - pzz) * idz
                )

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                vxr = vax2[i, j, k] / rho_x[i, j, k]
                vyr = vay2[i, j, k] / rho_y[i, j, k]
                vzr = vaz2[i, j, k] / rho_z[i, j, k]
                vxr_xm = vax2[i - 1, j, k] / rho_x[i - 1, j, k]
                vyr_ym = vay2[i, j - 1, k] / rho_y[i, j - 1, k]
                vzr_zm = vaz2[i, j, k - 1] / rho_z[i, j, k - 1]
                div = (vxr - vxr_xm) * idx + (vyr - vyr_ym) * idy + (vzr - vzr_zm) * idz
                saxx2[i, j, k] = saxx[i, j, k] - dt * div
                sayy2[i, j, k] = sayy[i, j, k] - dt * div
                sazz2[i, j, k] = sazz[i, j, k] - dt * div

                vxr_yp = vax2[i, j + 1, k] / rho_x[i, j + 1, k]
                vyr_xp = vay2[i + 1, j, k] / rho_y[i + 1, j, k]
                vxr_zp = vax2[i, j, k + 1] / rho_x[i, j, k + 1]
                vzr_xp = vaz2[i + 1, j, k] / rho_z[i + 1, j, k]
                vyr_zp = vay2[i, j, k + 1] / rho_y[i, j, k + 1]
                vzr_yp = vaz2[i, j + 1, k] / rho_z[i, j + 1, k]
                saxy2[i, j, k] = saxy[i, j, k] - dt * ((vxr_yp - vxr) * idy + (vyr_xp - vyr) * idx)
                saxz2[i, j, k] = saxz[i, j, k] - dt * ((vxr_zp - vxr) * idz + (vzr_xp - vzr) * idx)
                sayz2[i, j, k] = sayz[i, j, k] - dt * ((vyr_zp - vyr) * idz + (vzr_yp - vzr) * idy)


@njit(cache=True, fastmath=True)
def adjoint_stiffness_gradient3d(
    saxx, sayy, sazz, saxy, saxz, sayz,
    exx, eyy, ezz, exy, exz, eyz, dt,
    g_lambda, g_mu,
):
    nx, ny, nz = saxx.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                g_lambda[i, j, k] += dt * (saxx[i, j, k] + sayy[i, j, k] + sazz[i, j, k]) * (
                    exx[i, j, k] + eyy[i, j, k] + ezz[i, j, k]
                )
                g_mu[i, j, k] += dt * (
                    2.0 * (saxx[i, j, k] * exx[i, j, k] + sayy[i, j, k] * eyy[i, j, k] + sazz[i, j, k] * ezz[i, j, k])
                    + saxy[i, j, k] * exy[i, j, k]
                    + saxz[i, j, k] * exz[i, j, k]
                    + sayz[i, j, k] * eyz[i, j, k]
                )


@njit(cache=True, fastmath=True)
def adjoint_density_gradient3d(
    vax, vay, vaz, ax, ay, az, rho_x, rho_y, rho_z, dt, gx, gy, gz,
):
    nx, ny, nz = vax.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                gx[i, j, k] += -dt * vax[i, j, k] * ax[i, j, k] / rho_x[i, j, k]
                gy[i, j, k] += -dt * vay[i, j, k] * ay[i, j, k] / rho_y[i, j, k]
                gz[i, j, k] += -dt * vaz[i, j, k] * az[i, j, k] / rho_z[i, j, k]
