"""三维各向同性弹性波正演/伴随算子（P0 基础实现）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import kernels3d
from .model3d import ElasticModel3D, Grid3D, Survey3D, TrilinearWeights, _trilinear


@dataclass
class Storage3D:
    exx: np.ndarray
    eyy: np.ndarray
    ezz: np.ndarray
    exy: np.ndarray
    exz: np.ndarray
    eyz: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    stride: int

    def fields_at(self, n: int):
        k = min(n // self.stride, self.exx.shape[0] - 1)
        return (
            self.exx[k], self.eyy[k], self.ezz[k],
            self.exy[k], self.exz[k], self.eyz[k],
            self.ax[k], self.ay[k], self.az[k],
        )


@dataclass
class Gradient3D:
    misfit: float
    obs_energy: float
    g_vp: np.ndarray
    g_vs: np.ndarray
    g_rho: np.ndarray
    synthetic: np.ndarray
    residuals: np.ndarray


class Elastic3DOperator:
    def __init__(
        self,
        model: ElasticModel3D,
        survey: Survey3D,
        grid: Grid3D | None = None,
        *,
        dtype: str = "float64",
        sponge_strength: float = 3.0,
        storage_mb: float = 512.0,
        verbose: bool = False,
    ) -> None:
        if grid is None:
            if model.dx is None or model.dy is None or model.dz is None:
                raise ValueError("模型缺少 dx/dy/dz")
            grid = Grid3D(model.shape[0], model.shape[1], model.shape[2],
                          model.dx, model.dy, model.dz)
        if grid.shape != model.shape:
            raise ValueError("grid 与模型形状不一致")
        self.grid = grid
        self.model = model
        self.survey = survey
        self.dtype = np.dtype(dtype)
        self.sponge_strength = float(sponge_strength)
        self.storage_mb = float(storage_mb)
        self.verbose = verbose
        self.pad = grid.pad
        self.NX = grid.nx + 2 * self.pad
        self.NY = grid.ny + 2 * self.pad
        self.NZ = grid.nz + 2 * self.pad
        self.dx, self.dy, self.dz = grid.dx, grid.dy, grid.dz
        self.dt = survey.dt
        self._damping_vmax: float | None = None
        self._setup_geometry()
        self.set_model(model)

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        return np.pad(np.asarray(arr, dtype=np.float64), self.pad, mode="edge")

    def _setup_geometry(self) -> None:
        self._src = _trilinear(self.grid, self.survey.shots, (0.0, 0.0, 0.0))
        rvx = _trilinear(self.grid, self.survey.receivers, (-0.5, 0.0, 0.0))
        rvy = _trilinear(self.grid, self.survey.receivers, (0.0, -0.5, 0.0))
        rvz = _trilinear(self.grid, self.survey.receivers, (0.0, 0.0, -0.5))
        self._rec = (rvx, rvy, rvz)
        self._src_i = [self._src.i[:, s].copy() for s in range(self.survey.n_shots)]
        self._src_j = [self._src.j[:, s].copy() for s in range(self.survey.n_shots)]
        self._src_k = [self._src.k[:, s].copy() for s in range(self.survey.n_shots)]
        self._src_w = [self._src.w[:, s].copy() for s in range(self.survey.n_shots)]

    def set_model(self, model: ElasticModel3D) -> None:
        if model.shape != self.grid.shape:
            raise ValueError("模型形状与网格不一致")
        self.model = model
        vp = self._pad(model.vp)
        vs = self._pad(model.vs)
        rho = self._pad(model.rho)
        c11p, c13p, c44p = model.stiffness()
        self.c11 = self._pad(c11p)
        self.c13 = self._pad(c13p)
        self.c44 = self._pad(c44p)
        self.rho_p = rho

        def avg_x(a):
            o = np.empty_like(a); o[:-1] = 0.5 * (a[:-1] + a[1:]); o[-1] = o[-2]; return o

        def avg_y(a):
            o = np.empty_like(a); o[:, :-1] = 0.5 * (a[:, :-1] + a[:, 1:]); o[:, -1] = o[:, -2]; return o

        def avg_z(a):
            o = np.empty_like(a); o[:, :, :-1] = 0.5 * (a[:, :, :-1] + a[:, :, 1:]); o[:, :, -1] = o[:, :, -2]; return o

        self.rho_x = avg_x(rho)
        self.rho_y = avg_y(rho)
        self.rho_z = avg_z(rho)
        mu = self.c44
        self.mu_xy = 0.25 * (mu + np.roll(mu, -1, axis=0) + np.roll(mu, -1, axis=1) + np.roll(np.roll(mu, -1, axis=0), -1, axis=1))
        self.mu_xz = 0.25 * (mu + np.roll(mu, -1, axis=0) + np.roll(mu, -1, axis=2) + np.roll(np.roll(mu, -1, axis=0), -1, axis=2))
        self.mu_yz = 0.25 * (mu + np.roll(mu, -1, axis=1) + np.roll(mu, -1, axis=2) + np.roll(np.roll(mu, -1, axis=1), -1, axis=2))
        # 边界复制，避免 roll 造成周期
        self.mu_xy[-1, :, :] = self.mu_xy[-2, :, :]; self.mu_xy[:, -1, :] = self.mu_xy[:, -2, :]
        self.mu_xz[-1, :, :] = self.mu_xz[-2, :, :]; self.mu_xz[:, :, -1] = self.mu_xz[:, :, -2]
        self.mu_yz[:, -1, :] = self.mu_yz[:, -2, :]; self.mu_yz[:, :, -1] = self.mu_yz[:, :, -2]
        self.vmax = model.max_velocity()
        if self._damping_vmax is None:
            self._damping_vmax = self.vmax
        limit = self.grid.cfl_dt(self.vmax, safety=0.4)
        if self.dt > limit * 1.02:
            raise ValueError(f"三维 dt={self.dt} 超过稳定上限 {limit}")
        self._build_damping()

    def _build_damping(self) -> None:
        g = self.grid
        vref = self._damping_vmax if self._damping_vmax is not None else self.vmax
        dmax = self.sponge_strength * vref / max(g.sponge * min(self.dx, self.dy, self.dz), 1e-9)
        ix = np.arange(self.NX) - self.pad
        jy = np.arange(self.NY) - self.pad
        kz = np.arange(self.NZ) - self.pad
        px = np.zeros(self.NX); py = np.zeros(self.NY); pz = np.zeros(self.NZ)
        for arr, n, prof in ((ix, g.nx, px), (jy, g.ny, py), (kz, g.nz, pz)):
            dist = np.minimum(arr, n - 1 - arr)
            inside = (dist >= 0) & (dist < g.sponge)
            prof[inside] = dmax * ((g.sponge - dist[inside]) / g.sponge) ** 2
            prof[dist < 0] = dmax
        self._damp = np.exp(-(px[:, None, None] + py[None, :, None] + pz[None, None, :]) * self.dt)

    def _make_storage(self, nt: int) -> Storage3D:
        nbytes = 9 * self.NX * self.NY * self.NZ * self.dtype.itemsize
        max_steps = max(2, int(self.storage_mb * 1e6 / max(nbytes, 1)))
        stride = max(1, int(np.ceil(nt / max_steps)))
        nstore = int(np.ceil(nt / stride))
        shape = (nstore, self.NX, self.NY, self.NZ)
        arrs = [np.zeros(shape, dtype=self.dtype) for _ in range(9)]
        return Storage3D(*arrs, stride)

    def forward_shot(self, ishot: int, *, store: bool = False) -> tuple[np.ndarray, Storage3D | None]:
        nt = self.survey.nt
        dtype = self.dtype
        shape = (self.NX, self.NY, self.NZ)
        vx = np.zeros(shape, dtype); vy = np.zeros(shape, dtype); vz = np.zeros(shape, dtype)
        sxx = np.zeros(shape, dtype); syy = np.zeros(shape, dtype); szz = np.zeros(shape, dtype)
        sxy = np.zeros(shape, dtype); sxz = np.zeros(shape, dtype); syz = np.zeros(shape, dtype)
        ax = np.zeros(shape, dtype); ay = np.zeros(shape, dtype); az = np.zeros(shape, dtype)
        exx = np.zeros(shape, dtype); eyy = np.zeros(shape, dtype); ezz = np.zeros(shape, dtype)
        exy = np.zeros(shape, dtype); exz = np.zeros(shape, dtype); eyz = np.zeros(shape, dtype)
        storage = self._make_storage(nt) if store else None
        data = np.zeros((nt, self.survey.n_receivers, 3), dtype=np.float64)
        for n in range(nt):
            amp = self.survey.amplitude * self.survey.wavelet[n]
            kernels3d.inject_source_stress3d(
                sxx, syy, szz, self._src_i[ishot], self._src_j[ishot],
                self._src_k[ishot], self._src_w[ishot], amp, self.dt,
            )
            kernels3d.update_velocity3d(
                vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                self.rho_x, self.rho_y, self.rho_z, self.dt,
                self.dx, self.dy, self.dz, ax, ay, az,
            )
            kernels3d.update_stress3d(
                vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                self.c11, self.c13, self.mu_xy, self.mu_xz, self.mu_yz,
                self.dt, self.dx, self.dy, self.dz,
                exx, eyy, ezz, exy, exz, eyz,
            )
            if store and n % storage.stride == 0:
                k = n // storage.stride
                storage.exx[k] = exx; storage.eyy[k] = eyy; storage.ezz[k] = ezz
                storage.exy[k] = exy; storage.exz[k] = exz; storage.eyz[k] = eyz
                storage.ax[k] = ax; storage.ay[k] = ay; storage.az[k] = az
            if store and n % storage.stride != 0:
                # still need ax/ay/az for gradient; stored only at stride steps
                pass
            kernels3d.record_data3d(
                vx, vy, vz,
                *[self._rec[0].i, self._rec[0].j, self._rec[0].k, self._rec[0].w],
                *[self._rec[1].i, self._rec[1].j, self._rec[1].k, self._rec[1].w],
                *[self._rec[2].i, self._rec[2].j, self._rec[2].k, self._rec[2].w],
                data[n],
            )
            for f in (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz):
                kernels3d.apply_damping3d(f, self._damp)
        return data, storage

    def forward_all(self, model: ElasticModel3D | None = None) -> np.ndarray:
        if model is not None:
            self.set_model(model)
        out = np.empty((self.survey.n_shots, self.survey.nt, self.survey.n_receivers, 3), dtype=np.float64)
        for s in range(self.survey.n_shots):
            out[s] = self.forward_shot(s)[0]
        return out

    def misfit(self, model: ElasticModel3D, obs: np.ndarray) -> float:
        self.set_model(model)
        e = float(np.sum(obs * obs)) or 1.0
        total = 0.0
        for s in range(self.survey.n_shots):
            syn = self.forward_shot(s)[0]
            total += 0.5 * float(np.sum((syn - obs[s]) ** 2))
        return total / e

    def compute_gradient(
        self,
        model: ElasticModel3D,
        obs: np.ndarray,
        *,
        allow_unvalidated: bool = False,
    ) -> Gradient3D:
        """三维伴随梯度（实验性，默认禁用）。

        当前实现还未通过三维离散内积/有限差分验收；为避免误用，默认抛出
        ``NotImplementedError``。三维正演已可用；正式三维 FWI 需先修复
        伴随核并通过 ``tests/test_fd3d.py`` 的梯度检验。
        """
        if not allow_unvalidated:
            raise NotImplementedError(
                "三维伴随梯度尚未通过数值验收；"
                "如需实验，请显式设置 allow_unvalidated=True"
            )
        self.set_model(model)
        obs = np.asarray(obs, dtype=np.float64)
        e = float(np.sum(obs * obs)) or 1.0
        shape = self.grid.shape
        g_vp = np.zeros(shape); g_vs = np.zeros(shape); g_rho = np.zeros(shape)
        g_lambda = np.zeros((self.NX, self.NY, self.NZ))
        g_mu = np.zeros((self.NX, self.NY, self.NZ))
        gx = np.zeros((self.NX, self.NY, self.NZ))
        gy = np.zeros((self.NX, self.NY, self.NZ))
        gz = np.zeros((self.NX, self.NY, self.NZ))
        syn_all = np.empty_like(obs)
        res_all = np.empty_like(obs)
        total = 0.0
        for s in range(self.survey.n_shots):
            syn, storage = self.forward_shot(s, store=True)
            assert storage is not None
            r = syn - obs[s]
            syn_all[s] = syn; res_all[s] = r
            total += 0.5 * float(np.sum(r * r))
            nt = self.survey.nt
            shape = (self.NX, self.NY, self.NZ)
            vax = np.zeros(shape); vay = np.zeros(shape); vaz = np.zeros(shape)
            saxx = np.zeros(shape); sayy = np.zeros(shape); sazz = np.zeros(shape)
            saxy = np.zeros(shape); saxz = np.zeros(shape); sayz = np.zeros(shape)
            b = [np.zeros(shape) for _ in range(9)]
            for n in range(nt - 1, -1, -1):
                for f in (vax, vay, vaz, saxx, sayy, sazz, saxy, saxz, sayz):
                    kernels3d.apply_damping3d(f, self._damp)
                exx, eyy, ezz, exy, exz, eyz, ax, ay, az = storage.fields_at(n)
                kernels3d.inject_adjoint_source3d(
                    vax, vay, vaz,
                    *[self._rec[0].i, self._rec[0].j, self._rec[0].k, self._rec[0].w],
                    *[self._rec[1].i, self._rec[1].j, self._rec[1].k, self._rec[1].w],
                    *[self._rec[2].i, self._rec[2].j, self._rec[2].k, self._rec[2].w],
                    r[n],
                )
                kernels3d.adjoint_stiffness_gradient3d(
                    saxx, sayy, sazz, saxy, saxz, sayz,
                    exx, eyy, ezz, exy, exz, eyz, self.dt, g_lambda, g_mu,
                )
                kernels3d.adjoint_step3d(
                    vax, vay, vaz, saxx, sayy, sazz, saxy, saxz, sayz,
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8],
                    self.c11, self.c13, self.mu_xy, self.mu_xz, self.mu_yz,
                    self.rho_x, self.rho_y, self.rho_z, self.dt, self.dx, self.dy, self.dz,
                )
                kernels3d.adjoint_density_gradient3d(
                    b[0], b[1], b[2], ax, ay, az,
                    self.rho_x, self.rho_y, self.rho_z, self.dt, gx, gy, gz,
                )
                vax, vay, vaz = b[0], b[1], b[2]
                saxx, sayy, sazz = b[3], b[4], b[5]
                saxy, saxz, sayz = b[6], b[7], b[8]
                b = [np.zeros(shape) for _ in range(9)]
        total /= e
        for arr in (g_lambda, g_mu, gx, gy, gz):
            arr /= e
        sl = (slice(self.pad, self.pad + self.grid.nx),
              slice(self.pad, self.pad + self.grid.ny),
              slice(self.pad, self.pad + self.grid.nz))
        g_l = g_lambda[sl]; g_m = g_mu[sl]
        rho = model.rho; vp = model.vp; vs = model.vs
        g_vp = 2.0 * rho * vp * g_l
        g_vs = -4.0 * rho * vs * g_l + 2.0 * rho * vs * g_m
        # 密度直接项各方向 1/2 分配到中心（边界近似）
        g_rho = 0.5 * (gx[sl] + np.roll(gx, 1, axis=0)[sl])
        g_rho += 0.5 * (gy[sl] + np.roll(gy, 1, axis=1)[sl])
        g_rho += 0.5 * (gz[sl] + np.roll(gz, 1, axis=2)[sl])
        g_rho += (vp ** 2 - 2 * vs ** 2) * g_l + vs ** 2 * g_m
        return Gradient3D(total, e, g_vp, g_vs, g_rho, syn_all, res_all)
