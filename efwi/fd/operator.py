"""二维弹性波正演/伴随算子（交错网格 + 吸收层）。

对外接口::

    op = Elastic2DOperator(model, survey, grid)
    data = op.forward_all(model)                  # 合成炮集
    res  = op.compute_gradient(model, obs)        # 目标函数 + 原生梯度 + 伪 Hessian

原生梯度分别对应刚度分量 ``C11, C13, C33, C44`` 与密度 ``rho``，
再由 :mod:`efwi.gradient` 链式转换到 ``(vp, vs, rho[, epsilon, delta])``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import GridConfig
from ..geometry import BilinearWeights, Survey
from ..model import ElasticModel
from . import kernels


@dataclass
class ForwardStorage:
    """按时间抽稀保存的正演敏感场。

    梯度互相关需要 ``exx/ezz/exz``（应变率）和 ``ax/az``（加速度）。
    当内存不足时按 ``stride`` 抽稀，梯度仍可在相邻存储时刻近似使用。
    """

    exx: np.ndarray
    ezz: np.ndarray
    exz: np.ndarray
    ax: np.ndarray
    az: np.ndarray
    stride: int

    @property
    def nstore(self) -> int:
        return int(self.exx.shape[0])

    def fields_at(self, n: int) -> tuple[np.ndarray, ...]:
        k = min(n // self.stride, self.nstore - 1)
        return (
            self.exx[k],
            self.ezz[k],
            self.exz[k],
            self.ax[k],
            self.az[k],
        )


@dataclass
class ShotForward:
    data: np.ndarray
    storage: ForwardStorage | None = None
    hessian: dict[str, np.ndarray] | None = None


@dataclass
class ShotGradient:
    """单炮的原始（未按观测能量归一化）梯度、残差与目标函数。"""

    shot: int
    misfit_raw: float
    obs_energy: float
    g: dict[str, np.ndarray]  # c11/c13/c33/c44/rho（物理网格）
    h: dict[str, np.ndarray]  # 同键的伪 Hessian 对角
    synthetic: np.ndarray     # (nt, nr, 2)
    residuals: np.ndarray     # (nt, nr, 2)
    storage_stride: int


@dataclass
class GradientResult:
    misfit: float
    obs_energy: float
    g: dict[str, np.ndarray]
    h: dict[str, np.ndarray]
    synthetic: np.ndarray
    residuals: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    shot_gradients: list[ShotGradient] | None = None

    def gradient_norm(self) -> float:
        return float(np.sqrt(sum(np.sum(v * v) for v in self.g.values())))


class Elastic2DOperator:
    def __init__(
        self,
        model: ElasticModel,
        survey: Survey,
        grid: GridConfig | None = None,
        *,
        dtype: str = "float64",
        sponge_strength: float = 3.0,
        storage_mb: float = 512.0,
        strict_dt: bool = True,
        free_surface: bool = False,
        verbose: bool = False,
    ) -> None:
        if grid is None:
            if model.dx is None or model.dz is None:
                raise ValueError("未提供 grid，且模型没有 dx/dz")
            grid = GridConfig(model.nx, model.nz, model.dx, model.dz)
        if (grid.nx, grid.nz) != model.shape:
            raise ValueError("grid 与模型形状不一致")
        survey.validate_against_grid(grid, allow_sponge=True)
        self.grid = grid
        self.survey = survey
        self.dtype = np.dtype(dtype)
        self.sponge_strength = float(sponge_strength)
        self.storage_mb = float(storage_mb)
        self.strict_dt = bool(strict_dt)
        self.free_surface = bool(free_surface)
        self.verbose = bool(verbose)

        self.pad = grid.pad
        self.NX = grid.nx + 2 * self.pad
        self.NZ = grid.nz + 2 * self.pad
        self.dx = grid.dx
        self.dz = grid.dz
        self.dt = survey.dt
        self._damping_vmax: float | None = None

        self._setup_geometry()
        self.set_model(model)

    # ------------------------------------------------------------------
    # 几何 / 模型准备
    # ------------------------------------------------------------------
    def _setup_geometry(self) -> None:
        sw = self.survey.source_weights(self.grid)
        self._src_weights = sw
        self._src_i: dict[str, list[np.ndarray]] = {}
        self._src_j: dict[str, list[np.ndarray]] = {}
        self._src_w: dict[str, list[np.ndarray]] = {}
        for key, w in sw.items():
            self._src_i[key] = [w.i[:, s].copy() for s in range(self.survey.n_shots)]
            self._src_j[key] = [w.j[:, s].copy() for s in range(self.survey.n_shots)]
            self._src_w[key] = [w.w[:, s].copy() for s in range(self.survey.n_shots)]

        rw = self.survey.receiver_weights(self.grid)
        self._rec = rw
        self._rix = rw["vx"].i
        self._rjx = rw["vx"].j
        self._rwx = rw["vx"].w
        self._riz = rw["vz"].i
        self._rjz = rw["vz"].j
        self._rwz = rw["vz"].w

    def set_survey(self, survey: Survey) -> None:
        """切换观测系统（多尺度频带使用带限子波时复用同一个算子）。"""
        survey.validate_against_grid(self.grid, allow_sponge=True)
        if survey.nt != self.survey.nt or survey.n_receivers != self.survey.n_receivers:
            raise ValueError("新观测系统的 nt/nr 必须与原观测系统一致")
        self.survey = survey
        self.dt = survey.dt
        self._setup_geometry()
        self.set_model(self.model)

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        return np.pad(np.asarray(arr, dtype=np.float64), self.pad, mode="edge")

    def set_model(self, model: ElasticModel) -> None:
        if model.shape != self.grid.shape:
            raise ValueError("模型形状与网格不一致")
        model.validate()
        self.model = model
        self.vp_p = self._pad(model.vp)
        self.vs_p = self._pad(model.vs)
        self.rho_p = self._pad(model.rho)
        c11, c13, c33, c44 = model.stiffness()
        self.c11 = self._pad(c11)
        self.c13 = self._pad(c13)
        self.c33 = self._pad(c33)
        self.c44 = self._pad(c44)

        rho = self.rho_p
        rho_x = np.empty_like(rho)
        rho_x[:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
        rho_x[-1, :] = rho_x[-2, :]
        rho_z = np.empty_like(rho)
        rho_z[:, :-1] = 0.5 * (rho[:, :-1] + rho[:, 1:])
        rho_z[:, -1] = rho_z[:, -2]
        self.rho_x = rho_x
        self.rho_z = rho_z

        c44c = np.empty_like(self.c44)
        c44c[:-1, :-1] = 0.25 * (
            self.c44[:-1, :-1]
            + self.c44[1:, :-1]
            + self.c44[:-1, 1:]
            + self.c44[1:, 1:]
        )
        c44c[-1, :] = c44c[-2, :]
        c44c[:, -1] = c44c[:, -2]
        self.c44c = c44c

        self.vmax = max(model.max_velocity(), 1e-6)
        if self._damping_vmax is None:
            # 吸收层强度固定为首次建模时的速度尺度，避免反演中 vmax 微变
            # 导致阻尼剖面变化（这会使有限差分梯度混入非模型扰动）。
            self._damping_vmax = self.vmax
        limit = self.grid.cfl_dt(self.vmax, safety=0.5)
        if self.dt > limit * 1.02:
            msg = (
                f"时间步 dt={self.dt:.6g}s 超过当前模型稳定上限 {limit:.6g}s "
                f"(vmax={self.vmax:.0f} m/s)"
            )
            if self.strict_dt and self.dt > limit * 1.10:
                raise ValueError(msg + "；请减小 dt 或降低 sponge/模型速度")
            if self.verbose:
                print("警告: " + msg)
        self._build_damping()

    def _build_damping(self) -> None:
        g = self.grid
        vref = self._damping_vmax if self._damping_vmax is not None else self.vmax
        dmax = self.sponge_strength * vref / max(g.sponge * min(self.dx, self.dz), 1e-9)
        ix = np.arange(self.NX) - self.pad
        jz = np.arange(self.NZ) - self.pad
        dxp = np.minimum(ix, g.nx - 1 - ix)
        dzp = np.minimum(jz, g.nz - 1 - jz)
        px = np.zeros(self.NX, dtype=np.float64)
        pz = np.zeros(self.NZ, dtype=np.float64)
        for arr, dist, prof in ((ix, dxp, px), (jz, dzp, pz)):
            inside = (dist >= 0) & (dist < g.sponge)
            prof[inside] = dmax * ((g.sponge - dist[inside]) / g.sponge) ** 2
            prof[dist < 0] = dmax
        if self.free_surface:
            # 自由表面上方不设吸收层，避免把地表反射吃掉
            pz[: self.pad + g.sponge] = 0.0
        self._damp = np.exp(-(px[:, None] + pz[None, :]) * self.dt)

    def _apply_free_surface(self, vx, vz, sxx, szz, sxz) -> None:
        """平自由表面：σzz=σxz=0，地表以上场量为零。

        真正逐列起伏自由表面需要改变每列差分模板；当前先提供平自由表面
        + 高程静校正/浮动 datum 的组合（见 efwi/topography.py）。
        """
        j0 = self.pad
        vx[:, :j0] = 0.0
        vz[:, :j0] = 0.0
        sxx[:, :j0] = 0.0
        szz[:, :j0] = 0.0
        sxz[:, :j0] = 0.0
        szz[:, j0] = 0.0
        sxz[:, j0] = 0.0

    # ------------------------------------------------------------------
    # 存储
    # ------------------------------------------------------------------
    def _make_storage(self, nt: int) -> ForwardStorage:
        nbytes_step = 5 * self.NX * self.NZ * self.dtype.itemsize
        max_steps = max(2, int(self.storage_mb * 1e6 / max(nbytes_step, 1)))
        stride = max(1, int(np.ceil(nt / max_steps)))
        nstore = int(np.ceil(nt / stride))
        shape = (nstore, self.NX, self.NZ)
        dtype = self.dtype
        return ForwardStorage(
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            stride,
        )

    # ------------------------------------------------------------------
    # 正演
    # ------------------------------------------------------------------
    def forward_shot(
        self,
        ishot: int,
        *,
        store: bool = False,
        hessian: bool = False,
        record: bool = True,
    ) -> ShotForward:
        nt = self.survey.nt
        dtype = self.dtype
        vx = np.zeros((self.NX, self.NZ), dtype=dtype)
        vz = np.zeros((self.NX, self.NZ), dtype=dtype)
        sxx = np.zeros((self.NX, self.NZ), dtype=dtype)
        szz = np.zeros((self.NX, self.NZ), dtype=dtype)
        sxz = np.zeros((self.NX, self.NZ), dtype=dtype)

        scratch_ax = np.zeros((self.NX, self.NZ), dtype=dtype)
        scratch_az = np.zeros((self.NX, self.NZ), dtype=dtype)
        scratch_exx = np.zeros((self.NX, self.NZ), dtype=dtype)
        scratch_ezz = np.zeros((self.NX, self.NZ), dtype=dtype)
        scratch_exz = np.zeros((self.NX, self.NZ), dtype=dtype)

        if hessian:
            h = {
                "c11": np.zeros((self.NX, self.NZ), dtype=np.float64),
                "c13": np.zeros((self.NX, self.NZ), dtype=np.float64),
                "c33": np.zeros((self.NX, self.NZ), dtype=np.float64),
                "c44": np.zeros((self.NX, self.NZ), dtype=np.float64),
                "rho": np.zeros((self.NX, self.NZ), dtype=np.float64),
                "theta": np.zeros((self.NX, self.NZ), dtype=np.float64),
            }
        else:
            h = None
        dummy_h = np.zeros((1, 1), dtype=np.float64)
        hc11 = h["c11"] if h is not None else dummy_h
        hc13 = h["c13"] if h is not None else dummy_h
        hc33 = h["c33"] if h is not None else dummy_h
        hc44 = h["c44"] if h is not None else dummy_h
        hrho = h["rho"] if h is not None else dummy_h
        htheta = h["theta"] if h is not None else dummy_h

        storage = self._make_storage(nt) if store else None
        data = np.zeros((nt, self.survey.n_receivers, 2), dtype=np.float64)

        src_kind = self.survey.source_kind
        if src_kind == "explosive":
            si = self._src_i["stress"][ishot]
            sj = self._src_j["stress"][ishot]
            sw = self._src_w["stress"][ishot]
        else:
            six = self._src_i["vx"][ishot]
            sjx = self._src_j["vx"][ishot]
            swx = self._src_w["vx"][ishot]
            siz = self._src_i["vz"][ishot]
            sjz = self._src_j["vz"][ishot]
            swz = self._src_w["vz"][ishot]

        for n in range(nt):
            amp = self.survey.amplitude * self.survey.wavelet[n]
            if src_kind == "explosive":
                kernels.inject_source_stress(sxx, szz, si, sj, sw, amp, self.dt)
            else:
                kernels.inject_source_force_x(vx, self.rho_x, six, sjx, swx, amp, self.dt)
                kernels.inject_source_force_z(vz, self.rho_z, siz, sjz, swz, amp, self.dt)

            if store and n % storage.stride == 0:
                k = n // storage.stride
                ax_out, az_out = storage.ax[k], storage.az[k]
                exx_out, ezz_out, exz_out = storage.exx[k], storage.ezz[k], storage.exz[k]
            else:
                ax_out, az_out = scratch_ax, scratch_az
                exx_out, ezz_out, exz_out = scratch_exx, scratch_ezz, scratch_exz

            kernels.update_velocity(
                vx, vz, sxx, szz, sxz, self.rho_x, self.rho_z,
                self.dt, self.dx, self.dz, ax_out, az_out, hrho,
                1 if hessian else 0,
            )
            kernels.update_stress(
                vx, vz, sxx, szz, sxz, self.c11, self.c13, self.c33, self.c44c,
                self.dt, self.dx, self.dz, exx_out, ezz_out, exz_out,
                hc11, hc13, hc33, hc44, 1 if hessian else 0,
            )
            if hessian:
                kernels.accumulate_theta_hessian(exx_out, ezz_out, self.dt, htheta)
            if self.free_surface:
                self._apply_free_surface(vx, vz, sxx, szz, sxz)
            if record:
                kernels.record_data(
                    vx, vz, self._rix, self._rjx, self._rwx,
                    self._riz, self._rjz, self._rwz, data[n],
                )
            for field in (vx, vz, sxx, szz, sxz):
                kernels.apply_damping(field, self._damp)

        return ShotForward(data=data, storage=storage, hessian=h)

    def forward_all(
        self,
        model: ElasticModel | None = None,
        *,
        syn_filter: Any | None = None,
    ) -> np.ndarray:
        if model is not None:
            self.set_model(model)
        out = np.empty(
            (self.survey.n_shots, self.survey.nt, self.survey.n_receivers, 2),
            dtype=np.float64,
        )
        for s in range(self.survey.n_shots):
            data = self.forward_shot(s).data
            if syn_filter is not None:
                data = syn_filter.apply(data, axis=0)
            out[s] = data
        return out

    def misfit(
        self,
        model: ElasticModel,
        obs: np.ndarray,
        *,
        syn_filter: Any | None = None,
    ) -> float:
        self.set_model(model)
        obs = np.asarray(obs, dtype=np.float64)
        energy = float(np.sum(obs * obs))
        if energy <= 0:
            energy = 1.0
        total = 0.0
        for s in range(self.survey.n_shots):
            syn = self.forward_shot(s).data
            if syn_filter is not None:
                syn = syn_filter.apply(syn, axis=0)
            r = syn - obs[s]
            total += 0.5 * float(np.sum(r * r))
        return total / energy

    # ------------------------------------------------------------------
    # 梯度（伴随态）
    # ------------------------------------------------------------------
    def _physical_native(
        self,
        g_pad: dict[str, np.ndarray],
        h_pad: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """把带 padding 的原生梯度/照明场取回物理网格，并处理平均分配。"""
        nx, nz = self.grid.nx, self.grid.nz
        first_i, last_i = self.pad, self.pad + nx - 1
        first_j, last_j = self.pad, self.pad + nz - 1
        ii, jj = np.meshgrid(
            np.arange(1, self.NX - 1), np.arange(1, self.NZ - 1), indexing="ij"
        )

        def fold_center(arr: np.ndarray) -> np.ndarray:
            """把 padding 中与物理边界等价的模型单元梯度折叠回物理网格。"""
            out = np.zeros((nx, nz), dtype=np.float64)
            ci = np.clip(ii, first_i, last_i) - self.pad
            cj = np.clip(jj, first_j, last_j) - self.pad
            np.add.at(out, (ci, cj), arr[ii, jj])
            return out

        g_phys = {
            "c11": fold_center(g_pad["c11"]),
            "c13": fold_center(g_pad["c13"]),
            "c33": fold_center(g_pad["c33"]),
        }
        # 边界处 np.pad(mode="edge") 会让平均算子出现重复节点；下面用
        # “角点/速度点 -> 中心”的显式反向平均，并把越界索引 clip 到物理边界，
        # 这样边界单元的梯度也与离散正演严格一致。
        # C44 角点：c44c=0.25*(c44[i,j]+c44[i+1,j]+c44[i,j+1]+c44[i+1,j+1])
        gc44 = g_pad["c44"]
        c44_center = np.zeros((nx, nz), dtype=np.float64)
        vals = 0.25 * gc44[ii, jj]
        for di in (0, 1):
            ci = np.clip(ii + di, first_i, last_i) - self.pad
            for dj in (0, 1):
                cj = np.clip(jj + dj, first_j, last_j) - self.pad
                np.add.at(c44_center, (ci, cj), vals)
        g_phys["c44"] = c44_center

        # rho_x/rho_z：速度点上的密度是两个中心密度的平均，各分 1/2
        gx = g_pad["rho_x"]
        gz = g_pad["rho_z"]
        rho_center = np.zeros((nx, nz), dtype=np.float64)
        vals = 0.5 * gx[ii, jj]
        for di in (0, 1):
            ci = np.clip(ii + di, first_i, last_i) - self.pad
            cj = np.clip(jj, first_j, last_j) - self.pad
            np.add.at(rho_center, (ci, cj), vals)
        vals = 0.5 * gz[ii, jj]
        for dj in (0, 1):
            ci = np.clip(ii, first_i, last_i) - self.pad
            cj = np.clip(jj + dj, first_j, last_j) - self.pad
            np.add.at(rho_center, (ci, cj), vals)
        g_phys["rho"] = rho_center
        # 照明场同样按边界等价节点折叠，保持预条件在边界连续
        h_phys = {k: fold_center(v) for k, v in h_pad.items()}
        return g_phys, h_phys

    def _shot_gradient(
        self,
        ishot: int,
        obs_shot: np.ndarray,
        *,
        syn_filter: Any | None = None,
    ) -> ShotGradient:
        """单炮伴随循环：正演 -> 残差 -> 伴随源 -> 反向传播 -> 原始梯度。

        调用前必须已经 ``set_model``；本方法不做观测能量归一化。
        """
        obs_shot = np.asarray(obs_shot, dtype=np.float64)
        expected = (self.survey.nt, self.survey.n_receivers, 2)
        if obs_shot.shape != expected:
            raise ValueError(f"单炮 obs 形状应为 {expected}，实际 {obs_shot.shape}")
        shot = self.forward_shot(ishot, store=True, hessian=True)
        storage = shot.storage
        assert storage is not None and shot.hessian is not None
        syn = shot.data
        if syn_filter is not None:
            syn = syn_filter.apply(syn, axis=0)
        residual = syn - obs_shot
        misfit_raw = 0.5 * float(np.sum(residual * residual))

        g = {
            k: np.zeros((self.NX, self.NZ), dtype=np.float64)
            for k in ("c11", "c13", "c33", "c44", "rho_x", "rho_z")
        }
        nt = self.survey.nt
        field_dtype = self.dtype
        vax = np.zeros((self.NX, self.NZ), dtype=field_dtype)
        vaz = np.zeros((self.NX, self.NZ), dtype=field_dtype)
        saxx = np.zeros((self.NX, self.NZ), dtype=field_dtype)
        sazz = np.zeros((self.NX, self.NZ), dtype=field_dtype)
        saxz = np.zeros((self.NX, self.NZ), dtype=field_dtype)
        vax2 = np.zeros_like(vax)
        vaz2 = np.zeros_like(vaz)
        saxx2 = np.zeros_like(saxx)
        sazz2 = np.zeros_like(sazz)
        saxz2 = np.zeros_like(saxz)

        for n in range(nt - 1, -1, -1):
            # 正向在每步末尾乘阻尼；反向必须先做其转置（仍乘 damp），
            # 再注入伴随源/累加梯度/推进，保证离散一致。
            for field in (vax, vaz, saxx, sazz, saxz):
                kernels.apply_damping(field, self._damp)
            exx, ezz, exz, ax, az = storage.fields_at(n)
            # 伴随源 = 检波器采样算子的转置作用在残差上（双线性散加）
            kernels.inject_adjoint_source(
                vax, vaz, self._rix, self._rjx, self._rwx,
                self._riz, self._rjz, self._rwz, residual[n],
            )
            kernels.adjoint_stiffness_gradient(
                saxx, sazz, saxz, exx, ezz, exz, self.dt,
                g["c11"], g["c13"], g["c33"], g["c44"],
            )
            kernels.adjoint_step(
                vax, vaz, saxx, sazz, saxz,
                vax2, vaz2, saxx2, sazz2, saxz2,
                self.c11, self.c13, self.c33, self.c44c,
                self.rho_x, self.rho_z, self.dt, self.dx, self.dz,
            )
            # 密度项使用本步推进后的速度伴随（速度更新方程的 Lagrange 乘子）
            kernels.adjoint_density_gradient(
                vax2, vaz2, ax, az, self.rho_x, self.rho_z, self.dt,
                g["rho_x"], g["rho_z"],
            )
            vax, vax2 = vax2, vax
            vaz, vaz2 = vaz2, vaz
            saxx, saxx2 = saxx2, saxx
            sazz, sazz2 = sazz2, sazz
            saxz, saxz2 = saxz2, saxz
            if self.free_surface:
                self._apply_free_surface(vax, vaz, saxx, sazz, saxz)

        g_phys, h_phys = self._physical_native(g, shot.hessian)
        return ShotGradient(
            shot=ishot,
            misfit_raw=misfit_raw,
            obs_energy=float(np.sum(obs_shot * obs_shot)),
            g=g_phys,
            h=h_phys,
            synthetic=syn,
            residuals=residual,
            storage_stride=storage.stride,
        )

    def compute_shot_gradient(
        self,
        ishot: int,
        model: ElasticModel,
        obs_shot: np.ndarray,
        *,
        syn_filter: Any | None = None,
    ) -> ShotGradient:
        """计算单炮原始梯度（用于逐炮检查或自定义炮集累积）。"""
        self.set_model(model)
        return self._shot_gradient(ishot, obs_shot, syn_filter=syn_filter)

    def compute_gradient(
        self,
        model: ElasticModel,
        obs: np.ndarray,
        *,
        syn_filter: Any | None = None,
        keep_shot_gradients: bool = False,
    ) -> GradientResult:
        """计算归一化目标函数、原生梯度和伪 Hessian 对角。

        ``syn_filter`` 只作用于合成数据；传入时 ``obs`` 必须已经用**同一个**
        滤波器处理过（EFWI 主循环就是这么做的），否则残差频带不一致。
        ``keep_shot_gradients=True`` 时保留逐炮梯度/数据（用于诊断，内存占用
        更大）；EFWI 主循环默认关闭。
        """
        self.set_model(model)
        obs = np.asarray(obs, dtype=np.float64)
        expected = (self.survey.n_shots, self.survey.nt, self.survey.n_receivers, 2)
        if obs.shape != expected:
            raise ValueError(f"obs 形状应为 {expected}，实际 {obs.shape}")
        energy = float(np.sum(obs * obs))
        if energy <= 0:
            energy = 1.0

        g_sum = {
            k: np.zeros(self.grid.shape, dtype=np.float64)
            for k in ("c11", "c13", "c33", "c44", "rho")
        }
        h_sum = {
            k: np.zeros(self.grid.shape, dtype=np.float64)
            for k in ("c11", "c13", "c33", "c44", "rho", "theta")
        }
        synthetic = np.empty(expected, dtype=np.float64)
        residuals = np.empty(expected, dtype=np.float64)
        total = 0.0
        shot_gradients: list[ShotGradient] | None = [] if keep_shot_gradients else None
        last_stride = 1

        for s in range(self.survey.n_shots):
            shot = self._shot_gradient(s, obs[s], syn_filter=syn_filter)
            last_stride = shot.storage_stride
            if shot_gradients is not None:
                shot_gradients.append(shot)
            synthetic[s] = shot.synthetic
            residuals[s] = shot.residuals
            total += shot.misfit_raw
            for key in g_sum:
                g_sum[key] += shot.g[key]
            for key in h_sum:
                h_sum[key] += shot.h[key]

        total /= energy
        # 目标函数按观测能量归一化，梯度同步归一化以保持线搜索斜率一致
        for key in g_sum:
            g_sum[key] /= energy

        return GradientResult(
            misfit=total,
            obs_energy=energy,
            g=g_sum,
            h=h_sum,
            synthetic=synthetic,
            residuals=residuals,
            meta={
                "vmax": self.vmax,
                "dt": self.dt,
                "n_shots": self.survey.n_shots,
                "storage_stride": last_stride,
            },
            shot_gradients=shot_gradients,
        )

    # ------------------------------------------------------------------
    # RTM：正演源波场 + 接收波场反传 + 互相关成像
    # ------------------------------------------------------------------
    def _make_wavefield_storage(self, nt: int, storage_mb: float) -> tuple[np.ndarray, int]:
        nbytes_step = self.NX * self.NZ * self.dtype.itemsize
        max_steps = max(2, int(storage_mb * 1e6 / max(nbytes_step, 1)))
        stride = max(1, int(np.ceil(nt / max_steps)))
        nstore = int(np.ceil(nt / stride))
        return np.zeros((nstore, self.NX, self.NZ), dtype=self.dtype), stride

    def _source_wavefield(
        self, ishot: int, kind: str, storage_mb: float
    ) -> tuple[np.ndarray, int]:
        """正演并保存源波场的 P（散度）或 S（旋度）分量。"""
        nt = self.survey.nt
        dtype = self.dtype
        vx = np.zeros((self.NX, self.NZ), dtype=dtype)
        vz = np.zeros((self.NX, self.NZ), dtype=dtype)
        sxx = np.zeros((self.NX, self.NZ), dtype=dtype)
        szz = np.zeros((self.NX, self.NZ), dtype=dtype)
        sxz = np.zeros((self.NX, self.NZ), dtype=dtype)
        ax = np.zeros((self.NX, self.NZ), dtype=dtype)
        az = np.zeros((self.NX, self.NZ), dtype=dtype)
        exx = np.zeros((self.NX, self.NZ), dtype=dtype)
        ezz = np.zeros((self.NX, self.NZ), dtype=dtype)
        exz = np.zeros((self.NX, self.NZ), dtype=dtype)
        store, stride = self._make_wavefield_storage(nt, storage_mb)
        scratch = np.zeros((self.NX, self.NZ), dtype=np.float64)

        if self.survey.source_kind == "explosive":
            si = self._src_i["stress"][ishot]
            sj = self._src_j["stress"][ishot]
            sw = self._src_w["stress"][ishot]
        else:
            si = self._src_i["vx"][ishot]
            sj = self._src_j["vx"][ishot]
            sw = self._src_w["vx"][ishot]
            # 力源 vz 权重单独处理
            siz = self._src_i["vz"][ishot]
            sjz = self._src_j["vz"][ishot]
            swz = self._src_w["vz"][ishot]

        for n in range(nt):
            amp = self.survey.amplitude * self.survey.wavelet[n]
            if self.survey.source_kind == "explosive":
                kernels.inject_source_stress(sxx, szz, si, sj, sw, amp, self.dt)
            else:
                kernels.inject_source_force_x(vx, self.rho_x, si, sj, sw, amp, self.dt)
                kernels.inject_source_force_z(vz, self.rho_z, siz, sjz, swz, amp, self.dt)
            kernels.update_velocity(
                vx, vz, sxx, szz, sxz, self.rho_x, self.rho_z,
                self.dt, self.dx, self.dz, ax, az, scratch, 0,
            )
            kernels.update_stress(
                vx, vz, sxx, szz, sxz, self.c11, self.c13, self.c33, self.c44c,
                self.dt, self.dx, self.dz, exx, ezz, exz,
                scratch, scratch, scratch, scratch, 0,
            )
            if self.free_surface:
                self._apply_free_surface(vx, vz, sxx, szz, sxz)
            if n % stride == 0:
                k = n // stride
                if kind == "p":
                    kernels.divergence(vx, vz, store[k], self.dx, self.dz)
                elif kind == "s":
                    kernels.curl(vx, vz, store[k], self.dx, self.dz)
                else:
                    raise ValueError("kind 只能是 p/s")
            for field in (vx, vz, sxx, szz, sxz):
                kernels.apply_damping(field, self._damp)
        return store, stride

    def rtm_shot(
        self,
        ishot: int,
        data: np.ndarray,
        *,
        wavefield: str = "p",
        imaging: str = "source_normalized",
        storage_mb: float = 256.0,
        eps: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """单炮弹性 RTM。

        ``wavefield="p"`` 使用散度（PP 成像），``"s"`` 使用旋度（SS 成像）。
        ``imaging="source_normalized"`` 为保幅的源照明归一化互相关。
        """
        data = np.asarray(data, dtype=np.float64)
        if data.shape[:2] != (self.survey.nt, self.survey.n_receivers):
            raise ValueError("RTM 数据形状应为 (nt, nr, 2)")
        store, stride = self._source_wavefield(ishot, wavefield, storage_mb)
        image = np.zeros((self.NX, self.NZ), dtype=np.float64)
        illum = np.zeros((self.NX, self.NZ), dtype=np.float64)

        dtype = self.dtype
        vax = np.zeros((self.NX, self.NZ), dtype=dtype)
        vaz = np.zeros((self.NX, self.NZ), dtype=dtype)
        saxx = np.zeros((self.NX, self.NZ), dtype=dtype)
        sazz = np.zeros((self.NX, self.NZ), dtype=dtype)
        saxz = np.zeros((self.NX, self.NZ), dtype=dtype)
        vax2 = np.zeros_like(vax)
        vaz2 = np.zeros_like(vaz)
        saxx2 = np.zeros_like(saxx)
        sazz2 = np.zeros_like(sazz)
        saxz2 = np.zeros_like(saxz)
        recv_field = np.zeros((self.NX, self.NZ), dtype=np.float64)

        nt = self.survey.nt
        for n in range(nt - 1, -1, -1):
            for field in (vax, vaz, saxx, sazz, saxz):
                kernels.apply_damping(field, self._damp)
            kernels.inject_adjoint_source(
                vax, vaz, self._rix, self._rjx, self._rwx,
                self._riz, self._rjz, self._rwz, data[n],
            )
            if wavefield == "p":
                kernels.divergence(vax, vaz, recv_field, self.dx, self.dz)
            else:
                kernels.curl(vax, vaz, recv_field, self.dx, self.dz)
            k = min(n // stride, store.shape[0] - 1)
            kernels.accumulate_image(image, illum, store[k], recv_field, self.dt)
            kernels.adjoint_step(
                vax, vaz, saxx, sazz, saxz,
                vax2, vaz2, saxx2, sazz2, saxz2,
                self.c11, self.c13, self.c33, self.c44c,
                self.rho_x, self.rho_z, self.dt, self.dx, self.dz,
            )
            vax, vax2 = vax2, vax
            vaz, vaz2 = vaz2, vaz
            saxx, saxx2 = saxx2, saxx
            sazz, sazz2 = sazz2, sazz
            saxz, saxz2 = saxz2, saxz
            if self.free_surface:
                self._apply_free_surface(vax, vaz, saxx, sazz, saxz)

        if imaging == "source_normalized":
            image = image / (illum + eps * max(float(np.max(illum)), 1e-30))
        elif imaging != "crosscorrelation":
            raise ValueError("imaging 只能是 source_normalized/crosscorrelation")
        return image, illum

    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "grid": (self.grid.nx, self.grid.nz),
            "padded": (self.NX, self.NZ),
            "dx": self.dx,
            "dz": self.dz,
            "dt": self.dt,
            "nt": self.survey.nt,
            "n_shots": self.survey.n_shots,
            "n_receivers": self.survey.n_receivers,
            "vmax": self.vmax,
            "dtype": str(self.dtype),
            "storage_mb": self.storage_mb,
            "free_surface": self.free_surface,
        }
