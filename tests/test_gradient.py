import unittest
import numpy as np
from scipy.ndimage import gaussian_filter

from efwi.adjoint import (
    adjoint_source,
    check_gradient,
    compute_adjoint_gradients,
    data_residual,
    objective,
    shot_gradient_table,
)
from efwi.config import GridConfig
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.gradient import convert_gradient
from efwi.model import ElasticModel


def setup():
    grid = GridConfig(36, 30, 20.0, 20.0, sponge=5)
    nx, nz = grid.shape
    x = np.arange(nx) * 20.0
    z = np.arange(nz) * 20.0
    X, Z = np.meshgrid(x, z, indexing="ij")
    vp0 = np.full((nx, nz), 3000.0)
    vs0 = np.full((nx, nz), 1600.0)
    rho0 = np.full((nx, nz), 2200.0)
    true = ElasticModel(
        vp0 + 120 * np.exp(-((X - 300) ** 2 + (Z - 300) ** 2) / (2 * 60 ** 2)),
        vs0 + 60 * np.exp(-((X - 300) ** 2 + (Z - 300) ** 2) / (2 * 60 ** 2)),
        rho0 + 50 * np.exp(-((X - 300) ** 2 + (Z - 300) ** 2) / (2 * 60 ** 2)),
        dx=20.0, dz=20.0,
    )
    init = ElasticModel(vp0.copy(), vs0.copy(), rho0.copy(), dx=20.0, dz=20.0)
    shots = np.array([[140.0, 100.0], [500.0, 100.0]])
    recv = np.array([[x, 100.0] for x in np.linspace(120.0, 520.0, 11)])
    survey = Survey.from_positions(
        grid, shots, recv, f0=12.0, vmin=1500.0, vmax=3300.0, dt=0.0012, nt=160
    )
    return grid, true, init, survey


class TestGradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.true, cls.init, cls.survey = setup()
        cls.op = Elastic2DOperator(
            cls.true, cls.survey, cls.grid, dtype="float64", storage_mb=100
        )
        cls.obs = cls.op.forward_all(cls.true)
        x = np.arange(cls.grid.nx) * cls.grid.dx
        z = np.arange(cls.grid.nz) * cls.grid.dz
        cls.X, cls.Z = np.meshgrid(x, z, indexing="ij")

    def _directions(self):
        d1 = np.exp(-((self.X - 300) ** 2 + (self.Z - 280) ** 2) / (2 * 90 ** 2))
        d1 /= np.max(d1)
        rng = np.random.default_rng(3)
        d2 = gaussian_filter(rng.normal(size=self.grid.shape), sigma=3.0, mode="nearest")
        d2 /= np.max(np.abs(d2))
        return {"gaussian": d1, "random_smooth": d2}

    def test_directional_derivative_matches_fd(self):
        """核心验收：伴随梯度 vs 中心差分方向导数（三个参数、两个方向）。"""
        for name, d in self._directions().items():
            checks = check_gradient(
                self.op,
                self.init,
                self.obs,
                {"vp": d, "vs": d, "rho": d},
                eps={"vp": 2.0, "vs": 1.0, "rho": 1.0},
            )
            for key, row in checks.items():
                self.assertLess(
                    row["rel_error"], 1e-3,
                    msg=f"{name}/{key}: adj={row['adjoint']:.6g}, "
                        f"fd={row['finite_difference']:.6g}, "
                        f"rel={row['rel_error']:.3g}",
                )

    def test_random_observed_data_gradient(self):
        """任意观测数据下的时间循环伴随（不依赖“真模型”结构）。"""
        rng = np.random.default_rng(9)
        obs = rng.normal(
            0.0, 1e-5,
            size=(self.survey.n_shots, self.survey.nt, self.survey.n_receivers, 2),
        )
        d = np.exp(-((self.X - 300) ** 2 + (self.Z - 280) ** 2) / (2 * 90 ** 2))
        d /= np.max(d)
        checks = check_gradient(
            self.op, self.init, obs,
            {"vp": d, "vs": d, "rho": d},
            eps={"vp": 1.0, "vs": 1.0, "rho": 1.0},
        )
        for key, row in checks.items():
            self.assertLess(row["rel_error"], 1e-3, msg=f"{key}: {row}")

    def test_shot_gradient_accumulation(self):
        """多炮梯度必须等于逐炮原始梯度之和 / 全局观测能量。"""
        full = self.op.compute_gradient(self.init, self.obs)
        result = compute_adjoint_gradients(
            self.op, self.init, self.obs, params=("vp", "vs", "rho")
        )
        pg = convert_gradient(full, self.init, ("vp", "vs", "rho"))
        for key in ("vp", "vs", "rho"):
            np.testing.assert_allclose(pg.g[key], result.g[key], rtol=1e-10, atol=1e-16)
        for key in ("c11", "c13", "c33", "c44", "rho"):
            np.testing.assert_allclose(full.h[key], result.h_native[key], rtol=1e-12, atol=1e-16)
        self.assertAlmostEqual(full.misfit, result.misfit, places=12)
        self.assertEqual(len(result.shots), self.survey.n_shots)
        raw_sum = sum(s.misfit_raw for s in result.shots)
        self.assertAlmostEqual(raw_sum / result.obs_energy, full.misfit, places=12)
        table = shot_gradient_table(result)
        self.assertEqual(len(table), self.survey.n_shots)
        self.assertGreater(table[0]["vp_norm"], 0.0)
        self.assertGreater(table[1]["vp_norm"], 0.0)

    def test_shot_gradient_api(self):
        shot = self.op.compute_shot_gradient(0, self.init, self.obs[0])
        self.assertEqual(shot.shot, 0)
        self.assertGreater(shot.misfit_raw, 0.0)
        for key in ("c11", "c13", "c33", "c44", "rho"):
            self.assertEqual(shot.g[key].shape, self.grid.shape)
            self.assertTrue(np.all(np.isfinite(shot.g[key])))

    def test_residual_and_objective(self):
        syn = self.obs.copy()
        res = data_residual(syn, self.obs)
        self.assertEqual(res.r.shape, self.obs.shape)
        self.assertLess(np.max(np.abs(res.r)), 1e-20)
        self.assertEqual(objective(res), 0.0)
        src = adjoint_source(res)
        self.assertEqual(src.shape, self.obs.shape)

        syn2 = syn + 1e-6
        res2 = data_residual(syn2, self.obs)
        expected = 0.5 * np.sum(res2.r ** 2) / res2.energy
        self.assertAlmostEqual(objective(res2), expected, places=18)
        self.assertTrue(np.allclose(adjoint_source(res2, gain=2.0), 2.0 * res2.r))

    def test_hessian_positive(self):
        res = self.op.compute_gradient(self.init, self.obs)
        for key, arr in res.h.items():
            self.assertTrue(np.all(arr >= -1e-12))
            self.assertGreater(np.max(arr), 0)


if __name__ == "__main__":
    unittest.main()
