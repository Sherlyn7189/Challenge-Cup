import unittest
import numpy as np

from efwi.config import GridConfig
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.gradient import convert_gradient, precondition
from efwi.model import ElasticModel
from efwi.pseudo_hessian import PseudoHessian


def make_case():
    grid = GridConfig(32, 28, 20.0, 20.0, sponge=5)
    nx, nz = grid.shape
    x = np.arange(nx) * 20.0
    z = np.arange(nz) * 20.0
    X, Z = np.meshgrid(x, z, indexing="ij")
    vp0 = np.full((nx, nz), 3000.0)
    vs0 = np.full((nx, nz), 1600.0)
    rho0 = np.full((nx, nz), 2200.0)
    true = ElasticModel(
        vp0 + 120 * np.exp(-((X - 300) ** 2 + (Z - 280) ** 2) / (2 * 55 ** 2)),
        vs0 + 60 * np.exp(-((X - 300) ** 2 + (Z - 280) ** 2) / (2 * 55 ** 2)),
        rho0 + 40 * np.exp(-((X - 300) ** 2 + (Z - 280) ** 2) / (2 * 55 ** 2)),
        dx=20.0, dz=20.0,
    )
    init = ElasticModel(vp0.copy(), vs0.copy(), rho0.copy(), dx=20.0, dz=20.0)
    shots = np.array([[140.0, 100.0], [460.0, 100.0]])
    recv = np.array([[xx, 100.0] for xx in np.linspace(120.0, 480.0, 11)])
    survey = Survey.from_positions(
        grid, shots, recv, f0=12.0, vmin=1500.0, vmax=3300.0, dt=0.0012, nt=150
    )
    return grid, true, init, survey


class TestPrecondition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.true, cls.init, cls.survey = make_case()
        cls.op = Elastic2DOperator(cls.true, cls.survey, cls.grid, dtype="float64", storage_mb=100)
        cls.obs = cls.op.forward_all(cls.true)
        cls.res = cls.op.compute_gradient(cls.init, cls.obs)
        cls.pg = convert_gradient(cls.res, cls.init, ("vp", "vs", "rho"))

    def test_pseudo_hessian_positive_and_theta(self):
        for key in ("c11", "c13", "c33", "c44", "rho", "theta"):
            self.assertIn(key, self.res.h)
            self.assertTrue(np.all(self.res.h[key] >= -1e-12))
            self.assertGreater(np.max(self.res.h[key]), 0)
        for key in ("vp", "vs", "rho"):
            self.assertTrue(np.all(self.pg.h[key] >= 0))
            self.assertGreater(np.max(self.pg.h[key]), 0)

    def test_precondition_formula_without_cap(self):
        pre = precondition(
            self.pg.g, self.pg.h,
            damping=1e-3, smooth=0.0, normalize=False, max_gain=0,
        )
        for key, gk in self.pg.g.items():
            hk = self.pg.h[key]
            expected = gk / (hk + 1e-3 * np.max(hk))
            np.testing.assert_allclose(pre[key], expected, rtol=1e-12, atol=1e-20)

    def test_precondition_stability(self):
        g = {k: v.copy() for k, v in self.pg.g.items()}
        h = {k: v.copy() for k, v in self.pg.h.items()}
        # 人为制造极弱照明/噪声区
        h["vp"][:, :3] = 0.0
        g["vp"][:, :3] += 1e-3 * np.max(np.abs(g["vp"]))
        pre = precondition(
            g, h, damping=1e-12, smooth=0.0, normalize=False,
            max_gain=5.0, water_level=1e-6,
        )
        for key, arr in pre.items():
            self.assertTrue(np.all(np.isfinite(arr)))
        # 深部/弱照明区的增益被限制，不出现数量级爆炸
        raw_scale = np.percentile(np.abs(g["vp"]), 99)
        pre_scale = np.percentile(np.abs(pre["vp"]), 99)
        self.assertLess(pre_scale, 100.0 * raw_scale)

    def test_precondition_changes_spatial_pattern(self):
        pre = precondition(
            self.pg.g, self.pg.h,
            damping=1e-3, smooth=1.0, normalize=False, max_gain=20.0,
        )
        for key in ("vp", "vs", "rho"):
            raw = self.pg.g[key]
            ratio = pre[key] / (raw + 1e-30)
            # 预条件增益不是常数：空间上随照明变化
            self.assertGreater(np.std(ratio), 1e-6)
            self.assertGreater(
                np.max(np.abs(pre[key] - raw)),
                1e-3 * np.max(np.abs(raw)),
            )

    def test_pseudo_hessian_wrapper(self):
        ph = PseudoHessian.from_gradient_result(self.res, self.init, ("vp", "vs", "rho"))
        stats = ph.stats(self.pg.g)
        self.assertEqual(set(stats), {"vp", "vs", "rho"})
        self.assertGreater(stats["vp"]["h_max"], 0)
        illum = ph.illumination()
        self.assertEqual(illum.shape, self.grid.shape)
        self.assertAlmostEqual(float(np.max(illum)), 1.0, places=6)
        pre = ph.precondition(self.pg.g, damping=1e-3, smooth=0.0, max_gain=0)
        self.assertEqual(set(pre), {"vp", "vs", "rho"})


if __name__ == "__main__":
    unittest.main()
