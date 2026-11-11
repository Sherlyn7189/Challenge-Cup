import unittest
import numpy as np

from efwi.config import ConstraintConfig
from efwi.constraints import (
    GardnerTube,
    ModelConstraints,
    admm_projection,
    dykstra_projection,
    project_gradient_bound,
)


class TestConstraints(unittest.TestCase):
    def test_gradient_projection(self):
        m = np.zeros((50, 40))
        m[:, 20:] = 1.0
        q = project_gradient_bound(m, 0.05, 0.05, 10.0, 10.0, n_iter=80, rho=10.0)
        gx = np.max(np.abs((q[1:] - q[:-1]) / 10.0))
        gz = np.max(np.abs((q[:, 1:] - q[:, :-1]) / 10.0))
        self.assertLess(gx, 0.052)
        self.assertLess(gz, 0.052)
        self.assertAlmostEqual(float(np.mean(q)), float(np.mean(m)), places=6)

    def test_gardner_tube(self):
        rng = np.random.default_rng(0)
        vp = rng.uniform(2000, 4000, (30, 20))
        rho = 310 * vp ** 0.25 * np.exp(rng.normal(0, 0.3, vp.shape))
        g = GardnerTube(tol=0.1)
        out = g.project({"vp": vp, "rho": rho})
        res = g.residual(out)
        self.assertLessEqual(np.max(np.abs(res)), 0.100001)

    def test_dykstra_and_admm(self):
        rng = np.random.default_rng(1)
        shape = (25, 20)
        vp = 3000 + 300 * rng.normal(size=shape)
        vs = vp / 1.8
        rho = 310 * vp ** 0.25
        values = {"vp": vp, "vs": vs, "rho": rho}
        cfg = ConstraintConfig(
            method="dykstra", n_iter=4, grad_proj_iter=30,
            vp_grad_max=(5.0, 8.0), vs_grad_max=(4.0, 6.0), rho_grad_max=(3.0, 5.0),
        )
        mc = ModelConstraints(cfg, 10.0, 10.0, ("vp", "vs", "rho"), shape)
        out = mc.project(values)
        self.assertGreaterEqual(out["vp"].min(), cfg.vp_bounds[0] - 1e-8)
        self.assertLessEqual(out["vp"].max(), cfg.vp_bounds[1] + 1e-8)
        self.assertTrue(np.all(out["vs"] < out["vp"]))
        g = GardnerTube(tol=cfg.gardner_tol)
        self.assertLessEqual(np.max(np.abs(g.residual(out))), cfg.gardner_tol + 1e-6)

        cfg2 = ConstraintConfig(method="admm", n_iter=4, grad_proj_iter=20)
        mc2 = ModelConstraints(cfg2, 10.0, 10.0, ("vp", "vs", "rho"), shape)
        out2 = mc2.project(values)
        self.assertTrue(np.all(np.isfinite(out2["vp"])))

    def test_projection_functions_direct(self):
        values = {"x": np.array([[1.0, 2.0], [3.0, 4.0]])}
        # 用恒等集合验证 Dykstra/ADMM 不改变输入
        class Identity:
            keys = ("x",)

            def project(self, v):
                return {"x": v["x"].copy()}

        np.testing.assert_allclose(dykstra_projection(values, [Identity()], 2)["x"], values["x"])
        np.testing.assert_allclose(admm_projection(values, [Identity()], 2)["x"], values["x"])


if __name__ == "__main__":
    unittest.main()
