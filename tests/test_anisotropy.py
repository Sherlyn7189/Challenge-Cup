import unittest
import numpy as np

from efwi.config import GridConfig
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.gradient import convert_gradient
from efwi.model import ElasticModel


class TestAnisotropyGradient(unittest.TestCase):
    def test_epsilon_delta_directional_derivative(self):
        grid = GridConfig(30, 26, 20.0, 20.0, sponge=5)
        nx, nz = grid.shape
        x = np.arange(nx) * 20.0
        z = np.arange(nz) * 20.0
        X, Z = np.meshgrid(x, z, indexing="ij")
        vp0 = np.full((nx, nz), 3000.0)
        vs0 = np.full((nx, nz), 1600.0)
        rho0 = np.full((nx, nz), 2200.0)
        true = ElasticModel(
            vp0 + 100 * np.exp(-((X - 300) ** 2 + (Z - 260) ** 2) / (2 * 60 ** 2)),
            vs0 + 50 * np.exp(-((X - 300) ** 2 + (Z - 260) ** 2) / (2 * 60 ** 2)),
            rho0,
            np.full((nx, nz), 0.15),
            np.full((nx, nz), 0.08),
            0.10,
            "vti",
            dx=20.0,
            dz=20.0,
        )
        init = ElasticModel(
            vp0.copy(), vs0.copy(), rho0.copy(),
            np.full((nx, nz), 0.12), np.full((nx, nz), 0.06), 0.10,
            "vti", dx=20.0, dz=20.0,
        )
        shots = np.array([[140.0, 100.0], [450.0, 100.0]])
        recv = np.array([[xx, 100.0] for xx in np.linspace(120.0, 460.0, 9)])
        survey = Survey.from_positions(
            grid, shots, recv, f0=12.0, vmin=1500.0, vmax=3300.0, dt=0.0012, nt=140
        )
        op = Elastic2DOperator(true, survey, grid, dtype="float64", storage_mb=100)
        obs = op.forward_all(true)
        res = op.compute_gradient(init, obs)
        pg = convert_gradient(res, init, ("vp", "vs", "rho", "epsilon", "delta"))

        direction = np.exp(-((X - 300) ** 2 + (Z - 250) ** 2) / (2 * 80 ** 2))
        direction /= np.max(direction)
        for key in ("epsilon", "delta"):
            eps = 0.01
            mp = init.copy()
            mm = init.copy()
            setattr(mp, key, getattr(init, key) + eps * direction)
            setattr(mm, key, getattr(init, key) - eps * direction)
            fd = (op.misfit(mp, obs) - op.misfit(mm, obs)) / (2 * eps)
            adj = float(np.sum(pg.g[key] * direction))
            rel = abs(adj - fd) / max(abs(fd), 1e-30)
            self.assertLess(rel, 0.02, msg=f"{key}: adj={adj:.6g}, fd={fd:.6g}")


if __name__ == "__main__":
    unittest.main()
