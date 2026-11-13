import unittest
import numpy as np

from efwi.fd3d import Elastic3DOperator, ElasticModel3D, Grid3D, Survey3D
from efwi.wavelet import make_wavelet


def make_case():
    grid = Grid3D(14, 14, 14, 20.0, 20.0, 20.0, pad=2, sponge=3)
    nx, ny, nz = grid.shape
    vp = np.full((nx, ny, nz), 3000.0)
    vs = vp / 1.8
    rho = np.full((nx, ny, nz), 2200.0)
    model = ElasticModel3D(vp, vs, rho, 20.0, 20.0, 20.0)
    shots = np.array([[140.0, 140.0, 100.0]])
    receivers = np.array([[x, 140.0, 100.0] for x in np.linspace(100.0, 200.0, 5)])
    nt = 60
    dt = grid.cfl_dt(3000.0, safety=0.25)
    survey = Survey3D(shots, receivers, nt, dt, make_wavelet(nt, dt, 15.0))
    return grid, model, survey


class Test3D(unittest.TestCase):
    def test_forward(self):
        grid, model, survey = make_case()
        op = Elastic3DOperator(model, survey, grid, dtype="float64", storage_mb=200)
        data = op.forward_all()
        self.assertEqual(data.shape, (survey.n_shots, survey.nt, survey.n_receivers, 3))
        self.assertTrue(np.all(np.isfinite(data)))
        self.assertGreater(np.max(np.abs(data)), 0)

    def test_model_validation(self):
        with self.assertRaises(ValueError):
            Grid3D(5, 5, 5, 10, 10, 10)
        with self.assertRaises(ValueError):
            ElasticModel3D(
                np.ones((12, 12, 12)) * 1000,
                np.ones((12, 12, 12)) * 2000,
                np.ones((12, 12, 12)) * 2200,
            )

    def test_gradient_guard(self):
        grid, model, survey = make_case()
        op = Elastic3DOperator(model, survey, grid, dtype="float64", storage_mb=200)
        obs = op.forward_all()
        with self.assertRaises(NotImplementedError):
            op.compute_gradient(model, obs)
        res = op.compute_gradient(model, obs, allow_unvalidated=True)
        self.assertTrue(np.all(np.isfinite(res.g_vp)))
        self.assertEqual(res.g_vp.shape, grid.shape)


if __name__ == "__main__":
    unittest.main()
