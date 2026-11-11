import unittest
import numpy as np

from efwi.config import GridConfig
from efwi.geometry import Survey
from efwi.model import ElasticModel, constant_model
from efwi.fd import Elastic2DOperator


def make_setup(anisotropy="iso"):
    grid = GridConfig(50, 40, 15.0, 15.0, sponge=5)
    model = constant_model(
        grid.nx, grid.nz, 3000.0, 1600.0, 2200.0,
        dx=15.0, dz=15.0, symmetry=anisotropy,
        epsilon=0.15, delta=0.08, gamma=0.12,
    )
    shots = np.array([[150.0, 120.0]])
    receivers = np.array([[x, 120.0] for x in np.linspace(120.0, 600.0, 17)])
    survey = Survey.from_positions(
        grid, shots, receivers, f0=12.0, vmin=1500.0, vmax=3200.0, dt=0.001, nt=180
    )
    return grid, model, survey


class TestForward(unittest.TestCase):
    def test_isotropic_forward_finite(self):
        grid, model, survey = make_setup()
        op = Elastic2DOperator(model, survey, grid, dtype="float64", storage_mb=100)
        data = op.forward_all()
        self.assertEqual(data.shape, (1, survey.nt, survey.n_receivers, 2))
        self.assertTrue(np.all(np.isfinite(data)))
        self.assertGreater(np.max(np.abs(data)), 0)

    def test_anisotropic_stability(self):
        for sym in ("vti", "hti"):
            grid, model, survey = make_setup(sym)
            op = Elastic2DOperator(model, survey, grid, dtype="float64", storage_mb=100)
            data = op.forward_all()
            self.assertTrue(np.all(np.isfinite(data)))
            self.assertLess(np.max(np.abs(data)), 1e3)

    def test_source_force(self):
        grid, model, survey = make_setup()
        survey_force = Survey(
            survey.shots, survey.receivers, survey.nt, survey.dt,
            survey.wavelet, source_kind="force",
        )
        op = Elastic2DOperator(model, survey_force, grid, dtype="float64")
        data = op.forward_all()
        self.assertTrue(np.all(np.isfinite(data)))

    def test_misfit_zero_for_same_data(self):
        grid, model, survey = make_setup()
        op = Elastic2DOperator(model, survey, grid, dtype="float64")
        obs = op.forward_all()
        self.assertLess(op.misfit(model, obs), 1e-20)


if __name__ == "__main__":
    unittest.main()
