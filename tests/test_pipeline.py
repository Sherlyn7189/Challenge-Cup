import unittest
import numpy as np

from efwi.config import ConstraintConfig, FWIConfig, GridConfig, StageConfig
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.imaging import extract_dips, reverse_time_migration, top_mute
from efwi.inversion import ElasticFWI
from efwi.model import ElasticModel


def build_case():
    grid = GridConfig(36, 30, 20.0, 20.0, sponge=5)
    nx, nz = grid.shape
    x = np.arange(nx) * 20.0
    z = np.arange(nz) * 20.0
    X, Z = np.meshgrid(x, z, indexing="ij")
    vp = np.full((nx, nz), 3000.0) + 160 * np.exp(
        -((X - 300) ** 2 + (Z - 320) ** 2) / (2 * 70 ** 2)
    )
    vs = vp / 1.8
    rho = 310 * vp ** 0.25
    true = ElasticModel(vp, vs, rho, dx=20.0, dz=20.0)
    init = ElasticModel(
        np.full((nx, nz), 3000.0),
        np.full((nx, nz), 1666.0),
        np.full((nx, nz), 2310.0),
        dx=20.0, dz=20.0,
    )
    shots = np.array([[140.0, 120.0], [500.0, 120.0]])
    recv = np.array([[xx, 120.0] for xx in np.linspace(120.0, 520.0, 13)])
    survey = Survey.from_positions(
        grid, shots, recv, f0=12.0, vmin=1500.0, vmax=3300.0, dt=0.0012, nt=160
    )
    return grid, true, init, survey


class TestPipeline(unittest.TestCase):
    def test_efwi_reduces_misfit(self):
        grid, true, init, survey = build_case()
        cfg = FWIConfig(
            stages=[StageConfig(2.0, 8.0, 2, "test")],
            dtype="float64",
            storage_mb=100.0,
            verbose=False,
            constraints=ConstraintConfig(method="none"),
        )
        fwi = ElasticFWI(init, survey, true_model=true, grid=grid, config=cfg)
        result = fwi.run()
        self.assertGreater(len(result.history), 2)
        self.assertLess(result.history[-1]["misfit"], result.history[0]["misfit"])
        self.assertTrue(np.all(np.isfinite(result.model.vp)))

    def test_rtm_and_dips(self):
        grid, true, init, survey = build_case()
        op = Elastic2DOperator(true, survey, grid, dtype="float64")
        data = op.forward_all()
        rtm = reverse_time_migration(
            true, survey, data, grid=grid, storage_mb=100.0, dtype="float64"
        )
        self.assertEqual(rtm.image.shape, grid.shape)
        self.assertTrue(np.all(np.isfinite(rtm.image)))
        self.assertGreater(np.max(np.abs(rtm.image)), 0)
        dip = extract_dips(rtm.image, grid.dx, grid.dz, sigma=2.0)
        self.assertEqual(dip.dip.shape, grid.shape)
        self.assertTrue(np.all(np.isfinite(dip.dip)))
        self.assertGreater(int(dip.reflection_mask.sum()), 0)

    def test_top_mute(self):
        grid, true, init, survey = build_case()
        op = Elastic2DOperator(true, survey, grid, dtype="float64")
        data = op.forward_all()
        out = top_mute(data, survey.dt, survey.shots, survey.receivers, 1500.0)
        self.assertEqual(out.shape, data.shape)
        self.assertLessEqual(np.sum(np.abs(out)), np.sum(np.abs(data)))


if __name__ == "__main__":
    unittest.main()
