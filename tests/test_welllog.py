import os
import tempfile
import unittest
import numpy as np

from efwi.config import GridConfig
from efwi.welllog import (
    WellLog,
    build_initial_model,
    design_scale,
    extrapolate_curve,
    logging_seismic_scales,
)


LAS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "example_well.las")


class TestWellLog(unittest.TestCase):
    def test_read_las(self):
        log = WellLog.from_las(LAS_PATH)
        self.assertGreater(log.depth.size, 100)
        self.assertTrue(np.all(np.diff(log.depth) > 0))
        self.assertTrue(np.all(log.vp > 1000))
        self.assertTrue(np.all(log.vs < log.vp))
        self.assertTrue(np.all(log.rho > 1500))
        self.assertAlmostEqual(float(log.vp[0]), 1650.0, delta=80.0)

    def test_extrapolate(self):
        depth = np.linspace(100, 500, 50)
        vp = 2000 + 2.0 * depth
        z = np.linspace(0, 800, 81)
        out = extrapolate_curve(depth, vp, z, positive_trend=True)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(out[0], 1500)
        self.assertGreater(out[-1], out[0])

    def test_build_initial_model(self):
        log = WellLog.from_las(LAS_PATH)
        grid = GridConfig(30, 40, 20.0, 20.0, sponge=4)
        model = build_initial_model([log], grid)
        self.assertEqual(model.shape, grid.shape)
        self.assertTrue(np.all(np.isfinite(model.vp)))
        self.assertTrue(np.all(model.vs < model.vp))

    def test_scale_design(self):
        s = design_scale(1600.0, 4200.0, 30.0, ppw=8.0)
        self.assertGreater(s["dx"], 0)
        self.assertGreater(s["dt"], 0)
        scales = logging_seismic_scales(3000.0, 10000.0, 30.0)
        self.assertAlmostEqual(scales["ratio"]["dx"], 0.003, places=3)
        self.assertGreater(scales["ratio"]["frequency"], 300)

    def test_missing_vs_density(self):
        log = WellLog(depth=np.linspace(0, 100, 11), vp=np.linspace(2000, 3000, 11))
        self.assertTrue(np.all(log.vs > 0))
        self.assertTrue(np.all(log.rho > 0))


if __name__ == "__main__":
    unittest.main()
