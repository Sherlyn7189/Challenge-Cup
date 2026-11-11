import unittest
import numpy as np

from efwi.config import FWIConfig, GridConfig, StageConfig
from efwi.filters import (
    WienerFilter,
    band_energy_fractions,
    schedule_table,
    validate_stages,
)
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.inversion import ElasticFWI
from efwi.model import ElasticModel
from efwi.wavelet import make_wavelet


def make_case():
    grid = GridConfig(30, 26, 20.0, 20.0, sponge=5)
    nx, nz = grid.shape
    x = np.arange(nx) * 20.0
    z = np.arange(nz) * 20.0
    X, Z = np.meshgrid(x, z, indexing="ij")
    vp = np.full((nx, nz), 3000.0) + 180 * np.exp(
        -((X - 300) ** 2 + (Z - 260) ** 2) / (2 * 60 ** 2)
    )
    true = ElasticModel(vp, vp / 1.8, 310 * vp ** 0.25, dx=20.0, dz=20.0)
    init = ElasticModel(
        np.full((nx, nz), 3000.0),
        np.full((nx, nz), 1666.0),
        np.full((nx, nz), 2310.0),
        dx=20.0, dz=20.0,
    )
    shots = np.array([[140.0, 100.0], [450.0, 100.0]])
    recv = np.array([[xx, 100.0] for xx in np.linspace(120.0, 460.0, 9)])
    survey = Survey.from_positions(
        grid, shots, recv, f0=12.0, vmin=1500.0, vmax=3300.0, dt=0.0012, nt=150
    )
    return grid, true, init, survey


class TestMultiscale(unittest.TestCase):
    def test_schedule_validation(self):
        good = [StageConfig(2, 8, 3, "low"), StageConfig(3, 15, 3, "high")]
        validate_stages(good, 0.001)
        table = schedule_table(good, 0.001)
        self.assertEqual(len(table), 2)
        self.assertLess(table[0]["f_high"], table[1]["f_high"])
        with self.assertRaises(ValueError):
            validate_stages([StageConfig(3, 15, 1), StageConfig(2, 8, 1)], 0.001)
        with self.assertRaises(ValueError):
            validate_stages([StageConfig(2, 600, 1)], 0.001)

    def test_filter_responses(self):
        rng = np.random.default_rng(0)
        dt = 0.001
        nt = 600
        data = rng.normal(size=(nt, 20))
        low = WienerFilter.design(data, dt, 2.0, 8.0)
        high = WienerFilter.design(data, dt, 3.0, 15.0)
        freqs = np.array([2.5, 5.0, 7.5, 10.0, 12.0, 25.0, 50.0])
        rl = low.amplitude_response(freqs)
        rh = high.amplitude_response(freqs)
        self.assertGreater(rl[1], 0.2)  # 5 Hz 在低频带内
        self.assertGreater(rh[4], 0.2)  # 12 Hz 在高频带内
        self.assertLess(rl[5], 1e-2)    # 25 Hz 在 8 Hz 带外
        self.assertLess(rh[5], 1e-2)    # 25 Hz 在 15 Hz 带外
        self.assertGreaterEqual(np.min(rl), 0.0)
        self.assertGreaterEqual(np.min(rh), 0.0)

    def test_band_energy(self):
        dt = 0.001
        nt = 800
        rng = np.random.default_rng(1)
        data = rng.normal(size=(nt, 10))
        stages = [StageConfig(2, 8, 1), StageConfig(3, 15, 1)]
        frac = band_energy_fractions(data, dt, stages)
        self.assertEqual(len(frac), 2)
        self.assertTrue(all(0 <= x <= 1 for x in frac))
        self.assertGreater(sum(frac), 0.0)

    def test_two_stage_inversion_progression(self):
        grid, true, init, survey = make_case()
        cfg = FWIConfig(
            stages=[StageConfig(2, 8, 1, "band_8Hz"), StageConfig(3, 15, 1, "band_15Hz")],
            dtype="float64",
            storage_mb=100.0,
            verbose=False,
        )
        fwi = ElasticFWI(init, survey, true_model=true, grid=grid, config=cfg)
        result = fwi.run()
        self.assertEqual(len(result.stage_summaries), 2)
        s0, s1 = result.stage_summaries
        self.assertLessEqual(s0["f_high"], s1["f_high"])
        self.assertEqual(s0["iter_done"], 1)
        self.assertEqual(s1["iter_done"], 1)
        self.assertGreater(s0["misfit_reduction_pct"], 0)
        self.assertGreater(s1["misfit_reduction_pct"], 0)


if __name__ == "__main__":
    unittest.main()
