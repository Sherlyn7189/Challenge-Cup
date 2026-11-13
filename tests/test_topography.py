import unittest
import numpy as np

from efwi.config import GridConfig
from efwi.fd import Elastic2DOperator
from efwi.geometry import Survey
from efwi.model import ElasticModel
from efwi.io import SegyGathers
from efwi.topography import (
    Topography,
    apply_elevation_statics,
    fill_air_with_replacement,
)


def make_setup(free_surface=False):
    grid = GridConfig(40, 32, 15.0, 15.0, sponge=5)
    nx, nz = grid.shape
    x = np.arange(nx) * 15.0
    z = np.arange(nz) * 15.0
    X, Z = np.meshgrid(x, z, indexing="ij")
    vp = 3000.0 + 100.0 * np.tanh((Z - 240.0) / 20.0)
    model = ElasticModel(vp, vp / 1.8, 310 * vp ** 0.25, dx=15.0, dz=15.0)
    shots = np.array([[120.0, 90.0], [400.0, 90.0]])
    receivers = np.array([[xx, 90.0] for xx in np.linspace(100.0, 500.0, 15)])
    survey = Survey.from_positions(
        grid, shots, receivers, f0=12.0, vmin=1500.0, vmax=3300.0,
        dt=0.0012, nt=150, allow_sponge=True,
    )
    return grid, model, survey


class TestTopography(unittest.TestCase):
    def test_flat_and_hill_statics(self):
        x = np.array([0.0, 100.0, 200.0])
        flat = Topography.flat(x, 0.0)
        self.assertTrue(np.allclose(flat.at(x), 0.0))
        hill = Topography(x, np.array([0.0, 30.0, 10.0]))
        stat = hill.statics(np.array([[0.0, 0.0], [100.0, 0.0]]), datum=0.0, replacement_velocity=1500.0)
        self.assertAlmostEqual(float(stat[0]), 0.0)
        self.assertAlmostEqual(float(stat[1]), -0.02)

    def test_fill_air(self):
        grid = GridConfig(20, 15, 10.0, 10.0, sponge=4)
        m = ElasticModel(
            np.full((20, 15), 3000.0),
            np.full((20, 15), 1600.0),
            np.full((20, 15), 2200.0),
            dx=10.0, dz=10.0,
        )
        topo = Topography(np.linspace(0, 190, 20), np.full(20, 30.0))
        out = fill_air_with_replacement(m, topo, replacement_velocity=1500.0)
        self.assertEqual(out.vp.shape, m.vp.shape)
        self.assertTrue(np.all(out.vp[0, :3] == 1500.0))
        self.assertTrue(np.all(out.vp[0, 4:] == 3000.0))

    def test_apply_statics(self):
        nt, dt, nr, ns = 120, 0.002, 5, 2
        shots = np.array([[100.0, 0.0], [300.0, 0.0]])
        receivers = np.array([[x, 0.0] for x in np.linspace(50.0, 250.0, nr)])
        data = np.zeros((ns, nt, nr, 1))
        data[:, 40, :, 0] = 1.0
        mask = np.ones((ns, nr, 1), dtype=bool)
        g = SegyGathers(shots, receivers, data, dt, mask, np.array([1, 2]), np.arange(nr))
        topo = Topography.flat(np.array([0.0, 300.0]), 0.0)
        out, qc = apply_elevation_statics(
            g, topo, datum=10.0, replacement_velocity=1000.0,
            source_sign=0.0, receiver_sign=0.0,
        )
        # 静校正为零时数据不变
        self.assertTrue(np.allclose(out.data, g.data))
        self.assertEqual(qc["datum"], 10.0)

    def test_free_surface_forward(self):
        grid, model, survey = make_setup()
        op_abs = Elastic2DOperator(model, survey, grid, dtype="float64", free_surface=False)
        op_fs = Elastic2DOperator(model, survey, grid, dtype="float64", free_surface=True)
        self.assertTrue(op_fs.info()["free_surface"])
        d1 = op_abs.forward_all()
        d2 = op_fs.forward_all()
        self.assertTrue(np.all(np.isfinite(d2)))
        self.assertGreater(np.max(np.abs(d2)), 0)
        self.assertFalse(np.allclose(d1, d2))


if __name__ == "__main__":
    unittest.main()
