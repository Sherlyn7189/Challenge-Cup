import os
import tempfile
import unittest
import numpy as np

from efwi.imaging import DipField
from efwi.model import constant_model
from efwi.workflow import Model3D, build_3d_model


class TestWorkflow(unittest.TestCase):
    def test_build_3d(self):
        model = constant_model(20, 15, 3000, 1600, 2200, dx=10, dz=10)
        dip = np.zeros((20, 15))
        dip[:, 5:] = np.radians(20.0)
        field = DipField(
            dip=dip,
            coherence=np.ones_like(dip),
            reflection_mask=np.ones_like(dip, dtype=bool),
            image=np.ones_like(dip),
        )
        out = build_3d_model(model, field, ny=7, dy=10.0, max_shift_cells=4)
        self.assertEqual(out.vp.shape, (20, 7, 15))
        self.assertTrue(np.all(np.isfinite(out.vp)))
        self.assertTrue(np.all(out.vs < out.vp))
        center = out.central_slice()
        self.assertEqual(center.shape, model.shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m3d.npz")
            out.save_npz(path)
            self.assertTrue(os.path.exists(path))

    def test_model3d_central_slice(self):
        arr = np.ones((4, 3, 5))
        m = Model3D(
            x=np.arange(4) * 10.0,
            y=np.arange(3) * 10.0,
            z=np.arange(5) * 10.0,
            vp=arr * 3000,
            vs=arr * 1600,
            rho=arr * 2200,
            epsilon=arr * 0.1,
            delta=arr * 0.05,
            gamma=arr * 0.1,
            symmetry="vti",
        )
        s = m.central_slice()
        self.assertEqual(s.shape, (4, 5))
        self.assertEqual(s.symmetry, "vti")


if __name__ == "__main__":
    unittest.main()
