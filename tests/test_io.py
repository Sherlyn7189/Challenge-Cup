import os
import tempfile
import unittest
import numpy as np

from efwi.io import (
    SegyGathers,
    gather_quality_control,
    read_segy_gathers,
    read_segy_shot_list,
    write_segy_gathers,
)
from efwi.wavelet import make_wavelet


def make_gathers():
    nt, dt, nr, ns = 96, 0.002, 7, 3
    shots = np.array([[100.0, 0.0], [400.0, 0.0], [700.0, 0.0]])
    receivers = np.array([[x, 0.0] for x in np.linspace(0.0, 1200.0, nr)])
    data = np.zeros((ns, nt, nr, 1), dtype=np.float64)
    w = make_wavelet(nt, dt, 15.0)
    for s in range(ns):
        for r in range(nr):
            data[s, :, r, 0] = w * (0.5 + 0.1 * r)
    mask = np.ones((ns, nr, 1), dtype=bool)
    mask[1, 2, 0] = False
    return SegyGathers(
        shots=shots,
        receivers=receivers,
        data=data,
        dt=dt,
        mask=mask,
        shot_ids=np.array([11, 12, 13]),
        receiver_ids=np.arange(1, nr + 1),
        component_ids=np.array([1]),
    )


class TestSegyIO(unittest.TestCase):
    def test_roundtrip(self):
        gathers = make_gathers()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "line.sgy")
            write_segy_gathers(path, gathers, coordinate_scalar=1)
            self.assertTrue(os.path.exists(path))
            out = read_segy_gathers(path)
        self.assertEqual(out.data.shape, gathers.data.shape)
        self.assertAlmostEqual(out.dt, gathers.dt, places=6)
        np.testing.assert_allclose(out.shots, gathers.shots, atol=1e-6)
        np.testing.assert_allclose(out.receivers, gathers.receivers, atol=1e-6)
        np.testing.assert_allclose(out.data, gathers.data, atol=1e-6)
        self.assertEqual(int(out.mask.sum()), gathers.n_shots * gathers.n_receivers)
        self.assertGreater(out.shot_ids.size, 0)
        self.assertTrue(out.text_header is not None)

    def test_component_field_and_scalar(self):
        gathers = make_gathers()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "line_scalar.sgy")
            write_segy_gathers(path, gathers, coordinate_scalar=100)
            out = read_segy_gathers(
                path, coordinate_scale=1.0,
                component_field="TraceIdentificationCode",
            )
        np.testing.assert_allclose(out.receivers, gathers.receivers, atol=1e-6)
        np.testing.assert_array_equal(out.component_ids, np.array([1]))
        np.testing.assert_allclose(out.data, gathers.data, atol=1e-6)

    def test_shot_list_and_qc(self):
        gathers = make_gathers()
        shots = read_segy_shot_list  # just ensure symbol available
        self.assertTrue(callable(shots))
        qc = gather_quality_control(gathers)
        self.assertEqual(qc["n_shots"], 3)
        self.assertEqual(qc["n_receivers"], 7)
        self.assertGreater(qc["max_amplitude"], 0)
        self.assertIn("missing_fraction", qc)


if __name__ == "__main__":
    unittest.main()
