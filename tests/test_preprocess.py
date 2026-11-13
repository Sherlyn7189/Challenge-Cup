import unittest
import numpy as np

from efwi.io import SegyGathers
from efwi.preprocess import (
    PreprocessConfig,
    estimate_source_wavelet,
    fk_filter,
    mute_direct_wave,
    offsets_from_geometry,
    phase_rotate,
    preprocess_gathers,
    trace_balance,
)
from efwi.wavelet import make_wavelet


def make_case():
    nt, dt, nr, ns = 160, 0.002, 21, 2
    shots = np.array([[200.0, 100.0], [600.0, 100.0]])
    receivers = np.array([[x, 100.0] for x in np.linspace(100.0, 900.0, nr)])
    t = np.arange(nt) * dt
    data = np.zeros((ns, nt, nr, 1), dtype=np.float64)
    for s in range(ns):
        for r in range(nr):
            x = receivers[r, 0]
            off = abs(x - shots[s, 0])
            # direct at v=1800, low-velocity surface wave at 500
            data[s, :, r, 0] += 1.0 * np.exp(-((t - 0.15 - off / 1800.0) / 0.015) ** 2)
            data[s, :, r, 0] += 0.4 * np.sin(2 * np.pi * 12.0 * (t - off / 500.0)) * np.exp(
                -((t - off / 500.0) / 0.08) ** 2
            )
    mask = np.ones((ns, nr, 1), dtype=bool)
    return SegyGathers(
        shots, receivers, data, dt, mask,
        np.array([1, 2]), np.arange(1, nr + 1), component_ids=np.array([1]),
    )


class TestPreprocess(unittest.TestCase):
    def test_geometry_and_mute(self):
        g = make_case()
        off = offsets_from_geometry(g.shots, g.receivers)
        self.assertEqual(off.shape, (g.n_shots, g.n_receivers))
        muted = mute_direct_wave(g.data, g.dt, off, velocity=1800.0, factor=0.8)
        self.assertEqual(muted.shape, g.data.shape)
        self.assertTrue(np.all(np.isfinite(muted)))
        # 近偏移距直达波到时之前的能量应被压制
        self.assertLess(np.mean(np.abs(muted[0, :20, 0, 0])), np.mean(np.abs(g.data[0, :20, 0, 0])))

    def test_fk_and_balance(self):
        g = make_case()
        dx = float(np.median(np.diff(g.receivers[:, 0])))
        out = fk_filter(g.data, g.dt, dx, vmin=1000, vmax=6000)
        self.assertEqual(out.shape, g.data.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        bal = trace_balance(out, max_gain=8.0)
        self.assertEqual(bal.shape, out.shape)
        self.assertTrue(np.all(np.isfinite(bal)))

    def test_phase_rotate(self):
        g = make_case()
        out = phase_rotate(g.data, 90.0)
        self.assertEqual(out.shape, g.data.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_wavelet_estimate(self):
        rng = np.random.default_rng(0)
        nt = 200
        dt = 0.002
        ref = make_wavelet(nt, dt, 15.0)
        # 同一 Green's function，用不同子波生成观测/合成
        G = rng.normal(0, 0.05, size=(nt, 5))
        G = np.convolve(G[:, 0], np.ones(5) / 5, mode="same")[:, None] * np.ones((1, 5))
        true = make_wavelet(nt, dt, 12.0)
        obs = np.stack([np.convolve(G[:, i], true, mode="same") for i in range(5)], axis=1)
        syn = np.stack([np.convolve(G[:, i], ref, mode="same") for i in range(5)], axis=1)
        est = estimate_source_wavelet(obs, syn, ref, dt, eps=1e-6)
        self.assertEqual(est.wavelet.shape, (nt,))
        self.assertTrue(np.all(np.isfinite(est.wavelet)))
        self.assertGreater(est.coherence, 0.5)
        self.assertGreater(np.max(np.abs(est.wavelet)), 0)

    def test_pipeline(self):
        g = make_case()
        cfg = PreprocessConfig(
            direct_mute_velocity=1800.0,
            fk_filter=True,
            trace_balance=True,
            phase_deg=15.0,
        )
        out, qc = preprocess_gathers(g, cfg)
        self.assertEqual(out.data.shape, g.data.shape)
        self.assertTrue(np.all(np.isfinite(out.data)))
        self.assertEqual(out.mask.shape, g.mask.shape)
        self.assertIn("output_rms", qc)


if __name__ == "__main__":
    unittest.main()
