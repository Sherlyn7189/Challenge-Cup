import unittest
import numpy as np

from efwi.filters import WienerFilter, butter_bandpass, wiener_bandpass
from efwi.wavelet import make_wavelet


class TestWienerBandpass(unittest.TestCase):
    def setUp(self):
        self.dt = 0.001
        self.nt = 600
        rng = np.random.default_rng(0)
        t = np.arange(self.nt) * self.dt
        traces = []
        for k in range(8):
            w = make_wavelet(self.nt, self.dt, 10.0 + k, taper=False)
            traces.append(0.6 * w + 0.4 * rng.normal(0, 0.2, self.nt))
        self.data = np.stack(traces, axis=1)

    def test_shape_and_zero_phase(self):
        out = wiener_bandpass(self.data, self.dt, 4.0, 15.0)
        self.assertEqual(out.shape, self.data.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        # 零相位：输出与输入的互相关峰值应在零延迟附近
        a = self.data[:, 0]
        b = out[:, 0]
        lags = np.arange(-20, 21)
        cc = [np.dot(a[20:-20], np.roll(b, lag)[20:-20]) for lag in lags]
        self.assertLessEqual(abs(lags[int(np.argmax(cc))]), 3)

    def test_out_of_band_attenuation(self):
        filt = WienerFilter.design(self.data, self.dt, 5.0, 12.0)
        out = filt.apply(self.data)
        x = np.fft.rfft(self.data, axis=0)
        y = np.fft.rfft(out, axis=0)
        freqs = np.fft.rfftfreq(self.nt, self.dt)
        e_in = np.mean(np.abs(x) ** 2, axis=1)
        e_out = np.mean(np.abs(y) ** 2, axis=1)
        inband = (freqs >= 5.0) & (freqs <= 12.0)
        outband = (freqs > 30.0) & (freqs < 120.0)
        ratio_in = np.sum(e_out[inband]) / max(np.sum(e_in[inband]), 1e-30)
        ratio_out = np.sum(e_out[outband]) / max(np.sum(e_in[outband]), 1e-30)
        self.assertGreater(ratio_in, 0.2)
        self.assertLess(ratio_out, 1e-2)

    def test_butter_bandpass(self):
        out = butter_bandpass(self.data, self.dt, 5.0, 15.0)
        self.assertEqual(out.shape, self.data.shape)


if __name__ == "__main__":
    unittest.main()
