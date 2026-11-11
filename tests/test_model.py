import unittest
import numpy as np

from efwi.model import ElasticModel, constant_model, gardner_density, model_from_1d


class TestElasticModel(unittest.TestCase):
    def test_isotropic_stiffness(self):
        m = constant_model(5, 4, vp=3000.0, vs=1500.0, rho=2200.0)
        c11, c13, c33, c44 = m.stiffness()
        self.assertAlmostEqual(float(c11[0, 0]), 2200.0 * 3000.0 ** 2)
        self.assertAlmostEqual(float(c13[0, 0]), 2200.0 * (3000.0 ** 2 - 2 * 1500.0 ** 2))
        self.assertAlmostEqual(float(c44[0, 0]), 2200.0 * 1500.0 ** 2)
        np.testing.assert_allclose(c11, c33)

    def test_vti_hti_stiffness(self):
        vti = constant_model(3, 3, vp=3000, vs=1500, rho=2200, symmetry="vti", epsilon=0.2, delta=0.1)
        c11, c13, c33, c44 = vti.stiffness()
        self.assertAlmostEqual(float(c33[0, 0]), 2200 * 3000 ** 2)
        self.assertAlmostEqual(float(c11[0, 0]), 2200 * 3000 ** 2 * 1.4)
        self.assertTrue(np.all(c13 > 0))
        hti = constant_model(3, 3, vp=3000, vs=1500, rho=2200, symmetry="hti", epsilon=0.2, delta=0.1)
        h11, h13, h33, h44 = hti.stiffness()
        np.testing.assert_allclose(h11, c33)
        np.testing.assert_allclose(h33, c11)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ElasticModel(np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            ElasticModel(np.ones((3, 3)) * 1000, np.ones((3, 3)) * 2000, np.ones((3, 3)))

    def test_smooth_and_gardner(self):
        m = constant_model(20, 10, vp=3000, vs=1600, rho=2200)
        s = m.smoothed(1.5)
        self.assertEqual(s.shape, m.shape)
        rho = gardner_density(3000.0)
        self.assertTrue(2200 < rho < 2400)
        arr = model_from_1d(np.linspace(2000, 3000, 10), None, None, nz=10, nx=4)
        self.assertEqual(arr.vp.shape, (4, 10))
        np.testing.assert_allclose(arr.vp[:, 0], 2000.0)


if __name__ == "__main__":
    unittest.main()
