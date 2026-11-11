import unittest
import numpy as np

from efwi.optimize import LBFGS, limit_step, parabolic_line_search


class TestOptimize(unittest.TestCase):
    def test_lbfgs_quadratic_direction(self):
        rng = np.random.default_rng(0)
        n = 20
        A = np.diag(np.linspace(1.0, 10.0, n))
        x = rng.normal(size=n)
        g = A @ x
        opt = LBFGS(memory=5)
        d0 = opt.direction(g)
        self.assertLess(np.dot(g, d0), 0)
        s = -0.01 * g
        y = A @ s
        opt.update(s, y)
        d1 = opt.direction(g)
        self.assertLess(np.dot(g, d1), 0)

    def test_limit_step(self):
        d = np.array([3.0, -1.0, 0.5])
        out = limit_step(d, 1.0)
        self.assertAlmostEqual(np.max(np.abs(out)), 1.0)
        np.testing.assert_allclose(out, d / 3.0)

    def test_parabolic_line_search(self):
        x = np.array([2.0])
        d = np.array([-1.0])
        f0 = float(x[0] ** 2)
        g0 = np.array([2.0 * x[0]])
        calls = []

        def evaluate(alpha):
            calls.append(alpha)
            return float((x[0] + alpha * d[0]) ** 2)

        ls = parabolic_line_search(evaluate, x, d, f0, g0, alpha_init=1.0, max_eval=6)
        self.assertTrue(ls.success)
        self.assertLess(ls.f, f0)
        self.assertAlmostEqual(x[0] + ls.alpha * d[0], 0.0, delta=0.15)

    def test_non_descent_rejected(self):
        x = np.array([0.0])
        d = np.array([1.0])
        g0 = np.array([1.0])
        ls = parabolic_line_search(lambda a: float(a), x, d, 0.0, g0, alpha_init=1.0)
        self.assertFalse(ls.success)


if __name__ == "__main__":
    unittest.main()
