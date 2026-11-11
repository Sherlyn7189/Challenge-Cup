import unittest
import numpy as np

from efwi.fd import kernels


class TestAdjointTranspose(unittest.TestCase):
    def test_one_step_inner_product(self):
        rng = np.random.default_rng(7)
        nx, nz = 9, 8
        rho = rng.uniform(1800, 2400, (nx, nz))
        c11 = rng.uniform(1e10, 3e10, (nx, nz))
        c13 = rng.uniform(0.5e10, 1.5e10, (nx, nz))
        c33 = c11.copy()
        c44 = rng.uniform(1e9, 5e9, (nx, nz))
        rho_x = np.empty_like(rho)
        rho_x[:-1] = 0.5 * (rho[:-1] + rho[1:])
        rho_x[-1] = rho_x[-2]
        rho_z = np.empty_like(rho)
        rho_z[:, :-1] = 0.5 * (rho[:, :-1] + rho[:, 1:])
        rho_z[:, -1] = rho_z[:, -2]
        c44c = np.empty_like(c44)
        c44c[:-1, :-1] = 0.25 * (
            c44[:-1, :-1] + c44[1:, :-1] + c44[:-1, 1:] + c44[1:, 1:]
        )
        c44c[-1, :] = c44c[-2, :]
        c44c[:, -1] = c44c[:, -2]
        dt, dx, dz = 1e-3, 10.0, 10.0
        h = np.zeros((1, 1))

        def rnd():
            a = rng.normal(size=(nx, nz))
            a[0, :] = a[-1, :] = 0.0
            a[:, 0] = a[:, -1] = 0.0
            return a

        x = [rnd() for _ in range(5)]
        y = [rnd() for _ in range(5)]
        vx, vz, sxx, szz, sxz = [a.copy() for a in x]
        ax = np.zeros_like(vx)
        az = np.zeros_like(vx)
        exx = np.zeros_like(vx)
        ezz = np.zeros_like(vx)
        exz = np.zeros_like(vx)
        kernels.update_velocity(
            vx, vz, sxx, szz, sxz, rho_x, rho_z, dt, dx, dz, ax, az, h, 0
        )
        kernels.update_stress(
            vx, vz, sxx, szz, sxz, c11, c13, c33, c44c, dt, dx, dz,
            exx, ezz, exz, h, h, h, h, 0,
        )
        vax, vaz, saxx, sazz, saxz = y
        va1 = np.zeros_like(vax)
        va2 = np.zeros_like(vax)
        sa1 = np.zeros_like(vax)
        sa2 = np.zeros_like(vax)
        sa3 = np.zeros_like(vax)
        kernels.adjoint_step(
            vax, vaz, saxx, sazz, saxz,
            va1, va2, sa1, sa2, sa3,
            c11, c13, c33, c44c, rho_x, rho_z, dt, dx, dz,
        )
        sl = np.s_[1:-1, 1:-1]
        # 恒等式：<F x, y> = <x, F^T y>
        lhs = sum(
            np.sum(a[sl] * b[sl])
            for a, b in zip((vx, vz, sxx, szz, sxz), y)
        )
        rhs = sum(
            np.sum(a[sl] * b[sl])
            for a, b in zip(x, (va1, va2, sa1, sa2, sa3))
        )
        self.assertAlmostEqual(float(lhs), float(rhs), places=5)


if __name__ == "__main__":
    unittest.main()
