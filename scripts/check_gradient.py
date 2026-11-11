"""伴随梯度有限差分方向导数检验。

    .venv\\Scripts\\python.exe scripts\\check_gradient.py
    .venv\\Scripts\\python.exe scripts\\check_gradient.py --full
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efwi.adjoint import check_gradient  # noqa: E402
from efwi.fd import Elastic2DOperator  # noqa: E402
from efwi.synthetic import demo_setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--receivers", type=int, default=31)
    parser.add_argument("--out", default="runs/gradient_check")
    args = parser.parse_args()

    setup = demo_setup(
        quick=not args.full,
        n_shots=args.shots if not args.full else None,
        n_receivers=args.receivers if not args.full else None,
    )
    grid, model, survey = setup.grid, setup.initial_model, setup.survey
    op = Elastic2DOperator(model, survey, grid, dtype="float64", storage_mb=512)
    observed = op.forward_all(setup.true_model)
    print(f"model={model.describe()}")
    print(f"survey: {survey.n_shots} shots, {survey.n_receivers} receivers, nt={survey.nt}")

    x = np.arange(grid.nx) * grid.dx
    z = np.arange(grid.nz) * grid.dz
    X, Z = np.meshgrid(x, z, indexing="ij")
    d1 = np.exp(-((X - 0.5 * x[-1]) ** 2 + (Z - 0.55 * z[-1]) ** 2) / (2 * (0.18 * x[-1]) ** 2))
    d1 /= np.max(d1)
    rng = np.random.default_rng(0)
    d2 = gaussian_filter(rng.normal(size=grid.shape), sigma=3.0, mode="nearest")
    d2 /= np.max(np.abs(d2))

    all_checks = {}
    for name, d in (("gaussian", d1), ("random_smooth", d2)):
        checks = check_gradient(
            op, model, observed, {"vp": d, "vs": d, "rho": d},
            eps={"vp": 2.0, "vs": 1.0, "rho": 1.0},
        )
        all_checks[name] = checks
        print(f"\n[{name}]")
        print(f"{'param':>5s} {'adjoint':>16s} {'finite-diff':>16s} {'rel error':>12s}")
        for key, row in checks.items():
            print(
                f"{key:>5s} {row['adjoint']:16.6e} {row['finite_difference']:16.6e} "
                f"{row['rel_error']:12.3e}"
            )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gradient_check.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "grid": [grid.nx, grid.nz],
                "dx": grid.dx,
                "dt": survey.dt,
                "nt": survey.nt,
                "n_shots": survey.n_shots,
                "checks": all_checks,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n结果写入 {args.out}/gradient_check.json")


if __name__ == "__main__":
    main()
