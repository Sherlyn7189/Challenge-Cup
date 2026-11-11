"""快速 EFWI + RTM 演示。

    .venv\\Scripts\\python.exe scripts\\run_demo.py
    .venv\\Scripts\\python.exe scripts\\run_demo.py --full --iterations 25
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efwi.demo import run_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--anisotropy", choices=["iso", "vti", "hti"], default="iso")
    parser.add_argument("--no-constraints", action="store_true")
    args = parser.parse_args()
    run_demo(
        quick=not args.full,
        outdir=args.out,
        n_iter=args.iterations,
        anisotropy=args.anisotropy,
        with_constraints=not args.no_constraints,
    )


if __name__ == "__main__":
    main()
