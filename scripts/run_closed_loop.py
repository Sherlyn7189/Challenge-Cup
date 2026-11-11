"""反演—成像—三维建模闭环演示。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efwi.demo import run_closed_loop_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/closed_loop")
    parser.add_argument("--outer", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    run_closed_loop_demo(outdir=args.out, n_outer=args.outer, n_iter=args.iterations)


if __name__ == "__main__":
    main()
