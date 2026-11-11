"""测井 LAS -> 初始模型 / 尺度设计演示。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efwi.demo import run_welllog_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--las", default="data/example_well.las")
    parser.add_argument("--out", default="runs/welllog")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--anisotropy", choices=["iso", "vti", "hti"], default="vti")
    args = parser.parse_args()
    run_welllog_demo(args.las, outdir=args.out, x=args.x, anisotropy=args.anisotropy)


if __name__ == "__main__":
    main()
