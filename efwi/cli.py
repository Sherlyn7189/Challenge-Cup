"""命令行入口。

用法::

    python -m efwi.cli demo --quick
    python -m efwi.cli demo --full --iterations 25 --out runs/full
    python -m efwi.cli welllog --las data/example_well.las
    python -m efwi.cli loop --outer 2 --iterations 2
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="efwi",
        description="二维弹性波全波形反演 / RTM / 测井尺度建模",
    )
    parser.add_argument("--version", action="version", version=f"efwi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="运行 EFWI + RTM 端到端示例")
    p_demo.add_argument("--full", action="store_true", help="使用较大网格（较慢）")
    p_demo.add_argument("--iterations", type=int, default=3, help="每个频带迭代次数")
    p_demo.add_argument("--out", default="runs/demo", help="输出目录")
    p_demo.add_argument("--shots", type=int, default=None)
    p_demo.add_argument("--receivers", type=int, default=None)
    p_demo.add_argument("--anisotropy", choices=["iso", "vti", "hti"], default="iso")
    p_demo.add_argument("--no-constraints", action="store_true", help="关闭 Dykstra/ADMM 约束")

    p_well = sub.add_parser("welllog", help="LAS 测井解析与初始模型构建")
    p_well.add_argument("--las", default="data/example_well.las")
    p_well.add_argument("--out", default="runs/welllog")
    p_well.add_argument("--x", type=float, default=0.0, help="测井在模型中的 x 坐标 (m)")
    p_well.add_argument("--anisotropy", choices=["iso", "vti", "hti"], default="iso")

    p_loop = sub.add_parser("loop", help="反演—成像—三维建模闭环")
    p_loop.add_argument("--out", default="runs/closed_loop")
    p_loop.add_argument("--outer", type=int, default=2)
    p_loop.add_argument("--iterations", type=int, default=2)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        from .demo import run_demo

        run_demo(
            quick=not args.full,
            outdir=args.out,
            n_iter=args.iterations,
            with_constraints=not args.no_constraints,
            anisotropy=args.anisotropy,
            n_shots=args.shots,
            n_receivers=args.receivers,
        )
    elif args.command == "welllog":
        from .demo import run_welllog_demo

        run_welllog_demo(
            args.las,
            outdir=args.out,
            anisotropy=args.anisotropy,
            x=args.x,
        )
    elif args.command == "loop":
        from .demo import run_closed_loop_demo

        run_closed_loop_demo(outdir=args.out, n_outer=args.outer, n_iter=args.iterations)
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
