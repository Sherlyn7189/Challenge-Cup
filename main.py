"""项目入口：环境自检 / 快速演示 / 单元测试。

    .venv\\Scripts\\python.exe main.py --check
    .venv\\Scripts\\python.exe main.py --demo
    .venv\\Scripts\\python.exe main.py --demo --full --iterations 25
    .venv\\Scripts\\python.exe main.py --test
"""

from __future__ import annotations

import argparse
import sys


def check_versions() -> None:
    import numpy, scipy, matplotlib, numba, pandas

    print(f"Python      {sys.version.split()[0]}")
    print(f"NumPy       {numpy.__version__}")
    print(f"SciPy       {scipy.__version__}")
    print(f"Matplotlib  {matplotlib.__version__}")
    print(f"Numba       {numba.__version__}")
    print(f"Pandas      {pandas.__version__}")

    for name in ("pyvista", "segyio"):
        try:
            from importlib.metadata import version

            print(f"{name:11s} {version(name)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:11s} 不可用: {exc}")


def run_tests() -> int:
    import unittest

    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SeismicProjectEFWI 入口")
    parser.add_argument("--check", action="store_true", help="只做环境自检")
    parser.add_argument("--demo", action="store_true", help="运行 EFWI+RTM 快速演示")
    parser.add_argument("--full", action="store_true", help="演示使用较大网格")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--test", action="store_true", help="运行单元测试")
    args = parser.parse_args(argv)

    if args.test:
        return run_tests()

    check_versions()
    if args.check and not args.demo:
        return 0

    # 无参数时直接跑快速演示，方便在 PyCharm 里右键运行
    from efwi.demo import run_demo

    run_demo(
        quick=not args.full,
        outdir=args.out,
        n_iter=args.iterations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
