"""把配置里的层状模型导出成 npz 速度模型文件。

用途：演示"速度模型从文件读入"的流程——先由配置生成模型（或替换成层析/反演
结果），导出为 npz，再用 ``model.source="file"`` 读回。

用法::

    .venv\\Scripts\\python.exe scripts\\export_velocity_model.py
    .venv\\Scripts\\python.exe main.py --config config/file_model_demo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ertm.config import load_config  # noqa: E402
from ertm.models import build_model_builder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 npz 速度模型")
    parser.add_argument("--config", default="config/default.json", help="源配置")
    parser.add_argument("--output", default="data/velocity_model.npz", help="输出路径")
    args = parser.parse_args(argv)

    config = load_config(PROJECT_ROOT / args.config)
    model = build_model_builder(config).build(config)
    target = PROJECT_ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        qp=model.qp,
        qs=model.qs,
        dx=np.float32(model.grid.dx),
        dz=np.float32(model.grid.dz),
        x0=np.float32(model.grid.x0),
        z0=np.float32(model.grid.z0),
    )
    print(
        f"速度模型已导出: {target} "
        f"({model.grid.nx}×{model.grid.nz}, dx={model.grid.dx}, dz={model.grid.dz}, "
        f"vp={float(model.vp.min()):.0f}–{float(model.vp.max()):.0f} m/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
