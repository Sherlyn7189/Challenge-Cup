"""生成用于演示 SEG-Y 输入链路的合成数据。

用法::

    .venv\\Scripts\\python.exe scripts\\make_demo_segy.py

默认读取 ``config/default.json``，用其中的模型与观测系统正演若干炮，把所有炮
写进**同一个 SEG-Y 文件**（按 ``SourceX`` 分组），供
``config/segy_demo.json`` 以“实测资料”的方式读入。

这样就把"合成数据 → SEG-Y → 读取 → 成像"的完整链路串起来：文件是真实
SEG-Y 格式（道头、二进制头齐全），流水线走的是与实测数据相同的读取分支。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ertm import segy  # noqa: E402
from ertm.config import load_config  # noqa: E402
from ertm.models import LayeredModelBuilder  # noqa: E402
from ertm.pipeline import _build_receivers, _build_source, ERTMPipeline  # noqa: E402
from ertm.solver import build_solver  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成演示用 SEG-Y（把多炮写进一个文件）")
    parser.add_argument("--config", default="config/default.json", help="源配置")
    parser.add_argument("--output", default="data/demo_shots.sgy", help="输出 SEG-Y 路径")
    parser.add_argument("--shots", default="700,1000,1300", help="炮点 x 坐标，逗号分隔")
    args = parser.parse_args(argv)

    config = load_config(PROJECT_ROOT / args.config)
    model = LayeredModelBuilder().build(config)
    solver = build_solver(config.solver, config.q)
    receivers = _build_receivers(config)
    positions = [float(item) for item in str(args.shots).split(",") if item.strip()]

    records = []
    for position in positions:
        source = _build_source(config, x=position)
        shot = solver.forward(model, source, receivers)
        records.append(shot)
        print(f"正演完成: x={position:.0f} m, {shot.data.shape[0]} 样点 × {shot.data.shape[1]} 道")

    target = PROJECT_ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    segy.write_segy_shots(records, target)

    n_trace = sum(record.data.shape[1] for record in records)
    print(f"SEG-Y 已写出: {target} ({len(records)} 炮, {n_trace} 道, dt={records[0].dt:.2e} s)")
    # 立即用读取器验证一遍
    loaded = segy.read_segy_shots(target)
    print(f"回读校验: {len(loaded)} 炮, 首炮 {loaded[0].receivers.nrec} 道, "
          f"数据一致={np.allclose(loaded[0].data, records[0].data, atol=2e-3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
