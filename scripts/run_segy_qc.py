"""SEG-Y 实际资料读取、QC 与预处理示例。

    .venv\\Scripts\\python.exe scripts\\run_segy_qc.py path\\to\\line.sgy --out runs/segy_qc
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efwi.io import gather_quality_control, read_segy_gathers, write_segy_gathers  # noqa: E402
from efwi.preprocess import PreprocessConfig, preprocess_gathers  # noqa: E402


def plot_gather(gathers, path: str, title: str, shot: int = 0, component: int = 0) -> None:
    data = gathers.data[shot, :, :, component]
    clip = np.percentile(np.abs(data), 99) + 1e-12
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        data.T, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip,
        extent=(0, (data.shape[0] - 1) * gathers.dt, data.shape[1], 0),
    )
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("receiver")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segy")
    parser.add_argument("--out", default="runs/segy_qc")
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--no-fk", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gathers = read_segy_gathers(args.segy, max_shots=args.max_shots)
    qc_in = gather_quality_control(gathers)
    print("输入 QC:", json.dumps(qc_in, ensure_ascii=False, indent=2))
    np.savez_compressed(
        os.path.join(args.out, "gathers_raw.npz"),
        data=gathers.data, shots=gathers.shots, receivers=gathers.receivers,
        dt=gathers.dt, mask=gathers.mask,
    )
    plot_gather(gathers, os.path.join(args.out, "raw_gather.png"), "raw SEG-Y gather")

    cfg = PreprocessConfig(fk_filter=not args.no_fk)
    processed, qc_proc = preprocess_gathers(gathers, cfg)
    print("预处理 QC:", json.dumps(qc_proc, ensure_ascii=False, indent=2))
    np.savez_compressed(
        os.path.join(args.out, "gathers_processed.npz"),
        data=processed.data, shots=processed.shots, receivers=processed.receivers,
        dt=processed.dt, mask=processed.mask,
    )
    plot_gather(processed, os.path.join(args.out, "processed_gather.png"), "preprocessed gather")
    write_segy_gathers(os.path.join(args.out, "gathers_processed.sgy"), processed)
    with open(os.path.join(args.out, "qc.json"), "w", encoding="utf-8") as fh:
        json.dump({"input": qc_in, "preprocess": qc_proc}, fh, ensure_ascii=False, indent=2)
    print("输出:", args.out)


if __name__ == "__main__":
    main()
