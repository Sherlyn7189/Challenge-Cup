"""Q 补偿的 QC 与结果输出。

提供三类产物，便于验收"补偿是否真的改善了成像"而不是只看图漂亮：

1. **Q 模型导出**：``q_model.npz``（Qp/Qs）与 ``q_model.png``（剖面图）；
2. **补偿前后对比**：``q_compare.png``（无补偿 / 补偿 / 差值三幅 + 深度振幅曲线）
   与 ``q_compare.json``（能量、深部/浅部比、窗口振幅增益、光谱质心等指标）；
3. **增益曲线**：``q_gain.png``（各机制的补偿强度与最大增益随时间的变化）。

所有函数只读内存中的数据，不参与数值计算。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .exceptions import DataError  # noqa: E402
from .logging_utils import get_logger  # noqa: E402
from .types import ElasticModel, ImageResult  # noqa: E402

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def comparison_metrics(
    baseline: ImageResult,
    compensated: ImageResult,
    model: ElasticModel,
    *,
    band: tuple[float, float] | None = None,
) -> dict[str, float]:
    """补偿前后的定量对比。

    ``band`` 为关注的深度窗口（米），缺省取模型深度的 40%–80%，
    用于衡量"深部界面"的振幅恢复情况。
    """
    base = np.asarray(baseline.image, dtype=np.float64)
    comp = np.asarray(compensated.image, dtype=np.float64)
    if base.shape != comp.shape:
        raise DataError(f"两幅图像形状不一致: {base.shape} vs {comp.shape}")
    nz = base.shape[0]
    dz = float(model.grid.dz)
    if band is None:
        band = (0.4 * nz * dz, 0.8 * nz * dz)
    i0 = int(max(band[0] / dz, 0))
    i1 = int(min(band[1] / dz, nz))
    if i1 <= i0:
        raise DataError(f"对比窗口无效: {band}")

    def block(data: np.ndarray) -> dict[str, float]:
        magnitude = np.abs(data)
        return {
            "total_energy": float(np.sum(data**2)),
            "band_rms": float(np.sqrt(np.mean(data[i0:i1, :] ** 2))),
            "band_peak": float(np.max(magnitude[i0:i1, :])),
            "deep_shallow_ratio": _deep_shallow(magnitude),
        }

    base_stats = block(base)
    comp_stats = block(comp)
    gain = comp_stats["band_rms"] / max(base_stats["band_rms"], 1e-30)
    diff = comp - base
    metrics = {
        "band_top_m": float(band[0]),
        "band_bottom_m": float(band[1]),
        "baseline_band_rms": base_stats["band_rms"],
        "compensated_band_rms": comp_stats["band_rms"],
        "band_amplitude_gain": float(gain),
        "baseline_deep_shallow_ratio": base_stats["deep_shallow_ratio"],
        "compensated_deep_shallow_ratio": comp_stats["deep_shallow_ratio"],
        "difference_rms": float(np.sqrt(np.mean(diff**2))),
        "correlation": _correlation(base, comp),
    }
    return metrics


def _deep_shallow(magnitude: np.ndarray) -> float:
    half = max(magnitude.shape[0] // 2, 1)
    shallow = float(np.sum(magnitude[:half] ** 2))
    deep = float(np.sum(magnitude[half:] ** 2))
    return deep / shallow if shallow > 0 else 0.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel() - float(np.mean(a))
    y = b.ravel() - float(np.mean(b))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


# ----------------------------------------------------------------------------
# 图件
# ----------------------------------------------------------------------------
def save_q_model(
    model: ElasticModel,
    directory: str | Path,
    *,
    name: str = "q_model",
) -> tuple[Path, Path]:
    """输出 Q 模型：``.npz``（Qp/Qs 数组）与 ``.png``（剖面）。"""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{name}.npz"
    png_path = root / f"{name}.png"
    try:
        np.savez_compressed(
            npz_path,
            qp=model.qp,
            qs=model.qs,
            vp=model.vp,
            vs=model.vs,
            grid=np.array([model.grid.nx, model.grid.nz, model.grid.dx, model.grid.dz]),
        )
    except OSError as exc:  # pragma: no cover - 磁盘异常
        raise DataError(f"保存 Q 模型失败: {npz_path}: {exc}") from exc

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    extent = model.grid.extent
    for ax, data, title, cmap in (
        (axes[0], model.qp, "Qp", "magma"),
        (axes[1], model.qs, "Qs", "magma"),
    ):
        image = ax.imshow(data, extent=extent, origin="upper", aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Attenuation model (quality factor)", fontsize=11)
    try:
        fig.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
    except OSError as exc:  # pragma: no cover
        raise DataError(f"保存 Q 模型图失败: {png_path}: {exc}") from exc
    finally:
        plt.close(fig)
    logger.info("Q 模型已输出: %s, %s", npz_path, png_path)
    return npz_path, png_path


def save_comparison(
    baseline: ImageResult,
    compensated: ImageResult,
    model: ElasticModel,
    directory: str | Path,
    *,
    metrics: dict[str, float] | None = None,
    name: str = "q_compare",
) -> tuple[Path, Path]:
    """输出补偿前后对比图与指标 JSON。"""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    png_path = root / f"{name}.png"
    json_path = root / f"{name}.json"

    base = np.asarray(baseline.image, dtype=np.float64)
    comp = np.asarray(compensated.image, dtype=np.float64)
    if metrics is None:
        metrics = comparison_metrics(baseline, compensated, model)

    scale = float(np.percentile(np.abs(comp), 99.5)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = (
        (base, "Before compensation", "seismic"),
        (comp, "After compensation", "seismic"),
        (comp - base, "Difference (after - before)", "seismic"),
    )
    for ax, (data, title, cmap) in zip(axes, panels):
        image = ax.imshow(
            data,
            extent=model.grid.extent,
            origin="upper",
            aspect="auto",
            cmap=cmap,
            vmin=-scale,
            vmax=scale,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    axes[1].text(
        0.02,
        0.95,
        f"band gain = {metrics['band_amplitude_gain']:.2f}x\n"
        f"deep/shallow = {metrics['compensated_deep_shallow_ratio']:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.suptitle("Q compensation: before / after / difference", fontsize=11)
    try:
        fig.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
    except OSError as exc:  # pragma: no cover
        raise DataError(f"保存对比图失败: {png_path}: {exc}") from exc
    finally:
        plt.close(fig)

    import json

    try:
        json_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover
        raise DataError(f"保存对比指标失败: {json_path}: {exc}") from exc
    logger.info("补偿对比已输出: %s, %s", png_path, json_path)
    return png_path, json_path


def save_gain_curve(
    q_meta: dict[str, Any],
    directory: str | Path,
    *,
    dt: float,
    nt: int,
    name: str = "q_gain",
) -> Path | None:
    """输出补偿增益曲线（各机制的强度与累积增益随时间变化）。"""
    alphas = q_meta.get("alpha_mechanisms") or []
    peaks = q_meta.get("f_peak_mechanisms_hz") or []
    if not alphas:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"
    time = np.arange(int(nt)) * float(dt)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for alpha, peak in zip(alphas, peaks):
        # 单位 Q 下的累积增益曲线，直观显示各机制的放大能力
        gain_db = 20.0 / np.log(10.0) * np.pi * float(peak) * time * float(alpha)
        axes[0].plot(time, gain_db, label=f"f≈{peak:.0f} Hz, α={alpha:.2f}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Cumulative gain (dB, per unit t/Q)")
    axes[0].set_title("Compensation strength per mechanism")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(len(alphas)), alphas, color="#3b6ea5")
    axes[1].set_xticks(range(len(alphas)))
    axes[1].set_xticklabels([f"{p:.0f} Hz" for p in peaks])
    axes[1].set_ylabel("Effective strength α_l")
    axes[1].set_title(
        f"Band limit: f_max={q_meta.get('f_max')}, budget={q_meta.get('gain_budget_db')} dB"
    )
    axes[1].grid(alpha=0.3, axis="y")
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
    except OSError as exc:  # pragma: no cover
        raise DataError(f"保存增益曲线失败: {path}: {exc}") from exc
    finally:
        plt.close(fig)
    logger.info("Q 增益曲线已输出: %s", path)
    return path
