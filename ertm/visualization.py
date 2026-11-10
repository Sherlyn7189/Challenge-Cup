"""可视化模块。

只负责“把已有数据画出来”，不计算任何物理量。统一使用非交互后端
``Agg``，保证在 PyCharm、无显示器服务器和 CI 中都能保存图片。

约定：图片标题/坐标轴用英文，避免不同系统缺少中文字体导致方框；
报告和日志用中文。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .exceptions import VisualizationError  # noqa: E402
from .logging_utils import get_logger  # noqa: E402
from .types import ElasticModel, ImageResult, Metrics, ShotRecord  # noqa: E402

logger = get_logger(__name__)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MatplotlibVisualizer:
    """基于 Matplotlib 的静态图输出。"""

    name = "matplotlib-visualizer"

    def __init__(self, output_dir: str | Path, *, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.enabled = bool(enabled)
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _target(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _save(self, fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except OSError as exc:
            raise VisualizationError(f"保存图片失败: {path}: {exc}") from exc
        finally:
            plt.close(fig)
        logger.info("图片已保存: %s", path)
        return path

    def plot_model(self, model: ElasticModel, name: str = "model") -> Path | None:
        if not self.enabled:
            return None
        model.validate()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        panels = (
            ("Vp (m/s)", model.vp, "viridis"),
            ("Vs (m/s)", model.vs, "plasma"),
            ("Density (kg/m3)", model.rho, "cividis"),
        )
        for ax, (title, data, cmap) in zip(axes, panels):
            im = ax.imshow(
                data,
                extent=model.grid.extent,
                origin="upper",
                aspect="auto",
                cmap=cmap,
            )
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(f"Geological model: {model.name}", fontsize=11)
        return self._save(fig, self._target(name))

    def plot_shot(self, shot: ShotRecord, name: str = "shot") -> Path | None:
        if not self.enabled:
            return None
        shot.validate()
        fig, ax = plt.subplots(figsize=(9, 5))
        data = shot.data.T
        vmax = float(np.max(np.abs(data))) or 1.0
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            extent=(float(shot.t[0]), float(shot.t[-1]),
                    float(np.min(shot.receivers.x)), float(np.max(shot.receivers.x))),
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Receiver x (m)")
        ax.set_title(f"Shot record ({shot.receivers.component})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Amplitude")
        return self._save(fig, self._target(name))

    def plot_image(self, image: ImageResult, name: str = "image") -> Path | None:
        if not self.enabled:
            return None
        image.validate()
        fig, ax = plt.subplots(figsize=(9, 5))
        data = image.image
        vmax = float(np.max(np.abs(data))) or 1.0
        im = ax.imshow(
            data,
            extent=image.grid.extent,
            origin="upper",
            aspect="auto",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"ERTM image ({image.mode.upper()})")
        if image.meta.get("placeholder", False):
            ax.text(
                0.5,
                0.5,
                "PLACEHOLDER\nno physical migration",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="black",
                alpha=0.65,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Amplitude")
        return self._save(fig, self._target(name))

    def plot_metrics(self, metrics: Metrics, name: str = "metrics") -> Path | None:
        if not self.enabled:
            return None
        items = sorted(metrics.values.items(), key=lambda kv: kv[0])
        if not items:
            return None
        rows = [[key, f"{float(value):.6g}"] for key, value in items]
        fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.32 * len(rows) + 1.4)))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.25)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1F3A5F")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F2F2F2")
        ax.set_title("Basic metrics", pad=10)
        return self._save(fig, self._target(name))


def build_visualizer(output_dir: str | Path, enabled: bool = True) -> MatplotlibVisualizer:
    """可视化器工厂。"""
    return MatplotlibVisualizer(output_dir, enabled=enabled)
