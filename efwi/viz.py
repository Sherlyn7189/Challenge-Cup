"""Matplotlib 绘图工具（无界面，直接保存 PNG）。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model import ElasticModel  # noqa: E402


def _extent(grid) -> tuple[float, float, float, float]:
    return grid.extent


def plot_model_panel(
    models: Sequence[ElasticModel],
    grid,
    path: str,
    *,
    keys: Iterable[str] = ("vp", "vs", "rho"),
    titles: Sequence[str] | None = None,
) -> str:
    keys = tuple(keys)
    nrow, ncol = len(keys), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    for j, model in enumerate(models):
        for i, key in enumerate(keys):
            ax = axes[i, j]
            arr = np.asarray(getattr(model, key))
            im = ax.imshow(arr.T, cmap="jet", aspect="auto", extent=grid.extent, origin="upper")
            ax.set_title((titles[j] if titles else f"model {j}") + f" — {key}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rtm(image: np.ndarray, grid, path: str, *, title: str = "RTM image", clip: float = 99.0) -> str:
    vmax = np.percentile(np.abs(image), clip)
    vmax = max(vmax, 1e-12)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image.T, cmap="seismic", aspect="auto", extent=grid.extent,
                   vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dip(dip_deg: np.ndarray, coherence: np.ndarray, grid, path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = axes[0].imshow(dip_deg.T, cmap="twilight", aspect="auto", extent=grid.extent,
                         vmin=-90, vmax=90, origin="upper")
    axes[0].set_title("reflector dip (deg)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(coherence.T, cmap="viridis", aspect="auto", extent=grid.extent,
                         origin="upper")
    axes[1].set_title("structure-tensor coherence")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_history(history: list[dict], path: str) -> str:
    if not history:
        return path
    idx = [h["index"] for h in history]
    misfit = [h["misfit"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(idx, misfit, "o-", ms=3, lw=1.2)
    stages = {}
    for h in history:
        stages.setdefault(h["stage"], h["index"])
    for name, i0 in stages.items():
        ax.axvline(i0, color="gray", ls="--", lw=0.8)
        ax.text(i0, max(misfit), f" {name}", fontsize=8, va="top")
    ax.set_xlabel("iteration")
    ax.set_ylabel("normalized misfit")
    ax.set_title("EFWI convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gather(data: np.ndarray, dt: float, path: str, *, title: str = "shot gather") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    nt = data.shape[0]
    t = np.arange(nt) * dt
    for k, (ax, comp) in enumerate(zip(axes, ("vx", "vz"))):
        clip = np.percentile(np.abs(data[:, :, k]), 99) + 1e-12
        im = ax.imshow(data[:, :, k].T, cmap="seismic", aspect="auto",
                       extent=(t[0], t[-1], data.shape[1], 0), vmin=-clip, vmax=clip)
        ax.set_title(f"{title} — {comp}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("receiver")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_data_fit(
    observed: np.ndarray,
    synthetic: np.ndarray,
    dt: float,
    path: str,
    *,
    shot: int = 0,
    component: int = 0,
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, arr, title in (
        (axes[0], observed[shot, :, :, component], "observed"),
        (axes[1], synthetic[shot, :, :, component], "synthetic"),
        (axes[2], synthetic[shot, :, :, component] - observed[shot, :, :, component], "residual"),
    ):
        clip = np.percentile(np.abs(arr), 99) + 1e-12
        im = ax.imshow(arr.T, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip,
                       extent=(0, (arr.shape[0] - 1) * dt, arr.shape[1], 0))
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("receiver")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
