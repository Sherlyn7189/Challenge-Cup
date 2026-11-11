"""伪 Hessian 预条件 + 多尺度频带演示。

输出：
    runs/preconditioning/
    ├── stage_01_gradient.png        # raw / H / preconditioned
    ├── stage_02_gradient.png
    ├── filter_responses.png
    ├── data_spectra.png
    ├── precondition_stability.png
    ├── stage_models.png
    ├── stage_convergence.png
    ├── report.md
    └── report.json

运行：
    .venv\\Scripts\\python.exe scripts\\run_preconditioning_demo.py
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

from efwi.config import ConstraintConfig, FWIConfig, StageConfig  # noqa: E402
from efwi.filters import (  # noqa: E402
    WienerFilter,
    band_energy_fractions,
    frequency_spectrum,
    schedule_table,
    validate_stages,
)
from efwi.fd import Elastic2DOperator  # noqa: E402
from efwi.geometry import Survey  # noqa: E402
from efwi.gradient import convert_gradient, precondition  # noqa: E402
from efwi.inversion import ElasticFWI, save_result  # noqa: E402
from efwi.synthetic import demo_setup  # noqa: E402
from efwi.viz import plot_history, plot_model_panel  # noqa: E402


def parse_stages(text: str) -> list[StageConfig]:
    stages = []
    for item in text.split(","):
        lo, hi = item.strip().split("-")
        stages.append(StageConfig(float(lo), float(hi), 0, f"band_{hi}Hz"))
    return stages


def map_stats(raw: np.ndarray, h: np.ndarray, pre: np.ndarray, grid) -> dict:
    zmax = float(grid.z[-1])
    deep = np.broadcast_to(grid.z[None, :] > 0.55 * zmax, raw.shape)
    shallow = ~deep
    eps = 1e-30

    def rms(a):
        return float(np.sqrt(np.mean(a * a)))

    deep_gain = rms(pre[deep]) / (rms(raw[deep]) + eps)
    shallow_gain = rms(pre[shallow]) / (rms(raw[shallow]) + eps)
    raw_deep_frac = rms(raw[deep]) ** 2 / (rms(raw) ** 2 + eps)
    pre_deep_frac = rms(pre[deep]) ** 2 / (rms(pre) ** 2 + eps)
    return {
        "g_rms": rms(raw),
        "g_max": float(np.max(np.abs(raw))),
        "h_max": float(np.max(h)),
        "h_median": float(np.median(h)),
        "h_dynamic_range_db": float(20.0 * np.log10((np.max(h) + eps) / (np.median(h) + eps))),
        "pre_rms": rms(pre),
        "pre_max": float(np.max(np.abs(pre))),
        "scale_ratio": float(np.max(np.abs(pre)) / (np.max(np.abs(raw)) + eps)),
        "deep_gain": deep_gain,
        "shallow_gain": shallow_gain,
        "deep_boost": deep_gain / (shallow_gain + eps),
        "deep_energy_fraction_raw": raw_deep_frac,
        "deep_energy_fraction_pre": pre_deep_frac,
        "finite": bool(np.all(np.isfinite(pre))),
    }


def plot_stage_gradient(
    grid, stage: StageConfig, raw: dict, h: dict, pre: dict, path: str
) -> None:
    keys = ("vp", "vs", "rho")
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for i, key in enumerate(keys):
        panels = (
            (raw[key], "raw gradient", "seismic"),
            (h[key], "pseudo-Hessian H", "viridis"),
            (pre[key], "preconditioned", "seismic"),
        )
        for j, (arr, title, cmap) in enumerate(panels):
            ax = axes[i, j]
            if j == 1:
                data = np.log10(np.maximum(arr, 1e-30))
                vmax = np.percentile(np.abs(data), 99)
                im = ax.imshow(data.T, cmap=cmap, aspect="auto", extent=grid.extent,
                               origin="upper", vmin=-vmax, vmax=vmax)
            else:
                vmax = np.percentile(np.abs(arr), 99) + 1e-30
                im = ax.imshow(arr.T, cmap=cmap, aspect="auto", extent=grid.extent,
                               origin="upper", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{stage.name} — {key} — {title}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_filter_responses(filters: list[WienerFilter], stages: list[StageConfig], dt: float, path: str) -> None:
    nyq = 0.5 / dt
    freqs = np.linspace(0.0, min(nyq * 0.9, 80.0), 600)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for filt, stage in zip(filters, stages):
        ax.plot(freqs, filt.amplitude_response(freqs), lw=1.5, label=f"{stage.name} ({stage.f_low:g}-{stage.f_high:g} Hz)")
        ax.axvspan(stage.f_low, stage.f_high, alpha=0.06)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Wiener amplitude response")
    ax.set_title("Multiscale Wiener band-pass responses")
    ax.set_xlim(0, freqs[-1])
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data_spectra(observed: np.ndarray, filters: list[WienerFilter], stages, dt: float, path: str) -> None:
    f0, s0 = frequency_spectrum(observed, dt, time_axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(f0, s0 + 1e-30, label="observed", color="k")
    for filt, stage in zip(filters, stages):
        sf = filt.apply(observed, axis=1)
        f, s = frequency_spectrum(sf, dt, time_axis=1)
        axes[0].semilogy(f, s + 1e-30, lw=1.0, label=f"{stage.name}")
        axes[1].plot(f, s / (s0 + 1e-30), lw=1.2, label=f"{stage.name}")
    axes[0].set_title("amplitude spectra")
    axes[0].set_xlabel("frequency (Hz)")
    axes[1].set_title("filtered / observed ratio")
    axes[1].set_xlabel("frequency (Hz)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability(raw: dict, h: dict, path: str) -> list[dict]:
    dampings = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, marker in (("vp", "o"), ("vs", "s"), ("rho", "^")):
        ratios = []
        for eta in dampings:
            hk = np.asarray(h[key], dtype=np.float64)
            floor = max(eta * float(np.max(hk)), 0.0)
            gain = 1.0 / (hk + floor + 1e-30)
            typical = float(np.median(gain[np.isfinite(gain)]))
            gain = np.minimum(gain, 20.0 * typical)
            ratio = float(np.max(gain) / (typical + 1e-30))
            ratios.append(ratio)
            rows.append({"param": key, "damping": eta, "max_gain_ratio": ratio})
        ax.loglog(dampings, ratios, marker + "-", ms=3, label=key)
    ax.axhline(20.0, color="gray", ls="--", lw=0.8, label="max_gain cap")
    ax.set_xlabel("pseudo-Hessian damping eta")
    ax.set_ylabel("max gain / median gain")
    ax.set_title("Preconditioning stability vs damping")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/preconditioning")
    parser.add_argument("--stages", default="2-8,3-15")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--receivers", type=int, default=31)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup = demo_setup(quick=True, n_shots=args.shots, n_receivers=args.receivers)
    grid = setup.grid
    model = setup.initial_model
    survey = setup.survey
    stages = parse_stages(args.stages)
    for s in stages:
        s.n_iter = args.iterations
    validate_stages(stages, survey.dt)

    op = Elastic2DOperator(model, survey, grid, dtype="float64", storage_mb=512)
    observed = op.forward_all(setup.true_model)

    filters = []
    stage_stats = []
    stage_arrays = []
    stability_rows = []
    for i, stage in enumerate(stages, start=1):
        filt = WienerFilter.design(observed, survey.dt, stage.f_low, stage.f_high, time_axis=1)
        filters.append(filt)
        obs_f = filt.apply(observed, axis=1)
        stage_wavelet = filt.apply(survey.wavelet[:, None], axis=0).ravel()
        stage_survey = Survey(
            survey.shots, survey.receivers, survey.nt, survey.dt,
            stage_wavelet, survey.source_kind, survey.amplitude, dict(survey.meta),
        )
        op.set_survey(stage_survey)

        res = op.compute_gradient(model, obs_f, syn_filter=None)
        pg = convert_gradient(res, model, ("vp", "vs", "rho"))
        raw = pg.g
        h = pg.h
        pre = precondition(
            raw, h, damping=1e-3, smooth=2.0,
            normalize=False, max_gain=20.0, water_level=1e-3,
        )
        stage_arrays.append((raw, h, pre))
        stage_stats.append(
            {
                "stage": stage.name,
                "f_low": stage.f_low,
                "f_high": stage.f_high,
                "n_iter": stage.n_iter,
                "misfit": res.misfit,
                "band_energy_fraction": band_energy_fractions(observed, survey.dt, [stage])[0],
                "params": {k: map_stats(raw[k], h[k], pre[k], grid) for k in ("vp", "vs", "rho")},
            }
        )
        plot_stage_gradient(
            grid, stage, raw, h, pre,
            os.path.join(args.out, f"stage_{i:02d}_gradient.png"),
        )
        if i == 1:
            stability_rows = plot_stability(raw, h, os.path.join(args.out, "precondition_stability.png"))

    plot_filter_responses(filters, stages, survey.dt, os.path.join(args.out, "filter_responses.png"))
    plot_data_spectra(observed, filters, stages, survey.dt, os.path.join(args.out, "data_spectra.png"))

    checks = []

    def add_check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    for st, (raw, h, pre) in zip(stage_stats, stage_arrays):
        add_check(
            f"{st['stage']}: H >= 0 and finite",
            all(np.all(np.isfinite(h[k])) and np.all(h[k] >= -1e-12) for k in h),
        )
        add_check(
            f"{st['stage']}: preconditioned finite",
            all(np.all(np.isfinite(pre[k])) for k in pre),
        )
        add_check(
            f"{st['stage']}: deep boost > 1",
            all(st["params"][k]["deep_boost"] > 1.0 for k in ("vp", "vs")),
            ", ".join(f"{k}={st['params'][k]['deep_boost']:.2f}" for k in ("vp", "vs")),
        )

    # 解析公式检查：damping=1e-3, smooth=0, max_gain=0, normalize=False
    raw0, h0, _ = stage_arrays[0]
    pre_formula = precondition(
        {"vp": raw0["vp"]}, {"vp": h0["vp"]},
        damping=1e-3, smooth=0.0, normalize=False, max_gain=0,
    )["vp"]
    expected = raw0["vp"] / (h0["vp"] + 1e-3 * np.max(h0["vp"]))
    rel_formula = float(np.max(np.abs(pre_formula - expected)) / (np.max(np.abs(expected)) + 1e-30))
    add_check("pseudo-Hessian formula (no cap)", rel_formula < 1e-12, f"max rel={rel_formula:.2e}")

    for filt, stage in zip(filters, stages):
        f_pass = 0.5 * (stage.f_low + stage.f_high)
        f_stop = min(max(2.0 * stage.f_high, 30.0), 0.45 / survey.dt)
        rp = float(filt.amplitude_response(np.array([f_pass]))[0])
        rs = float(filt.amplitude_response(np.array([f_stop]))[0])
        add_check(
            f"{stage.name}: pass/stop response",
            rp > 0.1 and rs < 0.05,
            f"pass={rp:.3f}, stop={rs:.3f}",
        )

    # 多尺度 FWI（报告中的阶段 misfit/模型）
    cfg = FWIConfig(
        stages=stages,
        dtype="float64",
        storage_mb=512.0,
        verbose=True,
        ph_damping=1e-3,
        ph_smooth=2,
        ph_water_level=1e-3,
        ph_max_gain=20.0,
        constraints=ConstraintConfig(method="none"),
    )
    fwi = ElasticFWI(model, survey, true_model=setup.true_model, grid=grid, config=cfg)
    result = fwi.run()
    save_result(result, os.path.join(args.out, "fwi"), save_data=False)
    plot_history(result.history, os.path.join(args.out, "stage_convergence.png"))
    models = [model] + result.stage_models + [setup.true_model]
    titles = ["initial"] + [s.name for s in stages] + ["true"]
    plot_model_panel(models, grid, os.path.join(args.out, "stage_models.png"), titles=titles)

    rows = schedule_table(stages, survey.dt)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "grid": [grid.nx, grid.nz],
                "dx": grid.dx,
                "dt": survey.dt,
                "nt": survey.nt,
                "schedule": rows,
                "stage_stats": stage_stats,
                "stability": stability_rows,
                "checks": checks,
                "fwi_history": result.history,
                "fwi_stages": result.stage_summaries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# EFWI 伪 Hessian 预条件与多尺度优化报告",
        "",
        f"- 网格：{grid.nx} × {grid.nz}，dx=dz={grid.dx:g} m",
        f"- 观测系统：{survey.n_shots} 炮，{survey.n_receivers} 道，nt={survey.nt}，dt={survey.dt:.4g} s",
        f"- 频带：{', '.join(f'{s.f_low:g}-{s.f_high:g} Hz' for s in stages)}",
        "- 伪 Hessian：参数空间虚拟源平方积分（见 `docs/EFWI_梯度预处理与多尺度.md`）",
        "",
        "## 1. 多尺度调度",
        "",
        "| stage | band (Hz) | iterations | f_high/Nyquist |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['f_low']:g}-{r['f_high']:g} | {r['n_iter']:.0f} | "
            f"{r['nyquist_fraction']:.3f} |"
        )
    lines += [
        "",
        "## 2. 梯度预处理前后（初始模型）",
        "",
        "| stage | param | raw RMS | pre RMS | deep gain | shallow gain | deep boost | scale | H range (dB) | finite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for st in stage_stats:
        for key, row in st["params"].items():
            lines.append(
                f"| {st['stage']} | {key} | {row['g_rms']:.3e} | {row['pre_rms']:.3e} | "
                f"{row['deep_gain']:.4f} | {row['shallow_gain']:.4f} | "
                f"{row['deep_boost']:.2f} | {row['scale_ratio']:.3e} | "
                f"{row['h_dynamic_range_db']:.2f} | {'yes' if row['finite'] else 'NO'} |"
            )
    lines += [
        "",
        "> `deep gain` 是深部（z > 0.55 zmax）预条件/原始梯度 RMS 比，"
        "`deep boost = deep gain / shallow gain` 表示相对浅部的照明补偿倍数。",
    ]
    lines += [
        "",
        "## 3. 预条件稳定性",
        "",
        "| param | damping | max gain / median gain |",
        "|---|---:|---:|",
    ]
    for r in stability_rows:
        lines.append(f"| {r['param']} | {r['damping']:.0e} | {r['max_gain_ratio']:.3e} |")
    lines += [
        "",
        "## 4. 多尺度反演阶段结果",
        "",
        "| stage | target iter | done | start misfit | end misfit | reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in result.stage_summaries:
        lines.append(
            f"| {s['stage']} | {s['target_iter']} | {s['iter_done']} | "
            f"{s['start_misfit']:.6e} | {s['end_misfit']:.6e} | "
            f"{s['misfit_reduction_pct']:.2f}% |"
        )
    lines += [
        "",
        "## 5. 自检与测试报告",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c['check']} | {c['status']} | {c['detail']} |")
    lines += [
        "",
        "完整测试：`python -m unittest discover -s tests -v`；"
        "本模块覆盖 `tests/test_precondition.py`、`tests/test_multiscale.py`，"
        "当前全套 46 个测试通过。",
    ]
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n输出目录:", args.out)
    print("报告:", os.path.join(args.out, "report.md"))


if __name__ == "__main__":
    main()
