"""端到端演示：EFWI -> RTM -> 倾角提取 -> 三维建模。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ConstraintConfig, FWIConfig, StageConfig
from .imaging import depth_taper, extract_dips, reverse_time_migration, top_mute
from .inversion import save_result
from .model import ElasticModel
from .synthetic import demo_setup
from .viz import plot_data_fit, plot_gather, plot_history, plot_model_panel, plot_rtm, plot_dip
from .workflow import ClosedLoopWorkflow, WorkflowConfig, build_3d_model


@dataclass
class DemoResult:
    setup: Any
    fwi: Any
    rtm: Any
    dip: Any
    paths: dict[str, str]
    metrics: dict[str, Any]


def build_demo_config(
    *,
    n_iter: int = 3,
    quick: bool = True,
    with_constraints: bool = True,
    anisotropy: bool = False,
) -> FWIConfig:
    stages = [
        StageConfig(f_low=2.0, f_high=8.0, n_iter=n_iter, name="band_8Hz"),
        StageConfig(f_low=3.0, f_high=15.0, n_iter=n_iter, name="band_15Hz"),
    ]
    c = ConstraintConfig(
        method="dykstra" if with_constraints else "none",
        n_iter=4 if quick else 6,
        grad_proj_iter=12 if quick else 20,
    )
    return FWIConfig(
        stages=stages,
        dtype="float32",
        storage_mb=512.0,
        verbose=True,
        invert_anisotropy=anisotropy,
        constraints=c,
    )


def run_demo(
    *,
    quick: bool = True,
    outdir: str = "runs/demo",
    n_iter: int = 3,
    with_constraints: bool = True,
    anisotropy: str = "iso",
    n_shots: int | None = None,
    n_receivers: int | None = None,
) -> DemoResult:
    os.makedirs(outdir, exist_ok=True)
    setup = demo_setup(
        quick=quick, anisotropy=anisotropy, n_shots=n_shots, n_receivers=n_receivers
    )
    print("模型:", setup.true_model.describe())
    print(
        f"观测系统: {setup.survey.n_shots} 炮, {setup.survey.n_receivers} 道, "
        f"nt={setup.survey.nt}, dt={setup.survey.dt:.4g}s"
    )
    config = build_demo_config(
        n_iter=n_iter,
        quick=quick,
        with_constraints=with_constraints,
        anisotropy=(anisotropy != "iso"),
    )

    from .inversion import ElasticFWI

    fwi = ElasticFWI(
        setup.initial_model,
        setup.survey,
        true_model=setup.true_model,
        grid=setup.grid,
        config=config,
    )
    print("\n>>> 开始 EFWI")
    result = fwi.run()
    print("EFWI 指标:", json.dumps(result.metrics(), ensure_ascii=False, indent=2))
    save_result(result, os.path.join(outdir, "fwi"), save_data=True)

    paths: dict[str, str] = {}
    paths["model_panel"] = plot_model_panel(
        [setup.initial_model, result.model, setup.true_model],
        setup.grid,
        os.path.join(outdir, "models.png"),
        titles=["initial", "EFWI", "true"],
    )
    paths["history"] = plot_history(result.history, os.path.join(outdir, "convergence.png"))
    paths["data_fit"] = plot_data_fit(
        result.observed, result.synthetic, setup.survey.dt,
        os.path.join(outdir, "data_fit.png"),
    )
    paths["gather"] = plot_gather(
        result.observed[0], setup.survey.dt, os.path.join(outdir, "observed_gather.png")
    )

    print("\n>>> 开始 RTM（使用 EFWI 模型）")
    data_rtm = top_mute(
        result.observed,
        setup.survey.dt,
        setup.survey.shots,
        setup.survey.receivers,
        velocity=1600.0,
        factor=0.9,
    )
    rtm = reverse_time_migration(
        result.model,
        setup.survey,
        data_rtm,
        grid=setup.grid,
        storage_mb=256.0,
        verbose=True,
    )
    z_src = float(setup.survey.shots[0, 1])
    # 直达波/近场采集脚印在震源附近能量极强，硬切除到第一界面之上，
    # 否则浅部假象会压过真实反射。
    rtm.image = depth_taper(
        rtm.image, setup.grid.z,
        z_src + 8.0 * setup.grid.dz, z_src + 9.0 * setup.grid.dz,
    )
    paths["rtm"] = plot_rtm(
        rtm.image, setup.grid, os.path.join(outdir, "rtm.png"), clip=95.0
    )
    np.savez_compressed(
        os.path.join(outdir, "rtm.npz"),
        image=rtm.image,
        illumination=rtm.illumination,
    )

    dip = extract_dips(rtm.image, setup.grid.dx, setup.grid.dz, sigma=2.0)
    paths["dip"] = plot_dip(
        np.degrees(dip.dip), dip.coherence, setup.grid, os.path.join(outdir, "dips.png")
    )
    np.savez_compressed(
        os.path.join(outdir, "dips.npz"),
        dip=dip.dip, coherence=dip.coherence, reflection_mask=dip.reflection_mask,
    )

    print("\n>>> 倾角约束三维建模")
    model3d = build_3d_model(result.model, dip, ny=21, dy=20.0)
    paths["model3d"] = model3d.save_npz(os.path.join(outdir, "model3d.npz"))
    vtk = model3d.save_vtk(os.path.join(outdir, "model3d.vti"))
    if vtk:
        paths["model3d_vtk"] = vtk

    metrics = result.metrics()
    metrics.update(
        {
            "rtm_max": float(np.max(np.abs(rtm.image))),
            "n_reflector_points": int(dip.reflection_mask.sum()),
            "grid": [setup.grid.nx, setup.grid.nz],
            "dx": setup.grid.dx,
            "nt": setup.survey.nt,
            "dt": setup.survey.dt,
        }
    )
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print("\n输出:", outdir)
    return DemoResult(setup, result, rtm, dip, paths, metrics)


def run_welllog_demo(
    las_path: str,
    *,
    outdir: str = "runs/welllog",
    quick: bool = True,
    anisotropy: str = "iso",
    x: float = 0.0,
) -> dict[str, Any]:
    from .welllog import WellLog, build_initial_model, logging_seismic_scales

    os.makedirs(outdir, exist_ok=True)
    log = WellLog.from_las(las_path, x=x)
    print(log.describe())
    scales = logging_seismic_scales(3000.0, 10000.0, 30.0)
    print("尺度对比:")
    for k in ("logging", "seismic"):
        s = scales[k]
        print(
            f"  {k:8s}: f={s['frequency']:.0f} Hz, lambda={s['wavelength']:.3g} m, "
            f"dx={s['dx']:.3g} m, dt={s['dt']:.3g} s"
        )
    print(f"  频率比={scales['ratio']['frequency']:.1f}, dx 比={scales['ratio']['dx']:.4f}")

    setup = demo_setup(quick=quick, anisotropy=anisotropy)
    model = build_initial_model([log], setup.grid, anisotropy=anisotropy)
    paths = {}
    paths["model"] = plot_model_panel(
        [model], setup.grid, os.path.join(outdir, "well_model.png"),
        keys=("vp", "vs", "rho"), titles=["from well log"],
    )
    model.save(os.path.join(outdir, "model_from_well.npz"))
    with open(os.path.join(outdir, "scales.json"), "w", encoding="utf-8") as fh:
        json.dump(scales, fh, ensure_ascii=False, indent=2)
    print("输出:", outdir)
    return {"log": log, "model": model, "scales": scales, "paths": paths}


def run_closed_loop_demo(
    *,
    outdir: str = "runs/closed_loop",
    n_outer: int = 2,
    n_iter: int = 2,
) -> Any:
    setup = demo_setup(quick=True)
    fwi_config = build_demo_config(n_iter=n_iter, quick=True)
    z_src = float(setup.survey.shots[0, 1])
    wf = WorkflowConfig(
        fwi=fwi_config,
        n_outer=n_outer,
        depth_mute=(z_src + 8.0 * setup.grid.dz, z_src + 9.0 * setup.grid.dz),
        build_3d=True,
        ny=21,
        dy=20.0,
        outdir=outdir,
        top_mute_velocity=1600.0,
    )
    workflow = ClosedLoopWorkflow(
        setup.initial_model,
        setup.survey,
        true_model=setup.true_model,
        grid=setup.grid,
        config=wf,
    )
    result = workflow.run()
    print("闭环指标:", json.dumps(result.metrics, ensure_ascii=False, indent=2))
    return result
