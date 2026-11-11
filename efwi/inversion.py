"""多尺度弹性波全波形反演主循环。

流程（对应文档“四项解决方案”）::

    初始模型
      -> 对炮集做 Wiener 带通（8 Hz -> 15 Hz）
      -> 正演 + 伴随求梯度
      -> 伪 Hessian 对角预条件
      -> L-BFGS 方向 + 抛物线线搜索
      -> 地质/岩石物理凸集投影
      -> 下一频带 / 下一次迭代

目标函数采用观测能量归一化::

    J = 0.5 * ||d_syn - d_obs||^2 / ||d_obs||^2

因此打印的 misfit 无量纲，便于跨频带比较。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import FWIConfig, GridConfig, StageConfig
from .constraints import ModelConstraints
from .fd import Elastic2DOperator
from .filters import WienerFilter
from .gradient import PhysicalGradient, convert_gradient, precondition
from .model import ElasticModel
from .optimize import LBFGS, limit_step, parabolic_line_search
from .params import ParameterSpace


@dataclass
class FWIResult:
    model: ElasticModel
    initial_model: ElasticModel
    observed: np.ndarray
    synthetic: np.ndarray
    history: list[dict[str, Any]]
    config: FWIConfig
    true_model: ElasticModel | None = None
    elapsed: float = 0.0
    stage_summaries: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        out = {
            "n_iter": len(self.history),
            "elapsed_s": self.elapsed,
            "final_misfit": self.history[-1]["misfit"] if self.history else float("nan"),
            "initial_misfit": self.history[0]["misfit"] if self.history else float("nan"),
        }
        if self.true_model is not None:
            for key in ("vp", "vs", "rho"):
                out[f"{key}_rel_error"] = self.model.relative_error(self.true_model, key)
        return out


class ElasticFWI:
    def __init__(
        self,
        initial_model: ElasticModel,
        survey,
        *,
        true_model: ElasticModel | None = None,
        observed: np.ndarray | None = None,
        grid: GridConfig | None = None,
        config: FWIConfig | None = None,
        verbose: bool | None = None,
    ) -> None:
        if true_model is None and observed is None:
            raise ValueError("必须提供 true_model 或 observed 数据")
        self.config = config or FWIConfig()
        self.verbose = self.config.verbose if verbose is None else bool(verbose)
        self.initial_model = initial_model.copy()
        self.true_model = true_model.copy() if true_model is not None else None
        if grid is None:
            if initial_model.dx is None or initial_model.dz is None:
                raise ValueError("未提供 grid，模型也没有 dx/dz")
            grid = GridConfig(initial_model.nx, initial_model.nz, initial_model.dx, initial_model.dz)
        self.grid = grid
        self.survey = survey
        self.op = Elastic2DOperator(
            initial_model,
            survey,
            grid,
            dtype=self.config.dtype,
            storage_mb=self.config.storage_mb,
            verbose=self.verbose,
        )
        self.space = ParameterSpace(self.config.params)
        self.space.set_reference(initial_model)
        self.constraints = ModelConstraints(
            self.config.constraints,
            grid.dx,
            grid.dz,
            tuple(self.config.params),
            grid.shape,
        )
        if observed is None:
            assert true_model is not None
            observed = self.op.forward_all(true_model)
        self.observed = np.asarray(observed, dtype=np.float64)
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 约束/参数辅助
    # ------------------------------------------------------------------
    def _clip_bounds(self, model: ElasticModel) -> ElasticModel:
        c = self.config.constraints
        vals = {}
        for key in self.config.params:
            arr = np.asarray(getattr(model, key), dtype=np.float64)
            lo, hi = {
                "vp": c.vp_bounds,
                "vs": c.vs_bounds,
                "rho": c.rho_bounds,
                "epsilon": c.epsilon_bounds,
                "delta": c.delta_bounds,
            }[key]
            vals[key] = np.clip(arr, lo, hi)
        if "vp" in vals and "vs" in vals:
            vals["vs"] = np.minimum(vals["vs"], 0.98 * vals["vp"])
        return model.with_params(**vals)

    # ------------------------------------------------------------------
    # 梯度 + 预条件
    # ------------------------------------------------------------------
    def _evaluate_gradient(
        self,
        model: ElasticModel,
        obs: np.ndarray,
        filt: WienerFilter,
    ) -> tuple[Any, PhysicalGradient, dict[str, np.ndarray], np.ndarray, np.ndarray]:
        result = self.op.compute_gradient(model, obs, syn_filter=filt)
        pg = convert_gradient(result, model, self.config.params)
        g_true_x = self.space.gradient_to_space(pg.g, model)
        if self.config.pseudo_hessian:
            g_pre = precondition(
                pg.g,
                pg.h,
                damping=self.config.ph_damping,
                smooth=float(self.config.ph_smooth),
            )
        else:
            g_pre = {k: v.copy() for k, v in pg.g.items()}
        g_x = self.space.gradient_to_space(g_pre, model)
        return result, pg, g_pre, g_x, g_true_x

    # ------------------------------------------------------------------
    def run(self) -> FWIResult:
        t_start = time.time()
        model = self.initial_model.copy()
        x = self.space.pack(model)
        total_iter = 0
        stage_summaries: list[dict[str, Any]] = []
        last_filter: WienerFilter | None = None

        for si, stage in enumerate(self.config.stages):
            if stage.n_iter <= 0:
                continue
            if self.verbose:
                print(
                    f"\n===== Stage {si + 1}/{len(self.config.stages)} "
                    f"{stage.name or ''}: {stage.f_low:g}-{stage.f_high:g} Hz, "
                    f"{stage.n_iter} iters ====="
                )
            filt = WienerFilter.design(
                self.observed, self.survey.dt, stage.f_low, stage.f_high,
                time_axis=1,
            )
            last_filter = filt
            obs_f = filt.apply(self.observed, axis=1)
            lbfgs = LBFGS(memory=self.config.lbfgs_memory)
            res, pg, g_pre, g_x, g_true_x = self._evaluate_gradient(model, obs_f, filt)
            f_current = res.misfit
            stage_start_misfit = f_current
            self._record(total_iter, si, stage, f_current, 0.0, 0, 0.0, res, model, "start")

            for it in range(stage.n_iter):
                d = lbfgs.direction(g_x)
                d = limit_step(d, self.config.max_rel_step)
                # 预条件梯度用于 L-BFGS 方向；Armijo 必须用真实梯度斜率
                if float(np.dot(g_true_x, d)) >= 0.0 or not np.all(np.isfinite(d)):
                    d = -g_true_x
                    d = limit_step(d, self.config.max_rel_step)
                    lbfgs.reset()

                def evaluate(alpha: float, _x=x, _d=d, _model=model, _obs=obs_f, _filt=filt) -> float:
                    trial = self.space.unpack(_x + alpha * _d, _model)
                    trial = self._clip_bounds(trial)
                    return self.op.misfit(trial, _obs, syn_filter=_filt)

                ls = parabolic_line_search(
                    evaluate,
                    x,
                    d,
                    f_current,
                    g_true_x,
                    alpha_init=self.config.alpha_init,
                    c1=self.config.c1,
                    max_eval=self.config.max_line_search,
                    min_alpha=self.config.min_alpha,
                    max_alpha=4.0,
                    verbose=False,
                )
                if not ls.success:
                    if self.verbose:
                        print(f"  iter {it + 1}: 线搜索失败（{ls.message}），退回小步长")
                    d = -g_true_x / (np.linalg.norm(g_true_x) + 1e-30)
                    d = limit_step(d, 0.25 * self.config.max_rel_step)
                    ls = parabolic_line_search(
                        evaluate, x, d, f_current, g_true_x,
                        alpha_init=1.0, c1=self.config.c1,
                        max_eval=max(3, self.config.max_line_search // 2),
                        min_alpha=self.config.min_alpha,
                        max_alpha=1.0,
                    )
                    if not ls.success:
                        break

                x_trial = x + ls.alpha * d
                model_new = self.space.unpack(x_trial, model)
                model_new = self.constraints.project_model(model_new)
                x_new = self.space.pack(model_new)

                res_new, pg_new, g_pre_new, g_x_new, g_true_new = self._evaluate_gradient(
                    model_new, obs_f, filt
                )
                f_new = res_new.misfit
                s = x_new - x
                y = g_x_new - g_x
                lbfgs.update(s, y)

                total_iter += 1
                self._record(
                    total_iter, si, stage, f_new, ls.alpha, ls.n_eval,
                    float(np.linalg.norm(g_x_new)), res_new, model_new, "iter",
                )
                if self.verbose:
                    print(
                        f"  iter {it + 1:3d}: misfit={f_new:.6e} "
                        f"(stage start {stage_start_misfit:.3e}), alpha={ls.alpha:.3g}, "
                        f"ls={ls.n_eval}, |g|={np.linalg.norm(g_x_new):.3e}"
                    )

                x, model = x_new, model_new
                res, pg, g_pre, g_x, g_true_x = res_new, pg_new, g_pre_new, g_x_new, g_true_new
                f_current = f_new

                if f_current < 1e-10:
                    break
                if len(self.history) >= 4:
                    recent = [h["misfit"] for h in self.history[-4:]]
                    if abs(recent[-1] - recent[0]) <= 1e-5 * max(recent[0], 1e-30):
                        if self.verbose:
                            print("  目标函数已基本不再下降，提前结束该频带")
                        break

            stage_summaries.append(
                {
                    "stage": stage.name,
                    "f_low": stage.f_low,
                    "f_high": stage.f_high,
                    "start_misfit": stage_start_misfit,
                    "end_misfit": f_current,
                }
            )
            if self.verbose:
                print(
                    f"  stage end: {stage_start_misfit:.6e} -> {f_current:.6e}, "
                    f"约束违反={self.constraints.violations(model)}"
                )

        synthetic = self.op.forward_all(model)
        elapsed = time.time() - t_start
        return FWIResult(
            model=model,
            initial_model=self.initial_model,
            observed=self.observed,
            synthetic=synthetic,
            history=self.history,
            config=self.config,
            true_model=self.true_model,
            elapsed=elapsed,
            stage_summaries=stage_summaries,
        )

    # ------------------------------------------------------------------
    def _record(
        self,
        index: int,
        stage_index: int,
        stage: StageConfig,
        misfit: float,
        alpha: float,
        n_ls: int,
        gnorm: float,
        res: Any,
        model: ElasticModel,
        kind: str,
    ) -> None:
        self.history.append(
            {
                "index": index,
                "stage": stage.name,
                "stage_index": stage_index,
                "kind": kind,
                "misfit": float(misfit),
                "alpha": float(alpha),
                "n_line_search": int(n_ls),
                "grad_norm": float(gnorm),
                "obs_energy": float(res.obs_energy),
                "vp_min": float(model.vp.min()),
                "vp_max": float(model.vp.max()),
                "vs_min": float(model.vs.min()),
                "vs_max": float(model.vs.max()),
                "rho_min": float(model.rho.min()),
                "rho_max": float(model.rho.max()),
            }
        )


# ----------------------------------------------------------------------
def save_result(result: FWIResult, outdir: str, *, save_data: bool = False) -> str:
    os.makedirs(outdir, exist_ok=True)
    result.model.save(os.path.join(outdir, "model_final.npz"))
    result.initial_model.save(os.path.join(outdir, "model_initial.npz"))
    if result.true_model is not None:
        result.true_model.save(os.path.join(outdir, "model_true.npz"))
    with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"history": result.history, "stages": result.stage_summaries, "metrics": result.metrics()},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    if save_data:
        np.savez_compressed(
            os.path.join(outdir, "data.npz"),
            observed=result.observed,
            synthetic=result.synthetic,
        )
    return outdir
