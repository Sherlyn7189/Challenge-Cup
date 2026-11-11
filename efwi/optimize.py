"""L-BFGS 拟牛顿优化与抛物线线搜索。

文档要求“L-BFGS 近似 Hessian + 抛物线拟合搜索步长”。这里实现：

* 两循环 (two-loop recursion) L-BFGS，历史长度默认 8；
* 抛物线线搜索：先用 Armijo 条件保证下降，再用 ``f(0), f(alpha), g(0)·d``
  拟合抛物线取极小点，必要时回溯；
* 步长上限由 ``max_rel_step`` 控制，避免对数参数更新过大。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LineSearchResult:
    alpha: float
    f: float
    n_eval: int
    success: bool
    message: str = ""


def parabolic_line_search(
    evaluate: Callable[[float], float],
    x: np.ndarray,
    d: np.ndarray,
    f0: float,
    g0: np.ndarray,
    *,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    max_eval: int = 8,
    min_alpha: float = 1e-8,
    max_alpha: float = 10.0,
    verbose: bool = False,
) -> LineSearchResult:
    """沿 ``d`` 搜索满足 Armijo 条件的步长，并用抛物线预测极小点。

    ``evaluate(alpha)`` 需返回该步长处的目标函数值（内部可做约束投影）。
    """
    slope = float(np.dot(g0, d))
    if not np.isfinite(slope) or slope >= 0:
        return LineSearchResult(0.0, f0, 0, False, "方向不是下降方向")

    alpha = float(np.clip(alpha_init, min_alpha, max_alpha))
    n_eval = 0
    best_alpha = 0.0
    best_f = f0

    def eval_at(a: float) -> float:
        nonlocal n_eval
        n_eval += 1
        return float(evaluate(a))

    f1 = eval_at(alpha)
    if not np.isfinite(f1):
        return LineSearchResult(0.0, f0, n_eval, False, "目标函数出现 NaN/Inf")

    # Armijo 回溯
    while n_eval < max_eval and f1 > f0 + c1 * alpha * slope:
        alpha *= 0.5
        if alpha < min_alpha:
            break
        f1 = eval_at(alpha)

    if f1 <= f0 + c1 * alpha * slope:
        best_alpha, best_f = alpha, f1
    else:
        return LineSearchResult(0.0, f0, n_eval, False, "Armijo 回溯失败")

    # 抛物线拟合：p(a)=f0 + slope*a + c*a^2, p(alpha)=f1
    if n_eval < max_eval:
        denom = 2.0 * (f1 - f0 - slope * alpha)
        if denom > 0:
            a_star = -slope * alpha * alpha / denom
            a_star = float(np.clip(a_star, 0.1 * alpha, 3.0 * alpha))
            if abs(a_star - alpha) > 0.05 * alpha:
                f_star = eval_at(a_star)
                if np.isfinite(f_star) and f_star < best_f:
                    best_alpha, best_f = a_star, f_star

    if verbose:
        print(f"    line search: alpha={best_alpha:.4g}, f={best_f:.6g}, evals={n_eval}")
    return LineSearchResult(best_alpha, best_f, n_eval, True)


class LBFGS:
    """有限内存 BFGS 方向计算器。"""

    def __init__(self, memory: int = 8, scale: float = 1.0) -> None:
        self.memory = int(memory)
        self.scale = float(scale)
        self.s_history: list[np.ndarray] = []
        self.y_history: list[np.ndarray] = []
        self.rho_history: list[float] = []

    def reset(self) -> None:
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()

    @property
    def n_pairs(self) -> int:
        return len(self.s_history)

    def update(self, s: np.ndarray, y: np.ndarray) -> bool:
        s = np.asarray(s, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sy = float(np.dot(s, y))
        if not np.isfinite(sy) or sy <= 1e-14 * max(float(np.linalg.norm(s) * np.linalg.norm(y)), 1e-30):
            return False
        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.rho_history.append(1.0 / sy)
        if len(self.s_history) > self.memory:
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)
        return True

    def _initial_hessian(self) -> float:
        if not self.s_history:
            return self.scale
        s = self.s_history[-1]
        y = self.y_history[-1]
        yy = float(np.dot(y, y))
        if yy <= 0:
            return self.scale
        gamma = float(np.dot(s, y) / yy)
        return max(gamma, 1e-12)

    def direction(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64)
        if not self.s_history:
            return -g.copy()
        q = g.copy()
        alphas = []
        for s, y, rho in zip(
            reversed(self.s_history), reversed(self.y_history), reversed(self.rho_history)
        ):
            a = rho * float(np.dot(s, q))
            q -= a * y
            alphas.append(a)
        r = q * self._initial_hessian()
        for s, y, rho, a in zip(
            self.s_history, self.y_history, self.rho_history, reversed(alphas)
        ):
            beta = rho * float(np.dot(y, r))
            r += s * (a - beta)
        return -r


def limit_step(d: np.ndarray, max_abs: float) -> np.ndarray:
    """限制方向的最大分量，等价于信赖域式的步长上限。"""
    if max_abs <= 0:
        return d
    m = float(np.max(np.abs(d)))
    if m > max_abs:
        return d * (max_abs / m)
    return d
