# EFWI 伪 Hessian 预条件与多尺度优化

本文档对应代码：

* `efwi/fd/kernels.py`：正演敏感场、`H_theta`、原生刚度/密度照明累积；
* `efwi/gradient.py`：参数空间伪 Hessian、预条件与链式转换；
* `efwi/pseudo_hessian.py`：伪 Hessian 诊断/封装；
* `efwi/filters.py`：Wiener 带通、频带校验、能量统计；
* `efwi/inversion.py`：多尺度调度、带限子波、预条件 EFWI；
* `scripts/run_preconditioning_demo.py`：对比图与测试报告生成。

---

## 1. 伪 Hessian 的定义

全 Hessian 为

$$
H_{ij}(x)=\sum_s\int_0^T
\left(\frac{\partial d_s}{\partial m_i(x)}\right)^T
\left(\frac{\partial d_s}{\partial m_j(x)}\right)dt,
$$

直接计算不可行。伪 Hessian 采用**高频近似 + 无限孔径 + 只保留对角元**：

$$
\boxed{
H_{pp}(x)=\sum_s\int_0^T
\left\|S_p(x,t)\right\|^2dt
}
$$

其中 \(S_p(x,t)\) 是参数 \(p\) 的**虚拟源/散射源**：当模型参数发生
\(\delta p\) 扰动时，正演方程右端新增的源项。它不是对梯度做平滑，也不是
对反演结果做图像处理。

### 1.1 各向同性弹性参数

记

$$
\theta=\dot\varepsilon_{xx}+\dot\varepsilon_{zz},\qquad
\dot\varepsilon_{xz}=\partial_z v_x+\partial_x v_z.
$$

由

$$
C_{11}=C_{33}=\rho v_p^2,\quad
C_{13}=\rho(v_p^2-2v_s^2),\quad
C_{44}=\rho v_s^2
$$

可得：

**vp 的虚拟源**（sxx、szz 各一份）：

$$
S_{v_p}^{xx}=S_{v_p}^{zz}=2\rho v_p\theta,\qquad S_{v_p}^{xz}=0,
$$

$$
\boxed{
H_{v_p}=2\int(2\rho v_p\theta)^2dt
=8\rho^2v_p^2\int\theta^2dt
}
$$

代码中正演时累积 `H_theta=∫θ²dt`（`accumulate_theta_hessian`），再乘
`8*rho^2*vp^2`。

**vs 的虚拟源**：

$$
S_{v_s}^{xx}=-4\rho v_s\dot\varepsilon_{zz},\quad
S_{v_s}^{zz}=-4\rho v_s\dot\varepsilon_{xx},\quad
S_{v_s}^{xz}=2\rho v_s\dot\varepsilon_{xz},
$$

$$
\boxed{
H_{v_s}=16\rho^2v_s^2\int(\dot\varepsilon_{xx}^2+\dot\varepsilon_{zz}^2)dt
+4\rho^2v_s^2\int\dot\varepsilon_{xz}^2dt
}
$$

**rho**：密度同时出现在质量项和刚度中。代码使用

$$
H_\rho=H_\rho^{\text{mass}}
+\left(\frac{\partial C_{11}}{\partial\rho}\right)^2H_{C11}
+\left(\frac{\partial C_{13}}{\partial\rho}\right)^2H_{C13}
+\left(\frac{\partial C_{33}}{\partial\rho}\right)^2H_{C33}
+\left(\frac{\partial C_{44}}{\partial\rho}\right)^2H_{C44},
$$

其中

$$
H_\rho^{\text{mass}}
=\int\left[\left(\frac{a_x}{\rho_x}\right)^2+
\left(\frac{a_z}{\rho_z}\right)^2\right]dt,
$$

\(a=(Ds+f)/\rho\) 是加速度。这是密度参数在质量项 + 刚度链式下的对角近似。

### 1.2 各向异性

VTI/HTI 使用刚度分量对角照明 \(H_{C11},H_{C13},H_{C33},H_{C44}\)，再按
`dC/dp` 链式平方：

$$
H_p\approx\sum_C\left(\frac{\partial C}{\partial p}\right)^2H_C.
$$

各向异性目前不做 vp 的 \(\theta\) 修正；这是已知的近似，测试中只对
各向同性 vp 使用直接虚拟源公式。

### 1.3 计算流程

每个正演时间步（仅 `hessian=True`）：

```
exx, ezz, exz = E v
H_C11 += dt*exx^2
H_C13 += dt*(exx^2+ezz^2)
H_C33 += dt*ezz^2
H_C44 += dt*exz^2
H_rho += dt*((ax/rho_x)^2+(az/rho_z)^2)
H_theta += dt*(exx+ezz)^2
```

所有炮集求和后，转换到 `(vp, vs, rho)`。`H` 与梯度由**同一组正演敏感场**
得到，避免额外正演。

---

## 2. 梯度预处理

### 2.1 公式

$$
\boxed{
g_p^{\text{pre}}=
\frac{g_p}{H_p+\eta\max(H_p)+\eta_w\max(H_p)+h_{\min}}
}
$$

其中：

* \(\eta\)：`ph_damping`，默认 `1e-3`；
* \(\eta_w\)：`ph_water_level`，默认 `0.0`（演示中使用 `1e-3`）；
* \(h_{\min}\)：`ph_min_h`，绝对水层；
* 对 \(H\) 做高斯平滑（`ph_smooth=2`）**只用于稳定对角估计**。H 本身的
  计算由虚拟源平方积分给出，不是“简单平滑替代”。

### 2.2 数值稳定措施

1. **相对水层**：\(\eta\max H\) 防止 \(H\to0\) 时除零；
2. **额外水层/绝对水层**：控制深部极弱照明；
3. **最大增益 cap**：以中位增益为典型值，限制

$$
\text{gain}\le \text{max\_gain}\times \text{median(gain)},
$$

   默认 `ph_max_gain=20`。这防止深部噪声被放大 3–4 个数量级；
4. **非有限值清零**：`nan_to_num`；
5. **可选归一化**：预条件后按原始梯度 RMS 归一到同一量级，保持
   vp/vs/rho 参数间尺度一致（`ph_normalize=True`）。归一化不改变空间
   照明补偿模式，只改变整体尺度。

### 2.3 预条件前后差异

以演示（81×61，2 炮，初始模型）为例：

| stage | param | raw RMS | pre RMS | deep gain | shallow gain | deep boost |
|---|---|---:|---:|---:|---:|---:|
| 2–8 Hz | vp | 1.403e-06 | 2.028e-07 | 2.48 | 0.144 | 17.2 |
| 2–8 Hz | vs | 1.320e-06 | 3.317e-07 | 4.89 | 0.250 | 19.5 |
| 2–8 Hz | rho | 6.018e-07 | 8.971e-07 | 20.4 | 1.485 | 13.7 |
| 3–15 Hz | vp | 1.296e-06 | 3.462e-08 | 0.120 | 0.0267 | 4.50 |
| 3–15 Hz | vs | 1.152e-06 | 5.128e-08 | 0.238 | 0.0445 | 5.34 |
| 3–15 Hz | rho | 5.392e-07 | 8.322e-08 | 0.840 | 0.154 | 5.45 |

（数值由 `scripts/run_preconditioning_demo.py` 自动写入 `runs/preconditioning/report.md`；
换网格/炮集后会变化。）

`deep gain` 是深部（\(z>0.55z_{\max}\)）预条件/原始梯度 RMS 比；
`deep boost = deep gain / shallow gain` 表示相对浅部的照明补偿倍数。可以看到
预条件把深部相对浅部的梯度能量提高了约 4.5–19.5 倍（随频带/参数变化）。

> 对比图中 raw/H/pre 三列各自独立裁剪显示，因此不能用图的颜色幅度直接
> 比较绝对大小；绝对比例见 `report.md` 的 `scale` 列。

---

## 3. 多尺度滤波

### 3.1 滤波器类型

`WienerFilter` 是**零相位 Wiener 带通**，不是简单 Butterworth：

1. 以当前炮集为参考，去直流、加 5% Tukey 窗；
2. 用 4 阶 Butterworth 带通构造理想目标 \(y\)；
3. 频域最小二乘

$$
H(f)=\frac{P_{xy}(f)}{P_{xx}(f)+\epsilon\max P_{xx}},
$$

4. 对 \(H\) 做 1 Hz 平滑、限幅 `[0,1.5]`；
5. 在 `[f_low/√2, f_high*√2]` 外做余弦掩膜，防止噪声频段放大。

参数：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `order` | 4 | Butterworth 目标阶数 |
| `eps` | 1e-3 | Wiener 正则化 |
| `mask_octaves` | 0.5 | 频带外掩膜过渡 |
| `smooth` | 1.0 Hz | 响应平滑 |
| `taper` | 5% | 时窗，抑制周期延拓 |

### 3.2 频带与子波

默认：

| 阶段 | 频带 | 迭代 | 目的 |
|---|---|---:|---|
| `band_8Hz` | 2–8 Hz | 25 | 建立大尺度构造，缓解周跳 |
| `band_15Hz` | 3–15 Hz | 25 | 刻划细节 |

**带限子波**：每个阶段把 Wiener 响应作用到震源子波上，再用带限子波正演。
这样伪 Hessian 是**当前频带的照明**，而不是全频带照明；在线性滤波假设下，
这与“对合成数据做同带滤波”等价，但 H 的物理含义更合理。

---

## 4. 多尺度调度策略

`efwi.filters.validate_stages` 强制：

1. \(0<f_{low}<f_{high}\)；
2. \(f_{high}\) 单调不减（从低频到高频递进）；
3. \(f_{high}<0.95f_{Nyquist}\)。

调度流程：

```
for stage in stages:
    filt = WienerFilter.design(observed, dt, stage.f_low, stage.f_high)
    stage_wavelet = filt.apply(survey.wavelet)
    obs_f = filt.apply(observed)
    op.set_survey(stage_survey)        # 带限子波
    reset L-BFGS
    for it in stage.n_iter:
        gradient + pseudo-Hessian (当前频带)
        precondition
        L-BFGS + parabolic line search
        constraints
```

* 频带间**保留模型**（warm start），不重置模型；
* 每个频带重置 L-BFGS 历史（不同频带曲率差异大）；
* 目标函数按观测能量归一化，跨频带可比；
* 阶段结果记录 `stage_summaries`、`stage_models`，供报告/图使用。

可自定义三阶段：

```powershell
.venv\Scripts\python.exe scripts\run_preconditioning_demo.py --stages "2-8,3-12,3-18" --iterations 3
```

---

## 5. 稳定性测试

`scripts/run_preconditioning_demo.py` 对 \(\eta=10^{-8}\dots10^{-1}\) 扫描，
并启用 `max_gain=20`：

| param | \(\eta\) | max gain / median gain |
|---|---:|---:|
| vp | 1e-8 | 20.0（cap） |
| vp | 1e-5 | 5.12 |
| vp | 1e-3 | 1.04 |
| vp | 1e-1 | 1.00 |
| vs | 1e-8 | 20.0 |
| vs | 1e-3 | 1.07 |
| rho | 1e-8 | 20.0 |
| rho | 1e-3 | 1.10 |

结论：

* 小阻尼时 cap 把极端增益限制在 20× 中位增益；
* 阻尼增大后增益趋于 1（预条件逐渐退化为原始梯度）；
* 所有预条件梯度 `finite=True`，无 NaN/Inf；
* `tests/test_precondition.py` 验证了“无 cap 时严格等于解析公式”、
  “零 H 输出零梯度”、“弱照明+噪声不爆炸”。

---

## 6. 输出

运行：

```powershell
.venv\Scripts\python.exe scripts\run_preconditioning_demo.py
```

输出 `runs/preconditioning/`：

```
report.md / report.json              # 调度、梯度统计、稳定性、阶段 misfit
stage_01_gradient.png               # raw / H / pre 三列对比
stage_02_gradient.png
filter_responses.png                # 各频带 Wiener 响应
data_spectra.png                    # 原始/分频带振幅谱
precondition_stability.png          # 阻尼-增益 cap 曲线
stage_models.png                    # 初始/各频带模型/真模型
stage_convergence.png
fwi/model_stage_01.npz ...
```

测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本模块新增 `tests/test_precondition.py`、`tests/test_multiscale.py`，当前
**46 个测试全部通过**（含梯度、RTM、测井、各向异性、闭环）。

---

## 7. 配置项

| 字段 | 默认 | 说明 |
|---|---:|---|
| `FWIConfig.pseudo_hessian` | True | 启用伪 Hessian |
| `ph_damping` | 1e-3 | 相对水层 |
| `ph_water_level` | 0.0 | 额外相对水层 |
| `ph_min_h` | 0.0 | 绝对水层 |
| `ph_max_gain` | 20.0 | 最大/中位增益比 |
| `ph_smooth` | 2 | 仅平滑 H 对角估计 |
| `ph_normalize` | True | 预条件后 RMS 归一化 |
| `StageConfig.f_low/f_high/n_iter` | — | 频带与迭代数 |
