# Q 补偿成像：模型、设计、实现与验证

本文档对应代码中的 `ertm/viscoelastic.py`、`ertm/propagation.py::ViscoacousticPropagator`、
`ertm/qc.py` 与 `config/viscoacoustic.json`，回答四个问题：**用什么衰减模型、
Q 是什么、衰减波场怎么算、补偿怎么做得既真实又稳定**。

---

## 1. 衰减介质数学模型

### 1.1 为什么选 GSLS

可选模型有三类：

| 模型 | 优点 | 缺点 |
|---|---|---|
| 常 Q（Kjartansson 分数阶） | 频带内 Q 严格为常数 | 分数阶导数，需全局 FFT 或分数阶拉普拉斯，边界条件与并行都麻烦 |
| 单机制 SLS | 解析简单 | 只在峰值频率附近近似常 Q |
| **GSLS（L 个 SLS 并联）** | 因果、满足 Kramers-Kronig、有限个记忆变量、可用同一套时间步进 | 参数需拟合 |

本项目选 **GSLS**：它既能用有限内存逐时间步推进（与现有交错网格求解器同构），
又能在目标频带内逼近常 Q，符合技术方案 2.3 的选择。

### 1.2 本构方程

体积模量的松弛函数（L 个机制并联）：

$$
K(t)=K_R\Big[1-\sum_{l=1}^{L}\Delta_l\,e^{-t/\tau^{\sigma}_l}\Big],
\qquad \Delta_l=1-\frac{\tau^{\varepsilon}_l}{\tau^{\sigma}_l}
$$

对应复模量（$\omega=2\pi f$）：

$$
K(\omega)=K_R\Big[1-\sum_l \frac{\Delta_l}{1+i\omega\tau^{\sigma}_l}\Big],
\qquad
Q(\omega)=\frac{\operatorname{Re}K(\omega)}{\operatorname{Im}K(\omega)}
$$

`τ^ε/τ^σ` 由 $\Delta$ 给出；未松弛模量 $K_U=K_R(1+\sum_l\Delta_l)$。
参考频率处标定 $K_R=\rho v_p^2/\operatorname{Re}K_{\text{norm}}(\omega_{\text{ref}})$，
因此**模型里写的 $v_p$ 是 $f_{\text{ref}}$ 处的相速度**，与无衰减情形在同一标定下可比。

### 1.3 参数拟合

给定 $(Q,f_{\text{ref}})$，在频带 $[f_{\text{ref}}/\beta,\,f_{\text{ref}}\beta]$
（默认 $\beta=1.5$）上最小化 $\sum|\ln Q(\omega)-\ln Q|^2$，未知量为
$\{\ln\tau^\sigma_l,\ln\Delta_l\}$（`scipy.optimize.least_squares`，L=3）。
实测：3 机制在 20–45 Hz 内的 Q 偏差 **5e-6**；单机制解析解的峰值 Q 精确命中目标。

---

## 2. 品质因子 Q 的物理意义

- **定义**：$Q=2\pi\times\dfrac{\text{每周期存储的弹性能}}{\text{每周期耗散的能量}}$，
  即 $1/Q$ 是每周期的相对能量损失。
- **振幅**：单色波传播距离 $x$ 后 $A=A_0e^{-\pi f x/(Qv)}$，时间域
  $A(t)=A_0e^{-\pi f t/Q}$。所以 **$t^*=\int \mathrm{d}t/Q$ 是"衰减时间"**，
  振幅衰减 $\exp(-\pi f t^*)$，频率越高衰减越快。
- **相位**：衰减与频散由 Kramers-Kronig 关系绑定，$v_p(f)$ 随频率变化（正常频散），
  波形会"低频先到、高频后到"，这就是为什么**只放大振幅无法恢复波形**。
- **量级**：岩石通常 $Q_p\approx 20\text{–}200$，$Q_s\approx 10\text{–}100$；
  $Q\to\infty$ 退化为弹性；$Q<2$ 时 GSLS 已难以稳定拟合，代码直接拒绝。

---

## 3. 衰减介质中的波场传播

### 3.1 连续方程（黏声，一阶速度—压力）

$$
\rho\frac{\partial \mathbf v}{\partial t}=\nabla p,\qquad
\frac{\partial \varepsilon}{\partial t}=\nabla\cdot\mathbf v,\qquad
p=K(t)*\dot\varepsilon
$$

其中 $\varepsilon$ 为体应变，$*$ 为时间卷积。对 GSLS 引入记忆变量
$r_l$ 后化为**局部**的一阶系统：

$$
\dot r_l=-\frac{r_l}{\tau^\sigma_l}+K_R\Delta_l\,\dot\varepsilon,
\qquad
p=K_R\,\varepsilon+\sum_{l=1}^{L}r_l
$$

（等价写法 $p=K_U\varepsilon-\sum_l r_l'$，两者只差 $K_R\Delta_l\varepsilon$ 的整理；
本项目用上式，使 $\Delta=0$ 时严格退化为声波方程。）

### 3.2 离散格式（与正演求解器一致）

时间二阶、空间八阶交错网格，一次步进：

```text
1) v^{n+1/2} = v^{n-1/2} + Δt/ρ · ∇p^n          （含 CFS-PML）
2) ε^{n+1}   = ε^n + Δt ∇·v^{n+1/2}
3) r_l^{n+1} = a_l r_l^n + b_l (ε^{n+1} − ε^n)
4) p^{n+1}   = K_R ε^{n+1} + Σ_l r_l^{n+1}
   a_l = exp(−Δt/τ^σ_l),  b_l = K_R Δ_l (1−a_l) τ^σ_l/Δt
```

步进顺序与技术方案 3.2.3 相同（先速度、再应变、再记忆、最后应力）。
压力源注入等价于注入体应变 $\varepsilon\mathrel{+}=s/K_R$，与声波求解器的软源一致。

### 3.3 频散关系

由 $K(\omega)$ 得相速度 $v_p(\omega)=\sqrt{\operatorname{Re}K(\omega)/\rho}$、
衰减系数 $\alpha(\omega)=\omega\operatorname{Im}K/(2\rho v_p^3)$；
在 GSLS 参数下 $\alpha\propto f$ 近似成立，即 **常 Q**。

---

## 4. 稳定的 Q 补偿方案

### 4.1 为什么是"负 Q 反传"而不是振幅增益

RTM 的成像条件是源波场与**反传**波场的互相关：

$$
I(\mathbf x)=\int W_s(\mathbf x,t)\,W_r(\mathbf x,t)\,\mathrm dt .
$$

物理上 $W_r$ 应是观测记录按时间反演的传播结果。若介质有衰减，正向传播按
$e^{-\pi ft/Q}$ 衰减；要抵消它，反传算子必须提供 $e^{+\pi ft/Q}$ 的**增长**，
且**频散也要反向**（把被"抹开"的波形重新聚焦）。因此正确的做法是让反传
使用**同一个 GSLS 模型、同一个 $K_R$、同一组 $\tau$，只把松弛强度取反**：

$$
\Delta_l\;\longrightarrow\;-\alpha_l\,\Delta_l
$$

这就在 PDE 层面把"衰减 + 频散"整体反号，而不是在图像上乘一个增益曲线。
两者有本质区别（5.2 节的实测：往返增益随频率从 1.48 变到 3.70，
且与理论 $e^{2\pi ft/Q}$ 吻合；常数放大不可能做到这一点）。

### 4.2 为什么不能 $\alpha_l=1$ 直接用

反传算子的增益随时间指数增长：

$$
G(f,t)=\exp\!\Big(\pi f\,\frac{\alpha_c t}{Q}\Big).
$$

高频（$f$ 大）与长记录（$t$ 大）时 $G$ 可达 $10^{6}$ 以上，任何噪声、离散误差、
边界反射都会被同样放大 → 数值发散。因此必须做**频率限制 + 增益预算 + 保底手段**
三层正则化。

---

## 5. 稳定性设计（本项目实际实现）

### 5.1 频率限制（机制选择）

GSLS 的 $L$ 个机制各自覆盖一个频段，峰值频率
$f_l=\sqrt{1-\Delta_l}/(2\pi\tau^\sigma_l)$。给定 `q.f_max`：

$$
\alpha_l=0\quad\text{当 } f_l>f_{\max}
$$

即**只补偿信噪比足够的频带**，高于 $f_{\max}$ 的机制保持（正向的）衰减。
默认 `f_max=null`（不限制）；演示配置取 60 Hz，3 个机制（7.8/29.8/114.9 Hz）
中最高频机制被剔除——这正是"高频噪声不放大"的实现方式。

### 5.2 增益预算（逐机制强度）

给定最大允许增益 $G_{\max}$（`q.gain_budget_db`，默认 40 dB = 100×）与记录时长
$T$，机制 $l$ 的强度取

$$
\alpha_l=\min\!\Big(\alpha_c,\;
\frac{\ln G_{\max}\cdot Q}{\pi f_l T}\Big),
$$

保证该机制在记录末端的总增益不超过 $G_{\max}$。这一条是**正则化**的核心：
它把"完全逆"换成"在信噪比允许范围内尽可能逆"，避免了对高频的过度放大。

### 5.3 保底手段（数值安全网）

- **振幅上限**：反传场超过 `amplitude_cap × 初始峰值` 时整体缩放（默认 50×），
  并计数 `clipped_steps`；
- **时变低通**：每 `lowpass_interval` 步对波场做一次波数域低通，截止频率随时间
  从 0.9 倍 Nyquist 降到 0.35 倍（高频增长最快，先压高频）；
- 两者都写入报告（`clipped_steps`、`lowpass_steps`），可判断是否"靠裁剪硬撑"。

> 说明：低通在**波数域**实现，物理依据是频散关系把高时间频率与高空间波数绑定；
> 它只是安全网，真正限制放大的是 5.1/5.2 两条。

### 5.4 参数建议

| 参数 | 默认 | 说明 |
|---|---|---|
| `alpha_c` | 0.9 | 补偿强度；Q 小或噪声大时降到 0.6–0.7 |
| `gain_budget_db` | 40 | 最大增益 100×；数据噪声大时降到 20–30 dB |
| `f_max` | null | 有强高频噪声时设为 1.5–2 倍主频 |
| `amplitude_cap` | 50 | 安全网；频繁触发说明预算过松 |
| `lowpass_interval` | 20 | 步数；越小越稳但越伤分辨率 |

---

## 6. 与正演物理模型的一致性

**同构保证**：补偿与正演使用同一个类 `ViscoacousticPropagator`，同一套
`τ^σ_l、Δ_l、K_R`，同一离散格式与边界；唯一差别是 `Δ_l → −α_lΔ_l`。
因此

- 补偿算子的**相速度频散**与正演严格反向（同一组 $\tau$）；
- $\alpha_l=1$、无预算/带限时，往返增益 $G_{\text{round}}(f,t)\approx e^{2\pi ft/Q}$
  （实测 1.48/2.34/3.70 vs 理论 1.40/2.31/3.82，误差 < 8%）；
- 反向传播的每一步都写入报告元数据，可复查实际用了多少补偿。

---

## 7. 修改的文件与数学原因

| 文件 | 修改 | 数学原因 |
|---|---|---|
| `ertm/viscoelastic.py`（新） | GSLS 拟合、复模量、$Q(\omega)$、相速度、谱比法 | 提供 $K(\omega)$ 与参数表，是补偿算子的参数源 |
| `ertm/propagation.py` | 新增 `ViscoacousticPropagator`：记忆变量步进、$\varepsilon$ 注入、`state/load_state` 含 $r_l$、`QOptions`、补偿分支（$\Delta\to-\alpha\Delta$、带限、预算）、稳定化 | 本构由 $p=K\varepsilon$ 变为 $p=K_R\varepsilon+\sum r_l$；补偿需要同一离散系统的负松弛 |
| `ertm/config.py` | `QConfig` 增加 `model/f_max/gain_budget_db/compare/n_mechanisms/...`；`solver.engine` 增加 `viscoacoustic`；校验 gsls 必须配 viscoacoustic | 保证"补偿模型 = 正演模型"不会被配置错配打破 |
| `ertm/solver.py` | 引擎选择、`q_options()` 透传 | 正演必须用黏声算子，否则补偿与正演不一致 |
| `ertm/imaging.py` | 正/反传分别构造 `QOptions`（反传 compensation=True）；元数据记录 $\alpha_l$、$f_l$、预算、截断步数 | 补偿必须发生在反传算子内，而不是成像后处理 |
| `ertm/pipeline.py` | `q.compare` 时额外跑一遍"不补偿"的 RTM 并输出对比 | 提供补偿前/后对照（要求 7） |
| `ertm/qc.py`（新） | 对比指标与图件、Q 模型导出、增益曲线 | 输出与验收 |
| `ertm/diagnostics.py` | 必需文件/模块清单加入新模块 | 结构自检 |

**没有采用的方案**：把记录乘一条 $e^{\pi ft/Q}$ 的增益曲线再成像。它只放大振幅、
不反演频散，且无法区分路径（不同炮检距的 $t^*$ 不同），已用 5.2 的往返增益测试
证明本方案不是这种近似。

---

## 8. 补偿前后对比与结果输出

启用 `q.compare=true` 后，`pipeline` 会用**同一批记录、同一成像参数**再跑一遍
`q.enabled=false` 的 RTM，然后输出：

| 文件 | 内容 |
|---|---|
| `output/.../images/q_compare.png` | 补偿前 / 补偿后 / 差值 三幅 + 深部窗口增益标注 |
| `output/.../images/q_compare.json` | 能量、深部窗口 RMS、增益倍数、深/浅能量比、相关系数 |
| `output/.../images/q_gain.png` | 各机制的补偿强度与累积增益曲线 |
| `output/.../models/q_model.npz` | Qp/Qs（及 vp/vs、网格）数组 |
| `output/.../models/q_model.png` | Qp/Qs 剖面图 |
| `run_report.json → image_meta.q` | 机制数、拟合误差、$\alpha_l$、$f_l$、$f_{\max}$、预算、截断/低通步数 |

### 演示结果（`config/viscoacoustic.json`，3 炮，141×121）

```
band_amplitude_gain          1.13    （484–968 m 窗口）
deep_shallow_energy_ratio    0.610 → 0.957
alpha_mechanisms             [0.90, 0.88, 0.00]   （f_max=60 Hz 剔除 115 Hz 机制）
q_fit_error                  1.0e-05
clipped_steps / lowpass     51 / 73
```

深部/浅部能量比从 0.61 提到 0.96，说明深部照明被明显补回；被剔除的高频机制
保证了高频噪声不被放大。

---

## 9. 验证清单

| 测试（`tests/test_rtm.py`） | 判据 | 结果 |
|---|---|---|
| `test_constant_q_fit_over_band` | 频带内 Q 偏差 < 2% | 5e-6 |
| `test_single_mechanism_peaks_at_reference_frequency` | $Q(f_{\text{ref}})=40$ | 40.0 |
| `test_viscoacoustic_measures_target_q` | 谱比法偏差 < 20% | 42.4（目标 40） |
| `test_round_trip_gain_matches_theory_and_is_frequency_dependent` | 增益与 $e^{2\pi ft/Q}$ 偏差 < 15%，且 40 Hz ≫ 10 Hz | 1.48/2.34/3.70 vs 1.40/2.31/3.82 |
| `test_band_limit_and_gain_budget_regularization` | $f_l>f_{\max}$ 的机制 $\alpha_l=0$；预算越小 $\alpha$ 越小 | 通过 |
| `test_comparison_metrics_detect_deep_amplitude_recovery` | 对比指标能测出 2× 深部增益 | 通过 |
| `test_compensation_recovers_deep_amplitude` | 补偿后深部能量 > 1.15× | 2.4× |

复现：

```powershell
.venv\Scripts\python.exe main.py --config config/viscoacoustic.json    # 约 33 s（含对比）
.venv\Scripts\python.exe -m unittest tests.test_rtm -v                 # 25 项数值测试
```

---

## 10. 边界与后续

- GSLS 只覆盖 **P 波（体积模量松弛）**；弹性 S 波的 $Q_s$ 路径（$\mu$ 松弛）未实现，
  PS 成像暂用弹性引擎；
- 反传补偿是**同一模型的负松弛**，在 $\Delta$ 较大（$Q\lesssim10$）时与严格逆算子
  有可测差异，需靠预算/带限控制；
- 未实现逆散射条件与 LSRTM 框架下的联合反演。
