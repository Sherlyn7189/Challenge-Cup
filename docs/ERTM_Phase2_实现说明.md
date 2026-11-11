# ERTM 第二阶段实现说明（双程波 RTM）

本文件记录“把第一阶段骨架升级为可用 RTM”这一版的实际实现范围、公式约定、
验证结果与已知边界，便于对照代码与需求验收。

## 1. 实现范围

| 需求项 | 落点 | 状态 |
|---|---|---|
| 双程波延拓：源正传 + 观测反传 + 同时刻互相关 | `ertm/propagation.py`、`ertm/imaging.py` | 已实现 |
| 交错网格八阶、二阶时间、CFS-PML、自由表面 | `ertm/numerics.py`、`ertm/propagation.py` | 已实现 |
| 成像条件：零延迟 / 归一化（源照明）/ 激发振幅 | `ertm/imaging.py` | 已实现 |
| 源波场检查点存储与分块重算 | `ertm/imaging.py`（`CheckpointStore`） | 已实现，重放逐位一致 |
| Laplacian 去低波数 + 波数域低通 | `ertm/numerics.py`、`ertm/imaging.py` | 已实现 |
| 四分量 COG→CSG 旋转 | `ertm/preprocess.py` | 已实现（二维 `vx,vz` 旋转） |
| 时间采样 sinc 插值（36 μs / 40 μs） | `ertm/numerics.py::resample_sinc` | 已实现，双向 |
| 直达波切除与首尾 taper | `ertm/preprocess.py` | 已实现 |
| 弹性一阶速度—应力、P/S 分离（PP/PS/SP/SS） | `ertm/propagation.py` | 已实现（`divcurl` / `stress`） |
| Q 衰减建模与反传侧补偿（增益缩放 + 时变低通 + 上限） | `ertm/qcomp.py` | 已实现（频域算子形式） |
| 多炮按炮归一化叠加 | `ertm/pipeline.py::stack_images` | 已实现 |
| GSLS 记忆变量黏声正演（P 波） | `ertm/viscoelastic.py`、`ViscoacousticPropagator` | 已实现 |
| 伴随算子内的 Q 补偿（反传负衰减） | `ViscoacousticPropagator(compensation=True)` | 已实现，含稳定化 |
| GSLS 弹性（K、μ 双松弛） | — | 未实现，列入第三阶段 |
| GPU / Numba 加速、三维、各向异性 | — | 未实现 |

## 2. 离散格式与参数

- 空间：交错网格八阶，系数 `c₁=1225/1024, c₂=−245/3072, c₃=49/5120, c₄=−5/7168`
  （技术方案式 3.4）；`order` 可降到 2/4/6 用于调试。
- 时间：二阶蛙跳；`dt` 必须满足 CFL：

  ```text
  dt ≤ 0.5 / (v_max · sqrt((C/dx)² + (C/dz)²)),  C = Σ|c_i|
  ```

  不满足时 `check_stability` 直接报错并给出建议值，不做静默降精度。
- 边界：`cpml`（四面 PML）、`free`（顶面自由表面 + 三面 PML）、`none`。
  PML 记忆变量 `ψⁿ = bψⁿ⁻¹ + aDf`，`∂f/∂x|_PML = Df/κ + ψ`；
  `α_max = 2πf₀`（CFS），`σ_max` 由 `reflection=1e-5`、二阶剖面确定。
- 成像网格裁剪：PML 区域不满足伴随对称性，成像后置零（`_pml_mask`）。

## 3. 成像条件

```text
zero_lag   : I(x) = Σ_s ∫ W_s(x,t) W_r(x,t) dt                     (6.7)
normalized : I(x) = Σ_s ∫ W_s W_r dt / (Σ_s ∫ W_s² dt + ε)         (6.10)
excitation : I(x) = A_s(x) · W_r(x, t_exc)                         (激发振幅)
```

- `normalized` 的照明项 `Σ∫W_s²dt` 按**全时段**累计；互相关分子只累计
  `t > delay + start_periods/f0`，避免近源强能量被“小分母”放大（实测中
  这一步决定了浅部假象是否压住真实界面）。
- `excitation` 只需保存“峰值时刻 + 峰值振幅”两张图，不保存整条源波场轨迹；
  峰值取 `|P|` 的逐点时程最大值。
- 弹性 P/S 分离两种方式：
  - `divcurl`：`P = ∇·v`、`S = (∇×v)_y`（技术方案式 3.7），对 S 场是精确分离；
  - `stress`：`P = (σ_xx+σ_zz)/2`、`S = σ_xz`，纯代数、无微分相位；但
    `σ_xz` 含 P 波能量（各向同性介质的 P 波也产生剪应力），故只作对照。

## 4. 验证结果（`tests/test_rtm.py`，25 项全绿，约 11 s）

| 验证项 | 判据 | 实测 |
|---|---|---|
| 八阶交错导数相对误差 | < 1e-3 | 6.2e-5 |
| 四阶交错导数（应更差） | < 1e-2 且 > 八阶 | 3.5e-3 |
| sinc 重采样波形误差 | < 2e-2 | 8.8e-4 |
| CFL 超限 | 抛错并给建议 dt | 通过 |
| CPML 吸收 | 出界后全场峰值显著下降 | 通过 |
| 检查点重放 | 与全新运行逐位一致 | 差值 0.0 |
| 常速走时差 | 与解析差 < 10 ms | 0.1 s vs 0.1 s |
| 层状界面归位 | 误差 < 100 m | 峰值 z=470–500 m（真值 500） |
| 70° 陡倾角界面 | 沿界面/偏离界面能量比 > 5 | ≈ 357 |
| 爆炸源 S 能量 | curl²/div² < 5% | 0.4% |
| 剪切源 S 主导 | curl²/div² > 0.5 | 通过 |
| Q 衰减→补偿 | 峰值恢复 0.7–3 倍，无 NaN | 1.06 倍 |
| 四分量旋转 | 能量守恒、90° 交换分量 | 通过 |

默认演示（`config/default.json`，201×151、dt=0.5 ms、nt=2000、三炮）：
约 15 s 完成正演+成像+叠加+出图，成像中 400 m、900 m 界面与 650 m 溶洞均
可见，最强能量集中在 400 m 界面。

## 4.1 GSLS 黏声正演与 Q 补偿（本轮新增）

本构：``K(t) = K_R[1 − Σ_l Δ_l e^(−t/τ^σ_l)]``，复模量
``K(ω) = K_R[1 − Σ_l Δ_l/(1+iωτ^σ_l)]``，``Q = Re K/Im K``。离散用
Robertsson 记忆变量格式（技术方案 2.4.1）：

```text
εⁿ⁺¹ = εⁿ + Δt∇·v
r_lⁿ⁺¹ = a_l r_lⁿ + b_l(εⁿ⁺¹ − εⁿ),  a_l = e^(−Δt/τ^σ),  b_l = K_R Δ_l (1−a_l) τ^σ/Δt
pⁿ⁺¹ = K_R εⁿ⁺¹ + Σ_l r_lⁿ⁺¹
```

- `fit_gsls` 对目标 Q 做常 Q 频带拟合（`scipy.optimize.least_squares`，
  变量为 ``ln τ^σ_l, ln Δ_l``）；L=3 时 0.67–1.5 倍频带内 Q 与目标的最大偏差
  **5e-6**（实测），单机制的解析解在 ``f_ref`` 处 Q 精确等于目标；
- 模型里每个出现的 Q 值单独拟合，再按点装配系数数组，因此分层/异常体可以有
  不同 Q；超过 32 个不同 Q 值时退化为中位数（避免逐点拟合的开销）；
- 压力源注入等价于注入体应变（``ε += s/K_R``），与声波求解器的软源一致；
- 补偿：``Δ_l → −α_c Δ_l``，即 Q 的逆过程；指数增长由
  ``amplitude_cap``（振幅上限）与每 ``lowpass_interval`` 步一次的时变低通控制。

**验证**（`tests/test_rtm.py::TestGSLSViscoacoustic`）：

| 项目 | 判据 | 实测 |
|---|---|---|
| 常 Q 拟合 | 频带内偏差 < 2% | 3 机制偏差 5e-6 |
| 单机制峰值 | Q(f_ref)=目标 | 40.0（1 机制） |
| 衰减后测 Q（同炮两接收点谱比） | 偏差 < 20% | 42.4（目标 40） |
| 反传补偿 | 600 m 界面能量提升 > 15% | 2.4 倍 |

## 5. 井中四分量演示（`config/borehole_4c.json`）

- 观测：源在地面 x=100 m，61 级三分量检波器布于 x=1000 m 井中（z=100–1300 m）；
- 引擎：`elastic`，记录 `p, vx, vz, sxz`，分量旋转 15°（COG→CSG）；
- 成像：PP、`stress` 分离、归一化条件、Laplacian；
- Q：开启，`alpha_c=0.7`、`gain_cap=30`、正演记录叠加衰减；
- 运行约 7 s。

受单侧照明限制，井中 demo 的近源/近井壁假象较强、深部照明弱，属于
VSP-RTM 的固有特点（可通过多方位炮、逆散射条件、照明补偿改善）。
该配置主要用于验证“四分量→CSG→弹性延拓→PP 成像→Q 补偿”的链路完整性。

## 6. 资源与性能

- 声波延拓：201×151 网格、2000 步约 1.3 s/次；
- 弹性延拓约为声波的 3–4 倍（6 个场 + 8 个 PML 导数项）；
- 检查点内存：`state_bytes = 场 + PML 记忆`（声波约 1.1 MB/状态），
  `imaging.checkpoint_mb` 控制上限，超出则增大保存间隔、反传时重算；
- 全流程无外部数值库依赖（仅 NumPy），绘图用 Matplotlib。

## 7. 复现

```powershell
.venv\Scripts\python.exe main.py --check                       # 10 项结构自检
.venv\Scripts\python.exe -m unittest tests.test_rtm -v         # 18 项数值测试
.venv\Scripts\python.exe -m unittest tests.test_structure -v   # 7 项契约测试
.venv\Scripts\python.exe main.py                               # 声波三炮演示
.venv\Scripts\python.exe main.py --config config/borehole_4c.json --output output/borehole
.venv\Scripts\python.exe main.py --config config/viscoacoustic.json   # GSLS 黏声 + Q 补偿
```

## 8. 已知边界（第三阶段建议）

1. GSLS 只覆盖 P 波（体积模量松弛）；弹性波的 K、μ 双松弛与 PS 的 Qs 路径未实现；
2. 井中 demo 照明不足，需多方位炮 + 照明补偿 + 逆散射成像条件；
3. `σ_xz` 的 P 波污染说明 `stress` 分离不能作为 S 波主传播器；
4. 三维、各向异性、自由表面精确镜像（含记忆变量）均未实现；
5. 未使用 GPU/Numba，大网格需靠检查点预算与分块控制内存。
