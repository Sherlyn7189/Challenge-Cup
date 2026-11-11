# ERTM 项目：双程波逆时偏移（RTM）

弹性波逆时偏移（Reverse Time Migration）工程实现。当前版本把第一阶段的
**占位骨架**替换为真实的双程波逆时偏移：

- 震源波场**正传** + 观测波场**反传**，同一时刻按成像条件做互相关；
- 不做高频渐近或单程波近似，因此不受地层倾角与偏移孔径限制，可对高陡
  构造、回转波、棱柱波乃至多次波成像，绕射波收敛归位；
- 二维一阶速度—应力方程、交错网格八阶空间差分、CFS-PML 吸收边界；
- 井中场景：四分量 COG→CSG 旋转、时间采样 sinc 插值、直达波切除、
  多分量弹性 RTM（PP/PS）、Q 衰减与反传侧补偿。

> 适用范围：二维 P-SV、各向同性弹性介质，CPU（NumPy）实现。三维、
> 各向异性、GSLS 记忆变量形式的黏弹正演列入下一阶段，见
> `docs/ERTM_Phase2_实现说明.md`。

## 1. 快速开始

```powershell
# 完整 RTM 流水线（默认 config/default.json，三炮叠加，约 15 s）
.venv\Scripts\python.exe main.py

# 结构与接口自检（不跑数值流程）
.venv\Scripts\python.exe main.py --check

# 只跑数值测试（差分精度、CFL、检查点、归位、陡倾角、P/S 分离、Q 补偿）
.venv\Scripts\python.exe -m unittest tests.test_rtm -v

# 结构与契约测试 + 端到端流水线
.venv\Scripts\python.exe -m unittest tests.test_structure -v
```

PyCharm 中右键 `main.py` → Run 即可；解释器选项目内 `.venv\Scripts\python.exe`。

## 2. 产物

一次运行在 `output/` 下生成：

```text
output/
├── logs/ertm.log
├── models/model.npz            # 弹性参数模型（vp/vs/rho/qp/qs）
├── shots/shot.npz              # 首炮记录（含多分量）
├── images/image_pp.npz/.png    # 叠加后的深度域成像
├── images/model.png            # 模型、炮集、指标图
├── images/shot.png
├── images/metrics.png
├── images/q_compare.png        # 仅 q.compare=true：补偿前/后/差值
├── images/q_gain.png           # 仅 q.compare=true：各机制补偿强度
├── models/q_model.npz/.png     # 仅 q.compare=true：Qp/Qs 模型
└── reports/
    ├── config_snapshot.json
    ├── metrics.json
    └── run_report.json         # 配置、稳定性、检查点、成像元数据
```

`run_report.json` 记录正演引擎、成像条件、检查点间隔与内存、Q 补偿参数、
稳定性诊断（CFL 比值、PPW）与全部产物路径，可直接复现一次运行。

## 3. 目录结构

```text
SeismicProjectERTM/
├── main.py                     # 命令行入口
├── config/
│   ├── default.json            # 声波三炮演示（地面观测）
│   └── borehole_4c.json        # 井中四分量 + 弹性 RTM 演示
├── ertm/
│   ├── numerics.py             # 八阶交错差分、CPML、CFL、sinc 插值、滤波
│   ├── propagation.py          # 声波/弹性一阶速度—应力正演与反传延拓
│   ├── solver.py               # 求解器封装（配置→规格→多分量 ShotRecord）
│   ├── imaging.py              # RTM 装配：检查点、成像条件、归一化、后处理
│   ├── preprocess.py           # COG→CSG 旋转、时间重采样、直达波切除
│   ├── qcomp.py                # Q 衰减频响与反传侧补偿（时变增益+低通）
│   ├── models.py               # 层状背景 + 溶洞/裂缝带/断层
│   ├── pipeline.py             # 多炮正演→成像→叠加→评价→输出
│   ├── evaluation.py           # 统计指标 + 成像质量代理指标
│   ├── data.py / visualization.py / logging_utils.py / exceptions.py
│   └── config.py / types.py / interfaces.py / diagnostics.py
├── tests/
│   ├── test_structure.py       # 接口/契约/IO/流水线
│   └── test_rtm.py             # 数值正确性与成像质量
└── docs/                       # 需求与技术方案（第一、二阶段）
```

## 4. 核心实现

### 4.1 正演与反传（`propagation.py`）

- 声波：`ρ∂v/∂t = ∇p`，`(1/κ)∂p/∂t = ∇·v`，`κ = ρv_p²`；
- 弹性：一阶速度—应力 P-SV 系统，变量位置与技术方案表 3.1 一致
  （`σ_xx, σ_zz` 在单元中心，`v_x`/`v_z` 在面心，`σ_xz` 在角点）；
- 空间八阶交错差分（`c₁ = 1225/1024 …`），时间二阶蛙跳；
- CFS-PML：`ψⁿ = b·ψⁿ⁻¹ + a·Df`，`∂f/∂x|_PML = Df/κ + ψ`；
- 自由表面（`boundary="free"`）：顶部不设 PML，声波 `p=0`、弹性
  `σ_zz = σ_xz = 0`；
- 反传：记录按时间倒序注入检波点（速度分量注入速度场，`p` 注入两个正应力）。

### 4.2 成像（`imaging.py`）

| 条件 | 公式 | 说明 |
|---|---|---|
| `zero_lag` | `I = Σ_s ∫ W_s·W_r dt` | 技术方案式 (6.7) |
| `normalized` | `I = Σ∫W_sW_r dt / (Σ∫W_s² dt + ε)` | 式 (6.10)，压深浅能量失衡 |
| `excitation` | `I = A_s(x)·W_r(x, t_exc)` | 激发振幅/波至时间约束，免存完整源波场 |

- 源波场用**检查点**保存（内存预算 `imaging.checkpoint_mb`），反传时分块
  重算；状态含场与 PML 记忆变量，重放与全新运行逐位一致（有测试）；
- 时间切除：只累计 `t > delay + start_periods/f0`，压制近源强能量；
- PML 区域置零（该区不满足伴随对称性）；
- 后处理：Laplacian 去低波数 → 波数域低通 → 多炮按炮归一化叠加。

### 4.3 井中四分量链路（`preprocess.py`）

1. **COG→CSG**：按方位角旋转 `v_x, v_z`（正交变换，保能量）；
2. **时间采样插值**：Kaiser 窗 sinc，36 μs/40 μs 与求解器 `dt` 互换；
3. **直达波切除**：`t_mute = delay + 1.2/f0 + 1.05·|x − x_s|/v_ref`，余弦过渡；
4. **首尾 taper**：压制记录截断引起的边界反射。

### 4.4 Q 衰减与补偿（`qcomp.py` + `viscoelastic.py`）

两种模式，由 `q.model` 选择：

- **`filter`（数据域）**：频响 `H(f,t) = exp(−πft/Q)·exp(iφ)`，
  `φ = −(2ft/Q)ln(f/f_ref)`，振幅/相位可分别开关；补偿作用在反传边界条件上
  （增益指数乘 `α_c`、上限 `gain_cap`、时变低通）。
- **`gsls`（传播算子内，推荐）**：`solver.engine="viscoacoustic"` 使用一阶
  速度—压力 + GSLS 记忆变量：

  ```text
  εⁿ⁺¹ = εⁿ + Δt∇·v
  r_lⁿ⁺¹ = a_l r_lⁿ + b_l(εⁿ⁺¹ − εⁿ)
  pⁿ⁺¹ = K_R εⁿ⁺¹ + Σ_l r_lⁿ⁺¹
  a_l = exp(−Δt/τ^σ_l),  b_l = K_R Δ_l (1−a_l) τ^σ_l/Δt
  ```

  `τ^σ_l、Δ_l` 由 `fit_gsls` 对目标 Q 做常 Q 频带拟合（L=3 时 0.67–1.5 倍频带内
  Q 偏差 < 1e-4）；模型里每个 Q 值单独拟合，逐点装配 `a_l, b_l, K_R`，因此
  分层/异常体可以有不同 Q。补偿时把 `Δ_l` 乘 `−α_l`（Q 的逆过程），并做三层正则化：

1. **带限**：峰值频率 `f_l > f_max` 的机制不参与补偿（默认不限，演示取 60 Hz）；
2. **增益预算**：`α_l = min(α_c, ln(G_max)·Q/(π·f_l·T))`，保证记录末端总增益
   不超过 `gain_budget_db`（默认 40 dB = 100 倍）；
3. **安全网**：振幅上限 `amplitude_cap` + 每 `lowpass_interval` 步的时变低通。

`q.compare=true` 时额外跑一遍不补偿的 RTM，输出 `q_compare.png/.json`
（补偿前/后/差值 + 指标）、`q_gain.png`（增益曲线）、`models/q_model.npz/.png`
（Q 模型）。完整推导、稳定性分析与验证见
[docs/Q补偿成像_设计与实现.md](docs/Q补偿成像_设计与实现.md)。

## 5. 关键配置（`config/default.json`）

| 段 | 字段 | 说明 |
|---|---|---|
| `grid` | `nx, nz, dx, dz` | 二维规则网格，单位米 |
| `model` | `layers, anomaly` | 层状背景 + 溶洞/裂缝带/断层（`kind`, `dip`） |
| `source` | `x, z, f0, delay, mechanism` | 震源位置/主频/延迟/机制（爆炸/剪切/力源） |
| `receivers` | `count, x_start, x_end, z` | 地面排列；井中排列由 `acquisition` 决定 |
| `acquisition` | `geometry, components, record_dt, rotation_deg, shots` | 观测系统、四分量、实测采样率、多炮 |
| `solver` | `nt, dt, order, boundary, engine, npml` | 时间步、阶数、边界、声波/弹性引擎 |
| `imaging` | `mode, condition, separation, laplacian, checkpoint_mb` | 成像条件与后处理 |
| `q` | `enabled, model, alpha_c, gain_cap, n_mechanisms, amplitude_cap` | Q 模式（filter/gsls）、补偿强度与稳定化 |

`dt` 必须满足 `dt ≤ 0.5/(v_max·√((C/dx)² + (C/dz)²))`；不满足时求解器
直接报错并给出建议值（而不是静默降精度）。

## 6. 演示配置

- `config/default.json`：地面观测、声波、三炮（x = 700/1000/1300 m），
  层状模型 + 溶洞，`condition=normalized`，约 15 s 出图；
- `config/borehole_4c.json`：井中四分量、弹性波、PP/PS 成像、Q 补偿，
  用于演示第 ②③④ 项关键技术（运行时间明显更长）；
- `config/viscoacoustic.json`：GSLS 黏声正演 + 伴随内 Q 补偿（α_c=0.9），
  三炮、141×121 网格，约 19 s，可对照 `q.enabled=false` 观察深部振幅恢复。

## 7. 测试与验证

`tests/test_rtm.py` 覆盖：

- 八阶交错差分相对误差 < 1e-3（四阶 < 1e-2 且明显更差）；
- sinc 插值波形误差 < 2e-2；CFL 超限时抛错；
- CPML 吸收（波场出界后能量衰减）；
- 检查点重放与全新运行逐位一致（`allclose atol=1e-6`）；
- 常速走时差与解析解一致（误差 < 10 ms）；
- 层状界面归位误差 < 100 m；70° 陡倾角界面成像能量比 > 5；
- 爆炸源不产生可观 S 能量（curl/div < 5%），剪切源以 S 为主；
- PP/PS 弹性多分量成像有限且非零；
- Q 衰减后振幅下降、补偿后恢复到 0.7–3 倍区间且无 NaN；
- 四分量旋转保能量、90° 旋转交换分量；
- 端到端流水线可产出报告与图像。

## 8. 已知边界（下一阶段）

- 二维 P-SV，未做三维与 SH；四分量中的水平横向分量在二维下不参与；
- GSLS 黏声正演/补偿只覆盖 P 波（体积模量松弛）；弹性波的 K、μ 双松弛
  （技术方案 2.4.1 的完整黏弹系统）尚未实现，PS 成像仍用弹性引擎；
- 未使用 GPU/Numba，大网格需按 `checkpoint_mb` 调整内存；
- 逆散射成像条件、角度域道集、LSRTM 未实现。

## 9. 依赖

以 `requirements.txt` 为准；核心链路只用 NumPy，绘图用 Matplotlib，
SciPy 仅用于可选的高阶信号处理。

## 10. Qt 求解器工作台（workbench）

仓库中的 `workbench/` 是基于 PySide6 / Qt Quick 的 ERTM 可视化工作台：读取根目录的
配置、模型、单炮记录、成像产物和源码，并可直接调用 `main.py` 运行流水线。界面按
“参数配置 → 典型模型成像对比 → 实测资料对比 → Q 补偿 → 求解器源码”组织，
工作台读取的是本版 RTM 的真实产物（模型、炮集、成像与指标），可与 `python main.py` 的输出直接对照。

![Qt 工作台](workbench/screenshots/01_config.png)

“单炮记录”卡片展示时间 × 检波点的记录形式，点击可展开采集参数、直达波公式以及
RTM 中检波点波场的用途说明。

安装与运行方式见 [workbench/README.md](workbench/README.md)。
