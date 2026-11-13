# SeismicProjectEFWI — 二维弹性波全波形反演 / RTM / 测井尺度建模

本项目按《项目文档 2.2.6 EFWI（弹性波全波形反演）》的实现规格，提供一套可直接运行、
可测试、可扩展的 Python 代码：

* **EFWI**：交错网格弹性波正演 + 伴随态梯度，反演 `(vp, vs, rho)`，可扩展到
  VTI/HTI 各向异性；
* **多尺度**：炮集 Wiener 带通，默认 8 Hz → 15 Hz 两个频带；
* **伪 Hessian 预条件**：对角照明补偿，增强中深部梯度；
* **L-BFGS + 抛物线线搜索**；
* **地质/岩石物理约束**：上下界、侧向/垂向梯度上限、Gardner 关系；
  用 Dykstra 交替投影 / consensus ADMM 投影到交集；
* **RTM 保幅成像**：P/S 波场分离、源照明归一化成像条件、倾角/反射点提取；
* **测井适配**：LAS 解析、曲线延拓、井间插值、kHz↔Hz 尺度换算、VTI/HTI 参数；
* **闭环**：EFWI → RTM → 倾角约束三维建模 → 反馈下一轮反演。

---

## 1. 快速开始

环境已经配好独立虚拟环境 `.venv`（Python 3.14）。所有命令在项目根目录执行：

```powershell
# 环境自检（可选）
.venv\Scripts\python.exe main.py --check

# 快速端到端演示：EFWI + RTM + 倾角 + 三维建模
.venv\Scripts\python.exe scripts\run_demo.py --iterations 3

# 或者用模块入口
.venv\Scripts\python.exe -m efwi.cli demo --quick

# 测井 LAS -> 初始模型 + 尺度对比
.venv\Scripts\python.exe scripts\run_welllog.py

# 反演—成像—三维建模闭环
.venv\Scripts\python.exe scripts\run_closed_loop.py --outer 2 --iterations 2

# 梯度有限差分方向导数检验（打印逐参数误差并写 JSON）
.venv\Scripts\python.exe scripts\check_gradient.py

# 伪 Hessian 预条件 + 多尺度频带对比图/报告
.venv\Scripts\python.exe scripts\run_preconditioning_demo.py

# 实际 SEG-Y：读取 → QC → 预处理 → 写回
.venv\Scripts\python.exe scripts\run_segy_qc.py data\line.sgy --out runs/segy_qc

# 全部单元测试
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

快速演示默认网格 `81×61`、4 炮、41 道，单次运行约 10–30 秒（CPU）；`--full` 使用
`121×101`、6 炮、81 道，并把每个频带迭代数调到 25 次，耗时显著增加，适合正式实验。

输出默认在 `runs/demo/`：

```
runs/demo/
├── fwi/
│   ├── model_initial.npz / model_final.npz / model_true.npz
│   ├── history.json
│   └── data.npz
├── models.png              # 初始 / 反演 / 真模型 vp-vs-rho 对比
├── convergence.png         # 目标函数收敛曲线
├── data_fit.png            # 观测/合成/残差炮集
├── rtm.png                 # RTM 保幅成像
├── dips.png                # 反射面倾角 + 结构张量相干性
├── rtm.npz / dips.npz
└── model3d.npz / model3d.vti
```

---

## 2. 目录结构

```
SeismicProjectEFWI/
├── efwi/
│   ├── config.py          # 网格/频带/拟牛顿/约束配置
│   ├── model.py           # 弹性参数模型、刚度矩阵、Gardner
│   ├── wavelet.py         # Ricker/Gabor/高斯导数子波
│   ├── geometry.py        # 观测系统、双线性插值权重
│   ├── filters.py         # 零相位 Wiener 带通、多尺度频带
│   ├── params.py          # 参数打包/对数重参数化
│   ├── fd/
│   │   ├── kernels.py     # Numba 正演/伴随/成像核
│   │   └── operator.py    # 弹性波正演、梯度、RTM 算子
│   ├── io.py              # SEG-Y 读取/写回、观测系统映射、QC
│   ├── preprocess.py      # 直达波/FK/均衡/相位/子波估计
│   ├── topography.py      # 高程静校正、替换速度、浮动 datum
│   ├── fd3d/              # 三维弹性波正演（梯度实验性，默认禁用）
│   ├── adjoint.py         # 残差/目标函数/伴随源/逐炮累积/有限差分检验
│   ├── gradient.py        # 刚度梯度 -> vp/vs/rho/epsilon/delta
│   ├── pseudo_hessian.py  # 伪 Hessian 照明补偿
│   ├── optimize.py        # L-BFGS + 抛物线线搜索
│   ├── constraints.py     # Dykstra / ADMM / 凸集投影
│   ├── inversion.py       # 多尺度 EFWI 主循环
│   ├── imaging.py         # RTM、P/S 分离、倾角/反射点
│   ├── welllog.py         # LAS 解析、初始模型、尺度换算
│   ├── synthetic.py       # 合成模型与演示观测系统
│   ├── workflow.py        # 反演—成像—三维建模闭环
│   ├── viz.py             # Matplotlib 出图
│   ├── demo.py            # 端到端演示封装
│   └── cli.py             # 命令行入口
├── scripts/               # 可直接右键运行的脚本
├── tests/                 # unittest 回归测试
├── data/example_well.las  # 示例测井
├── docs/                  # 设计与公式说明
├── main.py                # 环境自检 / 统一入口
└── requirements.txt
```

---

## 3. 文档要求与代码对应

| 文档要求 | 实现位置 | 关键参数/说明 |
|---|---|---|
| 弹性波正演，反演 λ、μ、ρ 或 vp、vs、ρ | `fd/operator.py`, `gradient.py` | 原生梯度 `C11/C13/C33/C44/rho`，链式转换 |
| 残差/目标函数/伴随源 | `adjoint.py` | `r=d_syn-d_obs`，`J=0.5||r||²/E_obs`，`P^T r` |
| 伴随态法梯度 | `adjoint.py`, `fd/kernels.py::adjoint_step` | 离散转置，单炮/多炮累积，有限差分校验 |
| 梯度数值检验 | `adjoint.py::check_gradient`, `scripts/check_gradient.py` | 中心差分方向导数 |
| 多尺度 Wiener 带通：8 Hz → 15 Hz | `filters.py`, `inversion.py` | 默认 `StageConfig(2,8,25)`、`(3,15,25)` |
| 伪 Hessian 对角照明补偿 | `fd/operator.py`, `pseudo_hessian.py` | 高频+无限孔径近似，阻尼倒数预条件 |
| L-BFGS + 抛物线线搜索 | `optimize.py` | memory=8，Armijo + 三点抛物线 |
| 参数上下界 | `constraints.py::BoxSet` | vp/vs/rho/epsilon/delta |
| 侧向/垂向梯度上限 | `constraints.py::GradientBoundSet` | ADMM + DCT-II（Neumann 边界） |
| Gardner 密度—速度关系 | `constraints.py::GardnerTube` | log-log 空间线性管道，解析投影 |
| Dykstra 交替投影 + ADMM | `constraints.py` | 每个集合独立增量，避免错误收敛 |
| 测井初始模型 | `welllog.py` | LAS 解析、曲线延拓、井间插值 |
| 测井/地震尺度换算 | `welllog.py::design_scale` | 每波长 8 点 + CFL |
| VTI/HTI 各向异性 | `model.py`, `fd/` | 二维 P-SV 刚度 C11/C13/C33/C44 |
| RTM 保幅成像 | `imaging.py`, `fd/operator.py::rtm_shot` | P/S 分离 + 源照明归一化 |
| 反射点/倾角 | `imaging.py::extract_dips` | 结构张量 + 局部极大值 |
| 反演—成像闭环 | `workflow.py` | 三维建模并反馈中心剖面 |

---

## 4. 核心 API 示例

### 4.1 正演

```python
import numpy as np
from efwi.config import GridConfig
from efwi.model import ElasticModel, gardner_density
from efwi.geometry import Survey
from efwi.fd import Elastic2DOperator

grid = GridConfig(nx=121, nz=101, dx=10.0, dz=10.0, sponge=16)
vp = 1800 + 1.7 * grid.z[None, :] * np.ones((grid.nx, 1))
model = ElasticModel(vp, vp / 1.8, gardner_density(vp), dx=10.0, dz=10.0)

shots = np.array([[200.0, 180.0], [600.0, 180.0], [1000.0, 180.0]])
recv = np.array([[x, 180.0] for x in np.linspace(200.0, 1000.0, 81)])
survey = Survey.from_positions(grid, shots, recv, f0=10.0, vmin=1600.0, vmax=4200.0)

op = Elastic2DOperator(model, survey, grid, dtype="float32", storage_mb=512)
data = op.forward_all()             # (ns, nt, nr, 2)
```

### 4.2 EFWI

```python
from efwi.config import FWIConfig
from efwi.inversion import ElasticFWI

config = FWIConfig()                # 默认 8 Hz/25 次 + 15 Hz/25 次
fwi = ElasticFWI(initial_model, survey, true_model=true_model, grid=grid, config=config)
result = fwi.run()
print(result.metrics())
result.model.save("model_final.npz")
```

### 4.3 RTM + 倾角

```python
from efwi.imaging import reverse_time_migration, top_mute, depth_taper, extract_dips

data_rtm = top_mute(data, survey.dt, survey.shots, survey.receivers, velocity=1600.0)
rtm = reverse_time_migration(result.model, survey, data_rtm, grid=grid)
rtm.image = depth_taper(rtm.image, grid.z, 180.0, 300.0)
dip = extract_dips(rtm.image, grid.dx, grid.dz)
```

### 4.4 测井

```python
from efwi.welllog import WellLog, build_initial_model, logging_seismic_scales

log = WellLog.from_las("data/example_well.las", x=0.0)
model0 = build_initial_model([log], grid, anisotropy="vti")
print(logging_seismic_scales(vp=3000, f_logging=10000, f_seismic=30))
```

### 4.5 残差、伴随源与梯度

```python
from efwi.adjoint import (
    data_residual, objective, adjoint_source,
    compute_adjoint_gradients, check_gradient,
)
from efwi.fd import Elastic2DOperator

op = Elastic2DOperator(model, survey, grid, dtype="float64", storage_mb=512)
observed = op.forward_all(true_model)
result = compute_adjoint_gradients(op, model, observed)   # vp/vs/rho
g_vp, g_vs, g_rho = result.g_vp, result.g_vs, result.g_rho
print("misfit =", result.misfit, "|g| =", result.gradient_norm())

# 单炮原始梯度与逐炮累积
shot0 = op.compute_shot_gradient(0, model, observed[0])
print(shot0.misfit_raw, shot0.g["c11"].shape)

# 中心差分方向导数检验
direction = np.exp(-((grid.x[:, None] - 0.5 * grid.x[-1]) ** 2
                     + (grid.z[None, :] - 0.5 * grid.z[-1]) ** 2) / (2 * 150.0 ** 2))
checks = check_gradient(
    op, model, observed,
    {"vp": direction, "vs": direction, "rho": direction},
    eps={"vp": 2.0, "vs": 1.0, "rho": 1.0},
)
for param, row in checks.items():
    print(param, row["adjoint"], row["finite_difference"], row["rel_error"])
```

完整公式与测试结果见 [docs/EFWI_梯度推导.md](docs/EFWI_梯度推导.md)。

---

## 5. 数值方法要点

### 5.1 正演

二维 P-SV 速度—应力一阶方程（Virieux 交错网格）：

```
rho dvx/dt = d(sxx)/dx + d(sxz)/dz
rho dvz/dt = d(sxz)/dx + d(szz)/dz
dsxx/dt = C11 dvx/dx + C13 dvz/dz
dszz/dt = C13 dvx/dx + C33 dvz/dz
dsxz/dt = C44 (dvx/dz + dvz/dx)
```

空间二阶、时间二阶；四周海绵吸收层；CFL 自动检查。各向同性时
`C11=C33=rho*vp^2`、`C13=rho*(vp^2-2vs^2)`、`C44=rho*vs^2`；VTI/HTI 用 Thomsen
参数计算刚度。

### 5.2 梯度

正演写成 `v' = v + dt/rho D s`、`s' = s + dt C E v'`，伴随是其离散转置：

```
v~ = v~' + dt E^T C^T s~'
s~ = s~' + dt D^T (v~/rho)
```

梯度：

```
g_C11 = ∫ s~xx exx dt
g_C13 = ∫ (s~xx ezz + s~zz exx) dt
g_C33 = ∫ s~zz ezz dt
g_C44 = ∫ s~xz exz dt
g_rho = -∫ v~^{n-1/2} · a / rho dt
```

密度项必须用**本步推进后的**速度伴随 `v~^{n-1/2}`（速度更新方程的
Lagrange 乘子）；用推进前的 `v~^{n+1/2}` 会随记录长度累积误差。再链式转换到
`(vp, vs, rho)`。完整推导、符号约定与实现顺序见
[docs/EFWI_梯度推导.md](docs/EFWI_梯度推导.md)。

`tests/test_gradient.py` 和 `scripts/check_gradient.py` 用中心差分方向导数
逐参数检验，vp/vs/rho 的相对误差均在 `10^-3` 以内；演示脚本中 vp/vs 约
`10^-6`、rho 约 `10^-7`。多炮累积通过 `compute_shot_gradient` +
`ShotGradientAccumulator` 显式实现。

### 5.3 多尺度 Wiener 与调度

以当前炮集为参考，构造 Butterworth 带通目标 `y`，频域最小二乘
`H(f)=Pxy/(Pxx+eps)`，加频带掩膜和平滑；作用到所有炮集。零相位，不引入走时偏差。

每一阶段同时把响应作用到震源子波，使用**带限子波正演**，因此当前阶段的
伪 Hessian 对应当前频带照明。默认 2–8 Hz、3–15 Hz 两个阶段；`validate_stages`
强制高截频单调递进且低于 0.95 Nyquist。详见
[docs/EFWI_梯度预处理与多尺度.md](docs/EFWI_梯度预处理与多尺度.md)。

### 5.4 伪 Hessian

高频 + 无限孔径近似下只保留参数空间虚拟源的对角：

```
H_vp = 8*rho^2*vp^2*∫theta^2 dt,  theta=exx+ezz
H_vs = 16*rho^2*vs^2*∫(exx^2+ezz^2)dt + 4*rho^2*vs^2*∫exz^2dt
H_rho = H_rho_mass + 刚度链式项
```

预条件：

```
g_pre = g / (H + eta*max(H) + water_level*max(H) + min_h)
```

并用中位增益 cap（`ph_max_gain=20`）、非有限值清零防止深部弱照明放大。
`H` 的高斯平滑只用于稳定对角估计，不替代上述虚拟源计算。

### 5.5 约束投影

* 上下界、Gardner 管道：解析投影；
* 梯度上限：ADMM 求解 `min 0.5||u-m||² s.t. |Dx u|<=gx, |Dz u|<=gz`，
  用 DCT-II 对角化 Neumann 边界下的 `D^T D`；
* 外层 Dykstra 交替投影到交集；也提供 consensus ADMM。
* 每个凸集维护**独立**的 Dykstra 增量，这是正确收敛的关键。

---

## 6. 测井尺度

`efwi.welllog.logging_seismic_scales` 给出的典型对比（vp=3000 m/s）：

| 尺度 | 主频 | 波长 | dx（8 点/波长） | dt |
|---|---:|---:|---:|---:|
| 测井 | 10 kHz | 0.3 m | 0.0375 m | 4.0 μs |
| 地面地震 | 30 Hz | 100 m | 12.5 m | 1.33 ms |

频率比约 333，网格/时间步比约 0.003。直接把测井网格用于地震频带会导致
严重频散与不稳定，因此代码中所有网格/时间步由 `design_scale` 或
`Survey.from_positions` 按目标频带重新设计。

---

## 7. 性能与内存

* 正演/伴随核用 Numba `njit(cache=True)`，CPU 多核并行留给后续扩展；
* 梯度需要保存正演敏感场 `exx/ezz/exz/ax/az`，默认 `storage_mb=512` 自动
  抽稀（`stride`）；小模型 `stride=1`；
* 快速演示在普通笔记本 CPU 上约 10–30 秒；
* 全尺寸 `121×101`、6 炮、50 次迭代的耗时约数十分钟到数小时，取决于抽稀与线搜索次数。

---

## 8. 限制与后续工作

* 当前为 **2D** P-SV；三维扩展需要把刚度张量扩展到 3D 并改用三维交错网格；
* 海绵吸收层不是 CPML，强反射/大角度入射时仍有少量边界反射；
* 震源子波假设已知；实际资料可加子波估计（维纳反褶积/伴随源反演）；
* 密度参数在弹性 FWI 中仍是最不敏感的参数，建议结合 Gardner 约束或分参数反演；
* 三维建模目前是“倾角引导的走向外推”，真实工区应使用多测线/层位插值；
* 各向异性反演默认固定 `epsilon/delta`，打开 `FWIConfig.invert_anisotropy=True`
  可同时反演，但需要更强约束。

详细公式与设计说明见：

* [docs/EFWI_设计与实现.md](docs/EFWI_设计与实现.md) — 正演、多尺度、伪 Hessian、约束、RTM、测井、闭环；
* [docs/EFWI_梯度推导.md](docs/EFWI_梯度推导.md) — 残差/目标函数、伴随源、离散伴随、密度时间顺序、链式梯度、逐炮累积、有限差分检验结果与接口；
* [docs/EFWI_梯度预处理与多尺度.md](docs/EFWI_梯度预处理与多尺度.md) — 伪 Hessian 定义/计算、预条件稳定性、Wiener 频带与调度策略、对比图和测试报告；
* [docs/EFWI_实际资料与P0.md](docs/EFWI_实际资料与P0.md) — SEG-Y、预处理、自由表面/地形和三维模块的状态、接口与限制。
