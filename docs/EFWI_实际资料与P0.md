# 实际资料适配（P0）实现说明

本文档记录 P0 四项工作的代码、接口、验证状态和边界。

| P0 项 | 状态 | 说明 |
|---|---|---|
| SEG-Y 读取与观测系统映射 | 完成 | `efwi/io.py`，支持分炮、固定接收点、坐标 scalar、缺失道 mask、写回 |
| 实际资料预处理 | 完成 | `efwi/preprocess.py`，直达波切除、FK、均衡、相位、尖峰、子波估计 |
| 自由表面与地形 | 部分完成 | 平自由表面（2D）+ 高程静校正/浮动 datum；逐列不规则自由表面未完成 |
| 三维弹性 FWI | 部分完成 | 三维正演可用；三维伴随梯度未通过验收，默认禁用 |

---

## 1. SEG-Y 读取

```python
from efwi.io import read_segy_gathers, gather_quality_control

g = read_segy_gathers(
    "data/line.sgy",
    shot_field="FieldRecord",
    source_x_field="SourceX",
    source_z_field="SourceDepth",
    group_x_field="GroupX",
    group_z_field="ReceiverGroupElevation",
    scalar_field="SourceGroupScalar",
)
print(g.summary())
print(gather_quality_control(g))
```

内部数据形状：

```
data:  (ns, nt, nr, ncomp)
mask:  (ns, nr, ncomp)      # 缺失道为 False
```

特点：

* 按 `FieldRecord` 分炮，按 `GroupX/GroupZ` 合并共同接收点；
* 支持 `SourceGroupScalar` 正/负值；
* 支持 `component_field="TraceIdentificationCode"` 多分量；
* 缺失道补零并保留 mask；
* `to_survey(wavelet)` 可直接转成 EFWI 的 `Survey`；
* `write_segy_gathers` 可导出处理结果/QC。

注意：SEG-Y 是 3D 坐标；当前 EFWI 正演是 2D，默认把 `(SourceX, SourceZ)`
映射成 `(x,z)`。三维数据请使用 `SourceX/SourceY` 或后续 3D 模块。

---

## 2. 实际资料预处理

`PreprocessConfig` 默认步骤：

1. **直达波切除**：`t_mute = offset / velocity * factor`，切到前能量；
2. **FK 视速度滤波**：保留 `vmin <= f/|k| <= vmax`，压制低速面波/线性干扰；
3. **道均衡**：每道 RMS 归到中位 RMS，最大增益 8；
4. **球面扩散补偿**（可选 `power`）；
5. **常相位旋转**（Hilbert 解析信号）；
6. **尖峰压制**（中值滤波 + MAD 阈值）。

```python
from efwi.preprocess import PreprocessConfig, preprocess_gathers

cfg = PreprocessConfig(
    direct_mute_velocity=1500.0,
    fk_vmin=900.0, fk_vmax=7000.0,
    trace_balance=True,
    phase_deg=15.0,
)
processed, qc = preprocess_gathers(gathers, cfg)
```

子波估计：

```python
from efwi.preprocess import estimate_source_wavelet

est = estimate_source_wavelet(
    observed=obs_direct,       # (nt, ntraces)
    synthetic=syn_direct,      # 平滑模型正演直达波
    reference_wavelet=w_ref,   # 当前使用的子波
    dt=g.dt,
    time_window=(0.05, 0.35),
)
w_est = est.wavelet
print("coherence:", est.coherence)
```

原理：`H = Pxy/(Pxx+eps)`，`w_est = IFFT(FFT(w_ref)*H)`；合成/观测的
Green's 函数近似相同。

---

## 3. 自由表面与地形

### 3.1 平自由表面

```python
from efwi.fd import Elastic2DOperator

op = Elastic2DOperator(
    model, survey, grid,
    free_surface=True,          # 顶部不设海绵层
)
data = op.forward_all()
```

实现：

* 顶部 `σzz=0`、`σxz=0`；
* 顶部吸收层关闭；
* 正向/伴随/RTM 都应用相同边界条件；
* 适用于平地表或已做静校正的浮动 datum。

### 3.2 起伏地形：高程静校正 + 替换速度

```python
from efwi.topography import Topography, apply_elevation_statics
from efwi.topography import fill_air_with_replacement

topo = Topography(x_surface, z_surface)       # z 正向下
corrected, qc = apply_elevation_statics(
    gathers, topo, datum=0.0, replacement_velocity=1500.0
)
model_float = fill_air_with_replacement(model, topo, replacement_velocity=1500.0)
```

这是工程上常用的“地形静校正 + 浮动 datum”路径。**逐列不规则自由表面**
（每列不同差分模板、起伏地表上的真实应力自由边界）尚未实现，需要修改
三维/二维差分核，属于后续工作。

---

## 4. 三维弹性波

`efwi/fd3d/` 提供：

* `Grid3D`、`ElasticModel3D`、`Survey3D`；
* 三维各向同性速度—应力正演（`Elastic3DOperator.forward_all`）；
* 三维伴随/梯度代码（实验性，默认 **禁用**）。

```python
from efwi.fd3d import Grid3D, ElasticModel3D, Survey3D, Elastic3DOperator

grid = Grid3D(nx=64, ny=64, nz=48, dx=10, dy=10, dz=10)
model3 = ElasticModel3D(vp3, vs3, rho3, 10, 10, 10)
survey3 = Survey3D(shots3, receivers3, nt, dt, wavelet)
op3 = Elastic3DOperator(model3, survey3, grid)
data3 = op3.forward_all()          # (ns, nt, nr, 3)
```

**状态**：三维正演已测试可运行；三维伴随/梯度尚未通过离散内积和有限差分
验收，`compute_gradient` 默认抛出 `NotImplementedError`，避免误用。正式
三维 FWI 需要先修复 `fd3d/kernels3d.py::adjoint_step3d` 并完成：

1. 一步正演/伴随内积恒等式；
2. 单炮方向导数有限差分；
3. 多炮累积与 2D 退化测试；
4. 三维自由表面/地形与 VTI/HTI。

---

## 5. 快速 QC 脚本

```powershell
.venv\Scripts\python.exe scripts\run_segy_qc.py path\to\line.sgy --out runs/segy_qc
```

输出：

```
runs/segy_qc/
├── gathers_raw.npz
├── gathers_processed.npz
├── gathers_processed.sgy
├── raw_gather.png
├── processed_gather.png
└── qc.json
```

---

## 6. 测试

```powershell
.venv\Scripts\python.exe -m unittest tests.test_io tests.test_preprocess tests.test_topography tests.test_fd3d -v
```

当前全套 **61 个测试通过**。已覆盖：

* SEG-Y 写回/读回、坐标 scalar、多分量、QC；
* 直达波切除、FK、均衡、相位、子波估计、预处理流水线；
* 地形静校正、替换速度填充、平自由表面正演；
* 三维正演可运行、三维梯度默认禁用。

**未完成**：三维伴随/梯度数值验收、逐列不规则自由表面、三维 VTI/HTI、
实际 SEG-Y 观测系统到三维 Survey 的自动映射。
