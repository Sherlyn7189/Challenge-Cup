# EFWI 伴随源构造与梯度计算

本文档对应代码：

* `efwi/fd/kernels.py`：正演、伴随、梯度累积核；
* `efwi/fd/operator.py`：`Elastic2DOperator` 正演/单炮梯度/多炮累积；
* `efwi/adjoint.py`：残差、目标函数、伴随源、梯度累积与有限差分检验 API；
* `efwi/gradient.py`：刚度梯度到 vp/vs/rho/各向异性参数的链式转换。

---

## 1. 离散正演与符号

二维 P-SV 速度—应力方程（与正演代码完全一致）：

$$
\begin{aligned}
\rho \dot v_x &= \partial_x \sigma_{xx}+\partial_z \sigma_{xz},\\
\rho \dot v_z &= \partial_x \sigma_{xz}+\partial_z \sigma_{zz},\\
\dot\sigma_{xx} &= C_{11}\partial_x v_x+C_{13}\partial_z v_z,\\
\dot\sigma_{zz} &= C_{13}\partial_x v_x+C_{33}\partial_z v_z,\\
\dot\sigma_{xz} &= C_{44}(\partial_z v_x+\partial_x v_z).
\end{aligned}
$$

记 $v=(v_x,v_z)$、$s=(\sigma_{xx},\sigma_{zz},\sigma_{xz})$，交错网格离散后一步为

$$
\boxed{
v^{n+1/2}=v^{n-1/2}+\Delta t\,M^{-1}\left(Ds^n+f^n\right),\qquad
s^{n+1}=s^n+\Delta t\,C E v^{n+1/2}
}
$$

其中：

* $D$：应力散度算子（中心/角点 → 速度点）；
* $E$：应变率算子（速度点 → 中心/角点）；
* $M=\operatorname{diag}(\rho_x,\rho_z)$：密度质量矩阵；
* $C$：刚度矩阵（各向同性或 VTI/HTI）；
* $f^n$：震源项，爆炸源注入 $s$，力源注入 $v$。

数据 $d^n=P v^{n+1/2}$，$P$ 为检波点双线性采样算子。记录后乘吸收层阻尼
$G=\operatorname{diag}(e^{-d\Delta t})$。

---

## 2. 残差与目标函数

合成数据 $d^{syn}$、观测数据 $d^{obs}$：

$$
r=d^{syn}-d^{obs},\qquad
E_{obs}=\sum_{s}\|d_s^{obs}\|^2,
$$

$$
\boxed{
J=\frac{1}{2}\frac{\|r\|^2}{E_{obs}}
=\frac{1}{2E_{obs}}\sum_{s}\sum_{n,r,c}\left(d^{syn}-d^{obs}\right)^2
}
$$

代码：

```python
from efwi.adjoint import data_residual, objective

res = data_residual(synthetic, observed)   # r = syn - obs, energy = ||obs||^2
J = objective(res)                         # 0.5 * ||r||^2 / energy
```

`ShotGradient` 保存每炮的**未归一化** $0.5\|r_s\|^2$，最后统一除以全局
$E_{obs}$，避免逐炮归一化改变炮间权重。

---

## 3. 伴随源

对 $J=\frac{1}{2E_{obs}}\|Pv-d\|^2$ 有

$$
\boxed{\frac{\partial J}{\partial v^{n+1/2}}=\frac{1}{E_{obs}}P^T r^n}
$$

数据空间伴随源就是 $P^T r^n$。代码中：

1. `adjoint_source(residual, gain=...)` 返回数据空间的 $r$（可选增益）；
2. `kernels.inject_adjoint_source(...)` 用与正演 `record_data` **完全相同**的
   双线性权重，把 $r$ 散加到伴随速度场。因此 $P^T$ 是 $P$ 的严格转置；
3. 若目标函数已在 `compute_gradient` 中按 $E_{obs}$ 归一化，伴随源本身
   不必再乘 $1/E_{obs}$，最终梯度统一除 $E_{obs}$ 即可。

实现位置：`Elastic2DOperator._shot_gradient`。

---

## 4. 伴随波场

### 4.1 反向递推

将正演两步的转置按**逆序**作用，得到实现采用的离散伴随：

$$
\boxed{
\tilde v^{n-1/2}=\tilde v^{n+1/2}+\Delta t\,E^T C^T\tilde s^{n+1},
\qquad
\tilde s^n=\tilde s^{n+1}+\Delta t\,D^T M^{-1}\tilde v^{n-1/2}
}
$$

注意：

1. 先注入残差：
   $\tilde v^{n+1/2}\leftarrow\tilde v^{n+1/2}+P^T r^n$；
2. $\tilde s^n$ 必须用**刚更新完的** $\tilde v^{n-1/2}$，而不是旧的
   $\tilde v^{n+1/2}$；这对应完整 Jacobian 中的
   $\Delta t^2 C E M^{-1}D$ 耦合项；
3. 正向每步末尾乘 $G$，反向在每步推进前先乘 $G$（$G^T=G$）。

代码 `kernels.adjoint_step` 用双缓冲实现：第一遍算新速度伴随，第二遍算新
应力伴随，最后清零边界。`tests/test_adjoint.py` 直接用随机场验证

$$
\langle F x,y\rangle=\langle x,F^T y\rangle .
$$

### 4.2 完全匹配层/海绵层

阻尼是对角算子，转置仍为乘 $G$。由于正演在记录之后阻尼，反向必须先
“解阻尼”（仍乘 $G$）再注入残差/累加梯度，否则伴随方程与离散正演不一致。

---

## 5. 梯度表达式

### 5.1 刚度梯度

$$
\boxed{
\begin{aligned}
g_{C11}&=\sum_n \Delta t\,\tilde\sigma_{xx}^{n+1}\,\dot\varepsilon_{xx}^{n},\\
g_{C13}&=\sum_n \Delta t\left(\tilde\sigma_{xx}^{n+1}\dot\varepsilon_{zz}^{n}
+\tilde\sigma_{zz}^{n+1}\dot\varepsilon_{xx}^{n}\right),\\
g_{C33}&=\sum_n \Delta t\,\tilde\sigma_{zz}^{n+1}\,\dot\varepsilon_{zz}^{n},\\
g_{C44}&=\sum_n \Delta t\,\tilde\sigma_{xz}^{n+1}\,\dot\varepsilon_{xz}^{n}.
\end{aligned}}
$$

这里 $\tilde s^{n+1}$ 是**本步对应的伴随应力**（代码中在 `adjoint_step` 之前的
`saxx/sazz/saxz`），$\dot\varepsilon^n$ 是正演保存的应变率。$C_{44}$ 位于
角点，反传时按正演的 $1/4$ 平均分配到四个中心。

### 5.2 密度梯度

密度只出现在动量方程的质量项：

$$
\frac{\partial v^{n+1/2}}{\partial \rho}
=-\Delta t\,\frac{Ds^n+f^n}{\rho^2}
=-\Delta t\,\frac{a^{n+1/2}}{\rho}.
$$

因此

$$
\boxed{
g_\rho^{dir}=-\sum_n \Delta t\,\tilde v^{n-1/2}\cdot
\frac{a^{n+1/2}}{\rho}
}
$$

**关键细节：使用 $\tilde v^{n-1/2}$（本步推进后的速度伴随），而不是
$\tilde v^{n+1/2}$。** 它是速度更新方程的 Lagrange 乘子。若误用推进前的
$\tilde v^{n+1/2}$，短记录时误差很小，但随记录长度累积，测试中 nt=30 时
密度方向导数会偏大约 6%；修正后误差降到 $10^{-7}$ 量级（见第 8 节）。

代码：

```python
kernels.adjoint_stiffness_gradient(...)   # 在 adjoint_step 之前
kernels.adjoint_step(...)                 # -> vax2, vaz2 (v~^{n-1/2})
kernels.adjoint_density_gradient(vax2, vaz2, ax, az, rho_x, rho_z, ...)
```

`rho_x/rho_z` 是正演时的算术平均；反传时各分 $1/2$，边界重复节点按
`np.pad(mode="edge")` 的等价关系折叠回物理边界。

### 5.3 链式转换到 vp/vs/rho

各向同性：

$$
\begin{aligned}
g_{v_p}&=2\rho v_p\left(g_{C11}+g_{C13}+g_{C33}\right),\\
g_{v_s}&=2\rho v_s\left(-2g_{C13}+g_{C44}\right),\\
g_\rho&=g_\rho^{dir}
+v_p^2\left(g_{C11}+g_{C33}\right)
+\left(v_p^2-2v_s^2\right)g_{C13}
+v_s^2g_{C44}.
\end{aligned}
$$

VTI/HTI 的 $C_{ij}(v_p,v_s,\rho,\epsilon,\delta)$ 偏导见
`efwi/gradient.py::_stiffness_derivatives`；若 `invert_anisotropy=True`，
额外给出

$$
g_\epsilon = 2\rho v_p^2\,g_{C11}\ \text{(VTI)},\qquad
g_\delta = \rho\frac{\partial\sqrt{Q}}{\partial\delta}g_{C13}.
$$

---

## 6. 多炮累积与归一化

对每炮独立计算原始梯度 $g_p^{(s)}$，然后

$$
\boxed{
g_p=\frac{1}{E_{obs}}\sum_{s=1}^{N_s}g_p^{(s)},\qquad
J=\frac{1}{E_{obs}}\sum_{s=1}^{N_s}\frac12\|r_s\|^2
}
$$

伪 Hessian 也按炮求和（不做能量归一化，用于相对照明补偿）。`AdjointResult.g`
已按 $E_{obs}$ 归一化，`AdjointResult.h` 保留原始照明量纲。

代码接口：

```python
from efwi.adjoint import ShotGradientAccumulator, compute_adjoint_gradients

result = compute_adjoint_gradients(op, model, observed)   # vp/vs/rho
acc = ShotGradientAccumulator(model, ("vp", "vs", "rho"), obs_energy=E)
for s in range(ns):
    acc.add_shot(op.compute_shot_gradient(s, model, observed[s]))
result2 = acc.finalize()
```

`result.shots` 保存逐炮的原始梯度、misfit、合成/残差，可直接检查哪一炮
贡献异常。

---

## 7. 离散一致性要点

1. **转置严格**：`adjoint_step` 是 $F^T$，不是连续方程的近似离散；用
   `tests/test_adjoint.py` 的内积恒等式验证。
2. **时间顺序**：残差注入、刚度梯度、速度伴随推进、密度梯度、应力伴随
   推进的顺序不能交换；密度必须用推进后的速度伴随。
3. **阻尼**：正演末尾乘 $G$，反向开始乘 $G$。
4. **平均算子**：$C_{44}$ 角点平均、$\rho_x/\rho_z$ 算术平均的反向分配
   与正演一致；`np.pad(mode="edge")` 产生的重复节点在边界折叠。
5. **存储抽稀**：内存不足时 `ForwardStorage.stride>1`，梯度用最近存储时刻
   的场近似；`stride=1` 时是严格离散梯度。测试均用 `stride=1`。

---

## 8. 有限差分方向导数检验

### 8.1 方法

选方向 $d$，中心差分

$$
D_{FD}=\frac{J(m+\epsilon d)-J(m-\epsilon d)}{2\epsilon},
$$

与伴随方向导数

$$
D_{ADJ}=\sum_p g_p\,d_p
$$

比较相对误差。检验脚本：`scripts/check_gradient.py`；单元测试：
`tests/test_gradient.py::test_directional_derivative_matches_fd`。

### 8.2 当前测试结果

**单元测试**（36×30、2 炮、160 步）：

| 方向 | 参数 | 伴随 | 有限差分 | 相对误差 |
|---|---|---:|---:|---:|
| gaussian | vp | -5.312574e-09 | -5.312430e-09 | 2.72e-05 |
| gaussian | vs | -9.141950e-09 | -9.141787e-09 | 1.79e-05 |
| gaussian | rho | 2.137459e-11 | 2.136891e-11 | 2.66e-04 |
| random_smooth | vp | -2.137544e-09 | -2.137610e-09 | 3.07e-05 |
| random_smooth | vs | -1.820402e-09 | -1.820393e-09 | 4.93e-06 |
| random_smooth | rho | -7.983630e-10 | -7.983629e-10 | 8.51e-08 |

**演示脚本**（81×61、2 炮、658 步，`scripts/check_gradient.py`）：

| 方向 | 参数 | 伴随 | 有限差分 | 相对误差 |
|---|---|---:|---:|---:|
| gaussian | vp | 4.642620e-04 | 4.642592e-04 | 5.84e-06 |
| gaussian | vs | 2.175769e-05 | 2.175763e-05 | 2.68e-06 |
| gaussian | rho | -1.996298e-06 | -1.996299e-06 | 2.31e-07 |
| random_smooth | vp | 7.764439e-05 | 7.764429e-05 | 1.26e-06 |
| random_smooth | vs | -2.123332e-05 | -2.123332e-05 | 8.11e-09 |
| random_smooth | rho | -1.861060e-06 | -1.861062e-06 | 9.32e-07 |

rho 的高斯方向在单元测试中误差略大，是因为密度总梯度中**质量项与刚度
链式项接近抵消**（两项约 $7\times10^{-9}$，总梯度约 $2\times10^{-11}$），
相对误差被放大；绝对误差约 $4\times10^{-13}$。演示脚本中方向/尺度不同，
相对误差为 $2.3\times10^{-7}$。逐项检验：

* 仅质量项（固定 C，只改 $\rho$）：方向导数误差 $<10^{-6}$；
* 仅刚度链式项（固定质量，只改 C）：误差 $4.5\times10^{-8}$。

### 8.3 密度时间顺序回归

同一随机观测、同模型，只改记录长度，比较密度方向导数：

| nt | 修正前 adj/fd | 修正后 adj/fd |
|---:|---:|---:|
| 3 | 0.9982 | 0.9999998 |
| 5 | 1.0474 | 1.0000001 |
| 10 | 1.0632 | 1.0000001 |
| 30 | 1.0608 | 1.0000001 |

说明密度项必须用 $ṽ^{n-1/2}$ 的顺序修正。

### 8.4 运行

```powershell
.venv\Scripts\python.exe scripts\check_gradient.py
.venv\Scripts\python.exe -m unittest tests.test_gradient -v
```

当前完整测试集 **46 个测试全部通过**（含各向异性 epsilon/delta 梯度、
单炮/多炮累积、伴随转置恒等式）。

---

## 9. 接口说明

| 接口 | 作用 |
|---|---|
| `efwi.adjoint.data_residual(syn, obs)` | 残差与观测能量 |
| `efwi.adjoint.objective(res)` | 归一化/原始目标函数 |
| `efwi.adjoint.adjoint_source(res, gain=1)` | 数据空间伴随源 |
| `Elastic2DOperator.compute_shot_gradient(s, model, obs_s)` | 单炮原始梯度 |
| `Elastic2DOperator.compute_gradient(model, obs, keep_shot_gradients=False)` | 全部炮归一化梯度（默认不保留逐炮数据） |
| `efwi.adjoint.ShotGradientAccumulator` | 逐炮累积 + 归一化 |
| `efwi.adjoint.compute_adjoint_gradients(op, model, obs)` | 高层 API，返回 vp/vs/rho |
| `efwi.adjoint.directional_derivative(...)` | 中心差分方向导数 |
| `efwi.adjoint.check_gradient(...)` | 逐参数有限差分检验 |
| `efwi.gradient.convert_gradient(...)` | 原生 C/rho -> vp/vs/rho |

`ShotGradient` 字段：

| 字段 | 含义 |
|---|---|
| `misfit_raw` | $0.5\sum r^2$（未归一化） |
| `g` | `c11/c13/c33/c44/rho` 原生梯度（物理网格） |
| `h` | 伪 Hessian 对角 |
| `synthetic/residuals` | 单炮合成数据与残差 |
| `storage_stride` | 正演存储抽稀步长 |

---

## 10. 复现命令

```powershell
# 梯度方向导数检验（打印表格并写 JSON）
.venv\Scripts\python.exe scripts\check_gradient.py

# 全部测试
.venv\Scripts\python.exe -m unittest discover -s tests -v

# 完整 EFWI + RTM 演示
.venv\Scripts\python.exe scripts\run_demo.py --iterations 3
```
