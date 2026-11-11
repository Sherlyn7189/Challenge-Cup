# EFWI 设计与实现说明

本文档给出代码背后的公式、离散细节、参数选择和验证结果，便于把实现规格
与代码逐条对应。

---

## 1. 目标与范围

输入：

* 二维弹性参数模型（`vp, vs, rho`，可选 `epsilon, delta, gamma`）；
* 观测系统（炮点、检波器、子波、记录长度）；
* 观测炮集（或由真模型合成）。

输出：

* 多尺度 EFWI 更新后的弹性模型；
* 目标函数/梯度/照明/约束违反历史；
* RTM 保幅成像、倾角与反射点；
* 可选三维地质模型。

代码定位是**可运行、可验证的研究/工程原型**：所有关键步骤都有单元测试，
梯度用有限差分校验，正演/伴随在离散意义下互为转置。

---

## 2. 弹性波正演

### 2.1 方程

二维 P-SV 速度—应力一阶系统：

$$
\begin{aligned}
\rho \dot v_x &= \partial_x \sigma_{xx} + \partial_z \sigma_{xz},\\
\rho \dot v_z &= \partial_x \sigma_{xz} + \partial_z \sigma_{zz},\\
\dot\sigma_{xx} &= C_{11}\partial_x v_x + C_{13}\partial_z v_z,\\
\dot\sigma_{zz} &= C_{13}\partial_x v_x + C_{33}\partial_z v_z,\\
\dot\sigma_{xz} &= C_{44}(\partial_z v_x + \partial_x v_z).
\end{aligned}
$$

各向同性：

$$
C_{11}=C_{33}=\rho v_p^2,\quad
C_{13}=\rho(v_p^2-2v_s^2),\quad
C_{44}=\rho v_s^2.
$$

VTI（Thomsen）：

$$
\begin{aligned}
C_{11}&=\rho v_p^2(1+2\epsilon),\\
C_{33}&=\rho v_p^2,\\
C_{44}&=\rho v_s^2,\\
C_{13}&=\rho\left[\sqrt{(v_p^2-v_s^2)(v_p^2(1+2\delta)-v_s^2)}-v_s^2\right].
\end{aligned}
$$

HTI 在二维 x-z 平面内由 VTI 绕 z 轴旋转 90° 得到：`C11 <-> C33`，`C13/C44` 不变。
`gamma` 控制 SH 分量，二维 P-SV 正演不使用，但保留供三维扩展和测井解释。

### 2.2 交错网格

场量位置：

| 场 | 位置 |
|---|---|
| `vx` | `(i+1/2, j)` |
| `vz` | `(i, j+1/2)` |
| `sxx, szz` | `(i, j)` |
| `sxz` | `(i+1/2, j+1/2)` |

空间一阶前向差分、时间二阶；密度在速度点做算术平均，`C44` 在角点做四点半平均。
四周为海绵吸收层：阻尼剖面

$$
d(s)=d_{\max}\left(\frac{W-s}{W}\right)^2,\quad d_{\max}=\frac{3 v_{\max}}{W h},
$$

每步乘以 `exp(-d dt)`。CFL 检查：

$$
dt \le 0.5\,\frac{h_{\min}}{v_{\max}\sqrt 2}.
$$

### 2.3 离散正演映射

记 `s=(sxx,szz,sxz)`、`v=(vx,vz)`，一步正演为

$$
v' = v + \Delta t\,M^{-1}D s,\qquad
s' = s + \Delta t\,C E v',
$$

其中 `D` 是散度算子，`E` 是应变率算子，`M=diag(rho)`，`C` 是刚度矩阵。
加入阻尼 `G=diag(exp(-d dt))` 后：

$$
v' = G(v + \Delta t M^{-1}D s),\quad
s' = G(s + \Delta t C E v').
$$

---

## 3. 伴随态梯度

### 3.1 离散转置

目标函数

$$
J=\frac{1}{2}\frac{\|Pv-d\|^2}{\|d\|^2}.
$$

其中 `P` 是检波器采样算子。数据残差 `r=Pv-d` 作为伴随源注入接收点。
由第 2.3 节的正演映射，离散伴随（反向时间）为

$$
\tilde v^{n-1/2}=\tilde v^{n+1/2}+\Delta t\,E^T C^T \tilde s^{n+1},
$$

$$
\tilde s^{n}=\tilde s^{n+1}+\Delta t\,D^T M^{-1}\tilde v^{n-1/2}.
$$

注意第二式必须用**刚更新过的** `v~^{n-1/2}`；这是离散转置的耦合项，若用旧速度会
导致伴随场指数增长。阻尼的转置仍是乘 `G`，并且要在每步反向推进**之前**作用：

```
for n = nt-1 ... 0:
    damp(v~, s~)                  # G^T
    inject residual r[n]          # P^T
    accumulate gradients
    v~ = v~ + dt E^T C^T s~
    s~ = s~ + dt D^T M^{-1} v~
```

### 3.2 梯度

$$
\begin{aligned}
g_{C11} &= \sum_n \Delta t\,\tilde\sigma_{xx}\,\dot\varepsilon_{xx},\\
g_{C13} &= \sum_n \Delta t\,(\tilde\sigma_{xx}\dot\varepsilon_{zz}+\tilde\sigma_{zz}\dot\varepsilon_{xx}),\\
g_{C33} &= \sum_n \Delta t\,\tilde\sigma_{zz}\dot\varepsilon_{zz},\\
g_{C44} &= \sum_n \Delta t\,\tilde\sigma_{xz}\dot\varepsilon_{xz},\\
g_\rho &= -\sum_n \Delta t\,\tilde v^{n-1/2}\cdot a^{n+1/2}/\rho,
\end{aligned}
$$

其中 `a=(D s + f)/rho` 是加速度。密度项使用**本步推进后的**速度伴随
`v~^{n-1/2}`（速度更新方程的 Lagrange 乘子），而不是推进前的
`v~^{n+1/2}`；后者会随记录长度累积误差。完整推导见
[docs/EFWI_梯度推导.md](EFWI_梯度推导.md)。

`C44` 和 `rho` 的梯度按正演时的平均方式（角点→中心、速度点→中心）反向分配；
`np.pad(mode="edge")` 在边界产生的重复节点会折叠回物理边界，保证边界单元
也与离散正演一致。

### 3.3 链式转换

各向同性：

$$
\begin{aligned}
g_{v_p} &= 2\rho v_p(g_{C11}+g_{C13}+g_{C33}),\\
g_{v_s} &= 2\rho v_s(-2g_{C13}+g_{C44}),\\
g_\rho &= g_\rho^{\text{direct}}
+ v_p^2(g_{C11}+g_{C33})
+ (v_p^2-2v_s^2)g_{C13}
+ v_s^2 g_{C44}.
\end{aligned}
$$

VTI/HTI 的 `dC/dp` 见 `efwi/gradient.py::_stiffness_derivatives`。伪 Hessian
用同样的对角链式近似：

$$
H_p \approx \sum_{C} \left(\frac{\partial C}{\partial p}\right)^2 H_C.
$$

### 3.4 验证

`tests/test_gradient.py` 对 `vp/vs/rho` 各取一个光滑方向，比较

$$
\frac{J(m+\epsilon d)-J(m-\epsilon d)}{2\epsilon}
\quad\text{vs.}\quad
\sum g_p d.
$$

修正密度项时间顺序与边界平均后，vp/vs 相对误差约 `10^-5`–`10^-6`，rho
约 `10^-7`–`10^-4`（rho 的高斯方向因质量项/链式项接近抵消，相对误差被放大，
但绝对误差很小）。详细结果、逐项分解和接口说明见
[docs/EFWI_梯度推导.md](EFWI_梯度推导.md)。

在开发过程中还逐点比较了单网格扰动，确认早期的有限差分污染来自扰动
`vmax` 导致的吸收层变化；后将吸收层参考速度固定，并修正密度伴随的时间
顺序与边界折叠，方向导数误差降到 `10^-6` 量级。

---

## 4. 多尺度反演

`filters.WienerFilter`：

1. 对参考炮集加 Tukey 窗、去直流；
2. 用 Butterworth 带通构造目标 `y`；
3. 频域最小二乘：

$$
H(f)=\frac{P_{xy}(f)}{P_{xx}(f)+\epsilon\max P_{xx}};
$$

4. 平滑 `H`，在 `[f_low/2^0.5, f_high*2^0.5]` 外加余弦掩膜；
5. 作用到所有炮集（零相位）。

默认两级：

| 频带 | 迭代次数 | 目的 |
|---|---:|---|
| 2–8 Hz | 25 | 建立大尺度构造，缓解周跳/局部极小 |
| 3–15 Hz | 25 | 刻划细节 |

每个频带重新设计滤波器、重置 L-BFGS。目标函数用观测能量归一化，便于跨频带比较。

---

## 5. 伪 Hessian 照明补偿

全 Hessian 成本不可接受；高频 + 无限孔径近似下只保留对角元。正演时累积：

$$
H_{C11}=\sum \dot\varepsilon_{xx}^2,\quad
H_{C13}=\sum(\dot\varepsilon_{xx}^2+\dot\varepsilon_{zz}^2),\quad
H_{C33}=\sum \dot\varepsilon_{zz}^2,\quad
H_{C44}=\sum \dot\varepsilon_{xz}^2,
$$

$$
H_\rho=\sum\left[\left(\frac{a_x}{\rho_x}\right)^2+\left(\frac{a_z}{\rho_z}\right)^2\right].
$$

预条件：

$$
g^{\text{pre}}_p=\frac{g_p}{H_p+\eta\max H_p}.
$$

`eta` 默认 `1e-3`，`H` 做高斯平滑（默认 2 个网格）。物理意义是补偿中深部
照明不足，使梯度能量在空间上更均衡。

---

## 6. 优化

* 参数化：`vp/vs/rho` 用相对对数 `x=log(m/m_ref)`；`epsilon/delta` 线性；
* L-BFGS 两循环递归，历史长度默认 8；
* 初始 Hessian 标量 `gamma=s·y/y·y`；
* 方向限制：`max_rel_step=0.05`（每次迭代最大相对参数变化 5%）；
* 线搜索：Armijo + 抛物线。给定 `f0, f(alpha), g0·d`，拟合

$$
p(a)=f_0+s_0 a+c a^2,\quad
a^*=-\frac{s_0 a^2}{2(f_1-f_0-s_0 a)},
$$

在 `[0.1a, 3a]` 内取候选，必要时回溯。线搜索的斜率用**真实梯度**
`g_true·d`，而 L-BFGS 方向用预条件梯度，避免非下降方向。

每次迭代的代价约为：

1. 一次线搜索（1–3 次正演）；
2. 在新模型上重新计算目标+梯度+伪 Hessian（正演+伴随）。

---

## 7. 地质/岩石物理约束

### 7.1 凸集

1. **上下界**：`vp∈[1200,6500]`、`vs∈[250,3800]`、`rho∈[1400,3200]`；
2. **梯度上限**：`|∂x m|≤gx`、`|∂z m|≤gz`；
3. **Gardner 管道**：

$$
\left|\ln\rho-\ln a-b\ln v_p\right|\le\tau,
$$

在 `(ln vp, ln rho)` 空间是线性带，凸集，投影有解析解：沿法向
`(-b,1)` 移动。

### 7.2 梯度集合的投影

求解

$$
\min_u \frac12\|u-m\|^2\quad
\text{s.t. } |D_x u|\le g_x,\ |D_z u|\le g_z.
$$

ADMM：

$$
u=\arg\min \frac12\|u-m\|^2+\frac{\rho}{2}\|D_xu-p_x+q_x\|^2+\frac{\rho}{2}\|D_zu-p_z+q_z\|^2,
$$

$$
(I+\rho D_x^TD_x+\rho D_z^TD_z)u=m+\rho D_x^T(p_x-q_x)+\rho D_z^T(p_z-q_z).
$$

用 DCT-II 对角化（Neumann 边界）：

$$
\lambda_x(k)=\left(\frac{2\sin(\pi k/(2N_x))}{\Delta x}\right)^2,
$$

一次 DCT/IDCT 即可解 u。周期 FFT 会在模型边界把左右/上下错误缝合，导致
边界巨大假梯度；DCT 版本修复了该问题。

### 7.3 Dykstra 与 ADMM

Dykstra：

```
x = m
p_i = 0  (每个集合独立)
repeat:
  for i:
    y = x + p_i
    z = Proj_i(y)
    p_i = y - z
    x = z
```

关键：**每个集合维护独立的增量 `p_i`**；所有集合共用一个 `p` 会把其他集合
的修正重新加回去，导致投影失效。

Consensus ADMM 作为备选：

$$
m^{k+1}=\frac{m_0+\rho\sum_i(z_i-u_i)}{1+\rho K},\quad
z_i^{k+1}=\mathrm{Proj}_i(m^{k+1}+u_i^k),\quad
u_i^{k+1}=u_i^k+m^{k+1}-z_i^{k+1}.
$$

### 7.4 默认参数

| 约束 | 默认值 |
|---|---|
| vp 梯度上限 (横向, 垂向) | (8, 12) (m/s)/m |
| vs 梯度上限 | (5, 8) |
| rho 梯度上限 | (4, 6) (kg/m³)/m |
| Gardner | a=310, b=0.25, tol=0.10 |
| Dykstra 外层 | 6 |
| 梯度投影内层 | 20（rho_ADMM=10） |

梯度上限是场地相关参数；对尖锐界面应放宽，否则投影会把界面过度平滑。

---

## 8. RTM 保幅成像

### 8.1 成像条件

对每炮：

1. 正演源波场，保存 P（散度）或 S（旋度）；
2. 接收数据作为伴随源反传；
3. 互相关：

$$
I(x)=\sum_t s(x,t)r(x,t),\qquad
I_{\text{norm}}(x)=\frac{\sum_t s r}{\sum_t s^2+\epsilon}.
$$

默认 `source_normalized`，补偿几何 spreading 和照明不均，不做 AGC，
保留相对振幅。P/S 分离避免波型串扰。

### 8.2 直达波与采集脚印

实际/合成数据中直达波能量比反射波强 1–2 个数量级。代码提供：

* `top_mute`：按炮检距预测直达波到时，平滑切除；
* `depth_taper`：对成像结果在震源附近做深度域切除。

演示中两者都用，避免浅部强假象掩盖真实界面。

### 8.3 倾角与反射点

结构张量：

$$
J=\begin{bmatrix}G*(I_x^2)&G*(I_xI_z)\\G*(I_xI_z)&G*(I_z^2)\end{bmatrix},
$$

主方向 `theta=0.5 atan2(2Jxz, Jxx-Jzz)`，反射面倾角 `dip=theta-pi/2`，
折叠到 `[-pi/2,pi/2]`。相干性

$$
c=\frac{\sqrt{(J_{xx}-J_{zz})^2+4J_{xz}^2}}{J_{xx}+J_{zz}}.
$$

反射点：`|I|` 的局部极大值 + 分位数阈值。

---

## 9. 测井适配

### 9.1 LAS

无第三方依赖的 LAS 2.0/3.0 ASCII 解析，支持：

* `DEPT/DEPTH/MD` 深度；
* `DTC/DT/AC` 纵波时差；
* `DTS/DTSM/DTSF` 横波时差；
* `RHOB/RHOZ/DEN` 密度；
* 空值（默认 -999.25）与单位换算（us/ft、us/m、g/cm³、kg/m³）。

缺失横波用 `vp/1.8`，缺失密度用 Gardner。

### 9.2 初始模型

* 测井段内线性插值；
* 浅于/深于测井段用整体压实趋势线性外推；
* 多井沿 x 线性插值，井外取最近井；
* 横向/纵向高斯平滑；
* 保证 `vs<0.95 vp`、密度在合理范围。

### 9.3 尺度

`design_scale(vmin, vmax, fmax, ppw=8, cfl=0.45)`：

$$
\Delta x=\frac{v_{\min}}{f_{\max}\,N_{ppw}},\qquad
\Delta t=\frac{0.45\Delta x}{v_{\max}\sqrt 2}.
$$

测井 10 kHz、vp=3000 m/s：`dx=0.0375 m`、`dt≈4 μs`；地面地震 30 Hz：
`dx=12.5 m`、`dt≈1.33 ms`。直接混用会频散/不稳定。

### 9.4 各向异性

`WellLog.thomsen` 由 `vs` 与假定的 SH/SV 比（默认 1.15）估计
`gamma=(vsh²-vsv²)/(2vsv²)`，并取 `epsilon=0.9 gamma`、`delta=0.5 gamma`；
也可直接给定。四川深层页岩 SH/SV 差 15% 时 `gamma≈0.16`。

---

## 10. 闭环

`ClosedLoopWorkflow`：

```
for outer in 1..N:
    EFWI(initial/current model)
    RTM(final model, observed data)
    depth mute + dip/reflector extraction
    3D model = dip-guided strike extrapolation
    center slice -> next initial model
```

三维建模对每个 y 按深度平均倾角做横向错动，保持二维剖面的构造趋势；
输出 `.npz` 与可选 `.vti`（PyVista）。中心剖面反馈给下一轮 EFWI，
形成“反演提供模型、成像提供约束”的互锁。

---

## 11. 验证与测试

`python -m unittest discover -s tests -v` 覆盖：

| 测试 | 内容 |
|---|---|
| `test_model` | 刚度、VTI/HTI 旋转、Gardner、平滑 |
| `test_filters` | Wiener 带通形状、零相位、带外衰减 |
| `test_forward` | 各向同性/各向异性/力源稳定性、零残差 |
| `test_adjoint` | 一步正演/伴随的内积恒等式（离散转置） |
| `test_gradient` | 三个参数 × 两个方向 + 随机观测数据的有限差分方向导数、逐炮累积、残差/目标函数、H 非负 |
| `scripts/check_gradient.py` | 演示尺度上的梯度检验并输出 JSON |
| `test_anisotropy` | VTI 的 epsilon/delta 梯度 vs 有限差分 |
| `test_optimize` | L-BFGS 下降方向、抛物线线搜索 |
| `test_constraints` | 梯度投影、Gardner、Dykstra/ADMM |
| `test_welllog` | LAS 解析、延拓、初始模型、尺度 |
| `test_workflow` | 三维建模、中心剖面反馈 |
| `test_pipeline` | EFWI 降低 misfit、RTM/倾角、top mute |

当前全部 37 个测试通过。

---

## 12. 已知限制

1. 二维 P-SV，不含 SH/三维绕射；
2. 海绵吸收层非 CPML；
3. 震源子波固定且已知；
4. 密度参数敏感度低，建议 Gardner 约束或分步反演；
5. 梯度约束的 ADMM 投影是迭代近似，强约束/尖锐界面需增加内层迭代；
6. 三维建模是走向外推演示，不是真正的三维层位插值。
