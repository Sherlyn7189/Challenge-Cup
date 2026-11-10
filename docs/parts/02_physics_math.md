---

# 2. 物理模型与数学模型

本章回答任务书第 2 项，并给出后续离散、成像和 Q 补偿所依赖的全部控制方程。为避免“公式看似齐全、物理却互相矛盾”，本章先声明假设，再按“弹性—黏弹性—一阶系统—边界与源—一致性约束”的顺序展开。所有方程在进入离散之前必须闭合，离散只做近似，不引入新的物理。

## 2.1 建模假设与适用范围

| 编号 | 假设 | 物理含义 | 不满足时的后果 | 首期处理 |
|---|---|---|---|---|
| A1 | 线性小应变 | 应变远小于 1，应力与应变成线性关系 | 强震源附近、裂缝张开/闭合、岩石破碎 | 限制源强，异常体附近单独评估 |
| A2 | 各向同性 | 弹性张量由 K、μ 两个参数描述 | 页岩、裂缝带、层理发育地层 | 首期用等效各向同性 + 低 Q 近似；HTI 列入阶段二 |
| A3 | 连续介质 | 网格尺度远大于孔隙尺度 | 溶洞、裂缝、井孔等强非均质体 | 用等效介质参数；井孔流体不做耦合 |
| A4 | 线性黏弹性 | 衰减用松弛机制描述，响应满足因果性 | 非线性衰减、腾涌、局部流动 | GSLS 拟合目标 Q；超出频带外推需标注 |
| A5 | 二维平面应变（首期） | 介质与波场在 y 方向不变，v_y=0（P-SV） | 三维散射、方位裂缝、点源几何扩散 | 二维基线 + 2.5D/3D 评估；见第 4 章 |
| A6 | 常 Q 近似 | 在目标频带内 Q 近似为常数 | Q 强频变（如部分饱和岩石） | 分段常 Q；频带外不保证 |
| A7 | 局部均匀化 | 每个网格点的参数代表该体积的平均性质 | 小于网格的裂缝/溶洞 | 抗混叠体素化；最小异常体不少于 3—5 个网格 |
| A8 | 初值静止 | 初始位移、速度、应力、记忆变量为零 | 持续振动、残余应力 | 首期零初值；后期可加初始场 |

**关于“二维还是三维”的物理前提**：二维平面应变对应的是沿 y 方向无限延伸的线源，几何扩散为 r 的 -1/2 次方；三维点源为 r 的 -1 次方。两者的振幅不能直接比较，走时在均匀介质中则可比较。因此二维结果用于算法验证和构造归位，不直接用于三维振幅解释。该问题在第 4 章展开。

**关于“井旁远探测”与“勘探尺度”的尺度差异**：策划书技术章节给出的正演示例是 30 Hz、200×200 网格、1 ms 采样，属于勘探尺度；而井旁偶极远探测常用 0.5—5 kHz、厘米—分米级网格。两者相差 2—3 个数量级，不能共用同一套网格参数。本方案把传播核设计为尺度无关（参数由配置给定），但首期只锁定一种尺度做闭环。该选择列为待确认问题 Q1、Q3。

## 2.2 弹性动力学基本方程

### 2.2.1 动量守恒与几何方程

设介质密度为 ρ(x)，质点速度矢量为 v_i，应力张量为 σ_ij，体力为 f_i。忽略体力中的高阶项，动量守恒为

$$
\rho \frac{\partial v_i}{\partial t} = \frac{\partial \sigma_{ij}}{\partial x_j} + f_i
\tag{2.1}
$$

应变率张量与速度的关系为

$$
\dot{\varepsilon}_{ij} = \frac{1}{2}\left(\frac{\partial v_i}{\partial x_j}+\frac{\partial v_j}{\partial x_i}\right)
\tag{2.2}
$$

其中重复指标求和，i,j = 1,2,3 分别对应 x,y,z。本文用 `v_x, v_y, v_z` 表示速度分量，用 `σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz` 表示应力分量。

### 2.2.2 各向同性弹性本构

各向同性线性弹性介质的本构为

$$
\sigma_{ij} = \lambda \varepsilon_{kk}\delta_{ij} + 2\mu \varepsilon_{ij}
\tag{2.3}
$$

其中 λ、μ 为拉梅常数，δ_ij 为 Kronecker 符号，ε_kk = ε_xx+ε_yy+ε_zz 为体积应变。等价地，用体积模量 K 和剪切模量 μ 表示为

$$
\sigma_{ij} = K \varepsilon_{kk}\delta_{ij} + 2\mu\left(\varepsilon_{ij}-\frac{1}{3}\varepsilon_{kk}\delta_{ij}\right)
\tag{2.4}
$$

式 (2.3) 与 (2.4) 的等价条件是 λ = K - 2μ/3。后续黏弹性公式采用 K-μ 分解，因为 Qp 与 Qs 分别对应体积和剪切松弛，物理意义更清晰；但离散时必须保证 λ 与 K、μ 的换算一致，不能在同一处混用两套定义。

对式 (2.3) 求时间导数，并代入 (2.2)，得到一阶速度—应力形式的本构更新：

$$
\frac{\partial \sigma_{ij}}{\partial t} = \lambda \delta_{ij}\frac{\partial v_k}{\partial x_k} + \mu\left(\frac{\partial v_i}{\partial x_j}+\frac{\partial v_j}{\partial x_i}\right) + m_{ij}
\tag{2.5}
$$

其中 m_ij 为震源引起的应力率项（矩张量源）。式 (2.1) 与 (2.5) 构成一阶双曲系统，是交错网格有限差分的标准出发点。

### 2.2.3 二维 P-SV 系统

取 x 为水平方向、z 向下为正，介质与波场在 y 方向不变，且 v_y = 0、σ_xy = σ_yz = 0。非零分量为 v_x、v_z、σ_xx、σ_zz、σ_xz，满足

$$
\rho \frac{\partial v_x}{\partial t} = \frac{\partial \sigma_{xx}}{\partial x}+\frac{\partial \sigma_{xz}}{\partial z}+f_x
\tag{2.6a}
$$

$$
\rho \frac{\partial v_z}{\partial t} = \frac{\partial \sigma_{xz}}{\partial x}+\frac{\partial \sigma_{zz}}{\partial z}+f_z
\tag{2.6b}
$$

$$
\frac{\partial \sigma_{xx}}{\partial t} = (\lambda+2\mu)\frac{\partial v_x}{\partial x}+\lambda\frac{\partial v_z}{\partial z}+m_{xx}
\tag{2.6c}
$$

$$
\frac{\partial \sigma_{zz}}{\partial t} = \lambda\frac{\partial v_x}{\partial x}+(\lambda+2\mu)\frac{\partial v_z}{\partial z}+m_{zz}
\tag{2.6d}
$$

$$
\frac{\partial \sigma_{xz}}{\partial t} = \mu\left(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x}\right)+m_{xz}
\tag{2.6e}
$$

式 (2.6) 描述 P 波与 SV 波的耦合传播。二维情况下，P-SV 与 SH 是**精确解耦**的：只要介质参数和源都不随 y 变化，y 方向运动不会泄漏到 x-z 平面，反之亦然。这是第 3 章“解耦”的第一层含义。

### 2.2.4 二维 SH 系统

SH 波只有 y 方向速度 v_y 和非零剪应力 σ_xy、σ_yz：

$$
\rho \frac{\partial v_y}{\partial t} = \frac{\partial \sigma_{xy}}{\partial x}+\frac{\partial \sigma_{yz}}{\partial z}+f_y
\tag{2.7a}
$$

$$
\frac{\partial \sigma_{xy}}{\partial t} = \mu\frac{\partial v_y}{\partial x}+m_{xy}
\tag{2.7b}
$$

$$
\frac{\partial \sigma_{yz}}{\partial t} = \mu\frac{\partial v_y}{\partial z}+m_{yz}
\tag{2.7c}
$$

SH 系统不含体积模量，因此只受 Qs 控制。这一性质可用于单独验证剪切衰减，而不受 Qp 拟合误差的污染。

### 2.2.5 三维一阶系统

三维情况下有 3 个速度分量和 6 个独立应力分量。以指标形式写为

$$
\rho \frac{\partial v_i}{\partial t} = \frac{\partial \sigma_{ij}}{\partial x_j}+f_i
\tag{2.8a}
$$

$$
\frac{\partial \sigma_{ij}}{\partial t} = \lambda \delta_{ij}\frac{\partial v_k}{\partial x_k}+\mu\left(\frac{\partial v_i}{\partial x_j}+\frac{\partial v_j}{\partial x_i}\right)+m_{ij}
\tag{2.8b}
$$

展开后共 9 个一阶方程。三维不存在二维那样“P-SV 与 SH 精确分开”的全局解耦；P 与 S 在非均匀介质中通过界面耦合，成像时必须做波场分离或使用矢量成像条件。这是三维 ERTM 比二维复杂得多的根本原因之一。

### 2.2.6 波速与能量

弹性介质的 P 波和 S 波相速度为

$$
v_p=\sqrt{\frac{\lambda+2\mu}{\rho}},\qquad v_s=\sqrt{\frac{\mu}{\rho}}
\tag{2.9}
$$

弹性系统的总能量为动能与应变能之和：

$$
E(t)=\frac{1}{2}\int\left(\rho v_i v_i+\sigma_{ij}\varepsilon_{ij}\right)d\Omega
\tag{2.10}
$$

无衰减、无外源时 E(t) 守恒；数值格式的能流平衡是检验格式稳定性和边界条件的重要指标。黏弹性情况下能量单调耗散，耗散率由记忆变量与应变率的乘积给出，可用于验证 Q 模型是否“只衰减、不凭空增能”。

## 2.3 黏弹性本构：广义标准线性体

### 2.3.1 为什么选 GSLS

描述常 Q 的时域模型主要有四类：

| 模型 | 优点 | 缺点 | 本项目取舍 |
|---|---|---|---|
| 广义标准线性体（GSLS） | 常 Q 近似好；时域为整数阶 ODE；可用记忆变量高效更新；因果性自动满足 | 需要多个松弛机制；参数拟合非唯一 | **采用** |
| Cole-Cole / 分数阶 | 宽频带拟合好 | 分数阶导数，时域实现昂贵；边界和伴随复杂 | 不采用 |
| Kjartansson 模型 | 常 Q 解析形式简洁 | 分数阶；难以直接嵌入一阶速度—应力格式 | 不采用 |
| 经验指数增益 | 实现简单 | 非因果、破坏相位；无法与正演自洽 | **禁止单独使用** |

选择 GSLS 的核心理由不是“它最精确”，而是它**能同时进入正演、伴随和成像，且保持因果性与数值可控**。这与本项目“物理一致性优先”的要求一致。

### 2.3.2 单机制标准线性体

单个标准线性体的应力—应变关系为

$$
\sigma+\tau_\sigma\frac{\partial \sigma}{\partial t}
= M_R\left(\varepsilon+\tau_\varepsilon\frac{\partial \varepsilon}{\partial t}\right)
\tag{2.11}
$$

其中 M_R 为松弛模量（低频极限），τ_σ 为应力松弛时间，τ_ε 为应变松弛时间。对式 (2.11) 做 Fourier 变换（约定 e^{-iωt}），得到复模量

$$
M(\omega)=M_R\frac{1-i\omega\tau_\varepsilon}{1-i\omega\tau_\sigma}
\tag{2.12}
$$

为保证衰减为正、高频模量大于低频模量，必须满足

$$
\tau_\varepsilon>\tau_\sigma>0
\tag{2.13}
$$

此时 Im M(ω) < 0，品质因子定义为

$$
Q(\omega)=\frac{\mathrm{Re}\,M(\omega)}{-\mathrm{Im}\,M(\omega)}
\tag{2.14}
$$

式 (2.12) 的实部给出相速度，虚部给出衰减；两者由同一组 τ_σ、τ_ε 决定。这就是 Kramers-Kronig 关系在 GSLS 中的具体体现：**不能独立指定 Q 和速度频散，二者必须由同一复模量导出。**

### 2.3.3 多机制并联与常 Q 拟合

将 L 个标准线性体并联，总复模量为

$$
M(\omega)=M_R\left[1+\sum_{l=1}^{L}
\frac{-i\omega\left(\tau_{\varepsilon l}-\tau_{\sigma l}\right)}
{1-i\omega\tau_{\sigma l}}\right]
\tag{2.15}
$$

对应的松弛函数为

$$
M(t)=M_R\left[1+\sum_{l=1}^{L}
\left(\frac{\tau_{\varepsilon l}}{\tau_{\sigma l}}-1\right)
e^{-t/\tau_{\sigma l}}\right]H(t)
\tag{2.16}
$$

其中 H(t) 为 Heaviside 函数。式 (2.15) 在频带内可以逼近常 Q；机制数 L 越多，逼近越好，但内存和计算量线性增加。工程上 L=3—4 通常足够；L=2 在宽频带内误差偏大。

拟合步骤如下：

1. 给定目标频带 [f_1, f_2]、目标 Q(f)（首期取常数 Q）和机制数 L。
2. 将 L 个应力松弛频率 f_l 在 [f_1, f_2] 内对数等距分布，令 τ_σl = 1/(2π f_l)。
3. 以 τ_εl（或等价的 Q_l 权重）为未知量，最小化频带内 Q 的相对误差与相速度误差的加权和。
4. 检查 τ_εl > τ_σl、Q > 0、相速度单调/平滑；若违反，增加 L 或缩窄频带。
5. 固定参数后，计算参考频率 f_ref 处的复模量，用于把用户输入的 vp、vs 转换为 K_R、μ_R（见 2.3.5）。

拟合目标可写为

$$
J=\sum_{k=1}^{K}\left[
w_Q\left(\frac{Q^{-1}(\omega_k)-Q_{\mathrm{target}}^{-1}}{Q_{\mathrm{target}}^{-1}}\right)^2
+w_v\left(\frac{v(\omega_k)-v_{\mathrm{target}}}{v_{\mathrm{target}}}\right)^2
\right]+\lambda_R J_{\mathrm{reg}}
\tag{2.17}
$$

其中 w_Q、w_v 为权重，λ_R J_reg 为正则项（抑制 τ_ε 的剧烈振荡）。首期建议 w_Q : w_v = 1 : 0.1—0.3，优先保证 Q，其次保证走时。

典型精度量级（实施时以实际拟合结果为准）：

| 机制数 L | 频带内 Q 相对误差 | 相速度误差 | 每单元额外记忆场（二维） | 备注 |
|---|---|---|---|---|
| 2 | 5%—10% | 1%—2% | 8 | 仅适合窄带 |
| 3 | 2%—5% | 0.5%—1% | 12 | **首期推荐** |
| 4 | 1%—2% | 0.3%—0.5% | 16 | 宽频带或强 Q 对比 |
| 6 | <1% | <0.3% | 24 | 研究用，成本高 |

### 2.3.4 Qp、Qs 与 QK、Qμ 的关系

GSLS 分别对 K 和 μ 拟合，得到 QK、Qμ；但用户通常给出 Qp、Qs。弱衰减条件下，二者近似满足

$$
Q_p^{-1}=\left(1-\frac{4}{3}\frac{\beta^2}{\alpha^2}\right)Q_K^{-1}
+\frac{4}{3}\frac{\beta^2}{\alpha^2}Q_\mu^{-1}
\tag{2.18a}
$$

$$
Q_s^{-1}=Q_\mu^{-1}
\tag{2.18b}
$$

其中 α、β 为 P、S 波速度。式 (2.18) 说明：

- Qs 直接等于剪切衰减 Qμ；
- Qp 是 QK 与 Qμ 的加权组合，权重由速度比决定；
- 不能把“Qp”直接当作“QK”使用，否则 P 波衰减会被系统性高估或低估。

首期实现建议：用户输入 Qp、Qs，先由 (2.18b) 得 Qμ = Qs，再反解 QK：

$$
Q_K^{-1}=\frac{Q_p^{-1}-\frac{4}{3}\frac{\beta^2}{\alpha^2}Q_\mu^{-1}}
{1-\frac{4}{3}\frac{\beta^2}{\alpha^2}}
\tag{2.19}
$$

若分母接近零（极端泊松比），改用数值拟合，直接以 Qp、Qs 为目标优化 QK、Qμ，避免解析反演的奇异性。

### 2.3.5 参考频率与松弛模量的确定

用户输入的 vp、vs 通常是某一参考频率 f_ref（例如 20 Hz 或 1 kHz）的相速度，而不是低频极限速度。若直接把 vp 当作低频速度，会在强衰减介质中产生系统性走时偏差。正确做法是：

1. 由 QK、Qμ 拟合得到归一化复模量

$$
m_K(\omega)=\frac{K(\omega)}{K_R},\qquad m_\mu(\omega)=\frac{\mu(\omega)}{\mu_R}
\tag{2.20}
$$

2. 在 f_ref 处要求

$$
K_R\,\mathrm{Re}\,m_K(\omega_{\mathrm{ref}})
+\frac{4}{3}\mu_R\,\mathrm{Re}\,m_\mu(\omega_{\mathrm{ref}})
=\rho v_p^2
\tag{2.21a}
$$

$$
\mu_R\,\mathrm{Re}\,m_\mu(\omega_{\mathrm{ref}})=\rho v_s^2
\tag{2.21b}
$$

3. 解这个 2×2 线性系统得到 K_R、μ_R。由于 m_K、m_μ 的虚部同时决定 Qp、Qs，若拟合的 Qp 与目标不符，需回到 2.3.4 迭代 QK、Qμ，而不是事后修改速度。

这一步是“速度—衰减一致性”的关键。很多实现把 vp、vs 当作弹性速度，再单独加 Q，结果在强衰减时走时和振幅都对不上；本方案从复模量出发，强制二者一致。

### 2.3.6 频带外推的风险

GSLS 只在拟合频带内保证常 Q；频带外 Q 会偏离目标，极端情况下可能出现非物理的负衰减或异常频散。首期约束：

- 正演子波的主频带必须落在拟合频带内；
- 成像的低通/高通滤波不得把能量搬到拟合频带外再解释；
- 若必须宽带计算，增加 L 或采用分段拟合；
- 输出 Q(f) 与相速度曲线，作为 QC 必检项。

## 2.4 一阶黏弹性速度—应力方程

### 2.4.1 二维 P-SV 黏弹性系统

将应力分解为体积部分和偏斜部分，并对 K、μ 分别引入 L 个记忆机制。二维 P-SV 的非零应力为 σ_xx、σ_zz、σ_xz，定义为

$$
\sigma_{xx}=K_R\theta+2\mu_R e_{xx}
+\sum_{l=1}^{L}\left(r^{K}_{l}+2r^{S}_{xx,l}\right)
\tag{2.22a}
$$

$$
\sigma_{zz}=K_R\theta+2\mu_R e_{zz}
+\sum_{l=1}^{L}\left(r^{K}_{l}+2r^{S}_{zz,l}\right)
\tag{2.22b}
$$

$$
\sigma_{xz}=2\mu_R e_{xz}+2\sum_{l=1}^{L}r^{S}_{xz,l}
\tag{2.22c}
$$

其中 θ 为体积应变率积分对应的体积应变，e_ij 为偏斜应变：

$$
\theta=\frac{\partial u_x}{\partial x}+\frac{\partial u_z}{\partial z}
\tag{2.23a}
$$

$$
e_{xx}=\frac{\partial u_x}{\partial x}-\frac{\theta}{3},\qquad
e_{zz}=\frac{\partial u_z}{\partial z}-\frac{\theta}{3}
\tag{2.23b}
$$

$$
e_{xz}=\frac{1}{2}\left(\frac{\partial u_x}{\partial z}+\frac{\partial u_z}{\partial x}\right)
\tag{2.23c}
$$

**关于 θ/3 与 θ/2 的说明**：二维平面应变是三维状态在 ε_yy=0 下的投影，偏斜应变应按三维定义减去 θ/3，而不是二维的 θ/2。若误用 θ/2，则 σ_xx 的等效拉梅常数会变成 K-μ 而不是 λ=K-2μ/3，导致 P 波速度系统性错误。这是二维黏弹性实现中最常见、也最隐蔽的错误之一。实现时建议直接用 λ=K-2μ/3 的等价形式做单元测试，两种写法必须逐点一致。

记忆变量满足

$$
\frac{\partial r^{K}_{l}}{\partial t}+\frac{r^{K}_{l}}{\tau^{K}_{\sigma l}}
=K_R\left(\frac{\tau^{K}_{\varepsilon l}}{\tau^{K}_{\sigma l}}-1\right)
\frac{\partial \theta}{\partial t}
\tag{2.24a}
$$

$$
\frac{\partial r^{S}_{xx,l}}{\partial t}+\frac{r^{S}_{xx,l}}{\tau^{S}_{\sigma l}}
=\mu_R\left(\frac{\tau^{S}_{\varepsilon l}}{\tau^{S}_{\sigma l}}-1\right)
\frac{\partial e_{xx}}{\partial t}
\tag{2.24b}
$$

$$
\frac{\partial r^{S}_{zz,l}}{\partial t}+\frac{r^{S}_{zz,l}}{\tau^{S}_{\sigma l}}
=\mu_R\left(\frac{\tau^{S}_{\varepsilon l}}{\tau^{S}_{\sigma l}}-1\right)
\frac{\partial e_{zz}}{\partial t}
\tag{2.24c}
$$

$$
\frac{\partial r^{S}_{xz,l}}{\partial t}+\frac{r^{S}_{xz,l}}{\tau^{S}_{\sigma l}}
=\mu_R\left(\frac{\tau^{S}_{\varepsilon l}}{\tau^{S}_{\sigma l}}-1\right)
\frac{\partial e_{xz}}{\partial t}
\tag{2.24d}
$$

速度更新仍为式 (2.6a)、(2.6b)，但应力由 (2.22) 给出，不再用弹性本构直接更新。这样，衰减、色散和记忆效应全部进入同一个应力—应变关系，正演与伴随可以共用同一套记忆变量。

二维 P-SV 每网格点需要 5 个主变量（v_x、v_z、σ_xx、σ_zz、σ_xz）和 4L 个记忆变量（r^K、r^S_xx、r^S_zz、r^S_xz 各 L 个）。L=3 时为 17 个场变量。这个数量直接决定内存和带宽，是第 10 章估算的基础。

### 2.4.2 二维 SH 黏弹性系统

SH 系统只有剪切记忆：

$$
\sigma_{xy}=2\mu_R e_{xy}+2\sum_{l=1}^{L}r^{S}_{xy,l}
\tag{2.25a}
$$

$$
\sigma_{yz}=2\mu_R e_{yz}+2\sum_{l=1}^{L}r^{S}_{yz,l}
\tag{2.25b}
$$

$$
e_{xy}=\frac{1}{2}\frac{\partial u_y}{\partial x},\qquad
e_{yz}=\frac{1}{2}\frac{\partial u_y}{\partial z}
\tag{2.25c}
$$

记忆变量方程与 (2.24b)—(2.24d) 同形，只需把 e_xx 换成 e_xy、e_yz。SH 每网格点 3 个主变量（v_y、σ_xy、σ_yz）和 2L 个记忆变量，L=3 时为 9 个场。

### 2.4.3 三维黏弹性系统

三维情况下，应力为 6 个分量，记忆变量为 1 个体积机制加 6 个剪切机制，共 7L 个：

$$
\sigma_{ij}=K_R\theta\delta_{ij}+2\mu_R e_{ij}
+\sum_{l=1}^{L}\left(r^{K}_{l}\delta_{ij}+2r^{S}_{ij,l}\right)
\tag{2.26a}
$$

$$
\theta=\frac{\partial u_k}{\partial x_k},\qquad
e_{ij}=\frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}+\frac{\partial u_j}{\partial x_i}\right)
-\frac{\theta}{3}\delta_{ij}
\tag{2.26b}
$$

记忆变量方程与二维同形，只是指标从 xx、zz、xz 扩展为 xx、yy、zz、xy、xz、yz。三维每网格点 9 个主变量（3 速度 + 6 应力）和 7L 个记忆变量，L=3 时为 30 个场。400³ 网格、单精度时单层场约 7.7 GB，这是三维必须用 GPU 或区域分解的直接原因。

### 2.4.4 记忆变量的精确指数更新

记忆变量方程是一阶线性 ODE。若在一个时间步内把应变率近似为常数，则可以得到无条件稳定的精确更新：

$$
r^{n+1}_{l}=a_{l}r^{n}_{l}+b_{l}\left(\varepsilon^{n+1}-\varepsilon^{n}\right)
\tag{2.27a}
$$

$$
a_{l}=e^{-\Delta t/\tau_{\sigma l}},\qquad
b_{l}=M_R\frac{\tau_{\varepsilon l}-\tau_{\sigma l}}{\Delta t}
\left(1-a_{l}\right)
\tag{2.27b}
$$

其中 ε 代表对应的应变分量（θ、e_xx 等），M_R 代表对应的松弛模量（K_R 或 μ_R）。式 (2.27) 的关键优点是：

- 对任意 Δt/τ_σ 都稳定，不受松弛时间远小于时间步的影响；
- 不需要把 ODE 和波场耦合做显式子步；
- Q=∞ 时 b_l=0、a_l=1，记忆变量保持为零，自动退化为弹性。

若用简单的显式 Euler 更新记忆变量，当 τ_σ 接近或小于 Δt 时会数值爆炸；这是 Q 实现中最常见的稳定性陷阱之一。

**实现顺序建议**：先用当前速度场计算应变率，更新记忆变量到 n+1；再用更新后的记忆变量和应变计算应力到 n+1；最后用新应力更新速度到 n+3/2。具体半步同步在第 5 章给出。无论采用哪种顺序，正演和伴随必须使用**同一套顺序的转置**，否则点积测试不通过。

### 2.4.5 高频极限速度与 CFL

黏弹性介质的瞬时（高频）模量为

$$
K_U=K_R\left[1+\sum_{l=1}^{L}
\left(\frac{\tau^{K}_{\varepsilon l}}{\tau^{K}_{\sigma l}}-1\right)\right]
\tag{2.28a}
$$

$$
\mu_U=\mu_R\left[1+\sum_{l=1}^{L}
\left(\frac{\tau^{S}_{\varepsilon l}}{\tau^{S}_{\sigma l}}-1\right)\right]
\tag{2.28b}
$$

对应的高频 P/S 速度为

$$
v_{p,U}=\sqrt{\frac{K_U+\frac{4}{3}\mu_U}{\rho}},\qquad
v_{s,U}=\sqrt{\frac{\mu_U}{\rho}}
\tag{2.29}
$$

CFL 条件必须用 v_p,U（而不是参考频率的 vp）计算。强衰减介质的 v_p,U 可能比 vp(f_ref) 高 5%—15%，若忽略这一点，时间步会取得过大，长时间计算后出现高频振荡。

## 2.5 初始条件、边界条件与震源模型

### 2.5.1 初始条件

首期采用零初始条件：

$$
v_i(x,0)=0,\quad \sigma_{ij}(x,0)=0,\quad r^{K}_{l}(x,0)=0,\quad r^{S}_{ij,l}(x,0)=0
\tag{2.30}
$$

若后续需要模拟残余应力或持续振动，可加非零初始场，但必须同时给出一致的记忆变量初值，否则会激发非物理瞬态。

### 2.5.2 自由表面

自由表面 z=0 的应力条件为

$$
\sigma_{zz}(x,0,t)=0,\qquad \sigma_{xz}(x,0,t)=0
\tag{2.31}
$$

二维 P-SV 可用应力镜像法：在 z=0 上方设置虚拟网格，令 σ_zz、σ_xz 关于表面对称/反对称，使总应力在表面为零。黏弹性情况下，边界条件作用于**总应力**，而总应力包含记忆变量；因此镜像时必须同时镜像弹性部分和记忆变量，不能只镜像弹性应力。实现细节在第 5 章给出。

自由表面是一个强散射源，会产生面波、多次波和转换波。它既是实际观测的一部分，也是成像假象的来源。首期把自由表面作为可选项：算法验证时用 CPML 全包围，避免面波干扰；与实际观测对比时再打开自由表面。

### 2.5.3 CPML 吸收边界

对一阶速度—应力系统，坐标拉伸 CPML 把法向导数替换为

$$
\frac{\partial}{\partial x}\rightarrow
\frac{1}{\kappa_x}\frac{\partial}{\partial x}+\psi_x
\tag{2.32}
$$

其中 ψ_x 为辅助记忆变量，满足

$$
\frac{\partial \psi_x}{\partial t}
+\left(\frac{d_x}{\kappa_x}+\alpha_x\right)\psi_x
=-\frac{d_x}{\kappa_x}\frac{\partial f}{\partial x}
\tag{2.33}
$$

d_x 为阻尼剖面，κ_x 为坐标拉伸因子，α_x 为频移参数。离散后采用标准 CPML 更新：

$$
\psi^{n+1}_{x}=b_{x}\psi^{n}_{x}+a_{x}\left(\frac{\partial f}{\partial x}\right)^{n+1/2}
\tag{2.34a}
$$

$$
b_{x}=e^{-\left(d_x/\kappa_x+\alpha_x\right)\Delta t},\qquad
a_{x}=\frac{d_x}{\kappa_x\left(d_x+\kappa_x\alpha_x\right)}
\left(b_{x}-1\right)
\tag{2.34b}
$$

CPML 的参数（层数、d_0、κ_max、α）在第 5 章给出建议值。需要注意的是：黏弹性介质的 Q 衰减和 CPML 的阻尼是两种不同的耗散，不能互相替代；CPML 只应吸收向外传播的波，不应在物理区域内部产生额外衰减。

### 2.5.4 震源模型

首期支持以下震源：

1. **爆炸源（explosion）**：各向同性矩张量，m_xx = m_zz = M_0 w(t)，三维时 m_xx=m_yy=m_zz。
2. **单力源（force）**：在 v_x 或 v_z 上直接加 f_i = F_i w(t)。
3. **偶极源（dipole）**：两个等值反向的单力，间距 d，方向由单位矢量给定；用于模拟偶极声源的低频辐射。
4. **矩张量源（moment）**：给定 3×3 对称矩张量 M_ij，按 m_ij = M_ij w(t) 注入。
5. **自定义源**：用户提供时间序列，按采样率插值到 dt。

Ricker 子波定义为

$$
w(t)=\left[1-2\pi^2 f_0^2\left(t-t_0\right)^2\right]
e^{-\pi^2 f_0^2\left(t-t_0\right)^2}
\tag{2.35}
$$

其中 f_0 为主频，t_0 为延迟。Ricker 子波的频谱在 f_0 附近集中，但仍有低频和高频拖尾；若拟合频带较窄，应在源端做带通滤波，避免把频带外能量注入。

**震源注入位置**必须与交错网格变量位置一致：力源注入速度节点，矩张量源注入应力节点；若源坐标不在节点上，用面积加权或双线性插值分配。直接把源加到最近节点会引入与网格相关的相位误差，在高频（井旁）尺度尤其明显。

**井孔流体耦合**首期不做。若用固体中的等效偶极源近似井中偶极，必须意识到：井孔流体会改变源辐射的频谱和指向性，固体中的力偶只是近似。该近似在低频（波长远大于井径）下较好，高频下误差增大。是否需要在首期加入流体—固体耦合，列为待确认问题 Q6。

## 2.6 参数体系与单位

| 参数 | 符号 | 单位 | 典型范围（沉积岩） | 来源/备注 |
|---|---|---|---|---|
| 纵波速度 | vp | m/s | 1800—6500 | 测井、VSP、文献 |
| 横波速度 | vs | m/s | 800—3800 | 测井、经验关系 |
| 密度 | ρ | kg/m³ | 1800—2900 | 测井、Gardner 关系 |
| 体积模量 | K | Pa | 10⁹—5×10¹⁰ | 由 vp、vs、ρ 导出 |
| 剪切模量 | μ | Pa | 10⁹—3×10¹⁰ | μ=ρvs² |
| 纵波品质因子 | Qp | 无量纲 | 20—200 | 谱比法、质心频率偏移 |
| 横波品质因子 | Qs | 无量纲 | 10—100 | 谱比法、上升时间法 |
| 参考频率 | f_ref | Hz | 10—30（勘探）；500—3000（井旁） | 与子波主频一致 |
| 网格间距 | dx, dz | m | 1—10（勘探）；0.05—0.25（井旁） | 由频散和模型尺度决定 |
| 时间步 | dt | s | 0.5—2 ms（勘探）；0.01—0.05 ms（井旁） | 由 CFL 决定 |

黏弹性参数之间的换算关系：

$$
K=\rho\left(v_p^2-\frac{4}{3}v_s^2\right),\qquad
\mu=\rho v_s^2,\qquad
\lambda=\rho\left(v_p^2-2v_s^2\right)
\tag{2.36}
$$

式 (2.36) 适用于弹性或参考频率处的等效参数；在 GSLS 中应使用复模量，不能把参考频率速度直接代入低频模量。

## 2.7 物理一致性约束

“弹性波方程 + 衰减模型 + Q 补偿”必须在以下七个方面保持一致。任何一条不满足，都会导致成像振幅、相位或走时出现系统性错误。

| 编号 | 约束 | 检查方法 | 不满足的典型症状 |
|---|---|---|---|
| C1 | 正演与伴随使用同一套 GSLS 参数和同一离散顺序 | 点积测试；Q=∞ 对照 | 伴随不收敛、图像偏移 |
| C2 | Q 与相速度由同一复模量导出 | 输出 Q(f)、v(f) 曲线；与解析 GSLS 对比 | 走时偏差、频散不自然 |
| C3 | Qp/Qs 由 QK/Qμ 按 (2.18) 导出，不混用 | 均匀介质 P/S 衰减率对比 | P 波衰减被高估/低估 |
| C4 | 衰减为正：Im M<0，τ_ε>τ_σ>0 | 参数扫描；能量单调性 | 数值增长、非物理放大 |
| C5 | Q=∞ 严格退化为弹性 | 逐点对比弹性解 | 残余记忆变量、相位漂移 |
| C6 | 参考频率速度与复模量匹配 | 均匀介质走时对比 | 强衰减时走时整体偏移 |
| C7 | 补偿不引入独立后处理滤波器 | 代码路径审查；增益曲线 | 振幅随偏移距规律被破坏 |

### 2.7.1 一致性验证的最小用例集

1. **均匀全空间、Q=∞**：与弹性解析 Green 函数对比 P/S 走时和振幅。
2. **均匀全空间、常 Q**：与黏弹性解析解或一维参考解对比，检查衰减率和相速度。
3. **两层模型、Q=∞**：与 Zoeppritz 反射系数对比 PP/PS 振幅。
4. **两层模型、层间 Q 对比**：检查透射损失和频率依赖。
5. **Q=∞ 与 Q=1000 对比**：后者应非常接近前者，差异随传播距离增大。
6. **正演—伴随点积测试**：随机源、随机残差，验证转置正确。
7. **能量测试**：弹性无源时能量守恒；黏弹性时能量单调下降。
8. **互易性测试**：交换源和接收点，记录应一致（弹性）；黏弹性下需同时反转时间。

这些用例将在第 9 章的测试接口中固化为自动化测试。

## 2.8 本章结论

1. 采用一阶速度—应力方程 + GSLS 记忆变量作为唯一传播模型；二维 P-SV 与 SH 分别实现，三维按指标形式扩展。
2. K 与 μ 分别拟合 QK、Qμ，再由 (2.18) 得到 Qp、Qs；用户输入的 vp、vs 是参考频率相速度，通过 (2.21) 转换为松弛模量。
3. 记忆变量用精确指数更新，保证强衰减、大时间步下的稳定性；Q=∞ 自动退化为弹性。
4. 自由表面、CPML、震源注入都必须与黏弹性变量一致；边界条件作用于总应力，不能只作用于弹性部分。
5. 物理一致性通过七条约束和八组最小用例验证；其中 C1（正演/伴随同源）和 C7（禁止后处理 Q 滤波）是硬约束，不满足则方案不成立。
