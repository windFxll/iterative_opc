# 可微 ILT 的梯度下降：公式与计算流程

> 对应代码：`pr-1-ilt-gpu` 分支的 `ILT_GPU/` 子项目。
> 本文把 **代码实际实现** 与 **理论上更严格的做法** 分开写，便于核对哪些地方是近似、哪些地方需要修正。
>
> **公式渲染**：本文使用 `$...$`（行内）和 `$$...$$`（独立公式）；GitHub、VS Code Markdown Preview、Typora 等均可直接渲染。

---

## 0. 符号约定

| 符号 | 含义 |
|---|---|
| $m_{ij}\in[0,1]$ | 软掩模，1 = 不透明（暗图形），0 = 透光 |
| $t_{ij}\in\{0,1\}$ | 目标版图，1 = 要印刷的暗图形 |
| $z_{ij}$ | 掩模 logits，**优化变量** |
| $I_{ij}$ | 归一化空间像 |
| $p_{ij}$ | 软光刻胶印刷图案（1 = 显影后保留的图形） |
| $w_s$ | 光源点权重，$\sum_s w_s=1$ |
| $P_s(f)$ | 光源点 $s$ 对应的光瞳 |
| $f_c=NA/\lambda$ | 光瞳截止频率 |
| $\beta$ | 掩模 sigmoid 硬度（退火） |
| $\beta_r$ | 光刻胶 sigmoid 硬度（退火） |
| $I_{th}$ | 光刻胶阈值 |

代码约定：**亮场、暗图形**。透射场 $T=1-m$；背景归一化强度为 1，图形内部趋近 0；$I<I_{th}$ 处形成图形。

---

## 1. 问题定义

ILT 是一个像素级逆问题：

$$
\min_{z\in\mathbb{R}^{N\times N}}\; L\big(A(m(z)),\,t\big)
$$

其中 $A$ 是 Hopkins 部分相干成像模型（代码里用 Abbe 点源求和实现）+ 光刻胶模型，$L$ 是标量损失。

关键点：

- 优化变量是 $N^2$ 个 **logits**，每个像素一个；
- 前向 $A$ 是确定性的物理仿真，不含神经网络；
- 反向传播一次即可得到全部 $N^2$ 个偏导，而不是逐像素做有限差分。

---

## 2. 掩模参数化

代码：`ILT_GPU/ilt_gpu/optimization/parameterization.py` 的 `FilteredMask`。

$$
\begin{aligned}
\tilde z &= G_\sigma * z,\\
m &= W\odot \sigma(\beta\tilde z).
\end{aligned}
$$

其中：

- $G_\sigma$：高斯卷积（制造低通 / 最小特征尺寸约束），demo 中 `mask_filter_sigma_px = 1.25`；
- $\sigma(\cdot)$：sigmoid；
- $W$：design window，内部为 1、最外 `border_margin_px` 像素为 0；
- $\beta$：从 `mask_beta_start=1.8` 退火到 `mask_beta_end=10.0`，在前 `hardening_fraction=0.30` 的步数内线性增加，之后固定。

初始化：

$$
z_{ij}^{(0)}=
\begin{cases}
+1.5, & t_{ij}=1\\
-1.5, & t_{ij}=0
\end{cases}
$$

注意：是 $m=W\odot\sigma(\beta\tilde z)$，不是 $\sigma(\beta W\tilde z)$。因此 design window 的边界像素 $m=0$，且对 $z$ 的梯度也为 0，优化器无法在那些位置放置掩模。

---

## 3. 光学前向：Hopkins 模型（Abbe 点源求和实现）

代码：`ILT_GPU/ilt_gpu/optics/abbe.py` 和 `pupil.py`。

物理模型是 Hopkins 部分相干成像；代码用 Abbe 点源求和来数值实现它（两者在数学上等价，见 §10.11）。

### 3.1 透射场与频谱

$$
T = 1-m
$$

$$
\hat T = \mathcal{F}\{\operatorname{ifftshift}(T)\}
$$

`ifftshift` 把图像中心（原点）移到数组 `[0,0]`，与 `fftfreq` 的自然序网格一致。

### 3.2 光源采样

代码：`ILT_GPU/ilt_gpu/optics/source.py`。当前只实现环形照明。采样公式为

$$
\begin{aligned}
\sigma_r &= \sigma_{\text{in}}+\left(r+\frac12\right)\Delta\sigma,
& \Delta\sigma &= \frac{\sigma_{\text{out}}-\sigma_{\text{in}}}{N_{\text{rings}}},\\[2pt]
\theta_j &= \frac{2\pi j}{N_\theta},\\[2pt]
(\sigma_{sx},\sigma_{sy}) &= \sigma_r(\cos\theta_j,\sin\theta_j).
\end{aligned}
$$

其中 $r=0,\dots,N_{\text{rings}}-1$，$j=0,\dots,N_\theta-1$。面积权重 $\propto \sigma_r$，最后归一化 $\sum_s w_s=1$。

### 3.3 光瞳

对光源点 $s$，物理频移 $f_s=f_c(\sigma_{sx},\sigma_{sy})$，归一化光瞳坐标

$$
\nu_s(f)=\frac{|f+f_s|}{f_c}
$$

$$
P_s(f)=
\begin{cases}
e^{j\phi_s(f)}, & \nu_s(f)\le 1\\
0, & \nu_s(f)>1
\end{cases}
$$

离焦相位（**代码实际写法**）：

$$
\phi_s(f)=\frac{2\pi}{\lambda}\Delta z
\left(
\sqrt{\max\!\left(1-(NA\,\nu_s(f))^2,\,0\right)}-1
\right)
$$

**这里有一个需要确认的近似**：该式隐含折射率 $n=1$。浸没式光刻 $NA=1.35$ 时，
$NA\,\nu>1$ 的区域会被 `clamp` 成 0，外层光瞳的离焦相位变成常数，物理上不严格。
若显式考虑浸没介质 $n\approx1.44$，更合理的形式是

$$
\phi_s(f)=\frac{2\pi}{\lambda}\Delta z
\left(
\sqrt{n^2-(NA\,\nu_s(f))^2}-n
\right)
$$

（或等价的 reduced form；相位参考项可整体平移，不影响 $|E|^2$）。
本次 demo 的 `defocus_nm=0`，所以这个问题没有暴露。

### 3.4 像场与空间像

$$
\begin{aligned}
E_s(x) &= \mathcal{F}^{-1}\{\hat T(f)P_s(f)\}(x),\\[2pt]
I(x) &= \frac{1}{I_{\text{clear}}}\sum_s w_s\,|E_s(x)|^2.
\end{aligned}
$$

其中

$$
I_{\text{clear}}=\sum_s w_s\,|P_s(0)|^2
$$

当前环形光源 $\sigma_{\text{out}}=0.9<1$，所有源点都在光瞳内，因此 $I_{\text{clear}}=1$。

代码中 $|E_s|^2$ 用 `field.real**2 + field.imag**2` 计算；`ifft2` 自带 $1/N^2$ 归一化；`fftshift` 把原点移回图像中心。

### 3.5 边界条件

- **`isolated`**：把 $T$ 零填充到 $2N\times2N$（填充值 1.0，表示视场外无图形），算完裁中心 $N\times N$。等价于线性卷积，避免相邻周期串扰。`fft_size=2N`。
- **`periodic`**：不填充，直接 FFT。循环卷积 = 周期边界；视场就是一个周期。`fft_size=N`，更快、更省显存。

### 3.6 可选的胶内扩散

$$
I_b = G_{\sigma_r} * I
$$

- `periodic`：circular padding；
- `isolated`：replicate padding。

demo 的 `resist_blur_sigma_nm=0`，未启用。

---

## 4. 光刻胶模型

代码：`ILT_GPU/ilt_gpu/resist/models.py`。

### 4.1 优化时：软阈值（可微）

$$
p=\sigma\!\big(\beta_r(I_{th}-I)\big)
$$

- $I<I_{th}$ 时 $p\to1$（形成暗图形）；
- $\beta_r$ 从 `resist_beta_start=8.0` 退火到 `resist_beta_end=40.0`。

### 4.2 验证/输出时：硬阈值（不可微）

$$
p_{\text{hard}}=\mathbb{1}[I<I_{th}]
$$

硬阈值只用于前向验证、IoU、最终输出，不参与反向传播。

---

## 5. 损失函数

代码：`ILT_GPU/ilt_gpu/optimization/losses.py` 的 `ilt_loss`。

$$
\begin{aligned}
L ={}&
\underbrace{\frac{1}{N^2}\sum_{ij} w_{ij}(p_{ij}-t_{ij})^2}_{\text{image}} \\[2pt]
&+ \lambda_{TV}\underbrace{\frac{1}{N^2}\sum_{ij}|\nabla m_{ij}|}_{\text{TV}}
 + \lambda_b\underbrace{\frac{1}{N^2}\sum_{ij}m_{ij}(1-m_{ij})}_{\text{binary}}
 + \lambda_a\underbrace{\left(\bar m-\bar t\right)^2}_{\text{area}}.
\end{aligned}
$$

其中 $\bar m=\frac{1}{N^2}\sum m_{ij}$，$\bar t=\frac{1}{N^2}\sum t_{ij}$。

边界权重：

$$
w_{ij}=1+\lambda_e\cdot \text{boundary}_{ij}
$$

`boundary` 由目标 $t$ 的膨胀减腐蚀得到（`boundary_radius_px=2`）。默认参数：

$$
\lambda_e=3.0,\quad
\lambda_{TV}=5\times10^{-4},\quad
\lambda_b=10^{-3},\quad
\lambda_a=0.08
$$

各项作用：

- **image**：主数据项，边界加权让 EPE 附近的误差更重要；
- **TV**：抑制掩模高频噪声，近似周长/制造约束；
- **binary**：推动 $m$ 接近 0/1；
- **area**：防止整体掩模面积漂移。

---

## 6. 反向传播：从标量损失到每个像素

### 6.1 链式结构

$$
\begin{aligned}
\frac{\partial L}{\partial z}
&= \frac{\partial L}{\partial m}
\odot \frac{\partial m}{\partial \tilde z}
\odot \frac{\partial \tilde z}{\partial z},\\[2pt]
\frac{\partial m}{\partial \tilde z}
&= W\odot \beta\,\sigma(\beta\tilde z)\big(1-\sigma(\beta\tilde z)\big),\\[2pt]
\frac{\partial \tilde z}{\partial z} &= G_\sigma^{T}.
\end{aligned}
$$

高斯核对称，因此 $G_\sigma^{T}=G_\sigma$；严格写是卷积的伴随（反向卷积）。

### 6.2 通过光刻胶

$$
\begin{aligned}
\frac{\partial L}{\partial I}
&= \frac{\partial L}{\partial p}\odot \frac{\partial p}{\partial I},\\[2pt]
\frac{\partial p}{\partial I}
&= -\beta_r\,p\,(1-p).
\end{aligned}
$$

（因为 $p=\sigma(\beta_r(I_{th}-I))$。）

### 6.3 通过 Abbe 前向（伴随）

令 $g_I=\partial L/\partial I$。对每个源点：

$$
\begin{aligned}
I &= \sum_s w_s |E_s|^2,\\[2pt]
\bar E_s &= 2\,w_s\,g_I\odot E_s,\\[2pt]
\bar{\hat T}
&= \sum_s \operatorname{conj}(P_s)\odot \mathcal{F}\{\bar E_s\},\\[2pt]
\bar T &= \mathcal{F}^{-1}\{\bar{\hat T}\},\\[2pt]
\bar m &= -\bar T.
\end{aligned}
$$

这里 $\bar E_s$ 表示对 $E_s$ 的共轭梯度/伴随。具体共轭位置和 FFT 归一化由 PyTorch 的 Wirtinger 约定处理；手写 NumPy 版本时需要按所选内积约定重新推导。

（因为 $T=1-m$。）

这四步对应：`|E|²` → `ifft2` 的伴随 → 乘光瞳的伴随（乘共轭）→ `fft2` 的伴随 → `T=1-m` 的导数。

### 6.4 为什么一次反向传播能算出全部像素的梯度

反向模式自动微分（reverse-mode AD）计算的是 vector-Jacobian product：

$$
\bar z=\left(\frac{\partial L}{\partial z}\right)^{T}
$$

它从标量 $L$ 出发，沿计算图反向走一次，就得到所有 $N^2$ 个偏导。成本大约相当于几次前向，**与变量个数基本无关**。

如果改用有限差分，每个像素都要扰动一次、跑一次前向，代价是 $N^2$ 次前向——对 256×256 就是 65536 次，完全不可行。

---

## 7. 优化循环

代码：`ILT_GPU/ilt_gpu/optimization/optimizer.py` 的 `PixelILT.optimize`。

1. 初始化 logits：目标内 $+1.5$，目标外 $-1.5$。
2. 优化器：Adam，`learning_rate=0.08`；
   scheduler：CosineAnnealingLR，`T_max=iterations`，`eta_min=0.002`。
3. 每一步：
   - 计算当前 $\beta,\beta_r$（前 30% 步数退火，之后固定）；
   - 前向：$z\to m\to I\to p$；
   - 计算损失 $L$；
   - `loss.backward()`；
   - 梯度裁剪（max norm 5.0）；
   - `optimizer.step()`，`scheduler.step()`。
4. 每 `log_every=5` 步验证：
   - 硬掩模 $m_{\text{hard}}=\mathbb{1}[m\ge0.5]$；
   - 用硬掩模重新跑前向 + 硬光刻胶；
   - 计算 hard-mask IoU；
   - 保存 IoU 最好的硬掩模（`best_mask`）。
5. 早停条件（仅当已过 `max(early_stop_min_step, hardening_steps)` 且窗口足够）：
   - loss 相对变化 < `5e-4`；
   - IoU 波动 < `2e-4`；
   - mask change 最大值 < `2e-4`；
   - 连续 `patience=3` 次满足则停止。
6. 输出 `best_mask`，并用更密的光源采样（demo：36 点优化 → 144 点验证）重新计算空间像和印刷轮廓。

---

## 8. 计算流程伪代码

```python
z = init_logits(target)                    # +1.5 / -1.5
for step in range(iterations):
    beta, beta_r = beta_schedule(step)

    # ---- forward ----
    m = sigmoid(beta * gaussian_filter(z)) * design_window
    I = abbe_forward(m, pupils, weights, boundary_mode)
    p = sigmoid(beta_r * (threshold - I))
    L = image_loss(p, target, boundary_weights) \
        + tv_weight * tv(m) \
        + binary_weight * binary(m) \
        + area_weight * area(m)

    # ---- backward ----
    L.backward()
    clip_grad_norm_(z, 5.0)
    adam.step()
    scheduler.step()

    # ---- hard-mask validation ----
    if step % 5 == 0:
        m_hard = (m >= 0.5).float()
        I_hard = abbe_forward(m_hard, pupils, weights, boundary_mode)
        p_hard = (I_hard < threshold)
        iou = binary_iou(p_hard, target)
        keep_best(m_hard, iou)
```

---

## 9. 代码映射

| 公式 / 步骤 | 代码位置 |
|---|---|
| 掩模参数化 | `ILT_GPU/ilt_gpu/optimization/parameterization.py: FilteredMask` |
| 光源采样 | `ILT_GPU/ilt_gpu/optics/source.py` |
| 光瞳构建 | `ILT_GPU/ilt_gpu/optics/pupil.py` |
| Abbe 前向 | `ILT_GPU/ilt_gpu/optics/abbe.py: AbbeImaging` |
| 软/硬光刻胶 | `ILT_GPU/ilt_gpu/resist/models.py` |
| 损失 | `ILT_GPU/ilt_gpu/optimization/losses.py` |
| 优化循环 | `ILT_GPU/ilt_gpu/optimization/optimizer.py: PixelILT` |
| 运行配置 | `ILT_GPU/ilt_gpu/config.py`, `profiles.py` |
| demo 入口 | `ILT_GPU/scripts/run_demo.py` |
| 前向一致性测试 | `ILT_GPU/tests/forward_consistency.py` |

---

## 10. 已知近似 / 需要修正或确认的点

下面按“对当前 demo 的影响”从大到小排列。

### 10.1 离焦相位的浸没折射率（物理）

当前代码：

```python
torch.sqrt(torch.clamp(1.0 - (NA * nu).square(), min=0.0)) - 1.0
```

在 $NA=1.35$、$\nu>0.74$ 时根号内为负，被 clamp 到 0，外层光瞳相位变成常数。若使用浸没介质 $n=1.44$，应改为：

$$
\sqrt{n^2-(NA\,\nu)^2}-n
$$

或至少显式引入折射率参数。当前 demo `defocus_nm=0`，所以不影响结果；但做离焦实验前必须确认。

### 10.2 标量偏振（物理）

整个模型是标量 Abbe，没有高 NA 矢量投影、TE/TM、z 分量。需要偏振效应时，要升级为矢量 Abbe（每个衍射级做偏振投影，像强为 $|E_x|^2+|E_y|^2+|E_z|^2$）。

### 10.3 周期边界下的掩模滤波（实现）

`FilteredMask` 中：

```python
filtered = gaussian_blur2d(self.logits, self.filter_sigma_px)
```

`gaussian_blur2d` 默认 `circular=False`（replicate padding）。在 `periodic` 模式下，严格来说制造滤波也应该是 circular，否则接缝处不周期。当前 demo `border_margin_px=10`、滤波半径 4px，边界 40nm 被 design window 置零，所以影响被掩盖；一旦 `border_margin_px=0`，这个问题就会显现。

建议：把 `boundary_mode == "periodic"` 传给 `FilteredMask`，让 `gaussian_blur2d(..., circular=True)`。

### 10.4 design window 与周期目标（实现/建模）

`border_margin_px > 0` 会把最外 `border_margin_px × pixel_size_nm` 的掩模强制置零（demo：10×4nm=40nm；ICCAD：40×1nm=40nm）。

- 对 `isolated` 边界和边界本来为空的版图，这是合理的 guard band；
- 对填满视场的周期目标，这会在周期接缝处切出 40nm 空白，导致接缝处印不出来。

因此真正的 periodic + 无缝目标应设：

```python
boundary_mode = "periodic"
border_margin_px = 0
```

并配合 10.3 的 circular mask filter。

### 10.5 软模型 / 硬模型的差距（优化）

优化用 soft mask + soft resist；最终输出 hard mask + hard resist。代码用 hard-mask IoU 选择 best，但优化轨迹本身是在软模型上走的。可以增加：

- 最后若干步用 straight-through estimator（前向硬阈值、反向用 sigmoid 梯度）；
- 或加一个 hard-mask-aware 的 finetune 阶段。

### 10.6 优化光源数与验证光源数不一致（数值）

demo：36 个光源点优化，144 个点验证。若 36 点采样有明显误差，优化器可能追逐采样伪影，导致验证指标与优化 loss 不完全一致。

可选方案：

- source-count continuation：先用少点优化，再逐步增加；
- 或直接用验证点数优化（更慢）；
- 或在 loss 中加入多点采样的平均。

### 10.7 TV / binary / area 只是制造约束的代理（建模）

这些正则项不是 DRC/MRC：没有最小线宽/间距的硬约束，也没有掩模多边形化。若关心可制造性，需要额外加：

- 最小线宽/间距检查；
- 曲率/周长约束；
- 多边形提取和掩模规则检查。

### 10.8 目标必须与边界条件匹配（建模）

`periodic` 模式要求目标本身是一个周期单元；否则视场外重复的副本会参与中心单元的成像，而目标只描述中心这一份，指标会下降。这不是算法错误，而是模型/目标不匹配。

### 10.9 `clear_intensity` 与源点裁剪（数值）

当前 annular $\sigma_{\text{out}}=0.9$，所有源点在光瞳内，`clear_intensity=1`。若以后支持 $\sigma>1$ 或自定义光源，需要重新检查：

- 源点是否应裁到单位圆盘；
- `clear_intensity` 是否仍为 1；
- 归一化是否与 `main` 分支一致。

### 10.10 复数梯度的共轭约定（实现）

本文 §6.3 的伴随公式是示意。PyTorch 对复数张量使用 Wirtinger 微积分，`z.grad` 的具体共轭位置由框架处理。若要手写 NumPy 版本，需要按所选内积约定重新推导：

- `ifft2` 的伴随是 `fft2` 还是带共轭/归一化的 `fft2`；
- 乘光瞳的伴随是乘 `conj(P)`；
- $|E|^2$ 的导数位置。

建议用 `torch.autograd.gradcheck` 或小规模有限差分做数值验证。

### 10.11 Hopkins TCC 与 Abbe 点源求和（概念澄清）

Hopkins 方程把像强写成掩模频谱的二次型：

$$
I(x)
= \iint TCC(f',f'')\,M(f')\,M^*(f'')\,
e^{j2\pi(f'-f'')x}\,df'\,df''
$$

$$
TCC(f',f'')
= \int S(f)\,P(f+f')\,P^*(f+f'')\,df
$$

Abbe 方法则对每个光源点先算相干像，再按强度求和：

$$
E_s(x)=\mathcal{F}^{-1}\{M(f)P(f+f_s)\}(x),
\qquad
I(x)=\sum_s w_s\,|E_s(x)|^2
$$

把 $S(f)=\sum_s w_s\delta(f-f_s)$ 代入 Hopkins 的 TCC，就得到

$$
TCC(f',f'')
= \sum_s w_s\,P(f'+f_s)\,P^*(f''+f_s)
$$

代回 Hopkins 方程并整理，正好等于 Abbe 的 $\sum_s w_s|E_s|^2$。所以：

- **物理模型是 Hopkins 部分相干成像**；
- **数值实现是 Abbe 点源求和**（等价于对光源做离散化，不显式构造 4D TCC）；
- ILT_GPU 选择 Abbe 是因为它显存友好、容易 GPU 批处理；代价是每个光源点都要做一次 iFFT；
- 如果显式构造 TCC，光源离散后 TCC 是若干“移位光瞳外积”的和，存储/计算量更大，通常只在特定小规模场景使用。

因此，说“光学模型用的是 Abbe”指的是**求解方式**，不是说物理模型不是 Hopkins。

---

## 11. 一页公式速查

```text
m = W ⊙ sigmoid(β · G_σ * z)
T = 1 - m
M = FFT(ifftshift(T))
E_s = fftshift(IFFT(M ⊙ P_s))
I = (1 / I_clear) Σ_s w_s |E_s|²
p = sigmoid(β_r (I_th - I))
L = mean(w ⊙ (p - t)²) + λ_TV·TV(m) + λ_b·mean(m(1-m)) + λ_a·(mean(m)-mean(t))²
z ← Adam(z, ∂L/∂z)
```

其中 $P_s$ 由光源点 $s$ 和光瞳函数决定，$\partial L/\partial z$ 由反向模式自动微分一次算出。
