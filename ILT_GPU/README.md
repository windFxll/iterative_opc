# ILT_GPU

基于 PyTorch 的可微像素型逆向光刻（ILT）原型。光学核心与上级目录的
`iterative_opc/optics.py` 保持相同约定：亮场、暗图形 mask，Abbe 光源点求和，
归一化空间像，以及可选的 isolated 零填充边界。

## 目录结构

```text
ILT_GPU/
├─ ilt_gpu/
│  ├─ config.py             参数 dataclass
│  ├─ optics/               光源采样、光瞳、可微 Abbe 前向
│  ├─ resist/               软/硬光刻胶模型
│  ├─ layouts/              可复用参数化版图
│  ├─ optimization/         mask 参数化、损失、ILT 主循环
│  └─ utils/                评估指标、输出和绘图
├─ scripts/run_demo.py       GTX 1080 demo
├─ tests/smoke_test.py       GPU 前向/梯度自检
└─ requirements.txt
```

## 运行

使用已有的 PyTorch CUDA 环境：

```powershell
cd ILT_GPU
python -m pip install -r requirements.txt
python tests\smoke_test.py --device cuda:0
python tests\forward_consistency.py --device cuda:0
python scripts\run_demo.py --device cuda:0
```

如果目标 GPU 不是第 0 张卡，请相应修改 `--device`。

更复杂的多拐弯/密集细线版图：

```powershell
python scripts\preview_layout.py --layout complex
python scripts\run_demo.py --device cuda:0 --layout complex --iterations 1000
```

ICCAD-2013 原生 1024×1024、1 nm/px 配置（GTX 1080）：

```powershell
python scripts\run_demo.py --device cuda:0 --profile iccad-native --target benchmarks\ICCAD2013\M1_test3.glp --iterations 1000
```

`M1_test3.glp` 来自公开的
[OpenILT ICCAD-2013 benchmark](https://github.com/OpenOPC/OpenILT/tree/main/benchmark/ICCAD2013)。
可用 `scripts/prepare_glp_benchmark.py` 将 GLP 转成 NPY/PNG；这些生成文件默认不纳入 Git。

demo 最多运行 1000 步，在 beta 硬化结束后同时监控 loss、IoU 和 hard-mask
变化率并可提前收敛。demo profile 用 36 个光源点优化，再用 144 个光源点独立验证；
原生 ICCAD profile 使用 72 点验证。最终输出在 `output/`。

## 设计要点

- mask 由可训练 logits 经高斯制造滤波、设计窗边界约束和 sigmoid 生成。
- 优化期间用可微 soft resist，验证用硬阈值 resist。
- 损失包含像素/边界加权误差、TV、二值化和面积正则。
- 光源点分批传播，控制 8 GB 显存上的峰值占用。
- 保存 hard-mask 验证 IoU 最佳的状态，而不是盲目使用末轮。

## GTX 1080 demo 实测

- 1000 步上限，第 716 步收敛早停（保存第 375 步最佳 hard mask）
- 优化时间：21.25 s
- 144 光源点独立验证：0.15 s
- 峰值显存：651 MB
- hard-resist IoU：0.83043 → 0.97837
- 轮廓平均距离：8.35 nm → 0.96 nm
- 与 NumPy 参考前向的 RMS 差：1.1e-7

