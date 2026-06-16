# DD2424 小组 148 — CIFAR-10 统一训练基础设施

在 CIFAR-10 上研究现代 ConvNet 的 **精度–效率权衡** 与 **缩放规律**。  
本仓库提供三人共用的训练底座：相同的数据划分、超参数、FLOPs 统计方式。各自只需在 `models/` 中实现自己的网络并注册即可。

---

## 环境要求

已在 conda 环境 `dd2424` 下测试：

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| GPU | CUDA（如 RTX 5070） |

### 安装依赖

```bash
conda activate dd2424
cd ~/Desktop/final_prj
pip install -r requirements.txt
```

---

## 项目结构

```
final_prj/
├── configs/default.yaml      # 统一超参（seed、学习率、增强等）
├── data/
│   ├── splits.py             # 固定 train/val 划分逻辑
│   ├── cifar10/              # CIFAR-10 数据（首次训练自动下载，不提交 git）
│   └── artifacts/
│       └── cifar10_val_indices.pt   # 持久化的验证集索引（三人必须一致）
├── train/
│   ├── pipeline.py           # DataLoader、数据增强、AdamW、Cosine 调度
│   └── trainer.py            # 训练 / 验证循环
├── utils/
│   └── metrics.py            # 统一 FLOPs 统计（thop）
├── models/
│   ├── vgg_baseline.py       # VGG 风格基线
│   ├── depthwise.py          # 深度可分离卷积模板
│   ├── convnext_style.py     # ConvNeXt 风格模板
│   └── registry.py           # 模型注册表
├── scripts/
│   ├── train.py              # 训练入口
│   └── profile_model.py        # 仅统计参数量 / FLOPs
├── checkpoints/              # 最佳模型权重（按 val acc 保存）
└── logs/results.csv          # 实验结果汇总
```

---

## 数据集说明

- **数据集**：CIFAR-10（5 万张训练图 + 1 万张测试图，10 类，32×32）
- **存放路径**：`data/cifar10/`（由 `torchvision` 首次运行时自动下载）
- **划分方式**：
  - 从 5 万张训练图中按 `seed=42`、`val_ratio=0.1` 划出 **45000 训练 / 5000 验证**
  - 索引保存在 `data/artifacts/cifar10_val_indices.pt`，生成一次后全员复用
  - **10000 张官方测试集** 仅用于最终汇报，调参和选模型请只看 **验证集**

队友克隆仓库后：

1. 若已有数据：直接复制整个 `data/cifar10/` 和 `data/artifacts/cifar10_val_indices.pt`
2. 若没有：运行一次训练脚本会自动下载数据并生成索引文件

---

## 快速开始

### 1. 查看模型算力（无需完整训练）

```bash
python scripts/profile_model.py --model vgg_baseline
python scripts/profile_model.py --model depthwise_small
python scripts/profile_model.py --model convnext_small
```

### 2. 冒烟测试（1 个 epoch，确认环境正常）

```bash
python scripts/train.py --model vgg_baseline --epochs 1
```

### 3. 完整训练（默认 200 epoch）

```bash
python scripts/train.py --model vgg_baseline
python scripts/train.py --model depthwise_small
python scripts/train.py --model convnext_small
```

### 4. 命令行覆盖部分超参

```bash
python scripts/train.py --model vgg_baseline --epochs 100 --batch-size 64 --lr 0.0005 --seed 42
```

### 5. 仅评估已有权重

```bash
python scripts/train.py --model vgg_baseline --eval-only --checkpoint checkpoints/vgg_baseline_seed42_ep200.pt
```

---

## 内置模型

| 注册名 | 源文件 | 说明 |
|--------|--------|------|
| `vgg_baseline` | `models/vgg_baseline.py` | 标准卷积 + BN 基线 |
| `depthwise_small` | `models/depthwise.py` | 深度可分离卷积，可调 `width`、`kernel_size` |
| `convnext_small` | `models/convnext_style.py` | 大核 DW + LayerNorm + 倒瓶颈，可调 `width`、`kernel_size`、`depths` |

做 Pareto / 缩放实验时，请通过 `width`、`depths`、`kernel_size` 等把 **FLOPs 对齐到相近预算**，再比较精度。

---

## 训练输出

每次训练结束后会自动：

1. 在终端打印每个 epoch 的 `train_loss / train_acc / val_loss / val_acc / lr`
2. 将 **验证集准确率最高** 的权重保存到：
   `checkpoints/{模型名}_seed{seed}_ep{epochs}.pt`
3. 向 `logs/results.csv` 追加一行（时间戳、模型名、参数量、MACs、FLOPs、best val acc、test acc 等）

---

## 组内约定（请勿擅自修改）

1. **数据划分**  
   使用同一份 `cifar10_val_indices.pt`，`configs/default.yaml` 中的 `seed` 和 `val_ratio` 需组内一致后再改。

2. **训练超参**  
   统一使用 `AdamW` + `Cosine Annealing`（含 5 epoch warmup）+ 基础增强（`RandomCrop(padding=4)` + `RandomHorizontalFlip`）。  
   永久修改请编辑 `configs/default.yaml` 并告知组员。

3. **FLOPs 统计**  
   必须使用 `utils/metrics.count_flops()`（基于 `thop`）。  
   汇报约定：**FLOPs = 2 × MACs**。禁止各自手写公式。

4. **测试集**  
   测试集只用于最终一次评估；模型选择以 **验证集准确率** 为准。

5. **可选技巧**  
   `use_mixup`、`use_amp` 在配置中默认关闭；若做消融需全组统一后再开启。

---

## 如何添加新模型

1. 在 `models/` 下新建文件，实现 `__init__(self, num_classes=10, ...)` 和 `forward`。
2. 在 `models/registry.py` 中注册：

```python
from models.your_model import YourModel

MODEL_REGISTRY["your_model"] = YourModel
```

3. 训练：

```bash
python scripts/profile_model.py --model your_model
python scripts/train.py --model your_model
```

---

## 配置文件说明（`configs/default.yaml`）

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `seed` | 42 | 随机种子（影响划分与训练可复现性） |
| `data_root` | `./data/cifar10` | 数据目录 |
| `val_ratio` | 0.1 | 从训练集划出验证集比例 |
| `batch_size` | 128 | 批大小 |
| `epochs` | 200 | 训练轮数 |
| `lr` | 0.001 | AdamW 初始学习率 |
| `weight_decay` | 0.05 | 权重衰减 |
| `warmup_epochs` | 5 | 线性 warmup 轮数 |
| `min_lr` | 1e-6 | Cosine 最低学习率 |

---

## 常见问题

**Q：队友机器上没有数据怎么办？**  
A：运行 `python scripts/train.py --model vgg_baseline --epochs 1`，会自动下载；或拷贝 `data/cifar10/` 文件夹。

**Q：如何确认三人用的是同一验证集？**  
A：对比 `data/artifacts/cifar10_val_indices.pt` 文件是否一致（或相同 `seed` 下重新生成应得到相同索引）。

**Q：为什么 depthwise / ConvNeXt 的 FLOPs 和 VGG 差很多？**  
A：模板默认 `width=64` 未做预算对齐；扩展模型时请调节 `width` / 层数使 FLOPs 落在同一档位再比较准确率。

---

## 相关文档

- 项目提案：`DD2424_Project.md`
- 课程说明：`DefaultProject2.pdf`
- 报告草稿（含现有数据）：`REPORT_zh.md` / `REPORT.md`
- 报告大纲：`REPORT_OUTLINE_zh.md` / `REPORT_OUTLINE.md`
- 实验数据快照：`REPORT_DATA_zh.md`（`python scripts/generate_report_assets.py` 生成）
- **Overleaf 上传包：** `paper_overleaf_en.zip`（英文）、`paper_overleaf_zh.zip`（中文）；源码见 `paper/`，说明见 `paper/README_OVERLEAF.md`
