# DD2424 自评报告 — Group 148

*可填入 `self_assessment_form.pdf` 或作为单独 PDF 提交。若 GenAI 实际用法与下文不符，请自行修改相关段落。*

**English version:** `SELF_ASSESSMENT.md`

---

## 小组信息

| 字段 | 内容 |
|------|------|
| **组号** | 148 |
| **组员** | Boyi Shi, Puhao Zhu, Jinye Gong |
| **项目题目** | Scaling Laws and Accuracy–Efficiency Trade-offs in Modern ConvNets on CIFAR-10（现代 ConvNet 在 CIFAR-10 上的缩放规律与精度–效率权衡） |

---

## 建议等级

**Grade A**

我们认为本项目达到了 **A 档** 的预期，主要是因为完成了 proposal 中的 **B/A 扩展**：在相近 FLOPs 下比较三种 block，并对 width、kernel size、depth 做了缩放实验。

---

## 建议等级的依据

**1. 我们按 proposal 完成了主要工作**

在 CIFAR-10 上比较了 VGG、深度可分离和 ConvNeXt 风格 block，并将它们对齐到低、中、高 FLOPs 三档。实验队列中的训练均已完成（17 组，每组 200 epoch，seed 42）。最好结果是 `vgg_baseline` width 64 的 **test 92.78%**。

**2. 项目超出了基础要求**

除 E 级 VGG 基线外，我们还实现了另外两种架构、统一训练管线，并做了 width、kernel、depth 等消融。我们也用精度–FLOPs 图和简单的算力预算建议总结了结果。

**3. 报告和代码便于复现**

报告包含动机、相关工作、方法、结果和局限。训练设置在 `configs/default.yaml`，结果在 `logs/results.csv`，图表由脚本从日志生成，避免手改数字。

**4. 我们尝试解释「有用」和「没用」的结果**

例如，在 CIFAR-10 上增大 kernel 往往会降低精度，VGG 加宽到 width 80 也不一定更好。我们在报告中讨论了这些现象，并说明结论主要适用于本项目的设置（32×32、固定训练配方），而不是所有数据集或训练技巧。

---

## 局限

- 只在 **CIFAR-10** 上测试，结论主要针对小图分类。
- 使用 **单 seed（42）**，多种子时绝对精度可能略有变化。
- 报告的是 **thop 统计的 FLOPs**，未测量实际 GPU 速度或能耗。
- 主实验中 **关闭 MixUp 和 AMP**，以便各模型使用相同训练配方。
- 一组 ConvNeXt depth 实验曾中断；**已补训至 200 epoch** 后再写入终稿。

这些局限在报告中有讨论。它们也提醒了我们若有更多时间可以改进什么（多种子、更多数据集、测运行时间等）。

---

## 个人贡献与学习

### Boyi Shi（施博艺）

**贡献：** 在 `models/` 实现三族模型及 kernel/depth 变体；撰写架构与消融相关章节。

**收获：**

1. 在 PyTorch 中实现并比较不同 ConvNet block（VGG、深度可分离、ConvNeXt 风格）。
2. 学习如何设计消融：尽量只改一个因素（如 kernel 或 depth），其余设置保持稳定。

---

### Puhao Zhu（朱璞皓）

**贡献：** 负责 FLOPs 统计、档位对齐和精度–FLOPs 图；撰写算力对比相关章节。

**收获：**

1. 理解当默认配置算力差很大时，如何更公平地比较模型。
2. 学习如何把实验日志整理成能反映精度–效率权衡的图表。

---

### Jinye Gong（龚金烨）

**贡献：** 搭建训练管线和实验队列；管理配置与结果记录；参与讨论、结论与复现说明。

**收获：**

1. 学习如何用一致设置跑多组实验（数据划分、优化器、日志、按 val 选模型）。
2. 学习如何把实验结果与研究问题对应起来，并总结「能说什么、不能说什么」。

---

## GenAI 工具使用

我们有时使用 LLM 工具（如 ChatGPT、Cursor）辅助写作、排版、解释 PyTorch/LaTeX，或提供调试思路。

报告中的 **实验数字** 均来自本组训练结果，保存在 **`logs/results.csv`**。提交前我们对照 checkpoint 检查过，并用脚本重新生成了图表。

如有需要，各组员可补充个人使用方式。

---

*自评 — Group 148，2026 年 5 月*
